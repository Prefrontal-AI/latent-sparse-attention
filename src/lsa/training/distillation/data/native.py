"""Compute the checkpoint's own sparse selection offline, from the cached `x`.

    lsa distill native-picks ${LSA_ROOT}/data/targets/glm53_flash/long_16k_1m_heavy_x --workers 8 --wait
    lsa distill native-picks <root> --check <old root with native_key_ids> --samples 12

`dense_on_native` keeps the teacher's dense keys that GLM's own DSA indexer also picked. The extraction did not
keep the picks, but the indexer is a function of the layer's attention input and nothing else: its index keys
are `k_norm(wk(x))`, its queries come from `q_a_layernorm(q_a_proj(x))` through `wq_b`, its head weights are
`weights_proj(x)`, and it pools keys and takes the top-k over those causally visible. `x` is exactly that input,
so the picks at the cached query positions are one call of the model's own indexer over the cached `x`.

The key and query input, `indexer.input: key_and_query`, is read off the cached `x` the same way: GLM's key latent at
every position and each head's query folded through its key map at the sampled rows, the layer's own modules over the
layer's own input. `lsa distill served-read` reads it per head, and a run trained on it reads every head summed.
A model that stores its keys per head - Qwen3.8-27B - has no latent, and its input is the position-free part of every
key head side by side with each group's queries summed beside them; `key_and_query_input` is where a model is told
which of the two it reads.

Nothing is reimplemented. The checkpoint is loaded with everything but the sparse layers' attention left out,
so no expert is read, and [the adapter's blocked indexer](../../../evaluation/adapter/glm5_next.py) scores the
cached query rows against the keys of the whole document - the same call the live forward made, whose rows
at those positions are what it returns.

Each worker claims a (document, layer) with a claim file before computing it, so any number of them work one
cache, and the pass may start while the extraction is still writing: it only reads layer files that exist -
those are renamed into place whole - and with `--wait` a worker keeps looking until the manifest's documents
are all done.

The result is `<document>_layer<index>_native.pt` beside the layer file, holding `native_key_ids` `[Q, W]`
int32 in the convention the live capture had: the indexer's own output, `-1` where nothing was picked, the
sinks and the window included where it picked them. The target masks them as it masks everything else.
"""

import argparse
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lsa import env, io, slurm
from lsa.core import RotaryLayout, position_free, regroup_partial
from lsa.evaluation.adapter.layer import Projection
from lsa.evaluation.constants import Layer
from lsa.sglang.index.layer_index import KeyAndQuery
from lsa.sglang.settings.design import IndexerConfig
from lsa.training import require_extra
from lsa.training.constants import Targets
from lsa.training.distillation.constants import Cache, Native
from lsa.training.distillation.data.cache import Document, native_path, write

LAYER_FILE = re.compile(Native.LAYER_PATTERN)

# What a job asks for when the command line says nothing
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": Native.GPUS,
    "cpus_per_task": Native.CPUS,
    "mem_gb": Native.MEM_GB,
    "timeout_min": Native.HOURS * 60,
}


def load_attention(model: str, layers: list[int], device: str) -> dict[int, Any]:
    """Load the sparse layers' attention modules alone, with their indexers.

    The checkpoint's own model class is built on the meta device and pruned before a weight is read, as the
    adapter does for a span - here to the attention of the wanted layers, so the experts, the norms, the
    embedding and the vision tower are never read. The quantizer still swaps the modules and the fp8
    weights are read through the same path a full load takes.

    Args:
        model: The model card.
        layers: Layer indices whose attention is wanted.
        device: Where the weights land.

    Returns:
        Each layer's `self_attn`, in eval mode, its indexer scoring a block of queries at a time.
    """
    from transformers import AutoConfig

    from lsa.evaluation import adapter
    from lsa.evaluation.adapter.glm5_next import chunk_indexer_queries
    from lsa.evaluation.config import ModelConfig
    from lsa.evaluation.constants import Layer

    config = ModelConfig(name=model)
    model_adapter = adapter.adapter_for(config)
    auto_class = model_adapter.model_class()
    model_class = auto_class._model_mapping[type(AutoConfig.from_pretrained(config.checkpoint))]
    wanted = set(layers)

    def keep_attention(cls: Any, built: Any, distributed_config: Any, device_mesh: Any) -> Any:
        """Prune the built model to the wanted layers' attention, in the hook that runs before a weight is read."""
        text_model = model_adapter.text_model(built)
        for index in range(len(text_model.layers)):
            if index not in wanted:
                text_model.layers[index] = torch.nn.Identity()
                continue
            for name, _ in list(text_model.layers[index].named_children()):
                if name != "self_attn":
                    setattr(text_model.layers[index], name, torch.nn.Identity())
        text_model.embed_tokens = torch.nn.Identity()
        text_model.norm = torch.nn.Identity()
        if hasattr(built, "lm_head"):
            built.lm_head = torch.nn.Identity()
            built.all_tied_weights_keys = {}
        inner = model_adapter.inner_model(built)
        for tower in Layer.VISION_TOWERS:
            if hasattr(inner, tower):
                setattr(inner, tower, torch.nn.Identity())
        return built

    pruned: Any = type(
        f"{model_class.__name__}Attention",
        (model_class,),
        {"maybe_distribute_model": classmethod(keep_attention)},
    )
    from lsa.evaluation.adapter.model import torch_dtype

    loaded = pruned.from_pretrained(
        config.checkpoint,
        config=model_adapter.pretrained_config(config),
        dtype=torch_dtype(config.dtype),
        attn_implementation="sdpa",
        device_map=device,
    )
    loaded.eval()
    chunk_indexer_queries(loaded)
    text_model = model_adapter.text_model(loaded)
    return {layer: text_model.layers[layer].self_attn for layer in layers}


@torch.no_grad()
def pick(attention: Any, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Return the keys a layer's own indexer selects for some queries of a document.

    Args:
        attention: The layer's `self_attn`.
        x: The attention's input at every position [T, model_dim], on the attention's device.
        positions: Absolute query positions [Q] int64, on the same device.

    Returns:
        Selected key ids [Q, W] int32, `-1` where nothing was selected.
    """
    q_residual = attention.q_a_layernorm(attention.q_a_proj(x[positions][None]))
    mask = torch.ones(1, x.shape[0], dtype=torch.bool, device=x.device)
    selected = attention.indexer(
        hidden_states=x[None],
        q_resid=q_residual,
        attention_mask=mask,
        past_key_values=None,
        positions=positions,
    )
    return selected[0]


def key_latents(attention: Any, x: torch.Tensor) -> torch.Tensor:
    """Return GLM's key latent at every position, `kv_a_layernorm(kv_a_proj_with_mqa(x))`, a slice at a time.

    Args:
        attention: The layer's `self_attn`, as `load_attention` loads it.
        x: The attention's input [T, model_dim] bf16, on the attention's device.

    Returns:
        The key latents [T, kv_lora_rank] bf16, normed as attention caches them.
    """
    from lsa.evaluation.constants import Glm

    rank = attention.kv_lora_rank
    slices = []
    for start in range(0, x.shape[0], Glm.LATENT_SLICE):
        compressed = attention.kv_a_proj_with_mqa(x[None, start : start + Glm.LATENT_SLICE])
        slices.append(attention.kv_a_layernorm(compressed[..., :rank])[0])
    return torch.cat(slices).to(torch.bfloat16)


def head_queries(attention: Any, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Return each head's query at some positions, folded through its key map so it scores the key latent.

    Each head's query `q_b_proj(q_a_layernorm(q_a_proj(x)))` meets its key `K_h latent`, so `q_h K_h` is the head's
    query against the latent itself - folded in fp32 and kept in bf16, as the server's absorbed query is.

    Args:
        attention: The layer's `self_attn`, as `load_attention` loads it.
        x: The attention's input [T, model_dim] bf16, on the attention's device.
        positions: The query positions [R] int64, on the same device.

    Returns:
        The heads' queries [R, heads, kv_lora_rank] bf16.
    """
    from lsa.evaluation.adapter.glm5_next import key_maps

    q = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(x[None, positions])))[0]
    q = q.view(len(positions), -1, attention.qk_head_dim)[..., : attention.qk_nope_head_dim]
    return torch.einsum("rhd,hdc->rhc", q.float(), key_maps(attention).float()).to(torch.bfloat16)


def key_and_query(attention: Any, x: torch.Tensor, positions: torch.Tensor) -> KeyAndQuery:
    """Make the key and query input of a cached document from its `x` with the layer's own modules.

    What the freeze's probe extraction read off a layer (the adapter's `teacher_view`): the key latent at every
    position and each head's folded query at the sampled rows.

    Args:
        attention: The layer's `self_attn`, as `load_attention` loads it.
        x: The attention's input [T, model_dim] bf16, on the attention's device.
        positions: The sampled rows' positions [R] int64, on the same device.

    Returns:
        The key latents [T, kv_lora_rank] bf16 and the heads' queries [R, heads, kv_lora_rank] bf16.
    """
    return KeyAndQuery(key_latents(attention, x), head_queries(attention, x, positions))


@torch.no_grad()
def summed_queries(attention: Any, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Return every head's folded query summed at some positions, as `query_heads: all` serves it before the projection.

    The rows go `Native.QUERY_ROWS` at a time, since a million-token document samples some 9,400 and their 64 heads
    are 1.2 GB a pass in fp32.

    Args:
        attention: The layer's `self_attn`.
        x: The attention's input [T, model_dim] bf16, on the attention's device.
        positions: The query positions [R] int64, on the same device.

    Returns:
        The summed queries [R, kv_lora_rank] fp32.
    """
    return torch.cat(
        [
            head_queries(attention, x, positions[start : start + Native.QUERY_ROWS]).sum(
                1, dtype=torch.float32
            )
            for start in range(0, len(positions), Native.QUERY_ROWS)
        ]
    )


def latent_bases(attention: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the bases the freeze's key and query input was distilled in - `L` for the key, `L^-T` for the query.

    `L` factors the metric the key maps impose on the latent, `L L^T = sum_h K_h^T K_h`, and the freeze's cache held
    `latent @ L` and `summed_query @ L^-T` - the same logits, `q . latent`, in coordinates the concatenated keys make
    orthonormal. A map trained there on the key and query and folded back, `L @ W_K` and `L^-T @ W_Q`, reads the
    plain latent and summed query a server holds.

    Args:
        attention: The layer's `self_attn`.

    Returns:
        The key basis `L` and the query basis `L^-T`, each [kv_lora_rank, kv_lora_rank] fp32 on the attention's device.
    """
    from lsa.core import LatentMetric
    from lsa.evaluation.adapter.glm5_next import key_maps

    factor = LatentMetric.from_key_maps(key_maps(attention).detach()).metric_factor.double()
    return factor.float(), torch.linalg.inv(factor).t().float()


class LatentKeyAndQuery:
    """GLM's key and query input - an MLA layer's key latent at every position and every head's query folded into it.

    The heads share one latent, so the folded queries sum into one vector a row, and the maps act in the bases the
    freeze distilled them in.
    """

    def read(
        self, attention: Any, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the key latents and the summed folded queries of one document.

        Args:
            attention: The layer's `self_attn`.
            x: The attention's input [T, model_dim] bf16, on the attention's device.
            positions: The query positions [R] int64, on the same device.

        Returns:
            key: The key latents [T, kv_lora_rank] bf16.
            summed_query: The heads' folded queries summed [R, kv_lora_rank] fp32.
        """
        return key_latents(attention, x), summed_queries(attention, x, positions)

    def bases(self, attention: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the layer's key basis and query basis, as `latent_bases` makes them."""
        return latent_bases(attention)

    @torch.no_grad()
    def fields(self, attention: Any, x: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return what a cache under the key and query keeps of one (document, layer) in place of `x`, on the host.

        The summed queries are summed off the very head queries kept beside them, in `summed_queries`' slices, so a
        run reading them gets the tensors `read` makes of `x` - bit for bit.

        Args:
            attention: The layer's `self_attn`.
            x: The attention's input [T, model_dim] bf16, on the attention's device.
            positions: The query positions [R] int64, on the same device.

        Returns:
            The key latents [T, kv_lora_rank] bf16, the summed queries [R, kv_lora_rank] fp32 and each head's query
            [R, heads, kv_lora_rank] bf16.
        """
        heads = [
            head_queries(attention, x, positions[start : start + Native.QUERY_ROWS])
            for start in range(0, len(positions), Native.QUERY_ROWS)
        ]
        return {
            Cache.KEY_AND_QUERY_FIELDS[0]: key_latents(attention, x).cpu(),
            Cache.KEY_AND_QUERY_FIELDS[1]: torch.cat(
                [one.sum(1, dtype=torch.float32) for one in heads]
            ).cpu(),
            Cache.HEAD_QUERIES: torch.cat(heads).cpu(),
        }


class HeadKeyAndQuery:
    """The key and query input of a model that stores its keys per head - Qwen3.8-27B's grouped-query attention.

    The input is the position-free part of the logit, the rotated dims dropped. Every key head's slice sits side by
    side, and each group's query heads are summed into the slice of the key head they read, so the dot of the two is
    what every head's logit sums to on those dims and one selection serves every head - GLM's latent, without the
    latent. Nothing rotates, so there are no bases and the maps read the vectors as they are.
    """

    def __init__(self, attention_class: Any, rotary: RotaryLayout) -> None:
        """Hold what wraps a layer's attention and where its rotation sits.

        Args:
            attention_class: The adapter's per-layer module, whose `project` is the model's own projections.
            rotary: Where the rotation sits in a head vector, as the extraction measured it.
        """
        self.attention_class = attention_class
        self.rotary = rotary

    @torch.no_grad()
    def read(
        self, attention: Any, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the key heads side by side at every position and the grouped queries at some, position-free.

        A slice of positions at a time, keeping only the sampled rows' queries.

        Args:
            attention: The layer's `self_attn`.
            x: The attention's input [T, model_dim] bf16, on the attention's device.
            positions: The query positions [R] int64, on the same device.

        Returns:
            key: The key heads' position-free dims side by side [T, Hkv * (dh - rotary_dim)] bf16.
            summed_query: Each group's queries summed on the same dims [R, Hkv * (dh - rotary_dim)] fp32.
        """
        layer = self.attention_class(attention, IndexerConfig(), self.rotary)
        # `x` is cached at bf16, which the weights read in their own dtype
        x = x.to(next(attention.parameters()).dtype)
        keys = []
        summed_query: torch.Tensor | None = None
        # The slices a teacher projects a document in, so the online teacher's read of a document is this one
        for start in range(0, x.shape[0], Layer.TEACHER_SLICE):
            stop = min(start + Layer.TEACHER_SLICE, x.shape[0])
            projection = layer.project(x[None, start:stop])
            grouped_queries, grouped_keys = self.grouped(projection)
            keys.append(grouped_keys[0].to(torch.bfloat16))
            rows = ((positions >= start) & (positions < stop)).nonzero().squeeze(1)
            if summed_query is None:
                summed_query = grouped_queries.new_empty((len(positions), grouped_queries.shape[-1]))
            summed_query[rows] = grouped_queries[0, positions[rows] - start]
        if summed_query is None:
            raise ValueError("a document of no positions has no key and query to read")
        return torch.cat(keys), summed_query

    def grouped(self, projection: Projection) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one slice's queries and keys grouped into the input - position-free, keys side by side, queries summed.

        Args:
            projection: The slice's pre-RoPE projections.

        Returns:
            grouped_queries: Each group's queries summed [1, S, Hkv * (dh - rotary_dim)] fp32.
            grouped_keys: The key heads side by side [1, S, Hkv * (dh - rotary_dim)] fp32.
        """
        return regroup_partial(*position_free(projection.q, projection.k, self.rotary))

    def bases(self, attention: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return None - per-head keys read in no basis of their own."""
        return None

    def fields(self, attention: Any, x: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return what a cache under the key and query keeps of one (document, layer) in place of `x`, on the host.

        Args:
            attention: The layer's `self_attn`.
            x: The attention's input [T, model_dim] bf16, on the attention's device.
            positions: The query positions [R] int64, on the same device.

        Returns:
            The key heads side by side and the grouped queries, as `read` makes them - a group's heads are summed
            already, so there are no heads of their own to keep.
        """
        key, summed_query = self.read(attention, x, positions)
        return {Cache.KEY_AND_QUERY_FIELDS[0]: key.cpu(), Cache.KEY_AND_QUERY_FIELDS[1]: summed_query.cpu()}


type KeyAndQueryInput = LatentKeyAndQuery | HeadKeyAndQuery


def key_and_query_input(model: str, rotary: RotaryLayout) -> KeyAndQueryInput:
    """Return how one model's key and query input is read off its attention's input - the per-model glue.

    Args:
        model: The model card, or a checkpoint path.
        rotary: Where its rotation sits, as its extraction measured it.

    Returns:
        GLM's latent reading for an MLA model, and the per-head reading for every model whose keys are per head.
    """
    from lsa.evaluation import adapter
    from lsa.evaluation.config import ModelConfig

    config = ModelConfig(name=model)
    # The card's family where there is a card, so an MLA model is known without reading its checkpoint
    if config.asset is not None and config.asset.family == Native.LATENT_FAMILY:
        return LatentKeyAndQuery()
    model_adapter = adapter.adapter_for(config)
    if isinstance(model_adapter, adapter.Glm5NextAdapter):
        return LatentKeyAndQuery()
    return HeadKeyAndQuery(model_adapter.attention_class, rotary)


class KeyAndQueryReader:
    """Reads the key and query input off a document's `x` with its layer's own attention - `indexer.input: key_and_query`."""

    def __init__(
        self,
        attentions: dict[int, Any],
        closed_block: int | None = None,
        reading: KeyAndQueryInput | None = None,
    ) -> None:
        """Hold the layers' attention.

        Args:
            attentions: Each layer's `self_attn`, as `load_attention` loads it.
            closed_block: The cluster block, in keys, to lay over every document's visibility rule, or None.
            reading: How the model's key and query are read, `key_and_query_input`'s - GLM's latent when left out.
        """
        self.attentions = attentions
        self.closed_block = closed_block
        self.reading = reading if reading is not None else LatentKeyAndQuery()

    def bases(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the layer's key basis and query basis, or None for a model whose input reads in none."""
        return self.reading.bases(self.attentions[layer])

    @torch.no_grad()
    def __call__(self, document: Document) -> Document:
        """Return the document reading its keys and summed queries in place of `x`, which it lets go - or as it is, when its
        cache holds them already.

        Args:
            document: A cached document, on the device its layer's attention is on.

        Returns:
            The document under the key and query input.

        Raises:
            ValueError: The document holds neither `x` nor the key and query.
        """
        attention = self.attentions[document.layer]
        if document.index_keys is not None and document.index_queries is not None:
            return key_and_query_from_index_vectors(document, attention).with_closed_block(self.closed_block)
        if document.x is None and document.key is not None:
            # A cache extracted under the key and query holds them already
            return document.with_closed_block(self.closed_block)
        if document.x is None:
            raise ValueError(
                f"{document.name} layer {document.layer} has no x left to read the key and query off"
            )
        positions = document.query_positions.to(document.x.device)
        key, summed_query = self.reading.read(attention, document.x, positions)
        return document.with_key_and_query(key, summed_query).with_closed_block(self.closed_block)


def key_and_query_from_index_vectors(document: Document, attention: Any) -> Document:
    """Return an index-vector cache's document reading the key and summed query its index vectors were carried from.

    An index-vector cache holds `rownorm(latent @ L)` and `rownorm(summed_query @ L^-T)`. Taken back through `L^-1` and `L^T` they
    are the plain latent and summed query up to each row's scale, which every projection normalizes away - so the maps,
    their bases and their PCA init read them exactly as they read the ones made from `x`.

    Args:
        document: An index-vector document, on the device the attention is on.
        attention: Its layer's `self_attn`, whose key maps give `L`.

    Returns:
        The document under the key and query input, its keys [T, key_dim] bf16 and summed queries [Q, key_dim] fp32.
    """
    if document.index_keys is None or document.index_queries is None:
        raise ValueError(f"{document.name} layer {document.layer} holds no index vectors")
    key_basis, query_basis = latent_bases(attention)
    # [T, key_dim] bf16, as the key latent off x is, and [Q, key_dim] fp32 - `rownorm(latent @ L) @ L^-1` and
    # `rownorm(q @ L^-T) @ L^T`
    key = (document.index_keys.float() @ query_basis.t().to(document.device)).to(torch.bfloat16)
    summed_query = document.index_queries.float() @ key_basis.t().to(document.device)
    return document.with_key_and_query(key, summed_query)


def compute(path: Path, attention: Any, device: str) -> dict[str, Any]:
    """Compute the native selection of one cached (document, layer).

    Args:
        path: The layer file, read through a memory map.
        attention: That layer's `self_attn`.
        device: The device the attention is on.

    Returns:
        The blob to write: `native_key_ids` and a meta block.
    """
    blob = torch.load(path, mmap=True, weights_only=False)
    positions = blob["query_positions"].long().to(device)
    x = blob[Cache.X].to(device)
    picked = pick(attention, x, positions).cpu()
    meta = dict(blob[Cache.META])
    return {
        Cache.NATIVE: picked,
        Cache.META: {
            "document": meta.get("document"),
            "layer": meta.get("layer"),
            "length": int(x.shape[0]),
            "queries": int(positions.numel()),
            "width": int(picked.shape[1]),
        },
    }


def layer_files(directory: Path) -> list[tuple[Path, int]]:
    """List the finished layer files in a split directory, with their layer.

    A file being written is a `.tmp` and a sidecar ends in `_native`, and neither matches.

    Args:
        directory: A collection's split directory.

    Returns:
        (path, layer) per finished layer file.
    """
    found: list[tuple[Path, int]] = []
    if not directory.is_dir():
        return found
    with os.scandir(directory) as entries:
        for entry in entries:
            matched = LAYER_FILE.match(entry.name)
            if matched:
                found.append((directory / entry.name, int(matched["layer"])))
    return found


def pending(collection: Path) -> list[tuple[Path, int]]:
    """List the layer files that exist and have no selection yet.

    Args:
        collection: A collection directory, holding `train` and `eval`.

    Returns:
        (path, layer) per file still to do.
    """
    return [
        (path, layer)
        for split in ("train", "eval")
        for path, layer in layer_files(collection / split)
        if not native_path(path).exists()
    ]


def expected_files(collection: Path) -> int:
    """Return how many layer files the extraction will have written, from its manifest.

    Args:
        collection: A collection directory.

    Returns:
        Documents times layers, or 0 when there is no manifest yet.
    """
    manifest = collection / Targets.MANIFEST
    if not manifest.exists():
        return 0
    payload = io.read_json(manifest)
    return len(payload[Targets.DOCUMENTS]) * len(payload["layers"])


def claim_of(path: Path) -> io.Claim:
    """Return the claim that covers one (document, layer)'s selection."""
    return io.Claim(native_path(path).with_suffix(Native.CLAIM_SUFFIX), Native.CLAIM_STALE)


def work(
    root: str, collection: str, model: str, wait: bool, limit: int | None = None, device: str = "cuda:0"
) -> int:
    """Compute every selection that is missing, taking what no other worker holds.

    Returns when nothing is left among the files that exist and, with `wait`, when the extraction has written
    every file its manifest names too.

    Args:
        root: The cache root.
        collection: The collection under it.
        model: The model card.
        wait: Keep looking for new layer files until the extraction is complete.
        limit: Stop after this many selections, or None for no limit - a timing run.
        device: Where the weights land.

    Returns:
        How many selections this worker wrote.
    """
    require_extra("transformers", "omegaconf")
    directory = Path(root) / collection
    expected = expected_files(directory)
    layers = io.read_json(directory / Targets.MANIFEST)["layers"]
    attentions = load_attention(model, layers, device)
    rng = random.Random(os.getpid())
    written = 0
    while True:
        todo = pending(directory)
        rng.shuffle(todo)
        did = 0
        for path, layer in todo:
            if limit is not None and written + did >= limit:
                return written + did
            claim = claim_of(path)
            if not claim.take():
                continue
            try:
                if native_path(path).exists():
                    continue
                began = time.monotonic()
                write(native_path(path), compute(path, attentions[layer], device))
                did += 1
                print(f"{path.name}: {time.monotonic() - began:.1f} s", flush=True)
            finally:
                claim.release()
        written += did
        if not todo:
            total = sum(len(layer_files(directory / split)) for split in ("train", "eval"))
            if not wait or total >= expected:
                return written
        if not did:
            time.sleep(Native.POLL)


def agreement(picked: torch.Tensor, stored: torch.Tensor) -> dict[str, np.ndarray]:
    """Compare two selections row by row.

    Args:
        picked: Selection computed here [Q, W] int.
        stored: The live capture's [Q, W] int.

    Returns:
        Per row: whether the id sequences are identical, whether the id sets are, and the share of the stored
        row's ids the computed row also holds.
    """
    same_order, same_set, overlap = [], [], []
    for mine, theirs in zip(picked.numpy(), stored.numpy(), strict=True):
        a, b = np.unique(mine[mine >= 0]), np.unique(theirs[theirs >= 0])
        same_order.append(np.array_equal(mine, theirs))
        same_set.append(np.array_equal(a, b))
        overlap.append(len(np.intersect1d(a, b)) / max(len(b), 1))
    return {
        "same_order": np.array(same_order),
        "same_set": np.array(same_set),
        "overlap": np.array(overlap),
    }


def check(
    root: str, old_root: str, collection: str, model: str, samples: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Compare the offline selection with a cache whose forward captured it live, over a sample of its documents.

    The picks are computed from this cache's `x` at the old cache's own query positions and set beside the
    stored `native_key_ids`.

    Args:
        root: The cache whose `x` is read.
        old_root: A cache holding `native_key_ids` captured live.
        collection: The collection in both.
        model: The model card.
        samples: (document, layer) pairs per length.
        seed: Which pairs.

    Returns:
        One record per pair compared.
    """
    require_extra("transformers", "omegaconf")
    new, old = Path(root) / collection, Path(old_root) / collection
    pairs: dict[int, list[tuple[Path, Path, int]]] = {}
    for split in ("train", "eval"):
        for path, layer in layer_files(new / split):
            twin = old / split / path.name
            if twin.exists():
                length = int(re.search(r"_l(\d+)_", path.name)[1])  # type: ignore[index]
                pairs.setdefault(length, []).append((path, twin, layer))
    rng = random.Random(seed)
    chosen = [
        pair
        for length in sorted(pairs)
        for pair in rng.sample(pairs[length], min(samples, len(pairs[length])))
    ]
    layers = sorted({layer for _, _, layer in chosen})
    attentions = load_attention(model, layers, "cuda:0")
    records: list[dict[str, Any]] = []
    for path, twin, layer in chosen:
        blob = torch.load(twin, mmap=True, weights_only=False)
        positions = blob["query_positions"].long().cuda()
        x = torch.load(path, mmap=True, weights_only=False)[Cache.X].cuda()
        picked = pick(attentions[layer], x, positions).cpu()
        result = agreement(picked, blob[Cache.NATIVE])
        records.append(
            {
                "file": path.name,
                "layer": layer,
                "length": int(x.shape[0]),
                "rows": len(picked),
                "same_order": float(result["same_order"].mean()),
                "same_set": float(result["same_set"].mean()),
                "overlap": float(result["overlap"].mean()),
                "min_overlap": float(result["overlap"].min()),
            }
        )
        print(records[-1], flush=True)
    rows = sum(r["rows"] for r in records)
    print(
        f"{len(records)} pairs, {rows} rows: identical sequence "
        f"{sum(r['same_order'] * r['rows'] for r in records) / rows:.4f}, identical set "
        f"{sum(r['same_set'] * r['rows'] for r in records) / rows:.4f}, mean overlap "
        f"{sum(r['overlap'] * r['rows'] for r in records) / rows:.6f}",
        flush=True,
    )
    return records


def main() -> None:
    """Parse the command line and submit the workers, or the check."""
    parser = argparse.ArgumentParser(
        description="Compute GLM's own sparse selection from a cache's x, as slurm jobs"
    )
    parser.add_argument("root", help="the cache root, holding <collection>/{train,eval}")
    parser.add_argument("--collection", default="glm53_flash")
    parser.add_argument("--model", default="glm53_flash", help="the model card the cache was extracted from")
    parser.add_argument("--workers", type=int, default=1, help="jobs, each claiming (document, layer) items")
    parser.add_argument(
        "--wait", action="store_true", help="keep looking until the extraction has written every file"
    )
    parser.add_argument("--limit", type=int, help="stop each worker after this many selections")
    parser.add_argument(
        "--check", metavar="OLD_ROOT", help="compare with the live capture in this cache instead"
    )
    parser.add_argument("--samples", type=int, default=3, help="with --check, pairs sampled per length")
    slurm.add_flags(parser)
    args = parser.parse_args()
    root = str(Path(args.root).resolve())
    directory = Path(root) / args.collection
    print(
        f"{directory}: {len(pending(directory))} of {expected_files(directory)} files have layer files and no selection",
        flush=True,
    )
    if args.check:
        slurm.submit(
            args,
            "lsa-native-picks-check",
            env.OFFLINE_HF,
            PARAMETERS,
            check,
            root,
            str(Path(args.check).resolve()),
            args.collection,
            args.model,
            args.samples,
        )
        return
    if args.dry_run:
        print(f"would submit {args.workers} workers with {PARAMETERS | slurm.overrides(args)}")
        return
    executor = slurm.executor(
        Native.JOB_NAME, env.OFFLINE_HF, args.folder, **{**PARAMETERS, **slurm.overrides(args)}
    )
    jobs = executor.map_array(
        work,
        [root] * args.workers,
        [args.collection] * args.workers,
        [args.model] * args.workers,
        [args.wait] * args.workers,
        [args.limit] * args.workers,
    )
    for job in jobs:
        print(f"submitted {job.job_id}  {Native.JOB_NAME}", flush=True)
