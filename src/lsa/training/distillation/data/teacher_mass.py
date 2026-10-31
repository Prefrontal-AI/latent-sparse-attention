"""Dump the teacher's dense attention mass - every head's softmax over all causal keys, not a selection's share of it.

    lsa distill targets configs/data/targets/glm53_flash/dense_mass.yaml

This is `data.mass` of the [extraction](teacher.py), which runs the same documents, the same prefill and the same
[adapter](../../../evaluation/adapter/__init__.py) as a target extraction and writes this in place of the layer files.
It is a mode of that entry point rather than a second one because everything that is expensive and fiddly is shared -
the chunked prefill, the layer inputs captured on the way past, the spans, the claims and the plan - and what differs
is only what a layer does with its input.
The layer files stay the trainer's and are unchanged.

The format is language-neutral on purpose: another dumper, in the serving stack, writes the same files.

## Layout

    <root>/<collection>/<split>/<window>/layer<index>.safetensors

One [safetensors](https://huggingface.co/docs/safetensors) file per (window, layer), written atomically.
`<window>` is the extraction's window name, `<split>` is `train` or `eval`, and the root also holds the extraction's
`manifest.json` as any target extraction does.
A reader takes the rows without the key by opening the file lazily and slicing - the key is nearly all of its bytes.

## Fields

`Q` is the rows, the last `queries.tail` positions of the window.
`H` is the query heads, 64 on GLM-5.3-Flash.
`T` is the window's token count and `key_dim` the key's dim, 512.
`K` is `data.mass.top`, 4096, and `J` the count of `data.mass.head_tops`, two.

| field | shape | dtype | what |
|---|---|---|---|
| `query_positions` | `[Q]` | int32 | absolute position of each row |
| `head_queries` | `[Q, H, key_dim]` | bf16 | every head's query folded into the key's space - the model's logit is its dot with `key` times `scale` |
| `key` | `[T, key_dim]` | bf16 | the key at every position, the latent every head reads, as the layer's own modules build it |
| `shared_key_ids` | `[Q, K]` | int32 | the keys of the largest all-causal mass, strongest first |
| `shared_key_mass` | `[Q, K]` | fp32 | their mass - each head's softmax over all causal keys, summed over heads and divided by `H` |
| `shared_key_flags` | `[Q, K]` | uint8 | bit 0 set for a sink key, bit 1 for a window key that is not a sink |
| `sink_mass` | `[Q]` | fp32 | the mass on the sinks, summed over heads and divided by `H` |
| `window_mass` | `[Q]` | fp32 | the mass on the window outside the sinks, the same |
| `head_sink_mass` | `[Q, H]` | fp32 | each head's mass on the sinks |
| `head_window_mass` | `[Q, H]` | fp32 | each head's mass on the window outside the sinks |
| `head_top_mass` | `[Q, J, H]` | fp32 | the mass each head's own `head_tops[j]` strongest keys hold |
| `head_lse` | `[Q, H]` | fp32 | each head's log-sum-exp of its scaled logits over the causal keys |
| `cache_key_ids` | `[Q, K]` | int32 | the keys of the largest cache-style mass, strongest first |
| `cache_key_mass` | `[Q, K]` | fp32 | their mass - each head's softmax over the keys a read may select, summed over heads and not divided |

The mass on the rest of the keys is one minus the sinks and the window, per head, because the all-causal softmax of a head sums to one.
A key is in the window when it is less than `window_size` behind the row and past the sinks, so a row's decomposition holds at any position.
The cache-style set is the one `top_key_ids` and `top_key_mass` of a layer file hold - the sinks and the window are masked, the softmax is
per head over the readable keys, and the heads are summed - so its first `k` entries are a layer file's at `queries.top: k`.

The metadata holds strings only - `format` (`dense_mass`), `version`, `model`, `document`, `layer`, `length` (T), `sink_size`,
`window_size`, `scale` (the model's own attention scale, applied to the logits and never changed), `head_count`, `top`,
`head_tops` (a json list) and `queries` (the rows' count).

The softmax and the log-sum-exp are fp32 throughout.
The key is stored as bf16 and is bit-for-bit the model's, which computes it in bf16; the folded queries are rounded to bf16 from fp32, as the
`key_and_query` input is.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lsa import constants as const
from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.constants import Mass
from lsa.training.distillation.data.teacher_attention import causal_keys, head_logits, readable_keys

SHARED_KEY_IDS = "shared_key_ids"
SHARED_KEY_MASS = "shared_key_mass"
SHARED_KEY_FLAGS = "shared_key_flags"
SINK_MASS = "sink_mass"
WINDOW_MASS = "window_mass"
HEAD_SINK_MASS = "head_sink_mass"
HEAD_WINDOW_MASS = "head_window_mass"
HEAD_TOP_MASS = "head_top_mass"
HEAD_LSE = "head_lse"
CACHE_KEY_IDS = "cache_key_ids"
CACHE_KEY_MASS = "cache_key_mass"
QUERY_POSITIONS = "query_positions"
HEAD_QUERIES = "head_queries"
KEY = "key"


def mass_path(root: str | Path, collection: str, split: str, document: str, layer: int) -> Path:
    """Return where one (window, layer) of a dump lives.

    Args:
        root: The dump's root.
        collection: Directory under the root.
        split: `train` or `eval`.
        document: The window's name.
        layer: Layer index.

    Returns:
        The file path, which may not exist.
    """
    return Path(root) / collection / split / document / f"{Mass.LAYER_PREFIX}{layer}{Mass.SUFFIX}"


@dataclass(frozen=True)
class MassRequest:
    """What a dump keeps per row.

    Args:
        top: Keys the shared sets keep.
        head_tops: The key counts each head's own mass is taken at, ascending.
    """

    top: int
    head_tops: tuple[int, ...]


@torch.no_grad()
def dense_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    visibility: IndexerConfig,
    request: MassRequest,
) -> dict[str, torch.Tensor]:
    """Read off every head's softmax over all causal keys, and the cache-style set beside it.

    The logits are taken a block of rows at a time and the block narrows as the document grows, as in
    [teacher_attention](teacher_attention.py), so one block's logits are the same size at 1M as at 64k.
    Everything stays fp32 through the softmax and the log-sum-exp.

    Args:
        q: Query vectors at the sampled positions [Hq, Q, dh].
        k: The key, one latent shared by every head [1, T, dh], whose dot with `q` is the model's logit.
        scale: The model's own attention scale.
        positions: Absolute query positions [Q] int64.
        visibility: The rule that says which keys the cache-style set may take.
        request: How many keys each set keeps.

    Returns:
        The fields of the module's table other than the key and the folded queries, on the device of `q`, each
        with its leading dim Q.

    Raises:
        ValueError: The key has more than one head, or the document has fewer keys than a set keeps.
    """
    head_count, row_count = q.shape[0], q.shape[1]
    key_count = k.shape[1]
    if k.shape[0] != 1:
        raise ValueError(
            f"a dense mass dump reads one latent shared by every head - got {k.shape[0]} key-value heads"
        )
    if key_count < max(request.top, *request.head_tops):
        raise ValueError(
            f"a {key_count}-token document has fewer keys than the {max(request.top, *request.head_tops)} a dump keeps"
        )
    keys = k.float()
    block_rows = max(1, Mass.SCORE_CELLS // (head_count * key_count))
    blocks: list[dict[str, torch.Tensor]] = []
    for start in range(0, row_count, block_rows):
        chunk = slice(start, start + block_rows)
        blocks.append(
            mass_block(
                head_logits(q[:, chunk], keys, scale), positions[chunk], key_count, visibility, request
            )
        )
    return {name: torch.cat([block[name] for block in blocks]) for name in blocks[0]}


def mass_block(
    logits: torch.Tensor,
    positions: torch.Tensor,
    key_count: int,
    visibility: IndexerConfig,
    request: MassRequest,
) -> dict[str, torch.Tensor]:
    """Return a block of rows' masses from their logits.

    Args:
        logits: The block's scaled logits [Hq, R, T] fp32, unmasked.
        positions: The block's absolute query positions [R] int64.
        key_count: Tokens in the document.
        visibility: The rule that says which keys the cache-style set may take.
        request: How many keys each set keeps.

    Returns:
        The block's fields, each with leading dim R.
    """
    head_count = logits.shape[0]
    causal = causal_keys(positions, key_count)
    keys = torch.arange(key_count, device=logits.device)
    sink = keys < visibility.sink_size
    # Past the sinks and less than a window behind the row
    window = (positions[:, None] - keys[None] < visibility.window_size) & causal & ~sink[None]

    full = logits.masked_fill(~causal[None], const.Math.NINF)
    lse = torch.logsumexp(full, dim=-1)
    attention = torch.softmax(full, dim=-1)
    del full

    shared = (attention.sum(0) / head_count).topk(request.top, dim=-1)
    head_sink = attention[..., : visibility.sink_size].sum(-1)
    head_window = attention.masked_fill(~window[None], 0.0).sum(-1)
    strongest = attention.topk(max(request.head_tops), dim=-1).values
    head_top = torch.stack([strongest[..., :count].sum(-1) for count in request.head_tops])
    del attention, strongest

    readable = readable_keys(visibility, positions, key_count)
    cache_attention = torch.softmax(logits.masked_fill(~readable[None], const.Math.NINF), dim=-1)
    cache = cache_attention.sum(0).topk(request.top, dim=-1)
    del cache_attention

    ids = shared.indices
    in_sink = ids < visibility.sink_size
    in_window = (positions[:, None] - ids < visibility.window_size) & ~in_sink
    flags = in_sink.to(torch.uint8) * Mass.SINK_BIT + in_window.to(torch.uint8) * Mass.WINDOW_BIT
    return {
        SHARED_KEY_IDS: ids.int(),
        SHARED_KEY_MASS: shared.values,
        SHARED_KEY_FLAGS: flags,
        SINK_MASS: head_sink.sum(0) / head_count,
        WINDOW_MASS: head_window.sum(0) / head_count,
        HEAD_SINK_MASS: head_sink.t(),
        HEAD_WINDOW_MASS: head_window.t(),
        # [J, H, R] to [R, J, H]
        HEAD_TOP_MASS: head_top.permute(2, 0, 1),
        HEAD_LSE: lse.t(),
        CACHE_KEY_IDS: cache.indices.int(),
        CACHE_KEY_MASS: cache.values,
    }


@torch.no_grad()
def mass_fields(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    visibility: IndexerConfig,
    request: MassRequest,
) -> dict[str, torch.Tensor]:
    """Return everything one (window, layer) file holds, on the host.

    Args:
        q: Query vectors at the sampled positions [Hq, Q, dh], the heads' queries folded into the key's space.
        k: The key, one latent shared by every head [1, T, dh].
        scale: The model's own attention scale.
        positions: Absolute query positions [Q] int64.
        visibility: The rule that says which keys the cache-style set may take.
        request: How many keys each set keeps.

    Returns:
        The table's fields, contiguous, in the dtypes it states.
    """
    fields = dense_mass(q, k, scale, positions, visibility, request)
    fields = {
        QUERY_POSITIONS: positions.int(),
        HEAD_QUERIES: q.transpose(0, 1).to(torch.bfloat16),
        KEY: k[0].to(torch.bfloat16),
        **fields,
    }
    return {name: tensor.contiguous().cpu() for name, tensor in fields.items()}


def mass_metadata(
    model: str,
    document: str,
    layer: int,
    length: int,
    visibility: IndexerConfig,
    scale: float,
    head_count: int,
    request: MassRequest,
    query_count: int,
) -> dict[str, str]:
    """Return the strings a file carries beside its tensors.

    Args:
        model: The model card's name.
        document: The window's name.
        layer: Layer index.
        length: Tokens in the window.
        visibility: The sinks and window the masses are split by.
        scale: The model's own attention scale.
        head_count: Query heads.
        request: How many keys each set keeps.
        query_count: Rows.

    Returns:
        Every value a string.
    """
    return {
        "format": Mass.FORMAT,
        "version": str(Mass.VERSION),
        "model": model,
        "document": document,
        "layer": str(layer),
        "length": str(length),
        "sink_size": str(visibility.sink_size),
        "window_size": str(visibility.window_size),
        "scale": repr(float(scale)),
        "head_count": str(head_count),
        "top": str(request.top),
        "head_tops": json.dumps(list(request.head_tops)),
        "queries": str(query_count),
    }


def write_mass(path: Path, fields: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """Write one (window, layer) atomically.

    Args:
        path: Destination file, whose parents are created.
        fields: The table's tensors, contiguous and on the host.
        metadata: The strings beside them.
    """
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    save_file(fields, str(scratch), metadata=metadata)
    os.replace(scratch, path)


def read_mass(path: Path, names: list[str] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read a (window, layer) file whole, or only the fields named.

    Args:
        path: The file.
        names: The fields to load, or None for all of them.

    Returns:
        The tensors and the metadata.

    Raises:
        ValueError: The file is not a dump of this format and version, or lacks a field asked for.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != Mass.FORMAT or metadata.get("version") != str(Mass.VERSION):
            raise ValueError(
                f"{path} is {metadata.get('format')!r} version {metadata.get('version')!r} - "
                f"wanted {Mass.FORMAT!r} version {Mass.VERSION}"
            )
        wanted = list(handle.keys()) if names is None else names
        missing = [name for name in wanted if name not in handle.keys()]
        if missing:
            raise ValueError(f"{path} holds no {missing}")
        return {name: handle.get_tensor(name) for name in wanted}, metadata


def dump_layer(
    reader: Any,
    hidden: torch.Tensor,
    position_embeddings: Any,
    positions: torch.Tensor,
    path: Path,
    model: str,
    document: str,
    visibility: IndexerConfig,
    request: MassRequest,
) -> None:
    """Dump one (window, layer).

    Args:
        reader: The layer's sparse-attention wrapper, used for its projections and its latent.
        hidden: The layer input [1, T, model_dim], on its layer's device or on the host.
        position_embeddings: The model's own `(cos, sin)` tables, or None for a model that does not rotate.
        positions: Absolute query positions [Q] int64.
        path: Where the file goes.
        model: The model card's name.
        document: The window's name.
        visibility: The sinks and window the masses are split by.
        request: How many keys each set keeps.
    """
    # A model spread over several cards reads each layer on that layer's own device
    device = next(reader.attention.parameters()).device
    positions = positions.to(device)
    view = reader.teacher_view(hidden, positions, position_embeddings)
    scale = float(reader.attention.scaling)
    fields = mass_fields(view.q, view.k, scale, positions, visibility, request)
    metadata = mass_metadata(
        model,
        document,
        int(reader.attention.layer_idx),
        int(view.k.shape[1]),
        visibility,
        scale,
        int(view.q.shape[0]),
        request,
        len(positions),
    )
    write_mass(path, fields, metadata)
