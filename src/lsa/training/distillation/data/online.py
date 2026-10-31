"""The frozen teacher run inside a distillation job - every drawn document read once, its targets handed to the pools.

    data:
      teacher:
        extraction: configs/data/targets/qwen38_27b/long_16k_1m_heavy.yaml

An `x` cache of Qwen3.8-27B over the heavy mix would be some 60 TiB - 403M tokens, sixteen layers and 10 KiB a token a
layer - so a run that names `data.teacher` reads no cache. It loads the frozen model beside its maps, reads every
document its steps draw through it once, and hands each layer's targets to the same rolling pool a cache feeds.
Nothing is written.

Nothing numeric is new. The documents, their query positions and their targets are the extraction's own - `plan`,
`query_positions`, the layer's own projections in the extraction's slices, and `targets_blob` - called on the extraction
config the run names, and a document is built under the name the extraction would have written it, so it is the
document a pool reading that cache gets.

What differs is what is held. An extraction keeps each layer's input whole until the forward ends and projects it
after - a 1M document's sixteen inputs are 168 GB of host memory, and seven copies of the teacher would hold a node's.
Under `indexer.input: key_and_query` the teacher projects each chunk as the forward hands it over instead
(`TeacherViews`), in the slices the extraction projects in, and keeps only what the targets and the input need: the
rotated key heads, the sampled rows' query heads, and the key and query input - 3.5 GB a layer where the input is 10.
The attention input `x` itself is the one input this cannot serve, and a run on it keeps each layer's input as an
extraction does.

A teacher reads every layer at once, so the layers share one arrival order (`shared_order`) and each pool meets its
documents in the order the teacher reads them. Every card the trainer leaves holds a copy of the model, each reading
the next document while the documents already read wait in host memory, up to `ahead_gb`, for the pools to take them.
A pool waiting on a document no copy has started lets the copies read past that bound, because the trainer's steps go
round the layers and one waiting pool stops them all.

A job of several nodes has copies on every card of the others too, which take a share of the same documents and leave
them on the shared disk for the trainer - see [teacher_nodes](teacher_nodes.py).
"""

import gc
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.baselines import qsa
from lsa.core import RotaryLayout
from lsa.sglang.settings.design import KEY_AND_QUERY, IndexerConfig, QSAConfig
from lsa.training import require_extra
from lsa.training.distillation.constants import Cache, Online, Pool
from lsa.training.distillation.data.cache import Document, layer_path, selected_entries
from lsa.training.distillation.data.native import (
    HeadKeyAndQuery,
    key_and_query_input,
)
from lsa.training.distillation.data.pool import Entry, Item
from lsa.training.distillation.data.teacher import layer_blob, targets_blob
from lsa.training.distillation.data.teacher_capture import capture_layer_inputs, input_device, read_through
from lsa.training.distillation.data.teacher_config import TeacherConfig, load
from lsa.training.distillation.data.teacher_nodes import Request, Staging
from lsa.training.distillation.data.teacher_plan import (
    Planned,
    collection_entries,
    flatten_plan,
    plan,
    query_positions,
    selection_payload,
    training_query_count,
    window_length,
)

if TYPE_CHECKING:
    from lsa.training.distillation.config import DocumentSet, TeacherSource


def extraction_of(source: "TeacherSource") -> TeacherConfig:
    """Return the one extraction a teacher source names.

    Args:
        source: The run's `data.teacher`.

    Returns:
        The extraction, with its overrides applied.

    Raises:
        ValueError: The file describes a sweep rather than one extraction.
    """
    extractions = load(source.extraction, source.overrides)
    if len(extractions) != 1:
        raise ValueError(
            f"{source.extraction} describes {len(extractions)} extractions - a teacher reads one"
        )
    return extractions[0]


def teacher_layers(extraction: TeacherConfig) -> list[int]:
    """Return the layers an extraction captures, off the checkpoint's config alone.

    Args:
        extraction: The extraction.

    Returns:
        Layer indices, in order.
    """
    from lsa.evaluation import adapter

    if extraction.run.layers:
        return list(extraction.run.layers)
    model_adapter = adapter.adapter_for(extraction.model)
    return model_adapter.full_attention_layers(model_adapter.text_config(extraction.model), extraction.model)


class Stopped(Exception):
    """The teacher was closed while a copy read a document or a pool waited on one."""


@dataclass
class Replica:
    """One copy of the frozen model, on its own card, and what it reads each layer with."""

    device: str
    text_model: Any
    inputs: torch.device
    # Each layer's sparse-attention wrapper, built for its projections and its rotation
    readers: dict[int, Any]
    # How the key and query input is read off the projections, or None to keep `x`
    reading: HeadKeyAndQuery | None


class TeacherViews:
    """What each wanted layer's attention is handed, projected as a document goes through it - a chunk at a time.

    Each chunk is projected in the extraction's slices by the layer's own wrapper, and what is kept is the rotated key
    heads of every position, the sampled rows' query heads and the key and query input, never the layer's input.
    """

    def __init__(self, readers: dict[int, Any], positions: torch.Tensor, reading: HeadKeyAndQuery) -> None:
        """Start empty.

        Args:
            readers: Each wanted layer's sparse-attention wrapper.
            positions: Absolute query positions [Q] int64, on the layers' device.
            reading: How the key and query input is grouped off a slice's projections.
        """
        self.readers = readers
        self.positions = positions
        self.reading = reading
        self.covered = dict.fromkeys(readers, 0)
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, list[torch.Tensor]] = {layer: [] for layer in readers}
        self.key: dict[int, list[torch.Tensor]] = {layer: [] for layer in readers}
        self.summed_query: dict[int, torch.Tensor] = {}

    def hook(self, text_model: Any) -> list[Any]:
        """Hook every wanted layer's attention, and return the handles to remove once the document is through."""
        return [
            text_model.layers[layer].self_attn.register_forward_pre_hook(
                partial(self.keep, layer), with_kwargs=True
            )
            for layer in self.readers
        ]

    @torch.no_grad()
    def keep(self, layer: int, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Project one chunk of one layer's input and keep what the targets and the input need of it.

        Args:
            layer: The layer.
            _module: The attention module, unused.
            args: Its positional arguments, the hidden states first when they are not a keyword.
            kwargs: Its keyword arguments, which carry the rotary tables.
        """
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        start = self.covered[layer]
        relative = self.positions - start
        for piece in self.readers[layer].teacher_slices(hidden, relative, kwargs.get("position_embeddings")):
            grouped_queries, grouped_keys = self.reading.grouped(piece.projection)
            if layer not in self.q:
                self.q[layer] = piece.q.new_empty((piece.q.shape[0], len(self.positions), piece.q.shape[-1]))
                self.summed_query[layer] = grouped_queries.new_empty(
                    (len(self.positions), grouped_queries.shape[-1])
                )
            self.q[layer][:, piece.rows] = piece.q[:, piece.offsets]
            self.summed_query[layer][piece.rows] = grouped_queries[0, piece.offsets]
            self.k[layer].append(piece.k.to(Cache.HOST))
            self.key[layer].append(grouped_keys[0].to(torch.bfloat16).to(Cache.HOST))
        self.covered[layer] = start + hidden.shape[1]

    def take(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one layer's view of the whole document, and let it go.

        Args:
            layer: The layer.

        Returns:
            q: The sampled rows' rotated query heads [Hq, Q, dh], on the layer's device.
            k: Every position's rotated key heads [Hkv, T, dh], on the host.
            key: The key and query input's keys [T, d] bf16, on the host.
            summed_query: Its grouped queries at the sampled rows [Q, d] fp32, on the layer's device.
        """
        k, key = self.k.pop(layer), self.key.pop(layer)
        return self.q.pop(layer), torch.cat(k, dim=1), torch.cat(key), self.summed_query.pop(layer)


class BlockViews:
    """What a QSA indexer needs of each wanted layer, kept as a document goes through it - a chunk at a time.

    The indexer reads `x`, which at 10 KiB a token a layer is not kept. Its keys are pooled before they are projected,
    and the projection is linear, so a block's mean `x` is all a key needs - a quarter of `x` at blocks of four - and
    a query needs `x` at the sampled rows alone. The rotated key heads and the sampled rows' query heads are kept for
    the targets, as `TeacherViews` keeps them.
    """

    def __init__(
        self,
        readers: dict[int, Any],
        positions: torch.Tensor,
        block_size: int,
        reading: HeadKeyAndQuery | None = None,
    ) -> None:
        """Start empty.

        Args:
            readers: Each wanted layer's sparse-attention wrapper.
            positions: Absolute query positions [Q] int64, on the layers' device.
            block_size: The keys a block pools.
            reading: How the key and query input is grouped off a slice's projections, to pool it in place of `x`, or
                None to pool `x`.
        """
        self.readers = readers
        self.positions = positions
        self.block_size = block_size
        self.reading = reading
        self.covered = dict.fromkeys(readers, 0)
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, list[torch.Tensor]] = {layer: [] for layer in readers}
        self.block_x: dict[int, list[torch.Tensor]] = {layer: [] for layer in readers}
        self.query_x: dict[int, torch.Tensor] = {}

    def hook(self, text_model: Any) -> list[Any]:
        """Hook every wanted layer's attention, and return the handles to remove once the document is through."""
        return [
            text_model.layers[layer].self_attn.register_forward_pre_hook(
                partial(self.keep, layer), with_kwargs=True
            )
            for layer in self.readers
        ]

    @torch.no_grad()
    def keep(self, layer: int, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Pool one chunk of one layer's input into blocks, and keep its sampled rows and the heads the targets need.

        Args:
            layer: The layer.
            _module: The attention module, unused.
            args: Its positional arguments, the hidden states first when they are not a keyword.
            kwargs: Its keyword arguments, which carry the rotary tables.

        Raises:
            ValueError: The chunk starts inside a block, which a prefill of whole blocks never does.
        """
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        start = self.covered[layer]
        if start % self.block_size:
            raise ValueError(f"a chunk starts at {start}, inside a block of {self.block_size}")
        relative = self.positions - start
        if self.reading is None:
            self._keep_rows(layer, hidden[0], relative)
        for piece in self.readers[layer].teacher_slices(hidden, relative, kwargs.get("position_embeddings")):
            if layer not in self.q:
                self.q[layer] = piece.q.new_empty((piece.q.shape[0], len(self.positions), piece.q.shape[-1]))
            self.q[layer][:, piece.rows] = piece.q[:, piece.offsets]
            self.k[layer].append(piece.k.to(Cache.HOST))
            if self.reading is not None:
                # A slice starts at a multiple of the teacher's slice, so its blocks are the document's
                grouped_queries, grouped_keys = self.reading.grouped(piece.projection)
                self._keep_rows(layer, grouped_keys[0], piece.offsets, piece.rows, grouped_queries[0])
        self.covered[layer] = start + hidden.shape[1]

    def _keep_rows(
        self,
        layer: int,
        keys: torch.Tensor,
        offsets: torch.Tensor,
        rows: torch.Tensor | None = None,
        queries: torch.Tensor | None = None,
    ) -> None:
        """Pool some positions' key input into blocks and keep the sampled rows' query input among them.

        Args:
            layer: The layer.
            keys: The key input of consecutive positions [S, d], starting at a block's first.
            offsets: The sampled rows' offsets into them [R] int64 - every row's, out of range where outside, for `x`.
            rows: Which sampled rows the offsets are, or None for every row.
            queries: The query input of the same positions [S, d], or None to read the rows off `keys` - `x` is both.
        """
        if rows is None:
            rows = ((offsets >= 0) & (offsets < keys.shape[0])).nonzero().squeeze(1)
            offsets = offsets[rows]
        source = keys if queries is None else queries
        if layer not in self.query_x:
            dtype = keys.dtype if queries is None else torch.float32
            self.query_x[layer] = source.new_empty((len(self.positions), source.shape[-1]), dtype=dtype)
        self.query_x[layer][rows] = source[offsets].to(self.query_x[layer].dtype)
        self.block_x[layer].append(qsa.pool_blocks(keys, self.block_size).to(torch.bfloat16).to(Cache.HOST))

    def take(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one layer's view of the whole document, and let it go.

        Args:
            layer: The layer.

        Returns:
            q: The sampled rows' rotated query heads [Hq, Q, dh], on the layer's device.
            k: Every position's rotated key heads [Hkv, T, dh], on the host.
            block_x: Each closed block's mean key input [T // block_size, d] bf16, on the host - `x` or the key.
            query_x: The query input at the sampled rows [Q, d], on the host - `x` bf16, or the summed queries fp32.
        """
        k, block_x = self.k.pop(layer), self.block_x.pop(layer)
        query_x = self.query_x.pop(layer)
        query_x = (query_x if self.reading is not None else query_x.to(torch.bfloat16)).to(Cache.HOST)
        return self.q.pop(layer), torch.cat(k, dim=1), torch.cat(block_x), query_x


class OnlineTeacher:
    """The extraction a run names, run as the run goes - one copy of the model per card, read on threads."""

    def __init__(
        self,
        source: "TeacherSource",
        indexer: IndexerConfig,
        model: str,
        staging: Staging | None = None,
        qsa_indexer: QSAConfig | None = None,
    ) -> None:
        """Resolve the extraction's documents, without loading a weight.

        Args:
            source: The run's `data.teacher`.
            indexer: The run's indexer, whose input says what a document holds and whose visibility rule must be the
                extraction's.
            model: The run's model card, which must be the one the extraction reads.
            staging: Where the copies on the job's other nodes leave what they read, or None for a job of one node.
            qsa_indexer: The QSA indexer the run trains in place of an LSA map, whose blocks and target a document
                holds, or None.

        Raises:
            ValueError: The extraction reads another model, or under another visibility rule, or prefills in chunks
                that cut a QSA block.
        """
        require_extra("omegaconf", "transformers", "pyarrow")
        self.source = source
        self.extraction = extraction_of(source)
        if self.extraction.model.name != model:
            raise ValueError(
                f"{source.extraction} reads {self.extraction.model.name!r} and the run trains {model!r} - a map is "
                "only valid on the weights whose states trained it"
            )
        extracted = self.extraction.model.indexer
        if (extracted.sink_size, extracted.window_size) != (indexer.sink_size, indexer.window_size):
            raise ValueError(
                f"{source.extraction} takes its targets past {extracted.sink_size} sinks and a {extracted.window_size}-key "
                f"window, and the run reads {indexer.sink_size} and {indexer.window_size}"
            )
        self.indexer = indexer
        self.qsa_indexer = qsa_indexer
        if qsa_indexer is not None and self.extraction.run.prefill % qsa_indexer.block_size:
            raise ValueError(
                f"{source.extraction} prefills {self.extraction.run.prefill} tokens a chunk, which cuts a block of "
                f"{qsa_indexer.block_size}"
            )
        self.key_and_query = indexer.input == KEY_AND_QUERY and qsa_indexer is None
        # Whether a copy groups the attention's key and query off its projections - an LSA map's or a QSA indexer's
        self.reads_key_and_query = (
            self.key_and_query if qsa_indexer is None else qsa_indexer.input == KEY_AND_QUERY
        )
        self.selection = selection_payload(self.extraction) if self.extraction.data.selection else None
        planned = plan(self.extraction, None, self.selection)
        self.index = collection_entries(self.extraction, planned)
        self.planned = {(one.documents.collection, one.selected.name): one for one in flatten_plan(planned)}
        self.layers = teacher_layers(self.extraction)
        self.root = Path(self.extraction.run.output)
        self.replicas: list[Replica] = []
        self.rotary: RotaryLayout | None = None
        # The model's rotary frequencies [rotary_dim / 2] fp32 and the scale its tables carry, once a copy is loaded
        self.rotary_frequencies: tuple[torch.Tensor, float] | None = None

        self.staging = staging
        self._cond = threading.Condition()
        # Each request as every node numbers it - the document and the layers it is read for
        self._queue: deque[tuple[int, Planned, list[int]]] = deque()
        self._requested = 0
        # A document's requests not yet taken whole, oldest first, and the layers each has left to be taken
        self._requests_of: dict[tuple[str, str], deque[int]] = {}
        self._layers_left: dict[int, set[int]] = {}
        # Documents a copy is reading now, and the (document, layer) pairs a pool is waiting on
        self._reading: set[tuple[str, str]] = set()
        self._waiting: dict[tuple[str, str, int], int] = {}
        self._produced: dict[tuple[str, str, int], Document] = {}
        # Documents queued ahead of the `read_now` that will take them, with the layers they are read for
        self._prefetched: dict[tuple[str, str], list[int]] = {}
        self._held = 0
        self._ahead = source.ahead_gb * 1e9
        self._stop = False
        self._error: BaseException | None = None
        self._threads: list[threading.Thread] = []

        # A read document's bytes a token and a query row, under the run's input - None until one is read
        self.unit_bytes: tuple[float, float] | None = None
        self.documents_read = 0
        # Documents another node read and this one loaded off the shared disk
        self.documents_staged = 0
        self.tokens_read = 0
        self.forward_seconds = 0.0
        self.target_seconds = 0.0

    def entries(self, document_set: "DocumentSet") -> list[dict[str, Any]]:
        """Return the index entries one set selects, as the extraction's manifest would record them.

        Args:
            document_set: The set, whose `source` names the extraction's collection.

        Returns:
            The set's entries, sorted by name.

        Raises:
            ValueError: The extraction writes no such collection, or the set selects nothing in it.
        """
        if document_set.source not in self.index:
            raise ValueError(
                f"{document_set.name} reads collection {document_set.source!r} and {self.source.extraction} "
                f"writes {sorted(self.index)}"
            )
        return selected_entries(self.index[document_set.source], document_set, self.source.extraction)

    def catalog(self, document_sets: list["DocumentSet"]) -> list[Entry]:
        """Return every document of some sets, as the pool's entries, set by set.

        Args:
            document_sets: The sets.

        Returns:
            One entry per document, by name inside a set.
        """
        return [
            Entry((self.root, document_set.source, document_set.split), entry["name"], int(entry["length"]))
            for document_set in document_sets
            for entry in self.entries(document_set)
        ]

    def provenance(self) -> dict[str, Any]:
        """Return what a run's maps record of the teacher they were trained against."""
        selection = None
        if self.selection is not None:
            encoded = json.dumps(self.selection, sort_keys=True).encode()
            selection = hashlib.sha256(encoded).hexdigest()
        return {
            "collection": self.extraction.run.collection or next(iter(self.index)),
            "model": self.extraction.model.name,
            "layers": self.layers,
            "online": {
                "extraction": self.source.extraction,
                "overrides": self.source.overrides,
                "run": asdict(self.extraction),
                "selection_sha256": selection,
            },
        }

    def start(self, devices: list[str], layers: list[int]) -> None:
        """Load one copy of the model per card group and start reading.

        Copies are loaded one after another, because a load patches torch globally while it builds the model.

        Args:
            devices: The card each copy lands on.
            layers: The layers the run trains, which every copy reads.
        """
        from lsa.evaluation import adapter

        model_adapter = adapter.adapter_for(self.extraction.model)
        for device in devices:
            began = time.monotonic()
            model, _ = model_adapter.load(self.extraction.model, device, attention="sdpa")
            if self.extraction.run.prefill:
                model_adapter.prepare_long_prefill(model)
            text_model = model_adapter.text_model(model)
            # A copy told to stop leaves its document at the next decoder layer rather than reading it to the end
            for decoder_layer in text_model.layers:
                decoder_layer.register_forward_pre_hook(self._halt_if_stopped)
            rotary = model_adapter.rotary_layout(model, self.extraction.model, self.layers)
            self.rotary = self.rotary or rotary
            if self.rotary_frequencies is None:
                embedding = model_adapter.rotary_embedding(model)
                scaling = float(getattr(embedding, "attention_scaling", 1.0))
                self.rotary_frequencies = (embedding.inv_freq.detach().float().cpu(), scaling)
            attentions = {index: text_model.layers[index].self_attn for index in layers}
            reading = None
            if self.reads_key_and_query:
                read = key_and_query_input(self.extraction.model.name, rotary)
                if not isinstance(read, HeadKeyAndQuery):
                    raise NotImplementedError(
                        f"{self.extraction.model.name} reads a latent its heads share, which the online teacher does "
                        "not project - an MLA model distils from its cache"
                    )
                reading = read
            self.replicas.append(
                Replica(
                    device=device,
                    text_model=text_model,
                    inputs=input_device(model),
                    readers={
                        index: model_adapter.attention_class(attention, self.extraction.model.indexer, rotary)
                        for index, attention in attentions.items()
                    },
                    reading=reading,
                )
            )
            print(
                f"teacher: {self.extraction.model.name} on {device} in {time.monotonic() - began:.0f} s, "
                f"rotary {rotary}, layers {layers}",
                flush=True,
            )
        for replica in self.replicas:
            thread = threading.Thread(
                target=self._work, args=(replica,), name=f"teacher-{replica.device}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def bases(self, layer: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return None - the per-head key and query the teacher reads acts in no basis of its own."""
        return None

    def submit(self, order: list[tuple[str, str]], wanted: dict[tuple[str, str], list[int]]) -> None:
        """Queue documents for the copies to read, in order.

        Args:
            order: Each document as `(collection, name)`, in the order the pools will take them.
            wanted: The layers each document is read for.
        """
        requests = []
        with self._cond:
            for key in order:
                request = Request(self._requested, key[0], key[1], list(wanted[key]))
                self._requested += 1
                self._requests_of.setdefault(key, deque()).append(request.index)
                self._layers_left[request.index] = set(request.layers)
                requests.append(request)
            if self.staging is not None:
                self.staging.publish(requests)
            self._enqueue(requests)

    def _enqueue(self, requests: list[Request]) -> None:
        """Queue requests for this node's copies, under the lock."""
        for request in requests:
            planned = self.planned[(request.collection, request.name)]
            self._queue.append((request.index, planned, request.layers))
        self._cond.notify_all()

    def serve(self) -> None:
        """Read the trainer's requests off the shared disk as it makes them, until it says it takes nothing more.

        Raises:
            ValueError: The teacher has no staging directory to serve through.
            RuntimeError: A copy failed.
        """
        if self.staging is None:
            raise ValueError("a teacher serves another node's trainer only through a staging directory")
        while not self.staging.finished():
            with self._cond:
                if self._error is not None:
                    raise RuntimeError("a serving copy failed") from self._error
                self._enqueue(self.staging.requests())
                self._cond.wait(timeout=Online.POLL_SECONDS)

    def take(self, entry: Entry, layer: int) -> Document:
        """Return one document's layer once a copy has read it, and let it go from the teacher.

        A copy on another node leaves it on the shared disk instead, and it is loaded from there.

        Args:
            entry: The document, as the pool's catalog holds it.
            layer: The layer.

        Returns:
            The document on the host, under the run's input.

        Raises:
            RuntimeError: A copy failed.
        """
        key = (entry.where[1], entry.name)
        began = time.monotonic()
        said = began
        document = None
        staged = False
        with self._cond:
            index = self._requests_of[key][0]
            self._waiting[(*key, layer)] = self._waiting.get((*key, layer), 0) + 1
            self._cond.notify_all()
        while document is None:
            if self.staging is not None and self.staging.ready(index):
                where = layer_path(self.root, key[0], entry.where[2], entry.name, layer)
                document = Document.from_blob(where, self.staging.take(index, layer), self.indexer)
                staged = True
                with self._cond:
                    if self.unit_bytes is None:
                        self.unit_bytes = unit_bytes(document)
                    self.documents_staged += int(len(self._layers_left[index]) == 1)
                break
            with self._cond:
                if (*key, layer) in self._produced:
                    document = self._produced.pop((*key, layer))
                    self._held -= document.resident_bytes
                    break
                if self._error is not None:
                    raise RuntimeError(f"the teacher failed before reading {entry.name}") from self._error
                if self._stop:
                    raise Stopped(f"the teacher was closed before reading {entry.name}")
                self._cond.wait(timeout=Pool.LOG_SECONDS if self.staging is None else Online.POLL_SECONDS)
                # Every document another copy finishes wakes this one, so it says it is waiting once a minute
                if time.monotonic() - said >= Pool.LOG_SECONDS:
                    said = time.monotonic()
                    print(
                        f"teacher: layer {layer} has waited {time.monotonic() - began:.0f} s for {entry.name} - "
                        f"{self.describe()}",
                        flush=True,
                    )
        with self._cond:
            self._waiting[(*key, layer)] -= 1
            if not self._waiting[(*key, layer)]:
                del self._waiting[(*key, layer)]
            self._layers_left[index].discard(layer)
            if not self._layers_left[index]:
                del self._layers_left[index]
                self._requests_of[key].popleft()
                # Every layer of a document comes from one reader, so the take that empties it frees the staged copy
                if staged and self.staging is not None:
                    self.staging.release(index)
            self._cond.notify_all()
        return document

    def take_item(self, item: Item) -> Document:
        """Return one pool item's document once a copy has read it - what a pool loads with."""
        return self.take(item.entry, item.layer)

    def prefetch(self, entries: list[Entry], layers: list[int]) -> None:
        """Queue documents a later `read_now` takes, longest first, so every copy reads one of them at once.

        Args:
            entries: The documents.
            layers: The layers each is read for, which the `read_now` that takes them must ask for.
        """
        ordered = sorted(entries, key=lambda entry: entry.tokens, reverse=True)
        keys = [(entry.where[1], entry.name) for entry in ordered]
        keys = [key for key in dict.fromkeys(keys) if key not in self._prefetched]
        self._prefetched.update(dict.fromkeys(keys, layers))
        self.submit(keys, dict.fromkeys(keys, layers))

    def read_now(self, entries: list[Entry], layers: list[int]) -> dict[int, list[Document]]:
        """Read some documents for every layer and wait for all of them - a report's sets and a map's init.

        A document `prefetch` queued already is taken rather than queued again.

        Args:
            entries: The documents.
            layers: The layers each is read for.

        Returns:
            Each layer's documents, in the order given, on the host.

        Raises:
            ValueError: A prefetched document was queued for other layers than these.
        """
        keys = [(entry.where[1], entry.name) for entry in entries]
        queued = [key for key in keys if key in self._prefetched]
        if any(self._prefetched[key] != layers for key in queued):
            raise ValueError("a prefetched document is read for other layers than it was queued for")
        for key in queued:
            del self._prefetched[key]
        missing = [key for key in keys if key not in queued]
        self.submit(missing, dict.fromkeys(missing, layers))
        return {layer: [self.take(entry, layer) for entry in entries] for layer in layers}

    def size_of(self, entry: Entry, layer: int) -> int:
        """Return what one document's layer will occupy, scaled from the documents already read.

        Args:
            entry: The document.
            layer: The layer, which every layer's copy shares a size with.

        Returns:
            Bytes - a read document's cost a token and a query row, at this document's length.

        Raises:
            ValueError: Nothing has been read yet to scale from.
        """
        if self.unit_bytes is None:
            raise ValueError("the teacher has read no document yet, so it cannot say what one costs")
        token_bytes, row_bytes = self.unit_bytes
        queries = training_query_count(self.extraction, entry.tokens) + self.extraction.data.queries.tail
        return int(entry.tokens * token_bytes + queries * row_bytes)

    def describe(self) -> str:
        """Return what the teacher has read and holds, for the lines a run prints."""
        forward = self.tokens_read / self.forward_seconds if self.forward_seconds else 0.0
        staged = ""
        if self.staging is not None and not self.staging.serving:
            staged = f", {self.documents_staged} documents taken from {self.staging.nodes - 1} other nodes"
        return (
            f"{self.documents_read} documents, {self.tokens_read / 1e6:.0f}M tokens read by {len(self.replicas)} "
            f"copies at {forward:.0f} tokens/s a copy, targets {self.target_seconds:.0f} s, "
            f"{len(self._queue)} queued, holding {self._held / 1e9:.0f}/{self._ahead / 1e9:.0f} GB{staged}"
        )

    def close(self) -> None:
        """Stop the copies, wait for each to end, and let go of the models and the cards' memory.

        A copy reading a document leaves it at its next layer. A copy still running when the interpreter exits takes
        the process down with it - a thread inside a torch call that releases the GIL is ended by force at exit, and
        that unwinds through C++ that may not throw, so the process aborts after the run has finished.
        """
        with self._cond:
            self._stop = True
            self._queue.clear()
            self._produced.clear()
            self._cond.notify_all()
        if self.staging is not None and not self.staging.serving:
            self.staging.finish()
        for thread in self._threads:
            thread.join()
        self._threads.clear()
        cards = {replica.inputs for replica in self.replicas if replica.inputs.type == "cuda"}
        self.replicas.clear()
        gc.collect()
        for card in cards:
            torch.cuda.synchronize(card)
        if cards:
            torch.cuda.empty_cache()

    def _halt_if_stopped(self, *_hook_args: Any) -> None:
        """Leave the document a copy is reading once the teacher is closed - also a hook before each decoder layer.

        Raises:
            Stopped: The teacher is closed.
        """
        if self._stop:
            raise Stopped

    def _may_read(self) -> bool:
        """Return whether a copy may start a document - room to hold it, or past the bound one a pool waits on."""
        if not self._queue:
            return False
        if self.staging is not None and self.staging.serving:
            return self.staging.has_room()
        if self._held < self._ahead:
            return True
        return self._waited_position() is not None

    def _waited_position(self) -> int | None:
        """Return where in the queue the first document a pool waits on and nobody has under way sits, or None.

        Past the read-ahead bound a copy reads only that document. The queue's head need not be it - across nodes the
        head may be one the other nodes would read once their staging has room - and reading the head instead fills the
        host without bringing the waited document any sooner.
        """
        waited = {wanted[:2] for wanted in self._waiting if not self._under_way(wanted)}
        for position, (_, planned, _) in enumerate(self._queue):
            if (planned.documents.collection, planned.selected.name) in waited:
                return position
        return None

    def _next_request(self) -> tuple[int, Planned, list[int]]:
        """Take the request a copy reads next - the queue's head, or past the bound the document a pool waits on."""
        if (self.staging is not None and self.staging.serving) or self._held < self._ahead:
            return self._queue.popleft()
        position = self._waited_position()
        if position is None:
            return self._queue.popleft()
        request = self._queue[position]
        del self._queue[position]
        return request

    def _under_way(self, wanted: tuple[str, str, int]) -> bool:
        """Return whether a waited-for layer is read or being read - by a copy here, or claimed by one on any node.

        A document another node's copy has claimed is coming through the shared disk, so reading past the bound here
        would not bring it sooner and would fill the host while it waits.

        Args:
            wanted: The document's collection and name, and the layer.
        """
        if wanted in self._produced or wanted[:2] in self._reading:
            return True
        if self.staging is None:
            return False
        return self.staging.claimed(self._requests_of[wanted[:2]][0])

    def _work(self, replica: Replica) -> None:
        """Read the queued documents one at a time on one copy, until stopped or failed.

        Across nodes a copy claims a document before reading it and passes over one another node's copy has, and a
        serving node's copy leaves what it read on the shared disk.
        """
        if replica.inputs.type == "cuda":
            torch.cuda.set_device(replica.inputs)
        # Another node frees room on the shared disk without waking this one
        patience = None if self.staging is None else Online.POLL_SECONDS
        while True:
            with self._cond:
                while not (self._stop or self._error is not None or self._may_read()):
                    self._cond.wait(timeout=patience)
                if self._stop or self._error is not None:
                    return
                index, planned, layers = self._next_request()
            if self.staging is not None and not self.staging.claim(index):
                continue
            key = (planned.documents.collection, planned.selected.name)
            with self._cond:
                self._reading.add(key)
            try:
                blobs = self.read_blobs(replica, planned, layers)
                if self.staging is not None and self.staging.serving:
                    self.staging.put(index, blobs)
                    read = {}
                else:
                    read = self.documents_of(planned, blobs)
            except Stopped:
                return
            except BaseException as failure:
                with self._cond:
                    self._error = failure
                    self._cond.notify_all()
                return
            with self._cond:
                for layer, document in read.items():
                    self._produced[(*key, layer)] = document
                    self._held += document.resident_bytes
                self._reading.discard(key)
                self._cond.notify_all()

    def read_document(self, replica: Replica, planned: Planned, layers: list[int]) -> dict[int, Document]:
        """Read one document through one copy and build each layer's document as a pool would have read it.

        Args:
            replica: The copy.
            planned: The document and its place in the plan, which seeds its query positions.
            layers: The layers to keep.

        Returns:
            Each layer's document on the host, under the run's input.
        """
        return self.documents_of(planned, self.read_blobs(replica, planned, layers))

    def documents_of(self, planned: Planned, blobs: dict[int, dict[str, Any]]) -> dict[int, Document]:
        """Build each layer's document from the blobs a copy read, as a pool would have read it off the cache.

        Args:
            planned: The document.
            blobs: Each layer's blob.

        Returns:
            Each layer's document, where its blob's tensors are.
        """
        read = {}
        for layer, blob in blobs.items():
            selected = planned.selected
            where = layer_path(self.root, planned.documents.collection, selected.split, selected.name, layer)
            read[layer] = Document.from_blob(where, blob, self.indexer)
            if self.unit_bytes is None:
                self.unit_bytes = unit_bytes(read[layer])
        return read

    @torch.no_grad()
    def read_blobs(self, replica: Replica, planned: Planned, layers: list[int]) -> dict[int, dict[str, Any]]:
        """Read one document through one copy into the blob an extraction would write for each layer.

        Args:
            replica: The copy.
            planned: The document and its place in the plan, which seeds its query positions.
            layers: The layers to keep.

        Returns:
            Each layer's blob on the host, under the run's input.

        Raises:
            ValueError: The extraction asks for probe targets, which only the cached extraction keeps.
        """
        if self.extraction.data.queries.targets:
            raise ValueError("queries.targets is kept by `lsa distill targets`, not the online teacher")
        began = time.monotonic()
        selected = planned.selected
        ids = planned.documents.tokens(selected, window_length(self.extraction, selected)).to(replica.inputs)
        positions = query_positions(self.extraction, ids.shape[1], planned.index).to(replica.inputs)
        if self.qsa_indexer is not None:
            blobs, forwarded = self._block_blobs(replica, ids, positions, layers, selected.name)
        elif replica.reading is None:
            blobs = self._x_blobs(replica, ids, positions, layers, selected.name)
            forwarded = began
        else:
            views = TeacherViews(
                {layer: replica.readers[layer] for layer in layers}, positions, replica.reading
            )
            hooks = views.hook(replica.text_model)
            try:
                read_through(replica.text_model, ids, self.extraction.run.prefill)
            finally:
                for hook in hooks:
                    hook.remove()
            forwarded = time.monotonic()
            blobs = {}
            for layer in layers:
                self._halt_if_stopped()
                q, k, key, summed_query = views.take(layer)
                blob = targets_blob(
                    replica.readers[layer], q, k.to(replica.inputs), positions, self.extraction, selected.name
                )
                blobs[layer] = {
                    **blob,
                    Cache.KEY_AND_QUERY_FIELDS[0]: key,
                    Cache.KEY_AND_QUERY_FIELDS[1]: summed_query.to(Cache.HOST),
                }
                del q, k
        # The card's peak since the copy started, which says how much room the longest document leaves. No cache is
        # emptied here: a copy's thread that empties it releases every card's, under the other copies' kernels
        peak = ""
        if replica.inputs.type == "cuda":
            peak = f", card peak {torch.cuda.max_memory_allocated(replica.inputs) / 2**30:.1f} GiB"
        finished = time.monotonic()
        with self._cond:
            self.documents_read += 1
            self.tokens_read += ids.shape[1]
            self.forward_seconds += forwarded - began
            self.target_seconds += finished - forwarded
        print(
            f"teacher: read {planned.documents.collection}/{selected.split}/{selected.name} at {ids.shape[1]} tokens "
            f"on {replica.device} - forward {forwarded - began:.0f} s, targets {finished - forwarded:.0f} s for "
            f"{len(layers)} layers{peak}",
            flush=True,
        )
        return blobs

    def _block_blobs(
        self, replica: Replica, ids: torch.Tensor, positions: torch.Tensor, layers: list[int], name: str
    ) -> tuple[dict[int, dict[str, Any]], float]:
        """Read one document for a QSA indexer - each layer's blocks and sampled rows of `x`, and QSA's target.

        Args:
            replica: The copy.
            ids: Token ids [1, T] int64, on the copy's card.
            positions: Absolute query positions [Q] int64.
            layers: The layers to keep.
            name: The document's name.

        Returns:
            blobs: Each layer's blob, its blocks and rows in place of `x`.
            forwarded: When the forward finished, on the monotonic clock.

        Raises:
            ValueError: The teacher was built for no QSA indexer.
        """
        if self.qsa_indexer is None:
            raise ValueError("the teacher reads blocks for a QSA indexer, and was built for none")
        block_size = self.qsa_indexer.block_size
        views = BlockViews(
            {layer: replica.readers[layer] for layer in layers}, positions, block_size, replica.reading
        )
        fields = Cache.QSA_KEY_AND_QUERY_FIELDS if replica.reading is not None else Cache.QSA_FIELDS
        hooks = views.hook(replica.text_model)
        try:
            read_through(replica.text_model, ids, self.extraction.run.prefill)
        finally:
            for hook in hooks:
                hook.remove()
        forwarded = time.monotonic()
        blobs = {}
        for layer in layers:
            q, k, block_x, query_x = views.take(layer)
            blob = targets_blob(
                replica.readers[layer],
                q,
                k.to(replica.inputs),
                positions,
                self.extraction,
                name,
                block_size=block_size,
            )
            blobs[layer] = {**blob, fields[0]: block_x, fields[1]: query_x}
            del q, k
        return blobs, forwarded

    def _x_blobs(
        self, replica: Replica, ids: torch.Tensor, positions: torch.Tensor, layers: list[int], name: str
    ) -> dict[int, dict[str, Any]]:
        """Read one document as an extraction does, each layer's input kept whole - a run on `x` itself.

        Args:
            replica: The copy.
            ids: Token ids [1, T] int64, on the copy's card.
            positions: Absolute query positions [Q] int64.
            layers: The layers to keep.
            name: The document's name.

        Returns:
            Each layer's blob, `x` included, as the extraction writes it.
        """
        captured = capture_layer_inputs(replica.text_model, layers, ids, self.extraction.run.prefill)
        blobs = {}
        for layer in layers:
            self._halt_if_stopped()
            hidden, position_embeddings = captured.pop(layer)
            blobs[layer] = layer_blob(
                replica.readers[layer], hidden, position_embeddings, positions, self.extraction, name
            )
            del hidden, position_embeddings
        return blobs


def unit_bytes(document: Document) -> tuple[float, float]:
    """Return what a document costs a token and a query row - its inputs over its keys, its targets over its rows.

    Args:
        document: A document, under the run's input.

    Returns:
        token_bytes: Bytes a position, of everything held for every position.
        row_bytes: Bytes a query row, of everything held for every sampled row.
    """
    rows = len(document.query_positions)
    row_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in document.tensors()
        if tensor.ndim and tensor.shape[0] == rows and tensor.shape[0] != document.key_count
    )
    return (document.resident_bytes - row_bytes) / document.key_count, row_bytes / rows


def teacher_devices(trainer_device: str | None) -> list[str]:
    """Return where each copy of the teacher goes - every card the trainer's leaves, or the trainer's own.

    Args:
        trainer_device: The card the maps train on, or None on a node the trainer is not on.

    Returns:
        One card per copy, the trainer's own when the job has no other.
    """
    if trainer_device is None:
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    trainer = torch.device(trainer_device)
    if trainer.type != "cuda":
        return [trainer_device]
    left = [f"cuda:{index}" for index in range(torch.cuda.device_count()) if index != (trainer.index or 0)]
    return left or [trainer_device]
