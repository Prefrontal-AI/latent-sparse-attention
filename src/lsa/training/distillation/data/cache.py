"""The targets - what one extracted (document, layer) holds, and how a run reads it.

The model is not in the loop.
Everything the objective and the metrics need is extracted once by [teacher](teacher.py) and read back here.

| field | shape | what it is |
|---|---|---|
| `x` | `[T, model_dim]` bf16 | the attention's input at every position - the residual streams mixed and normed |
| `query_positions` | `[Q]` int32 | where the sampled queries sit |
| `top_key_ids` | `[Q, N]` int32 | the teacher's top-`N` visible long-range keys by head-summed mass |
| `top_key_mass` | `[Q, N]` fp16 | their masses, summed over heads |
| `peak_key_id` | `[Q, H]` int32 | per head, the single strongest visible long-range key |
| `query_kind` | `[Q]` int8 | a probe extraction only: which of `meta.query_kinds` - `question`, `answer`, `control` - a query is |
| `target_key_ids` | `[Q, N]` int32 | a probe extraction only: the keys a query is meant to find, `-1` padded and all `-1` where it has none |
| `line_key_ids` | `[Q, L]` int32 | a probe extraction on a needle task only: the keys of the needle's whole line, `-1` padded |
| `attention_keys` | `[T, kv_head_count, head_dim]` bf16 | `queries.attention` only: the model's own rotated keys, with `meta.attention_scale` |
| `attention_queries` | `[Q, query_head_count, head_dim]` bf16 | `queries.attention` only: each head's rotated query at the sampled positions |
| `key` | `[T, key_dim]` bf16 | under `key_and_query`, in place of `x`: GLM's key latent at every position |
| `summed_query` | `[Q, key_dim]` fp32 | under `key_and_query`: every head's folded query at the sampled rows, summed |
| `head_queries` | `[Q, query_head_count, key_dim]` bf16 | under `key_and_query` on GLM: each head's folded query, the `dense_full` target's logits are these dotted with `key`, scaled by `meta.attention_scale` or the model's own |

The index keys and index queries are not stored, because they are `x` projected by the map being trained.
An index query is the projection of `x` at a query position, so the sampled queries need no tensor of their own.
Under `indexer.input: key_and_query` they are projected from GLM's key latent and the heads' queries summed instead,
which the layer's own weights read off `x` once a document is on the card - see [native](native.py).

Visible means what the read may select - past the sinks and outside the window.
The cluster block is the window, so every such key sits in a block that closed before the query's own and
the block adds nothing to the rule.
The same rule that decides what a selection may return decides what the target is defined over.

An index-vector cache is read too: no `x`, but its index keys `rownorm(latent @ L)` and index queries
`rownorm(summed_query @ L^-T)` (`Cache.INDEX_FIELDS`), written under the closed-block rule.
A key is visible to a query, and carries target mass, once the cluster block of `closed_block` keys it sits in has
closed before the query's own, and a query row with no closed block behind it is not trained on.
The file names its block, and `closed_block` can be laid over an `x` cache's rule as well.

The first `train_queries` rows are the training positions and the rest are the held-out tail, and the two
never mix.
A probe extraction (`queries.targets`) has no training positions to draw, so its rows are the queries a prepared prompt
names for itself, in position order, with the last `tail` the question - and `train_queries` is every row before those.

A layer's share of a cache is some 3 TiB, so no run holds it: [pool](pool.py) streams documents through a
bounded part of host memory and a card, and this module is what one of them is once it has been read.
"""

import copy
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.baselines import qsa
from lsa.core import RotaryLayout
from lsa.sglang.settings.design import KEY_AND_QUERY, IndexerConfig
from lsa.training.constants import Targets
from lsa.training.data.targets import read_manifest
from lsa.training.distillation.constants import Cache, Target
from lsa.training.distillation.data.teacher_attention import full_attention_target, latent_attention_scale

# The config module reaches omegaconf and hydra, and this name is only ever an annotation
# => importing it here would defeat the extras guard in `run.main`
if TYPE_CHECKING:
    from lsa.training.distillation.config import DocumentSet


def layer_path(root: str | Path, collection: str, split: str, document: str, layer: int) -> Path:
    """Return where one (document, layer) lives.

    The split is a directory rather than a field, so a set that reads the training side cannot reach an
    evaluation document by any spelling of its name.

    Args:
        root: The cache root.
        collection: Directory under the root - `natural`, or a RULER task.
        split: `train` or `eval`.
        document: Document name inside the split.
        layer: Layer index.

    Returns:
        The file path, which may not exist.
    """
    return Path(root) / collection / split / f"{document}{Cache.LAYER_TAG}{layer}{Cache.SUFFIX}"


def recorded_rotary(manifest: dict[str, Any]) -> RotaryLayout:
    """Return where the rotation sits in the heads, as an extraction recorded it in its manifest.

    Args:
        manifest: A collection's manifest.

    Returns:
        The layout - none for a manifest that records none.
    """
    recorded = manifest.get("rotary")
    if not recorded:
        return RotaryLayout()
    return RotaryLayout(end=recorded["end"], dim=int(recorded["dim"]))


def native_path(path: str | Path) -> Path:
    """Return where the checkpoint's own selection of a (document, layer) is kept beside its file.

    Args:
        path: The layer file.

    Returns:
        `<document>_layer<index>_native.pt` next to it, which may not exist.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}{Cache.NATIVE_TAG}{path.suffix}")


def set_root(root: str | Path, document_set: "DocumentSet") -> Path:
    """Return the cache a set is read from - its own root when it names one, else the run's.

    Args:
        root: The run's cache root.
        document_set: The set.

    Returns:
        The root the set's collection sits under.
    """
    return Path(document_set.root or root)


def listing(root: str | Path, document_set: "DocumentSet") -> list[dict[str, Any]]:
    """Return the manifest entries one set holds, sorted by name.

    The collection's manifest is the index rather than the directory listing, because a name is an
    identifier and the things a set selects on - the domain, the side, the document it stands for - are
    recorded beside it rather than encoded in it.

    Args:
        root: The cache root.
        document_set: The set to resolve.

    Returns:
        The set's entries, each with at least `name` and `length`.

    Raises:
        FileNotFoundError: The collection has no manifest, so nothing has been extracted into it.
        ValueError: The set selects no document, which would train or score on nothing.
    """
    cache_root = set_root(root, document_set)
    manifest = cache_root / document_set.source / Targets.MANIFEST
    entries = read_manifest(cache_root, document_set.source)[Targets.DOCUMENTS]
    return selected_entries(entries, document_set, str(manifest))


def selected_entries(
    entries: list[dict[str, Any]], document_set: "DocumentSet", index: str
) -> list[dict[str, Any]]:
    """Return the entries of a collection's document index one set selects, sorted by name.

    Args:
        entries: The collection's entries, as its manifest records them.
        document_set: The set to resolve.
        index: Where the entries came from, for the message.

    Returns:
        The set's entries.

    Raises:
        ValueError: The set selects no document, which would train or score on nothing.
    """
    wanted = set(document_set.sources) if document_set.sources else None
    lengths = set(document_set.lengths) if document_set.lengths else None
    chosen = sorted(
        (
            entry
            for entry in entries
            if entry["split"] == document_set.split
            and (wanted is None or entry["source"] in wanted)
            and (lengths is None or entry["length"] in lengths)
        ),
        key=lambda entry: entry["name"],
    )
    if not chosen:
        raise ValueError(
            f"{document_set.name} selects no document - {index} holds no {document_set.split} "
            f"documents for sources {document_set.sources or 'any'} at lengths {document_set.lengths or 'any'}"
        )
    return chosen


def documents(root: str | Path, document_set: "DocumentSet") -> list[str]:
    """Return the document names one set holds, in a stable order.

    Args:
        root: The cache root.
        document_set: The set to resolve.

    Returns:
        Document names, sorted.
    """
    return [entry["name"] for entry in listing(root, document_set)]


def cached_layers(root: str | Path, document_set: "DocumentSet") -> list[int]:
    """Return the layers a collection holds, in order.

    Args:
        root: The cache root.
        document_set: The set to look under.

    Returns:
        Layer indices, sorted.
    """
    under = set_root(root, document_set) / document_set.source / document_set.split
    return sorted(
        {
            int(path.stem.rsplit(Cache.LAYER_TAG, 1)[1])
            for path in under.glob(Cache.GLOB)
            if not path.stem.endswith(Cache.NATIVE_TAG)
        }
    )


def write(path: Path, blob: dict[str, Any]) -> None:
    """Write one cached (document, layer) atomically.

    Two extraction jobs may cover overlapping documents, and a half-written file read by a trainer is a
    failure a long run finds late.

    Args:
        path: Destination file, whose parents are created.
        blob: The tensors and the meta block.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(blob, scratch)
    os.replace(scratch, path)


class Document:
    """One cached (document, layer), on a device, with `x` as the extraction wrote it or the key and query read off it."""

    def __init__(
        self, path: Path, indexer: IndexerConfig, device: str | torch.device = "cpu", native: bool = False
    ) -> None:
        """Read one cached (document, layer).

        Args:
            path: The cache file.
            indexer: The design's knobs, whose visibility rule must be the one the cache was written under.
            device: Where the tensors land.
            native: Read the checkpoint's own selection too - from the file when the extraction kept it, else
                from the sidecar beside it - and refuse a document that has neither.

        Raises:
            ValueError: The file is missing a field, was written under a different visibility rule, or holds the key
                and query for a run on `x`.
            FileNotFoundError: `native` is asked for and the document has no selection anywhere.
        """
        blob = torch.load(path, map_location=device, weights_only=False)
        if native and Cache.NATIVE not in blob:
            beside = native_path(path)
            if not beside.exists():
                raise FileNotFoundError(
                    f"{path} holds no native selection and {beside} does not exist - "
                    "compute it with `lsa distill native-picks`"
                )
            blob[Cache.NATIVE] = torch.load(beside, map_location=device, weights_only=False)[Cache.NATIVE]
        self._take(path, blob, indexer)

    @classmethod
    def from_blob(cls, path: Path, blob: dict[str, Any], indexer: IndexerConfig) -> "Document":
        """Build one (document, layer) from the blob an extraction would have written, without the disk.

        The online teacher hands its targets to the pool this way, so a document it builds is the document a read of
        the same blob from `path` gives - or, holding the key and query in place of `x`, the one the run's reader makes
        of it.

        Args:
            path: Where the extraction would have written it, which names the document and its collection.
            blob: The tensors and the meta block, as `layer_blob` builds them.
            indexer: The design's knobs, whose visibility rule must be the one the blob was built under.

        Returns:
            The document, on whatever device the blob's tensors are.
        """
        document = cls.__new__(cls)
        document._take(path, blob, indexer)
        return document

    def _take(self, path: Path, blob: dict[str, Any], indexer: IndexerConfig) -> None:
        """Fill this document from one (document, layer)'s blob.

        Args:
            path: The cache file, or where it would be, which names the document and its collection.
            blob: The tensors and the meta block.
            indexer: The design's knobs, whose visibility rule must be the one the blob was written under.

        Raises:
            ValueError: The blob is missing a field, was written under a different visibility rule, or holds the key
                and query for a run on `x`.
        """
        tensors = {name: value for name, value in blob.items() if name != Cache.META}
        indexed = Cache.X not in tensors and all(name in tensors for name in Cache.INDEX_FIELDS)
        # The online teacher's blob holds the key and query read already, in place of `x`
        read = Cache.X not in tensors and all(name in tensors for name in Cache.KEY_AND_QUERY_FIELDS)
        # Or, for a QSA indexer, each closed block's mean input and the sampled rows' - `x`, or the key and query
        pooled_fields = (
            Cache.QSA_KEY_AND_QUERY_FIELDS
            if Cache.QSA_KEY_AND_QUERY_FIELDS[0] in tensors
            else Cache.QSA_FIELDS
        )
        pooled = Cache.X not in tensors and all(name in tensors for name in pooled_fields)
        fields = Cache.FIELDS
        if indexed:
            fields = (*Cache.INDEX_FIELDS, *Cache.FIELDS[1:])
        elif read:
            fields = (*Cache.KEY_AND_QUERY_FIELDS, *Cache.FIELDS[1:])
        elif pooled:
            fields = (*pooled_fields, *Cache.FIELDS[1:])
        missing = [name for name in fields if name not in tensors]
        if missing:
            raise ValueError(f"{path} is missing {missing} - it holds {sorted(tensors)}")
        if read and indexer.input != KEY_AND_QUERY:
            raise ValueError(
                f"{path} holds the key and query in place of x, and the run reads {indexer.input!r} - "
                "a run on x needs a cache of x"
            )

        self.meta: dict[str, Any] = dict(blob[Cache.META])
        self.name = path.stem.rsplit(Cache.LAYER_TAG, 1)[0]
        self.collection = path.parent.name
        self.layer = int(self.meta["layer"])
        # Keys visible only once their cluster block of this many has closed before the query's own, or None for this
        # cache's rule alone
        self.closed_block: int | None = None
        if indexed:
            _check_index_visibility(path, self.meta, indexer)
            self.closed_block = int(self.meta[Cache.INDEX_BLOCK])
        else:
            _check_visibility(path, self.meta, indexer)

        # [T, model_dim] bf16 the attention's input at every position, None once the key and query are read off it
        self.x: torch.Tensor | None = tensors.get(Cache.X)
        # An index-vector cache's index keys [T, key_dim] and index queries [Q, key_dim] fp16 unit rows in the latent
        # bases, None otherwise and once the key and query are read off them
        self.index_keys: torch.Tensor | None = tensors.get(Cache.INDEX_FIELDS[0])
        self.index_queries: torch.Tensor | None = tensors.get(Cache.INDEX_FIELDS[1])
        # Under `key_and_query`, the key latents [T, key_dim] bf16 and each query row's heads' queries summed
        # [Q, key_dim] fp32 - see `with_key_and_query`
        self.key: torch.Tensor | None = tensors.get(Cache.KEY_AND_QUERY_FIELDS[0])
        self.summed_query: torch.Tensor | None = tensors.get(Cache.KEY_AND_QUERY_FIELDS[1])
        # Each head's folded query at the sampled rows [Q, heads, key_dim] bf16, which a cache extracted under the key and
        # query keeps for a served read - None otherwise
        self.head_queries: torch.Tensor | None = tensors.get(Cache.HEAD_QUERIES)
        # Under `queries.attention`, the model's own rotated keys [T, kv_head_count, head_dim] bf16 and the sampled rows'
        # queries [Q, query_head_count, head_dim] bf16, with the layer's softmax scale - None otherwise
        self.attention_keys: torch.Tensor | None = tensors.get(Cache.ATTENTION_KEYS)
        self.attention_queries: torch.Tensor | None = tensors.get(Cache.ATTENTION_QUERIES)
        self.attention_scale: float | None = self.meta.get(Cache.ATTENTION_SCALE)
        # For a QSA indexer, each closed block's mean `x` [T // block_size, model_dim] bf16 and `x` at the sampled rows
        # [Q, model_dim] bf16, with the block they are pooled in - None otherwise
        self.block_x: torch.Tensor | None = tensors.get(Cache.QSA_FIELDS[0])
        self.query_x: torch.Tensor | None = tensors.get(Cache.QSA_FIELDS[1])
        # Or, on the key and query, each closed block's mean key input [T // block_size, key_dim] bf16, its rows' summed
        # queries in `summed_query`
        self.block_key: torch.Tensor | None = tensors.get(Cache.QSA_KEY_AND_QUERY_FIELDS[0])
        self.block_size: int | None = self.meta.get(Cache.BLOCK_SIZE)
        # [Q] int64 absolute query positions
        self.query_positions = tensors["query_positions"].long()
        # [Q, N] int64 the teacher's strongest visible long-range keys
        self.top_key_ids = tensors["top_key_ids"].long()
        # [Q, N] fp32 their head-summed mass
        self.top_key_mass = tensors["top_key_mass"].float()
        # [Q, H] int64 the single strongest visible key of each head
        self.peak_key_id = tensors["peak_key_id"].long()
        # [Q, W] int32 the checkpoint's own sparse selection at each position, -1 where empty, if kept
        self.native_key_ids: torch.Tensor | None = tensors.get(Cache.NATIVE)
        # [Q, W] fp16 the head-summed attention the served model spends on each of those keys, if kept
        self.native_key_mass: torch.Tensor | None = tensors.get(Cache.NATIVE_MASS)
        # A probe extraction's [Q] int8 query kinds, [Q, N] int32 targets and [Q, L] int32 needle line, -1 padded
        self.query_kind: torch.Tensor | None = tensors.get(Cache.QUERY_KIND)
        self.target_key_ids: torch.Tensor | None = tensors.get(Cache.TARGET_KEY_IDS)
        self.line_key_ids: torch.Tensor | None = tensors.get(Cache.LINE_KEY_IDS)

        if self.native_key_ids is not None and len(self.native_key_ids) != len(self.query_positions):
            raise ValueError(
                f"{path} has {len(self.query_positions)} query rows and a native selection of "
                f"{len(self.native_key_ids)} - it was computed for other positions"
            )

        inputs = Cache.INDEX_FIELDS[0] if indexed else Cache.KEY_AND_QUERY_FIELDS[0] if read else Cache.X
        if pooled:
            self.key_count, self.model_dim = (
                int(self.meta[Cache.LENGTH]),
                tensors[pooled_fields[0]].shape[1],
            )
        else:
            self.key_count, self.model_dim = tensors[inputs].shape
        self.head_count = self.peak_key_id.shape[1]
        self.train_query_count = _train_query_count(path, self.meta, len(self.query_positions))
        self.sink_size = indexer.sink_size
        self.window_size = indexer.window_size

    @property
    def device(self) -> torch.device:
        """Where this document's tensors are"""
        return self.query_positions.device

    @property
    def input_dim(self) -> int:
        """Dims of what the projections read - `model_dim` for `x`, the key's dim under the key and query"""
        return int(self.key_inputs().shape[1])

    @property
    def resident_bytes(self) -> int:
        """What this document occupies wherever it is.

        `x` is `[T, model_dim]` and everything else is indexed by the sampled rows first, so at 1M tokens it
        is 8.7 GB of 8.9 - which is why a budget can be counted in whole documents without weighing each field.
        Under the key and query the key latents take `x`'s place, an eighth of it on GLM.
        """
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def to(self, device: str | torch.device) -> "Document":
        """Return this document with its tensors on `device`, leaving this one where it is.

        Every derived field is a tensor or a plain int, so this copies the object and moves what is a
        tensor. Torch returns the same tensor when it is already on the device asked for, which makes
        staging a document that never left the card free rather than a copy of it.

        Args:
            device: Where the returned document's tensors land.

        Returns:
            The document, on `device`.
        """
        return self._with_tensors(lambda tensor: tensor.to(device))

    def tensors(self) -> list[torch.Tensor]:
        """Return every tensor this document holds."""
        return [value for value in vars(self).values() if isinstance(value, torch.Tensor)]

    def _with_tensors(self, move: Callable[[torch.Tensor], torch.Tensor]) -> "Document":
        """Return a copy of this document with `move` applied to every tensor it holds.

        Args:
            move: What to do to each tensor.

        Returns:
            The copy.
        """
        moved = copy.copy(self)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(moved, name, move(value))
        return moved

    def with_key_and_query(self, key: torch.Tensor, summed_query: torch.Tensor) -> "Document":
        """Return this document reading the key and the heads' summed queries in place of `x`, which it lets go.

        Args:
            key: The key latent at every position [T, key_dim] bf16.
            summed_query: Every head's query at each query row, folded through its key map and summed [Q, key_dim] fp32.

        Returns:
            The document under `indexer.input: key_and_query`, on the device its new tensors are on.
        """
        read = copy.copy(self)
        read.x = None
        read.index_keys = None
        read.index_queries = None
        read.key = key
        read.summed_query = summed_query
        return read

    def with_closed_block(self, closed_block: int | None) -> "Document":
        """Return this document under the closed-block rule on top of its own, or as it is for None.

        Args:
            closed_block: Keys a cluster block holds, or None to leave the rule the cache's.

        Returns:
            The document, its visible keys and its training rows those of the stricter rule.

        Raises:
            ValueError: The cache was written under a closed block of another size, whose target mass cannot be re-cut.
        """
        if closed_block is None:
            return self
        if self.closed_block is not None and self.closed_block != closed_block:
            raise ValueError(
                f"{self.collection}/{self.name} layer {self.layer} was written under a closed block of "
                f"{self.closed_block} keys - objective.closed_block asks for {closed_block}"
            )
        read = copy.copy(self)
        read.closed_block = closed_block
        return read

    def key_inputs(self) -> torch.Tensor:
        """Return what the key projection reads at every position - `x`, the key latents, or a QSA indexer's blocks.

        Returns:
            [T, input_dim] - `x` bf16, or the key latents bf16 - or each closed block's mean `x` [T // block_size,
            model_dim] bf16 for a QSA indexer.

        Raises:
            ValueError: The document holds neither.
        """
        inputs = (
            self.key if self.key is not None else self.block_x if self.block_x is not None else self.block_key
        )
        if inputs is None:
            inputs = self.x
        if inputs is None and self.index_keys is not None:
            raise ValueError(
                f"{self.collection}/{self.name} layer {self.layer} holds index vectors - read it on indexer.input: key_and_query"
            )
        if inputs is None:
            raise ValueError(f"{self.collection}/{self.name} layer {self.layer} holds neither x nor the key")
        return inputs

    def query_inputs(self, rows: torch.Tensor) -> torch.Tensor:
        """Return what the query projection reads at some query rows - `x` at their positions, or their summed queries.

        Args:
            rows: Query rows [R] int64.

        Returns:
            [R, input_dim] - `x` at the rows' positions bf16, or the heads' queries summed fp32.
        """
        rows = rows.to(self.device)
        if self.summed_query is not None:
            return self.summed_query[rows]
        if self.query_x is not None:
            return self.query_x[rows]
        return self.key_inputs()[self.query_positions[rows]]

    def train_rows(self) -> torch.Tensor:
        """Return the query rows a run may train on [n_train] int64 - under a closed block, those with a block closed behind them."""
        rows = torch.arange(self.train_query_count, device=self.device)
        if self.closed_block is None:
            return rows
        return rows[self.query_positions[rows] >= self.sink_size + self.closed_block]

    def eval_rows(self) -> torch.Tensor:
        """Return the held-out tail query rows [Q - n_train] int64."""
        return torch.arange(self.train_query_count, len(self.query_positions), device=self.device)

    def visible_keys(self, rows: torch.Tensor) -> torch.Tensor:
        """Return which keys each query row may select.

        A key is readable when it is past the sinks and at least a window behind the query, which is the
        set a selection returns from and so the set the target is defined over. Under a QSA indexer it is any key
        of a block the query has closed.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Readable-key mask [R, T] bool.
        """
        keys = torch.arange(self.key_count, device=self.device)
        positions = self.query_positions[rows]
        if self.block_size is not None:
            return keys[None] < qsa.closed_blocks(positions, self.block_size)[:, None] * self.block_size
        if self.closed_block is not None:
            return self.closed_block_keys(keys, positions, self.closed_block)
        behind = positions[:, None] - keys[None]
        return (keys[None] >= self.sink_size) & (behind >= self.window_size)

    def selectable(self, key_ids: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Return which of some keys each row may select, under the rule `visible_keys` applies.

        Args:
            key_ids: Keys of each row [R, N] int64.
            rows: Query rows [R] int64.

        Returns:
            Selectable mask [R, N] bool.
        """
        positions = self.query_positions[rows][:, None]
        if self.block_size is not None:
            return key_ids < qsa.closed_blocks(positions, self.block_size) * self.block_size
        if self.closed_block is not None:
            return self.closed_block_keys(key_ids, positions[:, 0], self.closed_block)
        return (key_ids >= self.sink_size) & (positions - key_ids >= self.window_size)

    def visible_blocks(self, rows: torch.Tensor) -> torch.Tensor:
        """Return which blocks each query row may select under a QSA indexer - those its position has closed.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Selectable-block mask [R, T // block_size] bool.
        """
        block_size = self.qsa_block_size()
        return qsa.visible_blocks(self.query_positions[rows], self.key_count // block_size, block_size)

    def block_target(self, rows: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        """Return the teacher's attention over blocks - each block's strongest key, renormalised over what a row may select.

        The max rather than the sum keeps a block's one strong key from being diluted by its weak ones, which is QSA's
        own target. The teacher's keys are cut to its strongest, and a block's max is its strongest key's mass
        wherever that key is among them, so a cut drops whole blocks and never misprices one it keeps.

        Args:
            rows: Query rows [R] int64.
            depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.

        Returns:
            Teacher distribution over blocks [R, T // block_size] fp32, zero off the selectable set.
        """
        block_size = self.qsa_block_size()
        ids, values = self.top_key_ids[rows], self.top_key_mass[rows]
        if depth is not None:
            ids, values = ids[:, :depth], values[:, :depth]
        block_count = self.key_count // block_size
        # One column past the last block absorbs the keys of the open block, which carry no mass
        mass = torch.zeros(len(rows), block_count + 1, device=self.device)
        mass.scatter_reduce_(1, (ids // block_size).clamp_max(block_count), values, "amax")
        mass = mass[:, :block_count] * self.visible_blocks(rows)
        return mass / mass.sum(-1, keepdim=True).clamp_min(Cache.MASS_EPS)

    def qsa_block_size(self) -> int:
        """Return the block a QSA document's keys are pooled in.

        Raises:
            ValueError: The document was read for an LSA map.
        """
        if self.block_size is None:
            raise ValueError(
                f"{self.collection}/{self.name} layer {self.layer} was not read for a QSA indexer"
            )
        return int(self.block_size)

    def closed_block_keys(
        self, keys: torch.Tensor, positions: torch.Tensor, closed_block: int
    ) -> torch.Tensor:
        """Return the keys the closed-block rule leaves readable - past the sinks, in a cluster block that closed before the query's own.

        The window is not subtracted: the cache's target already excludes it, so the few window keys a query's last
        closed block holds carry no mass and only sit in the denominator.

        Args:
            keys: Key ids [T] or [R, N] int64.
            positions: The rows' positions [R] int64.
            closed_block: Keys a cluster block holds.

        Returns:
            Readable-key mask [R, T] or [R, N] bool.
        """
        closed_blocks = (self.key_count - self.sink_size) // closed_block
        key_block = (keys - self.sink_size).clamp_min(0) // closed_block
        key_block = torch.where((keys >= self.sink_size) & (key_block < closed_blocks), key_block, -1)
        query_block = (positions - self.sink_size).clamp_min(0) // closed_block
        key_block = key_block[None] if key_block.dim() == 1 else key_block
        return (key_block >= 0) & (key_block < query_block[:, None])

    def key_target(
        self, rows: torch.Tensor, depth: int | None = None, target: str = Target.DENSE
    ) -> torch.Tensor:
        """Return the teacher's attention over keys, renormalized over what those rows may read.

        The cache holds the teacher's strongest keys in rank order, so `depth` is a prefix of that ranking
        and the mass is renormalised over what is kept. A selector is scored on its top 16 and its top 64,
        and the tail of a 256-key target is mass it never has to rank correctly.

        The native targets read the checkpoint's own sparse selection instead of the dense attention it never
        runs. `dense_on_native` keeps the dense keys the selector also picked, `native_attention` is the
        attention the served model spends over everything it picked, and `native_selection` weighs each pick
        equally. Whatever the target, what a row may not read carries nothing.

        `dense_full` is none of those: it is the dense attention over every readable key, computed here from the key latent
        and each head's query rather than read from the cached ranking, so there is no depth to cut it at.

        Args:
            rows: Query rows [R] int64.
            depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.
            target: One of `Target.ALL`.

        Returns:
            Teacher distribution over keys [R, T] fp32, zero off the readable set.

        Raises:
            ValueError: `target` is not one the cache can build, or `depth` is given with `dense_full`.
        """
        if target not in Target.ALL:
            raise ValueError(f"target must be one of {Target.ALL} - got {target!r}")
        if target == Target.DENSE_FULL:
            if depth is not None:
                raise ValueError(f"target {target!r} is not truncated - got depth {depth}")
            return self.full_target(rows)
        if target == Target.NATIVE_SELECTION:
            mass = self.native_members(rows).float()
        elif target == Target.NATIVE_ATTENTION:
            mass = self.native_attention(rows)
        else:
            mass = self._ranked_mass(rows, depth, target)
        mass = mass * self.visible_keys(rows)
        return mass / mass.sum(-1, keepdim=True).clamp_min(Cache.MASS_EPS)

    def key_share(
        self, rows: torch.Tensor, depth: int | None = None, target: str = Target.DENSE
    ) -> torch.Tensor:
        """Return the share of the teacher's whole attention each kept key holds, which the kept keys hold only in part.

        Every head softmaxes over the visible keys before the masses are summed, so the cached mass sums to
        the query head count over them and a key's share is its mass divided by that count.
        Nothing is renormalised, so a row's shares sum to what its kept keys hold of the whole, at most one.

        Args:
            rows: Query rows [R] int64.
            depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.
            target: One of `Target.RANKED`.

        Returns:
            Share of the whole attention [R, T] fp32, zero off the kept and visible keys.

        Raises:
            ValueError: `target` is not read off the dense ranking.
        """
        if target not in Target.RANKED:
            raise ValueError(f"key_share reads the dense ranking, one of {Target.RANKED} - got {target!r}")
        return self._ranked_mass(rows, depth, target) * self.visible_keys(rows) / self.head_count

    def _ranked_mass(self, rows: torch.Tensor, depth: int | None, target: str) -> torch.Tensor:
        """Return the head-summed mass of the teacher's kept keys, off the dense ranking.

        Args:
            rows: Query rows [R] int64.
            depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.
            target: One of `Target.RANKED`.

        Returns:
            Mass over keys [R, T] fp32, zero off the kept keys.
        """
        ids, values = self.top_key_ids[rows], self.top_key_mass[rows]
        if depth is not None:
            ids, values = ids[:, :depth], values[:, :depth]
        mass = torch.zeros(len(rows), self.key_count, device=self.device)
        mass.scatter_add_(1, ids, values)
        if target == Target.DENSE_ON_NATIVE:
            mass = mass * self.native_members(rows)
        return mass

    def full_target(self, rows: torch.Tensor) -> torch.Tensor:
        """Return the teacher's attention over every readable key, from the key latent and each head's query.

        Every head's softmax over the keys a row may read, averaged over heads - see `full_attention_target`.
        The document holds the key latent and the heads' queries, `[T, key_dim]` and `[Q, H, key_dim]`, in place of
        the `[Q, T]` that would be, and the step computes the rows it trains on.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Teacher distribution over keys [R, T] fp32, zero off the readable set and all zero for a row that
            may read nothing.

        Raises:
            ValueError: The document is read for a QSA indexer, holds no key latent or heads' queries, or comes from a model
                whose heads rotate.
        """
        where = f"{self.collection}/{self.name} layer {self.layer}"
        if self.block_size is not None:
            raise ValueError(
                f"{where} is read for a QSA indexer, whose target is over blocks, not the dense one"
            )
        key, head_queries = self.key, self.head_queries
        if key is None or head_queries is None:
            held = {Cache.KEY_AND_QUERY_FIELDS[0]: key, Cache.HEAD_QUERIES: head_queries}
            missing = [name for name, value in held.items() if value is None]
            raise ValueError(
                f"{where} holds no {missing} - target {Target.DENSE_FULL!r} needs a cache written under the key and query"
            )
        # The model's own scale, and its refusal of a rotating head, whatever the cache recorded
        scale = latent_attention_scale(str(self.meta["model"]))
        if self.attention_scale is not None:
            scale = self.attention_scale
        rows = rows.to(self.device)
        return full_attention_target(
            head_queries[rows], key, scale, self.query_positions[rows], self.sink_size, self.window_size
        )

    def native_members(self, rows: torch.Tensor) -> torch.Tensor:
        """Return which keys the checkpoint's own selector picked for each query row.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Native-selection mask [R, T] bool.
        """
        picked = self._native_picks(rows)
        # One column past the end absorbs the empty slots
        members = torch.zeros(len(rows), self.key_count + 1, dtype=torch.bool, device=picked.device)
        members.scatter_(1, picked, True)
        return members[:, : self.key_count]

    def native_attention(self, rows: torch.Tensor) -> torch.Tensor:
        """Return the attention the served model spends on each key its selector picked.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Head-summed attention over keys [R, T] fp32, zero off the native selection.

        Raises:
            ValueError: The document was extracted without the served attention.
        """
        if self.native_key_mass is None:
            raise ValueError(
                f"{self.collection}/{self.name} layer {self.layer} has no native attention - extract it with queries.native"
            )
        picked = self._native_picks(rows)
        mass = torch.zeros(len(rows), self.key_count + 1, device=picked.device)
        mass.scatter_add_(1, picked, self.native_key_mass[rows].float())
        return mass[:, : self.key_count]

    def _native_picks(self, rows: torch.Tensor) -> torch.Tensor:
        """Return the native selection of each row, empty slots pointing one past the last key.

        Args:
            rows: Query rows [R] int64.

        Returns:
            Key ids [R, W] int64.

        Raises:
            ValueError: The document was extracted without the native selection.
        """
        if self.native_key_ids is None:
            raise ValueError(
                f"{self.collection}/{self.name} layer {self.layer} has no native selection - extract it with queries.native"
            )
        picked = self.native_key_ids[rows].long()
        return torch.where(picked >= 0, picked, self.key_count)


def _check_visibility(path: Path, meta: dict[str, Any], indexer: IndexerConfig) -> None:
    """Reject a cache written under a visibility rule this run does not use.

    The target is defined over the keys a read may see, so a window or sink that differs makes every number
    mean something else. The cluster block has to be the window, or the rule would hide keys a read selects.

    Args:
        path: The cache file, for the message.
        meta: The file's meta block.
        indexer: The design's knobs of the run.

    Raises:
        ValueError: A visibility field in the meta disagrees with the indexer, or the cluster block is not the window.
    """
    asked = {Cache.SINK_SIZE: indexer.sink_size, Cache.WINDOW_SIZE: indexer.window_size}
    disagree = {name: (meta[name], value) for name, value in asked.items() if meta[name] != value}
    if disagree:
        spelled = ", ".join(f"{name} {cached} != {wanted}" for name, (cached, wanted) in disagree.items())
        raise ValueError(f"{path} was written under a different visibility rule - {spelled}")
    if meta[Cache.CLUSTER_BLOCK_SIZE] != meta[Cache.WINDOW_SIZE]:
        raise ValueError(
            f"{path} was written with a cluster block of {meta[Cache.CLUSTER_BLOCK_SIZE]} against a window of "
            f"{meta[Cache.WINDOW_SIZE]} - the cluster block is the window, or the rule hides keys a read selects"
        )


def _check_index_visibility(path: Path, meta: dict[str, Any], indexer: IndexerConfig) -> None:
    """Reject an index-vector cache written under a window or sinks this run does not use.

    Args:
        path: The cache file, for the message.
        meta: The file's meta block, in the index-vector cache's names.
        indexer: The design's knobs of the run.

    Raises:
        ValueError: The window or the sinks differ from the run's.
    """
    asked = {Cache.INDEX_SINK: indexer.sink_size, Cache.INDEX_WINDOW: indexer.window_size}
    disagree = {name: (meta[name], value) for name, value in asked.items() if meta[name] != value}
    if disagree:
        spelled = ", ".join(f"{name} {cached} != {wanted}" for name, (cached, wanted) in disagree.items())
        raise ValueError(f"{path} was written under a different visibility rule - {spelled}")


def _train_query_count(path: Path, meta: dict[str, Any], query_count: int) -> int:
    """Return how many of a document's query rows are training positions.

    Args:
        path: The cache file, for the message.
        meta: The file's meta block.
        query_count: How many query rows the file holds.

    Returns:
        The count of leading rows that are training positions.

    Raises:
        ValueError: The meta names a count that does not fit.
    """
    count = int(meta[Cache.TRAIN_QUERIES])
    if not 0 <= count < query_count:
        raise ValueError(f"{path} says {count} training queries of {query_count}")
    return count
