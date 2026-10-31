"""The cards - the five kinds of dataset card, and the model card.

A card is what is declared about a dataset or a checkpoint before anything is prepared from it.

A source is text as it was collected, and says where it came from and what it may be used for.
A tokenized dataset is some sources as one tokenizer's ids in fixed-length rows.
Token shards are flat ids another pipeline wrote, read where they are.
A benchmark is what an accuracy run asks - a hub dataset read as it is, or a generator's grid rendered once.
A workload is what a performance run sends, built from a tokenized dataset.

Nothing on a card is measured. What preparing a dataset found - its vocabulary, its rows, its hashes - is in
the dataset's [manifest](../data/manifests.py) beside the data, which `manifest()` reads and holds to the card.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import env
from lsa.assets.constants import Licence, Preparer, Unpinned
from lsa.constants import Attention
from lsa.data import manifests
from lsa.data.constants import CorpusFiles, ShardFiles
from lsa.data.manifests import (
    BenchmarkManifest,
    Manifest,
    ShardManifest,
    TokenizedManifest,
    WorkloadManifest,
)


def _require_absolute(name: str, path: str) -> None:
    """Refuse a path a compute node could not find.

    Raises:
        ValueError: `path` is not absolute.
    """
    if not path.startswith("/"):
        raise ValueError(f"{name}: path must be absolute - got {path!r}")


@dataclass(frozen=True)
class SourceCard:
    """One source of collected text, which is its own origin.

    `dataset` is the hub dataset it was streamed from and `collected_from` the directory of a source that never
    left the cluster - exactly one of them. `licence` is an SPDX id, `internal`, or `per-document` when the
    manifest beside its shards carries each document's own.
    """

    name: str
    # One folder of parquet shards, each carrying a `doc_id` and a `text` column
    path: str
    licence: str
    about: str = ""
    dataset: str = ""
    # The hub commit that was the dataset's head when it was collected, empty where nobody can read it
    revision: str = ""
    # The configs or data directories the collector read, and the split
    subsets: list[str] = field(default_factory=list)
    split: str = ""
    collected_from: str = ""

    def __post_init__(self) -> None:
        """Reject a source that does not say where it came from or what it may be used for.

        Raises:
            ValueError: `path` is not absolute, neither or both of `dataset` and `collected_from` are set, or
                `licence` is empty.
        """
        _require_absolute(self.name, self.path)
        if bool(self.dataset) == bool(self.collected_from):
            raise ValueError(
                f"{self.name}: a source comes from a hub dataset or a local path, and exactly one"
            )
        if not self.licence:
            raise ValueError(f"{self.name}: a source needs a licence, or `{Licence.PER_DOCUMENT}`")

    def document_licences(self) -> dict[str, str]:
        """Return each document's own licence, read off the manifest beside the shards.

        Returns:
            Document id to the licence upstream states for it.

        Raises:
            ValueError: The card gives one licence for every document, so there is no column to read.
        """
        if self.licence != Licence.PER_DOCUMENT:
            raise ValueError(
                f"{self.name}: every document is {self.licence} - there is no per-document licence"
            )
        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(self.path) / Licence.MANIFEST, columns=[Licence.DOCUMENT_COLUMN, Licence.LICENCE_COLUMN]
        ).to_pydict()
        return dict(zip(table[Licence.DOCUMENT_COLUMN], table[Licence.LICENCE_COLUMN], strict=True))


@dataclass(frozen=True)
class TokenizedCard:
    """Some sources as one tokenizer's ids, one `<source>.bin` of `sample_length`-token rows each."""

    name: str
    path: str
    # Source cards, in the order a source index resolves
    sources: list[str]
    # The model card whose tokenizer writes the ids
    tokenizer: str
    # Tokens per row, which is the longest contiguous span a sample can come from
    sample_length: int
    about: str = ""

    def __post_init__(self) -> None:
        """Reject a tokenized dataset no reader could open.

        Raises:
            ValueError: `path` is not absolute, no source or tokenizer is named, or `sample_length` is not
                positive.
        """
        _require_absolute(self.name, self.path)
        if not self.sources or not self.tokenizer:
            raise ValueError(f"{self.name}: a tokenized dataset names its sources and its tokenizer")
        if self.sample_length <= 0:
            raise ValueError(f"{self.name}: sample_length must be positive - got {self.sample_length}")

    def bin_path(self, source: str) -> Path:
        """Return one source's tokenized rows."""
        return Path(self.path) / f"{source}{CorpusFiles.BIN_SUFFIX}"

    def manifest(self) -> TokenizedManifest:
        """Return what tokenizing measured, held to this card.

        Raises:
            FileNotFoundError: The dataset has not been tokenized.
            ValueError: It was tokenized under a card that differs from this one.
        """
        found = manifests.read_tokenized(self.path)
        manifests.check_declared(self.name, found.declared, self, self.path)
        return found


@dataclass(frozen=True)
class ShardCard:
    """A directory of flat token files another pipeline wrote - training shards and a validation file.

    A shard is ids and nothing else: no rows, no document boundaries beyond the end-of-document ids the writer
    put in, so a reader cuts consecutive windows. The directory is shared and not ours, so a card points at it
    where it is and nothing prepares it.
    """

    name: str
    path: str
    # The model card whose tokenizer wrote the ids
    tokenizer: str
    about: str = ""

    def __post_init__(self) -> None:
        """Reject a shard directory no reader could open.

        Raises:
            ValueError: `path` is not absolute, or no tokenizer is named.
        """
        _require_absolute(self.name, self.path)
        if not self.tokenizer:
            raise ValueError(f"{self.name}: token shards name the tokenizer that wrote them")

    def train_paths(self) -> list[Path]:
        """Return the training shards in name order, which is the order a reader's shard index means."""
        return sorted(Path(self.path).glob(ShardFiles.TRAIN_GLOB))

    def validation_path(self) -> Path:
        """Return the validation file the writer fixed."""
        return Path(self.path) / ShardFiles.VALIDATION

    def manifest(self) -> ShardManifest:
        """Return what the writer recorded, read where it is.

        Raises:
            FileNotFoundError: The directory holds no metadata.
        """
        return manifests.read_shards(self.path)


@dataclass(frozen=True)
class BenchmarkLength:
    """One length of a generated benchmark's grid, and the rows a file of it holds."""

    length: int
    rows: int


@dataclass(frozen=True)
class BenchmarkCard:
    """What an accuracy run asks - a hub dataset, or a generator and its grid.

    `dataset` is read straight from the hub at its pinned `revision`. A benchmark with `generator` instead
    is rendered by that repository at that commit. Either is prepared into files when `path` is set, one per
    combination of the grid that `file` names - `{task}` and `{length}` for a generated one, `{context}` for
    one truncated to a context - and the files' lengths are counted in `tokenizer`, a model card.

    `prompt` and `join` say which fields of a prepared row compose one prompt, for a reader that is not the
    harness that wrote it. `preparer` names the generator that writes the files, one of `Preparer.ALL`, and
    says how they are made rather than what they hold, so a manifest does not record it.
    """

    name: str
    licence: str
    about: str = ""
    dataset: str = ""
    generator: str = ""
    revision: str = ""
    subsets: list[str] = field(default_factory=list)
    split: str = ""
    path: str = ""
    file: str = ""
    tokenizer: str = ""
    tasks: list[str] = field(default_factory=list)
    lengths: list[BenchmarkLength] = field(default_factory=list)
    # The tokens a prompt is truncated to
    context: int = 0
    seed: int = 0
    prompt: list[str] = field(default_factory=list)
    join: str = "\n"
    preparer: str = ""

    def __post_init__(self) -> None:
        """Reject a benchmark that does not say where its rows come from.

        Raises:
            ValueError: Neither or both of `dataset` and `generator` are set, `revision` or `licence` is
                empty, or a prepared benchmark leaves out its file or the tokenizer its lengths are counted in.
            ValueError: `preparer` names no generator this repository has.
        """
        if self.preparer and self.preparer not in Preparer.ALL:
            raise ValueError(f"{self.name}: preparer must be one of {Preparer.ALL} - got {self.preparer!r}")
        if bool(self.dataset) == bool(self.generator):
            raise ValueError(f"{self.name}: a benchmark is a hub dataset or a generator, and exactly one")
        if not self.revision or not self.licence:
            raise ValueError(f"{self.name}: a benchmark pins the revision it reads and says its licence")
        if self.path:
            _require_absolute(self.name, self.path)
            if not self.file or not self.tokenizer:
                raise ValueError(f"{self.name}: a prepared benchmark names its file and its tokenizer")

    @property
    def length_values(self) -> list[int]:
        """Return the grid's lengths, in order."""
        return [one.length for one in self.lengths]

    def rows_at(self, length: int) -> int:
        """Return how many rows a file at one length of the grid holds.

        Raises:
            KeyError: The grid has no such length.
        """
        for one in self.lengths:
            if one.length == length:
                return one.rows
        raise KeyError(f"{self.name} has no length {length} - it has {self.length_values}")

    def data_root(self) -> str:
        """Return the root this benchmark's directory sits in, which is where its manifest is."""
        return str(Path(self.path).parent)

    def file_path(self, data_root: str | None = None, **grid: int | str) -> str:
        """Return one prepared file, under the shared root or a data root of the same layout.

        Args:
            data_root: A root other than the card's own, holding a directory of the same name.
            grid: The values `file` names, as `task` and `length`.

        Returns:
            The file's path, which may not exist.
        """
        directory = os.path.join(data_root, Path(self.path).name) if data_root else self.path
        return os.path.join(directory, self.file.format(**grid))

    def manifest(self, data_root: str | None = None) -> BenchmarkManifest:
        """Return what generation recorded about every prepared file of this benchmark, held to this card.

        Args:
            data_root: A root other than the card's own.

        Raises:
            FileNotFoundError: Nothing of this benchmark has been prepared there.
            ValueError: A file was prepared under a card that differs from this one.
        """
        root = data_root or self.data_root()
        prefix = f"{Path(self.path).name}/"
        files = {
            key: record
            for key, record in manifests.read_benchmark_files(root).items()
            if key.startswith(prefix)
        }
        if not files:
            raise FileNotFoundError(f"{self.name}: the manifest under {root} records no file of {prefix}")
        for key, record in files.items():
            manifests.check_declared(self.name, record.declared, self, os.path.join(root, key))
        return BenchmarkManifest(declared=manifests.declared(self), files=files)


@dataclass(frozen=True)
class WorkloadCard:
    """What a performance run sends - one prompt file per length, built from a tokenized dataset."""

    name: str
    path: str
    # The tokenized dataset the text is decoded from, and its source
    tokenized: str
    source: str
    # One file per length, `{length}` in the name
    file: str
    lengths: list[int]
    # Prompts per file - one count, entry i at every length a prefix of the next length's, or a count per
    # length, each length's entries a span of their own and not nested
    entries: int | dict[int, int]
    about: str = ""

    def __post_init__(self) -> None:
        """Reject a workload that would build no file.

        Raises:
            ValueError: `path` is not absolute, or `lengths` or `entries` leaves nothing to send.
            ValueError: A count per length does not name exactly the card's lengths.
        """
        _require_absolute(self.name, self.path)
        counts = [self.entries] if isinstance(self.entries, int) else list(self.entries.values())
        if not self.lengths or min(counts) <= 0:
            raise ValueError(f"{self.name}: a workload has lengths and at least one entry")
        if isinstance(self.entries, dict) and sorted(self.entries) != sorted(self.lengths):
            raise ValueError(
                f"{self.name}: entries counts lengths {sorted(self.entries)}, the card lists {sorted(self.lengths)}"
            )

    @property
    def nested(self) -> bool:
        """Return whether entry i at every length is a prefix of the same entry at the next."""
        return isinstance(self.entries, int)

    def entries_at(self, length: int) -> int:
        """Return the prompts one length's file holds."""
        return self.entries if isinstance(self.entries, int) else self.entries[length]

    def file_path(self, length: int, root: str | None = None) -> str:
        """Return one length's prompt file, under the card's directory or another."""
        return os.path.join(root or self.path, self.file.format(length=length))

    def manifest(self) -> WorkloadManifest:
        """Return what building the files measured, held to this card.

        Raises:
            FileNotFoundError: The workload has not been built.
            ValueError: It was built under a card that differs from this one.
        """
        found = manifests.read_workload(self.path)
        manifests.check_declared(self.name, found.declared, self, self.path)
        return found


type DatasetCard = SourceCard | TokenizedCard | ShardCard | BenchmarkCard | WorkloadCard


@dataclass
class ModelCard:
    """One checkpoint this repository knows by name.

    `family` and `flavor` are the trainer's - which module builds the architecture, and which entry of that
    family's registry gives the dims. `adapter`, `layer_types` and the rotary pair are the harness's, and
    every one of them is optional because the harness works them out from the checkpoint when it can.
    """

    name: str
    # One sentence saying what the checkpoint is
    about: str = ""
    # A checkpoint this repository wrote, under the shared root - empty for a published one
    path: str = ""
    # What it is called upstream, which is what a record names it by
    repo: str = ""
    # The hub commit of the published snapshot, which is what the hub cache keeps it under
    revision: str = ""
    # The architecture, which decides both the model module and the index space
    family: str = ""
    # The size, as that family's registry names it
    flavor: str = ""
    # The harness code that drives it, or `auto` to resolve it from the checkpoint's own `model_type`
    adapter: str = Unpinned.ADAPTER
    # What this stack calls the quadratic layers LSA replaces, when it is not `full_attention`
    layer_types: list[str] = field(default_factory=list)
    # Where the rotation sits, pinned against the measurement rather than supplying it
    rotary_end: str = Unpinned.ROTARY_END
    rotary_dim: int = 0
    # The rotation it is served and distilled under, replacing the checkpoint's own `rope_parameters` - YaRN past
    # its native context - or empty for the checkpoint's as published
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    # The attention the checkpoint ships - `full`, or `block_sparse` for a learned indexer of its own - empty for a
    # snapshot read by path, which nothing has looked at
    attention: str = ""

    def __post_init__(self) -> None:
        """Reject a card that names a checkpoint no caller could find, or an attention no checkpoint ships.

        Raises:
            ValueError: Neither `path` nor a `repo` pinned to a `revision` says where the checkpoint is.
            ValueError: `path` is set and is not absolute.
            ValueError: `rotary_dim` is given without an end for it to sit at.
            ValueError: `attention` is not one a checkpoint ships.
        """
        if not self.path and not (self.repo and self.revision):
            raise ValueError(
                f"{self.name}: a model card needs a path, or a repo and the revision it is pinned to"
            )
        if self.path:
            _require_absolute(self.name, self.path)
        if self.attention and self.attention not in Attention.SHIPPED:
            raise ValueError(
                f"{self.name}: attention must say what the checkpoint ships, one of {Attention.SHIPPED} - "
                f"got {self.attention!r}"
            )
        if self.rotary_dim and self.rotary_end == Unpinned.ROTARY_END:
            raise ValueError(
                f"{self.name}: rotary_dim needs an end to sit at - name rotary_end too, or drop both "
                "and let the layout be measured from the model"
            )

    @property
    def checkpoint(self) -> str:
        """Return the directory to hand a loader - the checkpoint's own, or the pinned snapshot in the hub cache.

        The hub cache is this user's `HF_HOME`, which a job exports before it starts, and the snapshot is a
        directory there - so a loader that needs a directory and one that takes a hub id read the same files.
        """
        if self.path:
            return self.path
        # The hub cache's own folder naming, spelled here so a bare install needs no huggingface_hub
        folder = "--".join(("models", *self.repo.split("/")))
        return str(env.hf_home() / "hub" / folder / Manifest.SNAPSHOTS / self.revision)
