"""What preparing a dataset measured, kept on disk beside the data rather than on its card.

A card says what a dataset is meant to be before anything is prepared from it - its sources, its tokenizer,
its grid. A manifest says what preparing it found - the vocabulary, the rows, the hashes, the commit and the
date - and repeats the card's declared fields verbatim as `declared`.
That copy is what catches a card edited after its data was made: reading the manifest through the card refuses
when the two disagree, rather than a run trusting a name whose bytes say something else.

The declared fields are every field of the card but `path`, `about` and `preparer`. Where a dataset sits, how
it is described in prose and which generator wrote it do not change what its bytes are, so moving the root or
rewording a card leaves every manifest true.

Writing a manifest is the prepare step's job, and every write here goes through [lsa.io](../io.py), so a
reader never sees half of one.
"""

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import config_loader, io
from lsa.data.constants import ShardFiles

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# Fields of a card that say where it is, how it reads and what writes it, not what its data is
_UNDECLARED: tuple[str, ...] = ("path", "about", "preparer")

# What every benchmark file's entry records, beside whatever its generator adds
_BENCHMARK_FIELDS: tuple[str, ...] = ("rows", "sha256", "tokenizer", "date")
_DECLARED: str = "declared"


class Manifest:
    """Where a prepared dataset's measured manifest sits, beside the data it measures."""

    # A tokenized dataset's, in its own directory - the name tokenizing has always written
    TOKENIZED: str = "corpus.json"

    # A benchmark's, under the data root its directory sits in - one entry file per prepared file, because a
    # hundred generators writing one file at once on a filesystem whose flock does not serialise lost entries
    BENCHMARK_FOLDER: str = "manifest"
    # The single-writer form an entry may also be in, which an entry file wins over
    BENCHMARK_FILE: str = "manifest.json"

    # A workload's, in its own directory
    # The same name as the benchmark's single-writer file but a different file, one directory down in another tree
    WORKLOAD: str = "manifest.json"

    # What the hub cache calls the folder a pinned snapshot sits in
    SNAPSHOTS: str = "snapshots"


def declared(card: "DataclassInstance") -> dict[str, Any]:
    """Return the fields a card declares, in the form a manifest stores them.

    Round-tripped through json, so a list and a tuple, or an int key and its string, compare as the file
    will hold them.

    Args:
        card: Any dataset card.

    Returns:
        Field name to value, without `path`, `about` and `preparer`.
    """
    fields_of = {name: value for name, value in asdict(card).items() if name not in _UNDECLARED}
    normalised: dict[str, Any] = json.loads(json.dumps(fields_of))
    return normalised


def check_declared(name: str, recorded: dict[str, Any], card: "DataclassInstance", where: str | Path) -> None:
    """Refuse a manifest whose declared fields are not the card's.

    Args:
        name: The card's name, for the message.
        recorded: What the manifest says was declared when the data was prepared.
        card: The card as it is now.
        where: The manifest, for the message.

    Raises:
        ValueError: The two differ, naming every field that does.
    """
    expected = declared(card)
    differing = sorted(key for key in set(expected) | set(recorded) if expected.get(key) != recorded.get(key))
    if differing:
        raise ValueError(
            f"{name}: {where} was prepared under a card that differs in {differing} - prepare it again, or "
            f"put the card back to { ({key: recorded.get(key) for key in differing}) }"
        )


def today() -> str:
    """Return the date a manifest is stamped with."""
    return datetime.date.today().isoformat()


def snapshot_revision(snapshot: str | Path) -> str:
    """Return the hub commit a snapshot directory is, or empty for one that is not in a hub cache.

    Args:
        snapshot: A snapshot directory, or a link to one.

    Returns:
        The commit the hub cache keeps it under.
    """
    resolved = Path(snapshot).resolve()
    return resolved.name if resolved.parent.name == Manifest.SNAPSHOTS else ""


@dataclass(frozen=True)
class TokenizedManifest:
    """What tokenizing a dataset measured, `corpus.json` in its directory.

    `tokenizer`, `vocab_size`, `sample_length` and `sources` are what tokenizing has always written there;
    the rest is what a card used to claim and only tokenizing can know.
    """

    declared: dict[str, Any]
    # The snapshot directory tokenizing read the tokenizer from, and the hub commit it is
    tokenizer: str
    tokenizer_revision: str
    # How many ids the rows may contain
    vocab_size: int
    sample_length: int
    # The sources on disk, in the order a source index resolves
    sources: list[str]
    # Per source, the rows it contributed and the documents that gave at least one whole row
    rows: dict[str, int]
    documents: dict[str, int]
    git_sha: str = ""
    date: str = ""

    @property
    def tokens(self) -> int:
        """Return the tokens the whole dataset holds."""
        return sum(self.rows.values()) * self.sample_length


@dataclass(frozen=True)
class ShardManifest:
    """What the pipeline that wrote a directory of token shards recorded, `metadata.json` in it.

    Read in place and never written here, so it carries no `declared`: the card cannot be held to a file this
    repository does not own, and a reader checks what it relies on - the vocabulary, the dtype - instead.
    """

    # The ids' tokenizer by its hub name, how wide its vocabulary is, and the id that ends a document
    tokenizer: str
    vocab_size: int
    eos_token_id: int
    # The numpy dtype of a token in a shard
    dtype: str
    # Where the text came from, which the writer records and nothing here reads
    dataset: str = ""
    dataset_config: str = ""
    dataset_split: str = ""
    text_field: str = ""
    # Tokens in the validation file and in a full training shard, and how many training shards there are
    num_val_tokens: int = 0
    shard_size: int = 0
    num_train_shards: int = 0


def read_shards(path: str | Path) -> ShardManifest:
    """Read the metadata a directory of token shards keeps beside them.

    Args:
        path: The shards' directory.

    Returns:
        What the writer recorded.

    Raises:
        FileNotFoundError: The directory holds no metadata.
    """
    where = Path(path) / ShardFiles.METADATA
    if not where.is_file():
        raise FileNotFoundError(f"{where} does not exist - it is written by whatever made the shards")
    return config_loader.construct(ShardManifest, io.read_json(where))


def read_tokenized(path: str | Path) -> TokenizedManifest:
    """Read the manifest a tokenized dataset keeps in its directory.

    Args:
        path: The tokenized dataset's directory.

    Returns:
        What tokenizing it measured.

    Raises:
        FileNotFoundError: The directory holds no manifest.
        ValueError: The manifest predates cards and declares nothing.
    """
    return _read_manifest(Path(path) / Manifest.TOKENIZED, TokenizedManifest, "prepare the dataset")


def _read_manifest[Manifested: DataclassInstance](
    where: Path, schema: type[Manifested], remedy: str
) -> Manifested:
    """Read one manifest file and build its dataclass.

    Args:
        where: The manifest file.
        schema: The dataclass it holds.
        remedy: What to do about a missing file, for the message.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: There is no file.
        ValueError: The manifest predates cards and declares nothing.
    """
    if not where.is_file():
        raise FileNotFoundError(f"{where} does not exist - {remedy}, which writes it")
    recorded = io.read_json(where)
    if _DECLARED not in recorded:
        raise ValueError(f"{where} predates dataset cards and declares nothing - write its manifest again")
    return config_loader.construct(schema, recorded)


def write_tokenized(path: str | Path, manifest: TokenizedManifest) -> None:
    """Write a tokenized dataset's manifest into its directory, atomically.

    Args:
        path: The tokenized dataset's directory.
        manifest: What tokenizing it measured.
    """
    io.write_json(Path(path) / Manifest.TOKENIZED, asdict(manifest))


@dataclass(frozen=True)
class BenchmarkFile:
    """What generating one prepared benchmark file recorded about it.

    `details` is whatever else its generator wrote - the seed, the generator's commit, the prompt sizes - and
    an accuracy record carries all of it beside `rows`, `sha256`, `tokenizer` and `date`.
    `declared` is the card the file was prepared under, and is empty for a file no card prepared, such as the
    ones lm-eval's port made.
    """

    rows: int
    sha256: str
    tokenizer: str
    date: str
    details: dict[str, Any] = field(default_factory=dict)
    declared: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "BenchmarkFile":
        """Build one from an entry as a manifest file holds it.

        Args:
            entry: The flat entry, with `declared` beside what was measured.

        Returns:
            The file's record.
        """
        measured = {key: entry[key] for key in _BENCHMARK_FIELDS}
        details = {key: value for key, value in entry.items() if key not in (*_BENCHMARK_FIELDS, _DECLARED)}
        return cls(**measured, details=details, declared=entry.get(_DECLARED, {}))

    def recorded(self) -> dict[str, Any]:
        """Return what generation recorded, without the card it was prepared under."""
        return {**{name: getattr(self, name) for name in _BENCHMARK_FIELDS}, **self.details}

    def entry(self) -> dict[str, Any]:
        """Return the flat entry a manifest file holds."""
        return {**self.recorded(), _DECLARED: self.declared} if self.declared else self.recorded()


@dataclass(frozen=True)
class BenchmarkManifest:
    """Every prepared file of one benchmark, keyed by its path under the data root."""

    declared: dict[str, Any]
    files: dict[str, BenchmarkFile]


def read_benchmark_files(data_root: str | Path) -> dict[str, BenchmarkFile]:
    """Return what generation recorded about every prepared file under a data root.

    `manifest.json` first, then one entry file each under `manifest/`, which wins - nothing is
    read-modify-written, so concurrent generators never lose each other's entries.

    Args:
        data_root: The root the benchmark directories sit in.

    Returns:
        File, relative to the root, to its record.
    """
    entries: dict[str, Any] = {}
    single = Path(data_root) / Manifest.BENCHMARK_FILE
    if single.is_file():
        entries.update(io.read_json(single))
    folder = Path(data_root) / Manifest.BENCHMARK_FOLDER
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        entries.update(io.read_json(path))
    return {key: BenchmarkFile.from_entry(entry) for key, entry in entries.items()}


def write_benchmark_file(data_root: str | Path, key: str, record: BenchmarkFile) -> None:
    """Write one prepared file's entry as its own file, `<root>/manifest/<key with / as _>.json`.

    Args:
        data_root: The root the benchmark directories sit in.
        key: The prepared file, relative to the root.
        record: What generation recorded about it.
    """
    name = key.replace("/", "_") + ".json"
    io.write_json(Path(data_root) / Manifest.BENCHMARK_FOLDER / name, {key: record.entry()}, indent=1)


@dataclass(frozen=True)
class WorkloadManifest:
    """What building a workload's prompt files measured, `manifest.json` in its directory.

    `files` is keyed by length, each with its file's name, its sha256 and every entry's rows, tokens and
    characters.
    """

    declared: dict[str, Any]
    # The tokenized rows the text was decoded from, the tokens per row, and the tokenizer's hub id
    corpus: str
    row: int
    tokenizer: str
    entries: int | dict[int, int]
    files: dict[str, Any]
    git_sha: str = ""
    date: str = ""


def read_workload(path: str | Path) -> WorkloadManifest:
    """Read the manifest a workload keeps in its directory.

    Args:
        path: The workload's directory.

    Returns:
        What building it measured.

    Raises:
        FileNotFoundError: The directory holds no manifest.
        ValueError: The manifest predates cards and declares nothing.
    """
    return _read_manifest(Path(path) / Manifest.WORKLOAD, WorkloadManifest, "build the workload")


def write_workload(path: str | Path, manifest: WorkloadManifest) -> None:
    """Write a workload's manifest into its directory, atomically.

    Args:
        path: The workload's directory.
        manifest: What building it measured.
    """
    io.write_json(os.path.join(path, Manifest.WORKLOAD), asdict(manifest), indent=1)
