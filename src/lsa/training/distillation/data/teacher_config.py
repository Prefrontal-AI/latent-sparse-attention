"""The schema of one target extraction - which documents, which positions, which checkpoint - and the yaml loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import assets, config_loader
from lsa.evaluation.config import ModelConfig
from lsa.sglang.settings.design import ATTENTION_INPUT, INDEXER_INPUTS, IndexerConfig
from lsa.training.constants import Corpus
from lsa.training.distillation.constants import Probe, Split
from lsa.training.distillation.data import selection as document_selection


@dataclass
class TokenizedConfig:
    """Which tokenized documents to extract, and how they divide.

    Counts are per source and per side of the boundary, so a source contributes what is asked of it whatever
    its hash happened to do. Which documents fall on which side is the dataset's own decision - a hash of the
    document id - and never a range of indices written here.

    `windows` past the first is the most correlated data available, another 65,536 tokens of a document
    already read, so it stays at one and breadth comes from raising the counts instead.

    `documents` names a document-selection manifest specifying each document, split and window length.
    With a manifest, counts, `windows` and `length` are ignored. `lengths` extracts only the selected
    length buckets while the collection manifest still records the full selection.

    `nest` also reads each document of the sides `splits` names at every shorter length in `lengths`, from the
    same first token, so a document the manifest took at 1M is read at 512k and below too and the lengths are
    comparable on the same text.
    """

    # A tokenized dataset [card](../../../assets/__init__.py), or `path` for one that has no card yet
    name: str = assets.Default.DATASET
    path: str | None = None
    sources: list[str] | None = None
    train: int = 9
    evaluate: int = 3
    windows: int = 1
    # Tokens taken from the front of a window, or 0 for the whole of it
    length: int = 0
    held_out_fraction: float = Corpus.HELD_OUT_FRACTION
    eval_share: float = Split.EVAL_SHARE
    documents: str | None = None
    lengths: list[int] | None = None
    splits: list[str] | None = None
    nest: bool = False

    def __post_init__(self) -> None:
        """Reject a selection that would extract nothing.

        Raises:
            ValueError: Neither side takes a document, or a count is negative.
            ValueError: `windows` is not positive, or `length` is negative.
            ValueError: `held_out_fraction` or `eval_share` is out of range.
            ValueError: `sources` is present but empty.
            ValueError: `lengths` is given without a document-selection manifest, or is empty.
            ValueError: `splits` is empty or names a side that does not exist.
            ValueError: `nest` is set without `documents`, `lengths` and `splits`.
        """
        if self.nest and not (self.documents and self.lengths and self.splits):
            raise ValueError(
                "tokenized.nest needs tokenized.documents, tokenized.lengths and tokenized.splits"
            )
        if self.lengths is not None and not self.lengths:
            raise ValueError("tokenized.lengths must name at least one length")
        if self.splits is not None and (not self.splits or not set(self.splits) <= set(Split.BOTH)):
            raise ValueError(f"tokenized.splits must name sides out of {Split.BOTH} - got {self.splits!r}")
        if self.train < 0 or self.evaluate < 0:
            raise ValueError(f"tokenized.train and tokenized.evaluate must not be negative - got {self!r}")
        config_loader.require_positive(self, "tokenized", "windows")
        if self.length < 0:
            raise ValueError(f"tokenized.length must not be negative - got {self.length}")
        if not 0 < self.held_out_fraction <= 1:
            raise ValueError(f"tokenized.held_out_fraction must be in (0, 1] - got {self.held_out_fraction}")
        if not 0 < self.eval_share < 1:
            raise ValueError(f"tokenized.eval_share must be in (0, 1) - got {self.eval_share}")
        if self.sources is not None and not self.sources:
            raise ValueError("tokenized.sources must name at least one source, or be left out entirely")

    @property
    def asset(self) -> assets.TokenizedCard:
        """Return the dataset this extraction reads, by name or off its own manifest."""
        return assets.resolve(self.name, self.path)

    @property
    def tokens_per_document(self) -> int:
        """Return how much of a window one document contributes, defaulting to the whole of it."""
        return self.length or self.asset.sample_length


@dataclass
class PreparedConfig:
    """Which prepared benchmark prompts to extract beside the tokenized documents.

    A benchmark card by name alone - left empty the extraction is the tokenized dataset by itself. Everything
    about how the files are laid out and which of their fields compose a prompt belongs to that card, because
    it describes that benchmark and nothing else. `sources` are the card's tasks to read.

    Every prepared prompt scores and none of them trains, so this has no split to state.

    `rows` takes those rows of each file in place of the first `samples` - the served records' `doc_id`s, so a
    cached prompt is the one a served sample scored. `root` reads files of the card's layout from another data
    root, such as a cell generated with more rows than the shared one holds.

    `join` and `chat` read a prompt as the accuracy harness serves it rather than as the card's bare text: lm-eval
    joins RULER's haystack and question with a space, and the request is one user turn the server renders in the
    model's chat template and answers past a closed think block. The last positions are then the ones the
    answer is read from. `chat_template_kwargs` renders the template with those arguments instead, the way a run
    whose generation block names them is served - Qwen3.8-27B's `enable_thinking: false` closes the think block
    inside the template, and nothing is appended.

    `source_rows` is `rows` one task at a time, for a selection that draws its own rows for each - a task named
    there takes its list and a task not named takes none. It is the only one of the two that may be set.
    """

    name: str = ""
    sources: list[str] | None = None
    samples: int = 2
    # The prepared length, which names the file - the same prompts exist at several and this picks one
    length: int = 65536
    rows: list[int] | None = None
    source_rows: dict[str, list[int]] | None = None
    root: str | None = None
    join: str | None = None
    chat: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a selection that would read nothing.

        Raises:
            ValueError: `samples` or `length` is not positive.
            ValueError: `sources` or `rows` is present but empty, or `rows` holds a negative row or a repeat.
            ValueError: `source_rows` is set beside `rows`, names a task outside `sources`, or holds a bad list.
        """
        config_loader.require_positive(self, "prepared", "samples", "length")
        if self.sources is not None and not self.sources:
            raise ValueError("prepared.sources must name at least one source, or be left out entirely")
        if self.rows is not None and (
            not self.rows or min(self.rows) < 0 or len(set(self.rows)) != len(self.rows)
        ):
            raise ValueError(
                f"prepared.rows must name distinct rows from 0, or be left out - got {self.rows}"
            )
        if self.source_rows is None:
            return
        if self.rows is not None:
            raise ValueError("prepared.rows and prepared.source_rows cannot both be set")
        outside = set(self.source_rows) - set(self.sources) if self.sources is not None else set()
        if outside:
            raise ValueError(f"prepared.source_rows names {sorted(outside)}, which prepared.sources does not")
        for source, rows in self.source_rows.items():
            if not rows or min(rows) < 0 or len(set(rows)) != len(rows):
                raise ValueError(
                    f"prepared.source_rows[{source}] must name distinct rows from 0 - got {rows}"
                )

    @property
    def asset(self) -> assets.BenchmarkCard:
        """Return the benchmark this extraction reads."""
        return assets.benchmark(self.name)


@dataclass
class MassConfig:
    """A dense attention-mass dump in place of the layer files - see [teacher_mass](teacher_mass.py) for its format.

    The rows are the last `queries.tail` positions of each window, and `queries.train` may be zero.
    `top` is how many keys each shared set keeps and `head_tops` the key counts each head's own mass is taken at.
    """

    top: int = 4096
    head_tops: list[int] = field(default_factory=lambda: [2048, 4096])

    def __post_init__(self) -> None:
        """Reject a dump that keeps nothing or whose key counts do not run up to `top` at most.

        Raises:
            ValueError: `top` is not positive, `head_tops` is empty, not ascending or not positive, or past `top`.
        """
        config_loader.require_positive(self, "mass", "top")
        if not self.head_tops or self.head_tops != sorted(set(self.head_tops)) or self.head_tops[0] <= 0:
            raise ValueError(
                f"mass.head_tops must be ascending distinct positive counts - got {self.head_tops}"
            )
        if self.head_tops[-1] > self.top:
            raise ValueError(f"mass.head_tops must not pass mass.top ({self.top}) - got {self.head_tops}")


@dataclass
class QueryConfig:
    """Which positions a document is sampled at, and how much of the teacher is kept for each.

    Training positions are drawn uniformly from the first one with a readable key - past the sinks and the window -
    because a query with nothing readable behind it has no long-range read to judge.
    The tail positions are the last `tail` of the document and are what a report scores - they see the whole
    document, which is the regime the design is for.

    `density` makes the training count follow the document - that many per position a query may sit at,
    and never fewer than `train`. Left at 0 every document gets `train`, which over documents of different
    lengths samples a long one far more thinly than a short one.

    `native` also keeps what the checkpoint's own sparse selector picked at every sampled position, for a
    model that ships one - the keys its attention actually reads, beside the dense target.

    `attention` also keeps the model's own rotated keys at every position and each head's query at the sampled ones,
    with the layer's softmax scale - what an offline PISA arm on a model with key-value heads reads.

    `targets` replaces the random draw with the positions a prepared prompt itself names and keeps, beside each, the keys
    that position is meant to find - `ruler` reads them off a RULER row, and `train`, `density` and `seed` then
    do nothing. A needle task is queried at its last `tail` positions, aimed at the needle's value and its line. A
    counting task is queried at the list items that repeat an earlier word - every one of the answer's words, and as
    many of the others, drawn by the row - aimed at that word's earlier occurrences, and at its last `tail` positions too.
    """

    train: int = 512
    tail: int = 32
    top: int = 256
    seed: int = 1000
    density: float = 0.0
    native: bool = False
    attention: bool = False
    targets: str = ""

    def __post_init__(self) -> None:
        """Reject a sampling that leaves nothing to train or score on.

        Raises:
            ValueError: A count is not positive (`train` may be zero), or `density` is negative.
            ValueError: `targets` names a mode that does not exist.
        """
        if self.targets and self.targets not in Probe.ALL:
            raise ValueError(f"queries.targets must be one of {Probe.ALL} or left out - got {self.targets!r}")
        config_loader.require_positive(self, "queries", "tail", "top", "seed")
        if self.train < 0:
            raise ValueError(f"queries.train must not be negative - got {self.train}")
        if self.density < 0:
            raise ValueError(f"queries.density must not be negative - got {self.density}")


@dataclass
class TeacherModelConfig(ModelConfig):
    """Describe the checkpoint and the design whose visibility rule the targets are taken under.

    Only `sink_size` and `window_size` are read - the cluster block is the window, so a key is readable when it
    is past the sinks and at least a window behind the query.
    """

    indexer: IndexerConfig = field(default_factory=IndexerConfig)


@dataclass
class DataConfig:
    """Choose documents and query positions for target extraction."""

    tokenized: TokenizedConfig = field(default_factory=TokenizedConfig)
    prepared: PreparedConfig = field(default_factory=PreparedConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
    selection: document_selection.SelectionConfig | None = None
    mass: MassConfig | None = None


@dataclass(kw_only=True)
class RunConfig:
    """Name the extraction, its required output and its execution layout."""

    output: str
    name: str = "teacher"
    layers: list[int] | None = None
    prefill: int = 0
    stages: list[int] | None = None
    collection: str = ""
    # What a layer file keeps of the attention's input - `x` whole, or the key and query its layer reads off it
    input: str = ATTENTION_INPUT

    def __post_init__(self) -> None:
        """Reject empty output, an empty layer selection, negative prefill or an input no run reads."""
        if not self.output:
            raise ValueError("run.output must name the targets directory")
        if self.input not in INDEXER_INPUTS:
            raise ValueError(f"run.input must be one of {INDEXER_INPUTS} - got {self.input!r}")
        if self.layers is not None and not self.layers:
            raise ValueError("layers must name at least one layer")
        if self.prefill < 0:
            raise ValueError("prefill must be nonnegative")
        if self.collection and (
            Path(self.collection).name != self.collection or self.collection in {".", ".."}
        ):
            raise ValueError("run.collection must be one directory name")


@dataclass(kw_only=True)
class TeacherConfig:
    """Describe target extraction through model, data and run blocks."""

    model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig

    def __post_init__(self) -> None:
        """Reject ambiguous selection inputs, missing collection names and an extraction with no rows.

        Raises:
            ValueError: A selection and a documents manifest are both named, a selection has no collection, or
                an extraction that keeps layer files samples no training row.
        """
        if self.data.mass is None and self.data.queries.train == 0 and self.data.queries.density == 0:
            raise ValueError("queries.train may be zero only for a dense mass dump (data.mass)")
        if self.data.selection is not None:
            if not self.run.collection:
                raise ValueError("data.selection requires an explicit run.collection")
            if self.data.tokenized.documents:
                raise ValueError("data.selection cannot also name tokenized.documents")
            if self.data.tokenized.held_out_fraction != Corpus.HELD_OUT_FRACTION:
                raise ValueError("data.selection uses the corpus held-out fraction")
        if self.data.tokenized.lengths is not None and not (
            self.data.selection or self.data.tokenized.documents
        ):
            raise ValueError(
                "tokenized.lengths selects buckets - provide data.selection or tokenized.documents"
            )
        if (
            not self.data.selection
            and not self.data.tokenized.documents
            and not (self.data.tokenized.train or self.data.tokenized.evaluate)
        ):
            raise ValueError("tokenized.train and tokenized.evaluate cannot both be zero")


def load(path: str | Path, overrides: list[str] | None = None) -> list[TeacherConfig]:
    """Read an extraction file and return the extractions it describes.

    Args:
        path: Path to the extraction yaml.
        overrides: Dotted `key=value` assignments applied after everything else.

    Returns:
        One `TeacherConfig` per extraction, in sweep order.
    """
    return config_loader.load(TeacherConfig, path, overrides, name_path="run.name")
