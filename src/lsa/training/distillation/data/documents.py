"""Where the documents the targets are extracted from come from.

One dataset card is one collection in the cache, and every document in it carries the domain it came from,
which of its windows this is, and which side of the train/eval boundary it falls on. A run then names a
collection, a side and optionally the domains it wants, and can express nothing else - a set that reads the
training side cannot reach an evaluation document by any spelling.

Two hashes of the document id decide where a document may be used, so both are properties of the corpus
rather than of a seed or of a config. `document_share` holds a document out of language-model training, and
`document_split` then sends it to one side of the distillation boundary. The side a document falls on is
written into its path.

A document selection is the other way to choose - a file naming each document, its side and the length of its one
window. It trains on documents language-model training also reads and scores on held-out ones alone, and
reading one refuses an entry that says otherwise.

A document is identified by where it starts on disk and by which window this is, never by its position in a
filtered list, which would move whenever the number of documents taken per source changed.

Tokenized text and prepared prompts are two readers because the mechanics genuinely differ - a memory map over
`uint32` rows against a tokenizer over jsonl - but neither knows anything about the other and both answer the
same three questions. What a prepared benchmark's files are called and which of its fields compose a prompt
are its card's to state, not this module's, and how many rows each file holds is its manifest's.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from lsa.assets import BenchmarkCard, TokenizedCard
from lsa.data.benchmarks import BenchmarkFiles
from lsa.data.corpus import SourceRows, document_extents
from lsa.training.constants import Corpus
from lsa.training.data.corpus import document_share
from lsa.training.distillation.constants import Split


def document_split(document_id: str, eval_share: float = Split.EVAL_SHARE) -> str:
    """Return whether a held-out document trains the index map or scores it.

    Salted apart from `document_share`, so being held out of language-model training says nothing about
    which side of this a document falls on.

    Args:
        document_id: The manifest's own id for a document.
        eval_share: Share of held-out documents that score rather than train.

    Returns:
        `Split.TRAIN` or `Split.EVAL`.
    """
    return Split.EVAL if document_share(document_id, Split.SALT) < eval_share else Split.TRAIN


@dataclass(frozen=True)
class Prompt:
    """One rendered prepared sample - the text a tokenizer was given and the tokens it made of it.

    `spans` holds where the leading tokens sit in `text`, as [start, end) characters, and is None unless asked
    for. The tokens appended after the text, the closed think block, have no span, so `spans` can be shorter than `ids`.
    `sample` is the row as the harness wrote it, which carries the answer.
    """

    text: str
    ids: torch.Tensor
    spans: torch.Tensor | None
    sample: dict[str, Any]


@dataclass(frozen=True)
class Selected:
    """One window of one document, named and split before anything is read.

    `name` is what the cache calls it and `document` is what the dataset calls it. The second cannot be the
    first, because a document id carries `/` and `:` and a file name cannot - the collection manifest is
    what binds the pair, and it is also what a run's selection is resolved against.
    """

    source: str
    document: str
    window: int
    split: str
    name: str
    # Where the document starts on disk - its first tokenized row, or its line in a prepared file
    start: int
    # Tokens into the document the window starts at, and how many it holds - 0 for what the run takes
    offset: int = 0
    length: int = 0


class Documents(Protocol):
    """What an extraction asks of any dataset it reads."""

    @property
    def collection(self) -> str:
        """The cache directory this dataset's documents live under."""

    def tokens(self, selected: Selected, length: int) -> torch.Tensor:
        """Return one document's token ids [1, tokens] int64."""


class CorpusDocuments(SourceRows):
    """A tokenized corpus, read as whole documents and split at the distillation boundary.

    The files are the ones [the corpus reader](../../data/corpus.py) reads, and the difference is the unit: a
    training run reads rows, and an extraction reads one window of one document across as many rows as it
    spans.
    """

    # `<source>_r<row>_w<window>` - where the document starts in the bin, and which window of it this is
    NAME: str = "{source}_r{start}_w{window}"
    # A selection window also carries its length, since one document may be read at several across datasets
    SELECTED_NAME: str = "{source}_r{start}_l{length}_w{window}"

    def __init__(
        self,
        asset: TokenizedCard,
        sources: list[str] | None = None,
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
        eval_share: float = Split.EVAL_SHARE,
        collection: str | None = None,
    ) -> None:
        """Open every source whose bin exists under the dataset.

        Args:
            asset: The dataset to read, which carries its own root and row length.
            sources: Sources to open, or None for every one the card names that is present.
            held_out_fraction: Share of documents held out of language-model training.
            eval_share: Share of those that score the index map rather than train it.
            collection: An inline selection's collection, or the dataset's name.

        Raises:
            FileNotFoundError: No source bin exists under the root.
        """
        super().__init__(asset, sources)
        self.asset = asset
        self._collection = collection or asset.name
        self.held_out_fraction = held_out_fraction
        # Document id to its first tokenized row and its row count, per source, in manifest order
        self.document_extents = {name: document_extents(self.root, name) for name in self.sources}
        self.held_out = {name: self._held_out(name, eval_share) for name in self.sources}
        # Row a document starts at, to the number of whole windows it was tokenized into
        self.window_counts = {name: dict(self.document_extents[name].values()) for name in self.sources}

    @property
    def collection(self) -> str:
        """The cache directory this dataset's documents live under, which is the dataset's own name."""
        return self._collection

    def _held_out(self, source: str, eval_share: float) -> list[Selected]:
        """Return every held-out document of a source, one entry each.

        A long document is tokenized into several consecutive rows. Which of them an extraction reads is
        `select`'s business; this enumerates the documents themselves, so that nothing downstream can
        mistake two windows of one document for two documents.

        Args:
            source: Source name.
            eval_share: Share of held-out documents that score rather than train.

        Returns:
            One entry per held-out document naming its first window, in manifest order.
        """
        return [
            self._window(source, document, document_split(document, eval_share), start, 0)
            for document, (start, _) in self.document_extents[source].items()
            if document_share(document) < self.held_out_fraction
        ]

    def _window(self, source: str, document: str, split: str, start: int, window: int) -> Selected:
        """Build one window's entry.

        Args:
            source: Source name.
            document: The corpus's own id for the document.
            split: Which side it falls on.
            start: The document's first tokenized row.
            window: Which window of it this is.

        Returns:
            The entry, named the way the cache names it.
        """
        return Selected(
            source=source,
            document=document,
            window=window,
            split=split,
            name=self.NAME.format(source=source, start=start, window=window),
            start=start,
            offset=window * self.sample_length,
        )

    def select(self, train: int, evaluate: int, windows: int = 1) -> list[Selected]:
        """Choose the documents an extraction covers, by side and by source.

        Counts are per source and per side, so a source contributes the number of training documents asked
        for whatever its hash happened to do, and a source holding fewer than that contributes what it has.

        Windows past the first are the most correlated data available - another 65,536 tokens of a document
        already read - so this defaults to one and the breadth comes from taking more documents.

        Args:
            train: Training documents to take per source.
            evaluate: Evaluation documents to take per source.
            windows: Windows to take from each chosen document.

        Returns:
            One entry per cached document, source-major and then in manifest order.
        """
        wanted = {Split.TRAIN: train, Split.EVAL: evaluate}
        chosen: list[Selected] = []
        for source in self.sources:
            taken = dict.fromkeys(Split.BOTH, 0)
            for held in self.held_out[source]:
                if taken[held.split] >= wanted[held.split]:
                    continue
                taken[held.split] += 1
                for window in range(min(windows, self.window_counts[source][held.start])):
                    chosen.append(self._window(source, held.document, held.split, held.start, window))
        return chosen

    def select_documents(self, entries: list[dict[str, Any]]) -> list[Selected]:
        """Resolve a selection's entries into windows, refusing one that breaks the split.

        A selection trains on documents language-model training also reads and scores on held-out documents alone,
        so an entry on the wrong side of the held-out hash is refused rather than extracted. So is a window
        that runs past the rows its document was tokenized into.

        Args:
            entries: The selection's entries naming this dataset, each carrying its source, document, split,
                length and window.

        Returns:
            One entry per window, in the selection's order.

        Raises:
            ValueError: An entry names a document this dataset does not hold, sits on the wrong side of the
                held-out hash, or does not fit inside its document.
        """
        chosen = []
        for entry in entries:
            source, document, split = entry["source"], str(entry["document"]), entry["split"]
            length, window = int(entry["length"]), int(entry["window"])
            if document not in self.document_extents.get(source, {}):
                raise ValueError(f"{self.collection} holds no {source} document {document!r}")
            held_out = document_share(document) < self.held_out_fraction
            if held_out != (split == Split.EVAL):
                side = "held out" if held_out else "on the training side"
                raise ValueError(
                    f"{source} document {document!r} is {side} and the selection puts it in {split}"
                )
            start, rows = self.document_extents[source][document]
            if length <= 0 or (window + 1) * length > rows * self.sample_length:
                raise ValueError(
                    f"{source} document {document!r} holds {rows * self.sample_length} tokens - "
                    f"window {window} at {length} does not fit"
                )
            chosen.append(
                Selected(
                    source=source,
                    document=document,
                    window=window,
                    split=split,
                    name=self.SELECTED_NAME.format(source=source, start=start, length=length, window=window),
                    start=start,
                    offset=window * length,
                    length=length,
                )
            )
        return chosen

    def tokens(self, selected: Selected, length: int) -> torch.Tensor:
        """Read one window of one document, across as many consecutive rows as it spans.

        Args:
            selected: The window to read.
            length: Tokens to take from where it starts.

        Returns:
            Token ids [1, length] int64.
        """
        import numpy as np

        rows = self.source_rows(selected.source)
        first_row = selected.start + selected.offset // self.sample_length
        column = selected.offset % self.sample_length
        row_count = -(-(column + length) // self.sample_length)
        # [rows * sample_length] the rows the window touches, which are one contiguous span of the document
        spanned = rows[first_row : first_row + row_count].reshape(-1)
        return torch.from_numpy(spanned[column : column + length].astype(np.int64))[None]


class PreparedDocuments:
    """Prompts a harness already rendered, read as documents rather than as an evaluation.

    Every prompt scores and none of them trains, so the whole of this lands on the evaluation side. The map
    is fit on natural prose and read on synthetic retrieval, which is the shift the numbers are for.

    The file layout and the fields that compose a prompt are the card's, because they describe that benchmark
    and nothing else, and the rows each file holds are what its manifest measured. A prompt's token count
    belongs to the tokenizer it was rendered against, so a file prepared for one model is the wrong length for
    another and this reads only its own card's.
    """

    # A prepared prompt is one window by construction
    NAME: str = "{source}_s{start}_w{window}"

    def __init__(
        self,
        asset: BenchmarkCard,
        sources: list[str] | None,
        length: int,
        tokenizer: Any,
        root: str | None = None,
        join: str | None = None,
        chat: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Open every named task that has a prepared file at this length.

        Args:
            asset: The benchmark to read, which carries its root, its layout and its prompt schema.
            sources: Tasks to open, or None for every one the card names.
            length: The prepared length, which names the file rather than truncating anything.
            tokenizer: The model's tokenizer.
            root: A data root of the card's layout to read in place of the card's own, or None.
            join: What joins the card's prompt fields in place of the card's own separator, or None.
            chat: Read a prompt as a served request - one user turn in the model's chat template, answered past
                a closed think block - rather than as bare text.
            chat_template_kwargs: What the template renders a chat prompt with, which then closes the think block
                itself - or None to append the closed block.

        Raises:
            FileNotFoundError: None of the named tasks has a prepared file the manifest records.
            ValueError: The files were prepared under a card that differs from this one.
        """
        self.asset = asset
        self.length = length
        self.tokenizer = tokenizer
        self.join = join
        self.chat = chat
        self.chat_template_kwargs = chat_template_kwargs or {}
        wanted = list(sources or asset.tasks)
        self.files = BenchmarkFiles(asset, root)
        self.sample_counts = {
            source: count
            for source in wanted
            if (count := self.files.row_count(task=source, length=length)) is not None
        }
        self.sources = [source for source in wanted if source in self.sample_counts]
        if not self.sources:
            raise FileNotFoundError(
                f"no prepared samples of {asset.name} under {asset.path} - looked for {wanted} at {length}"
            )

    @property
    def collection(self) -> str:
        """The cache directory this dataset's documents live under, which is the dataset's own name."""
        return self.asset.name

    def path(self, source: str) -> Path:
        """Return the prepared file one task's samples are read from.

        Args:
            source: Task name.

        Returns:
            The file path, which may not exist.
        """
        return self.files.path(task=source, length=self.length)

    def select(
        self, samples: int, rows: list[int] | None = None, source_rows: dict[str, list[int]] | None = None
    ) -> list[Selected]:
        """Choose the prepared samples an extraction covers.

        A prompt is read at its prepared length, so the corpus side's window never cuts it - a 1M prompt read
        at a 64k window would keep the question and lose the haystack the needle sits in.

        Args:
            samples: Samples to take per source, the first that many.
            rows: The rows of each file to take in place of the first `samples`, or None.
            source_rows: The rows to take of each named file, in place of `rows` - a file not named takes none.

        Returns:
            One entry per document, source-major, every one of them on the evaluation side.

        Raises:
            ValueError: A row is past what a file holds, or `source_rows` names a task with no file.
        """
        if source_rows is not None:
            absent = sorted(set(source_rows) - set(self.sources))
            if absent:
                raise ValueError(f"source_rows names {absent}, which have no prepared file at {self.length}")
        wanted: dict[str, list[int]] = {}
        for source in self.sources:
            if source_rows is not None:
                wanted[source] = source_rows.get(source, [])
            elif rows is not None:
                wanted[source] = rows
            else:
                wanted[source] = list(range(min(samples, self.sample_counts[source])))
            outside = [row for row in wanted[source] if not 0 <= row < self.sample_counts[source]]
            if outside:
                raise ValueError(f"{source} holds {self.sample_counts[source]} samples - asked for {outside}")
        return [
            Selected(
                source=source,
                document=f"{source}:{index}",
                window=0,
                split=Split.EVAL,
                name=self.NAME.format(source=source, start=index, window=0),
                start=index,
                length=self.length,
            )
            for source in self.sources
            for index in wanted[source]
        ]

    def tokens(self, selected: Selected, length: int) -> torch.Tensor:
        """Tokenise one prepared sample.

        A prompt longer than `length` loses its front rather than its end, because the question is the last
        thing in it and the tail positions a report scores are the rows that read back for the answer.

        Args:
            selected: The sample to read.
            length: Tokens to keep.

        Returns:
            Token ids [1, tokens] int64, which is `length` or shorter.
        """
        return self.prompt(selected, offsets=False).ids[:, -length:]

    def prompt(self, selected: Selected, offsets: bool) -> "Prompt":
        """Render and tokenise one prepared sample as the accuracy harness serves it.

        Args:
            selected: The sample to read.
            offsets: Also keep where each token sits in the text, which needs a tokenizer that reports it.

        Returns:
            The text, its tokens and, when asked, their character spans.
        """
        sample = self.sample(selected.source, selected.start)
        text = self.files.prompt(sample, self.join)
        suffix: list[int] = []
        if self.chat:
            # What the accuracy harness sends with think off, as sglang builds it for `continue_final_message`:
            # the user turn rendered with the generation prompt, then the closed think block encoded on its own
            # and appended - so GLM's prompt ends `<|assistant|><think><think></think>`, one token more than the
            # template's own continuation would give, and the count is the served record's `prompt_tokens`
            from lsa.evaluation.accuracy.client import NO_THINK

            messages = [{"role": "user", "content": text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )
            # A template told to think or not closes the block itself, as the served request has it
            suffix = [] if self.chat_template_kwargs else self.tokenizer.encode(NO_THINK)
            if suffix and suffix[0] == self.tokenizer.bos_token_id:
                suffix = suffix[1:]
        extra = {"return_offsets_mapping": True} if offsets else {}
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, **extra)
        ids = encoded.input_ids
        if suffix:
            ids = torch.cat([ids, torch.tensor([suffix], dtype=ids.dtype)], dim=1)
        spans = encoded.offset_mapping[0].to(torch.int64) if offsets else None
        return Prompt(text, ids.to(torch.int64), spans, sample)

    def sample(self, source: str, index: int) -> dict[str, Any]:
        """Read one line of a prepared file.

        The files are hundreds of megabytes of prompt, so a line is found by walking rather than by reading
        the whole of it into memory.

        Args:
            source: Source name.
            index: Line to read, counting from zero.

        Returns:
            The sample, as the harness wrote it.

        Raises:
            ValueError: The source has fewer samples than that.
        """
        return self.files.sample(index, task=source, length=self.length)
