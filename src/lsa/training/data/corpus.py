"""Select training and held-out samples from shared tokenized source rows."""

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from lsa.assets import TokenizedCard
from lsa.data.corpus import SourceRows, document_extents
from lsa.training.constants import Corpus, CorpusSplit, Loss


def document_share(document_id: str, salt: str = "") -> float:
    """Map a document id to a fixed point in `[0, 1)`, which every decision taken on a document reads.

    Unsalted it is the held-out hash every stage splits the corpus by. Every other decision - which side of
    the distillation boundary a held-out document falls on, the order a mix takes documents in - salts it
    apart, so none of them says anything about another.

    Args:
        document_id: The manifest's own id for a document, or that id and a slot.
        salt: What sets this decision apart from the others, or nothing for the held-out hash.

    Returns:
        The document's position in the hash space.
    """
    digest = hashlib.md5(f"{salt}{document_id}".encode()).hexdigest()[: Corpus.HASH_DIGITS]
    return int(digest, 16) / Corpus.HASH_SPACE


@dataclass(frozen=True)
class Sample:
    """Identify a sequence by its source, starting row and slice within that row."""

    source: int
    row: int
    slice_index: int


class TokenizedCorpus(SourceRows):
    """The tokenized sources on disk, split into the rows a run may read."""

    def __init__(
        self,
        corpus: TokenizedCard,
        sources: list[str] | None = None,
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
        split: str = CorpusSplit.TRAIN,
        excluded: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        """Open every source whose bin exists under the corpus and take one side of the split.

        Args:
            corpus: The corpus to read, which carries its own root and row length.
            sources: Sources to open, or None for every one the card names that is present.
            held_out_fraction: Share of documents held out of training.
            split: `"train"` for the documents the targets never saw, `"held_out"` for the ones they did.
            excluded: `(source, document id)` pairs to leave out whichever side they fall on - the documents
                a targets extraction read, when the held-out side is a validation set.

        Raises:
            FileNotFoundError: No source bin exists under the root.
            ValueError: `split` is neither half.
        """
        if split not in CorpusSplit.ALL:
            raise ValueError(f"split must be one of {CorpusSplit.ALL} - got {split!r}")
        super().__init__(corpus, sources)
        self.split = split
        self.excluded = excluded

        # Per source, the document id of every row this side holds
        self.documents = [self._documents(name, held_out_fraction) for name in self.sources]
        self.rows = [list(documents) for documents in self.documents]

    def _documents(self, source: str, held_out_fraction: float) -> dict[int, str]:
        """Return the tokenized rows of one source that fall on this run's side of the split.

        Args:
            source: Source name.
            held_out_fraction: Share of documents held out.

        Returns:
            Each row's document id, keyed by row index in manifest order.
        """
        documents: dict[int, str] = {}
        for document, (start, count) in document_extents(self.root, source).items():
            held_out = document_share(document) < held_out_fraction
            if held_out != (self.split == CorpusSplit.HELD_OUT) or (source, document) in self.excluded:
                continue
            for row in range(start, start + count):
                documents[row] = document
        return documents

    def index(self, seq_len: int) -> list[Sample]:
        """Enumerate every sample the split holds at one sequence length.

        Args:
            seq_len: Tokens per sample, dividing a row or spanning a whole number of rows.

        Returns:
            One `Sample` per sequence, source-major then row-major.

        Raises:
            ValueError: Neither the row nor `seq_len` divides the other.
        """
        if seq_len <= 0 or (self.sample_length % seq_len and seq_len % self.sample_length):
            raise ValueError(
                f"seq_len must divide the tokenized row or span whole rows - {seq_len} vs {self.sample_length}"
            )
        if seq_len > self.sample_length:
            return self._long_samples(seq_len // self.sample_length)
        per_row = self.sample_length // seq_len
        return [
            Sample(source, row, slice_index)
            for source, rows in enumerate(self.rows)
            for row in rows
            for slice_index in range(per_row)
        ]

    def _long_samples(self, rows_per_sequence: int) -> list[Sample]:
        """Join whole consecutive rows within each document.

        Args:
            rows_per_sequence: Rows a complete sequence spans.

        Returns:
            Non-overlapping sequences, leaving each document's incomplete tail out.
        """
        samples: list[Sample] = []
        for source, documents in enumerate(self.documents):
            document_rows: dict[str, list[int]] = {}
            for row, document in documents.items():
                document_rows.setdefault(document, []).append(row)
            for rows in document_rows.values():
                for offset in range(0, len(rows) - rows_per_sequence + 1, rows_per_sequence):
                    first = rows[offset]
                    last = rows[offset + rows_per_sequence - 1]
                    if last - first == rows_per_sequence - 1:
                        samples.append(Sample(source, first, 0))
        return samples

    def read(self, sample: Sample, seq_len: int) -> torch.Tensor:
        """Read one sample plus the token its last label needs.

        Args:
            sample: Which slice of which row of which source.
            seq_len: Tokens per sample.

        Returns:
            Token ids [n] int64, including the next label when its row or document continues.
        """
        import numpy as np

        rows = self.source_rows(self.sources[sample.source])
        if seq_len > self.sample_length:
            row_count = seq_len // self.sample_length
            tokens = rows[sample.row : sample.row + row_count].reshape(-1)
            next_row = sample.row + row_count
            documents = self.documents[sample.source]
            if documents.get(next_row) == documents[sample.row]:
                tokens = np.concatenate((tokens, rows[next_row, :1]))
            return torch.from_numpy(tokens.astype(np.int64))
        start = sample.slice_index * seq_len
        # One past the sample, so the last position has a next token to predict
        stop = min(start + seq_len + 1, self.sample_length)
        return torch.from_numpy(rows[sample.row, start:stop].astype(np.int64))


def batch_of(windows: list[Any], seq_len: int) -> dict[str, Any]:
    """Lay windows end to end as one batch in the trainer's own contract.

    Args:
        windows: Each a sequence of ids, `seq_len + 1` of them or fewer at the end of a row.
        seq_len: Tokens per sequence.

    Returns:
        Flat `input`, `labels`, `positions` and `padding_mask`, and the count of labels the loss will see.
    """
    import numpy as np

    size = len(windows) * seq_len
    # [size] int64 token ids, zero where a sequence ran short and the label is ignored anyway
    input_ids = torch.zeros(size, dtype=torch.long)
    labels = torch.full((size,), Loss.IGNORE_INDEX, dtype=torch.long)
    positions = torch.arange(size, dtype=torch.long) % seq_len
    padding_mask = torch.ones(size, dtype=torch.bool)

    for sequence, window in enumerate(windows):
        tokens = torch.as_tensor(np.asarray(window, dtype=np.int64))
        taken = min(tokens.numel() - 1, seq_len)
        start = sequence * seq_len
        input_ids[start : start + taken] = tokens[:taken]
        labels[start : start + taken] = tokens[1 : taken + 1]
        padding_mask[start : start + taken] = False

    return {
        "input": input_ids,
        "labels": labels,
        "positions": positions,
        "padding_mask": padding_mask,
        "num_valid_tokens": int((labels != Loss.IGNORE_INDEX).sum()),
    }


def assemble_batch(corpus: TokenizedCorpus, samples: list[Sample], seq_len: int) -> dict[str, Any]:
    """Lay samples end to end as one batch in the trainer's own contract.

    Args:
        corpus: Where the samples are read from.
        samples: The sequences, in batch order.
        seq_len: Tokens per sequence.

    Returns:
        Flat `input`, `labels`, `positions` and `padding_mask`, and the count of labels the loss will see.
    """
    return batch_of([corpus.read(sample, seq_len) for sample in samples], seq_len)


def one_per_document(corpus: TokenizedCorpus, seq_len: int, count: int, seed: int) -> list[Sample]:
    """Choose a fixed set of samples, no two from one document.

    One per document because the samples of one document share a topic and a style, and a set that
    counts a long book eight times measures that book rather than the corpus.

    Args:
        corpus: The side of the split to choose from.
        seq_len: Tokens per sample.
        count: How many to choose.
        seed: Seed of the order they are chosen in.

    Returns:
        The samples, in the order they were chosen.

    Raises:
        ValueError: The split holds fewer than `count` documents.
    """
    samples = corpus.index(seq_len)
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed))
    chosen: list[Sample] = []
    seen: set[tuple[int, str]] = set()
    for position in order.tolist():
        sample = samples[position]
        document = (sample.source, corpus.documents[sample.source][sample.row])
        if document in seen:
            continue
        seen.add(document)
        chosen.append(sample)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"the split holds {len(seen)} documents - asked for {count}")
