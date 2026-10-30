"""Storage - what each kind of dataset card reads off disk, and nothing about training.

A source is a list of independent units - a shard, a parquet row group, one sample - and reading a unit yields its
records in order, so the layers above can deal units out to readers, skip into one and know where they are.
It reads either ids that something already wrote or text, says which it holds in `field`, and never renames
or tokenizes either: that is [the task](sequencing.py)'s.

| source | card | unit | record |
|---|---|---|---|
| `TokenShards` | shards | a shard file | a slab of consecutive ids |
| `PackedRows` | tokenized | one sample of a row | the sample's ids |
| `ParquetText` | source | a row group of one parquet file | a few documents' text |

The held-out side of a split is each kind's own: the validation file for shards, and a hash of the document id for
the other two, which is the hash every stage splits a corpus by.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from lsa import assets
from lsa.data.constants import Collected
from lsa.training.constants import Corpus, CorpusSplit, Field, Packing, Stream
from lsa.training.data.corpus import TokenizedCorpus, document_share


class Source(ABC):
    """Units of records, read by unit and resumable by record."""

    # Whether a record is ids or text
    field: str
    # The ways documents may be laid into sequences, the first being the one a run gets when it says nothing
    packings: tuple[str, ...]

    @property
    @abstractmethod
    def units(self) -> int:
        """Return how many units the source holds."""

    @abstractmethod
    def read(self, unit: int, start: int) -> Iterator[list[Any]]:
        """Yield one unit's records in order, each a list of documents - ids as an array, or text.

        Args:
            unit: Which unit.
            start: Records of the unit to skip.

        Returns:
            An iterator of records.
        """


class TokenShards(Source):
    """Flat ids another pipeline wrote, read as slabs of consecutive tokens that end up cut into windows."""

    field = Field.TOKENS
    packings = (Packing.CONCATENATE,)

    def __init__(
        self, card: assets.ShardCard, split: str = CorpusSplit.TRAIN, slab_tokens: int = Stream.SLAB_TOKENS
    ) -> None:
        """Take the training shards, or the validation file.

        Args:
            card: The shards to read.
            split: `"train"` for the training shards, `"held_out"` for the writer's validation file.
            slab_tokens: Tokens per record, or None for the usual.

        Raises:
            FileNotFoundError: The directory holds none of the files the split needs.
            ValueError: `split` is neither half.
        """
        if split not in CorpusSplit.ALL:
            raise ValueError(f"split must be one of {CorpusSplit.ALL} - got {split!r}")
        self.paths: list[Path] = (
            card.train_paths() if split == CorpusSplit.TRAIN else [card.validation_path()]
        )
        missing = [path for path in self.paths if not path.is_file()]
        if not self.paths or missing:
            raise FileNotFoundError(
                f"{card.name}: no {split} shards under {card.path} - missing {missing or 'all'}"
            )
        self.dtype = np.dtype(card.manifest().dtype)
        self.slab_tokens = slab_tokens or Stream.SLAB_TOKENS

    @property
    def units(self) -> int:
        """Return how many shard files there are."""
        return len(self.paths)

    def read(self, unit: int, start: int) -> Iterator[list[Any]]:
        """Yield a shard's slabs, mapped rather than read.

        Args:
            unit: Which shard.
            start: Slabs to skip.

        Returns:
            An iterator of one-document records [slab] of the shard's dtype.
        """
        data = np.memmap(self.paths[unit], dtype=self.dtype, mode="r")
        for first in range(start * self.slab_tokens, len(data), self.slab_tokens):
            yield [data[first : first + self.slab_tokens]]


class PackedRows(Source):
    """Read tokenized spans within documents, with the corpus's own held-out split."""

    field = Field.TOKENS
    packings = (Packing.SPAN,)

    def __init__(
        self,
        card: assets.TokenizedCard,
        seq_len: int,
        sources: list[str] | None = None,
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
        split: str = CorpusSplit.TRAIN,
    ) -> None:
        """Index the samples of the split.

        Args:
            card: The corpus to read.
            seq_len: Tokens per sample, dividing a row or spanning whole rows.
            sources: Sources to open, or None for every one present.
            held_out_fraction: Share of documents held out.
            split: Which side of the document hash.
        """
        self.corpus = TokenizedCorpus(card, sources=sources, held_out_fraction=held_out_fraction, split=split)
        self.seq_len = seq_len
        self.samples = self.corpus.index(seq_len)

    @property
    def units(self) -> int:
        """Return how many samples the split holds."""
        return len(self.samples)

    def read(self, unit: int, start: int) -> Iterator[list[Any]]:
        """Yield one sample, the slice and the id its last label needs.

        Args:
            unit: Which sample.
            start: Records to skip, of which there is one.

        Returns:
            An iterator of one record [ids] that is one document's span.
        """
        if start == 0:
            yield [self.corpus.read(self.samples[unit], self.seq_len).numpy()]


class ParquetText(Source):
    """Collected text, one parquet row group a unit, read a few documents at a time."""

    field = Field.TEXT
    packings = (Packing.CONCATENATE, Packing.SPAN)

    def __init__(
        self,
        card: assets.SourceCard,
        held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
        split: str = CorpusSplit.TRAIN,
        documents_per_chunk: int | None = None,
    ) -> None:
        """List the row groups of every file the card holds.

        Args:
            card: The source to read, a folder of parquet shards.
            held_out_fraction: Share of documents held out, by the hash of their id.
            split: Which side of the document hash.
            documents_per_chunk: Documents per record, or None for the usual.

        Raises:
            FileNotFoundError: The folder holds no parquet shard.
            ValueError: `split` is neither half, or a file lacks a column a read needs.
        """
        import pyarrow.parquet as pq

        if split not in CorpusSplit.ALL:
            raise ValueError(f"split must be one of {CorpusSplit.ALL} - got {split!r}")
        files = sorted(Path(card.path).glob(Collected.SHARD_GLOB))
        if not files:
            raise FileNotFoundError(f"{card.name}: no {Collected.SHARD_GLOB} under {card.path}")
        self.units_list: list[tuple[Path, int]] = []
        for file in files:
            parquet = pq.ParquetFile(file)
            lacking = {Collected.DOC_ID, Collected.TEXT} - set(parquet.schema_arrow.names)
            if lacking:
                raise ValueError(
                    f"{file} has columns {parquet.schema_arrow.names} - reading text needs {sorted(lacking)} too"
                )
            self.units_list += [(file, group) for group in range(parquet.metadata.num_row_groups)]
        self.held_out_fraction = held_out_fraction
        self.held_out = split == CorpusSplit.HELD_OUT
        self.documents_per_chunk = documents_per_chunk or Stream.DOCUMENTS_PER_CHUNK

    @property
    def units(self) -> int:
        """Return how many row groups there are."""
        return len(self.units_list)

    def read(self, unit: int, start: int) -> Iterator[list[Any]]:
        """Yield a row group's documents of this side of the split, a chunk at a time.

        Args:
            unit: Which row group.
            start: Chunks to skip.

        Returns:
            An iterator of records, each a list of document texts.
        """
        import pyarrow.parquet as pq

        file, group = self.units_list[unit]
        table = (
            pq.ParquetFile(file).read_row_group(group, columns=[Collected.DOC_ID, Collected.TEXT]).to_pydict()
        )
        texts = [
            text
            for identifier, text in zip(table[Collected.DOC_ID], table[Collected.TEXT], strict=True)
            if (document_share(str(identifier)) < self.held_out_fraction) == self.held_out
        ]
        for first in range(start * self.documents_per_chunk, len(texts), self.documents_per_chunk):
            yield texts[first : first + self.documents_per_chunk]


def open_source(
    card: assets.SourceCard | assets.TokenizedCard | assets.ShardCard,
    *,
    seq_len: int,
    split: str = CorpusSplit.TRAIN,
    sources: list[str] | None = None,
    held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
) -> Source:
    """Open the source a card's kind names.

    Args:
        card: A source, tokenized or shards card.
        seq_len: Tokens per sequence, which a tokenized corpus's samples are cut to.
        split: Which side of the split.
        sources: Sources of a tokenized corpus to read, which no other kind has.
        held_out_fraction: Share of documents held out, of the kinds that hold out by document id.

    Returns:
        The source.
    """
    if isinstance(card, assets.ShardCard):
        return TokenShards(card, split)
    if isinstance(card, assets.TokenizedCard):
        return PackedRows(card, seq_len, sources, held_out_fraction, split)
    return ParquetText(card, held_out_fraction, split)
