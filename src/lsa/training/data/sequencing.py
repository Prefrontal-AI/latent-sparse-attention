"""The language-model task - documents into the windows a sequence is read from, and windows into a batch.

A source hands over ids or text. Text is tokenized here, with the run's tokenizer, and ids pass through, so the
one thing every source has in common by this point is a document as an array of ids.
A window is `seq_len + 1` ids - the inputs and the one id the last label needs - or fewer at the end of a document span.

**Concatenate** joins documents, each ended by the end-of-document id, into one stream and cuts it every
`seq_len` ids. A window starts where the last ended, one id shared, and what is left at the end of a record is the
carry that the next record continues from, so where a source cuts its records does not show in the windows.
**Span** cuts each document alone, and drops what is left past its last whole window, which is what keeps every
window one contiguous stretch of one document.
"""

from typing import Any

import numpy as np

from lsa.data.tokenizer import load_tokenizer
from lsa.training.constants import Field, Packing
from lsa.training.data.sources import Source


class TextEncoder:
    """A tokenizer that turns text into ids and knows the id that ends a document."""

    def __init__(self, tokenizer_path: str) -> None:
        """Load the fast tokenizer a snapshot holds.

        Args:
            tokenizer_path: The snapshot directory.

        Raises:
            ValueError: The tokenizer has no fast implementation or no end-of-document id.
        """
        self.tokenizer = load_tokenizer(tokenizer_path, require_fast=True)
        if self.tokenizer.eos_token_id is None:
            raise ValueError(f"{tokenizer_path} has no end-of-document id to join documents with")
        self.eos_id: int = int(self.tokenizer.eos_token_id)

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        """Tokenize documents without special tokens.

        Args:
            texts: The documents.

        Returns:
            Each document's ids [n] int64.
        """
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [np.asarray(one, dtype=np.int64) for one in ids]


class LanguageModelTask:
    """Documents of one source cut into windows, under one way of packing them."""

    def __init__(
        self, source: Source, seq_len: int, packing: str | None, encoder: TextEncoder | None
    ) -> None:
        """Hold the source's fields to what this task can make windows from.

        Args:
            source: Where the documents come from.
            seq_len: Tokens per sequence.
            packing: One of `Packing.ALL`, or None for the source's own first choice.
            encoder: What tokenizes text, which a source of text needs.

        Raises:
            ValueError: The packing is not one this source supports, or the source holds text and no encoder is given.
        """
        self.seq_len = seq_len
        self.packing = packing or source.packings[0]
        if self.packing not in source.packings:
            raise ValueError(
                f"{type(source).__name__} supports packing {source.packings} - got {self.packing!r}"
            )
        if source.field == Field.TEXT and encoder is None:
            raise ValueError(f"{type(source).__name__} holds text - the run needs a tokenizer to read it")
        self.encoder = encoder if source.field == Field.TEXT else None

    def documents(self, record: list[Any]) -> list[np.ndarray]:
        """Turn one record into documents of ids, each ended when concatenating text.

        Args:
            record: What the source yielded.

        Returns:
            The documents' ids, each [n].
        """
        if self.encoder is None:
            return [np.asarray(document) for document in record]
        encoded = self.encoder.encode(record)
        if self.packing == Packing.SPAN:
            return encoded
        return [np.append(ids, self.encoder.eos_id) for ids in encoded]

    def cut(self, carry: np.ndarray, documents: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        """Cut documents into windows.

        Args:
            carry: Ids left over from the record before, [n] int64, empty at the start of a unit.
            documents: This record's documents.

        Returns:
            The windows, each a view of at most `seq_len + 1` ids, and the carry the next record starts from.
        """
        length = self.seq_len
        if self.packing == Packing.SPAN:
            windows = [
                document[first * length : min(first * length + length + 1, len(document))]
                for document in documents
                for first in range(len(document) // length)
            ]
            return windows, carry
        joined = np.concatenate([carry, *documents])
        # One id past the window, so each window's last position has a next token to predict
        whole = (len(joined) - 1) // length if len(joined) > length else 0
        windows = [joined[first * length : first * length + length + 1] for first in range(whole)]
        return windows, joined[whole * length :].copy()
