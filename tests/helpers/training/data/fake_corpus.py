"""A tokenized corpus at test size, structurally identical to the one on disk.

Same two files per source - a `uint32` bin of fixed-length rows and a parquet manifest naming the document
each row came from - plus the `corpus.json` manifest preparing a corpus writes, so the loader is exercised
on the format rather than on a mock of it.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lsa import assets
from lsa.data import manifests


def write_corpus(
    root,
    sources=("arxiv", "code"),
    documents=8,
    rows_per_document=2,
    sample_length=64,
    vocab_size=1_000_000,
):
    """Write one tokenized corpus and return the card it was tokenized under.

    Token ids encode where they came from, so a test can tell one row from another.
    """
    root.mkdir(parents=True, exist_ok=True)
    for source_index, name in enumerate(sources):
        rows = documents * rows_per_document
        ids = np.arange(rows * sample_length, dtype=np.uint32).reshape(rows, sample_length)
        ids += np.uint32(source_index * 1_000_000)
        ids.tofile(root / f"{name}.bin")

        manifest = {
            "doc_id": [f"{name}-doc-{i}" for i in range(documents)],
            "row_start": [i * rows_per_document for i in range(documents)],
            "n_samples": [rows_per_document] * documents,
        }
        pq.write_table(pa.table(manifest), root / f"{name}.manifest.parquet")

    card = assets.TokenizedCard(
        name=root.name,
        path=str(root),
        sources=list(sources),
        tokenizer="qwen35_4b",
        sample_length=sample_length,
    )
    manifests.write_tokenized(
        root,
        manifests.TokenizedManifest(
            declared=manifests.declared(card),
            tokenizer="/fake/tokenizer",
            tokenizer_revision="",
            vocab_size=vocab_size,
            sample_length=sample_length,
            sources=list(sources),
            rows=dict.fromkeys(sources, documents * rows_per_document),
            documents=dict.fromkeys(sources, documents),
        ),
    )
    return assets.unregistered(root)
