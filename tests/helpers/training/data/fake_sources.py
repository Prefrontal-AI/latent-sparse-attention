"""Token shards and parquet text at test size, structurally identical to what is on disk.

Ids are consecutive, so a test can tell a window from another and where in a shard it was cut.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lsa import assets

VOCAB_SIZE = 50_257


def write_shards(root, shards=8, tokens=70, validation_tokens=33):
    """Write a directory of uint16 shards with the metadata its writer keeps, and return its card.

    Shard `i` holds the ids `i * 1000 ... i * 1000 + tokens - 1`.
    """
    root.mkdir(parents=True, exist_ok=True)
    for index in range(shards):
        (np.arange(tokens, dtype=np.uint16) + np.uint16(index * 1000)).tofile(root / f"train_{index:05d}.bin")
    (np.arange(validation_tokens, dtype=np.uint16) + np.uint16(40_000)).tofile(root / "val.bin")
    metadata = {
        "dataset": "fake/fineweb",
        "tokenizer": "gpt2",
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": VOCAB_SIZE - 1,
        "dtype": "uint16",
        "num_val_tokens": validation_tokens,
        "shard_size": tokens,
        "num_train_shards": shards,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return assets.ShardCard(name=root.name, path=str(root), tokenizer="gpt2")


def write_text(root, files=2, row_groups=2, documents=6):
    """Write a folder of parquet shards of `doc_id` and `text` columns, and return its card.

    A document's text is its id repeated, so a fake encoder can give it a length.
    """
    root.mkdir(parents=True, exist_ok=True)
    for file in range(files):
        rows = [(f"doc-{file}-{i}", chr(ord("a") + i) * (3 + i)) for i in range(row_groups * documents)]
        table = pa.table({"doc_id": [row[0] for row in rows], "text": [row[1] for row in rows]})
        pq.write_table(table, root / f"shard_{file:05d}.parquet", row_group_size=documents)
    return assets.SourceCard(
        name=root.name, path=str(root), licence="internal", collected_from="/nowhere/fake"
    )


class FakeEncoder:
    """A tokenizer of characters, which is what a test needs of one: ids that say which text they came from."""

    eos_id = 999

    def encode(self, texts):
        return [np.array([ord(character) for character in text], dtype=np.int64) for text in texts]
