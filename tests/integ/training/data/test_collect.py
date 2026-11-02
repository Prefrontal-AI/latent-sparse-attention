"""Collected shards on disk, and a lock file that tells a collection that reproduced them from one that did not."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lsa import assets
from lsa.assets.constants import Collector
from lsa.data import lock
from lsa.data.collect import Collection, long_documents, write_shards


def card(path: Path) -> assets.SourceCard:
    return assets.SourceCard(
        name="books",
        path=str(path),
        licence="per-document",
        dataset="a/b",
        revision="r",
        collector=Collector.TEXT,
        min_tokens=3,
        count_tokenizer="t/t",
        count_tokenizer_revision="r",
    )


def words(texts):
    return [len(text.split()) for text in texts]


# A word is four characters, so a document of three words clears the 2.5 characters a token the prefilter allows for
DOCUMENTS = [(f"d{index}", " ".join(["word"] * (3 + index)), 3 + index) for index in range(5)]


def test_documents_are_written_in_shards_with_the_marker_last(tmp_path, monkeypatch):
    monkeypatch.setattr(Collection, "SHARD_DOCS", 2)
    out = tmp_path / "books"

    summary = write_shards(DOCUMENTS, out, "books", "per-document")

    assert summary == {"n_docs": 5, "kept_tokens": sum(n for _, _, n in DOCUMENTS), "n_shards": 3}
    assert sorted(path.name for path in out.iterdir()) == [
        "_DONE",
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_00002.parquet",
    ]
    assert json.loads((out / "_DONE").read_text()) == summary
    table = pq.read_table(out / "shard_00000.parquet")
    assert table.column_names == ["doc_id", "source", "license", "n_tokens", "text"]
    assert table.column("doc_id").to_pylist() == ["d0", "d1"]
    assert set(table.column("source").to_pylist()) == {"books"}


def test_a_collection_never_replaces_one_that_is_there(tmp_path):
    out = tmp_path / "books"
    write_shards(DOCUMENTS, out, "books", "per-document")

    with pytest.raises(FileExistsError, match="move it aside"):
        write_shards(DOCUMENTS, out, "books", "per-document")


def test_a_collection_that_did_not_finish_leaves_no_folder_that_looks_collected(tmp_path):
    def dies():
        yield DOCUMENTS[0]
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        write_shards(dies(), tmp_path / "books", "books", "per-document")

    assert not (tmp_path / "books").exists()
    with pytest.raises(FileExistsError, match="did not finish"):
        write_shards(DOCUMENTS, tmp_path / "books", "books", "per-document")


def collected(tmp_path, documents):
    out = tmp_path / "books"
    write_shards(long_documents([(d, t) for d, t, _ in documents], words, min_tokens=3), out, "books", "x")
    return card(out)


def test_a_collection_that_reproduces_the_lock_verifies(tmp_path):
    source = collected(tmp_path, DOCUMENTS)
    locked = lock.write(source, str(tmp_path / "books.lock.parquet"))

    assert lock.verify(source, str(locked)).ok


def test_the_lock_records_each_document_in_order_with_its_digest_and_tokens(tmp_path):
    source = collected(tmp_path, DOCUMENTS)

    table = pq.read_table(lock.write(source, str(tmp_path / "books.lock.parquet")))

    assert table.column("doc_id").to_pylist() == [doc_id for doc_id, _, _ in DOCUMENTS]
    assert table.column("n_tokens").to_pylist() == [n for _, _, n in DOCUMENTS]
    assert table.schema.metadata[b"source"] == b"books"
    assert table.column("sha256").to_pylist()[0] == lock.digest(DOCUMENTS[0][1])


def test_a_collection_whose_text_drifted_is_refused_with_the_documents_named(tmp_path):
    locked = lock.write(collected(tmp_path, DOCUMENTS), str(tmp_path / "books.lock.parquet"))
    drifted = [(doc_id, text + " extra", n) for doc_id, text, n in DOCUMENTS[:2]] + DOCUMENTS[2:]
    other = tmp_path / "again"
    other.mkdir()

    with pytest.raises(ValueError, match="does not match its lock"):
        lock.verify(collected(other, drifted), str(locked))


def test_a_lock_is_refused_for_a_source_it_is_not_the_lock_of(tmp_path):
    source = collected(tmp_path, DOCUMENTS)
    locked = lock.write(source, str(tmp_path / "books.lock.parquet"))
    other = assets.SourceCard(
        **{**{field: getattr(source, field) for field in source.__dataclass_fields__}, "name": "papers"}
    )

    with pytest.raises(ValueError, match="is the lock of books"):
        lock.verify(other, str(locked))


def test_a_source_with_no_shards_cannot_be_locked(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shard"):
        lock.write(card(tmp_path / "missing"), str(tmp_path / "x.parquet"))
