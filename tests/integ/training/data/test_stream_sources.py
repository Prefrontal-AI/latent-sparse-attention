"""The three storage backends - what each reads, and how it holds out a split."""

import dataclasses

import numpy as np
import pytest

from lsa.training.constants import CorpusSplit, Field
from lsa.training.data.sources import PackedRows, ParquetText, TokenShards, open_source
from tests.helpers.training.data.fake_corpus import write_corpus
from tests.helpers.training.data.fake_sources import VOCAB_SIZE, write_shards, write_text


def test_token_shards_are_read_in_slabs_of_one_shard(tmp_path):
    source = TokenShards(write_shards(tmp_path / "fw", shards=2, tokens=25), slab_tokens=10)
    assert source.units == 2
    slabs = [record[0].tolist() for record in source.read(1, 0)]
    assert [len(slab) for slab in slabs] == [10, 10, 5]
    assert np.concatenate(slabs).tolist() == list(range(1000, 1025))


def test_a_slab_can_be_skipped_into(tmp_path):
    source = TokenShards(write_shards(tmp_path / "fw", shards=1, tokens=25), slab_tokens=10)
    assert [slab[0][0] for slab in source.read(0, 2)] == [20]


def test_the_held_out_side_of_shards_is_the_validation_file(tmp_path):
    card = write_shards(tmp_path / "fw", validation_tokens=33)
    held_out = TokenShards(card, CorpusSplit.HELD_OUT)
    assert held_out.units == 1
    assert next(held_out.read(0, 0))[0][0] == 40_000


def test_shards_are_sorted_by_name_and_typed_by_their_metadata(tmp_path):
    source = TokenShards(write_shards(tmp_path / "fw", shards=11))
    assert [path.name for path in source.paths] == [f"train_{i:05d}.bin" for i in range(11)]
    assert source.dtype == np.dtype("uint16")


def test_a_directory_with_no_shards_is_refused(tmp_path):
    card = write_shards(tmp_path / "fw")
    for path in (tmp_path / "fw").glob("train_*.bin"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="no train shards"):
        TokenShards(card)


def test_shards_without_metadata_are_refused(tmp_path):
    card = write_shards(tmp_path / "fw")
    (tmp_path / "fw" / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        TokenShards(card)


def test_packed_rows_are_one_sample_a_unit(tmp_path):
    asset = write_corpus(tmp_path / "corpus", sample_length=64)
    source = PackedRows(asset, seq_len=16, sources=["arxiv"], held_out_fraction=0.0)
    assert source.units == 16 * 4
    (record,) = list(source.read(0, 0))
    assert record[0].shape == (17,)
    assert list(source.read(0, 1)) == []


def test_text_is_one_row_group_a_unit_and_read_in_chunks(tmp_path):
    source = ParquetText(write_text(tmp_path / "text"), held_out_fraction=0.0, documents_per_chunk=4)
    assert source.units == 4
    chunks = list(source.read(0, 0))
    assert [len(chunk) for chunk in chunks] == [4, 2]
    assert [chunk[0] for chunk in list(source.read(0, 1))] == [chunks[1][0]]


def test_the_two_sides_of_the_text_split_are_a_partition_by_document_id(tmp_path):
    card = write_text(tmp_path / "text")
    sides = {
        split: [
            text
            for unit in range(4)
            for chunk in ParquetText(card, held_out_fraction=0.5, split=split).read(unit, 0)
            for text in chunk
        ]
        for split in CorpusSplit.ALL
    }
    assert sorted(sides[CorpusSplit.TRAIN] + sides[CorpusSplit.HELD_OUT]) == sorted(
        text
        for unit in range(4)
        for chunk in ParquetText(card, held_out_fraction=0.0).read(unit, 0)
        for text in chunk
    )
    assert sides[CorpusSplit.TRAIN] and sides[CorpusSplit.HELD_OUT]


def test_a_parquet_file_without_the_text_column_is_refused_at_the_boundary(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    card = write_text(tmp_path / "text")
    pq.write_table(pa.table({"doc_id": ["a"], "body": ["x"]}), tmp_path / "text" / "shard_00009.parquet")
    with pytest.raises(ValueError, match=r"shard_00009.parquet has columns .* needs \['text'\]"):
        ParquetText(card)


def test_a_folder_with_no_parquet_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    card = dataclasses.replace(write_text(tmp_path / "text"), path=str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="no shard_"):
        ParquetText(card)


def test_each_card_kind_opens_its_own_source(tmp_path):
    shards = open_source(write_shards(tmp_path / "fw"), seq_len=8)
    rows = open_source(write_corpus(tmp_path / "corpus", sample_length=64), seq_len=16)
    text = open_source(write_text(tmp_path / "text"), seq_len=8)
    assert [type(one) for one in (shards, rows, text)] == [TokenShards, PackedRows, ParquetText]
    assert [one.field for one in (shards, rows, text)] == [Field.TOKENS, Field.TOKENS, Field.TEXT]


def test_token_shards_read_their_writers_metadata_in_place(tmp_path):
    manifest = write_shards(tmp_path / "fw", shards=3).manifest()
    assert (manifest.vocab_size, manifest.dtype, manifest.num_train_shards) == (VOCAB_SIZE, "uint16", 3)
