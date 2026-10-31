"""The tokenized corpus loader - the split, the sharding, and exact resumption.

Resumption is the property worth a test: a run long enough to be interrupted is a run whose loss curve has
to be one curve, and a loader that resumes approximately makes it two.
"""

import dataclasses

import pytest

from lsa.training.data.corpus import (
    TokenizedCorpus,
    document_share,
    one_per_document,
)
from tests.helpers.training.data.fake_corpus import write_corpus

SAMPLE_LENGTH = 64


def corpus(tmp_path, split="train", held_out_fraction=0.02, **kwargs):
    asset = write_corpus(tmp_path / "corpus", sample_length=SAMPLE_LENGTH, **kwargs)
    return TokenizedCorpus(
        asset,
        sources=["arxiv", "code"],
        held_out_fraction=held_out_fraction,
        split=split,
    )


def test_the_split_is_a_partition(tmp_path):
    """Every row is on exactly one side, which is what keeps a teacher document out of a training set"""
    training = corpus(tmp_path, split="train", held_out_fraction=0.5)
    held_out = corpus(tmp_path, split="held_out", held_out_fraction=0.5)
    for source in range(2):
        both = set(training.rows[source]) & set(held_out.rows[source])
        every = set(training.rows[source]) | set(held_out.rows[source])
        assert not both
        assert every == set(range(16))


def test_the_held_out_split_is_a_property_of_the_document_id():
    assert document_share("abc") == document_share("abc")
    assert 0.0 <= document_share("abc") < 1.0
    assert document_share("abc") != document_share("abd")


def test_a_salt_sets_one_decision_apart_from_another():
    ids = [f"doc-{index}" for index in range(200)]
    assert [document_share(one) for one in ids] != [document_share(one, "salt:") for one in ids]
    assert document_share("abc", "salt:") == document_share("salt:abc")


def test_a_source_with_no_bin_is_refused(tmp_path):
    asset = dataclasses.replace(
        write_corpus(tmp_path / "corpus", sample_length=SAMPLE_LENGTH), path=str(tmp_path / "nothing")
    )
    with pytest.raises(FileNotFoundError, match="no tokenized sources"):
        TokenizedCorpus(asset, sources=["arxiv"])


def test_a_sample_is_a_slice_of_one_row(tmp_path):
    held = corpus(tmp_path)
    samples = held.index(seq_len=16)
    assert len(samples) == len(held.rows[0]) * 4 + len(held.rows[1]) * 4
    tokens = held.read(samples[0], seq_len=16)
    assert tokens.numel() == 17
    assert (tokens.diff() == 1).all()


def test_seq_len_must_divide_the_row(tmp_path):
    with pytest.raises(ValueError, match="seq_len must divide the tokenized row"):
        corpus(tmp_path).index(seq_len=7)


def test_an_excluded_document_is_on_neither_side(tmp_path):
    """How a validation set leaves out the documents a targets extraction read"""
    asset = write_corpus(tmp_path / "corpus", sample_length=SAMPLE_LENGTH)
    whole = TokenizedCorpus(asset, sources=["arxiv", "code"], held_out_fraction=0.0)
    less = TokenizedCorpus(
        asset,
        sources=["arxiv", "code"],
        held_out_fraction=0.0,
        excluded=frozenset({("arxiv", "arxiv-doc-3")}),
    )
    assert set(whole.rows[0]) - set(less.rows[0]) == {6, 7}
    assert less.rows[1] == whole.rows[1]
    assert "arxiv-doc-3" not in less.documents[0].values()


def test_a_document_is_excluded_only_under_its_own_source(tmp_path):
    asset = write_corpus(tmp_path / "corpus", sample_length=SAMPLE_LENGTH)
    less = TokenizedCorpus(
        asset, sources=["arxiv"], held_out_fraction=0.0, excluded=frozenset({("code", "arxiv-doc-3")})
    )
    assert len(less.rows[0]) == 16


def test_a_validation_set_takes_one_sample_per_document(tmp_path):
    held = corpus(tmp_path, held_out_fraction=0.0)
    chosen = one_per_document(held, seq_len=16, count=16, seed=0)
    documents = [(sample.source, held.documents[sample.source][sample.row]) for sample in chosen]
    assert len(set(documents)) == len(documents) == 16


def test_a_validation_set_is_fixed_by_its_seed(tmp_path):
    held = corpus(tmp_path, held_out_fraction=0.0)
    assert one_per_document(held, 16, 8, seed=1) == one_per_document(held, 16, 8, seed=1)
    assert one_per_document(held, 16, 8, seed=1) != one_per_document(held, 16, 8, seed=2)


def test_a_validation_set_larger_than_the_documents_is_refused(tmp_path):
    held = corpus(tmp_path, held_out_fraction=0.0)
    with pytest.raises(ValueError, match="holds 16 documents - asked for 17"):
        one_per_document(held, seq_len=16, count=17, seed=0)


@pytest.mark.parametrize("rows_per_sequence", [2, 3])
def test_long_samples_join_whole_rows_of_one_document_and_drop_its_tail(tmp_path, rows_per_sequence):
    held = corpus(tmp_path, held_out_fraction=0.0, documents=3, rows_per_document=5)
    length = rows_per_sequence * SAMPLE_LENGTH
    samples = held.index(length)
    assert len(samples) == 2 * 3 * (5 // rows_per_sequence)
    for sample in samples:
        documents = held.documents[sample.source]
        assert len({documents[row] for row in range(sample.row, sample.row + rows_per_sequence)}) == 1
        tokens = held.read(sample, length)
        assert tokens.numel() >= length
        assert (tokens.diff() == 1).all()
        assert tokens[0] == sample.source * 1_000_000 + sample.row * SAMPLE_LENGTH


def test_a_long_samples_next_label_never_crosses_a_document_boundary(tmp_path):
    held = corpus(tmp_path, held_out_fraction=0.0, documents=3, rows_per_document=4)
    samples = held.index(2 * SAMPLE_LENGTH)
    assert held.read(samples[0], 2 * SAMPLE_LENGTH).numel() == 2 * SAMPLE_LENGTH + 1
    assert held.read(samples[1], 2 * SAMPLE_LENGTH).numel() == 2 * SAMPLE_LENGTH


def test_long_sequences_do_not_join_short_documents_to_fill_the_context(tmp_path):
    held = corpus(tmp_path, held_out_fraction=0.0, documents=8, rows_per_document=1)
    assert held.index(2 * SAMPLE_LENGTH) == []
