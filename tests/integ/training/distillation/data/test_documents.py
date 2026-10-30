"""Where extraction gets its documents - that a window is not a document, and that the split is not a guess."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lsa import assets
from lsa.data import manifests
from lsa.training.data.corpus import document_share
from lsa.training.distillation.constants import Split
from lsa.training.distillation.data.documents import CorpusDocuments, PreparedDocuments, document_split
from lsa.training.distillation.data.teacher_plan import window_length
from tests.helpers.training.data.fake_corpus import write_corpus

SAMPLE_LENGTH = 64


def tokenized(tmp_path, rows_per_document=3, documents=8):
    """A corpus whose every document is held out, so the held-out hash is not what is under test"""
    asset = write_corpus(
        tmp_path / "corpus",
        sources=("arxiv",),
        documents=documents,
        rows_per_document=rows_per_document,
        sample_length=SAMPLE_LENGTH,
    )
    return CorpusDocuments(asset, sources=["arxiv"], held_out_fraction=1.0)


def test_the_split_is_a_property_of_the_document_rather_than_of_a_position():
    """Taking more documents later must not move one that is already cached"""
    first = [document_split(f"doc-{index}") for index in range(50)]
    again = [document_split(f"doc-{index}") for index in range(50)]

    assert first == again
    assert set(first) == {Split.TRAIN, Split.EVAL}


def test_the_split_is_independent_of_the_held_out_hash():
    """Salted apart, so being held out of pretraining says nothing about which side a document lands on"""
    ids = [f"doc-{index}" for index in range(400)]
    held = [document_share(one) < 0.25 for one in ids]
    scored = [document_split(one) == Split.EVAL for one in ids]

    agree = sum(one == other for one, other in zip(held, scored, strict=True))
    assert 0.4 < agree / len(ids) < 0.9


def test_a_document_is_one_entry_however_many_windows_it_packs_into(tmp_path):
    corpus = tokenized(tmp_path, rows_per_document=3, documents=8)

    assert len(corpus.held_out["arxiv"]) == 8
    assert [entry.start for entry in corpus.held_out["arxiv"]] == [0, 3, 6, 9, 12, 15, 18, 21]


def test_every_window_of_a_document_lands_on_the_same_side(tmp_path):
    """A window is not a document, so a split that could separate two of them would leak one into the other"""
    corpus = tokenized(tmp_path, rows_per_document=3, documents=8)

    chosen = corpus.select(train=8, evaluate=8, windows=3)

    sides = {}
    for window in chosen:
        sides.setdefault(window.document, set()).add(window.split)
    assert all(len(seen) == 1 for seen in sides.values())


def test_windows_of_one_document_are_consecutive_rows_of_it(tmp_path):
    corpus = tokenized(tmp_path, rows_per_document=3, documents=8)

    chosen = [one for one in corpus.select(train=8, evaluate=8, windows=3) if one.start == 0]

    assert [one.window for one in chosen] == [0, 1, 2]
    assert [one.name for one in chosen] == ["arxiv_r0_w0", "arxiv_r0_w1", "arxiv_r0_w2"]


def test_the_two_sides_share_no_document(tmp_path):
    corpus = tokenized(tmp_path, rows_per_document=2, documents=20)

    chosen = corpus.select(train=6, evaluate=4)

    training = {one.document for one in chosen if one.split == Split.TRAIN}
    scored = {one.document for one in chosen if one.split == Split.EVAL}
    assert training and scored
    assert not training & scored


def test_a_side_takes_the_count_it_asks_for(tmp_path):
    corpus = tokenized(tmp_path, rows_per_document=1, documents=40)

    chosen = corpus.select(train=5, evaluate=2)

    assert sum(one.split == Split.TRAIN for one in chosen) == 5
    assert sum(one.split == Split.EVAL for one in chosen) == 2


def test_a_source_contributes_what_it_has_rather_than_inventing_any(tmp_path):
    """`games` holds six held-out documents against `swe`'s 681, and asking for more must not invent one"""
    corpus = tokenized(tmp_path, rows_per_document=1, documents=4)

    chosen = corpus.select(train=50, evaluate=50)

    assert len(chosen) == 4


def test_a_window_reads_its_own_row(tmp_path):
    corpus = tokenized(tmp_path, rows_per_document=3, documents=4)

    chosen = [one for one in corpus.select(train=4, evaluate=4, windows=3) if one.start == 0]
    rows = [corpus.tokens(one, SAMPLE_LENGTH) for one in chosen]

    assert all(row.shape == (1, SAMPLE_LENGTH) for row in rows)
    assert not torch.equal(rows[0], rows[1])


def a_selection_corpus(tmp_path, rows_per_document=4, documents=200):
    """A corpus under the real held-out hash, big enough that both sides hold documents"""
    asset = write_corpus(
        tmp_path / "corpus",
        sources=("arxiv",),
        documents=documents,
        rows_per_document=rows_per_document,
        sample_length=SAMPLE_LENGTH,
    )
    return CorpusDocuments(asset, sources=["arxiv"])


def one_of(corpus, held_out):
    return next(
        document
        for document in corpus.document_extents["arxiv"]
        if (document_share(document) < corpus.held_out_fraction) == held_out
    )


def an_entry(document, split, length=SAMPLE_LENGTH, window=0):
    return {"source": "arxiv", "document": document, "split": split, "length": length, "window": window}


def test_a_selection_window_longer_than_a_row_reads_consecutive_rows_of_its_document(tmp_path):
    corpus = a_selection_corpus(tmp_path)
    document = one_of(corpus, held_out=False)
    (selected,) = corpus.select_documents(
        [an_entry(document, Split.TRAIN, length=2 * SAMPLE_LENGTH, window=1)]
    )

    ids = corpus.tokens(selected, selected.length)

    start, _ = corpus.document_extents["arxiv"][document]
    first = (start + 2) * SAMPLE_LENGTH
    assert torch.equal(ids[0], torch.arange(first, first + 2 * SAMPLE_LENGTH))


def test_a_selection_window_shorter_than_a_row_is_a_slice_of_one(tmp_path):
    corpus = a_selection_corpus(tmp_path)
    document = one_of(corpus, held_out=False)
    (selected,) = corpus.select_documents(
        [an_entry(document, Split.TRAIN, length=SAMPLE_LENGTH // 2, window=3)]
    )

    ids = corpus.tokens(selected, selected.length)

    start, _ = corpus.document_extents["arxiv"][document]
    first = start * SAMPLE_LENGTH + 3 * SAMPLE_LENGTH // 2
    assert torch.equal(ids[0], torch.arange(first, first + SAMPLE_LENGTH // 2))


def test_a_selection_window_is_named_by_its_length(tmp_path):
    corpus = a_selection_corpus(tmp_path)
    document = one_of(corpus, held_out=False)
    (selected,) = corpus.select_documents(
        [an_entry(document, Split.TRAIN, length=2 * SAMPLE_LENGTH, window=1)]
    )

    start, _ = corpus.document_extents["arxiv"][document]
    assert selected.name == f"arxiv_r{start}_l{2 * SAMPLE_LENGTH}_w1"


def test_a_selection_cannot_train_on_a_held_out_document(tmp_path):
    corpus = a_selection_corpus(tmp_path)

    with pytest.raises(ValueError, match="is held out"):
        corpus.select_documents([an_entry(one_of(corpus, held_out=True), Split.TRAIN)])


def test_a_selection_cannot_score_on_a_document_language_model_training_reads(tmp_path):
    corpus = a_selection_corpus(tmp_path)

    with pytest.raises(ValueError, match="on the training side"):
        corpus.select_documents([an_entry(one_of(corpus, held_out=False), Split.EVAL)])


def test_a_selection_window_past_the_rows_its_document_was_tokenized_into_is_refused(tmp_path):
    corpus = a_selection_corpus(tmp_path, rows_per_document=4)

    with pytest.raises(ValueError, match="does not fit"):
        corpus.select_documents(
            [an_entry(one_of(corpus, held_out=False), Split.TRAIN, length=2 * SAMPLE_LENGTH, window=2)]
        )


def test_a_selection_naming_a_document_the_dataset_lacks_is_refused(tmp_path):
    corpus = a_selection_corpus(tmp_path)

    with pytest.raises(ValueError, match="holds no arxiv document"):
        corpus.select_documents([an_entry("no-such-document", Split.TRAIN)])


def prepared_card(root, sources=("niah_single_1",)):
    """A prepared benchmark's card, which is where the layout and the prompt schema live"""
    return assets.BenchmarkCard(
        name="ruler_test",
        licence="Apache-2.0",
        generator="NVIDIA/RULER",
        revision="abc",
        path=str(root),
        file="{task}_{length}.jsonl",
        tokenizer="glm53_flash",
        tasks=list(sources),
        prompt=["input", "gen_prefix"],
        join="\n",
    )


def write_samples(card, source, length, rows):
    """Write a prepared file the way a harness writes one, and its manifest entry"""
    path = Path(card.file_path(task=source, length=length))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"index": index, "input": haystack, "gen_prefix": question, "length": length})
        for index, (haystack, question) in enumerate(rows)
    ]
    path.write_text("\n".join(lines) + "\n")
    entry = {"rows": len(rows), "sha256": "", "tokenizer": "t", "date": "2026-10-01"}
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(card)})
    manifests.write_benchmark_file(card.data_root(), f"{path.parent.name}/{path.name}", record)


class WordTokenizer:
    """One id per word, which is enough to check what was tokenised and how much of it was kept"""

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [len(word) for word in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


def test_a_prepared_dataset_is_one_collection_named_after_itself(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc")])
    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    assert corpus.collection == "ruler_test"
    assert [one.name for one in corpus.select(4)] == ["niah_single_1_s0_w0"]


def test_every_prepared_prompt_scores_and_none_of_them_trains(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc"), ("d ee", "fff")])
    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    assert {one.split for one in corpus.select(2)} == {Split.EVAL}


def test_a_source_with_no_prepared_file_is_not_opened(tmp_path):
    card = prepared_card(tmp_path / "ruler", sources=("niah_single_1", "absent"))
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc")])

    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    assert corpus.sources == ["niah_single_1"]


def test_a_prompt_is_the_haystack_and_the_question_joined_as_the_asset_says(tmp_path):
    """Joined bare, the haystack's last word and the question's first become one token that never occurs"""
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc")])
    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    ids = corpus.tokens(corpus.select(1)[0], 64)

    assert ids.tolist() == [[1, 2, 3]]


def test_a_prompt_is_read_at_its_prepared_length_rather_than_the_tokenized_window(tmp_path):
    """A 1M prompt read at the tokenized side's 64k window kept its question and lost the needle's haystack"""
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 1000000, [("a bb", "ccc")])
    corpus = PreparedDocuments(card, None, 1000000, WordTokenizer())
    run = SimpleNamespace(tokenized=SimpleNamespace(tokens_per_document=65536))

    assert window_length(run, corpus.select(1)[0]) == 1000000


def test_an_overlong_prompt_loses_its_front_rather_than_its_question(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb ccc", "dddd")])
    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    ids = corpus.tokens(corpus.select(1)[0], 2)

    assert ids.tolist() == [[3, 4]]


def test_a_prepared_benchmark_reads_its_row_counts_off_the_manifest(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc"), ("d ee", "fff")])

    assert PreparedDocuments(card, None, 64, WordTokenizer()).sample_counts == {"niah_single_1": 2}


def test_a_prepared_benchmark_under_another_card_is_refused(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc")])

    with pytest.raises(ValueError, match=r"differs in \['join'\]"):
        PreparedDocuments(dataclasses.replace(card, join=" "), None, 64, WordTokenizer())


def test_prepared_rows_take_those_rows_in_place_of_the_first(tmp_path):
    """The served records' doc ids, so a cached prompt is the one a served sample scored"""
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a", "b"), ("c", "d"), ("e", "f")])
    corpus = PreparedDocuments(card, None, 64, WordTokenizer())

    assert [one.start for one in corpus.select(1, rows=[2, 0])] == [2, 0]
    with pytest.raises(ValueError, match="holds 3 samples"):
        corpus.select(1, rows=[3])


def test_a_prepared_root_reads_a_cell_of_the_same_layout_elsewhere(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    elsewhere = dataclasses.replace(card, path=str(tmp_path / "other" / "ruler"))
    write_samples(elsewhere, "niah_single_1", 64, [("a", "b"), ("c", "d")])

    corpus = PreparedDocuments(card, None, 64, WordTokenizer(), root=str(tmp_path / "other"))

    assert corpus.sample_counts == {"niah_single_1": 2}


class ChatTokenizer(WordTokenizer):
    """Words as ids, a template that wraps the user turn, and a think block encoded as two ids after a BOS"""

    bos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        (turn,) = messages
        return f"u {turn['content']} a" + (" think" if add_generation_prompt else "")

    def encode(self, text):
        return [99, 7, 8]


def test_a_chat_prompt_is_the_served_request_with_the_think_block_appended_as_sglang_does(tmp_path):
    card = prepared_card(tmp_path / "ruler")
    write_samples(card, "niah_single_1", 64, [("a bb", "ccc")])
    corpus = PreparedDocuments(card, None, 64, ChatTokenizer(), join=" ", chat=True)

    ids = corpus.tokens(corpus.select(1)[0], 64)

    # `u a bb ccc a think` by word length, then the think block encoded with its BOS dropped
    assert ids.tolist() == [[1, 1, 2, 3, 1, 5, 7, 8]]
