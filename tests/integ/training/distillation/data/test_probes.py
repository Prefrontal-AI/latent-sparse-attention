"""What a probe extraction queries and what it holds each query to - read off a RULER row and nothing else."""

import pytest
import torch

from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data.documents import PreparedDocuments
from lsa.training.distillation.data.teacher import layer_blob
from lsa.training.distillation.data.teacher_config import (
    DataConfig,
    PreparedConfig,
    QueryConfig,
    RunConfig,
    TeacherConfig,
)
from lsa.training.distillation.data.teacher_plan import (
    document_queries,
    line_around,
    probe_prompt,
    query_positions,
)
from tests.helpers.training.distillation.data.fake_ruler import (
    LENGTH,
    SpanTokenizer,
    prepared_card,
    write_rows,
)

TAIL = 3
QUESTION = "Question: what are the top words? Answer:"
# One word each, so a word's index is its token's
LIST = [
    "cat",
    "p",
    "dog",
    "q",
    "cat",
    "r",
    "cat",
    "p",
    "dog",
    "x",
    "cat",
    "q",
    "dog",
    "r",
    "cat",
    "p",
    "q",
    "dog",
    "r",
    "p",
    "q",
    "r",
]
ANSWERS = ["cat", "dog"]
# The user turn's opening, the header, then a number and a word an item - past which the question starts
FIRST_WORD = 3
SUFFIX = 2


def a_run():
    return TeacherConfig(
        data=DataConfig(queries=QueryConfig(tail=TAIL, targets="ruler")), run=RunConfig(output="targets")
    )


def a_list_prompt():
    items = " ".join(f"{i + 1}. {word}" for i, word in enumerate(LIST))
    return f"Words: {items}", QUESTION


def a_corpus(tmp_path, source, rows):
    card = prepared_card(tmp_path / "ruler", [source])
    write_rows(card, source, rows)
    corpus = PreparedDocuments(card, None, LENGTH, SpanTokenizer(), join=" ", chat=True)
    return corpus, corpus.select(1)[0]


def item_position(i):
    return FIRST_WORD + 2 * i


def test_a_list_is_queried_at_every_repeat_of_an_answer_word_and_as_many_others(tmp_path):
    corpus, selected = a_corpus(tmp_path, "cwe", [(*a_list_prompt(), ANSWERS)])

    ids, probes = probe_prompt(a_run(), corpus, selected, LENGTH)

    kinds = [Cache.QUERY_KINDS[kind] for kind in probes.kinds.tolist()]
    repeats = sum(LIST[:i].count(word) > 0 for i, word in enumerate(LIST) if word in ANSWERS)
    assert repeats == 7
    assert kinds.count("answer") == repeats and kinds.count("control") == repeats
    assert kinds[-TAIL:] == ["question"] * TAIL
    assert probes.positions[-TAIL:].tolist() == list(range(ids.shape[1] - TAIL, ids.shape[1]))
    assert bool((probes.positions[1:] > probes.positions[:-1]).all())


def test_an_item_is_aimed_at_the_first_tokens_of_its_words_earlier_occurrences(tmp_path):
    corpus, selected = a_corpus(tmp_path, "cwe", [(*a_list_prompt(), ANSWERS)])

    _, probes = probe_prompt(a_run(), corpus, selected, LENGTH)

    asked = probes.positions[:-TAIL].tolist()
    for row, position in enumerate(asked):
        i = next(item for item in range(len(LIST)) if item_position(item) == position)
        earlier = [item_position(j) for j in range(i) if LIST[j] == LIST[i]]
        held = probes.target_key_ids[row]
        assert held[held >= 0].tolist() == earlier
        assert earlier, "a query with nothing earlier to find is not a probe"
    assert bool((probes.target_key_ids[-TAIL:] == Cache.NO_KEY).all())
    assert probes.line_key_ids is None


def test_a_words_first_occurrence_is_never_a_query(tmp_path):
    corpus, selected = a_corpus(tmp_path, "cwe", [(*a_list_prompt(), ANSWERS)])

    _, probes = probe_prompt(a_run(), corpus, selected, LENGTH)

    firsts = {item_position(LIST.index(word)) for word in set(LIST)}
    assert not firsts & set(probes.positions[:-TAIL].tolist())


def test_the_controls_are_the_rows_own_draw(tmp_path):
    rows = [(*a_list_prompt(), ANSWERS)] * 2
    corpus, _ = a_corpus(tmp_path, "cwe", rows)
    first, second = corpus.select(2)

    again = probe_prompt(a_run(), corpus, first, LENGTH)[1]
    alike = probe_prompt(a_run(), corpus, first, LENGTH)[1]
    other = probe_prompt(a_run(), corpus, second, LENGTH)[1]

    assert torch.equal(again.positions, alike.positions)
    controls = Cache.QUERY_KINDS.index("control")
    drawn = {tuple(p.positions[p.kinds == controls].tolist()) for p in (again, other)}
    assert len(drawn) == 2


def a_needle_prompt():
    filler = " ".join(f"w{i}" for i in range(20))
    return (
        f"{filler}. One special number for key is: 1234. {filler}\nWhat is the special number for key?",
        "Answer:",
    )


def test_a_needle_task_is_queried_at_its_question_and_aimed_at_the_value_and_its_sentence(tmp_path):
    corpus, selected = a_corpus(tmp_path, "niah_single_2", [(*a_needle_prompt(), ["1234"])])

    ids, probes = probe_prompt(a_run(), corpus, selected, LENGTH)

    words = ["user", *" ".join(a_needle_prompt()).split()]
    value = words.index("1234.")
    sentence = words.index("One")
    assert probes.positions.tolist() == list(range(ids.shape[1] - TAIL, ids.shape[1]))
    assert set(probes.kinds.tolist()) == {Cache.QUERY_KINDS.index("question")}
    assert probes.target_key_ids.shape == (TAIL, 1)
    assert bool((probes.target_key_ids == value).all())
    assert probes.line_key_ids is not None
    assert probes.line_key_ids[0].tolist() == list(range(sentence, value + 1))


def test_a_line_is_its_row_where_the_haystack_has_rows_and_its_sentence_where_it_has_none():
    rows = "One number for a is: 1.\nOne number for b is: 22.\nOne number for c is: 3.\nQuestion"
    at = rows.index("22")
    assert rows[slice(*line_around(rows, at, at + 2))] == "One number for b is: 22."

    prose = "Filler words here. One number for b is: 22. More filler. Question"
    at = prose.index("22")
    assert prose[slice(*line_around(prose, at, at + 2))] == "One number for b is: 22."


def test_an_answer_that_is_not_in_its_prompt_is_refused(tmp_path):
    corpus, selected = a_corpus(tmp_path, "niah_single_2", [(*a_needle_prompt(), ["9999"])])

    with pytest.raises(ValueError, match="does not occur"):
        probe_prompt(a_run(), corpus, selected, LENGTH)


def test_a_prompt_cut_at_the_front_moves_its_queries_and_targets_with_it(tmp_path):
    corpus, selected = a_corpus(tmp_path, "niah_single_2", [(*a_needle_prompt(), ["1234"])])
    whole, probes = probe_prompt(a_run(), corpus, selected, LENGTH)
    value = int(probes.target_key_ids[0, 0])

    kept, cut = probe_prompt(a_run(), corpus, selected, whole.shape[1] - 5)

    assert kept.shape[1] == whole.shape[1] - 5
    assert int(cut.target_key_ids[0, 0]) == value - 5
    assert cut.positions.tolist() == [p - 5 for p in probes.positions.tolist()]


def test_a_cut_through_the_needle_is_refused(tmp_path):
    corpus, selected = a_corpus(tmp_path, "niah_single_2", [(*a_needle_prompt(), ["1234"])])

    with pytest.raises(ValueError, match="hold a query or a target"):
        probe_prompt(a_run(), corpus, selected, 30)


def test_probes_need_a_prepared_prompt():
    with pytest.raises(ValueError, match="needs `prepared` documents"):
        probe_prompt(a_run(), object(), None, 10)


class TokenizedWindow:
    """A tokenized document, which has tokens and no row - the tokenized side every extraction plans"""

    def tokens(self, selected, length):
        return torch.arange(length)[None]


def test_the_tokenized_side_of_a_probe_extraction_is_queried_by_the_draw():
    ids, positions, probes = document_queries(a_run(), TokenizedWindow(), None, 4096, 2)
    assert probes is None
    assert ids.shape == (1, 4096)
    assert torch.equal(positions, query_positions(a_run(), 4096, 2))


def test_a_prepared_prompt_under_targets_is_queried_at_its_probes(tmp_path):
    corpus, selected = a_corpus(tmp_path, "cwe", [(*a_list_prompt(), ANSWERS)])
    ids, positions, probes = document_queries(a_run(), corpus, selected, LENGTH, 0)
    assert probes is not None
    assert torch.equal(positions, probes.positions)


def test_a_layer_file_keeps_the_kinds_and_targets_beside_the_attention(tmp_path):
    """The blob grows the probe fields and the meta names the kinds, and a cache of no training rows reads"""
    from types import SimpleNamespace

    from lsa.evaluation.adapter.layer import TeacherView
    from lsa.sglang.settings.design import IndexerConfig
    from lsa.training.distillation.data.cache import Document, layer_path, write
    from lsa.training.distillation.data.teacher_plan import Probes

    length, heads, head_dim = 96, 2, 6
    attention = torch.nn.Linear(1, 1)
    attention.layer_idx, attention.scaling = 3, head_dim**-0.5
    positions = torch.arange(length - TAIL, length)

    def view(hidden, rows, embeddings):
        index = torch.randn(1, 1)
        return TeacherView(
            index, index, torch.randn(heads, len(rows), head_dim), torch.randn(1, length, head_dim)
        )

    reader = SimpleNamespace(attention=attention, teacher_view=view)
    probes = Probes(
        positions,
        torch.zeros(TAIL, dtype=torch.int8),
        torch.tensor([[40, 41]] * TAIL, dtype=torch.int32),
        None,
    )
    indexer = IndexerConfig(
        window_size=8, sink_size=4, bucket_size=4, bucket_cap=8, pool_size=None, top_k=8, top_p=1
    )
    run = TeacherConfig(
        data=DataConfig(queries=QueryConfig(tail=TAIL, top=4, targets="ruler")),
        run=RunConfig(output="targets"),
    )
    run.model.indexer = indexer

    blob = layer_blob(reader, torch.randn(1, length, 12), None, positions, run, "doc", probes=probes)

    assert blob[Cache.QUERY_KIND].dtype is torch.int8 and blob[Cache.TARGET_KEY_IDS].shape == (TAIL, 2)
    assert Cache.LINE_KEY_IDS not in blob
    assert blob[Cache.META][Cache.QUERY_KIND_NAMES] == list(Cache.QUERY_KINDS)
    assert blob[Cache.META][Cache.TRAIN_QUERIES] == 0

    path = layer_path(tmp_path, "ruler", "eval", "doc", 3)
    write(path, blob)
    document = Document(path, indexer)
    assert document.train_query_count == 0 and len(document.eval_rows()) == TAIL
    assert torch.equal(document.target_key_ids, blob[Cache.TARGET_KEY_IDS])
    assert document.line_key_ids is None


def test_a_prepared_selection_may_draw_its_rows_a_task_at_a_time(tmp_path):
    card = prepared_card(tmp_path / "ruler", ["niah_single_2", "cwe"])
    for source in card.tasks:
        write_rows(card, source, [("a", "b", ["c"])] * 4)
    corpus = PreparedDocuments(card, None, LENGTH, SpanTokenizer())

    chosen = corpus.select(1, source_rows={"niah_single_2": [3, 1], "cwe": [2]})

    assert [(one.source, one.start) for one in chosen] == [
        ("niah_single_2", 3),
        ("niah_single_2", 1),
        ("cwe", 2),
    ]
    with pytest.raises(ValueError, match="holds 4 samples"):
        corpus.select(1, source_rows={"cwe": [4]})


def test_the_row_lists_and_the_query_mode_are_checked_when_the_config_is_read():
    with pytest.raises(ValueError, match="cannot both be set"):
        PreparedConfig(name="ruler", rows=[1], source_rows={"cwe": [1]})
    with pytest.raises(ValueError, match="does not"):
        PreparedConfig(name="ruler", sources=["cwe"], source_rows={"vt": [1]})
    with pytest.raises(ValueError, match="distinct rows"):
        PreparedConfig(name="ruler", source_rows={"cwe": [1, 1]})
    with pytest.raises(ValueError, match="queries.targets must be one of"):
        QueryConfig(targets="every-word")


def test_a_chat_prompt_rendered_with_template_arguments_closes_the_think_block_in_the_template(tmp_path):
    card = prepared_card(tmp_path / "ruler", ["niah_single_2"])
    write_rows(card, "niah_single_2", [(*a_list_prompt(), ANSWERS)])
    appended = PreparedDocuments(card, None, LENGTH, SpanTokenizer(), join=" ", chat=True)
    templated = PreparedDocuments(
        card,
        None,
        LENGTH,
        SpanTokenizer(),
        join=" ",
        chat=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    selected = appended.select(1)[0]
    appended_ids = appended.prompt(selected, offsets=False).ids
    templated_ids = templated.prompt(selected, offsets=False).ids
    # The closed block appended as its own tokens on one, rendered by the template as one more word on the other
    assert appended_ids[0, -2:].tolist() == [7, 8]
    assert templated_ids.shape[1] == appended_ids.shape[1] - 1
