import pytest

from lsa.evaluation.accuracy.grading import graphwalks

DOC = {"answer_nodes": ["a1", "b2", "c3"]}
EMPTY: dict[str, list[str]] = {"answer_nodes": []}


def scores(doc, reply):
    return graphwalks.process_results(doc, [reply])


def test_the_last_line_states_the_answer():
    assert scores(DOC, "Walking the edges...\nFinal Answer: [a1, b2, c3]") == {"f1": 1.0, "flexible_f1": 1.0}


def test_a_partial_answer_scores_its_set_f1():
    # Precision 1, recall 1/3
    assert scores(DOC, "Final Answer: [a1]")["f1"] == pytest.approx(0.5)
    # Precision 1/2, recall 1/3
    assert scores(DOC, "Final Answer: [a1, zz]")["f1"] == pytest.approx(0.4)


def test_repeats_count_once():
    assert scores(DOC, "Final Answer: [a1, a1, b2, c3]")["f1"] == 1.0


def test_an_empty_answer_to_an_empty_set_is_right():
    """The card's grader scores it 1, where lm-eval's port scores it 0"""
    assert scores(EMPTY, "Final Answer: []") == {"f1": 1.0, "flexible_f1": 1.0}
    assert scores(EMPTY, "Final Answer: [a1]")["f1"] == 0.0
    assert scores(DOC, "Final Answer: []")["f1"] == 0.0


def test_no_stated_answer_scores_zero_even_against_an_empty_set():
    assert scores(EMPTY, "There are no parents.") == {"f1": 0.0, "flexible_f1": 0.0}
    assert scores(EMPTY, "") == {"f1": 0.0, "flexible_f1": 0.0}


def test_only_the_last_line_counts_but_flexible_finds_an_earlier_one():
    reply = "Final Answer: [a1, b2, c3]\nHope this helps!"
    assert scores(DOC, reply) == {"f1": 0.0, "flexible_f1": 1.0}


def test_trailing_whitespace_does_not_hide_the_last_line():
    assert scores(DOC, "Final Answer: [a1, b2, c3]\n\n")["f1"] == 1.0


def test_quoted_ids_count_only_under_the_flexible_reading():
    assert scores(DOC, "Final Answer: ['a1', 'b2', 'c3']") == {"f1": 0.0, "flexible_f1": 1.0}


def test_the_answer_read_back_is_the_strict_one():
    assert graphwalks.answer_nodes("Final Answer: [a1,  b2 ,]") == ["a1", "b2"]
    assert graphwalks.answer_nodes("Final Answer: []") == []
    assert graphwalks.answer_nodes("Final Answer: a1, b2") is None
