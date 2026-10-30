import re
import zlib

import pytest
import torch

from lsa.training.native.constants import Retrieval
from lsa.training.native.probes import ProbeBuilder, draw_facts, fits

SEQ_LEN = 4096
SPACE, STOP = 3, 4


def encode(text):
    """Words become ids of their own, digits their value, so a code is one token a digit as in Qwen3.5."""
    ids = []
    for piece in re.findall(r"\d|[A-Za-z]+|\.| ", text):
        if piece.isdigit():
            ids.append(10 + int(piece))
        elif piece == " ":
            ids.append(SPACE)
        elif piece == ".":
            ids.append(STOP)
        else:
            ids.append(1000 + zlib.crc32(piece.encode()) % 5000)
    return ids


def document():
    return torch.randint(10_000, 20_000, (SEQ_LEN,), generator=torch.Generator().manual_seed(1))


def build(distance=1024, index=0, seed=0, distractors=4):
    return ProbeBuilder(encode, seed, distractors, SEQ_LEN).build(document(), index, distance)


def statement(fact):
    return encode(f"{Retrieval.HEAD.format(name=fact.name)} {fact.code}.")


def test_the_sequence_keeps_its_length_and_the_distance_is_exact():
    for distance in (256, 1024, 2048):
        probe = build(distance)
        assert probe.tokens.shape == probe.control.shape == (SEQ_LEN,)
        assert probe.distance == distance
        assert probe.query_start - probe.needle_stop == distance


def test_the_needle_is_the_statement_and_the_query_ends_the_sequence_with_the_digits():
    probe = build()
    needle, *_ = draw_facts(0, 0, 5)
    assert probe.tokens[probe.needle_start : probe.needle_stop].tolist() == statement(needle)
    query = encode(f"{Retrieval.HEAD.format(name=needle.name)} ")
    assert probe.tokens[probe.query_start : probe.answer_start].tolist() == query
    assert probe.tokens[probe.answer_start :].tolist() == probe.answer.tolist() == encode(needle.code)
    assert probe.answer.numel() == Retrieval.DIGITS


def test_the_inputs_that_predict_the_answer_are_the_positions_before_it():
    probe = build()
    assert probe.answer_inputs.tolist() == list(range(SEQ_LEN - Retrieval.DIGITS - 1, SEQ_LEN - 1))
    assert probe.tokens[probe.answer_inputs + 1].tolist() == probe.answer.tolist()


def test_the_control_differs_from_the_probe_at_the_needle_alone():
    probe = build()
    changed = (probe.tokens != probe.control).nonzero().flatten()
    assert changed.numel() > 0
    assert changed.min() >= probe.needle_start
    assert changed.max() < probe.needle_stop


def test_the_control_holds_the_distractors_and_the_query_the_probe_does():
    probe = build()
    assert probe.control[probe.query_start :].tolist() == probe.tokens[probe.query_start :].tolist()
    for start in probe.distractor_starts:
        assert probe.control[start : start + 10].tolist() == probe.tokens[start : start + 10].tolist()


def test_distractors_state_other_names_and_overlap_nothing():
    probe = build()
    facts = draw_facts(0, 0, 5)
    assert len({fact.name for fact in facts}) == 5
    spans = [(probe.needle_start, probe.needle_stop), (probe.query_start, SEQ_LEN)]
    for fact, start in zip(facts[1:], probe.distractor_starts, strict=True):
        tokens = statement(fact)
        assert probe.tokens[start : start + len(tokens)].tolist() == tokens
        spans.append((start, start + len(tokens)))
    spans.sort()
    assert all(stop <= next_start for (_, stop), (next_start, _) in zip(spans, spans[1:], strict=False))


def test_everything_outside_the_spliced_spans_is_the_document():
    probe = build()
    changed = (probe.tokens != document()).nonzero().flatten()
    spliced = Retrieval.DIGITS + 9 + 30 * 5
    assert 0 < changed.numel() <= spliced + 100
    assert probe.tokens[: min(probe.needle_start, *probe.distractor_starts)].tolist() == (
        document()[: min(probe.needle_start, *probe.distractor_starts)].tolist()
    )


def test_a_seed_fixes_the_facts_and_the_positions():
    assert torch.equal(build().tokens, build().tokens)
    assert draw_facts(0, 3, 5) == draw_facts(0, 3, 5)
    assert draw_facts(1, 3, 5) != draw_facts(0, 3, 5)
    assert draw_facts(0, 4, 5) != draw_facts(0, 3, 5)
    assert not torch.equal(build(seed=1).tokens, build().tokens)


def test_a_distance_that_leaves_no_room_for_the_needle_is_refused_and_dropped():
    with pytest.raises(ValueError, match="no room"):
        build(SEQ_LEN)
    assert fits(256, SEQ_LEN, encode)
    assert fits(SEQ_LEN - 100, SEQ_LEN, encode)
    assert not fits(SEQ_LEN - 20, SEQ_LEN, encode)


def test_a_sequence_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="needs"):
        ProbeBuilder(encode, 0, 4, SEQ_LEN).build(document()[:-1], 0, 256)


def test_no_distractors_leaves_the_needle_alone():
    assert build(distractors=0).distractor_starts == ()
