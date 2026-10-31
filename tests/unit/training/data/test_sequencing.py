"""The language-model task - how documents become windows, and what it refuses at the boundary."""

import numpy as np
import pytest

from lsa.training.constants import Field, Packing
from lsa.training.data.sequencing import LanguageModelTask
from tests.helpers.training.data.fake_sources import FakeEncoder


class Source:
    def __init__(self, field=Field.TOKENS, packings=(Packing.CONCATENATE, Packing.SPAN)):
        self.field = field
        self.packings = packings


def task(packing=Packing.CONCATENATE, seq_len=4, source=None, encoder=None):
    return LanguageModelTask(source or Source(), seq_len, packing, encoder)


def ids(*lengths, start=0):
    out, next_id = [], start
    for length in lengths:
        out.append(np.arange(next_id, next_id + length, dtype=np.int64))
        next_id += length
    return out


def test_concatenated_windows_overlap_by_the_id_the_last_label_needs():
    windows, carry = task().cut(np.empty(0, dtype=np.int64), ids(10))
    assert [window.tolist() for window in windows] == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]
    assert carry.tolist() == [8, 9]


def test_where_a_record_ends_does_not_show_in_the_windows():
    whole = task()
    stream = np.arange(23, dtype=np.int64)
    expected, _ = whole.cut(np.empty(0, dtype=np.int64), [stream])
    seen, carry = [], np.empty(0, dtype=np.int64)
    for piece in (stream[:3], stream[3:4], stream[4:15], stream[15:]):
        windows, carry = whole.cut(carry, [piece])
        seen += windows
    assert [w.tolist() for w in seen] == [w.tolist() for w in expected]


def test_a_stream_shorter_than_one_window_is_all_carry():
    windows, carry = task().cut(np.empty(0, dtype=np.int64), ids(4))
    assert windows == []
    assert carry.tolist() == [0, 1, 2, 3]


def test_a_text_document_ends_with_the_end_id_when_concatenated():
    text = task(source=Source(Field.TEXT), encoder=FakeEncoder())
    (document,) = text.documents(["ab"])
    assert document.tolist() == [ord("a"), ord("b"), FakeEncoder.eos_id]


def test_a_span_is_one_document_and_its_remainder_is_dropped():
    spans = task(Packing.SPAN)
    windows, carry = spans.cut(np.empty(0, dtype=np.int64), ids(10, 3, 8, start=0))
    # 10 ids give two windows, 3 give none, 8 give two the second of which has no id past it
    assert [window.tolist() for window in windows] == [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
        [13, 14, 15, 16, 17],
        [17, 18, 19, 20],
    ]
    assert carry.size == 0


def test_span_text_carries_no_end_id():
    text = task(Packing.SPAN, source=Source(Field.TEXT), encoder=FakeEncoder())
    (document,) = text.documents(["ab"])
    assert document.tolist() == [ord("a"), ord("b")]


def test_a_packing_the_source_does_not_support_is_refused():
    with pytest.raises(ValueError, match="supports packing"):
        task(Packing.SPAN, source=Source(packings=(Packing.CONCATENATE,)))


def test_a_source_of_text_needs_a_tokenizer():
    with pytest.raises(ValueError, match="needs a tokenizer"):
        task(source=Source(Field.TEXT))


def test_the_default_packing_is_the_sources_first():
    assert task(None, source=Source(packings=(Packing.SPAN,))).packing == Packing.SPAN
