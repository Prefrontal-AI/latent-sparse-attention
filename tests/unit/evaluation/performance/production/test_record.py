"""A production record's rows - the arithmetic of a request, and what a failed one says."""

import pytest

from lsa.evaluation.performance.production import record


def test_a_request_s_tpot_is_its_mean_gap_in_milliseconds():
    result = {"ttfts": [2.0], "itls": [[0.01, 0.03]], "input_lens": [1000], "output_lens": [3]}
    (row,) = record.request_rows(result)
    assert row["tpot_ms"] == pytest.approx(20.0)
    assert row["e2e_s"] == pytest.approx(2.04)


def test_a_length_where_every_request_failed_says_why_instead_of_dividing_by_zero():
    result = {
        "ttfts": [0.0, 0.0],
        "itls": [[], []],
        "input_lens": [1_048_576, 1_048_576],
        "output_lens": [0, 0],
        "errors": ["context length 1050000 exceeds the pool of 739433 tokens", ""],
    }
    with pytest.raises(
        ValueError, match="1048576 input tokens streamed no tokens.*exceeds the pool of 739433"
    ):
        record.request_rows(result)


def test_a_failed_request_with_no_error_recorded_still_raises():
    result = {"ttfts": [0.0], "itls": [[]], "input_lens": [8], "output_lens": [0]}
    with pytest.raises(ValueError, match="no error recorded"):
        record.request_rows(result)
