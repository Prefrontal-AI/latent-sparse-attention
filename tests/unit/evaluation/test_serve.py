"""The server's flags a run names, and the command line they become."""

import pytest

from lsa.evaluation.config import serve_argv


def test_a_scalar_is_the_flag_s_argument():
    assert serve_argv(
        {"context-length": 10500000, "mem-fraction-static": 0.75, "attention-backend": "dsa"}
    ) == [
        "--context-length",
        "10500000",
        "--mem-fraction-static",
        "0.75",
        "--attention-backend",
        "dsa",
    ]


def test_true_is_the_bare_flag():
    assert serve_argv({"enable-metrics": True}) == ["--enable-metrics"]


@pytest.mark.parametrize("value", [None, False])
def test_null_and_false_leave_the_flag_out(value):
    assert serve_argv({"enable-metrics": value, "page-size": 1}) == ["--page-size", "1"]


def test_a_list_is_one_flag_with_each_item():
    assert serve_argv({"cuda-graph-bs": [1, 2, 4]}) == ["--cuda-graph-bs", "1", "2", "4"]


def test_the_flags_keep_the_order_they_were_written_in():
    assert serve_argv({"b": 1, "a": 2}) == ["--b", "1", "--a", "2"]


@pytest.mark.parametrize("value", [{"size": 1}, [], [{"size": 1}], [True]])
def test_a_value_no_flag_takes_is_refused_saying_what_arrived_and_what_was_wanted(value):
    with pytest.raises(ValueError, match=r"run\.serve\.page-size must be a scalar.*got"):
        serve_argv({"page-size": value})


def test_a_key_with_its_dashes_is_refused():
    with pytest.raises(ValueError, match="without the leading dashes - got '--page-size'"):
        serve_argv({"--page-size": 1})
