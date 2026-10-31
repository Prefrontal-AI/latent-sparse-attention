"""Extraction stage geometry and capture coverage."""

import pytest

from lsa.training.distillation.data.stage import stage_spans

GLM_LAYERS = list(range(3, 44, 4))


def test_the_last_span_ends_past_the_last_captured_layer():
    assert stage_spans([16, 30], GLM_LAYERS) == [(0, 16), (16, 30), (30, 44)]


def test_no_cuts_is_one_span():
    assert stage_spans([], GLM_LAYERS) == [(0, 44)]


@pytest.mark.parametrize("cuts", [[30, 16], [16, 16], [0, 16], [44]])
def test_cuts_that_do_not_increase_inside_the_stack_are_refused(cuts):
    with pytest.raises(ValueError, match="must increase"):
        stage_spans(cuts, GLM_LAYERS)


def test_a_span_that_captures_nothing_is_refused():
    """It would pass the streams on and write no layer, so nothing would say it was done"""
    with pytest.raises(ValueError, match=r"span \[4, 7\) captures none"):
        stage_spans([4, 7], GLM_LAYERS)
