"""The anchors a GPU measures for its roofline."""

import pytest

from lsa.evaluation.performance.roofline import anchors


@pytest.mark.gpu
def test_the_anchors_are_rates_under_any_card_s_roof(monkeypatch) -> None:
    monkeypatch.setattr(anchors, "COPY_BYTES", 1 << 28)
    monkeypatch.setattr(anchors, "GEMM_DIM", 2048)
    monkeypatch.setattr(anchors, "REPEATS", 3)
    measured = anchors.measure()
    assert 1e11 < measured["copy_bytes_per_second"] < 2e13
    assert 1e13 < measured["bf16_gemm_flops"] < 1e16
    assert measured["sm_count"] > 0
