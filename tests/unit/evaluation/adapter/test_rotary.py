"""Where the rotation sits, measured rather than declared.

Reading a prefix as a suffix indexes on the rotary half and nothing fails - it just scores noise - so this
is the one per-model fact the harness refuses to take on trust.
"""

import pytest
import torch

from lsa import constants as const
from lsa.core import RotaryLayout
from lsa.evaluation.adapter.rotary import detect_rotary, resolve_rotary
from tests.helpers.evaluation.adapter.fake_model import (
    FakeAttention,
    FakeRotaryEmbedding,
    MropeRotaryEmbedding,
    rotation_over,
)

HEAD_DIM = 16


def probe(dims):
    attention = FakeAttention(32, 2, 1, HEAD_DIM)
    return detect_rotary(rotation_over(dims), attention, FakeRotaryEmbedding(dims.numel()), "fake/model")


def test_a_rotated_prefix_is_found():
    assert probe(torch.arange(0, 8)) == RotaryLayout(const.Rotary.PREFIX, 8)


def test_a_rotated_suffix_is_found():
    assert probe(torch.arange(HEAD_DIM - 8, HEAD_DIM)) == RotaryLayout(const.Rotary.SUFFIX, 8)


def test_a_model_with_no_rotation_is_found():
    assert probe(torch.arange(0, 0)) == RotaryLayout(const.Rotary.NONE, 0)


def test_a_fully_rotated_head_has_no_position_free_sub_term():
    with pytest.raises(ValueError, match="LSA's training-free index space"):
        probe(torch.arange(HEAD_DIM))


def test_an_interleaved_rotation_is_rejected_rather_than_read_as_a_prefix():
    with pytest.raises(ValueError, match="neither a prefix nor a suffix"):
        probe(torch.arange(0, 16, 2))


def test_a_narrow_rotation_is_still_found():
    assert probe(torch.arange(0, 2)) == RotaryLayout(const.Rotary.PREFIX, 2)


MEASURED = RotaryLayout(const.Rotary.PREFIX, 8)


def test_auto_takes_the_measurement():
    assert resolve_rotary(const.Rotary.AUTO, None, MEASURED, "fake/model") == MEASURED


def test_a_declaration_that_agrees_is_kept():
    assert resolve_rotary(const.Rotary.PREFIX, 8, MEASURED, "fake/model") == MEASURED


def test_a_declared_end_with_no_dim_takes_the_measured_dim():
    assert resolve_rotary(const.Rotary.PREFIX, None, MEASURED, "fake/model") == MEASURED


def test_the_wrong_end_raises_rather_than_indexing_on_the_rotary_half():
    with pytest.raises(ValueError, match="drop both fields"):
        resolve_rotary(const.Rotary.SUFFIX, 8, MEASURED, "fake/model")


def test_the_wrong_dim_raises():
    with pytest.raises(ValueError, match="drop both fields"):
        resolve_rotary(const.Rotary.PREFIX, 4, MEASURED, "fake/model")


def test_declaring_no_rotation_on_a_model_that_rotates_raises():
    with pytest.raises(ValueError, match="drop both fields"):
        resolve_rotary(const.Rotary.NONE, None, MEASURED, "fake/model")


def test_declaring_no_rotation_on_a_model_with_none_agrees():
    unrotated = RotaryLayout(const.Rotary.NONE, 0)
    assert resolve_rotary(const.Rotary.NONE, None, unrotated, "fake/model") == unrotated


def test_a_model_that_wants_mrope_position_ids_is_still_measured():
    """Qwen3.5 takes [3, B, T], a plain decoder takes [B, T], and no config reliably says which"""
    attention = FakeAttention(32, 2, 1, HEAD_DIM)
    dims = torch.arange(0, 8)
    detected = detect_rotary(rotation_over(dims), attention, MropeRotaryEmbedding(dims.numel()), "fake/mrope")
    assert detected == RotaryLayout(const.Rotary.PREFIX, 8)


def test_the_slowest_dims_of_a_long_context_rope_still_read_as_rotated():
    """At rope_theta 1e7 over 64 dims the slowest turn by ~5e-7 radians at position 3 and read as untouched.

    Qwen3.5's own numbers - the probe measured 59 of its 64 rotated dims and called the result interleaved
    until it started probing at positions large enough to move them.
    """
    attention = FakeAttention(256, 2, 1, 128)
    dims = torch.arange(0, 64)
    detected = detect_rotary(
        rotation_over(dims), attention, FakeRotaryEmbedding(dims.numel(), theta=1e7), "fake/long-context"
    )
    assert detected == RotaryLayout(const.Rotary.PREFIX, 64)
