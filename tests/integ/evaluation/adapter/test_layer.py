"""The generic layer - the part of reading an attention module that is the same for every model.

Everything here runs on CPU against [fake_model](fake_model.py), so it gates in CI with no checkpoint and
nothing from the evaluation extra installed.
"""

import pytest
import torch

from lsa.core import RotaryLayout
from lsa.evaluation.adapter.layer import Projection, SparseAttention, heads, rotary_function
from lsa.sglang.settings.design import IndexerConfig
from tests.helpers.evaluation.adapter.fake_model import (
    FakeAttention,
    FakeRotaryEmbedding,
)

MODEL_DIM = 64
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
ROTARY_DIM = 8

COVERING = IndexerConfig(window_size=8, sink_size=2, top_k=64, bucket_size=4, bucket_cap=8, pool_size=None)


def build(qk_norm=False, config=COVERING):
    torch.manual_seed(0)
    attention = FakeAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM, qk_norm=qk_norm)
    layer = SparseAttention(attention, config, RotaryLayout("prefix", ROTARY_DIM))
    return attention, layer, FakeRotaryEmbedding(ROTARY_DIM)


def tables(rotary_embedding, hidden_states, start=0):
    query_count = hidden_states.shape[1]
    position_ids = torch.arange(start, start + query_count)[None]
    return rotary_embedding(hidden_states, position_ids)


def inputs(query_count, seed=1):
    torch.manual_seed(seed)
    return torch.randn(1, query_count, MODEL_DIM)


def test_qk_norm_is_applied_when_the_model_carries_it():
    plain, plain_layer, _ = build(qk_norm=False)
    normed, normed_layer, _ = build(qk_norm=True)
    normed.load_state_dict(plain.state_dict(), strict=False)
    normed.q_norm.weight.data.mul_(3.0)

    hidden_states = inputs(16)
    assert not torch.allclose(plain_layer.project(hidden_states).q, normed_layer.project(hidden_states).q)


def test_the_default_projection_has_no_gate():
    _, layer, _ = build()
    assert layer.project(inputs(8)).gate is None


def test_heads_splits_a_projection_into_the_layout_the_index_wants():
    projected = torch.randn(2, 5, QUERY_HEADS * HEAD_DIM)
    assert heads(projected, HEAD_DIM).shape == (2, QUERY_HEADS, 5, HEAD_DIM)


def test_the_rotation_is_resolved_from_the_model_that_defines_it():
    attention, _, _ = build()
    from tests.helpers.evaluation.adapter import fake_model

    assert rotary_function(attention) is fake_model.apply_rotary_pos_emb


def test_a_model_with_no_rotation_to_resolve_says_so():
    class Stray(torch.nn.Module):
        pass

    Stray.__module__ = "builtins"
    with pytest.raises(ValueError, match="apply_rotary_pos_emb"):
        rotary_function(Stray())


def test_a_projection_carries_the_heads_the_index_and_the_read_want():
    _, layer, _ = build()
    projection = layer.project(inputs(12))

    assert isinstance(projection, Projection)
    assert projection.q.shape == (1, QUERY_HEADS, 12, HEAD_DIM)
    assert projection.k.shape == (1, KV_HEADS, 12, HEAD_DIM)
    assert projection.v.shape == (1, KV_HEADS, 12, HEAD_DIM)
