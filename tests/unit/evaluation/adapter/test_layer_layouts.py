"""The attribute layouts the generic path has to cover beyond a Qwen-shaped decoder.

Every one of these was found by running the generic adapter against a real checkpoint of the family, and
each failed differently - two raised and one would have scored.
"""

import pytest
import torch

from lsa.core import RotaryLayout
from lsa.evaluation.adapter.layer import SparseAttention, qk_norms
from lsa.sglang.settings.design import IndexerConfig
from tests.helpers.evaluation.adapter.fake_model import FakeAttention, FakeRotaryEmbedding
from tests.helpers.evaluation.adapter.fake_partial_model import (
    NopeAttention,
    PartialAttention,
)

MODEL_DIM = 64
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
ROTARY_DIM = 4

# A budget past every key any query here can see, so selection covers the context and the read is exact
COVERING = IndexerConfig(window_size=8, sink_size=2, top_k=64, bucket_size=4, bucket_cap=8, pool_size=None)


def inputs(query_count, seed=1):
    torch.manual_seed(seed)
    return torch.randn(1, query_count, MODEL_DIM)


def tables(rotary_embedding, hidden_states):
    positions = torch.arange(hidden_states.shape[1])[None]
    return rotary_embedding(hidden_states, positions)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_a_rotary_function_that_does_not_slice_is_given_only_the_dims_it_rotates(qk_norm):
    """stablelm and phi slice in their attention forward, so a whole head must not reach it"""
    torch.manual_seed(0)
    attention = PartialAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM, qk_norm=qk_norm)
    layer = SparseAttention(attention, COVERING, RotaryLayout("prefix", ROTARY_DIM))

    hidden_states = inputs(24)
    cos, sin = tables(FakeRotaryEmbedding(ROTARY_DIM), hidden_states)

    q, k = layer.rotated(layer.project(hidden_states), (cos, sin))
    assert q.shape == (1, QUERY_HEADS, 24, HEAD_DIM)
    assert k.shape == (1, KV_HEADS, 24, HEAD_DIM)
    # The dims past the rotation are the position-free sub-term, and they must come back untouched
    projection = layer.project(hidden_states)
    assert torch.equal(q[..., ROTARY_DIM:], projection.q[..., ROTARY_DIM:])


def test_qk_norm_is_found_under_the_name_its_family_uses():
    """A lookup that knows only `q_norm` skips a `q_layernorm` silently, which scores rather than raises"""
    torch.manual_seed(0)
    attention = PartialAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM, qk_norm=True)
    query_norm, key_norm = qk_norms(attention)
    assert query_norm is attention.q_layernorm
    assert key_norm is attention.k_layernorm


def test_a_model_with_no_qk_norm_reports_neither():
    attention = PartialAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    assert qk_norms(attention) == (None, None)


def test_a_nope_model_reads_without_rotary_tables():
    """GLM-5.3-Flash's DSA layers are NoPE - their decoder layer passes position_embeddings=None"""
    torch.manual_seed(0)
    attention = NopeAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    layer = SparseAttention(attention, COVERING, RotaryLayout())

    projection = layer.project(inputs(24))
    q, k = layer.rotated(projection, None)
    assert q is projection.q
    assert k is projection.k


def test_a_rotating_model_handed_no_tables_raises_rather_than_skipping_the_rotation():
    torch.manual_seed(0)
    attention = FakeAttention(MODEL_DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    layer = SparseAttention(attention, COVERING, RotaryLayout("prefix", ROTARY_DIM))

    with pytest.raises(ValueError, match="passed no rotary tables"):
        layer.rotated(layer.project(inputs(24)), None)
