"""The window arm against the reference training code's sliding window with sinks, ported as a few lines of masking.

The reference training code lets query i read key j when j <= i and either i - j < window_size or j < num_sink_tokens
(`SlidingWindowAttention.pattern`).
Its gated attention layer reaches the same read with flash's left window: i - j <= left, plus the sinks, so
`window_size=[1535, 0]` with four sinks is a window of 1536 and four sinks.
Both are written here from those definitions and nothing is imported from the reference training code.
"""

import pytest
import torch

from lsa.training.native.config import AttentionConfig
from lsa.training.native.layers import MultiresInnerAttention

HEAD_DIM = 8
SCALE = HEAD_DIM**-0.5
LENGTH = 40
# Eight query heads over one key-value head, the multi-query shape the sparse arms were compared on
QUERY_HEADS = 8


def sliding_window_pattern(window_size, sink_tokens):
    """Which keys each query reads under `SlidingWindowAttention.pattern`, [LENGTH, LENGTH] bool."""
    delta = torch.arange(LENGTH)[:, None] - torch.arange(LENGTH)[None, :]
    sink = torch.arange(LENGTH)[None, :] < sink_tokens
    return (delta >= 0) & ((delta < window_size) | sink)


def flash_window_pattern(left, sink_tokens):
    """Which keys each query reads under `Attention.pattern` with `window_size=(left, 0)` and sinks."""
    delta = torch.arange(LENGTH)[:, None] - torch.arange(LENGTH)[None, :]
    allowed = (delta >= 0) & (delta <= left)
    sink = (delta >= 0) & (torch.arange(LENGTH)[None, :] < sink_tokens)
    return allowed | sink


def reference_context(q, k, v, pattern):
    """Masked softmax attention of `[T, H, K]` queries over `[T, 1, K]` keys, as the reference training code's SDPA path."""
    scores = torch.einsum("thd,sd->hts", q, k[:, 0]) * SCALE
    scores = scores.masked_fill(~pattern[None], float("-inf"))
    return torch.einsum("hts,sd->thd", scores.softmax(-1), v[:, 0])


@pytest.fixture
def tokens():
    generator = torch.Generator().manual_seed(5)
    q = torch.randn(LENGTH, QUERY_HEADS, HEAD_DIM, generator=generator, dtype=torch.float64)
    k = torch.randn(LENGTH, 1, HEAD_DIM, generator=generator, dtype=torch.float64)
    v = torch.randn(LENGTH, 1, HEAD_DIM, generator=generator, dtype=torch.float64)
    return q, k, v


def module(window_size, sink_size):
    config = AttentionConfig(kind="window", window_size=window_size, sink_size=sink_size)
    return MultiresInnerAttention.Config(attention=config, seq_len=LENGTH, rotary_dim=0).build()


@pytest.mark.parametrize(
    ("window_size", "sink_size"),
    [(6, 0), (6, 3), (6, 4), (1, 4), (1, 0), (LENGTH, 4), (3, 25)],
    ids=[
        "window",
        "window+sinks",
        "sinks inside window early",
        "single key+sinks",
        "self only",
        "whole",
        "sinks past window",
    ],
)
def test_the_window_arm_reads_what_the_reference_training_codes_sliding_window_reads(
    tokens, window_size, sink_size
):
    q, k, v = tokens
    got = module(window_size, sink_size)(q, k, v, scale=SCALE)
    expected = reference_context(q, k, v, sliding_window_pattern(window_size, sink_size))
    assert torch.allclose(got, expected, atol=1e-12)


def test_the_window_arm_is_the_flash_left_window_plus_sinks_at_one_key_more_than_the_left(tokens):
    # window_size=[1535, 0], num_sink_tokens=4 in the reference training code is window_size 1536, sink_size 4 here
    left, sinks = 5, 4
    q, k, v = tokens
    got = module(left + 1, sinks)(q, k, v, scale=SCALE)
    assert torch.allclose(got, reference_context(q, k, v, flash_window_pattern(left, sinks)), atol=1e-12)


def test_the_window_arm_matches_the_reference_gradient_on_q_k_and_v(tokens):
    q, k, v = (tensor.clone().requires_grad_() for tensor in tokens)
    arm = module(6, 3)
    weights = torch.randn(
        LENGTH, QUERY_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(9), dtype=torch.float64
    )
    (arm(q, k, v, scale=SCALE) * weights).sum().backward()
    got = [tensor.grad.clone() for tensor in (q, k, v)]
    q2, k2, v2 = (tensor.detach().clone().requires_grad_() for tensor in tokens)
    (reference_context(q2, k2, v2, sliding_window_pattern(6, 3)) * weights).sum().backward()
    for have, want in zip(got, (q2.grad, k2.grad, v2.grad), strict=True):
        assert torch.allclose(have, want, atol=1e-12)


def test_a_key_past_the_window_and_the_sinks_gets_no_gradient(tokens):
    q, k, v = (tensor.clone().requires_grad_() for tensor in tokens)
    module(6, 3)(q, k, v, scale=SCALE)[-1].sum().backward()
    # The last query reads the last six keys and the first three, and nothing between
    assert (k.grad[3 : LENGTH - 6] == 0).all()
    assert (v.grad[3 : LENGTH - 6] == 0).all()
    assert (v.grad[:3] != 0).any()


def test_sequences_laid_end_to_end_do_not_read_each_others_sinks(tokens):
    q, k, v = (torch.cat([tensor, tensor.flip(0)]) for tensor in tokens)
    got = module(6, 3)(q, k, v, scale=SCALE)
    assert torch.allclose(got[:LENGTH], reference_context(*tokens, sliding_window_pattern(6, 3)), atol=1e-12)
    flipped = [tensor.flip(0) for tensor in tokens]
    assert torch.allclose(got[LENGTH:], reference_context(*flipped, sliding_window_pattern(6, 3)), atol=1e-12)


def test_a_negative_sink_size_is_refused():
    with pytest.raises(ValueError, match="sink_size"):
        AttentionConfig(kind="window", sink_size=-1)
