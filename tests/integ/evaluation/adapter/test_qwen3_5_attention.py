"""The bottom-right causal read a chunked prefill attends through - a chunk sees every key up to its own position."""

import pytest
import torch

from lsa.evaluation.adapter.qwen3_5 import lower_right_causal_attention


def heads(query_count, key_count, device="cpu", dtype=torch.float32, head_dim=16):
    """Six query heads on two key heads, the chunk the last `query_count` positions of the keys"""
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 6, query_count, head_dim, generator=generator, dtype=dtype)
    key = torch.randn(1, 2, key_count, head_dim, generator=generator, dtype=dtype)
    value = torch.randn(1, 2, key_count, head_dim, generator=generator, dtype=dtype)
    return query.to(device), key.to(device), value.to(device)


def dense(query, key, value):
    """The read written out - each query head on its group's key head, masked past its own position"""
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    query_count, key_count = query.shape[2], key.shape[2]
    scores = query.float() @ key.float().transpose(-1, -2) / query.shape[-1] ** 0.5
    visible = torch.arange(key_count)[None] <= torch.arange(query_count)[:, None] + key_count - query_count
    weights = scores.masked_fill(~visible.to(scores.device), float("-inf")).softmax(-1)
    return (weights @ value.float()).transpose(1, 2)


def test_a_chunk_reads_every_key_up_to_its_own_position():
    query, key, value = heads(8, 24)

    read, weights = lower_right_causal_attention(None, query, key, value, None, scaling=16**-0.5)

    assert weights is None
    assert read.shape == (1, 8, 6, 16)
    torch.testing.assert_close(read, dense(query, key, value), rtol=1e-5, atol=1e-5)


def test_a_padding_mask_is_refused():
    query, key, value = heads(4, 4)
    with pytest.raises(ValueError, match="no padding mask"):
        lower_right_causal_attention(None, query, key, value, torch.ones(1, 1, 4, 4))


@pytest.mark.gpu
def test_on_a_card_the_grouped_kernel_reads_what_the_dense_read_does():
    query, key, value = heads(256, 1024, "cuda", torch.bfloat16, head_dim=256)

    read, _ = lower_right_causal_attention(None, query, key, value, None, scaling=256**-0.5)
    again, _ = lower_right_causal_attention(None, query, key, value, None, scaling=256**-0.5)

    torch.testing.assert_close(read.float(), dense(query, key, value), rtol=2e-2, atol=2e-2)
    assert torch.equal(read, again)
