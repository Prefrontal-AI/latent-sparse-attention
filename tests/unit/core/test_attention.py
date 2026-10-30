"""Invariant 15 - at a budget of `T` or more the sparse read reproduces dense causal attention.

This is the anchor that says the read adds no error of its own, so any gap against the base model is the
indexer's and nothing else's.
"""

import torch

from lsa.core.attention import keep_set, topk_attention

torch.manual_seed(0)
B, T, HQ, HKV, DH = 2, 96, 8, 2, 16
SCALE = DH**-0.5


def qkv(t=T):
    return (
        torch.randn(B, t, HQ, DH),
        torch.randn(B, t, HKV, DH),
        torch.randn(B, t, HKV, DH),
    )


def dense_causal(q, k, v):
    group = torch.arange(HQ) // (HQ // HKV)
    mask = torch.arange(q.shape[1])[:, None] >= torch.arange(k.shape[1])[None, :]
    out = torch.empty_like(q)
    for b in range(q.shape[0]):
        s = torch.einsum("qhd,shd->qhs", q[b], k[b][:, group, :]) * SCALE
        p = torch.softmax(s.masked_fill(~mask[:, None, :], float("-inf")), dim=-1)
        out[b] = torch.einsum("qhs,shd->qhd", p, v[b][:, group, :])
    return out


def everything_selected(t=T):
    return torch.arange(t, dtype=torch.int32).expand(B, t, t).contiguous()


def test_full_budget_reproduces_dense_attention():
    q, k, v = qkv()
    got = topk_attention(q, k, v, everything_selected(), window=0, sink=0, scale=SCALE)
    assert torch.allclose(got, dense_causal(q, k, v), atol=1e-5)


def test_the_window_alone_reproduces_dense_attention_when_it_spans_the_sequence():
    q, k, v = qkv()
    empty = torch.full((B, T, 1), -1, dtype=torch.int32)
    got = topk_attention(q, k, v, empty, window=T, sink=0, scale=SCALE)
    assert torch.allclose(got, dense_causal(q, k, v), atol=1e-5)


def test_query_blocking_does_not_change_the_answer():
    q, k, v = qkv()
    sel = everything_selected()
    whole = topk_attention(q, k, v, sel, window=0, sink=0, scale=SCALE, query_block_size=T)
    split = topk_attention(q, k, v, sel, window=0, sink=0, scale=SCALE, query_block_size=7)
    assert torch.equal(whole, split)


def test_the_keep_set_is_the_union_of_window_sinks_and_selection():
    sel = torch.tensor([[[40, -1, -1]]], dtype=torch.int32)[0]
    keep = keep_set(
        sel, query_start=50, query_count=1, key_count=T, window=4, sink=2, device=torch.device("cpu")
    )
    assert set(keep[0].nonzero(as_tuple=True)[0].tolist()) == {0, 1, 40, 47, 48, 49, 50}


def test_padding_selects_nothing():
    sel = torch.full((1, 3), -1, dtype=torch.int32)
    keep = keep_set(
        sel, query_start=50, query_count=1, key_count=T, window=1, sink=0, device=torch.device("cpu")
    )
    assert keep[0].nonzero(as_tuple=True)[0].tolist() == [50]


def test_causality_beats_a_selected_future_key():
    """A selection may not reach forward even if an id says it should"""
    sel = torch.tensor([[80]], dtype=torch.int32)
    keep = keep_set(
        sel, query_start=10, query_count=1, key_count=T, window=1, sink=0, device=torch.device("cpu")
    )
    assert keep[0].nonzero(as_tuple=True)[0].tolist() == [10]


def test_a_decode_step_matches_its_prefill_row():
    q, k, v = qkv()
    sel = everything_selected()
    prefill = topk_attention(q, k, v, sel, window=8, sink=2, scale=SCALE)
    step = topk_attention(
        q[:, T - 1 :], k, v, sel[:, T - 1 :], window=8, sink=2, scale=SCALE, query_start=T - 1
    )
    assert torch.allclose(step[:, 0], prefill[:, T - 1], atol=1e-6)
