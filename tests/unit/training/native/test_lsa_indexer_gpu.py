"""Hold learned LSA's training operations to the reference training code on the cluster."""

import pytest
import torch

from lsa.training.native import lsa_indexer, lsa_summaries
from lsa.training.native.config import AttentionConfig
from lsa.training.native.lsa_gradients import StraightThrough
from lsa.training.native.lsa_indexer import (
    LsaInnerAttention,
    bucket_scores,
    read_counts,
    selection_table,
    window_scores,
)
from tests.helpers.training.kernels.inputs import (
    attention_inputs,
    dense_reference,
    make_buckets,
    random_selection,
    relative_error,
    run_reference,
    unit_rows,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]


def test_bucket_score_forward_and_backward_match_reference_training_code():
    generator = torch.Generator(device="cuda").manual_seed(31)
    queries = unit_rows((2, 128, 32), generator).requires_grad_()
    keys = unit_rows((2, 128, 32), generator).requires_grad_()
    _, members = make_buckets(keys.detach(), 32, 4)
    owner, scores = bucket_scores(queries, keys, members)
    # The oracle returns owners first, so compare the score gradient independently
    outputs, _ = run_reference("lsa.bucket_scores", [queries, keys, members])
    assert torch.equal(owner.cpu(), outputs[0])
    assert relative_error(scores, outputs[1]) < 1e-6
    gradient = torch.randn_like(scores)
    got = torch.autograd.grad(scores, (queries, keys), gradient)
    q2, k2 = queries.detach().clone().requires_grad_(), keys.detach().clone().requires_grad_()
    units = torch.nn.functional.normalize(k2.float(), dim=-1)
    held = members >= 0
    gathered = units[torch.arange(2, device="cuda")[:, None, None], members.clamp_min(0)]
    centroids = torch.nn.functional.normalize((gathered * held[..., None]).sum(2), dim=-1)
    expected = torch.autograd.grad(q2 @ centroids.transpose(1, 2), (q2, k2), gradient)
    for ours, want in zip(got, expected, strict=True):
        torch.testing.assert_close(ours, want)


def test_straight_through_gradients_match_reference_training_code():
    q, k, v, generator = attention_inputs(1, 256, 4, 1, 32, 7)
    selected = random_selection(1, 256, 32, generator, window_size=32)
    kept = selection_table(selected, 256)
    index_queries = unit_rows((1, 256, 32), generator).requires_grad_()
    index_keys = unit_rows((1, 256, 32), generator).requires_grad_()
    _, members = make_buckets(index_keys.detach(), 32, 4)
    owner, picked = bucket_scores(index_queries, index_keys, members)
    recent_scores = window_scores(index_queries, index_keys, 32)
    context, lse = dense_reference(q, k, v, selected, 32, 4, 32**-0.5)
    context = context.transpose(1, 2).to(q.dtype)
    gradient = torch.randn_like(context)
    args = [
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        context,
        gradient,
        lse,
        kept,
        owner,
        32,
        32,
        4,
        32**-0.5,
        True,
    ]
    outputs, _ = run_reference("straight_through.gradients", args)
    expected = torch.autograd.grad(
        (picked, recent_scores), (index_queries, index_keys), tuple(x.cuda() for x in outputs)
    )
    ours = StraightThrough.apply(
        context,
        index_queries,
        index_keys,
        members,
        selected,
        tuple(x.detach() for x in args[:3]) + (lse.detach(),),
        (32, 4, 32**-0.5, True),
        None,
    )
    gradients = torch.autograd.grad(ours, (index_queries, index_keys), gradient)
    for got, want in zip(gradients, expected, strict=True):
        print(f"sparse_ste_relative_error={relative_error(got, want):.7g}")
        assert relative_error(got, want) < 1e-4


def test_summary_forward_and_query_gradients_match_reference_training_code():
    q, k, v, generator = attention_inputs(1, 128, 4, 1, 32, 7)
    _, members = make_buckets(unit_rows((1, 128, 32), generator), 32, 4)
    selected = random_selection(1, 128, 32, generator, window_size=32)
    _, lse = dense_reference(q, k, v, selected, 32, 4, 32**-0.5)
    counts = read_counts(selected, members, 128)
    args = [
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        lse.detach(),
        members,
        counts,
        32,
        32,
        4,
        32**-0.5,
        0.7,
    ]
    gradient = torch.randn_like(lse)
    outputs, grads = run_reference("lsa_summaries.terms", args, leaves=(0, 1), grad_output=gradient)
    ours = lsa_summaries.terms(*args)
    got_grads = torch.autograd.grad(ours[0], (args[0], args[1]), gradient)
    for got, want in zip(ours, outputs, strict=True):
        assert relative_error(got, want) < 1e-6
    for got, want in zip(got_grads, grads, strict=True):
        print(f"summary_gradient_relative_error={relative_error(got, want):.7g}")
        assert relative_error(got, want) < 1e-4


@pytest.mark.parametrize("fast_read", [False, True])
def test_indexer_free_gpu_read_preserves_legacy_or_matches_its_sparse_math(fast_read):
    from lsa.training.native.config import AttentionConfig
    from lsa.training.native.layers import MultiresInnerAttention

    config = AttentionConfig(
        kind="lsa",
        coarse=False,
        window_size=64,
        block_size=64,
        top_k=32,
        pool_size=4,
        bucket_size=8,
        bucket_cap=16,
        fast_read=fast_read,
    )
    old = MultiresInnerAttention.Config(attention=config, seq_len=256, rotary_dim=0).build().cuda()
    new = LsaInnerAttention.Config(attention=config, seq_len=256, rotary_dim=0).build().cuda()
    q, k, v, generator = attention_inputs(1, 256, 4, 1, 64, 73)
    tensors = [tensor.flatten(0, 1).detach().requires_grad_() for tensor in (q, k, v)]
    weights = torch.randn(256, 4, 64, generator=generator, device="cuda").bfloat16()
    expected = old(*tensors)
    expected_grads = torch.autograd.grad(expected, tensors, weights)
    actual = new(*tensors)
    actual_grads = torch.autograd.grad(actual, tensors, weights)
    for got, want in [(actual, expected), *zip(actual_grads, expected_grads, strict=True)]:
        if fast_read:
            assert relative_error(got, want) < 0.005
        else:
            assert torch.equal(got, want)


def test_fast_read_asks_the_cuda_kernels_for_v2(monkeypatch):
    calls = []
    original = lsa_indexer.cuda_sparse.attention

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(lsa_indexer.cuda_sparse, "attention", record)
    attention = AttentionConfig(
        kind="lsa",
        coarse=False,
        fast_read=True,
        window_size=512,
        block_size=512,
        pool_size=4,
        bucket_size=8,
        bucket_cap=16,
        sink_size=4,
        top_k=64,
        query_head_reduction="max",
    )
    layer = (
        LsaInnerAttention.Config(
            attention=attention, seq_len=2048, rotary_dim=0, query_head_count=8, kv_head_count=1, head_dim=128
        )
        .build()
        .cuda()
    )
    q, k, v = (torch.randn(2048, h, 128, device="cuda").bfloat16() for h in (8, 1, 1))
    layer(q, k, v)
    assert calls == [{"v2": True}]
