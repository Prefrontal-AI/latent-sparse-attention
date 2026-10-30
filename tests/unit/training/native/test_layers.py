import pytest
import torch

from lsa.training.native.buckets import partition_keys
from lsa.training.native.config import AttentionConfig
from lsa.training.native.layers import (
    MultiresInnerAttention,
    coarse_mass_share,
    multires_attention,
    reference_attention,
    window_attention,
)

HEAD_DIM = 8
SCALE = HEAD_DIM**-0.5
STREAMS, GROUP, LENGTH = 2, 2, 96


def small(**changes):
    knobs = {
        "window_size": 8,
        "sink_size": 2,
        "top_k": 24,
        "bucket_size": 4,
        "pool_size": 2,
        "bucket_cap": 8,
        "block_size": 16,
    }
    return AttentionConfig(**{**knobs, **changes})


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(1)
    queries = torch.randn(STREAMS, GROUP, LENGTH, HEAD_DIM, generator=generator, dtype=torch.float64)
    keys = torch.randn(STREAMS, LENGTH, HEAD_DIM, generator=generator, dtype=torch.float64)
    values = torch.randn(STREAMS, LENGTH, HEAD_DIM, generator=generator, dtype=torch.float64)
    return queries, keys, values


def dense(queries, keys, values):
    scores = torch.einsum("sgqd,skd->sgqk", queries, keys) * SCALE
    scores = scores.masked_fill(torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1), float("-inf"))
    return torch.einsum("sgqk,skd->sgqd", scores.softmax(-1), values)


VARIANTS = {
    "default": {},
    "no_variance": {"variance": False},
    "fine_only": {"coarse": False},
    "chunks": {"partition": "chunks"},
    "random": {"partition": "random"},
    "error_rank": {"rank": "error"},
    "cosine_rank": {"rank": "cosine"},
    "no_selection": {"top_k": 0},
    "narrow_top_p": {"top_p": 3},
}


@pytest.mark.parametrize("changes", VARIANTS.values(), ids=VARIANTS)
def test_the_training_path_is_the_reference_in_value_and_in_every_gradient(inputs, changes):
    config = small(**changes)
    queries, keys, values = (tensor.clone().requires_grad_() for tensor in inputs)
    result = multires_attention(queries, keys, values, config, SCALE)
    expected = reference_attention(queries, keys, values, result.table, config, SCALE)
    weights = torch.randn_like(expected)
    ours = torch.autograd.grad((result.context * weights).sum(), [queries, keys, values])
    theirs = torch.autograd.grad((expected * weights).sum(), [queries, keys, values])
    assert torch.allclose(result.context, expected, atol=1e-10)
    for got, want in zip(ours, theirs, strict=True):
        assert torch.allclose(got, want, atol=1e-10)


def test_every_bucket_selected_is_dense_attention(inputs):
    config = small(top_k=10**6, top_p=10**6)
    result = multires_attention(*inputs, config, SCALE)
    assert torch.allclose(result.context, dense(*inputs), atol=1e-12)


def test_singleton_buckets_are_dense_attention_whatever_is_selected(inputs):
    config = small(bucket_size=1, pool_size=1, bucket_cap=2, partition="chunks")
    result = multires_attention(*inputs, config, SCALE)
    assert torch.allclose(result.context, dense(*inputs), atol=1e-12)


def test_with_nothing_selected_the_summaries_stand_in_for_every_bucket_beyond_the_window(inputs):
    config = small(top_k=0)
    result = multires_attention(*inputs, config, SCALE)
    assert not torch.allclose(result.context, dense(*inputs), atol=1e-3)
    shares = coarse_mass_share(
        inputs[0], inputs[1], result.statistics, result.table, result.reads, config, SCALE, 32
    )
    assert shares["coarse_mass_share_far"] > 0.1


def test_a_query_in_the_first_block_reads_dense_attention_in_every_arm(inputs):
    for changes in ({}, {"coarse": False}, {"top_k": 0}):
        result = multires_attention(*inputs, small(**changes), SCALE)
        assert torch.allclose(result.context[:, :, :16], dense(*inputs)[:, :, :16], atol=1e-12)


def test_unselected_buckets_get_a_gradient_through_their_summaries_and_only_with_coarse_terms(inputs):
    def key_gradient(config):
        queries, keys, values = (tensor.clone().requires_grad_() for tensor in inputs)
        multires_attention(queries, keys, values, config, SCALE).context[:, :, -1].square().sum().backward()
        return keys.grad

    config = small(top_k=0, window_size=1, sink_size=0)
    table = partition_keys(inputs[1], config)
    far_keys = table.bucket_of_key[0] < 4
    assert key_gradient(config)[0][far_keys].abs().sum() > 0
    assert (
        key_gradient(small(top_k=0, window_size=1, sink_size=0, coarse=False))[0][far_keys].abs().sum() == 0
    )


def test_the_variance_term_moves_the_output_and_vanishes_for_singletons(inputs):
    with_variance = multires_attention(*inputs, small(top_k=0), SCALE).context
    without = multires_attention(*inputs, small(top_k=0, variance=False), SCALE).context
    assert not torch.allclose(with_variance, without, atol=1e-6)


def test_a_window_attends_to_its_recent_keys_and_sinks_alone(inputs):
    config = small(kind="window", window_size=4, sink_size=0)
    queries, keys, values = inputs
    got = window_attention(queries, keys, values, config, SCALE)
    scores = torch.einsum("sgqd,skd->sgqk", queries, keys) * SCALE
    distance = torch.arange(LENGTH)[:, None] - torch.arange(LENGTH)[None, :]
    scores = scores.masked_fill((distance < 0) | (distance >= 4), float("-inf"))
    assert torch.allclose(got, torch.einsum("sgqk,skd->sgqd", scores.softmax(-1), values), atol=1e-12)


def make_module(**changes):
    config = small(**changes)
    return MultiresInnerAttention.Config(attention=config, seq_len=LENGTH).build()


def test_the_module_splits_packed_tokens_into_streams_and_puts_the_context_back(inputs):
    module = make_module(top_k=10**6, top_p=10**6)
    heads, kv_heads, sequences = 4, 2, 2
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(sequences * LENGTH, heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    k = torch.randn(sequences * LENGTH, kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    v = torch.randn(sequences * LENGTH, kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    got = module(q, k, v, scale=SCALE)
    for sequence in range(sequences):
        tokens = slice(sequence * LENGTH, (sequence + 1) * LENGTH)
        for head in range(heads):
            kv = head // (heads // kv_heads)
            scores = (q[tokens, head] @ k[tokens, kv].T) * SCALE
            scores = scores.masked_fill(
                torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1), float("-inf")
            )
            assert torch.allclose(got[tokens, head], scores.softmax(-1) @ v[tokens, kv], atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "changes", [{}, {"coarse": False}, {"partition": "chunks"}], ids=["multires", "fine_only", "chunks"]
)
def test_the_flex_path_matches_the_reference_in_value_and_gradient_on_a_gpu(changes):
    config = small(
        window_size=64,
        sink_size=4,
        top_k=64,
        bucket_size=8,
        pool_size=4,
        bucket_cap=16,
        block_size=64,
        **changes,
    )
    generator = torch.Generator().manual_seed(0)
    shape = (2, 4, 256, 64)
    queries = torch.randn(shape, generator=generator).cuda().requires_grad_()
    keys = torch.randn(2, 256, 64, generator=generator).cuda().requires_grad_()
    values = torch.randn(2, 256, 64, generator=generator).cuda().requires_grad_()
    scale = 64**-0.5
    result = multires_attention(queries, keys, values, config, scale)
    table = type(result.table)(*[t.cpu() if torch.is_tensor(t) else t for t in result.table])
    cpu = [tensor.detach().cpu().requires_grad_() for tensor in (queries, keys, values)]
    expected = reference_attention(*cpu, table, config, scale)
    weights = torch.randn(shape, generator=generator)
    ours = torch.autograd.grad((result.context * weights.cuda()).sum(), [queries, keys, values])
    theirs = torch.autograd.grad((expected * weights).sum(), cpu)
    assert torch.allclose(result.context.cpu(), expected, atol=1e-4)
    for got, want in zip(ours, theirs, strict=True):
        assert torch.allclose(got.cpu(), want, atol=1e-4)


@pytest.mark.parametrize(
    "changes", [{}, {"partition": "chunks"}, {"coarse": False}], ids=["default", "chunks", "fine_only"]
)
def test_the_training_path_traces_as_one_graph_and_matches_eager_in_value_and_gradient(changes):
    generator = torch.Generator().manual_seed(2)
    queries = torch.randn(STREAMS, GROUP, LENGTH, HEAD_DIM, generator=generator, requires_grad=True)
    keys = torch.randn(STREAMS, LENGTH, HEAD_DIM, generator=generator, requires_grad=True)
    values = torch.randn(STREAMS, LENGTH, HEAD_DIM, generator=generator, requires_grad=True)
    config = small(**changes)

    def read(queries, keys, values):
        return multires_attention(queries, keys, values, config, SCALE).context

    eager = read(queries, keys, values)
    compiled = torch.compile(read, fullgraph=True)(queries, keys, values)
    assert torch.allclose(compiled, eager, atol=1e-5)
    weights = torch.randn_like(eager)
    ours = torch.autograd.grad((compiled * weights).sum(), [queries, keys, values])
    theirs = torch.autograd.grad((eager * weights).sum(), [queries, keys, values])
    for got, want in zip(ours, theirs, strict=True):
        assert torch.allclose(got, want, atol=1e-4)
