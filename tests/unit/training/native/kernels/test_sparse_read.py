import pytest
import torch

from lsa.training.native.kernels import cuda_sparse, dsa_sparse, gather_sparse
from lsa.training.native.kernels.dsa_sparse.topk import pack
from tests.helpers.training.kernels.inputs import (
    SINK_SIZE,
    WINDOW_SIZE,
    attention_inputs,
    dense_reference,
    keep_table,
    random_selection,
    relative_error,
    require_hopper,
)

pytestmark = pytest.mark.gpu

# Batch, length, query heads, key heads, head dim and top_k
SHAPES = [
    pytest.param((2, 300, 8, 1, 128, 1024), id="shorter-than-the-window"),
    pytest.param((2, 1000, 1, 1, 128, 1024), id="one-head"),
    pytest.param((2, 1000, 8, 1, 128, 512), id="eight-over-one"),
    pytest.param((1, 2200, 8, 1, 128, 1024), id="eight-over-one-longer"),
    pytest.param((2, 1500, 1, 1, 64, 512), id="head-dim-64"),
    pytest.param((1, 2000, 6, 1, 64, 512), id="six-over-one"),
    pytest.param((2, 1200, 4, 2, 64, 512), id="grouped"),
    pytest.param((1, 4096, 8, 2, 256, 1024), id="lsa-indexer-head-dim-256"),
    pytest.param((1, 4096, 8, 1, 128, 4096), id="lsa-mqa8-top-k-4096"),
]
CUDA_SHAPES = [shape for shape in SHAPES if shape.id != "lsa-indexer-head-dim-256"]
TOLERANCE = 1e-2


def check_against_dense(read, shape, seed=0):
    batch, length, query_heads, kv_heads, head_dim, top_k = shape
    q, k, v, generator = attention_inputs(batch, length, query_heads, kv_heads, head_dim, seed)
    ids = random_selection(batch, length, top_k, generator)
    scale = head_dim**-0.5
    output, lse = read(q, k, v, ids, scale)
    grad_output = torch.randn_like(output)
    got = torch.autograd.grad(output, (q, k, v), grad_output)
    leaves = [x.detach().float().requires_grad_() for x in (q, k, v)]
    want_output, want_lse = dense_reference(*leaves, ids, WINDOW_SIZE, SINK_SIZE, scale)
    want = torch.autograd.grad(want_output, leaves, grad_output.float())
    assert relative_error(output, want_output) < TOLERANCE
    torch.testing.assert_close(lse, want_lse, atol=1e-3, rtol=1e-4)
    for name, g, w in zip("qkv", got, want, strict=True):
        assert relative_error(g, w) < TOLERANCE, f"d{name}"


@pytest.mark.parametrize("shape", SHAPES)
def test_gathered_read_matches_masked_attention_forward_and_backward(shape):
    def read(q, k, v, ids, scale):
        return gather_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale)

    check_against_dense(read, shape)


@pytest.mark.parametrize("backward", ["keys", "scatter"])
def test_gathered_backward_matches_by_either_route(backward, monkeypatch):
    monkeypatch.setattr(gather_sparse, "BACKWARD", backward)

    def read(q, k, v, ids, scale):
        return gather_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale)

    check_against_dense(read, (1, 1300, 8, 1, 64, 512))


@pytest.mark.parametrize("shape", CUDA_SHAPES)
def test_cuda_read_matches_masked_attention_forward_and_backward(shape):
    require_hopper()
    if shape[2] // shape[3] not in cuda_sparse.GROUPS:
        pytest.skip("the CUDA kernels take no such group of query heads")

    def read(q, k, v, ids, scale):
        return cuda_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale)

    check_against_dense(read, shape)


def test_cuda_read_refuses_a_head_dim_it_has_no_kernel_for():
    q, k, v, generator = attention_inputs(1, 1024, 8, 2, 256, 0)
    ids = random_selection(1, 1024, 512, generator)
    with pytest.raises(ValueError, match="bf16 d 64 or 128"):
        cuda_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, 256**-0.5)


@pytest.mark.parametrize("length", [1000, 1024, 3000])
def test_cuda_v2_read_matches_masked_attention_forward_and_backward(length):
    require_hopper()

    def read(q, k, v, ids, scale):
        return cuda_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale, v2=True)

    check_against_dense(read, (2, length, 8, 1, 128, 1024))


@pytest.mark.parametrize("shape", SHAPES[:5])
def test_flash_read_over_a_selection_matches_masked_attention(shape):
    batch, length, query_heads, kv_heads, head_dim, top_k = shape
    if head_dim not in dsa_sparse.HEAD_DIMS:
        pytest.skip("not a head dim of the read")

    def read(q, k, v, ids, scale):
        selection = pack(keep_table(ids, length)[:, None].expand(-1, query_heads, -1, -1))
        return dsa_sparse.attention(q, k, v, selection, WINDOW_SIZE, SINK_SIZE, scale)

    check_against_dense(read, shape)


@pytest.mark.parametrize("pipeline", ["union-hybrid", "split-v14"])
def test_cuda_v2_pipelines_each_match_masked_attention(pipeline, monkeypatch):
    # Each pipeline is forced, since the dispatch takes the hybrid backward from 32,768 tokens and the split forward from 65,536
    require_hopper()
    union = pipeline == "union-hybrid"
    monkeypatch.setattr(cuda_sparse, "HYBRID_TOKENS", 0)
    monkeypatch.setattr(cuda_sparse._vendor("fwd_hard"), "SHARED_THR", 0.0 if union else 1.0)
    monkeypatch.setattr(cuda_sparse._vendor("bwd_hybrid"), "MODE", 1 if union else 2)

    def read(q, k, v, ids, scale):
        return cuda_sparse.attention(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale, v2=True)

    check_against_dense(read, (4, 1024, 8, 1, 128, 1024))
