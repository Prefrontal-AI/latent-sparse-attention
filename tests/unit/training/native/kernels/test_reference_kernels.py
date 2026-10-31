"""Every kernel against the reference training code's own, on small shapes, forward and gradient.

Reads the clone `REFERENCE_CLONE` names, so it runs on the cluster only.
The reference training code answers from a worker process of its own, since two builds of one CUDA source cannot share a process.
Each test prints the errors it measured, for the pull request that ports the kernels.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.training.native.kernels import (
    cuda_sparse,
    dsa_sparse,
    fa3_index,
    gather_sparse,
    lsa_rank,
    lsa_select,
)
from tests.helpers.training.kernels.inputs import (
    BUCKET_CAP,
    BUCKET_SIZE,
    POOL_SIZE,
    SINK_SIZE,
    WINDOW_SIZE,
    attention_inputs,
    make_buckets,
    max_abs_error,
    random_selection,
    relative_error,
    require_hopper,
    run_reference,
    unit_rows,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]

# One Triton or CUDA source, so a read is expected to agree to rounding and usually exactly
TOLERANCE = 1e-3


def report(name, pairs):
    for label, got, want in pairs:
        print(
            f"ERRORS {name} {label}: max_abs={max_abs_error(got, want):.3e} relative={relative_error(got, want):.3e}"
        )


def compare_reads(name, ours, function, shape, patches=(), **kwargs):
    batch, length, query_heads, kv_heads, head_dim, top_k = shape
    q, k, v, generator = attention_inputs(batch, length, query_heads, kv_heads, head_dim, 0)
    ids = random_selection(batch, length, top_k, generator)
    scale = head_dim**-0.5
    grad_output = torch.randn(
        batch, length, query_heads, head_dim, device="cuda", generator=generator
    ).bfloat16()
    args = [q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale]
    outputs, grads = run_reference(
        function, args, kwargs, leaves=(0, 1, 2), grad_output=grad_output, patches=patches
    )
    got_output, got_lse = ours(q, k, v, ids, WINDOW_SIZE, SINK_SIZE, scale, **kwargs)
    got_grads = torch.autograd.grad(got_output, (q, k, v), grad_output)
    pairs = [("output", got_output, outputs[0]), ("lse", got_lse, outputs[1])]
    pairs += [(label, g, w) for label, g, w in zip(("dq", "dk", "dv"), got_grads, grads, strict=True)]
    report(name, pairs)
    for label, g, w in pairs:
        assert relative_error(g, w) < TOLERANCE, label


GATHER_SHAPES = [
    pytest.param((2, 1000, 8, 1, 128, 512), id="eight-over-one"),
    pytest.param((2, 1200, 4, 2, 64, 512), id="grouped"),
    pytest.param((1, 1500, 1, 1, 64, 512), id="one-head"),
]


@pytest.mark.parametrize("shape", GATHER_SHAPES)
def test_gathered_read_matches_reference_training_code(shape):
    compare_reads("gather_sparse", gather_sparse.attention, "gather_sparse.attention", shape)


@pytest.mark.parametrize("shape", GATHER_SHAPES)
def test_cuda_read_matches_reference_training_code(shape):
    require_hopper()
    compare_reads("cuda_sparse", cuda_sparse.attention, "cuda_sparse.attention", shape)


@pytest.mark.parametrize("length", [1000, 3000])
def test_cuda_v2_read_matches_reference_training_code(length):
    require_hopper()
    compare_reads(
        "cuda_sparse v2",
        cuda_sparse.attention,
        "cuda_sparse.attention",
        (2, length, 8, 1, 128, 1024),
        v2=True,
    )


def test_top_k_bitmask_and_flash_read_match_reference_training_code():
    generator = torch.Generator(device="cuda").manual_seed(0)
    batch, heads, kv_heads, length, dim, top_k = 2, 4, 2, 1024, 64, 256
    index_queries = torch.randn(batch, heads, length, dim, device="cuda", generator=generator).bfloat16()
    index_keys = torch.randn(batch, kv_heads, length, dim, device="cuda", generator=generator).bfloat16()
    ours = dsa_sparse.select(index_queries, index_keys, 128, 4, top_k, dim**-0.5)
    outputs, _ = run_reference("dsa_sparse.select", [index_queries, index_keys, 128, 4, top_k, dim**-0.5])
    assert torch.equal(ours.bits.cpu(), outputs[0])
    assert torch.equal(ours.occupancy.cpu(), outputs[1])
    print("ERRORS dsa_sparse.select bits and occupancy: mismatched words=0")

    q, k, v, _ = attention_inputs(batch, length, heads, kv_heads, 64, 1)
    grad_output = torch.randn(batch, length, heads, 64, device="cuda", generator=generator).bfloat16()
    selection = {"selection": [ours.bits.cpu(), ours.occupancy.cpu()]}
    outputs, grads = run_reference(
        "dsa_sparse.attention", [q, k, v, selection, 128, 4, 0.125], leaves=(0, 1, 2), grad_output=grad_output
    )
    got_output, got_lse = dsa_sparse.attention(q, k, v, ours, 128, 4, 0.125)
    got_grads = torch.autograd.grad(got_output, (q, k, v), grad_output)
    pairs = [("output", got_output, outputs[0]), ("lse", got_lse, outputs[1])]
    pairs += [(label, g, w) for label, g, w in zip(("dq", "dk", "dv"), got_grads, grads, strict=True)]
    report("dsa_sparse.attention", pairs)
    for label, g, w in pairs:
        assert relative_error(g, w) < TOLERANCE, label


@pytest.mark.parametrize(
    ("batch", "length", "heads", "dim", "top_k"),
    [
        pytest.param(2, 1000, 1, 128, 1024, id="one-head"),
        pytest.param(1, 4096, 8, 128, 1024, id="eight-heads"),
        pytest.param(2, 2048, 1, 64, 512, id="head-dim-64"),
    ],
)
def test_selections_match_reference_training_code(batch, length, heads, dim, top_k):
    top_p = top_k * BUCKET_CAP // BUCKET_SIZE**2
    generator = torch.Generator(device="cuda").manual_seed(0)
    keys = unit_rows((batch, length, dim), generator)
    queries = unit_rows((batch, length, heads, dim), generator)
    queries = (queries[:, :, 0] if heads == 1 else queries).bfloat16()
    centroids, members = make_buckets(keys, WINDOW_SIZE, seed=1)
    positions = torch.arange(length, device="cuda")
    args = (WINDOW_SIZE, BUCKET_SIZE, top_p, top_k, WINDOW_SIZE, SINK_SIZE)
    got = lsa_select.select_indices(queries, positions, (centroids, members), *args)
    outputs, _ = run_reference("lsa_select.select_indices", [queries, positions, (centroids, members), *args])
    rows = (got.cpu() == outputs[0]).all(-1).float().mean().item()
    print(f"ERRORS lsa_select.select_indices {heads}h d{dim}: rows equal={rows:.6f}")
    assert rows > 0.999
    if torch.cuda.get_device_capability() != (9, 0):
        return
    got = lsa_rank.select_indices(queries, (centroids, members), *args, POOL_SIZE)
    outputs, _ = run_reference("lsa_rank.select_indices", [queries, (centroids, members), *args, POOL_SIZE])
    rows = (got.cpu() == outputs[0]).all(-1).float().mean().item()
    print(f"ERRORS lsa_rank.select_indices {heads}h d{dim}: rows equal={rows:.6f}")
    assert rows > 0.999


@pytest.mark.parametrize(
    ("heads", "kv_heads", "length", "top_k"), [(1, 1, 1500, 512), (8, 1, 2000, 512), (16, 2, 1000, 64)]
)
def test_the_fa3_indexer_matches_reference_training_code(heads, kv_heads, length, top_k):
    require_hopper()
    generator = torch.Generator(device="cuda").manual_seed(0)
    index_queries = F.rms_norm(
        torch.randn(1, heads, length, 64, device="cuda", generator=generator), (64,)
    ).bfloat16()
    index_keys = F.rms_norm(
        torch.randn(1, kv_heads, length, 64, device="cuda", generator=generator), (64,)
    ).bfloat16()
    got = fa3_index.select(index_queries, index_keys, 128, 4, top_k, 0.125)
    outputs, _ = run_reference("fa3_index.select", [index_queries, index_keys, 128, 4, top_k, 0.125])
    # Positions come in no order
    same = (got.sort(-1).values.cpu() == outputs[0].sort(-1).values).all(-1).float().mean().item()
    print(f"ERRORS fa3_index.select {heads}h/{kv_heads}: rows equal={same:.6f}")
    assert same > 0.999


@pytest.mark.parametrize("pipeline", ["union-hybrid", "split-v14"])
def test_cuda_v2_pipelines_each_match_reference_training_code(pipeline, monkeypatch):
    require_hopper()
    union = pipeline == "union-hybrid"
    monkeypatch.setattr(cuda_sparse, "HYBRID_TOKENS", 0)
    monkeypatch.setattr(cuda_sparse._vendor("fwd_hard"), "SHARED_THR", 0.0 if union else 1.0)
    monkeypatch.setattr(cuda_sparse._vendor("bwd_hybrid"), "MODE", 1 if union else 2)
    patches = (
        ("cuda_sparse", "HYBRID_TOKENS", 0),
        ("sparse_attn_vendor.fwd_hard", "SHARED_THR", 0.0 if union else 1.0),
        ("sparse_attn_vendor.bwd_hybrid", "MODE", 1 if union else 2),
    )
    compare_reads(
        f"cuda_sparse v2 {pipeline}",
        cuda_sparse.attention,
        "cuda_sparse.attention",
        (4, 1024, 8, 1, 128, 1024),
        patches,
        v2=True,
    )
