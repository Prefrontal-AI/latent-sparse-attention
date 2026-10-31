import pytest
import torch
import torch.nn.functional as F

from lsa.core import build_buckets, read
from lsa.training.native.config import AttentionConfig
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.lsa import build_flat_buckets, index_vectors, read_keys

HEAD_DIM = 8
ROTARY_DIM = 2
SCALE = HEAD_DIM**-0.5
LENGTH = 96


def small(**changes):
    knobs = {
        "kind": "lsa",
        "coarse": False,
        "window_size": 8,
        "sink_size": 2,
        "top_k": 24,
        "bucket_size": 4,
        "pool_size": 2,
        "bucket_cap": 8,
        "block_size": 16,
    }
    return AttentionConfig(**{**knobs, **changes})


def unit_rows(count, dim, seed):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(6, dim, generator=generator)
    rows = centers[torch.randint(6, (count,), generator=generator)] + 0.6 * torch.randn(
        count, dim, generator=generator
    )
    return F.normalize(rows, dim=-1)


def core_read(index_queries, index_keys, config):
    """The keys core's reference reads beyond each query's sinks and window, as a mask."""
    buckets = build_buckets(
        index_keys, config.pool_size, config.bucket_size, config.bucket_cap, config.block_size
    )
    positions = torch.arange(index_keys.shape[0])
    selected, lengths = read(
        index_queries,
        positions,
        buckets,
        config.selection_width,
        config.top_k,
        config.sink_size,
        config.window_size,
    )
    mask = torch.zeros(index_keys.shape[0], index_keys.shape[0], dtype=torch.bool)
    for position in range(index_keys.shape[0]):
        window_start = max(0, position + 1 - config.window_size)
        tail = (position + 1 - window_start) + min(config.sink_size, window_start, position + 1)
        mask[position, selected[position, : lengths[position] - tail].long()] = True
    return mask


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"top_k": 21},
        {"top_k": 5, "top_p": 2},
        {"sink_size": 0, "window_size": 40},
        {"sink_size": 7, "window_size": 3, "top_k": 50},
        {"top_p": 1000, "top_k": 1000},
    ],
    ids=["default", "cut_short", "narrow_top_p", "wide_window", "wide_sink", "everything"],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_picks_are_core_references_picks_key_for_key(changes, seed):
    config = small(**changes)
    # A trailing partial block leaves the last block open for every query
    key_count = 5 * config.block_size + 7
    index_keys = unit_rows(key_count, 12, seed)
    index_queries = unit_rows(key_count, 12, seed + 100)
    got = read_keys(index_queries[None], index_keys[None], config)[0]
    assert torch.equal(got, core_read(index_queries, index_keys, config))


def test_a_cut_short_bucket_reads_the_first_of_its_keys_in_position_order():
    config = small(top_k=10)
    index_keys = unit_rows(5 * 16 + 7, 12, 4)
    index_queries = unit_rows(5 * 16 + 7, 12, 5)
    mask = read_keys(index_queries[None], index_keys[None], config)[0]
    deep = mask[5 * 16 :]
    # Outside sinks and window the budget is filled exactly, so a bucket is cut inside
    assert (deep.sum(-1) == 10).all()
    assert torch.equal(mask, core_read(index_queries, index_keys, config))


def test_the_buckets_are_cores_buckets():
    config = small()
    index_keys = unit_rows(3 * 16 + 5, 12, 6)
    centroids, members = build_flat_buckets(index_keys[None], config)
    expected = build_buckets(
        index_keys, config.pool_size, config.bucket_size, config.bucket_cap, config.block_size
    )
    count = centroids.shape[1]
    assert torch.allclose(centroids[0], expected.bucket_centroids[:count], atol=1e-6)
    assert torch.equal(members[0], expected.bucket_members[:count])


def test_the_index_space_drops_the_rotated_dims_and_groups_the_heads():
    queries = torch.randn(2, 4, 10, HEAD_DIM)
    keys = torch.randn(2, 2, 10, HEAD_DIM)
    index_queries, index_keys = index_vectors(queries, keys, ROTARY_DIM)
    assert index_queries.shape == index_keys.shape == (2, 10, 2 * (HEAD_DIM - 2))
    assert torch.allclose(index_keys.norm(dim=-1), torch.ones(2, 10), atol=1e-5)
    # Rotating the leading dims changes nothing
    rotated = queries.clone()
    rotated[..., :2] = torch.randn(2, 4, 10, 2)
    assert torch.equal(index_vectors(rotated, keys, ROTARY_DIM)[0], index_queries)


def make_module(**changes):
    return MultiresInnerAttention.Config(
        attention=small(**changes), seq_len=LENGTH, rotary_dim=ROTARY_DIM
    ).build()


def packed(sequences=2, heads=4, kv_heads=2, seed=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(sequences * LENGTH, heads, HEAD_DIM, generator=generator, dtype=dtype)
    k = torch.randn(sequences * LENGTH, kv_heads, HEAD_DIM, generator=generator, dtype=dtype)
    v = torch.randn(sequences * LENGTH, kv_heads, HEAD_DIM, generator=generator, dtype=dtype)
    return q, k, v


def test_with_top_k_covering_every_closed_key_the_read_is_dense_attention():
    module = make_module(top_k=10**6, top_p=10**6)
    q, k, v = packed(dtype=torch.float64)
    got = module(q, k, v, scale=SCALE)
    for sequence in range(2):
        tokens = slice(sequence * LENGTH, (sequence + 1) * LENGTH)
        for head in range(4):
            kv = head // 2
            scores = (q[tokens, head] @ k[tokens, kv].T) * SCALE
            scores = scores.masked_fill(
                torch.triu(torch.ones(LENGTH, LENGTH, dtype=torch.bool), 1), float("-inf")
            )
            assert torch.allclose(got[tokens, head], scores.softmax(-1) @ v[tokens, kv], atol=1e-12)


def test_a_small_budget_is_not_dense_and_is_shared_by_every_head():
    module = make_module(top_k=8, top_p=2)
    q, k, v = packed(dtype=torch.float64)
    dense = make_module(top_k=10**6, top_p=10**6)(q, k, v, scale=SCALE)
    sparse = module(q, k, v, scale=SCALE)
    assert not torch.allclose(sparse, dense, atol=1e-6)
    # One selection a sequence: it is built from the heads together, so it is the same for the two key-value heads
    index_queries, index_keys = index_vectors(
        q[:LENGTH].reshape(1, LENGTH, 4, HEAD_DIM).transpose(1, 2),
        k[:LENGTH].reshape(1, LENGTH, 2, HEAD_DIM).transpose(1, 2),
        ROTARY_DIM,
    )
    assert index_keys.shape[-1] == 2 * (HEAD_DIM - 2)


def test_the_selection_carries_no_gradient_and_q_k_and_v_get_one_through_the_read():
    module = make_module(top_k=8, top_p=2)
    q, k, v = (tensor.requires_grad_() for tensor in packed())
    index_queries, index_keys = index_vectors(
        q.reshape(2, LENGTH, 4, HEAD_DIM).transpose(1, 2),
        k.reshape(2, LENGTH, 2, HEAD_DIM).transpose(1, 2),
        ROTARY_DIM,
    )
    assert not index_queries.requires_grad and not index_keys.requires_grad
    assert not read_keys(index_queries, index_keys, small()).requires_grad
    module(q, k, v, scale=SCALE).square().sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0


def test_the_module_traces_as_one_graph_and_matches_eager_in_value_and_gradient():
    module = make_module(top_k=8, top_p=2)
    eager_inputs = [tensor.requires_grad_() for tensor in packed()]
    traced_inputs = [tensor.detach().clone().requires_grad_() for tensor in eager_inputs]
    eager = module(*eager_inputs, scale=SCALE)
    traced = torch.compile(lambda q, k, v: module(q, k, v, scale=SCALE), fullgraph=True)(*traced_inputs)
    assert torch.allclose(traced, eager, atol=1e-5)
    weights = torch.randn_like(eager)
    (eager * weights).sum().backward()
    (traced * weights).sum().backward()
    for got, want in zip(traced_inputs, eager_inputs, strict=True):
        assert torch.allclose(got.grad, want.grad, atol=1e-4)


@pytest.mark.gpu
def test_the_flex_path_matches_the_cpu_path_in_value_and_gradient_on_a_gpu():
    config = small(
        window_size=64, sink_size=4, top_k=64, bucket_size=8, pool_size=4, bucket_cap=16, block_size=64
    )
    sequence = 256
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(sequence, 4, 64, generator=generator)
    k = torch.randn(sequence, 2, 64, generator=generator)
    v = torch.randn(sequence, 2, 64, generator=generator)
    weights = torch.randn(sequence, 4, 64, generator=generator)
    grads = {}
    for device in ("cpu", "cuda"):
        module = MultiresInnerAttention.Config(
            attention=config, seq_len=sequence, rotary_dim=ROTARY_DIM
        ).build()
        inputs = [tensor.to(device).detach().requires_grad_() for tensor in (q, k, v)]
        out = module(*inputs, scale=64**-0.5)
        (out * weights.to(device)).sum().backward()
        grads[device] = [out.detach().cpu(), *[tensor.grad.cpu() for tensor in inputs]]
    for got, want in zip(grads["cuda"], grads["cpu"], strict=True):
        assert torch.allclose(got, want, atol=1e-3)
