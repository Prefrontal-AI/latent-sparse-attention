import pytest
import torch
import torch.nn.functional as F

from lsa.training.native.kernels import fa3_index, lsa_rank, lsa_select, reference
from lsa.training.native.kernels.dsa_sparse import select
from lsa.training.native.kernels.dsa_sparse.topk import unpack
from tests.helpers.training.kernels.inputs import (
    BUCKET_CAP,
    BUCKET_SIZE,
    POOL_SIZE,
    SINK_SIZE,
    WINDOW_SIZE,
    make_buckets,
    require_hopper,
    unit_rows,
)

pytestmark = pytest.mark.gpu


def index_inputs(batch, length, heads, dim, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    keys = unit_rows((batch, length, dim), generator)
    queries = unit_rows((batch, length, heads, dim), generator)
    queries = queries[:, :, 0] if heads == 1 else queries
    return queries.bfloat16(), keys.bfloat16()


@pytest.mark.parametrize(
    ("batch", "length", "heads", "dim", "top_k"),
    [
        pytest.param(2, 1000, 1, 128, 1024, id="one-head"),
        pytest.param(1, 4096, 8, 128, 1024, id="eight-heads"),
        pytest.param(2, 2048, 1, 64, 512, id="head-dim-64"),
        pytest.param(2, 300, 1, 128, 1024, id="shorter-than-a-block"),
    ],
)
def test_the_fused_selection_is_the_reference_selection(batch, length, heads, dim, top_k):
    top_p = top_k * BUCKET_CAP // BUCKET_SIZE**2
    queries, keys = index_inputs(batch, length, heads, dim, 0)
    buckets = make_buckets(keys.float(), WINDOW_SIZE, seed=1)
    positions = torch.arange(length, device="cuda")
    args = (WINDOW_SIZE, BUCKET_SIZE, top_p, top_k, WINDOW_SIZE, SINK_SIZE)
    got = lsa_select.select_indices(queries, positions, buckets, *args)
    # The reference scores in fp32 where the kernel's three bf16 parts sum to the same, up to the last bits of a tie
    want = reference.select_indices(queries.float(), positions, buckets, *args)
    rows = (got == want).all(-1).float().mean().item()
    print(f"ERRORS fused selection against the reference, {heads}h d{dim}: rows equal={rows:.6f}")
    assert rows > 0.995
    assert (got >= 0).any() == (length > WINDOW_SIZE + BUCKET_SIZE)


@pytest.mark.parametrize(
    ("batch", "length", "heads", "dim", "top_k"),
    [
        pytest.param(2, 1000, 1, 128, 1024, id="one-head"),
        pytest.param(1, 4096, 8, 128, 1024, id="eight-heads"),
        pytest.param(2, 2048, 1, 64, 512, id="head-dim-64"),
        pytest.param(1, 16384, 8, 128, 2048, id="top-k-2048"),
    ],
)
def test_the_ranking_kernel_selects_what_the_fused_selection_does(batch, length, heads, dim, top_k):
    require_hopper()
    top_p = top_k * BUCKET_CAP // BUCKET_SIZE**2
    queries, keys = index_inputs(batch, length, heads, dim, 0)
    buckets = make_buckets(keys.float(), WINDOW_SIZE, seed=1)
    assert lsa_rank.supported(queries, buckets, WINDOW_SIZE, top_p, POOL_SIZE)
    got = lsa_rank.select_indices(
        queries, buckets, WINDOW_SIZE, BUCKET_SIZE, top_p, top_k, WINDOW_SIZE, SINK_SIZE, POOL_SIZE
    )
    want = lsa_select.select_indices(
        queries,
        torch.arange(length, device="cuda"),
        buckets,
        WINDOW_SIZE,
        BUCKET_SIZE,
        top_p,
        top_k,
        WINDOW_SIZE,
        SINK_SIZE,
    )
    assert (got == want).all(-1).float().mean() > 0.999


def exact_top_k_mask(index_queries, index_keys, window_size, sink_size, top_k, scale):
    """Rank in the indexer's dtype, an exact tie to the later key, as the selection documents."""
    length = index_queries.shape[2]
    keys = index_keys.repeat_interleave(index_queries.shape[1] // index_keys.shape[1], dim=1)
    ranking = (torch.einsum("bhqd,bhsd->bhqs", index_queries, keys) * scale).float()
    position = torch.arange(length, device=ranking.device)
    long_range = (position[:, None] - position[None, :] >= window_size) & (position[None, :] >= sink_size)
    ranking = ranking.masked_fill(~long_range, float("-inf"))
    # Reversed, a stable sort puts the later of two tied keys first
    order = ranking.flip(-1).sort(dim=-1, descending=True, stable=True).indices
    chosen = (length - 1 - order)[..., :top_k]
    keep = torch.zeros_like(ranking, dtype=torch.bool).scatter(-1, chosen, True)
    return keep & long_range


@pytest.mark.parametrize(
    ("batch", "length", "heads", "kv_heads", "dim", "window_size", "sink_size", "top_k"),
    [
        pytest.param(2, 1024, 4, 4, 64, 128, 4, 256, id="own-keys"),
        pytest.param(1, 1000, 8, 2, 32, 192, 64, 300, id="grouped"),
        pytest.param(1, 777, 1, 1, 64, 4, 2, 200, id="short-window"),
    ],
)
def test_the_top_k_bitmask_is_the_exact_top_k(
    batch, length, heads, kv_heads, dim, window_size, sink_size, top_k
):
    generator = torch.Generator(device="cuda").manual_seed(0)
    index_queries = torch.randn(batch, heads, length, dim, device="cuda", generator=generator).bfloat16()
    index_keys = torch.randn(batch, kv_heads, length, dim, device="cuda", generator=generator).bfloat16()
    scale = dim**-0.5
    selection = select(index_queries, index_keys, window_size, sink_size, top_k, scale)
    want = exact_top_k_mask(index_queries, index_keys, window_size, sink_size, top_k, scale)
    assert torch.equal(unpack(selection.bits, length), want)


def check_within_tolerance(index_queries, index_keys, ids, window_size, sink_size, top_k, scale):
    """Every row lists its candidates where it has at most top_k, else top_k distinct ones within tolerance of the k-th."""
    batch, heads, length, _ = index_queries.shape
    group = heads // index_keys.shape[1]
    position = torch.arange(length, device="cuda")
    valid = (position[None] >= sink_size) & (position[None] <= position[:, None] - window_size)
    count = valid.sum(1)
    for b in range(batch):
        for h in range(heads):
            picked = ids[b, h].long()
            small = count <= top_k
            slots = torch.arange(top_k, device="cuda")[None]
            assert (picked[small] == torch.where(slots < count[small, None], sink_size + slots, -1)).all()
            wide = ~small
            if not wide.any():
                continue
            scores = (index_queries[b, h].float() @ index_keys[b, h // group].float().T * scale)[wide]
            row_valid, row_ids = valid[wide], picked[wide]
            scores = scores.masked_fill(~row_valid, float("-inf"))
            assert row_valid.gather(1, row_ids.clamp(0, length - 1)).all() and (row_ids >= 0).all()
            ordered = row_ids.sort(1).values
            assert (ordered[:, 1:] != ordered[:, :-1]).all()
            kth = scores.topk(top_k, dim=1).values[:, -1]
            rms = torch.sqrt((scores.masked_fill(~row_valid, 0) ** 2).sum(1) / count[wide])
            tolerance = 2 * 0.015625 * 0.75 * rms
            chosen = torch.zeros_like(row_valid).scatter_(1, row_ids, True)
            assert not ((scores > (kth + tolerance)[:, None]) & ~chosen).any()
            assert (scores.gather(1, row_ids).min(1).values >= kth - tolerance).all()


@pytest.mark.parametrize(
    ("batch", "heads", "kv_heads", "length", "top_k", "window_size", "sink_size"),
    [
        pytest.param(2, 1, 1, 1500, 512, 512, 4, id="one-head"),
        pytest.param(2, 8, 8, 1200, 256, 128, 0, id="eight-own-keys"),
        pytest.param(1, 8, 1, 2000, 512, 512, 4, id="eight-over-one"),
        pytest.param(1, 16, 2, 1000, 64, 64, 4, id="sixteen-over-two"),
        pytest.param(2, 1, 1, 300, 1024, 512, 4, id="shorter-than-the-window"),
    ],
)
def test_the_fa3_indexer_selects_the_top_k_within_its_tolerance(
    batch, heads, kv_heads, length, top_k, window_size, sink_size
):
    require_hopper()
    generator = torch.Generator(device="cuda").manual_seed(0)
    index_queries = F.rms_norm(
        torch.randn(batch, heads, length, 64, device="cuda", generator=generator), (64,)
    ).bfloat16()
    index_keys = F.rms_norm(
        torch.randn(batch, kv_heads, length, 64, device="cuda", generator=generator), (64,)
    ).bfloat16()
    scale = 64**-0.5
    ids = fa3_index.select(index_queries, index_keys, window_size, sink_size, top_k, scale)
    assert ids.shape == (batch, heads, length, top_k)
    assert ids.dtype == torch.int32
    check_within_tolerance(index_queries, index_keys, ids, window_size, sink_size, top_k, scale)


def test_the_fa3_indexer_refuses_what_it_does_not_take():
    index_queries = torch.randn(1, 1, 256, 64, device="cuda")
    assert not fa3_index.supported(index_queries, index_queries, 0)
    with pytest.raises(ValueError, match="fa3_index"):
        fa3_index.select(index_queries, index_queries, 0, 0, 16, 1.0)
