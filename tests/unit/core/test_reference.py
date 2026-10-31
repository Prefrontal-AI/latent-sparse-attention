"""The torch reference of the flat design - its boundary, and invariants 2-9 and 17-21.

A cache of four closed cluster blocks of 64 keys and a partial fifth, pools of 4 and buckets of 8 keys at a cap
of 16, so each block ends with 8 buckets.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.budget import EarlyBudget
from lsa.core import build_buckets, read

POOL, BUCKET, CAP, BLOCK, SINK, WINDOW = 4, 8, 16, 64, 4, 64
KEYS = 4 * BLOCK + 20


def unit(count, index_dim=16, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(count, index_dim, generator=generator), dim=-1).to(dtype)


def build(keys, pool=POOL):
    return build_buckets(keys, pool, BUCKET, CAP, BLOCK)


def reading(queries, positions, buckets, top_p=8, top_k=48):
    return read(queries, positions, buckets, top_p=top_p, top_k=top_k, sink_size=SINK, window_size=WINDOW)


@pytest.fixture(scope="module")
def cache():
    keys = unit(KEYS)
    return keys, build(keys)


def test_every_key_of_a_closed_block_is_in_exactly_one_bucket_of_it(cache):
    _, buckets = cache
    assert buckets.bucket_members.shape == (32, CAP)
    for block in range(4):
        members = buckets.bucket_members[block * 8 : (block + 1) * 8]
        held = members[members >= 0].sort().values
        assert torch.equal(held, torch.arange(block * BLOCK, (block + 1) * BLOCK, dtype=torch.int32))


@pytest.mark.parametrize("pool_size", [1, 4])
def test_a_pool_is_never_split_and_a_bucket_holds_pools_up_to_the_cap(pool_size):
    buckets = build(unit(2 * BLOCK, seed=2), pool_size)
    sizes = buckets.bucket_sizes
    assert sizes.max() <= CAP
    assert (sizes % pool_size == 0).all()
    for row in buckets.bucket_members:
        row = row[row >= 0]
        for pool in (row // pool_size).unique().tolist():
            assert set(range(pool * pool_size, (pool + 1) * pool_size)) <= set(row.tolist())


def test_a_block_ends_with_exactly_the_buckets_its_size_gives(cache):
    _, buckets = cache
    assert buckets.bucket_centroids.shape[0] == 4 * (BLOCK // BUCKET)
    assert buckets.block_bucket_count == BLOCK // BUCKET


def test_a_buckets_members_are_its_keys_and_its_size_counts_them(cache):
    _, buckets = cache
    assert torch.equal(buckets.bucket_sizes, (buckets.bucket_members >= 0).sum(1).to(torch.int32))
    for row in buckets.bucket_members:
        held = row[row >= 0]
        # In position order, with the padding after
        assert torch.equal(held, held.sort().values)
        assert (row[held.numel() :] == -1).all()


def test_a_centroid_is_the_normalized_sum_of_its_keys_and_of_its_pools(cache):
    keys, buckets = cache
    for row, centroid in zip(buckets.bucket_members, buckets.bucket_centroids, strict=True):
        held = row[row >= 0].long()
        by_keys = F.normalize(keys[held].sum(0), dim=0)
        pools = keys[held].view(-1, POOL, keys.shape[1]).sum(1)
        by_pools = F.normalize(pools.sum(0), dim=0)
        torch.testing.assert_close(centroid, by_keys, rtol=0, atol=1e-6)
        torch.testing.assert_close(by_pools, by_keys, rtol=0, atol=1e-6)


def test_ward_is_deterministic(cache):
    keys, buckets = cache
    again = build(keys)
    assert torch.equal(again.bucket_members, buckets.bucket_members)
    assert torch.equal(again.bucket_centroids, buckets.bucket_centroids)


def test_a_block_clustered_alone_gives_the_buckets_it_gives_in_the_cache(cache):
    keys, buckets = cache
    alone = build(keys[2 * BLOCK : 3 * BLOCK])
    together = buckets.bucket_members[16:24]
    assert torch.equal(alone.bucket_members + 2 * BLOCK * (alone.bucket_members >= 0), together)


def test_the_trailing_partial_block_is_in_no_bucket(cache):
    _, buckets = cache
    assert buckets.bucket_members.max() < 4 * BLOCK


def test_a_cap_under_twice_the_bucket_size_is_refused():
    with pytest.raises(ValueError, match="at least twice"):
        build_buckets(unit(BLOCK), POOL, BUCKET, 15, BLOCK)


@pytest.mark.parametrize(
    "sizes", [(3, 8, 16, 64), (4, 8, 16, 60), (0, 8, 16, 64)], ids=["pool", "block", "zero"]
)
def test_sizes_that_do_not_nest_are_refused(sizes):
    with pytest.raises(ValueError, match="divide|positive"):
        build_buckets(unit(BLOCK), *sizes)


def test_index_keys_of_the_wrong_dtype_or_shape_or_norm_are_refused():
    keys = unit(BLOCK)
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        build(keys.half())
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        build(keys.double())
    with pytest.raises(ValueError, match=r"\[N, index_dim\]"):
        build(keys[None])
    with pytest.raises(ValueError, match="unit rows"):
        build(keys * 1.1)
    broken = keys.clone()
    broken[3, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build(broken)


def test_bf16_unit_rows_are_accepted():
    keys = unit(2 * BLOCK, dtype=torch.bfloat16)
    assert build(keys).bucket_members.shape[0] == 16


def test_queries_and_positions_that_are_wrong_are_refused(cache):
    _, buckets = cache
    queries = unit(2, seed=5)
    positions = torch.tensor([100, 200])
    with pytest.raises(ValueError, match="unit rows"):
        reading(queries * 2, positions, buckets)
    with pytest.raises(ValueError, match="have 8 dims"):
        reading(unit(2, index_dim=8), positions, buckets)
    with pytest.raises(ValueError, match="int32 or int64"):
        reading(queries, positions.float(), buckets)
    with pytest.raises(ValueError, match="one a query"):
        reading(queries, positions[:1], buckets)
    with pytest.raises(ValueError, match="non-negative"):
        reading(queries, torch.tensor([100, -1]), buckets)
    with pytest.raises(ValueError, match="positive"):
        reading(queries, positions, buckets, top_p=0)


def test_a_query_scores_only_the_buckets_of_the_blocks_closed_before_its_own(cache):
    _, buckets = cache
    # Position 2 * BLOCK + 5 is in block 2, so the first bucket of block 2 is out of reach even as the best match
    position = 2 * BLOCK + 5
    query = buckets.bucket_centroids[16:17]
    selected, _ = reading(query, torch.tensor([position]), buckets, top_p=1, top_k=100)
    best = (query @ buckets.bucket_centroids[:16].T).argmax()
    expected = buckets.bucket_members[best]
    expected = expected[(expected >= SINK) & (expected < position + 1 - WINDOW)]
    assert expected.numel() > 0
    assert torch.equal(selected[0, : expected.numel()], expected)


def test_every_selected_id_is_below_the_position_and_none_is_read_twice(cache):
    keys, buckets = cache
    positions = torch.arange(10, KEYS, 7)
    selected, lengths = reading(keys[positions], positions, buckets)
    for row, position in enumerate(positions.tolist()):
        held = selected[row, : lengths[row]]
        assert held.max() <= position
        assert held.unique().numel() == held.numel()
        assert (selected[row, lengths[row] :] == -1).all()
        long_range = held[(held >= SINK) & (held < position + 1 - WINDOW)]
        assert (long_range < position + 1 - WINDOW).all()


def test_a_read_has_sink_top_k_and_window_keys_and_only_its_last_bucket_is_cut_short(cache):
    keys, buckets = cache
    position = KEYS - 1
    query = keys[position : position + 1]
    selected, lengths = reading(query, torch.tensor([position]), buckets, top_p=24, top_k=100)
    assert int(lengths[0]) == 100 + WINDOW + SINK
    held = selected[0, :100].long()
    picked = torch.argsort((query @ buckets.bucket_centroids.T)[0], descending=True, stable=True)[:24]
    rows = buckets.bucket_members[picked]
    walk = rows[rows >= SINK].long()
    walk = walk[walk < position + 1 - WINDOW][:100]
    assert torch.equal(held, walk)


def test_with_every_bucket_picked_the_read_covers_every_causal_position_once(cache):
    keys, buckets = cache
    position = KEYS - 1
    selected, lengths = reading(
        keys[position : position + 1], torch.tensor([position]), buckets, top_p=32, top_k=1000
    )
    covered = selected[0, : lengths[0]].sort().values
    # The partial block's keys are in the window, the closed blocks' in the buckets
    assert torch.equal(covered, torch.arange(position + 1, dtype=torch.int32))


def test_buckets_are_read_best_first(cache):
    keys, buckets = cache
    position = 3 * BLOCK + 10
    query = keys[position : position + 1]
    top_p = 6
    selected, _ = reading(query, torch.tensor([position]), buckets, top_p=top_p, top_k=1000)
    scores = (query @ buckets.bucket_centroids.T)[0, : 3 * 8]
    order = torch.argsort(scores, descending=True, stable=True)[:top_p]
    expected = buckets.bucket_members[order].flatten()
    expected = expected[(expected >= SINK) & (expected < position + 1 - WINDOW)]
    assert torch.equal(selected[0, : expected.numel()], expected)


def test_a_query_alone_reads_what_it_reads_in_a_chunk(cache):
    keys, buckets = cache
    positions = torch.tensor([70, 150, 240])
    chunk, lengths = reading(keys[positions], positions, buckets)
    for row in range(3):
        alone, alone_lengths = reading(keys[positions[row : row + 1]], positions[row : row + 1], buckets)
        assert torch.equal(alone[0], chunk[row])
        assert alone_lengths[0] == lengths[row]


def test_a_read_early_in_the_document_takes_every_visible_key():
    keys = unit(BLOCK + 30)
    buckets = build(keys)
    position = BLOCK + 3
    selected, lengths = reading(keys[position : position + 1], torch.tensor([position]), buckets, top_p=8)
    assert torch.equal(selected[0, : lengths[0]].sort().values, torch.arange(position + 1, dtype=torch.int32))
    # Under the first block there is no bucket, and the window reads everything
    selected, lengths = reading(keys[5:6], torch.tensor([5]), buckets)
    assert selected[0, : lengths[0]].tolist() == [0, 1, 2, 3, 4, 5]


def test_a_chunk_across_the_early_position_reads_each_row_at_its_own_budget(cache):
    keys, buckets = cache
    early = EarlyBudget(position=150, top_k=16, top_p=4)
    positions = torch.tensor([70, 140, 149, 150, 151, 200, 270])
    mixed, lengths = read(
        keys[positions],
        positions,
        buckets,
        top_p=8,
        top_k=48,
        sink_size=SINK,
        window_size=WINDOW,
        early_budget=early,
    )
    # Every row keeps the design's width, -1 padded
    assert mixed.shape == (7, 48 + WINDOW + SINK)
    for row, position in enumerate(positions.tolist()):
        small = position < early.position
        alone, alone_lengths = reading(
            keys[position : position + 1],
            positions[row : row + 1],
            buckets,
            top_p=early.top_p if small else 8,
            top_k=early.top_k if small else 48,
        )
        assert torch.equal(mixed[row, : alone.shape[1]], alone[0])
        assert (mixed[row, alone.shape[1] :] == -1).all()
        assert lengths[row] == alone_lengths[0]
    # The rows after the position read more than the rows before it, as their buckets allow
    assert lengths[2] < lengths[4]


def test_an_early_budget_changes_nothing_for_rows_at_or_past_its_position(cache):
    keys, buckets = cache
    positions = torch.tensor([150, 200, 270])
    plain = reading(keys[positions], positions, buckets)
    early = read(
        keys[positions],
        positions,
        buckets,
        top_p=8,
        top_k=48,
        sink_size=SINK,
        window_size=WINDOW,
        early_budget=EarlyBudget(position=150, top_k=16, top_p=4),
    )
    assert torch.equal(early[0], plain[0])
    assert torch.equal(early[1], plain[1])


def test_an_early_budget_over_the_reads_own_is_refused(cache):
    keys, buckets = cache
    positions = torch.tensor([200])
    with pytest.raises(
        ValueError, match="the early budget is at most top_p 8 and top_k 48 - got top_p 9 and top_k 16"
    ):
        read(
            keys[positions],
            positions,
            buckets,
            top_p=8,
            top_k=48,
            sink_size=SINK,
            window_size=WINDOW,
            early_budget=EarlyBudget(position=150, top_k=16, top_p=9),
        )
