"""The bucket cache's bookkeeping - pages, requests, the row maps, the counts a decode step reads.

Runs on a CPU. The cache clusters through the torch rounds there, so these tests hold what the cache does with
buckets - where it puts them, who may see them, what it gives back when a request ends - and not the buckets,
which `test_clustering.py` and the goldens hold.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.shape import build_shapes
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.layer_index import page_fingerprints
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

WINDOW = 64
INDEXER = {
    "window_size": WINDOW,
    "index_dim": 16,
    "pool_size": 4,
    "bucket_size": 8,
    "bucket_cap": 16,
    "top_k": 64,
    "top_p": 16,
}
ROWS_PER_PAGE = PAGE_SIZE // 8


def keys_of(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(count, 16, generator=generator), dim=-1).to(torch.bfloat16)


def lookup(keys):
    return lambda positions: keys[positions.long()]


def make_cache(
    kv_size=3 * PAGE_SIZE, requests=3, max_context=4 * PAGE_SIZE, reuse=False, spare_contexts=0, **indexer
):
    settings = DesignSettings.from_blocks({**INDEXER, **indexer}, {"spare_contexts": spare_contexts})
    return BucketCache(settings, kv_size, requests, max_context, "cpu", reuse)


def fill(cache, row, rid, keys, step=1, chunk=3000, active=None):
    cache.ensure(row, rid, step)
    end = 0
    while end < len(keys):
        end = min(end + chunk, len(keys))
        cache.advance(row, end, lookup(keys), active or {row})


def rows_of_request(cache, row):
    return cache.logical_rows(row, cache.visible(row))[: cache.visible(row)]


def test_a_request_takes_a_page_as_its_keys_reach_it():
    cache = make_cache()
    keys = keys_of(PAGE_SIZE + 100)
    fill(cache, 1, "a", keys)
    assert cache.pages[1] == [cache.request_pages[1, 0].item(), cache.request_pages[1, 1].item()]
    assert cache.request_pages[1, 2].item() == -1
    assert cache.page_positions[cache.pages[1][1]].item() == PAGE_SIZE


def test_a_lone_requests_pages_are_consecutive_rows_in_order():
    cache = make_cache()
    fill(cache, 1, "a", keys_of(2 * PAGE_SIZE + 10))
    assert cache.run[1] == 3
    assert cache.pages[1] == [0, 1, 2]
    rows = cache.run_rows(1, cache.visible(1))
    assert rows is not None
    assert torch.equal(torch.arange(rows.start, rows.stop), cache.logical_rows(1, cache.visible(1)))


def test_interleaved_requests_leave_a_gather_where_the_run_breaks():
    cache = make_cache()
    first, second = keys_of(2 * PAGE_SIZE, 1), keys_of(2 * PAGE_SIZE, 2)
    cache.ensure(0, "a", 1)
    cache.ensure(1, "b", 1)
    for end in range(PAGE_SIZE // 2, 2 * PAGE_SIZE + 1, PAGE_SIZE // 2):
        cache.advance(0, end, lookup(first), {0, 1})
        cache.advance(1, end, lookup(second), {0, 1})
    assert cache.pages[0] != cache.pages[1]
    assert cache.run[0] < len(cache.pages[0])
    count = cache.visible(0)
    assert cache.run_rows(0, count) is None
    # The gather still reads each page's own rows, not the other request's
    assert torch.equal(
        cache.gather(0, count)[:count], cache.bucket_centroids[cache.logical_rows(0, count)[:count]]
    )


def test_a_request_alone_or_beside_another_leaves_the_same_buckets():
    # invariant 27
    first, second = keys_of(2 * PAGE_SIZE, 1), keys_of(2 * PAGE_SIZE, 2)
    alone = make_cache()
    fill(alone, 0, "a", first)
    shared = make_cache()
    shared.ensure(0, "a", 1)
    shared.ensure(1, "b", 1)
    for end in range(PAGE_SIZE // 2, 2 * PAGE_SIZE + 1, PAGE_SIZE // 2):
        shared.advance(0, end, lookup(first), {0, 1})
        shared.advance(1, end, lookup(second), {0, 1})
    count = alone.visible(0)
    assert shared.visible(0) == count
    for name in ("bucket_members", "bucket_sizes", "bucket_centroids"):
        assert torch.equal(
            getattr(alone, name)[rows_of_request(alone, 0)], getattr(shared, name)[rows_of_request(shared, 0)]
        )


def test_a_cluster_block_closes_with_its_last_key():
    cache = make_cache()
    cache.ensure(1, "a", 1)
    keys = keys_of(1000)
    cache.advance(1, WINDOW - 1, lookup(keys), {1})
    assert (cache.visible(1), cache.closed_keys[1], cache.end[1]) == (0, 0, WINDOW - 1)
    cache.advance(1, WINDOW, lookup(keys), {1})
    assert (cache.visible(1), cache.closed_keys[1]) == (WINDOW // 8, WINDOW)
    cache.advance(1, 3 * WINDOW + 5, lookup(keys), {1})
    assert (cache.visible(1), cache.closed_keys[1], cache.end[1]) == (
        3 * WINDOW // 8,
        3 * WINDOW,
        3 * WINDOW + 5,
    )


def test_every_key_of_a_request_is_in_one_bucket_across_pages():
    cache = make_cache()
    fill(cache, 1, "a", keys_of(2 * PAGE_SIZE + 3 * WINDOW + 7))
    members = cache.bucket_members[rows_of_request(cache, 1)]
    held = members[members >= 0].long()
    assert torch.equal(held.sort().values, torch.arange(2 * PAGE_SIZE + 3 * WINDOW))
    assert torch.equal(cache.bucket_sizes[rows_of_request(cache, 1)], (members >= 0).sum(1).int())


def test_the_query_lengths_count_the_buckets_of_the_blocks_before_a_querys_own():
    cache = make_cache()
    positions = torch.tensor(
        [0, WINDOW - 1, WINDOW, 3 * WINDOW + 5, PAGE_SIZE + 2 * WINDOW], dtype=torch.int32
    )
    lengths = cache.query_lengths(positions)
    assert lengths.dtype == torch.int32
    per_block = WINDOW // 8
    assert lengths.tolist() == [0, 0, per_block, 3 * per_block, PAGE_SIZE // 8 + 2 * per_block]


def test_the_decode_counts_follow_the_closed_keys_of_each_request():
    cache = make_cache()
    fill(cache, 0, "a", keys_of(5 * WINDOW + 1))
    fill(cache, 2, "b", keys_of(2 * WINDOW + 40), step=2)
    counts = cache.decode_counts(torch.tensor([0, 1, 2]))
    assert counts.dtype == torch.int32
    assert counts.tolist() == [5 * WINDOW // 8, 0, 2 * WINDOW // 8]
    assert counts.tolist() == [cache.visible(0), cache.visible(1), cache.visible(2)]


def test_physical_maps_logical_buckets_through_the_request_pages_and_keeps_none():
    cache = make_cache()
    other = keys_of(PAGE_SIZE, 3)
    cache.ensure(0, "other", 1)
    cache.advance(0, PAGE_SIZE, lookup(other), {0})
    fill(cache, 1, "a", keys_of(2 * PAGE_SIZE), step=2)
    logical = torch.tensor([[0, 5, ROWS_PER_PAGE + 3, -1, 2 * ROWS_PER_PAGE]], dtype=torch.int32)
    physical = cache.physical(torch.tensor([1]), logical)
    first, second = cache.pages[1]
    assert physical.tolist() == [
        [first * ROWS_PER_PAGE, first * ROWS_PER_PAGE + 5, second * ROWS_PER_PAGE + 3, -1, -1]
    ]


def test_a_released_request_leaves_nothing_visible_and_its_pages_free():
    # invariant 29
    cache = make_cache()
    fill(cache, 1, "a", keys_of(PAGE_SIZE + 2 * WINDOW))
    held = list(cache.pages[1])
    free = len(cache.free)
    cache.release(1)
    assert cache.visible(1) == 0
    assert cache.decode_counts(torch.tensor([1])).tolist() == [0]
    assert (cache.request_pages[1] == -1).all()
    assert (cache.end[1], cache.closed_keys[1], cache.pages[1], cache.rid[1]) == (0, 0, [], None)
    assert len(cache.free) == free + len(held)
    assert set(held) <= set(cache.free)


def test_a_row_taken_by_a_new_request_drops_what_the_old_one_indexed():
    # invariant 26
    keys = keys_of(PAGE_SIZE + 2 * WINDOW, 4)
    reused = make_cache()
    fill(reused, 1, "old", keys_of(PAGE_SIZE + 5 * WINDOW, 5))
    fill(reused, 1, "new", keys, step=2)
    fresh = make_cache()
    fill(fresh, 1, "new", keys)
    assert reused.visible(1) == fresh.visible(1)
    for name in ("bucket_members", "bucket_sizes", "bucket_centroids"):
        assert torch.equal(
            getattr(reused, name)[rows_of_request(reused, 1)], getattr(fresh, name)[rows_of_request(fresh, 1)]
        )


def test_the_same_request_keeps_its_pages_across_calls():
    cache = make_cache()
    cache.ensure(1, "a", 1)
    cache.advance(1, 100, lookup(keys_of(200)), {1})
    pages = list(cache.pages[1])
    cache.ensure(1, "a", 2)
    cache.advance(1, 200, lookup(keys_of(200)), {1})
    assert cache.pages[1] == pages
    assert cache.seen[1] == 2


def test_a_full_cache_reclaims_the_oldest_idle_request_and_never_a_running_one():
    # invariant 29
    cache = make_cache(kv_size=0, requests=3, max_context=2 * PAGE_SIZE)
    # One page a request and the three the table adds
    assert len(cache.free) == 3
    for row, step in ((0, 1), (1, 2)):
        cache.ensure(row, f"r{row}", step)
        cache.advance(row, PAGE_SIZE, lookup(keys_of(PAGE_SIZE, row)), {row})
    cache.ensure(2, "r2", 3)
    cache.advance(2, PAGE_SIZE, lookup(keys_of(PAGE_SIZE, 2)), {2})
    assert not cache.free
    # The next page of row 2 takes the oldest idle request's - row 0, seen at step 1
    cache.advance(2, PAGE_SIZE + 10, lookup(keys_of(PAGE_SIZE + 10, 2)), {2})
    assert cache.pages[0] == [] and cache.rid[0] is None
    assert cache.pages[1] != [] and len(cache.pages[2]) == 2


def test_a_cache_with_every_page_in_a_running_batch_is_exhausted():
    cache = make_cache(kv_size=0, requests=3, max_context=2 * PAGE_SIZE)
    for row in range(3):
        cache.ensure(row, f"r{row}", row)
        cache.advance(row, PAGE_SIZE, lookup(keys_of(PAGE_SIZE, row)), {0, 1, 2})
    with pytest.raises(RuntimeError, match="every page belongs to a running request"):
        cache.advance(0, PAGE_SIZE + 1, lookup(keys_of(PAGE_SIZE + 1)), {0, 1, 2})


def run_requests_over_one_shared_prefix(cache, requests):
    keys = keys_of(2 * PAGE_SIZE)
    active = set(range(requests))
    for row in range(requests):
        cache.ensure(row, f"r{row}", row)
        cache.advance(row, 2 * PAGE_SIZE, lookup(keys), active)


def test_requests_over_one_cached_prefix_exhaust_a_cache_with_no_spare_contexts():
    # One prefix of 2 pages in the pool, held by 4 running requests that each index it alone
    cache = make_cache(kv_size=2 * PAGE_SIZE, requests=4, max_context=2 * PAGE_SIZE)
    with pytest.raises(RuntimeError, match="raise model.serving.spare_contexts"):
        run_requests_over_one_shared_prefix(cache, 4)


def test_spare_contexts_cover_requests_that_index_one_cached_prefix_each():
    cache = make_cache(kv_size=2 * PAGE_SIZE, requests=4, max_context=2 * PAGE_SIZE, spare_contexts=4)
    run_requests_over_one_shared_prefix(cache, 4)
    assert [len(pages) for pages in cache.pages] == [2, 2, 2, 2]


@pytest.mark.parametrize(
    ("design", "serving", "levels"),
    [
        ("flat", {}, None),
        ("fixed", {"node_centroid_precision": "bf16"}, 1),
        ("tree", {"node_centroid_precision": "fp8"}, 16),
        ("tree descend", {"node_centroid_precision": "fp8", "decode": "descend"}, 16),
    ],
)
def test_each_designs_settings_build_a_cache_with_the_hierarchy_it_asks_for(design, serving, levels):
    nodes = {
        "top_level": levels,
        "node_bucket_size": 4,
        "node_cluster_block_size": 4096,
        "beam_width": 2048,
        "start_nodes": 32768 if levels == 16 else None,
    }
    indexer = {**INDEXER, "window_size": 512, "index_dim": 64, "top_k": 4096, "top_p": 1024}
    settings = DesignSettings.from_blocks({**indexer, **(nodes if levels else {})}, serving)
    cache = BucketCache(settings, 2 * PAGE_SIZE, 3, 2 * PAGE_SIZE, "cpu")
    assert cache.rows_per_page == PAGE_SIZE // 8
    assert cache.bucket_centroids.shape == (len(cache.free) * cache.rows_per_page, 64)
    assert cache.bucket_members.shape[1] == 16
    if levels is None:
        assert cache.hierarchies is None
        return
    assert len(cache.hierarchies) == 3
    assert cache.hierarchies[0].fp8 == (serving["node_centroid_precision"] == "fp8")
    assert all(hierarchy.levels == 0 for hierarchy in cache.hierarchies)


def test_a_flat_cache_without_pools_places_every_key_once():
    cache = make_cache(pool_size=None)
    fill(cache, 1, "a", keys_of(PAGE_SIZE + 4 * WINDOW))
    members = cache.bucket_members[rows_of_request(cache, 1)]
    held = members[members >= 0].long()
    assert torch.equal(held.sort().values, torch.arange(PAGE_SIZE + 4 * WINDOW))
    assert int(cache.bucket_sizes[rows_of_request(cache, 1)].min()) >= 1


def test_a_shaped_cache_shapes_each_bucket_from_its_keys_as_its_block_closes():
    generator = torch.Generator().manual_seed(4)
    keys = F.normalize(torch.randn(PAGE_SIZE + 3 * WINDOW + 10, 32, generator=generator), dim=-1).bfloat16()
    cache = make_cache(index_dim=32, bucket_shape={"rank": 2, "deviations": 2})
    fill(cache, 1, "request", keys)
    rows = rows_of_request(cache, 1)
    assert len(rows) == (PAGE_SIZE + 3 * WINDOW) // 8
    members = cache.bucket_members[rows]
    held = members >= 0
    centroids = F.normalize((keys.float()[members.clamp_min(0).long()] * held[..., None]).sum(1), dim=-1)
    want = build_shapes(keys, centroids, members, 2)
    table = cache.bucket_shapes.rows(rows)
    torch.testing.assert_close(table.variances, want.variances, rtol=2e-2, atol=1e-5)
    torch.testing.assert_close(table.floor, want.floor, rtol=2e-2, atol=1e-5)
    gathered = cache.gather_shapes(1, len(rows))
    assert torch.equal(gathered.basis[: len(rows)], table.basis)


def test_an_unshaped_cache_holds_no_shapes():
    cache = make_cache()
    assert cache.bucket_shapes is None and cache.gather_shapes(1, 16) is None


class Pool:
    """A request's keys as the pool holds them - index keys by position, and the fingerprints of whole pages"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, positions):
        return self.keys[positions.long()]

    def fingerprints(self, first, count):
        stored = self.keys[first * PAGE_SIZE : (first + count) * PAGE_SIZE]
        return page_fingerprints(stored, torch.ones(len(stored), dtype=torch.bool)).tolist()


def serve(cache, row, rid, keys, prefix=0, step=1, chunk=3000, active=None):
    """Maintenance of one request forward by forward - admitted with `prefix` keys cached, then `chunk` at a time"""
    pool, active = Pool(keys), active or {row}
    cache.ensure(row, rid, step)
    cache.adopt(row, prefix, pool)
    end = prefix
    while True:
        cache.advance(row, end, pool, active)
        cache.fingerprint(row, end, pool)
        if end == len(keys):
            return
        end = min(end + chunk, len(keys))


def buckets_of(cache, row):
    rows = rows_of_request(cache, row)
    return [getattr(cache, name)[rows] for name in ("bucket_members", "bucket_sizes", "bucket_centroids")]


def assert_same_buckets(cache, row, other, other_row):
    assert cache.visible(row) == other.visible(other_row)
    for mine, theirs in zip(buckets_of(cache, row), buckets_of(other, other_row), strict=True):
        assert torch.equal(mine, theirs)


def test_a_request_takes_a_retained_prefixs_pages_and_holds_the_buckets_it_would_have_built():
    keys = keys_of(2 * PAGE_SIZE + 700, 1)
    other = torch.cat([keys[: 2 * PAGE_SIZE], keys_of(900, 2)])
    cache = make_cache(reuse=True)
    serve(cache, 1, "a", keys)
    shared = cache.pages[1][:2]
    # The scheduler gives a's row to b, and a's two full pages stay retained
    serve(cache, 1, "b", keys_of(100, 3), step=2)
    assert [entry.pages for entry in cache.retained] == [shared]
    serve(cache, 2, "c", other, prefix=2 * PAGE_SIZE + 300, step=3)
    assert cache.pages[2][:2] == shared
    assert [cache.holders[page] for page in shared] == [2, 2]
    fresh = make_cache()
    serve(fresh, 2, "c", other, prefix=2 * PAGE_SIZE + 300)
    assert_same_buckets(cache, 2, fresh, 2)


def test_a_page_whose_keys_differ_is_not_taken_nor_any_page_after_it():
    keys = keys_of(3 * PAGE_SIZE, 1)
    second = keys.clone()
    second[PAGE_SIZE + 17] = keys[0]
    first = keys.clone()
    first[5] = keys[6]
    for changed, taken in ((second, 1), (first, 0)):
        cache = make_cache(kv_size=6 * PAGE_SIZE, reuse=True)
        serve(cache, 1, "a", keys)
        serve(cache, 2, "c", changed, prefix=3 * PAGE_SIZE, step=2)
        assert cache.pages[2][:taken] == cache.pages[1][:taken]
        assert not set(cache.pages[2][taken:]) & set(cache.pages[1])
        fresh = make_cache(kv_size=6 * PAGE_SIZE)
        serve(fresh, 2, "c", changed, prefix=3 * PAGE_SIZE)
        assert_same_buckets(cache, 2, fresh, 2)


def test_a_running_request_shares_its_pages_and_a_page_is_free_once_its_last_holder_lets_go():
    keys = keys_of(2 * PAGE_SIZE + 100, 1)
    cache = make_cache(reuse=True)
    free = len(cache.free)
    serve(cache, 1, "a", keys)
    serve(cache, 2, "c", keys, prefix=2 * PAGE_SIZE + 100, step=2)
    assert cache.pages[2][:2] == cache.pages[1][:2]
    # Both rows taken by new requests - a's prefix retained, and c's, the same pages, not kept again
    cache.ensure(1, "d", 3)
    assert [cache.holders[page] for page in cache.retained[0].pages] == [2, 2]
    cache.ensure(2, "e", 3)
    assert len(cache.retained) == 1
    assert [cache.holders[page] for page in cache.retained[0].pages] == [1, 1]
    cache.drop_oldest()
    assert len(cache.free) == free


def test_a_retained_prefix_is_kept_once_and_a_longer_one_over_it_replaces_it():
    keys = keys_of(3 * PAGE_SIZE, 1)
    cache = make_cache(kv_size=6 * PAGE_SIZE, reuse=True)
    serve(cache, 1, "a", keys[: 2 * PAGE_SIZE + 50])
    built = cache.pages[1][:2]
    serve(cache, 1, "b", keys, prefix=2 * PAGE_SIZE, step=2)
    # a's two pages retained, then b's three over them - b's replace a's
    serve(cache, 1, "c", keys_of(10, 2), step=3)
    assert [len(entry.prints) for entry in cache.retained] == [3]
    assert cache.retained[0].pages[:2] == built
    assert all(cache.holders[page] == 1 for page in cache.retained[0].pages)


def test_retained_prefixes_go_before_idle_requests_when_pages_run_short():
    cache = make_cache(kv_size=0, requests=3, max_context=2 * PAGE_SIZE, reuse=True)
    serve(cache, 0, "a", keys_of(PAGE_SIZE, 0))
    serve(cache, 0, "b", keys_of(PAGE_SIZE, 1), step=2)
    serve(cache, 1, "c", keys_of(PAGE_SIZE, 2), step=3)
    assert not cache.free and len(cache.retained) == 1
    serve(cache, 2, "d", keys_of(10, 3), step=4)
    # d's page came from a's retained prefix, not from the idle requests b and c
    assert not cache.retained and cache.rid[0] == "b" and cache.rid[1] == "c"


def test_without_reuse_a_released_request_retains_nothing_and_none_takes_a_prefix():
    keys = keys_of(2 * PAGE_SIZE, 1)
    cache = make_cache()
    serve(cache, 1, "a", keys)
    assert cache.prints[1] == []
    serve(cache, 1, "b", keys, prefix=2 * PAGE_SIZE, step=2)
    assert not cache.retained
    assert all(cache.holders[page] == 1 for page in cache.pages[1])
