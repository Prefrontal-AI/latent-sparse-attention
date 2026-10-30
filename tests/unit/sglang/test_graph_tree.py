"""The graph tree - a request row's node buckets mirrored into pooled pages at fixed addresses.

Runs on a CPU through the torch pairing, with node cluster blocks of 64 and a growth at every page end so a few pages
grow several levels. The mirror is held to the hierarchy it copies, level by level and in creation order.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.graph_tree import NODE_BUCKET_PAGE, GraphTree
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
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 64,
    "beam_width": 2048,
}


@pytest.fixture(autouse=True)
def every_page(monkeypatch):
    # The first level pairs at every page end, so pages grow, close and stack levels one by one
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)


def spread_keys(count, seed=0):
    """Index keys in runs of 12 around one anchor each, so groups of buckets have structure to pair"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, 16, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 16, generator=generator)
    return F.normalize(keys, dim=-1).to(torch.bfloat16)


class Pool:
    """A request's keys as the pool holds them - index keys by position, and the fingerprints of whole pages"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, positions):
        return self.keys[positions.long()]

    def fingerprints(self, first, count):
        stored = self.keys[first * PAGE_SIZE : (first + count) * PAGE_SIZE]
        return page_fingerprints(stored, torch.ones(len(stored), dtype=torch.bool)).tolist()


def make_cache(precision="bf16", requests=3, max_context=5 * PAGE_SIZE, reuse=False, serving=None, **indexer):
    serving = {"node_centroid_precision": precision, "decode": "descend", **(serving or {})}
    settings = DesignSettings.from_blocks({**INDEXER, **indexer}, serving)
    return BucketCache(settings, max_context, requests, max_context, "cpu", reuse)


def serve(cache, row, rid, keys, prefix=0, chunk=PAGE_SIZE):
    """One request forward by forward - admitted with `prefix` keys cached, then `chunk` keys at a time"""
    pool = Pool(keys)
    cache.ensure(row, rid, 1)
    cache.adopt(row, prefix, pool)
    end = prefix
    while end < len(keys):
        end = min(end + chunk, len(keys))
        cache.advance(row, end, pool, {row})
        cache.fingerprint(row, end, pool)


def creation_order(hierarchy, level, name):
    """One field of a level's node buckets in creation order - the closed ones, then the open"""
    parts = [
        rows[name]
        for rows in (hierarchy.closed_node_buckets[level - 1], hierarchy.open_node_buckets[level - 1])
        if len(rows)
    ]
    return torch.cat(parts)


def mirrored(tree, row, level, name):
    """One field of a row's level, read back through its page table in creation order"""
    made = int(tree.made[level - 1, row])
    index = torch.arange(made)
    physical = (
        tree.pages[level - 1][row][index // NODE_BUCKET_PAGE].long() * NODE_BUCKET_PAGE
        + index % NODE_BUCKET_PAGE
    )
    return getattr(tree, name)[level - 1][physical]


def assert_mirrors(cache, row):
    """The row's mirror holds its hierarchy's node buckets and counts"""
    tree, hierarchy = cache.graph_tree, cache.hierarchies[row]
    for level in range(1, tree.level_count + 1):
        if level > hierarchy.levels:
            assert int(tree.made[level - 1, row]) == int(tree.closed[level - 1, row]) == 0
            assert (tree.pages[level - 1][row] == -1).all()
            continue
        assert int(tree.made[level - 1, row]) == len(hierarchy.open_node_buckets[level - 1]) + len(
            hierarchy.closed_node_buckets[level - 1]
        )
        assert int(tree.closed[level - 1, row]) == len(hierarchy.closed_node_buckets[level - 1])
        for name in ("node_centroids", "children"):
            want, got = creation_order(hierarchy, level, name), mirrored(tree, row, level, name)
            assert torch.equal(got.float(), want.float())
    assert int(tree.start[row]) == (hierarchy.start_level() if hierarchy.levels else 0)
    assert int(tree.covered[row]) == hierarchy.covered_buckets


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("start_nodes", [None, 300])
def test_the_mirror_reads_back_every_levels_node_buckets_in_creation_order(precision, start_nodes):
    cache = make_cache(precision, start_nodes=start_nodes)
    keys = spread_keys(4 * PAGE_SIZE + 3 * WINDOW + 100)
    serve(cache, 1, "request", keys)
    hierarchy = cache.hierarchies[1]
    assert hierarchy.levels == 4
    assert any(len(shut) for shut in hierarchy.closed_node_buckets)
    assert_mirrors(cache, 1)


def test_the_mirror_follows_the_hierarchy_page_by_page_as_levels_close_and_stack():
    cache = make_cache()
    pool = Pool(spread_keys(4 * PAGE_SIZE))
    cache.ensure(1, "request", 1)
    levels = []
    for end in range(PAGE_SIZE, 4 * PAGE_SIZE + 1, PAGE_SIZE):
        cache.advance(1, end, pool, {1})
        assert_mirrors(cache, 1)
        levels.append(cache.hierarchies[1].levels)
    assert levels == sorted(levels) and levels[-1] > levels[0]


def test_rows_at_different_lengths_mirror_independently_and_one_below_the_first_level_holds_nothing():
    cache = make_cache()
    serve(cache, 0, "short", spread_keys(PAGE_SIZE // 4, seed=1))
    serve(cache, 1, "long", spread_keys(4 * PAGE_SIZE + 100, seed=2))
    serve(cache, 2, "middle", spread_keys(2 * PAGE_SIZE, seed=3))
    tree = cache.graph_tree
    assert cache.hierarchies[0].levels == 0
    assert int(tree.start[0]) == 0 and int(tree.made[:, 0].sum()) == 0
    assert tree.held(0) == {"graph_node_centroids": 0, "graph_children": 0, "graph_pages": 0}
    assert cache.hierarchies[1].levels > cache.hierarchies[2].levels > 0
    for row in range(3):
        assert_mirrors(cache, row)


def test_a_row_takes_a_page_a_level_for_its_first_node_buckets_and_another_past_a_pages_worth():
    cache = make_cache()
    serve(cache, 1, "request", spread_keys(4 * PAGE_SIZE))
    tree, hierarchy = cache.graph_tree, cache.hierarchies[1]
    first = len(hierarchy.open_node_buckets[0]) + len(hierarchy.closed_node_buckets[0])
    taken = [len(held) for held in tree.held_pages[1]]
    assert taken[0] == -(-first // NODE_BUCKET_PAGE) > 1
    assert all(count == 1 for count in taken[1:])
    assert (tree.pages[0][1, : taken[0]] >= 0).all() and (tree.pages[0][1, taken[0] :] == -1).all()


def test_a_released_row_gives_its_pages_back_and_zeroes_its_counts():
    cache = make_cache()
    tree = cache.graph_tree
    free = [len(free) for free in tree.free]
    serve(cache, 1, "request", spread_keys(4 * PAGE_SIZE))
    assert [len(free) for free in tree.free] != free
    assert sum(tree.held(1).values()) > 0
    cache.release(1)
    assert [len(free) for free in tree.free] == free
    assert all((pages == -1).all() for pages in tree.pages)
    assert (
        int(tree.made.sum())
        == int(tree.closed.sum())
        == int(tree.start.sum())
        == int(tree.covered.sum())
        == 0
    )
    assert sum(tree.held(1).values()) == 0


def test_a_new_request_in_a_released_row_mirrors_its_own_hierarchy():
    cache = make_cache()
    serve(cache, 1, "first", spread_keys(4 * PAGE_SIZE, seed=1))
    serve(cache, 1, "second", spread_keys(2 * PAGE_SIZE + 100, seed=2))
    assert cache.hierarchies[1].levels > 0
    assert_mirrors(cache, 1)


def test_adopting_a_prefix_resyncs_the_row_to_the_truncated_hierarchy_and_growth_goes_on_from_it():
    cache = make_cache(reuse=True)
    keys = spread_keys(4 * PAGE_SIZE + 100)
    serve(cache, 0, "donor", keys)
    deep = cache.hierarchies[0].levels
    serve(cache, 1, "taker", keys[: 2 * PAGE_SIZE + 50], prefix=2 * PAGE_SIZE)
    assert cache.hierarchies[1].covered_buckets > 0
    assert cache.hierarchies[1].levels < deep
    assert_mirrors(cache, 1)
    assert_mirrors(cache, 0)
    pool = Pool(keys)
    cache.advance(1, 4 * PAGE_SIZE, pool, {1})
    assert_mirrors(cache, 1)


def test_the_mirror_holds_the_bytes_of_the_pages_a_row_holds():
    cache = make_cache("fp8")
    serve(cache, 1, "request", spread_keys(4 * PAGE_SIZE))
    tree = cache.graph_tree
    pages = sum(len(held) for held in tree.held_pages[1])
    held = tree.held(1)
    assert held["graph_node_centroids"] == pages * NODE_BUCKET_PAGE * 16
    assert held["graph_children"] == pages * NODE_BUCKET_PAGE * 4 * 4
    assert held["graph_pages"] == pages * 4
    assert set(held).issubset(cache.held(1))


def test_the_pool_bytes_are_what_the_tensors_allocate():
    cache = make_cache("fp8", requests=5)
    tree = cache.graph_tree
    allocated = sum(
        tensor.numel() * tensor.element_size()
        for tensor in [
            *tree.node_centroids,
            *tree.children,
            *tree.pages,
            tree.made,
            tree.closed,
            tree.start,
            tree.covered,
        ]
    )
    rows = cache.bucket_centroids.shape[0]
    assert GraphTree.pool_bytes(cache.settings, rows, 5, 5 * PAGE_SIZE) == allocated


def test_the_pools_hold_the_levels_the_longest_context_grows_and_pages_for_every_bucket_row():
    cache = make_cache()
    tree, rows = cache.graph_tree, cache.bucket_centroids.shape[0]
    assert tree.level_count == 4
    for level in range(1, 5):
        assert tree.node_centroids[level - 1].shape[0] >= rows // 4**level
        assert tree.node_centroids[level - 1].shape[0] % NODE_BUCKET_PAGE == 0


def test_a_context_that_grows_more_levels_than_the_cache_was_sized_for_raises():
    cache = make_cache(max_context=PAGE_SIZE)
    other = make_cache()
    serve(other, 1, "request", spread_keys(4 * PAGE_SIZE))
    small = GraphTree(cache.settings, 10, 3, 0, "cpu")
    assert small.level_count == 0
    with pytest.raises(ValueError, match="holds 0 levels"):
        small.sync(1, other.hierarchies[1])


def test_a_hierarchy_that_lost_node_buckets_since_the_last_sync_raises_until_the_row_is_reset():
    cache = make_cache()
    serve(cache, 1, "request", spread_keys(4 * PAGE_SIZE))
    donor = make_cache()
    serve(donor, 1, "request", spread_keys(2 * PAGE_SIZE))
    with pytest.raises(ValueError, match="reset the row first"):
        cache.graph_tree.sync(1, donor.hierarchies[1])
    cache.graph_tree.reset(1)
    cache.graph_tree.sync(1, donor.hierarchies[1])
    assert_mirrors_of(cache.graph_tree, donor.hierarchies[1], 1)


def assert_mirrors_of(tree, hierarchy, row):
    for level in range(1, hierarchy.levels + 1):
        for name in ("node_centroids", "children"):
            assert torch.equal(
                mirrored(tree, row, level, name).float(), creation_order(hierarchy, level, name).float()
            )


def test_a_pool_without_a_free_page_raises():
    cache = make_cache()
    tree = cache.graph_tree
    tree.free[0].clear()
    with pytest.raises(RuntimeError, match="out of pages"):
        serve(cache, 1, "request", spread_keys(PAGE_SIZE))


def test_shaped_node_buckets_raise_at_construction_saying_what_arrived():
    shaped = DesignSettings.from_blocks(
        {**INDEXER, "index_dim": 32, "node_bucket_shape": {"rank": 2, "deviations": 2}}, {"decode": "descend"}
    )
    with pytest.raises(ValueError, match="node_bucket_shape"):
        GraphTree(shaped, 4 * PAGE_SIZE, 3, 4 * PAGE_SIZE, "cpu")


def test_a_cache_of_shaped_node_buckets_builds_no_mirror():
    assert make_cache(index_dim=32, node_bucket_shape={"rank": 2, "deviations": 2}).graph_tree is None


def test_only_a_descending_hierarchical_design_builds_a_mirror():
    flat = {
        key: value for key, value in INDEXER.items() if not key.startswith(("node_", "beam_", "top_level"))
    }
    flat_design = DesignSettings.from_blocks(flat, {"decode": "flat"})
    assert BucketCache(flat_design, PAGE_SIZE, 3, PAGE_SIZE, "cpu").graph_tree is None
    assert make_cache(serving={"decode": "flat"}).graph_tree is None
