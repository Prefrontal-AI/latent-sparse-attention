"""The hierarchy's levels over a cache's buckets - growth, cover, centroids and the start level.

Runs on a CPU through the torch pairing, with node cluster blocks of 64 so a few pages grow several levels. The
levels are held to the torch reference's (`lsa.core.build_hierarchy`), the beam over them is `test_selector.py`,
on a GPU, and the freeze's own reads of the core sizes are `test_goldens.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import Buckets, build_hierarchy
from lsa.core.hierarchy import growth_at
from lsa.core.shape import build_shapes
from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.hierarchy import GROWTH_BUCKETS, Hierarchy, Rows
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings, IndexerConfig

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
BUCKETS_PER_PAGE = PAGE_SIZE // 8


def spread_keys(count, seed=0):
    """Index keys in runs of 12 around one anchor each, so groups of buckets have structure to pair"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, 16, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 16, generator=generator)
    return F.normalize(keys, dim=-1).to(torch.bfloat16)


def build(pages, tail=0, precision="bf16", chunk=3000, **indexer):
    """A cache holding request 1's first `pages` pages and `tail` keys, and its keys"""
    keys = spread_keys(5 * PAGE_SIZE + 100)[: pages * PAGE_SIZE + tail]
    settings = DesignSettings.from_blocks({**INDEXER, **indexer}, {"node_centroid_precision": precision})
    cache = BucketCache(settings, len(keys), 3, len(keys), "cpu")
    cache.ensure(1, "request", 1)
    end = 0
    while end < len(keys):
        end = min(end + chunk, len(keys))
        cache.advance(1, end, lambda positions: keys[positions.long()], {1})
    return cache, keys


@pytest.fixture(scope="module")
def deep():
    return build(4, 3 * WINDOW + 100)


@pytest.fixture(scope="module")
def deep_fp8():
    return build(4, 3 * WINDOW + 100, precision="fp8")


def node_buckets_of(hierarchy, level, name):
    """One field of every level-`level` node bucket, the closed ones then the open"""
    parts = [
        rows[name]
        for rows in (hierarchy.closed_node_buckets[level - 1], hierarchy.open_node_buckets[level - 1])
        if len(rows)
    ]
    return torch.cat(parts)


def keys_under(cache, hierarchy, level):
    """The positions each level-`level` node holds [N, 4 ** level * 16], -1 padded"""
    children = node_buckets_of(hierarchy, level, "children")
    return cache.bucket_members[hierarchy.buckets_under(level, children)].long().flatten(1)


def centroid_of(keys, members):
    held = members >= 0
    return F.normalize((keys.float()[members.clamp_min(0)] * held[..., None]).sum(1), dim=-1)


def test_four_pages_grow_four_levels_and_cover_every_closed_bucket(deep):
    cache, _ = deep
    hierarchy = cache.hierarchies[1]
    assert hierarchy.levels == 4
    assert hierarchy.covered_buckets == 4 * BUCKETS_PER_PAGE
    assert [
        len(hierarchy.open_node_buckets[level]) + len(hierarchy.closed_node_buckets[level])
        for level in range(4)
    ] == [
        1024,
        256,
        64,
        16,
    ]


def test_every_closed_bucket_is_under_one_node_a_level_and_a_block_not_yet_full_under_none():
    # invariant 10 - the fifth page's 1,024 buckets fill 16 of the 64 blocks a growth waits for
    cache, _ = build(5)
    hierarchy = cache.hierarchies[1]
    assert (cache.visible(1), hierarchy.covered_buckets) == (5 * BUCKETS_PER_PAGE, 4 * BUCKETS_PER_PAGE)
    covered = cache.logical_rows(1, hierarchy.covered_buckets)[: hierarchy.covered_buckets]
    for level in range(1, hierarchy.levels + 1):
        children = node_buckets_of(hierarchy, level, "children")
        under = hierarchy.buckets_under(level, children).flatten()
        assert torch.equal(under.sort().values, covered.sort().values)


def test_a_node_holds_exactly_its_node_bucket_size_of_children_and_none_is_empty(deep):
    # invariant 11
    cache, _ = deep
    hierarchy = cache.hierarchies[1]
    for level in range(1, hierarchy.levels + 1):
        children = node_buckets_of(hierarchy, level, "children")
        buckets = hierarchy.buckets_under(level, children)
        assert buckets.shape == (len(children), 4**level)
        assert all(len(set(row)) == 4**level for row in buckets.tolist())
        assert bool((cache.bucket_sizes[buckets] > 0).all())
    level_one = hierarchy.closed_node_buckets[0]["children"]
    assert level_one.shape == (len(level_one), 4)


@pytest.mark.parametrize("fixture", ["deep", "deep_fp8"])
def test_a_nodes_centroid_is_the_normalized_sum_of_every_index_key_beneath_it(fixture, request):
    # invariant 12
    cache, keys = request.getfixturevalue(fixture)
    hierarchy = cache.hierarchies[1]
    for level in range(1, hierarchy.levels + 1):
        stored = node_buckets_of(hierarchy, level, "node_centroids")
        exact = centroid_of(keys, keys_under(cache, hierarchy, level))
        if fixture == "deep_fp8":
            stored = stored.float() / 16
        torch.testing.assert_close(stored.float(), exact, rtol=0.07, atol=2e-2)


def test_every_key_is_under_one_node_of_each_level(deep):
    cache, _ = deep
    hierarchy = cache.hierarchies[1]
    covered = 4 * PAGE_SIZE
    for level in range(1, hierarchy.levels + 1):
        held = keys_under(cache, hierarchy, level).flatten()
        assert torch.equal(held[held >= 0].sort().values, torch.arange(covered))


def test_a_level_below_the_top_holds_under_one_block_of_open_node_buckets(deep):
    cache, _ = deep
    hierarchy = cache.hierarchies[1]
    for level in range(1, hierarchy.levels):
        assert len(hierarchy.open_node_buckets[level - 1]) < 64


def test_an_fp8_node_dequantizes_within_the_tolerance_of_its_stored_scale(deep, deep_fp8):
    # invariant 16 - the same node buckets in bf16 and in fp8, the fp8 ones a sixteenth of the way to e4m3
    (bf16_cache, _), (fp8_cache, _) = deep, deep_fp8
    bf16, fp8 = bf16_cache.hierarchies[1], fp8_cache.hierarchies[1]
    for level in range(1, fp8.levels + 1):
        stored = node_buckets_of(fp8, level, "node_centroids")
        assert stored.dtype == torch.float8_e4m3fn
        assert node_buckets_of(bf16, level, "node_centroids").dtype == torch.bfloat16
        dequantized = stored.float() / 16
        exact = node_buckets_of(bf16, level, "node_centroids").float()
        # e4m3 keeps three mantissa bits - a relative 2 ** -4 - and rounds the smallest components to its floor
        assert bool(((dequantized - exact).abs() <= exact.abs() * 2**-3 + 2**-9 / 16 + 1e-2).all())


def test_growing_a_level_changes_no_bucket_below_it():
    # invariant 15 - the buckets after the fourth page are the third page's, with the fourth's beside them
    before, _ = build(3, 100)
    hierarchy = before.hierarchies[1]
    assert hierarchy.levels == 0
    after, keys = build(4, 100)
    assert after.hierarchies[1].levels > 0

    def bucket_sets(cache):
        members = cache.bucket_members[cache.logical_rows(1, cache.visible(1))[: cache.visible(1)]]
        return {tuple(row[row >= 0].tolist()) for row in members}

    assert bucket_sets(before) < bucket_sets(after)
    # And no row moved: the third page's rows hold what they held, each its own bucket's centroid
    first = before.logical_rows(1, before.visible(1))[: before.visible(1)]
    assert torch.equal(before.bucket_members[first], after.bucket_members[first])
    rows = after.logical_rows(1, after.visible(1))[: after.visible(1)]
    exact = centroid_of(keys, after.bucket_members[rows].long()).to(torch.bfloat16)
    assert torch.equal(after.bucket_centroids[rows], exact)
    assert torch.equal(after.bucket_sizes[rows], (after.bucket_members[rows] >= 0).sum(1).int())


def test_growth_waits_for_a_growth_buckets_worth_of_blocks():
    # The first level pairs GROWTH_BUCKETS buckets at once at least, so a block of 1,024 grows every fourth page
    assert GROWTH_BUCKETS == 4096
    one = dict(top_level=1, node_cluster_block_size=1024)
    covered = []
    for pages in (1, 2, 3, 4):
        cache, _ = build(pages, **one)
        covered.append((cache.hierarchies[1].levels, cache.hierarchies[1].covered_buckets))
    assert covered == [(0, 0), (0, 0), (0, 0), (1, 4096)]


def test_the_top_level_caps_the_levels_grown():
    cache, _ = build(4, top_level=2)
    hierarchy = cache.hierarchies[1]
    assert hierarchy.levels == 2
    # The top level pairs no further, so its node buckets keep no key sums
    assert "key_sums" not in hierarchy.open_node_buckets[1].parts
    assert "key_sums" in hierarchy.open_node_buckets[0].parts


def test_each_level_pairs_blocks_of_its_own_node_cluster_block_size():
    cache, _ = build(4, top_level=2, node_cluster_block_size=[64, 128])
    hierarchy = cache.hierarchies[1]
    # 1,024 level 1 node buckets make 8 blocks of 128, and 32 level 2 node buckets from each
    assert len(hierarchy.open_node_buckets[0]) + len(hierarchy.closed_node_buckets[0]) == 1024
    assert len(hierarchy.open_node_buckets[1]) + len(hierarchy.closed_node_buckets[1]) == 256


@pytest.mark.parametrize("start_nodes", [1024, 600, 256, 64, 16, 15, None])
def test_the_beam_starts_at_the_lowest_level_holding_at_most_the_start_nodes(deep, start_nodes):
    # invariant 14
    cache, _ = deep
    hierarchy = cache.hierarchies[1]
    counts = [
        len(hierarchy.open_node_buckets[level]) + len(hierarchy.closed_node_buckets[level])
        for level in range(4)
    ]
    configured = Hierarchy(IndexerConfig(**{**INDEXER, "start_nodes": start_nodes}), "bf16")
    configured.open_node_buckets, configured.closed_node_buckets = (
        hierarchy.open_node_buckets,
        hierarchy.closed_node_buckets,
    )
    expected = next(
        (level for level, count in enumerate(counts, 1) if start_nodes and count <= start_nodes), 4
    )
    assert configured.start_level() == expected


def test_a_hierarchy_of_one_level_starts_at_it():
    cache, _ = build(4, top_level=1)
    assert cache.hierarchies[1].start_level() == 1


def test_a_released_row_starts_a_fresh_hierarchy(deep):
    cache, _ = build(4)
    assert cache.hierarchies[1].levels == 4
    cache.release(1)
    assert cache.hierarchies[1].levels == 0
    assert cache.hierarchies[1].covered_buckets == 0
    assert cache.hierarchies[0] == cache.hierarchies[1]


def test_a_hierarchy_holds_one_node_bucket_size_at_every_level():
    config = IndexerConfig(
        **{**INDEXER, "top_level": 2, "node_bucket_size": [4, 2], "node_cluster_block_size": 64}
    )
    with pytest.raises(ValueError, match="one node bucket size at every level"):
        Hierarchy(config, "bf16")


@pytest.mark.parametrize("block", [48, 32, 8192])
def test_a_node_cluster_block_the_pairing_cannot_take_is_refused(block):
    config = IndexerConfig(**{**INDEXER, "top_level": 1, "node_cluster_block_size": block})
    with pytest.raises(ValueError, match="power of two from 16 node bucket sizes to 4,096"):
        Hierarchy(config, "bf16")


def test_rows_append_at_the_end_and_are_taken_from_the_front():
    rows = Rows()
    for first in range(0, 300, 100):
        rows.extend(
            values=torch.arange(first, first + 100).float()[:, None].expand(-1, 3), ids=torch.arange(100)
        )
    assert len(rows) == 300
    taken = rows.take(120)
    assert taken["values"][:, 0].tolist() == list(range(120))
    assert len(rows) == 180
    assert rows["values"][0, 0].item() == 120
    rows.extend(values=torch.zeros(5, 3), ids=torch.zeros(5, dtype=torch.long))
    assert len(rows) == 185
    assert rows["values"][-1].tolist() == [0.0, 0.0, 0.0]
    other = Rows()
    other.extend(values=rows["values"].clone(), ids=rows["ids"].clone())
    assert rows == other


def test_two_builds_of_the_same_keys_grow_equal_hierarchies():
    first, _ = build(4)
    second, _ = build(4)
    assert first.hierarchies[1] == second.hierarchies[1]
    third, _ = build(4, chunk=1000)
    assert first.hierarchies[1] == third.hierarchies[1]


def shaped(rank=2):
    """A cache of four pages of 32-wide keys whose node buckets keep their shapes, and its keys"""
    generator = torch.Generator().manual_seed(3)
    count = 4 * PAGE_SIZE + 3 * WINDOW
    anchors = torch.randn(count // 12 + 1, 32, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 32, generator=generator)
    keys = F.normalize(keys, dim=-1).to(torch.bfloat16)
    indexer = {**INDEXER, "index_dim": 32, "node_bucket_shape": {"rank": rank, "deviations": 2}}
    settings = DesignSettings.from_blocks(indexer, {})
    cache = BucketCache(settings, len(keys), 3, len(keys), "cpu")
    cache.ensure(1, "request", 1)
    for end in range(3000, len(keys) + 3000, 3000):
        cache.advance(1, min(end, len(keys)), lambda positions: keys[positions.long()], {1})
    return cache, keys


@pytest.mark.parametrize("rank", [0, 2])
def test_a_node_buckets_shape_is_the_shape_of_every_index_key_beneath_it(rank):
    cache, keys = shaped(rank)
    hierarchy = cache.hierarchies[1]
    assert hierarchy.levels == 4
    for level in range(1, hierarchy.levels + 1):
        members = keys_under(cache, hierarchy, level)
        want = build_shapes(keys, centroid_of(keys, members), members, rank)
        assert node_buckets_of(hierarchy, level, "basis").shape == (len(members), rank, 4)
        torch.testing.assert_close(
            node_buckets_of(hierarchy, level, "variances"), want.variances, rtol=2e-2, atol=1e-5
        )
        torch.testing.assert_close(
            node_buckets_of(hierarchy, level, "floor"), want.floor, rtol=2e-2, atol=1e-5
        )


def test_a_node_bucket_shape_leaves_the_partition_and_the_centroids_as_they_were():
    cache, keys = shaped()
    settings = DesignSettings.from_blocks({**INDEXER, "index_dim": 32}, {})
    plain = BucketCache(settings, len(keys), 3, len(keys), "cpu")
    plain.ensure(1, "request", 1)
    for end in range(3000, len(keys) + 3000, 3000):
        plain.advance(1, min(end, len(keys)), lambda positions: keys[positions.long()], {1})
    for level in range(1, 5):
        for name in ("node_centroids", "children"):
            assert torch.equal(
                node_buckets_of(cache.hierarchies[1], level, name),
                node_buckets_of(plain.hierarchies[1], level, name),
            )


# Designs whose levels grow, close and reopen within four pages once the first level pairs at every page end
CUT_DESIGNS = [
    pytest.param({}, "bf16", id="tree"),
    pytest.param({}, "fp8", id="tree-fp8"),
    pytest.param({"top_level": 1, "node_cluster_block_size": 1024}, "bf16", id="fixed"),
    pytest.param({"top_level": 2, "node_cluster_block_size": [64, 128]}, "bf16", id="two-levels"),
]


@pytest.fixture
def every_page(monkeypatch):
    # The first level pairs at every page end, so four pages grow, close and stack levels page by page
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)


@pytest.mark.parametrize(("indexer", "precision"), CUT_DESIGNS)
def test_the_counts_replay_the_growth_of_every_page_end(every_page, indexer, precision):
    for pages in (1, 2, 3, 4):
        cache, _ = build(pages, precision=precision, **indexer)
        hierarchy = cache.hierarchies[1]
        covered, made, closed = hierarchy.counts_at(pages * BUCKETS_PER_PAGE)
        levels = zip(hierarchy.open_node_buckets, hierarchy.closed_node_buckets, strict=True)
        assert covered == hierarchy.covered_buckets
        assert made[: hierarchy.levels] == [len(opened) + len(shut) for opened, shut in levels]
        assert closed[: hierarchy.levels] == [len(shut) for shut in hierarchy.closed_node_buckets]
        assert not any(made[hierarchy.levels :])


@pytest.mark.parametrize(("indexer", "precision"), CUT_DESIGNS)
def test_a_hierarchy_cut_back_to_a_page_end_is_the_one_grown_to_it(every_page, indexer, precision):
    donor, keys = build(4, 3 * WINDOW + 100, precision=precision, **indexer)
    hierarchy = donor.hierarchies[1]
    lookup = lambda positions: keys[positions.long()]  # noqa: E731
    reopened = False
    for pages in (1, 2, 3, 4):
        grown = build(pages, precision=precision, **indexer)[0].hierarchies[1]
        closed_buckets = pages * BUCKETS_PER_PAGE
        assert hierarchy.truncated(closed_buckets, donor, lookup) == grown
        # What a retained prefix keeps, opened again when a request takes it
        assert hierarchy.trimmed(closed_buckets).truncated(closed_buckets, donor, lookup) == grown
        _, _, closed = hierarchy.counts_at(closed_buckets)
        reopened |= any(
            len(shut) > count for shut, count in zip(hierarchy.closed_node_buckets, closed, strict=False)
        )
    if not indexer:
        # Tree's third level closes at the fourth page end, so cutting before it sums key sums again
        assert reopened


@pytest.mark.parametrize(
    "indexer",
    [
        pytest.param({}, id="tree"),
        pytest.param({"top_level": 1, "node_cluster_block_size": 1024}, id="fixed"),
        pytest.param({"top_level": 2, "node_cluster_block_size": [64, 128]}, id="two-levels"),
    ],
)
@pytest.mark.parametrize("pages", [4, 5])
def test_the_levels_are_the_torch_references_levels(every_page, indexer, pages):
    cache, keys = build(pages, 3 * WINDOW + 100, **indexer)
    hierarchy = cache.hierarchies[1]
    config = cache.config
    visible = cache.visible(1)
    rows = cache.logical_rows(1, visible)[:visible]
    buckets = Buckets(
        cache.bucket_centroids[rows].float(),
        cache.bucket_members[rows],
        cache.bucket_sizes[rows],
        config.cluster_block_size // config.bucket_size,
        config.cluster_block_size,
    )
    block_sizes = [config.at("node_cluster_block_size", level) for level in range(1, config.top_level + 1)]
    reference = build_hierarchy(keys, buckets, config.top_level, hierarchy.node_bucket_size, block_sizes)
    growth = growth_at(reference, pages * BUCKETS_PER_PAGE, BUCKETS_PER_PAGE, hierarchy_module.GROWTH_BUCKETS)
    assert growth.covered_buckets == hierarchy.covered_buckets
    levels = zip(hierarchy.open_node_buckets, hierarchy.closed_node_buckets, strict=True)
    assert growth.made[: hierarchy.levels] == [len(opened) + len(shut) for opened, shut in levels]
    for level in range(1, hierarchy.levels + 1):
        children = reference.children[level - 1][: growth.made[level - 1]]
        # Level 1 names cache rows, the reference logical buckets
        want = rows[children] if level == 1 else children
        assert torch.equal(node_buckets_of(hierarchy, level, "children").long(), want)
        torch.testing.assert_close(
            node_buckets_of(hierarchy, level, "node_centroids").float(),
            reference.node_centroids[level - 1][: len(children)],
            rtol=0,
            atol=8e-3,
        )
