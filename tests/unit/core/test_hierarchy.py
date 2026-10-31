"""The torch reference of the hierarchy - its levels over the buckets, and the beam down them.

Small sizes, so a few thousand keys grow four levels: cluster blocks of 64 keys in buckets of 8, node cluster blocks
of 16 and pages of 512 keys. The served hierarchy is held to these levels in `tests/unit/sglang/test_hierarchy.py`
and its beam to this read in `tests/unit/sglang/test_selector.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import build_buckets, build_hierarchy, read, read_hierarchy, read_hierarchy_counted
from lsa.core.hierarchy import Scorer, growth_at

WINDOW = 64
SINKS = 4
TOP_K = 64
TOP_P = 16
PAGE = 512
NODE_BLOCK = 16
KEYS = 8192
EVERY_NODE = 4096


def spread_keys(count, seed=0):
    """Index keys in runs of 12 around one anchor each, so groups of buckets have structure to pair"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, 16, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 16, generator=generator)
    return F.normalize(keys, dim=-1)


def queries_near(keys, positions, seed=1):
    """A query near an earlier key, one a position"""
    generator = torch.Generator().manual_seed(seed)
    earlier = (torch.rand(len(positions), generator=generator) * positions).long()
    return F.normalize(keys[earlier] + 0.1 * torch.randn(len(positions), 16, generator=generator), dim=-1)


@pytest.fixture(scope="module")
def store():
    keys = spread_keys(KEYS)
    buckets = build_buckets(keys, 4, 8, 16, WINDOW)
    hierarchy = build_hierarchy(keys, buckets, 16, 4, NODE_BLOCK)
    return keys, buckets, hierarchy


def flat_read(store, queries, positions):
    _, buckets, _ = store
    return read(queries, positions, buckets, TOP_P, TOP_K, SINKS, WINDOW)[0]


def beam_read(store, queries, positions, beam_width, start_nodes=None, growth_buckets=0):
    _, buckets, hierarchy = store
    return read_hierarchy(
        queries,
        positions,
        buckets,
        hierarchy,
        beam_width,
        start_nodes,
        TOP_P,
        TOP_K,
        SINKS,
        WINDOW,
        PAGE,
        growth_buckets,
    )[0]


def sets(lists):
    """Each row's ids as a sorted list, -1 padding dropped"""
    return [sorted(row[row >= 0].tolist()) for row in lists]


def test_every_full_node_cluster_block_of_every_level_is_paired(store):
    _, buckets, hierarchy = store
    counts = [len(centroids) for centroids in hierarchy.node_centroids]
    assert len(buckets.bucket_sizes) == 1024
    assert counts == [256, 64, 16, 4]


def test_a_node_holds_exactly_its_node_bucket_size_of_distinct_children(store):
    _, _, hierarchy = store
    for children in hierarchy.children:
        assert children.shape[1] == 4
        assert (children.sort(1).values.diff(dim=1) > 0).all()


def test_every_bucket_of_a_full_block_is_under_one_node_a_level(store):
    _, buckets, hierarchy = store
    below = len(buckets.bucket_sizes)
    for children in hierarchy.children:
        covered = children.flatten().sort().values
        assert torch.equal(covered, torch.arange(len(covered)))
        assert len(covered) <= below
        below = len(children)


def test_a_node_holds_children_from_its_own_node_cluster_block(store):
    _, _, hierarchy = store
    for children in hierarchy.children:
        blocks = children // NODE_BLOCK
        assert (blocks == blocks[:, :1]).all()


def test_a_nodes_centroid_is_the_normalized_sum_of_every_index_key_beneath_it(store):
    keys, buckets, hierarchy = store
    rows = torch.arange(len(buckets.bucket_sizes))[:, None]
    for level, children in enumerate(hierarchy.children, start=1):
        rows = rows[children].flatten(1)
        members = buckets.bucket_members[rows].flatten(1).long()
        sums = (keys[members.clamp_min(0)] * (members >= 0)[..., None]).sum(1)
        torch.testing.assert_close(hierarchy.node_centroids[level - 1], F.normalize(sums, dim=-1))


def test_the_top_level_caps_the_levels_built(store):
    keys, buckets, _ = store
    assert len(build_hierarchy(keys, buckets, 2, 4, NODE_BLOCK).node_centroids) == 2


def test_each_level_pairs_blocks_of_its_own_node_cluster_block_size(store):
    keys, buckets, _ = store
    hierarchy = build_hierarchy(keys, buckets, 2, 4, [64, 32])
    assert [len(centroids) for centroids in hierarchy.node_centroids] == [256, 64]
    blocks = hierarchy.children[1] // 32
    assert (blocks == blocks[:, :1]).all()


def test_growth_waits_for_a_page_end_and_a_growth_buckets_worth_of_blocks(store):
    _, _, hierarchy = store
    page_buckets = PAGE // 8
    assert growth_at(hierarchy, page_buckets - 1, page_buckets, 0).made[0] == 0
    assert growth_at(hierarchy, page_buckets, page_buckets, 0).covered_buckets == page_buckets
    assert growth_at(hierarchy, page_buckets, page_buckets, 2 * page_buckets).made[0] == 0
    assert (
        growth_at(hierarchy, 2 * page_buckets, page_buckets, 2 * page_buckets).covered_buckets
        == 2 * page_buckets
    )


def test_a_level_above_takes_only_full_blocks_of_the_node_buckets_below(store):
    _, _, hierarchy = store
    growth = growth_at(hierarchy, 1024, PAGE // 8, 0)
    assert growth.made == [256, 64, 16, 4] + [0] * 12
    assert growth.closed == [256, 64, 16] + [0] * 13


@pytest.mark.parametrize(
    ("top_level", "start_nodes"),
    [(1, None), (16, None), (16, 300), (16, 64)],
    ids=["fixed", "top", "level 1", "level 2"],
)
def test_a_beam_wide_enough_for_every_node_reads_what_the_flat_read_reads(store, top_level, start_nodes):
    # invariant 13, from the top and from a start level scored whole
    keys, buckets, _ = store
    hierarchy = build_hierarchy(keys, buckets, top_level, 4, NODE_BLOCK)
    positions = torch.arange(KEYS - 200, KEYS, 3)
    queries = queries_near(keys, positions)
    beam = read_hierarchy(
        queries, positions, buckets, hierarchy, EVERY_NODE, start_nodes, TOP_P, TOP_K, SINKS, WINDOW, PAGE, 0
    )[0]
    assert sets(beam) == sets(flat_read(store, queries, positions))


def test_a_narrow_beam_fills_its_budget_but_reads_what_the_flat_read_would_not(store):
    keys, _, _ = store
    positions = torch.arange(KEYS - 200, KEYS, 3)
    queries = queries_near(keys, positions, seed=2)
    got, want = sets(beam_read(store, queries, positions, 4)), sets(flat_read(store, queries, positions))
    assert got != want
    assert all(len(ids) == TOP_K + WINDOW + SINKS for ids in got)


def test_a_query_before_any_growth_reads_flat(store):
    keys, _, _ = store
    positions = torch.arange(PAGE // 2, PAGE, 5)
    queries = queries_near(keys, positions, seed=3)
    assert sets(beam_read(store, queries, positions, 4)) == sets(flat_read(store, queries, positions))


def test_a_query_reads_the_levels_as_its_page_saw_them(store):
    # The first query of a page sees the growth at its start, the last one of the page before does not
    keys, _, _ = store
    positions = torch.tensor([4 * PAGE - 1, 4 * PAGE])
    queries = queries_near(keys, positions, seed=4)
    narrow = beam_read(store, queries, positions, 4, growth_buckets=4 * PAGE // 8)
    flat = flat_read(store, queries, positions)
    assert sets(narrow[:1]) == sets(flat[:1])
    assert sets(narrow[1:]) != sets(flat[1:])


def test_every_selected_id_is_below_the_query_and_none_is_read_twice(store):
    keys, _, _ = store
    positions = torch.arange(KEYS - 200, KEYS, 3)
    queries = queries_near(keys, positions, seed=5)
    for position, ids in zip(positions.tolist(), sets(beam_read(store, queries, positions, 4)), strict=True):
        assert max(ids) <= position
        assert len(set(ids)) == len(ids)


def test_a_node_bucket_size_that_is_not_a_power_of_two_is_refused(store):
    keys, buckets, _ = store
    with pytest.raises(ValueError, match="power of two"):
        build_hierarchy(keys, buckets, 1, 3, 12)


def read_with(store, queries, positions, beam_width, **options):
    keys, buckets, hierarchy = store
    return read_hierarchy(
        queries,
        positions,
        buckets,
        hierarchy,
        beam_width,
        None,
        TOP_P,
        TOP_K,
        SINKS,
        WINDOW,
        PAGE,
        0,
        **options,
    )[0]


def test_pages_hold_runs_of_consecutive_keys(store):
    keys, _, _ = store
    pages = build_buckets(keys, 4, 8, 16, WINDOW, cluster_algorithm="position")
    assert torch.equal(pages.bucket_members[:, :8].long(), torch.arange(KEYS).view(-1, 8))
    assert bool((pages.bucket_members[:, 8:] == -1).all())


def test_positional_grouping_takes_adjacent_node_buckets(store):
    keys, buckets, _ = store
    hierarchy = build_hierarchy(keys, buckets, 16, 4, NODE_BLOCK, node_cluster_algorithm="position")
    for children in hierarchy.children:
        assert torch.equal(children, torch.arange(children.numel()).view(-1, 4))


def test_a_nodes_mean_is_the_mean_of_every_index_key_beneath_it(store):
    keys, buckets, hierarchy = store
    rows = torch.arange(len(buckets.bucket_sizes))[:, None]
    for level, children in enumerate(hierarchy.children, start=1):
        rows = rows[children].flatten(1)
        members = buckets.bucket_members[rows].flatten(1).long()
        held = (members >= 0)[..., None]
        want = (keys[members.clamp_min(0)] * held).sum(1) / held.sum(1)
        torch.testing.assert_close(hierarchy.node_means[level - 1], want)


@pytest.mark.parametrize("node_cluster_algorithm", ["ward", "position"])
@pytest.mark.parametrize("node_bucket_score", ["centroid", "lse"])
def test_any_grouping_or_node_score_under_a_beam_wide_enough_reads_what_the_flat_read_reads(
    store, node_cluster_algorithm, node_bucket_score
):
    keys, buckets, _ = store
    hierarchy = build_hierarchy(
        keys, buckets, 16, 4, NODE_BLOCK, node_cluster_algorithm=node_cluster_algorithm
    )
    positions = torch.arange(KEYS - 200, KEYS, 7)
    queries = queries_near(keys, positions, seed=6)
    beam = read_hierarchy(
        queries,
        positions,
        buckets,
        hierarchy,
        EVERY_NODE,
        None,
        TOP_P,
        TOP_K,
        SINKS,
        WINDOW,
        PAGE,
        0,
        node_bucket_score=node_bucket_score,
        tau=0.05,
    )[0]
    assert sets(beam) == sets(flat_read(store, queries, positions))


def test_the_lse_node_score_at_a_low_temperature_ranks_by_the_best_childs_score(store):
    keys, _, hierarchy = store
    query = queries_near(keys, torch.tensor([KEYS - 1]), seed=7)[0]
    rows = torch.arange(len(hierarchy.children[1]))

    scorer = Scorer(store[1], hierarchy, "lse", "centroid", 1e-4, None)
    scores = scorer.node_buckets(query, 2, rows)
    best_child = (hierarchy.node_means[0][hierarchy.children[1]] @ query).max(1).values
    torch.testing.assert_close(scores * 1e-4, best_child, rtol=0, atol=1e-3)


def test_the_lse_bucket_score_with_every_bucket_picked_ranks_buckets_by_their_keys(store):
    keys, buckets, _ = store
    positions = torch.tensor([KEYS - 1])
    queries = queries_near(keys, positions, seed=8)
    got = read_with(store, queries, positions, EVERY_NODE, bucket_score="lse", tau=0.05, index_keys=keys)
    visible = (KEYS - 1) // WINDOW * 8
    members = buckets.bucket_members[:visible].long()
    logits = (keys[members.clamp_min(0)] @ queries[0] / 0.05).masked_fill(members < 0, -torch.inf)
    order = torch.argsort(torch.logsumexp(logits, 1), descending=True, stable=True)[:TOP_P]
    held = members[order].flatten()
    window_start = KEYS - WINDOW
    want = held[(held >= SINKS) & (held < window_start)][:TOP_K]
    assert sorted(got[0][got[0] >= 0].tolist()) == sorted(
        want.tolist() + list(range(window_start, KEYS)) + list(range(SINKS))
    )


def test_a_narrow_beam_reads_otherwise_under_positional_grouping(store):
    keys, buckets, hierarchy = store
    positional = build_hierarchy(keys, buckets, 16, 4, NODE_BLOCK, node_cluster_algorithm="position")
    positions = torch.arange(KEYS - 200, KEYS, 3)
    queries = queries_near(keys, positions, seed=9)
    args = (queries, positions, buckets)
    rest = (4, None, TOP_P, TOP_K, SINKS, WINDOW, PAGE, 0)
    assert sets(read_hierarchy(*args, hierarchy, *rest)[0]) != sets(
        read_hierarchy(*args, positional, *rest)[0]
    )


def test_unknown_options_and_a_bucket_lse_without_keys_are_refused(store):
    keys, buckets, hierarchy = store
    positions = torch.tensor([KEYS - 1])
    queries = queries_near(keys, positions)
    with pytest.raises(ValueError, match="node_cluster_algorithm"):
        build_hierarchy(keys, buckets, 1, 4, NODE_BLOCK, node_cluster_algorithm="hash")
    with pytest.raises(ValueError, match="cluster_algorithm"):
        build_buckets(keys, 4, 8, 16, WINDOW, cluster_algorithm="hash")
    with pytest.raises(ValueError, match="node_bucket_score"):
        read_with(store, queries, positions, 4, node_bucket_score="max")
    with pytest.raises(ValueError, match="index_keys"):
        read_with(store, queries, positions, 4, bucket_score="lse")
    with pytest.raises(ValueError, match="tau"):
        read_with(store, queries, positions, 4, node_bucket_score="lse", tau=0.0)


def test_the_counted_read_is_the_read_and_counts_the_vectors_each_query_scored(store):
    keys, buckets, hierarchy = store
    positions = torch.tensor([PAGE // 2, KEYS - 1])
    queries = queries_near(keys, positions, seed=10)
    args = (queries, positions, buckets, hierarchy, 4, None, TOP_P, TOP_K, SINKS, WINDOW, PAGE, 0)
    selected, lengths, scored = read_hierarchy_counted(*args)
    assert torch.equal(selected, read_hierarchy(*args)[0])
    assert torch.equal(lengths, read_hierarchy(*args)[1])
    # Before any growth a query reads flat - every bucket of the blocks it can see, a centroid each
    assert scored[0] == PAGE // 2 // WINDOW * (WINDOW // 8)
    # A narrow beam scores far fewer than the 1,016 buckets flat would
    assert 0 < scored[1] < 200


def test_an_lse_score_counts_each_vector_it_reads(store):
    keys, buckets, hierarchy = store
    positions = torch.tensor([PAGE // 2])
    queries = queries_near(keys, positions, seed=11)
    args = (queries, positions, buckets, hierarchy, 4, None, TOP_P, TOP_K, SINKS, WINDOW, PAGE, 0)
    scored = read_hierarchy_counted(*args, bucket_score="lse", tau=0.05, index_keys=keys)[2]
    # Every key of the blocks it can see, once
    assert scored[0] == PAGE // 2 // WINDOW * WINDOW
