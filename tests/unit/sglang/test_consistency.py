"""The check that every rank of a decode context-parallel group holds the same index - on a CPU, ranks as threads.

Each thread is a rank with a bucket cache of its own, and a sum across the group is every thread's term added once
they all arrive, as a collective would. The real collectives are `tests/integ/sglang/test_context_parallel_ranks.py`.
"""

import threading

import pytest
import torch

from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index.consistency import check_same_index, checksum, request_checksum
from lsa.sglang.index.graph_tree import NODE_BUCKET_PAGE
from lsa.sglang.settings.design import PAGE_SIZE
from tests.unit.sglang.test_graph_tree import make_cache, serve, spread_keys
from tests.unit.sglang.test_hierarchy import WINDOW, build

ROW = 1


class Group:
    """A decode group of threads - each rank's sum is every rank's term added"""

    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=60)
        self.terms = [None] * size

    def reduce(self, rank):
        def total(values):
            self.terms[rank] = values.clone()
            self.barrier.wait()
            summed = sum(self.terms[1:], self.terms[0]).clone()
            self.barrier.wait()
            return summed

        return total


def check_ranks(caches):
    """Run the check on every rank at once, and return what each raised"""
    group, errors = Group(len(caches)), [None] * len(caches)

    def run(rank):
        try:
            check_same_index(caches[rank], [ROW], group.reduce(rank), len(caches))
        except RuntimeError as error:
            errors[rank] = error

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(len(caches))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


@pytest.fixture
def ranks(monkeypatch):
    """Three ranks' caches, each holding the same request's first page and a grown tree over it"""
    # A page's 1,024 buckets grow the tree
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)
    caches = [build(1, 3 * WINDOW + 5)[0] for _ in range(3)]
    assert caches[0].hierarchies[ROW].levels == 3
    return caches


def test_ranks_holding_the_same_index_pass_the_check(ranks):
    assert check_ranks(ranks) == [None, None, None]
    assert len({int(request_checksum(cache, ROW)) for cache in ranks}) == 1


def test_a_rank_whose_tree_differs_fails_the_check_on_every_rank(ranks):
    # Two nodes of level 1 trade children - same counts and shapes, another tree
    children = ranks[1].hierarchies[ROW].closed_node_buckets[0]["children"]
    children[[0, 1]] = children[[1, 0]]
    errors = check_ranks(ranks)
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert "different indexes" in str(errors[0])


def test_a_rank_whose_buckets_differ_fails_the_check_on_every_rank(ranks):
    cache = ranks[2]
    cache.bucket_centroids[cache.logical_rows(ROW, 1)[0]] *= -1
    assert all(isinstance(error, RuntimeError) for error in check_ranks(ranks))


def test_a_checksum_follows_the_bytes_and_their_order():
    values = torch.arange(10, dtype=torch.int32)
    assert int(checksum(values)) == int(checksum(values.clone()))
    assert int(checksum(values.flip(0))) != int(checksum(values))
    assert int(checksum(values.bfloat16())) != int(checksum(values))


@pytest.fixture
def mirrored(monkeypatch):
    """Three ranks' caches that hold the graph tree, over the same request"""
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)
    keys = spread_keys(3 * PAGE_SIZE)
    caches = [make_cache() for _ in range(3)]
    for cache in caches:
        serve(cache, ROW, "request", keys)
    assert caches[0].graph_tree is not None
    assert caches[0].graph_tree.made[:, ROW].gt(0).any()
    return caches


def drift_counts(tree):
    tree.made[0, ROW] += 1


def drift_start(tree):
    tree.start[ROW] += 1


def drift_covered(tree):
    tree.covered[ROW] += 1


def drift_pages(tree):
    tree.pages[0][ROW, 0] += 1


def drift_centroids(tree):
    page = int(tree.pages[0][ROW, 0])
    tree.node_centroids[0][page * NODE_BUCKET_PAGE] *= -1


def drift_children(tree):
    page = int(tree.pages[0][ROW, 0])
    tree.children[0][page * NODE_BUCKET_PAGE, 0] += 1


@pytest.mark.parametrize(
    "drift", [drift_counts, drift_start, drift_covered, drift_pages, drift_centroids, drift_children]
)
def test_a_rank_whose_graph_tree_drifted_fails_the_check_on_every_rank(mirrored, drift):
    assert check_ranks(mirrored) == [None, None, None]
    drift(mirrored[1].graph_tree)
    assert all(isinstance(error, RuntimeError) for error in check_ranks(mirrored))


def test_a_mirror_of_one_level_checks_like_any_other(monkeypatch):
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)
    keys = spread_keys(2 * PAGE_SIZE)
    caches = [make_cache(top_level=1) for _ in range(2)]
    for cache in caches:
        serve(cache, ROW, "request", keys)
    assert caches[0].graph_tree.level_count == 1
    assert check_ranks(caches) == [None, None]
