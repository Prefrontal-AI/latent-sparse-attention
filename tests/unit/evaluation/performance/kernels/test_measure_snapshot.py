"""The kernels snapshot puts back the graph tree a trial grew."""

from types import SimpleNamespace

import pytest
import torch

from lsa.evaluation.performance.kernels.measure.snapshot import IndexState
from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.settings.design import PAGE_SIZE
from tests.unit.sglang.test_graph_tree import assert_mirrors, make_cache, serve, spread_keys


@pytest.fixture(autouse=True)
def every_page(monkeypatch):
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)


def index_over(cache):
    selector = SimpleNamespace(cache=cache, step=0)
    return SimpleNamespace(selector=selector, index_keys=torch.empty(0))


def test_a_trial_that_grows_the_graph_tree_is_put_back_exactly():
    cache = make_cache()
    serve(cache, 1, "kept", spread_keys(3 * PAGE_SIZE, seed=1))
    state = IndexState(index_over(cache), torch.arange(4))
    tree = cache.graph_tree
    saved = [tensor.clone() for tensor in (tree.made, tree.start, tree.covered, *tree.pages)]
    held = [[list(level) for level in row] for row in tree.held_pages]

    # A trial that frees the kept row's pages and fills them again for another request
    cache.release(1)
    serve(cache, 2, "other", spread_keys(4 * PAGE_SIZE, seed=2))
    with pytest.raises(AssertionError):
        state.check()

    state.restore()
    state.check()
    after = (tree.made, tree.start, tree.covered, *tree.pages)
    assert all(torch.equal(tensor, was) for tensor, was in zip(after, saved, strict=True))
    # The kept row's node buckets read back through its pages, which the other request had overwritten
    assert_mirrors(cache, 1)
    assert tree.held_pages == held


def test_a_mirror_out_of_step_with_its_hierarchy_fails_the_check():
    cache = make_cache()
    serve(cache, 1, "kept", spread_keys(3 * PAGE_SIZE, seed=1))
    state = IndexState(index_over(cache), torch.arange(4))
    cache.graph_tree.copied[1][0] -= 1
    state.tree.copied = [list(row) for row in cache.graph_tree.copied]
    with pytest.raises(AssertionError, match="holds"):
        state.check()
