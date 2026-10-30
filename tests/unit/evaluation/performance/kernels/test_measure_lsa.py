"""The design's half of the harness that a CPU can exercise - the snapshot, the selection check, the cases.

The bucket cache clusters through the torch rounds on a CPU, so the snapshot is held against a real cache. The
layer itself, its timing and the engine's top-k are `tests/integ/evaluation/performance/kernels/test_harness_gpu.py`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.cutedsl import top_k as cutedsl_top_k
from lsa.core.kernels.cutedsl.decode import DecodeSelect
from lsa.core.kernels.scan_select import ScanSelect
from lsa.core.kernels.triton import scan_select as scan_select_module
from lsa.core.kernels.triton.keys import pack_keys
from lsa.evaluation.constants import Kernels
from lsa.evaluation.performance.kernels.measure import decode
from lsa.evaluation.performance.kernels.measure.instrument import wrap_lsa
from lsa.evaluation.performance.kernels.measure.score_check import lsa_score_oracle
from lsa.evaluation.performance.kernels.measure.selection import (
    LSASelectionCheck,
    selection_rows,
    weigh,
)
from lsa.evaluation.performance.kernels.measure.snapshot import (
    IndexState,
    Snapshot,
    same_field,
    same_hierarchy,
)
from lsa.evaluation.performance.kernels.measure.workload import score_rows
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.hierarchy import Hierarchy
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
FIXED = {"top_level": 1, "node_bucket_size": 4, "node_cluster_block_size": 64, "beam_width": 16}


def keys_of(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(count, 16, generator=generator), dim=-1).to(torch.bfloat16)


def lookup(keys):
    return lambda positions: keys[positions.long()]


class Index:
    """A layer's index with the bucket cache real and the engine absent"""

    def __init__(self, **indexer):
        self.settings = DesignSettings.from_blocks({**INDEXER, **indexer}, {})
        slots = 3 * PAGE_SIZE
        self.cache = BucketCache(self.settings, slots, 2, 4 * PAGE_SIZE, "cpu")
        self.index_keys = torch.zeros(slots, 16, dtype=torch.bfloat16)
        self.selector = SimpleNamespace(cache=self.cache, step=1)
        self.keys = keys_of(2 * PAGE_SIZE)
        self.cache.ensure(1, "a", 1)

    def advance(self, end):
        self.cache.advance(1, end, lookup(self.keys), {1})

    def snapshot(self):
        indexer = SimpleNamespace(is_lsa=True, ready=lambda: self)
        pool = SimpleNamespace(get_key_buffer=lambda layer: torch.zeros(3 * PAGE_SIZE, 1, 4))
        module = SimpleNamespace(layer_id=3, indexer=indexer)
        return Snapshot(module, pool, SimpleNamespace(out_cache_loc=torch.tensor([5, 6, 7])))


def test_the_snapshot_restores_the_member_counts_block_positions_and_run_a_threshold_fill_reads():
    index = Index()
    index.advance(PAGE_SIZE // 2)
    snapshot = index.snapshot()
    cache = index.cache
    saved = {
        "bucket_sizes": cache.bucket_sizes.clone(),
        "page_positions": cache.page_positions.clone(),
        "bucket_members": cache.bucket_members.clone(),
    }
    run = list(cache.run)
    # A pass takes a second page, clusters blocks into it and writes index keys
    index.advance(PAGE_SIZE + 200)
    index.index_keys[[5, 6, 7]] = 1
    cache.run[1] = 7
    assert not torch.equal(cache.bucket_sizes, saved["bucket_sizes"])

    snapshot.restore()

    for name, expected in saved.items():
        assert torch.equal(getattr(cache, name), expected), name
    assert cache.run == run
    assert cache.pages[1] == [0]
    assert cache.end[1] == PAGE_SIZE // 2
    assert torch.count_nonzero(index.index_keys[[5, 6, 7]]) == 0
    snapshot.check()


def test_the_snapshot_restores_in_place_what_a_cuda_graph_holds_by_address():
    index = Index()
    index.advance(PAGE_SIZE // 2)
    snapshot = index.snapshot()
    held = {name: getattr(index.cache, name).data_ptr() for name in IndexState.TENSORS}
    index.advance(PAGE_SIZE + 200)

    snapshot.restore()

    assert {name: getattr(index.cache, name).data_ptr() for name in IndexState.TENSORS} == held


def test_the_snapshot_check_names_what_was_not_restored():
    index = Index()
    index.advance(PAGE_SIZE // 2)
    snapshot = index.snapshot()
    snapshot.check()
    index.cache.closed_keys[1] += WINDOW
    with pytest.raises(AssertionError, match="closed_keys"):
        snapshot.check()
    snapshot.restore()
    index.cache.bucket_sizes[3] += 1
    with pytest.raises(AssertionError):
        snapshot.check()
    snapshot.restore()
    index.selector.step += 1
    with pytest.raises(AssertionError, match="step"):
        snapshot.check()


def test_the_snapshot_restores_a_request_the_pass_released():
    index = Index()
    index.advance(PAGE_SIZE // 2)
    snapshot = index.snapshot()
    free = list(index.cache.free)
    index.cache.release(1)
    index.cache.ensure(1, "b", 2)

    snapshot.restore()

    assert index.cache.rid[1] == "a"
    assert index.cache.free == free
    snapshot.check()


def test_a_pass_is_named_by_what_it_did_to_the_cache():
    index = Index(**FIXED)
    index.advance(PAGE_SIZE - 1)
    state = index.snapshot().state

    assert state.changed(1) == dict.fromkeys(Kernels.CASES[:3], False)

    # One key closes the last cluster block of the page
    index.advance(PAGE_SIZE)
    assert state.changed(1) == {
        "page_allocation": False,
        "cluster_block_close": True,
        "hierarchy_growth": False,
    }
    # The next key takes the second page
    index.advance(PAGE_SIZE + 1)
    assert state.changed(1)["page_allocation"]
    index.cache.hierarchies[1].covered_buckets = 4096
    assert state.changed(1)["hierarchy_growth"]


def test_a_flat_cache_has_no_hierarchy_to_grow():
    index = Index()
    index.advance(PAGE_SIZE // 2)
    state = index.snapshot().state
    index.advance(PAGE_SIZE)
    assert index.cache.hierarchies is None
    assert state.changed(1)["hierarchy_growth"] is False


def test_the_restore_also_puts_a_grown_hierarchy_back():
    index = Index(**FIXED)
    index.advance(PAGE_SIZE // 2)
    snapshot = index.snapshot()
    hierarchy = index.cache.hierarchies[1]
    hierarchy.stage(1, torch.ones(4, 16), torch.arange(4, dtype=torch.int32))
    hierarchy.covered_buckets = 32
    assert index.cache.hierarchies[1] != snapshot.state.host["hierarchies"][1]
    with pytest.raises(AssertionError, match="hierarchies"):
        snapshot.check()

    snapshot.restore()

    assert index.cache.hierarchies[1].levels == 0
    assert index.cache.hierarchies[1].covered_buckets == 0
    snapshot.check()


def test_fp8_node_buckets_are_compared_by_their_bytes():
    """fp8 has no equality kernel on every build, and the tree design stores its node centroids as fp8."""
    settings = DesignSettings.from_blocks({**INDEXER, **FIXED}, {"node_centroid_precision": "fp8"})
    first, second = Hierarchy(settings.indexer, "fp8"), Hierarchy(settings.indexer, "fp8")
    for hierarchy in (first, second):
        hierarchy.stage(1, torch.arange(1.0, 33.0).reshape(2, 16), torch.arange(2, dtype=torch.int32))
    assert first.open_node_buckets[0].parts["node_centroids"].dtype == torch.float8_e4m3fn
    assert same_hierarchy(first, second)

    second.open_node_buckets[0].parts["node_centroids"][0, 0] = 0
    assert not same_hierarchy(first, second)
    assert not same_hierarchy(first, Hierarchy(settings.indexer, "fp8"))
    assert same_field("end", [1, 2], [1, 2])
    assert not same_field("end", [1, 2], [1, 3])
    assert same_field("hierarchies", None, None)


def rows(*listings, width=8):
    out = torch.full((len(listings), width), -1, dtype=torch.int32)
    for row, ids in enumerate(listings):
        out[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return out


def test_identical_selections_in_another_order_are_exact():
    rerun = selection_rows(rows([5, 1, 3], [2, 4]), rows([1, 3, 5], [4, 2]), bucket_cap=16)
    assert rerun["differing"] == 0
    assert rerun["exact"]
    assert rerun["first_differing"] == []


def test_a_row_that_differs_by_one_bucket_is_counted_and_not_exact():
    """Every pick is exact, so a rerun that differs on a row has parted from the prepared selection."""
    expected = rows([1, 2, 3, 4, 5, 6], [1, 2, 3])
    observed = rows([1, 2, 3, 4, 5, 7], [1, 2, 3])
    rerun = selection_rows(observed, expected, bucket_cap=2)
    assert rerun["differing"] == 1
    assert rerun["widest_side"] == 1
    assert rerun["first_differing"] == [0]
    assert not rerun["exact"]


def test_the_facts_a_rerun_wrote_are_weighed_again_by_the_record():
    """A ladder's record is rebuilt from the facts, so the bar needs no measurement."""
    facts = {"rows": 8192, "differing": 0, "widest_side": 0, "first_differing": [], "bucket_cap": 16}
    assert weigh(facts)["exact"]
    assert not weigh({**facts, "differing": 1, "widest_side": 4, "first_differing": [0]})["exact"]


def test_a_row_that_lost_its_tail_to_padding_differs_by_the_keys_it_lost():
    rerun = selection_rows(rows([1, 2, 3]), rows([1, 2, 3, 4]), bucket_cap=16)
    assert rerun["differing"] == 1
    assert rerun["widest_side"] == 1


class FakeTopK:
    """The top-k, with the picks a check reads them from"""

    def __init__(self, out):
        self.out = out

    def best(self):
        return pack_keys(torch.zeros(self.out.shape), self.out.to(torch.int32))

    def listed(self, scores, lengths):
        return self.out, lengths


def design_check(top_p=2, decode_step=False, out=None):
    prefill, decoding = (
        FakeTopK(out if out is not None else torch.tensor([[0, 2]])),
        FakeTopK(torch.tensor([[0, 1]])),
    )
    selector = SimpleNamespace(
        config=SimpleNamespace(top_p=top_p),
        prefill_top_k=prefill,
        decode_top_k=decoding,
        scan_select=ScanSelect.__new__(ScanSelect),
        decode_select=DecodeSelect(top_p),
    )
    module = SimpleNamespace(indexer=SimpleNamespace(index=SimpleNamespace(selector=selector)))
    check = LSASelectionCheck(module, torch.arange(20), 20, decode=decode_step)
    # One read by the top-k, as a capture of a short chunk keeps it
    check.parts = [{"fused": False, "listing": (None, None)}]
    return check, prefill


def test_the_flat_read_is_held_to_the_exact_top_p_of_its_scores():
    check, top_k = design_check()
    check.reference = {"scores": torch.tensor([[10.0, 9.0, 9.0, 1.0, 99.0]]), "lengths": torch.tensor([4])}
    check.prepare()
    selected = torch.tensor([[0, 1, 2, 3]])
    assert check.check(selected)
    top_k.out[:] = torch.tensor([[1, 2]])
    with pytest.raises(AssertionError, match="Omitted"):
        check.check(selected)
    top_k.out[:] = torch.tensor([[0, 3]])
    with pytest.raises(AssertionError, match="below the true top-k cutoff"):
        check.check(selected)


def test_a_row_that_sees_fewer_buckets_than_the_top_p_must_pick_all_of_them():
    check, top_k = design_check(top_p=4, out=torch.tensor([[0, 1, -1, -1]]))
    check.reference = {"scores": torch.tensor([[10.0, 9.0, 99.0, 99.0]]), "lengths": torch.tensor([2])}
    check.prepare()
    assert check.check(torch.tensor([[0, 1]]))
    top_k.out[:] = torch.tensor([[0, -1, -1, -1]])
    with pytest.raises(AssertionError):
        check.check(torch.tensor([[0, 1]]))


def test_a_short_prefill_s_cutedsl_listing_is_captured_and_picked_from(monkeypatch):
    listing = (torch.tensor([[7, 8]]), torch.tensor([2]))
    monkeypatch.setattr(cutedsl_top_k, "list_rows", lambda scores, lengths, top_k, ordered=False: listing)
    check, _ = design_check()
    scores, lengths = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([3])
    captured = {}
    with check.capture(captured):
        assert cutedsl_top_k.list_rows(scores, lengths, 2) is listing
    assert torch.equal(captured["scores"], scores)
    assert torch.equal(captured["lengths"], lengths)
    assert check.parts[0]["listing"] is listing
    assert not check.parts[0]["fused"]


def test_an_ordered_cutedsl_listing_passes_through_the_capture(monkeypatch):
    calls = []
    listing = (torch.tensor([[7, 8]]), torch.tensor([2]))
    monkeypatch.setattr(
        cutedsl_top_k,
        "list_rows",
        lambda scores, lengths, top_k, ordered=False: calls.append(ordered) or listing,
    )
    check, _ = design_check()
    scores, lengths = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([3])
    with check.capture({}):
        assert cutedsl_top_k.list_rows(scores, lengths, 2, ordered=True) is listing
    assert calls == [True]


def test_the_flat_decode_step_is_read_from_the_decode_top_k():
    check, _ = design_check(decode_step=True)
    check.reference = {
        "scores": torch.tensor([[10.0, 9.0, 1.0, float("nan")]]),
        "lengths": torch.tensor([3]),
        "all_lengths": torch.tensor([3]),
    }
    check.prepare()
    assert check.check(torch.tensor([[0, 1]]))


def test_a_fused_read_is_picked_again_from_the_listing_it_read():
    """An expansion by threshold reads the listing's best buckets without picking them, so the check picks them itself."""
    check, top_k = design_check()
    check.parts = [{"fused": True, "listing": "listing"}]
    listings = {"listing": torch.tensor([[0, 2]])}
    check.scan_select = SimpleNamespace(pick=lambda listing: listings[listing])
    check.reference = {"scores": torch.tensor([[10.0, 9.0, 9.0, 1.0, 99.0]]), "lengths": torch.tensor([4])}
    check.prepare()
    assert check.check(torch.tensor([[0, 1, 2, 3]]))
    # The top-k's own buffer is not what a fused read picked by
    top_k.out[:] = torch.tensor([[1, 3]])
    assert check.check(torch.tensor([[0, 1, 2, 3]]))
    listings["listing"] = torch.tensor([[0, 3]])
    with pytest.raises(AssertionError, match="below the true top-k cutoff"):
        check.check(torch.tensor([[0, 1, 2, 3]]))


def listing_through(monkeypatch, scored):
    """The fused select listing as the selector calls it - the bucket shapes the fifth argument, None when unshaped"""
    listed = []

    def listing(select, queries, centroids, lengths, shapes=None):
        listed.append(shapes)
        return "listing"

    def dense_scores(queries, centroids, lengths, shapes=None):
        scored.append(shapes)
        return (queries.double() @ centroids.double().t()).float()

    monkeypatch.setattr(ScanSelect, "listed", listing)
    monkeypatch.setattr(scan_select_module, "dense_scores", dense_scores)
    return listed


@pytest.mark.parametrize("shapes", [None, "shapes"])
def test_a_fused_read_is_scored_with_the_bucket_shapes_it_was_listed_with(monkeypatch, shapes):
    check, _ = design_check()
    scored = []
    listed = listing_through(monkeypatch, scored)
    queries, centroids = keys_of(3), keys_of(5, seed=1)
    target = {}
    with check.capture(target):
        listing = check.scan_select.listed(queries, centroids, torch.tensor([5, 4, 3]), shapes)
    assert listing == check.parts[0]["listing"] == "listing"
    assert check.parts[0]["fused"]
    assert listed == scored == [shapes]
    assert target["scores"].shape == (3, 5)


class OracleSelector:
    """A flat prefill read as the selector makes it - gather the centroids, list them through the fused select"""

    def __init__(self, centroids, bucket_shapes=None):
        self.cache = SimpleNamespace(gather=lambda row, count: centroids, bucket_shapes=bucket_shapes)
        self.prefill_top_k = FakeTopK(torch.tensor([[0, 1]]))
        self.scan_select = ScanSelect.__new__(ScanSelect)

    def read_flat(self, row, index_queries, lengths):
        centroids = self.cache.gather(row, lengths.max())
        shapes = None if self.cache.bucket_shapes is None else "shapes"
        return self.scan_select.listed(index_queries.to(torch.bfloat16), centroids, lengths, shapes)


def test_a_fused_read_is_scored_independently_through_the_selector_s_call(monkeypatch):
    scored = []
    listed = listing_through(monkeypatch, scored)
    selector = OracleSelector(keys_of(6, seed=2))
    indexer = SimpleNamespace(index=SimpleNamespace(selector=selector))
    result = {}
    with lsa_score_oracle(indexer, result):
        selector.read_flat(0, keys_of(4).float(), torch.tensor([6, 5, 4, 3], dtype=torch.int32))
    assert listed == scored == [None]
    assert result["passed"] is True


def test_a_shaped_read_has_no_dot_to_be_scored_against(monkeypatch):
    scored = []
    listed = listing_through(monkeypatch, scored)
    selector = OracleSelector(keys_of(6, seed=2), bucket_shapes="table")
    indexer = SimpleNamespace(index=SimpleNamespace(selector=selector))
    result = {}
    with lsa_score_oracle(indexer, result):
        selector.read_flat(0, keys_of(4).float(), torch.tensor([6, 5, 4, 3], dtype=torch.int32))
    assert listed == ["shapes"]
    assert scored == []
    assert result["passed"] is None
    assert "shaped" in result["skipped"]


def test_a_shaped_decode_step_has_no_dot_to_be_scored_against():
    cache = SimpleNamespace(bucket_shapes="table")
    indexer = SimpleNamespace(index=SimpleNamespace(selector=SimpleNamespace(cache=cache)))
    result = {}
    with decode.lsa_decode_score_oracle(indexer, result):
        pass
    assert result["passed"] is None
    assert "shaped" in result["skipped"]


def test_a_selection_past_the_position_is_not_causal():
    check, _ = design_check()
    check.positions = torch.tensor([1]).numpy()
    check.reference = {"scores": torch.tensor([[10.0, 9.0, 9.0, 1.0]]), "lengths": torch.tensor([4])}
    check.prepare()
    with pytest.raises(AssertionError, match="Non-causal"):
        check.check(torch.tensor([[0, 1, 2]]))


def test_a_read_through_a_hierarchy_has_no_top_p_verdict(monkeypatch):
    """A beam ranks a level at a time, with no one row of scores to hold - the check says so rather than pass."""
    check, top_k = design_check()
    scores, lengths = torch.rand(1, 5), torch.tensor([5])

    def search(hierarchy, *args, **kwargs):
        # The pick is the read's own top-k, called inside the search
        top_k.listed(scores, lengths)
        return "beam"

    monkeypatch.setattr(Hierarchy, "beam_search", search)
    target = {}
    with check.capture(target):
        assert Hierarchy.beam_search(None) == "beam"
    assert check.tree
    assert target == {}
    assert Hierarchy.beam_search is search
    check.compare_scores({})
    check.prepare()
    assert check.check(torch.tensor([[0, 1]])) is None


def test_the_flat_read_keeps_the_scores_it_ranked():
    check, top_k = design_check()
    scores, lengths = torch.rand(1, 5), torch.tensor([5])
    target = {}
    with check.capture(target):
        top_k.listed(scores, lengths)
    assert not check.tree
    assert torch.equal(target["scores"], scores)
    assert torch.equal(target["all_lengths"], lengths)
    assert not check.parts[0]["fused"]


def test_a_second_capture_into_a_filled_target_is_refused():
    check, _ = design_check()
    with pytest.raises(RuntimeError, match="one top-k capture"), check.capture({"scores": torch.zeros(1)}):
        pass


class ListedTopK:
    """The top-k as a read uses it - it lists, and picks the best from the keys put back into it"""

    def __init__(self):
        self.keys = self.counts = None
        # One buffer the picks are returned in, as the engine reuses its own
        self.buffer = torch.zeros(4, 2, dtype=torch.int32)

    def listed(self, scores, lengths):
        best = scores.topk(2, dim=1).indices.to(torch.int32)
        return best, lengths

    def best(self):
        rows = self.keys.shape[0]
        self.buffer[:rows] = self.keys
        return pack_keys(torch.zeros(rows, 2), self.buffer[:rows])


def segmented_check(rows=None):
    top_k = ListedTopK()
    selector = SimpleNamespace(
        config=SimpleNamespace(top_p=2),
        prefill_top_k=top_k,
        decode_top_k=top_k,
        scan_select=ScanSelect.__new__(ScanSelect),
        decode_select=DecodeSelect(2),
    )
    module = SimpleNamespace(indexer=SimpleNamespace(index=SimpleNamespace(selector=selector)))
    return LSASelectionCheck(module, torch.arange(40), 40, score_rows=rows), top_k


def test_a_chunk_read_in_segments_is_kept_and_picked_a_segment_at_a_time():
    """A chunk wider than a page of rows reads in segments over different visible counts, one listing each"""
    check, top_k = segmented_check()
    first = torch.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]])
    second = torch.tensor([[1.0, 2.0, 8.0, 7.0], [6.0, 5.0, 4.0, 3.0]])
    target = {}
    with check.capture(target):
        top_k.listed(first, torch.tensor([3, 3]))
        top_k.listed(second, torch.tensor([4, 4]))
    # The narrower segment's scores are padded to the wider one's buckets
    assert target["scores"].shape == (4, 4)
    assert torch.equal(target["lengths"], torch.tensor([3, 3, 4, 4]))
    check.reference = target
    check.prepare()
    assert check.check(torch.zeros(4, 4, dtype=torch.int32))


def test_a_segment_s_wrong_pick_fails_the_check():
    check, top_k = segmented_check()
    target = {}
    with check.capture(target):
        top_k.listed(torch.tensor([[1.0, 5.0, 3.0]]), torch.tensor([3]))
        top_k.listed(torch.tensor([[1.0, 2.0, 8.0, 7.0]]), torch.tensor([4]))
    check.reference = target
    check.prepare()
    keys, counts = check.parts[1]["listing"]
    check.parts[1]["listing"] = (torch.tensor([[0, 1]], dtype=torch.int32), counts)
    with pytest.raises(AssertionError, match="below the true top-k cutoff"):
        check.check(torch.zeros(2, 4, dtype=torch.int32))


def test_sampled_rows_are_offset_into_the_segment_that_read_them():
    """The sampled rows are numbered over the chunk, a segment's scores over its own rows"""
    check, top_k = segmented_check(rows=[1, 2])
    first = torch.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]])
    second = torch.tensor([[1.0, 2.0, 8.0, 7.0], [6.0, 5.0, 4.0, 3.0]])
    target = {}
    with check.capture(target):
        top_k.listed(first, torch.tensor([3, 3]))
        top_k.listed(second, torch.tensor([4, 4]))
    assert torch.equal(target["scores"], torch.tensor([[9.0, 2.0, 4.0, 0.0], [1.0, 2.0, 8.0, 7.0]]))
    assert torch.equal(target["all_lengths"], torch.tensor([3, 3, 4, 4]))
    check.reference = target
    check.prepare()
    assert check.check(torch.zeros(4, 4, dtype=torch.int32))


def test_a_fused_segment_and_a_short_one_are_each_picked_their_own_way(monkeypatch):
    """A chunk can cross the fused select's gate - one segment listed by the dense product, the next fused"""
    check, top_k = segmented_check()
    scored = []
    listing_through(monkeypatch, scored)
    queries, centroids = keys_of(1), keys_of(5, seed=1)
    target = {}
    with check.capture(target):
        top_k.listed(torch.tensor([[1.0, 5.0, 3.0]]), torch.tensor([3]))
        check.scan_select.listed(queries, centroids, torch.tensor([5]))
    assert [part["fused"] for part in check.parts] == [False, True]
    dense = (queries.double() @ centroids.double().t()).float()
    best = dense.topk(2, dim=1).indices.to(torch.int32)
    check.scan_select = SimpleNamespace(pick=lambda listing: best)
    check.reference = target
    check.prepare()
    assert check.check(torch.zeros(2, 4, dtype=torch.int32))


def test_rows_sampled_for_the_scores_leave_the_rest_unchecked():
    assert score_rows(8192, 131072, "lsa", 1024, bucket_size=8) is None
    rows = score_rows(8192, 10485760, "lsa", 1024, bucket_size=8)
    assert (rows[0], rows[-1]) == (0, 8191)
    # Each sampled row is scored against every bucket the context holds, fp32
    assert len(rows) * (10485760 // 8) * 4 <= 1024 * 2**20


def design(**indexer):
    return SimpleNamespace(
        design=DesignSettings.from_blocks({**INDEXER, **indexer}, {}), lsa=True, pisa=False, indexer=True
    )


def runner_with(indexer, block=None):
    """A runner whose layer 3 holds the given indexer - what `Variant` reads off an engine it never starts"""
    module = SimpleNamespace(indexer=indexer, skip_topk=False)
    layers = [SimpleNamespace(is_linear_attn=False) for _ in range(4)]
    layers[3] = SimpleNamespace(self_attn=module, is_linear_attn=False)
    backend = SimpleNamespace()
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(lsa=block), hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
        ),
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        attn_backend=SimpleNamespace(full_attn_backend=backend, linear_attn_backend=backend),
        token_to_kv_pool=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(),
        device="cpu",
    )


def test_the_design_is_timed_only_where_the_built_indexer_is_the_design():
    from lsa.evaluation.performance.kernels.measure.variant import Variant

    block = DesignSettings.from_blocks(INDEXER, {}).body()
    args = SimpleNamespace(indexer="lsa", layer=3)

    # The plugin did not apply, so the layer still holds the checkpoint's own indexer
    with pytest.raises(ValueError, match="the checkpoint of its own"):
        Variant(runner_with(SimpleNamespace(is_lsa=False), block), args)
    built = Variant(runner_with(SimpleNamespace(is_lsa=True), block), args)
    assert built.lsa and built.kind == "sparse"
    assert built.design.indexer.window_size == WINDOW

    # The checkpoint's own indexer under a label that says it is not the design's is just as refused
    with pytest.raises(ValueError, match="the design"):
        Variant(runner_with(SimpleNamespace(is_lsa=True)), SimpleNamespace(indexer="native", layer=3))


def test_the_cases_recur_at_the_periods_the_design_maintains_them():
    flat = decode.maintenance_periods(design())
    assert flat == {"cluster_block_close": WINDOW, "page_allocation": PAGE_SIZE}

    fixed = decode.maintenance_periods(design(**FIXED))
    # Growth waits for a node cluster block of buckets, or the gate if more, at the end of a page
    assert fixed["hierarchy_growth"] == 4096 * 8
    assert {name: period for name, period in fixed.items() if name != "hierarchy_growth"} == flat

    native = SimpleNamespace(lsa=False, pisa=False, indexer=SimpleNamespace(index_kpool=4))
    assert decode.maintenance_periods(native) == {"pool_close": 4}
    assert decode.maintenance_periods(SimpleNamespace(lsa=False, pisa=False, indexer=None)) == {}


def test_a_decode_token_names_every_case_the_record_reads():
    index = Index()
    index.advance(PAGE_SIZE - 1)
    snapshot = index.snapshot()
    index.advance(PAGE_SIZE + 1)
    variant = design()
    case = decode.decode_case(SimpleNamespace(context=PAGE_SIZE), variant, snapshot, PAGE_SIZE + 2, 1)

    assert set(Kernels.CASES) <= set(case)
    assert case["offset"] == 2
    assert case["prefix_tokens"] == PAGE_SIZE + 1
    assert case["page_allocation"] and case["cluster_block_close"]
    assert not case["pool_close"]
    assert case["periods"] == decode.maintenance_periods(variant)


def test_the_native_decode_token_closes_a_pool_every_fourth_key():
    snapshot = SimpleNamespace(state=None)
    variant = SimpleNamespace(lsa=False, pisa=False, indexer=SimpleNamespace(index_kpool=4))
    at_four = decode.decode_case(SimpleNamespace(context=128), variant, snapshot, 132, 0)
    at_five = decode.decode_case(SimpleNamespace(context=128), variant, snapshot, 133, 0)
    assert at_four["pool_close"] and not at_five["pool_close"]
    assert not any(at_four[name] for name in Kernels.CASES if name != "pool_close")


def test_qsa_closes_a_block_every_compress_ratio_keys():
    qsa = SimpleNamespace(lsa=False, pisa=False, indexer=SimpleNamespace(compress_ratio=4))
    assert decode.maintenance_periods(qsa) == {"pool_close": 4}
    snapshot = SimpleNamespace(state=None)
    assert decode.decode_case(SimpleNamespace(context=128), qsa, snapshot, 132, 0)["pool_close"]
    assert not decode.decode_case(SimpleNamespace(context=128), qsa, snapshot, 133, 0)["pool_close"]


def test_both_languages_decode_scans_are_charged_to_the_scan():
    charged = {}

    def wrap(obj, attribute, name, stack):
        charged[(getattr(obj, "__name__", None), attribute)] = name

    indexer = SimpleNamespace(index=SimpleNamespace(projection=SimpleNamespace(), write=None))
    wrap_lsa(indexer, wrap, None)
    assert charged[("DecodeSelect", "listed")] == "index q scan"
    assert charged[("lsa.sglang.index.selector", "scan_paged")] == "index q scan"
