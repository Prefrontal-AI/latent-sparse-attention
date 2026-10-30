"""The harness's validation cycle over the real selector, on a GPU - snapshot, rerun, hold the selection to the prepared one.

The engine is absent: torch's exact top-k stands in for the radix one, behind the same `out` buffer the check
reads its picks from, so no sglang is needed. The cycle is the one `measure` runs - prepare a state, snapshot
it, run a chunk, snapshot what it left, put the first back, run the chunk again and hold the rerun to the first.
What the timing paths record from it is `tests/unit/evaluation/performance/kernels/test_measure_lsa.py`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.evaluation.performance.kernels.measure import decode
from lsa.evaluation.performance.kernels.measure.score_check import lsa_score_oracle
from lsa.evaluation.performance.kernels.measure.selection import LSASelectionCheck, selection_rows
from lsa.evaluation.performance.kernels.measure.snapshot import IndexState
from lsa.sglang.index import selector as selector_module
from lsa.sglang.index.selector import Selector
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
WINDOW = 64
FLAT = {
    "window_size": WINDOW,
    "sink_size": 4,
    "index_dim": 64,
    "pool_size": 4,
    "bucket_size": 8,
    "bucket_cap": 16,
    "top_k": 512,
    "top_p": 128,
}
FIXED = {
    **FLAT,
    "top_level": 1,
    "node_bucket_size": 4,
    "node_cluster_block_size": 64,
    "beam_width": 2048,
}
SHAPED_FLAT = {**FLAT, "bucket_shape": {"rank": 2, "deviations": 2.0}}
ROW = 1
CHUNK = 1024


class Layer:
    """A selector over random index keys with the identity slot table, in the shape the harness holds a layer"""

    def __init__(self, indexer, total, seed=0, chunk=CHUNK):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        anchors = torch.randn(total // 12 + 1, 64, device=DEVICE, generator=generator)
        noise = 0.5 * torch.randn(total, 64, device=DEVICE, generator=generator)
        self.keys = F.normalize(anchors[torch.arange(total, device=DEVICE) // 12] + noise, dim=-1).to(
            torch.bfloat16
        )
        earlier = (
            torch.rand(total, device=DEVICE, generator=generator) * torch.arange(total, device=DEVICE)
        ).long()
        self.queries = F.normalize(
            self.keys[earlier].float() + 0.1 * torch.randn(total, 64, device=DEVICE, generator=generator),
            dim=-1,
        )
        self.table = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(3, -1).contiguous()
        self.settings = DesignSettings.from_blocks(indexer, {})
        self.selector = Selector(
            self.settings,
            total,
            3,
            total,
            chunk,
            DEVICE,
            lambda: self.table,
        )
        self.index = SimpleNamespace(index_keys=self.keys.clone(), selector=self.selector)
        self.module = SimpleNamespace(indexer=SimpleNamespace(index=self.index))
        self.total = total

    def keys_of(self, positions):
        return self.keys[positions.long()]

    def catch_up(self, end):
        self.selector.maintain(["request"], [ROW], [end], lambda row: self.keys_of)

    def prefill(self, start, stop):
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        return self.selector.prefill(ROW, self.queries[start:stop], positions, self.keys_of, {ROW})

    def decode(self, position):
        rows = torch.tensor([ROW], device=DEVICE)
        positions = torch.tensor([position], device=DEVICE, dtype=torch.int32)
        return self.selector.decode(self.queries[position : position + 1], positions, rows)

    def check(self, decoding=False, rows=None):
        slots = torch.arange(self.total, device=DEVICE)
        return LSASelectionCheck(self.module, slots, self.total, None, rows, decoding)


def cycle(layer, start, length=CHUNK, rows=None):
    """Prepare, snapshot, run a chunk, snapshot what it left, put the first back and run it again.

    Returns the prepared selection, the rerun's, the check that held the rerun and the state it left.
    """
    layer.catch_up(start)
    locations = torch.arange(start, start + length, device=DEVICE)
    before = IndexState(layer.index, locations)
    check = layer.check(rows=rows)
    prepared_scores = {}
    with check.capture(prepared_scores):
        prepared = layer.prefill(start, start + length)
    after = IndexState(layer.index, locations)
    before.restore()
    rerun_scores = {}
    with check.capture(rerun_scores):
        rerun = layer.prefill(start, start + length)
    return prepared, rerun, check, after, (prepared_scores, rerun_scores)


def test_a_flat_rerun_from_the_snapshot_chooses_the_prepared_selection_and_leaves_the_state_it_left():
    layer = Layer(FLAT, 2 * PAGE_SIZE + CHUNK)
    prepared, rerun, check, after, (first, second) = cycle(layer, 2 * PAGE_SIZE)

    assert not check.tree and not any(part["fused"] for part in check.parts)
    after.check()
    verdict = selection_rows(rerun, prepared, FLAT["bucket_cap"])
    assert verdict["differing"] == 0, verdict
    check.reference = first
    check.compare_scores(second)
    check.prepare()
    assert check.check(rerun)


def test_a_fused_rerun_is_held_to_the_exact_top_p_of_the_listing_s_scores(monkeypatch):
    monkeypatch.setattr(selector_module, "SCAN_SELECT_BUCKETS", 0)
    layer = Layer(FLAT, 2 * PAGE_SIZE + CHUNK, seed=4)
    prepared, rerun, check, after, (first, second) = cycle(layer, 2 * PAGE_SIZE)

    assert all(part["fused"] for part in check.parts)
    after.check()
    assert selection_rows(rerun, prepared, FLAT["bucket_cap"])["exact"]
    check.reference = first
    check.compare_scores(second)
    check.prepare()
    assert check.check(rerun)


@pytest.mark.parametrize("fused", [False, True])
def test_a_chunk_wider_than_a_page_is_read_in_segments_and_held_whole(monkeypatch, fused):
    """A read holds at most a page of rows, so a chunk wider than a page reads in segments, one listing each"""
    if fused:
        monkeypatch.setattr(selector_module, "SCAN_SELECT_BUCKETS", 0)
    start, length = 2 * PAGE_SIZE, PAGE_SIZE + CHUNK
    layer = Layer(FLAT, start + length, seed=6, chunk=length)
    # Sampled rows on both sides of the segments' border, numbered over the chunk
    rows = [0, PAGE_SIZE - 1, PAGE_SIZE, length - 1]
    prepared, rerun, check, after, (first, second) = cycle(layer, start, length, rows)

    assert len(check.parts) == 2
    assert [part["fused"] for part in check.parts] == [fused, fused]
    assert first["scores"].shape[0] == len(rows)
    after.check()
    assert selection_rows(rerun, prepared, FLAT["bucket_cap"])["exact"]
    check.reference = first
    check.compare_scores(second)
    check.prepare()
    assert check.check(rerun)


def test_a_flat_read_is_scored_independently():
    layer = Layer(FLAT, 2 * PAGE_SIZE + CHUNK, seed=5)
    layer.catch_up(2 * PAGE_SIZE)
    result = {}
    with lsa_score_oracle(layer.module.indexer, result):
        layer.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + CHUNK)
    assert result["passed"] is True
    assert result["max_error_over_bound"] <= 1


def test_a_fused_read_is_scored_independently(monkeypatch):
    monkeypatch.setattr(selector_module, "SCAN_SELECT_BUCKETS", 0)
    layer = Layer(FLAT, 2 * PAGE_SIZE + CHUNK, seed=5)
    layer.catch_up(2 * PAGE_SIZE)
    result = {}
    with lsa_score_oracle(layer.module.indexer, result):
        layer.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + CHUNK)
    assert result["passed"] is True
    assert result["max_error_over_bound"] <= 1


def test_a_shaped_fused_rerun_is_held_to_the_exact_top_p_of_its_shaped_scores(monkeypatch):
    monkeypatch.setattr(selector_module, "SCAN_SELECT_BUCKETS", 0)
    layer = Layer(SHAPED_FLAT, 2 * PAGE_SIZE + CHUNK, seed=4)
    prepared, rerun, check, after, (first, second) = cycle(layer, 2 * PAGE_SIZE)

    assert layer.selector.cache.bucket_shapes is not None
    assert all(part["fused"] for part in check.parts)
    after.check()
    assert selection_rows(rerun, prepared, FLAT["bucket_cap"])["exact"]
    check.reference = first
    check.compare_scores(second)
    check.prepare()
    assert check.check(rerun)
    # A shaped score is the estimated best key, not a dot => no independent score verdict
    result = {}
    with lsa_score_oracle(layer.module.indexer, result):
        pass
    assert result["passed"] is None
    assert "shaped" in result["skipped"]


def test_a_tree_rerun_has_no_top_p_verdict_and_no_score_verdict():
    # Four pages hold the 4,096 buckets the first growth waits for, so the chunk after them reads through a level
    layer = Layer(FIXED, 4 * PAGE_SIZE + CHUNK, seed=6)
    prepared, rerun, check, after, _ = cycle(layer, 4 * PAGE_SIZE)

    assert layer.selector.cache.hierarchies[ROW].levels >= 1
    assert check.tree
    assert check.check(rerun) is None
    after.check()
    assert selection_rows(rerun, prepared, FIXED["bucket_cap"])["exact"]
    result = {}
    with lsa_score_oracle(layer.module.indexer, result, tree=True):
        pass
    assert result["passed"] is None


def test_a_flat_decode_step_is_held_to_the_exact_top_p_and_scored_independently():
    layer = Layer(FLAT, 2 * PAGE_SIZE + 8, seed=7)
    layer.catch_up(2 * PAGE_SIZE)
    position = 2 * PAGE_SIZE
    check = layer.check(decoding=True)
    scores = {}
    with check.capture(scores):
        selected = layer.decode(position)
    check.reference = scores
    check.prepare()
    assert check.check(selected)
    result = {}
    with decode.lsa_decode_score_oracle(layer.module.indexer, result):
        layer.decode(position)
    assert result["passed"] is True


def test_a_pass_that_clusters_a_block_is_named_by_the_cases_it_triggered():
    layer = Layer(FLAT, 2 * PAGE_SIZE + CHUNK, seed=8)
    layer.catch_up(2 * PAGE_SIZE - 1)
    state = IndexState(layer.index, torch.arange(2 * PAGE_SIZE, 2 * PAGE_SIZE + 4, device=DEVICE))
    layer.catch_up(2 * PAGE_SIZE + WINDOW)
    assert state.changed(ROW) == {
        "page_allocation": True,
        "cluster_block_close": True,
        "hierarchy_growth": False,
    }
