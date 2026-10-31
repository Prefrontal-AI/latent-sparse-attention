"""The batched descend - every decode row read down the graph tree at fixed shapes, eagerly and in a CUDA graph.

The eager `decode_descend`, a request at a time down its own hierarchy, is the reference. The batched descend scores
the same candidates in the same order, but its scores come from a gather kernel where the reference's come from a
matrix product, so a score may differ by fp32 rounding. A row is held to the reference as a set of KV slots, and a slot
that differs is allowed only where its bucket's score sits within `ROUNDING` of the score the reference stopped at.
"""

import pytest
import torch

from lsa.core.kernels.cutedsl import beam as cutedsl_beam
from lsa.core.kernels.cutedsl import expansion as cutedsl_expansion
from lsa.core.kernels.cutedsl import spread as cutedsl_spread
from lsa.sglang.index import graph_descend
from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.settings.design import PAGE_SIZE
from tests.unit.sglang.test_selector import DEEP, DEVICE, TOP_START, WINDOW, Store, sets

pytestmark = pytest.mark.gpu

# Scores of one bucket differ by the order fp32 adds 64 products in, about 1e-6 of the score - 1e-3 is a thousand
# times that, and far under the gap between neighbouring picks' scores in a hundred or more
ROUNDING = 1e-3
SINK = 4
LOW_START = {**DEEP, "start_nodes": 150}
DESIGNS = {"deep": DEEP, "from the top": TOP_START, "low start": LOW_START}
# No level yet, three levels, four levels
ENDS = [PAGE_SIZE - 64, 2 * PAGE_SIZE + 320, 4 * PAGE_SIZE + 320]


@pytest.fixture(autouse=True)
def every_page(monkeypatch):
    # The first level pairs at every page end, so a few pages grow several levels
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", 256)


def make_store(indexer, ends, kernels="triton", requests=None, seed=0, longest=0):
    """A store whose rows 0.. hold requests grown to `ends`, with `requests` rows in all, and keys to `longest`"""
    requests = requests or len(ends)
    # Every row holds its own pages of buckets, so the cache is sized for as many contexts as there are rows
    serving = {"decode": "descend", "kernels": kernels, "spare_contexts": requests}
    store = Store(indexer, max(*ends, longest) + 400, serving, seed=seed, requests=requests)
    grow(store, ends)
    return store


def grow(store, ends):
    """Bring row i's cache up to ends[i], as the engine does before a forward"""
    rows = list(range(len(ends)))
    store.selector.maintain([f"request {row}" for row in rows], rows, ends, store.index_keys_of)


def batch(store, ends, rows=None):
    """The decode inputs of rows at their ends - a query each, at its own position"""
    rows = list(range(len(ends))) if rows is None else rows
    positions = torch.tensor([ends[row] for row in rows], device=DEVICE, dtype=torch.int32)
    return store.queries[positions.long()], positions, torch.tensor(rows, device=DEVICE)


def captured(store, queries, positions, rows):
    """The graph of one decode step over static inputs, warmed up first, and the list it writes"""
    selector = store.selector
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        selector.decode_descend(queries, positions, rows, batched=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lists = selector.decode_descend(queries, positions, rows)
    return graph, lists


class Judge:
    """Tells whether a slot a read differs on is a tie of the reference's cutoff"""

    def __init__(self, store, row, query, position, reference):
        cache = store.cache
        visible = cache.visible(row)
        bucket_rows = cache.logical_rows(row, visible)[:visible]
        members = cache.bucket_members[bucket_rows].long()
        self.owner = torch.full((store.keys.shape[0],), -1, device=DEVICE, dtype=torch.long)
        index = torch.arange(visible, device=DEVICE)[:, None].expand_as(members)
        self.owner[members[members >= 0]] = index[members >= 0]
        query = query.to(torch.bfloat16).float()
        self.scores = cache.bucket_centroids[bucket_rows].float() @ query
        self.position = position
        # The reference stops at the worst bucket it read past the window and the sinks
        slots = reference[reference >= 0]
        outside = slots[(slots >= SINK) & (slots <= position - WINDOW)]
        owners = self.owner[outside.long()]
        self.cutoff = self.scores[owners[owners >= 0]].min()

    def tied(self, slot):
        bucket = int(self.owner[slot])
        if bucket < 0:
            return False, float("nan")
        score = float(self.scores[bucket])
        gap = abs(score - float(self.cutoff))
        return gap <= ROUNDING * abs(float(self.cutoff)), gap


def compare(store, got, want, queries, positions, rows, label):
    """Hold each row of `got` to `want` as a set of slots - differing ones only at a tie - and return how many differ"""
    differing, largest = 0, 0.0
    for i, row in enumerate(rows.tolist()):
        have, expected = set(sets(got[i : i + 1])[0]), set(sets(want[i : i + 1])[0])
        if have == expected:
            continue
        judge = Judge(store, row, queries[i], int(positions[i]), want[i])
        for slot in have ^ expected:
            ok, gap = judge.tied(slot)
            assert ok, (
                f"{label}: row {row} differs on slot {slot} at {gap} from the cutoff {float(judge.cutoff)}"
            )
            differing += 1
            largest = max(largest, gap)
    print(f"{label}: {differing} slots differ, largest gap {largest:.3g}")
    return differing


def levels(store, ends):
    return [store.cache.hierarchies[row].levels for row in range(len(ends))]


@pytest.mark.parametrize("kernels", ["triton", "cutedsl"])
@pytest.mark.parametrize("name", DESIGNS)
def test_the_batched_descend_eagerly_and_in_a_graph_reads_what_the_per_request_descend_reads(name, kernels):
    store = make_store(DESIGNS[name], ENDS, kernels, seed=1)
    assert levels(store, ENDS) == [0, 3, 4]
    queries, positions, rows = batch(store, ENDS)
    reference = store.selector.decode_descend(queries, positions, rows)
    assert all(len(ids) > WINDOW for ids in sets(reference))
    batched = store.selector.decode_descend(queries, positions, rows, batched=True)
    assert batched.shape == reference.shape and batched.dtype == reference.dtype
    differing = compare(store, batched, reference, queries, positions, rows, f"{name} {kernels} eager")
    graph, lists = captured(store, queries, positions, rows)
    for _ in range(3):
        lists.zero_()
        graph.replay()
        torch.cuda.synchronize()
        differing += compare(store, lists, reference, queries, positions, rows, f"{name} {kernels} graph")
    # The graph reads what the same code reads outside it
    assert torch.equal(lists, batched)
    print(
        f"{name} {kernels}: {differing} slots differ in all, starts {store.cache.graph_tree.start.tolist()}"
    )


@pytest.mark.parametrize("name", ["deep", "from the top"])
def test_levels_read_the_same_through_the_triton_chain_as_through_the_cutedsl_kernels(name, monkeypatch):
    monkeypatch.setattr(graph_descend, "keep_kernel", lambda *_: "triton chain")
    test_the_batched_descend_eagerly_and_in_a_graph_reads_what_the_per_request_descend_reads(name, "cutedsl")


def test_the_widths_and_kernels_of_each_level():
    store = make_store(TOP_START, ENDS)
    tree = store.cache.graph_tree
    plans = graph_descend.plan_levels(store.settings, tree)
    kernels = {}
    for kernel in ("triton", "cutedsl"):
        for level, plan in enumerate(plans, 1):
            kernels[kernel, level] = graph_descend.keep_kernel(
                plan, store.queries[:1].bfloat16(), tree.node_centroids[level - 1], kernel
            )
    for level, plan in enumerate(plans, 1):
        print(
            f"level {level}: {plan} triton: {kernels['triton', level]} cutedsl: {kernels['cutedsl', level]}"
        )
        assert kernels["triton", level] == "triton chain"
        assert kernels["cutedsl", level] == (
            "beam step" if plan.width <= graph_descend.SINGLE_CTA_WIDTH else "scored keep"
        )
        # The trees here are narrower than a cluster is spread over
        assert cutedsl_spread.layout(1, plan.width, plan.keep) is None
    wide = graph_descend.LevelPlan(32768, 4095, 2048, 32768)
    assert cutedsl_spread.layout(1, wide.width, wide.keep) == (
        cutedsl_spread.SPREAD_THREADS,
        cutedsl_spread.CLUSTER,
    )
    assert cutedsl_spread.layout(cutedsl_spread.FEW_ROWS, wide.width, wide.keep) == (
        cutedsl_spread.SPREAD_THREADS,
        cutedsl_spread.CLUSTER,
    )
    assert cutedsl_spread.layout(cutedsl_spread.FEW_ROWS + 1, wide.width, wide.keep) == (
        cutedsl_spread.SPREAD_THREADS,
        cutedsl_spread.CLUSTER // 2,
    )
    tree_start = tree.start
    for level, plan in enumerate(plans, 1):
        # No request of the store outgrows its plan - a start level holds its node buckets in the start width, a level
        # below its start its open ones in the open width
        made, closed = tree.made[level - 1], tree.closed[level - 1]
        assert bool(((tree_start != level) | (made <= plan.start_width)).all())
        assert bool(((tree_start <= level) | (made - closed <= plan.open_width)).all())
    assert graph_descend.uncovered_width(store.settings, store.cache.rows_per_page) > 0


def spy(monkeypatch, module, name):
    """Count the calls of a function of a module, which still runs"""
    calls = []
    original = getattr(module, name)
    monkeypatch.setattr(
        module, name, lambda *arguments, **options: calls.append(1) or original(*arguments, **options)
    )
    return calls


@pytest.mark.parametrize("name", ["deep", "from the top"])
def test_the_scored_keep_and_the_scored_pick_run_in_a_graph_and_read_what_they_read_eagerly(
    name, monkeypatch
):
    # The trees here are too small for a level past the single CTA's columns, so every level is made to be scored apart
    # => and none is spread over a cluster
    monkeypatch.setattr(graph_descend, "SINGLE_CTA_WIDTH", 0)
    monkeypatch.setattr(cutedsl_spread, "SPREAD_FROM", 2**31)
    store = make_store(DESIGNS[name], ENDS, "cutedsl", seed=1)
    tree = store.cache.graph_tree
    plans = graph_descend.plan_levels(store.settings, tree)
    scored_levels = [
        level
        for level, plan in enumerate(plans, 1)
        if plan.keep < plan.width
        and graph_descend.keep_kernel(
            plan, store.queries[:1].bfloat16(), tree.node_centroids[level - 1], "cutedsl"
        )
        == "scored keep"
    ]
    assert scored_levels
    keeps = spy(monkeypatch, cutedsl_beam, "keep_scored")
    picks = spy(monkeypatch, cutedsl_beam, "pick_scored")
    queries, positions, rows = batch(store, ENDS)
    reference = store.selector.decode_descend(queries, positions, rows)
    eager = store.selector.decode_descend(queries, positions, rows, batched=True)
    assert len(keeps) == len(scored_levels) and len(picks) == 1
    compare(store, eager, reference, queries, positions, rows, f"{name} scored eager")
    graph, lists = captured(store, queries, positions, rows)
    # The warm-up and the capture each ran them
    assert len(keeps) == 3 * len(scored_levels) and len(picks) == 3
    for _ in range(3):
        lists.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(lists, eager)


@pytest.mark.parametrize("single_cta_width", [0, graph_descend.SINGLE_CTA_WIDTH])
def test_a_capture_compiles_nothing_at_batch_sizes_either_side_of_the_threads_threshold(
    single_cta_width, monkeypatch
):
    monkeypatch.setattr(graph_descend, "SINGLE_CTA_WIDTH", single_cta_width)
    # The scored kernels are the ones held here, so none is spread over a cluster
    monkeypatch.setattr(cutedsl_spread, "SPREAD_FROM", 2**31)
    # Nothing is compiled yet, so the warm-up is what compiles
    for module, name in (
        (cutedsl_beam, "COMPILED"),
        (cutedsl_beam, "COMPILED_PICKS"),
        (cutedsl_beam, "COMPILED_SCORED_PICKS"),
        (cutedsl_expansion, "COMPILED"),
        (cutedsl_expansion, "COMPILED_IN_ORDER"),
    ):
        monkeypatch.setattr(module, name, {})
    compiles = spy(monkeypatch, cutedsl_beam.cute, "compile")
    ends = ENDS[1:]
    store = make_store(TOP_START, ends, "cutedsl", requests=2, seed=6)
    limit = cutedsl_expansion.ROWS_PER_SM * cutedsl_expansion.multiprocessors(torch.device(DEVICE))
    for count in (limit, limit + 1):
        queries, positions, rows = batch(store, ends, rows=[i % 2 for i in range(count)])
        before = len(compiles)
        with torch.cuda.stream(torch.cuda.Stream()):
            eager = store.selector.decode_descend(queries, positions, rows, batched=True)
        torch.cuda.synchronize()
        warmed = len(compiles)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lists = store.selector.decode_descend(queries, positions, rows)
        assert len(compiles) == warmed
        # The warm-up compiled what this size needs and the other size shares, or the count would be blind
        assert warmed > before or count > limit
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(lists, eager)
        print(f"{count} rows, single CTA width {single_cta_width}: {warmed - before} compiles in the warm-up")
    threads = {key[-2] for key in cutedsl_expansion.COMPILED}
    assert threads == {cutedsl_expansion.FEW_ROWS_THREADS, cutedsl_expansion.MANY_ROWS_THREADS}
    assert cutedsl_beam.COMPILED_SCORED_PICKS
    assert any(key[-1] for key in cutedsl_beam.COMPILED)


@pytest.mark.parametrize("name", ["deep", "from the top"])
def test_the_spread_keep_and_the_spread_pick_run_in_a_graph_and_read_what_they_read_eagerly(
    name, monkeypatch
):
    # The trees here are too narrow for a cluster, so every level is made wide enough to be spread
    monkeypatch.setattr(graph_descend, "SINGLE_CTA_WIDTH", 0)
    monkeypatch.setattr(cutedsl_spread, "SPREAD_FROM", 0)
    store = make_store(DESIGNS[name], ENDS, "cutedsl", seed=1)
    plans = graph_descend.plan_levels(store.settings, store.cache.graph_tree)
    spread_levels = [level for level, plan in enumerate(plans, 1) if plan.keep < plan.width]
    assert spread_levels
    keeps = spy(monkeypatch, cutedsl_spread, "keep_scored")
    picks = spy(monkeypatch, cutedsl_spread, "pick_scored")
    queries, positions, rows = batch(store, ENDS)
    reference = store.selector.decode_descend(queries, positions, rows)
    eager = store.selector.decode_descend(queries, positions, rows, batched=True)
    assert len(keeps) == len(spread_levels) and len(picks) == 1
    compare(store, eager, reference, queries, positions, rows, f"{name} spread eager")
    graph, lists = captured(store, queries, positions, rows)
    # The warm-up and the capture each ran them
    assert len(keeps) == 3 * len(spread_levels) and len(picks) == 3
    for _ in range(3):
        lists.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(lists, eager)


def test_a_capture_compiles_nothing_at_batch_sizes_either_side_of_the_cluster_threshold(monkeypatch):
    monkeypatch.setattr(graph_descend, "SINGLE_CTA_WIDTH", 0)
    monkeypatch.setattr(cutedsl_spread, "SPREAD_FROM", 0)
    # Nothing is compiled yet, so the warm-up is what compiles
    for name in ("COMPILED_KEEPS", "COMPILED_PICKS"):
        monkeypatch.setattr(cutedsl_spread, name, {})
    compiles = spy(monkeypatch, cutedsl_beam.cute, "compile")
    ends = ENDS[1:]
    store = make_store(TOP_START, ends, "cutedsl", requests=2, seed=6)
    for count in (cutedsl_spread.FEW_ROWS, cutedsl_spread.FEW_ROWS + 1):
        queries, positions, rows = batch(store, ends, rows=[i % 2 for i in range(count)])
        before = len(compiles)
        with torch.cuda.stream(torch.cuda.Stream()):
            eager = store.selector.decode_descend(queries, positions, rows, batched=True)
        torch.cuda.synchronize()
        warmed = len(compiles)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lists = store.selector.decode_descend(queries, positions, rows)
        assert len(compiles) == warmed
        # Each size compiles its own cluster, or the count would be blind
        assert warmed > before
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(lists, eager)
    clusters = {key[-1] for key in cutedsl_spread.COMPILED_KEEPS}
    assert clusters == {cutedsl_spread.CLUSTER, cutedsl_spread.CLUSTER // 2}
    assert {key[-1] for key in cutedsl_spread.COMPILED_PICKS} == clusters


@pytest.mark.parametrize("name", ["from the top", "low start"])
def test_a_batch_mixing_start_levels_reads_each_row_from_its_own(name):
    store = make_store(DESIGNS[name], ENDS, seed=2)
    starts = store.cache.graph_tree.start.tolist()
    print(f"{name}: start levels {starts}")
    assert starts[0] == 0 and len(set(starts[1:])) == 2
    queries, positions, rows = batch(store, ENDS)
    reference = store.selector.decode_descend(queries, positions, rows)
    batched = store.selector.decode_descend(queries, positions, rows, batched=True)
    compare(store, batched, reference, queries, positions, rows, f"{name} mixed")
    # A row reads the same alone as in the batch
    for i in range(len(ENDS)):
        one = slice(i, i + 1)
        alone = store.selector.decode_descend(queries[one], positions[one], rows[one], batched=True)
        assert sets(alone) == sets(batched[one])


def test_a_graph_replays_after_a_growth_that_adds_a_level_without_a_recapture():
    ends = [3 * PAGE_SIZE + 320, 2 * PAGE_SIZE + 320]
    store = make_store(TOP_START, ends, seed=3, longest=4 * PAGE_SIZE + 320)
    assert levels(store, ends) == [3, 3]
    queries, positions, rows = batch(store, ends)
    static = queries.clone(), positions.clone(), rows
    graph, lists = captured(store, *static)
    graph.replay()
    torch.cuda.synchronize()
    before = lists.clone()
    assert sets(before) == sets(store.selector.decode_descend(queries, positions, rows))
    # Row 0 crosses a page end and grows a fourth level, in the eager step before the next forward
    grown = [4 * PAGE_SIZE + 320, 2 * PAGE_SIZE + 320]
    grow(store, grown)
    assert levels(store, grown) == [4, 3]
    queries, positions, rows = batch(store, grown)
    static[0].copy_(queries)
    static[1].copy_(positions)
    reference = store.selector.decode_descend(queries, positions, rows)
    graph.replay()
    torch.cuda.synchronize()
    assert sets(lists) != sets(before)
    compare(store, lists, reference, queries, positions, rows, "growth replay")


@pytest.mark.parametrize("kernels", ["triton", "cutedsl"])
def test_rows_padded_into_a_captured_batch_read_nothing_but_minus_one(kernels):
    ends = ENDS[1:]
    store = make_store(TOP_START, ends, kernels, requests=4, seed=4)
    queries, positions, rows = batch(store, ends)
    # Two padded rows, as a graph captured for four requests is handed two - a request row that holds nothing, at no position
    pad = torch.full((2,), 3, device=DEVICE)
    queries = torch.cat([queries, store.queries[:2]])
    positions = torch.cat([positions, torch.full((2,), -1, device=DEVICE, dtype=torch.int32)])
    rows = torch.cat([rows, pad])
    graph, lists = captured(store, queries, positions, rows)
    lists.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((lists[2:] == -1).all())
    assert bool((lists[:2] >= 0).any(1).all())
    live = store.selector.decode_descend(queries[:2], positions[:2], rows[:2])
    compare(store, lists[:2], live, queries[:2], positions[:2], rows[:2], "padded batch")


@pytest.mark.parametrize("kernels", ["triton", "cutedsl"])
def test_rows_padded_as_sglang_pads_them_name_request_row_0_at_position_0_and_read_inside_the_tables(kernels):
    # sglang zeroes the padded tail of a replayed batch's request rows and positions - the padded rows name row 0, which
    # a live request may hold, at position 0, and the queries are whatever the zeroed tokens projected to
    ends = ENDS[1:]
    store = make_store(TOP_START, ends, kernels, requests=4, seed=5)
    queries, positions, rows = batch(store, ends, rows=[1, 0])
    live = store.selector.decode_descend(queries, positions, rows)
    queries = torch.cat([queries, store.queries[:3]])
    positions = torch.cat([positions, torch.zeros(3, device=DEVICE, dtype=torch.int32)])
    rows = torch.cat([rows, torch.zeros(3, device=DEVICE, dtype=torch.long)])
    graph, lists = captured(store, queries, positions, rows)
    lists.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    slots = store.table.shape[1]
    assert bool(((lists >= -1) & (lists < slots)).all())
    compare(store, lists[:2], live, queries[:2], positions[:2], rows[:2], "padded as sglang pads")
