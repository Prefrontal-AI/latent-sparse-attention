"""What a query reads - the selector's routing for flat, fixed and tree, prefill and decode.

On a GPU, with torch's exact top-k standing in for the engine's radix one so no sglang is needed. The torch
reference is the oracle for a flat read (`lsa.core.read`) and for a beam of any width (`lsa.core.read_hierarchy`),
and a beam as wide as a level reads what the flat scan reads. The freeze's own reads of every design are
`tests/integ/sglang/test_goldens.py`.
"""

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from lsa.core import Buckets, Shapes, build_hierarchy, read, read_hierarchy
from lsa.core import hierarchy as core_hierarchy
from lsa.core.kernels.triton.keys import unpack_ids
from lsa.core.shape import build_shapes as reference_build_shapes
from lsa.core.shape import unpack_shape_basis
from lsa.sglang import context_buckets
from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index import selector as selector_module
from lsa.sglang.index.selector import Selector
from lsa.sglang.settings.design import DECODE_CONTEXT_LIMITS, PAGE_SIZE, DesignSettings

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
NODES = {"node_bucket_size": 4, "node_cluster_block_size": 64, "beam_width": 2048}
FIXED = {**FLAT, "top_level": 1, **NODES}
DEEP = {**FLAT, "top_level": 16, **NODES, "start_nodes": 32768}
# No start level, so the beam descends from the top level
TOP_START = {**FLAT, "top_level": 16, **NODES}
WIDTH = 512 + WINDOW + 4
ROW = 1
# The most query rows a prefill call is handed
CHUNK = 8192


class Store:
    """A selector over random index keys with the identity slot table, and the keys and queries it reads"""

    def __init__(self, indexer, total, serving=None, seed=0, max_context=None, requests=3):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        anchors = torch.randn(total // 12 + 1, 64, device=DEVICE, generator=generator)
        keys = anchors[torch.arange(total, device=DEVICE) // 12] + 0.5 * torch.randn(
            total, 64, device=DEVICE, generator=generator
        )
        self.keys = F.normalize(keys, dim=-1).to(torch.bfloat16)
        earlier = (
            torch.rand(total, device=DEVICE, generator=generator) * torch.arange(total, device=DEVICE)
        ).long()
        noisy = self.keys[earlier].float() + 0.1 * torch.randn(total, 64, device=DEVICE, generator=generator)
        self.queries = F.normalize(noisy, dim=-1)
        self.table = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(requests, -1).contiguous()
        self.settings = DesignSettings.from_blocks(indexer, serving or {})
        self.selector = Selector(
            self.settings, total, requests, max_context or total, CHUNK, DEVICE, lambda: self.table
        )
        self.cache = self.selector.cache

    def index_keys_of(self, row=ROW):
        return lambda positions: self.keys[positions.long()]

    def catch_up(self, end):
        self.selector.maintain(["request"], [ROW], [end], self.index_keys_of)

    def prefill(self, start, stop):
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        return self.selector.prefill(ROW, self.queries[start:stop], positions, self.index_keys_of(), {ROW})

    def decode(self, position):
        rows = torch.tensor([ROW], device=DEVICE)
        positions = torch.tensor([position], device=DEVICE, dtype=torch.int32)
        return self.selector.decode(self.queries[position : position + 1], positions, rows)

    def oracle(self, start, stop):
        """The torch reference's read of queries [start, stop) over the buckets the cache holds now"""
        config = self.settings.indexer
        visible = self.cache.visible(ROW)
        rows = self.cache.logical_rows(ROW, visible)[:visible]
        buckets = Buckets(
            self.cache.bucket_centroids[rows].float().cpu(),
            self.cache.bucket_members[rows].cpu(),
            self.cache.bucket_sizes[rows].cpu(),
            config.window_size // config.bucket_size,
            config.window_size,
        )
        selected, _ = read(
            self.queries[start:stop].bfloat16().cpu(),
            torch.arange(start, stop, dtype=torch.int32),
            buckets,
            config.top_p,
            config.top_k,
            config.sink_size,
            config.window_size,
            early_budget=config.early_budget,
        )
        return selected

    def paired_on_the_gpu(self, key_sums, node_bucket_size, node_cluster_algorithm="ward"):
        """The reference's node buckets, paired by the kernel the selector's growth runs"""
        labels = hierarchy_module.pair_into_node_buckets(
            key_sums.to(DEVICE), node_bucket_size, self.settings.serving.kernels
        )
        return labels.cpu()

    def tree_oracle(self, positions):
        """The torch reference's beam read at `positions` over the buckets the cache holds now and their levels, the
        node centroids rounded to bf16 as the cache stores them"""
        config = self.settings.indexer
        visible = self.cache.visible(ROW)
        rows = self.cache.logical_rows(ROW, visible)[:visible]
        buckets = Buckets(
            self.cache.bucket_centroids[rows].float().cpu(),
            self.cache.bucket_members[rows].cpu(),
            self.cache.bucket_sizes[rows].cpu(),
            config.window_size // config.bucket_size,
            config.window_size,
        )
        levels = range(1, config.top_level + 1)
        # Two pairings in fp32 can break a tie at its last bit apart - a GPU's cheaper partner by 5e-7 at a cost of
        # 5.66 is the reference's dearer one
        # => the reference pairs with the selector's own kernel, which the pairing's own tests hold to the reference
        with mock.patch.object(core_hierarchy, "pair_into_node_buckets", self.paired_on_the_gpu):
            hierarchy = build_hierarchy(
                self.keys.cpu(),
                buckets,
                config.top_level,
                config.at("node_bucket_size", 1),
                [config.at("node_cluster_block_size", level) for level in levels],
            )
        hierarchy = hierarchy._replace(
            node_centroids=[centroids.bfloat16().float() for centroids in hierarchy.node_centroids]
        )
        selected, _ = read_hierarchy(
            self.queries[positions.long()].bfloat16().cpu(),
            positions.cpu(),
            buckets,
            hierarchy,
            [config.at("beam_width", level) for level in levels],
            config.start_nodes,
            config.top_p,
            config.top_k,
            config.sink_size,
            config.window_size,
            PAGE_SIZE,
            hierarchy_module.GROWTH_BUCKETS,
        )
        return selected


def sets(lists):
    """Each row's ids as a sorted list, -1 padding dropped"""
    return [sorted(row[row >= 0].tolist()) for row in lists.cpu()]


def test_a_flat_read_is_the_torch_references_read():
    store = Store(FLAT, 2 * PAGE_SIZE + 64)
    store.catch_up(2 * PAGE_SIZE)
    lists = store.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    assert lists.shape == (64, WIDTH)
    assert lists.dtype == torch.int32
    assert sets(lists) == sets(store.oracle(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64))


def test_a_prefill_across_a_page_end_reads_each_query_as_the_reference_does():
    # The segment is cut at the page, the page's buckets are all closed by the next, and a flat cache never moves a row
    store = Store(FLAT, 2 * PAGE_SIZE + 400, seed=1)
    store.catch_up(PAGE_SIZE - 300)
    lists = store.prefill(PAGE_SIZE - 300, PAGE_SIZE + 300)
    assert sets(lists) == sets(store.oracle(PAGE_SIZE - 300, PAGE_SIZE + 300))


def test_chunks_of_any_size_read_the_same_rows():
    store = Store(FLAT, 2 * PAGE_SIZE, seed=2)
    whole = store.prefill(0, 3000)
    chunked = Store(FLAT, 2 * PAGE_SIZE, seed=2)
    pieces = torch.cat([chunked.prefill(start, min(start + 700, 3000)) for start in range(0, 3000, 700)])
    assert sets(whole) == sets(pieces)


def test_a_flat_read_without_pools_is_the_torch_references_read():
    store = Store({**FLAT, "pool_size": None}, PAGE_SIZE + 64, seed=3)
    store.catch_up(PAGE_SIZE)
    lists = store.prefill(PAGE_SIZE, PAGE_SIZE + 64)
    assert sets(lists) == sets(store.oracle(PAGE_SIZE, PAGE_SIZE + 64))


def test_the_fused_select_reads_what_the_dense_product_reads():
    # invariant 22 - at a gate of zero every row takes the fused select, and the early rows the overflow's pick
    dense = Store(FLAT, 2 * PAGE_SIZE + 64, seed=4)
    dense.catch_up(2 * PAGE_SIZE)
    want = dense.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    fused = Store(FLAT, 2 * PAGE_SIZE + 64, seed=4)
    original = selector_module.SCAN_SELECT_BUCKETS
    selector_module.SCAN_SELECT_BUCKETS = 0
    try:
        early = fused.prefill(0, 300)
        fused.catch_up(2 * PAGE_SIZE)
        got = fused.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    finally:
        selector_module.SCAN_SELECT_BUCKETS = original
    assert sets(got) == sets(want)
    assert early.shape == (300, WIDTH)


def test_every_selected_id_is_below_the_query_and_none_is_read_twice():
    # invariants 17 and 18
    store = Store(FLAT, PAGE_SIZE + 3000, seed=5)
    lists = store.prefill(0, PAGE_SIZE + 3000)
    positions = torch.arange(PAGE_SIZE + 3000, device=DEVICE)
    valid = lists >= 0
    assert bool((lists <= positions[:, None]).all())
    assert bool(((lists == -1) | (lists >= 0)).all())
    for row in (10, 600, 5000, PAGE_SIZE + 2999):
        ids = lists[row][valid[row]].tolist()
        assert len(ids) == len(set(ids))
        window_start = max(0, row + 1 - WINDOW)
        sinks = {i for i in ids if i < 4}
        window = {i for i in ids if i >= window_start}
        longrange = [i for i in ids if i not in sinks and i not in window]
        assert all(4 <= i < window_start for i in longrange)
        assert window == set(range(window_start, row + 1))


def test_a_read_holds_its_full_budget_once_that_many_keys_are_visible():
    # invariant 19
    store = Store(FLAT, PAGE_SIZE, seed=6)
    lists = store.prefill(0, PAGE_SIZE)
    counts = (lists >= 0).sum(1)
    # Early on the window holds every key before the query, later the budget of top_k + window + sinks is full
    assert counts[:WINDOW].tolist() == list(range(1, WINDOW + 1))
    assert bool((counts[4000:] == WIDTH).all())


def test_with_every_bucket_picked_and_the_top_k_above_the_context_nothing_is_missing():
    # invariants 20 and 28 - the selection, the window and the sink cover every causal position
    indexer = {**FLAT, "top_p": 1024, "top_k": PAGE_SIZE}
    store = Store(indexer, PAGE_SIZE, seed=7)
    lists = store.prefill(0, PAGE_SIZE)
    for position in (3, 70, 500, 4096, PAGE_SIZE - 1):
        ids = lists[position][lists[position] >= 0].tolist()
        assert sorted(ids) == list(range(position + 1))


def test_a_query_alone_reads_what_it_reads_among_a_chunk():
    # invariant 23
    store = Store(FLAT, PAGE_SIZE + 64, seed=8)
    store.catch_up(PAGE_SIZE)
    cache = store.cache
    positions = torch.arange(PAGE_SIZE, PAGE_SIZE + 32, device=DEVICE, dtype=torch.int32)
    queries = store.queries[PAGE_SIZE : PAGE_SIZE + 32]
    rows = torch.full((32,), ROW, device=DEVICE, dtype=torch.int64)
    counts = cache.query_lengths(positions)
    chunk = store.selector.read_flat(ROW, queries, positions, counts, rows)
    for i in range(0, 32, 7):
        one = slice(i, i + 1)
        alone = store.selector.read_flat(ROW, queries[one], positions[one], counts[one], rows[one])
        assert sets(alone) == sets(chunk[one])


def test_a_decode_step_selects_what_that_position_selected_in_prefill():
    # invariant 24
    store = Store(FLAT, PAGE_SIZE + 64, seed=9)
    position = PAGE_SIZE
    store.catch_up(position)
    decoded = store.decode(position)
    prefilled = store.prefill(position, position + 1)
    assert decoded.shape == (1, WIDTH)
    assert sets(decoded) == sets(prefilled)


def test_a_decode_step_captured_in_a_cuda_graph_replays_the_eager_read_and_repeats():
    # invariant 30
    store = Store(FLAT, PAGE_SIZE + 64, seed=9)
    position = PAGE_SIZE
    store.catch_up(position)
    rows = torch.tensor([ROW], device=DEVICE)
    positions = torch.tensor([position], device=DEVICE, dtype=torch.int32)
    queries = store.queries[position : position + 1].clone()
    eager = store.selector.decode(queries, positions, rows).clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        store.selector.decode(queries, positions, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = store.selector.decode(queries, positions, rows)
    for _ in range(3):
        replayed.zero_()
        graph.replay()
        assert torch.equal(replayed, eager)


def decode_in_range(store, limit, monkeypatch, position, rows=(ROW,)):
    """One decode step as the graph for contexts up to `limit` reads it, and the score columns it sized to"""
    columns = []
    scan = selector_module.scan_paged

    def spy(*args):
        columns.append(args[-1])
        return scan(*args)

    monkeypatch.setattr(selector_module, "scan_paged", spy)
    monkeypatch.setattr(context_buckets, "captured_limit", lambda: limit)
    request_rows = torch.tensor(list(rows), device=DEVICE)
    positions = torch.tensor([position] * len(rows), device=DEVICE, dtype=torch.int32)
    lists = store.selector.decode(
        store.queries[position : position + 1].expand(len(rows), -1), positions, request_rows
    )
    return lists, columns[0]


@pytest.mark.parametrize("keys", [2 * PAGE_SIZE - 513, 2 * PAGE_SIZE - 1, 2 * PAGE_SIZE])
def test_a_range_graph_reads_what_the_full_graph_reads_up_to_its_limit(monkeypatch, keys):
    # The server holds 64 pages of context, the range graph 2
    store = Store(FLAT, 2 * PAGE_SIZE + 1, seed=11, max_context=64 * PAGE_SIZE)
    position = keys - 1
    store.catch_up(position)
    full, full_columns = decode_in_range(store, None, monkeypatch, position)
    ranged, ranged_columns = decode_in_range(store, 2 * PAGE_SIZE, monkeypatch, position)
    assert full_columns == 64 * store.cache.rows_per_page
    assert ranged_columns == 2 * store.cache.rows_per_page
    assert torch.equal(ranged, full)


@pytest.mark.parametrize("keys", [DECODE_CONTEXT_LIMITS[0] - 1, DECODE_CONTEXT_LIMITS[0]])
def test_the_default_first_range_reads_what_the_full_graph_reads_at_its_limit(monkeypatch, keys):
    limit = DECODE_CONTEXT_LIMITS[0]
    store = Store(FLAT, limit + 1, seed=13, max_context=DECODE_CONTEXT_LIMITS[-1] * 2)
    store.catch_up(keys - 1)
    full, _ = decode_in_range(store, None, monkeypatch, keys - 1)
    ranged, columns = decode_in_range(store, limit, monkeypatch, keys - 1)
    assert columns == limit // store.settings.indexer.bucket_size
    assert torch.equal(ranged, full)


def test_a_range_graph_with_a_padded_row_past_its_limit_reads_the_rows_within_it_as_the_full_graph_does(
    monkeypatch,
):
    # A replay pads the batch with a row that may hold a longer request than the range
    store = Store(FLAT, 3 * PAGE_SIZE, seed=12, max_context=64 * PAGE_SIZE)
    store.catch_up(PAGE_SIZE + 640)
    store.selector.cache.ensure(2, "long", 2)
    store.selector.cache.advance(2, 3 * PAGE_SIZE - 1, store.index_keys_of(2), {1, 2})
    position = PAGE_SIZE + 640
    full, _ = decode_in_range(store, None, monkeypatch, position, rows=(ROW, 2))
    ranged, columns = decode_in_range(store, 2 * PAGE_SIZE, monkeypatch, position, rows=(ROW, 2))
    assert columns == 2 * store.cache.rows_per_page
    assert torch.equal(ranged[0], full[0])


def test_a_range_graph_captured_in_a_cuda_graph_replays_the_full_graphs_read(monkeypatch):
    store = Store(FLAT, PAGE_SIZE + 64, seed=9, max_context=64 * PAGE_SIZE)
    position = PAGE_SIZE
    store.catch_up(position)
    full, _ = decode_in_range(store, None, monkeypatch, position)
    full = full.clone()
    rows = torch.tensor([ROW], device=DEVICE)
    positions = torch.tensor([position], device=DEVICE, dtype=torch.int32)
    queries = store.queries[position : position + 1].clone()
    monkeypatch.setattr(context_buckets, "captured_limit", lambda: 2 * PAGE_SIZE)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        store.selector.decode(queries, positions, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = store.selector.decode(queries, positions, rows)
    for _ in range(3):
        replayed.zero_()
        graph.replay()
        assert torch.equal(replayed, full)


def test_decoding_requests_of_different_lengths_in_one_batch_reads_each_alone():
    # invariant 27
    store = Store(FLAT, 2 * PAGE_SIZE, seed=10)
    store.catch_up(PAGE_SIZE + 640)
    store.selector.cache.ensure(2, "short", 2)
    store.selector.cache.advance(2, 2000, store.index_keys_of(2), {1, 2})
    rows = torch.tensor([1, 2], device=DEVICE)
    positions = torch.tensor([PAGE_SIZE + 640, 2000], device=DEVICE, dtype=torch.int32)
    queries = store.queries[torch.tensor([PAGE_SIZE + 640, 2000], device=DEVICE)]
    batch = store.selector.decode(queries, positions, rows)
    for i in range(2):
        alone = store.selector.decode(queries[i : i + 1], positions[i : i + 1], rows[i : i + 1])
        assert sets(batch[i : i + 1]) == sets(alone)


def decode_batch(store):
    rows = torch.tensor([1, 2], device=DEVICE)
    positions = torch.tensor([PAGE_SIZE + 640, 2000], device=DEVICE, dtype=torch.int32)
    return store.queries[positions.long()], positions, rows


def store_of_two(kernels, indexer=FLAT):
    store = Store(indexer, 2 * PAGE_SIZE, {"kernels": kernels}, seed=10)
    store.catch_up(PAGE_SIZE + 640)
    store.selector.cache.ensure(2, "short", 2)
    store.selector.cache.advance(2, 2000, store.index_keys_of(2), {1, 2})
    return store


def test_a_decode_step_reads_the_same_in_either_kernel_language(monkeypatch):
    listings = []
    listed = selector_module.cutedsl_decode.DecodeSelect.listed
    monkeypatch.setattr(
        selector_module.cutedsl_decode.DecodeSelect,
        "listed",
        lambda select, *arguments: listings.append(1) or listed(select, *arguments),
    )
    reads = {}
    for kernels in ("triton", "cutedsl"):
        store = store_of_two(kernels)
        reads[kernels] = sets(store.selector.decode(*decode_batch(store)))
    assert listings == [1]
    assert reads["cutedsl"] == reads["triton"]


def test_a_short_prefill_reads_the_same_in_either_kernel_language(monkeypatch):
    listings = []
    list_rows = selector_module.cutedsl_top_k.list_rows
    monkeypatch.setattr(
        selector_module.cutedsl_top_k,
        "list_rows",
        lambda *arguments: listings.append(1) or list_rows(*arguments),
    )
    reads = {}
    for kernels in ("triton", "cutedsl"):
        store = Store(FLAT, 2 * PAGE_SIZE + 64, {"kernels": kernels}, seed=10)
        store.catch_up(2 * PAGE_SIZE)
        reads[kernels] = sets(store.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64))
    assert listings == [1]
    assert reads["cutedsl"] == reads["triton"]


EARLY = {"position": 4000, "top_k": 128, "top_p": 32}


def test_a_prefill_across_the_early_position_reads_each_query_as_the_reference_does():
    # The segment straddles the position, so one chunk holds queries of both budgets
    store = Store({**FLAT, "early_budget": {**EARLY, "position": PAGE_SIZE}}, 2 * PAGE_SIZE + 400, seed=1)
    store.catch_up(PAGE_SIZE - 300)
    lists = store.prefill(PAGE_SIZE - 300, PAGE_SIZE + 300)
    assert lists.shape == (600, WIDTH)
    assert sets(lists) == sets(store.oracle(PAGE_SIZE - 300, PAGE_SIZE + 300))
    lengths = (lists >= 0).sum(1)
    assert lengths[:300].max() < lengths[300:].max()


@pytest.mark.parametrize("kernels", ["triton", "cutedsl"])
def test_a_decode_batch_across_the_early_position_reads_each_row_at_its_own_budget(kernels):
    # Row 2 sits at 2000, before the position, and row 1 at PAGE_SIZE + 640, after it
    store = store_of_two(kernels, {**FLAT, "early_budget": EARLY})
    queries, positions, rows = decode_batch(store)
    mixed = store.selector.decode(queries, positions, rows)
    assert mixed.shape == (2, WIDTH)
    whole = store_of_two(kernels, FLAT).selector.decode(queries, positions, rows)
    early = store_of_two(kernels, {**FLAT, "top_k": EARLY["top_k"], "top_p": EARLY["top_p"]})
    small = early.selector.decode(queries, positions, rows)
    assert sets(mixed[:1]) == sets(whole[:1])
    assert sets(mixed[1:]) == sets(small[1:])
    assert sets(mixed[1:]) != sets(whole[1:])


@pytest.mark.parametrize("kernels", ["triton", "cutedsl"])
def test_a_decode_batch_across_the_early_position_captured_in_a_cuda_graph_replays_the_eager_read(kernels):
    store = store_of_two(kernels, {**FLAT, "early_budget": EARLY})
    queries, positions, rows = decode_batch(store)
    eager = store.selector.decode(queries, positions, rows).clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        store.selector.decode(queries, positions, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = store.selector.decode(queries, positions, rows)
    for _ in range(3):
        replayed.zero_()
        graph.replay()
        assert torch.equal(replayed, eager)


def test_a_cutedsl_decode_step_captured_in_a_cuda_graph_replays_the_eager_read():
    store = store_of_two("cutedsl")
    queries, positions, rows = decode_batch(store)
    eager = store.selector.decode(queries, positions, rows).clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        store.selector.decode(queries, positions, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = store.selector.decode(queries, positions, rows)
    for _ in range(3):
        replayed.zero_()
        graph.replay()
        assert torch.equal(replayed, eager)


@pytest.mark.parametrize("indexer", [FIXED, DEEP, TOP_START], ids=["fixed", "tree", "from the top"])
def test_a_beam_wide_enough_for_every_node_reads_what_the_flat_scan_reads(indexer):
    # invariant 13 - four pages grow the levels, and the next block's queries see every closed bucket
    end = 4 * PAGE_SIZE + 5 * 64
    store = Store(indexer, end + 64, seed=11)
    store.catch_up(end)
    hierarchy = store.cache.hierarchies[ROW]
    assert hierarchy.levels == (1 if indexer is FIXED else 4)
    assert hierarchy.covered_buckets == 4 * PAGE_SIZE // 8
    positions = torch.arange(end, end + 64, device=DEVICE, dtype=torch.int32)
    queries = store.queries[end : end + 64]
    rows = torch.full((64,), ROW, device=DEVICE, dtype=torch.int64)
    counts = store.cache.query_lengths(positions)
    flat = store.selector.read_flat(ROW, queries, positions, counts, rows)
    beam = store.selector.expand_bucket_keys(
        hierarchy.beam_search(
            store.cache, ROW, queries, counts, store.selector.beam_top_k, store.selector.tree_top_k
        ),
        positions,
        rows,
    )
    assert sets(beam) == sets(flat)


@pytest.mark.parametrize("indexer", [FIXED, DEEP, TOP_START], ids=["fixed", "tree", "from the top"])
def test_a_prefill_read_after_growth_routes_through_the_beam(indexer):
    end = 4 * PAGE_SIZE + 5 * 64
    store = Store(indexer, end + 64, seed=12)
    store.catch_up(end)
    lists = store.prefill(end, end + 64)
    flat = Store(FLAT, end + 64, seed=12)
    flat.catch_up(end)
    # The same keys, the same buckets - the hierarchy moves no rows - and a beam as wide as a level
    assert sets(lists) == sets(flat.prefill(end, end + 64))


def test_a_level_started_whole_reads_what_the_flat_scan_reads_too():
    # invariant 14 - the start level scored whole, as though the level above had kept every node
    end = 4 * PAGE_SIZE + 5 * 64
    store = Store({**DEEP, "start_nodes": 300}, end + 64, seed=13)
    store.catch_up(end)
    assert store.cache.hierarchies[ROW].start_level() == 2
    flat = Store(FLAT, end + 64, seed=13)
    flat.catch_up(end)
    assert sets(store.prefill(end, end + 64)) == sets(flat.prefill(end, end + 64))


def test_a_narrow_beam_fills_its_budget_but_reads_what_the_flat_scan_would_not():
    end = 4 * PAGE_SIZE + 5 * 64
    narrow = Store({**FIXED, "beam_width": 64}, end + 64, seed=14)
    narrow.catch_up(end)
    wide = Store(FIXED, end + 64, seed=14)
    wide.catch_up(end)
    got, want = sets(narrow.prefill(end, end + 64)), sets(wide.prefill(end, end + 64))
    assert got != want
    assert all(len(ids) == WIDTH for ids in got)


def test_growing_a_level_changes_no_earlier_querys_read():
    # invariant 15 - the fourth page's growth comes after the page's last read, so the reads are the flat ones
    grown = Store(FIXED, 4 * PAGE_SIZE, seed=15)
    reads = grown.prefill(0, 4 * PAGE_SIZE)
    assert grown.cache.hierarchies[ROW].levels == 1
    flat = Store(FLAT, 4 * PAGE_SIZE, seed=15)
    assert sets(reads) == sets(flat.prefill(0, 4 * PAGE_SIZE))


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
@pytest.mark.parametrize("indexer", [FIXED, DEEP, TOP_START], ids=["fixed", "tree", "from the top"])
def test_a_tree_reads_the_same_buckets_in_either_kernel_language(indexer, precision, monkeypatch):
    end = 4 * PAGE_SIZE + 5 * 64
    calls = {"beam_step": [], "keep_start_level": [], "pick_top_p": []}
    for name, launched in calls.items():
        fused = getattr(hierarchy_module.cutedsl_beam, name)
        monkeypatch.setattr(
            hierarchy_module.cutedsl_beam,
            name,
            lambda *arguments, fused=fused, launched=launched: launched.append(1) or fused(*arguments),
        )
    reads = {}
    for kernels in ("triton", "cutedsl"):
        store = Store(indexer, end + 64, {"node_centroid_precision": precision, "kernels": kernels}, seed=17)
        store.catch_up(end)
        reads[kernels] = sets(store.prefill(end, end + 64))
    # Every beam keeps its top level and picks its buckets in CuTeDSL, and one started from the top carries children
    # down through a level that scores them - a keep launches as a level that scores none
    assert (len(calls["beam_step"]) > len(calls["keep_start_level"])) == (indexer is TOP_START)
    assert calls["keep_start_level"]
    assert calls["pick_top_p"]
    assert reads["cutedsl"] == reads["triton"]


def test_the_beam_picks_only_buckets_a_query_sees():
    end = 4 * PAGE_SIZE + 5 * 64
    store = Store(DEEP, end + 128, seed=16)
    store.catch_up(end)
    hierarchy = store.cache.hierarchies[ROW]
    covered = hierarchy.covered_buckets
    queries = store.queries[end : end + 4]
    counts = torch.tensor([covered, covered, covered + 8, covered + 16], device=DEVICE, dtype=torch.int32)
    keys = hierarchy.beam_search(
        store.cache, ROW, queries, counts, store.selector.beam_top_k, store.selector.tree_top_k
    )
    rows = unpack_ids(keys)
    logical = {
        row: index
        for index, row in enumerate(store.cache.logical_rows(ROW, store.cache.visible(ROW)).tolist())
    }
    for query in range(4):
        picked = [logical[row] for row in rows[query].tolist() if row >= 0]
        assert picked
        assert max(picked) < int(counts[query])


@pytest.mark.parametrize("indexer", [FIXED, DEEP, TOP_START], ids=["fixed", "tree", "from the top"])
def test_a_tree_decode_reads_what_a_prefill_query_at_that_position_reads(indexer):
    # invariant 25
    position = 4 * PAGE_SIZE + 5 * 64
    tree = Store(indexer, position + 64, {"decode": "descend", "node_centroid_precision": "bf16"}, seed=17)
    tree.catch_up(position)
    decoded = tree.decode(position)
    prefilled = tree.prefill(position, position + 1)
    assert sets(decoded) == sets(prefilled)


def test_a_tree_decode_before_any_level_scans_the_buckets_as_a_flat_decode_does():
    position = 2 * PAGE_SIZE + 128
    tree = Store(FIXED, position + 64, {"decode": "descend"}, seed=18)
    tree.catch_up(position)
    assert tree.cache.hierarchies[ROW].levels == 0
    flat = Store(FLAT, position + 64, seed=18)
    flat.catch_up(position)
    assert sets(tree.decode(position)) == sets(flat.decode(position))


def test_a_tree_decode_of_shaped_node_buckets_refuses_to_run_inside_a_cuda_graph_capture(monkeypatch):
    # Shaped node buckets are not mirrored, so such a design has no graph tree to read
    indexer = {**FIXED, "node_bucket_shape": SHAPE}
    store = Store(indexer, PAGE_SIZE + 64, {"decode": "descend"}, seed=19)
    store.catch_up(PAGE_SIZE)
    assert store.cache.graph_tree is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eagerly"):
        store.decode(PAGE_SIZE)


CORE_FIXED = {
    **FLAT,
    "top_level": 1,
    "node_bucket_size": 4,
    "node_cluster_block_size": 1024,
    "beam_width": 2048,
}
CORE_TREE = {
    **FLAT,
    "top_level": 16,
    "node_bucket_size": 4,
    "node_cluster_block_size": 4096,
    "beam_width": 2048,
    "start_nodes": 32768,
}


@pytest.mark.parametrize(
    "indexer",
    [
        CORE_FIXED,
        CORE_TREE,
    ],
    ids=["fixed", "tree"],
)
def test_a_read_just_past_a_growth_holds_its_full_budget(indexer):
    # invariant 19 - each of seven queries wants the bucket of a different key of the last closed block, which its
    # window partly covers, and a bucket is judged by its own keys' block
    end = 4 * PAGE_SIZE
    store = Store(indexer, end + 64, seed=22)
    store.catch_up(end)
    assert store.cache.hierarchies[ROW].covered_buckets == end // 8
    for query in range(7):
        store.queries[end + query] = store.keys[end - 1 - 8 * query].float()
    lists = store.prefill(end, end + 64)
    assert bool(((lists[:7] >= 0).sum(1) == WIDTH).all())


def test_a_fp8_hierarchy_reads_what_a_bf16_one_reads_when_the_beam_keeps_every_node():
    end = 4 * PAGE_SIZE + 5 * 64
    fp8 = Store(TOP_START, end + 64, {"node_centroid_precision": "fp8"}, seed=20)
    fp8.catch_up(end)
    bf16 = Store(TOP_START, end + 64, {"node_centroid_precision": "bf16"}, seed=20)
    bf16.catch_up(end)
    assert fp8.cache.hierarchies[ROW].open_node_buckets[0]["node_centroids"].dtype == torch.float8_e4m3fn
    assert sets(fp8.prefill(end, end + 64)) == sets(bf16.prefill(end, end + 64))


def test_a_fused_beam_keeps_the_nodes_the_top_k_keeps():
    end = 4 * PAGE_SIZE + 5 * 64
    plain = Store(TOP_START, end + 64, seed=21)
    plain.catch_up(end)
    want = plain.prefill(end, end + 64)
    fused = Store(TOP_START, end + 64, seed=21)
    fused.catch_up(end)
    original = hierarchy_module.SCAN_SELECT_BEAM_WIDTH
    hierarchy_module.SCAN_SELECT_BEAM_WIDTH = 0
    try:
        got = fused.prefill(end, end + 64)
    finally:
        hierarchy_module.SCAN_SELECT_BEAM_WIDTH = original
    assert sets(got) == sets(want)


NARROW = {"node_bucket_size": 4, "node_cluster_block_size": 64, "beam_width": 64}


@pytest.mark.parametrize(
    "indexer",
    [
        {**FLAT, "top_level": 1, **NARROW},
        {**FLAT, "top_level": 16, **NARROW, "start_nodes": 32768},
        {**FLAT, "top_level": 16, **NARROW, "start_nodes": 300},
        {**FLAT, "top_level": 16, **NARROW},
    ],
    ids=["fixed", "tree", "started at level 2", "from the top"],
)
def test_a_narrow_beam_reads_what_the_torch_reference_reads(indexer):
    # Five pages read a page at a time, so the levels grow at the fourth page end between two queries
    total = 5 * PAGE_SIZE + 300
    store = Store(indexer, total, seed=31)
    whole = torch.cat(
        [store.prefill(start, min(start + PAGE_SIZE, total)) for start in range(0, total, PAGE_SIZE)]
    )
    generator = torch.Generator().manual_seed(32)
    drawn = torch.randperm(total - 100, generator=generator)[:300] + 100
    around_growth = torch.arange(4 * PAGE_SIZE - 4, 4 * PAGE_SIZE + 4)
    positions = torch.cat([drawn, around_growth]).unique().to(device=DEVICE, dtype=torch.int32)
    assert store.cache.hierarchies[ROW].levels >= 1
    assert sets(whole[positions.long()]) == sets(store.tree_oracle(positions))


@pytest.mark.parametrize(
    "indexer",
    [FLAT, {**FLAT, "top_level": 1, **NARROW}, {**FLAT, "top_level": 16, **NARROW, "start_nodes": 300}],
    ids=["flat", "fixed", "tree"],
)
def test_sampled_queries_read_what_the_whole_prefill_reads_at_them(indexer):
    total = 5 * PAGE_SIZE + 300
    store = Store(indexer, total, seed=3)
    pages = range(0, total, PAGE_SIZE)
    whole = torch.cat([store.prefill(start, min(start + PAGE_SIZE, total)) for start in pages])
    generator = torch.Generator().manual_seed(4)
    positions = torch.randperm(total - 100, generator=generator)[:400].sort().values + 100
    positions = positions.to(device=DEVICE, dtype=torch.int32)
    queries = store.queries[positions.long()]
    sampled = selector_module.read_sampled(store.settings, queries, store.keys, positions)
    assert sets(sampled) == sets(whole[positions.long()])


def test_a_query_a_rank_reads_each_ranks_list_as_its_own_query_would():
    total = 3 * PAGE_SIZE + 100
    store = Store(FLAT, total, seed=5)
    positions = torch.arange(200, total, 97, device=DEVICE, dtype=torch.int32)
    first = store.queries[positions.long()]
    second = store.queries[(positions - 150).long()]
    ranked = selector_module.read_sampled(
        store.settings, torch.stack([first, second], 1), store.keys, positions
    )
    assert ranked.shape[:2] == (len(positions), 2)
    for rank, queries in enumerate((first, second)):
        alone = selector_module.read_sampled(store.settings, queries, store.keys, positions)
        assert sets(ranked[:, rank]) == sets(alone)


SHAPE = {"rank": 2, "deviations": 2.0}
SHAPED_FLAT = {**FLAT, "bucket_shape": SHAPE}


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_a_node_shaped_tree_reads_the_same_buckets_in_either_kernel_language(precision, monkeypatch):
    end = 4 * PAGE_SIZE + 5 * 64
    levels = []
    beam_step = hierarchy_module.cutedsl_beam.beam_step
    monkeypatch.setattr(
        hierarchy_module.cutedsl_beam,
        "beam_step",
        lambda *arguments: levels.append(arguments[-1] is not None) or beam_step(*arguments),
    )
    reads = {}
    for kernels in ("triton", "cutedsl"):
        store = Store(
            {**TOP_START, "node_bucket_shape": SHAPE},
            end + 64,
            {"node_centroid_precision": precision, "kernels": kernels},
            seed=17,
        )
        store.catch_up(end)
        reads[kernels] = sets(store.prefill(end, end + 64))
    # A level that carries children scores their shapes in CuTeDSL
    assert any(levels)
    assert reads["cutedsl"] == reads["triton"]


def shaped_oracle(store, start, stop):
    """The torch reference's shaped read of queries [start, stop) over the buckets and shapes the cache holds now"""
    config = store.settings.indexer
    visible = store.cache.visible(ROW)
    rows = store.cache.logical_rows(ROW, visible)[:visible]
    buckets = Buckets(
        store.cache.bucket_centroids[rows].float().cpu(),
        store.cache.bucket_members[rows].cpu(),
        store.cache.bucket_sizes[rows].cpu(),
        config.window_size // config.bucket_size,
        config.window_size,
    )
    table = store.cache.bucket_shapes.rows(rows)
    shapes = Shapes(unpack_shape_basis(table.basis).cpu(), table.variances.cpu(), table.floor.cpu())
    selected, _ = read(
        store.queries[start:stop].bfloat16().cpu(),
        torch.arange(start, stop, dtype=torch.int32),
        buckets,
        config.top_p,
        config.top_k,
        config.sink_size,
        config.window_size,
        shapes,
        table.deviations,
    )
    return selected


def overlap(got, want):
    """The share of each row's ids the two reads hold in common"""
    return [
        len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
        for a, b in zip(sets(got), sets(want), strict=True)
    ]


def test_a_shaped_flat_read_is_the_references_shaped_read_to_the_queries_planes():
    store = Store(SHAPED_FLAT, 2 * PAGE_SIZE + 64, seed=21)
    store.catch_up(2 * PAGE_SIZE)
    lists = store.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    shared = overlap(lists, shaped_oracle(store, 2 * PAGE_SIZE, 2 * PAGE_SIZE + 64))
    plain = Store(FLAT, 2 * PAGE_SIZE + 64, seed=21)
    plain.catch_up(2 * PAGE_SIZE)
    unshaped = overlap(lists, plain.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64))
    # The planes move a bucket at the cut at most, where the shape moves many - one of up to 16 keys in, as many out
    assert min(shared) > 1 - 2 * FLAT["bucket_cap"] / WIDTH and sum(shared) / len(shared) > 0.995
    assert sum(unshaped) / len(unshaped) < 0.95


def test_a_shaped_cache_holds_each_buckets_shape_from_its_keys():
    store = Store(SHAPED_FLAT, PAGE_SIZE, seed=22)
    store.catch_up(PAGE_SIZE)
    visible = store.cache.visible(ROW)
    rows = store.cache.logical_rows(ROW, visible)[:visible]
    members = store.cache.bucket_members[rows]
    centroids = F.normalize(
        (store.keys.float()[members.clamp_min(0).long()] * (members >= 0)[..., None]).sum(1), dim=-1
    )
    want = reference_build_shapes(store.keys, centroids, members, 2)
    table = store.cache.bucket_shapes.rows(rows)
    torch.testing.assert_close(table.variances, want.variances, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(table.floor, want.floor, rtol=1e-3, atol=1e-6)
    assert table.basis.shape == (visible, 2, 8)
    # 16 bytes of signs, 8 of variances and 4 of floor a bucket
    per_bucket = sum(part[0].numel() * part.element_size() for part in store.cache.bucket_shapes[:3])
    assert per_bucket == 28


def test_the_shaped_fused_select_reads_what_the_shaped_dense_product_reads():
    dense = Store(SHAPED_FLAT, 2 * PAGE_SIZE + 64, seed=23)
    dense.catch_up(2 * PAGE_SIZE)
    want = dense.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    fused = Store(SHAPED_FLAT, 2 * PAGE_SIZE + 64, seed=23)
    original = selector_module.SCAN_SELECT_BUCKETS
    selector_module.SCAN_SELECT_BUCKETS = 0
    try:
        fused.catch_up(2 * PAGE_SIZE)
        got = fused.prefill(2 * PAGE_SIZE, 2 * PAGE_SIZE + 64)
    finally:
        selector_module.SCAN_SELECT_BUCKETS = original
    assert sets(got) == sets(want)


def test_a_shaped_decode_step_selects_what_that_position_selected_in_prefill():
    store = Store(SHAPED_FLAT, PAGE_SIZE + 64, seed=24)
    position = PAGE_SIZE
    store.catch_up(position)
    assert sets(store.decode(position)) == sets(store.prefill(position, position + 1))


def test_a_shaped_decode_step_captured_in_a_cuda_graph_replays_the_eager_read():
    store = Store(SHAPED_FLAT, PAGE_SIZE + 64, seed=25)
    store.catch_up(PAGE_SIZE)
    rows = torch.tensor([ROW], device=DEVICE)
    positions = torch.tensor([PAGE_SIZE], device=DEVICE, dtype=torch.int32)
    queries = store.queries[PAGE_SIZE : PAGE_SIZE + 1].clone()
    eager = store.selector.decode(queries, positions, rows)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        store.selector.decode(queries, positions, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = store.selector.decode(queries, positions, rows)
    graph.replay()
    torch.cuda.synchronize()
    assert sets(captured) == sets(eager)


@pytest.mark.parametrize(
    ("indexer", "flat"),
    [
        ({**DEEP, "bucket_shape": SHAPE}, SHAPED_FLAT),
        ({**FIXED, "bucket_shape": SHAPE}, SHAPED_FLAT),
        ({**DEEP, "node_bucket_shape": SHAPE}, FLAT),
        ({**TOP_START, "node_bucket_shape": SHAPE, "bucket_shape": SHAPE}, SHAPED_FLAT),
    ],
    ids=["bucket shape", "fixed bucket shape", "node shape", "both from the top"],
)
def test_a_shaped_beam_wide_enough_for_every_node_reads_what_the_flat_scan_reads(indexer, flat):
    end = 4 * PAGE_SIZE + 5 * 64
    store = Store(indexer, end + 64, seed=26)
    store.catch_up(end)
    assert store.cache.hierarchies[ROW].levels >= 1
    lists = store.prefill(end, end + 64)
    reference = Store(flat, end + 64, seed=26)
    reference.catch_up(end)
    assert sets(lists) == sets(reference.prefill(end, end + 64))


def test_a_narrow_beam_over_node_shapes_reads_otherwise_than_over_centroids():
    end = 4 * PAGE_SIZE + 5 * 64
    narrow = {**DEEP, "beam_width": 32}
    shaped = Store({**narrow, "node_bucket_shape": SHAPE}, end + 64, seed=27)
    shaped.catch_up(end)
    plain = Store(narrow, end + 64, seed=27)
    plain.catch_up(end)
    got, want = sets(shaped.prefill(end, end + 64)), sets(plain.prefill(end, end + 64))
    assert got != want
    assert all(len(ids) == WIDTH for ids in got)


def test_a_shaped_tree_decode_reads_what_a_prefill_query_at_that_position_reads():
    end = 4 * PAGE_SIZE + 5 * 64
    indexer = {**DEEP, "node_bucket_shape": SHAPE, "bucket_shape": SHAPE, "beam_width": 64}
    store = Store(indexer, end + 64, {"decode": "descend"}, seed=28)
    store.catch_up(end)
    assert sets(store.decode(end)) == sets(store.prefill(end, end + 1))
