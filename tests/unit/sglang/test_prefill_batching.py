"""A prefill that reads a chunk's pages together selects what reading them a page at a time did.

The oracle is the loop `Selector.prefill` ran before it was batched - append a run of keys up to the page's end, read
the queries inside it, settle at a page end - run on a second selector over the same keys and queries. The lists must
hold the same ids and the caches the same buckets, pages and hierarchies, over chunk sizes and alignments, chunks
smaller than a page, several requests, and under prefill context parallel, where a rank queries a span of a chunk and
indexes all of it. Run with the fused select's gate and the hierarchy's growth lowered, so the routes the production
sizes reach only in the later pages are all reached in a few.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index import selector as selector_module
from lsa.sglang.index.layer_index import KeyLookup, Shard
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
NODES = {"node_bucket_size": 4, "node_cluster_block_size": 64, "beam_width": 2048}
FIXED = {**FLAT, "top_level": 1, **NODES}
DEEP = {**FLAT, "top_level": 16, **NODES, "start_nodes": 32768}
SHAPED = {**FLAT, "bucket_shape": {"rank": 2, "deviations": 2.0}}
DESIGNS = {"flat": FLAT, "fixed": FIXED, "tree": DEEP, "shaped": SHAPED}
ROWS = (1, 2)
CHUNK = 8192
WIDTH = 512 + WINDOW + 4
# Two growths a page, and the fused select from the second page on
GROWTH_BUCKETS = 256
FUSED_GATE = 1500


def paged_prefill(
    selector, row, index_queries, positions, index_keys_of, active, new_keys=None, ragged_start=None
):
    """The prefill loop as it ran before the pages were read together - the oracle."""
    cache = selector.cache
    key_count = len(positions) if new_keys is None else new_keys
    query_first = int(positions[0]) if len(positions) else 0
    ragged = ragged_start is not None
    lists, done = [], 0
    while done < key_count:
        end = cache.end[row]
        take = min(key_count - done, PAGE_SIZE - end % PAGE_SIZE)
        cache.advance(row, end + take, index_keys_of, active, settle=False)
        first = max(end, query_first) - query_first
        stop = min(end + take, query_first + len(positions)) - query_first
        if first < stop:
            segment = slice(first, stop)
            lists.append(selector.read_segment(row, index_queries[segment], positions[segment], ragged))
        if cache.end[row] % PAGE_SIZE == 0:
            cache.settle(row, index_keys_of)
        done += take
    read = torch.cat(lists) if lists else positions.new_empty((0, selector.width))
    if ragged_start is None:
        return read
    return torch.where(read >= 0, read + ragged_start, -1).to(torch.int32)


class Pair:
    """Two selectors over the same keys, one read the old way and one the new, a request a row of keys"""

    def __init__(self, indexer, total, seed=0, serving=None):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        anchors = torch.randn(total // 12 + 1, 64, device=DEVICE, generator=generator)
        keys = []
        for _ in ROWS:
            noise = 0.5 * torch.randn(total, 64, device=DEVICE, generator=generator)
            keys.append(F.normalize(anchors[torch.arange(total, device=DEVICE) // 12] + noise, dim=-1))
        self.keys = torch.stack(keys).to(torch.bfloat16)
        earlier = (
            torch.rand(total, device=DEVICE, generator=generator) * torch.arange(total, device=DEVICE)
        ).long()
        noisy = self.keys[0][earlier].float() + 0.1 * torch.randn(
            total, 64, device=DEVICE, generator=generator
        )
        self.queries = F.normalize(noisy, dim=-1)
        self.table = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(3, -1).contiguous()
        settings = DesignSettings.from_blocks(indexer, serving or {})
        self.old, self.new = (
            Selector(settings, 2 * total, 3, total, CHUNK, DEVICE, lambda: self.table) for _ in range(2)
        )

    def keys_of(self, row):
        return lambda positions: self.keys[row - 1][positions.long()]

    def prefill(self, row, start, stop, span=None, ragged_start=None):
        """Read [start, stop) of a request both ways, from a rank's span [a, b) of it under context parallel"""
        first, last = (start, stop) if span is None else span
        positions = torch.arange(first, last, device=DEVICE, dtype=torch.int32)
        queries = self.queries[first:last]
        keys_of, active = self.keys_of(row), set(ROWS)
        for selector in (self.old, self.new):
            selector.cache.ensure(row, f"request {row}", 0)
        want = paged_prefill(self.old, row, queries, positions, keys_of, active, stop - start, ragged_start)
        got = self.new.prefill(row, queries, positions, keys_of, active, stop - start, ragged_start, first)
        return got, want

    def same_state(self, row):
        old, new = self.old.cache, self.new.cache
        for name in ("end", "closed_keys", "pages", "run", "free"):
            assert getattr(old, name) == getattr(new, name), name
        for name in (
            "bucket_centroids",
            "bucket_members",
            "bucket_sizes",
            "page_positions",
            "request_pages",
            "closed_keys_device",
        ):
            assert torch.equal(getattr(old, name), getattr(new, name)), name
        if old.bucket_shapes is not None:
            for name in ("basis", "variances", "floor"):
                assert torch.equal(getattr(old.bucket_shapes, name), getattr(new.bucket_shapes, name)), name
        if old.hierarchies is not None:
            assert old.hierarchies[row] == new.hierarchies[row]


def sets(lists):
    return [sorted(row[row >= 0].tolist()) for row in lists.cpu()]


@pytest.fixture(autouse=True)
def lowered_thresholds(monkeypatch):
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", GROWTH_BUCKETS)
    monkeypatch.setattr(selector_module, "SCAN_SELECT_BUCKETS", FUSED_GATE)


# Chunks of a request, as [start, stop) - aligned, ragged, inside a page, and across several
CHUNKINGS = {
    "aligned": [(0, PAGE_SIZE), (PAGE_SIZE, 3 * PAGE_SIZE), (3 * PAGE_SIZE, 4 * PAGE_SIZE)],
    "unaligned": [(0, 3000), (3000, 3000 + PAGE_SIZE + 1500), (4500 + PAGE_SIZE, 3 * PAGE_SIZE + 700)],
    "inside a page": [(0, 100), (100, 164), (164, 5000), (5000, 5001), (5001, PAGE_SIZE + 17)],
    "across pages": [
        (0, PAGE_SIZE - 300),
        (PAGE_SIZE - 300, 3 * PAGE_SIZE + 700),
        (3 * PAGE_SIZE + 700, 5 * PAGE_SIZE),
    ],
}


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("design", DESIGNS)
def test_a_chunk_read_together_selects_what_its_pages_read_one_at_a_time_did(design, chunking):
    chunks = CHUNKINGS[chunking]
    pair = Pair(DESIGNS[design], chunks[-1][1], seed=1)
    for start, stop in chunks:
        got, want = pair.prefill(1, start, stop)
        assert got.shape == want.shape == (stop - start, WIDTH)
        assert sets(got) == sets(want)
        pair.same_state(1)


@pytest.mark.parametrize("design", DESIGNS)
def test_requests_interleaved_in_the_batch_keep_their_own_reads_and_caches(design):
    pair = Pair(DESIGNS[design], 4 * PAGE_SIZE, seed=2)
    for start, stop in ((0, 3000), (3000, PAGE_SIZE + 2000), (PAGE_SIZE + 2000, 3 * PAGE_SIZE + 5)):
        for row in ROWS:
            got, want = pair.prefill(row, start, stop)
            assert sets(got) == sets(want)
    for row in ROWS:
        pair.same_state(row)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("design", DESIGNS)
def test_a_rank_reading_a_span_of_a_chunk_selects_as_the_loop_did(design, rank):
    # Under prefill context parallel a rank queries block `rank` of 4 of each chunk and indexes all of it
    size, chunks = 4, [(0, 3 * PAGE_SIZE + 123), (3 * PAGE_SIZE + 123, 6 * PAGE_SIZE + 1)]
    pair = Pair(DESIGNS[design], chunks[-1][1], seed=3)
    for start, stop in chunks:
        length = stop - start
        span = (start + length * rank // size, start + length * (rank + 1) // size)
        got, want = pair.prefill(1, start, stop, span)
        assert got.shape == want.shape == (span[1] - span[0], WIDTH)
        assert sets(got) == sets(want)
        pair.same_state(1)


@pytest.mark.parametrize("design", ["flat", "tree"])
def test_a_read_that_returns_rows_of_prefix_and_chunk_keys_selects_as_the_loop_did(design):
    # Decode context parallel's prefill reads rows of [prefix, chunk] keys, from where the request starts in them
    size, chunks = 2, [(0, 2 * PAGE_SIZE + 77), (2 * PAGE_SIZE + 77, 5 * PAGE_SIZE)]
    pair = Pair(DESIGNS[design], chunks[-1][1], seed=4)
    for start, stop in chunks:
        length = stop - start
        span = (start + length // size, stop)
        got, want = pair.prefill(1, start, stop, span, ragged_start=1000)
        assert sets(got) == sets(want)
    pair.same_state(1)


def test_every_row_picked_in_full_still_selects_as_the_loop_did():
    # A gate of zero sends the early pages through the fused select, whose short rows are scored in full
    pair = Pair(FLAT, 3 * PAGE_SIZE, seed=5)
    selector_module.SCAN_SELECT_BUCKETS = 0
    for start, stop in ((0, 100), (100, PAGE_SIZE + 400), (PAGE_SIZE + 400, 3 * PAGE_SIZE)):
        got, want = pair.prefill(1, start, stop)
        assert sets(got) == sets(want)
    pair.same_state(1)


def test_runs_read_together_where_only_some_may_miss_a_bucket_are_picked_apart(monkeypatch):
    # With top_p 1024 a query sees fewer buckets than that until its page's end, so the first run's listing may miss and the second's cannot
    indexer = {**FLAT, "top_p": 1024, "top_k": 2048}
    pair = Pair(indexer, 2 * PAGE_SIZE, seed=6)
    selector_module.SCAN_SELECT_BUCKETS = 0
    answers = []
    read_missed = Selector.read_missed

    def spy(self, listing, spans, missed, *rest):
        answers.append(missed)
        return read_missed(self, listing, spans, missed, *rest)

    monkeypatch.setattr(Selector, "read_missed", spy)
    got, want = pair.prefill(1, 0, PAGE_SIZE - 300)
    assert sets(got) == sets(want)
    got, want = pair.prefill(1, PAGE_SIZE - 300, PAGE_SIZE + 300)
    assert [True, False] in answers
    assert got.shape[0] == 600
    assert sets(got) == sets(want)
    pair.same_state(1)


@pytest.mark.parametrize("shard", [Shard(), Shard(2, 1, lambda part: part)], ids=["whole", "split"])
def test_a_lookup_reads_a_span_as_it_reads_the_positions_of_it(shard):
    keys = F.normalize(torch.randn(900, 64, device=DEVICE), dim=-1).to(torch.bfloat16)

    class Buffer:
        def read(self, slots):
            return keys[slots]

    table = torch.arange(900, device=DEVICE, dtype=torch.int32)[None]
    chunk = (400, keys[400:700])
    for held in (None, chunk):
        lookup = KeyLookup(Buffer(), lambda: table, shard, 0, held)
        for start, stop in ((0, 300), (300, 500), (400, 600), (450, 700)):
            assert torch.equal(lookup.span(start, stop), lookup(torch.arange(start, stop, device=DEVICE)))
