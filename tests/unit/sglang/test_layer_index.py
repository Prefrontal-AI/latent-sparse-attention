"""One layer's index read off `x` - the projection, the index-key buffer and the lists a forward batch gets back.

The projection runs anywhere and holds invariant 1. The rest needs a GPU, with torch's exact top-k standing in for the
engine's so no sglang is needed: requests laid out through a request table that is not the identity, so a list of KV
slots is told apart from a list of positions, are read by a `LayerIndex` and by a bare `Selector` over the keys
the projection gives, and the two must select the same keys.
"""

from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from lsa.sglang.index import hierarchy as hierarchy_module
from lsa.sglang.index.consistency import request_checksum
from lsa.sglang.index.layer_index import (
    KeyAndQuery,
    KeyAndQueryProjection,
    KeyBuffer,
    KeyLookup,
    KeySlot,
    KeySplit,
    LayerIndex,
    PoolKeys,
    Projection,
    Shard,
    Span,
    exact_group_sum,
    page_fingerprints,
)
from lsa.sglang.index.selector import Selector
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

MODEL_DIM = 96
DEVICE = "cuda"
# The most query rows a prefill call is handed
CHUNK = 8192
DESIGN: dict[str, Any] = {
    "window_size": 64,
    "sink_size": 4,
    "top_k": 512,
    "top_p": 128,
    "index_dim": 64,
    "pool_size": 4,
    "bucket_size": 8,
    "bucket_cap": 16,
    # The attention input's layer - the key and query's, the default, is `LATENT_SETTINGS` below
    "input": "attention_input",
}
SETTINGS = DesignSettings.from_blocks(DESIGN, {})
# Node cluster blocks of 64 and a narrow beam, so a page grows several levels and the beam reads other lists than flat
NODES = {"node_bucket_size": 4, "node_cluster_block_size": 64, "beam_width": 64}
TREES = {
    "fixed": {**DESIGN, "top_level": 1, **NODES},
    "tree": {**DESIGN, "top_level": 16, **NODES, "start_nodes": 256},
}
# The first level pairs this many buckets at once at least - under a page's 1,024, so a tree grows at every page end
TREE_GROWTH_BUCKETS = 256
# Three pages in prefill chunks that cross the first two page ends, then decode steps across the third
TREE_CHUNKS = ((0, 6000), (6000, 14000), (14000, 22000), (22000, 3 * PAGE_SIZE - 6))
TREE_LENGTH = 3 * PAGE_SIZE + 30


def maps(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(MODEL_DIM, 64, generator=generator), torch.randn(MODEL_DIM, 64, generator=generator)


def test_an_index_key_and_an_index_query_are_unit_and_depend_on_x_alone():
    # invariant 1
    projection = Projection(*maps(), "cpu")
    x = torch.randn(40, MODEL_DIM).bfloat16()
    # The same x at two positions, in different company
    x[30] = x[3]
    queries, keys = projection(x)
    assert queries.dtype == torch.float32 and keys.dtype == torch.bfloat16
    assert queries.shape == keys.shape == (40, 64)
    assert torch.allclose(queries.norm(dim=-1), torch.ones(40), atol=1e-5)
    assert torch.allclose(keys.float().norm(dim=-1), torch.ones(40), atol=1e-2)
    assert torch.equal(queries[30], queries[3]) and torch.equal(keys[30], keys[3])
    alone_queries, alone_keys = projection(x[3:4])
    # A GEMM over fewer rows may round its last bit differently
    assert torch.allclose(alone_queries[0], queries[3], atol=1e-6)
    assert torch.allclose(alone_keys[0].float(), keys[3].float(), atol=1e-2)


def test_the_two_maps_are_applied_to_x_as_they_were_trained():
    query_projection, key_projection = maps(1)
    projection = Projection(query_projection, key_projection, "cpu")
    x = torch.randn(7, MODEL_DIM).bfloat16()
    queries, keys = projection(x)
    assert torch.allclose(queries, F.normalize(x.float() @ query_projection, dim=-1), atol=1e-5)
    assert torch.allclose(keys.float(), F.normalize(x.float() @ key_projection, dim=-1), atol=1e-2)


class Layout:
    """Two requests whose positions sit at shuffled KV slots, a LayerIndex over them and a bare Selector as the oracle"""

    def __init__(self, lengths, seed=0, settings=SETTINGS):
        self.lengths = lengths
        self.seed = seed
        self.settings = settings
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        total = max(lengths)
        slot_count = len(lengths) * total + 8
        self.total, self.slot_count = total, slot_count
        # Each request's slots are a shuffle of its own run, so a slot is never a position
        self.table = torch.full((3, total), -1, device=DEVICE, dtype=torch.int32)
        self.rows = [0, 2][: len(lengths)]
        for i, row in enumerate(self.rows):
            order = torch.randperm(total, device=DEVICE, generator=generator) + i * total + 1
            self.table[row] = order.to(torch.int32)
        self.x = [torch.randn(n, MODEL_DIM, device=DEVICE, generator=generator).bfloat16() for n in lengths]
        self.layer = LayerIndex(
            settings,
            maps(seed),
            slot_count,
            slot_count - 8,
            lambda: self.table,
            total,
            CHUNK,
            DEVICE,
        )
        self.inverse = torch.full((slot_count,), -1, device=DEVICE, dtype=torch.int64)
        for row, n in zip(self.rows, lengths, strict=True):
            self.inverse[self.table[row, :n].long()] = torch.arange(n, device=DEVICE)

    def slots(self, request, start, stop):
        return self.table[self.rows[request], start:stop].long()

    def positions_of(self, lists):
        """A list of KV slots as the positions they hold, -1 padding kept"""
        return torch.where(lists >= 0, self.inverse[lists.clamp(min=0).long()], -1)

    def oracle(self, request, start, stop):
        """A bare selector over this request's keys and queries on the identity table, reading [start, stop)"""
        queries, keys = self.layer.projection(self.x[request])
        total = self.lengths[request]
        identity = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(1, -1).contiguous()
        selector = Selector(SETTINGS, total, 1, total, CHUNK, DEVICE, lambda: identity)
        lookup = lambda positions: keys[positions.long()]  # noqa: E731
        selector.maintain(["oracle"], [0], [start], lambda row: lookup)
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        return selector.prefill(0, queries[start:stop], positions, lookup, {0})


def sets(lists):
    return [sorted(row[row >= 0].tolist()) for row in lists.cpu()]


@pytest.mark.gpu
def test_a_prefill_batch_of_two_requests_reads_each_as_a_selector_over_its_own_keys():
    layout = Layout([5000, 3200])
    layer = layout.layer
    layer.maintain(["a", "b"], layout.rows, [0, 0])
    x = torch.cat(layout.x)
    positions = torch.cat([torch.arange(n, device=DEVICE, dtype=torch.int32) for n in layout.lengths])
    slots = torch.cat([layout.slots(i, 0, n) for i, n in enumerate(layout.lengths)])
    lists = layer.extend(x, positions, slots, layout.rows, layout.lengths, [0, 0])
    assert lists.shape == (sum(layout.lengths), 512 + 64 + 4)
    first, second = lists[:5000], lists[5000:]
    assert sets(layout.positions_of(first)) == sets(layout.oracle(0, 0, 5000))
    assert sets(layout.positions_of(second)) == sets(layout.oracle(1, 0, 3200))
    # Every entry is a slot of its own request
    mine = set(layout.table[0, :5000].tolist())
    assert {int(slot) for slot in first[first >= 0].tolist()} <= mine


@pytest.mark.gpu
def test_the_keys_land_at_the_slots_the_tokens_were_given():
    layout = Layout([300])
    layout.layer.write(layout.x[0], layout.slots(0, 0, 300))
    _, keys = layout.layer.projection(layout.x[0])
    found = layout.layer.keys_of(layout.rows[0])(torch.arange(300, device=DEVICE))
    assert torch.equal(found, keys)
    # Nothing else was written
    written = layout.table[0, :300].long()
    untouched = torch.ones(layout.layer.index_keys.shape[0], dtype=torch.bool, device=DEVICE)
    untouched[written] = False
    assert not bool(layout.layer.index_keys[untouched].any())


@pytest.mark.gpu
def test_a_decode_step_reads_what_that_position_read_in_prefill():
    # invariant 24, through the layer: the same x, projected and written for one new token
    layout = Layout([3001])
    layer = layout.layer
    layer.maintain(["a"], layout.rows[:1], [0])
    positions = torch.arange(3000, device=DEVICE, dtype=torch.int32)
    layer.extend(layout.x[0][:3000], positions, layout.slots(0, 0, 3000), layout.rows[:1], [3000], [0])
    layer.maintain(["a"], layout.rows[:1], [3000])
    rows = torch.tensor(layout.rows[:1], device=DEVICE)
    decoded = layer.decode(
        layout.x[0][3000:3001],
        torch.tensor([3000], device=DEVICE, dtype=torch.int32),
        layout.slots(0, 3000, 3001),
        rows,
    )
    assert sets(layout.positions_of(decoded)) == sets(layout.oracle(0, 3000, 3001))


@pytest.mark.gpu
def test_a_prefill_that_only_writes_keys_leaves_them_for_the_next_forward():
    # A short prompt goes through the dense path, which calls the indexer for its keys alone
    layout = Layout([400])
    layer = layout.layer
    layer.write(layout.x[0], layout.slots(0, 0, 400))
    layer.maintain(["a"], layout.rows[:1], [400])
    assert layer.selector.cache.end[layout.rows[0]] == 400


@pytest.mark.gpu
def test_with_the_top_k_above_the_context_the_hooked_read_is_the_dense_attention(monkeypatch):
    # invariant 28, attention half: the lists the layer hands the backend, through the fa3 hook's exact lengths,
    # read every causal key once, so attention over them is the dense causal attention
    import sys
    from types import ModuleType

    from lsa.sglang import plugin

    precompute = ModuleType("sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute")
    precompute.compute_cu_seqlens = lambda lengths: F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute", precompute)

    layout = Layout([300])
    layer = layout.layer
    layer.maintain(["a"], layout.rows[:1], [0])
    positions = torch.arange(300, device=DEVICE, dtype=torch.int32)
    lists = layer.extend(layout.x[0], positions, layout.slots(0, 0, 300), layout.rows[:1], [300], [0])

    generator = torch.Generator(device=DEVICE).manual_seed(7)
    queries = torch.randn(300, 32, device=DEVICE, generator=generator)
    cache = torch.randn(layer.index_keys.shape[0], 32, device=DEVICE, generator=generator)

    def original(self, q, page_table, cache_seqlens, cu_seqlens_k):
        # Attention of each token over the first cache_seqlens slots of its row
        out = torch.zeros_like(q)
        for i in range(q.shape[0]):
            slots = page_table[i, : int(cache_seqlens[i])].long()
            weights = (cache[slots] @ q[i]).softmax(0)
            out[i] = weights @ cache[slots]
        return out

    hooked = plugin.fa3(original, None, queries, lists, None, None)
    keys = cache[layout.table[0, :300].long()]
    dense = torch.zeros_like(queries)
    for i in range(300):
        weights = (keys[: i + 1] @ queries[i]).softmax(0)
        dense[i] = weights @ keys[: i + 1]
    assert torch.allclose(hooked, dense, atol=1e-5)


class Ranks:
    """The ranks of a context-parallel group in one process, each a LayerIndex of its own over one layout

    A decode group's sum across its ranks is replaced by reading every rank's owned rows, and a prefill group's
    gather by handing each rank the chunk's index keys, so ranks can be run one after another.
    """

    def __init__(self, layout, decode_size=1, prefill_size=1):
        self.layout, self.decode_size, self.prefill_size = layout, decode_size, prefill_size
        rows = -(-layout.slot_count // decode_size)
        self.layers = [
            LayerIndex(
                layout.settings,
                maps(layout.seed),
                rows,
                rows,
                lambda: layout.table,
                layout.total,
                CHUNK,
                DEVICE,
                Shard(decode_size, rank % decode_size, lambda local: local),
            )
            for rank in range(max(decode_size, prefill_size))
        ]
        group = self.layers[:decode_size]

        def read(slots):
            terms = []
            for layer in group:
                buffer_rows, owned = layer.buffer.rows(slots)
                terms.append(layer.buffer.keys[buffer_rows] * owned[:, None])
            return sum(terms[1:], terms[0])

        if decode_size > 1:
            for layer in group:
                layer.buffer.read = read

    def maintain(self, rids, ends):
        for layer in self.layers:
            layer.maintain(rids, self.layout.rows[: len(rids)], ends)

    def prefill(self, requests, start, stop, ragged=None):
        """Every rank's lists for the chunk [start, stop) of each request, indexed by rank"""
        layout = self.layout
        x = [layout.x[i][start:stop] for i in requests]
        counts = [len(part) for part in x]
        positions = torch.cat(
            [torch.arange(start, start + n, device=DEVICE, dtype=torch.int32) for n in counts]
        )
        slots = torch.cat([layout.slots(i, start, start + n) for i, n in zip(requests, counts, strict=True)])
        rows = [layout.rows[i] for i in requests]
        full = torch.cat(x)
        chunk_keys = self.layers[0].projection(full)[1]
        out = []
        for rank, layer in enumerate(self.layers):
            if self.prefill_size == 1:
                out.append(
                    layer.extend(full, positions, slots, rows, counts, [start] * len(counts), None, ragged)
                )
                continue
            # This rank's share of each request: its part of every request's rows, in request order
            mine = [self.share(n, rank) for n in counts]
            local_x = torch.cat([part[each] for part, each in zip(x, mine, strict=True)])
            local_positions = torch.cat(
                [
                    positions[a : a + n][each]
                    for a, n, each in zip(self.offsets(counts), counts, mine, strict=True)
                ]
            )
            span = Span(
                positions,
                counts,
                [each.stop - each.start for each in mine],
                [start + each.start for each in mine],
                lambda local: chunk_keys,
            )
            out.append(
                layer.extend(
                    local_x, local_positions, slots, rows, counts, [start] * len(counts), span, ragged
                )
            )
        return out

    def share(self, count, rank):
        """The rows of a request's `count` new tokens a rank holds under prefill context parallel"""
        size = self.prefill_size
        return slice(count * (rank % size) // size, count * (rank % size + 1) // size)

    def decode(self, request, position):
        """Every rank's list for one request's decode step at `position`, indexed by rank"""
        layout = self.layout
        rows = torch.tensor(layout.rows[request : request + 1], device=DEVICE)
        token = torch.tensor([position], device=DEVICE, dtype=torch.int32)
        x = layout.x[request][position : position + 1]
        return [
            layer.decode(x, token, layout.slots(request, position, position + 1), rows)
            for layer in self.layers
        ]

    @staticmethod
    def offsets(counts):
        return [sum(counts[:i]) for i in range(len(counts))]


@pytest.mark.gpu
def test_a_decode_group_selects_what_one_rank_holding_every_key_selects():
    layout = Layout([3100])
    alone, group = Ranks(layout), Ranks(layout, decode_size=2)
    for ranks in (alone, group):
        ranks.maintain(["a"], [0])
    # The second chunk starts inside a cluster block, so its keys come partly from the pool and partly from the chunk
    wanted = alone.prefill([0], 0, 1500)
    for lists in group.prefill([0], 0, 1500):
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(wanted[0]))
    for ranks in (alone, group):
        ranks.maintain(["a"], [1500])
    wanted = alone.prefill([0], 1500, 3000)
    for lists in group.prefill([0], 1500, 3000):
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(wanted[0]))
    # Each rank holds half of the index keys
    assert group.layers[0].buffer.keys.shape[0] == -(-layout.slot_count // 2) + 1
    # And the decode step the group takes reads what a rank holding every key reads
    rows = torch.tensor(layout.rows[:1], device=DEVICE)
    token = torch.tensor([3000], device=DEVICE, dtype=torch.int32)
    for ranks in (alone, group):
        ranks.maintain(["a"], [3000])
    decoded = [
        layer.decode(layout.x[0][3000:3001], token, layout.slots(0, 3000, 3001), rows)
        for layer in alone.layers + group.layers
    ]
    for lists in decoded[1:]:
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(decoded[0]))


@pytest.mark.gpu
def test_decode_context_parallel_reads_a_prefill_as_rows_of_the_prefix_and_the_chunk():
    layout = Layout([2200])
    alone, group = Ranks(layout), Ranks(layout, decode_size=2)
    for ranks in (alone, group):
        ranks.maintain(["a"], [0])
    wanted = alone.prefill([0], 0, 2200)[0]
    offset = 9000
    for lists in group.prefill([0], 0, 2200, ragged=[offset]):
        positions = layout.positions_of(wanted)
        assert sets(lists) == [[p + offset for p in row] for row in sets(positions)]
        assert lists.shape == wanted.shape


@pytest.mark.gpu
def test_a_prefill_group_selects_each_ranks_rows_as_one_rank_holding_the_chunk_does():
    layout = Layout([2500, 1800])
    alone, group = Ranks(layout), Ranks(layout, prefill_size=2)
    for ranks in (alone, group):
        ranks.maintain(["a", "b"], [0, 0])
    wanted = alone.prefill([0, 1], 0, 2500)[0]
    first, second = group.prefill([0, 1], 0, 2500)
    # Rank r holds its half of each request - the requests' rows end to end
    halves = [(0, 1250), (2500, 2500 + 900)], [(1250, 2500), (2500 + 900, 4300)]
    for lists, spans in zip((first, second), halves, strict=True):
        expected = torch.cat([wanted[a:b] for a, b in spans])
        assert lists.shape == expected.shape
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(expected))


@pytest.mark.gpu
def test_prefill_and_decode_context_parallel_together_select_as_one_rank_does():
    layout = Layout([2600])
    alone, both = Ranks(layout), Ranks(layout, decode_size=2, prefill_size=2)
    for ranks in (alone, both):
        ranks.maintain(["a"], [0])
    wanted = alone.prefill([0], 0, 2600)[0]
    first, second = both.prefill([0], 0, 2600)
    for lists, part in zip((first, second), (wanted[:1300], wanted[1300:]), strict=True):
        assert sets(layout.positions_of(lists)) == sets(layout.positions_of(part))


def assert_same_index(groups, row):
    """Every rank of every group holds the same buckets and the same tree for the request"""
    caches = [layer.selector.cache for ranks in groups for layer in ranks.layers]
    first = caches[0]
    visible = first.visible(row)
    held = first.logical_rows(row, visible)[:visible]
    for cache in caches[1:]:
        assert cache.visible(row) == visible
        assert cache.hierarchies[row] == first.hierarchies[row]
        rows = cache.logical_rows(row, visible)[:visible]
        for name in ("bucket_centroids", "bucket_members", "bucket_sizes"):
            assert torch.equal(getattr(cache, name)[rows], getattr(first, name)[held])
    assert len({int(request_checksum(cache, row)) for cache in caches}) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("decode", ["flat", "descend"])
@pytest.mark.parametrize("design", ["fixed", "tree"])
def test_context_parallel_groups_grow_the_tree_one_rank_grows_and_select_as_it_does(
    design, decode, monkeypatch
):
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", TREE_GROWTH_BUCKETS)
    layout = Layout([TREE_LENGTH], settings=DesignSettings.from_blocks(TREES[design], {"decode": decode}))
    alone = Ranks(layout)
    groups = [
        Ranks(layout, decode_size=2),
        Ranks(layout, prefill_size=2),
        Ranks(layout, decode_size=2, prefill_size=2),
    ]
    row = layout.rows[0]
    covered = set()
    for start, stop in TREE_CHUNKS:
        for ranks in (alone, *groups):
            ranks.maintain(["a"], [start])
        wanted = sets(layout.positions_of(alone.prefill([0], start, stop)[0]))
        for ranks in groups:
            # Decode context parallel's prefill reads rows of [prefix, chunk] keys - here from 0, so positions
            ragged = [0] if ranks.decode_size > 1 else None
            for rank, lists in enumerate(ranks.prefill([0], start, stop, ragged)):
                read = sets(lists if ragged else layout.positions_of(lists))
                assert read == wanted[ranks.share(stop - start, rank)]
        assert_same_index([alone, *groups], row)
        covered.add(alone.layers[0].selector.cache.hierarchies[row].covered_buckets)
    # The last page fills in a decode step's maintenance, which grows the tree from index keys the group sums
    for position in range(TREE_CHUNKS[-1][1], TREE_LENGTH):
        for ranks in (alone, *groups):
            ranks.maintain(["a"], [position])
        wanted = sets(layout.positions_of(alone.decode(0, position)[0]))
        for ranks in groups:
            for lists in ranks.decode(0, position):
                assert sets(layout.positions_of(lists)) == wanted
        assert_same_index([alone, *groups], row)
        covered.add(alone.layers[0].selector.cache.hierarchies[row].covered_buckets)
    buckets_per_page = PAGE_SIZE // DESIGN["bucket_size"]
    assert sorted(covered) == [0, buckets_per_page, 2 * buckets_per_page, 3 * buckets_per_page]
    assert alone.layers[0].selector.cache.hierarchies[row].levels == (1 if design == "fixed" else 3)


KEY_DIM = 48
HEADS = 4
LATENT_SETTINGS = DesignSettings.from_blocks({**DESIGN, "input": "key_and_query"}, {})


def latent_maps(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(KEY_DIM, 64, generator=generator), torch.randn(KEY_DIM, 64, generator=generator)


def test_the_key_and_query_projection_is_the_freezes_bf16_arithmetic():
    query_projection, key_projection = latent_maps(3)
    projection = KeyAndQueryProjection(query_projection, key_projection, "cpu")
    key = torch.randn(9, KEY_DIM).bfloat16()
    query = torch.randn(9, HEADS, KEY_DIM).bfloat16()
    expected_keys = F.normalize((key @ key_projection.bfloat16()).float(), dim=-1).bfloat16()
    summed = query.sum(1, dtype=torch.float32).bfloat16()
    expected_queries = F.normalize((summed @ query_projection.bfloat16()).float(), dim=-1)
    assert torch.equal(projection.index_keys(key), expected_keys)
    assert torch.equal(projection.index_queries(query), expected_queries)
    assert projection.index_queries(query).dtype == torch.float32


def test_the_key_and_query_input_is_refused_without_the_pool():
    table = torch.zeros(1, 8, dtype=torch.int32)
    with pytest.raises(ValueError, match="KV pool"):
        LayerIndex(LATENT_SETTINGS, latent_maps(), 8, 8, lambda: table, 8, CHUNK, "cpu")


def owner_pools(keys, size):
    """Each rank's KV pool of key latents, slot `s` at row `s // size` of rank `s % size`, as decode CP places it"""
    pools = [torch.zeros(-(-len(keys) // size), 1, KEY_DIM + 8, dtype=torch.bfloat16) for _ in range(size)]
    for slot, key in enumerate(keys):
        pools[slot % size][slot // size, 0, :KEY_DIM] = key
    return pools


def test_a_decode_group_projects_each_key_on_the_rank_that_owns_it_and_sums_the_index_keys():
    size = 3
    keys = torch.randn(20, KEY_DIM).bfloat16()
    table = torch.zeros(1, 8, dtype=torch.int32)
    pools, terms = owner_pools(keys, size), []
    layers = [
        LayerIndex(
            LATENT_SETTINGS,
            latent_maps(),
            8,
            8,
            lambda: table,
            8,
            CHUNK,
            "cpu",
            Shard(size, rank, lambda term: terms.append(term) or term),
            key_buffer=lambda pool=pool: pool,
        )
        for rank, pool in enumerate(pools)
    ]
    slots = torch.tensor([7, 0, 19, 4, 4, 11])
    for layer in layers:
        layer.buffer.read(slots)
    assert torch.equal(sum(terms[1:], terms[0]), layers[0].projection.index_keys(keys[slots]))
    # Each row has one owner, and the others add zeros
    for rank, term in enumerate(terms):
        assert torch.equal(term.float().abs().sum(-1) > 0, slots % size == rank)
    # A decode group writes nothing to the pool - attention writes each key to the rank that owns it
    before = [pool.clone() for pool in pools]
    for layer in layers:
        layer.write(KeyAndQuery(keys[:2], torch.randn(2, HEADS, KEY_DIM).bfloat16()), slots[:2])
    assert all(torch.equal(pool, kept) for pool, kept in zip(pools, before, strict=True))


def test_a_prefill_chunk_s_key_must_be_the_whole_chunk_s():
    table = torch.arange(8, dtype=torch.int32)[None]
    pool = torch.zeros(9, 1, KEY_DIM + 8, dtype=torch.bfloat16)
    layer = LayerIndex(
        LATENT_SETTINGS, latent_maps(), 8, 8, lambda: table, 8, CHUNK, "cpu", key_buffer=lambda: pool
    )
    with pytest.raises(ValueError, match="whole chunk"):
        layer.write(
            KeyAndQuery(torch.randn(2, KEY_DIM).bfloat16(), torch.randn(2, HEADS, KEY_DIM).bfloat16()),
            torch.arange(4),
        )


class LatentLayout(Layout):
    """A Layout whose layer reads the key and the heads' queries, its index keys projected from a KV pool's keys"""

    def __init__(self, lengths, seed=0):
        super().__init__(lengths, seed)
        self.settings = LATENT_SETTINGS
        generator = torch.Generator(device=DEVICE).manual_seed(seed + 1)
        # A pool slot holds the key latent and rope dims the index never reads
        self.pool = torch.zeros(self.slot_count, 1, KEY_DIM + 8, device=DEVICE, dtype=torch.bfloat16)
        self.keys = [torch.randn(n, KEY_DIM, device=DEVICE, generator=generator).bfloat16() for n in lengths]
        self.queries = [
            torch.randn(n, HEADS, KEY_DIM, device=DEVICE, generator=generator).bfloat16() for n in lengths
        ]
        self.layer = LayerIndex(
            LATENT_SETTINGS,
            latent_maps(seed),
            self.slot_count,
            self.slot_count - 8,
            lambda: self.table,
            self.total,
            CHUNK,
            DEVICE,
            key_buffer=lambda: self.pool,
        )

    def inputs(self, request, start, stop):
        return KeyAndQuery(self.keys[request][start:stop], self.queries[request][start:stop])

    def oracle(self, request, start, stop):
        """A bare selector over this request's projected keys and queries on the identity table, reading [start, stop)"""
        projection = self.layer.projection
        queries = projection.index_queries(self.queries[request])
        keys = projection.index_keys(self.keys[request])
        total = self.lengths[request]
        identity = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(1, -1).contiguous()
        selector = Selector(LATENT_SETTINGS, total, 1, total, CHUNK, DEVICE, lambda: identity)
        lookup = lambda positions: keys[positions.long()]  # noqa: E731
        selector.maintain(["oracle"], [0], [start], lambda row: lookup)
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        return selector.prefill(0, queries[start:stop], positions, lookup, {0})


@pytest.mark.gpu
def test_the_key_and_query_input_reads_as_a_selector_over_its_projected_keys():
    layout = LatentLayout([5000, 3200])
    layer = layout.layer
    layer.maintain(["a", "b"], layout.rows, [0, 0])
    inputs = KeyAndQuery(torch.cat(layout.keys), torch.cat(layout.queries))
    positions = torch.cat([torch.arange(n, device=DEVICE, dtype=torch.int32) for n in layout.lengths])
    slots = torch.cat([layout.slots(i, 0, n) for i, n in enumerate(layout.lengths)])
    lists = layer.extend(inputs, positions, slots, layout.rows, layout.lengths, [0, 0])
    assert sets(layout.positions_of(lists[:5000])) == sets(layout.oracle(0, 0, 5000))
    assert sets(layout.positions_of(lists[5000:])) == sets(layout.oracle(1, 0, 3200))
    # The keys are the pool's, written at the slots the tokens were given, and the layer holds none of its own
    assert torch.equal(layout.pool[slots, 0, :KEY_DIM], inputs.key)
    assert layer.index_keys.numel() == 0


@pytest.mark.gpu
def test_a_key_and_query_decode_step_reads_what_that_position_read_in_prefill():
    layout = LatentLayout([3001])
    layer = layout.layer
    layer.maintain(["a"], layout.rows[:1], [0])
    positions = torch.arange(3000, device=DEVICE, dtype=torch.int32)
    layer.extend(layout.inputs(0, 0, 3000), positions, layout.slots(0, 0, 3000), layout.rows[:1], [3000], [0])
    layer.maintain(["a"], layout.rows[:1], [3000])
    decoded = layer.decode(
        layout.inputs(0, 3000, 3001),
        torch.tensor([3000], device=DEVICE, dtype=torch.int32),
        layout.slots(0, 3000, 3001),
        torch.tensor(layout.rows[:1], device=DEVICE),
    )
    assert sets(layout.positions_of(decoded)) == sets(layout.oracle(0, 3000, 3001))


@pytest.mark.gpu
def test_keys_attention_wrote_to_the_pool_are_indexed_with_no_indexer_call():
    # A short prompt the dense path prefills writes its keys to the pool through attention alone
    layout = LatentLayout([400])
    layer = layout.layer
    layout.pool[layout.slots(0, 0, 400), 0, :KEY_DIM] = layout.keys[0]
    layer.maintain(["a"], layout.rows[:1], [400])
    assert layer.selector.cache.end[layout.rows[0]] == 400
    found = layer.keys_of(layout.rows[0])(torch.arange(400, device=DEVICE))
    assert torch.equal(found, layer.projection.index_keys(layout.keys[0]))


class LatentRanks:
    """The ranks of a prefill and decode context-parallel group under the key and query, in one process

    Each rank's pool holds the key latents of the slots it owns, which attention writes after every forward, and the
    decode group's sum is replaced by adding every rank's term.
    """

    def __init__(self, layout, size):
        self.layout, self.size = layout, size
        rows = -(-layout.slot_count // size)
        self.pools = [
            torch.zeros(rows, 1, KEY_DIM + 8, device=DEVICE, dtype=torch.bfloat16) for _ in range(size)
        ]
        self.layers = [
            LayerIndex(
                LATENT_SETTINGS,
                latent_maps(layout.seed),
                rows,
                rows,
                lambda: layout.table,
                layout.total,
                CHUNK,
                DEVICE,
                Shard(size, rank, lambda term: term),
                key_buffer=lambda pool=pool: pool,
            )
            for rank, pool in enumerate(self.pools)
        ]
        reads = [layer.buffer.read for layer in self.layers]

        def read(slots):
            terms = [own(slots) for own in reads]
            return sum(terms[1:], terms[0])

        for layer in self.layers:
            layer.buffer.read = read

    def attend(self, slots, keys):
        """Write a forward's key latents as attention does, each to the rank that owns its slot"""
        for rank, pool in enumerate(self.pools):
            owned = slots % self.size == rank
            pool[slots[owned] // self.size, 0, :KEY_DIM] = keys[owned]

    def prefill(self, start, stop):
        """Every rank's lists for [start, stop) of request 0, its queries this rank's rows and its key the chunk's"""
        layout, count = self.layout, stop - start
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        slots, key, query = (
            layout.slots(0, start, stop),
            layout.keys[0][start:stop],
            layout.queries[0][start:stop],
        )
        out = []
        for rank, layer in enumerate(self.layers):
            mine = slice(count * rank // self.size, count * (rank + 1) // self.size)
            span = Span(
                positions, [count], [mine.stop - mine.start], [start + mine.start], lambda local: local
            )
            inputs = KeyAndQuery(key, query[mine])
            out.append(layer.extend(inputs, positions[mine], slots, layout.rows[:1], [count], [start], span))
        self.attend(slots, key)
        return out


@pytest.mark.gpu
def test_prefill_and_decode_context_parallel_select_the_key_and_query_input_as_one_rank_does():
    layout = LatentLayout([3100])
    alone, group = layout.layer, LatentRanks(layout, 2)
    rows = layout.rows[:1]
    # The second chunk starts inside a cluster block, so its keys come partly from the pool and partly from the chunk
    for start, stop in ((0, 1500), (1500, 3000)):
        alone.maintain(["a"], rows, [start])
        for layer in group.layers:
            layer.maintain(["a"], rows, [start])
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        wanted = alone.extend(
            layout.inputs(0, start, stop),
            positions,
            layout.slots(0, start, stop),
            rows,
            [stop - start],
            [start],
        )
        half = (stop - start) // 2
        first, second = group.prefill(start, stop)
        assert sets(layout.positions_of(first)) == sets(layout.positions_of(wanted[:half]))
        assert sets(layout.positions_of(second)) == sets(layout.positions_of(wanted[half:]))
    # The decode step reads every earlier key through the group's sum, and each rank has its every head
    alone.maintain(["a"], rows, [3000])
    for layer in group.layers:
        layer.maintain(["a"], rows, [3000])
    token = torch.tensor([3000], device=DEVICE, dtype=torch.int32)
    slots, request_rows = layout.slots(0, 3000, 3001), torch.tensor(rows, device=DEVICE)
    wanted = alone.decode(layout.inputs(0, 3000, 3001), token, slots, request_rows)
    for layer in group.layers:
        decoded = layer.decode(layout.inputs(0, 3000, 3001), token, slots, request_rows)
        assert sets(layout.positions_of(decoded)) == sets(layout.positions_of(wanted))


def test_an_attention_input_reaching_a_key_and_query_index_is_refused():
    table = torch.arange(8, dtype=torch.int32)[None]
    pool = torch.zeros(9, 1, KEY_DIM + 8, dtype=torch.bfloat16)
    layer = LayerIndex(
        LATENT_SETTINGS, latent_maps(), 8, 8, lambda: table, 8, CHUNK, "cpu", key_buffer=lambda: pool
    )
    with pytest.raises(TypeError, match="attention input"):
        layer.write(torch.randn(2, MODEL_DIM).bfloat16(), torch.arange(2))


def test_query_heads_all_sums_every_ranks_heads_before_the_projection():
    # Two ranks of two heads each, summed across the group, read as one rank holding all four heads
    query_projection, key_projection = latent_maps(5)
    query = torch.randn(6, 4, KEY_DIM).bfloat16()
    first, second = query[:, :2], query[:, 2:]
    whole = KeyAndQueryProjection(query_projection, key_projection, "cpu")
    group = [
        KeyAndQueryProjection(
            query_projection,
            key_projection,
            "cpu",
            sum_heads=lambda sums, other=other: sums + other.sum(1, dtype=torch.float32),
        )
        for other in (second, first)
    ]
    expected = whole.index_queries(query)
    assert torch.allclose(group[0].index_queries(first), expected, atol=1e-6)
    assert torch.allclose(group[1].index_queries(second), expected, atol=1e-6)
    assert torch.equal(group[0].index_queries(first), group[1].index_queries(second))


# A grouped-query pool slot - Qwen3.5's shape scaled down: four key heads of 32 dims, the first 8 rotated
GROUPED_SLOT = KeySlot(heads=4, head_dim=32, offset=8, width=24)


class GroupedLayout(Layout):
    """A Layout whose layer reads a grouped-query pool - every key head past its rotated prefix, side by side"""

    def __init__(self, lengths, seed=0, settings=LATENT_SETTINGS):
        super().__init__(lengths, seed)
        self.grouped_settings = settings
        generator = torch.Generator(device=DEVICE).manual_seed(seed + 2)
        key_dim = GROUPED_SLOT.key_dim
        self.pool = torch.zeros(self.slot_count, 4, 32, device=DEVICE, dtype=torch.bfloat16)
        self.heads = [torch.randn(n, 4, 32, device=DEVICE, generator=generator).bfloat16() for n in lengths]
        self.queries = [torch.randn(n, 1, key_dim, device=DEVICE, generator=generator) for n in lengths]
        generator = torch.Generator().manual_seed(seed)
        self.maps = (
            torch.randn(key_dim, 64, generator=generator),
            torch.randn(key_dim, 64, generator=generator),
        )
        self.layer = LayerIndex(
            settings,
            self.maps,
            self.slot_count,
            self.slot_count - 8,
            lambda: self.table,
            self.total,
            CHUNK,
            DEVICE,
            key_buffer=lambda: self.pool,
            key_slot=GROUPED_SLOT,
        )

    def key(self, request, start, stop):
        return self.heads[request][start:stop, :, 8:].reshape(stop - start, -1)

    def inputs(self, request, start, stop):
        return KeyAndQuery(self.key(request, start, stop), self.queries[request][start:stop])

    def attend(self, request, start, stop):
        """What attention writes - the whole of every head, its rotated prefix included"""
        self.pool[self.slots(request, start, stop)] = self.heads[request][start:stop]

    def oracle(self, request, start, stop):
        projection = self.layer.projection
        queries = projection.index_queries(self.queries[request])
        keys = projection.index_keys(self.key(request, 0, self.lengths[request]))
        total = self.lengths[request]
        identity = torch.arange(total, device=DEVICE, dtype=torch.int32).expand(1, -1).contiguous()
        selector = Selector(LATENT_SETTINGS, total, 1, total, CHUNK, DEVICE, lambda: identity)
        lookup = lambda positions: keys[positions.long()]  # noqa: E731
        selector.maintain(["oracle"], [0], [start], lambda row: lookup)
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        return selector.prefill(0, queries[start:stop], positions, lookup, {0})


@pytest.mark.gpu
def test_a_grouped_query_pool_reads_as_a_selector_over_its_projected_keys():
    layout = GroupedLayout([5000, 3200])
    layer = layout.layer
    for request, n in enumerate(layout.lengths):
        layout.attend(request, 0, n)
    layer.maintain(["a", "b"], layout.rows, [0, 0])
    inputs = KeyAndQuery(
        torch.cat([layout.key(i, 0, n) for i, n in enumerate(layout.lengths)]), torch.cat(layout.queries)
    )
    positions = torch.cat([torch.arange(n, device=DEVICE, dtype=torch.int32) for n in layout.lengths])
    slots = torch.cat([layout.slots(i, 0, n) for i, n in enumerate(layout.lengths)])
    lists = layer.extend(inputs, positions, slots, layout.rows, layout.lengths, [0, 0])
    assert sets(layout.positions_of(lists[:5000])) == sets(layout.oracle(0, 0, 5000))
    assert sets(layout.positions_of(lists[5000:])) == sets(layout.oracle(1, 0, 3200))
    # The write left each slot's rotated prefix as attention wrote it
    assert torch.equal(layout.pool[slots[:5000]], layout.heads[0])


@pytest.mark.gpu
def test_a_grouped_query_decode_step_reads_what_that_position_read_in_prefill():
    layout = GroupedLayout([3001])
    layer = layout.layer
    layout.attend(0, 0, 3000)
    layer.maintain(["a"], layout.rows[:1], [0])
    positions = torch.arange(3000, device=DEVICE, dtype=torch.int32)
    layer.extend(layout.inputs(0, 0, 3000), positions, layout.slots(0, 0, 3000), layout.rows[:1], [3000], [0])
    layout.attend(0, 3000, 3001)
    layer.maintain(["a"], layout.rows[:1], [3000])
    decoded = layer.decode(
        layout.inputs(0, 3000, 3001),
        torch.tensor([3000], device=DEVICE, dtype=torch.int32),
        layout.slots(0, 3000, 3001),
        torch.tensor(layout.rows[:1], device=DEVICE),
    )
    assert sets(layout.positions_of(decoded)) == sets(layout.oracle(0, 3000, 3001))


class GroupedRanks:
    """A decode context-parallel group over a grouped-query pool, in one process - Qwen under tensor parallel 8

    Each rank's pool holds every key head of the slots it owns, slot s at row s // size, and the group's sum is
    replaced by adding every rank's term. Every rank is handed the chunk's keys whole and the gathered query heads.
    """

    def __init__(self, layout, size):
        self.layout, self.size = layout, size
        rows = -(-layout.slot_count // size)
        self.pools = [torch.zeros(rows, 4, 32, device=DEVICE, dtype=torch.bfloat16) for _ in range(size)]
        self.layers = [
            LayerIndex(
                layout.grouped_settings,
                layout.maps,
                rows,
                rows,
                lambda: layout.table,
                layout.total,
                CHUNK,
                DEVICE,
                Shard(size, rank, lambda term: term),
                key_buffer=lambda pool=pool: pool,
                key_slot=GROUPED_SLOT,
            )
            for rank, pool in enumerate(self.pools)
        ]
        reads = [layer.buffer.read for layer in self.layers]

        def read(slots):
            terms = [own(slots) for own in reads]
            return sum(terms[1:], terms[0])

        for layer in self.layers:
            layer.buffer.read = read

    def attend(self, request, start, stop):
        """Write a forward's every head as attention does, each slot to the rank that owns it"""
        layout = self.layout
        slots, heads = layout.slots(request, start, stop), layout.heads[request][start:stop]
        for rank, pool in enumerate(self.pools):
            owned = slots % self.size == rank
            pool[slots[owned] // self.size] = heads[owned]

    def maintain(self, rids, ends):
        for layer in self.layers:
            layer.maintain(rids, self.layout.rows[: len(rids)], ends)


@pytest.mark.gpu
@pytest.mark.parametrize("design", ["flat", "fixed"])
def test_a_grouped_query_decode_group_selects_what_one_rank_holding_every_key_selects(design):
    settings = (
        LATENT_SETTINGS
        if design == "flat"
        else DesignSettings.from_blocks({**TREES[design], "input": "key_and_query"}, {})
    )
    layout = GroupedLayout([3100], settings=settings)
    alone, group = layout.layer, GroupedRanks(layout, 8)
    rows = layout.rows[:1]
    # The second chunk starts inside a cluster block, so its keys come partly from the pool and partly from the chunk
    for start, stop in ((0, 1500), (1500, 3000)):
        alone.maintain(["a"], rows, [start])
        group.maintain(["a"], [start])
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        slots = layout.slots(0, start, stop)
        layout.attend(0, start, stop)
        group.attend(0, start, stop)
        wanted = alone.extend(layout.inputs(0, start, stop), positions, slots, rows, [stop - start], [start])
        for layer in group.layers:
            lists = layer.extend(
                layout.inputs(0, start, stop), positions, slots, rows, [stop - start], [start]
            )
            assert sets(layout.positions_of(lists)) == sets(layout.positions_of(wanted))
    # A decode step reads every earlier key through the group's sum
    alone.maintain(["a"], rows, [3000])
    group.maintain(["a"], [3000])
    layout.attend(0, 3000, 3001)
    group.attend(0, 3000, 3001)
    token = torch.tensor([3000], device=DEVICE, dtype=torch.int32)
    slots, request_rows = layout.slots(0, 3000, 3001), torch.tensor(rows, device=DEVICE)
    wanted = alone.decode(layout.inputs(0, 3000, 3001), token, slots, request_rows)
    for layer in group.layers:
        decoded = layer.decode(layout.inputs(0, 3000, 3001), token, slots, request_rows)
        assert sets(layout.positions_of(decoded)) == sets(layout.positions_of(wanted))


def test_a_fingerprint_reads_every_bit_of_a_page_and_its_order():
    stored = torch.randn(2 * PAGE_SIZE, 64).bfloat16()
    owned = torch.ones(2 * PAGE_SIZE, dtype=torch.bool)
    prints = page_fingerprints(stored, owned)
    assert prints.shape == (2,) and prints.dtype == torch.int64
    assert torch.equal(page_fingerprints(stored.clone(), owned), prints)
    flipped = stored.clone()
    flipped.view(torch.int16)[PAGE_SIZE + 5, 3] ^= 1
    assert page_fingerprints(flipped, owned)[0] == prints[0]
    assert page_fingerprints(flipped, owned)[1] != prints[1]
    swapped = stored.clone()
    swapped[[0, 1]] = stored[[1, 0]]
    assert page_fingerprints(swapped, owned)[0] != prints[0]


class Group:
    """Ranks called one after another, each `reduce` returning the sum of what every rank last handed it"""

    def __init__(self, size):
        self.handed = {}
        self.size = size

    def reduce_of(self, rank):
        def reduce(local):
            self.handed[rank] = local
            return sum(self.handed.values()) if len(self.handed) == self.size else local

        return reduce


def test_a_decode_group_sums_its_ranks_fingerprints_into_the_one_rank_holding_every_slot_takes():
    stored = torch.randn(PAGE_SIZE, 64).bfloat16()
    whole = page_fingerprints(stored, torch.ones(PAGE_SIZE, dtype=torch.bool))
    owners = torch.arange(PAGE_SIZE) % 3
    # A rank reads some row of its own where another owns the slot - the fingerprint ignores it
    parts = [
        page_fingerprints(torch.where((owners == rank)[:, None], stored, stored[0]), owners == rank)
        for rank in range(3)
    ]
    group = Group(3)
    sums = [exact_group_sum(parts[rank], group.reduce_of(rank)) for rank in (0, 1, 2, 0, 1)]
    assert all(torch.equal(total, whole) for total in sums[2:])
    # The sum wraps as an int64 sum does
    near = torch.tensor([(1 << 63) - 5, -(1 << 63) + 7])
    group = Group(2)
    sums = [exact_group_sum(near, group.reduce_of(rank)) for rank in (0, 1)]
    assert torch.equal(sums[1], near + near)


def test_requests_holding_the_same_keys_at_other_slots_fingerprint_alike_until_a_slot_is_written_again():
    # Two requests' page of keys at disjoint shuffled slots - the latents in the pool, and index keys in a buffer
    keys = torch.randn(PAGE_SIZE, KEY_DIM).bfloat16()
    index_keys = torch.randn(PAGE_SIZE, 64).bfloat16()
    table = torch.stack([torch.randperm(PAGE_SIZE) + offset for offset in (0, PAGE_SIZE)]).int()
    slot = int(table[1, 17])
    projection = KeyAndQueryProjection(*latent_maps(), "cpu")
    for size in (1, 2):
        latents = torch.zeros(2 * PAGE_SIZE, KEY_DIM).bfloat16()
        for row in (0, 1):
            latents[table[row].long()] = keys
        pools = owner_pools(latents, size)
        group = Group(size)
        shards = [Shard(size, rank, group.reduce_of(rank) if size > 1 else None) for rank in range(size)]
        stores = [
            PoolKeys(lambda pool=pool: pool, projection, shard, KeySlot.latent(KEY_DIM, 8))
            for pool, shard in zip(pools, shards, strict=True)
        ]
        buffers = [KeyBuffer(2 * PAGE_SIZE // size + 1, 64, "cpu", shard) for shard in shards]
        for buffer in buffers:
            for row in (0, 1):
                buffer.write(table[row].long(), index_keys)

        def prints(held, row):
            # Every rank asks, as a group does - the last to ask has every rank's term
            return [
                KeyLookup(store, lambda: table, store.shard, row, None).fingerprints(0, 1) for store in held
            ][-1]

        for held in (stores, buffers):
            assert prints(held, 0) == prints(held, 1)
        # Request 1's slot 17 freed and written again by another token
        pools[slot % size][slot // size, 0, :KEY_DIM] = keys[3]
        for buffer in buffers:
            buffer.write(torch.tensor([slot]), index_keys[3:4])
        for held in (stores, buffers):
            assert prints(held, 0) != prints(held, 1)


REUSE_LENGTHS = (3 * PAGE_SIZE + 700, 3 * PAGE_SIZE + 300)
SHARED = 2 * PAGE_SIZE


def prefix_run(settings, reuse, rewrite, retained):
    """Request a prefilled and decoded a step at shuffled slots, then b admitted with a's first two pages cached
    at a's slots - as the prefix cache hands them - and a tail of its own; b's prefill lists and a decode step's.

    Without reuse the pool holds the cached prefix alone, so b indexes it from the pool. Under `rewrite` a slot of
    the second page was freed and written again by another token; under `retained` the scheduler gave a's row to
    another request before b came.
    """
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    first, second = REUSE_LENGTHS
    table = torch.full((3, first), -1, device=DEVICE, dtype=torch.int32)
    table[0] = (torch.randperm(first, device=DEVICE, generator=generator) + 1).int()
    table[2, :SHARED] = table[0, :SHARED]
    table[2, SHARED:second] = (
        torch.randperm(second - SHARED, device=DEVICE, generator=generator) + first + 1
    ).int()
    x_a = torch.randn(first, MODEL_DIM, device=DEVICE, generator=generator).bfloat16()
    x_b = torch.cat(
        [x_a[:SHARED], torch.randn(second - SHARED, MODEL_DIM, device=DEVICE, generator=generator).bfloat16()]
    )
    other = torch.randn(1, MODEL_DIM, device=DEVICE, generator=generator).bfloat16()
    slot_count = first + second + 8
    layer = LayerIndex(
        settings, maps(), slot_count, slot_count - 8, lambda: table, first, CHUNK, DEVICE, prefix_reuse=reuse
    )

    def prefill(rid, row, x, start, stop):
        lists = []
        for begin in range(start, stop, CHUNK):
            end = min(stop, begin + CHUNK)
            layer.maintain([rid], [row], [begin])
            positions = torch.arange(begin, end, device=DEVICE, dtype=torch.int32)
            lists.append(
                layer.extend(
                    x[begin:end], positions, table[row, begin:end].long(), [row], [end - begin], [begin]
                )
            )
        return torch.cat(lists)

    if reuse:
        prefill("a", 0, x_a, 0, first)
        layer.maintain(["a"], [0], [first])
        if retained:
            layer.maintain(["z"], [0], [0])
    else:
        layer.write(x_b[:SHARED], table[2, :SHARED].long())
    if rewrite:
        layer.write(other, table[2, PAGE_SIZE + 17 : PAGE_SIZE + 18].long())
    lists = prefill("b", 2, x_b, SHARED, second - 1)
    layer.maintain(["b"], [2], [second - 1])
    rows = torch.tensor([2], device=DEVICE)
    position = torch.tensor([second - 1], device=DEVICE, dtype=torch.int32)
    decoded = layer.decode(x_b[second - 1 :], position, table[2, second - 1 : second].long(), rows)
    return layer.selector.cache, sets(lists), sets(decoded)


@pytest.mark.gpu
@pytest.mark.parametrize("retained", [False, True], ids=["running", "retained"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["same", "rewritten"])
@pytest.mark.parametrize("design", ["flat", "fixed", "tree"])
def test_a_request_with_a_cached_prefix_takes_its_index_and_selects_what_indexing_it_again_selects(
    design, rewrite, retained, monkeypatch
):
    monkeypatch.setattr(hierarchy_module, "GROWTH_BUCKETS", TREE_GROWTH_BUCKETS)
    settings = SETTINGS if design == "flat" else DesignSettings.from_blocks(TREES[design], {})
    cache, lists, decoded = prefix_run(settings, True, rewrite, retained)
    cold, cold_lists, cold_decoded = prefix_run(settings, False, rewrite, retained)
    taken = 1 if rewrite else 2
    donor = cache.retained[0].pages if retained else cache.pages[0]
    assert cache.pages[2][:taken] == donor[:taken]
    assert cache.pages[2][taken] not in donor
    assert lists == cold_lists
    assert decoded == cold_decoded


class SplitRanks:
    """Tensor parallel over a grouped-query pool in one process - Qwen under tensor parallel 8, or 4

    Each rank holds the key head or heads of its share and no pool to read the others' from, so it keeps an index key
    buffer of its own. A replica of a head adds its key to the sum only on the first rank, and its queries are the
    head's group sum halved between the replicas. The group's sum is replaced by adding every rank's term, taken in
    a first pass over the ranks before the layer reads it.
    """

    def __init__(self, layout, size):
        self.layout, self.size = layout, size
        self.replicas = max(1, size // 4)
        held = 4 // size if size < 4 else 1
        width = GROUPED_SLOT.width
        self.dims = []
        self.layers = []
        for rank in range(size):
            first = rank // self.replicas if size >= 4 else rank * held
            start, stop = first * width, (first + held) * width
            self.dims.append((start, stop))
            self.layers.append(
                LayerIndex(
                    layout.grouped_settings,
                    layout.maps,
                    layout.slot_count,
                    layout.slot_count - 8,
                    lambda: layout.table,
                    layout.total,
                    CHUNK,
                    DEVICE,
                    split=KeySplit(start, stop, rank % self.replicas == 0, lambda term: term),
                )
            )

    def inputs(self, request, start, stop):
        """Each rank's share of a forward's key and grouped query"""
        layout = self.layout
        key, query = layout.key(request, start, stop), layout.queries[request][start:stop]
        return [KeyAndQuery(key[:, lo:hi], query[..., lo:hi] / self.replicas) for lo, hi in self.dims]

    def sum(self, inputs):
        """Add every rank's term of a forward, and have every layer's reduce return the sum"""
        terms = []
        for layer, rank_input in zip(self.layers, inputs, strict=True):
            record = replace(layer.projection.split, reduce=lambda term: terms.append(term) or term)
            layer.projection.split = record
            layer.projection.split_index(rank_input.key, rank_input.query)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        for layer in self.layers:
            layer.projection.split = replace(layer.projection.split, reduce=lambda term: total)


def overlap(layout, ours, theirs):
    """The share of the positions one list holds that the other holds, over every row"""
    held, shared = 0, 0
    for row_ours, row_theirs in zip(
        sets(layout.positions_of(ours)), sets(layout.positions_of(theirs)), strict=True
    ):
        held += len(row_theirs)
        shared += len(set(row_ours) & set(row_theirs))
    return shared / max(held, 1)


@pytest.mark.gpu
@pytest.mark.parametrize("size", [4, 8])
def test_a_tensor_parallel_group_selects_what_one_rank_holding_every_key_head_selects(size):
    layout = GroupedLayout([3100])
    alone, group = layout.layer, SplitRanks(layout, size)
    rows = layout.rows[:1]
    # The pool the one rank reads its keys from - the group's ranks keep an index key of their own instead
    for start, stop in ((0, 1500), (1500, 3000)):
        alone.maintain(["a"], rows, [start])
        for layer in group.layers:
            layer.maintain(["a"], rows, [start])
        positions = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        slots = layout.slots(0, start, stop)
        layout.attend(0, start, stop)
        wanted = alone.extend(layout.inputs(0, start, stop), positions, slots, rows, [stop - start], [start])
        inputs = group.inputs(0, start, stop)
        group.sum(inputs)
        for layer, rank_input in zip(group.layers, inputs, strict=True):
            lists = layer.extend(rank_input, positions, slots, rows, [stop - start], [start])
            # The sum is exact, but the one rank rounds its products to bf16 where the group's partials do not
            assert overlap(layout, lists, wanted) > 0.97
    # Every rank's buffer holds the same index keys, and they are what the pool projects to
    for layer in group.layers[1:]:
        assert torch.equal(layer.index_keys, group.layers[0].index_keys)
    stored = group.layers[0].buffer.read(layout.slots(0, 0, 3000))
    projected = alone.buffer.read(layout.slots(0, 0, 3000))
    assert torch.allclose(stored.float(), projected.float(), atol=2e-2)
    # A decode step reads every earlier key from the buffer
    alone.maintain(["a"], rows, [3000])
    for layer in group.layers:
        layer.maintain(["a"], rows, [3000])
    layout.attend(0, 3000, 3001)
    token = torch.tensor([3000], device=DEVICE, dtype=torch.int32)
    slots, request_rows = layout.slots(0, 3000, 3001), torch.tensor(rows, device=DEVICE)
    wanted = alone.decode(layout.inputs(0, 3000, 3001), token, slots, request_rows)
    inputs = group.inputs(0, 3000, 3001)
    group.sum(inputs)
    for layer, rank_input in zip(group.layers, inputs, strict=True):
        assert overlap(layout, layer.decode(rank_input, token, slots, request_rows), wanted) > 0.97
