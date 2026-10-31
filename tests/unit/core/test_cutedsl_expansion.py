"""The CuTeDSL fill by threshold against the Triton fill it replaces.

One GPU. Both find the same last key and write the buckets above it in listing order, each bucket's members in slot
order, so a row's read is the same entry for entry - order and padding included.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.budget import EarlyBudget
from lsa.core.kernels import expansion
from lsa.core.kernels.cutedsl import expansion as cutedsl_expansion
from lsa.core.kernels.scan_select import ScanSelect
from lsa.core.kernels.triton.keys import pack_keys, unpack_ids
from tests.helpers.expansion import (
    BLOCK,
    DEVICE,
    PER_BLOCK,
    SINK,
    WINDOW,
    listing_of,
    make_cache,
    oracle,
    physical,
    queries_at,
    visible,
)

pytestmark = pytest.mark.gpu


def assert_same_fill(*arguments, **options):
    expected = expansion.expand_by_threshold(*arguments, **options)
    for threads in (cutedsl_expansion.MANY_ROWS_THREADS, cutedsl_expansion.FEW_ROWS_THREADS):
        filled = cutedsl_expansion.expand_by_threshold(*arguments, **options, threads=threads)
        assert torch.equal(filled, expected)
    return filled


@pytest.mark.parametrize(("top_p", "top_k", "repeated"), [(16, 64, False), (32, 128, False), (16, 96, True)])
def test_each_row_reads_what_the_triton_fill_reads(top_p, top_k, repeated):
    cache = make_cache(24, seed=5, repeated=repeated)
    queries, positions = queries_at(cache, 128, seed=6)
    listed_keys, listed, _ = listing_of(cache, queries, positions, top_p, seed=1)
    requests = torch.full((128,), 1, device=DEVICE, dtype=torch.int64)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    for mapped in (None, slot_table):
        assert_same_fill(
            listed_keys,
            listed,
            cache.bucket_members,
            cache.bucket_sizes,
            cache.pages,
            requests,
            positions,
            PER_BLOCK,
            BLOCK,
            top_p,
            WINDOW,
            SINK,
            top_k,
            mapped,
        )


@pytest.mark.parametrize(("top_p", "top_k", "window"), [(512, 2048, WINDOW), (1024, 4096, 512)])
def test_the_scans_own_listing_reads_what_the_triton_fill_reads(top_p, top_k, window):
    cache = make_cache(8200, seed=11)
    queries, positions = queries_at(cache, 256, seed=12, first=8000 * BLOCK)
    select = ScanSelect(256, top_p)
    listing = select.listed(queries, cache.centroids, visible(positions))
    requests = torch.full((256,), 1, device=DEVICE, dtype=torch.int64)
    filled = assert_same_fill(
        listing.keys,
        listing.survivor_counts,
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        requests,
        positions,
        PER_BLOCK,
        BLOCK,
        top_p,
        window,
        SINK,
        top_k,
    )
    # Late rows fill the top-k whole
    assert ((filled >= 0).sum(1) == top_k + window + SINK).all()


def test_keys_that_hold_cache_rows_read_what_the_triton_fill_reads():
    cache = make_cache(24, seed=7)
    queries, positions = queries_at(cache, 64, seed=8)
    top_p, top_k = 16, 64
    rows = physical(cache, ScanSelect(64, top_p)(queries, cache.centroids, visible(positions)))
    scores = (top_p - torch.arange(top_p, device=DEVICE, dtype=torch.float32)).expand(64, top_p).contiguous()
    assert_same_fill(
        pack_keys(scores, rows),
        torch.full((64,), top_p, device=DEVICE, dtype=torch.int32),
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        torch.zeros(64, device=DEVICE, dtype=torch.int64),
        positions,
        PER_BLOCK,
        BLOCK,
        top_p,
        WINDOW,
        SINK,
        top_k,
        None,
        cache.block_positions,
    )


def test_a_row_early_in_the_document_reads_what_the_triton_fill_reads():
    cache = make_cache(8, seed=9)
    queries, positions = queries_at(cache, 8, seed=10)
    positions[:] = torch.tensor([0, 1, 3, 4, 10, 30, 63, 64], dtype=torch.int32, device=DEVICE)
    listed_keys, listed, _ = listing_of(cache, queries, positions, 16, seed=2)
    assert_same_fill(
        listed_keys,
        listed,
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        torch.zeros(8, device=DEVICE, dtype=torch.int64),
        positions,
        PER_BLOCK,
        BLOCK,
        16,
        WINDOW,
        SINK,
        64,
    )


def test_a_listing_the_kernel_does_not_take_is_refused():
    cache = make_cache(8, seed=13)
    keys = torch.zeros(4, 48, device=DEVICE, dtype=torch.int32)
    assert not cutedsl_expansion.supports(keys, cache.bucket_members, cache.pages)
    with pytest.raises(ValueError, match="int64 keys"):
        cutedsl_expansion.expand_by_threshold(
            keys,
            torch.zeros(4, device=DEVICE, dtype=torch.int32),
            cache.bucket_members,
            cache.bucket_sizes,
            cache.pages,
            torch.zeros(4, device=DEVICE, dtype=torch.int64),
            torch.zeros(4, device=DEVICE, dtype=torch.int32),
            PER_BLOCK,
            BLOCK,
            16,
            WINDOW,
            SINK,
            64,
        )


def test_both_thread_counts_read_the_same_around_the_rows_that_call_for_each():
    cache = make_cache(24, seed=5)
    limit = cutedsl_expansion.ROWS_PER_SM * cutedsl_expansion.multiprocessors(torch.device(DEVICE))
    for count in (limit, limit + 1):
        queries, positions = queries_at(cache, count, seed=6)
        listed_keys, listed, _ = listing_of(cache, queries, positions, 16, seed=1)
        arguments = (
            listed_keys,
            listed,
            cache.bucket_members,
            cache.bucket_sizes,
            cache.pages,
            torch.zeros(count, device=DEVICE, dtype=torch.int64),
            positions,
            PER_BLOCK,
            BLOCK,
            16,
            WINDOW,
            SINK,
            64,
        )
        many = cutedsl_expansion.expand_by_threshold(*arguments, threads=cutedsl_expansion.MANY_ROWS_THREADS)
        few = cutedsl_expansion.expand_by_threshold(*arguments, threads=cutedsl_expansion.FEW_ROWS_THREADS)
        chosen = cutedsl_expansion.expand_by_threshold(*arguments)
        assert torch.equal(many, few)
        assert torch.equal(chosen, few)
        assert cutedsl_expansion.threads_for(count, DEVICE) == (
            cutedsl_expansion.FEW_ROWS_THREADS if count <= limit else cutedsl_expansion.MANY_ROWS_THREADS
        )


def test_the_fill_takes_cutedsl_when_asked(monkeypatch):
    calls = []
    fill = cutedsl_expansion.expand_by_threshold
    monkeypatch.setattr(
        cutedsl_expansion, "expand_by_threshold", lambda *arguments: calls.append(1) or fill(*arguments)
    )
    cache = make_cache(24, seed=15)
    queries, positions = queries_at(cache, 32, seed=16)
    listed_keys, listed, _ = listing_of(cache, queries, positions, 16, seed=3)
    arguments = (
        listed_keys,
        listed,
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        torch.zeros(32, device=DEVICE, dtype=torch.int64),
        positions,
        PER_BLOCK,
        BLOCK,
        16,
        WINDOW,
        SINK,
        64,
    )
    by_triton = expansion.expand_by_threshold(*arguments)
    assert not calls
    by_cutedsl = expansion.expand_by_threshold(*arguments, kernels="cutedsl")
    assert calls
    assert torch.equal(by_triton, by_cutedsl)


def picks_of(cache, rows, top_p, seed, holes=0.1):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    selected = torch.randint(
        0, cache.bucket_members.shape[0], (rows, top_p), device=DEVICE, generator=generator
    )
    selected[torch.rand(selected.shape, device=DEVICE, generator=generator) < holes] = -1
    return selected.int()


def assert_same_read(*arguments, **options):
    expected = expansion.expand_in_order(*arguments, **options)
    for threads in (None, cutedsl_expansion.MANY_ROWS_THREADS, cutedsl_expansion.FEW_ROWS_THREADS):
        read = cutedsl_expansion.expand_in_order(*arguments, **options, threads=threads)
        assert torch.equal(read[0], expected[0])
        assert torch.equal(read[1], expected[1])


@pytest.mark.parametrize("mapped", [False, True])
def test_a_row_reads_its_picks_in_order_as_the_triton_read_does(mapped):
    cache = make_cache(24, seed=5)
    queries, positions = queries_at(cache, 128, seed=6)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    slot_rows = torch.randint(0, 3, (128,), device=DEVICE)
    assert_same_read(
        picks_of(cache, 128, 16, seed=1),
        cache.bucket_members,
        positions,
        WINDOW,
        SINK,
        64,
        slot_table if mapped else None,
        slot_rows if mapped else None,
    )


def test_picks_at_serving_shapes_read_as_the_triton_read_does():
    cache = make_cache(8200, seed=11)
    queries, positions = queries_at(cache, 256, seed=12, first=8000 * BLOCK)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    slot_rows = torch.randint(0, 3, (256,), device=DEVICE)
    selected = picks_of(cache, 256, 1024, seed=2)
    # A row whose picks run out before the top-k fills, and one with no picks at all
    selected[0, 40:] = -1
    selected[1] = -1
    assert_same_read(selected, cache.bucket_members, positions, 512, SINK, 4096, slot_table, slot_rows)


def test_a_row_early_in_the_document_reads_its_picks_as_the_triton_read_does():
    cache = make_cache(8, seed=9)
    queries, positions = queries_at(cache, 8, seed=10)
    positions[:] = torch.tensor([0, 1, 3, 4, 10, 30, 63, 64], dtype=torch.int32, device=DEVICE)
    assert_same_read(picks_of(cache, 8, 16, seed=3), cache.bucket_members, positions, WINDOW, SINK, 64)


def test_both_thread_counts_read_picks_the_same_around_the_rows_that_call_for_each():
    cache = make_cache(24, seed=5)
    limit = cutedsl_expansion.ROWS_PER_SM * cutedsl_expansion.multiprocessors(torch.device(DEVICE))
    for count in (limit, limit + 1):
        queries, positions = queries_at(cache, count, seed=6)
        arguments = (picks_of(cache, count, 16, seed=1), cache.bucket_members, positions, WINDOW, SINK, 64)
        many = cutedsl_expansion.expand_in_order(*arguments, threads=cutedsl_expansion.MANY_ROWS_THREADS)
        few = cutedsl_expansion.expand_in_order(*arguments, threads=cutedsl_expansion.FEW_ROWS_THREADS)
        chosen = cutedsl_expansion.expand_in_order(*arguments)
        assert torch.equal(many[0], few[0]) and torch.equal(many[1], few[1])
        assert torch.equal(chosen[0], few[0]) and torch.equal(chosen[1], few[1])
        assert torch.equal(chosen[0], expansion.expand_in_order(*arguments)[0])


def test_the_in_order_read_takes_cutedsl_when_asked(monkeypatch):
    calls = []
    read = cutedsl_expansion.expand_in_order
    monkeypatch.setattr(
        cutedsl_expansion, "expand_in_order", lambda *arguments: calls.append(1) or read(*arguments)
    )
    cache = make_cache(24, seed=15)
    queries, positions = queries_at(cache, 32, seed=16)
    arguments = (picks_of(cache, 32, 16, seed=4), cache.bucket_members, positions, WINDOW, SINK, 64)
    by_triton = expansion.expand_in_order(*arguments)
    assert not calls
    by_cutedsl = expansion.expand_in_order(*arguments, kernels="cutedsl")
    assert calls
    assert torch.equal(by_triton[0], by_cutedsl[0])


EARLY = EarlyBudget(position=800, top_k=32, top_p=8)


@pytest.mark.parametrize(("count", "repeated"), [(128, False), (128, True), (16, False)])
def test_rows_before_the_early_position_read_the_early_budget_as_the_triton_fill_does(count, repeated):
    cache = make_cache(24, seed=21, repeated=repeated)
    queries, positions = queries_at(cache, count, seed=22)
    top_p, top_k = 16, 64
    unit_queries = F.normalize(queries.float(), dim=-1).bfloat16()
    assert (positions < EARLY.position).any() and (positions >= EARLY.position).any()
    listed_keys, listed, best = listing_of(cache, unit_queries, positions, top_p, seed=1)
    requests = torch.full((count,), 1, device=DEVICE, dtype=torch.int64)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    for mapped in (None, slot_table):
        assert_same_fill(
            listed_keys,
            listed,
            cache.bucket_members,
            cache.bucket_sizes,
            cache.pages,
            requests,
            positions,
            PER_BLOCK,
            BLOCK,
            top_p,
            WINDOW,
            SINK,
            top_k,
            mapped,
            early=EARLY,
        )
    # The picks in rank order read the reference's early budget, entry for entry
    chosen = physical(cache, unpack_ids(best))
    read, lengths = cutedsl_expansion.expand_in_order(
        chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, early=EARLY
    )
    assert torch.equal(read, oracle(cache, unit_queries, positions, top_p, top_k, EARLY))
    assert torch.equal(lengths, (read >= 0).sum(1).int())
    assert_same_read(
        chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, slot_table, requests, early=EARLY
    )
    # The dispatcher hands the early budget to either kernel
    fill = expansion.expand_by_threshold(
        listed_keys,
        listed,
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        requests,
        positions,
        PER_BLOCK,
        BLOCK,
        top_p,
        WINDOW,
        SINK,
        top_k,
        early=EARLY,
        kernels="cutedsl",
    )
    assert torch.equal(fill.sort(dim=1).values, read.sort(dim=1).values)


def test_an_early_budget_at_the_serving_shapes_reads_as_the_triton_fill_does():
    cache = make_cache(8200, seed=11)
    queries, positions = queries_at(cache, 256, seed=12, first=7000 * BLOCK)
    top_p, top_k = 1024, 4096
    early = EarlyBudget(position=7800 * BLOCK, top_k=2048, top_p=512)
    assert (positions < early.position).any() and (positions >= early.position).any()
    select = ScanSelect(256, top_p)
    listing = select.listed(queries, cache.centroids, visible(positions))
    assert select.holds_top_p(listing)
    requests = torch.full((256,), 1, device=DEVICE, dtype=torch.int64)
    assert_same_fill(
        listing.keys,
        listing.survivor_counts,
        cache.bucket_members,
        cache.bucket_sizes,
        cache.pages,
        requests,
        positions,
        PER_BLOCK,
        BLOCK,
        top_p,
        512,
        SINK,
        top_k,
        early=early,
    )
