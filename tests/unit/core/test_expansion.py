"""Expansion on the GPU against the torch reference's read, and its two forms against each other.

The expansion in order (`expand_in_order`) and by threshold (`expand_by_threshold`) choose the same keys - invariant 21 - and the
reference's `read` is the oracle for both. Every case keeps its buckets in permuted cache rows, so a kernel that
confused a bucket with its row would fail.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.budget import EarlyBudget
from lsa.core.kernels.expansion import expand_by_threshold, expand_in_order
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


@pytest.mark.parametrize(("top_p", "top_k", "repeated"), [(16, 64, False), (32, 128, False), (16, 96, True)])
def test_the_walk_reads_what_the_reference_reads(top_p, top_k, repeated):
    cache = make_cache(24, seed=1, repeated=repeated)
    queries, positions = queries_at(cache, 128, seed=2)
    # The reference scores a unit query, and the ranking is the same for any norm
    unit_queries = F.normalize(queries.float(), dim=-1).bfloat16()
    chosen = ScanSelect(128, top_p)(unit_queries, cache.centroids, visible(positions))
    walked, lengths = expand_in_order(
        physical(cache, chosen), cache.bucket_members, positions, WINDOW, SINK, top_k
    )
    assert walked.shape == (128, top_k + WINDOW + SINK)
    assert torch.equal(walked, oracle(cache, unit_queries, positions, top_p, top_k))
    assert torch.equal(lengths, (walked >= 0).sum(1).int())


def test_the_walk_stores_kv_slots_when_it_is_given_a_slot_table():
    cache = make_cache(24, seed=3)
    queries, positions = queries_at(cache, 64, seed=4)
    chosen = physical(cache, ScanSelect(64, 16)(queries, cache.centroids, visible(positions)))
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    requests = torch.full((64,), 1, device=DEVICE, dtype=torch.int64)
    plain, _ = expand_in_order(chosen, cache.bucket_members, positions, WINDOW, SINK, 64)
    mapped, _ = expand_in_order(
        chosen, cache.bucket_members, positions, WINDOW, SINK, 64, slot_table, requests
    )
    expected = torch.where(
        plain >= 0, slot_table[1].gather(0, plain.clamp_min(0).flatten().long()).view_as(plain), -1
    )
    assert torch.equal(mapped, expected)


@pytest.mark.parametrize(("top_p", "top_k", "repeated"), [(16, 64, False), (32, 128, False), (16, 96, True)])
def test_the_fill_by_threshold_reads_what_the_walk_of_the_best_picks_reads(top_p, top_k, repeated):
    cache = make_cache(24, seed=5, repeated=repeated)
    queries, positions = queries_at(cache, 128, seed=6)
    listed_keys, listed, best = listing_of(cache, queries, positions, top_p, seed=1)
    chosen = physical(cache, unpack_ids(best))
    requests = torch.full((128,), 1, device=DEVICE, dtype=torch.int64)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    for mapped in (None, slot_table):
        walked, _ = expand_in_order(
            chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, mapped, requests
        )
        filled = expand_by_threshold(
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
        assert torch.equal(filled.sort(dim=1).values, walked.sort(dim=1).values)


def test_the_fill_reads_what_the_walk_reads_from_the_scans_own_listing():
    # A cache past the scan's gate, where the listing holds the picks of every row that sees enough buckets
    cache = make_cache(8200, seed=11)
    queries, positions = queries_at(cache, 256, seed=12, first=8000 * BLOCK)
    top_p, top_k = 512, 2048
    select = ScanSelect(256, top_p)
    listing = select.listed(queries, cache.centroids, visible(positions))
    assert select.holds_top_p(listing)
    chosen = physical(cache, select.pick(listing))
    requests = torch.full((256,), 1, device=DEVICE, dtype=torch.int64)
    walked, _ = expand_in_order(chosen, cache.bucket_members, positions, WINDOW, SINK, top_k)
    filled = expand_by_threshold(
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
        WINDOW,
        SINK,
        top_k,
    )
    assert torch.equal(filled.sort(dim=1).values, walked.sort(dim=1).values)


def test_the_fill_over_keys_that_hold_cache_rows_reads_what_the_walk_reads():
    cache = make_cache(24, seed=7)
    queries, positions = queries_at(cache, 64, seed=8)
    top_p, top_k = 16, 64
    chosen = ScanSelect(64, top_p)(queries, cache.centroids, visible(positions))
    rows = physical(cache, chosen)
    # Packed keys for the cache rows, best first - any score order that is consistent with the chosen
    scores = (top_p - torch.arange(top_p, device=DEVICE, dtype=torch.float32)).expand(64, top_p).contiguous()
    keys = pack_keys(scores, rows)
    walked, _ = expand_in_order(rows, cache.bucket_members, positions, WINDOW, SINK, top_k)
    filled = expand_by_threshold(
        keys,
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
    assert torch.equal(filled.sort(dim=1).values, walked.sort(dim=1).values)


def test_a_row_early_in_the_document_reads_only_its_window_and_sinks():
    cache = make_cache(8, seed=9)
    queries, positions = queries_at(cache, 8, seed=10)
    positions[:] = torch.tensor([0, 1, 3, 4, 10, 30, 63, 64], dtype=torch.int32, device=DEVICE)
    chosen = physical(cache, ScanSelect(8, 16)(queries, cache.centroids, visible(positions)))
    lists, lengths = expand_in_order(chosen, cache.bucket_members, positions, WINDOW, SINK, 64)
    assert lengths.tolist() == [1, 2, 4, 5, 11, 31, 64, 65]
    for row, position in enumerate(positions.tolist()):
        assert lists[row, : lengths[row]].sort().values.tolist() == list(range(position + 1))
        assert (lists[row, lengths[row] :] == -1).all()


EARLY = EarlyBudget(position=800, top_k=32, top_p=8)


@pytest.mark.parametrize(("count", "repeated"), [(128, False), (128, True), (16, False)])
def test_rows_before_the_early_position_read_the_early_budget_in_both_forms(count, repeated):
    cache = make_cache(24, seed=21, repeated=repeated)
    queries, positions = queries_at(cache, count, seed=22)
    # A decode batch is a few rows and takes the wide tiles, a prefill chunk the narrow
    top_p, top_k = 16, 64
    unit_queries = F.normalize(queries.float(), dim=-1).bfloat16()
    assert (positions < EARLY.position).any() and (positions >= EARLY.position).any()
    listed_keys, listed, best = listing_of(cache, unit_queries, positions, top_p, seed=1)
    chosen = physical(cache, unpack_ids(best))
    requests = torch.full((count,), 1, device=DEVICE, dtype=torch.int64)
    slot_table = torch.stack([torch.randperm(cache.key_count + 16, device=DEVICE) for _ in range(3)]).int()
    expected = oracle(cache, unit_queries, positions, top_p, top_k, EARLY)
    walked, lengths = expand_in_order(
        chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, early=EARLY
    )
    # The list keeps the design's width and each row holds what its own budget reads
    assert walked.shape == (count, top_k + WINDOW + SINK)
    assert torch.equal(walked, expected)
    assert torch.equal(lengths, (walked >= 0).sum(1).int())
    early_rows = positions < EARLY.position
    assert (lengths[early_rows] <= EARLY.top_k + WINDOW + SINK).all()
    for mapped in (None, slot_table):
        filled = expand_by_threshold(
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
        walked_as_stored, _ = expand_in_order(
            chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, mapped, requests, early=EARLY
        )
        assert torch.equal(filled.sort(dim=1).values, walked_as_stored.sort(dim=1).values)


def test_an_early_budget_equal_to_the_designs_own_reads_as_no_early_budget():
    cache = make_cache(24, seed=23)
    queries, positions = queries_at(cache, 64, seed=24)
    unit_queries = F.normalize(queries.float(), dim=-1).bfloat16()
    top_p, top_k = 16, 64
    listed_keys, listed, best = listing_of(cache, unit_queries, positions, top_p, seed=2)
    chosen = physical(cache, unpack_ids(best))
    requests = torch.full((64,), 1, device=DEVICE, dtype=torch.int64)
    same = EarlyBudget(position=2**30, top_k=top_k, top_p=top_p)
    plain = expand_in_order(chosen, cache.bucket_members, positions, WINDOW, SINK, top_k)
    early = expand_in_order(chosen, cache.bucket_members, positions, WINDOW, SINK, top_k, early=same)
    assert torch.equal(plain[0], early[0]) and torch.equal(plain[1], early[1])
    arguments = (
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
    )
    assert torch.equal(expand_by_threshold(*arguments), expand_by_threshold(*arguments, early=same))
