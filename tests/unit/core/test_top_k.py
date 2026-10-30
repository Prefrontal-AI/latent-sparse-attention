"""The exact top-k - invariant 30: a top-p pick is the exact top-p of its scores, and the same on every run.

The oracle is a stable descending sort of each row's scores within its length, which puts a tie at the lower column.
The long rows are the ones the engine's radix top-k parted from the exact cutoff on.
"""

import pytest
import torch
import triton

from lsa.core.kernels.triton import top_k as top_k_module
from lsa.core.kernels.triton.keys import EMPTY, unpack_ids
from lsa.core.kernels.triton.top_k import TopK, keep_plan, listing_plan, top_keys

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def ascending(columns):
    """Columns in ascending order with the -1 padding last"""
    held = torch.where(columns >= 0, columns, torch.iinfo(torch.int32).max).sort(dim=1).values
    return torch.where(held == torch.iinfo(torch.int32).max, -1, held)


def picked(scores, lengths, k, exact=False):
    """The best k columns of each row, best first - from a superset listing, or an exact one"""
    if not exact:
        return unpack_ids(top_keys(scores, lengths, k))
    pick = TopK(k, exact=True)
    pick.listed(scores, lengths)
    return unpack_ids(pick.best())


def oracle(scores, lengths, k):
    """The best k columns of each row within its length, best first and a tie to the lower column, -1 padded"""
    out = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        order = scores[row, :length].cpu().sort(descending=True, stable=True).indices[:k]
        out[row, : order.numel()] = order.to(torch.int32)
    return out


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("columns", [1000, 6000, 40000, 262144])
@pytest.mark.parametrize("k", [64, 1024, 2048])
def test_a_row_of_any_length_picks_its_exact_top_k(columns, k, exact):
    generator = torch.Generator(device=DEVICE).manual_seed(columns + k)
    scores = 0.9 + 0.05 * torch.randn(6, columns, device=DEVICE, generator=generator)
    lengths = torch.tensor(
        [columns, columns, columns - 3, columns // 2, k + 1, k // 2], dtype=torch.int32, device=DEVICE
    )
    assert torch.equal(picked(scores, lengths, k, exact).cpu(), oracle(scores, lengths, k))


@pytest.mark.parametrize("exact", [False, True])
def test_a_tie_goes_to_the_lower_column_whatever_the_run(exact):
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    # A few distinct values over many columns - nearly every cut is a tie
    scores = torch.randint(0, 5, (4, 100000), device=DEVICE, generator=generator).float()
    lengths = torch.full((4,), 100000, dtype=torch.int32, device=DEVICE)
    first = picked(scores, lengths, 1024, exact)
    assert torch.equal(first.cpu(), oracle(scores, lengths, 1024))
    for _ in range(3):
        assert torch.equal(picked(scores, lengths, 1024, exact), first)


def test_fewer_columns_than_k_leave_the_empty_key_last():
    scores = torch.randn(3, 10, device=DEVICE)
    lengths = torch.tensor([10, 4, 0], dtype=torch.int32, device=DEVICE)
    keys = top_keys(scores, lengths, 16)
    assert keys.shape == (3, 16)
    assert (keys[:, 10:] == EMPTY).all()
    assert (keys == EMPTY).sum(1).tolist() == [6, 12, 16]
    assert torch.equal(picked(scores, lengths, 16).cpu(), oracle(scores, lengths, 16))


def test_a_row_of_a_wider_matrix_is_read_at_its_own_stride():
    scores = torch.randn(4, 3000, device=DEVICE)
    lengths = torch.full((4,), 1000, dtype=torch.int32, device=DEVICE)
    assert torch.equal(picked(scores[:, :1000], lengths, 50).cpu(), oracle(scores[:, :1000], lengths, 50))


def test_the_pick_is_captured_in_a_cuda_graph_and_replays_to_the_same_columns():
    scores = torch.randn(4, 70000, device=DEVICE)
    lengths = torch.full((4,), 70000, dtype=torch.int32, device=DEVICE)
    pick = TopK(512, 4, DEVICE)
    pick.listed(scores, lengths)
    pick(scores, lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pick.listed(scores, lengths)
        best = unpack_ids(pick.best())
        kept = pick(scores, lengths)
    scores.copy_(torch.randn(4, 70000, device=DEVICE))
    graph.replay()
    assert torch.equal(best.cpu(), oracle(scores, lengths, 512))
    assert torch.equal(kept.cpu(), ascending(oracle(scores, lengths, 512)))


def test_a_listing_holds_the_best_k_of_each_row_and_counts_what_it_lists():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    scores = torch.randn(6, 50000, device=DEVICE, generator=generator)
    lengths = torch.tensor([50000, 50000, 40000, 4000, 900, 100], dtype=torch.int32, device=DEVICE)
    pick = TopK(256)
    keys, counts = pick.listed(scores, lengths)
    assert keys.shape[1] == listing_plan(256, 50000)[2]
    listed = torch.arange(keys.shape[1], device=DEVICE)[None] < counts[:, None]
    assert (keys[listed] != EMPTY).all()
    assert (counts >= torch.minimum(lengths, torch.tensor(256, device=DEVICE))).all()
    best = pick.best()
    assert torch.equal(unpack_ids(best).cpu(), oracle(scores, lengths, 256))


def test_a_decode_batch_lists_into_buffers_sized_once():
    scores = torch.randn(2, 70000, device=DEVICE)
    lengths = torch.full((2,), 70000, dtype=torch.int32, device=DEVICE)
    pick = TopK(128, 4, DEVICE)
    keys, _ = pick.listed(scores, lengths)
    assert keys.data_ptr() == pick.held[0].data_ptr()
    assert torch.equal(unpack_ids(pick.best()).cpu(), oracle(scores, lengths, 128))


def test_an_exact_listing_is_the_best_k_of_each_row_in_column_order():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    scores = torch.randn(6, 50000, device=DEVICE, generator=generator)
    lengths = torch.tensor([50000, 50000, 40000, 4000, 900, 100], dtype=torch.int32, device=DEVICE)
    keys, counts = TopK(256, exact=True).listed(scores, lengths)
    assert keys.shape == (6, 256)
    assert torch.equal(counts.cpu(), torch.minimum(lengths, torch.tensor(256, device=DEVICE)).cpu())
    listed = torch.arange(256, device=DEVICE)[None] < counts[:, None]
    assert (keys[listed] != EMPTY).all()
    assert (keys[~listed] == EMPTY).all()
    assert torch.equal(unpack_ids(keys).cpu(), ascending(oracle(scores, lengths, 256)))


def test_an_exact_decode_batch_lists_into_buffers_sized_once():
    scores = torch.randn(2, 70000, device=DEVICE)
    lengths = torch.full((2,), 70000, dtype=torch.int32, device=DEVICE)
    pick = TopK(128, 4, DEVICE, exact=True)
    keys, counts = pick.listed(scores, lengths)
    assert keys.data_ptr() == pick.held[0].data_ptr()
    assert counts.data_ptr() == pick.held[1].data_ptr()
    assert torch.equal(unpack_ids(pick.best()).cpu(), oracle(scores, lengths, 128))


class Threshold:
    """Stands in for the sample's kernel, writing the same threshold for every row"""

    def __init__(self, guess):
        self.guess = guess

    def __getitem__(self, grid):
        return lambda scores, lengths, thresholds, *args, **kwargs: thresholds.fill_(self.guess)


@pytest.mark.parametrize("guess", [float("inf"), float("-inf"), 1.0])
def test_a_wrong_estimate_costs_time_and_not_the_answer(monkeypatch, guess):
    """A row that lists too few columns or too many is searched bit by bit instead"""
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    scores = torch.randint(0, 50, (5, 90000), device=DEVICE, generator=generator).float()
    lengths = torch.tensor([90000, 90000, 70000, 3000, 500], dtype=torch.int32, device=DEVICE)
    monkeypatch.setattr(
        top_k_module,
        "_sample_thresholds",
        Threshold(guess),
    )
    assert torch.equal(picked(scores, lengths, 512).cpu(), oracle(scores, lengths, 512))


class Spy:
    """Stands in for the search kernel, keeping the totals it was handed and searching nothing"""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((args, kwargs))


@pytest.mark.parametrize("rows, columns, k", [(3, 262144, 1024), (256, 20000, 256), (64, 100000, 512)])
def test_the_estimate_lists_enough_columns_and_few_enough_that_a_normal_row_is_never_searched(
    monkeypatch, rows, columns, k
):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    scores = 0.9 + 0.05 * torch.randn(rows, columns, device=DEVICE, generator=generator)
    lengths = torch.full((rows,), columns, dtype=torch.int32, device=DEVICE)
    spy = Spy()
    monkeypatch.setattr(top_k_module, "_search_rows", spy)
    found = picked(scores, lengths, k)
    totals = spy.calls[0][0][2]
    assert int(((totals < k) | (totals > listing_plan(k, columns)[2])).sum()) == 0
    assert torch.equal(found.cpu(), oracle(scores, lengths, k))


@pytest.mark.parametrize("columns", [500, 3000, 6000, 70000])
def test_a_beam_keeps_exactly_the_best_k_columns_in_ascending_order(columns):
    generator = torch.Generator(device=DEVICE).manual_seed(columns)
    scores = torch.randint(0, 200, (6, columns), device=DEVICE, generator=generator).float()
    lengths = torch.tensor(
        [columns, columns, columns - 7, columns // 2, 300, 100], dtype=torch.int32, device=DEVICE
    )
    kept = TopK(256)(scores, lengths).cpu()
    assert torch.equal(kept, ascending(oracle(scores, lengths, 256)))


@pytest.mark.parametrize("k", [64, 512])
def test_a_few_outliers_do_not_stretch_the_estimate_into_a_search(monkeypatch, k):
    """A query matching a repeat of its own text scores a few buckets far above the rest"""
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    scores = 0.3 + 0.05 * torch.randn(8, 262144, device=DEVICE, generator=generator)
    scores[:, ::50000] = 1.0
    lengths = torch.full((8,), 262144, dtype=torch.int32, device=DEVICE)
    spy = Spy()
    monkeypatch.setattr(top_k_module, "_search_rows", spy)
    found = picked(scores, lengths, k)
    totals = spy.calls[0][0][2]
    assert int(((totals < k) | (totals > listing_plan(k, 262144)[2])).sum()) == 0
    assert torch.equal(found.cpu(), oracle(scores, lengths, k))


@pytest.mark.parametrize("rows, columns", [(2, 100000), (3, 262144), (200, 100000)])
def test_a_keep_split_across_programs_picks_what_one_program_does(monkeypatch, rows, columns):
    """Ties, a constant row, a row of -inf and short rows land on and across the slices' edges"""
    generator = torch.Generator(device=DEVICE).manual_seed(rows + columns)
    scores = torch.randint(0, 40, (rows, columns), device=DEVICE, generator=generator).float()
    scores[0] = 1.0
    scores[1 % rows, : columns // 2] = -torch.inf
    lengths = torch.full((rows,), columns, dtype=torch.int32, device=DEVICE)
    lengths[-1] = 20000
    monkeypatch.setattr(top_k_module, "SPLIT_BELOW", 1)
    whole = TopK(1000)
    alone = whole(scores, lengths)
    monkeypatch.setattr(top_k_module, "SPLIT_BELOW", 10**6)
    split = TopK(1000)
    assert torch.equal(alone, split(scores, lengths))
    assert torch.equal(alone.cpu(), ascending(oracle(scores, lengths, 1000)))


class Estimate:
    """Stands in for a keep's sample kernel, writing the same threshold for every row"""

    def __init__(self, guess):
        self.guess = guess

    def __getitem__(self, grid):
        return lambda scores, lengths, thresholds, *args, **kwargs: thresholds.fill_(self.guess)


@pytest.mark.parametrize("guess", [float("inf"), float("-inf"), 25.0])
@pytest.mark.parametrize("rows", [5, 2])
def test_a_wrong_estimate_costs_a_keep_time_and_not_the_answer(monkeypatch, guess, rows):
    """A row split across programs that lists too few columns or too many is searched across the whole row instead"""
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    scores = torch.randint(0, 50, (rows, 90000), device=DEVICE, generator=generator).float()
    lengths = torch.tensor([90000, 70000, 3000, 500, 90000][:rows], dtype=torch.int32, device=DEVICE)
    monkeypatch.setattr(top_k_module, "_estimate", Estimate(guess))
    assert torch.equal(TopK(512)(scores, lengths).cpu(), ascending(oracle(scores, lengths, 512)))


@pytest.mark.parametrize("target, capacity", [(0.01, 4096), (50.0, 1024)])
def test_a_keep_in_one_program_is_searched_when_its_listing_falls_short_or_overflows(
    monkeypatch, target, capacity
):
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    scores = torch.randint(0, 50, (70, 90000), device=DEVICE, generator=generator).float()
    lengths = torch.full((70,), 90000, dtype=torch.int32, device=DEVICE)
    lengths[:4] = torch.tensor([70000, 3000, 500, 0], dtype=torch.int32)
    planned = top_k_module.keep_plan
    monkeypatch.setattr(
        top_k_module, "keep_plan", lambda k, columns: (*planned(k, columns)[:2], target, capacity)
    )
    assert torch.equal(TopK(512)(scores, lengths).cpu(), ascending(oracle(scores, lengths, 512)))


def keep_listing_totals(scores, k):
    """What a keep's estimate lists for every row of full length, by running its kernels"""
    rows, columns = scores.shape
    samples, stride, target, capacity = keep_plan(k, columns)
    lengths = torch.full((rows,), columns, dtype=torch.int32, device=DEVICE)
    thresholds = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    top_k_module._estimate[(rows,)](
        scores,
        lengths,
        thresholds,
        scores.stride(0),
        columns,
        stride,
        target,
        SAMPLES=samples,
        RUN=top_k_module.RUN,
        K=k,
    )
    listing = torch.empty(rows, capacity, dtype=torch.int64, device=DEVICE)
    totals = torch.empty(rows, dtype=torch.int32, device=DEVICE)
    sums = torch.empty(rows, 1, dtype=torch.int32, device=DEVICE)
    top_k_module._list[(rows, 1)](
        scores,
        lengths,
        thresholds,
        sums,
        totals,
        listing,
        scores.stride(0),
        listing.stride(0),
        columns,
        triton.cdiv(columns, top_k_module.BLOCK) * top_k_module.BLOCK,
        CAPACITY=capacity,
        BLOCK=top_k_module.BLOCK,
        PARTS=1,
        COUNTING=False,
        STAGES=1,
    )
    return totals, capacity


@pytest.mark.parametrize("rows, columns, k", [(3, 262144, 1024), (256, 20000, 256), (64, 100000, 512)])
def test_a_keeps_estimate_lists_enough_columns_and_few_enough_that_a_normal_row_is_never_searched(
    rows, columns, k
):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    scores = 0.9 + 0.05 * torch.randn(rows, columns, device=DEVICE, generator=generator)
    totals, capacity = keep_listing_totals(scores, k)
    assert int(((totals < k) | (totals > capacity)).sum()) == 0
