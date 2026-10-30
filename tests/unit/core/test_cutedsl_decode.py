"""The CuTeDSL decode select against the paged scan and the exact pick it replaces.

One GPU. Its scores are the paged scan's bit for bit, so a listing holds the exact pick's keys - every one of a row's
top-p, among others the expansion passes over - and a row whose sample misled it is listed exactly instead.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.cutedsl import decode
from lsa.core.kernels.triton.keys import unpack_scores
from lsa.core.kernels.triton.paged import scan_paged
from lsa.core.kernels.triton.top_k import RUN, TopK, keep_plan

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
PAGE_ROWS = 64


def paged(batch, buckets, seed=0, short=None, holes=False):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pages_each = -(-buckets // PAGE_ROWS)
    total = batch * pages_each
    centroids = F.normalize(
        torch.randn(total * PAGE_ROWS, 64, device=DEVICE, generator=generator), dim=-1
    ).bfloat16()
    # A zero page scores +0 against one query and -0 against another
    centroids[:PAGE_ROWS] = 0
    pages = torch.randperm(total, device=DEVICE, generator=generator).int().view(batch, pages_each)
    if holes:
        pages[:, 1::7] = -1
    counts = torch.full((batch,), buckets, device=DEVICE, dtype=torch.int32)
    if short is not None:
        counts[0] = short
    index_queries = F.normalize(torch.randn(batch, 64, device=DEVICE, generator=generator), dim=-1).bfloat16()
    return index_queries, centroids, pages, counts


def exact_keys(index_queries, centroids, pages, counts, max_rows, top_p=1024):
    scores = scan_paged(index_queries, centroids, pages, counts, PAGE_ROWS, max_rows)
    keys, listed = TopK(top_p, index_queries.shape[0], DEVICE, exact=True).listed(scores, counts)
    return [keys[b, : listed[b]].sort(descending=True).values for b in range(len(listed))]


def assert_holds_the_top_p(index_queries, centroids, pages, counts, max_rows, top_p=1024):
    keys, listed = decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, max_rows, top_p)
    for b, want in enumerate(exact_keys(index_queries, centroids, pages, counts, max_rows, top_p)):
        got = keys[b, : listed[b]].sort(descending=True).values
        assert torch.equal(got[: want.numel()], want)
        assert int(listed[b]) <= keys.shape[1]


def test_a_listing_held_whole_scores_every_bucket_as_the_paged_scan_does():
    index_queries, centroids, pages, counts = paged(3, 1500, seed=1, holes=True)
    index_queries[1] = -index_queries[0]
    keys, listed = decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, 1500, 1024)
    scores = scan_paged(index_queries, centroids, pages, counts, PAGE_ROWS, 1500)
    assert listed.tolist() == [1500] * 3
    for b in range(3):
        got = torch.zeros(1500, device=DEVICE)
        held = keys[b, :1500]
        got[0x7FFFFFFF - (held & 0xFFFFFFFF)] = unpack_scores(held)
        assert torch.equal(got.view(torch.int32), scores[b].view(torch.int32))


@pytest.mark.parametrize(
    "batch, buckets, short, holes",
    [
        (1, 1310720, None, False),
        (8, 131072, None, False),
        (4, 131072, 700, True),
        (2, 3000, None, False),
        (3, 20000, None, True),
    ],
)
def test_a_listing_holds_the_exact_picks_top_p(batch, buckets, short, holes):
    index_queries, centroids, pages, counts = paged(batch, buckets, seed=batch, short=short, holes=holes)
    assert_holds_the_top_p(index_queries, centroids, pages, counts, buckets)


def test_a_row_of_ties_overflows_and_is_listed_exactly():
    index_queries, centroids, pages, counts = paged(2, 65536, seed=6)
    centroids[:] = centroids[PAGE_ROWS + 3]
    keys, listed = decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, 65536, 1024)
    assert listed.tolist() == [1024, 1024]
    for b, want in enumerate(exact_keys(index_queries, centroids, pages, counts, 65536)):
        assert torch.equal(keys[b, :1024].sort(descending=True).values, want)


def test_a_row_its_sample_overrates_lists_too_few_and_is_listed_exactly():
    # Only the sampled buckets score high, so the threshold the sample sets leaves fewer than the top-p above it
    buckets = 20000
    samples, stride, _, _ = keep_plan(1024, buckets)
    sample = torch.arange(samples, device=DEVICE)
    sampled = (sample // RUN) * stride + sample % RUN
    sampled = sampled[sampled < buckets]
    level = torch.full((buckets,), -0.9, device=DEVICE)
    level[sampled] = torch.linspace(0.95, 0.1, len(sampled), device=DEVICE)
    centroids = torch.zeros(buckets, 64, device=DEVICE)
    centroids[:, 0] = level
    centroids[:, 1] = (1 - level**2).sqrt()
    centroids = centroids.bfloat16()
    pages = torch.arange(buckets // PAGE_ROWS + 1, device=DEVICE, dtype=torch.int32)[None]
    centroids = torch.cat([centroids, torch.zeros(PAGE_ROWS, 64, device=DEVICE).bfloat16()])
    counts = torch.tensor([buckets], device=DEVICE, dtype=torch.int32)
    index_queries = torch.zeros(1, 64, device=DEVICE).bfloat16()
    index_queries[0, 0] = 1
    keys, listed = decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, buckets, 1024)
    assert listed.tolist() == [1024]
    want = exact_keys(index_queries, centroids, pages, counts, buckets)[0]
    assert torch.equal(keys[0, :1024].sort(descending=True).values, want)


def test_the_pick_is_each_rows_best_buckets_best_first():
    index_queries, centroids, pages, counts = paged(2, 131072, seed=8, short=300)
    select = decode.DecodeSelect(1024)
    picked = select.pick(select.listed(index_queries, centroids, pages, counts, PAGE_ROWS, 131072))
    for b, want in enumerate(exact_keys(index_queries, centroids, pages, counts, 131072)):
        assert torch.equal(picked[b, : want.numel()], (0x7FFFFFFF - (want & 0xFFFFFFFF)).int())
        assert bool((picked[b, want.numel() :] == -1).all())


def test_a_row_whose_best_buckets_fill_whole_spans_lists_them_in_bucket_order():
    # The best buckets sit together, so the threshold stages whole spans of them
    buckets = 131072
    index_queries, centroids, pages, counts = paged(2, buckets, seed=9)
    for b in range(2):
        logical = torch.arange(1280, device=DEVICE) + 40000 * b
        rows = pages[b].long()[logical // PAGE_ROWS] * PAGE_ROWS + logical % PAGE_ROWS
        centroids[rows] = (index_queries[b].float() + 0.1 * centroids[rows].float()).bfloat16()
    assert_holds_the_top_p(index_queries, centroids, pages, counts, buckets)
    keys, listed = decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, buckets, 1024)
    for b in range(2):
        bucket = 0x7FFFFFFF - (keys[b, : listed[b]] & 0xFFFFFFFF)
        assert bool((bucket[1:] > bucket[:-1]).all())


def test_a_captured_listing_replays_the_same_and_follows_its_inputs():
    buckets = 131072
    index_queries, centroids, pages, counts = paged(3, buckets, seed=11, short=900)
    decode.decode_listing(index_queries, centroids, pages, counts, PAGE_ROWS, buckets, 1024)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keys, listed = decode.decode_listing(
            index_queries, centroids, pages, counts, PAGE_ROWS, buckets, 1024
        )
    lists = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        lists.append([keys[b, : listed[b]].clone() for b in range(3)])
    for again in lists[1:]:
        assert all(torch.equal(a, b) for a, b in zip(lists[0], again, strict=True))
    for b, want in enumerate(exact_keys(index_queries, centroids, pages, counts, buckets)):
        assert torch.equal(lists[0][b].sort(descending=True).values[: want.numel()], want)
    # A new query, written in place, lists its own top-p on the next replay
    index_queries.copy_(F.normalize(torch.randn(3, 64, device=DEVICE), dim=-1).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    for b, want in enumerate(exact_keys(index_queries, centroids, pages, counts, buckets)):
        got = keys[b, : listed[b]].sort(descending=True).values
        assert torch.equal(got[: want.numel()], want)


def test_rows_the_kernel_does_not_take_are_refused():
    index_queries, centroids, pages, counts = paged(1, 4096)
    assert not decode.supports(index_queries.float(), centroids, 4096)
    assert not decode.supports(index_queries, centroids, decode.MOST_SPANS * decode.SPAN + 1)
    with pytest.raises(ValueError, match="64 index dims"):
        decode.decode_listing(
            index_queries[:, :48].contiguous(),
            centroids[:, :48].contiguous(),
            pages,
            counts,
            PAGE_ROWS,
            4096,
            1024,
        )
