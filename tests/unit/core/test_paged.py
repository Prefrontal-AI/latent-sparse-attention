"""The decode scan over a paged cache against a dense reference.

The freeze and `main` never tested it: each request's buckets live in pages scattered through the cache, and the
scan has to read the right rows of the right pages. The reference gathers a request's centroids into a dense
matrix and takes one product.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.paged import scan_paged

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def paged_cache(counts, page_size, d, seed=0):
    """Bucket centroids [rows, d] bf16 laid in shuffled pages, the pages of each request [B, max_pages] int32"""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pages_needed = [-(-count // page_size) for count in counts]
    max_pages = max(pages_needed) + 1
    total_pages = sum(pages_needed) + 3
    centroids = F.normalize(
        torch.randn(total_pages * page_size, d, device=DEVICE, generator=generator), dim=-1
    )
    order = torch.randperm(total_pages, device=DEVICE, generator=generator).tolist()
    pages = torch.full((len(counts), max_pages), -1, dtype=torch.int32)
    for request, needed in enumerate(pages_needed):
        pages[request, :needed] = torch.tensor([order.pop() for _ in range(needed)], dtype=torch.int32)
    return centroids.bfloat16(), pages.to(DEVICE)


def dense_scores(index_queries, centroids, pages, counts, page_size, max_rows):
    scores = torch.full((len(counts), max_rows), -torch.inf, device=DEVICE)
    for request, count in enumerate(counts):
        logical = torch.arange(count, device=DEVICE)
        rows = pages[request].long()[logical // page_size] * page_size + logical % page_size
        scores[request, :count] = centroids[rows].float() @ index_queries[request].float()
    return scores


@pytest.mark.parametrize(("d", "page_size"), [(64, 16), (64, 8), (512, 8)])
def test_the_scan_scores_each_requests_buckets_through_its_pages(d, page_size):
    counts = [1, 37, 200, 513, 8]
    centroids, pages = paged_cache(counts, page_size, d)
    index_queries = F.normalize(torch.randn(len(counts), d, device=DEVICE), dim=-1).bfloat16()
    max_rows = 600
    got = scan_paged(
        index_queries,
        centroids,
        pages,
        torch.tensor(counts, device=DEVICE, dtype=torch.int32),
        page_size,
        max_rows,
    )
    want = dense_scores(index_queries, centroids, pages, counts, page_size, max_rows)
    assert got.shape == (len(counts), max_rows)
    assert torch.equal(got == -torch.inf, want == -torch.inf)
    seen = want > -torch.inf
    torch.testing.assert_close(got[seen], want[seen], rtol=0, atol=1e-5)


def test_a_bucket_on_a_page_the_request_has_not_got_scores_minus_infinity():
    counts = [40]
    centroids, pages = paged_cache(counts, 8, 64)
    index_queries = F.normalize(torch.randn(1, 64, device=DEVICE), dim=-1).bfloat16()
    pages[0, 2] = -1
    got = scan_paged(
        index_queries, centroids, pages, torch.tensor([40], device=DEVICE, dtype=torch.int32), 8, 48
    )
    assert (got[0, 16:24] == -torch.inf).all()
    assert (got[0, 40:] == -torch.inf).all()
    assert torch.isfinite(got[0, :16]).all() and torch.isfinite(got[0, 24:40]).all()


def test_the_scan_of_a_request_is_the_scan_of_it_in_a_batch():
    counts = [90, 300, 17]
    centroids, pages = paged_cache(counts, 16, 64, seed=1)
    index_queries = F.normalize(torch.randn(3, 64, device=DEVICE), dim=-1).bfloat16()
    all_counts = torch.tensor(counts, device=DEVICE, dtype=torch.int32)
    together = scan_paged(index_queries, centroids, pages, all_counts, 16, 320)
    alone = scan_paged(index_queries[1:2], centroids, pages[1:2], all_counts[1:2], 16, 320)
    assert torch.equal(together[1], alone[0])
