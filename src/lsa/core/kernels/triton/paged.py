"""Centroid scan over paged bucket rows - the decode path of the engine index.

One kernel, `_scan_paged_kernel`: each bucket row's centroid dotted with the request's index query, over a page
table. Centroids are bf16 and the dot is an fp32 sum of their products.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan_paged_kernel(
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    pages_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    out_ptr: tl.tensor,
    max_rows: tl.tensor,
    pages_stride: tl.tensor,
    out_stride: tl.tensor,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DK: tl.constexpr,
    ROWS: tl.constexpr,
) -> None:
    """Score a block of one request's bucket rows against its index query through its pages.

    Args:
        index_query_ptr: Index queries [B, D] bf16.
        bucket_centroids_ptr: Centroid rows [rows, D] bf16.
        pages_ptr: Pages of each request [B, pages] int32.
        counts_ptr: Bucket rows each request holds [B] int32.
        out_ptr: Scores [B, max_rows] fp32, -inf for a row not held or on no page.
        max_rows: Columns scored a request.
    """
    program = tl.program_id(0)
    request = tl.program_id(1)
    rows = program * ROWS + tl.arange(0, ROWS)
    inside = rows < max_rows
    visible = rows < tl.load(counts_ptr + request)
    page = tl.load(
        pages_ptr + request.to(tl.int64) * pages_stride + rows // PAGE_SIZE, mask=visible, other=-1
    )
    valid = visible & (page >= 0)
    cache_row = tl.where(valid, page.to(tl.int64) * PAGE_SIZE + rows % PAGE_SIZE, 0)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for k in tl.static_range(D // DK):
        dims = k * DK + tl.arange(0, DK)
        q = tl.load(index_query_ptr + request * D + dims).to(tl.float32)
        c = tl.load(
            bucket_centroids_ptr + cache_row[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0
        )
        acc += tl.sum(c.to(tl.float32) * q[None, :], axis=1)
    out = tl.where(valid, acc, float("-inf"))
    tl.store(out_ptr + request.to(tl.int64) * out_stride + rows, out, mask=inside)


def scan_paged(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    pages: torch.Tensor,
    counts: torch.Tensor,
    page_size: int,
    max_rows: int,
) -> torch.Tensor:
    """Score each request's buckets against its index query, reading the centroids through its pages.

    Args:
        index_queries: One index query a request [B, index_dim] bf16, index_dim a power of two.
        bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16, `page_size` rows to a page.
        pages: Each request's pages [B, max_pages] int32, -1 for none - logical bucket `i` is row
            `i % page_size` of page `pages[:, i // page_size]`.
        counts: Visible buckets of each request [B] int32.
        page_size: Cache rows a page holds.
        max_rows: Columns of the result, at least the largest count.

    Returns:
        Scores [B, max_rows] fp32, -inf past a request's count or on a page it has not got.
    """
    batch_size, d = index_queries.shape
    out = torch.empty(batch_size, max_rows, device=index_queries.device, dtype=torch.float32)
    _scan_paged_kernel[(triton.cdiv(max_rows, 32), batch_size)](
        index_queries,
        bucket_centroids,
        pages,
        counts,
        out,
        max_rows,
        pages.stride(0),
        out.stride(0),
        PAGE_SIZE=page_size,
        D=d,
        DK=min(128, d),
        ROWS=32,
        num_warps=8,
        num_stages=2,
    )
    return out
