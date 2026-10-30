"""Beam search between levels - scoring the children of the node buckets a level kept, in one launch each.

A level scores its open node buckets, then the children of the node buckets the level above kept, into one row of
scores a query; a top-k keeps the best columns. A node bucket keeps its children as a row of `node_bucket_size` row
numbers of the level below - cache rows under level 1 - wherever those rows lie. The children a level carries are
those rows, `node_bucket_size` a kept node bucket, side by side. A kept column is an open node bucket, whose own
children are its row of `open_children`, or a carried child, whose children are its row of `closed_children` - and
a column the top-k returned from under -inf, or from under a node bucket that was never there, keeps nothing.
"""

import torch
import triton
import triton.language as tl

from lsa.core.kernels.triton.keys import EMPTY, key_bucket, ordered_bits

# Node centroids stored in fp8 hold this times the unit row they stand for, to keep small components in range
NODE_CENTROID_SCALE = 16.0
EMPTY_KEY: tl.constexpr = tl.constexpr(EMPTY)


@triton.jit
def _kept_children_kernel(
    scores_ptr: tl.tensor,
    kept_ptr: tl.tensor,
    open_children_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    closed_children_ptr: tl.tensor,
    out_ptr: tl.tensor,
    score_stride: tl.tensor,
    kept_stride: tl.tensor,
    carried_stride: tl.tensor,
    open_count: tl.tensor,
    beam_width: tl.tensor,
    width: tl.tensor,
    NODE_BUCKET_SIZE: tl.constexpr,
    CARRIED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Gather the children of the node buckets each query kept, one program a block of kept columns.

    Args:
        scores_ptr: The level's scores [S, width] fp32.
        kept_ptr: The top-k's columns of `scores_ptr` [S, beam_width] int32, -1 padded.
        open_children_ptr: Children of the open node buckets [open_count, NODE_BUCKET_SIZE] int32.
        carried_ptr: Rows scored after the open node buckets [S, carried] int32 - read only if CARRIED.
        closed_children_ptr: Children of closed node buckets [N, NODE_BUCKET_SIZE] int32 - read if CARRIED.
        out_ptr: Children of the kept node buckets [S, beam_width, NODE_BUCKET_SIZE] int32, -1 for none.
        open_count: Open node buckets scored, the first columns.
        width: Columns scored.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = slots < beam_width
    column = tl.load(kept_ptr + row * kept_stride + slots, mask=inside, other=-1).to(tl.int64)
    valid = inside & (column >= 0) & (column < width)
    score = tl.load(scores_ptr + row * score_stride + tl.maximum(column, 0), mask=valid, other=float("-inf"))
    valid &= score > float("-inf")
    from_open = column < open_count
    # The kept node bucket's children row, from the open table or the closed one
    offsets = tl.arange(0, NODE_BUCKET_SIZE)
    children = tl.load(
        open_children_ptr + tl.maximum(column, 0)[:, None] * NODE_BUCKET_SIZE + offsets[None, :],
        mask=(valid & from_open)[:, None],
        other=-1,
    )
    if CARRIED:
        node_bucket = tl.load(
            carried_ptr + row * carried_stride + tl.maximum(column - open_count, 0),
            mask=valid & ~from_open,
            other=-1,
        ).to(tl.int64)
        valid &= from_open | (node_bucket >= 0)
        closed = tl.load(
            closed_children_ptr + tl.maximum(node_bucket, 0)[:, None] * NODE_BUCKET_SIZE + offsets[None, :],
            mask=(valid & ~from_open)[:, None],
            other=-1,
        )
        children = tl.where(from_open[:, None], children, closed)
    tl.store(
        out_ptr + (row * beam_width + slots)[:, None] * NODE_BUCKET_SIZE + offsets[None, :],
        tl.where(valid[:, None], children, -1),
        mask=inside[:, None],
    )


def kept_children(
    scores: torch.Tensor,
    kept: torch.Tensor,
    open_count: int,
    open_children: torch.Tensor,
    carried_children: torch.Tensor | None,
    closed_children: torch.Tensor | None,
    node_bucket_size: int,
) -> torch.Tensor:
    """Name the children of the node buckets each query kept.

    Args:
        scores: The level's scores [S, open_count + carried] fp32, the open node buckets first.
        kept: The top-k's columns of `scores` [S, beam_width] int32, -1 padded.
        open_count: Open node buckets the level scored.
        open_children: Their children [open_count, node_bucket_size] int32.
        carried_children: The rows the level scored after its open node buckets [S, carried] int32, -1 padded, or
            None at the top - the children of the node buckets the level above kept.
        closed_children: The children of the level's closed node buckets [N, node_bucket_size] int32, which
            `carried_children` index - None at the top.
        node_bucket_size: Children a node bucket holds.

    Returns:
        The children of the kept node buckets [S, beam_width * node_bucket_size] int32, a kept bucket's side by
        side, -1 where a query kept nothing.
    """
    count, beam_width = kept.shape
    out = torch.empty(count, beam_width * node_bucket_size, dtype=torch.int32, device=kept.device)
    block = min(triton.next_power_of_2(beam_width), 64)
    _kept_children_kernel[(count, triton.cdiv(beam_width, block))](
        scores,
        kept,
        open_children,
        kept if carried_children is None else carried_children,
        open_children if closed_children is None else closed_children,
        out,
        scores.stride(0),
        kept.stride(0),
        0 if carried_children is None else carried_children.stride(0),
        open_count,
        beam_width,
        scores.shape[1],
        NODE_BUCKET_SIZE=node_bucket_size,
        CARRIED=carried_children is not None,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _children_of_kernel(
    node_buckets_ptr: tl.tensor,
    children_ptr: tl.tensor,
    out_ptr: tl.tensor,
    node_buckets_stride: tl.tensor,
    width: tl.tensor,
    NODE_BUCKET_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Gather the children of the node buckets each query picked, one program a block of picks.

    Args:
        node_buckets_ptr: Picked rows of `children_ptr` [S, width] int32 or int64, -1 for none.
        children_ptr: The node buckets' children [N, NODE_BUCKET_SIZE] int32.
        out_ptr: Their children [S, width, NODE_BUCKET_SIZE] int32, -1 under a -1.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = slots < width
    node_bucket = tl.load(node_buckets_ptr + row * node_buckets_stride + slots, mask=inside, other=-1).to(
        tl.int64
    )
    offsets = tl.arange(0, NODE_BUCKET_SIZE)
    children = tl.load(
        children_ptr + tl.maximum(node_bucket, 0)[:, None] * NODE_BUCKET_SIZE + offsets[None, :],
        mask=(node_bucket >= 0)[:, None],
        other=-1,
    )
    tl.store(
        out_ptr + (row * width + slots)[:, None] * NODE_BUCKET_SIZE + offsets[None, :],
        children,
        mask=inside[:, None],
    )


def children_of(node_buckets: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
    """Name the children of the node buckets each query picked, by their rows.

    Args:
        node_buckets: Picked rows of `children` [S, W] - int32 or int64, -1 for none.
        children: The node buckets' children [N, node_bucket_size] int32.

    Returns:
        Their children [S, W * node_bucket_size] int32, a picked bucket's side by side, -1 under a -1.
    """
    count, width = node_buckets.shape
    size = children.shape[1]
    out = torch.empty(count, width * size, dtype=torch.int32, device=node_buckets.device)
    block = min(triton.next_power_of_2(width), 64)
    _children_of_kernel[(count, triton.cdiv(width, block))](
        node_buckets,
        children,
        out,
        node_buckets.stride(0),
        width,
        NODE_BUCKET_SIZE=size,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _bucket_keys_kernel(
    keys_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    pages_ptr: tl.tensor,
    out_ptr: tl.tensor,
    key_stride: tl.tensor,
    carried_stride: tl.tensor,
    open_count: tl.tensor,
    covered_buckets: tl.tensor,
    PAGE_SIZE: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Replace the picked columns in keys with cache rows, keeping the scores.

    An uncovered bucket is a cache row through the request's pages, a carried child the row its table names.

    Args:
        keys_ptr: Picked columns packed with their scores [S, WIDTH] int64.
        counts_ptr: Keys each row holds [S] int32.
        carried_ptr: Cache rows scored after the uncovered buckets [S, carried] int32 - read only if CARRIED.
        pages_ptr: The request's pages [max_pages] int32.
        out_ptr: Scores packed with cache rows [S, WIDTH] int64, EMPTY for an invalid key.
        open_count: Uncovered buckets scored.
        covered_buckets: Buckets under a node bucket, where the logical bucket numbers start.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = slots < WIDTH
    key = tl.load(keys_ptr + row * key_stride + slots, mask=inside, other=EMPTY_KEY)
    column = key_bucket(key).to(tl.int64)
    never = ordered_bits(tl.full([], float("-inf"), tl.float32))
    valid = (
        inside & (slots < tl.load(counts_ptr + row)) & (key != EMPTY_KEY) & ((key >> 32).to(tl.int32) > never)
    )
    from_open = column < open_count
    logical = covered_buckets + column
    page = tl.load(pages_ptr + logical // PAGE_SIZE, mask=valid & from_open, other=-1)
    cache_row = page * PAGE_SIZE + logical % PAGE_SIZE
    valid &= ~from_open | (page >= 0)
    if CARRIED:
        child = tl.load(
            carried_ptr + row * carried_stride + tl.maximum(column - open_count, 0),
            mask=valid & ~from_open,
            other=-1,
        ).to(tl.int64)
        valid &= from_open | (child >= 0)
        cache_row = tl.where(from_open, cache_row, child)
    packed = ((key >> 32) << 32) | (0x7FFFFFFF - cache_row).to(tl.int64)
    tl.store(out_ptr + row * WIDTH + slots, tl.where(valid, packed, EMPTY_KEY), mask=inside)


def bucket_keys(
    keys: torch.Tensor,
    counts: torch.Tensor,
    carried_children: torch.Tensor | None,
    pages: torch.Tensor,
    open_count: int,
    covered_buckets: int,
    page_size: int,
) -> torch.Tensor:
    """Name the cache rows of the buckets each query picked, keeping their scores in the keys.

    Args:
        keys: The picked columns of the last level's scores packed with them [S, W] int64.
        counts: Keys each row holds [S] int32, at most W.
        carried_children: The cache rows the last level scored after the uncovered buckets [S, carried] int32,
            -1 padded, or None.
        pages: The request's pages [max_pages] int32.
        open_count: Uncovered buckets scored, from logical bucket `covered_buckets`.
        covered_buckets: Buckets under a node bucket.
        page_size: Cache rows a page holds.

    Returns:
        The scores packed with the cache rows [S, W] int64, the empty key for a column that is no bucket, was scored
        -inf or was never picked.
    """
    count, width = keys.shape
    out = torch.empty(count, width, dtype=torch.int64, device=keys.device)
    block = min(triton.next_power_of_2(width), 256)
    _bucket_keys_kernel[(count, triton.cdiv(width, block))](
        keys,
        counts,
        keys if carried_children is None else carried_children,
        pages,
        out,
        keys.stride(0),
        0 if carried_children is None else carried_children.stride(0),
        open_count,
        covered_buckets,
        PAGE_SIZE=page_size,
        CARRIED=carried_children is not None,
        WIDTH=width,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _score_rows(
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    child: tl.tensor,
    row: tl.tensor,
    rescale: tl.tensor,
    D: tl.constexpr,
) -> tl.tensor:
    """Return the scores of a block of centroid rows against a query - -inf under a -1 row.

    Args:
        index_query_ptr: Index queries [S, D] bf16.
        bucket_centroids_ptr: Centroid rows [rows, D] bf16 or fp8.
        child: The block's centroid rows [BLOCK] int64, -1 for none.
        row: The query.
        rescale: Factor undoing the fp8 centroid scale.
    """
    held = child >= 0
    dims = tl.arange(0, D)
    query = tl.load(index_query_ptr + row * D + dims).to(tl.float32)
    centroid = tl.load(
        bucket_centroids_ptr + child[:, None] * D + dims[None, :], mask=held[:, None], other=0.0
    )
    score = tl.sum(centroid.to(tl.float32) * query[None, :], axis=1) * rescale
    return tl.where(held, score, float("-inf"))


@triton.jit
def _score_children_kernel(
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    out_ptr: tl.tensor,
    rescale: tl.tensor,
    rows_stride: tl.tensor,
    out_stride: tl.tensor,
    D: tl.constexpr,
    COLUMNS: tl.constexpr,
    BOUNDED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Score each query's children against its index query, one program a block of children.

    Args:
        index_query_ptr: Index queries [S, D] bf16.
        bucket_centroids_ptr: Centroid rows [rows, D] bf16 or fp8.
        rows_ptr: Children of each query [S, COLUMNS] int32, -1 for none.
        lengths_ptr: Columns each query holds [S] int32 - read if BOUNDED.
        out_ptr: Scores [S, COLUMNS] fp32, -inf under a -1, unwritten past a query's length if BOUNDED.
        rescale: Factor undoing the fp8 centroid scale.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    columns = COLUMNS
    if BOUNDED:
        columns = tl.minimum(tl.load(lengths_ptr + row), COLUMNS)
    child = tl.load(rows_ptr + row * rows_stride + slots, mask=slots < columns, other=-1).to(tl.int64)
    score = _score_rows(index_query_ptr, bucket_centroids_ptr, child, row, rescale, D)
    tl.store(out_ptr + row * out_stride + slots, score, mask=slots < columns)


def score_children(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    children: torch.Tensor,
    out: torch.Tensor | None = None,
    centroid_scale: float = 1.0,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score the children of each query's kept node buckets, as the flat scan scores a bucket.

    Args:
        index_queries: One index query a row [S, index_dim] bf16.
        bucket_centroids: The centroid rows [rows, index_dim] the children name - bf16, or fp8 holding `centroid_scale` times the
            row.
        children: The rows of each query's children [S, C] int32, -1 for none, wherever the rows lie.
        out: A buffer for the scores [S, C] fp32, a view of rows of any stride.
        centroid_scale: What fp8 centroids are multiplied by on storing, `NODE_CENTROID_SCALE`; 1 for bf16.
        lengths: Columns each query holds [S] int32 - a query's scores past its length are left unwritten - or None for
            all of them.

    Returns:
        Scores [S, C] fp32, the fp32 sum of the centroid's products with the query, -inf under a -1 row.
    """
    count, width = children.shape
    if out is None:
        out = torch.empty(count, width, device=index_queries.device, dtype=torch.float32)
    block = 64
    _score_children_kernel[(count, triton.cdiv(width, block))](
        index_queries,
        bucket_centroids,
        children,
        children if lengths is None else lengths,
        out,
        1.0 / centroid_scale,
        children.stride(0),
        out.stride(0),
        D=index_queries.shape[1],
        COLUMNS=width,
        BOUNDED=lengths is not None,
        BLOCK=block,
    )
    return out


@triton.jit
def _level_pooled(
    pages_ptr: tl.tensor,
    made_ptr: tl.tensor,
    closed_ptr: tl.tensor,
    start_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    query: tl.tensor,
    columns: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    level: tl.tensor,
    max_pages: tl.tensor,
    PAGE: tl.constexpr,
    CARRIED: tl.constexpr,
) -> tl.tensor:
    """Return the pooled rows of a block of one request's columns at a level.

    Args:
        pages_ptr: Each request's node bucket pages at the level.
        made_ptr: Node buckets each request has made at the level.
        closed_ptr: Node buckets each request has closed at the level.
        start_ptr: Each request's start level.
        rows_ptr: The batch's request table rows.
        carried_ptr: Creation indices the level above kept the children of.
        query: The query of the program.
        columns: The block's columns.
        pages_stride: Stride of a request's pages.
        carried_stride: Stride of a query's carried indices.
        level: The level, from 1.
        max_pages: Pages a request holds at most.
        PAGE: Node buckets a page holds.
        CARRIED: Columns a query carries.

    Returns:
        The pooled rows of the columns, -1 for none.
    """
    row = tl.load(rows_ptr + query)
    made = tl.load(made_ptr + row)
    closed = tl.load(closed_ptr + row)
    start = tl.load(start_ptr + row)
    opened = made - closed
    from_start = start == level
    below = start > level
    # A start level scores its open node buckets then its closed ones, a level below its open ones then the carried
    in_open = columns < opened
    carried_column = columns - opened
    carried_held = below & ~in_open & (carried_column < CARRIED)
    carried_index = tl.load(
        carried_ptr + query * carried_stride + carried_column, mask=carried_held, other=-1
    )
    creation = tl.where(in_open, closed + columns, tl.where(from_start, carried_column, carried_index))
    creation = tl.where((from_start & (columns < made)) | (below & in_open) | carried_held, creation, -1)
    page = tl.load(
        pages_ptr + row * pages_stride + tl.minimum(tl.maximum(creation, 0) // PAGE, max_pages - 1),
        mask=creation >= 0,
        other=-1,
    )
    return tl.where((creation >= 0) & (page >= 0), page * PAGE + creation % PAGE, -1)


@triton.jit
def _level_length(
    made_ptr: tl.tensor,
    closed_ptr: tl.tensor,
    start_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    query: tl.tensor,
    carried_stride: tl.tensor,
    level: tl.tensor,
    WIDTH: tl.constexpr,
    CARRIED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> tl.tensor:
    """Return the columns one request holds at a level up to its last valid one.

    Args:
        made_ptr: Node buckets each request has made at the level.
        closed_ptr: Node buckets each request has closed at the level.
        start_ptr: Each request's start level.
        rows_ptr: The batch's request table rows.
        carried_ptr: Creation indices the level above kept the children of.
        query: The query of the program.
        carried_stride: Stride of a query's carried indices.
        level: The level, from 1.
        WIDTH: Columns a row holds at most.
        CARRIED: Columns a query carries.
        BLOCK: Carried columns read at a time.

    Returns:
        The columns the request holds.
    """
    row = tl.load(rows_ptr + query)
    made = tl.load(made_ptr + row)
    opened = made - tl.load(closed_ptr + row)
    start = tl.load(start_ptr + row)
    # The length is the last carried index that holds a child, after the open ones
    last = tl.zeros([BLOCK], dtype=tl.int32)
    for base in tl.static_range(0, CARRIED, BLOCK):
        index = base + tl.arange(0, BLOCK)
        held = tl.load(carried_ptr + query * carried_stride + index, mask=index < CARRIED, other=-1) >= 0
        last = tl.maximum(last, tl.where(held, index + 1, 0))
    length = tl.where(start == level, made, tl.where(start > level, opened + tl.max(last, axis=0), 0))
    return tl.minimum(length, WIDTH)


@triton.jit
def _level_candidates_kernel(
    pages_ptr: tl.tensor,
    made_ptr: tl.tensor,
    closed_ptr: tl.tensor,
    start_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    out_stride: tl.tensor,
    level: tl.tensor,
    max_pages: tl.tensor,
    PAGE: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Write the node buckets one request scores at a level, as pooled rows, valid ones first, one program a block.

    Args:
        pages_ptr: The level's page table [R, max_pages] int32, -1 for none.
        made_ptr: Node buckets each request made of the level [R] int32.
        closed_ptr: Its closed ones [R] int32.
        start_ptr: Each request's start level [R] int32.
        rows_ptr: The batch's request table rows [B] int64.
        carried_ptr: Creation indices the level above kept the children of [B, CARRIED] int32, -1 for none.
        out_ptr: Pooled rows [B, WIDTH] int32, -1 for none.
        lengths_ptr: Columns each query holds up to its last valid one [B] int32.
        level: The level, from 1.
    """
    query = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pooled = _level_pooled(
        pages_ptr,
        made_ptr,
        closed_ptr,
        start_ptr,
        rows_ptr,
        carried_ptr,
        query,
        columns,
        pages_stride,
        carried_stride,
        level,
        max_pages,
        PAGE,
        CARRIED,
    )
    tl.store(out_ptr + query * out_stride + columns, pooled, mask=columns < WIDTH)
    if tl.program_id(1) == 0:
        tl.store(
            lengths_ptr + query,
            _level_length(
                made_ptr,
                closed_ptr,
                start_ptr,
                rows_ptr,
                carried_ptr,
                query,
                carried_stride,
                level,
                WIDTH,
                CARRIED,
                BLOCK,
            ),
        )


def level_candidates(
    pages: torch.Tensor,
    made: torch.Tensor,
    closed: torch.Tensor,
    start: torch.Tensor,
    rows: torch.Tensor,
    carried: torch.Tensor,
    level: int,
    page_size: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Name the node buckets each request scores at a level as rows of the level's pool, in the eager beam's order.

    A request whose start level is this one scores all of its node buckets, the open ones (creation `closed` to
    `made`) first and the closed ones (0 to `closed`) after, and drops what the level above carried. A request below
    its start scores its open node buckets, then the carried ones. A request above its start scores none. Creation
    indices map to pooled rows through the request's pages.

    Args:
        pages: The level's page table [R, max_pages] int32, -1 for none.
        made: Node buckets each request made of the level [R] int32.
        closed: Its closed ones [R] int32.
        start: Each request's start level [R] int32, 0 for none.
        rows: The batch's request table rows [B] int64.
        carried: Creation indices the level above kept the children of [B, C] int32, -1 for none.
        level: The level, from 1.
        page_size: Node buckets a page holds.
        width: Columns of the result.

    Returns:
        Pooled rows [B, width] int32, valid ones first and -1 for none, and the columns each query holds up to its last
        valid one [B] int32 - 0 for a request above its start.
    """
    count = len(rows)
    out = torch.empty(count, width, device=rows.device, dtype=torch.int32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    block = 1024
    _level_candidates_kernel[(count, triton.cdiv(width, block))](
        pages,
        made,
        closed,
        start,
        rows,
        carried,
        out,
        lengths,
        pages.stride(0),
        carried.stride(0),
        out.stride(0),
        level,
        pages.shape[1],
        PAGE=page_size,
        CARRIED=carried.shape[1],
        WIDTH=width,
        BLOCK=block,
        num_warps=4,
    )
    return out, lengths


@triton.jit
def _level_scores_kernel(
    pages_ptr: tl.tensor,
    made_ptr: tl.tensor,
    closed_ptr: tl.tensor,
    start_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    candidates_ptr: tl.tensor,
    scores_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    out_stride: tl.tensor,
    rescale: tl.tensor,
    level: tl.tensor,
    max_pages: tl.tensor,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    LENGTH_BLOCK: tl.constexpr,
) -> None:
    """Name the node buckets one request scores at a level and score them, one program a block of columns.

    One program past the blocks writes the query's length. Scores are those of `_score_children_kernel` bit for bit,
    as the block and the dims are the same.

    Args:
        pages_ptr: The level's page table [R, max_pages] int32, -1 for none.
        made_ptr: Node buckets each request made of the level [R] int32.
        closed_ptr: Its closed ones [R] int32.
        start_ptr: Each request's start level [R] int32.
        rows_ptr: The batch's request table rows [B] int64.
        carried_ptr: Creation indices the level above kept the children of [B, CARRIED] int32, -1 for none.
        index_query_ptr: Index queries [B, D] bf16.
        bucket_centroids_ptr: The level's centroid rows [rows, D] bf16 or fp8.
        candidates_ptr: Pooled rows [B, WIDTH] int32, -1 for none.
        scores_ptr: Scores [B, WIDTH] fp32, -inf under a -1.
        lengths_ptr: Columns each query holds up to its last valid one [B] int32.
        rescale: Factor undoing the fp8 centroid scale.
        level: The level, from 1.
    """
    query = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) < tl.num_programs(1) - 1:
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        pooled = _level_pooled(
            pages_ptr,
            made_ptr,
            closed_ptr,
            start_ptr,
            rows_ptr,
            carried_ptr,
            query,
            columns,
            pages_stride,
            carried_stride,
            level,
            max_pages,
            PAGE,
            CARRIED,
        )
        inside = columns < WIDTH
        tl.store(candidates_ptr + query * out_stride + columns, pooled, mask=inside)
        score = _score_rows(index_query_ptr, bucket_centroids_ptr, pooled.to(tl.int64), query, rescale, D)
        tl.store(scores_ptr + query * out_stride + columns, score, mask=inside)
    else:
        tl.store(
            lengths_ptr + query,
            _level_length(
                made_ptr,
                closed_ptr,
                start_ptr,
                rows_ptr,
                carried_ptr,
                query,
                carried_stride,
                level,
                WIDTH,
                CARRIED,
                LENGTH_BLOCK,
            ),
        )


@triton.jit
def _level_children_kernel(
    pages_ptr: tl.tensor,
    made_ptr: tl.tensor,
    closed_ptr: tl.tensor,
    start_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    children_ptr: tl.tensor,
    out_ptr: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    rescale: tl.tensor,
    level: tl.tensor,
    max_pages: tl.tensor,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    NODE_BUCKET_SIZE: tl.constexpr,
) -> None:
    """Name the node buckets one request scores at a level and write all their children, one program a block of columns.

    Args:
        pages_ptr: The level's page table [R, max_pages] int32, -1 for none.
        made_ptr: Node buckets each request made of the level [R] int32.
        closed_ptr: Its closed ones [R] int32.
        start_ptr: Each request's start level [R] int32.
        rows_ptr: The batch's request table rows [B] int64.
        carried_ptr: Creation indices the level above kept the children of [B, CARRIED] int32, -1 for none.
        index_query_ptr: Index queries [B, D] bf16.
        bucket_centroids_ptr: The level's centroid rows [rows, D] bf16 or fp8.
        children_ptr: The pool's children [rows, NODE_BUCKET_SIZE] int32.
        out_ptr: The children of every column [B, WIDTH * NODE_BUCKET_SIZE] int32, -1 under a column with no row or
            scored -inf.
        rescale: Factor undoing the fp8 centroid scale.
        level: The level, from 1.
    """
    query = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pooled = _level_pooled(
        pages_ptr,
        made_ptr,
        closed_ptr,
        start_ptr,
        rows_ptr,
        carried_ptr,
        query,
        columns,
        pages_stride,
        carried_stride,
        level,
        max_pages,
        PAGE,
        CARRIED,
    )
    score = _score_rows(index_query_ptr, bucket_centroids_ptr, pooled.to(tl.int64), query, rescale, D)
    # The scored keep names no children at a word at or under -inf's, which a NaN with the sign bit set is
    # => the same test
    never = ordered_bits(tl.full([], float("-inf"), tl.float32))
    named = (columns < WIDTH) & (pooled >= 0) & (ordered_bits(score) > never)
    slots = tl.arange(0, NODE_BUCKET_SIZE)
    kept = tl.load(
        children_ptr + pooled.to(tl.int64)[:, None] * NODE_BUCKET_SIZE + slots[None, :],
        mask=named[:, None],
        other=-1,
    )
    tl.store(
        out_ptr + query * WIDTH * NODE_BUCKET_SIZE + columns[:, None] * NODE_BUCKET_SIZE + slots[None, :],
        kept,
        mask=(columns < WIDTH)[:, None],
    )


def level_children(
    pages: torch.Tensor,
    made: torch.Tensor,
    closed: torch.Tensor,
    start: torch.Tensor,
    rows: torch.Tensor,
    carried: torch.Tensor,
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    children: torch.Tensor,
    level: int,
    page_size: int,
    width: int,
    centroid_scale: float = 1.0,
) -> torch.Tensor:
    """Name the node buckets each request scores at a level and write all their children, in one launch.

    For a level that keeps every column it scores: the same as `level_scores` then a keep of the whole row, but the
    scores and the pooled rows never reach memory.

    Args:
        pages: The level's page table [R, max_pages] int32, -1 for none.
        made: Node buckets each request made of the level [R] int32.
        closed: Its closed ones [R] int32.
        start: Each request's start level [R] int32, 0 for none.
        rows: The batch's request table rows [B] int64.
        carried: Creation indices the level above kept the children of [B, C] int32, -1 for none.
        index_queries: One index query a row [B, index_dim] bf16.
        bucket_centroids: The level's centroid rows [rows, index_dim] - bf16, or fp8 holding `centroid_scale` times the row.
        children: The pool's children [rows, node_bucket_size] int32, a power of two wide.
        level: The level, from 1.
        page_size: Node buckets a page holds.
        width: Columns scored and kept.
        centroid_scale: What fp8 centroids are multiplied by on storing; 1 for bf16.

    Returns:
        The children of every column [B, width * node_bucket_size] int32 in column order - -1 under a column with no row
        or scored -inf.
    """
    count = len(rows)
    node_bucket_size = children.shape[1]
    out = torch.empty(count, width * node_bucket_size, device=rows.device, dtype=torch.int32)
    block = 64
    _level_children_kernel[(count, triton.cdiv(width, block))](
        pages,
        made,
        closed,
        start,
        rows,
        carried,
        index_queries,
        bucket_centroids,
        children,
        out,
        pages.stride(0),
        carried.stride(0),
        1.0 / centroid_scale,
        level,
        pages.shape[1],
        D=index_queries.shape[1],
        PAGE=page_size,
        CARRIED=carried.shape[1],
        WIDTH=width,
        BLOCK=block,
        NODE_BUCKET_SIZE=node_bucket_size,
    )
    return out


def level_scores(
    pages: torch.Tensor,
    made: torch.Tensor,
    closed: torch.Tensor,
    start: torch.Tensor,
    rows: torch.Tensor,
    carried: torch.Tensor,
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    level: int,
    page_size: int,
    width: int,
    centroid_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Name the node buckets each request scores at a level and score them, in one launch.

    The same as `level_candidates` then `score_children`, but every column's score is written, -inf under a -1 row.

    Args:
        pages: The level's page table [R, max_pages] int32, -1 for none.
        made: Node buckets each request made of the level [R] int32.
        closed: Its closed ones [R] int32.
        start: Each request's start level [R] int32, 0 for none.
        rows: The batch's request table rows [B] int64.
        carried: Creation indices the level above kept the children of [B, C] int32, -1 for none.
        index_queries: One index query a row [B, index_dim] bf16.
        bucket_centroids: The level's centroid rows [rows, index_dim] - bf16, or fp8 at `centroid_scale` times.
        level: The level, from 1.
        page_size: Node buckets a page holds.
        width: Columns of the result.
        centroid_scale: What fp8 centroids are multiplied by on storing; 1 for bf16.

    Returns:
        Pooled rows [B, width] int32, their scores [B, width] fp32, and the columns each query holds up to its last
        valid one [B] int32.
    """
    count = len(rows)
    candidates = torch.empty(count, width, device=rows.device, dtype=torch.int32)
    scores = torch.empty(count, width, device=rows.device, dtype=torch.float32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    block = 64
    _level_scores_kernel[(count, triton.cdiv(width, block) + 1)](
        pages,
        made,
        closed,
        start,
        rows,
        carried,
        index_queries,
        bucket_centroids,
        candidates,
        scores,
        lengths,
        pages.stride(0),
        carried.stride(0),
        candidates.stride(0),
        1.0 / centroid_scale,
        level,
        pages.shape[1],
        D=index_queries.shape[1],
        PAGE=page_size,
        CARRIED=carried.shape[1],
        WIDTH=width,
        BLOCK=block,
        LENGTH_BLOCK=1024,
    )
    return candidates, scores, lengths


@triton.jit
def _pick_pooled(
    positions_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    covered_ptr: tl.tensor,
    pages_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    query: tl.tensor,
    columns: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    max_pages: tl.tensor,
    PAGE_ROWS: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_BUCKETS: tl.constexpr,
    CARRIED: tl.constexpr,
) -> tl.tensor:
    """Return the cache rows of a block of one request's level 0 columns.

    Args:
        positions_ptr: Each query's position.
        rows_ptr: The batch's request table rows.
        covered_ptr: Buckets of each request that node buckets cover.
        pages_ptr: Each request's bucket pages.
        carried_ptr: Cache rows the level 1 beam kept.
        query: The query of the program.
        columns: The block's columns.
        pages_stride: Stride of a request's pages.
        carried_stride: Stride of a query's carried rows.
        max_pages: Pages a request holds at most.
        PAGE_ROWS: Bucket rows a page holds.
        BLOCK_POSITIONS: Positions a cluster block covers.
        BLOCK_BUCKETS: Buckets a cluster block covers.
        CARRIED: Columns a query carries.

    Returns:
        The cache rows of the columns, -1 for none.
    """
    row = tl.load(rows_ptr + query)
    covered = tl.load(covered_ptr + row)
    # A query sees the buckets of the cluster blocks before its own
    visible = tl.load(positions_ptr + query) // BLOCK_POSITIONS * BLOCK_BUCKETS
    uncovered = tl.maximum(visible - covered, 0)
    in_own = columns < uncovered
    logical = covered + columns
    page = tl.load(
        pages_ptr + row * pages_stride + tl.minimum(logical // PAGE_ROWS, max_pages - 1),
        mask=in_own,
        other=-1,
    )
    own = tl.where(in_own & (page >= 0), page * PAGE_ROWS + logical % PAGE_ROWS, -1)
    carried_column = columns - uncovered
    carried_held = ~in_own & (carried_column < CARRIED)
    child = tl.load(carried_ptr + query * carried_stride + carried_column, mask=carried_held, other=-1)
    return tl.where(in_own, own, child)


@triton.jit
def _pick_length(
    positions_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    covered_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    query: tl.tensor,
    carried_stride: tl.tensor,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_BUCKETS: tl.constexpr,
    WIDTH: tl.constexpr,
    CARRIED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> tl.tensor:
    """Return the columns one request holds at level 0 up to its last valid one.

    Args:
        positions_ptr: Each query's position.
        rows_ptr: The batch's request table rows.
        covered_ptr: Buckets of each request that node buckets cover.
        carried_ptr: Cache rows the level 1 beam kept.
        query: The query of the program.
        carried_stride: Stride of a query's carried rows.
        BLOCK_POSITIONS: Positions a cluster block covers.
        BLOCK_BUCKETS: Buckets a cluster block covers.
        WIDTH: Columns a row holds at most.
        CARRIED: Columns a query carries.
        BLOCK: Carried columns read at a time.

    Returns:
        The columns the request holds.
    """
    covered = tl.load(covered_ptr + tl.load(rows_ptr + query))
    visible = tl.load(positions_ptr + query) // BLOCK_POSITIONS * BLOCK_BUCKETS
    uncovered = tl.maximum(visible - covered, 0)
    # The length is the last carried index that holds a child, after the uncovered ones
    last = tl.zeros([BLOCK], dtype=tl.int32)
    for base in tl.static_range(0, CARRIED, BLOCK):
        index = base + tl.arange(0, BLOCK)
        held = tl.load(carried_ptr + query * carried_stride + index, mask=index < CARRIED, other=-1) >= 0
        last = tl.maximum(last, tl.where(held, index + 1, 0))
    return tl.minimum(uncovered + tl.max(last, axis=0), WIDTH)


@triton.jit
def _pick_candidates_kernel(
    positions_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    covered_ptr: tl.tensor,
    pages_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    out_stride: tl.tensor,
    max_pages: tl.tensor,
    PAGE_ROWS: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_BUCKETS: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Write the buckets one request scores at level 0 as cache rows, the uncovered ones then the carried, one program a block.

    Args:
        positions_ptr: Each query's position [B] int32.
        rows_ptr: The batch's request table rows [B] int64.
        covered_ptr: Buckets under a node bucket, a request [R] int32.
        pages_ptr: The cache's page table [R, max_pages] int32, -1 for none.
        carried_ptr: Cache rows the level 1 beam kept [B, CARRIED] int32, -1 for none.
        out_ptr: Cache rows [B, WIDTH] int32, -1 for none.
        lengths_ptr: Columns each query holds up to its last valid one [B] int32.
        BLOCK_POSITIONS: Positions of a cluster block, and BLOCK_BUCKETS the buckets of one.
    """
    query = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + query * out_stride + columns,
        _pick_pooled(
            positions_ptr,
            rows_ptr,
            covered_ptr,
            pages_ptr,
            carried_ptr,
            query,
            columns,
            pages_stride,
            carried_stride,
            max_pages,
            PAGE_ROWS,
            BLOCK_POSITIONS,
            BLOCK_BUCKETS,
            CARRIED,
        ),
        mask=columns < WIDTH,
    )
    if tl.program_id(1) == 0:
        tl.store(
            lengths_ptr + query,
            _pick_length(
                positions_ptr,
                rows_ptr,
                covered_ptr,
                carried_ptr,
                query,
                carried_stride,
                BLOCK_POSITIONS,
                BLOCK_BUCKETS,
                WIDTH,
                CARRIED,
                BLOCK,
            ),
        )


def pick_candidates(
    positions: torch.Tensor,
    rows: torch.Tensor,
    covered: torch.Tensor,
    pages: torch.Tensor,
    carried: torch.Tensor,
    page_rows: int,
    block_positions: int,
    block_buckets: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Name the buckets each request scores at level 0 as cache rows, the uncovered ones then the carried.

    A request sees the buckets of the cluster blocks before its position. The uncovered ones are those from bucket
    `covered` on, as cache rows through the request's pages.

    Args:
        positions: Each query's position [B] int32.
        rows: The batch's request table rows [B] int64.
        covered: Buckets under a node bucket, a request [R] int32.
        pages: The cache's page table [R, max_pages] int32, -1 for none.
        carried: Cache rows the level 1 beam kept [B, C] int32, -1 for none.
        page_rows: Cache rows a page holds.
        block_positions: Positions of a cluster block.
        block_buckets: Buckets of a cluster block.
        width: Columns of the result.

    Returns:
        Cache rows [B, width] int32, valid ones first and -1 for none, and the columns each query holds up to its
        last valid one [B] int32.
    """
    count = len(rows)
    out = torch.empty(count, width, device=rows.device, dtype=torch.int32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    block = 1024
    _pick_candidates_kernel[(count, triton.cdiv(width, block))](
        positions,
        rows,
        covered,
        pages,
        carried,
        out,
        lengths,
        pages.stride(0),
        carried.stride(0),
        out.stride(0),
        pages.shape[1],
        PAGE_ROWS=page_rows,
        BLOCK_POSITIONS=block_positions,
        BLOCK_BUCKETS=block_buckets,
        CARRIED=carried.shape[1],
        WIDTH=width,
        BLOCK=block,
        num_warps=4,
    )
    return out, lengths


@triton.jit
def _pick_scores_kernel(
    positions_ptr: tl.tensor,
    rows_ptr: tl.tensor,
    covered_ptr: tl.tensor,
    pages_ptr: tl.tensor,
    carried_ptr: tl.tensor,
    index_query_ptr: tl.tensor,
    bucket_centroids_ptr: tl.tensor,
    candidates_ptr: tl.tensor,
    scores_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    pages_stride: tl.tensor,
    carried_stride: tl.tensor,
    out_stride: tl.tensor,
    max_pages: tl.tensor,
    D: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    BLOCK_POSITIONS: tl.constexpr,
    BLOCK_BUCKETS: tl.constexpr,
    CARRIED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    LENGTH_BLOCK: tl.constexpr,
) -> None:
    """Name the buckets one request scores at level 0 and score them, one program a block of columns.

    One program past the blocks writes the query's length.

    Args:
        positions_ptr: Each query's position [B] int32.
        rows_ptr: The batch's request table rows [B] int64.
        covered_ptr: Buckets under a node bucket, a request [R] int32.
        pages_ptr: The cache's page table [R, max_pages] int32, -1 for none.
        carried_ptr: Cache rows the level 1 beam kept [B, CARRIED] int32, -1 for none.
        index_query_ptr: Index queries [B, D] bf16.
        bucket_centroids_ptr: The cache's centroid rows [rows, D] bf16.
        candidates_ptr: Cache rows [B, WIDTH] int32, -1 for none.
        scores_ptr: Scores [B, WIDTH] fp32, -inf under a -1.
        lengths_ptr: Columns each query holds up to its last valid one [B] int32.
    """
    query = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) < tl.num_programs(1) - 1:
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        pooled = _pick_pooled(
            positions_ptr,
            rows_ptr,
            covered_ptr,
            pages_ptr,
            carried_ptr,
            query,
            columns,
            pages_stride,
            carried_stride,
            max_pages,
            PAGE_ROWS,
            BLOCK_POSITIONS,
            BLOCK_BUCKETS,
            CARRIED,
        )
        inside = columns < WIDTH
        tl.store(candidates_ptr + query * out_stride + columns, pooled, mask=inside)
        score = _score_rows(index_query_ptr, bucket_centroids_ptr, pooled.to(tl.int64), query, 1.0, D)
        tl.store(scores_ptr + query * out_stride + columns, score, mask=inside)
    else:
        tl.store(
            lengths_ptr + query,
            _pick_length(
                positions_ptr,
                rows_ptr,
                covered_ptr,
                carried_ptr,
                query,
                carried_stride,
                BLOCK_POSITIONS,
                BLOCK_BUCKETS,
                WIDTH,
                CARRIED,
                LENGTH_BLOCK,
            ),
        )


def pick_scores(
    positions: torch.Tensor,
    rows: torch.Tensor,
    covered: torch.Tensor,
    pages: torch.Tensor,
    carried: torch.Tensor,
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    page_rows: int,
    block_positions: int,
    block_buckets: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Name the buckets each request scores at level 0 and score them, in one launch.

    The same as `pick_candidates` then `score_children`, but every column's score is written, -inf under a -1 row.

    Args:
        positions: Each query's position [B] int32.
        rows: The batch's request table rows [B] int64.
        covered: Buckets under a node bucket, a request [R] int32.
        pages: The cache's page table [R, max_pages] int32, -1 for none.
        carried: Cache rows the level 1 beam kept [B, C] int32, -1 for none.
        index_queries: One index query a row [B, index_dim] bf16.
        bucket_centroids: The cache's centroid rows [rows, index_dim] bf16.
        page_rows: Cache rows a page holds.
        block_positions: Positions of a cluster block.
        block_buckets: Buckets of a cluster block.
        width: Columns of the result.

    Returns:
        Cache rows [B, width] int32, their scores [B, width] fp32, and the columns each query holds up to its last
        valid one [B] int32.
    """
    count = len(rows)
    candidates = torch.empty(count, width, device=rows.device, dtype=torch.int32)
    scores = torch.empty(count, width, device=rows.device, dtype=torch.float32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    block = 64
    _pick_scores_kernel[(count, triton.cdiv(width, block) + 1)](
        positions,
        rows,
        covered,
        pages,
        carried,
        index_queries,
        bucket_centroids,
        candidates,
        scores,
        lengths,
        pages.stride(0),
        carried.stride(0),
        candidates.stride(0),
        pages.shape[1],
        D=index_queries.shape[1],
        PAGE_ROWS=page_rows,
        BLOCK_POSITIONS=block_positions,
        BLOCK_BUCKETS=block_buckets,
        CARRIED=carried.shape[1],
        WIDTH=width,
        BLOCK=block,
        LENGTH_BLOCK=1024,
    )
    return candidates, scores, lengths


@triton.jit
def _cache_row_keys_kernel(
    keys_ptr: tl.tensor,
    listed_ptr: tl.tensor,
    candidates_ptr: tl.tensor,
    out_ptr: tl.tensor,
    key_stride: tl.tensor,
    candidates_stride: tl.tensor,
    COLUMNS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Replace the columns a listing names with the cache rows they stand for, keeping the scores in the keys.

    Args:
        keys_ptr: Listed columns packed with their scores [S, WIDTH] int64.
        listed_ptr: Keys each row lists [S] int32.
        candidates_ptr: The cache row of each column [S, COLUMNS] int32, -1 for none.
        out_ptr: Scores packed with cache rows [S, WIDTH] int64, EMPTY for an invalid key.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = slots < WIDTH
    key = tl.load(keys_ptr + row * key_stride + slots, mask=inside, other=EMPTY_KEY)
    never = ordered_bits(tl.full([], float("-inf"), tl.float32))
    valid = (
        inside & (slots < tl.load(listed_ptr + row)) & (key != EMPTY_KEY) & ((key >> 32).to(tl.int32) > never)
    )
    column = tl.minimum(tl.maximum(key_bucket(key), 0), COLUMNS - 1)
    cache_row = tl.load(candidates_ptr + row * candidates_stride + column, mask=valid, other=-1).to(tl.int64)
    valid &= cache_row >= 0
    packed = ((key >> 32) << 32) | (0x7FFFFFFF - cache_row)
    tl.store(out_ptr + row * WIDTH + slots, tl.where(valid, packed, EMPTY_KEY), mask=inside)


def cache_row_keys(keys: torch.Tensor, listed: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Replace the columns a listing names with the cache rows they stand for, keeping the scores in the keys.

    Args:
        keys: Listed columns packed with their scores [S, W] int64.
        listed: Keys each row lists [S] int32.
        candidates: The cache row of each column [S, N] int32, -1 for none.

    Returns:
        Scores packed with cache rows [S, W] int64, the empty key for a column that is no bucket, was scored -inf or
        was never listed.
    """
    count, width = keys.shape
    out = torch.empty(count, width, dtype=torch.int64, device=keys.device)
    block = min(triton.next_power_of_2(width), 256)
    _cache_row_keys_kernel[(count, triton.cdiv(width, block))](
        keys,
        listed,
        candidates,
        out,
        keys.stride(0),
        candidates.stride(0),
        COLUMNS=candidates.shape[1],
        WIDTH=width,
        BLOCK=block,
        num_warps=4,
    )
    return out
