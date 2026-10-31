"""PISA's block scores in one kernel - for each query row and candidate block, the LogSumExp over the block's children of
the scaled dot with each head's query, summed over the heads.

A leaf block's children are its keys, read where they sit in the KV pool through the request table, and a key past the
row's query is masked. A block above the leaves has a few children, the summaries of the level below.
Nothing is gathered first, so a whole prefill chunk is scored at once - the torch search gathers every candidate's keys
of a tile of rows, and its launches, not its arithmetic, bound a long prefill.

A program scores one row's group of candidates: `CHILDREN` columns a candidate, `CANDIDATES` candidates, their dots
accumulated over the key dims in steps of `BLOCK_DIM`. A launch's second dim is the rows, which caps a call. A leaf program is one block of 64 keys, an upper one 32 blocks of
two summaries, so both read 64 columns.

The summaries are fp32 and the queries bf16, so an upper level dots each query with a summary's three bf16 parts on the
tensor cores - every product exact in fp32, as the fp32 dot's are, and only the order of the sum differs.
"""

import torch
import triton
import triton.language as tl

# Columns a program dots against - a leaf block's keys, or the summaries of a group of upper blocks
COLUMNS: int = 64

# Key dims accumulated a step
BLOCK_DIM: int = 64

# Rows a call scores at most - the launch's second dim
MAX_ROWS: int = 65535

# Fewest heads `tl.dot` takes - fewer are padded with zero queries, whose LogSumExp is dropped
MIN_HEADS: int = 16


@triton.jit
def block_scores_kernel(
    queries_ptr: tl.tensor,
    candidates_ptr: tl.tensor,
    requests_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    children_ptr: tl.tensor,
    table_ptr: tl.tensor,
    row_index_ptr: tl.tensor,
    scores_ptr: tl.tensor,
    width: tl.tensor,
    head_count: tl.tensor,
    key_dim: tl.tensor,
    child_limit: tl.tensor,
    query_row_stride: tl.tensor,
    query_head_stride: tl.tensor,
    candidate_row_stride: tl.tensor,
    child_stride: tl.tensor,
    child_request_stride: tl.tensor,
    table_row_stride: tl.tensor,
    logit_scale: tl.tensor,
    LEAF: tl.constexpr,
    CHILDREN: tl.constexpr,
    CANDIDATES: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    IEEE: tl.constexpr,
    SPLIT: tl.constexpr,
) -> None:
    """Score `CANDIDATES` candidate blocks of one query row."""
    group = tl.program_id(0)
    row = tl.program_id(1)
    columns = tl.arange(0, CANDIDATES * CHILDREN)
    slot_in_row = group * CANDIDATES + columns // CHILDREN
    candidate_live = slot_in_row < width
    candidate = tl.load(
        candidates_ptr + row * candidate_row_stride + slot_in_row, mask=candidate_live, other=0
    )
    child = candidate * CHILDREN + columns % CHILDREN
    request = tl.load(requests_ptr + row)
    position = tl.load(positions_ptr + row)
    table_row = tl.load(row_index_ptr + request)
    if LEAF:
        # A key past the query is masked, and read from the query's own position meanwhile
        valid = candidate_live & (child <= position)
        at = tl.minimum(child, position)
        source = tl.load(table_ptr + table_row * table_row_stride + at, mask=candidate_live, other=0).to(
            tl.int64
        )
        child_base = children_ptr + source * child_stride
    else:
        valid = candidate_live & (child < child_limit)
        at = tl.minimum(child, child_limit - 1)
        child_base = children_ptr + table_row * child_request_stride + at * child_stride

    heads = tl.arange(0, HEADS)
    head_live = heads < head_count
    dots = tl.zeros((HEADS, CANDIDATES * CHILDREN), dtype=tl.float32)
    for start in range(0, key_dim, BLOCK_DIM):
        dims = start + tl.arange(0, BLOCK_DIM)
        dim_live = dims < key_dim
        query = tl.load(
            queries_ptr + row * query_row_stride + heads[:, None] * query_head_stride + dims[None, :],
            mask=head_live[:, None] & dim_live[None, :],
            other=0.0,
        )
        keys = tl.load(
            child_base[None, :] + dims[:, None], mask=dim_live[:, None] & candidate_live[None, :], other=0.0
        )
        if SPLIT:
            # An fp32 summary is three bf16 parts exactly, and a bf16 query times each is exact in fp32
            # => three tensor-core dots give the fp32 dot but for the order of the sum
            high = keys.to(tl.bfloat16)
            rest = keys - high.to(tl.float32)
            middle = rest.to(tl.bfloat16)
            low = (rest - middle.to(tl.float32)).to(tl.bfloat16)
            dots = tl.dot(query, high, dots)
            dots = tl.dot(query, middle, dots)
            dots = tl.dot(query, low, dots)
        elif IEEE:
            dots = tl.dot(query, keys, dots, input_precision="ieee")
        else:
            dots = tl.dot(query, keys, dots)

    logits = tl.where(valid[None, :], dots * logit_scale, float("-inf"))
    logits = tl.reshape(logits, (HEADS, CANDIDATES, CHILDREN))
    peak = tl.max(logits, axis=2)
    # A block with no valid child scores -inf, as torch's LogSumExp of nothing does
    finite_peak = tl.where(peak == float("-inf"), 0.0, peak)
    total = tl.sum(tl.exp(logits - finite_peak[:, :, None]), axis=2)
    head_scores = tl.where(peak == float("-inf"), float("-inf"), finite_peak + tl.log(total))
    head_scores = tl.where(head_live[:, None], head_scores, 0.0)
    scores = tl.sum(head_scores, axis=0)
    slots = group * CANDIDATES + tl.arange(0, CANDIDATES)
    tl.store(scores_ptr + row * width + slots, scores, mask=slots < width)


def block_scores(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    children: torch.Tensor,
    children_per_block: int,
    logit_scale: float,
    table: torch.Tensor | None = None,
    row_index: torch.Tensor | None = None,
    child_limit: int | None = None,
) -> torch.Tensor:
    """Score every candidate block of every row by the LogSumExp of its children's scaled dots, summed over the heads.

    Args:
        queries: Each row's heads' queries [rows, head_count, key_dim] bf16 for leaves, any float for summaries.
        candidates: Each row's candidate blocks [rows, width] int64.
        requests: Each row's request, an index into `row_index` [rows] int64.
        positions: Each row's position [rows] int64.
        children: The keys, a KV pool viewed [slots, key_dim] with any slot stride, when `table` is given -
            else the summaries of the level below [summary_rows, block_count, key_dim] fp32.
        children_per_block: Keys a leaf block holds, or blocks an upper block holds.
        logit_scale: What the dots are multiplied by.
        table: The request table [table_rows, context] int32, which turns a key's position into its slot, for leaves.
        row_index: Each request's table row [requests] int64 - its row of the summaries above the leaves, and
            for leaves its row of the table. The summaries' rows are the requests' own when it is not given.
        child_limit: Blocks of the summaries' level a search spans, the rest masked - all of them when not given.

    Returns:
        The candidates' scores [rows, width] fp32, -inf where a candidate has no valid child.

    Raises:
        ValueError: The children per block do not divide the columns a program reads, a leaf block was asked
            for without the table, or there are more rows than a launch holds.
    """
    leaf = table is not None
    if COLUMNS % children_per_block != 0:
        raise ValueError(f"children_per_block must divide {COLUMNS} - got {children_per_block}")
    if leaf and row_index is None:
        raise ValueError("a leaf block's keys are read through the table, which needs each request's row")
    row_count, width = candidates.shape
    if row_count > MAX_ROWS:
        raise ValueError(f"a call scores at most {MAX_ROWS} rows - got {row_count}, so tile them")
    head_count, key_dim = queries.shape[1], queries.shape[2]
    scores = torch.empty(row_count, width, dtype=torch.float32, device=queries.device)
    if row_count == 0 or width == 0:
        return scores
    queries = queries if queries.stride(2) == 1 else queries.contiguous()
    candidates = candidates.contiguous()
    candidate_count = COLUMNS // children_per_block
    heads = max(MIN_HEADS, triton.next_power_of_2(head_count))
    if table is not None:
        child_stride, child_request_stride, child_limit = children.stride(0), 0, 0
        table_row_stride = table.stride(0)
    else:
        child_stride, child_request_stride = children.stride(1), children.stride(0)
        child_limit = children.shape[1] if child_limit is None else child_limit
        if row_index is None:
            row_index = torch.arange(children.shape[0], device=children.device)
        table, table_row_stride = requests, 0
    # bf16 queries against fp32 summaries take the split dots, any other pair is brought to the children's dtype
    split = queries.dtype == torch.bfloat16 and children.dtype == torch.float32
    if not split and queries.dtype != children.dtype:
        queries = queries.to(children.dtype)
    # A row's groups launch together, then the next row's - neighbouring rows keep much the same blocks, so the
    # programs in flight read keys the cache already holds
    grid = (triton.cdiv(width, candidate_count), row_count)
    block_scores_kernel[grid](
        queries,
        candidates,
        requests,
        positions,
        children,
        table,
        row_index,
        scores,
        width,
        head_count,
        key_dim,
        child_limit,
        queries.stride(0),
        queries.stride(1),
        candidates.stride(0),
        child_stride,
        child_request_stride,
        table_row_stride,
        logit_scale,
        LEAF=leaf,
        CHILDREN=children_per_block,
        CANDIDATES=candidate_count,
        HEADS=heads,
        BLOCK_DIM=BLOCK_DIM,
        IEEE=children.dtype == torch.float32,
        SPLIT=split,
    )
    return scores
