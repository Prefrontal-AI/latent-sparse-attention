"""PISA's pyramid kept up in place - the summaries above a request's leaf blocks pooled again over a range of leaves
that just closed or reopened, in one launch.

A summary is the mean of the children it has, its children's sum over their count, the order the torch pyramid sums
them in, so a level kept this way holds what a pyramid built from scratch would.

triton-ascend: the original one-launch form (a program pooling a dim slice at every level, 2D tiles over parents x
dims) miscompiles on this backend - 2D gather loads, 2D reductions and comparisons against sorted values all
corrupt their results depending on surrounding code. The port therefore runs one launch per level, one program per
(parent, dim slice), and keeps every load, store and reduction 1D.
"""

import logging

import torch
import triton
import triton.language as tl

# Key dims a program pools
BLOCK_DIM: int = 64


@triton.jit
def lift_level_kernel(
    means_ptr: tl.tensor,
    upper_ptr: tl.tensor,
    widths_ptr: tl.tensor,
    offsets_ptr: tl.tensor,
    row: tl.tensor,
    level: tl.tensor,
    level_first: tl.tensor,
    key_dim: tl.tensor,
    means_row_stride: tl.tensor,
    means_block_stride: tl.tensor,
    upper_row_stride: tl.tensor,
    BRANCHING: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> None:
    """Pool one parent's summary at one level over one slice of the key dims."""
    dims = tl.program_id(0) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_live = dims < key_dim
    parent = tl.program_id(1)
    width_below = tl.load(widths_ptr + level - 1)
    offset_below = tl.load(offsets_ptr + level - 1)
    offset = tl.load(offsets_ptr + level)
    means_base = means_ptr + row.to(tl.int64) * means_row_stride
    upper_base = upper_ptr + row.to(tl.int64) * upper_row_stride
    p = level_first + parent
    total = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    count = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    for i in tl.static_range(BRANCHING):
        child = p * BRANCHING + i
        child_live = child < width_below
        if level == 1:
            source = means_base + child.to(tl.int64) * means_block_stride + dims
        else:
            source = upper_base + (offset_below + child).to(tl.int64) * key_dim + dims
        total += tl.load(source, mask=dim_live & child_live, other=0.0).to(tl.float32)
        count += child_live.to(tl.float32)
    summary = total / tl.maximum(count, 1.0)
    at = upper_base + (offset + p).to(tl.int64) * key_dim + dims
    tl.store(at, summary, mask=dim_live)


def lift(
    means: torch.Tensor,
    upper: torch.Tensor,
    widths: torch.Tensor,
    offsets: torch.Tensor,
    row: int,
    first: int,
    stop: int,
    branching: int,
) -> None:
    """Pool again every summary above leaves `[first, stop)` of one request, in place.

    Args:
        means: Each request's leaf means [requests, leaf_count, key_dim], any float dtype, any block stride.
        upper: Each request's summaries of every level above the leaves, end to end [requests, upper_count, key_dim]
            fp32, contiguous.
        widths: Blocks of every level, the leaves first [level_count] int64.
        offsets: Where each level starts in `upper`, the leaves' entry unused [level_count] int64.
        row: The request's row.
        first: The first leaf whose mean changed.
        stop: One past the last.
        branching: Children of a block above the leaves.
    """
    if stop <= first or widths.shape[0] < 2:
        return
    # triton-ascend: the per-level bounds are resolved on the host, since a runtime `for level`
    # loop carrying kernel arguments left every level above the first unwritten
    widths_list = widths.tolist()
    offsets_list = offsets.tolist()
    key_dim = means.shape[2]
    span = 1
    for level in range(1, len(widths_list)):
        span *= branching
        level_first = first // span
        level_stop = (stop + span - 1) // span
        if level_stop <= level_first:
            continue
        logging.debug(
            "lift level %d parents [%d, %d) of row %d", level, level_first, level_stop, row
        )
        lift_level_kernel[(triton.cdiv(key_dim, BLOCK_DIM), level_stop - level_first)](
            means,
            upper,
            widths,
            offsets,
            row,
            level,
            level_first,
            key_dim,
            means.stride(0),
            means.stride(1),
            upper.stride(0),
            BRANCHING=branching,
            BLOCK_DIM=BLOCK_DIM,
        )
