"""PISA's pyramid kept up in place - the summaries above a request's leaf blocks pooled again over a range of leaves
that just closed or reopened, in one launch.

A summary is the mean of the children it has, its children's sum over their count, the order the torch pyramid sums
them in, so a level kept this way holds what a pyramid built from scratch would.
A program pools one slice of the key dims at every level in turn, since a level reads what the level below it just
wrote, and the slices never meet.
"""

import torch
import triton
import triton.language as tl

# Key dims a program pools
BLOCK_DIM: int = 64

# Summaries of a level pooled at once
PARENTS: int = 32


@triton.jit(do_not_specialize=["row", "first", "stop"])
def lift_kernel(
    means_ptr: tl.tensor,
    upper_ptr: tl.tensor,
    widths_ptr: tl.tensor,
    offsets_ptr: tl.tensor,
    row: tl.tensor,
    first: tl.tensor,
    stop: tl.tensor,
    level_count: tl.tensor,
    key_dim: tl.tensor,
    means_row_stride: tl.tensor,
    means_block_stride: tl.tensor,
    upper_row_stride: tl.tensor,
    BRANCHING: tl.constexpr,
    PARENTS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> None:
    """Pool the ancestors of leaves `[first, stop)` of one row, level by level, over one slice of the key dims."""
    dims = tl.program_id(0) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    dim_live = dims < key_dim
    means_base = means_ptr + row.to(tl.int64) * means_row_stride
    upper_base = upper_ptr + row.to(tl.int64) * upper_row_stride
    # Kernel arguments reassigned in this loop left every level above the first unwritten
    # => each level's bounds from the leaves' own
    span = 1
    for level in range(1, level_count):
        width_below = tl.load(widths_ptr + level - 1)
        offset_below = tl.load(offsets_ptr + level - 1)
        offset = tl.load(offsets_ptr + level)
        span = span * BRANCHING
        level_first = first // span
        level_stop = (stop + span - 1) // span
        for start in range(level_first, level_stop, PARENTS):
            parents = start + tl.arange(0, PARENTS)
            live = parents < level_stop
            total = tl.zeros((PARENTS, BLOCK_DIM), dtype=tl.float32)
            count = tl.zeros((PARENTS,), dtype=tl.float32)
            for i in tl.static_range(BRANCHING):
                child = parents * BRANCHING + i
                child_live = live & (child < width_below)
                mask = child_live[:, None] & dim_live[None, :]
                if level == 1:
                    leaf = means_base + child[:, None].to(tl.int64) * means_block_stride + dims[None, :]
                    total += tl.load(leaf, mask=mask, other=0.0).to(tl.float32)
                else:
                    summary = (
                        upper_base + (offset_below + child)[:, None].to(tl.int64) * key_dim + dims[None, :]
                    )
                    total += tl.load(summary, mask=mask, other=0.0)
                count += child_live.to(tl.float32)
            parent = tl.math.div_rn(total, tl.maximum(count, 1.0)[:, None])
            at = upper_base + (offset + parents)[:, None].to(tl.int64) * key_dim + dims[None, :]
            tl.store(at, parent, mask=live[:, None] & dim_live[None, :])
        # The next level reads what this one wrote, across the program's threads
        tl.debug_barrier()


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
    key_dim = means.shape[2]
    lift_kernel[(triton.cdiv(key_dim, BLOCK_DIM),)](
        means,
        upper,
        widths,
        offsets,
        row,
        first,
        stop,
        widths.shape[0],
        key_dim,
        means.stride(0),
        means.stride(1),
        upper.stride(0),
        BRANCHING=branching,
        PARENTS=PARENTS,
        BLOCK_DIM=BLOCK_DIM,
    )
