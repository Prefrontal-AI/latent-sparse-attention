"""One level of PISA's descent in one launch - a row's candidates masked, the forced blocks put first, the best kept, and
their children listed for the level below.

A program holds one query row's candidates, packs each one's priority with its slot into an int64 and sorts them, so
the keep is the first `keep` of the sort: forced blocks first, then the best scores, a tie toward the lower slot.
Above the leaves it writes the kept blocks' children, at the leaves the kept blocks in ascending order, which is the
order their keys are listed in.
"""

import torch
import triton
import triton.language as tl

from lsa.core.kernels.triton.keys import EMPTY, key_bucket, ordered_bits, pack_key

# The empty key, which sorts last
EMPTY_KEY: tl.constexpr = tl.constexpr(EMPTY)

# A kept slot past the row's blocks, which sorts last at the leaves
NO_BLOCK: tl.constexpr = tl.constexpr(1 << 40)


@triton.jit
def keep_kernel(
    scores_ptr: tl.tensor,
    candidates_ptr: tl.tensor,
    valid_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    kept_ptr: tl.tensor,
    kept_valid_ptr: tl.tensor,
    width: tl.tensor,
    keep: tl.tensor,
    blocks_below: tl.tensor,
    block_size: tl.tensor,
    out_width: tl.tensor,
    LEAF: tl.constexpr,
    FORCED: tl.constexpr,
    BRANCHING: tl.constexpr,
    WIDTH: tl.constexpr,
) -> None:
    """Keep the best `keep` eligible candidates of one row and write their children, or at the leaves themselves."""
    row = tl.program_id(0).to(tl.int64)
    slots = tl.arange(0, WIDTH)
    live = slots < width
    candidates = tl.load(candidates_ptr + row * width + slots, mask=live, other=0)
    valid = tl.load(valid_ptr + row * width + slots, mask=live, other=0) != 0
    scores = tl.load(scores_ptr + row * width + slots, mask=live, other=float("-inf"))
    position = tl.load(positions_ptr + row)
    # Keys a block of this level spans => a candidate starting after the query is out
    eligible = valid & (candidates * block_size * blocks_below <= position)
    priority = tl.where(eligible, scores, float("-inf"))
    if FORCED:
        current = position // block_size
        previous = tl.where(current > 0, (current - 1) // blocks_below, -1)
        forced = (candidates == 0) | (candidates == previous) | (candidates == current // blocks_below)
        priority = tl.where(eligible & forced, float("inf"), priority)
    keys = tl.where(live, pack_key(priority, slots), EMPTY_KEY)
    ordered = tl.sort(keys, descending=True)
    kept_live = (slots < keep) & (ordered != EMPTY_KEY)
    lowest = ordered_bits(tl.full((WIDTH,), float("-inf"), tl.float32)).to(tl.int64)
    kept_valid = kept_live & ((ordered >> 32) > lowest)
    kept = tl.load(candidates_ptr + row * width + key_bucket(ordered), mask=kept_valid, other=0)
    if LEAF:
        blocks = tl.sort(tl.where(kept_valid, kept, NO_BLOCK))
        tl.store(
            kept_ptr + row * out_width + slots,
            tl.where(blocks == NO_BLOCK, -1, blocks),
            mask=slots < out_width,
        )
    else:
        children = tl.arange(0, BRANCHING)
        at = row * out_width + slots[:, None] * BRANCHING + children[None, :]
        written = (slots < out_width // BRANCHING)[:, None]
        tl.store(kept_ptr + at, kept[:, None] * BRANCHING + children[None, :], mask=written)
        tl.store(kept_valid_ptr + at, kept_valid[:, None] & (children[None, :] >= 0), mask=written)


def keep(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    positions: torch.Tensor,
    keep_count: int,
    blocks_below: int,
    block_size: int,
    branching: int,
    forced_blocks: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep each row's best eligible candidates of one level, the forced ones first.

    Args:
        scores: The candidates' scores [rows, width] fp32.
        candidates: Each row's candidate blocks [rows, width] int64.
        candidate_valid: Which candidates exist [rows, width] bool.
        positions: Each row's position [rows] int64.
        keep_count: Blocks a row keeps.
        blocks_below: Leaf blocks a block of this level holds - 1 at the leaves.
        block_size: Keys in a leaf block.
        branching: Children of a block above the leaves.
        forced_blocks: Whether the first, previous and current blocks are kept first.

    Returns:
        Above the leaves, the kept blocks' children and whether each is valid [rows, kept * branching] int64 and
        bool. At the leaves, the kept blocks ascending, -1 past them, and whether each is valid [rows, kept].
    """
    row_count, width = candidates.shape
    leaf = blocks_below == 1
    kept_count = min(keep_count, width)
    out_width = kept_count if leaf else kept_count * branching
    kept = torch.empty(row_count, out_width, dtype=torch.long, device=candidates.device)
    kept_valid = torch.empty(row_count, out_width, dtype=torch.bool, device=candidates.device)
    if row_count == 0:
        return kept, kept_valid
    keep_kernel[(row_count,)](
        scores.contiguous(),
        candidates.contiguous(),
        candidate_valid.contiguous(),
        positions.contiguous(),
        kept,
        kept_valid,
        width,
        keep_count,
        blocks_below,
        block_size,
        out_width,
        LEAF=leaf,
        FORCED=forced_blocks,
        BRANCHING=branching,
        WIDTH=max(16, triton.next_power_of_2(width)),
    )
    if leaf:
        kept_valid = kept >= 0
    return kept, kept_valid
