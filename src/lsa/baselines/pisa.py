"""PISA's block selection, in torch - a positional pyramid of mean-pooled key blocks searched coarse to fine.

    leaf block b      C consecutive keys                       level 1
    block above       g adjacent blocks of the level below     levels 2 ... top, one root
    summary           mean of the child summaries              leaves: mean of their keys
    score of a block  log sum_r exp(q . child_r / sqrt(head_dim)), the children's summaries, or for a leaf its raw keys

A query starts at the root, scores the candidates of a level, keeps the best `keep_count` and moves to their children.
The kept leaf blocks are its selection, so it scores O(g * keep_count * log(T / C)) blocks, not every block.
Query heads that share a KV head share one selection and their scores are summed.
The first, previous and current leaf blocks and the blocks above them are always kept, unscored, and count inside the keep count.

Tang et al., "Block Sparse Attention with Log-Linear Complexity", arXiv 2609.31093, Section 3 and Appendices A-B.
This is the reference - one query and one KV head at a time, for clarity over speed.
"""

import math
from typing import NamedTuple

import torch


class PisaSelection(NamedTuple):
    """What PISA selected for a batch of queries and what that cost.

    Attributes:
        selected_key_ids: Key ids of the kept leaf blocks [R, kv_head_count, keep_count * block_size] int32, ascending, -1 padded.
        scored_counts: Blocks scored per query, KV head and level [R, kv_head_count, level_count] int32, the leaf level first.
        boundary_gaps: At each level, how far the last kept free block scored above the best dropped one, over the
            kept score's magnitude or 1 if that is less [R, kv_head_count, level_count] fp32, inf where the level
            dropped no scored block.
    """

    selected_key_ids: torch.Tensor
    scored_counts: torch.Tensor
    boundary_gaps: torch.Tensor


def keep_count_for_budget(key_budget: int, block_size: int) -> int:
    """Return the leaf blocks that make a key budget, `key_budget / block_size`.

    Args:
        key_budget: Keys a query may read, a multiple of `block_size`.
        block_size: Keys in a leaf block.

    Returns:
        The keep count.

    Raises:
        ValueError: If the budget is not a whole number of blocks.
    """
    if key_budget <= 0 or key_budget % block_size != 0:
        raise ValueError(
            f"key_budget must be a positive multiple of block_size={block_size}, got {key_budget}"
        )
    return key_budget // block_size


def build_pyramid(keys: torch.Tensor, block_size: int, branching: int) -> list[torch.Tensor]:
    """Mean-pool the keys into leaf blocks and those into blocks above, up to a single root.

    A block's summary is the mean of its valid children's summaries, so a ragged last block averages what it has.
    Parents average their children unweighted, as the paper does.

    Args:
        keys: Keys [T, kv_head_count, head_dim], any float dtype.
        block_size: Keys in a leaf block.
        branching: Children of a block above the leaves.

    Returns:
        The summaries of each level [block_count, kv_head_count, head_dim] fp32, the keys themselves first, then the leaf blocks, up to the root.
    """
    levels = [keys.float()]
    children_per_block = block_size
    while levels[-1].shape[0] > 1 or len(levels) == 1:
        below = levels[-1]
        block_count = math.ceil(below.shape[0] / children_per_block)
        padded = torch.zeros(block_count * children_per_block, *below.shape[1:])
        padded[: below.shape[0]] = below
        valid = torch.zeros(block_count * children_per_block)
        valid[: below.shape[0]] = 1.0
        sums = padded.view(block_count, children_per_block, *below.shape[1:]).sum(1)
        counts = valid.view(block_count, children_per_block).sum(1)
        levels.append(sums / counts[:, None, None])
        children_per_block = branching
    return levels


def score_blocks(
    queries: torch.Tensor,
    children: torch.Tensor,
    child_valid: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """Score blocks by the LogSumExp of the query's dots with their children, summed over the heads.

    Args:
        queries: One KV head's query heads [group_size, head_dim] fp32.
        children: The blocks' children, summaries or raw keys [block_count, children_per_block, head_dim] fp32.
        child_valid: Which children exist and may be read [block_count, children_per_block] bool.
        logit_scale: What the dots are multiplied by.

    Returns:
        Block scores [block_count] fp32.
    """
    logits = torch.einsum("gd,bcd->gbc", queries, children) * logit_scale
    logits = logits.masked_fill(~child_valid[None], -torch.inf)
    return logits.logsumexp(-1).sum(0)


def select_for_head(
    queries: torch.Tensor,
    position: int,
    levels: list[torch.Tensor],
    kv_head: int,
    block_size: int,
    branching: int,
    keep_counts: list[int],
    logit_scale: float,
    forced_blocks: bool,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Search the pyramid for one query position and one KV head.

    Args:
        queries: The KV head's query heads at the position [group_size, head_dim] fp32.
        position: The query's absolute position.
        levels: The pyramid from `build_pyramid`.
        kv_head: Which KV head's summaries to read.
        block_size: Keys in a leaf block.
        branching: Children of a block above the leaves.
        keep_counts: Blocks kept at each level, the leaf level first.
        logit_scale: What the dots are multiplied by.
        forced_blocks: Whether the first, previous and current leaf blocks and their ancestors are kept unscored.

    Returns:
        kept_leaves: The kept leaf blocks [<= keep_counts[0]] int64, ascending.
        scored_counts: Blocks scored at each level, the leaf level first.
        boundary_gaps: At each level, the last kept free block's score less the best dropped one's, over the kept
            score's magnitude or 1 if that is less, inf where no scored block was dropped.
    """
    level_count = len(levels) - 1
    current = position // block_size
    forced_leaves = torch.tensor(
        sorted({0, current - 1, current} - {-1}) if forced_blocks else [], dtype=torch.long
    )
    scored_counts = [0] * level_count
    boundary_gaps = [math.inf] * level_count
    candidates = torch.zeros(1, dtype=torch.long)
    kept = candidates

    for level in range(level_count, 0, -1):
        # Keys a block of this level spans => candidates starting after the query are out
        span = block_size * branching ** (level - 1)
        eligible = candidates[candidates * span <= position]
        forced = (forced_leaves // branching ** (level - 1)).unique()
        is_forced = torch.isin(eligible, forced)
        keep_count = keep_counts[level - 1]

        if eligible.shape[0] <= keep_count:
            kept = eligible
        else:
            free = eligible[~is_forced]
            children_per_block = block_size if level == 1 else branching
            child_ids = free[:, None] * children_per_block + torch.arange(children_per_block)
            child_count = levels[level - 1].shape[0]
            child_valid = child_ids < child_count
            if level == 1:
                child_valid &= child_ids <= position
            children = levels[level - 1][child_ids.clamp(max=child_count - 1), kv_head]
            scores = score_blocks(queries, children, child_valid, logit_scale)
            scored_counts[level - 1] = free.shape[0]
            free_kept = keep_count - int(is_forced.sum())
            if free_kept < free.shape[0]:
                ranked = scores.topk(free_kept + 1).values
                if free_kept:
                    kept_score, dropped_score = float(ranked[free_kept - 1]), float(ranked[free_kept])
                    boundary_gaps[level - 1] = (kept_score - dropped_score) / max(1.0, abs(kept_score))
            best = free[scores.topk(free_kept).indices]
            kept = torch.cat((eligible[is_forced], best))

        if level > 1:
            child_ids = (kept[:, None] * branching + torch.arange(branching)).flatten()
            candidates = child_ids[child_ids < levels[level - 1].shape[0]]
    return kept.sort().values, scored_counts, boundary_gaps


def select(
    keys: torch.Tensor,
    queries: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    keep_count: int,
    branching: int = 2,
    beam_width: int | None = None,
    logit_scale: float | None = None,
    forced_blocks: bool = True,
) -> PisaSelection:
    """Select the leaf blocks each query reads, searching the block pyramid coarse to fine.

    Query heads are assigned to KV heads in consecutive groups, and a group's scores are summed.
    A level with no more candidates than it keeps keeps them all without scoring.
    Candidates starting after the query are dropped, and keys after it are never selected.

    Args:
        keys: Keys [T, kv_head_count, head_dim], any float dtype.
        queries: Queries [R, query_head_count, head_dim], any float dtype.
        positions: Each query's absolute position [R] int, below T.
        block_size: Keys in a leaf block, the paper's C.
        keep_count: Leaf blocks kept, the paper's K - a budget of B keys is `keep_count_for_budget(B, block_size)`.
        branching: Children of a block above the leaves, the paper's g.
        beam_width: Blocks kept at each level above the leaves, `keep_count` when None.
        logit_scale: What the dots are multiplied by, `1 / sqrt(head_dim)` when None.
        forced_blocks: Whether the first, previous and current leaf blocks and their ancestors are kept unscored.

    Returns:
        The selection and the blocks scored to make it.

    Raises:
        ValueError: If a shape, a position or a count is not one the method is defined for.
    """
    if keys.ndim != 3 or queries.ndim != 3 or positions.ndim != 1:
        raise ValueError(
            f"want keys [T, kv, d], queries [R, h, d], positions [R], got {keys.shape}, {queries.shape}, {positions.shape}"
        )
    key_count, kv_head_count, head_dim = keys.shape
    query_count, query_head_count, query_dim = queries.shape
    if query_dim != head_dim or positions.shape[0] != query_count:
        raise ValueError(
            f"queries {tuple(queries.shape)} and positions {tuple(positions.shape)} do not match keys {tuple(keys.shape)}"
        )
    if query_head_count % kv_head_count != 0:
        raise ValueError(
            f"query_head_count={query_head_count} is not a multiple of kv_head_count={kv_head_count}"
        )
    if query_count > 0 and (int(positions.min()) < 0 or int(positions.max()) >= key_count):
        raise ValueError(f"positions must lie in [0, {key_count})")
    if block_size < 1 or branching < 2:
        raise ValueError(f"want block_size >= 1 and branching >= 2, got {block_size} and {branching}")
    beam_width = keep_count if beam_width is None else beam_width
    smallest = 3 if forced_blocks else 1
    if keep_count < smallest or beam_width < smallest:
        raise ValueError(
            f"keep_count and beam_width must be at least {smallest}, got {keep_count} and {beam_width}"
        )

    scale = 1.0 / math.sqrt(head_dim) if logit_scale is None else logit_scale
    levels = build_pyramid(keys, block_size, branching)
    level_count = len(levels) - 1
    keep_counts = [keep_count] + [beam_width] * (level_count - 1)
    group_size = query_head_count // kv_head_count
    grouped = queries.float().view(query_count, kv_head_count, group_size, head_dim)

    selected_key_ids = torch.full(
        (query_count, kv_head_count, keep_count * block_size), -1, dtype=torch.int32
    )
    scored_counts = torch.zeros(query_count, kv_head_count, level_count, dtype=torch.int32)
    boundary_gaps = torch.full((query_count, kv_head_count, level_count), math.inf)
    for r in range(query_count):
        position = int(positions[r])
        for kv_head in range(kv_head_count):
            leaves, counts, gaps = select_for_head(
                grouped[r, kv_head],
                position,
                levels,
                kv_head,
                block_size,
                branching,
                keep_counts,
                scale,
                forced_blocks,
            )
            key_ids = (leaves[:, None] * block_size + torch.arange(block_size)).flatten()
            key_ids = key_ids[key_ids <= position]
            selected_key_ids[r, kv_head, : key_ids.shape[0]] = key_ids.int()
            scored_counts[r, kv_head] = torch.tensor(counts, dtype=torch.int32)
            boundary_gaps[r, kv_head] = torch.tensor(gaps)
    return PisaSelection(selected_key_ids, scored_counts, boundary_gaps)
