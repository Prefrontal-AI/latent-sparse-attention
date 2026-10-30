"""Qwen's own sparse attention, QSA - the indexer and the read, in torch, as Qwen's server computes them.

    q[i, h] = rotate( norm( x_i W_Q^h ),          i )      index queries   [R, H, head_dim]
    k[b]    = rotate( norm( mean_{s in b} x_s W_K ), r b )  index keys      [B, head_dim]
    I[i, b] = sum_h relu( q[i, h] . k[b] ) / sqrt(head_dim)

A block is `r` consecutive keys, pooled before the rotation so its keys' phases are never averaged, and rotated at
its first position. A query reads the blocks its own position has closed - the `(i + 1) // r` before it - picks the
`top_k / r` best and expands them to keys, and always reads the keys of its own open block.
The norm is Gemma's, `(1 + weight)` its gain, and the rotation the model's own: its frequencies and its YaRN scale on
the first `rotary_dim` dims of each head, halves rotated against each other.

The trainer, the reports and the check on the served selection all read the indexer through this module.
"""

import math

import torch

from lsa import constants as const


def gemma_norm(rows: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize rows to unit RMS and scale them by `1 + weight`, in fp32.

    Args:
        rows: Rows [..., head_dim], any float dtype.
        weight: The learned gain offset [head_dim].
        eps: Added to the mean square.

    Returns:
        The normalized rows [..., head_dim] fp32.
    """
    rows = rows.float()
    return rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())


def rotate(
    rows: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    """Rotate the leading dims of each row by its position, the two halves of the rotated part against each other.

    Args:
        rows: Rows [N, ..., head_dim] fp32, the position dim first.
        positions: Each row's position [N] int64.
        inverse_frequencies: The model's frequencies [rotary_dim / 2] fp32.
        attention_scaling: What the model scales its cosine and sine tables by - YaRN's, 1 without it.

    Returns:
        The rotated rows, same shape, fp32.
    """
    half = inverse_frequencies.shape[0]
    angles = positions.to(inverse_frequencies.device).float()[:, None] * inverse_frequencies.float()[None]
    # [N, 1, ..., half] so the tables broadcast over any head dims between
    view = (rows.shape[0], *([1] * (rows.ndim - 2)), half)
    cos = (angles.cos() * attention_scaling).view(view)
    sin = (angles.sin() * attention_scaling).view(view)
    first, second, rest = rows[..., :half], rows[..., half : 2 * half], rows[..., 2 * half :]
    return torch.cat((first * cos - second * sin, second * cos + first * sin, rest), dim=-1)


def pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool every closed block of consecutive rows, dropping an open one at the end.

    Args:
        x: Rows [T, d], any float dtype.
        block_size: Rows a block holds.

    Returns:
        Block means [T // block_size, d] fp32.
    """
    block_count = x.shape[0] // block_size
    return x[: block_count * block_size].float().view(block_count, block_size, x.shape[1]).mean(1)


def block_scores(index_queries: torch.Tensor, index_keys: torch.Tensor) -> torch.Tensor:
    """Score every block for every query row - the heads' rectified dots, summed and scaled.

    Args:
        index_queries: Index queries [R, H, head_dim] fp32.
        index_keys: Index keys [B, head_dim] fp32.

    Returns:
        Block scores [R, B] fp32, causality not applied.
    """
    dots = torch.einsum("rhd,bd->rbh", index_queries.float(), index_keys.float())
    return dots.relu().sum(-1) / math.sqrt(index_queries.shape[-1])


def closed_blocks(positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Return how many blocks each position has closed, which are the blocks it may select.

    Args:
        positions: Query positions [R] int64.
        block_size: Keys a block holds.

    Returns:
        Closed block counts [R] int64.
    """
    return torch.div(positions.long() + 1, block_size, rounding_mode="floor")


def visible_blocks(positions: torch.Tensor, block_count: int, block_size: int) -> torch.Tensor:
    """Return which blocks each query row may select.

    Args:
        positions: Query positions [R] int64.
        block_count: Blocks in the document.
        block_size: Keys a block holds.

    Returns:
        Selectable-block mask [R, block_count] bool.
    """
    blocks = torch.arange(block_count, device=positions.device)
    return blocks[None] < closed_blocks(positions, block_size)[:, None]


def select(scores: torch.Tensor, positions: torch.Tensor, block_size: int, top_blocks: int) -> torch.Tensor:
    """Read the best closed blocks as keys, then the keys of each row's open block.

    Args:
        scores: Block scores [R, B] fp32.
        positions: Query positions [R] int64.
        block_size: Keys a block holds.
        top_blocks: Blocks a row selects at most.

    Returns:
        Key ids [R, top_blocks * block_size + block_size - 1] int32 - the selected blocks' keys, best block first,
        then the open block's, each -1 padded.
    """
    row_count, block_count = scores.shape
    positions = positions.long().to(scores.device)
    visible = visible_blocks(positions, block_count, block_size)
    picked = min(top_blocks, block_count)
    best = scores.masked_fill(~visible, const.Math.NINF).topk(picked, dim=1)
    blocks = torch.where(visible.gather(1, best.indices), best.indices, -1)
    blocks = torch.nn.functional.pad(blocks, (0, top_blocks - picked), value=-1)
    offsets = torch.arange(block_size, device=scores.device)
    expanded = torch.where(blocks[..., None] >= 0, blocks[..., None] * block_size + offsets, -1)
    open_start = closed_blocks(positions, block_size) * block_size
    open_keys = open_start[:, None] + offsets[: block_size - 1]
    open_keys = torch.where(open_keys <= positions[:, None], open_keys, -1)
    return torch.cat([expanded.view(row_count, -1), open_keys], dim=1).to(torch.int32)
