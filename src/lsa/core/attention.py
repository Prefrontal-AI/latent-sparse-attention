"""The sparse read - full attention restricted to each query's keep-set.

keep = window U sinks U whatever the indexer selected, then one causal softmax over it.
The native q, k, v and scale are reused verbatim, so at a budget of `T` or more this reproduces dense causal
attention exactly - which is the anchor that says the read itself adds no error.

Blocked over queries, so a `[Tq, Tk]` matrix never exists beyond one block's worth.
"""

import torch

from lsa import constants as const


def keep_set(
    selected_key_ids: torch.Tensor,
    query_start: int,
    query_count: int,
    key_count: int,
    window: int,
    sink: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the read mask for one query block.

    Selection, then window, then sinks, and causality last so it overrides all three.

    Args:
        selected_key_ids: Absolute key positions [query_count, budget] in int32.
            Values in `[0, key_count)` are live and `PAD` entries select nothing.
        query_start: Absolute position of the first query row.
        query_count: Number of query rows in the block.
        key_count: Number of available key positions.
        window: Number of recent keys that each query can read.
        sink: Number of initial keys that every query can read.
        device: Device on which to construct the mask.

    Returns:
        A boolean read mask [query_count, key_count] on ``device``.
    """

    # Absolute query positions [query_count] and key positions [key_count], both int64
    query_positions = query_start + torch.arange(query_count, device=device)
    key_positions = torch.arange(key_count, device=device)

    # Selected keys [query_count, key_count] as a boolean mask
    read_mask = torch.zeros(query_count, key_count, dtype=torch.bool, device=device)
    read_mask.scatter_(
        1,
        selected_key_ids.clamp_min(0).long(),
        selected_key_ids != const.Sentinel.PAD,
    )

    # Add local-window and sink visibility to every query row
    read_mask |= (query_positions[:, None] - key_positions[None, :] < window) | (
        key_positions[None, :] < sink
    )

    # Apply causality after combining every visibility source
    read_mask &= query_positions[:, None] >= key_positions[None, :]

    return read_mask


def topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_key_ids: torch.Tensor,
    window: int,
    sink: int,
    scale: float,
    query_start: int = 0,
    query_block_size: int = const.Chunk.QUERY,
) -> torch.Tensor:
    """Apply causal attention over the window, sinks, and selected long-range keys.

    Query row i sits at absolute position `query_start + i`, and `PAD` in ``selected_key_ids`` is padding.
    Scores are taken in fp32 and the output is cast back to q's dtype.

    Args:
        q: Queries [B, Tq, Hq, Dh].
        k: Keys [B, Tk, Hkv, Dh].
        v: Values [B, Tk, Hkv, Dh].
        selected_key_ids: Long-range key positions [B, Tq, budget] in int32.
        window: Number of recent keys that each query can read.
        sink: Number of initial keys that every query can read.
        scale: Multiplier applied to query-key scores.
        query_start: Absolute position of the first query row.
        query_block_size: Maximum number of query rows processed together.

    Returns:
        Attention output with the same shape, dtype, and device as ``q``.
    """

    batch_size, query_count, query_head_count, head_dim = q.shape
    key_count = k.shape[1]
    kv_head_count = k.shape[2]

    # Key-value head read by each query head [query_head_count] int64
    query_to_kv_head = torch.arange(query_head_count, device=q.device) // (query_head_count // kv_head_count)

    # Attention output filled one query block at a time
    output = torch.empty_like(q)
    for batch_index in range(batch_size):
        # Keys and values [key_count, query_head_count, head_dim] broadcast across query-head groups
        keys_by_query_head = k[batch_index].float()[:, query_to_kv_head, :]
        values_by_query_head = v[batch_index].float()[:, query_to_kv_head, :]
        for query_block_start in range(0, query_count, query_block_size):
            query_block_end = min(query_block_start + query_block_size, query_count)

            # Boolean read mask [block_query_count, key_count]
            read_mask = keep_set(
                selected_key_ids[batch_index, query_block_start:query_block_end],
                query_start + query_block_start,
                query_block_end - query_block_start,
                key_count,
                window,
                sink,
                q.device,
            )

            # Scaled attention logits [block_query_count, query_head_count, key_count] fp32
            attention_logits = (
                torch.einsum(
                    "qhd,shd->qhs",
                    q[batch_index, query_block_start:query_block_end].float(),
                    keys_by_query_head,
                )
                * scale
            )

            # Probabilities over retained keys with the same leading dimensions as the logits
            attention_probabilities = torch.softmax(
                attention_logits.masked_fill(~read_mask[:, None, :], const.Math.NINF),
                dim=-1,
            )

            # Weighted values [block_query_count, query_head_count, head_dim]
            output[batch_index, query_block_start:query_block_end] = torch.einsum(
                "qhs,shd->qhd",
                torch.nan_to_num(attention_probabilities),
                values_by_query_head,
            ).to(q.dtype)

    return output
