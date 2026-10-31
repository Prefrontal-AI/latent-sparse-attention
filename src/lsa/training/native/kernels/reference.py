"""The bucket read in plain torch - the reference the fused selection kernels are held to and fall back on.

A query scores the unit centroids of the buckets closed before its block, keeps its `top_p` best, and takes their keys
best bucket first, skipping the sinks and the window it reads anyway, until `top_k` are taken.
Index queries are `[b, n, index_dim]`, or `[b, n, h, index_dim]` for several heads of one query, which rank a centroid
by their best head.
"""

import torch

# Position of a slot that holds no key
PAD = -1
# Elements of the [b, rows, h, m] per-head scores held at once
HEAD_SCORE_ELEMENTS = 2**26


def scores(index_queries: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Score every bucket for every query, by the query's best head where it has several.

    Args:
        index_queries: Index queries [b, n, index_dim] or [b, n, h, index_dim].
        centroids: Unit bucket centroids [b, m, index_dim].

    Returns:
        Scores [b, n, m] fp32.
    """
    q = index_queries.float()
    if q.dim() == 3:
        return torch.einsum("bnd,bmd->bnm", q, centroids)
    b, n, h, _ = q.shape
    rows = max(1, HEAD_SCORE_ELEMENTS // (b * h * max(1, centroids.shape[1])))
    out = [torch.einsum("bnhd,bmd->bnhm", q[:, s : s + rows], centroids).amax(2) for s in range(0, n, rows)]
    return torch.cat(out, dim=1)


def select_indices(
    index_queries: torch.Tensor,
    q_pos: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor] | None,
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    """Select each query's long-range keys, best bucket first.

    Each bucket's keys come in position order, PAD past the keys a query takes.

    Args:
        index_queries: Index queries [b, n_q, d] or [b, n_q, h, d], see scores.
        q_pos: Query positions [n_q].
        buckets: cluster's centroids and key positions, or None.
        Others: See the module docstring.

    Returns:
        Positions [b, n_q, top_k] int64.
    """
    b, n_q = index_queries.shape[:2]
    selected = torch.full((b, n_q, top_k), PAD, dtype=torch.long, device=index_queries.device)
    if buckets is None:
        return selected
    centroids, members = buckets
    m, cap = members.shape[1:]
    bucket_block = torch.arange(m, device=index_queries.device) // (block // bucket_size)
    pick_count = min(top_p, m)
    rows_at_once = max(1, 2**25 // (b * pick_count * cap))
    batch = torch.arange(b, device=index_queries.device)[:, None, None]
    for s in range(0, n_q, rows_at_once):
        pos = q_pos[s : s + rows_at_once]
        visible = bucket_block[None, :] < (pos // block)[:, None]
        ranked = scores(index_queries[:, s : s + rows_at_once], centroids)
        top = ranked.masked_fill(~visible, float("-inf")).topk(pick_count, dim=-1)
        ids = members[batch, top.indices].flatten(2)
        live = (top.values > float("-inf")).repeat_interleave(cap, dim=-1)
        valid = live & (ids >= sink_size) & (ids <= (pos - window_size)[None, :, None])
        rank = valid.cumsum(-1) - 1
        taken = valid & (rank < top_k)
        # Every key not taken lands in a spare last slot, dropped after
        packed = torch.full((b, len(pos), top_k + 1), PAD, dtype=torch.long, device=index_queries.device)
        packed.scatter_(-1, torch.where(taken, rank, top_k), torch.where(taken, ids, PAD))
        selected[:, s : s + rows_at_once] = packed[..., :top_k]
    return selected
