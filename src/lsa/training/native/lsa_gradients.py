"""Carry log-mass and straight-through gradients through sparse selected positions.

On a GPU with bf16 or fp16 heads the log-mass gradient is the gather kernels' backward fed a zero output gradient and a log-sum-exp gradient, and the straight-through gradient is a Triton kernel that sums each score's gradient over the heads in registers.
Anywhere else both run in PyTorch over query chunks that group the query heads over their key head instead of repeating the gathered keys.
"""

from typing import Any

import torch
from torch.nn import functional as functional

from lsa.training.native.kernels import gather_sparse, straight_through

# Largest bytes of gathered keys and values, or of picked-key table, a query chunk holds
GATHER_BYTES = 2**30
# Bytes a chunk holds on the CPU, which has no free-memory query
CPU_GATHER_BYTES = 2**28
# Fraction of free device memory a chunk may take is 1 / this
FREE_FRACTION = 8
# Fp32 copies of a gathered row a chunk holds at once: keys, values, their product and gradient
ROW_COPIES = 4
# Query rows a straight-through kernel launch covers in whole tiles of this many
TILE_ROWS = 64


def positions(
    selected: torch.Tensor, start: int, end: int, window: int, sink: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """List selected, recent and sink positions once for each query chunk."""
    batch, tokens, _ = selected.shape
    device = selected.device
    queries = torch.arange(start, end, device=device)[:, None]
    recent = queries - torch.arange(window, device=device)[None]
    recent = torch.where(recent >= 0, recent, -1).expand(batch, -1, -1)
    sinks = torch.arange(sink, device=device)[None].expand(end - start, -1)
    sinks = torch.where(sinks <= queries - window, sinks, -1).expand(batch, -1, -1)
    ids = torch.cat([selected[:, start:end].long(), recent, sinks], dim=-1)
    return ids, ids >= 0


def gather_rows(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Gather each query row's key positions from every key head.

    Args:
        x: Keys or values [B, Hkv, T, head_dim].
        ids: Positions to gather [B, R, J] int64, `-1` read as position 0.

    Returns:
        Gathered rows [B, Hkv, R, J, head_dim].
    """
    batch_ids = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x.transpose(1, 2)[batch_ids, ids.clamp_min(0)].permute(0, 3, 1, 2, 4)


def grouped_logits(q: torch.Tensor, keys: torch.Tensor, scale: float) -> torch.Tensor:
    """Score each query head against its key head's gathered keys, with no key repeated for the group.

    Args:
        q: Queries [B, H, R, head_dim].
        keys: Gathered keys [B, Hkv, R, J, head_dim].
        scale: The logits' scale.

    Returns:
        Scaled logits [B, H, R, J] fp32.
    """
    batch, heads, rows, dim = q.shape
    grouped = q.float().view(batch, keys.shape[1], heads // keys.shape[1], rows, dim)
    logits = torch.einsum("bkgqd,bkqjd->bkgqj", grouped, keys.float()) * scale
    return logits.reshape(batch, heads, rows, keys.shape[3])


def chunk_rows(q: torch.Tensor, k: torch.Tensor, selected: torch.Tensor, window: int, sink: int) -> int:
    """Size a query chunk by the bytes of its gathered rows against the free device memory.

    Args:
        q: Queries [B, H, T, head_dim].
        k: Keys [B, Hkv, T, head_dim].
        selected: Selected positions [B, T, top_k].
        window: Keys a query's window holds.
        sink: Leading keys every query keeps.

    Returns:
        Query rows a chunk holds, at least 1.
    """
    budget = CPU_GATHER_BYTES
    if q.is_cuda:
        free, _ = torch.cuda.mem_get_info(q.device)
        budget = min(GATHER_BYTES, free // FREE_FRACTION)
    reads = selected.shape[-1] + window + sink
    row_bytes = 4 * q.shape[0] * reads * (ROW_COPIES * k.shape[1] * q.shape[-1] + 3 * q.shape[1])
    return max(1, budget // row_bytes)


def kernel_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.dtype | None:
    """Return the dtype the CUDA kernels take these heads at, or None where they take none."""
    if not q.is_cuda or q.shape[-1] not in gather_sparse.HEAD_DIMS:
        return None
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return q.dtype if q.dtype in gather_sparse.DTYPES and k.dtype == v.dtype == q.dtype else None


class LogMass(torch.autograd.Function):
    """Attach sparse query and key gradients to a fast read's log mass."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        selected: torch.Tensor,
        lse: torch.Tensor,
        window: int,
        sink: int,
        scale: float,
        members: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the log mass while retaining sparse read metadata."""
        ctx.save_for_backward(q, k, selected, lse, members)
        ctx.window, ctx.sink, ctx.scale = window, sink, scale
        ctx.dtype = kernel_dtype(q, k, k)
        return lse.view_as(lse)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        """Accumulate each sparse softmax probability's query and key derivative."""
        q, k, selected, lse, members = ctx.saved_tensors
        window, sink, scale = ctx.window, ctx.sink, ctx.scale
        if ctx.dtype is not None:
            dq, dk = gather_sparse.log_mass_gradients(
                q.transpose(1, 2).to(ctx.dtype),
                k.transpose(1, 2).to(ctx.dtype),
                selected,
                members,
                lse,
                gradient,
                window,
                sink,
                scale,
            )
            return (
                dq.transpose(1, 2).to(q.dtype),
                dk.transpose(1, 2).to(k.dtype),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        dq, dk = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32)
        batch, heads, tokens, dim = q.shape
        kv_heads = k.shape[1]
        rows = chunk_rows(q, k, selected, window, sink)
        for start in range(0, tokens, rows):
            end = min(start + rows, tokens)
            ids, valid = positions(selected, start, end, window, sink)
            keys = gather_rows(k, ids)
            logits = grouped_logits(q[:, :, start:end], keys, scale)
            probability = (logits - lse[:, :, start:end, None]).exp().masked_fill(~valid[:, None], 0.0)
            ds = (probability * gradient[:, :, start:end, None] * scale).view(
                batch, kv_heads, heads // kv_heads, end - start, ids.shape[-1]
            )
            grouped = q[:, :, start:end].float().view(batch, kv_heads, heads // kv_heads, end - start, dim)
            dq[:, :, start:end] = torch.einsum("bkgqj,bkqjd->bkgqd", ds, keys.float()).reshape(
                batch, heads, end - start, dim
            )
            key_grad = torch.einsum("bkgqj,bkgqd->bkqjd", ds, grouped)
            at = ids.clamp_min(0)[:, None, :, :, None].expand(-1, kv_heads, -1, -1, dim).flatten(2, 3)
            dk.scatter_add_(2, at, key_grad.flatten(2, 3))
        return dq.to(q.dtype), dk.to(k.dtype), None, None, None, None, None, None


class StraightThrough(torch.autograd.Function):
    """Propagate sparse selected and window logit gradients into index vectors."""

    @staticmethod
    def forward(
        ctx: Any,
        context: torch.Tensor,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        members: torch.Tensor,
        selected: torch.Tensor,
        read: tuple[torch.Tensor, ...],
        shape: tuple[int, int, float, bool],
        joint: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the unchanged read and retain linear and sparse metadata."""
        ctx.read, ctx.shape = read, shape
        ctx.save_for_backward(
            context if joint is None else joint, index_queries, index_keys, members, selected
        )
        q, k, v, _ = read
        ctx.fast = (
            q.is_cuda
            and q.dtype in gather_sparse.DTYPES
            and k.dtype == v.dtype == q.dtype
            and q.shape[-1] in gather_sparse.HEAD_DIMS
        )
        return context.view_as(context)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        """Accumulate centroid and recent-key derivatives from the read's score gradients."""
        context, index_queries, index_keys, members, selected = ctx.saved_tensors
        q, k, v, lse = ctx.read
        window, sink, scale, recent = ctx.shape
        batch, tokens, dim = index_keys.shape
        with torch.enable_grad():
            differentiable_keys = index_keys.detach().requires_grad_()
            units = functional.normalize(differentiable_keys.float(), dim=-1)
            batch_ids = torch.arange(batch, device=q.device)[:, None, None]
            sums = (units[batch_ids, members.clamp_min(0)] * (members >= 0)[..., None]).sum(2)
            centroids = functional.normalize(sums, dim=-1)
        owner = torch.zeros(batch, tokens + 1, dtype=torch.long, device=q.device)
        bucket_ids = torch.arange(members.shape[1], device=q.device).repeat_interleave(members.shape[2])
        owner.scatter_(1, torch.where(members >= 0, members, tokens).flatten(1), bucket_ids.expand(batch, -1))
        delta = (gradient.float() * context.float()).sum(-1)
        score_gradients = fast_gradients if ctx.fast else chunked_gradients
        query_grad, centroid_grad, unit_grad = score_gradients(
            (q, k, v, lse), gradient, delta, index_queries, units.detach(), centroids.detach(), owner, selected,
            members.shape[1], (window, sink, scale, recent),
        )  # fmt: skip
        with torch.enable_grad():
            (key_grad,) = torch.autograd.grad(
                (centroids, units), differentiable_keys, (centroid_grad, unit_grad)
            )
        return gradient, query_grad.to(index_queries.dtype), key_grad, None, None, None, None, None


def chunked_gradients(
    read: tuple[torch.Tensor, ...],
    gradient: torch.Tensor,
    delta: torch.Tensor,
    index_queries: torch.Tensor,
    units: torch.Tensor,
    centroids: torch.Tensor,
    owner: torch.Tensor,
    selected: torch.Tensor,
    bucket_count: int,
    shape: tuple[int, int, float, bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the index vectors' score gradients in PyTorch, over query chunks bounded by bytes.

    Args:
        read: Queries [B, H, T, head_dim], keys and values [B, Hkv, T, head_dim], and the log-sum-exp [B, H, T].
        gradient: Gradient of the read's output [B, H, T, head_dim].
        delta: Dot of the output gradient and the joint output [B, H, T] fp32.
        index_queries: Index queries [B, T, index_dim].
        units: Unit index keys [B, T, index_dim] fp32.
        centroids: Bucket centroids [B, M, index_dim] fp32.
        owner: Bucket of each key [B, T + 1] int64.
        selected: Selected positions [B, T, top_k].
        bucket_count: Buckets `M`.
        shape: Window, sinks, scale, and whether window keys get a gradient.

    Returns:
        Gradient of the index queries [B, T, index_dim] fp32.
        Gradient of the centroids [B, M, index_dim] fp32.
        Gradient of the unit index keys [B, T, index_dim] fp32.
    """
    q, k, v, lse = read
    window, sink, scale, recent = shape
    batch, heads, tokens, _ = q.shape
    kv_heads = k.shape[1]
    dim = units.shape[-1]
    centroid_grad = torch.zeros_like(centroids)
    unit_grad = torch.zeros_like(units)
    query_grad = torch.zeros_like(index_queries, dtype=torch.float32)
    rows = chunk_rows(q, k, selected, window, sink)
    batch_ids = torch.arange(batch, device=q.device)[:, None, None]
    for start in range(0, tokens, rows):
        end = min(start + rows, tokens)
        ids, valid = positions(selected, start, end, window, sink)
        logits = grouped_logits(q[:, :, start:end], gather_rows(k, ids), scale)
        probability = (logits - lse[:, :, start:end, None]).exp().masked_fill(~valid[:, None], 0.0)
        grouped = gradient[:, :, start:end].float().view(batch, kv_heads, heads // kv_heads, end - start, -1)
        value_gradient = torch.einsum("bkgqd,bkqjd->bkgqj", grouped, gather_rows(v, ids).float())
        value_gradient = value_gradient.reshape(batch, heads, end - start, ids.shape[-1])
        scores_grad = (probability * (value_gradient - delta[:, :, start:end, None])).sum(1)
        picked_grad = scores_grad[..., : selected.shape[-1]]
        selected_ids = selected[:, start:end].long()
        buckets = owner.gather(1, selected_ids.clamp_min(0).flatten(1)).reshape_as(selected_ids)
        if bucket_count:
            chosen = centroids[batch_ids, buckets]
            query_grad[:, start:end] += (picked_grad[..., None] * chosen).sum(2)
            at = buckets[..., None].expand(-1, -1, -1, dim).flatten(1, 2)
            updates = picked_grad[..., None] * index_queries[:, start:end, None].float()
            centroid_grad.scatter_add_(1, at, updates.flatten(1, 2))
        if recent:
            near_ids = ids[..., selected.shape[-1] : selected.shape[-1] + window]
            near_grad = scores_grad[..., selected.shape[-1] : selected.shape[-1] + window]
            near_grad = near_grad.masked_fill(near_ids < sink, 0.0)
            near_units = units[batch_ids, near_ids.clamp_min(0)]
            query_grad[:, start:end] += (near_grad[..., None] * near_units).sum(2)
            updates = near_grad[..., None] * index_queries[:, start:end, None].float()
            at = near_ids.clamp_min(0)[..., None].expand(-1, -1, -1, dim).flatten(1, 2)
            unit_grad.scatter_add_(1, at, updates.flatten(1, 2))
    return query_grad, centroid_grad, unit_grad


def fast_gradients(
    read: tuple[torch.Tensor, ...],
    gradient: torch.Tensor,
    delta: torch.Tensor,
    index_queries: torch.Tensor,
    units: torch.Tensor,
    centroids: torch.Tensor,
    owner: torch.Tensor,
    selected: torch.Tensor,
    bucket_count: int,
    shape: tuple[int, int, float, bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the index vectors' score gradients with the straight-through kernel, then two small matmuls.

    Args:
        Those of `chunked_gradients`.

    Returns:
        Those of `chunked_gradients`.
    """
    q, k, v, lse = read
    window, sink, scale, recent = shape
    batch, _, tokens, _ = q.shape
    buckets = max(1, bucket_count)
    picked = torch.zeros(batch, tokens, buckets, dtype=torch.float32, device=q.device)
    near = torch.zeros(batch, tokens, window, dtype=torch.float32, device=q.device) if recent else None
    gradient = gradient.to(q.dtype).contiguous()
    keys = owner[:, :tokens].int()
    span = max(TILE_ROWS, GATHER_BYTES // (batch * tokens) // TILE_ROWS * TILE_ROWS)
    for start in range(0, tokens, span):
        end = min(start + span, tokens)
        table = torch.zeros(batch, end - start, tokens + 1, dtype=torch.bool, device=q.device)
        table.scatter_(
            2, torch.where(selected[:, start:end] >= 0, selected[:, start:end].long(), tokens), True
        )
        straight_through.gradients(
            q, k, v, gradient, delta.contiguous(), lse, table[..., :tokens], keys, (start, end), buckets, window, sink, scale, picked, near,
        )  # fmt: skip
    index = index_queries.float()
    query_grad = torch.zeros_like(index)
    centroid_grad = torch.zeros_like(centroids)
    if bucket_count:
        query_grad += picked @ centroids
        centroid_grad += picked.transpose(1, 2) @ index
    unit_grad = torch.zeros_like(units)
    if near is not None:
        window_query, unit_grad = window_gradients(near, index, units, window)
        query_grad += window_query
    return query_grad, centroid_grad, unit_grad


def window_gradients(
    near: torch.Tensor, index_queries: torch.Tensor, units: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn each query's window-score gradients into gradients of its index query and of the window's unit keys.

    Each chunk of `window` queries sees its own and the previous `window - 1` keys, so a dense banded matrix covers it.

    Args:
        near: Gradient of each query's window scores [B, T, window] fp32, offset o the key o before it.
        index_queries: Index queries [B, T, index_dim] fp32.
        units: Unit index keys [B, T, index_dim] fp32.
        window: Keys a query's window holds, itself included.

    Returns:
        Gradient of the index queries [B, T, index_dim] fp32.
        Gradient of the unit index keys [B, T, index_dim] fp32.
    """
    batch, tokens, dim = units.shape
    padded = functional.pad(units, (0, 0, window - 1, 0))
    query_grad = torch.zeros_like(index_queries)
    unit_grad = torch.zeros_like(padded)
    offsets = torch.arange(window, device=units.device)
    for start in range(0, tokens, window):
        end = min(start + window, tokens)
        column = torch.arange(end - start, device=units.device)[:, None] - offsets + window - 1
        dense = torch.zeros(batch, end - start, end - start + window - 1, device=units.device)
        dense.scatter_(2, column.expand(batch, -1, -1), near[:, start:end])
        query_grad[:, start:end] = dense @ padded[:, start : end + window - 1]
        unit_grad[:, start : end + window - 1] += dense.transpose(1, 2) @ index_queries[:, start:end]
    return query_grad, unit_grad[:, window - 1 :]
