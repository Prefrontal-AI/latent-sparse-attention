"""The gradient LSA's straight-through bias hands its index scores, computed after the read.

Every logit a query reads gains s - stopgrad(s): its bucket's centroid score for a picked key, the key's own index score for a window key past the sinks.
The term is zero, so only its backward matters, which gives s the logit's gradient summed over heads,

    g_ij = sum_h P_hij (dO_hi . v_j - D_hi),    P_hij = exp(scale q_hi . k_j - lse_hi),    D_hi = dO_hi . O_hi,

summed over each query's picked keys bucket by bucket, and taken as is for its window keys.
Each g_ij needs only its own logit and the row's log-sum-exp, so neither is recomputed for keys whose gradient nobody wants.
A Triton kernel, flash-style: a tile of queries walks the key tiles holding a picked or window key, sums g over every head in registers, then adds each picked key's into its bucket atomically and stores each window key's.
The kernel is the reference training code's `straight_through.gradients`, over a span of query rows so the picked-key table is one span's.
"""

import torch
import triton
import triton.language as tl

# Query tile, key tile, warps and stages: the fastest that fit an H200's shared memory on 8 heads of 256
TILING = (64, 128, 8, 1)


@triton.jit
def _gradients_kernel(
    q_ptr: tl.tensor, k_ptr: tl.tensor, v_ptr: tl.tensor, do_ptr: tl.tensor, lse_ptr: tl.tensor, delta_ptr: tl.tensor, kept_ptr: tl.tensor, owner_ptr: tl.tensor, picked_ptr: tl.tensor, recent_ptr: tl.tensor,
    sq_b: tl.tensor, sq_h: tl.tensor, sq_t: tl.tensor, sk_b: tl.tensor, sk_h: tl.tensor, sk_t: tl.tensor, sv_b: tl.tensor, sv_h: tl.tensor, sv_t: tl.tensor, sdo_b: tl.tensor, sdo_h: tl.tensor, sdo_t: tl.tensor,
    t: tl.tensor, row_start: tl.tensor, row_end: tl.tensor, n_buckets: tl.tensor, scale: tl.tensor,
    H_KV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr, SINK: tl.constexpr, HAS_RECENT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
) -> None:  # fmt: skip
    """Add one tile of queries' score gradients into the picked buckets and the window slots."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rows = row_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < row_end
    dims = tl.arange(0, D)
    hi = tl.minimum(row_start + (pid_m + 1) * BLOCK_M, row_end)
    span = row_end - row_start
    for start in range(0, hi, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_ok = cols < t
        valid = row_ok[:, None] & col_ok[None, :]
        distance = rows[:, None] - cols[None, :]
        kept = tl.load(
            kept_ptr + b * span * t + (rows[:, None] - row_start).to(tl.int64) * t + cols[None, :],
            mask=valid,
            other=0,
        ).to(tl.int1)
        near = (distance >= 0) & (distance < WINDOW) & (cols[None, :] >= SINK) & valid
        if HAS_RECENT:
            wanted = kept | near
        else:
            wanted = kept
        if tl.max(wanted.to(tl.int32)) > 0:
            g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kv in range(H_KV):
                k = tl.load(
                    k_ptr + b * sk_b + kv * sk_h + cols[:, None] * sk_t + dims[None, :],
                    mask=col_ok[:, None],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr + b * sv_b + kv * sv_h + cols[:, None] * sv_t + dims[None, :],
                    mask=col_ok[:, None],
                    other=0.0,
                )
                for j in range(GROUP):
                    h = kv * GROUP + j
                    q = tl.load(
                        q_ptr + b * sq_b + h * sq_h + rows[:, None] * sq_t + dims[None, :],
                        mask=row_ok[:, None],
                        other=0.0,
                    )
                    do = tl.load(
                        do_ptr + b * sdo_b + h * sdo_h + rows[:, None] * sdo_t + dims[None, :],
                        mask=row_ok[:, None],
                        other=0.0,
                    )
                    lse = tl.load(lse_ptr + (b * H_KV * GROUP + h) * t + rows, mask=row_ok, other=0.0)
                    delta = tl.load(delta_ptr + (b * H_KV * GROUP + h) * t + rows, mask=row_ok, other=0.0)
                    logits = tl.dot(q, tl.trans(k)) * scale
                    probs = tl.exp(logits - lse[:, None])
                    d_probs = tl.dot(do, tl.trans(v))
                    g += probs * (d_probs - delta[:, None])
            g = tl.where(wanted, g, 0.0)
            owner = tl.load(owner_ptr + b * t + cols, mask=col_ok, other=0)
            tl.atomic_add(
                picked_ptr + (b * t + rows[:, None]) * n_buckets + owner[None, :],
                g,
                mask=kept,
                sem="relaxed",
            )
            if HAS_RECENT:
                tl.store(recent_ptr + (b * t + rows[:, None]) * WINDOW + distance, g, mask=near)


def gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_context: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    kept: torch.Tensor,
    owner: torch.Tensor,
    rows: tuple[int, int],
    n_buckets: int,
    window: int,
    sink: int,
    scale: float,
    d_picked: torch.Tensor,
    d_recent: torch.Tensor | None,
) -> None:
    """Add the index scores' gradients of a span of query rows into `d_picked` and `d_recent`.

    Args:
        q: Queries [B, H, T, head_dim] bf16 or fp16, unit stride on the last dim.
        k: Keys [B, Hkv, T, head_dim], `Hkv` dividing `H`.
        v: Values [B, Hkv, T, head_dim].
        d_context: Gradient of the read's output [B, H, T, head_dim] in q's dtype, unit stride on the last dim.
        delta: Row term, the dot of the output gradient and the output [B, H, T] fp32, contiguous.
        lse: Log-sum-exp of the read's scaled scores [B, H, T] fp32, contiguous.
        kept: Picked long-range keys of each query in the span [B, end - start, T] bool.
        owner: Bucket of each key [B, T] int32.
        rows: The first and one past the last query row of the span.
        n_buckets: Buckets the picked scores span.
        window: Keys a query's window holds, itself included.
        sink: Leading keys every query keeps.
        scale: The scores' scale.
        d_picked: Gradient of each query's bucket scores [B, T, n_buckets] fp32, added into.
        d_recent: Gradient of each query's window scores [B, T, window] fp32, offset o the key o before it, stored into, or None to skip the window.
    """
    batch, heads, tokens, _ = q.shape
    block_m, block_n, warps, stages = TILING
    start, end = rows
    _gradients_kernel[(triton.cdiv(end - start, block_m), batch)](
        q, k, v, d_context, lse, delta, kept.contiguous().view(torch.uint8), owner,
        d_picked, d_picked if d_recent is None else d_recent,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d_context.stride(0), d_context.stride(1), d_context.stride(2),
        tokens, start, end, n_buckets, scale,
        H_KV=k.shape[1], GROUP=heads // k.shape[1], D=q.shape[-1], WINDOW=window, SINK=sink,
        HAS_RECENT=d_recent is not None, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
