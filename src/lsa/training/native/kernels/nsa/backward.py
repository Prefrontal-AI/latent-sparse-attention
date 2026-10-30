"""NSA's backward kernels: the gradients of the compressed and the selected reads.

Each takes the forward's context o and fp32 lse (reference.py's conventions) and the context's gradient do, and returns
the gradients of the forward's differentiable inputs in their dtypes.
The queries are every position of the sequence.

The kernels are adapted from fla's NSA kernels (flash-linear-attention, fla/ops/nsa, MIT licensed,
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li) and differ from them in three ways:

- Any group size, without duplicating heads.
  The compressed read and the selected read's dk and dv flatten the (position, query head) pairs of a key-value head
  into rows, so a tile holds any number of rows whatever the group size is.
  The selected read's dq, which holds one position's heads, pads them to a tile of max(16, group_size) rows whose extra
  rows load zeros.
- Overlapping compression blocks: compressed token i is visible from position
  i * compress_stride + compress_block_size - 1 on, for any compress_block_size and compress_stride.
- Deterministic sums and no atomics: dk and dv of a selected block sum the queries that chose it in ascending order, in
  chunks whose fp32 partials a second kernel adds in a fixed order.

The head dims and select_block_size are powers of two of at least 16, and tensors are made contiguous here.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

# Rows (position, query head pairs) one program of the selected dk/dv kernel handles at most
# A block every query selects (the first one) is so split over many programs
_CHUNK_ROWS = 4096
# Bytes the selected dk/dv kernel's fp32 partials may take; longer inputs get longer chunks
_PART_BYTES = 128 * 2**20
# Programs (token tiles times row segments) the compressed dk/dv kernel aims for
# About half of them are busy, since only the rows after a token tile see it
_CMP_PROGRAMS = 2048


def _check_dim(name: str, dim: int) -> None:
    """Refuse a dim that is not a power of two of at least 16, the kernels' tile constraint.

    Args:
        name: What the dim is, for the message.
        dim: The dim.

    Raises:
        ValueError: The dim is below 16 or not a power of two.
    """
    if dim < 16 or dim & (dim - 1):
        raise ValueError(f"{name} must be a power of two of at least 16 - got {dim}")


@triton.jit
def _delta_kernel(
    o: tl.tensor, do: tl.tensor, delta: tl.tensor, N: tl.tensor, V: tl.constexpr, BN: tl.constexpr
) -> None:
    # Delta is rowsum(o * do) in fp32, one value per (batch, position, query head)
    """Compute the delta kernel operation."""
    o_n = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN)
    m_n = o_n < N
    o_v = tl.arange(0, V)
    offs = o_n[:, None] * V + o_v[None, :]
    b_o = tl.load(o + offs, mask=m_n[:, None], other=0.0).to(tl.float32)
    b_do = tl.load(do + offs, mask=m_n[:, None], other=0.0).to(tl.float32)
    tl.store(delta + o_n, tl.sum(b_o * b_do, 1), mask=m_n)


def _delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """fp32 rowsum(o * do) over the value dim, shaped like lse."""
    n, v = o.numel() // o.shape[-1], o.shape[-1]
    delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)
    bn = 64
    _delta_kernel[(triton.cdiv(n, bn),)](o, do, delta, n, V=v, BN=bn, num_warps=4)
    return delta


_CMP_CONFIGS = [
    triton.Config({"BC": bc, "BM": bm}, num_warps=w, num_stages=s)
    for bc in [64, 128]
    for bm in [64, 128]
    for w in [4, 8]
    for s in [2, 3]
]


@triton.jit
def _cmp_dkv_step(
    q: tl.tensor, do: tl.tensor, lse: tl.tensor, delta: tl.tensor, b_k: tl.tensor, b_v: tl.tensor, b_dk: tl.tensor, b_dv: tl.tensor, c_vis: tl.tensor, r0: tl.tensor, i_b: tl.tensor, i_h: tl.tensor, T: tl.tensor, qk_scale: tl.tensor,
    H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BM: tl.constexpr,
    MASK: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:  # fmt: skip
    # One tile of BM rows against the program's tokens
    # Without MASK every token is visible to every row
    # Rows past the end load zeros, which add nothing since their do is zero
    """Compute the cmp dkv step operation."""
    HQ: tl.constexpr = H * G
    o_r = r0 + tl.arange(0, BM)
    m_r = o_r < T * G
    o_t = o_r // G
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    q_row = (i_b * T + o_t).to(tl.int64) * HQ + i_h * G + o_r % G
    b_q = tl.load(q + q_row[:, None] * K + o_k[None, :], mask=m_r[:, None], other=0.0)
    b_do = tl.load(do + q_row[:, None] * V + o_v[None, :], mask=m_r[:, None], other=0.0)
    b_lse = tl.load(lse + q_row, mask=m_r, other=0.0) * 1.4426950408889634
    b_delta = tl.load(delta + q_row, mask=m_r, other=0.0)
    # Scores are BC by BM
    b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
    b_p = tl.exp2(b_s - b_lse[None, :])
    if MASK:
        b_p = tl.where((c_vis[:, None] <= o_t[None, :]) & m_r[None, :], b_p, 0.0)
    b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
    b_dp = tl.dot(b_v, tl.trans(b_do))
    b_ds = b_p * (b_dp - b_delta[None, :])
    b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
    return b_dk, b_dv


@triton.autotune(
    configs=_CMP_CONFIGS, key=["G", "K", "V", "COMPRESS_BLOCK_SIZE", "COMPRESS_STRIDE", "DIRECT"]
)
@triton.jit(do_not_specialize=["T", "TC", "SEG"])
def _cmp_bwd_dkv_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    lse: tl.tensor,
    delta: tl.tensor,
    do: tl.tensor,
    dk: tl.tensor,
    dv: tl.tensor,
    part_dk: tl.tensor,
    part_dv: tl.tensor,
    scale: tl.tensor,
    T: tl.tensor,
    TC: tl.tensor,
    SEG: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESS_BLOCK_SIZE: tl.constexpr,
    COMPRESS_STRIDE: tl.constexpr,
    DIRECT: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
) -> None:
    # One program per tile of BC compressed tokens of one (batch, key head) and one segment
    # Segments hold SEG rows (position, query head of the group), a multiple of BM
    # The program walks the segment's rows from the first that sees the tile
    # With DIRECT (one segment) it writes dk and dv, otherwise an fp32 partial the caller sums
    """Compute the cmp bwd dkv kernel operation."""
    i_c, i_seg, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_c = i_c * BC + tl.arange(0, BC)
    m_c = o_c < TC
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    kv_row = (i_b * TC + o_c).to(tl.int64) * H + i_h
    b_k = tl.load(k + kv_row[:, None] * K + o_k[None, :], mask=m_c[:, None], other=0.0)
    b_v = tl.load(v + kv_row[:, None] * V + o_v[None, :], mask=m_c[:, None], other=0.0)
    # The position from which each token is visible
    c_vis = o_c * COMPRESS_STRIDE + COMPRESS_BLOCK_SIZE - 1
    b_dk = tl.zeros([BC, K], dtype=tl.float32)
    b_dv = tl.zeros([BC, V], dtype=tl.float32)
    qk_scale = scale * 1.4426950408889634
    seg_lo = i_seg * SEG
    seg_hi = tl.minimum(seg_lo + SEG, T * G)
    # Rows from r_first on see the tile's first token, rows from r_full on see all its tokens
    r_first = ((i_c * BC * COMPRESS_STRIDE + COMPRESS_BLOCK_SIZE - 1) * G) // BM * BM
    r_full = (((i_c * BC + BC - 1) * COMPRESS_STRIDE + COMPRESS_BLOCK_SIZE - 1) * G + BM - 1) // BM * BM
    r_lo = tl.maximum(r_first, seg_lo)
    r_mid = tl.minimum(tl.maximum(r_full, r_lo), seg_hi)
    for r0 in range(r_lo, r_mid, BM):
        b_dk, b_dv = _cmp_dkv_step(
            q, do, lse, delta, b_k, b_v, b_dk, b_dv, c_vis, r0, i_b, i_h, T, qk_scale,
            H, G, K, V, BM, True,
        )  # fmt: skip
    for r0 in range(r_mid, seg_hi, BM):
        b_dk, b_dv = _cmp_dkv_step(
            q, do, lse, delta, b_k, b_v, b_dk, b_dv, c_vis, r0, i_b, i_h, T, qk_scale,
            H, G, K, V, BM, False,
        )  # fmt: skip
    b_dk *= scale
    if DIRECT:
        p_dk = dk + kv_row[:, None] * K + o_k[None, :]
        p_dv = dv + kv_row[:, None] * V + o_v[None, :]
        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_c[:, None])
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_c[:, None])
    else:
        n_kv = tl.num_programs(2) * TC
        o_p = i_seg * n_kv + kv_row
        tl.store(part_dk + o_p[:, None] * K + o_k[None, :], b_dk, mask=m_c[:, None])
        tl.store(part_dv + o_p[:, None] * V + o_v[None, :], b_dv, mask=m_c[:, None])


@triton.jit
def _cmp_dq_step(
    k: tl.tensor, v: tl.tensor, b_q: tl.tensor, b_do: tl.tensor, b_lse: tl.tensor, b_delta: tl.tensor, b_dq: tl.tensor, o_t: tl.tensor, c0: tl.tensor, c_end: tl.tensor, i_b: tl.tensor, i_h: tl.tensor, TC: tl.tensor, qk_scale: tl.tensor,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, COMPRESS_BLOCK_SIZE: tl.constexpr,
    COMPRESS_STRIDE: tl.constexpr, BC: tl.constexpr, MASK: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    # One tile of BC tokens against the program's rows
    # Without MASK every token is visible to every row and lies below c_end
    """Compute the cmp dq step operation."""
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    o_c = c0 + tl.arange(0, BC)
    kv_row = (i_b * TC + o_c).to(tl.int64) * H + i_h
    if MASK:
        m_c = o_c < c_end
        b_k = tl.load(k + kv_row[:, None] * K + o_k[None, :], mask=m_c[:, None], other=0.0)
        b_v = tl.load(v + kv_row[:, None] * V + o_v[None, :], mask=m_c[:, None], other=0.0)
    else:
        b_k = tl.load(k + kv_row[:, None] * K + o_k[None, :])
        b_v = tl.load(v + kv_row[:, None] * V + o_v[None, :])
    # Scores are BM by BC
    b_s = tl.dot(b_q, tl.trans(b_k)) * qk_scale
    b_p = tl.exp2(b_s - b_lse[:, None])
    if MASK:
        vis = ((o_c * COMPRESS_STRIDE + COMPRESS_BLOCK_SIZE - 1)[None, :] <= o_t[:, None]) & m_c[None, :]
        b_p = tl.where(vis, b_p, 0.0)
    b_dp = tl.dot(b_do, tl.trans(b_v))
    b_ds = b_p * (b_dp - b_delta[:, None])
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    return b_dq


@triton.autotune(configs=_CMP_CONFIGS, key=["G", "K", "V", "COMPRESS_BLOCK_SIZE", "COMPRESS_STRIDE"])
@triton.jit(do_not_specialize=["T", "TC"])
def _cmp_bwd_dq_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    lse: tl.tensor,
    delta: tl.tensor,
    do: tl.tensor,
    dq: tl.tensor,
    scale: tl.tensor,
    T: tl.tensor,
    TC: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESS_BLOCK_SIZE: tl.constexpr,
    COMPRESS_STRIDE: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
) -> None:
    # One program per tile of BM rows (position, query head of the group) of one (batch, head)
    # It walks the compressed tokens the tile's last position can see
    """Compute the cmp bwd dq kernel operation."""
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    HQ: tl.constexpr = H * G
    n_rows = T * G
    o_r = i_m * BM + tl.arange(0, BM)
    m_r = o_r < n_rows
    o_t = o_r // G
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    q_row = (i_b * T + o_t).to(tl.int64) * HQ + i_h * G + o_r % G
    b_q = tl.load(q + q_row[:, None] * K + o_k[None, :], mask=m_r[:, None], other=0.0)
    b_do = tl.load(do + q_row[:, None] * V + o_v[None, :], mask=m_r[:, None], other=0.0)
    b_lse = tl.load(lse + q_row, mask=m_r, other=0.0) * 1.4426950408889634
    b_delta = tl.load(delta + q_row, mask=m_r, other=0.0)
    b_dq = tl.zeros([BM, K], dtype=tl.float32)
    qk_scale = scale * 1.4426950408889634
    # Token i is visible to position p when i * COMPRESS_STRIDE + COMPRESS_BLOCK_SIZE - 1 <= p
    # Tokens below n_full are visible to every row of the tile, tokens from n_vis on to none
    p_first = (i_m * BM) // G
    p_last = (tl.minimum(i_m * BM + BM, n_rows) - 1) // G
    n_full = tl.minimum(
        tl.maximum(p_first - COMPRESS_BLOCK_SIZE + 1 + COMPRESS_STRIDE, 0) // COMPRESS_STRIDE, TC
    )
    n_vis = tl.minimum(
        tl.maximum(p_last - COMPRESS_BLOCK_SIZE + 1 + COMPRESS_STRIDE, 0) // COMPRESS_STRIDE, TC
    )
    c_mid = n_full // BC * BC
    for c0 in range(0, c_mid, BC):
        b_dq = _cmp_dq_step(
            k, v, b_q, b_do, b_lse, b_delta, b_dq, o_t, c0, c0 + BC, i_b, i_h, TC, qk_scale,
            H, K, V, COMPRESS_BLOCK_SIZE, COMPRESS_STRIDE, BC, False,
        )  # fmt: skip
    for c0 in range(c_mid, n_vis, BC):
        b_dq = _cmp_dq_step(
            k, v, b_q, b_do, b_lse, b_delta, b_dq, o_t, c0, n_vis, i_b, i_h, TC, qk_scale,
            H, K, V, COMPRESS_BLOCK_SIZE, COMPRESS_STRIDE, BC, True,
        )  # fmt: skip
    b_dq *= scale
    p_dq = dq + q_row[:, None] * K + o_k[None, :]
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_r[:, None])


def compressed_attention_bwd(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the gradients of `reference.compressed_attention`'s context.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim], every position.
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        compressed_values: Compressed values [batch_size, compressed_key_count, kv_head_count, head_dim].
        o: The forward's context [batch_size, query_count, query_head_count, head_dim], typically in q's dtype.
        lse: The forward's log-sum-exp [batch_size, query_count, query_head_count] fp32, 0 for a row with no visible token.
        do: The context's gradient [batch_size, query_count, query_head_count, head_dim].
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        scale: The factor logits are scaled by.

    Returns:
        dq, dk and dv of the compressed keys and values, in the dtypes of q, compressed_keys and compressed_values,
        accumulated in fp32.
    """
    batch_size, query_count, query_head_count, head_dim = q.shape
    compressed_key_count, kv_head_count = compressed_keys.shape[1], compressed_keys.shape[2]
    value_dim = compressed_values.shape[-1]
    group_size = query_head_count // kv_head_count
    _check_dim("head_dim", head_dim)
    _check_dim("value_dim", value_dim)
    q, compressed_keys, compressed_values, o, do = (
        x.contiguous() for x in (q, compressed_keys, compressed_values, o, do)
    )
    lse = lse.float().contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(compressed_keys)
    dv = torch.empty_like(compressed_values)
    if compressed_key_count == 0:
        return dq.zero_(), dk, dv
    delta = _delta(o, do)
    constants = dict(
        H=kv_head_count,
        G=group_size,
        K=head_dim,
        V=value_dim,
        COMPRESS_BLOCK_SIZE=compress_block_size,
        COMPRESS_STRIDE=compress_stride,
    )
    # Split the rows into segments so that a long sequence still has enough programs to fill the GPU, each segment an
    # fp32 partial
    # => one segment writes dk and dv directly
    row_count = query_count * group_size
    tile_count = triton.cdiv(compressed_key_count, 64) * batch_size * kv_head_count
    segment_count = max(1, min(triton.cdiv(row_count, 4096), triton.cdiv(_CMP_PROGRAMS, tile_count)))
    segment_rows = triton.cdiv(triton.cdiv(row_count, segment_count), 128) * 128
    segment_count = triton.cdiv(row_count, segment_rows)
    direct = segment_count == 1
    partial_dk = partial_dv = dk
    if not direct:
        partial_dk = torch.empty(segment_count, *compressed_keys.shape, dtype=torch.float32, device=q.device)
        partial_dv = torch.empty(
            segment_count, *compressed_values.shape, dtype=torch.float32, device=q.device
        )
    _cmp_bwd_dkv_kernel[lambda meta: (triton.cdiv(compressed_key_count, meta["BC"]), segment_count, batch_size * kv_head_count)](
        q, compressed_keys, compressed_values, lse, delta, do, dk, dv, partial_dk, partial_dv, scale,
        query_count, compressed_key_count, segment_rows, DIRECT=direct, **constants,
    )  # fmt: skip
    if not direct:
        dk = partial_dk.sum(0).to(compressed_keys.dtype)
        dv = partial_dv.sum(0).to(compressed_values.dtype)
    _cmp_bwd_dq_kernel[lambda meta: (triton.cdiv(row_count, meta["BM"]), batch_size * kv_head_count)](
        q, compressed_keys, compressed_values, lse, delta, do, dq, scale, query_count, compressed_key_count,
        **constants,
    )  # fmt: skip
    return dq, dk, dv


@triton.jit
def _slc_dq_step(
    k: tl.tensor, v: tl.tensor, b_q: tl.tensor, b_do: tl.tensor, b_lse: tl.tensor, b_delta: tl.tensor, b_dq: tl.tensor, i_s: tl.tensor, ok: tl.tensor, i_t: tl.tensor, kv_base: tl.tensor, qk_scale: tl.tensor,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BS: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    # One selected block starting at key i_s, its keys after i_t masked, all of it unless ok
    """Compute the slc dq step operation."""
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    o_s = i_s + tl.arange(0, BS)
    m_s = (o_s <= i_t) & ok
    kv_row = kv_base + o_s * H
    # Keys and values load transposed, K by BS and V by BS
    b_k = tl.load(k + kv_row[None, :] * K + o_k[:, None], mask=m_s[None, :], other=0.0)
    b_v = tl.load(v + kv_row[None, :] * V + o_v[:, None], mask=m_s[None, :], other=0.0)
    b_s = tl.dot(b_q, b_k) * qk_scale
    b_p = tl.where(m_s[None, :], tl.exp2(b_s - b_lse[:, None]), 0.0)
    b_dp = tl.dot(b_do, b_v)
    b_ds = b_p * (b_dp - b_delta[:, None])
    b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    return b_dq


@triton.autotune(
    configs=[
        triton.Config({"PIPE": pipe}, num_warps=w, num_stages=s)
        for pipe in [False, True]
        for w in [1, 2, 4]
        for s in [1, 2, 3, 4]
    ],
    key=["BG", "K", "V", "BS", "S"],
)
@triton.jit(do_not_specialize=["T"])
def _slc_bwd_dq_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    lse: tl.tensor,
    delta: tl.tensor,
    do: tl.tensor,
    dq: tl.tensor,
    indices: tl.tensor,
    scale: tl.tensor,
    T: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    PIPE: tl.constexpr,
) -> None:
    # One program per (position, batch, key head) walks the S selected blocks
    # The group's G query heads are padded to BG rows that load zeros and contribute nothing
    """Compute the slc bwd dq kernel operation."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    HQ: tl.constexpr = H * G
    o_g = tl.arange(0, BG)
    m_g = o_g < G
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    q_row = (i_b * T + i_t) * HQ + i_h * G + o_g
    b_q = tl.load(q + q_row[:, None] * K + o_k[None, :], mask=m_g[:, None], other=0.0)
    b_do = tl.load(do + q_row[:, None] * V + o_v[None, :], mask=m_g[:, None], other=0.0)
    b_lse = tl.load(lse + q_row, mask=m_g, other=0.0) * 1.4426950408889634
    b_delta = tl.load(delta + q_row, mask=m_g, other=0.0)
    b_dq = tl.zeros([BG, K], dtype=tl.float32)
    qk_scale = scale * 1.4426950408889634
    kv_base = i_b * T * H + i_h
    p_idx = indices + ((i_b * T + i_t) * H + i_h) * S
    for i in range(S):
        blk = tl.load(p_idx + i)
        if PIPE:
            # Every slot runs, an unused one with fully masked loads
            # The compiler can then prefetch the next block's keys while this one computes
            i_s = tl.maximum(blk, 0).to(tl.int64) * BS
            ok = (blk >= 0) & (i_s <= i_t)
            b_dq = _slc_dq_step(
                k, v, b_q, b_do, b_lse, b_delta, b_dq, i_s, ok, i_t, kv_base, qk_scale,
                H, K, V, BS,
            )  # fmt: skip
        else:
            i_s = blk.to(tl.int64) * BS
            if (blk >= 0) & (i_s <= i_t):
                b_dq = _slc_dq_step(
                    k, v, b_q, b_do, b_lse, b_delta, b_dq, i_s, True, i_t, kv_base, qk_scale,
                    H, K, V, BS,
                )  # fmt: skip
    b_dq *= scale
    p_dq = dq + q_row[:, None] * K + o_k[None, :]
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_g[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BR": br}, num_warps=w, num_stages=s)
        for br in [64, 128]
        for w in [4, 8]
        for s in [1, 2, 3]
    ],
    key=["G", "K", "V", "BS", "CQ"],
)
@triton.jit(do_not_specialize=["T", "NB", "NBLK"])
def _slc_bwd_dkv_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    lse: tl.tensor,
    delta: tl.tensor,
    do: tl.tensor,
    dk: tl.tensor,
    dv: tl.tensor,
    part_dk: tl.tensor,
    part_dv: tl.tensor,
    csr_t: tl.tensor,
    csr_off: tl.tensor,
    chunk_off: tl.tensor,
    part_off: tl.tensor,
    item_blk: tl.tensor,
    scale: tl.tensor,
    T: tl.tensor,
    NB: tl.tensor,
    NBLK: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    CQ: tl.constexpr,
    BR: tl.constexpr,
) -> None:
    # One program per work item: up to CQ of the positions that selected one key block
    # The positions, ascending, are flattened with their G heads into rows
    # A block with a single item writes dk and dv directly
    # Otherwise each item writes an fp32 partial for the reduce kernel
    """Compute the slc bwd dkv kernel operation."""
    i_item = tl.program_id(0)
    blk = tl.load(item_blk + i_item)
    if blk < NB:
        HQ: tl.constexpr = H * G
        c_first = tl.load(chunk_off + blk)
        n_chunks = tl.load(chunk_off + blk + 1) - c_first
        q_lo = tl.load(csr_off + blk) + (i_item - c_first) * CQ
        q_hi = tl.minimum(q_lo + CQ, tl.load(csr_off + blk + 1))
        j = blk % NBLK
        i_bh = blk // NBLK
        i_b, i_h = i_bh // H, i_bh % H
        o_s = j * BS + tl.arange(0, BS)
        m_s = o_s < T
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, V)
        kv_row = (i_b * T + o_s).to(tl.int64) * H + i_h
        b_k = tl.load(k + kv_row[:, None] * K + o_k[None, :], mask=m_s[:, None], other=0.0)
        b_v = tl.load(v + kv_row[:, None] * V + o_v[None, :], mask=m_s[:, None], other=0.0)
        b_dk = tl.zeros([BS, K], dtype=tl.float32)
        b_dv = tl.zeros([BS, V], dtype=tl.float32)
        qk_scale = scale * 1.4426950408889634
        for r0 in range(q_lo * G, q_hi * G, BR):
            o_r = r0 + tl.arange(0, BR)
            m_r = o_r < q_hi * G
            o_t = tl.load(csr_t + o_r // G, mask=m_r, other=0)
            q_row = (i_b * T + o_t).to(tl.int64) * HQ + i_h * G + o_r % G
            b_q = tl.load(q + q_row[:, None] * K + o_k[None, :], mask=m_r[:, None], other=0.0)
            b_do = tl.load(do + q_row[:, None] * V + o_v[None, :], mask=m_r[:, None], other=0.0)
            b_lse = tl.load(lse + q_row, mask=m_r, other=0.0) * 1.4426950408889634
            b_delta = tl.load(delta + q_row, mask=m_r, other=0.0)
            # Scores are BS by BR
            b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
            vis = (o_s[:, None] <= o_t[None, :]) & m_r[None, :]
            b_p = tl.where(vis, tl.exp2(b_s - b_lse[None, :]), 0.0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dk *= scale
        if n_chunks == 1:
            p_dk = dk + kv_row[:, None] * K + o_k[None, :]
            p_dv = dv + kv_row[:, None] * V + o_v[None, :]
            tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_s[:, None])
            tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_s[:, None])
        else:
            slot = tl.load(part_off + blk) + i_item - c_first
            o_p = slot.to(tl.int64) * BS + tl.arange(0, BS)
            tl.store(part_dk + o_p[:, None] * K + o_k[None, :], b_dk)
            tl.store(part_dv + o_p[:, None] * V + o_v[None, :], b_dv)


@triton.jit(do_not_specialize=["T", "NBLK"])
def _slc_bwd_reduce_kernel(
    dk: tl.tensor,
    dv: tl.tensor,
    part_dk: tl.tensor,
    part_dv: tl.tensor,
    chunk_off: tl.tensor,
    part_off: tl.tensor,
    T: tl.tensor,
    NBLK: tl.tensor,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
) -> None:
    # One program per key block writes zeros for a block nobody selected
    # A block split over several items gets the in-order sum of its partials
    # A block one item wrote is left alone
    """Compute the slc bwd reduce kernel operation."""
    blk = tl.program_id(0)
    c_first = tl.load(chunk_off + blk)
    n_chunks = tl.load(chunk_off + blk + 1) - c_first
    if n_chunks != 1:
        j = blk % NBLK
        i_bh = blk // NBLK
        i_b, i_h = i_bh // H, i_bh % H
        o_s = j * BS + tl.arange(0, BS)
        m_s = o_s < T
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, V)
        b_dk = tl.zeros([BS, K], dtype=tl.float32)
        b_dv = tl.zeros([BS, V], dtype=tl.float32)
        p_first = tl.load(part_off + blk)
        for c in range(p_first, p_first + n_chunks):
            o_p = tl.cast(c, tl.int64) * BS + tl.arange(0, BS)
            b_dk += tl.load(part_dk + o_p[:, None] * K + o_k[None, :])
            b_dv += tl.load(part_dv + o_p[:, None] * V + o_v[None, :])
        kv_row = (i_b * T + o_s).to(tl.int64) * H + i_h
        p_dk = dk + kv_row[:, None] * K + o_k[None, :]
        p_dv = dv + kv_row[:, None] * V + o_v[None, :]
        tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_s[:, None])
        tl.store(p_dv, b_dv.to(dv.dtype.element_ty), mask=m_s[:, None])


class BlockQueries(NamedTuple):
    """A selection inverted into, for every key block, the positions that selected it, ascending."""

    # Positions grouped by block id [selection entries] int32
    positions: torch.Tensor
    # Where each block id's positions start in `positions` [block_id_count + 1] int32
    position_offsets: torch.Tensor
    # Where each block id's work items start [block_id_count + 1] int32
    item_offsets: torch.Tensor
    # Where each block id's fp32 partials start, for a block of several items [block_id_count] int32
    partial_offsets: torch.Tensor
    # Each work item's block id, block_id_count past the last item [max_items] int32
    item_blocks: torch.Tensor
    # Bounds on the work items and the partials, which need no host sync to know
    max_items: int
    partial_count: int


def invert_selection(
    indices: torch.Tensor, select_block_size: int, block_count: int, chunk_queries: int
) -> BlockQueries:
    """Invert the selection into, per key block, the positions that selected it, all without a host sync.

    Block ids are (batch * kv_head_count + head) * block_count + block.
    A block's work items are the ceil(count / chunk_queries) chunks of its positions.

    Args:
        indices: The selected blocks [batch_size, query_count, kv_head_count, select_block_count] int32, -1 unused.
        select_block_size: Keys a selection block holds.
        block_count: Selection blocks in a sequence.
        chunk_queries: Positions one work item handles at most.

    Returns:
        The inverted selection.
    """
    batch_size, query_count, kv_head_count, select_block_count = indices.shape
    block_id_count = batch_size * kv_head_count * block_count
    device = indices.device
    blocks = indices.long()
    positions = torch.arange(query_count, device=device).view(1, query_count, 1, 1)
    valid = (blocks >= 0) & (blocks * select_block_size <= positions)
    stream = torch.arange(batch_size * kv_head_count, device=device).view(batch_size, 1, kv_head_count, 1)
    block_ids = torch.where(valid, stream * block_count + blocks, block_id_count).flatten().int()
    # A stable sort keeps each block's entries in (batch, position, head, slot) order
    sorted_ids, order = torch.sort(block_ids, stable=True)
    sorted_positions = (order // (kv_head_count * select_block_count) % query_count).int()
    # The ids are sorted, so each block's first entry is a binary search away
    position_offsets = torch.searchsorted(
        sorted_ids, torch.arange(block_id_count + 1, device=device, dtype=torch.int32)
    )
    chunks = (position_offsets.diff() + chunk_queries - 1) // chunk_queries
    item_offsets = torch.zeros(block_id_count + 1, dtype=torch.int64, device=device)
    item_offsets[1:] = chunks.cumsum(0)
    # Only the items of a block split over several hold a partial
    # A block of count n has ceil(n / chunk_queries) < 2 n / chunk_queries items, so they number fewer than 2 * entries / chunk_queries
    partial_offsets = torch.zeros(block_id_count, dtype=torch.int64, device=device)
    partial_offsets[1:] = torch.where(chunks > 1, chunks, 0)[:-1].cumsum(0)
    partial_count = max(1, 2 * triton.cdiv(block_ids.numel(), chunk_queries))
    max_items = block_id_count + triton.cdiv(block_ids.numel(), chunk_queries)
    items = torch.arange(max_items, device=device)
    item_blocks = torch.searchsorted(item_offsets[1:], items, right=True)
    return BlockQueries(
        sorted_positions,
        position_offsets.int(),
        item_offsets.int(),
        partial_offsets.int(),
        item_blocks.int(),
        max_items,
        partial_count,
    )


def selected_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the gradients of `reference.selected_attention`'s context, the indices held fixed.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim], every position.
        k: Keys [batch_size, key_count, kv_head_count, head_dim].
        v: Values [batch_size, key_count, kv_head_count, head_dim].
        o: The forward's context [batch_size, query_count, query_head_count, head_dim].
        lse: The forward's log-sum-exp [batch_size, query_count, query_head_count] fp32.
        do: The context's gradient [batch_size, query_count, query_head_count, head_dim].
        indices: The selected blocks [batch_size, query_count, kv_head_count, select_block_count] int32, -1 for an
            unused slot, a block at most once a row, and blocks after the query's own ignored.
        select_block_size: Keys a selection block holds.
        scale: The factor logits are scaled by.

    Returns:
        dq, dk and dv in the dtypes of q, k and v, accumulated in fp32 and deterministic.
    """
    batch_size, query_count, query_head_count, head_dim = q.shape
    kv_head_count, value_dim = k.shape[2], v.shape[-1]
    group_size = query_head_count // kv_head_count
    select_block_count = indices.shape[-1]
    _check_dim("head_dim", head_dim)
    _check_dim("value_dim", value_dim)
    _check_dim("select_block_size", select_block_size)
    q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
    indices = indices.to(torch.int32).contiguous()
    lse = lse.float().contiguous()
    delta = _delta(o, do)

    dq = torch.empty_like(q)
    padded_group = max(16, triton.next_power_of_2(group_size))
    _slc_bwd_dq_kernel[(query_count, batch_size * kv_head_count)](
        q, k, v, lse, delta, do, dq, indices, scale, query_count,
        H=kv_head_count, G=group_size, BG=padded_group, K=head_dim, V=value_dim, S=select_block_count,
        BS=select_block_size,
    )  # fmt: skip

    block_count = triton.cdiv(query_count, select_block_size)
    block_id_count = batch_size * kv_head_count * block_count
    # The partials take under 2 * entries / chunk_queries blocks of fp32 dk and dv
    partial_block_bytes = select_block_size * (head_dim + value_dim) * 4
    chunk_queries = max(
        1, _CHUNK_ROWS // group_size, triton.cdiv(2 * indices.numel() * partial_block_bytes, _PART_BYTES)
    )
    inverted = invert_selection(indices, select_block_size, block_count, chunk_queries)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    partial_dk = torch.empty(
        inverted.partial_count, select_block_size, head_dim, dtype=torch.float32, device=q.device
    )
    partial_dv = torch.empty(
        inverted.partial_count, select_block_size, value_dim, dtype=torch.float32, device=q.device
    )
    _slc_bwd_dkv_kernel[(inverted.max_items,)](
        q, k, v, lse, delta, do, dk, dv, partial_dk, partial_dv,
        inverted.positions, inverted.position_offsets, inverted.item_offsets, inverted.partial_offsets,
        inverted.item_blocks, scale, query_count, block_id_count, block_count,
        H=kv_head_count, G=group_size, K=head_dim, V=value_dim, BS=select_block_size, CQ=chunk_queries,
    )  # fmt: skip
    _slc_bwd_reduce_kernel[(block_id_count,)](
        dk, dv, partial_dk, partial_dv, inverted.item_offsets, inverted.partial_offsets, query_count,
        block_count, H=kv_head_count, K=head_dim, V=value_dim, BS=select_block_size, num_warps=4,
    )  # fmt: skip
    return dq, dk, dv
