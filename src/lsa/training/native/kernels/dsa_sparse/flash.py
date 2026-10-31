"""Flash attention over a DSA selection: forward, and backward to q, k and v, in Triton.

Each query i attends over its causal window_size (i - j < window_size), the sink_size (j < sink_size) and the
long-range keys its selection bitmask sets, which are only ever keys outside both. The
window_size and sink_size come from positions, so the bitmask holds only the long-range bits, and
the key tiles a query tile visits split three ways:

- sink_size tiles before the window_size band: dense, every key kept;
- long-range tiles, every element outside the window_size: the bitmask alone, visited from a
  list of the tiles the selection's occupancy marks, so an empty tile costs nothing;
- band tiles, from the first key a query of the tile sees in its window_size to the diagonal:
  the positional mask, and the bitmask only where the tile has long-range elements.

With a 64-aligned window_size and sink_size the band is one tile that needs the bitmask, the dense
tiles inside the window_size and the causal diagonal; unaligned sizes only add masked elements.
A tile reads the bitmask a word per 32 keys and expands it in registers: a load per element
costs four times the tile's softmax. The backward pass recomputes the probabilities under
the same mask, dq in one kernel and dk, dv in another, which walks every query head of its
key/value head's group and so sums their gradients without atomics.
"""

from typing import Any

import torch
import triton
import triton.language as tl

from lsa.training.native.kernels.dsa_sparse import topk

OCC_BLOCK = tl.constexpr(topk.OCC_BLOCK)

LOG2E = 1.4426950408889634
HEAD_DIMS = (16, 32, 64, 128, 192, 256)

# (BLOCK_M, BLOCK_N, num_warps, num_stages) per head_dim: forward and dq walk query tiles of
# BLOCK_M over key tiles of BLOCK_N, dk/dv key tiles of BLOCK_N over query tiles of BLOCK_M
FWD_CONFIGS = {
    16: (128, 64, 4, 3),
    32: (128, 64, 4, 3),
    64: (128, 64, 4, 3),
    128: (64, 64, 4, 3),
    192: (64, 64, 4, 2),
    256: (64, 64, 4, 2),
}
DQ_CONFIGS = {
    16: (128, 64, 4, 3),
    32: (128, 64, 4, 3),
    64: (128, 64, 8, 3),
    128: (64, 64, 4, 2),
    192: (64, 64, 4, 2),
    256: (64, 32, 4, 1),
}
DKV_CONFIGS = {
    16: (64, 128, 4, 3),
    32: (64, 128, 4, 3),
    64: (64, 128, 8, 3),
    128: (64, 64, 4, 2),
    192: (32, 64, 4, 2),
    256: (32, 64, 4, 1),
}


@triton.jit
def _row_bits(
    BITS: tl.tensor,
    stride_bt: tl.tensor,
    rows: tl.tensor,
    n0: tl.tensor,
    n_words: tl.tensor,
    active: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> tl.tensor:
    """The bitmask of queries rows over keys n0 onwards, (BLOCK_M, BLOCK_N) bool."""
    W: tl.constexpr = BLOCK_N // 32
    widx = n0 // 32 + tl.arange(0, W)
    mask = active & (widx[None, :] < n_words)
    words = tl.load(BITS + rows[:, None] * stride_bt + widx[None, :], mask=mask, other=0)
    lane = tl.arange(0, 32)
    return tl.reshape((words[:, :, None] >> lane[None, None, :]) & 1, [BLOCK_M, BLOCK_N]) != 0


@triton.jit
def _col_bits(
    BITS: tl.tensor,
    stride_bt: tl.tensor,
    rows: tl.tensor,
    n0: tl.tensor,
    n_words: tl.tensor,
    active: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> tl.tensor:
    """The same transposed, keys down and queries across, (BLOCK_N, BLOCK_M) bool."""
    W: tl.constexpr = BLOCK_N // 32
    widx = n0 // 32 + tl.arange(0, W)
    mask = active & (widx[:, None] < n_words)
    words = tl.load(BITS + rows[None, :] * stride_bt + widx[:, None], mask=mask, other=0)
    lane = tl.arange(0, 32)
    return tl.reshape((words[:, None, :] >> lane[None, :, None]) & 1, [BLOCK_N, BLOCK_M]) != 0


@triton.jit
def _keep(
    rows: tl.tensor,
    cols: tl.tensor,
    bit: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    MODE: tl.constexpr,
) -> tl.tensor:
    """Which (rows, cols) pairs attend in a tile of MODE 1 (long-range) or 2 (band)."""
    if MODE == 2:
        delta = rows - cols
        return ((delta >= 0) & ((delta < WINDOW) | (cols < SINK))) | bit
    if SINK != 0:
        # A long-range tile may straddle the sink_size's end
        return (cols < SINK) | bit
    return bit


@triton.jit
def _query_segments(
    m0: tl.tensor,
    T: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    BLOCK_M: tl.tensor,
    BLOCK_N: tl.tensor,
) -> tl.tensor:
    """The key tiles of the query tile at m0: [0, a) sink_size, [a, b) long-range, [b, c) band.

    _tile_lists computes the same bounds to list the long-range tiles.
    """
    last_row = tl.minimum(m0 + BLOCK_M, T) - 1
    band_end = last_row // BLOCK_N + 1
    band_start = tl.minimum(tl.maximum(m0 - WINDOW + 1, 0) // BLOCK_N, band_end)
    sink_end = tl.minimum(SINK // BLOCK_N, band_start)
    return sink_end, band_start, band_end


@triton.jit
def _fwd_tiles(
    acc: tl.tensor,
    l_i: tl.tensor,
    m_i: tl.tensor,
    q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    stride_kt: tl.tensor,
    stride_vt: tl.tensor,
    stride_bt: tl.tensor,
    rows: tl.tensor,
    dims: tl.tensor,
    m0: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    MODE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> tl.tensor:
    """Online softmax over key tiles of one MODE: 0 dense, 1 long-range (listed), 2 band."""
    dmask = dims[None, :] < HEAD_DIM
    for i in range(lo, hi):
        if MODE == 1:
            n = tl.load(LIST + i)
        else:
            n = i
        n0 = n * BLOCK_N
        cols = n0 + tl.arange(0, BLOCK_N)
        kmask = (cols[:, None] < T) & dmask
        k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
        v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        if MODE != 0:
            active = rows[:, None] < T
            if MODE == 2:
                # Only a tile with elements a window_size away reads the bitmask
                active = active & ((m0 + BLOCK_M - 1) - n0 >= WINDOW)
            bit = _row_bits(BITS, stride_bt, rows, n0, n_words, active, BLOCK_M, BLOCK_N)
            keep = _keep(rows[:, None], cols[None, :], bit, WINDOW, SINK, MODE)
            s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.math.exp2(s - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    OUT: tl.tensor,
    LSE: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    stride_vb: tl.tensor,
    stride_vt: tl.tensor,
    stride_vh: tl.tensor,
    stride_ob: tl.tensor,
    stride_ot: tl.tensor,
    stride_oh: tl.tensor,
    stride_bb: tl.tensor,
    stride_bh: tl.tensor,
    stride_bt: tl.tensor,
    H_Q: tl.tensor,
    H_SEL: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    GROUP: tl.constexpr,
    SEL_DIV: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute the attention output and log-sum-exp of a tile of queries, the longest tiles first.

    Each query tile walks the sink tiles, the listed long-range tiles and the band tiles in turn.

    Args:
        Q: Queries [B, T, H_Q, HEAD_DIM].
        K: Keys [B, T, H_KV, HEAD_DIM].
        V: Values [B, T, H_KV, HEAD_DIM].
        BITS: Long-range selection bitmask, 32 keys a word [B, H_SEL, T, n_words] int32.
        LIST: Key tiles with a selected key, per query tile [B, H_SEL, query tiles, key tiles] int32.
        COUNT: Tiles each LIST row holds [B, H_SEL, query tiles] int32.
        OUT: The output [B, T, H_Q, HEAD_DIM].
        LSE: The natural-log log-sum-exp [B, H_Q, T] fp32, -inf for a row that keeps no key.
        qk_scale: The softmax scale times log2(e).
    """
    # The longest query tiles first, so the causal tail fills the grid's last wave
    n_q = tl.num_programs(0)
    pid_m = n_q - 1 - tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_Q
    h = bh % H_Q
    h_sel = h // SEL_DIV
    m0 = pid_m * BLOCK_M
    rows = m0 + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    qmask = (rows[:, None] < T) & (dims[None, :] < HEAD_DIM)

    Q += b * stride_qb + h * stride_qh
    K += b * stride_kb + (h // GROUP) * stride_kh
    V += b * stride_vb + (h // GROUP) * stride_vh
    BITS += b * stride_bb + h_sel * stride_bh
    row = (b * H_SEL + h_sel) * n_q + pid_m
    LIST += row * tl.cdiv(T, BLOCK_N)
    count = tl.load(COUNT + row)
    q = tl.load(Q + rows[:, None] * stride_qt + dims[None, :], mask=qmask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    sink_end, band_start, band_end = _query_segments(m0, T, WINDOW, SINK, BLOCK_M, BLOCK_N)
    for mode in tl.static_range(3):
        if mode == 0:
            lo, hi = 0, sink_end
        elif mode == 1:
            lo, hi = 0, count
        else:
            lo, hi = band_start, band_end
        acc, l_i, m_i = _fwd_tiles(
            acc,
            l_i,
            m_i,
            q,
            K,
            V,
            BITS,
            LIST,
            stride_kt,
            stride_vt,
            stride_bt,
            rows,
            dims,
            m0,
            lo,
            hi,
            T,
            n_words,
            qk_scale,
            WINDOW,
            SINK,
            mode,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
        )

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    OUT += b * stride_ob + h * stride_oh
    out = (acc / l_safe[:, None]).to(OUT.dtype.element_ty)
    tl.store(OUT + rows[:, None] * stride_ot + dims[None, :], out, mask=qmask)
    # The natural-log LSE, -inf for a row that keeps no key
    lse = tl.where(empty, float("-inf"), (m_i + tl.math.log2(l_safe)) * 0.6931471805599453)
    tl.store(LSE + bh * T + rows, lse, mask=rows < T)


@triton.jit
def _dq_tiles(
    dq: tl.tensor,
    q: tl.tensor,
    do: tl.tensor,
    lse2: tl.tensor,
    delta: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    stride_kt: tl.tensor,
    stride_vt: tl.tensor,
    stride_bt: tl.tensor,
    rows: tl.tensor,
    dims: tl.tensor,
    m0: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    MODE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_LIST: tl.constexpr,
) -> tl.tensor:
    """Dq over key tiles of one MODE, as _fwd_tiles walks them.

    Without PIPELINE_LIST the listed tiles' loop is not software-pipelined: Triton 3.7.1's
    two-stage pipelining of it, for a length not a multiple of 16, faults on a shared address
    (b 2, t 1000, eight query heads over one key head).
    """
    dmask = dims[None, :] < HEAD_DIM
    for i in tl.range(lo, hi, num_stages=None if PIPELINE_LIST or MODE != 1 else 1):
        if MODE == 1:
            n = tl.load(LIST + i)
        else:
            n = i
        n0 = n * BLOCK_N
        cols = n0 + tl.arange(0, BLOCK_N)
        kmask = (cols[:, None] < T) & dmask
        k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
        v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.math.exp2(s - lse2[:, None])
        if MODE != 0:
            active = rows[:, None] < T
            if MODE == 2:
                active = active & ((m0 + BLOCK_M - 1) - n0 >= WINDOW)
            bit = _row_bits(BITS, stride_bt, rows, n0, n_words, active, BLOCK_M, BLOCK_N)
            keep = _keep(rows[:, None], cols[None, :], bit, WINDOW, SINK, MODE)
            p = tl.where(keep, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    return dq


@triton.jit
def _dq_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    DQ: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    stride_vb: tl.tensor,
    stride_vt: tl.tensor,
    stride_vh: tl.tensor,
    stride_db: tl.tensor,
    stride_dt: tl.tensor,
    stride_dh: tl.tensor,
    stride_bb: tl.tensor,
    stride_bh: tl.tensor,
    stride_bt: tl.tensor,
    stride_xb: tl.tensor,
    stride_xt: tl.tensor,
    stride_xh: tl.tensor,
    H_Q: tl.tensor,
    H_SEL: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    sm_scale: tl.tensor,
    GROUP: tl.constexpr,
    SEL_DIV: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_LIST: tl.constexpr,
) -> None:
    """Compute dq for a tile of queries, recomputing the probabilities under the forward's mask.

    Args:
        Q: Queries [B, T, H_Q, HEAD_DIM].
        K: Keys [B, T, H_KV, HEAD_DIM].
        V: Values [B, T, H_KV, HEAD_DIM].
        DO: Output gradient [B, T, H_Q, HEAD_DIM].
        LSE: The forward's log-sum-exp [B, H_Q, T] fp32.
        DELTA: Each query's output dotted with its output gradient [B, H_Q, T] fp32.
        BITS: Long-range selection bitmask, 32 keys a word [B, H_SEL, T, n_words] int32.
        LIST: Key tiles with a selected key, per query tile [B, H_SEL, query tiles, key tiles] int32.
        COUNT: Tiles each LIST row holds [B, H_SEL, query tiles] int32.
        DQ: Query gradient [B, T, H_Q, HEAD_DIM].
        qk_scale: The softmax scale times log2(e).
        sm_scale: The softmax scale.
    """
    n_q = tl.num_programs(0)
    pid_m = n_q - 1 - tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_Q
    h = bh % H_Q
    h_sel = h // SEL_DIV
    m0 = pid_m * BLOCK_M
    rows = m0 + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    qmask = (rows[:, None] < T) & (dims[None, :] < HEAD_DIM)

    Q += b * stride_qb + h * stride_qh
    DO += b * stride_db + h * stride_dh
    K += b * stride_kb + (h // GROUP) * stride_kh
    V += b * stride_vb + (h // GROUP) * stride_vh
    BITS += b * stride_bb + h_sel * stride_bh
    row = (b * H_SEL + h_sel) * n_q + pid_m
    LIST += row * tl.cdiv(T, BLOCK_N)
    count = tl.load(COUNT + row)
    q = tl.load(Q + rows[:, None] * stride_qt + dims[None, :], mask=qmask, other=0.0)
    do = tl.load(DO + rows[:, None] * stride_dt + dims[None, :], mask=qmask, other=0.0)
    lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=float("inf"))
    # A row that keeps no key has LSE -inf and every probability 0
    lse2 = tl.where(lse == float("-inf"), float("inf"), lse * 1.4426950408889634)
    delta = tl.load(DELTA + bh * T + rows, mask=rows < T, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    sink_end, band_start, band_end = _query_segments(m0, T, WINDOW, SINK, BLOCK_M, BLOCK_N)
    for mode in tl.static_range(3):
        if mode == 0:
            lo, hi = 0, sink_end
        elif mode == 1:
            lo, hi = 0, count
        else:
            lo, hi = band_start, band_end
        dq = _dq_tiles(
            dq,
            q,
            do,
            lse2,
            delta,
            K,
            V,
            BITS,
            LIST,
            stride_kt,
            stride_vt,
            stride_bt,
            rows,
            dims,
            m0,
            lo,
            hi,
            T,
            n_words,
            qk_scale,
            WINDOW,
            SINK,
            mode,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            PIPELINE_LIST,
        )
    DQ += b * stride_xb + h * stride_xh
    dq = (dq * sm_scale).to(DQ.dtype.element_ty)
    tl.store(DQ + rows[:, None] * stride_xt + dims[None, :], dq, mask=qmask)


@triton.jit
def _dkv_tiles(
    dk: tl.tensor,
    dv: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    Q: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    stride_qt: tl.tensor,
    stride_dt: tl.tensor,
    stride_bt: tl.tensor,
    cols: tl.tensor,
    dims: tl.tensor,
    n0: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    MODE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> tl.tensor:
    """Dk and dv of the key tile at n0 over query tiles of one MODE (1 listed)."""
    dmask = dims[None, :] < HEAD_DIM
    for i in range(lo, hi):
        if MODE == 1:
            m = tl.load(LIST + i)
        else:
            m = i
        m0 = m * BLOCK_M
        rows = m0 + tl.arange(0, BLOCK_M)
        qmask = (rows[:, None] < T) & dmask
        q = tl.load(Q + rows[:, None] * stride_qt + dims[None, :], mask=qmask, other=0.0)
        do = tl.load(DO + rows[:, None] * stride_dt + dims[None, :], mask=qmask, other=0.0)
        lse = tl.load(LSE + rows, mask=rows < T, other=float("inf"))
        lse2 = tl.where(lse == float("-inf"), float("inf"), lse * 1.4426950408889634)
        delta = tl.load(DELTA + rows, mask=rows < T, other=0.0)
        # Transposed tiles: keys down, queries across
        s = tl.dot(k, tl.trans(q)) * qk_scale
        p = tl.math.exp2(s - lse2[None, :])
        if MODE != 0:
            active = rows[None, :] < T
            if MODE == 2:
                active = active & ((m0 + BLOCK_M - 1) - n0 >= WINDOW)
            bit = _col_bits(BITS, stride_bt, rows, n0, n_words, active, BLOCK_M, BLOCK_N)
            keep = _keep(rows[None, :], cols[:, None], bit, WINDOW, SINK, MODE)
            p = tl.where(keep, p, 0.0)
        dv += tl.dot(p.to(do.dtype), do)
        dp = tl.dot(v, tl.trans(do))
        ds = p * (dp - delta[None, :])
        dk += tl.dot(ds.to(q.dtype), q)
    return dk, dv


@triton.jit
def _dkv_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    BITS: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    DK: tl.tensor,
    DV: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    stride_vb: tl.tensor,
    stride_vt: tl.tensor,
    stride_vh: tl.tensor,
    stride_db: tl.tensor,
    stride_dt: tl.tensor,
    stride_dh: tl.tensor,
    stride_bb: tl.tensor,
    stride_bh: tl.tensor,
    stride_bt: tl.tensor,
    stride_xb: tl.tensor,
    stride_xt: tl.tensor,
    stride_xh: tl.tensor,
    H_Q: tl.tensor,
    H_KV: tl.tensor,
    H_SEL: tl.tensor,
    T: tl.tensor,
    n_words: tl.tensor,
    qk_scale: tl.tensor,
    sm_scale: tl.tensor,
    GROUP: tl.constexpr,
    SEL_DIV: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute dk and dv for a tile of keys, walking every query head of the key/value head's group.

    Summing the group's gradients here needs no atomics.

    Args:
        Q: Queries [B, T, H_Q, HEAD_DIM].
        K: Keys [B, T, H_KV, HEAD_DIM].
        V: Values [B, T, H_KV, HEAD_DIM].
        DO: Output gradient [B, T, H_Q, HEAD_DIM].
        LSE: The forward's log-sum-exp [B, H_Q, T] fp32.
        DELTA: Each query's output dotted with its output gradient [B, H_Q, T] fp32.
        BITS: Long-range selection bitmask, 32 keys a word [B, H_SEL, T, n_words] int32.
        LIST: Query tiles with a selected key, per key tile [B, H_SEL, key tiles, query tiles] int32.
        COUNT: Tiles each LIST row holds [B, H_SEL, key tiles] int32.
        DK: Key gradient [B, T, H_KV, HEAD_DIM].
        DV: Value gradient [B, T, H_KV, HEAD_DIM].
        qk_scale: The softmax scale times log2(e).
        sm_scale: The softmax scale.
    """
    n_k = tl.num_programs(0)
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_KV
    h_kv = bh % H_KV
    n0 = pid_n * BLOCK_N
    cols = n0 + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    kmask = (cols[:, None] < T) & (dims[None, :] < HEAD_DIM)
    K += b * stride_kb + h_kv * stride_kh
    V += b * stride_vb + h_kv * stride_vh
    k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
    v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # From the diagonal: band tiles, then every later tile of a sink_size tile, or the listed ones
    n_q = tl.cdiv(T, BLOCK_M)
    band_start = n0 // BLOCK_M
    band_end = tl.minimum((n0 + BLOCK_N - 2 + WINDOW) // BLOCK_M + 1, n_q)
    dense_end = tl.where(n0 + BLOCK_N <= SINK, n_q, band_end)
    for g in range(GROUP):
        h = h_kv * GROUP + g
        h_sel = h // SEL_DIV
        Qh = Q + b * stride_qb + h * stride_qh
        DOh = DO + b * stride_db + h * stride_dh
        LSEh = LSE + (b * H_Q + h) * T
        DELTAh = DELTA + (b * H_Q + h) * T
        BITSh = BITS + b * stride_bb + h_sel * stride_bh
        row = (b * H_SEL + h_sel) * n_k + pid_n
        LISTh = LIST + row * n_q
        count = tl.load(COUNT + row)
        for mode in tl.static_range(3):
            if mode == 0:
                lo, hi = band_start, band_end
            elif mode == 1:
                lo, hi = band_end, dense_end
            else:
                lo, hi = 0, count
            dk, dv = _dkv_tiles(
                dk,
                dv,
                k,
                v,
                Qh,
                DOh,
                LSEh,
                DELTAh,
                BITSh,
                LISTh,
                stride_qt,
                stride_dt,
                stride_bt,
                cols,
                dims,
                n0,
                lo,
                hi,
                T,
                n_words,
                qk_scale,
                WINDOW,
                SINK,
                (mode + 2) % 3,
                HEAD_DIM,
                BLOCK_M,
                BLOCK_N,
            )
    DK += b * stride_xb + h_kv * stride_xh
    DV += b * stride_xb + h_kv * stride_xh
    out = cols[:, None] * stride_xt + dims[None, :]
    tl.store(DK + out, (dk * sm_scale).to(DK.dtype.element_ty), mask=kmask)
    tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=kmask)


@triton.jit
def _delta_kernel(
    OUT: tl.tensor,
    DO: tl.tensor,
    DELTA: tl.tensor,
    stride_ob: tl.tensor,
    stride_ot: tl.tensor,
    stride_oh: tl.tensor,
    stride_db: tl.tensor,
    stride_dt: tl.tensor,
    stride_dh: tl.tensor,
    H_Q: tl.tensor,
    T: tl.tensor,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
) -> None:
    """Each row's sum of o * do in fp32, the softmax backward's correction, (b, h_q, t)."""
    bh = tl.program_id(1)
    b = bh // H_Q
    h = bh % H_Q
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < T) & (dims[None, :] < HEAD_DIM)
    o = tl.load(OUT + b * stride_ob + h * stride_oh + rows[:, None] * stride_ot + dims[None, :], mask=mask)
    do = tl.load(DO + b * stride_db + h * stride_dh + rows[:, None] * stride_dt + dims[None, :], mask=mask)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTA + bh * T + rows, delta, mask=rows < T)


@triton.jit
def _occupied(
    OCC: tl.tensor,
    stride_oq: tl.tensor,
    q_tile: tl.tensor,
    k_tile: tl.tensor,
    inside: tl.tensor,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> tl.tensor:
    """Whether any selected bit lies in each (q_tile, k_tile) pair, pooled from 64 x 64 tiles.

    Only pairs inside are read: the rest may lie past the occupancy's edge.
    """
    QS: tl.constexpr = BLOCK_Q // OCC_BLOCK if BLOCK_Q >= OCC_BLOCK else 1
    KS: tl.constexpr = BLOCK_K // OCC_BLOCK if BLOCK_K >= OCC_BLOCK else 1
    q64 = (q_tile * BLOCK_Q) // OCC_BLOCK
    k64 = (k_tile * BLOCK_K) // OCC_BLOCK
    hit = tl.zeros(q_tile.shape, dtype=tl.int32)
    for i in tl.static_range(QS):
        for j in tl.static_range(KS):
            hit = tl.maximum(hit, tl.load(OCC + (q64 + i) * stride_oq + k64 + j, mask=inside, other=0))
    return hit != 0


@triton.jit
def _list_kernel(
    OCC: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    stride_ob: tl.tensor,
    stride_oh: tl.tensor,
    stride_oq: tl.tensor,
    H: tl.tensor,
    T: tl.tensor,
    n_rows: tl.tensor,
    n_cols: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KEYS_MAJOR: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """The occupied long-range tiles each attention tile visits, in order, and their count.

    Rows are query tiles and the list holds key tiles, as _query_segments bounds them; or,
    KEYS_MAJOR, rows are key tiles and the list holds the query tiles past the window_size band, as
    _dkv_kernel bounds them, none for a key tile wholly inside the sink_size. A tile straddling the
    sink_size is always listed.
    """
    r = tl.program_id(0)
    bh = tl.program_id(1)
    OCC += (bh // H) * stride_ob + (bh % H) * stride_oh
    if KEYS_MAJOR:
        n0 = r * BLOCK_K
        lo = tl.minimum((n0 + BLOCK_K - 2 + WINDOW) // BLOCK_Q + 1, n_cols)
        hi = tl.where(n0 + BLOCK_K <= SINK, lo, n_cols)
    else:
        m0 = r * BLOCK_Q
        band_end = (tl.minimum(m0 + BLOCK_Q, T) - 1) // BLOCK_K + 1
        hi = tl.minimum(tl.maximum(m0 - WINDOW + 1, 0) // BLOCK_K, band_end)
        lo = tl.minimum(SINK // BLOCK_K, hi)
    base = 0
    for start in range(lo, hi, CHUNK):
        cols = start + tl.arange(0, CHUNK)
        inside = cols < hi
        if KEYS_MAJOR:
            hit = _occupied(OCC, stride_oq, cols, r + 0 * cols, inside, BLOCK_Q, BLOCK_K)
            hit = hit | (n0 < SINK)
        else:
            hit = _occupied(OCC, stride_oq, r + 0 * cols, cols, inside, BLOCK_Q, BLOCK_K)
            hit = hit | (cols * BLOCK_K < SINK)
        keep = inside & hit
        slot = base + tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(LIST + (bh * n_rows + r) * n_cols + slot, cols, mask=keep)
        base += tl.sum(keep.to(tl.int32), axis=0)
    tl.store(COUNT + bh * n_rows + r, base)


def _tile_lists(
    occupancy: torch.Tensor,
    t: int,
    window_size: int,
    sink_size: int,
    block_q: int,
    block_k: int,
    keys_major: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """_list_kernel's lists, (b, h, rows, cols) int32, and their counts, (b, h, rows)."""
    b, h = occupancy.shape[:2]
    n_q, n_k = triton.cdiv(t, block_q), triton.cdiv(t, block_k)
    n_rows, n_cols = (n_k, n_q) if keys_major else (n_q, n_k)
    tiles = torch.empty(b, h, n_rows, n_cols, dtype=torch.int32, device=occupancy.device)
    count = torch.empty(b, h, n_rows, dtype=torch.int32, device=occupancy.device)
    _list_kernel[(n_rows, b * h)](
        occupancy, tiles, count, occupancy.stride(0), occupancy.stride(1), occupancy.stride(2),
        h, t, n_rows, n_cols, WINDOW=window_size, SINK=sink_size, BLOCK_Q=block_q, BLOCK_K=block_k,
        KEYS_MAJOR=keys_major, CHUNK=min(triton.next_power_of_2(n_cols), 256), num_warps=4,
    )  # fmt: skip
    return tiles, count


def _strides(x: torch.Tensor) -> tuple[int, int, int]:
    """The batch, time and head strides of a (b, t, h, d) tensor with unit stride along d."""
    return x.stride(0), x.stride(1), x.stride(2)


def _layout(q: torch.Tensor, k: torch.Tensor, bits: torch.Tensor) -> tuple[int, int]:
    """GROUP, query heads per key/value head, and SEL_DIV, query heads per selection head."""
    h_q, h_kv, h_sel = q.shape[2], k.shape[2], bits.shape[1]
    group = h_q // h_kv
    return group, 1 if h_sel == h_q else group


def _unit_stride(x: torch.Tensor) -> torch.Tensor:
    """Return x, made contiguous unless its last dim has unit stride."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward over a selection.

    Args:
        q: Queries [B, T, H_q, D].
        k: Keys [B, T, H_kv, D].
        v: Values [B, T, H_kv, D].
        bits: Long-range selection bitmask, 32 keys a word [B, H_sel, T, n_words] int32.
        occupancy: Which tiles of the selection hold a key [B, H_sel, n, n] int32.
        window_size: Recent keys each query reads.
        sink_size: First keys each query reads.
        scale: The factor logits are scaled by.

    Returns:
        The output like q.
        The log-sum-exp [B, H_q, T] fp32.
    """
    b, t, h_q, d = q.shape
    q, k, v = _unit_stride(q), _unit_stride(k), _unit_stride(v)
    group, sel_div = _layout(q, k, bits)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty(b, h_q, t, dtype=torch.float32, device=q.device)
    block_m, block_n, warps, stages = FWD_CONFIGS[d]
    tiles, count = _tile_lists(occupancy, t, window_size, sink_size, block_m, block_n, keys_major=False)
    _fwd_kernel[(triton.cdiv(t, block_m), b * h_q)](
        q, k, v, bits, tiles, count, o, lse,
        *_strides(q), *_strides(k), *_strides(v), *_strides(o),
        bits.stride(0), bits.stride(1), bits.stride(2),
        h_q, bits.shape[1], t, bits.shape[3], scale * LOG2E,
        GROUP=group, SEL_DIV=sel_div, WINDOW=window_size, SINK=sink_size, HEAD_DIM=d,
        BLOCK_D=triton.next_power_of_2(d), BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return o, lse


def _backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward over a selection.

    Args:
        do: Output gradient like q.
        q: Queries [B, T, H_q, D].
        k: Keys [B, T, H_kv, D].
        v: Values [B, T, H_kv, D].
        o: The forward's output like q.
        lse: The forward's log-sum-exp [B, H_q, T] fp32.
        bits: Long-range selection bitmask, 32 keys a word [B, H_sel, T, n_words] int32.
        occupancy: Which tiles of the selection hold a key [B, H_sel, n, n] int32.
        window_size: Recent keys each query reads.
        sink_size: First keys each query reads.
        scale: The factor logits are scaled by.

    Returns:
        The gradients of q, k and v.
    """
    b, t, h_q, d = q.shape
    h_kv = k.shape[2]
    q, k, v, do = _unit_stride(q), _unit_stride(k), _unit_stride(v), _unit_stride(do)
    group, sel_div = _layout(q, k, bits)
    delta = torch.empty(b, h_q, t, dtype=torch.float32, device=q.device)
    block_t = max(16, 4096 // triton.next_power_of_2(d))
    _delta_kernel[(triton.cdiv(t, block_t), b * h_q)](
        o, do, delta, *_strides(o), *_strides(do), h_q, t,
        HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d), BLOCK_T=block_t,
    )  # fmt: skip
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    common = dict(
        GROUP=group, SEL_DIV=sel_div, WINDOW=window_size, SINK=sink_size, HEAD_DIM=d,
        BLOCK_D=triton.next_power_of_2(d),
    )  # fmt: skip
    bit_strides = (bits.stride(0), bits.stride(1), bits.stride(2))

    block_m, block_n, warps, stages = DQ_CONFIGS[d]
    tiles, count = _tile_lists(occupancy, t, window_size, sink_size, block_m, block_n, keys_major=False)
    _dq_kernel[(triton.cdiv(t, block_m), b * h_q)](
        q, k, v, do, lse, delta, bits, tiles, count, dq,
        *_strides(q), *_strides(k), *_strides(v), *_strides(do), *bit_strides, *_strides(dq),
        h_q, bits.shape[1], t, bits.shape[3], scale * LOG2E, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, PIPELINE_LIST=t % 16 == 0,
        num_warps=warps, num_stages=stages, **common,
    )  # fmt: skip
    block_m, block_n, warps, stages = DKV_CONFIGS[d]
    tiles, count = _tile_lists(occupancy, t, window_size, sink_size, block_m, block_n, keys_major=True)
    _dkv_kernel[(triton.cdiv(t, block_n), b * h_kv)](
        q, k, v, do, lse, delta, bits, tiles, count, dk, dv,
        *_strides(q), *_strides(k), *_strides(v), *_strides(do), *bit_strides, *_strides(dk),
        h_q, h_kv, bits.shape[1], t, bits.shape[3], scale * LOG2E, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, num_warps=warps, num_stages=stages, **common,
    )  # fmt: skip
    return dq, dk, dv


@torch.library.custom_op("lsa::dsa_sparse_attention", mutates_args=())
def sparse_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run attention over a selection; see dsa_sparse.attention."""
    return _forward(q, k, v, bits, occupancy, window_size, sink_size, scale)


@sparse_attention_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an empty output like q and an empty log-sum-exp [B, H_q, T] fp32."""
    b, t, h_q, _ = q.shape
    return q.new_empty(q.shape), q.new_empty((b, h_q, t), dtype=torch.float32)


@torch.library.custom_op("lsa::dsa_sparse_attention_backward", mutates_args=())
def sparse_attention_backward_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk and dv of sparse_attention_op."""
    return _backward(do, q, k, v, o, lse, bits, occupancy, window_size, sink_size, scale)


@sparse_attention_backward_op.register_fake
def _(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    bits: torch.Tensor,
    occupancy: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty gradients like q, k and v."""
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    """Save the tensors and settings the backward needs."""
    q, k, v, bits, occupancy, window_size, sink_size, scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, bits, occupancy)
    ctx.window_size, ctx.sink_size, ctx.scale = window_size, sink_size, scale


def _autograd_backward(ctx: Any, do: torch.Tensor, _dlse: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Return the gradients of q, k and v, and none for the other inputs."""
    q, k, v, o, lse, bits, occupancy = ctx.saved_tensors
    dq, dk, dv = sparse_attention_backward_op(
        do, q, k, v, o, lse, bits, occupancy, ctx.window_size, ctx.sink_size, ctx.scale
    )
    return dq, dk, dv, None, None, None, None, None


sparse_attention_op.register_autograd(_autograd_backward, setup_context=_setup_context)
