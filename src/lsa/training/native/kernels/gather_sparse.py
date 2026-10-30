"""Attention over each query's window_size, sink_size and selected keys, the selected ones gathered by index.

dsa_sparse.attention reads a selection as a bitmask over 64 x 64 tiles and skips only empty
tiles. LSA's and the indexer's long-range picks occupy every tile at a few percent fill, so it
does dense causal work. Here each query's long-range keys come as positions, (b, t, top_k)
lsa_select.select_indices' with PAD (-1) anywhere, and their rows of K and V are loaded by index:
the read is O(t * (window_size + sink_size + top_k)).

A program takes G consecutive queries and every query head of one key/value head, M = G * HP
rows (HP the group padded to a power of two), all sharing the one selection, and runs one
online softmax, so one log-sum-exp, over:

- the sink_size: keys below sink_size at least a window_size away, one tile;
- the band: the keys within each query's window_size, tiles of BLOCK_N from the first key the
  earliest query sees to the diagonal, masked by position;
- the gathered keys: KB of each query's positions at a time, a tile of G * KB rows of K and V
  loaded by index and scored by every row, masked block-diagonally to the query that picked
  them. A query reads one key row for all its heads, so with h_q = 8 the gathered rows are
  read at 8 rows of Q a key; G is the least that makes M the 16 rows tl.dot needs (2 queries
  of 8 heads, 16 of one), the mask's waste the price of a tensor-core tile.

A position outside [sink_size, i - window_size] is skipped, so the selection never doubles a window_size or
sink_size key; a position listed twice counts twice (select_indices lists none twice).

A query reads its first i - window_size - sink_size + 1 slots only, as many as the positions it can keep.

Backward: a dq kernel recomputes the forward's tiles, gathering, with no scatter. The gathered
keys' dk and dv come key-side: a query's consecutive listed keys of one group whose members rise
are a run, a query's take of one LSA bucket (members, given) or of 16 consecutive positions
(otherwise, and for keys no bucket holds). _runs_kernel counts each group's runs, a cumsum gives
each its offset, and a second pass lists each run there as its query and the bits of the members
it took, so a budget cut after a prefix of a bucket's members, or a member outside [sink_size, i -
window_size], is a missing bit. _group_kernel then loads a group's K and V rows once and walks the
runs that read it, a tile of query rows (runs times heads) at a time, accumulating dk and dv in
registers: q and do are read once a (bucket, query) pair, 512 B a key for eight heads at 8 keys
a bucket, where the earlier scatter wrote 1 KiB of fp32 a (query, key) pair (BACKWARD 'scatter'
keeps it). The sink_size's dk and dv come from a kernel over long query chunks, one atomic per chunk;
the band's from a key-tile kernel walking the queries whose window_size holds the tile, which adds
the gathered keys' and writes dk and dv.

At 1 x 32768, top_k 1024, on an H200 the backward takes about 4.4 ms for one head and 8.6 ms
for eight over one, against 12.5 and 12.2 by the scatter; the group kernel's share for eight
heads (3.9 ms) is its 4 KiB of q and do a run, which L2 holds little of.
"""

from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lsa.training.native.kernels.dsa_sparse.flash import _delta_kernel

LOG2E = 1.4426950408889634
HEAD_DIMS = (16, 32, 64, 128, 256)
DTYPES = (torch.float16, torch.bfloat16)
# Rows of a forward or dq program, tl.dot's least; 32 was slower for one head and eight
MIN_ROWS = 16
# Gathered keys a tile, forward (by padded group, 64 past it) and dq, and the kernels'
# (num_warps, num_stages): the fastest at 1 x 32768 of a sweep of 32-128 keys and 2-8 warps
FWD_GATHER = {1: 128}
DQ_GATHER = 64
FWD_LAUNCH = (4, 2)
DQ_LAUNCH = (4, 3)
# Keys a band tile
BLOCK_N = 64
# Query rows (queries times heads) a band or sink_size dk/dv tile, and query rows a sink_size program
DKV_ROWS = 64
SINK_ROWS = 8192
# The gathered keys' dk and dv: 'keys' by the key-side group kernel, 'scatter' by the dq kernel's
# Atomics a gathered row, kept to compare
BACKWARD = "keys"
# Keys a group: LSA's buckets padded, else consecutive positions
GROUP_KEYS = 16
# Counters a group's runs spread over, and the consecutive queries a counter takes in turn
RUN_LANES = 8
RUN_LANE_SPAN = 64
# A work item's bits: its first query from ITEM_QUERY up, its group from ITEM_GROUP, its chunk
ITEM_QUERY = tl.constexpr(44)
ITEM_GROUP = tl.constexpr(20)
# Runs a group-kernel work item, query rows a group tile by padded group (128 past it), programs
# An SM and (warps, stages): the fastest at 1 x 32768 of 64-256 rows and 64-1024 runs
GROUP_RUNS = 256
GROUP_ROWS = {1: 64}
GROUP_WAVES = 4
GROUP_LAUNCH = (4, 2)
# The band dk/dv kernel's (num_warps, num_stages), and its stages on Blackwell at a 256 head dim,
# where Triton's lowering needs 262720 bytes of shared memory at two against the 232448 there
DKV_LAUNCH = (4, 2)
DKV_BLACKWELL_STAGES = 1
# The same knobs on Blackwell, where a sweep at 8 x 4096, top_k 512-4096, of the 8-head by 128 and
# 4-head by 256 shapes took these: 128 gathered keys for eight query heads, and the dq kernel at two
# stages, the group kernel at three, a 64-row group tile for four heads, the band dk/dv at 8 warps
BLACKWELL_FWD_GATHER = {1: 128, 8: 128}
BLACKWELL_DQ_LAUNCH = (4, 2)
BLACKWELL_GROUP_LAUNCH = (4, 3)
BLACKWELL_GROUP_ROWS = {1: 64, 4: 64}
BLACKWELL_DKV_WARPS = 8


@triton.jit
def _update(
    acc: tl.tensor, l_i: tl.tensor, m_i: tl.tensor, s: tl.tensor, keep: tl.tensor, v: tl.tensor
) -> tl.tensor:
    """One online-softmax step over scores s (M, N) kept by keep, values v (N, D)."""
    s = tl.where(keep, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(s - m_safe[:, None])
    alpha = tl.math.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, l_i, m_new


@triton.jit
def _picks(
    IND: tl.tensor,
    stride_it: tl.tensor,
    m0: tl.tensor,
    j0: tl.tensor,
    T: tl.tensor,
    TOP_K: tl.tensor,
    WINDOW: tl.tensor,
    SINK: tl.tensor,
    G: tl.constexpr,
    KB: tl.constexpr,
) -> tl.tensor:
    """Positions j0.. of queries m0.., flat (G * KB,), and which lie in [sink_size, query - window_size]."""
    query = m0 + tl.arange(0, G)
    slot = j0 + tl.arange(0, KB)
    idx, valid = _listed(IND, stride_it, query[:, None], slot[None, :], T, TOP_K, WINDOW, SINK)
    return tl.reshape(idx, [G * KB]), tl.reshape(valid, [G * KB])


@triton.jit
def _listed(
    IND: tl.tensor,
    stride_it: tl.tensor,
    query: tl.tensor,
    slot: tl.tensor,
    T: tl.tensor,
    TOP_K: tl.tensor,
    WINDOW: tl.tensor,
    SINK: tl.tensor,
) -> tl.tensor:
    """The positions at slot of query, broadcast, and which the read keeps.

    Query i reads its first i - window_size - sink_size + 1 slots, as many as the positions it can keep.
    """
    held = (query < T) & (slot >= 0) & (slot < TOP_K) & (slot < query - WINDOW - SINK + 1)
    idx = tl.load(IND + query * stride_it + slot, mask=held, other=-1).to(tl.int32)
    return idx, held & (idx >= SINK) & (idx <= query - WINDOW) & (idx >= 0)


@triton.jit
def _gather_steps(
    m0: tl.tensor, G: tl.constexpr, WINDOW: tl.tensor, SINK: tl.tensor, TOP_K: tl.tensor
) -> tl.tensor:
    """Positions the queries m0.. can have picked at most: keys sink_size to query - window_size."""
    return tl.maximum(tl.minimum(m0 + G - WINDOW - SINK + 1, TOP_K), 0)


@triton.jit
def _sink_tile(
    K: tl.tensor,
    V: tl.tensor,
    stride_kt: tl.tensor,
    stride_vt: tl.tensor,
    query: tl.tensor,
    live: tl.tensor,
    dims: tl.tensor,
    T: tl.tensor,
    WINDOW: tl.tensor,
    SINK: tl.tensor,
    HEAD_DIM: tl.tensor,
    SINK_N: tl.tensor,
) -> tl.tensor:
    """The sink_size's keys and values, and which rows keep them: those a window_size or more away."""
    cols = tl.arange(0, SINK_N)
    kmask = (cols[:, None] < SINK) & (cols[:, None] < T) & (dims[None, :] < HEAD_DIM)
    k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
    v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
    far = query[:, None] - cols[None, :] >= WINDOW
    return k, v, live[:, None] & (cols[None, :] < SINK) & far


@triton.jit
def _band_tile(
    K: tl.tensor,
    V: tl.tensor,
    stride_kt: tl.tensor,
    stride_vt: tl.tensor,
    query: tl.tensor,
    live: tl.tensor,
    dims: tl.tensor,
    n0: tl.tensor,
    T: tl.tensor,
    WINDOW: tl.tensor,
    HEAD_DIM: tl.tensor,
    BLOCK_N: tl.tensor,
) -> tl.tensor:
    """The band's keys and values from n0, and which rows hold them in their window_size."""
    cols = n0 + tl.arange(0, BLOCK_N)
    kmask = (cols[:, None] < T) & (dims[None, :] < HEAD_DIM)
    k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
    v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
    gap = query[:, None] - cols[None, :]
    return k, v, live[:, None] & (gap >= 0) & (gap < WINDOW)


@triton.jit
def _band_bounds(
    m0: tl.tensor, T: tl.tensor, WINDOW: tl.tensor, G: tl.constexpr, BLOCK_N: tl.constexpr
) -> tl.tensor:
    """The band tiles of queries m0..: from the earliest query's first window_size key to the last."""
    return (tl.maximum(m0 - WINDOW + 1, 0) // BLOCK_N) * BLOCK_N, tl.minimum(m0 + G, T)


@triton.jit
def _fwd_sink(
    acc: tl.tensor, l_i: tl.tensor, m_i: tl.tensor, q: tl.tensor, K: tl.tensor, V: tl.tensor, stride_kt: tl.tensor, stride_vt: tl.tensor, query: tl.tensor, live: tl.tensor, dims: tl.tensor, T: tl.tensor, qk_scale: tl.tensor,
    WINDOW: tl.tensor, SINK: tl.tensor, HEAD_DIM: tl.tensor, SINK_N: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    """Fold the sink tile into the running online softmax and return it."""
    k, v, keep = _sink_tile(K, V, stride_kt, stride_vt, query, live, dims, T, WINDOW, SINK, HEAD_DIM, SINK_N)
    s = tl.dot(q, tl.trans(k)) * qk_scale
    return _update(acc, l_i, m_i, s, keep, v)


@triton.jit
def _fwd_band(
    acc: tl.tensor, l_i: tl.tensor, m_i: tl.tensor, q: tl.tensor, K: tl.tensor, V: tl.tensor, stride_kt: tl.tensor, stride_vt: tl.tensor, query: tl.tensor, live: tl.tensor, dims: tl.tensor, m0: tl.tensor, T: tl.tensor, qk_scale: tl.tensor,
    WINDOW: tl.tensor, HEAD_DIM: tl.tensor, G: tl.constexpr, BLOCK_N: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    """Fold the band tiles into the running online softmax and return it."""
    lo, hi = _band_bounds(m0, T, WINDOW, G, BLOCK_N)
    for n0 in range(lo, hi, BLOCK_N):
        k, v, keep = _band_tile(
            K, V, stride_kt, stride_vt, query, live, dims, n0, T, WINDOW, HEAD_DIM, BLOCK_N
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        acc, l_i, m_i = _update(acc, l_i, m_i, s, keep, v)
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    IND: tl.tensor,
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
    stride_ib: tl.tensor,
    stride_it: tl.tensor,
    H_Q: tl.tensor,
    H_KV: tl.tensor,
    HG: tl.tensor,
    T: tl.tensor,
    qk_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    TOP_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    G: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KB: tl.constexpr,
    SINK_N: tl.constexpr,
) -> None:
    """Compute the output and log-sum-exp of G consecutive queries and every query head of one key/value head.

    Args:
        Q: Queries [B, T, H_Q, HEAD_DIM].
        K: Keys [B, T, H_KV, HEAD_DIM].
        V: Values [B, T, H_KV, HEAD_DIM].
        IND: Selected key positions [B, T, TOP_K] int32, PAD (-1) anywhere.
        OUT: The output [B, T, H_Q, HEAD_DIM].
        LSE: The natural-log log-sum-exp [B, H_Q, T] fp32, -inf for a row that keeps no key.
        qk_scale: The softmax scale times log2(e).
    """
    M: tl.constexpr = G * HP
    m0 = tl.program_id(0) * G
    bk = tl.program_id(1)
    b = (bk // H_KV).to(tl.int64)
    kvh = bk % H_KV
    r = tl.arange(0, M)
    query = m0 + r // HP
    head = r % HP
    live = (query < T) & (head < HG)
    dims = tl.arange(0, BLOCK_D)
    dmask = dims[None, :] < HEAD_DIM
    Q += b * stride_qb + kvh * HG * stride_qh
    K += b * stride_kb + kvh * stride_kh
    V += b * stride_vb + kvh * stride_vh
    IND += b * stride_ib
    q_at = query[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :]
    q = tl.load(Q + q_at, mask=live[:, None] & dmask, other=0.0)

    acc = tl.zeros([M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    if SINK > 0:
        acc, l_i, m_i = _fwd_sink(
            acc, l_i, m_i, q, K, V, stride_kt, stride_vt, query, live, dims, T, qk_scale,
            WINDOW, SINK, HEAD_DIM, SINK_N,
        )  # fmt: skip
    acc, l_i, m_i = _fwd_band(
        acc, l_i, m_i, q, K, V, stride_kt, stride_vt, query, live, dims, m0, T, qk_scale,
        WINDOW, HEAD_DIM, G, BLOCK_N,
    )  # fmt: skip
    owner = (r // HP)[:, None] == (tl.arange(0, G * KB) // KB)[None, :]
    for j0 in range(0, _gather_steps(m0, G, WINDOW, SINK, TOP_K), KB):
        idx, valid = _picks(IND, stride_it, m0, j0, T, TOP_K, WINDOW, SINK, G, KB)
        kmask = valid[:, None] & dmask
        k = tl.load(K + idx[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
        v = tl.load(V + idx[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = live[:, None] & valid[None, :] & owner
        acc, l_i, m_i = _update(acc, l_i, m_i, s, keep, v)

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    OUT += b * stride_ob + kvh * HG * stride_oh
    o_at = query[:, None] * stride_ot + head[:, None] * stride_oh + dims[None, :]
    tl.store(OUT + o_at, (acc / l_safe[:, None]).to(OUT.dtype.element_ty), mask=live[:, None] & dmask)
    # The natural-log LSE, -inf for a row that keeps no key
    lse = tl.where(empty, float("-inf"), (m_i + tl.math.log2(l_safe)) * 0.6931471805599453)
    tl.store(LSE + (b * H_Q + kvh * HG + head) * T + query, lse, mask=live)


@triton.jit
def _grads(
    dq: tl.tensor,
    q: tl.tensor,
    do: tl.tensor,
    lse2: tl.tensor,
    delta: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    keep: tl.tensor,
    qk_scale: tl.tensor,
) -> tl.tensor:
    """Dq of one key tile, and its probabilities and score gradients for dk and dv."""
    s = tl.dot(q, tl.trans(k)) * qk_scale
    p = tl.where(keep, tl.math.exp2(s - lse2[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v))
    ds = p * (dp - delta[:, None])
    dq += tl.dot(ds.to(k.dtype), k)
    return dq, p, ds


@triton.jit
def _dq_sink(
    dq: tl.tensor, q: tl.tensor, do: tl.tensor, lse2: tl.tensor, delta: tl.tensor, K: tl.tensor, V: tl.tensor, stride_kt: tl.tensor, stride_vt: tl.tensor, query: tl.tensor, live: tl.tensor, dims: tl.tensor, T: tl.tensor, qk_scale: tl.tensor,
    WINDOW: tl.tensor, SINK: tl.tensor, HEAD_DIM: tl.tensor, SINK_N: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    """Add the sink tile's gradient to dq and return it."""
    k, v, keep = _sink_tile(K, V, stride_kt, stride_vt, query, live, dims, T, WINDOW, SINK, HEAD_DIM, SINK_N)
    dq, _, _ = _grads(dq, q, do, lse2, delta, k, v, keep, qk_scale)
    return dq


@triton.jit
def _dq_band(
    dq: tl.tensor, q: tl.tensor, do: tl.tensor, lse2: tl.tensor, delta: tl.tensor, K: tl.tensor, V: tl.tensor, stride_kt: tl.tensor, stride_vt: tl.tensor, query: tl.tensor, live: tl.tensor, dims: tl.tensor, m0: tl.tensor, T: tl.tensor, qk_scale: tl.tensor,
    WINDOW: tl.tensor, HEAD_DIM: tl.tensor, G: tl.constexpr, BLOCK_N: tl.constexpr,
) -> tl.tensor:  # fmt: skip
    """Add the band tiles' gradient to dq and return it."""
    lo, hi = _band_bounds(m0, T, WINDOW, G, BLOCK_N)
    for n0 in range(lo, hi, BLOCK_N):
        k, v, keep = _band_tile(
            K, V, stride_kt, stride_vt, query, live, dims, n0, T, WINDOW, HEAD_DIM, BLOCK_N
        )
        dq, _, _ = _grads(dq, q, do, lse2, delta, k, v, keep, qk_scale)
    return dq


@triton.jit
def _dq_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    IND: tl.tensor,
    DQ: tl.tensor,
    DK32: tl.tensor,
    DV32: tl.tensor,
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
    stride_xb: tl.tensor,
    stride_xt: tl.tensor,
    stride_xh: tl.tensor,
    stride_ib: tl.tensor,
    stride_it: tl.tensor,
    stride_ab: tl.tensor,
    stride_at: tl.tensor,
    stride_ah: tl.tensor,
    H_Q: tl.tensor,
    H_KV: tl.tensor,
    HG: tl.tensor,
    T: tl.tensor,
    qk_scale: tl.tensor,
    sm_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    TOP_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    G: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KB: tl.constexpr,
    SINK_N: tl.constexpr,
    SCATTER: tl.constexpr,
) -> None:
    """Compute dq for G consecutive queries and every query head of one key/value head.

    With SCATTER the gathered keys' dk and dv are also added to DK32 and DV32 by atomics.

    Args:
        Q: Queries [B, T, H_Q, HEAD_DIM].
        K: Keys [B, T, H_KV, HEAD_DIM].
        V: Values [B, T, H_KV, HEAD_DIM].
        DO: Output gradient [B, T, H_Q, HEAD_DIM].
        LSE: The forward's log-sum-exp [B, H_Q, T] fp32.
        DELTA: Each query's output dotted with its output gradient [B, H_Q, T] fp32.
        IND: Selected key positions [B, T, TOP_K] int32, PAD (-1) anywhere.
        DQ: Query gradient [B, T, H_Q, HEAD_DIM].
        DK32: Key gradient accumulator [B, T, H_KV, HEAD_DIM] fp32 - written only with SCATTER.
        DV32: Value gradient accumulator [B, T, H_KV, HEAD_DIM] fp32 - written only with SCATTER.
        qk_scale: The softmax scale times log2(e).
        sm_scale: The softmax scale.
    """
    M: tl.constexpr = G * HP
    m0 = tl.program_id(0) * G
    bk = tl.program_id(1)
    b = (bk // H_KV).to(tl.int64)
    kvh = bk % H_KV
    r = tl.arange(0, M)
    query = m0 + r // HP
    head = r % HP
    live = (query < T) & (head < HG)
    dims = tl.arange(0, BLOCK_D)
    dmask = dims[None, :] < HEAD_DIM
    Q += b * stride_qb + kvh * HG * stride_qh
    DO += b * stride_db + kvh * HG * stride_dh
    K += b * stride_kb + kvh * stride_kh
    V += b * stride_vb + kvh * stride_vh
    IND += b * stride_ib
    rmask = live[:, None] & dmask
    q_at = query[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :]
    q = tl.load(Q + q_at, mask=rmask, other=0.0)
    do_at = query[:, None] * stride_dt + head[:, None] * stride_dh + dims[None, :]
    do = tl.load(DO + do_at, mask=rmask, other=0.0)
    row = (b * H_Q + kvh * HG + head) * T + query
    lse = tl.load(LSE + row, mask=live, other=float("inf"))
    # A row that keeps no key has LSE -inf and every probability 0
    lse2 = tl.where(lse == float("-inf"), float("inf"), lse * 1.4426950408889634)
    delta = tl.load(DELTA + row, mask=live, other=0.0)

    dq = tl.zeros([M, BLOCK_D], dtype=tl.float32)
    if SINK > 0:
        dq = _dq_sink(
            dq, q, do, lse2, delta, K, V, stride_kt, stride_vt, query, live, dims, T, qk_scale,
            WINDOW, SINK, HEAD_DIM, SINK_N,
        )  # fmt: skip
    dq = _dq_band(
        dq, q, do, lse2, delta, K, V, stride_kt, stride_vt, query, live, dims, m0, T, qk_scale,
        WINDOW, HEAD_DIM, G, BLOCK_N,
    )  # fmt: skip
    owner = (r // HP)[:, None] == (tl.arange(0, G * KB) // KB)[None, :]
    DK32 += b * stride_ab + kvh * stride_ah
    DV32 += b * stride_ab + kvh * stride_ah
    for j0 in range(0, _gather_steps(m0, G, WINDOW, SINK, TOP_K), KB):
        idx, valid = _picks(IND, stride_it, m0, j0, T, TOP_K, WINDOW, SINK, G, KB)
        kmask = valid[:, None] & dmask
        k = tl.load(K + idx[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
        v = tl.load(V + idx[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
        keep = live[:, None] & valid[None, :] & owner
        dq, p, ds = _grads(dq, q, do, lse2, delta, k, v, keep, qk_scale)
        if SCATTER:
            # Each gathered row belongs to one query: its heads sum in the dot, one atomic a row
            dv = tl.dot(tl.trans(p.to(do.dtype)), do)
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
            at = idx[:, None] * stride_at + dims[None, :]
            tl.atomic_add(DV32 + at, dv, mask=kmask, sem="relaxed")
            tl.atomic_add(DK32 + at, dk, mask=kmask, sem="relaxed")

    DQ += b * stride_xb + kvh * HG * stride_xh
    dq_at = query[:, None] * stride_xt + head[:, None] * stride_xh + dims[None, :]
    tl.store(DQ + dq_at, (dq * sm_scale).to(DQ.dtype.element_ty), mask=rmask)


@triton.jit
def _dkv_rows(
    dk: tl.tensor,
    dv: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    cols: tl.tensor,
    Q: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_dt: tl.tensor,
    stride_dh: tl.tensor,
    lse_base: tl.tensor,
    q0: tl.tensor,
    q1: tl.tensor,
    T: tl.tensor,
    HG: tl.tensor,
    qk_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK_ONLY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP: tl.constexpr,
    ROWS: tl.constexpr,
) -> tl.tensor:
    """Dk and dv of keys cols over queries [q0, q1), ROWS query rows a tile, keys down."""
    QT: tl.constexpr = ROWS // HP
    r = tl.arange(0, ROWS)
    dims = tl.arange(0, BLOCK_D)
    dmask = dims[None, :] < HEAD_DIM
    for m0 in range(q0, q1, QT):
        query = m0 + r // HP
        head = r % HP
        live = (query < q1) & (query < T) & (head < HG)
        rmask = live[:, None] & dmask
        q_at = query[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :]
        q = tl.load(Q + q_at, mask=rmask, other=0.0)
        do_at = query[:, None] * stride_dt + head[:, None] * stride_dh + dims[None, :]
        do = tl.load(DO + do_at, mask=rmask, other=0.0)
        row = lse_base + head * T + query
        lse = tl.load(LSE + row, mask=live, other=float("inf"))
        lse2 = tl.where(lse == float("-inf"), float("inf"), lse * 1.4426950408889634)
        delta = tl.load(DELTA + row, mask=live, other=0.0)
        s = tl.dot(k, tl.trans(q)) * qk_scale
        gap = query[None, :] - cols[:, None]
        if SINK_ONLY:
            keep = live[None, :] & (gap >= WINDOW)
        else:
            keep = live[None, :] & (gap >= 0) & (gap < WINDOW)
        p = tl.where(keep, tl.math.exp2(s - lse2[None, :]), 0.0)
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
    DK32: tl.tensor,
    DV32: tl.tensor,
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
    stride_ab: tl.tensor,
    stride_at: tl.tensor,
    stride_ah: tl.tensor,
    stride_xb: tl.tensor,
    stride_xt: tl.tensor,
    stride_xh: tl.tensor,
    H_Q: tl.tensor,
    H_KV: tl.tensor,
    HG: tl.tensor,
    T: tl.tensor,
    qk_scale: tl.tensor,
    sm_scale: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    SINK_ONLY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """Band keys' dk and dv plus the gathered buffer, or SINK_ONLY the sink_size's added to it.

    A band program takes BLOCK_N keys and the queries whose window_size holds any of them, and writes
    dk and dv; a sink_size program takes the sink_size keys and CHUNK queries, and adds to the buffer.
    """
    bk = tl.program_id(1)
    b = (bk // H_KV).to(tl.int64)
    kvh = bk % H_KV
    dims = tl.arange(0, BLOCK_D)
    dmask = dims[None, :] < HEAD_DIM
    if SINK_ONLY:
        cols = tl.arange(0, BLOCK_N)
        kmask = (cols[:, None] < SINK) & (cols[:, None] < T) & dmask
        q0 = tl.program_id(0) * CHUNK
        q1 = tl.minimum(q0 + CHUNK, T)
    else:
        n0 = tl.program_id(0) * BLOCK_N
        cols = n0 + tl.arange(0, BLOCK_N)
        kmask = (cols[:, None] < T) & dmask
        q0 = n0
        q1 = tl.minimum(n0 + BLOCK_N - 1 + WINDOW, T)
    K += b * stride_kb + kvh * stride_kh
    V += b * stride_vb + kvh * stride_vh
    k = tl.load(K + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
    v = tl.load(V + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk, dv = _dkv_rows(
        dk,
        dv,
        k,
        v,
        cols,
        Q + b * stride_qb + kvh * HG * stride_qh,
        DO + b * stride_db + kvh * HG * stride_dh,
        LSE,
        DELTA,
        stride_qt,
        stride_qh,
        stride_dt,
        stride_dh,
        (b * H_Q + kvh * HG) * T,
        q0,
        q1,
        T,
        HG,
        qk_scale,
        WINDOW,
        SINK_ONLY,
        HEAD_DIM,
        BLOCK_D,
        HP,
        ROWS,
    )
    at = b * stride_ab + kvh * stride_ah + cols[:, None] * stride_at + dims[None, :]
    if SINK_ONLY:
        tl.atomic_add(DK32 + at, dk * sm_scale, mask=kmask, sem="relaxed")
        tl.atomic_add(DV32 + at, dv, mask=kmask, sem="relaxed")
    else:
        dk = dk * sm_scale + tl.load(DK32 + at, mask=kmask, other=0.0)
        dv = dv + tl.load(DV32 + at, mask=kmask, other=0.0)
        out = b * stride_xb + kvh * stride_xh + cols[:, None] * stride_xt + dims[None, :]
        tl.store(DK + out, dk.to(DK.dtype.element_ty), mask=kmask)
        tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=kmask)


@triton.jit
def _code(
    IND: tl.tensor,
    SLOTS: tl.tensor,
    stride_it: tl.tensor,
    stride_st: tl.tensor,
    query: tl.tensor,
    slot: tl.tensor,
    T: tl.tensor,
    TOP_K: tl.tensor,
    WINDOW: tl.tensor,
    SINK: tl.tensor,
) -> tl.tensor:
    """Whether query keeps its slot, and the position's group * KP + member."""
    idx, valid = _listed(IND, stride_it, query, slot, T, TOP_K, WINDOW, SINK)
    return valid, tl.load(SLOTS + idx * stride_st, mask=valid, other=0)


@triton.jit
def _joins(
    valid: tl.tensor, code: tl.tensor, valid_before: tl.tensor, code_before: tl.tensor, KP: tl.constexpr
) -> tl.tensor:
    """Whether a slot extends the run of the slot before: kept, one group, a later member."""
    same = code_before // KP == code // KP
    return valid & valid_before & same & (code_before % KP < code % KP)


@triton.jit
def _max(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    """Return the larger of two values, the combine of an associative scan."""
    return tl.maximum(a, b)


@triton.jit
def _runs_kernel(
    IND: tl.tensor,
    SLOTS: tl.tensor,
    COUNT: tl.tensor,
    CURSOR: tl.tensor,
    QUERY: tl.tensor,
    TAKEN: tl.tensor,
    stride_ib: tl.tensor,
    stride_it: tl.tensor,
    stride_sb: tl.tensor,
    stride_st: tl.tensor,
    T: tl.tensor,
    N_GROUPS: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    TOP_K: tl.constexpr,
    KP: tl.constexpr,
    K_PAD: tl.constexpr,
    LANES: tl.constexpr,
    LANE_SPAN: tl.constexpr,
    FILL: tl.constexpr,
) -> None:
    """A query's runs, counted a group, or FILL listed at the group's cursor with their members.

    A run is consecutive kept slots of one group whose members rise: a query's take of one LSA
    bucket, members in position order; a position listed twice falls in two runs. A group's
    counter is LANES counters, each taking LANE_SPAN consecutive queries in turn, so a bucket
    every query picks does not serialise its atomics. A run is listed as its query and the bits
    of the members it took: their sum from its first slot to its last, distinct bits a run.
    """
    query = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    IND += b * stride_ib
    SLOTS += b * stride_sb
    slot = tl.arange(0, K_PAD)
    valid, code = _code(IND, SLOTS, stride_it, stride_st, query, slot, T, TOP_K, WINDOW, SINK)
    valid_before, code_before = _code(
        IND, SLOTS, stride_it, stride_st, query, slot - 1, T, TOP_K, WINDOW, SINK
    )
    start = valid & ~_joins(valid, code, valid_before, code_before, KP)
    counter = (b * N_GROUPS + code // KP) * LANES + (query // LANE_SPAN) % LANES
    if FILL:
        valid_after, code_after = _code(
            IND, SLOTS, stride_it, stride_st, query, slot + 1, T, TOP_K, WINDOW, SINK
        )
        last = valid & ~_joins(valid_after, code_after, valid, code, KP)
        bit = tl.where(valid, tl.full([K_PAD], 1, tl.int64) << (code % KP), 0)
        through = tl.cumsum(bit, axis=0)
        before = tl.associative_scan(tl.where(start, through - bit, 0), 0, _max)
        at = tl.atomic_add(CURSOR + counter, 1, mask=last, sem="relaxed")
        tl.store(QUERY + at, tl.full([K_PAD], 0, tl.int32) + query, mask=last)
        tl.store(TAKEN + at, (through - before).to(tl.int32), mask=last)
    else:
        tl.atomic_add(COUNT + counter, 1, mask=start, sem="relaxed")


@triton.jit
def _items_kernel(
    QUERY: tl.tensor,
    OFFSET: tl.tensor,
    COUNT: tl.tensor,
    CHUNK_END: tl.tensor,
    ITEMS: tl.tensor,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """A group's work items, chunks of CHUNK runs, keyed by their first run's query to sort."""
    flat = tl.program_id(0)
    count = tl.load(COUNT + flat)
    n = tl.cdiv(count, CHUNK)
    first = tl.load(CHUNK_END + flat) - n
    base = tl.load(OFFSET + flat)
    for c0 in range(0, n, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        query = tl.load(QUERY + base + c * CHUNK, mask=c < n, other=0).to(tl.int64)
        item = (query << ITEM_QUERY) | (flat.to(tl.int64) << ITEM_GROUP) | c
        tl.store(ITEMS + first + c, item, mask=c < n)


@triton.jit
def _group_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    DO: tl.tensor,
    LSE: tl.tensor,
    DELTA: tl.tensor,
    TABLE: tl.tensor,
    QUERY: tl.tensor,
    TAKEN: tl.tensor,
    OFFSET: tl.tensor,
    COUNT: tl.tensor,
    CHUNK_END: tl.tensor,
    ITEMS: tl.tensor,
    DK32: tl.tensor,
    DV32: tl.tensor,
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
    stride_ab: tl.tensor,
    stride_at: tl.tensor,
    stride_ah: tl.tensor,
    H_Q: tl.tensor,
    H_KV: tl.tensor,
    HG: tl.tensor,
    T: tl.tensor,
    N_GROUPS: tl.tensor,
    N_ALL: tl.tensor,
    qk_scale: tl.tensor,
    sm_scale: tl.tensor,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HP: tl.constexpr,
    KP: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """Dk and dv of the gathered keys, a group's KP rows over CHUNK of the runs that read it.

    Persistent: a program takes work items pid, pid + programs, ..., each a chunk of one group's
    runs, in the order of their first queries, so the programs at work at once read nearby
    queries' rows, which L2 then holds for the next. The group's keys and values load once,
    ROWS query rows (runs times heads) a tile score them, each run's bits masking the members
    it took, and the sums add to the fp32 buffer, a few atomics a chunk.
    """
    QT: tl.constexpr = ROWS // HP
    kvh = tl.program_id(1)
    total = tl.load(CHUNK_END + N_ALL - 1)
    dims = tl.arange(0, BLOCK_D)
    dmask = dims[None, :] < HEAD_DIM
    members = tl.arange(0, KP)
    r = tl.arange(0, ROWS)
    head = r % HP
    pair = tl.arange(0, QT)
    for w in range(tl.program_id(0), total, tl.num_programs(0)):
        item = tl.load(ITEMS + w)
        flat = ((item >> ITEM_GROUP) & ((1 << (ITEM_QUERY - ITEM_GROUP)) - 1)).to(tl.int32)
        c = (item & ((1 << ITEM_GROUP) - 1)).to(tl.int32)
        count = tl.load(COUNT + flat)
        p0 = c * CHUNK
        p1 = tl.minimum(p0 + CHUNK, count)
        base = tl.load(OFFSET + flat)
        b = (flat // N_GROUPS).to(tl.int64)
        cols = tl.load(TABLE + flat * KP + members)
        held = cols >= 0
        kmask = held[:, None] & dmask
        Kb = K + b * stride_kb + kvh * stride_kh
        Vb = V + b * stride_vb + kvh * stride_vh
        k = tl.load(Kb + cols[:, None] * stride_kt + dims[None, :], mask=kmask, other=0.0)
        v = tl.load(Vb + cols[:, None] * stride_vt + dims[None, :], mask=kmask, other=0.0)
        Qb = Q + b * stride_qb + kvh * HG * stride_qh
        DOb = DO + b * stride_db + kvh * HG * stride_dh
        lse_base = (b * H_Q + kvh * HG) * T
        dk = tl.zeros([KP, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([KP, BLOCK_D], dtype=tl.float32)
        for s0 in range(p0, p1, QT):
            in_runs = s0 + pair < p1
            query = tl.load(QUERY + base + s0 + pair, mask=in_runs, other=0)
            bits = tl.load(TAKEN + base + s0 + pair, mask=in_runs, other=0)
            bits = tl.reshape(tl.broadcast_to(bits[:, None], [QT, HP]), [ROWS])
            query = tl.reshape(tl.broadcast_to(query[:, None], [QT, HP]), [ROWS])
            live = tl.reshape(tl.broadcast_to(in_runs[:, None], [QT, HP]), [ROWS]) & (head < HG)
            rmask = live[:, None] & dmask
            q_at = query[:, None] * stride_qt + head[:, None] * stride_qh + dims[None, :]
            q = tl.load(Qb + q_at, mask=rmask, other=0.0)
            do_at = query[:, None] * stride_dt + head[:, None] * stride_dh + dims[None, :]
            do = tl.load(DOb + do_at, mask=rmask, other=0.0)
            row = lse_base + head * T + query
            lse = tl.load(LSE + row, mask=live, other=float("inf"))
            lse2 = tl.where(lse == float("-inf"), float("inf"), lse * 1.4426950408889634)
            delta = tl.load(DELTA + row, mask=live, other=0.0)
            keep = (((bits[None, :] >> members[:, None]) & 1) != 0) & live[None, :]
            s = tl.dot(k, tl.trans(q)) * qk_scale
            p = tl.where(keep, tl.math.exp2(s - lse2[None, :]), 0.0)
            dv += tl.dot(p.to(do.dtype), do)
            dp = tl.dot(v, tl.trans(do))
            ds = p * (dp - delta[None, :])
            dk += tl.dot(ds.to(q.dtype), q)
        at = b * stride_ab + kvh * stride_ah + cols[:, None] * stride_at + dims[None, :]
        tl.atomic_add(DK32 + at, dk * sm_scale, mask=kmask, sem="relaxed")
        tl.atomic_add(DV32 + at, dv, mask=kmask, sem="relaxed")


def _groups(
    members: torch.Tensor | None, b: int, t: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The groups the key-side backward walks: (b, n, KP) positions and (b, t) group * KP + slot.

    LSA's buckets first, when given, then every KP consecutive positions, which hold the keys
    no bucket does.
    """
    cap = 1 if members is None else members.shape[2]
    kp = max(GROUP_KEYS, triton.next_power_of_2(cap))
    if kp > 32:
        raise ValueError(f"groups of more than 32 keys, got {cap}")
    spans = triton.cdiv(t, kp)
    pos = torch.arange(spans * kp, device=device, dtype=torch.int32)
    consecutive = torch.where(pos < t, pos, -1).view(1, spans, kp).expand(b, -1, -1)
    if members is None:
        slots = pos[:t].expand(b, -1).contiguous()
        return consecutive.contiguous(), slots, kp
    m = members.shape[1]
    buckets = F.pad(members.to(torch.int32), (0, kp - members.shape[2]), value=-1)
    table = torch.cat([buckets, consecutive], 1).contiguous()
    slots = torch.empty(b, t + 1, dtype=torch.int32, device=device)
    slots[:, :t] = pos[:t] + m * kp
    code = torch.arange(m * kp, device=device, dtype=torch.int32).expand(b, -1)
    slots.scatter_(1, torch.where(buckets >= 0, buckets, t).flatten(1).long(), code)
    return table, slots[:, :t], kp


def _key_side(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    dk32: torch.Tensor,
    dv32: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
    hp: int,
) -> None:
    """The gathered keys' dk and dv, added to dk32 and dv32, by _runs_kernel and _group_kernel."""
    b, t, h_q, d = q.shape
    h_kv = k.shape[2]
    top_k = indices.shape[2]
    table, slots, kp = _groups(members, b, t, q.device)
    n_groups = table.shape[1]
    counters = torch.zeros(b * n_groups * RUN_LANES, dtype=torch.int32, device=q.device)
    # Runs a query can hold: one a slot it reads
    reads = int(torch.arange(t).sub(window_size + sink_size - 1).clamp(0, top_k).sum().item())
    query = torch.empty(max(1, b * reads), dtype=torch.int32, device=q.device)
    taken = torch.empty_like(query)
    common = dict(
        WINDOW=window_size, SINK=sink_size, TOP_K=top_k, KP=kp, K_PAD=triton.next_power_of_2(top_k),
        LANES=RUN_LANES, LANE_SPAN=RUN_LANE_SPAN, num_warps=4,
    )  # fmt: skip
    strides = (indices.stride(0), indices.stride(1), slots.stride(0), slots.stride(1))
    _runs_kernel[(t, b)](
        indices, slots, counters, counters, query, taken, *strides, t, n_groups, FILL=False,
        **common,
    )  # fmt: skip
    cursor = counters.cumsum(0, dtype=torch.int32) - counters
    offset = cursor[::RUN_LANES].clone()
    count = counters.view(-1, RUN_LANES).sum(1, dtype=torch.int32)
    _runs_kernel[(t, b)](
        indices, slots, counters, cursor, query, taken, *strides, t, n_groups, FILL=True, **common
    )  # fmt: skip
    chunk = GROUP_RUNS
    chunk_end = (count + chunk - 1).div(chunk, rounding_mode="floor").cumsum(0, dtype=torch.int32)
    # Work items: at most one a group plus one a CHUNK of the runs, the unused sorting last
    items = torch.full((b * n_groups + query.numel() // chunk + 1,), torch.iinfo(torch.int64).max,
                       dtype=torch.int64, device=q.device)  # fmt: skip
    _items_kernel[(b * n_groups,)](query, offset, count, chunk_end, items, CHUNK=chunk, BLOCK=64, num_warps=1)
    items = items.sort().values
    programs = _sm_count(q.device) * GROUP_WAVES
    warps, stages = GROUP_LAUNCH
    rows = GROUP_ROWS
    if _blackwell(q.device):
        warps, stages = BLACKWELL_GROUP_LAUNCH
        rows = BLACKWELL_GROUP_ROWS
    _group_kernel[(programs, h_kv)](
        q, k, v, do, lse, delta, table, query, taken, offset, count, chunk_end, items, dk32, dv32,
        *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dk32),
        h_q, h_kv, h_q // h_kv, t, n_groups, b * n_groups, scale * LOG2E, scale,
        HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d), HP=hp, KP=kp,
        ROWS=max(rows.get(hp, 128), hp), CHUNK=chunk, num_warps=warps, num_stages=stages,
    )  # fmt: skip


def _dkv_launch(device: torch.device, block_d: int) -> dict[str, int]:
    """Return the warps and stages of the dk and dv kernels on a device."""
    warps, stages = DKV_LAUNCH
    if _blackwell(device):
        warps = BLACKWELL_DKV_WARPS
        stages = DKV_BLACKWELL_STAGES if block_d >= 256 else stages
    return {"num_warps": warps, "num_stages": stages}


def _blackwell(device: torch.device) -> bool:
    """Return whether the device is Blackwell or later."""
    return torch.cuda.get_device_capability(device)[0] >= 10


def _sm_count(device: torch.device) -> int:
    """Return the streaming multiprocessors of the device."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _strides(x: torch.Tensor) -> tuple[int, int, int]:
    """The batch, time and head strides of a (b, t, h, d) tensor with unit stride along d."""
    return x.stride(0), x.stride(1), x.stride(2)


def _unit_stride(x: torch.Tensor) -> torch.Tensor:
    """Return x, made contiguous unless its last dim has unit stride."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _shape(
    q: torch.Tensor, k: torch.Tensor, indices: torch.Tensor, gather: int | None = None
) -> dict[str, int]:
    """The kernels' shared constexprs and the group of query heads a key/value head."""
    d, h_q, h_kv = q.shape[3], q.shape[2], k.shape[2]
    group = h_q // h_kv
    hp = triton.next_power_of_2(group)
    g = max(1, MIN_ROWS // hp)
    fwd_gather = BLACKWELL_FWD_GATHER if _blackwell(q.device) else FWD_GATHER
    gather = fwd_gather.get(hp, 64) if gather is None else gather
    return {
        "group": group,
        "G": g,
        "HP": hp,
        "KB": max(1, gather // g),
        "TOP_K": indices.shape[2],
        "HEAD_DIM": d,
        "BLOCK_D": triton.next_power_of_2(d),
    }


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward over the gathered keys.

    Args:
        q: Queries [B, T, H_q, D].
        k: Keys [B, T, H_kv, D].
        v: Values [B, T, H_kv, D].
        indices: Selected key positions [B, T, top_k] int32, PAD (-1) anywhere.
        window_size: Recent keys each query reads.
        sink_size: First keys each query reads.
        scale: The factor logits are scaled by.

    Returns:
        The output like q.
        The log-sum-exp [B, H_q, T] fp32.
    """
    b, t, h_q, _ = q.shape
    h_kv = k.shape[2]
    q, k, v, indices = _unit_stride(q), _unit_stride(k), _unit_stride(v), _unit_stride(indices)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty(b, h_q, t, dtype=torch.float32, device=q.device)
    shape = _shape(q, k, indices)
    group = shape.pop("group")
    warps, stages = FWD_LAUNCH
    _fwd_kernel[(triton.cdiv(t, shape['G']), b * h_kv)](
        q, k, v, indices, o, lse,
        *_strides(q), *_strides(k), *_strides(v), *_strides(o),
        indices.stride(0), indices.stride(1),
        h_q, h_kv, group, t, scale * LOG2E,
        WINDOW=window_size, SINK=sink_size, BLOCK_N=BLOCK_N, SINK_N=max(16, triton.next_power_of_2(sink_size)),
        num_warps=warps, num_stages=stages, **shape,
    )  # fmt: skip
    return o, lse


def _backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
    dlse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward over the gathered keys.

    Args:
        do: Output gradient like q.
        q: Queries [B, T, H_q, D].
        k: Keys [B, T, H_kv, D].
        v: Values [B, T, H_kv, D].
        o: The forward's output like q.
        lse: The forward's log-sum-exp [B, H_q, T] fp32.
        indices: Selected key positions [B, T, top_k] int32, PAD (-1) anywhere.
        members: LSA bucket members, which group a query's listed keys into runs - 16 positions when None.
        window_size: Recent keys each query reads.
        sink_size: First keys each query reads.
        scale: The factor logits are scaled by.
        dlse: Gradient of the log-sum-exp [B, H_q, T], when it is used downstream.

    Returns:
        The gradients of q, k and v.
    """
    b, t, h_q, d = q.shape
    h_kv = k.shape[2]
    q, k, v, do = _unit_stride(q), _unit_stride(k), _unit_stride(v), _unit_stride(do)
    indices = _unit_stride(indices)
    delta = torch.empty(b, h_q, t, dtype=torch.float32, device=q.device)
    block_t = max(16, 4096 // triton.next_power_of_2(d))
    _delta_kernel[(triton.cdiv(t, block_t), b * h_q)](
        o, do, delta, *_strides(o), *_strides(do), h_q, t,
        HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d), BLOCK_T=block_t,
    )  # fmt: skip
    if dlse is not None:
        # A log-sum-exp gradient adds g * p to a score's, which is p * (dp - delta) at delta - g
        delta -= dlse.float()
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    dk32 = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv32 = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    shape = _shape(q, k, indices, DQ_GATHER)
    group = shape.pop("group")
    sink_n = max(16, triton.next_power_of_2(sink_size))
    warps, stages = BLACKWELL_DQ_LAUNCH if _blackwell(q.device) else DQ_LAUNCH
    scatter = BACKWARD == "scatter"
    _dq_kernel[(triton.cdiv(t, shape['G']), b * h_kv)](
        q, k, v, do, lse, delta, indices, dq, dk32, dv32,
        *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dq),
        indices.stride(0), indices.stride(1), *_strides(dk32),
        h_q, h_kv, group, t, scale * LOG2E, scale,
        WINDOW=window_size, SINK=sink_size, BLOCK_N=BLOCK_N, SINK_N=sink_n, SCATTER=scatter,
        num_warps=warps, num_stages=stages, **shape,
    )  # fmt: skip
    if not scatter:
        _key_side(
            q, k, v, do, lse, delta, indices, members, dk32, dv32, window_size, sink_size, scale,
            shape['HP'],
        )  # fmt: skip
    common = dict(
        HEAD_DIM=d, BLOCK_D=shape['BLOCK_D'], HP=shape['HP'], WINDOW=window_size, SINK=sink_size,
        ROWS=max(DKV_ROWS, shape['HP']), **_dkv_launch(q.device, shape['BLOCK_D']),
    )  # fmt: skip
    args = (
        q, k, v, do, lse, delta, dk32, dv32, dk, dv,
        *_strides(q), *_strides(k), *_strides(v), *_strides(do), *_strides(dk32), *_strides(dk),
        h_q, h_kv, group, t, scale * LOG2E, scale,
    )  # fmt: skip
    if sink_size:
        chunk = max(1, SINK_ROWS // shape["HP"])
        _dkv_kernel[(triton.cdiv(t, chunk), b * h_kv)](
            *args, SINK_ONLY=True, BLOCK_N=sink_n, CHUNK=chunk, **common
        )
    _dkv_kernel[(triton.cdiv(t, BLOCK_N), b * h_kv)](
        *args, SINK_ONLY=False, BLOCK_N=BLOCK_N, CHUNK=1, **common
    )
    return dq, dk, dv


@torch.library.custom_op("lsa::gather_sparse_attention", mutates_args=())
def gather_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run attention over gathered selected keys; see gather_sparse.attention."""
    return _forward(q, k, v, indices, window_size, sink_size, scale)


@gather_attention_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an empty output like q and an empty log-sum-exp [B, H_q, T] fp32."""
    b, t, h_q, _ = q.shape
    return q.new_empty(q.shape), q.new_empty((b, h_q, t), dtype=torch.float32)


@torch.library.custom_op("lsa::gather_sparse_attention_backward", mutates_args=())
def gather_attention_backward_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk and dv of gather_attention_op."""
    return _backward(do, q, k, v, o, lse, indices, members, window_size, sink_size, scale)


@gather_attention_backward_op.register_fake
def _(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty gradients like q, k and v."""
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    """Save the tensors and settings the backward needs."""
    q, k, v, indices, members, window_size, sink_size, scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, indices, members)
    ctx.window_size, ctx.sink_size, ctx.scale = window_size, sink_size, scale


def _autograd_backward(ctx: Any, do: torch.Tensor, _dlse: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Return the gradients of q, k and v, and none for the other inputs."""
    q, k, v, o, lse, indices, members = ctx.saved_tensors
    dq, dk, dv = gather_attention_backward_op(
        do, q, k, v, o, lse, indices, members, ctx.window_size, ctx.sink_size, ctx.scale
    )
    return dq, dk, dv, None, None, None, None, None


gather_attention_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
    members: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal softmax attention over each query's window_size, sink_size and selected keys.

    Query i, at position i, keeps keys j with 0 <= i - j < window_size, j < sink_size, and the positions indices lists for it that lie in [sink_size, i - window_size] in its first i - window_size - sink_size + 1 slots, the most distinct positions it can keep.
    That is as dsa_sparse.attention keeps a Selection of the same keys.
    Under autocast q, k and v are cast to its dtype.

    Args:
        q: Queries [b, t, h_q, d].
        k: Keys [b, t, h_kv, d], h_kv dividing h_q - query head h reads key head h // (h_q // h_kv).
        v: Values [b, t, h_kv, d].
        indices: Key positions [b, t, top_k] int32 or int64, one selection every head reads, PAD (-1) or any position out of range taking none.
            They are lsa_select.select_indices'.
        window_size: Keys a query's window_size holds, itself included.
        sink_size: Leading keys every query keeps.
        scale: The scores' scale.
        members: LSA's [b, m, cap] buckets of key positions, PAD past a bucket's size, cap at most 32, which the backward reads the selected keys' gradients by.
            A query's consecutive listed keys of one bucket, in position order, are one run of it.
            None groups consecutive positions instead, right for any selection but slower for LSA's.

    Returns:
        The [b, t, h_q, d] context and the [b, h_q, t] fp32 log-sum-exp of the scaled scores, which carries no gradient.
    """
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    b, t, h_q, d = q.shape
    if d not in HEAD_DIMS:
        raise ValueError(f"head_dim {d} is not one of {HEAD_DIMS}")
    if q.dtype not in DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"attention needs q, k and v of one dtype of {DTYPES}")
    if h_q % k.shape[2] or k.shape != v.shape or k.shape[:2] != (b, t) or k.shape[3] != d:
        raise ValueError(f"q {tuple(q.shape)} does not group over k and v {tuple(k.shape)}")
    if indices.shape[:2] != (b, t) or indices.dim() != 3:
        raise ValueError(f"indices {tuple(indices.shape)} are not (b, t, top_k) for q")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"indices must be int32 or int64, got {indices.dtype}")
    if members is not None and (members.dim() != 3 or len(members) != b or members.shape[2] > 32):
        raise ValueError(f"members {tuple(members.shape)} are not (b, m, cap <= 32) for q")
    with torch.autocast("cuda", enabled=False):
        o, lse = gather_attention_op(q, k, v, indices, members, window_size, sink_size, scale)
    return o, lse.detach()


def log_mass_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    members: torch.Tensor | None,
    lse: torch.Tensor,
    dlse: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate a gradient of the read's log-sum-exp into its queries and keys.

    The backward kernels take a zero output gradient and a delta shifted by `dlse`, so the scores' gradient is `dlse` times each probability.

    Args:
        q: Queries [b, t, h_q, d] bf16 or fp16.
        k: Keys [b, t, h_kv, d] of q's dtype.
        indices: The read's selection, as `attention` takes it.
        members: The read's buckets, as `attention` takes them.
        lse: The read's log-sum-exp [b, h_q, t] fp32.
        dlse: Gradient of the log-sum-exp [b, h_q, t].
        window_size: Keys a query's window holds, itself included.
        sink_size: Leading keys every query keeps.
        scale: The scores' scale.

    Returns:
        Gradient of the queries [b, t, h_q, d] in q's dtype.
        Gradient of the keys [b, t, h_kv, d] in k's dtype.
    """
    zero = torch.zeros_like(q)
    dq, dk, _ = _backward(zero, q, k, k, zero, lse, indices, members, window_size, sink_size, scale, dlse)
    return dq, dk
