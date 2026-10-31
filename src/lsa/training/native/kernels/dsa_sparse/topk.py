"""The DSA indexer's exact top-k selection as a bitmask, without materializing any T x T tensor.

Each query keeps its causal window_size and the sink_size by position, and the top_k long-range keys
(outside both) its indexer ranks highest. A Selection stores only those long-range keys:

- bits: (b, h_sel, t, w) int32, w = 2 * ceil(t / 64): bit j % 32 of word j // 32 of row i is
  set when query i keeps long-range key j;
- occupancy: (b, h_sel, n, n) int32, n = ceil(t / 64) rounded up to a multiple of OCC_PAD,
  nonzero where the OCC_BLOCK x OCC_BLOCK tile of queries and keys holds a bit.

The selection is the one dsa_select.sparse_selection makes from rank scores
dtype(dtype(index_queries . index_keys) * scale) in the indexer's half-precision dtype: distinct priorities
from the score's bits and the key index, so exactly top_k candidates and an exact tie goes to
the later key; a row with at most top_k candidates keeps them all. select finds it from the
raw fp32 dot products of tensor-core tiles, recomputed each pass and never stored, in a
search per row for the bracket of rank-score classes that holds its k-th score:

1. A stats pass: each row's mean and variance, so a Gaussian guess of its k-th score.
2. Search passes: one counts each row's candidates at four nested thresholds and keeps the
   keys between the middle two, the bracket where it expects the k-th, in a buffer of CAP
   slots. A threshold is the exact fp32 boundary of a class under the double rounding, so a
   float compare of the raw dot product decides the class. When the k-th is in the middle
   bracket and every key of it fit, the row is collected; else the counts give the next
   bracket. The first pass's middle bracket covers the k-th order statistic's rank noise,
   so most rows are collected there; later passes interpolate, split the bracket's scores
   when that stalls, step out of an open end, and in the end split its classes, which bounds
   them. Each pass regroups the rows still open, so it costs as much as those rows, not
   their blocks.
3. A resolve pass: of each row's collected keys, the least priority it keeps, by bisection.
4. An emit pass: the bits of every key above each row's bracket, and of its bracket's keys
   those at or past that priority. A bracket of one class with more than CAP keys, all
   tied, has no cut: the emit pass keeps the latest of them, dropping a tile's lowest set
   bits. A tile's bits are packed by a small tensor-core product.

The one difference from the layer's integer priorities: a raw dot product of exactly -0.0,
which needs every term to be a signed zero, ranks as +0.0 rather than just below it.
"""

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

OCC_BLOCK = 64
# Attention tiles up to OCC_PAD * OCC_BLOCK wide read occupancy past the last tile
OCC_PAD = 4
# Above any row's passes: one splits classes after any that shrank its bracket under twofold
MAX_PASSES = 24
# Passes that may interpolate before every one splits classes, which bounds the rest
VALUE_PASSES = 8
# Bracket keys a row collects, and its first middle bracket's half-width, see _stats_kernel
CAP = 128
HALF_SIGMAS = 1.5
HALF_MIN = 4.0
# Triton's CPU interpreter has no popc
INTERPRETED = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")
# A row's search flags: split its classes next, collected, a single class of TIED keys
SPLIT, COLLECTED, TIED = tl.constexpr(1), tl.constexpr(2), tl.constexpr(4)


class Selection(NamedTuple):
    """The long-range keys each query keeps, as a bitmask and the tiles that hold a bit.

    Attributes:
        bits: Bitmask [b, h_sel, t, w] int32, bit j % 32 of word j // 32 of row i set if query i keeps key j.
        occupancy: Nonzero where a tile of queries and keys holds a bit [b, h_sel, n, n] int32.
    """

    bits: torch.Tensor
    occupancy: torch.Tensor


def words_per_row(t: int) -> int:
    """Return the bitmask words per query: two per 64 keys, so a tile's bits are one aligned int64."""
    return 2 * triton.cdiv(t, 64)


def occupancy_size(t: int) -> int:
    """Return the occupancy tiles per side for t queries and keys, padded to a multiple of OCC_PAD."""
    return triton.cdiv(triton.cdiv(t, OCC_BLOCK), OCC_PAD) * OCC_PAD


def pack(long_range: torch.Tensor) -> Selection:
    """Pack a bool mask of the long-range keys each query keeps into a Selection.

    Args:
        long_range: Mask [b, h, t, t] bool.
    """
    b, h, t, n_k = long_range.shape
    words = words_per_row(n_k)
    padded = torch.nn.functional.pad(long_range, (0, words * 32 - n_k))
    weights = torch.ones((), dtype=torch.int64, device=long_range.device) << torch.arange(
        32, device=long_range.device
    )
    bits = (padded.view(b, h, t, words, 32).long() * weights).sum(-1)
    # Bit 31 wraps to the int32 sign bit
    bits = torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)
    n = occupancy_size(max(t, n_k))
    tiles = torch.nn.functional.pad(long_range, (0, n * OCC_BLOCK - n_k, 0, n * OCC_BLOCK - t))
    occupancy = tiles.view(b, h, n, OCC_BLOCK, n, OCC_BLOCK).any(5).any(3).to(torch.int32)
    return Selection(bits.contiguous(), occupancy.contiguous())


def unpack(bits: torch.Tensor, n_k: int) -> torch.Tensor:
    """Unpack a Selection's bits into a bool mask.

    Returns:
        Mask [b, h, t, n_k] bool.
    """
    shifts = torch.arange(32, device=bits.device, dtype=torch.int32)
    mask = (bits.unsqueeze(-1) >> shifts) & 1
    return mask.flatten(-2)[..., :n_k].bool()


@triton.jit
def _order(r: tl.tensor) -> tl.tensor:
    """The int32 of the same order as bf16 r: negative floats order backwards as integers."""
    o = r.to(tl.int16, bitcast=True).to(tl.int32)
    return tl.where(o < 0, o ^ 0x7FFF, o)


@triton.jit
def _class(x: tl.tensor, scale: tl.tensor, RANK: tl.constexpr) -> tl.tensor:
    """The rank score's order: RANK(RANK(x) * scale), as a half-precision einsum rounds it."""
    return _order((x.to(RANK).to(tl.float32) * scale).to(RANK))


def order_range(dtype: torch.dtype) -> tuple[int, int]:
    """Return the orders of -inf and +inf in a half-precision dtype, the ends of its classes."""
    infinities = torch.tensor([float("-inf"), float("inf")], dtype=dtype).view(torch.int16).int()
    low, high = torch.where(infinities < 0, infinities ^ 0x7FFF, infinities).tolist()
    return low, high


_boundary_tables: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}


def boundary_table(scale: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return each class's least raw score: entry c - low is the least fp32 x of _class >= c.

    Classes run over order_range(dtype) and one past it, every score to none, so the ends are -inf and +inf.
    A class is dtype(dtype(x) * scale)'s order, so its least x is the round-half-even midpoint below the least value of dtype that the scale lifts to it.
    """
    key = (float(scale), dtype, device)
    table = _boundary_tables.get(key)
    if table is not None:
        return table
    low, high = order_range(dtype)
    order = torch.arange(low, high + 1, dtype=torch.int32, device=device)

    def values(o: torch.Tensor) -> torch.Tensor:
        return torch.where(o < 0, o ^ 0x7FFF, o).to(torch.int16).view(dtype).float()

    lifted = (values(order) * torch.tensor(scale, dtype=torch.float32)).to(dtype)
    lifted = lifted.view(torch.int16).int()
    lifted = torch.where(lifted < 0, lifted ^ 0x7FFF, lifted)
    classes = torch.arange(low, high + 2, dtype=torch.int32, device=device)
    first = low + torch.searchsorted(lifted, classes).clamp(max=order.numel() - 1)
    y, below = values(first), values(torch.clamp(first - 1, min=low))
    # Halves are exact; a midpoint next to an infinity is half a step past the finite end
    above = values(torch.clamp(first + 1, max=high))
    mid = torch.where(torch.isinf(below), y - (above - y) / 2, below / 2 + y / 2)
    mid = torch.where(torch.isinf(y), below + (below - values(torch.clamp(first - 2, min=low))) / 2, mid)
    even = (torch.where(first < 0, first ^ 0x7FFF, first) & 1) == 0
    least = torch.where(even, mid, torch.nextafter(mid, torch.tensor(float("inf"), device=device)))
    least = torch.where(first == 0, 0.0, least)
    least = torch.where(classes <= low, float("-inf"), least)
    table = torch.where(classes > lifted[-1], float("inf"), least).contiguous()
    _boundary_tables[key] = table
    return table


@triton.jit
def _scores(
    q: tl.tensor,
    KI: tl.tensor,
    stride_kt: tl.tensor,
    n: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    D_R: tl.constexpr,
) -> tl.tensor:
    """The raw fp32 indexer scores of the block's queries over the 64 keys of tile n."""
    cols = n * 64 + tl.arange(0, 64)
    kmask = (cols[:, None] < T) & (dr[None, :] < D_R)
    k = tl.load(KI + cols[:, None] * stride_kt + dr[None, :], mask=kmask, other=0.0)
    return tl.dot(q, tl.trans(k)), cols


@triton.jit
def _moments(
    total: tl.tensor,
    square: tl.tensor,
    q: tl.tensor,
    KI: tl.tensor,
    stride_kt: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    last: tl.tensor,
    SINK: tl.constexpr,
    MASKED: tl.constexpr,
    D_R: tl.constexpr,
) -> tl.tensor:
    """Add the scores of tiles lo to hi to each row's running sum and sum of squares.

    Args:
        total: Running sum of the scores [64] fp32.
        square: Running sum of the squared scores [64] fp32.
        last: Last long-range key of each row [64] - with MASKED, only keys from the sink to it count.

    Returns:
        The updated sum and sum of squares.
    """
    for n in range(lo, hi):
        x, cols = _scores(q, KI, stride_kt, n, T, dr, D_R)
        if MASKED:
            x = tl.where((cols[None, :] >= SINK) & (cols[None, :] <= last[:, None]), x, 0.0)
        total += tl.sum(x, axis=1)
        square += tl.sum(x * x, axis=1)
    return total, square


@triton.jit
def _pack(keep: tl.tensor) -> tl.tensor:
    """Each row's 64 keep bits as two int32 words, in the first two columns of a (64, 4).

    A tensor-core product with powers of two sums each byte exactly in fp32, which costs
    less than reshaping the whole tile to sum its bits.
    """
    col = tl.arange(0, 64)
    byte = tl.arange(0, 16)
    weights = tl.where(col[:, None] // 8 == byte[None, :], 1 << (col[:, None] % 8), 0)
    ones = tl.where(keep, 1.0, 0.0).to(tl.float16)
    bytes_ = tl.dot(ones, weights.to(tl.float16)).to(tl.int32)
    return tl.sum(tl.reshape(bytes_ << (8 * (byte % 4))[None, :], [64, 4, 4]), axis=2)


@triton.jit
def _emit(
    ties_left: tl.tensor,
    q: tl.tensor,
    KI: tl.tensor,
    stride_kt: tl.tensor,
    BITS: tl.tensor,
    stride_bt: tl.tensor,
    OCC: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    rows: tl.tensor,
    last: tl.tensor,
    upper: tl.tensor,
    lower: tl.tensor,
    cut_low: tl.tensor,
    cut_key: tl.tensor,
    SINK: tl.constexpr,
    MASKED: tl.constexpr,
    D_R: tl.constexpr,
) -> tl.tensor:
    """Store the bits of tiles hi - 1 down to lo, keeping each row's latest ties_left ties.

    Besides every key at or above upper, a collected row keeps those at or above cut_low from
    key cut_key on: the class of its cut, from the cut's index.
    """
    bit = (1 << (tl.arange(0, 64) % 32))[None, :]
    for i in range(hi - lo):
        n = hi - 1 - i
        x, cols = _scores(q, KI, stride_kt, n, T, dr, D_R)
        above = (x >= upper[:, None]) | ((x >= cut_low[:, None]) & (cols[None, :] >= cut_key[:, None]))
        if MASKED:
            cand = (cols[None, :] >= SINK) & (cols[None, :] <= last[:, None])
            above = above & cand
            x = tl.where(cand, x, float("-inf"))
        words = _pack(above)
        tie = (x >= lower[:, None]) & (x < upper[:, None])
        n_ties = tl.sum(tie.to(tl.int32), axis=1)
        if tl.max(tie.to(tl.int32)) > 0:
            low, high = tl.split(tl.sum(tl.reshape(tl.where(tie, bit, 0), [64, 2, 32]), axis=2))
            ties = (high.to(tl.int64) << 32) | (low.to(tl.int64) & 0xFFFFFFFF)
            # Keep the latest ties_left of the tile's ties: drop its lowest set bits
            drop = tl.where((ties_left > 0) & (ties_left < n_ties), n_ties - ties_left, 0)
            for _ in range(tl.max(drop[:, None])):
                ties = tl.where(drop > 0, ties & (ties - 1), ties)
                drop -= 1
            ties = tl.where(ties_left > 0, ties, 0)
            pad = tl.zeros_like(ties).to(tl.int32)
            tie_words = tl.join(tl.join(ties.to(tl.int32), pad), tl.join((ties >> 32).to(tl.int32), pad))
            words = words | tl.reshape(tie_words, [64, 4])
            ties_left -= n_ties
        valid = rows < T
        pair = tl.arange(0, 4)
        tl.store(
            BITS + rows[:, None] * stride_bt + 2 * n + pair[None, :],
            words,
            mask=valid[:, None] & (pair[None, :] < 2),
        )
        occupied = tl.max(tl.where(valid[:, None] & (words != 0), 1, 0))
        tl.store(OCC + n, occupied.to(tl.int32))
    return ties_left


@triton.jit
def _inverse_normal(p: tl.tensor) -> tl.tensor:
    """The standard normal quantile of p, by Acklam's rational approximation."""
    q = p - 0.5
    r = q * q
    central = (
        (
            (((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r + 138.3577518672690)
            * r
            - 30.66479806614716
        )
        * r
        + 2.506628277459239
    ) * q
    central = central / (
        (
            (((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r + 66.80131188771972)
            * r
            - 13.28068155288572
        )
        * r
        + 1.0
    )
    t = tl.sqrt(-2.0 * tl.math.log(tl.minimum(p, 1.0 - p)))
    tail = (
        (((-0.007784894002430293 * t - 0.3223964580411365) * t - 2.400758277161838) * t - 2.549732539343734)
        * t
        + 4.374664141464968
    ) * t + 2.938163982698783
    tail = tail / (
        (((0.007784695709041462 * t + 0.3224671290700398) * t + 2.445134137142996) * t + 3.754408661907416)
        * t
        + 1.0
    )
    tail = tl.where(p < 0.5, tail, -tail)
    return tl.where(tl.abs(q) <= 0.47575, central, tail)


@triton.jit
def _block(
    QI: tl.tensor,
    KI: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kh: tl.tensor,
    T: tl.tensor,
    TOPK: tl.tensor,
    H_SEL: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
) -> tl.tensor:
    """This program's 64 consecutive queries: their indexer queries, keys and candidate tiles.

    Row i ranks keys [SINK, i - WINDOW], in tiles [tile_lo, tile_hi); every row ranks all of
    tiles [full_lo, full_hi). The longest rows go first, so the causal tail fills the grid's
    last wave.
    """
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H_SEL
    h = bh % H_SEL
    m0 = pid_m * 64
    rows = m0 + tl.arange(0, 64)
    dr = tl.arange(0, BLOCK_DR)
    QI += b * stride_qb + h * stride_qh
    qmask = (rows[:, None] < T) & (dr[None, :] < D_R)
    q = tl.load(QI + rows[:, None] * stride_qt + dr[None, :], mask=qmask, other=0.0)
    KI += b * stride_kb + (h // KV_GROUP) * stride_kh
    last = tl.where(rows < T, rows - WINDOW, -1)
    need = tl.maximum(last - SINK + 1, 0) > TOPK
    tile_lo = SINK // 64
    tile_hi = tl.maximum((tl.minimum(m0 + 63, T - 1) - WINDOW) // 64 + 1, tile_lo)
    tile_hi = tl.where(m0 + 63 - WINDOW >= SINK, tile_hi, tile_lo)
    full_lo = tl.minimum((SINK + 63) // 64, tile_hi)
    full_hi = tl.minimum(tl.maximum(tl.maximum(m0 - WINDOW + 1, 0) // 64, full_lo), tile_hi)
    return pid_m, bh, rows, q, KI, last, need, tile_lo, full_lo, full_hi, tile_hi


@triton.jit
def _load_state(STATE: tl.tensor, n_rows: tl.tensor, index: tl.tensor, valid: tl.tensor) -> tl.tensor:
    """Load the search state of the rows at `index`.

    STATE is nine words a row, laid out word by word [9, n_rows]: the bracket's lower and upper classes, the
    candidates at or above each, its score bounds, the guess and the step (these four fp32), and the flags.

    Returns:
        The state, as `_store_state` takes it.
    """
    c_lo = tl.load(STATE + 0 * n_rows + index, mask=valid, other=0)
    c_hi = tl.load(STATE + 1 * n_rows + index, mask=valid, other=0)
    n_lo = tl.load(STATE + 2 * n_rows + index, mask=valid, other=0)
    n_hi = tl.load(STATE + 3 * n_rows + index, mask=valid, other=0)
    floats = STATE.to(tl.pointer_type(tl.float32))
    b_lo = tl.load(floats + 4 * n_rows + index, mask=valid, other=0.0)
    b_hi = tl.load(floats + 5 * n_rows + index, mask=valid, other=0.0)
    guess = tl.load(floats + 6 * n_rows + index, mask=valid, other=0.0)
    step = tl.load(floats + 7 * n_rows + index, mask=valid, other=0.0)
    flags = tl.load(STATE + 8 * n_rows + index, mask=valid, other=0)
    return c_lo, c_hi, n_lo, n_hi, b_lo, b_hi, guess, step, flags


@triton.jit
def _store_state(
    STATE: tl.tensor, n_rows: tl.tensor, index: tl.tensor, valid: tl.tensor, state: tl.tensor
) -> None:
    """Store the search state of the rows at `index`, as `_load_state` loads it."""
    c_lo, c_hi, n_lo, n_hi, b_lo, b_hi, guess, step, flags = state
    tl.store(STATE + 0 * n_rows + index, c_lo, mask=valid)
    tl.store(STATE + 1 * n_rows + index, c_hi, mask=valid)
    tl.store(STATE + 2 * n_rows + index, n_lo, mask=valid)
    tl.store(STATE + 3 * n_rows + index, n_hi, mask=valid)
    floats = STATE.to(tl.pointer_type(tl.float32))
    tl.store(floats + 4 * n_rows + index, b_lo, mask=valid)
    tl.store(floats + 5 * n_rows + index, b_hi, mask=valid)
    tl.store(floats + 6 * n_rows + index, guess, mask=valid)
    tl.store(floats + 7 * n_rows + index, step, mask=valid)
    tl.store(STATE + 8 * n_rows + index, flags, mask=valid)


@triton.jit
def _stats_kernel(
    QI: tl.tensor,
    KI: tl.tensor,
    STATE: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    T: tl.tensor,
    TOPK: tl.tensor,
    H_SEL: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    ORDER_LO: tl.constexpr,
    ORDER_HI: tl.constexpr,
    HALF_SIGMAS: tl.constexpr,
    HALF_MIN: tl.constexpr,
) -> None:
    """Each row's empty bracket, and a Gaussian guess of its k-th score from its moments.

    step is the half-width of the first pass's middle bracket: HALF_SIGMAS standard
    deviations of the k-th order statistic's rank, plus HALF_MIN, in scores.
    """
    _, bh, rows, q, KI, last, need, tile_lo, full_lo, full_hi, tile_hi = _block(
        QI, KI, stride_qb, stride_qt, stride_qh, stride_kb, stride_kh, T, TOPK, H_SEL,
        KV_GROUP, WINDOW, SINK, D_R, BLOCK_DR,
    )  # fmt: skip
    dr = tl.arange(0, BLOCK_DR)
    n_cand = tl.maximum(last - SINK + 1, 0)
    guess = tl.zeros([64], dtype=tl.float32)
    step = tl.zeros([64], dtype=tl.float32)
    if tl.max(need.to(tl.int32), axis=0) > 0:
        total = tl.zeros([64], dtype=tl.float32)
        square = tl.zeros([64], dtype=tl.float32)
        for part in tl.static_range(3):
            if part == 0:
                lo, hi = tile_lo, full_lo
            elif part == 1:
                lo, hi = full_lo, full_hi
            else:
                lo, hi = full_hi, tile_hi
            total, square = _moments(
                total, square, q, KI, stride_kt, lo, hi, T, dr, last, SINK, part != 1, D_R
            )
        count = tl.maximum(n_cand, 1).to(tl.float32)
        mean = total / count
        sigma = tl.sqrt(tl.maximum(square / count - mean * mean, 0.0)) + 1e-30
        kept = TOPK / count
        z = _inverse_normal(tl.minimum(tl.maximum(1.0 - kept, 1e-6), 1.0 - 1e-6))
        guess = mean + sigma * z
        # Candidates per unit score at the guess, and the half-width in ranks to cover
        density = count * tl.exp(-0.5 * z * z) * 0.3989422804014327 / sigma
        ranks = HALF_SIGMAS * tl.sqrt(TOPK * tl.maximum(1.0 - kept, 0.0)) + HALF_MIN
        step = ranks / tl.maximum(density, 1e-30)
    c_lo = tl.full([64], ORDER_LO, tl.int32)
    c_hi = tl.full([64], ORDER_HI + 1, tl.int32)
    b_lo = tl.full([64], float("-inf"), tl.float32)
    b_hi = tl.full([64], float("inf"), tl.float32)
    state = (
        c_lo,
        c_hi,
        n_cand,
        tl.zeros([64], dtype=tl.int32),
        b_lo,
        b_hi,
        guess,
        step,
        tl.zeros([64], dtype=tl.int32),
    )
    n_rows = tl.num_programs(1) * T
    valid = rows < T
    _store_state(STATE, n_rows, bh * T + rows, valid, state)


@triton.jit
def _compact_kernel(
    STATE: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    T: tl.tensor,
    TOPK: tl.tensor,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """The rows of each (batch, head) whose search is still open, in order, and their count."""
    bh = tl.program_id(0)
    n_rows = tl.num_programs(0) * T
    base = 0
    for start in range(0, T, CHUNK):
        rows = start + tl.arange(0, CHUNK)
        valid = rows < T
        flags = tl.load(STATE + 8 * n_rows + bh * T + rows, mask=valid, other=0)
        need = valid & (rows - WINDOW - SINK + 1 > TOPK)
        open_ = need & ((flags & (COLLECTED | TIED)) == 0)
        slot = base + tl.cumsum(open_.to(tl.int32), axis=0) - 1
        tl.store(LIST + bh * T + slot, rows, mask=open_)
        base += tl.sum(open_.to(tl.int32), axis=0)
    tl.store(COUNT + bh, base)


@triton.jit
def _popcount(x: tl.tensor) -> tl.tensor:
    """Set bits of each int32: the hardware's count, or shifts and masks when interpreted."""
    if INTERPRETED:
        x = x.to(tl.uint32, bitcast=True)
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return ((x * 0x01010101) >> 24).to(tl.int32)
    return libdevice.popc(x)


@triton.jit
def _level_counts(
    f1: tl.tensor,
    f2: tl.tensor,
    f3: tl.tensor,
    f4: tl.tensor,
    q: tl.tensor,
    KI: tl.tensor,
    stride_kt: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    last: tl.tensor,
    x1: tl.tensor,
    x2: tl.tensor,
    x3: tl.tensor,
    x4: tl.tensor,
    SINK: tl.constexpr,
    MASKED: tl.constexpr,
    D_R: tl.constexpr,
) -> tl.tensor:
    """Per row, the candidates at each of four nested levels.

    A candidate's level is how many of x1 <= x2 <= x3 <= x4 it reaches. One int32 sum per tile
    counts all four, a byte each, as a tile holds at most 64 per level.
    """
    for n in range(lo, hi):
        x, cols = _scores(q, KI, stride_kt, n, T, dr, D_R)
        level = (x >= x1[:, None]).to(tl.int32) + (x >= x2[:, None]).to(tl.int32)
        level += (x >= x3[:, None]).to(tl.int32) + (x >= x4[:, None]).to(tl.int32)
        if MASKED:
            cand = (cols[None, :] >= SINK) & (cols[None, :] <= last[:, None])
            level = tl.where(cand, level, 0)
        packed = tl.sum(tl.where(level > 0, 1 << (8 * (level - 1)), 0), axis=1)
        f1 += packed & 255
        f2 += (packed >> 8) & 255
        f3 += (packed >> 16) & 255
        f4 += (packed >> 24) & 255
    return f1, f2, f3, f4


@triton.jit
def _collect(
    kept: tl.tensor,
    q: tl.tensor,
    KI: tl.tensor,
    BUF: tl.tensor,
    stride_kt: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    last: tl.tensor,
    index: tl.tensor,
    collect: tl.tensor,
    low: tl.tensor,
    high: tl.tensor,
    scale: tl.tensor,
    SINK: tl.constexpr,
    MASKED: tl.constexpr,
    D_R: tl.constexpr,
    CAP: tl.constexpr,
    RANK: tl.constexpr,
) -> tl.tensor:
    """The collecting rows' keys in [low, high), at their next slots of BUF.

    Each slot holds the key's rounded score bits over its index.

    A key's slot is the set bits below it in its row's packed take mask, so no atomics. Each
    64-key tile goes as two halves of 32, one word of the mask each, to halve what is live.
    """
    lane = tl.arange(0, 32)[None, :]
    col = tl.arange(0, 32)
    byte = tl.arange(0, 16)
    weights = tl.where(col[:, None] // 8 == byte[None, :], 1 << (col[:, None] % 8), 0).to(tl.float16)
    shifts = (8 * (byte % 4))[None, :]
    row = BUF + index.to(tl.int64)[:, None] * CAP
    for n in range(lo, hi):
        for half in tl.static_range(2):
            cols = n * 64 + half * 32 + col
            kmask = (cols[:, None] < T) & (dr[None, :] < D_R)
            k = tl.load(KI + cols[:, None] * stride_kt + dr[None, :], mask=kmask, other=0.0)
            x = tl.dot(q, tl.trans(k))
            take = collect[:, None] & (x >= low[:, None]) & (x < high[:, None])
            if MASKED:
                take = take & (cols[None, :] >= SINK) & (cols[None, :] <= last[:, None])
            if tl.max(take.to(tl.int32)) > 0:
                # The half's take bits, a byte per 8 keys, summed into one word
                ones = tl.where(take, 1.0, 0.0).to(tl.float16)
                bytes_ = tl.dot(ones, weights).to(tl.int32) << shifts
                word = tl.sum(tl.where(byte[None, :] < 4, bytes_, 0), axis=1)
                slot = kept[:, None] + _popcount(word[:, None] & ((1 << lane) - 1))
                # The score's own rounding and the key: _resolve_kernel ranks them by class
                rounded = x.to(RANK).to(tl.int16, bitcast=True).to(tl.int32)
                tl.store(row + slot, (rounded << 16) | cols[None, :], mask=take & (slot < CAP))
                kept += _popcount(word)
    return kept


@triton.jit
def _thresholds(
    state: tl.tensor,
    passes: tl.tensor,
    TABLE: tl.tensor,
    scale: tl.tensor,
    TOPK: tl.tensor,
    CAP: tl.constexpr,
    VALUE_PASSES: tl.constexpr,
    RANK: tl.constexpr,
    ORDER_LO: tl.constexpr,
) -> tl.tensor:
    """A pass's four nested threshold classes per row, their fp32 boundaries, and its step.

    A pure function of the row state, so a pass computes it before its count loop and again
    after, rather than keeping the state live across the loop.
    """
    c_lo, c_hi, n_lo, n_hi, b_lo, b_hi, guess, step, flags = state
    split = (flags & SPLIT) != 0
    stepped = flags >> 4
    width = c_hi - c_lo
    inside = n_lo - n_hi
    closed = (b_lo > float("-inf")) & (b_hi < float("inf"))
    if passes == 0:
        t1 = _class(guess - 8.0 * step, scale, RANK)
        t2 = _class(guess - step, scale, RANK)
        t3 = _class(guess + step, scale, RANK)
        t4 = _class(guess + 8.0 * step, scale, RANK)
    else:
        # Interpolate a closed bracket's counts; the middle bracket holds about CAP / 4 keys
        span = b_hi - b_lo
        frac = tl.minimum(tl.maximum((n_lo - TOPK).to(tl.float32) + 0.5, 0.0) / tl.maximum(inside, 1), 1.0)
        at = b_lo + frac * span
        half = span * (CAP * 0.125) / tl.maximum(inside, 1).to(tl.float32)
        t1 = _class(at - 4.0 * half, scale, RANK)
        t2 = _class(at - half, scale, RANK)
        t3 = _class(at + half, scale, RANK)
        t4 = _class(at + 4.0 * half, scale, RANK)
        # A closed bracket that barely shrank splits its scores in five
        t1 = tl.where(split, _class(b_lo + 0.2 * span, scale, RANK), t1)
        t2 = tl.where(split, _class(b_lo + 0.4 * span, scale, RANK), t2)
        t3 = tl.where(split, _class(b_lo + 0.6 * span, scale, RANK), t3)
        t4 = tl.where(split, _class(b_lo + 0.8 * span, scale, RANK), t4)
        # Step geometrically out of an open end, sixteenfold a pass
        miss_low = b_lo == float("-inf")
        miss_high = b_hi == float("inf")
        t1 = tl.where(miss_low, _class(b_hi - 16.0 * step, scale, RANK), t1)
        t2 = tl.where(miss_low, _class(b_hi - 6.35 * step, scale, RANK), t2)
        t3 = tl.where(miss_low, _class(b_hi - 2.52 * step, scale, RANK), t3)
        t4 = tl.where(miss_low, _class(b_hi - step, scale, RANK), t4)
        t1 = tl.where(miss_high, _class(b_lo + step, scale, RANK), t1)
        t2 = tl.where(miss_high, _class(b_lo + 2.52 * step, scale, RANK), t2)
        t3 = tl.where(miss_high, _class(b_lo + 6.35 * step, scale, RANK), t3)
        t4 = tl.where(miss_high, _class(b_lo + 16.0 * step, scale, RANK), t4)
        step = tl.where(miss_low | miss_high, 16.0 * step, step)
        # Split the classes in five after three steps out or VALUE_PASSES: the bound
        late = (stepped >= 3) | (passes >= VALUE_PASSES)
        t1 = tl.where(late, c_lo + width // 5, t1)
        t2 = tl.where(late, c_lo + (2 * width) // 5, t2)
        t3 = tl.where(late, c_lo + (3 * width) // 5, t3)
        t4 = tl.where(late, c_lo + (4 * width) // 5, t4)
    # A small closed bracket is collected whole
    whole = closed & (inside <= CAP)
    t1 = tl.where(whole, c_lo, t1)
    t2 = tl.where(whole, c_lo, t2)
    t3 = tl.where(whole, c_hi, t3)
    t4 = tl.where(whole, c_hi, t4)
    t1 = tl.minimum(tl.maximum(t1, c_lo), c_hi)
    t2 = tl.minimum(tl.maximum(t2, t1), c_hi)
    t3 = tl.minimum(tl.maximum(t3, t2), c_hi)
    t4 = tl.minimum(tl.maximum(t4, t3), c_hi)
    x1 = tl.load(TABLE + (t1 - ORDER_LO))
    x2 = tl.load(TABLE + (t2 - ORDER_LO))
    x3 = tl.load(TABLE + (t3 - ORDER_LO))
    x4 = tl.load(TABLE + (t4 - ORDER_LO))
    return t1, t2, t3, t4, x1, x2, x3, x4, step


@triton.jit
def _search_pass(
    STATE: tl.tensor,
    n_rows: tl.tensor,
    index: tl.tensor,
    valid: tl.tensor,
    passes: tl.tensor,
    q: tl.tensor,
    KI: tl.tensor,
    TABLE: tl.tensor,
    stride_kt: tl.tensor,
    T: tl.tensor,
    dr: tl.tensor,
    last: tl.tensor,
    tile_lo: tl.tensor,
    full_lo: tl.tensor,
    full_hi: tl.tensor,
    tile_hi: tl.tensor,
    scale: tl.tensor,
    TOPK: tl.tensor,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    CAP: tl.constexpr,
    VALUE_PASSES: tl.constexpr,
    RANK: tl.constexpr,
    ORDER_LO: tl.constexpr,
) -> None:
    """One pass over every candidate of open rows: four counts, and the middle bracket kept.

    The middle bracket [t2, t3) is where the pass expects the k-th score; if it is there and
    at most CAP keys fall in it, the row is collected and done. Otherwise the counts give the
    next bracket. A closed bracket of at most CAP keys is collected whole, so a row is done the
    pass after its bracket gets that small; one of a single class with more keys is TIED,
    and the emit pass keeps the latest of them. Reads and writes the rows' state in STATE;
    _collect_kernel then collects the rows marked COLLECTED.
    """
    state = _load_state(STATE, n_rows, index, valid)
    _, _, _, _, x1, x2, x3, x4, _ = _thresholds(
        state, passes, TABLE, scale, TOPK, CAP, VALUE_PASSES, RANK, ORDER_LO
    )
    f1 = tl.zeros([64], dtype=tl.int32)
    f2 = tl.zeros([64], dtype=tl.int32)
    f3 = tl.zeros([64], dtype=tl.int32)
    f4 = tl.zeros([64], dtype=tl.int32)
    for part in tl.static_range(3):
        if part == 0:
            lo, hi = tile_lo, full_lo
        elif part == 1:
            lo, hi = full_lo, full_hi
        else:
            lo, hi = full_hi, tile_hi
        f1, f2, f3, f4 = _level_counts(
            f1, f2, f3, f4, q, KI, stride_kt, lo, hi, T, dr, last, x1, x2, x3, x4, SINK,
            part != 1, D_R,
        )  # fmt: skip
    state = _load_state(STATE, n_rows, index, valid)
    t1, t2, t3, t4, x1, x2, x3, x4, step = _thresholds(
        state, passes, TABLE, scale, TOPK, CAP, VALUE_PASSES, RANK, ORDER_LO
    )
    c_lo, c_hi, n_lo, n_hi, b_lo, b_hi, guess, _, flags = state
    split = (flags & SPLIT) != 0
    stepped = flags >> 4
    open_ = valid & ((flags & (COLLECTED | TIED)) == 0)
    inside = n_lo - n_hi
    n4 = f4
    n3 = f3 + n4
    n2 = f2 + n3
    n1 = f1 + n2
    # Collected: the k-th is in the middle bracket, and every key of it fits
    done = open_ & (n2 >= TOPK) & (n3 < TOPK) & (n2 - n3 <= CAP)
    # Else the next bracket, between the thresholds whose counts straddle TOPK
    up4 = open_ & ~done & (n4 >= TOPK)
    up3 = open_ & ~done & ~up4 & (n3 >= TOPK)
    up2 = open_ & ~done & ~up4 & ~up3 & (n2 >= TOPK)
    up1 = open_ & ~done & ~up4 & ~up3 & ~up2 & (n1 >= TOPK)
    down = open_ & ~done & (n1 < TOPK)
    mid = done | up2
    c_lo = tl.where(up4, t4, tl.where(up3, t3, tl.where(mid, t2, tl.where(up1, t1, c_lo))))
    n_lo = tl.where(up4, n4, tl.where(up3, n3, tl.where(mid, n2, tl.where(up1, n1, n_lo))))
    b_lo = tl.where(up4, x4, tl.where(up3, x3, tl.where(mid, x2, tl.where(up1, x1, b_lo))))
    c_hi = tl.where(up3, t4, tl.where(mid, t3, tl.where(up1, t2, tl.where(down, t1, c_hi))))
    n_hi = tl.where(up3, n4, tl.where(mid, n3, tl.where(up1, n2, tl.where(down, n1, n_hi))))
    b_hi = tl.where(up3, x4, tl.where(mid, x3, tl.where(up1, x2, tl.where(down, x1, b_hi))))
    now_closed = (b_lo > float("-inf")) & (b_hi < float("inf"))
    tied = open_ & ~done & (c_hi - c_lo == 1) & (n_lo - n_hi > CAP)
    split = now_closed & ((n_lo - n_hi) * 2 > inside)
    stepped = tl.where(now_closed, stepped, stepped + 1)
    flags = tl.where(
        open_,
        split.to(tl.int32) * SPLIT
        + done.to(tl.int32) * COLLECTED
        + tied.to(tl.int32) * TIED
        + (stepped << 4),
        flags,
    )
    state = (c_lo, c_hi, n_lo, n_hi, b_lo, b_hi, guess, step, flags)
    _store_state(STATE, n_rows, index, valid, state)


@triton.jit
def _listed_rows(
    QI: tl.tensor,
    KI: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kh: tl.tensor,
    T: tl.tensor,
    H_SEL: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
) -> tl.tensor:
    """The 64 rows of _compact_kernel's list this program takes, and their candidate tiles."""
    bh = tl.program_id(1)
    open_rows = tl.load(COUNT + bh)
    slots = tl.program_id(0) * 64 + tl.arange(0, 64)
    valid = slots < open_rows
    rows = tl.load(LIST + bh * T + slots, mask=valid, other=0)
    b = bh // H_SEL
    h = bh % H_SEL
    dr = tl.arange(0, BLOCK_DR)
    QI += b * stride_qb + h * stride_qh
    qmask = valid[:, None] & (dr[None, :] < D_R)
    q = tl.load(QI + rows[:, None] * stride_qt + dr[None, :], mask=qmask, other=0.0)
    KI += b * stride_kb + (h // KV_GROUP) * stride_kh
    last = tl.where(valid, rows - WINDOW, -1)
    tile_lo = SINK // 64
    tile_hi = tl.max(last, axis=0) // 64 + 1
    full_lo = tl.minimum((SINK + 63) // 64, tile_hi)
    least_last = tl.min(tl.where(valid, last, T), axis=0)
    full_hi = tl.minimum(tl.maximum((least_last + 1) // 64, full_lo), tile_hi)
    return bh * T + rows, valid, q, KI, last, tile_lo, full_lo, full_hi, tile_hi


@triton.jit
def _collect_kernel(
    QI: tl.tensor,
    KI: tl.tensor,
    STATE: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    BUF: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    T: tl.tensor,
    scale: tl.tensor,
    H_SEL: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    CAP: tl.constexpr,
    RANK: tl.constexpr,
) -> None:
    """Collect the bracket keys of the listed rows the last pass marked COLLECTED."""
    if tl.program_id(0) * 64 < tl.load(COUNT + tl.program_id(1)):
        index, valid, q, KI, last, tile_lo, full_lo, full_hi, tile_hi = _listed_rows(
            QI, KI, LIST, COUNT, stride_qb, stride_qt, stride_qh, stride_kb, stride_kh, T, H_SEL,
            KV_GROUP, WINDOW, SINK, D_R, BLOCK_DR,
        )  # fmt: skip
        n_rows = tl.num_programs(1) * T
        flags = tl.load(STATE + 8 * n_rows + index, mask=valid, other=0)
        collect = valid & ((flags & COLLECTED) != 0)
        if tl.max(collect.to(tl.int32), axis=0) > 0:
            floats = STATE.to(tl.pointer_type(tl.float32))
            low = tl.load(floats + 4 * n_rows + index, mask=collect, other=0.0)
            high = tl.load(floats + 5 * n_rows + index, mask=collect, other=0.0)
            dr = tl.arange(0, BLOCK_DR)
            kept = tl.zeros([64], dtype=tl.int32)
            for part in tl.static_range(3):
                if part == 0:
                    lo, hi = tile_lo, full_lo
                elif part == 1:
                    lo, hi = full_lo, full_hi
                else:
                    lo, hi = full_hi, tile_hi
                kept = _collect(
                    kept, q, KI, BUF, stride_kt, lo, hi, T, dr, last, index, collect, low, high,
                    scale, SINK, part != 1, D_R, CAP, RANK,
                )  # fmt: skip


@triton.jit
def _pass_kernel(
    QI: tl.tensor,
    KI: tl.tensor,
    TABLE: tl.tensor,
    STATE: tl.tensor,
    LIST: tl.tensor,
    COUNT: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    T: tl.tensor,
    scale: tl.tensor,
    TOPK: tl.tensor,
    H_SEL: tl.tensor,
    passes: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    CAP: tl.constexpr,
    VALUE_PASSES: tl.constexpr,
    MAX_PASSES: tl.constexpr,
    LOOP: tl.constexpr,
    RANK: tl.constexpr,
    ORDER_LO: tl.constexpr,
) -> None:
    """A search pass over 64 of the rows _compact_kernel listed; with LOOP, until they are done.

    Rows in order of position share most key tiles, so the open rows of a (batch, head) are
    regrouped every pass: a pass costs as much as its open rows, not their blocks.
    """
    if tl.program_id(0) * 64 < tl.load(COUNT + tl.program_id(1)):
        index, valid, q, KI, last, tile_lo, full_lo, full_hi, tile_hi = _listed_rows(
            QI, KI, LIST, COUNT, stride_qb, stride_qt, stride_qh, stride_kb, stride_kh, T, H_SEL,
            KV_GROUP, WINDOW, SINK, D_R, BLOCK_DR,
        )  # fmt: skip
        dr = tl.arange(0, BLOCK_DR)
        n_rows = tl.num_programs(1) * T
        _search_pass(
            STATE, n_rows, index, valid, passes, q, KI, TABLE, stride_kt, T, dr, last,
            tile_lo, full_lo, full_hi, tile_hi, scale, TOPK, SINK, D_R, CAP, VALUE_PASSES, RANK,
            ORDER_LO,
        )  # fmt: skip
        if LOOP:
            more = passes + 1
            flags = tl.load(STATE + 8 * n_rows + index, mask=valid, other=COLLECTED)
            while (tl.max(((flags & (COLLECTED | TIED)) == 0).to(tl.int32), axis=0) > 0) & (
                more < MAX_PASSES
            ):
                _search_pass(
                    STATE, n_rows, index, valid, more, q, KI, TABLE, stride_kt, T, dr,
                    last, tile_lo, full_lo, full_hi, tile_hi, scale, TOPK, SINK, D_R, CAP,
                    VALUE_PASSES, RANK, ORDER_LO,
                )  # fmt: skip
                flags = tl.load(STATE + 8 * n_rows + index, mask=valid, other=COLLECTED)
                more += 1


@triton.jit
def _emit_kernel(
    QI: tl.tensor,
    KI: tl.tensor,
    TABLE: tl.tensor,
    STATE: tl.tensor,
    BITS: tl.tensor,
    OCC: tl.tensor,
    stride_qb: tl.tensor,
    stride_qt: tl.tensor,
    stride_qh: tl.tensor,
    stride_kb: tl.tensor,
    stride_kt: tl.tensor,
    stride_kh: tl.tensor,
    stride_bb: tl.tensor,
    stride_bh: tl.tensor,
    stride_bt: tl.tensor,
    stride_ob: tl.tensor,
    stride_oh: tl.tensor,
    stride_oq: tl.tensor,
    T: tl.tensor,
    TOPK: tl.tensor,
    H_SEL: tl.tensor,
    KV_GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    D_R: tl.constexpr,
    BLOCK_DR: tl.constexpr,
    ORDER_LO: tl.constexpr,
) -> None:
    """The bits of every key each row keeps, and the tiles they occupy.

    Every key above the row's bracket, and of its bracket's keys those at or past
    _resolve_kernel's cut, or a TIED row's latest; a row with at most TOPK candidates keeps
    them all.
    """
    pid_m, bh, rows, q, KI, last, need, tile_lo, full_lo, full_hi, tile_hi = _block(
        QI, KI, stride_qb, stride_qt, stride_qh, stride_kb, stride_kh, T, TOPK, H_SEL,
        KV_GROUP, WINDOW, SINK, D_R, BLOCK_DR,
    )  # fmt: skip
    dr = tl.arange(0, BLOCK_DR)
    n_rows = tl.num_programs(1) * T
    state = _load_state(STATE, n_rows, bh * T + rows, rows < T)
    n_hi, b_lo, b_hi, flags = state[3], state[4], state[5], state[8]
    tied = need & ((flags & TIED) != 0)
    collected = need & ((flags & COLLECTED) != 0)
    # A collected row keeps the classes above its cut's, and its cut's from the cut's key on
    cut_class = tl.load(STATE + 6 * n_rows + bh * T + rows, mask=collected, other=0)
    cut_key = tl.load(STATE + 7 * n_rows + bh * T + rows, mask=collected, other=0)
    above_cut = tl.load(TABLE + (cut_class + 1 - ORDER_LO), mask=collected, other=0.0)
    cut_low = tl.load(TABLE + (cut_class - ORDER_LO), mask=collected, other=0.0)
    upper = tl.where(collected, above_cut, tl.where(need, b_hi, float("-inf")))
    cut_low = tl.where(collected, cut_low, float("inf"))
    lower = tl.where(tied, b_lo, upper)
    ties_left = tl.where(tied, TOPK - n_hi, 0)
    b = bh // H_SEL
    h = bh % H_SEL
    BITS += b * stride_bb + h * stride_bh
    OCC += b * stride_ob + h * stride_oh + pid_m * stride_oq
    for part in tl.static_range(3):
        if part == 0:
            lo, hi = full_hi, tile_hi
        elif part == 1:
            lo, hi = full_lo, full_hi
        else:
            lo, hi = tile_lo, full_lo
        ties_left = _emit(
            ties_left, q, KI, stride_kt, BITS, stride_bt, OCC, lo, hi, T, dr, rows, last, upper,
            lower, cut_low, cut_key, SINK, part != 1, D_R,
        )  # fmt: skip


@triton.jit
def _resolve_kernel(
    STATE: tl.tensor,
    BUF: tl.tensor,
    T: tl.tensor,
    scale: tl.tensor,
    TOPK: tl.tensor,
    CAP: tl.constexpr,
    RANK: tl.constexpr,
) -> None:
    """Each collected row's least priority to keep, of its bracket keys, for _emit_kernel.

    A key's priority is its class shifted past its index, as dsa_select.selection_priority's; the
    row keeps the TOPK - n_hi highest of its n_lo - n_hi bracket keys, so the least of them
    is the largest p with that many at or above it, built bit by bit.
    """
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    bh = tl.program_id(1)
    valid = rows < T
    n_rows = tl.num_programs(1) * T
    index = bh * T + rows
    flags = tl.load(STATE + 8 * n_rows + index, mask=valid, other=0)
    collected = valid & ((flags & COLLECTED) != 0)
    if tl.max(collected.to(tl.int32), axis=0) > 0:
        n_lo = tl.load(STATE + 2 * n_rows + index, mask=collected, other=0)
        n_hi = tl.load(STATE + 3 * n_rows + index, mask=collected, other=0)
        slots = tl.arange(0, CAP)
        present = collected[:, None] & (slots[None, :] < (n_lo - n_hi)[:, None])
        at = index.to(tl.int64)[:, None] * CAP + slots[None, :]
        packed = tl.load(BUF + at, mask=present, other=0)
        rounded = (packed >> 16).to(tl.int16).to(RANK, bitcast=True).to(tl.float32)
        rank = _order((rounded * scale).to(RANK))
        priority = tl.where(present, rank * 65536 + (packed & 0xFFFF), -(2**31))
        # Unsigned, in the same order, so the cut is built bit by bit from the top
        ordered = (priority ^ -(2**31)).to(tl.uint32, bitcast=True)
        take = TOPK - n_hi
        cut = tl.zeros([64], dtype=tl.uint32)
        for bit in tl.static_range(31, -1, -1):
            trial = cut | (1 << bit)
            reach = tl.sum((ordered >= trial[:, None]).to(tl.int32), axis=1) >= take
            cut = tl.where(reach, trial, cut)
        least = cut.to(tl.int32, bitcast=True) ^ -(2**31)
        # The cut's class and key, for _emit_kernel to compare raw scores against
        tl.store(STATE + 6 * n_rows + index, least >> 16, mask=collected)
        tl.store(STATE + 7 * n_rows + index, least & 0xFFFF, mask=collected)


# Passes launched one per kernel over the rows still open, before one that loops to the end
FIXED_PASSES = 2


def _select(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the long-range keys each query keeps, by the passes of the module's search.

    Args:
        index_queries: Head-major indexer queries [b, h_sel, t, d_r].
        index_keys: Indexer keys [b, h_k, t, d_r].
        window_size: Recent keys each query keeps by position.
        sink_size: First keys each query keeps by position.
        top_k: Long-range keys each query keeps.
        scale: The factor scores are scaled by before rounding.

    Returns:
        The bitmask [b, h_sel, t, w] int32.
        The occupancy [b, h_sel, n, n] int32.
    """
    b, h_sel, t, d_r = index_queries.shape
    low, high = order_range(index_queries.dtype)
    n = occupancy_size(t)
    device = index_queries.device
    bits = torch.zeros(b, h_sel, t, words_per_row(t), dtype=torch.int32, device=device)
    occupancy = torch.zeros(b, h_sel, n, n, dtype=torch.int32, device=device)
    rows = b * h_sel * t
    state = torch.empty(9, rows, dtype=torch.int32, device=device)
    buf = torch.empty(rows, CAP, dtype=torch.int32, device=device)
    listed = torch.empty(b * h_sel, t, dtype=torch.int32, device=device)
    counts = torch.empty(b * h_sel, dtype=torch.int32, device=device)
    index_queries = index_queries if index_queries.stride(-1) == 1 else index_queries.contiguous()
    index_keys = index_keys if index_keys.stride(-1) == 1 else index_keys.contiguous()
    table = boundary_table(scale, index_queries.dtype, device)
    grid = (triton.cdiv(t, 64), b * h_sel)
    strides = (
        index_queries.stride(0), index_queries.stride(2), index_queries.stride(1),
        index_keys.stride(0), index_keys.stride(2), index_keys.stride(1),
    )  # fmt: skip
    shape = dict(
        KV_GROUP=h_sel // index_keys.shape[1], WINDOW=window_size, SINK=sink_size, D_R=d_r,
        BLOCK_DR=max(16, triton.next_power_of_2(d_r)),
    )  # fmt: skip
    rank = tl.bfloat16 if index_queries.dtype == torch.bfloat16 else tl.float16
    _stats_kernel[grid](
        index_queries, index_keys, state, *strides, t, top_k, h_sel, ORDER_LO=low, ORDER_HI=high,
        HALF_SIGMAS=HALF_SIGMAS, HALF_MIN=HALF_MIN, **shape, num_warps=4, num_stages=2,
    )  # fmt: skip
    for p in range(FIXED_PASSES + 1):
        _compact_kernel[(b * h_sel,)](
            state, listed, counts, t, top_k, WINDOW=window_size, SINK=sink_size, CHUNK=1024, num_warps=4
        )
        _pass_kernel[grid](
            index_queries, index_keys, table, state, listed, counts, *strides, t, scale, top_k,
            h_sel, p, CAP=CAP, VALUE_PASSES=VALUE_PASSES, MAX_PASSES=MAX_PASSES,
            LOOP=p == FIXED_PASSES, RANK=rank, ORDER_LO=low, **shape, num_warps=4, num_stages=2,
        )  # fmt: skip
        _collect_kernel[grid](
            index_queries, index_keys, state, listed, counts, buf, *strides, t, scale, h_sel, CAP=CAP,
            RANK=rank, **shape, num_warps=4, num_stages=2,
        )  # fmt: skip
    _resolve_kernel[grid](state, buf, t, scale, top_k, CAP=CAP, RANK=rank, num_warps=4)
    _emit_kernel[grid](
        index_queries, index_keys, table, state, bits, occupancy, *strides,
        bits.stride(0), bits.stride(1), bits.stride(2),
        occupancy.stride(0), occupancy.stride(1), occupancy.stride(2),
        t, top_k, h_sel, ORDER_LO=low, **shape, num_warps=4, num_stages=2,
    )  # fmt: skip
    return bits, occupancy


@torch.library.custom_op("lsa::dsa_sparse_select", mutates_args=())
def select_op(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the Selection of head-major indexer queries over keys.

    Args:
        index_queries: Head-major indexer queries [b, h_sel, t, d_r].
        index_keys: Indexer keys [b, h_k, t, d_r].

    Returns:
        The Selection's bits and occupancy.
    """
    return _select(index_queries, index_keys, window_size, sink_size, top_k, scale)


@select_op.register_fake
def _(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empty tensors shaped as the bitmask and occupancy."""
    b, h_sel, t, _ = index_queries.shape
    n = occupancy_size(t)
    bits = index_queries.new_empty((b, h_sel, t, words_per_row(t)), dtype=torch.int32)
    return bits, index_queries.new_empty((b, h_sel, n, n), dtype=torch.int32)
