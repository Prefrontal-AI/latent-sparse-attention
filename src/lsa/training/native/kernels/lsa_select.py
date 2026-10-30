"""LSA's bucket ranking and key gather (reference.select_indices) in two Triton kernels.

reference.select_indices scores every visible centroid in PyTorch, tops-k the scores and gathers,
filters, ranks and scatters each pick's members, a dozen passes over (rows, top_p * cap) tensors.
Here:

1. _scores_kernel: a tile of queries against the centroids visible to it, a tile at a time, by
   tensor-core dots, q . c or max_h q_h . c. Rows are scored ROW_ELEMENTS scores at a time, so no
   whole (b, n, m) matrix exists at once.
2. _select_kernel, a program a query row: each visible score packed with its bucket into one
   int64 - the score's order-preserving bits high, the bucket reversed low - so the order is by
   score and an exact tie goes to the lower bucket. Of up to WHOLE_BUCKETS, a row that sees up to
   512 sorts the narrowest of WHOLE_WIDTHS that holds them (a bitonic top-k where it keeps
   fewer), so a row of an early block sorts 64 rather than all M_PAD. Otherwise a row of up to
   SORT_BUCKETS sorts them whole, and a wider one bisects its scores for a floor that leaves
   fewer than top_p + P_PAD candidates, falling back to a search of the key bits on ties, and
   sorts the candidates. Each orders a row's keys exactly, so they pick alike. Its pick_count'
   members are then gathered best first, EXPAND buckets at a time, in position order,
   filtered to the long-range keys past the sinks, ranked by a cumsum and written up to top_k:
   as select_indices' (b, n, top_k) positions, PAD past the taken keys, and as dsa_sparse's
   Selection bits and occupancy, which dsa_sparse.attention reads with no dense table. It stops
   at the bucket that fills top_k.

redo_flagged runs both over a list of flagged rows that a kernel builds on the device, its
programs looping over the rows the list holds, so lsa_rank redoes the rows it leaves without a
host sync.

The scores' dot is one of PRECISIONS: 'bf16x3' splits each fp32 centroid into three bf16 parts,
so with bf16 index queries, exact in bf16, it is as exact as fp32 at half tf32x3's tensor-core
cost; 'ieee', 'tf32x3' and 'tf32' are tl.dot's fp32 precisions; 'auto' takes bf16x3 for bf16
queries and tf32x3 otherwise.

The selection is select_indices' up to the scores: the kernel's dot sums in another order than
cuBLAS's, so two buckets whose scores differ in the last bits can swap, about one query row in
300 at 8 x 4096 against fp32 scores. torch.topk breaks exact
ties in no documented order; the kernel breaks them to the lower bucket.
"""

import torch
import triton
import triton.language as tl

from lsa.training.native.kernels import reference
from lsa.training.native.kernels.dsa_sparse import topk
from lsa.training.native.kernels.dsa_sparse.topk import Selection, occupancy_size, words_per_row

# The scores' dot, see the module docstring
PRECISIONS = ("auto", "bf16x3", "ieee", "tf32x3", "tf32")
PREC = {"ieee": 0, "tf32x3": 1, "tf32": 2, "bf16x3": 3}
# Scores held at once, (b, rows, m) fp32 - select_indices' chunk
ROW_ELEMENTS = 2**25
# Most buckets a row ranks; more take the reference
MAX_BUCKETS = 8192
# Most buckets a row sorts whole; past them it searches its pick_count-th score and sorts the pick_count
SORT_BUCKETS = 512
# Query rows times heads per score tile, and buckets per tile
# HEAD_TILE_ROWS serves eight bf16x3 heads up to 128 wide
TILE_ROWS = 128
HEAD_TILE_ROWS = 256
TILE_BUCKETS = 64
# Fewest rows a pass of redo_flagged takes from its scratch rather than its own buffers
SCRATCH_ROWS = 1024
# Bisections of a row's scores before it falls back to searching its key bits
BISECTIONS = 48
# Widths a row of fewer buckets sorts whole, the narrowest that holds them, in kernels of up to
# WHOLE_BUCKETS padded buckets; wider kernels of 8 or 16 warps run slower with the branches
WHOLE_WIDTHS = (64, 128, 256, 512)
WHOLE_BUCKETS = 2048
# Query rows and warps of a program by the padded bucket count
SELECT_LAUNCH = {512: (1, 1), 1024: (1, 2), 2048: (1, 4), 4096: (1, 8)}
# Picked buckets a row expands at a time, stopping once it holds top_k keys: NARROW_EXPAND up to
# 512 padded buckets (0.24 ms against 32's 0.27 at 8 x 4,096), EXPAND past them
EXPAND = 32
NARROW_EXPAND = 8
INT64_MIN = tl.constexpr(-(2**63))
OCC_BLOCK = tl.constexpr(topk.OCC_BLOCK)


@triton.jit
def _split3_kernel(C: tl.tensor, OUT: tl.tensor, n: tl.tensor, part: tl.tensor, BLOCK: tl.constexpr) -> None:
    """Split fp32 centroids into bf16 hi, mid and lo parts, which sum to them.

    Args:
        C: Centroids [n] fp32.
        OUT: The parts [3, n] bf16, `part` elements apart.
    """
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    held = at < n
    c = tl.load(C + at, mask=held)
    hi = c.to(tl.bfloat16)
    mid = (c - hi.to(tl.float32)).to(tl.bfloat16)
    lo = (c - hi.to(tl.float32) - mid.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + at, hi, mask=held)
    tl.store(OUT + part + at, mid, mask=held)
    tl.store(OUT + 2 * part + at, lo, mask=held)


def split3(centroids: torch.Tensor) -> torch.Tensor:
    """Split fp32 centroids into bf16x3's hi, mid and lo bf16 parts, which sum to them.

    Args:
        centroids: Centroids, fp32.

    Returns:
        Parts [3, ...] bf16.
    """
    centroids = centroids.float().contiguous()
    if not centroids.is_cuda or not centroids.numel():
        hi = centroids.bfloat16()
        mid = (centroids - hi.float()).bfloat16()
        lo = (centroids - hi.float() - mid.float()).bfloat16()
        return torch.stack([hi, mid, lo])
    n = centroids.numel()
    out = torch.empty(3, *centroids.shape, dtype=torch.bfloat16, device=centroids.device)
    _split3_kernel[(triton.cdiv(n, 1024),)](centroids, out, n, n, BLOCK=1024, num_warps=4)
    return out


def supported(index_queries: torch.Tensor, buckets: tuple[torch.Tensor, torch.Tensor] | None) -> bool:
    """Check whether the kernels take these index queries and buckets."""
    if buckets is None or not index_queries.is_cuda:
        return False
    d = index_queries.shape[-1]
    heads = index_queries.shape[2] if index_queries.dim() == 4 else 1
    m, cap = buckets[1].shape[1:]
    return (
        16 <= d <= 256
        and d & (d - 1) == 0
        and heads & (heads - 1) == 0
        and heads <= TILE_ROWS // 16
        and cap & (cap - 1) == 0
        and m <= MAX_BUCKETS
    )


@triton.jit
def _dot(
    q: tl.tensor, c_ptr: tl.tensor, at: tl.tensor, mask: tl.tensor, PREC: tl.constexpr, C_STRIDE: tl.tensor
) -> tl.tensor:
    """Return the dot of the query rows with the centroids at `at`, at the precision PREC names."""
    if PREC == 3:
        # q is bf16, exact; the fp32 centroid split in three bf16 parts sums to it within 2^-24
        hi = tl.load(c_ptr + at, mask=mask, other=0.0)
        mid = tl.load(c_ptr + C_STRIDE + at, mask=mask, other=0.0)
        lo = tl.load(c_ptr + 2 * C_STRIDE + at, mask=mask, other=0.0)
        dot = tl.dot(q, tl.trans(lo))
        dot = tl.dot(q, tl.trans(mid), dot)
        return tl.dot(q, tl.trans(hi), dot)
    c = tl.load(c_ptr + at, mask=mask, other=0.0)
    if PREC == 0:
        return tl.dot(q, tl.trans(c), input_precision="ieee")
    elif PREC == 1:
        return tl.dot(q, tl.trans(c), input_precision="tf32x3")
    else:
        return tl.dot(q, tl.trans(c), input_precision="tf32")


@triton.jit
def _scores_rows(
    Q: tl.tensor, C: tl.tensor, S: tl.tensor, b: tl.tensor, q_rows: tl.tensor, s_rows: tl.tensor, in_rows: tl.tensor, pos: tl.tensor, m: tl.tensor, c_split: tl.tensor, stride_qb: tl.tensor, stride_qn: tl.tensor, stride_cb: tl.tensor,
    stride_sb: tl.tensor, stride_sn: tl.tensor, BLOCK: tl.constexpr, PER_BLOCK: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr,
) -> None:  # fmt: skip
    """The scores of BM query rows, rows q_rows of Q at positions pos, as rows s_rows of S."""
    visible = tl.max(tl.where(in_rows, tl.minimum((pos // BLOCK) * PER_BLOCK, m), 0), axis=0)
    dims = tl.arange(0, D)
    heads = tl.arange(0, H)
    # Rows times heads, contiguous in a (b, n, h, d) query
    at = (q_rows[:, None] * stride_qn + heads[None, :] * D).reshape(BM * H)
    held = (in_rows[:, None] & (heads[None, :] < H)).reshape(BM * H)
    q = tl.load(Q + b * stride_qb + at[:, None] + dims[None, :], mask=held[:, None], other=0.0)
    if PREC != 3:
        q = q.to(tl.float32)
    for first in range(0, visible, BN):
        cols = first + tl.arange(0, BN)
        c_at = cols[:, None] * D + dims[None, :]
        dot = _dot(q, C + b * stride_cb, c_at, cols[:, None] < m, PREC, c_split)
        if H > 1:
            dot = tl.max(dot.reshape(BM, H, BN), axis=1)
        tl.store(
            S + b * stride_sb + s_rows[:, None] * stride_sn + cols[None, :],
            dot,
            mask=in_rows[:, None] & (cols[None, :] < m),
        )


@triton.jit
def _scores_kernel(
    Q: tl.tensor,
    C: tl.tensor,
    POS: tl.tensor,
    S: tl.tensor,
    LIST: tl.tensor,
    n: tl.tensor,
    m: tl.tensor,
    c_split: tl.tensor,
    listed0: tl.tensor,
    t_list: tl.tensor,
    stride_qb: tl.tensor,
    stride_qn: tl.tensor,
    stride_cb: tl.tensor,
    stride_sb: tl.tensor,
    stride_sn: tl.tensor,
    stride_lb: tl.tensor,
    BLOCK: tl.constexpr,
    PER_BLOCK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PREC: tl.constexpr,
    LISTED: tl.constexpr,
) -> None:
    """Score a tile of query rows against the centroids visible to them, a tile of centroids at a time.

    A LISTED launch scores the rows of a list of flagged rows, a program looping over its tiles.
    Otherwise a program scores the tile of rows it is given.

    Args:
        Q: Index queries [b, n, H * D].
        C: Centroids [b, m, D] fp32, or [3, b, m, D] bf16 if split three ways.
        POS: Each row's position [n] int64.
        S: Scores [b, rows, m] fp32.
        LIST: Each batch's flagged rows, and their count at column t_list [b, t_list + 1] int32.
        n: Rows to score.
        m: Centroids.
        c_split: Elements between the three parts of split centroids - 0 if not split.
    """
    b = tl.program_id(1).to(tl.int64)
    if LISTED:
        # Entries listed0.. of the batch's row list (_list_kernel's), n at most, BM a tile
        count = tl.load(LIST + b * stride_lb + t_list)
        rows_left = tl.maximum(tl.minimum(count - listed0, n), 0)
        for tile in range(tl.program_id(0), tl.cdiv(rows_left, BM), tl.num_programs(0)):
            local = tile * BM + tl.arange(0, BM)
            in_rows = local < rows_left
            pos = tl.load(LIST + b * stride_lb + listed0 + local, mask=in_rows, other=0)
            pos = pos.to(tl.int64)
            _scores_rows(
                Q, C, S, b, pos, local, in_rows, pos, m, c_split, stride_qb, stride_qn,
                stride_cb, stride_sb, stride_sn, BLOCK, PER_BLOCK, H, D, BM, BN, PREC,
            )  # fmt: skip
    else:
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        in_rows = rows < n
        pos = tl.load(POS + rows, mask=in_rows, other=0)
        _scores_rows(
            Q, C, S, b, rows, rows, in_rows, pos, m, c_split, stride_qb, stride_qn, stride_cb,
            stride_sb, stride_sn, BLOCK, PER_BLOCK, H, D, BM, BN, PREC,
        )  # fmt: skip


@triton.jit
def _first(x: tl.tensor, R: tl.constexpr, N: tl.constexpr, P: tl.constexpr) -> tl.tensor:
    """The first P < N of each row of x (R, N), as a reduction over the rest of its 3-d view."""
    lead = (tl.arange(0, N // P) == 0)[None, :, None]
    return tl.sum(tl.where(lead, x.reshape(R, N // P, P), 0), axis=1)


@triton.jit
def _rank_whole(
    ROW: tl.tensor,
    visible: tl.tensor,
    live: tl.tensor,
    ORDER_ROW: tl.tensor,
    R: tl.constexpr,
    N: tl.constexpr,
    P_PAD: tl.constexpr,
) -> None:
    """The first live of a row's N >= visible buckets best first, as buckets at ORDER_ROW."""
    cols = tl.arange(0, N)
    in_reach = cols[None, :] < visible[:, None]
    score = tl.load(ROW + cols[None, :], mask=in_reach, other=0.0)
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)
    # The low word N - 1 - bucket: the lower bucket of a tie sorts first
    key = (ordered << 32) | (N - 1 - cols)[None, :].to(tl.int64)
    key = tl.where(in_reach, key, INT64_MIN)
    if P_PAD < N:
        ranked = tl.topk(key, P_PAD, dim=1)
        lane = tl.arange(0, P_PAD)[None, :]
    else:
        ranked = tl.sort(key, dim=1, descending=True)
        lane = tl.arange(0, N)[None, :]
    bucket = (N - 1 - (ranked & (N - 1))).to(tl.int32)
    tl.store(ORDER_ROW + lane, bucket, mask=lane < live[:, None])


@triton.jit
def _rank_wide(
    ROW: tl.tensor, visible: tl.tensor, live: tl.tensor, pick_count: tl.tensor, KEYS_ROW: tl.tensor, ORDER_ROW: tl.tensor, R: tl.constexpr, M_PAD: tl.constexpr,
    P_PAD: tl.constexpr, SEARCH: tl.constexpr, BISECTIONS: tl.constexpr,
) -> None:  # fmt: skip
    """The first live of a row's M_PAD-wide buckets best first, sorted whole or searched."""
    cols = tl.arange(0, M_PAD)
    in_reach = cols[None, :] < visible[:, None]
    score = tl.load(ROW + cols[None, :], mask=in_reach, other=0.0)
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)
    # The low word M_PAD - 1 - bucket: the lower bucket of a tie sorts first
    key = (ordered << 32) | (M_PAD - 1 - cols)[None, :].to(tl.int64)
    key = tl.where(in_reach, key, INT64_MIN)
    lane = tl.arange(0, P_PAD)[None, :]
    if SEARCH:
        # Bisect each row's scores for a floor at least pick_count clear and under pick_count + P_PAD do
        low = tl.min(tl.where(in_reach, score, float("inf")), axis=1)
        high = tl.max(tl.where(in_reach, score, float("-inf")), axis=1)
        low_count = visible
        high_count = tl.sum((in_reach & (score >= high[:, None])).to(tl.int32), axis=1)
        open_rows = (low_count - high_count > P_PAD) & (high_count < pick_count)
        steps = 0
        while (tl.max(open_rows.to(tl.int32), axis=0) > 0) & (steps < BISECTIONS):
            middle = low + (high - low) * 0.5
            count = tl.sum((in_reach & (score >= middle[:, None])).to(tl.int32), axis=1)
            enough = open_rows & (count >= pick_count)
            short = open_rows & (count < pick_count)
            low = tl.where(enough, middle, low)
            low_count = tl.where(enough, count, low_count)
            high = tl.where(short, middle, high)
            high_count = tl.where(short, count, high_count)
            open_rows = (low_count - high_count > P_PAD) & (high_count < pick_count)
            steps += 1
        kept = in_reach & (score >= low[:, None])
        # A row whose top score pick_count or more buckets share never bisects and keeps them all, so
        # Past the scratch's 2 * P_PAD it takes the exact search too
        crowded = tl.sum(kept.to(tl.int32), axis=1) > 2 * P_PAD
        open_rows = open_rows | (crowded & (visible > pick_count))
        if tl.max(open_rows.to(tl.int32), axis=0) > 0:
            # Ties too many to bisect: the pick_count-th key's bits, from the top of their order
            flipped = tl.where(bits >= 0, bits ^ (-(2**31)), ~bits)
            unsigned = tl.where(in_reach, flipped, 0).to(tl.uint32, bitcast=True)
            cut = tl.zeros([R], tl.uint32)
            for bit in tl.static_range(31, -1, -1):
                trial = cut | (tl.full([R], 1, tl.uint32) << bit)
                enough = tl.sum((unsigned >= trial[:, None]).to(tl.int32), axis=1) >= pick_count
                cut = tl.where(enough, trial, cut)
            above = unsigned > cut[:, None]
            tied = (unsigned == cut[:, None]) & in_reach
            room = pick_count - tl.sum(above.to(tl.int32), axis=1)
            exact = above | (tied & (tl.cumsum(tied.to(tl.int32), axis=1) <= room[:, None]))
            kept = tl.where(open_rows[:, None], exact, kept)
        kept = tl.where((visible <= pick_count)[:, None], in_reach, kept)
        # Under 2 * P_PAD candidates a row, listed in its scratch and sorted from there
        slot = tl.cumsum(kept.to(tl.int32), axis=1) - 1
        listed_count = tl.sum(kept.to(tl.int32), axis=1)
        tl.store(KEYS_ROW + slot, key, mask=kept)
        tl.debug_barrier()
        wide = tl.arange(0, 2 * P_PAD)[None, :]
        listed = tl.load(KEYS_ROW + wide, mask=wide < listed_count[:, None], other=INT64_MIN)
        ranked = _first(tl.sort(listed, dim=1, descending=True), R, 2 * P_PAD, P_PAD)
    else:
        ranked = tl.sort(key, dim=1, descending=True)
        if P_PAD < M_PAD:
            ranked = _first(ranked, R, M_PAD, P_PAD)
    bucket = (M_PAD - 1 - (ranked & (M_PAD - 1))).to(tl.int32)
    tl.store(ORDER_ROW + lane, bucket, mask=lane < live[:, None])


@triton.jit
def _select_rows(
    S: tl.tensor,
    MEMBERS: tl.tensor,
    IDS: tl.tensor,
    BITS: tl.tensor,
    OCC: tl.tensor,
    KEYS: tl.tensor,
    ORDER: tl.tensor,
    b: tl.tensor,
    local: tl.tensor,
    o_row: tl.tensor,
    row: tl.tensor,
    pos: tl.tensor,
    in_rows: tl.tensor,
    n_local: tl.tensor,
    m: tl.tensor,
    pick_count: tl.tensor,
    stride_sb: tl.tensor,
    stride_sn: tl.tensor,
    stride_mb: tl.tensor,
    stride_ib: tl.tensor,
    stride_in: tl.tensor,
    stride_bb: tl.tensor,
    stride_bn: tl.tensor,
    stride_ob: tl.tensor,
    stride_on: tl.tensor,
    BLOCK: tl.constexpr,
    PER_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    CAP: tl.constexpr,
    GROUP: tl.constexpr,
    R: tl.constexpr,
    M_PAD: tl.constexpr,
    P_PAD: tl.constexpr,
    K_PAD: tl.constexpr,
    EXPAND: tl.constexpr,
    WRITE_IDS: tl.constexpr,
    WRITE_BITS: tl.constexpr,
    SEARCH: tl.constexpr,
    BISECTIONS: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    N4: tl.constexpr,
) -> None:
    """Rows local of S, query rows o_row of IDS (row of BITS) at positions pos; see the kernel."""
    visible = tl.where(in_rows, tl.minimum((pos // BLOCK) * PER_BLOCK, m), 0).to(tl.int32)
    live = tl.minimum(visible, pick_count)
    scratch = ((b * n_local + local) * P_PAD)[:, None]
    row_scores = S + b * stride_sb + local[:, None] * stride_sn
    # A row of fewer buckets sorts only the next power of two of them
    # The order is exact, so any width ranks them alike
    widest = tl.max(visible, axis=0)
    if N4 == 0:
        _rank_wide(
            row_scores, visible, live, pick_count, KEYS + 2 * scratch, ORDER + scratch, R, M_PAD,
            P_PAD, SEARCH, BISECTIONS,
        )  # fmt: skip
    elif widest <= N1:
        _rank_whole(row_scores, visible, live, ORDER + scratch, R, N1, P_PAD)
    elif widest <= N2:
        _rank_whole(row_scores, visible, live, ORDER + scratch, R, N2, P_PAD)
    elif widest <= N3:
        _rank_whole(row_scores, visible, live, ORDER + scratch, R, N3, P_PAD)
    elif widest <= N4:
        _rank_whole(row_scores, visible, live, ORDER + scratch, R, N4, P_PAD)
    else:
        _rank_wide(
            row_scores, visible, live, pick_count, KEYS + 2 * scratch, ORDER + scratch, R, M_PAD,
            P_PAD, SEARCH, BISECTIONS,
        )  # fmt: skip
    tl.debug_barrier()

    # Picked buckets best first, EXPAND at a time, until every row holds top_k keys
    slots = tl.arange(0, CAP)[None, None, :]
    out = IDS + b * stride_ib + o_row[:, None] * stride_in
    words = BITS + b * stride_bb + row[:, None] * stride_bn
    tiles = OCC + b * stride_ob + (row[:, None] // OCC_BLOCK) * stride_on
    members = MEMBERS + b * stride_mb
    taken_count = tl.zeros([R], tl.int32)
    first = 0
    wanting = (live > 0) & in_rows
    while tl.max(wanting.to(tl.int32), axis=0) > 0:
        rank = first + tl.arange(0, EXPAND)[None, :]
        picked = (rank < live[:, None]) & wanting[:, None]
        chosen = tl.load(ORDER + scratch + rank, mask=picked, other=0)
        ids = tl.load(
            members + chosen[:, :, None].to(tl.int64) * CAP + slots,
            mask=picked[:, :, None],
            other=-1,
        ).to(tl.int32)
        valid = picked[:, :, None] & (ids >= SINK) & (ids <= (pos - WINDOW)[:, None, None])
        ids = ids.reshape(R, EXPAND * CAP)
        valid = valid.reshape(R, EXPAND * CAP)
        order = taken_count[:, None] + tl.cumsum(valid.to(tl.int32), axis=1) - 1
        taken = valid & (order < TOP_K)
        if WRITE_IDS:
            tl.store(out + order, ids.to(tl.int64), mask=taken)
        if WRITE_BITS:
            # A group of GROUP slots holds consecutive keys of one word: one atomic a group
            flags = tl.where(taken, tl.full([R, EXPAND * CAP], 1, tl.int32) << (ids & 31), 0)
            flags = tl.sum(flags.reshape(R, EXPAND * CAP // GROUP, GROUP), axis=2)
            head = tl.where(taken, ids, -1).reshape(R, EXPAND * CAP // GROUP, GROUP)
            head = tl.max(head, axis=2)
            tl.atomic_or(words + (head >> 5), flags, mask=head >= 0, sem="relaxed")
            tl.store(tiles + head // OCC_BLOCK, tl.full(head.shape, 1, tl.int32), mask=head >= 0)
        taken_count += tl.sum(valid.to(tl.int32), axis=1)
        first += EXPAND
        wanting = wanting & (first < live) & (taken_count < TOP_K)
    if WRITE_IDS:
        rest = tl.arange(0, K_PAD)[None, :]
        count = tl.minimum(taken_count, TOP_K)[:, None]
        pad = (rest >= count) & (rest < TOP_K) & in_rows[:, None]
        tl.store(out + rest, tl.full([R, K_PAD], -1, tl.int64), mask=pad)


@triton.jit
def _select_kernel(
    S: tl.tensor,
    MEMBERS: tl.tensor,
    POS: tl.tensor,
    IDS: tl.tensor,
    BITS: tl.tensor,
    OCC: tl.tensor,
    KEYS: tl.tensor,
    ORDER: tl.tensor,
    LIST: tl.tensor,
    row0: tl.tensor,
    n_local: tl.tensor,
    m: tl.tensor,
    pick_count: tl.tensor,
    listed0: tl.tensor,
    t_list: tl.tensor,
    stride_sb: tl.tensor,
    stride_sn: tl.tensor,
    stride_mb: tl.tensor,
    stride_ib: tl.tensor,
    stride_in: tl.tensor,
    stride_bb: tl.tensor,
    stride_bn: tl.tensor,
    stride_ob: tl.tensor,
    stride_on: tl.tensor,
    stride_lb: tl.tensor,
    BLOCK: tl.constexpr,
    PER_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    WINDOW: tl.constexpr,
    SINK: tl.constexpr,
    CAP: tl.constexpr,
    GROUP: tl.constexpr,
    R: tl.constexpr,
    M_PAD: tl.constexpr,
    P_PAD: tl.constexpr,
    K_PAD: tl.constexpr,
    EXPAND: tl.constexpr,
    WRITE_IDS: tl.constexpr,
    WRITE_BITS: tl.constexpr,
    SEARCH: tl.constexpr,
    BISECTIONS: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    N3: tl.constexpr,
    N4: tl.constexpr,
    LISTED: tl.constexpr,
) -> None:
    """Pick the best buckets of each query row from its scores and gather their members' positions.

    Writes the positions, and the selection's bits and occupancy, as WRITE_IDS and WRITE_BITS say.

    Args:
        S: Scores [b, rows, m] fp32.
        MEMBERS: Member positions of each bucket [b, m, CAP] int32.
        POS: Each row's position [n] int64.
        IDS: Selected key positions [b, n, TOP_K].
        BITS: Selection bitmask [b, n, w] int32.
        OCC: Selection occupancy [b, h, n, n] int32.
        KEYS: Scratch for the packed scores [b, rows, keys] int64.
        ORDER: Scratch for the picked buckets [b, rows, P_PAD] int32.
        LIST: Each batch's flagged rows, and their count at column t_list [b, t_list + 1] int32.
        row0: First row of S.
        n_local: Rows of S.
        m: Buckets.
        pick_count: Buckets picked a row.
    """
    b = tl.program_id(1).to(tl.int64)
    if LISTED:
        # Entries listed0.. of the batch's row list (_list_kernel's), n_local at most, a row each
        count = tl.load(LIST + b * stride_lb + t_list)
        rows_left = tl.maximum(tl.minimum(count - listed0, n_local), 0)
        for first in range(tl.program_id(0), rows_left, tl.num_programs(0)):
            local = (first + tl.arange(0, R)).to(tl.int64)
            in_rows = local < rows_left
            pos = tl.load(LIST + b * stride_lb + listed0 + local, mask=in_rows, other=0)
            pos = pos.to(tl.int64)
            _select_rows(
                S, MEMBERS, IDS, BITS, OCC, KEYS, ORDER, b, local, pos, pos, pos,
                in_rows, n_local, m, pick_count, stride_sb, stride_sn, stride_mb,
                stride_ib, stride_in, stride_bb, stride_bn, stride_ob, stride_on,
                BLOCK, PER_BLOCK, TOP_K, WINDOW, SINK, CAP, GROUP, R, M_PAD, P_PAD,
                K_PAD, EXPAND, WRITE_IDS, WRITE_BITS, SEARCH, BISECTIONS, N1, N2,
                N3, N4,
            )  # fmt: skip
    else:
        local = tl.program_id(0) * R + tl.arange(0, R)
        in_rows = local < n_local
        local = tl.where(in_rows, local, 0).to(tl.int64)
        row = row0 + local
        pos = tl.load(POS + row)
        _select_rows(
            S, MEMBERS, IDS, BITS, OCC, KEYS, ORDER, b, local, local, row, pos, in_rows,
            n_local, m, pick_count, stride_sb, stride_sn, stride_mb, stride_ib, stride_in,
            stride_bb, stride_bn, stride_ob, stride_on, BLOCK, PER_BLOCK, TOP_K, WINDOW,
            SINK, CAP, GROUP, R, M_PAD, P_PAD, K_PAD, EXPAND, WRITE_IDS, WRITE_BITS,
            SEARCH, BISECTIONS, N1, N2, N3, N4,
        )  # fmt: skip


@triton.jit
def _list_kernel(
    CUM: tl.tensor,
    LIST: tl.tensor,
    t: tl.tensor,
    stride_cb: tl.tensor,
    stride_lb: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    """Each batch's flagged rows in order, LIST[b, :count], and count at LIST[b, t]."""
    b = tl.program_id(1).to(tl.int64)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    held = i < t
    cum = tl.load(CUM + b * stride_cb + i, mask=held, other=0)
    before = tl.load(CUM + b * stride_cb + i - 1, mask=held & (i > 0), other=0)
    tl.store(LIST + b * stride_lb + cum - 1, i, mask=held & (cum != before))
    tl.store(LIST + b * stride_lb + t + i * 0, cum, mask=i == t - 1)


def _select(
    index_queries: torch.Tensor,
    q_pos: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor],
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    ids: torch.Tensor | None,
    selection: Selection | None,
    precision: str,
    group: int,
    flag: torch.Tensor | None = None,
    scratch: torch.Tensor | None = None,
) -> None:
    """Select the keys each query row reads, as positions and as the selection's bits and occupancy.

    Scores are made a chunk of rows at a time into scratch, then a select kernel picks from them.
    With `flag`, every pass is launched over the flagged rows a kernel lists, so the host never waits.

    Args:
        index_queries: Index queries [b, n, d] or [b, n, h, d].
        q_pos: Each row's position [n].
        buckets: Centroids [b, m, d] and member positions [b, m, cap] of the buckets.
        block: Positions a block covers.
        bucket_size: Keys a bucket holds.
        top_p: Buckets picked a row.
        top_k: Keys read a row.
        window_size: Recent keys each query reads.
        sink_size: First keys each query reads.
        ids: Positions [b, n, top_k], written in place - or None to write none.
        selection: Bits and occupancy, written in place - or None to write none.
        precision: The scores' dot, see PRECISIONS.
        group: Slots of a bucket that hold consecutive keys of one bitmask word, which take one atomic.
        flag: Rows to write [b, n] int32 - every row if None.
        scratch: Memory for the per-row buffers, as many rows a pass as it holds.

    Raises:
        ValueError: bf16x3 is asked of index queries that are not bf16.
    """
    centroids, members = buckets
    b, n = index_queries.shape[:2]
    heads = index_queries.shape[2] if index_queries.dim() == 4 else 1
    d = index_queries.shape[-1]
    m, cap = members.shape[1:]
    q = index_queries.reshape(b, n, heads * d)
    if precision == "auto":
        precision = "bf16x3" if q.dtype == torch.bfloat16 else "tf32x3"
    if precision == "bf16x3" and q.dtype != torch.bfloat16:
        raise ValueError(f"bf16x3 scores bf16 index queries, got {q.dtype}")
    if q.stride(-1) != 1:
        q = q.contiguous()
    centroids = split3(centroids) if precision == "bf16x3" else centroids.float().contiguous()
    # Positions fit int32, which halves the gathers' bytes
    members = members.to(torch.int32).contiguous()
    q_pos = q_pos.to(device=q.device, dtype=torch.int64).contiguous()
    pick_count = min(top_p, m)
    m_pad = triton.next_power_of_2(m)
    p_pad = triton.next_power_of_2(pick_count)
    search = m_pad > SORT_BUCKETS
    key_width = 2 * p_pad if search else 1
    if flag is None:
        rows_at_once = max(1, ROW_ELEMENTS // (b * m))
        chunk = min(n, rows_at_once)
        scores = torch.empty(b, chunk, m, device=q.device)
        keys = torch.empty(b, chunk, key_width, dtype=torch.long, device=q.device)
        order = torch.empty(b, chunk, p_pad, dtype=torch.int32, device=q.device)
    else:
        # A row's scores, keys and order, in scratch where it holds them
        row_bytes = 4 * m + 8 * key_width + 4 * p_pad
        room = scratch.numel() * scratch.element_size() if scratch is not None else 0
        held = room // (b * row_bytes)
        chunk = min(n, held if held >= SCRATCH_ROWS else max(1, ROW_ELEMENTS // (b * m)))
        if scratch is not None and chunk <= held:
            raw = scratch.view(-1).view(torch.uint8)
            sizes = (4 * m * b * chunk, 8 * key_width * b * chunk, 4 * p_pad * b * chunk)
            scores, keys, order = (
                raw[sum(sizes[:i]) : sum(sizes[: i + 1])].view(dt).view(b, chunk, -1)
                for i, dt in enumerate((torch.float32, torch.long, torch.int32))
            )
        else:
            scores = torch.empty(b, chunk, m, device=q.device)
            keys = torch.empty(b, chunk, key_width, dtype=torch.long, device=q.device)
            order = torch.empty(b, chunk, p_pad, dtype=torch.int32, device=q.device)
        cum = flag.cumsum(1, dtype=torch.int32)
        listed = torch.empty(b, n + 1, dtype=torch.int32, device=q.device)
        _list_kernel[(triton.cdiv(n, 1024), b)](
            cum, listed, n, cum.stride(0), listed.stride(0), BLOCK=1024, num_warps=4
        )  # fmt: skip
    # Any tile gives a row the same scores: its dots sum in the same order
    wide = heads == 8 and d <= 128 and precision == "bf16x3"
    bm = (HEAD_TILE_ROWS if wide else TILE_ROWS) // heads
    rows, warps = SELECT_LAUNCH.get(m_pad, (1, 1) if m_pad < 512 else (1, 16))
    bits, occupancy = (q_pos, q_pos) if selection is None else selection
    out = q_pos if ids is None else ids
    c_split = centroids[0].numel() if precision == "bf16x3" else 0
    common = dict(BLOCK=block, PER_BLOCK=block // bucket_size)
    shape = dict(H=heads, D=d, BM=bm, BN=TILE_BUCKETS, PREC=PREC[precision])
    shape |= dict(num_warps=4 if heads == 1 else 8, num_stages=3 if precision == "bf16x3" else 1)
    whole = m_pad <= WHOLE_BUCKETS
    expand = NARROW_EXPAND if m_pad <= SORT_BUCKETS else EXPAND
    picking = dict(
        TOP_K=top_k, WINDOW=window_size, SINK=sink_size, CAP=cap, GROUP=group, R=rows, M_PAD=m_pad,
        P_PAD=p_pad, K_PAD=triton.next_power_of_2(top_k), EXPAND=min(expand, p_pad),
        WRITE_IDS=ids is not None, WRITE_BITS=selection is not None, SEARCH=search,
        BISECTIONS=BISECTIONS, num_warps=warps,
        **{f'N{i + 1}': min(width, m_pad) * whole for i, width in enumerate(WHOLE_WIDTHS)},
    )  # fmt: skip
    strides = (
        scores.stride(0), scores.stride(1), members.stride(0),
        out.stride(0) if ids is not None else 0, out.stride(1) if ids is not None else 0,
        bits.stride(0) if selection is not None else 0,
        bits.stride(2) if selection is not None else 0,
        occupancy.stride(0) if selection is not None else 0,
        occupancy.stride(2) if selection is not None else 0,
    )  # fmt: skip
    if flag is not None:
        # Every pass is launched and loops over the listed rows it has, none past the count
        # So nothing waits for the host
        sms = _sm_count(q.device)
        for s in range(0, n, chunk):
            _scores_kernel[(min(sms, triton.cdiv(chunk, bm)), b)](
                q, centroids, q_pos, scores, listed, chunk, m, c_split, s, n,
                q.stride(0), q.stride(1), centroids.stride(-3), scores.stride(0),
                scores.stride(1), listed.stride(0), **common, **shape, LISTED=True,
            )  # fmt: skip
            _select_kernel[(min(sms * 16 // warps, chunk), b)](
                scores, members, q_pos, out, bits, occupancy, keys, order, listed, 0, chunk, m,
                pick_count, s, n, *strides, listed.stride(0), **common, **picking, LISTED=True,
            )  # fmt: skip
        return
    for s in range(0, n, rows_at_once):
        e = min(n, s + rows_at_once)
        _scores_kernel[(triton.cdiv(e - s, bm), b)](
            q[:, s:], centroids, q_pos[s:], scores, q_pos, e - s, m, c_split, 0, 0,
            q.stride(0), q.stride(1), centroids.stride(-3), scores.stride(0), scores.stride(1), 0,
            **common, **shape, LISTED=False,
        )  # fmt: skip
        _select_kernel[(triton.cdiv(e - s, rows), b)](
            scores, members, q_pos, out[:, s:] if ids is not None else out, bits, occupancy,
            keys, order, q_pos, s, e - s, m, pick_count, 0, 0, *strides, 0, **common, **picking,
            LISTED=False,
        )  # fmt: skip


_SMS: dict[torch.device, int] = {}


def _sm_count(device: torch.device) -> int:
    """Return the streaming multiprocessors of the device, cached."""
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


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
    precision: str = "auto",
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Select as reference.select_indices does, by the kernels where supported takes the shapes, by the reference otherwise.

    Args:
        precision: The scores' dot, see PRECISIONS.
        dtype: The positions' dtype, int64 or int32 (what cuda_sparse's kernels read).
        Others: reference.select_indices'.
    """
    if buckets is None or not supported(index_queries, buckets):
        return reference.select_indices(
            index_queries, q_pos, buckets, block, bucket_size, top_p, top_k, window_size, sink_size
        ).to(dtype)  # fmt: skip
    b, n = index_queries.shape[:2]
    ids = torch.empty(b, n, top_k, dtype=dtype, device=index_queries.device)
    _select(
        index_queries, q_pos, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, ids, None,
        precision, 1,
    )  # fmt: skip
    return ids


def redo_flagged(
    index_queries: torch.Tensor,
    flag: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor],
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    ids: torch.Tensor,
    scratch: torch.Tensor | None = None,
) -> None:
    """Write select_indices' positions of a whole sequence into the rows of ids that flag names.

    The flagged rows are listed on the device and the kernels loop over the list, so nothing waits for the host.
    A row's positions are those select_indices gives it alone.

    Args:
        index_queries: Index queries [b, t, d] or [b, t, h, d], query i at position i, which supported must take.
        flag: Rows to write [b, t] int32, 1 for the rows to write and 0 elsewhere.
        ids: Positions [b, t, top_k] int32 or int64, written in place.
        scratch: Memory the kernels may use for their per-row buffers, as many rows a pass as it holds.
        Others: select_indices'.
    """
    positions = torch.arange(index_queries.shape[1], device=index_queries.device)
    _select(
        index_queries, positions, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, ids, None,
        'auto', 1, flag=flag, scratch=scratch,
    )  # fmt: skip


def select_bits(
    index_queries: torch.Tensor,
    buckets: tuple[torch.Tensor, torch.Tensor] | None,
    block: int,
    bucket_size: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    pool: int = 1,
    with_indices: bool = False,
    precision: str = "auto",
) -> tuple[Selection, torch.Tensor | None]:
    """Select a whole sequence as a one-head dsa_sparse Selection, for dsa_sparse.attention.

    Query i sits at position i and the keys are the queries' own t positions.
    Without buckets the selection is empty.

    Args:
        index_queries: Index queries [b, t, d] or [b, t, h, d], which supported must take.
        pool: The pool size cluster made the buckets with, whose pools of consecutive keys set one word's bits with one atomic: 1, or a power of two dividing the cap and 32.
        with_indices: Also return select_indices' [b, t, top_k] positions.
        precision: The scores' dot, see PRECISIONS.
        Others: reference.select_indices'.

    Returns:
        The Selection, bits [b, 1, t, words_per_row(t)] int32 and occupancy [b, 1, n, n] int32, and the positions, or None.
    """
    b, t = index_queries.shape[:2]
    words, tiles = words_per_row(t), occupancy_size(t)
    bits = torch.zeros(b, 1, t, words, dtype=torch.int32, device=index_queries.device)
    occupancy = torch.zeros(b, 1, tiles, tiles, dtype=torch.int32, device=index_queries.device)
    selection = Selection(bits, occupancy)
    if buckets is None:
        ids: torch.Tensor | None = torch.full(
            (b, t, top_k), reference.PAD, dtype=torch.long, device=index_queries.device
        )
        return selection, ids if with_indices else None
    if not supported(index_queries, buckets):
        raise ValueError("select_bits takes only the shapes lsa_select.supported allows")
    cap = buckets[1].shape[2]
    if pool & (pool - 1) or cap % pool or 32 % pool:
        raise ValueError(f"a pool of {pool} keys does not tile the cap {cap} and a 32-bit word")
    ids = torch.empty(b, t, top_k, dtype=torch.long, device=index_queries.device) if with_indices else None
    positions = torch.arange(t, device=index_queries.device)
    _select(
        index_queries, positions, buckets, block, bucket_size, top_p, top_k, window_size, sink_size, ids,
        selection, precision, pool,
    )  # fmt: skip
    return selection, ids
