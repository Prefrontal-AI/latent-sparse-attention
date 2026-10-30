"""NSA's forward kernels: the compressed read, block selection and the selected read.

Each function matches its namesake in reference.py, whose docstring states the conventions.

The kernels are adapted from fla's NSA kernels (flash-linear-attention, fla/ops/nsa, MIT licensed, Songlin Yang,
Yu Zhang et al.), generalised to overlapping compression (compress_block_size, compress_stride), equation 9's selection
scores and any group size in 1, 2, 4, 8, 16 and 32.

- compressed_attention_fwd tiles BT consecutive queries times their group's heads into one BT * group_size row tile,
  so every group size fills a tensor-core tile without copying heads, and the visible compressed keys are read once per
  tile rather than once per query.
- select_blocks uses the same kind of query tile, sums the compressed probabilities over the group's heads, maps them to
  selection blocks with a small banded matrix (equation 9) and keeps a running top-k per query on unique int64 keys
  (score bits over the inverted block index, so ties go to the earlier block) with Triton's bitonic topk.
- selected_attention_fwd runs one query per program, its group's heads padded to max(16, group_size) rows (the padding
  rows load zeros and are never stored), with one online softmax across all selected blocks.

Non-contiguous inputs are made contiguous.
Head dims must be powers of two in [16, 256] (64 and 128 are tested), select_block_size a power of two of at least 16,
and compress_stride must divide compress_block_size and select_block_size.
"""

import math

import torch
import triton
import triton.language as tl

_LOG2E = math.log2(math.e)


@triton.autotune(
    configs=[
        triton.Config({"BM": bm, "BC": bc}, num_warps=w, num_stages=s)
        for bm in [64, 128]
        for bc in [32, 64]
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["G", "D", "DV", "COMPRESS_BLOCK_SIZE", "COMPRESS_STRIDE"],
)
@triton.jit
def _compressed_fwd_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    o: tl.tensor,
    lse: tl.tensor,
    scale_log2: tl.tensor,
    query_count: tl.tensor,
    compressed_key_count: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    COMPRESS_BLOCK_SIZE: tl.constexpr,
    COMPRESS_STRIDE: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
) -> None:
    # Row r of the tile is query t0 + r // G, head r % G of the group
    """Compute the compressed fwd kernel operation."""
    BT: tl.constexpr = BM // G
    HQ: tl.constexpr = H * G
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    t0 = i_t * BT

    rows = tl.arange(0, BM)
    tt = t0 + rows // G
    gg = rows % G
    row_ok = tt < query_count
    pos = tt
    # Visible compressed tokens per row, max(0, (pos + 1 - COMPRESS_BLOCK_SIZE) // COMPRESS_STRIDE + 1)
    nc = tl.where(
        pos + 1 >= COMPRESS_BLOCK_SIZE,
        (pos + 1 - COMPRESS_BLOCK_SIZE + COMPRESS_STRIDE) // COMPRESS_STRIDE,
        0,
    )
    nc = tl.minimum(nc, compressed_key_count)

    o_d = tl.arange(0, D)
    o_v = tl.arange(0, DV)
    q_row = (i_b.to(tl.int64) * query_count + tt) * HQ + i_h * G + gg
    b_q = tl.load(q + q_row[:, None] * D + o_d[None, :], mask=row_ok[:, None], other=0.0)

    k_base = k + i_b.to(tl.int64) * compressed_key_count * H * D + i_h * D
    v_base = v + i_b.to(tl.int64) * compressed_key_count * H * DV + i_h * DV

    # The first query sees the fewest tokens, the last valid one the most
    pos_lo = t0
    pos_hi = tl.minimum(t0 + BT, query_count) - 1
    nc_lo = tl.where(
        pos_lo + 1 >= COMPRESS_BLOCK_SIZE,
        (pos_lo + 1 - COMPRESS_BLOCK_SIZE + COMPRESS_STRIDE) // COMPRESS_STRIDE,
        0,
    )
    nc_hi = tl.where(
        pos_hi + 1 >= COMPRESS_BLOCK_SIZE,
        (pos_hi + 1 - COMPRESS_BLOCK_SIZE + COMPRESS_STRIDE) // COMPRESS_STRIDE,
        0,
    )
    nc_lo = tl.minimum(nc_lo, compressed_key_count)
    nc_hi = tl.minimum(nc_hi, compressed_key_count)
    c_split = (nc_lo // BC) * BC

    b_m = tl.full([BM], float("-inf"), dtype=tl.float32)
    b_l = tl.zeros([BM], dtype=tl.float32)
    b_o = tl.zeros([BM, DV], dtype=tl.float32)

    # Chunks every row sees in full need no mask
    for c0 in range(0, c_split, BC):
        o_c = c0 + tl.arange(0, BC)
        b_k = tl.load(k_base + o_c[None, :] * (H * D) + o_d[:, None])
        b_v = tl.load(v_base + o_c[:, None] * (H * DV) + o_v[None, :])
        b_s = tl.dot(b_q, b_k) * scale_log2
        b_mn = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.math.exp2(b_m - b_mn)
        b_p = tl.math.exp2(b_s - b_mn[:, None])
        b_l = b_l * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_mn

    # The tail is masked per row, and a row may still have seen nothing
    for c0 in range(c_split, nc_hi, BC):
        o_c = c0 + tl.arange(0, BC)
        m_c = o_c < nc_hi
        b_k = tl.load(k_base + o_c[None, :] * (H * D) + o_d[:, None], mask=m_c[None, :], other=0.0)
        b_v = tl.load(v_base + o_c[:, None] * (H * DV) + o_v[None, :], mask=m_c[:, None], other=0.0)
        b_s = tl.dot(b_q, b_k) * scale_log2
        b_s = tl.where(o_c[None, :] < nc[:, None], b_s, float("-inf"))
        b_mn = tl.maximum(b_m, tl.max(b_s, 1))
        b_mu = tl.where(b_mn == float("-inf"), 0.0, b_mn)
        b_r = tl.math.exp2(b_m - b_mu)
        b_p = tl.math.exp2(b_s - b_mu[:, None])
        b_l = b_l * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_mn

    # A row with no visible token reads 0 with lse 0
    seen = b_l > 0
    b_l = tl.where(seen, b_l, 1.0)
    b_o = b_o / b_l[:, None]
    b_lse = tl.where(seen, b_m * 0.6931471805599453 + tl.log(b_l), 0.0)
    tl.store(o + q_row[:, None] * DV + o_v[None, :], b_o.to(o.dtype.element_ty), mask=row_ok[:, None])
    tl.store(lse + q_row, b_lse, mask=row_ok)


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=w, num_stages=s)
        for bt in [16, 32]
        for w in [4, 8]
        for s in [1, 2, 3]
    ],
    key=["G", "D", "COMPRESS_BLOCK_SIZE", "COMPRESS_STRIDE", "SELECT_BLOCK_SIZE", "S"],
)
@triton.jit
def _select_kernel(
    q: tl.tensor,
    k: tl.tensor,
    lse: tl.tensor,
    out: tl.tensor,
    scale_log2: tl.tensor,
    query_count: tl.tensor,
    compressed_key_count: tl.tensor,
    n_blocks: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    COMPRESS_BLOCK_SIZE: tl.constexpr,
    COMPRESS_STRIDE: tl.constexpr,
    SELECT_BLOCK_SIZE: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
) -> None:
    # Each chunk scores W selection blocks from the BC compressed tokens from their first one
    """Compute the select kernel operation."""
    PER_SLC: tl.constexpr = SELECT_BLOCK_SIZE // COMPRESS_STRIDE
    PER_CMP: tl.constexpr = COMPRESS_BLOCK_SIZE // COMPRESS_STRIDE
    BC: tl.constexpr = W * PER_SLC
    NPASS: tl.constexpr = G // HG
    HQ: tl.constexpr = H * G
    LOG2E: tl.constexpr = 1.4426950408889634
    BIG: tl.constexpr = 0x7FFFFFFF
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    t0 = i_t * BT

    # Rows of a pass are query t0 + r // HG, head i_p * HG + r % HG
    rows = tl.arange(0, BT * HG)
    tt = t0 + rows // HG
    gg = rows % HG
    row_ok = tt < query_count
    pos = tt
    nc = tl.where(
        pos + 1 >= COMPRESS_BLOCK_SIZE,
        (pos + 1 - COMPRESS_BLOCK_SIZE + COMPRESS_STRIDE) // COMPRESS_STRIDE,
        0,
    )
    nc = tl.minimum(nc, compressed_key_count)
    o_d = tl.arange(0, D)
    q_row = (i_b.to(tl.int64) * query_count + tt) * HQ + i_h * G + gg
    b_q0 = tl.load(q + q_row[:, None] * D + o_d[None, :], mask=row_ok[:, None], other=0.0)
    b_lse0 = tl.load(lse + q_row, mask=row_ok, other=0.0) * LOG2E
    k_base = k + i_b.to(tl.int64) * compressed_key_count * H * D + i_h * D

    # Per query, its own block cur; the candidates are blocks 0..cur
    o_bt = t0 + tl.arange(0, BT)
    bt_ok = o_bt < query_count
    cur = o_bt // SELECT_BLOCK_SIZE
    cur_hi = (tl.minimum(t0 + BT, query_count) - 1) // SELECT_BLOCK_SIZE

    # Equation 9 as a matrix: token c0 + c weighs into block j0 + jj by the count of (m, n)
    # With PER_SLC * jj + m - n = c, m < PER_SLC, n < PER_CMP; columns W.. feed the next chunk
    o_cc = tl.arange(0, BC)[:, None]
    o_jj = tl.arange(0, 2 * W)[None, :]
    rel = o_cc - PER_SLC * o_jj
    w_lo = tl.maximum(rel, 0)
    w_hi = tl.minimum(rel + PER_CMP - 1, PER_SLC - 1)
    b_w = tl.maximum(w_hi - w_lo + 1, 0).to(tl.float32)

    b_carry = tl.zeros([BT, W], dtype=tl.float32)
    b_top = tl.full([BT, W], -1, dtype=tl.int64)
    o_w = tl.arange(0, W)
    for j0 in range(0, cur_hi + 1, W):
        o_c = j0 * PER_SLC + tl.arange(0, BC)
        b_k = tl.load(
            k_base + o_c[None, :] * (H * D) + o_d[:, None],
            mask=(o_c < compressed_key_count)[None, :],
            other=0.0,
        )
        vis = o_c[None, :] < nc[:, None]
        b_ps = tl.zeros([BT, BC], dtype=tl.float32)
        for i_p in tl.static_range(NPASS):
            if i_p == 0:
                b_q = b_q0
                b_lse = b_lse0
            else:
                q_row_p = q_row + i_p * HG
                b_q = tl.load(q + q_row_p[:, None] * D + o_d[None, :], mask=row_ok[:, None], other=0.0)
                b_lse = tl.load(lse + q_row_p, mask=row_ok, other=0.0) * LOG2E
            b_s = tl.dot(b_q, b_k) * scale_log2
            b_p = tl.where(vis, tl.math.exp2(b_s - b_lse[:, None]), 0.0)
            b_ps += tl.sum(tl.reshape(b_p, [BT, HG, BC]), 1)
        # Fp32 products keep the block scores as exact as the reference's
        b_sw = tl.dot(b_ps, b_w, input_precision="ieee")
        b_here, b_next = tl.split(tl.permute(tl.reshape(b_sw, [BT, 2, W]), (0, 2, 1)))
        b_score = b_here + b_carry
        b_carry = b_next

        blk = j0 + o_w
        cand = (blk[None, :] <= cur[:, None]) & (blk[None, :] < n_blocks)
        forced = (blk[None, :] == 0) | (blk[None, :] == cur[:, None])
        forced = forced | (blk[None, :] == cur[:, None] - 1)
        b_score = tl.where(forced, float("inf"), b_score)
        # Scores are >= 0, so their bits order like the floats; the low half breaks ties
        bits = b_score.to(tl.int32, bitcast=True).to(tl.int64)
        key = (bits << 32) | (BIG - blk[None, :]).to(tl.int64)
        key = tl.where(cand, key, -1)
        b_min = tl.min(b_top, 1)
        if tl.sum((key > b_min[:, None]).to(tl.int32)) > 0:
            both = tl.reshape(tl.join(b_top, key), [BT, 2 * W])
            b_top = tl.topk(both, W)

    # The first S keys are the selection; list their blocks ascending with -1 padding
    idx = tl.where(b_top >= 0, BIG - b_top.to(tl.int32), BIG)
    idx = tl.where(o_w[None, :] < S, idx, BIG)
    idx = tl.sort(idx, 1)
    idx = tl.where(idx == BIG, -1, idx)
    o_row = (i_b.to(tl.int64) * query_count + o_bt) * H + i_h
    tl.store(
        out + o_row[:, None] * S + o_w[None, :],
        idx.to(tl.int32),
        mask=bt_ok[:, None] & (o_w[None, :] < S),
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [2, 4, 8] for s in [1, 2, 3]],
    key=["G", "D", "DV", "BS", "S"],
)
@triton.jit
def _selected_fwd_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    indices: tl.tensor,
    o: tl.tensor,
    lse: tl.tensor,
    scale_log2: tl.tensor,
    query_count: tl.tensor,
    key_count: tl.tensor,
    H: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BS: tl.constexpr,
    S: tl.constexpr,
) -> None:
    """Compute the selected fwd kernel operation."""
    HQ: tl.constexpr = H * G
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    pos = i_t

    # Rows G..BG pad the group to a tensor-core tile; they load zeros and are never stored
    rows = tl.arange(0, BG)
    row_ok = rows < G
    o_d = tl.arange(0, D)
    o_v = tl.arange(0, DV)
    q_row = (i_b.to(tl.int64) * query_count + i_t) * HQ + i_h * G + rows
    b_q = tl.load(q + q_row[:, None] * D + o_d[None, :], mask=row_ok[:, None], other=0.0)

    k_base = k + i_b.to(tl.int64) * key_count * H * D + i_h * D
    v_base = v + i_b.to(tl.int64) * key_count * H * DV + i_h * DV
    idx_base = indices + ((i_b.to(tl.int64) * query_count + i_t) * H + i_h) * S

    b_m = tl.full([BG], float("-inf"), dtype=tl.float32)
    b_l = tl.zeros([BG], dtype=tl.float32)
    b_o = tl.zeros([BG, DV], dtype=tl.float32)
    o_s = tl.arange(0, BS)
    for i in tl.range(S):
        blk = tl.load(idx_base + i)
        start = blk * BS
        keys = start + o_s
        # Padding (-1) and blocks after the query load nothing and score -inf
        m_k = (blk >= 0) & (keys <= pos)
        b_k = tl.load(k_base + keys[None, :] * (H * D) + o_d[:, None], mask=m_k[None, :], other=0.0)
        b_v = tl.load(v_base + keys[:, None] * (H * DV) + o_v[None, :], mask=m_k[:, None], other=0.0)
        b_s = tl.dot(b_q, b_k) * scale_log2
        b_s = tl.where(m_k[None, :], b_s, float("-inf"))
        b_mn = tl.maximum(b_m, tl.max(b_s, 1))
        b_mu = tl.where(b_mn == float("-inf"), 0.0, b_mn)
        b_r = tl.math.exp2(b_m - b_mu)
        b_p = tl.math.exp2(b_s - b_mu[:, None])
        b_l = b_l * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_mn

    # A row that read no key (not possible with select_blocks' indices) gets o 0 and lse -inf
    seen = b_l > 0
    b_l = tl.where(seen, b_l, 1.0)
    b_o = b_o / b_l[:, None]
    b_lse = tl.where(seen, b_m * 0.6931471805599453 + tl.log(b_l), float("-inf"))
    tl.store(o + q_row[:, None] * DV + o_v[None, :], b_o.to(o.dtype.element_ty), mask=row_ok[:, None])
    tl.store(lse + q_row, b_lse, mask=row_ok)


def group_size_of(query_head_count: int, kv_head_count: int) -> int:
    """Return the query heads that read one key-value head.

    Args:
        query_head_count: Query heads.
        kv_head_count: Key-value heads.

    Returns:
        The group size, a power of two up to 32.

    Raises:
        ValueError: The query heads do not split into whole groups, or a group is not a power of two up to 32.
    """
    if query_head_count % kv_head_count:
        raise ValueError(f"{query_head_count} query heads do not split into {kv_head_count} groups")
    group_size = query_head_count // kv_head_count
    if group_size > 32 or group_size & (group_size - 1):
        raise ValueError(f"group size {group_size} must be a power of two up to 32")
    return group_size


def check_head_dim(head_dim: int) -> None:
    """Refuse a head dim that is not a whole tile.

    Args:
        head_dim: A head's dim.

    Raises:
        ValueError: The head dim is not a power of two in [16, 256].
    """
    if head_dim < 16 or head_dim > 256 or head_dim & (head_dim - 1):
        raise ValueError(f"head_dim {head_dim} must be a power of two in [16, 256]")


def check_select_block_size(select_block_size: int) -> None:
    """Refuse a selection block that is not a key tile.

    Args:
        select_block_size: Keys a selection block holds.

    Raises:
        ValueError: The selection block is not a power of two of at least 16 keys.
    """
    if select_block_size < 16 or select_block_size & (select_block_size - 1):
        raise ValueError(f"select_block_size {select_block_size} must be a power of two of at least 16")


def compressed_attention_fwd(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the compressed tokens each query sees, as `reference.compressed_attention` on a Triton kernel.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        compressed_values: Compressed values [batch_size, compressed_key_count, kv_head_count, head_dim].
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    q, compressed_keys = q.contiguous(), compressed_keys.contiguous()
    compressed_values = compressed_values.contiguous()
    batch_size, query_count, query_head_count, head_dim = q.shape
    compressed_key_count, kv_head_count = compressed_keys.shape[1], compressed_keys.shape[2]
    value_dim = compressed_values.shape[-1]
    group_size = group_size_of(query_head_count, kv_head_count)
    check_head_dim(head_dim)
    check_head_dim(value_dim)
    o = torch.empty(batch_size, query_count, query_head_count, value_dim, dtype=q.dtype, device=q.device)
    lse = torch.empty(batch_size, query_count, query_head_count, dtype=torch.float32, device=q.device)
    if compressed_key_count == 0 or query_count == 0:
        return o.zero_(), lse.zero_()

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        """Compute the grid operation."""
        return (triton.cdiv(query_count, meta["BM"] // group_size), batch_size * kv_head_count)

    _compressed_fwd_kernel[grid](
        q,
        compressed_keys,
        compressed_values,
        o,
        lse,
        scale * _LOG2E,
        query_count,
        compressed_key_count,
        H=kv_head_count,
        G=group_size,
        D=head_dim,
        DV=value_dim,
        COMPRESS_BLOCK_SIZE=compress_block_size,
        COMPRESS_STRIDE=compress_stride,
    )
    return o, lse


def select_blocks(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    select_block_count: int,
    scale: float,
) -> torch.Tensor:
    """Pick the blocks each key-value head reads, as `reference.select_blocks` on a Triton kernel.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        lse: The compressed read's log-sum-exp [batch_size, query_count, query_head_count] fp32.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        select_block_size: Keys a selection block holds.
        select_block_count: Blocks a key-value head reads, the forced ones included.
        scale: The factor logits are scaled by.

    Returns:
        The blocks [batch_size, query_count, kv_head_count, select_block_count] int32, ascending, -1 padded.

    Raises:
        ValueError: compress_stride does not divide the blocks, the selection block is not a key tile, or a compression
            block spans more selection blocks than a chunk scores.
    """
    if compress_block_size % compress_stride or select_block_size % compress_stride:
        raise ValueError(
            f"compress_stride {compress_stride} must divide compress_block_size {compress_block_size} "
            f"and select_block_size {select_block_size}"
        )
    check_select_block_size(select_block_size)
    q, compressed_keys, lse = q.contiguous(), compressed_keys.contiguous(), lse.contiguous()
    batch_size, query_count, query_head_count, head_dim = q.shape
    compressed_key_count, kv_head_count = compressed_keys.shape[1], compressed_keys.shape[2]
    group_size = group_size_of(query_head_count, kv_head_count)
    check_head_dim(head_dim)
    block_count = -(-query_count // select_block_size)
    indices = torch.empty(
        batch_size, query_count, kv_head_count, select_block_count, dtype=torch.int32, device=q.device
    )
    if query_count == 0:
        return indices
    # The running top-k is chunk_blocks wide, and each chunk scores that many blocks
    chunk_blocks = max(16, triton.next_power_of_2(select_block_count))
    if compress_block_size // compress_stride > chunk_blocks * (select_block_size // compress_stride):
        raise ValueError(
            f"compress_block_size {compress_block_size} spans more than {chunk_blocks} selection blocks"
        )

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        """Compute the grid operation."""
        return (triton.cdiv(query_count, meta["BT"]), batch_size * kv_head_count)

    _select_kernel[grid](
        q,
        compressed_keys,
        lse,
        indices,
        scale * _LOG2E,
        query_count,
        compressed_key_count,
        block_count,
        H=kv_head_count,
        G=group_size,
        HG=min(group_size, 8),
        D=head_dim,
        COMPRESS_BLOCK_SIZE=compress_block_size,
        COMPRESS_STRIDE=compress_stride,
        SELECT_BLOCK_SIZE=select_block_size,
        S=select_block_count,
        W=chunk_blocks,
    )
    return indices


def selected_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the keys of the selected blocks, as `reference.selected_attention` on a Triton kernel.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        k: Keys [batch_size, key_count, kv_head_count, head_dim].
        v: Values [batch_size, key_count, kv_head_count, head_dim].
        indices: The selected blocks [batch_size, query_count, kv_head_count, select_block_count] int32, -1 unused.
        select_block_size: Keys a selection block holds.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    check_select_block_size(select_block_size)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    indices = indices.to(torch.int32).contiguous()
    batch_size, query_count, query_head_count, head_dim = q.shape
    key_count, kv_head_count, value_dim = k.shape[1], k.shape[2], v.shape[-1]
    group_size = group_size_of(query_head_count, kv_head_count)
    check_head_dim(head_dim)
    check_head_dim(value_dim)
    o = torch.empty(batch_size, query_count, query_head_count, value_dim, dtype=q.dtype, device=q.device)
    lse = torch.empty(batch_size, query_count, query_head_count, dtype=torch.float32, device=q.device)
    if query_count == 0:
        return o, lse
    _selected_fwd_kernel[(query_count, batch_size * kv_head_count)](
        q,
        k,
        v,
        indices,
        o,
        lse,
        scale * _LOG2E,
        query_count,
        key_count,
        H=kv_head_count,
        G=group_size,
        BG=max(16, group_size),
        D=head_dim,
        DV=value_dim,
        BS=select_block_size,
        S=indices.shape[-1],
    )
    return o, lse
