"""PISA's pyramid top-K block selection (Tang et al., arXiv 2609.31093, section 3.2, appendix B).

Keys fall in leaf blocks of `block` consecutive keys, and every `branching` adjacent blocks of
one level make a block of the next, up to a single root: O(log N) levels. A block's summary is
the mean of its children's summaries, a leaf's the mean of its keys. A query picks its blocks
coarse to fine: at each level it scores its candidates and keeps the top_k, and the children of
those are the next level's candidates; a level with no more than top_k candidates keeps them all.
A leaf scores the exact log-sum-exp of the query's scaled logits over its own keys, a coarser
block the log-sum-exp over its children's summaries.

The query heads that share a key head share its selection, ranked by their scores summed. The
first leaf block and the `local_blocks` newest ones (the current block and the ones before it; 2,
the previous and current, by default) and their ancestors are kept whenever they exist, inside the
top_k: the current one is the only block holding keys after the query, so it is never ranked. A
block starting after the query is never a candidate. The selection has no gradient.

The paper's kernels score only the candidates. Here every block of every level is scored and the
pyramid then only masks: the same selection at O(N^2) cost rather than O(N log N). On CUDA the
leaves' scores come from one Triton kernel that never writes a logit; elsewhere from a dense q.k a
row block at a time.

Positions follow ops/nsa/reference.py: the queries are the last n_q of the n_k keys' positions,
query row r at q_offset + r, q_offset = n_k - n_q (0 in training, the cache length decoding).
select returns a (b, h_kv, n_q, M) bool table of leaves; select_blocks the same selection as the
block indices ops/nsa's selected read takes, int32 (b, n_q, h_kv, top_k), ascending, -1 padded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Leaf logits, (b, heads, rows, keys) in fp32, that one row block holds at once
_LEAF_ELEMENTS = 1 << 28
# Coarser levels' logits, (b, heads, rows, blocks) in fp32, that one row block holds at once
_LEVEL_ELEMENTS = 1 << 26
# Query rows of a leaf kernel program; times the group, the rows of its dot
_LEAF_ROWS = 16


def level_sizes(n_blocks: int, branching: int) -> list[int]:
    """The blocks of each level, leaves first, down to the single root."""
    sizes = [n_blocks]
    while sizes[-1] > 1:
        sizes.append(-(-sizes[-1] // branching))
    return sizes


def _pool(x: torch.Tensor, size: int, width: int) -> torch.Tensor:
    """Means of runs of width along dim 2 of x, (b, h, n, d), counting only real entries."""
    b, h, n, d = x.shape
    total = F.pad(x, (0, 0, 0, size * width - n)).view(b, h, size, width, d).sum(3)
    count = F.pad(x.new_ones(n), (0, size * width - n)).view(size, width).sum(-1)
    return total / count[:, None]


def summaries(k: torch.Tensor, block: int, branching: int) -> list[torch.Tensor]:
    """The summaries each scored level reads, (b, h_kv, M_l, d) fp32, leaves first.

    Args:
        k: (b, n, h_kv, d) keys at positions 0..n-1.
        block, branching: See the module docstring.
    """
    sizes = level_sizes(-(-k.shape[1] // block), branching)
    levels = [_pool(k.float().transpose(1, 2), sizes[0], block)]
    # The level below the root only ever scores the root, which is never ranked
    for size in sizes[1:-2]:
        levels.append(_pool(levels[-1], size, branching))
    return levels


def _lse_groups(logits: torch.Tensor, size: int, width: int) -> torch.Tensor:
    """Log-sum-exp of runs of width along the last dim, -inf past its end, (..., size)."""
    padded = F.pad(logits, (0, size * width - logits.shape[-1]), value=float("-inf"))
    return padded.unflatten(-1, (size, width)).logsumexp(-1)


@triton.jit
def _leaf_kernel(
    q: tl.tensor,
    k: tl.tensor,
    out: tl.tensor,
    scale: tl.tensor,
    n_q: tl.tensor,
    n_k: tl.tensor,
    n_blocks: tl.tensor,
    q_start: tl.tensor,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
) -> None:
    """out[b, h, i, j]: query i's log-sum-exp over leaf j, summed over heads, j before its own."""
    i_r, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H_KV, i_bh % H_KV
    # Row r of the dot is query r // G of the program, head i_h * G + r % G
    o_r = tl.arange(0, ROWS * G)
    row = i_r * ROWS + o_r // G
    o_d = tl.arange(0, D)
    head = i_h * G + o_r[:, None] % G
    p_q = q + ((i_b * n_q + row[:, None]) * H_KV * G + head) * D + o_d[None, :]
    b_q = tl.load(p_q, mask=row[:, None] < n_q, other=0.0)
    o_q = i_r * ROWS + tl.arange(0, ROWS)
    current = (q_start + o_q) // BLOCK
    last = (q_start + tl.minimum(i_r * ROWS + ROWS, n_q) - 1) // BLOCK
    o_s = tl.arange(0, BLOCK)
    p_out = out + (i_bh * n_q + o_q) * n_blocks
    for j in range(0, last):
        o_k = j * BLOCK + o_s
        p_k = k + ((i_b * n_k + o_k[None, :]) * H_KV + i_h) * D + o_d[:, None]
        b_k = tl.load(p_k, mask=o_k[None, :] < n_k, other=0.0)
        b_s = tl.dot(b_q, b_k) * scale
        b_m = tl.max(b_s, 1)
        b_lse = b_m + tl.log(tl.sum(tl.exp(b_s - b_m[:, None]), 1))
        b_u = tl.sum(tl.reshape(b_lse, (ROWS, G)), 1)
        tl.store(p_out + j, b_u, mask=(j < current) & (o_q < n_q))


def _leaf_scores_kernel(
    q: torch.Tensor, k: torch.Tensor, q_offset: int, block: int, scale: float
) -> torch.Tensor:
    """Score every completed leaf block with the Triton kernel."""
    b, n_q, h, d = q.shape
    n_k, h_kv = k.shape[1], k.shape[2]
    n_blocks = -(-n_k // block)
    out = torch.full((b, h_kv, n_q, n_blocks), float("-inf"), device=q.device)
    grid = (triton.cdiv(n_q, _LEAF_ROWS), b * h_kv)
    _leaf_kernel[grid](
        q.contiguous(),
        k.contiguous(),
        out,
        scale,
        n_q,
        n_k,
        n_blocks,
        q_offset,
        H_KV=h_kv,
        G=h // h_kv,
        D=d,
        BLOCK=block,
        ROWS=_LEAF_ROWS,
    )
    return out


def leaf_scores(
    q: torch.Tensor, k: torch.Tensor, block: int, scale: float, q_offset: int = 0
) -> torch.Tensor:
    """(b, h_kv, n_q, M) fp32: each query's summed log-sum-exp over every leaf before its own.

    Leaves from the query's own on score -inf. Args as select's.
    """
    b, n_q, h, d = q.shape
    n_k, h_kv = k.shape[1], k.shape[2]
    n_blocks = -(-n_k // block)
    group = h // h_kv
    kernel_ok = (
        d in (16, 32, 64, 128, 256) and group & (group - 1) == 0 and block >= 16 and block & (block - 1) == 0
    )
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and kernel_ok:
        return _leaf_scores_kernel(q, k.to(q.dtype), q_offset, block, scale)
    qg = q.unflatten(2, (h_kv, group)).permute(0, 2, 3, 1, 4)
    kh = k.transpose(1, 2)
    out = torch.full((b, h_kv, n_q, n_blocks), float("-inf"), device=q.device)
    step = max(1, _LEAF_ELEMENTS // (b * h * n_k))
    for s in range(0, n_q, step):
        pos = q_offset + torch.arange(s, min(s + step, n_q), device=q.device)
        n_vis = min(n_k, ((q_offset + min(s + step, n_q) - 1) // block) * block)
        logits = torch.einsum("bkgrd,bknd->bkgrn", qg[:, :, :, s : s + step], kh[:, :, :n_vis])
        leaf = _lse_groups(logits.float() * scale, n_vis // block, block).sum(2)
        before = torch.arange(n_vis // block, device=q.device) < (pos // block)[:, None]
        out[:, :, s : s + step, : n_vis // block] = leaf.masked_fill(~before, float("-inf"))
    return out


def select(
    q: torch.Tensor,
    k: torch.Tensor,
    block: int,
    branching: int,
    top_k: int,
    scale: float,
    q_offset: int = 0,
    local_blocks: int = 2,
) -> torch.Tensor:
    """Each query's selected leaf blocks, per key head, as a (b, h_kv, n_q, M) bool table.

    Args:
        q: (b, n_q, h, d) queries, after RoPE.
        k: (b, n_k, h_kv, d) keys at positions 0..n_k-1, after RoPE; h_kv divides h.
        block, branching, top_k, local_blocks: See the module docstring.
        scale: The attention's scale on q.k.
        q_offset: The first query's position, n_k - n_q for the last n_q positions.
    """
    b, n_q, h, _ = q.shape
    n_k, h_kv = k.shape[1], k.shape[2]
    sizes = level_sizes(-(-n_k // block), branching)
    pos = q_offset + torch.arange(n_q, device=q.device)
    with torch.no_grad():
        q, k = q.detach(), k.detach()
        leaf = leaf_scores(q, k, block, scale, q_offset)
        qg = q.float().unflatten(2, (h_kv, h // h_kv)).permute(0, 2, 3, 1, 4)
        levels = summaries(k, block, branching)
        selected = torch.empty(b, h_kv, n_q, sizes[0], dtype=torch.bool, device=q.device)
        step = max(1, _LEVEL_ELEMENTS // (b * h * sizes[0]))
        for s in range(0, n_q, step):
            rows = slice(s, s + step)
            scores = [leaf[:, :, rows]]
            for i, size in enumerate(sizes[1:-1]):
                child = torch.einsum("bkgrd,bkmd->bkgrm", qg[:, :, :, rows], levels[i]) * scale
                scores.append(_lse_groups(child, size, branching).sum(2))
            selected[:, :, rows] = _descend(scores, pos[rows], block, branching, top_k, sizes, local_blocks)
        return selected


def _descend(
    scores: list[torch.Tensor],
    pos: torch.Tensor,
    block: int,
    branching: int,
    top_k: int,
    sizes: list[int],
    local_blocks: int,
) -> torch.Tensor:
    """The coarse-to-fine top_k over each level's scores, (b, h_kv, rows, M) bool."""
    b, h_kv, rows = scores[0].shape[:3]
    current = pos // block
    newest = [current - back for back in range(local_blocks)]
    forced_leaves = torch.stack([torch.zeros_like(current), *newest], dim=-1)
    candidates = torch.ones(b, h_kv, rows, 1, dtype=torch.bool, device=pos.device)
    for level in range(len(sizes) - 1, -1, -1):
        size, span = sizes[level], block * branching**level
        index = torch.arange(size, device=pos.device)
        eligible = index[None, :] * span <= pos[:, None]
        ancestors = forced_leaves // branching**level
        forced = ((index == ancestors[..., None]) & (forced_leaves >= 0)[..., None]).any(1)
        live = candidates & eligible
        # The root holds every key and is never ranked
        root = level == len(scores)
        score = torch.zeros_like(live, dtype=torch.float32) if root else scores[level]
        score = score.masked_fill(~live, float("-inf")).masked_fill(live & forced, float("inf"))
        top = score.topk(min(top_k, size), dim=-1)
        kept = torch.zeros_like(live).scatter_(-1, top.indices, top.values > float("-inf"))
        if level == 0:
            return kept
        children = kept.repeat_interleave(branching, dim=-1)
        candidates = children[..., : sizes[level - 1]]
    raise RuntimeError("PISA pyramid has no leaf level")


def block_indices(selected: torch.Tensor, top_k: int) -> torch.Tensor:
    """The (b, h_kv, n_q, M) table of select as int32 (b, n_q, h_kv, top_k), ascending, -1 pads."""
    m = selected.shape[-1]
    ids = torch.arange(m, device=selected.device, dtype=torch.int32)
    # Unselected leaves sort last as m, then become padding
    ranked = torch.where(selected, ids, m).sort(-1).values[..., :top_k]
    ranked = torch.where(ranked == m, -1, ranked)
    if ranked.shape[-1] < top_k:
        ranked = F.pad(ranked, (0, top_k - ranked.shape[-1]), value=-1)
    return ranked.transpose(1, 2).contiguous()


@torch.library.custom_op("lsa::pisa_select", mutates_args=())
def select_op(
    q: torch.Tensor,
    k: torch.Tensor,
    block: int,
    branching: int,
    top_k: int,
    scale: float,
    q_offset: int,
    local_blocks: int,
) -> torch.Tensor:
    """block_indices of select, one opaque op so torch.compile leaves its loops eager."""
    selected = select(q, k, block, branching, top_k, scale, q_offset, local_blocks)
    return block_indices(selected, top_k)


@select_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    block: int,
    branching: int,
    top_k: int,
    scale: float,
    q_offset: int,
    local_blocks: int,
) -> torch.Tensor:
    """Describe the selected block tensor without reading data."""
    return q.new_empty((q.shape[0], q.shape[1], k.shape[2], top_k), dtype=torch.int32)


def select_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    block: int,
    branching: int,
    top_k: int,
    scale: float,
    q_offset: int = 0,
    local_blocks: int = 2,
) -> torch.Tensor:
    """Each key head's selected leaves as ops/nsa block indices; no gradient. Args as select's."""
    return select_op(q.detach(), k.detach(), block, branching, top_k, scale, q_offset, local_blocks)
