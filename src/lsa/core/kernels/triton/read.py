# The sparse read - causal attention restricted to each query's keep-set, tiled for the GPU.
#
# One program reads one (query row, kv head group): it walks the row's compact keep list in
# blocks, gathers the key/value rows through the ids, and runs an online softmax entirely in
# registers - two tensor-core dots per block (QK^T, PV), no 3D reductions, no atomics, and the
# only store happens after the loop. The keep list is built by [read_attention] below: selected
# long-range ids, then the local window, then the sinks, causality applied at build time so the
# kernel masks nothing but padding (-1).
#
# The list assumes the selected ids never collide with the window or the sinks (the indexer
# selects far buckets, the window and sinks are appended around them, and window ids under the
# sink count are dropped at build time since the sink list already covers them) - a selected
# collision would be counted twice, where the torch reference
# [lsa.core.attention.topk_attention] scatters a boolean mask and dedupes.
#
# Ascend notes: dot inside a plain range loop with a loop-carried accumulator is the one form
# the triton-ascend compiler had not crashed on when this was written - the crash forms were
# dot plus atomic, and stores of dot-derived tensors from inside the loop; here the store is
# after the loop and the accumulated tile is two-dimensional. fp32 dots are not supported on
# the NPU, so QK^T runs on the bf16 inputs and PV on bf16 probabilities (the flash form), and
# both accumulate in fp32.

import torch
import triton
import triton.language as tl

# Keep ids a program reads at a time, and the head dimension a head group is padded to for the dot.
# Wider head dims blow the unified buffer through the gather tiles' fp32 conversions, so the id
# block shrinks as the head dim grows.
BLOCK_C = 128
PAD_HEADS = 16
WARPS = 4


def block_c_for(dim: int) -> int:
    if dim <= 64:
        return 128
    if dim <= 128:
        return 64
    return 32


@triton.jit
def _read_tile_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    ids_ptr,
    out_ptr,
    scale,
    total_q,
    total_k,
    keep_length,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    HG: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    PQ: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // KVH
    g = pid % KVH
    b = row // total_q
    t = row % total_q
    d = tl.arange(0, DIM)
    h = tl.arange(0, PQ)
    hlive = h < HG

    q_tile = tl.load(
        q_ptr + ((b * total_q + t) * QH + g * HG + h[:, None]) * DIM + d[None, :],
        mask=hlive[:, None],
        other=0.0,
    )

    m = tl.full([PQ], -1e30, tl.float32)
    l = tl.zeros([PQ], tl.float32)
    acc = tl.zeros([PQ, DIM], tl.float32)
    for start in range(0, keep_length, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        live = offs < keep_length
        idx = tl.load(ids_ptr + row * keep_length + offs, mask=live, other=-1)
        live = live & (idx >= 0)
        idx = tl.maximum(idx, 0)

        kt = tl.load(
            k_ptr + ((b * total_k + idx[None, :]) * KVH + g) * DIM + d[:, None],
            mask=live[None, :],
            other=0.0,
        )
        s = tl.dot(q_tile, kt) * scale
        s = tl.where(live[None, :], s, -1e30)
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(live[None, :], p, 0.0)
        v = tl.load(
            v_ptr + ((b * total_k + idx[:, None]) * KVH + g) * DIM + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, 1)
        m = m_new

    l = tl.where(l > 0, l, 1.0)
    tl.store(
        out_ptr + ((b * total_q + t) * QH + g * HG + h[:, None]) * DIM + d[None, :],
        (acc / l[:, None]).to(out_ptr.dtype.element_ty),
        mask=hlive[:, None],
    )


def read_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_key_ids: torch.Tensor,
    window: int,
    sink: int,
    scale: float,
    query_start: int = 0,
) -> torch.Tensor:
    batch, total_q, q_heads, dim = q.shape
    total_k = k.shape[1]
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    device = q.device

    position = query_start + torch.arange(total_q, device=device, dtype=torch.int32)
    window_ids = position[:, None] - (window - 1) + torch.arange(window, device=device, dtype=torch.int32)[None, :]
    window_ids = torch.where(window_ids < 0, torch.tensor(-1, dtype=torch.int32, device=device), window_ids)
    window_ids = torch.where(window_ids < sink, torch.tensor(-1, dtype=torch.int32, device=device), window_ids)
    sink_ids = torch.arange(sink, device=device, dtype=torch.int32)[None, :].expand(total_q, sink).contiguous()
    sink_ids = torch.where(sink_ids <= position[:, None], sink_ids, torch.tensor(-1, dtype=torch.int32, device=device))
    selected = selected_key_ids.to(torch.int32)
    if selected.dim() == 2:
        selected = selected.unsqueeze(0).expand(batch, total_q, selected.shape[-1])
    ids = torch.cat([selected, window_ids.unsqueeze(0).expand(batch, total_q, window)], dim=-1)
    ids = torch.cat([ids, sink_ids.unsqueeze(0).expand(batch, total_q, sink)], dim=-1).contiguous()

    out = torch.empty_like(q)
    _read_tile_kernel[(batch * total_q * kv_heads,)](
        q,
        k,
        v,
        ids,
        out,
        scale,
        total_q,
        total_k,
        ids.shape[-1],
        QH=q_heads,
        KVH=kv_heads,
        HG=group,
        DIM=dim,
        BLOCK=block_c_for(dim),
        PQ=PAD_HEADS,
        num_warps=WARPS,
        num_stages=3,
    )
    return out
