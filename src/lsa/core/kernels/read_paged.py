"""Sparse read executed with torch_npu._npu_paged_attention (M1).

The oracle `topk_attention` reads keep = selection U window U sinks with one causal softmax.
This module produces the same keep set per query, gathers those kv rows into a compact
private block cache, and hands the whole batch to the production paged-attention op in
chunks of 256 queries (the op's batch ceiling).

Semantics matched to the oracle:
- keep ids are absolute key positions, causal, window then sinks merged, PAD dropped;
- one softmax over the whole keep set, fp32 accumulation inside the op;
- output cast back to q's dtype.

Known op constraints (see REPORT 20.5): the first call per process fails cleanly and must
be retried; context_lens must be a device tensor; the cache is 3D [NB, BSZ, Hkv * D].
"""

import logging

import torch

PAD = -1
_LOG = logging.getLogger(__name__)


def build_keep_ids(
    selected_key_ids: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    sink: int,
    key_count: int,
) -> torch.Tensor:
    """Return sorted, deduplicated, causal keep ids per query, right-padded with PAD.

    Args:
        selected_key_ids: Absolute key positions [N, budget] int32/64, PAD for none.
        positions: Absolute position of each query row [N] int64. Rows of one batch
            share the batch's key sequence, so callers pass per-batch positions, not
            one monotonically increasing range across batches.
        window: Recent keys each query reads, its own included.
        sink: Initial keys every query reads.
        key_count: Number of available key positions.

    Returns:
        Keep ids [N, L] int64 sorted ascending, PAD-padded on the right.
    """
    device = selected_key_ids.device
    n, budget = selected_key_ids.shape
    positions = positions.to(device=device, dtype=torch.int64)

    window_ids = positions[:, None] - torch.arange(window, device=device)[None, :]
    sink_ids = torch.arange(sink, device=device)[None, :].expand(n, -1)
    sel = selected_key_ids.long()
    sel_valid = sel != PAD
    sel = torch.where(sel_valid, sel, torch.zeros_like(sel))
    cat = torch.cat([sel, window_ids, sink_ids], dim=1)
    valid = torch.cat(
        [
            sel_valid,
            torch.ones_like(window_ids, dtype=torch.bool),
            torch.ones_like(sink_ids, dtype=torch.bool),
        ],
        dim=1,
    )

    cat = cat.clamp(0, key_count - 1)
    cat = torch.where(cat > positions[:, None], torch.zeros_like(cat), cat)
    cat = torch.where(valid & (cat <= positions[:, None]), cat, torch.full_like(cat, PAD))

    sorted_ids, order = torch.sort(cat, dim=1)
    live = sorted_ids != PAD
    first = live & torch.cat(
        [torch.ones_like(sorted_ids[:, :1], dtype=torch.bool), sorted_ids[:, 1:] != sorted_ids[:, :-1]], dim=1
    )
    counts = first.sum(dim=1)
    max_keep = int(counts.max().item()) if n else 0

    flat_rank = torch.cumsum(first.long(), dim=1) - 1
    keep = torch.full((n, max_keep), PAD, dtype=torch.int64, device=device)
    row = torch.arange(n, device=device)[:, None].expand_as(sorted_ids)
    keep[row[first], flat_rank[first]] = sorted_ids[first]
    _LOG.debug("keep ids: n=%d budget=%d max_keep=%d mean=%.1f", n, budget, max_keep, counts.float().mean().item())
    return keep, counts


def paged_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_key_ids: torch.Tensor,
    window: int,
    sink: int,
    scale: float,
    query_start: int = 0,
    block_size: int = 128,
    chunk: int = 256,
    backend: str = "v2",
) -> torch.Tensor:
    """Attention over each query's keep set via the production paged-attention op.

    Args:
        q: Queries [B, Tq, Hq, Dh] (bf16 or fp16 on npu).
        k: Keys [B, Tk, Hkv, Dh], same dtype as q.
        v: Values [B, Tk, Hkv, Dh], same dtype as q.
        selected_key_ids: Long-range key positions [B, Tq, budget] int32, PAD for none.
        window: Recent keys each query reads.
        sink: Initial keys every query reads.
        scale: Multiplier applied to query-key scores.
        query_start: Absolute position of the first query row.
        block_size: Cache block size the compact cache is built on.
        chunk: Queries per op call, at most 256.

    Returns:
        Attention output [B, Tq, Hq, Dh], q's dtype and device.
    """
    import torch_npu

    batch_size, query_count, head_count, head_dim = q.shape
    key_count = k.shape[1]
    kv_head_count = k.shape[2]
    n = batch_size * query_count

    sel = selected_key_ids.reshape(n, -1)
    row_positions = query_start + torch.arange(query_count, device=q.device).repeat(batch_size)
    keep, counts = build_keep_ids(sel, row_positions, window, sink, key_count)
    max_keep = keep.shape[1]
    max_blocks = (max_keep + block_size - 1) // block_size

    b_idx = torch.arange(batch_size, device=q.device)[:, None].expand(-1, query_count).reshape(n)
    pad = torch.where(keep == PAD, torch.zeros_like(keep), keep)
    gathered_k = k[b_idx[:, None], pad]
    gathered_v = v[b_idx[:, None], pad]
    live = keep != PAD
    gathered_k = torch.where(live[:, :, None, None], gathered_k, torch.zeros_like(gathered_k))
    gathered_v = torch.where(live[:, :, None, None], gathered_v, torch.zeros_like(gathered_v))

    padded = max_blocks * block_size
    if padded > max_keep:
        tail = torch.zeros(n, padded - max_keep, kv_head_count, head_dim, dtype=k.dtype, device=q.device)
        gathered_k = torch.cat([gathered_k, tail], dim=1)
        gathered_v = torch.cat([gathered_v, tail], dim=1)

    if backend == "v2":
        cache_k = gathered_k.view(n, max_blocks, block_size, kv_head_count, head_dim).reshape(-1, block_size, kv_head_count, head_dim).contiguous()
        cache_v = gathered_v.view(n, max_blocks, block_size, kv_head_count, head_dim).reshape(-1, block_size, kv_head_count, head_dim).contiguous()
    else:
        cache_k = gathered_k.view(n, max_blocks, block_size, kv_head_count * head_dim).reshape(-1, block_size, kv_head_count * head_dim).contiguous()
        cache_v = gathered_v.view(n, max_blocks, block_size, kv_head_count * head_dim).reshape(-1, block_size, kv_head_count * head_dim).contiguous()
    block_table = torch.arange(n * max_blocks, device=q.device, dtype=torch.int32).view(n, max_blocks)

    q_flat = q.reshape(n, head_count, head_dim).contiguous()
    out = torch.empty_like(q_flat)

    if backend == "v2":
        torch.ops.atb._npu_paged_attention_v2.out(
            q_flat, cache_k, block_table, counts.tolist(),
            value_cache=cache_v, num_kv_heads=kv_head_count, num_heads=head_count,
            scale_value=scale, out=out)
        torch.npu.synchronize()
        return out.view(batch_size, query_count, head_count, head_dim).to(q.dtype)

    context_lens = counts.to(torch.int32)
    done = 0
    while done < n:
        end = min(done + chunk, n)
        call = lambda: torch_npu._npu_paged_attention(
            query=q_flat[done:end],
            key_cache=cache_k,
            value_cache=cache_v,
            num_kv_heads=kv_head_count,
            num_heads=head_count,
            scale_value=scale,
            block_table=block_table[done:end],
            context_lens=context_lens[done:end],
            out=out[done:end],
        )
        try:
            call()
        except RuntimeError as setup_failure:
            for attempt in range(2, 7):
                torch.npu.synchronize()
                try:
                    call()
                    _LOG.info("paged attention call ok on attempt %d (setup failed: %s)", attempt, setup_failure)
                    break
                except RuntimeError:
                    if attempt == 6:
                        raise
        done = end

    return out.view(batch_size, query_count, head_count, head_dim).to(q.dtype)
