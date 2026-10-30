"""Two-path LSA's summary path: each visible key the read skips enters the softmax summarised.

From lsa's "Native LSA Training" design (section 4) and its multi-resolution read (branch
caciolai/native-training, `lsa.training.native.layers`). A query reads its sinks, its window and
the keys LSA picks exactly. Every other key of a closed block enters the same softmax through the
summary of the unit it belongs to: a bucket nothing read, or what is left of a bucket the read took
part of. A unit U of n keys enters as one term with the value mean_U(v) and the logit

    log n + scale q . mu + scale^2 |q|^2 sigma^2 / 2,    mu = mean_U(k),
                                                         sigma^2 = mean_U |k - mu|^2 / d,

an estimate of log sum_U exp(scale q . k) exact to second order for isotropic keys. The statistics
are differentiable in k and v over the fixed buckets, so the language-model loss reaches the keys
and values the read skipped; Ward and the selection stay without gradient.

A bucket's members run in position order, and of them a query reads its sinks (the first), the
keys its selection takes (the next, in order) and its window (the last), so what it leaves is one
run of the members. Prefix sums over each bucket's members give every run's statistics without a
key set per query. Two kinds of bucket are left partly read: the one where the read's budget runs
out, at most one a query, and the buckets of the block before the query's, which its window
overlaps. Every other closed bucket is read whole or not at all, and an unread one is summarised
from statistics every query shares.

`terms` returns the read's share of the joint softmax and the summaries' output, so the joint
output is share * read + summaries: the read keeps its own kernel and its log-sum-exp carries the
join's gradient into it.
"""

import math
from typing import Any

import torch
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

# The logit of a unit with no keys: finite, so a softmax with none left has no NaN gradient
NO_MASS = -1e30
# Bytes of fp32 summary terms a query tile holds, which its checkpoint recomputes rather than stores
TILE_BYTES = 2**28
# Fp32 copies of a tile's unit statistics held at once
TILE_COPIES = 4


def wide(x: torch.Tensor) -> torch.Tensor:
    """Promote a tensor to at least fp32 while retaining fp64 reference inputs."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def tile_rows(
    batch: int, query_heads: int, kv_heads: int, buckets: int, per_block: int, dim: int, block: int
) -> int:
    """Size a query tile by the bytes of its summary terms, up to one cluster block.

    Args:
        batch: Sequences `B`.
        query_heads: Query heads `H`.
        kv_heads: Key heads `Hkv`.
        buckets: Buckets of the whole sequence `M`.
        per_block: Buckets a block holds.
        dim: Head dim.
        block: Query rows a cluster block holds, which a tile never crosses.

    Returns:
        Query rows a tile holds, at least 1.
    """
    row_bytes = 4 * batch * (TILE_COPIES * kv_heads * (per_block + 2) * dim + 3 * query_heads * buckets)
    return max(1, min(block, TILE_BYTES // row_bytes))


def prefixes(
    k: torch.Tensor, v: torch.Tensor, members: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build differentiable prefix sums over each bucket's members in position order.

    Args:
        k, v: (b, h_kv, t, d) keys and values.
        members: (b, m, cap) key positions in position order, PAD (-1) past a bucket's size.

    Returns:
        (b, h_kv, m, cap + 1, d) key and value sums over each bucket's first c members, and
        (b, h_kv, m, cap + 1) sums of the keys' squared norms, all at least fp32.
    """
    b, h_kv, _, d = k.shape
    held = (members >= 0)[:, None, :, :, None]
    index = members.clamp_min(0).flatten(1)[:, None, :, None].expand(b, h_kv, -1, d)

    def gathered(x: torch.Tensor) -> torch.Tensor:
        """Gather bucket members with padded positions contributing zero."""
        rows = wide(x).gather(2, index).view(b, h_kv, *members.shape[1:], d)
        return rows * held

    keys, values = gathered(k), gathered(v)
    zero = keys.new_zeros(b, h_kv, members.shape[1], 1, d)
    key_sums = torch.cat([zero, keys.cumsum(3)], 3)
    value_sums = torch.cat([zero, values.cumsum(3)], 3)
    squares = torch.cat([zero[..., 0], keys.square().sum(-1).cumsum(3)], 3)
    return key_sums, value_sums, squares


def _runs(
    sums: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ids: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute key and value means, spread and size over each bucket run.

    Args:
        sums: prefixes' three sums.
        ids, lo, hi: (b, *shape) int64 bucket and member bounds, shared by every head.

    Returns:
        (b, h_kv, *shape, d) key and value means, (b, h_kv, *shape) spread sigma^2 and size, all
        at least fp32; a run of no keys has size 0 and zero statistics.
    """
    key_sums, value_sums, squares = sums
    b, h_kv, _, slots, d = key_sums.shape
    shape = ids.shape[1:]

    def at(table: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Gather a prefix endpoint for each requested bucket run."""
        flat = (ids * slots + slot).flatten(1)[:, None]
        if table.dim() == 4:
            return table.flatten(2).gather(2, flat.expand(b, h_kv, -1)).view(b, h_kv, *shape)
        rows = table.flatten(2, 3).gather(2, flat[..., None].expand(b, h_kv, -1, d))
        return rows.view(b, h_kv, *shape, d)

    size = (hi - lo).clamp_min(0).to(key_sums.dtype)[:, None].expand(b, h_kv, *shape)
    per = 1 / size.clamp_min(1)
    key_mean = (at(key_sums, hi) - at(key_sums, lo)) * per[..., None]
    value_mean = (at(value_sums, hi) - at(value_sums, lo)) * per[..., None]
    second = (at(squares, hi) - at(squares, lo)) * per
    spread = (second - key_mean.square().sum(-1)).clamp_min(0) / d
    return key_mean, value_mean, spread, size


def _logits(
    q: torch.Tensor,
    q_sq: torch.Tensor,
    key_mean: torch.Tensor,
    spread: torch.Tensor,
    size: torch.Tensor,
    scale: float,
    log_weight: float,
) -> torch.Tensor:
    """Compute grouped summary logits, with NO_MASS for empty units.

    Args:
        q: (b, h_kv, g, r, d) queries, at least fp32; q_sq their (b, h_kv, g, r) squared norms.
        key_mean: (b, h_kv, u, d) units every query shares, or (b, h_kv, r, u, d) a query's own.
        spread, size: Matching (b, h_kv, u) or (b, h_kv, r, u).
    """
    if key_mean.dim() == 4:
        dots = torch.einsum("bkgrd,bkud->bkgru", q, key_mean)
        spread, size = spread[:, :, None, None], size[:, :, None, None]
    else:
        dots = torch.einsum("bkgrd,bkrud->bkgru", q, key_mean)
        spread, size = spread[:, :, None], size[:, :, None]
    logits = size.clamp_min(1).log() + scale * dots
    logits = logits + 0.5 * scale**2 * q_sq[..., None] * spread + log_weight
    return logits.masked_fill(size == 0, NO_MASS)


def terms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    members: torch.Tensor,
    counts: torch.Tensor,
    block: int,
    window: int,
    sink: int,
    scale: float,
    weight: float = 1.0,
    bias: torch.Tensor | None = None,
    index_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
    selected_positions: torch.Tensor | None = None,
    buckets_per_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join the read and summaries under one softmax, in bounded query chunks.

    Args:
        q: (b, h, t, d) queries at positions 0.., h a multiple of h_kv.
        k, v: (b, h_kv, t, d) keys and values.
        lse: (b, h, t) log-sum-exp of the read's scaled logits.
        members: cluster's (b, m, cap) key positions, block // (m // blocks) buckets a block.
        counts: lsa.read_counts' (b, t, m) keys each query's selection takes from each bucket.
        block: Keys a cluster block holds, equal to the window.
        window, sink: The read's window and sinks.
        scale: The logits' scale.
        weight: lambda, the summaries' weight in the softmax, above 0.
        bias: (b, t, m) differentiable bucket scores s, whose s - stopgrad(s) every summary of
            the bucket's keys gains (the straight-through bias), or None.

    Returns:
        (b, h, t) share and (b, h, t, d) output, at least fp32: the joint output is share *
        read + output.
    """
    if window != block:
        raise ValueError(f"the summaries take the window as the cluster block, {window} != {block}")
    b, h, t, d = q.shape
    h_kv, m = k.shape[1], members.shape[1]
    per_block = m // (t // block) if buckets_per_block is None else buckets_per_block
    log_weight = math.log(weight)
    sums = prefixes(k, v, members)
    held = members >= 0
    sizes = held.sum(-1)
    sinks = (held & (members < sink)).sum(-1)
    shared = _runs(sums, torch.arange(m, device=q.device).expand(b, m), sinks, sizes)
    queries = wide(q).view(b, h_kv, h // h_kv, t, d)
    q_sq = queries.square().sum(-1)
    lse = wide(lse).to(queries.dtype).view(b, h_kv, h // h_kv, t)
    if bias is not None:
        bias = bias - bias.detach()
    index_centroids = None
    if index_sources is not None:
        index_keys = functional.normalize(index_sources[1].float(), dim=-1)
        batch_ids = torch.arange(b, device=q.device)[:, None, None]
        gathered = index_keys[batch_ids, members.clamp_min(0)]
        index_centroids = functional.normalize((gathered * held[..., None]).sum(2), dim=-1)
    owner = None
    if selected_positions is not None:
        owner = torch.full((b, t + 1), m, dtype=torch.long, device=q.device)
        ids = torch.arange(m, device=q.device).repeat_interleave(members.shape[2])
        owner.scatter_(1, torch.where(held, members, t).flatten(1), ids.expand(b, -1))
        owner[:, t] = m
    shares, outputs = [lse.new_ones(b, h_kv, h // h_kv, min(block, t))], []
    outputs.append(queries.new_zeros(b, h_kv, h // h_kv, min(block, t), d))
    step = tile_rows(b, h, h_kv, m, per_block, d, block)
    starts = [
        start
        for block_start in range(block, t, block)
        for start in range(block_start, min(block_start + block, t), step)
    ]
    for start in starts:
        end = min(start + step, (start // block + 1) * block, t)
        rows = slice(start, end)
        index_queries = None if index_sources is None else index_sources[0][:, rows]
        share, out = checkpoint(
            _block_terms,
            queries[:, :, :, rows],
            q_sq[:, :, :, rows],
            lse[:, :, :, rows],
            members,
            sizes,
            sinks,
            counts[:, rows] if selected_positions is None else counts,
            shared,
            sums,
            start // block,
            per_block,
            window,
            scale,
            log_weight,
            None if bias is None else bias[:, rows],
            start,
            index_queries,
            index_centroids,
            None if selected_positions is None else selected_positions[:, rows],
            owner,
            use_reentrant=False,
        )
        shares.append(share)
        outputs.append(out)
    share = torch.cat(shares, 3).view(b, h, t)
    return share, torch.cat(outputs, 3).view(b, h, t, d)


def _block_terms(
    queries: Any, q_sq: Any, lse: Any, members: Any, sizes: Any, sinks: Any, counts: Any, shared: Any, sums: Any, block_id: int, per_block: int, window: int,
    scale: float, log_weight: float, bias: Any,
    query_start: int | None = None, index_queries: torch.Tensor | None = None, index_centroids: torch.Tensor | None = None,
    selected_positions: torch.Tensor | None = None, owner: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Compute summary terms for queries sharing the same closed blocks."""
    b, r = queries.shape[0], queries.shape[3]
    device = queries.device
    if selected_positions is not None and owner is not None:
        buckets = shared[0].shape[2]
        tokens = owner.shape[1] - 1
        ids = torch.where(selected_positions >= 0, selected_positions.long(), tokens)
        owned = owner.gather(1, ids.flatten(1)).view_as(ids)
        counts = torch.zeros(b, r, buckets + 1, dtype=torch.long, device=device)
        counts.scatter_add_(2, owned, torch.ones_like(owned))
        counts = counts[..., :buckets]
    before = (block_id - 1) * per_block
    query_start = block_id * window if query_start is None else query_start
    positions = query_start + torch.arange(r, device=device)
    if index_queries is not None and index_centroids is not None:
        index_scores = index_queries.float() @ index_centroids.transpose(1, 2)
        bias = index_scores - index_scores.detach()
    counts = counts[:, :, : block_id * per_block].long()
    logits, values = [], []

    if before:
        # Blocks before the last closed one: no window key, each bucket read whole or not at all
        key_mean, value_mean, spread, size = (x[:, :, :before] for x in shared)
        untouched = (counts[:, :, :before] == 0)[:, None, None]
        whole = _logits(queries, q_sq, key_mean, spread, size, scale, log_weight)
        if bias is not None:
            whole = whole + bias[:, None, None, :, :before]
        logits.append(whole.masked_fill(~untouched, NO_MASS))
        values.append(("shared", value_mean))
        # The bucket where the budget ran out, its first count members outside the sinks read
        taken = counts[:, :, :before]
        cut = (taken > 0) & (taken < (sizes - sinks)[:, None, :before])
        ids = cut.int().argmax(-1, keepdim=True)
        lo = sinks.gather(1, ids.flatten(1)).view_as(ids) + taken.gather(2, ids)
        whole_size = sizes.gather(1, ids.flatten(1)).view_as(ids)
        hi = torch.where(cut.any(-1, keepdim=True), whole_size, lo)
        key_mean, value_mean, spread, size = _runs(sums, ids, lo, hi)
        rest = _logits(queries, q_sq, key_mean, spread, size, scale, log_weight)
        if bias is not None:
            rest = rest + bias[:, None, None].gather(-1, ids[:, None, None].expand(-1, 1, 1, r, 1))
        logits.append(rest)
        values.append(("own", value_mean))

    # The last closed block: the window covers its latest keys, the selection may take some
    ids = (before + torch.arange(per_block, device=device)).expand(b, r, per_block)
    latest = members[:, before : before + per_block]
    recent = (positions - window)[None, :, None, None]
    in_window = (latest[:, None] > recent) & (latest[:, None] >= 0)
    lo = sinks[:, None, before : before + per_block] + counts[:, :, before:]
    hi = sizes[:, None, before : before + per_block] - in_window.sum(-1)
    key_mean, value_mean, spread, size = _runs(sums, ids, lo, hi.clamp_min(lo))
    last = _logits(queries, q_sq, key_mean, spread, size, scale, log_weight)
    if bias is not None:
        last = last + bias[:, None, None, :, before : before + per_block]
    logits.append(last)
    values.append(("own", value_mean))

    joined = torch.cat(logits, -1)
    total = torch.logaddexp(lse, joined.logsumexp(-1))
    weights = (joined - total[..., None]).exp()
    out = torch.zeros_like(queries)
    offset = 0
    for kind, value_mean in values:
        unit_count = value_mean.shape[2] if kind == "shared" else value_mean.shape[3]
        part = weights[..., offset : offset + unit_count]
        offset += unit_count
        if kind == "shared":
            out = out + torch.einsum("bkgru,bkud->bkgrd", part, value_mean)
        else:
            out = out + torch.einsum("bkgru,bkrud->bkgrd", part, value_mean)
    return (lse - total).exp(), out
