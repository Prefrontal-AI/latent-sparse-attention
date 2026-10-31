"""The partition of a sequence's closed keys into buckets, what each bucket is summarised by, and which a query reads.

Three steps, and the gradient passes through one of them.
`partition_keys` cuts each closed block of keys into buckets - Ward, chunks or a shuffle - and is `no_grad`: an
assignment has no derivative.
`bucket_statistics` takes the means and the variance of each bucket's keys and the mean of its values, which are
differentiable functions of the keys and values over the fixed assignment - so the loss trains the projections
through the summaries of the buckets nothing selected.
`select_buckets` ranks the buckets a query may read at fine resolution and keeps the top of the list, and is
`no_grad` too: a top-p index has no derivative.

A key in the `S` streams below is one sequence's keys of one key-value head, because a head clusters its own
keys and the query heads of its group share that partition and that selection.
Everything is `torch` alone, batched over streams and blocks, and runs on a CPU.
"""

from typing import NamedTuple

import torch
from torch.nn import functional as F

from lsa.core.ward import pool_keys, ward_merge
from lsa.training.native.config import AttentionConfig
from lsa.training.native.constants import Partition, Rank

# Queries a selection scores at once, which bounds the [S, G, queries, buckets] score tensor
QUERY_CHUNK = 1024


class BucketTable(NamedTuple):
    """Which bucket every key of the streams is in, and what that says about the buckets."""

    # Global bucket of each key [S, key_count] int64, numbered block after block. `bucket_count` for a key in no bucket
    bucket_of_key: torch.Tensor
    # Keys in each bucket [S, bucket_count] fp32
    bucket_sizes: torch.Tensor
    # Earliest and latest position of each bucket's keys [S, bucket_count] int64
    first_position: torch.Tensor
    last_position: torch.Tensor
    # The block each bucket was cut from [bucket_count] int64
    block_of_bucket: torch.Tensor
    bucket_count: int
    # Keys held by a bucket, which is every key of the closed blocks
    covered_count: int


class BucketStatistics(NamedTuple):
    """What a bucket's softmax terms are approximated by."""

    # Mean key of each bucket [S, bucket_count, head_dim] fp32
    key_means: torch.Tensor
    # Mean value of each bucket [S, bucket_count, head_dim] fp32
    value_means: torch.Tensor
    # Variance of the keys per dimension about their mean [S, bucket_count] fp32
    variances: torch.Tensor


def at_least_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor in fp32, or in fp64 when that is what it was, so a test can run the math in double."""
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


def closed_block_count(key_count: int, block_size: int) -> int:
    """Return the blocks some query of a sequence sees closed - those before the last query's own.

    Args:
        key_count: Keys in the sequence.
        block_size: Keys in a block.

    Returns:
        The number of blocks that get buckets.
    """
    return (key_count - 1) // block_size


@torch.no_grad()
def partition_keys(keys: torch.Tensor, config: AttentionConfig) -> BucketTable:
    """Cut every closed block of every stream's keys into buckets.

    Ward clusters the row-normalised keys, because the cost it merges by is the alignment lost and a unit row
    is what its pools and costs are written for. Qwen3.5's key norm keeps the norms near one value, so this is
    close to clustering the raw keys, and the statistics that follow use the raw ones.

    Args:
        keys: Post-RoPE, post-norm keys [S, key_count, head_dim].
        config: The attention's knobs.

    Returns:
        The table of the `(key_count - 1) // block_size` closed blocks, whose last block's keys and any beyond
        are in no bucket.
    """
    stream_count, key_count, head_dim = keys.shape
    device = keys.device
    block_size, per_block = config.block_size, config.buckets_per_block
    blocks = closed_block_count(key_count, block_size)
    covered = blocks * block_size
    bucket_count = blocks * per_block

    local = _local_labels(keys[:, :covered], config)
    # [S, covered] bucket of each key across the sequence, block after block
    covered_ids = (
        local.reshape(stream_count, blocks, block_size)
        + per_block * torch.arange(blocks, device=device)[None, :, None]
    )
    covered_ids = covered_ids.reshape(stream_count, covered)
    uncovered = torch.full((stream_count, key_count - covered), bucket_count, dtype=torch.long, device=device)
    bucket_of_key = torch.cat([covered_ids, uncovered], dim=1)

    positions = torch.arange(key_count, device=device)[None, :].expand(stream_count, -1)
    sizes = torch.zeros(stream_count, bucket_count + 1, device=device)
    sizes.scatter_add_(1, bucket_of_key, torch.ones_like(bucket_of_key, dtype=sizes.dtype))
    first = torch.full((stream_count, bucket_count + 1), key_count, dtype=torch.long, device=device)
    last = torch.full((stream_count, bucket_count + 1), -1, dtype=torch.long, device=device)
    first.scatter_reduce_(1, bucket_of_key, positions, reduce="amin")
    last.scatter_reduce_(1, bucket_of_key, positions, reduce="amax")
    return BucketTable(
        bucket_of_key=bucket_of_key,
        bucket_sizes=sizes[:, :bucket_count],
        first_position=first[:, :bucket_count],
        last_position=last[:, :bucket_count],
        block_of_bucket=torch.arange(bucket_count, device=device) // per_block,
        bucket_count=bucket_count,
        covered_count=covered,
    )


def _local_labels(keys: torch.Tensor, config: AttentionConfig) -> torch.Tensor:
    """Label each key of the closed blocks with its bucket within its block.

    Args:
        keys: The closed blocks' keys [S, blocks * block_size, head_dim].
        config: The attention's knobs.

    Returns:
        Block-local bucket of each key [S, blocks, block_size] int64, in `[0, buckets_per_block)`.
    """
    stream_count, covered, head_dim = keys.shape
    block_size, per_block = config.block_size, config.buckets_per_block
    blocks = covered // block_size
    device = keys.device
    shape = (stream_count, blocks, block_size)
    if blocks == 0:
        return torch.zeros(shape, dtype=torch.long, device=device)
    if config.partition == Partition.CHUNKS:
        return (torch.arange(block_size, device=device) // config.bucket_size).expand(shape).clone()
    if config.partition == Partition.RANDOM:
        generator = torch.Generator(device=device).manual_seed(config.seed)
        order = torch.rand(shape, generator=generator, device=device).argsort(-1).argsort(-1)
        return order // config.bucket_size
    rows = F.normalize(keys.float(), dim=-1).reshape(stream_count * blocks, block_size, head_dim)
    pools = pool_keys(rows, config.pool_size)
    pool_sizes = torch.full(pools.shape[:2], float(config.pool_size), device=device)
    labels = ward_merge(pools, pool_sizes, per_block, config.bucket_cap)
    return labels.repeat_interleave(config.pool_size, dim=1).reshape(shape)


def bucket_statistics(keys: torch.Tensor, values: torch.Tensor, table: BucketTable) -> BucketStatistics:
    """Take the mean key, the mean value and the key variance of every bucket, differentiably.

    Args:
        keys: Post-RoPE, post-norm keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        table: The buckets, from `partition_keys`.

    Returns:
        The statistics of every bucket in fp32, the variance being `sum ||k - mean||^2 / (n head_dim)`.
    """
    stream_count, _, head_dim = keys.shape
    bucket_count, covered = table.bucket_count, table.covered_count
    flat_ids = (
        table.bucket_of_key[:, :covered]
        + bucket_count * torch.arange(stream_count, device=keys.device)[:, None]
    )
    flat_ids = flat_ids.reshape(-1)
    sizes = at_least_float32(table.bucket_sizes.reshape(-1, 1).clamp_min(1))
    wide = at_least_float32(keys)

    def mean_of(rows: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros(stream_count * bucket_count, head_dim, device=rows.device, dtype=wide.dtype)
        return sums.index_add(0, flat_ids, at_least_float32(rows)[:, :covered].reshape(-1, head_dim)) / sizes

    key_means = mean_of(keys)
    value_means = mean_of(values)
    residual = wide[:, :covered].reshape(-1, head_dim) - key_means[flat_ids]
    squares = torch.zeros(stream_count * bucket_count, device=keys.device, dtype=wide.dtype)
    squares = squares.index_add(0, flat_ids, residual.square().sum(-1))
    variances = squares / (sizes[:, 0] * head_dim)
    return BucketStatistics(
        key_means.view(stream_count, bucket_count, head_dim),
        value_means.view(stream_count, bucket_count, head_dim),
        variances.view(stream_count, bucket_count),
    )


def estimated_log_mass(
    queries: torch.Tensor, statistics: BucketStatistics, table: BucketTable, scale: float, variance: bool
) -> torch.Tensor:
    """Estimate the log of the softmax mass each bucket holds for each query.

    `log |b| + q . mean_b * scale` is the log of the bucket's size times the geometric mean of its keys' weights,
    and the cumulant term `||q||^2 variance_b scale^2 / 2` is what Jensen's gap is to second order - zero for
    a bucket whose keys coincide, which is why singleton buckets are exact.

    Args:
        queries: Queries [S, G, Q, head_dim], G query heads to a key-value head.
        statistics: The buckets' statistics.
        table: The buckets.
        scale: The factor logits are scaled by, `head_dim ** -0.5`.
        variance: Whether to add the cumulant term.

    Returns:
        The estimated log mass [S, G, Q, bucket_count] fp32.
    """
    queries = at_least_float32(queries)
    logits = torch.einsum("sgqd,sbd->sgqb", queries, statistics.key_means) * scale
    log_mass = logits + table.bucket_sizes.to(logits.dtype).log()[:, None, None, :]
    if not variance:
        return log_mass
    squared_norm = queries.square().sum(-1)
    return log_mass + 0.5 * scale**2 * squared_norm[..., None] * statistics.variances[:, None, None, :]


class Reads(NamedTuple):
    """The buckets each query reads at fine resolution."""

    # True where query reads the bucket's keys [S, key_count, bucket_count + 1], the last column never
    read: torch.Tensor
    # Keys of selected buckets each query reads, the pinned ones too [S, key_count] fp32
    key_counts: torch.Tensor


@torch.no_grad()
def select_buckets(
    queries: torch.Tensor,
    statistics: BucketStatistics,
    table: BucketTable,
    config: AttentionConfig,
    scale: float,
) -> Reads:
    """Pick, for every query, the buckets it reads at fine resolution.

    A bucket is a candidate for a query once its block is closed before the query's own and no key of it is a sink
    or inside the window - a bucket with one such key is read in full whatever it scores, which is what keeps every
    key in the softmax once. Candidates are ranked, the best `selection_width` kept, and the list is cut at
    the first bucket that would take the keys past `top_k`, so a bucket is read whole or summarised whole.

    The ranking of a key-value head is the sum over its query heads of each head's score, a mass normalised
    over the candidates so that no head with larger logits decides alone.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        statistics: The buckets' statistics.
        table: The buckets.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The reads of every query.
    """
    stream_count, group, query_count, _ = queries.shape
    device = queries.device
    bucket_count = table.bucket_count
    read = torch.zeros(stream_count, query_count, bucket_count + 1, dtype=torch.bool, device=device)
    key_counts = torch.zeros(stream_count, query_count, device=device)
    if bucket_count == 0:
        return Reads(read, key_counts)
    width = min(config.selection_width, bucket_count)
    sink_pinned = (table.first_position < config.sink_size)[:, None, :]
    for start in range(0, query_count, QUERY_CHUNK):
        positions = torch.arange(start, min(start + QUERY_CHUNK, query_count), device=device)
        chunk = slice(start, start + positions.numel())
        closed = table.block_of_bucket[None, :] < (positions // config.block_size)[:, None]
        pinned = closed[None] & (
            sink_pinned | (table.last_position[:, None, :] > (positions - config.window_size)[None, :, None])
        )
        candidates = closed[None] & ~pinned
        scores = _rank_scores(queries[:, :, chunk], statistics, table, candidates, config, scale)
        read[:, chunk, :bucket_count] = pinned
        if width > 0 and config.top_k > 0:
            values, ids = scores.topk(width, dim=-1)
            valid = values > float("-inf")
            sizes = table.bucket_sizes[:, None, :].expand(-1, positions.numel(), -1).gather(2, ids)
            keep = valid & ((sizes * valid).cumsum(-1) <= config.top_k)
            chosen = torch.zeros_like(pinned)
            chosen.scatter_(2, ids, keep)
            read[:, chunk, :bucket_count] |= chosen
        key_counts[:, chunk] = (read[:, chunk, :bucket_count] * table.bucket_sizes[:, None, :]).sum(-1)
    return Reads(read, key_counts)


def _rank_scores(
    queries: torch.Tensor,
    statistics: BucketStatistics,
    table: BucketTable,
    candidates: torch.Tensor,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Score the candidate buckets of a chunk of queries by the run's rank.

    Args:
        queries: The chunk's queries [S, G, Q, head_dim].
        statistics: The buckets' statistics.
        table: The buckets.
        candidates: Which buckets each query may select [S, Q, bucket_count].
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        Scores [S, Q, bucket_count] fp32, `-inf` where a bucket is no candidate.
    """
    mask = candidates[:, None]
    if config.rank == Rank.COSINE:
        unit_queries = F.normalize(at_least_float32(queries), dim=-1)
        unit_means = F.normalize(statistics.key_means, dim=-1)
        scores = torch.einsum("sgqd,sbd->sgqb", unit_queries, unit_means).sum(1)
        return scores.masked_fill(~candidates, float("-inf"))
    log_mass = estimated_log_mass(queries, statistics, table, scale, config.variance)
    if config.rank == Rank.ERROR:
        squared_norm = at_least_float32(queries).square().sum(-1)
        error = squared_norm[..., None] * statistics.variances[:, None, None, :]
        log_mass = log_mass + (error + 1e-30).log()
    log_mass = log_mass.masked_fill(~mask, float("-inf"))
    peak = log_mass.amax(-1, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = (log_mass - peak).exp()
    shares = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
    return shares.sum(1).masked_fill(~candidates, float("-inf"))
