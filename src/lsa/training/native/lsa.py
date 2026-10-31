"""The flat LSA read as a training arm - the keys a query reads are exactly the keys lsa serves.

The index space is the model's own q,k input and nothing is learned: the position-free dims of the post-norm query and
key heads, grouped-query regrouped into unit rows that every head of the layer shares, as `lsa.core` builds them.
Ward clusters the pools of each closed block of those index keys, a bucket's centroid is the normalised sum of its
index keys, and a query scores the centroids of the blocks closed before its own, takes its `top_p` best and reads their
keys best bucket first until `top_k` keys outside its sinks and window are held, the last bucket cut short.

`lsa.core.reference.build_buckets` and `read` are the oracle and run on a CPU one query at a time. This module is
the same selection batched over the closed blocks of a training sequence on the model's device, built on core's own
`pool_keys` and `ward_merge`, and a test holds its picks to the oracle's. The selection is `no_grad`: the model trains
through the scores and values of the keys it picks, never through the pick.
"""

import torch

from lsa import constants as const
from lsa.core import RotaryLayout, position_free, regroup
from lsa.core.ward import pool_keys, ward_merge
from lsa.training.native.buckets import closed_block_count
from lsa.training.native.config import AttentionConfig


def index_vectors(
    queries: torch.Tensor, keys: torch.Tensor, rotary_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the index queries and index keys of a batch of sequences, in the space core serves.

    Args:
        queries: Post-norm query heads [B, Hq, T, head_dim]. The rotated dims are dropped, so the rotation is no matter.
        keys: Post-norm key heads [B, Hkv, T, head_dim].
        rotary_dim: The leading head dims the model rotates, which are dropped. 0 drops none.

    Returns:
        Index queries [B, T, Hkv * (head_dim - rotary_dim)] fp32 unit rows.
        Index keys [B, T, Hkv * (head_dim - rotary_dim)] fp32 unit rows.
    """
    rotary = RotaryLayout(const.Rotary.PREFIX, rotary_dim) if rotary_dim else RotaryLayout()
    return regroup(*position_free(queries.detach(), keys.detach(), rotary))


@torch.no_grad()
def build_flat_buckets(
    index_keys: torch.Tensor, config: AttentionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster every closed block of every sequence's index keys into buckets with Ward.

    Args:
        index_keys: Index keys [N, T, index_dim] fp32 unit rows.
        config: The attention's knobs.

    Returns:
        Bucket centroids [N, buckets, index_dim] fp32, the normalised key sum of each bucket.
        Bucket members [N, buckets, bucket_cap] int32, each bucket's positions ascending and `-1` padded.
    """
    sequence_count, _, index_dim = index_keys.shape
    block_size, per_block = config.block_size, config.buckets_per_block
    blocks = closed_block_count(index_keys.shape[1], block_size)
    device = index_keys.device
    rows = (
        index_keys[:, : blocks * block_size].float().reshape(sequence_count * blocks, block_size, index_dim)
    )
    pools = pool_keys(rows, config.pool_size)
    pool_sizes = torch.full(pools.shape[:2], float(config.pool_size), device=device)
    labels = ward_merge(pools, pool_sizes, per_block, config.bucket_cap)
    # Bucket of each key within its block [N * blocks, block_size]
    key_labels = labels.repeat_interleave(config.pool_size, dim=1)
    key_sums = torch.zeros(rows.shape[0], per_block, index_dim, device=device)
    key_sums.scatter_add_(1, key_labels[..., None].expand(-1, -1, index_dim), rows)
    centroids = key_sums / key_sums.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    members = _members(key_labels, config)
    block_starts = block_size * (torch.arange(rows.shape[0], device=device) % blocks)
    members = torch.where(members >= 0, members + block_starts[:, None, None], -1)
    return (
        centroids.reshape(sequence_count, blocks * per_block, index_dim),
        members.reshape(sequence_count, blocks * per_block, config.bucket_cap),
    )


def _members(key_labels: torch.Tensor, config: AttentionConfig) -> torch.Tensor:
    """List each bucket's keys in position order from the bucket of every key.

    Args:
        key_labels: The bucket of each key [G, block_size] int64, from 0 to `buckets_per_block` - 1.
        config: The attention's knobs.

    Returns:
        Block-local members [G, buckets_per_block, bucket_cap] int32, `-1` past a bucket's size.
    """
    groups, block_size = key_labels.shape
    per_block = config.buckets_per_block
    device = key_labels.device
    order = key_labels.argsort(dim=1, stable=True)
    sorted_labels = key_labels.gather(1, order)
    counts = torch.nn.functional.one_hot(key_labels, per_block).sum(1)
    starts = counts.cumsum(1) - counts
    rank = torch.arange(block_size, device=device)[None] - starts.gather(1, sorted_labels)
    members = torch.full((groups, per_block, config.bucket_cap), -1, dtype=torch.int32, device=device)
    members[torch.arange(groups, device=device)[:, None], sorted_labels, rank] = order.to(torch.int32)
    return members


@torch.no_grad()
def read_keys(index_queries: torch.Tensor, index_keys: torch.Tensor, config: AttentionConfig) -> torch.Tensor:
    """Mark the long-range keys each query reads - the keys of its best buckets, best first, up to `top_k`.

    Sinks, window and the query's own open block are not marked, because the attention reads them anyway.

    Args:
        index_queries: Index queries [N, T, index_dim] fp32 unit rows.
        index_keys: Index keys [N, T, index_dim] fp32 unit rows.
        config: The attention's knobs.

    Returns:
        Whether query reads key [N, T, T] bool.
    """
    sequence_count, key_count, _ = index_keys.shape
    read = torch.zeros(sequence_count, key_count, key_count + 1, dtype=torch.bool, device=index_keys.device)
    blocks = closed_block_count(key_count, config.block_size)
    if blocks == 0:
        return read[..., :key_count].contiguous()
    centroids, members = build_flat_buckets(index_keys, config)
    for sequence in range(sequence_count):
        for block in range(1, blocks + 1):
            queries = slice(block * config.block_size, min((block + 1) * config.block_size, key_count))
            visible = block * config.buckets_per_block
            read[sequence, queries] = _read_block(
                index_queries[sequence, queries],
                centroids[sequence, :visible],
                members[sequence, :visible],
                queries.start,
                key_count,
                config,
            )
    return read[..., :key_count].contiguous()


def _read_block(
    index_queries: torch.Tensor,
    centroids: torch.Tensor,
    members: torch.Tensor,
    query_start: int,
    key_count: int,
    config: AttentionConfig,
) -> torch.Tensor:
    """Select the keys of the queries of one cluster block, which all see the same buckets.

    Args:
        index_queries: The block's index queries [Q, index_dim].
        centroids: The visible buckets' centroids [visible, index_dim].
        members: The visible buckets' members [visible, bucket_cap] int32.
        query_start: Position of the block's first query.
        key_count: Keys in the sequence.
        config: The attention's knobs.

    Returns:
        Whether each query reads each key, with one spare column at the end [Q, key_count + 1] bool.
    """
    device = index_queries.device
    positions = query_start + torch.arange(index_queries.shape[0], device=device)
    scores = index_queries @ centroids.T
    # Best score first, a tie to the lower bucket
    picked = scores.sort(dim=-1, descending=True, stable=True).indices[:, : config.selection_width]
    picked_members = members[picked].flatten(1).long()
    window_start = (positions + 1 - config.window_size).clamp_min(0)[:, None]
    outside = (picked_members >= config.sink_size) & (picked_members < window_start)
    kept = outside & (outside.cumsum(1) <= config.top_k)
    read = torch.zeros(positions.numel(), key_count + 1, dtype=torch.bool, device=device)
    # Keys not kept land in the spare column
    read.scatter_(1, torch.where(kept, picked_members, key_count), torch.ones_like(kept))
    return read
