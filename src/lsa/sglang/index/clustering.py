"""Ward over a cluster block's pools, and the buckets it leaves - their member rows, sizes and centroids.

A cluster block closes when its last key arrives. Its index keys are summed into pool keys, Ward merges the pools
down to the block's bucket count, and each bucket is the pools that merged into it. The merge is the whole
partition: there is no seed, no assignment pass and no fallback, because a cap of twice the bucket size always
leaves an allowed pair, and a cap under that is refused when the settings load.

A call clusters every block it is handed in one launch, so a prefill chunk's sixteen blocks cost one.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.ward import bucket_members_from_labels
from lsa.core.kernels.ward import cluster_ward
from lsa.core.ward import pool_keys as sum_pools
from lsa.core.ward import ward_merge
from lsa.sglang.settings.design import IndexerConfig

# Entries of the [pools, pools] matrices a torch round holds at once - blocks go through the rounds in groups
CPU_ROUND_ENTRIES = 2**25


class ClusteredBlocks(NamedTuple):
    """The buckets of some cluster blocks, block after block."""

    # Normalized key sum of each bucket [rows, index_dim] bf16
    bucket_centroids: torch.Tensor
    # Positions of each bucket's keys [rows, bucket_cap] int32, -1 padded
    bucket_members: torch.Tensor
    # Keys in each bucket [rows] int32
    bucket_sizes: torch.Tensor


def ward_labels(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    kernels: str = "triton",
) -> torch.Tensor:
    """Merge each block's pools down to exactly `bucket_count` buckets - the kernel on a GPU, torch elsewhere.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
        kernels: The language of the GPU merge - the same labels either way.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.

    Raises:
        ValueError: The cap is under twice the bucket size - a block stopping short of its bucket count fails a
            device assert on a GPU.
    """
    if not pool_keys.is_cuda:
        blocks, pools, _ = pool_keys.shape
        group = max(1, CPU_ROUND_ENTRIES // pools**2)
        return torch.cat(
            [
                ward_merge(
                    pool_keys[first : first + group],
                    pool_sizes[first : first + group],
                    bucket_count,
                    bucket_cap,
                )
                for first in range(0, blocks, group)
            ]
        )
    labels, reached = cluster_ward(pool_keys, pool_sizes, bucket_count, bucket_cap, kernels=kernels)
    # Checked on the device, since a host read of it would stall the stream
    torch._assert_async(reached.all())
    return labels


def cluster_blocks(
    index_keys: torch.Tensor, config: IndexerConfig, first_position: int, kernels: str = "triton"
) -> ClusteredBlocks:
    """Cluster consecutive closed cluster blocks into buckets, in one call.

    The centroid of a bucket is the normalized sum of its pool keys, which is the normalized sum of its index keys
    and equal to it in bf16.

    Args:
        index_keys: The blocks' index keys [G * window_size, index_dim] bf16, unit rows, in position order.
        config: The design.
        first_position: Position of the first block's first key.
        kernels: The language of the GPU Ward merge - the same buckets either way.

    Returns:
        The buckets of the G blocks, `window_size / bucket_size` a block, their members as absolute positions.
    """
    pool_size = config.pool_size or 1
    block_size = config.cluster_block_size
    bucket_count = block_size // config.bucket_size
    blocks = index_keys.shape[0] // block_size
    pools = sum_pools(index_keys.float().view(blocks, block_size, -1), pool_size)
    pool_sizes = torch.full(pools.shape[:2], float(pool_size), device=pools.device)
    labels = ward_labels(pools, pool_sizes, bucket_count, config.bucket_cap, kernels)
    local = bucket_members_from_labels(
        labels.repeat_interleave(pool_size, dim=1), bucket_count, config.bucket_cap
    )
    starts = first_position + block_size * torch.arange(blocks, device=pools.device)
    bucket_members = torch.where(local >= 0, local + starts[:, None, None], -1).to(torch.int32)
    pool_members = bucket_members_from_labels(labels, bucket_count, config.bucket_cap // pool_size)
    return ClusteredBlocks(
        pool_centroids(pools, pool_members).flatten(0, 1),
        bucket_members.flatten(0, 1),
        (bucket_members >= 0).sum(-1, dtype=torch.int32).flatten(),
    )


def pool_centroids(pool_keys: torch.Tensor, pool_members: torch.Tensor) -> torch.Tensor:
    """Normalize each bucket's pool keys summed.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32.
        pool_members: The pools of each bucket [G, m, bucket_cap / pool_size] int32, -1 past its size.

    Returns:
        The buckets' centroids [G, m, index_dim] bf16.
    """
    held = pool_members >= 0
    blocks = torch.arange(pool_keys.shape[0], device=pool_keys.device)[:, None, None]
    gathered = pool_keys[blocks, pool_members.clamp_min(0).long()]
    return F.normalize((gathered * held[..., None]).sum(2), dim=-1).to(torch.bfloat16)
