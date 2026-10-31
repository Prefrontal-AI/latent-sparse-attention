"""A cluster block's Ward merge on Blackwell - the Hopper merge's rounds on `tcgen05` tiles.

Every round, cluster exchange and label is [the Hopper merge](../hopper/ward.py)'s; only a tile of merge costs is taken
on [Blackwell's MMA](tile.py), so every round merges the same pools as the Triton merge on the same GPU.
"""

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.blackwell.tile import BlackwellTile
from lsa.core.kernels.cutedsl.hopper import ward as hopper_ward


class BlackwellWardClustering(BlackwellTile, hopper_ward.WardClustering):
    """The CuTeDSL ward clustering with its tiles on `tcgen05`."""


def supports(pool_keys: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL Ward merge takes these blocks - tf32x3, 64 index dims, 64 to 512 pools, Blackwell.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return (
        pool_keys.is_cuda
        and dispatch.capability(pool_keys.device) in dispatch.BLACKWELL
        and hopper_ward.takes_blocks(pool_keys, precision)
    )


def cluster_ward(
    pool_keys: torch.Tensor, pool_sizes: torch.Tensor, bucket_count: int, bucket_cap: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools down to `bucket_count` buckets - the Triton merge's labels.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32, as `supports` takes them.
        pool_sizes: Keys each pool holds [G, P] fp32, whole counts.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.

    Raises:
        ValueError: If the sizes are not [G, P] on the keys' device, or the merge does not take these blocks.
    """
    hopper_ward.check_sizes(pool_keys, pool_sizes)
    if not supports(pool_keys, "tf32x3"):
        raise ValueError(
            f"the CuTeDSL Ward merge takes tf32x3 blocks of {hopper_ward.TILE} to {hopper_ward.MOST_POOLS} pools, a power of two, "
            f"of 64 index dims, at most one block an SM, on a Blackwell GPU - got {tuple(pool_keys.shape)} on "
            f"{pool_keys.device}"
        )
    return hopper_ward.launch(pool_keys, pool_sizes, bucket_count, bucket_cap, BlackwellWardClustering)
