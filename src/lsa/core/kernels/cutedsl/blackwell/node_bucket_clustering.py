"""A hierarchy growth's pairing of buckets by the Ward cost on Blackwell - the Hopper pairing's rounds on `tcgen05` tiles.

Every round, check and leftover is [the Hopper node-bucket clustering](../hopper/node_bucket_clustering.py)'s; only a tile of merge costs is taken on
[Blackwell's MMA](tile.py), so the rounds pair the same buckets as the Triton pairing on the same GPU.
"""

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.blackwell.tile import BlackwellTile
from lsa.core.kernels.cutedsl.hopper import node_bucket_clustering as hopper_clustering


class BlackwellNodeBucketClustering(BlackwellTile, hopper_clustering.NodeBucketClustering):
    """The CuTeDSL node-bucket clustering with its tiles on `tcgen05`."""


def supports(key_sums: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL pairing takes these blocks - tf32x3, 64 index dims, 64 to 4,096 buckets, Blackwell.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return (
        key_sums.is_cuda
        and dispatch.capability(key_sums.device) in dispatch.BLACKWELL
        and hopper_clustering.takes_blocks(key_sums, precision)
    )


def cluster_node_buckets(key_sums: torch.Tensor) -> torch.Tensor:
    """Return each bucket's partner within its block, every round in one launch - the Triton rounds' partners.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32, as `supports` takes them.

    Returns:
        Each bucket's partner [G, P] int64.

    Raises:
        ValueError: If the pairing does not take these blocks.
    """
    if not supports(key_sums, "tf32x3"):
        raise ValueError(
            "the CuTeDSL pairing takes tf32x3 blocks of 64 to 4,096 buckets, a power of two, of 64 index dims, "
            f"at most one block an SM, on a Blackwell GPU - got {tuple(key_sums.shape)} on {key_sums.device}"
        )
    return hopper_clustering.launch(key_sums, BlackwellNodeBucketClustering)
