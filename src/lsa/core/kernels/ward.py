"""The ward clustering and the node-bucket clustering - where `cluster_ward` and `cluster_node_buckets` choose a language.

Under `kernels: cutedsl` a step runs its CuTeDSL kernel when [dispatch](dispatch.py)'s table names the GPU and the
kernel's own `supports` takes the blocks; anything else runs the [Triton kernel](triton/ward.py), which reads the same.
A merge of more blocks than the GPU has SMs runs in Triton, since a CTA a block would queue behind the others, and
says so once for each shape.
"""

import logging

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import node_bucket_clustering as cutedsl_node_bucket_clustering
from lsa.core.kernels.cutedsl import ward as cutedsl_ward
from lsa.core.kernels.triton import ward as triton_ward

logger = logging.getLogger(__name__)

# The (pools, blocks) of each merge already warned of
WARNED: set[tuple[int, int]] = set()


def warn_of_the_fallback(pool_keys: torch.Tensor, precision: str) -> None:
    """Say once for each shape that a merge asked for in CuTeDSL runs in Triton for want of SMs.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    blocks, pools, _ = pool_keys.shape
    # A single block is taken exactly when blocks are the only thing wrong
    # => the merge falls back for its block count alone
    if cutedsl_ward.fits_the_gpu(pool_keys) or not cutedsl_ward.supports(pool_keys[:1], precision):
        return
    if (pools, blocks) in WARNED:
        return
    WARNED.add((pools, blocks))
    logger.warning(
        "lsa-lsa: Ward merge of %d blocks of %d pools runs in Triton under kernels: cutedsl - "
        "more blocks than the GPU's %d SMs, where the CuTeDSL merge is slower",
        blocks,
        pools,
        torch.cuda.get_device_properties(pool_keys.device).multi_processor_count,
    )


def cluster_ward(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    precision: str = "tf32x3",
    kernels: str = "triton",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools down to `bucket_count` buckets on the GPU.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32, P a power of two from 16 to 4,096, index_dim a power of two from 16
            to 256.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
        precision: The Gram matrix's dot - `ieee`, `tf32x3` or `tf32`.
        kernels: `cutedsl` to hold a block in its CTAs' shared memory where the blocks allow it, `triton` otherwise -
            the same labels either way.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.

    Raises:
        ValueError: If P is not a power of two from 16 to 4,096, or index_dim is not a power of two from 16 to 256.
    """
    triton_ward.check_pools(pool_keys)
    if dispatch.runs_cutedsl("ward clustering", kernels, pool_keys.device):
        if cutedsl_ward.supports(pool_keys, precision):
            return cutedsl_ward.cluster_ward(pool_keys, pool_sizes, bucket_count, bucket_cap)
        warn_of_the_fallback(pool_keys, precision)
    return triton_ward.ward_merge(pool_keys, pool_sizes, bucket_count, bucket_cap, precision)


def cluster_node_buckets(
    key_sums: torch.Tensor, precision: str = "tf32x3", rounds_a_check: int = 8, kernels: str = "triton"
) -> torch.Tensor:
    """Pair every block's buckets by the Ward cost, the pairs each other's cheapest fixed round by round.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32, P a power of two from 32 to 4,096, index_dim a multiple
            of 16.
        precision: The Gram matrix's dot - `ieee`, `tf32x3` or `tf32`.
        rounds_a_check: Rounds launched between two looks at what is left.
        kernels: `cutedsl` to run every round in one launch where the blocks allow it, `triton` a launch a step -
            the same pairs either way.

    Returns:
        Each bucket's pair [G, P] int64, numbered from 0 within its block in the order of its first bucket.

    Raises:
        ValueError: If P is not a power of two from 32 to 4,096, or index_dim is not a multiple of 16.
    """
    triton_ward.check_buckets(key_sums)
    if (
        dispatch.runs_cutedsl("node-bucket clustering", kernels, key_sums.device)
        and rounds_a_check == cutedsl_node_bucket_clustering.CHECK
        and cutedsl_node_bucket_clustering.supports(key_sums, precision)
    ):
        return triton_ward.number_pairs(cutedsl_node_bucket_clustering.cluster_node_buckets(key_sums))
    return triton_ward.cluster_node_buckets(key_sums, precision, rounds_a_check)
