"""A cluster block's ward clustering on the GPU at hand - its tiles on Hopper's warpgroup MMA, or on Blackwell's `tcgen05` MMA.

The two run the same rounds and merge the same pools, so a caller asks this module and never names a GPU. A GPU that
has neither instruction merges in Triton.
"""

from types import ModuleType

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.blackwell import ward as blackwell_ward
from lsa.core.kernels.cutedsl.hopper import ward as hopper_ward


def kernel_for(device: torch.device) -> ModuleType:
    """Return the ward clustering module built for a device's GPU, Hopper's for any GPU that is not a Blackwell.

    Args:
        device: The device the pool keys are on.
    """
    if device.type == "cuda" and dispatch.capability(device) in dispatch.BLACKWELL:
        return blackwell_ward
    return hopper_ward


def fits_the_gpu(pool_keys: torch.Tensor) -> bool:
    """Tell whether every block's CTA is resident at once - no more blocks than the GPU has SMs.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
    """
    return hopper_ward.fits_the_gpu(pool_keys)


def supports(pool_keys: torch.Tensor, precision: str) -> bool:
    """Tell whether a CuTeDSL ward clustering takes these blocks on their GPU.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return bool(kernel_for(pool_keys.device).supports(pool_keys, precision))


def cluster_ward(
    pool_keys: torch.Tensor, pool_sizes: torch.Tensor, bucket_count: int, bucket_cap: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools down to `bucket_count` buckets, with the kernel built for its GPU.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.
    """
    merged: tuple[torch.Tensor, torch.Tensor] = kernel_for(pool_keys.device).cluster_ward(
        pool_keys, pool_sizes, bucket_count, bucket_cap
    )
    return merged
