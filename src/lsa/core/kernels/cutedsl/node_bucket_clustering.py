"""The node-bucket clustering of a hierarchy growth on the GPU at hand - its tiles on Hopper's warpgroup MMA, or on Blackwell's `tcgen05` MMA.

The two run the same rounds and pair the same buckets, so a caller asks this module and never names a GPU. A GPU that
has neither instruction pairs in Triton.

The CTAs of a launch wait on each other through counters in global memory, so every one must be resident at once.
The launch is cooperative, so a grid that does not fit fails at launch instead of hanging, and a caller keeps to at most one block an SM.
"""

from types import ModuleType

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.blackwell import node_bucket_clustering as blackwell_clustering
from lsa.core.kernels.cutedsl.hopper import node_bucket_clustering as hopper_clustering

# Rounds between two looks at what is left, as the Triton pairing looks
CHECK = hopper_clustering.CHECK


def kernel_for(device: torch.device) -> ModuleType:
    """Return the node-bucket clustering module built for a device's GPU, Hopper's for any GPU that is not a Blackwell.

    Args:
        device: The device the key sums are on.
    """
    if device.type == "cuda" and dispatch.capability(device) in dispatch.BLACKWELL:
        return blackwell_clustering
    return hopper_clustering


def supports(key_sums: torch.Tensor, precision: str) -> bool:
    """Tell whether a CuTeDSL node-bucket clustering takes these blocks on their GPU.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return bool(kernel_for(key_sums.device).supports(key_sums, precision))


def cluster_node_buckets(key_sums: torch.Tensor) -> torch.Tensor:
    """Return each bucket's partner within its block, with the kernel built for its GPU.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32.

    Returns:
        Each bucket's partner [G, P] int64.
    """
    partner: torch.Tensor = kernel_for(key_sums.device).cluster_node_buckets(key_sums)
    return partner
