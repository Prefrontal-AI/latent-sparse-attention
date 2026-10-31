"""The bucket scan on the GPU at hand - on Hopper's warpgroup MMA, or on Blackwell's `tcgen05` MMA.

The two kernels list the same buckets from the same tensor-core scores and take the same rows, so a caller asks this
module and never names a GPU. A GPU that has neither instruction raises here - the select asks `supports` first and
lists by the Triton survivor words when it is no ([scan_select](../scan_select.py)).

Scores and thresholds are finite. Both kernels' fast paths test the sign bit of a score less its threshold, which counts
a NaN as a hit where the slow path and Triton's `>=` do not. Scores are internal, so nothing checks.
"""

from types import ModuleType

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.blackwell import bucket_scan as blackwell_bucket_scan
from lsa.core.kernels.cutedsl.hopper import bucket_scan as hopper_bucket_scan

# Survivors a row holds in shared memory before they move to the global listing, and tiles a span holds at most -
# the same on either GPU
ROW_SLOTS = hopper_bucket_scan.ROW_SLOTS
MAX_SPAN_TILES = hopper_bucket_scan.MAX_SPAN_TILES


def kernel_for(device: torch.device) -> ModuleType:
    """Return the bucket scan module built for a device's GPU, Hopper's for any GPU that is not a Blackwell.

    Args:
        device: The device the rows are on.
    """
    if device.type == "cuda" and dispatch.capability(device) in dispatch.BLACKWELL:
        return blackwell_bucket_scan
    return hopper_bucket_scan


def supports(index_queries: torch.Tensor, bucket_centroids: torch.Tensor) -> bool:
    """Tell whether a CuTeDSL bucket scan takes these rows on their GPU.

    Args:
        index_queries: Index queries [R, index_dim].
        bucket_centroids: Bucket centroids [N, index_dim].
    """
    return bool(kernel_for(index_queries.device).supports(index_queries, bucket_centroids))


def scan_buckets(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    thresholds: torch.Tensor,
    capacity: int,
    span_tiles: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """List the buckets each row sees that score at or above its threshold, with the kernel built for its GPU.

    The scores and the thresholds must be finite, bar a threshold of -inf or inf - a NaN counts as a hit in a kernel's
    fast path.

    Args:
        index_queries: Index queries [R, index_dim] bf16.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        thresholds: Score each row lists at or above [R] fp32.
        capacity: Buckets a row's listing holds.
        span_tiles: Tiles a CTA scans - None to fill the GPU.

    Returns:
        The listing [R, capacity] int32 - a row's first `capacity` survivors in no defined order, unset after.
        Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.
    """
    listing = kernel_for(index_queries.device)
    result: tuple[torch.Tensor, torch.Tensor] = listing.scan_buckets(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity, span_tiles
    )
    return result
