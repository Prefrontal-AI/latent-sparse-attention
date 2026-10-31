"""Whether every rank of a decode context-parallel group holds the same index - a checksum summed across the group.

Every rank keeps the whole bucket cache, and the hierarchy of fixed and tree grows on every rank from it and
from index keys the group sums. Nothing compares the copies while serving: a rank whose cache or tree drifted would
select other lists without an error, and one that grew where the others did not would call the sum a different number
of times and hang the group. The check reads a request's buckets, node levels and graph tree mirror into one checksum, sums it across
the group and raises on every rank at once when any rank's differs. It costs two sums and a host sync a layer, so a
server runs it only when asked to.
"""

from collections.abc import Callable, Sequence

import torch

from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.graph_tree import NODE_BUCKET_PAGE, GraphTree

# Bytes a checksum reads at once - its int64 copies stay a few hundred MB whatever the context
CHECKSUM_BYTES = 1 << 24
# Bits of the checksum one fp32 term carries - a group of up to 8 ranks sums them exactly
PIECE_BITS = 20
PIECES = 3


def checksum(values: torch.Tensor) -> torch.Tensor:
    """Return an order-sensitive checksum of a tensor's bytes [] int64.

    An integer sum wraps instead of rounding, so the order a reduction adds in does not change it.

    Args:
        values: Any tensor.

    Returns:
        The checksum, on the tensor's device.
    """
    # Copied in row-major strides - a column of one element keeps its stride under `contiguous`, which a byte view refuses
    data = values.clone(memory_format=torch.contiguous_format).view(-1).view(torch.uint8)
    total = torch.zeros((), dtype=torch.int64, device=values.device)
    for first in range(0, len(data), CHECKSUM_BYTES):
        part = data[first : first + CHECKSUM_BYTES].long()
        weights = torch.arange(first + 1, first + len(part) + 1, device=values.device)
        total += (part * weights).sum()
    return total


def mirror_terms(
    graph_tree: GraphTree, row: int, counts: list[int], parts: list[torch.Tensor]
) -> tuple[list[int], list[torch.Tensor]]:
    """Add the graph tree's view of one request - its counts, start, covered, page tables and node bucket rows.

    Args:
        graph_tree: The mirror.
        row: The request table row.
        counts: The checksum's counts so far.
        parts: The checksum's tensors so far.

    Returns:
        The counts and tensors with the mirror's added.
    """
    # One host read of the row's counts, then the rows its pages hold in creation order
    made = graph_tree.made[:, row].tolist()
    parts = [
        *parts,
        graph_tree.made[:, row],
        graph_tree.closed[:, row],
        graph_tree.start[row : row + 1],
        graph_tree.covered[row : row + 1],
    ]
    for level, count in enumerate(made):
        counts.append(count)
        parts.append(graph_tree.pages[level][row])
        index = torch.arange(count, device=graph_tree.device)
        pages = graph_tree.pages[level][row][index // NODE_BUCKET_PAGE].long()
        physical = pages * NODE_BUCKET_PAGE + index % NODE_BUCKET_PAGE
        parts.append(graph_tree.node_centroids[level][physical])
        parts.append(graph_tree.children[level][physical])
    return counts, parts


def request_checksum(cache: BucketCache, row: int) -> torch.Tensor:
    """Return the checksum of one request's buckets and node levels [] int64.

    Args:
        cache: The bucket cache.
        row: The request table row.

    Returns:
        The checksum, on the cache's device.
    """
    visible = cache.visible(row)
    rows = cache.logical_rows(row, visible)[:visible]
    counts = [cache.end[row], cache.closed_keys[row]]
    parts = [cache.bucket_centroids[rows], cache.bucket_members[rows], cache.bucket_sizes[rows]]
    hierarchy = None if cache.hierarchies is None else cache.hierarchies[row]
    if hierarchy is not None:
        counts += [hierarchy.covered_buckets, hierarchy.levels]
        for node_buckets in (*hierarchy.open_node_buckets, *hierarchy.closed_node_buckets):
            counts.append(len(node_buckets))
            parts.extend(node_buckets[name] for name in sorted(node_buckets.parts))
    if cache.graph_tree is not None:
        counts, parts = mirror_terms(cache.graph_tree, row, counts, parts)
    parts.append(torch.tensor(counts, dtype=torch.int64, device=cache.bucket_sizes.device))
    total = torch.zeros((), dtype=torch.int64, device=cache.bucket_sizes.device)
    for index, part in enumerate(parts):
        # An odd factor each - the same bytes in another part check differently
        total += checksum(part) * (2 * index + 1)
    return total


def check_same_index(
    cache: BucketCache, rows: Sequence[int], reduce: Callable[[torch.Tensor], torch.Tensor], size: int
) -> None:
    """Raise on every rank unless every rank of the group holds the same buckets and node levels for each request.

    Every rank calls it with the same rows, as it holds the same batch.

    Args:
        cache: This rank's bucket cache.
        rows: The batch's request table rows.
        reduce: Sums a tensor across the group.
        size: Ranks in the group.

    Raises:
        RuntimeError: Some rank's index differs from another's.
    """
    if not rows:
        return
    totals = torch.stack([request_checksum(cache, row) for row in rows])
    shifts = PIECE_BITS * torch.arange(PIECES, device=totals.device)
    # Each rank's checksum in exact fp32 pieces, so the sum is exact whatever the collective adds in
    local = ((totals[:, None] >> shifts) & ((1 << PIECE_BITS) - 1)).float()
    differs = (reduce(local.clone()) != local * size).any().float().reshape(1)
    if float(reduce(differs).item()):
        raise RuntimeError(
            f"the decode context-parallel group's ranks hold different indexes for request table rows {list(rows)}"
        )
