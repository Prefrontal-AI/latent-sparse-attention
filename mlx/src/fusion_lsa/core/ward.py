import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)


def _nonzero_flat(mask: mx.array) -> tuple[mx.array, int]:
    # Indices where mask is true, ascending, plus the count - nonzero emulation via argsort
    flat = mask.reshape(-1)
    order = mx.argsort((~flat).astype(mx.int32))
    count = int(flat.sum().item())
    return order[:count], count


def pool_keys(index_keys: mx.array, pool_size: int) -> mx.array:
    # index_keys [G, T, index_dim] -> [G, T / pool_size, index_dim]
    blocks, key_count, index_dim = index_keys.shape
    if key_count % pool_size:
        raise ValueError(f"key_count {key_count} must be a multiple of pool_size {pool_size}")
    return index_keys.reshape(blocks, key_count // pool_size, pool_size, index_dim).sum(axis=2)


def ward_rounds(
    pool_keys: mx.array,
    pool_sizes: mx.array,
    bucket_count: int,
    bucket_cap: int,
    max_rounds: int = 256,
) -> tuple[mx.array, mx.array]:
    # Merge each block's pools down to bucket_count buckets; labels [G, P] int32, reached [G] bool
    blocks, pools, _ = pool_keys.shape
    key_sums = pool_keys.astype(mx.float32)
    bucket_sizes = pool_sizes.astype(mx.float32)
    alive = mx.ones((blocks, pools), dtype=mx.bool_)
    parent = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    index = mx.broadcast_to(mx.arange(pools), (blocks, pools))
    diagonal = mx.eye(pools, dtype=mx.bool_)
    rows = mx.arange(blocks)[:, None]
    cols = mx.arange(pools)[None, :]
    merges_done = 0
    for _ in range(max_rounds):
        merges_left = alive.sum(axis=1) - bucket_count
        if not bool((merges_left > 0).any()):
            break
        gram = key_sums @ key_sums.transpose(0, 2, 1)
        squares = mx.take_along_axis(gram, mx.arange(pools)[None, :, None], axis=2)[:, :, 0]
        merged_alignments = mx.maximum(squares[:, :, None] + squares[:, None, :] + 2 * gram, 0.0).sqrt()
        alignments = mx.maximum(squares, 0.0).sqrt()
        merge_costs = alignments[:, :, None] + alignments[:, None, :] - merged_alignments
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed = allowed & (bucket_sizes[:, :, None] + bucket_sizes[:, None, :] <= bucket_cap)
        masked = mx.where(allowed, merge_costs, mx.inf)
        best_costs = masked.min(axis=2)
        best = masked.argmin(axis=2)
        mutual = (best_costs < mx.inf) & (mx.take_along_axis(best, best, axis=1) == index) & (index < best)
        rank = mx.argsort(mx.argsort(mx.where(mutual, best_costs, mx.inf), axis=1), axis=1)
        mutual = mutual & (rank < mx.maximum(merges_left, 0)[:, None])
        flat_idx, count = _nonzero_flat(mutual)
        if count == 0:
            break
        block_ids = flat_idx // pools
        kept = flat_idx % pools
        absorbed = best[block_ids, kept]
        updates = key_sums[block_ids, absorbed]
        for i in range(count):
            row_hit = block_ids[i : i + 1] == rows
            kept_hit = row_hit & (cols == kept[i])
            abs_hit = row_hit & (cols == absorbed[i])
            key_sums = key_sums + kept_hit[:, :, None] * updates[i][None, None, :]
            bucket_sizes = bucket_sizes + mx.where(kept_hit, bucket_sizes[block_ids[i], absorbed[i]], 0.0)
            alive = alive & ~abs_hit
            parent = mx.where(abs_hit, mx.full(parent.shape, int(kept[i]), dtype=parent.dtype), parent)
        merges_done += count
    for _ in range(math.ceil(math.log2(max(pools, 2))) + 1):
        parent = mx.take_along_axis(parent, parent, axis=1)
    number = mx.cumsum(alive.astype(mx.int32), axis=1) - 1
    labels = mx.take_along_axis(number, parent, axis=1)
    reached = alive.sum(axis=1) <= bucket_count
    logger.debug("ward_rounds done: %d merges over %d blocks", merges_done, blocks)
    return labels, reached


def ward_merge(
    pool_keys: mx.array, pool_sizes: mx.array, bucket_count: int, bucket_cap: int
) -> mx.array:
    bucket_size = pool_sizes.sum(axis=1).max().item() / bucket_count
    if bucket_cap < 2 * bucket_size:
        raise ValueError(
            f"the bucket cap must be at least twice the bucket size, {2 * bucket_size:g} - got {bucket_cap}"
        )
    labels, reached = ward_rounds(pool_keys, pool_sizes, bucket_count, bucket_cap)
    if not bool(reached.all()):
        raise ValueError(
            f"{int((~reached).sum().item())} blocks stopped short of {bucket_count} buckets under the cap"
        )
    return labels


def members_from_labels(labels: mx.array, bucket_count: int, bucket_cap: int) -> mx.array:
    # labels [T] -> members [bucket_count, bucket_cap] int32, -1 padded, positions in order
    members = mx.full((bucket_count, bucket_cap), -1, dtype=mx.int32)
    for bucket in range(bucket_count):
        is_mem = labels == bucket
        count = int(is_mem.sum().item())
        if count == 0:
            continue
        if count > bucket_cap:
            raise ValueError(
                f"bucket {bucket} holds {count} keys against a cap of {bucket_cap} - labels are not a valid partition"
            )
        order = mx.argsort((~is_mem).astype(mx.int32))
        keys = order[:count]
        row = mx.full((1, bucket_cap), -1, dtype=mx.int32)
        row = mx.put_along_axis(row, mx.arange(count)[None, :], keys[None, :].astype(mx.int32), axis=1)
        members = mx.put_along_axis(members, mx.array([[bucket]]), row, axis=0)
    return members
