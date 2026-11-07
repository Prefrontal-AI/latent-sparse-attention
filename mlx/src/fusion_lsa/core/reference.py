import logging
from typing import NamedTuple

import mlx.core as mx

from fusion_lsa.budget import EarlyBudget, budget_at
from fusion_lsa.core.ward import members_from_labels, pool_keys, ward_merge

logger = logging.getLogger(__name__)

NORM_TOLERANCE = {mx.float32: 1e-4, mx.bfloat16: 2e-2}
CLUSTER_ALGORITHMS = ("ward", "position")


class Buckets(NamedTuple):
    # Normalized key sum of each bucket [m, index_dim] fp32
    bucket_centroids: mx.array
    # Positions of each bucket's keys [m, bucket_cap] int32, -1 padded
    bucket_members: mx.array
    # Keys in each bucket [m] int32
    bucket_sizes: mx.array
    # Buckets a cluster block ends with
    block_bucket_count: int
    # Keys in a cluster block
    cluster_block_size: int


def check_unit_rows(name: str, vectors: mx.array) -> None:
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be [N, index_dim] - got shape {vectors.shape}")
    if vectors.dtype not in NORM_TOLERANCE:
        raise ValueError(f"{name} must be float32 or bfloat16 - got {vectors.dtype}")
    if not bool(mx.isfinite(vectors).all()):
        raise ValueError(f"{name} holds a non-finite value")
    if vectors.size:
        off = float(mx.abs(mx.linalg.norm(vectors.astype(mx.float32), axis=1) - 1).max().item())
        if off > NORM_TOLERANCE[vectors.dtype]:
            raise ValueError(f"{name} must be unit rows - a row is {off:.3g} off unit length")


def check_sizes(pool_size: int, bucket_size: int, bucket_cap: int, cluster_block_size: int) -> None:
    if min(pool_size, bucket_size, bucket_cap, cluster_block_size) < 1:
        raise ValueError("pool_size, bucket_size, bucket_cap and cluster_block_size must be positive")
    if bucket_size % pool_size or cluster_block_size % bucket_size:
        raise ValueError(
            f"pool_size {pool_size} must divide bucket_size {bucket_size}, which must divide "
            f"cluster_block_size {cluster_block_size}"
        )
    if bucket_cap < 2 * bucket_size:
        raise ValueError(f"bucket_cap must be at least twice bucket_size {bucket_size} - got {bucket_cap}")


def build_buckets(
    index_keys: mx.array,
    pool_size: int,
    bucket_size: int,
    bucket_cap: int,
    cluster_block_size: int,
    cluster_algorithm: str = "ward",
) -> Buckets:
    check_unit_rows("index_keys", index_keys)
    check_sizes(pool_size, bucket_size, bucket_cap, cluster_block_size)
    if cluster_algorithm not in CLUSTER_ALGORITHMS:
        raise ValueError(f"cluster_algorithm must be one of {CLUSTER_ALGORITHMS} - got {cluster_algorithm!r}")
    keys = index_keys.astype(mx.float32)
    key_count, index_dim = keys.shape
    block_count = key_count // cluster_block_size
    block_bucket_count = cluster_block_size // bucket_size
    blocks = keys[: block_count * cluster_block_size].reshape(block_count, cluster_block_size, index_dim)
    pools = pool_keys(blocks, pool_size)
    pool_sizes = mx.full(pools.shape[:2], pool_size, dtype=mx.float32)
    if cluster_algorithm == "position":
        labels = mx.broadcast_to(
            mx.arange(pools.shape[1]) // (bucket_size // pool_size), (block_count, pools.shape[1])
        )
    else:
        labels = ward_merge(pools, pool_sizes, block_bucket_count, bucket_cap)
    key_labels = mx.repeat(labels, pool_size, axis=1)
    bucket_ids = mx.arange(block_bucket_count)
    sum_rows = []
    for block in range(block_count):
        onehot = (key_labels[block][:, None] == bucket_ids[None, :]).astype(mx.float32)
        sum_rows.append(onehot.T @ blocks[block])
    key_sums = mx.concatenate(sum_rows, axis=0) if sum_rows else mx.zeros((0, index_dim), dtype=mx.float32)
    bucket_centroids = key_sums / mx.maximum(mx.linalg.norm(key_sums, axis=1, keepdims=True), 1e-30)
    member_rows = []
    for block in range(block_count):
        local = members_from_labels(key_labels[block], block_bucket_count, bucket_cap)
        member_rows.append(mx.where(local >= 0, local + block * cluster_block_size, -1))
    members = mx.concatenate(member_rows, axis=0) if member_rows else mx.zeros((0, bucket_cap), dtype=mx.int32)
    bucket_sizes = (members >= 0).sum(axis=1).astype(mx.int32)
    logger.debug(
        "build_buckets: %d blocks, %d buckets each, %d keys", block_count, block_bucket_count, key_count
    )
    return Buckets(bucket_centroids, members, bucket_sizes, block_bucket_count, cluster_block_size)


def read(
    index_queries: mx.array,
    positions: mx.array,
    buckets: Buckets,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
    early_budget: EarlyBudget | None = None,
) -> tuple[mx.array, mx.array]:
    check_unit_rows("index_queries", index_queries)
    if index_queries.shape[1] != buckets.bucket_centroids.shape[1]:
        raise ValueError(
            f"index_queries have {index_queries.shape[1]} dims and the buckets {buckets.bucket_centroids.shape[1]}"
        )
    if positions.dtype not in (mx.int32, mx.int64, mx.uint32) or positions.shape != (index_queries.shape[0],):
        raise ValueError("positions must be an int32 or int64 vector, one a query")
    if positions.size and int(positions.min().item()) < 0:
        raise ValueError("positions must be non-negative")
    if min(top_p, top_k, window_size) < 1 or sink_size < 0:
        raise ValueError("top_p, top_k and window_size must be positive and sink_size non-negative")
    if early_budget is not None and (early_budget.top_p > top_p or early_budget.top_k > top_k):
        raise ValueError(
            f"the early budget is at most top_p {top_p} and top_k {top_k} - "
            f"got top_p {early_budget.top_p} and top_k {early_budget.top_k}"
        )
    width = top_k + window_size + sink_size
    scores_all = index_queries.astype(mx.float32) @ buckets.bucket_centroids.T
    selected_rows = []
    lengths = []
    for row, position in enumerate(positions.tolist()):
        row_top_k, row_top_p = budget_at(position, top_k, top_p, early_budget)
        keys = _read_row(scores_all[row], position, buckets, row_top_p, row_top_k, sink_size, window_size)
        padded = mx.full((1, width), -1, dtype=mx.int32)
        padded = mx.put_along_axis(
            padded, mx.arange(keys.shape[0])[None, :], keys[None, :].astype(mx.int32), axis=1
        )[0]
        selected_rows.append(padded)
        lengths.append(keys.shape[0])
    selected = mx.stack(selected_rows, axis=0) if selected_rows else mx.zeros((0, width), dtype=mx.int32)
    lengths_out = mx.array(lengths, dtype=mx.int32) if lengths else mx.zeros((0,), dtype=mx.int32)
    return selected, lengths_out


def _read_row(
    scores: mx.array,
    position: int,
    buckets: Buckets,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
) -> mx.array:
    visible = min(position // buckets.cluster_block_size * buckets.block_bucket_count, scores.shape[0])
    picked = mx.argsort(-scores[:visible])[:top_p]
    return expand_buckets(picked, position, buckets, top_k, sink_size, window_size)


def expand_buckets(
    picked: mx.array, position: int, buckets: Buckets, top_k: int, sink_size: int, window_size: int
) -> mx.array:
    window_start = max(0, position + 1 - window_size)
    members = buckets.bucket_members[picked.astype(mx.int32)].reshape(-1)
    keep = (members >= sink_size) & (members < window_start)
    order = mx.argsort((~keep).astype(mx.int32))
    kept_count = min(int(keep.sum().item()), top_k)
    members = mx.take(members, order[:kept_count])
    window = mx.arange(window_start, position + 1).astype(mx.int32)
    sinks = mx.arange(min(sink_size, window_start, position + 1)).astype(mx.int32)
    return mx.concatenate([members.astype(mx.int32), window, sinks])
