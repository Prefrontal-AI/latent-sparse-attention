import logging

import mlx.core as mx
import numpy as np
import pytest

from fusion_lsa.core.reference import build_buckets
from fusion_lsa.index.incremental import IncrementalBucketIndex

logger = logging.getLogger(__name__)

BLOCK = 512


def _keys(total: int, seed: int = 5) -> mx.array:
    keys = mx.random.normal((total, 64), key=mx.random.key(seed))
    return keys / mx.linalg.norm(keys, axis=1, keepdims=True)


def test_incremental_matches_batched():
    keys = _keys(BLOCK * 4)
    batched = build_buckets(keys, 4, 8, 16, BLOCK)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(4):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    incremental = index.buckets()
    assert np.array_equal(np.array(incremental.bucket_centroids), np.array(batched.bucket_centroids))
    assert np.array_equal(np.array(incremental.bucket_members), np.array(batched.bucket_members))
    assert np.array_equal(np.array(incremental.bucket_sizes), np.array(batched.bucket_sizes))


def test_incremental_refuses_partial_page():
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    with pytest.raises(ValueError, match="page"):
        index.append(_keys(BLOCK + 100))


def test_partial_tail_never_bucketed():
    # A key count that is not a whole cluster block leaves the tail outside the buckets
    keys = _keys(BLOCK * 3 + 200)
    batched = build_buckets(keys, 4, 8, 16, BLOCK)
    assert batched.bucket_centroids.shape[0] == 3 * (BLOCK // 8)
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    for block in range(3):
        index.append(keys[block * BLOCK : (block + 1) * BLOCK])
    with pytest.raises(ValueError, match="page"):
        index.append(keys[3 * BLOCK :])


def test_closed_blocks_stay_unchanged():
    index = IncrementalBucketIndex(pool_size=4, bucket_size=8, bucket_cap=16, cluster_block_size=BLOCK)
    index.append(_keys(BLOCK, seed=7))
    first = index.buckets()
    index.append(_keys(BLOCK, seed=8))
    second = index.buckets()
    assert np.array_equal(np.array(first.bucket_centroids), np.array(second.bucket_centroids[: first.bucket_centroids.shape[0]]))
    assert np.array_equal(np.array(first.bucket_members), np.array(second.bucket_members[: first.bucket_members.shape[0]]))
    assert second.bucket_centroids.shape[0] == first.bucket_centroids.shape[0] * 2
