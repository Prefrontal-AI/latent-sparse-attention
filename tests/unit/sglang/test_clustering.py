"""Ward over cluster blocks - the buckets a closed block leaves, held to the clustering invariants.

Runs on a CPU through the torch rounds, which the kernel answers to, so these hold the contract of
`cluster_blocks` and the clustering invariants that live at this level. The kernel against
the rounds is `tests/unit/core/test_ward_kernel.py`, and every design against the freeze is `test_goldens.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import build_buckets
from lsa.sglang.index.clustering import cluster_blocks, ward_labels
from lsa.sglang.settings.design import IndexerConfig

WINDOW = 64
SMALL = {"window_size": WINDOW, "index_dim": 16, "bucket_size": 8, "bucket_cap": 16, "top_k": 64, "top_p": 16}


def spread_keys(count, seed=0, index_dim=16):
    """Index keys in runs of 12 around one anchor each, unit rows in bf16, so Ward has structure to find"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, index_dim, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.4 * torch.randn(count, index_dim, generator=generator)
    return F.normalize(keys, dim=-1).to(torch.bfloat16)


def clustered(keys, pool_size=4, first_position=0):
    return cluster_blocks(keys, IndexerConfig(**SMALL, pool_size=pool_size), first_position)


def buckets_of(members):
    """Every bucket's keys as a sorted tuple, the buckets sorted - a partition whatever its row order"""
    return sorted(tuple(sorted(row[row >= 0].tolist())) for row in members)


@pytest.mark.parametrize("pool_size", [4, None, 2])
def test_every_key_of_a_block_is_in_exactly_one_bucket_of_that_block(pool_size):
    # invariant 2
    blocks = 3
    result = clustered(spread_keys(blocks * WINDOW), pool_size)
    per_block = WINDOW // SMALL["bucket_size"]
    for block in range(blocks):
        members = result.bucket_members[block * per_block : (block + 1) * per_block]
        held = members[members >= 0].long()
        assert torch.equal(held.sort().values, torch.arange(block * WINDOW, (block + 1) * WINDOW))


@pytest.mark.parametrize("pool_size", [4, 2])
def test_a_pool_is_never_split_across_buckets(pool_size):
    # invariant 3
    result = clustered(spread_keys(4 * WINDOW, seed=1), pool_size)
    for members in result.bucket_members:
        held = members[members >= 0].long()
        for pool in held.div(pool_size, rounding_mode="floor").unique():
            assert int((held // pool_size == pool).sum()) == pool_size


@pytest.mark.parametrize("pool_size", [4, None])
def test_a_bucket_holds_at_least_a_pool_and_at_most_the_cap_and_a_block_ends_with_its_count(pool_size):
    # invariant 4
    blocks = 5
    result = clustered(spread_keys(blocks * WINDOW, seed=2), pool_size)
    assert len(result.bucket_members) == blocks * WINDOW // SMALL["bucket_size"]
    sizes = result.bucket_sizes
    assert int(sizes.min()) >= (pool_size or 1)
    assert int(sizes.max()) <= SMALL["bucket_cap"]
    # Ward averages the bucket size, and a loose block does not give every bucket the mean
    assert int(sizes.sum()) == blocks * WINDOW
    assert bool((sizes != SMALL["bucket_size"]).any())


def test_a_membership_row_is_exactly_its_keys_padded_and_its_size_counts_them():
    # invariant 5
    result = clustered(spread_keys(3 * WINDOW, seed=3))
    members = result.bucket_members
    assert members.dtype == torch.int32
    assert members.shape[1] == SMALL["bucket_cap"]
    assert torch.equal(result.bucket_sizes, (members >= 0).sum(1).int())
    # Padding comes last, and a row lists its keys in position order
    for row in members:
        held = row[row >= 0]
        assert torch.equal(held, held.sort().values)
        assert bool((row[len(held) :] == -1).all())


@pytest.mark.parametrize("pool_size", [4, None])
def test_a_centroid_is_the_normalized_sum_of_its_keys_and_of_its_pool_keys_alike(pool_size):
    # invariant 6
    keys = spread_keys(4 * WINDOW, seed=4)
    result = clustered(keys, pool_size)
    members = result.bucket_members.long()
    held = members >= 0
    from_keys = F.normalize((keys.float()[members.clamp_min(0)] * held[..., None]).sum(1), dim=-1)
    # Equal in bf16, the one rounding both are stored under
    assert torch.equal(result.bucket_centroids, from_keys.to(torch.bfloat16))
    assert result.bucket_centroids.dtype == torch.bfloat16


def test_ward_is_deterministic():
    # invariant 7
    keys = spread_keys(4 * WINDOW, seed=5)
    first, second = clustered(keys), clustered(keys)
    for one, other in zip(first, second, strict=True):
        assert torch.equal(one, other)


def test_a_block_clustered_alone_gives_the_buckets_it_gives_in_a_batch():
    # invariant 8
    keys = spread_keys(4 * WINDOW, seed=6)
    batch = clustered(keys)
    per_block = WINDOW // SMALL["bucket_size"]
    for block in range(4):
        alone = clustered(keys[block * WINDOW : (block + 1) * WINDOW], first_position=block * WINDOW)
        rows = slice(block * per_block, (block + 1) * per_block)
        assert torch.equal(alone.bucket_members, batch.bucket_members[rows])
        assert torch.equal(alone.bucket_centroids, batch.bucket_centroids[rows])


def test_a_cap_under_twice_the_bucket_size_is_refused():
    # invariant 9
    with pytest.raises(ValueError, match="at least twice bucket_size"):
        IndexerConfig(**{**SMALL, "bucket_cap": 15}, pool_size=None)
    pools = torch.randn(1, 64, 16)
    sizes = torch.ones(1, 64)
    with pytest.raises(ValueError, match="twice the bucket size"):
        ward_labels(pools, sizes, 8, 15)


def test_ward_never_stops_short_of_its_count_at_twice_the_bucket_size():
    # invariant 9, on blocks of near-identical keys, where the cap binds hardest
    keys = F.normalize(torch.ones(8 * WINDOW, 16) + 1e-3 * torch.randn(8 * WINDOW, 16), dim=-1).to(
        torch.bfloat16
    )
    result = clustered(keys)
    assert len(result.bucket_members) == 8 * WINDOW // SMALL["bucket_size"]
    assert int(result.bucket_sizes.max()) <= SMALL["bucket_cap"]


@pytest.mark.parametrize("pool_size", [4, None])
def test_the_clustering_is_the_torch_references(pool_size):
    keys = spread_keys(4 * WINDOW, seed=7)
    result = clustered(keys, pool_size)
    reference = build_buckets(keys, pool_size or 1, SMALL["bucket_size"], SMALL["bucket_cap"], WINDOW)
    assert buckets_of(result.bucket_members) == buckets_of(reference.bucket_members)
    assert torch.equal(result.bucket_members, reference.bucket_members)
    torch.testing.assert_close(result.bucket_centroids.float(), reference.bucket_centroids, rtol=0, atol=2e-2)


def test_positions_are_absolute_from_the_first_position_given():
    keys = spread_keys(2 * WINDOW, seed=8)
    result = clustered(keys, first_position=5 * WINDOW)
    held = result.bucket_members[result.bucket_members >= 0].long()
    assert torch.equal(held.sort().values, torch.arange(5 * WINDOW, 7 * WINDOW))
