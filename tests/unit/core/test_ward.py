"""Invariants 7, 8 and 9 - Ward over a cluster block's pools, the torch reference the kernel is held to."""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.ward import members_from_labels, pool_keys, ward_merge, ward_rounds


def pool_sums(blocks, pools, index_dim=16, pool_size=4, seed=0):
    """Each block's pools as sums of random unit keys [G, P, index_dim] fp32, and their key counts [G, P]"""
    generator = torch.Generator().manual_seed(seed)
    keys = F.normalize(torch.randn(blocks, pools * pool_size, index_dim, generator=generator), dim=-1)
    return pool_keys(keys, pool_size), torch.full((blocks, pools), float(pool_size))


def test_a_pool_key_is_the_sum_of_its_keys():
    keys = F.normalize(torch.randn(2, 12, 8), dim=-1)
    pooled = pool_keys(keys, 4)
    assert pooled.shape == (2, 3, 8)
    torch.testing.assert_close(pooled[1, 2], keys[1, 8:12].sum(0))


def test_every_block_ends_at_its_count_with_buckets_numbered_by_their_first_pool():
    sums, sizes = pool_sums(3, 64)
    labels = ward_merge(sums, sizes, 32, 16)
    for block in labels:
        assert block.unique().tolist() == list(range(32))
        first = [int((block == bucket).nonzero()[0]) for bucket in range(32)]
        assert first == sorted(first)


def test_no_bucket_passes_the_cap():
    sums, sizes = pool_sums(2, 64)
    # 64 pools into 32 buckets of at most 2 pools - every bucket a pair
    labels, reached = ward_rounds(sums, sizes, 32, 8)
    assert reached.all()
    for block in labels:
        assert torch.bincount(block).tolist() == [2] * 32


def test_the_tightest_pair_merges_first():
    base = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(1)), dim=-1)
    near = F.normalize(base[0] + 0.01 * base[1], dim=-1)
    sums = torch.stack([base[0], near, base[2], base[3]])[None] * 4
    labels = ward_merge(sums, torch.full((1, 4), 4.0), 3, 16)
    assert labels[0].tolist() == [0, 0, 1, 2]


def test_a_cap_under_twice_the_bucket_size_is_refused():
    sums, _ = pool_sums(2, 16)
    sizes = torch.full((2, 16), 16.0)
    with pytest.raises(ValueError, match="at least twice"):
        ward_merge(sums, sizes, 8, 16)
    # The merge itself reports the block short, and leaves every pool alone
    labels, reached = ward_rounds(sums, sizes, 8, 16)
    assert not reached.any()
    assert torch.equal(labels, torch.arange(16).expand(2, 16))


def test_a_block_merges_as_it_does_alone():
    sums, sizes = pool_sums(4, 32, seed=3)
    together = ward_merge(sums, sizes, 16, 16)
    alone = ward_merge(sums[2:3], sizes[2:3], 16, 16)
    assert torch.equal(together[2], alone[0])


def test_ward_is_deterministic_and_a_tie_goes_to_the_lower_pool():
    sums, sizes = pool_sums(2, 32, seed=4)
    assert torch.equal(ward_merge(sums, sizes, 16, 16), ward_merge(sums, sizes, 16, 16))
    # Four identical pools to two buckets - every cost ties, so pool 0 takes pool 1, then the pair takes pool 2
    # (a basis vector, so that every length is exact and every cost is exactly zero)
    same = torch.eye(8)[:1].expand(1, 4, 8) * 4
    assert ward_merge(same, torch.full((1, 4), 4.0), 2, 16).tolist() == [[0, 0, 0, 1]]


@pytest.mark.parametrize("pool_size", [1, 4])
def test_ward_never_stops_short_at_a_cap_of_twice_the_bucket_size(pool_size):
    # Random, near-copied and rounded pools, with and without pooling
    for seed in range(6):
        sums, sizes = pool_sums(8, 128, pool_size=pool_size, seed=seed)
        if seed % 3 == 1:
            sums = sums[:, :1].expand_as(sums) + 1e-3 * sums
        if seed % 3 == 2:
            sums = sums.round()
        labels = ward_merge(sums, sizes, 128 * pool_size // 8, 16)
        for block in labels:
            assert block.unique().numel() == 128 * pool_size // 8


def test_a_member_table_lists_each_buckets_keys_in_position_order():
    labels = torch.tensor([1, 0, 1, 2, 0, 1])
    members = members_from_labels(labels, 3, 4)
    assert members.tolist() == [[1, 4, -1, -1], [0, 2, 5, -1], [3, -1, -1, -1]]
    assert members.dtype == torch.int32
