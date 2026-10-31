"""The Ward kernels against the torch rounds they are the GPU form of - merge, pairing, and the member table.

The kernel tests need a GPU. The member table is torch and runs anywhere.
"""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lsa.core.kernels.triton.ward import (
    _square_sum,
    bucket_members_from_labels,
    cluster_node_buckets,
    pair_leftovers,
    ward_merge,
)
from lsa.core.ward import members_from_labels, ward_rounds

DEVICE = "cuda"
gpu = pytest.mark.gpu


@triton.jit
def _square_sum_kernel(vectors_ptr, out_ptr, BLOCK: tl.constexpr, D: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    vectors = tl.load(vectors_ptr + rows[:, None] * D + tl.arange(0, D)[None, :])
    tl.store(out_ptr + rows, _square_sum(vectors, BLOCK, D))


def square_sum_in_order(vectors):
    """Each row's squared length in the kernel's order, every step rounded to fp32 once [R] fp32"""

    def rounded(value):
        return value.float().double()

    runs = vectors.double().view(vectors.shape[0], -1, 4)
    # fp64 holds a product of two fp32 exactly => an fma is the fp64 sum rounded to fp32
    sums = rounded(runs[..., 1] ** 2)
    for dim in (0, 2, 3):
        sums = rounded(runs[..., dim] ** 2 + sums)
    while sums.shape[1] > 1:
        half = sums.shape[1] // 2
        sums = rounded(sums[:, :half] + sums[:, half:])
    return sums[:, 0].float()


def pool_sums(blocks, pools, walk=False, seed=0):
    """Each block's pools as sums of 4 unit keys [G, P, 64] fp32 - independent, or a smooth walk as real keys drift"""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(blocks, pools * 4, 64, device=DEVICE, generator=generator)
    keys = F.normalize(torch.cumsum(noise * 0.3, 1) if walk else noise, dim=-1)
    return keys.view(blocks, pools, 4, 64).sum(2), torch.full((blocks, pools), 4.0, device=DEVICE)


def co_membership(labels):
    return labels[:, :, None] == labels[:, None, :]


@gpu
def test_pairing_is_the_merge_to_half_the_buckets_under_a_cap_of_two():
    for blocks, buckets, walk in ((5, 64, False), (2, 1024, True), (1, 4096, False)):
        key_sums, _ = pool_sums(blocks, buckets, walk)
        want, _ = ward_rounds(key_sums, torch.ones(blocks, buckets, device=DEVICE), buckets // 2, 2)
        got = cluster_node_buckets(key_sums, precision="ieee")
        assert torch.equal(got, want)
        sizes = torch.zeros(blocks, buckets // 2, device=DEVICE).scatter_add_(
            1, got, torch.ones_like(got).float()
        )
        assert bool((sizes == 2).all())


@gpu
def test_near_identical_buckets_are_all_paired_each_once():
    key_sums, _ = pool_sums(1, 4096)
    # A repeated passage - one bucket's sum 4,096 times, a rounding's worth apart, beside a block of distinct ones
    repeated = key_sums[:, :1].expand(1, 4096, 64) + 1e-6 * torch.randn(1, 4096, 64, device=DEVICE)
    for block in (repeated, torch.cat([repeated, key_sums])):
        labels = cluster_node_buckets(block.contiguous())
        sizes = torch.zeros(block.shape[0], block.shape[1] // 2, device=DEVICE).scatter_add_(
            1, labels, torch.ones_like(labels).float()
        )
        assert bool((sizes == 2).all())


@gpu
def test_buckets_left_open_are_paired_in_order():
    partner = torch.tensor([[1, 0, -1, -1, 5, 4, -1, -1], [1, 0, 3, 2, 5, 4, 7, 6]], device=DEVICE)
    pair_leftovers(partner)
    assert partner.tolist() == [[1, 0, 3, 2, 5, 4, 7, 6], [1, 0, 3, 2, 5, 4, 7, 6]]


@gpu
def test_the_kernel_merges_as_the_torch_rounds_do():
    for blocks, pools, walk in ((16, 128, False), (16, 128, True), (3, 2048, False)):
        key_sums, sizes = pool_sums(blocks, pools, walk)
        want, want_reached = ward_rounds(key_sums, sizes, pools // 2, 16)
        got, reached = ward_merge(key_sums, sizes, pools // 2, 16, precision="ieee")
        assert torch.equal(reached, want_reached)
        assert torch.equal(got, want), f"{blocks} blocks of {pools} pools"


@gpu
def test_near_ties_part_the_kernel_from_the_torch_rounds_at_no_cost():
    # A smooth walk makes neighbouring pools near-copies - the two sum a dot in different orders, so a tie
    # within rounding goes either way and the rounds after it follow
    key_sums, sizes = pool_sums(3, 2048, walk=True)
    want, _ = ward_rounds(key_sums, sizes, 1024, 16)
    got, _ = ward_merge(key_sums, sizes, 1024, 16, precision="ieee")
    moved = (co_membership(got) != co_membership(want)).any(2).float().mean().item()
    assert moved < 0.02

    def resultant_length(labels):
        summed = torch.zeros(3, 1024, 64, device=DEVICE).scatter_add_(
            1, labels[..., None].expand_as(key_sums), key_sums
        )
        return summed.norm(dim=-1).sum().item()

    assert resultant_length(got) / resultant_length(want) == pytest.approx(1.0, abs=1e-5)


@gpu
def test_tf32x3_merges_the_same_pools_almost_everywhere():
    key_sums, sizes = pool_sums(16, 128, walk=True, seed=1)
    exact, _ = ward_merge(key_sums, sizes, 64, 16, precision="ieee")
    fast, _ = ward_merge(key_sums, sizes, 64, 16)
    agreement = (co_membership(exact) == co_membership(fast)).float().mean().item()
    assert agreement > 0.999


@gpu
def test_squared_lengths_round_in_one_order_whatever_the_build():
    # Near-tied merge costs turn on a square's last bit => the kernel's squares are this order exactly
    for index_dim in (16, 64, 256):
        generator = torch.Generator(device=DEVICE).manual_seed(index_dim)
        vectors = F.normalize(torch.randn(64, 4, index_dim, device=DEVICE, generator=generator), dim=-1).sum(
            1
        )
        got = torch.empty(64, device=DEVICE)
        _square_sum_kernel[(1,)](vectors, got, BLOCK=64, D=index_dim)
        assert torch.equal(got, square_sum_in_order(vectors)), f"index_dim {index_dim}"


@gpu
def test_every_block_ends_at_its_count_under_the_cap():
    key_sums, sizes = pool_sums(8, 256, walk=True, seed=2)
    labels, reached = ward_merge(key_sums, sizes, 96, 16)
    assert bool(reached.all())
    for block in labels:
        assert block.unique().tolist() == list(range(96))
        assert int(torch.bincount(block).max()) * 4 <= 16


@gpu
def test_a_cap_that_leaves_no_pair_reports_the_block_short():
    key_sums, _ = pool_sums(2, 64)
    labels, reached = ward_merge(key_sums, torch.full((2, 64), 16.0, device=DEVICE), 32, 16)
    assert not bool(reached.any())
    assert torch.equal(labels, torch.arange(64, device=DEVICE).expand(2, 64))


def test_the_member_table_from_labels_is_the_references():
    generator = torch.Generator().manual_seed(0)
    labels = torch.stack([torch.randperm(64, generator=generator) // 4 for _ in range(3)])
    got = bucket_members_from_labels(labels, 16, 16)
    assert got.dtype == torch.int32
    for block in range(3):
        assert torch.equal(got[block], members_from_labels(labels[block], 16, 16))
