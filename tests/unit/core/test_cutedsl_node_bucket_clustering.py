"""The CuTeDSL growth pairing against the Triton rounds it replaces.

One Hopper or Blackwell GPU. Both take each merge cost as the same tf32x3 dot bit for bit, so every round pairs the same buckets, the
check every eight rounds stops at the same round, and the buckets left pair in the same order.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import dispatch, ward
from lsa.core.kernels.cutedsl import node_bucket_clustering
from lsa.core.kernels.triton import ward as triton_ward
from lsa.sglang.index.hierarchy import pair_into_node_buckets

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# A consumer Blackwell, which the dispatch table names for no step
UNNAMED_GPU = (12, 0)


def pool_sums(blocks, pools, walk=False, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(blocks, pools * 4, 64, device=DEVICE, generator=generator)
    keys = F.normalize(torch.cumsum(noise * 0.3, 1) if walk else noise, dim=-1)
    return keys.view(blocks, pools, 4, 64).sum(2).contiguous()


def near_copies(pools, seed=5):
    # A repeated passage - one bucket's sum over and over, a rounding's worth apart - pairs a bucket or two a round
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    one = pool_sums(1, pools)[:, :1]
    return (one + 1e-6 * torch.randn(1, pools, 64, device=DEVICE, generator=generator)).contiguous()


@pytest.mark.parametrize(
    "blocks, pools, walk, seed",
    [
        (1, 64, False, 0),
        (5, 64, False, 1),
        (1, 128, True, 0),
        (2, 1024, True, 0),
        (3, 2048, True, 2),
        (1, 4096, False, 0),
    ],
)
def test_the_pairs_are_the_triton_rounds_pairs(blocks, pools, walk, seed):
    key_sums = pool_sums(blocks, pools, walk, seed)
    assert torch.equal(
        triton_ward.number_pairs(node_bucket_clustering.cluster_node_buckets(key_sums)),
        ward.cluster_node_buckets(key_sums),
    )


def test_one_cta_a_block_holding_64_tiles_pairs_as_the_triton_rounds_do():
    blocks = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    key_sums = pool_sums(blocks, 4096, walk=True, seed=6)
    assert torch.equal(
        triton_ward.number_pairs(node_bucket_clustering.cluster_node_buckets(key_sums)),
        ward.cluster_node_buckets(key_sums),
    )


def test_near_copies_stall_and_pair_their_leftovers_as_the_triton_rounds_do():
    copies = near_copies(4096)
    for key_sums in (copies, torch.cat([copies, pool_sums(1, 4096)]).contiguous()):
        partner = node_bucket_clustering.cluster_node_buckets(key_sums)
        assert torch.equal(triton_ward.number_pairs(partner), ward.cluster_node_buckets(key_sums))
        index = torch.arange(key_sums.shape[1], device=DEVICE)
        assert bool((partner.gather(1, partner) == index).all())


def test_every_bucket_is_paired_once():
    partner = node_bucket_clustering.cluster_node_buckets(pool_sums(4, 512, walk=True, seed=3))
    index = torch.arange(512, device=DEVICE)
    assert bool((partner >= 0).all())
    assert bool((partner != index).all())
    assert bool((partner.gather(1, partner) == index).all())


def test_cluster_node_buckets_runs_the_cutedsl_kernel_when_asked(monkeypatch):
    calls = []
    real = node_bucket_clustering.cluster_node_buckets
    monkeypatch.setattr(
        node_bucket_clustering, "cluster_node_buckets", lambda key_sums: calls.append(1) or real(key_sums)
    )
    key_sums = pool_sums(2, 256, walk=True)
    want = ward.cluster_node_buckets(key_sums)
    assert not calls
    assert torch.equal(ward.cluster_node_buckets(key_sums, kernels="cutedsl"), want)
    assert calls == [1]


def test_a_growth_pairs_into_the_same_node_buckets():
    key_sums = pool_sums(2, 1024, walk=True, seed=4)
    want = pair_into_node_buckets(key_sums, 4)
    assert torch.equal(pair_into_node_buckets(key_sums, 4, "cutedsl"), want)


def test_blocks_the_kernel_does_not_take_go_to_triton(monkeypatch):
    calls = []
    monkeypatch.setattr(node_bucket_clustering, "cluster_node_buckets", lambda key_sums: calls.append(1))
    key_sums = pool_sums(1, 256)
    ward.cluster_node_buckets(key_sums, precision="ieee", kernels="cutedsl")
    ward.cluster_node_buckets(key_sums, rounds_a_check=4, kernels="cutedsl")
    ward.cluster_node_buckets(pool_sums(1, 32), kernels="cutedsl")
    monkeypatch.setattr(dispatch, "capability", lambda device: UNNAMED_GPU)
    ward.cluster_node_buckets(key_sums, kernels="cutedsl")
    assert not calls


def test_blocks_the_kernel_does_not_take_are_refused():
    assert not node_bucket_clustering.supports(pool_sums(1, 32), "tf32x3")
    assert not node_bucket_clustering.supports(pool_sums(1, 256), "ieee")
    assert not node_bucket_clustering.supports(torch.zeros(1, 256, 48, device=DEVICE), "tf32x3")
    with pytest.raises(ValueError, match="64 to 4,096 buckets"):
        node_bucket_clustering.cluster_node_buckets(torch.zeros(1, 96, 64, device=DEVICE))
