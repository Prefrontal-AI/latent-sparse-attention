"""The hierarchy's build and beam, and the members of a bucket, on CUDA tensors - the same answers as on the CPU.

One GPU. Every tensor these make sits on their input's device, so a caller on the GPU never mixes devices.
The buckets come from the CPU and are moved over, since `build_buckets` is not yet device-aware.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import build_buckets, build_hierarchy
from lsa.core.hierarchy import Scorer, beam_pick, growth_at, per_level
from lsa.core.reference import Buckets
from lsa.core.ward import members_from_labels

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
KEYS = 8192
WINDOW = 64
PAGE = 512
NODE_BLOCK = 16


def spread_keys(count, seed=0):
    """Index keys in runs of 12 around one anchor each, so groups of buckets have structure to pair"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, 16, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 16, generator=generator)
    return F.normalize(keys, dim=-1)


def on_gpu(buckets):
    return Buckets(*(field.to(DEVICE) if isinstance(field, torch.Tensor) else field for field in buckets))


@pytest.fixture(scope="module")
def store():
    keys = spread_keys(KEYS)
    buckets = build_buckets(keys, 4, 8, 16, WINDOW)
    return keys, buckets


def test_a_buckets_members_sit_on_their_labels_device():
    labels = torch.tensor([2, 0, 1, 0, 2, 2], device=DEVICE)
    members = members_from_labels(labels, 3, 4)
    assert members.device.type == "cuda"
    assert torch.equal(members.cpu(), members_from_labels(labels.cpu(), 3, 4))


@pytest.mark.parametrize("node_cluster_algorithm", ["ward", "position"])
def test_a_hierarchy_built_on_the_gpu_is_the_one_built_on_the_cpu(store, node_cluster_algorithm):
    keys, buckets = store
    on_cpu = build_hierarchy(keys, buckets, 16, 4, NODE_BLOCK, node_cluster_algorithm=node_cluster_algorithm)
    on_cuda = build_hierarchy(
        keys.to(DEVICE), on_gpu(buckets), 16, 4, NODE_BLOCK, node_cluster_algorithm=node_cluster_algorithm
    )
    assert len(on_cuda.children) == len(on_cpu.children) > 1
    for children, expected in zip(on_cuda.children, on_cpu.children, strict=True):
        assert children.device.type == "cuda"
        assert torch.equal(children.cpu(), expected)
    for centroids, expected in zip(on_cuda.node_centroids, on_cpu.node_centroids, strict=True):
        torch.testing.assert_close(centroids.cpu(), expected)


def test_a_beam_on_the_gpu_picks_the_buckets_it_picks_on_the_cpu(store):
    keys, buckets = store
    hierarchy = build_hierarchy(keys, buckets, 16, 4, NODE_BLOCK)
    gpu_buckets = on_gpu(buckets)
    gpu_hierarchy = build_hierarchy(keys.to(DEVICE), gpu_buckets, 16, 4, NODE_BLOCK)
    cpu_scorer = Scorer(buckets, hierarchy, "centroid", "centroid", 1.0, None)
    gpu_scorer = Scorer(gpu_buckets, gpu_hierarchy, "centroid", "centroid", 1.0, None)
    page_buckets = PAGE // buckets.cluster_block_size * buckets.block_bucket_count
    beam_widths = per_level(4, len(hierarchy.node_cluster_block_sizes))
    generator = torch.Generator().manual_seed(1)
    for position in (PAGE, 3000, KEYS - 1):
        query = F.normalize(keys[position // 2] + 0.1 * torch.randn(16, generator=generator), dim=-1)
        growth = growth_at(hierarchy, position // PAGE * page_buckets, page_buckets, 0)
        expected = beam_pick(query, position, buckets, growth, beam_widths, None, 16, cpu_scorer)
        picked = beam_pick(query.to(DEVICE), position, gpu_buckets, growth, beam_widths, None, 16, gpu_scorer)
        assert picked.device.type == "cuda"
        assert torch.equal(picked.cpu(), expected)
