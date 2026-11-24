"""The torch reference on CUDA tensors - the buckets, the hierarchy and both reads give the CPU's answers, on the GPU.

One GPU.
Every tensor the reference makes sits on its input's device, so a caller on the GPU never mixes devices.
The reads run on the CPU's buckets and hierarchy moved over, so the only arithmetic that differs is their own.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import build_buckets, build_hierarchy, read, read_hierarchy_counted
from lsa.core.hierarchy import Scorer, beam_pick, growth_at, per_level
from lsa.core.ward import members_from_labels

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
KEYS = 8192
WINDOW = 64
SINKS = 4
TOP_K = 64
TOP_P = 16
PAGE = 512
NODE_BLOCK = 16


def spread_keys(count, seed=0):
    """Index keys in runs of 12 around one anchor each, so groups of buckets have structure to pair"""
    generator = torch.Generator().manual_seed(seed)
    anchors = torch.randn(count // 12 + 1, 16, generator=generator)
    keys = anchors[torch.arange(count) // 12] + 0.5 * torch.randn(count, 16, generator=generator)
    return F.normalize(keys, dim=-1)


def queries_near(keys, positions, seed=1):
    """A query near an earlier key, one a position"""
    generator = torch.Generator().manual_seed(seed)
    earlier = (torch.rand(len(positions), generator=generator) * positions).long()
    return F.normalize(keys[earlier] + 0.1 * torch.randn(len(positions), 16, generator=generator), dim=-1)


def on_gpu(value):
    """A tensor, a list of them or a tuple of fields, moved to the GPU"""
    if isinstance(value, torch.Tensor):
        return value.to(DEVICE)
    if isinstance(value, list):
        return [on_gpu(item) for item in value]
    if isinstance(value, tuple):
        fields = [on_gpu(field) for field in value]
        return type(value)(*fields) if hasattr(value, "_fields") else tuple(fields)
    return value


def built_on(keys, device, cluster_algorithm):
    on_device = keys.to(device)
    buckets = build_buckets(on_device, 4, 8, 16, WINDOW, cluster_algorithm=cluster_algorithm)
    hierarchy = build_hierarchy(
        on_device, buckets, 16, 4, NODE_BLOCK, node_cluster_algorithm=cluster_algorithm
    )
    return on_device, buckets, hierarchy


@pytest.fixture(scope="module", params=["ward", "position"])
def stores(request):
    """The store built on the CPU, built on the GPU, and the CPU's moved to the GPU"""
    keys = spread_keys(KEYS)
    on_cpu = built_on(keys, "cpu", request.param)
    moved = on_gpu(on_cpu)
    assert moved[1].bucket_centroids.is_cuda and moved[2].node_centroids[0].is_cuda
    return on_cpu, built_on(keys, DEVICE, request.param), moved


def test_a_buckets_members_sit_on_their_labels_device():
    labels = torch.tensor([2, 0, 1, 0, 2, 2], device=DEVICE)
    members = members_from_labels(labels, 3, 4)
    assert members.device.type == "cuda"
    assert torch.equal(members.cpu(), members_from_labels(labels.cpu(), 3, 4))


def test_buckets_built_on_the_gpu_are_the_ones_built_on_the_cpu(stores):
    (_, on_cpu, _), (_, on_cuda, _), _ = stores
    assert on_cuda.bucket_members.device.type == "cuda"
    assert torch.equal(on_cuda.bucket_members.cpu(), on_cpu.bucket_members)
    assert torch.equal(on_cuda.bucket_sizes.cpu(), on_cpu.bucket_sizes)
    torch.testing.assert_close(on_cuda.bucket_centroids.cpu(), on_cpu.bucket_centroids)


def test_a_hierarchy_built_on_the_gpu_is_the_one_built_on_the_cpu(stores):
    (_, _, on_cpu), (_, _, on_cuda), _ = stores
    assert len(on_cuda.children) == len(on_cpu.children) > 1
    for children, expected in zip(on_cuda.children, on_cpu.children, strict=True):
        assert children.device.type == "cuda"
        assert torch.equal(children.cpu(), expected)
    for centroids, expected in zip(on_cuda.node_centroids, on_cpu.node_centroids, strict=True):
        torch.testing.assert_close(centroids.cpu(), expected)


def test_a_beam_on_the_gpu_picks_the_buckets_it_picks_on_the_cpu(stores):
    (keys, buckets, hierarchy), _, (_, gpu_buckets, gpu_hierarchy) = stores
    cpu_scorer = Scorer(buckets, hierarchy, "centroid", "centroid", 1.0, None)
    gpu_scorer = Scorer(gpu_buckets, gpu_hierarchy, "centroid", "centroid", 1.0, None)
    page_buckets = PAGE // buckets.cluster_block_size * buckets.block_bucket_count
    beam_widths = per_level(4, len(hierarchy.node_cluster_block_sizes))
    positions = torch.tensor([PAGE, 3000, KEYS - 1])
    for position, query in zip(positions.tolist(), queries_near(keys, positions), strict=True):
        growth = growth_at(hierarchy, position // PAGE * page_buckets, page_buckets, 0)
        expected = beam_pick(query, position, buckets, growth, beam_widths, None, TOP_P, cpu_scorer)
        picked = beam_pick(
            query.to(DEVICE), position, gpu_buckets, growth, beam_widths, None, TOP_P, gpu_scorer
        )
        assert picked.device.type == "cuda"
        assert torch.equal(picked.cpu(), expected)


def test_a_read_on_the_gpu_selects_what_it_selects_on_the_cpu(stores):
    (keys, buckets, _), _, (_, gpu_buckets, _) = stores
    positions = torch.arange(WINDOW, KEYS, 997)
    queries = queries_near(keys, positions)
    expected = read(queries, positions, buckets, TOP_P, TOP_K, SINKS, WINDOW)
    selected = read(queries.to(DEVICE), positions.to(DEVICE), gpu_buckets, TOP_P, TOP_K, SINKS, WINDOW)
    for got, want in zip(selected, expected, strict=True):
        assert got.device.type == "cuda"
        assert torch.equal(got.cpu(), want)


def test_a_counted_beam_read_on_the_gpu_is_the_one_on_the_cpu(stores):
    (keys, buckets, hierarchy), _, (_, gpu_buckets, gpu_hierarchy) = stores
    positions = torch.arange(WINDOW, KEYS, 997)
    queries = queries_near(keys, positions)
    options = {
        "beam_width": 4,
        "start_nodes": None,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "sink_size": SINKS,
        "window_size": WINDOW,
        "page_size": PAGE,
        "growth_buckets": 0,
    }
    expected = read_hierarchy_counted(queries, positions, buckets, hierarchy, **options)
    counted = read_hierarchy_counted(
        queries.to(DEVICE), positions.to(DEVICE), gpu_buckets, gpu_hierarchy, **options
    )
    for got, want in zip(counted, expected, strict=True):
        assert got.device.type == "cuda"
        assert torch.equal(got.cpu(), want)
