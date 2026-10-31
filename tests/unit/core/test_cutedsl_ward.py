"""The CuTeDSL Ward merge against the Triton merge it replaces.

The merge needs one Hopper or Blackwell GPU, bar the checks of its inputs.
Both take each merge cost as the same tf32x3 dot and each merged pool's squared length in the same order,
so every round merges the same pools and every pool gets the same label.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import dispatch, ward
from lsa.core.kernels.cutedsl import ward as cutedsl_ward
from lsa.core.ward import pool_keys

DEVICE = "cuda"
# A consumer Blackwell, which the dispatch table names for no step
UNNAMED_GPU = (12, 0)
POOL_SIZE = 4


def blocks_of(blocks, window, walk=False, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn(blocks, window, 64, device=DEVICE, generator=generator)
    keys = F.normalize(torch.cumsum(noise * 0.3, 1) if walk else noise, dim=-1)
    pools = pool_keys(keys, POOL_SIZE)
    return pools, torch.full(pools.shape[:2], float(POOL_SIZE), device=DEVICE)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("blocks", "window", "walk", "seed"),
    [
        (16, 512, True, 0),
        (11, 512, False, 1),
        (3, 256, True, 2),
        (64, 512, True, 3),
        (1, 512, True, 7),
        (16, 1024, True, 8),
        (64, 1024, False, 9),
        (1, 2048, True, 10),
        (16, 2048, True, 11),
        (64, 2048, False, 12),
    ],
    ids=[
        "served",
        "independent",
        "64 pools",
        "many blocks",
        "one block split",
        "256 pools",
        "256 pools in fewer CTAs",
        "512 pools",
        "512 pools unpooled served",
        "512 pools in fewer CTAs",
    ],
)
def test_the_labels_are_the_triton_merges(blocks, window, walk, seed):
    pools, sizes = blocks_of(blocks, window, walk, seed)
    want = ward.cluster_ward(pools, sizes, window // 8, 16)
    got = cutedsl_ward.cluster_ward(pools, sizes, window // 8, 16)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@pytest.mark.gpu
def test_near_copies_merge_the_cheapest_pairs_first_as_the_triton_merge_does():
    # A repeated passage - more pairs mutual in a round than the count leaves room for
    pools, sizes = blocks_of(4, 512, seed=5)
    pools = (pools[:, :1] + 1e-4 * pools).contiguous()
    for count in (64, 40):
        want = ward.cluster_ward(pools, sizes, count, 64)
        got = cutedsl_ward.cluster_ward(pools, sizes, count, 64)
        assert torch.equal(got[0], want[0])
        assert torch.equal(got[1], want[1])


@pytest.mark.gpu
def test_a_cap_that_stops_a_block_short_is_reported_as_the_triton_merge_reports_it():
    pools, sizes = blocks_of(2, 512, walk=True, seed=6)
    want = ward.cluster_ward(pools, sizes, 20, 16)
    got = cutedsl_ward.cluster_ward(pools, sizes, 20, 16)
    assert not bool(want[1].any())
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@pytest.mark.gpu
def test_cluster_ward_runs_the_cutedsl_merge_when_asked(monkeypatch):
    calls = []
    real = cutedsl_ward.cluster_ward
    monkeypatch.setattr(cutedsl_ward, "cluster_ward", lambda *arguments: calls.append(1) or real(*arguments))
    pools, sizes = blocks_of(4, 512, walk=True)
    want = ward.cluster_ward(pools, sizes, 64, 16)
    assert not calls
    got = ward.cluster_ward(pools, sizes, 64, 16, kernels="cutedsl")
    assert calls == [1]
    assert torch.equal(got[0], want[0])


@pytest.mark.gpu
def test_blocks_the_kernel_does_not_take_go_to_triton(monkeypatch):
    calls = []
    monkeypatch.setattr(cutedsl_ward, "cluster_ward", lambda *arguments: calls.append(1))
    pools, sizes = blocks_of(2, 512)
    ward.cluster_ward(pools, sizes, 64, 16, precision="ieee", kernels="cutedsl")
    wide, wide_sizes = blocks_of(2, 4096)
    ward.cluster_ward(wide, wide_sizes, 512, 16, kernels="cutedsl")
    many, many_sizes = blocks_of(torch.cuda.get_device_properties(DEVICE).multi_processor_count + 1, 512)
    ward.cluster_ward(many, many_sizes, 64, 16, kernels="cutedsl")
    monkeypatch.setattr(dispatch, "capability", lambda device: UNNAMED_GPU)
    ward.cluster_ward(pools, sizes, 64, 16, kernels="cutedsl")
    assert not calls


@pytest.mark.gpu
def test_more_blocks_than_sms_warn_once_for_each_shape(monkeypatch, caplog):
    monkeypatch.setattr(ward, "WARNED", set())
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(multi_processor_count=2)
    )
    pools, sizes = blocks_of(4, 512)
    with caplog.at_level(logging.WARNING, logger=ward.logger.name):
        for _ in range(2):
            ward.cluster_ward(pools, sizes, 64, 16, kernels="cutedsl")
        assert len(caplog.records) == 1
        assert "4 blocks of 128 pools" in caplog.text
        assert "2 SMs" in caplog.text
        other, other_sizes = blocks_of(3, 512)
        ward.cluster_ward(other, other_sizes, 64, 16, kernels="cutedsl")
        ward.cluster_ward(pools, sizes, 64, 16)
    assert len(caplog.records) == 2


@pytest.mark.gpu
def test_blocks_the_kernel_does_not_take_are_refused():
    assert not cutedsl_ward.supports(torch.zeros(1, 1024, 64, device=DEVICE), "tf32x3")
    assert not cutedsl_ward.supports(torch.zeros(1, 128, 64, device=DEVICE), "ieee")
    assert not cutedsl_ward.supports(torch.zeros(1, 128, 48, device=DEVICE), "tf32x3")
    with pytest.raises(ValueError, match="64 to 512 pools"):
        cutedsl_ward.cluster_ward(
            torch.zeros(1, 32, 64, device=DEVICE), torch.ones(1, 32, device=DEVICE), 16, 16
        )


@pytest.mark.gpu
@pytest.mark.parametrize("count", [128, 200])
def test_a_count_the_pools_already_meet_merges_nothing(count):
    pools, sizes = blocks_of(4, 512, walk=True, seed=13)
    want = ward.cluster_ward(pools, sizes, count, 16)
    got = cutedsl_ward.cluster_ward(pools, sizes, count, 16)
    assert torch.equal(got[0], torch.arange(128, device=DEVICE).expand(4, 128))
    assert bool(got[1].all())
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_pool_sizes_that_are_not_one_to_a_pool_are_refused():
    keys = torch.zeros(2, 128, 64)
    for sizes in (torch.ones(2, 64), torch.ones(128), torch.ones(2, 128, 1)):
        with pytest.raises(ValueError, match=r"\[G, P\].*got sizes"):
            cutedsl_ward.cluster_ward(keys, sizes, 16, 16)


def test_pool_sizes_on_another_device_are_refused():
    with pytest.raises(ValueError, match="got sizes on meta, keys on cpu"):
        cutedsl_ward.cluster_ward(torch.zeros(2, 128, 64), torch.ones(2, 128, device="meta"), 16, 16)
