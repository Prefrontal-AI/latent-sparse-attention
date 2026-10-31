"""The CuTeDSL sampled thresholds against the Triton ones they replace past the crossover.

One GPU. Both take each row's threshold from the same sampled scores, added and binned in the same order, so they
agree bit for bit - a row seeing too few samples to rank is -inf either way.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import scan_select
from lsa.core.kernels.cutedsl import thresholds as cutedsl_thresholds
from lsa.core.kernels.triton import scan_select as triton_scan_select

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def unit_rows(count, generator, index_dim=64):
    return F.normalize(torch.randn(count, index_dim, device=DEVICE, generator=generator), dim=-1).bfloat16()


def stride_and_rank(top_p):
    stride = max(1, round(triton_scan_select.TARGET * top_p / triton_scan_select.SAMPLE_RANK))
    return stride, max(1, round(triton_scan_select.TARGET * top_p / stride))


def assert_same_thresholds(rows, buckets, top_p, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = (unit_rows(rows, generator).float() * 30).bfloat16()
    # Rows end inside tiles and blocks, and the last few see too few samples to rank
    visible_counts = (
        buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) * 997 % buckets
    ).contiguous()
    visible_counts[-3:] = torch.tensor([0, 5, 40], device=DEVICE, dtype=torch.int32)
    stride, rank = stride_and_rank(top_p)
    expected = scan_select.sample_thresholds(index_queries, bucket_centroids, visible_counts, top_p)
    estimated = cutedsl_thresholds.sample_thresholds(
        index_queries, bucket_centroids, visible_counts, stride, rank
    )
    assert torch.equal(estimated, expected)
    assert expected[-3:].isneginf().all()


@pytest.mark.parametrize(
    ("rows", "buckets", "top_p"),
    [
        # 20001 samples - Triton adds a thread's columns one at a time
        (300, 40001, 64),
        # 18432 samples - a multiple of 16, so four at a time
        (128, 36864, 64),
        # Short contexts - one partial block, a partial tile, both orders
        (300, 1000, 1024),
        (130, 4096 * 24, 1024),
        (64, 3000, 64),
        (100, 1008 * 24, 1024),
        # Rank 75, past what the 64 group maxima bound
        (256, 40001, 100),
        # 25000 and 24576 samples at the serving top_p
        (1000, 600000, 1024),
        (517, 589824, 1024),
    ],
)
def test_each_row_has_the_triton_threshold(rows, buckets, top_p):
    assert_same_thresholds(rows, buckets, top_p, rows)


def test_one_process_takes_both_orders_in_turn():
    for buckets in (36864, 40001, 36864):
        assert_same_thresholds(200, buckets, 64, buckets)
    assert len(cutedsl_thresholds.COMPILED) == 6


def test_rows_the_kernel_does_not_take_are_refused():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    bucket_centroids = unit_rows(1000, generator, index_dim=48)
    index_queries = unit_rows(10, generator, index_dim=48)
    visible_counts = torch.full((10,), 1000, device=DEVICE, dtype=torch.int32)
    stride, rank = stride_and_rank(64)
    assert not cutedsl_thresholds.supports(index_queries, bucket_centroids)
    with pytest.raises(ValueError, match="centroids of 64 dims"):
        cutedsl_thresholds.sample_thresholds(index_queries, bucket_centroids, visible_counts, stride, rank)


@pytest.mark.parametrize("buckets", [16384, 40001])
def test_the_select_takes_cutedsl_thresholds_when_asked(monkeypatch, buckets):
    calls = []
    estimate = cutedsl_thresholds.sample_thresholds
    monkeypatch.setattr(
        cutedsl_thresholds, "sample_thresholds", lambda *arguments: calls.append(1) or estimate(*arguments)
    )
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = unit_rows(256, generator)
    visible_counts = torch.full((256,), buckets, device=DEVICE, dtype=torch.int32)
    by_triton = scan_select.ScanSelect(256, 64)(index_queries, bucket_centroids, visible_counts)
    assert not calls
    by_cutedsl = scan_select.ScanSelect(256, 64, kernels="cutedsl")(
        index_queries, bucket_centroids, visible_counts
    )
    assert calls
    assert torch.equal(by_triton, by_cutedsl)
