"""The CuTeDSL sampled thresholds against the Triton ones they replace past the crossover.

One GPU. A threshold only sets how many buckets a row lists, and a row that lists fewer than its top-p lists again or
is scored in full, so the picks must match exactly and the thresholds need only agree - see
[kernel parity](../../../docs/guide/development.md#kernel-parity).
The kernel does add in Triton's order today, so every row matches to the bit.
The bounds below are what a different order would still meet, and the exact equality is not asked for.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import scan_select
from lsa.core.kernels.constants import ThresholdEstimate, TritonScanSelect
from lsa.core.kernels.cutedsl import thresholds as cutedsl_thresholds

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def unit_rows(count, generator, index_dim=64):
    return F.normalize(torch.randn(count, index_dim, device=DEVICE, generator=generator), dim=-1).bfloat16()


def stride_and_rank(top_p):
    stride = max(1, round(TritonScanSelect.TARGET * top_p / TritonScanSelect.SAMPLE_RANK))
    return stride, max(1, round(TritonScanSelect.TARGET * top_p / stride))


# A threshold sits at a bin edge, and the bins span the upper half between a row's mean and best, at most 2 norm
# => a sample carried over an edge by an ulp moves the threshold a bin of at most norm / 32
BIN_SHARE = 2 / ThresholdEstimate.BINS
# A mean added in another order differs by ulps of itself, and a threshold follows it - measured on 4 shapes of 1,700
# rows at under 1.2e-7 of the norm, with adds reordered and every sample moved an ulp, and never a bin
# => 2**-20 of the norm, 8 times that, for all but the rare row whose sample crossed an edge
ORDER_SLACK = 2**-20
AGREEING_ROWS = 0.99


def assert_agreeing_thresholds(rows, buckets, top_p, seed):
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
    assert torch.equal(estimated.isneginf(), expected.isneginf())
    assert expected[-3:].isneginf().all()
    ranked = expected.isfinite()
    gap = (estimated - expected).abs()[ranked]
    norms = index_queries.float().norm(dim=1)[ranked]
    assert (gap <= ORDER_SLACK * norms).sum() >= AGREEING_ROWS * gap.numel()
    # A crossed edge empties no more than a few bins between the two it moves between
    assert (gap <= 4 * BIN_SHARE * norms).all()


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
def test_each_row_has_a_threshold_agreeing_with_the_triton_one(rows, buckets, top_p):
    assert_agreeing_thresholds(rows, buckets, top_p, rows)


def test_one_process_takes_both_orders_in_turn():
    for buckets in (36864, 40001, 36864):
        assert_agreeing_thresholds(200, buckets, 64, buckets)
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
