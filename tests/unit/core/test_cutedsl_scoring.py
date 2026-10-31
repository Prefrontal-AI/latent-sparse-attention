"""The CuTeDSL scoring of listed buckets against the Triton scoring it replaces.

One GPU. Both add a 64-dim dot in the same order and pack it the same way, so on the same listing a row's keys and its
count of buckets reached are the same bit for bit - the empty keys past its listed count included.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import scan_select
from lsa.core.kernels.cutedsl import bucket_scan, scoring
from lsa.core.kernels.triton import scan_select as triton_scan_select
from lsa.core.kernels.triton.keys import EMPTY

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def unit_rows(count, generator, index_dim=64):
    return F.normalize(torch.randn(count, index_dim, device=DEVICE, generator=generator), dim=-1).bfloat16()


def scored_by_triton(index_queries, bucket_centroids, buckets, survivor_counts, thresholds):
    rows, capacity = buckets.shape
    keys = torch.full((rows, capacity), EMPTY, device=DEVICE, dtype=torch.int64)
    reached = torch.empty(rows, device=DEVICE, dtype=torch.int32)
    triton_scan_select._score_buckets[(rows,)](
        index_queries,
        bucket_centroids,
        buckets,
        survivor_counts,
        thresholds,
        keys,
        reached,
        capacity,
        D=64,
        BLOCK=triton_scan_select.SCORE_BLOCK,
        SPAN=0,
        DENSE=False,
        num_warps=4,
        **triton_scan_select.shape_arguments(index_queries, None, None),
    )
    return keys, reached


@pytest.mark.parametrize(
    ("rows", "buckets", "top_p", "norm", "capacity_scale"),
    [
        (64, 3000, 64, 1.0, 1.0),
        (300, 70001, 512, 30.0, 1.0),
        (1000, 131071, 1024, 1.0, 1.0),
        # Every row overflows its listing
        (300, 70001, 512, 1.0, 0.25),
    ],
)
def test_each_row_has_the_triton_keys(rows, buckets, top_p, norm, capacity_scale):
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = (unit_rows(rows, generator).float() * norm).bfloat16()
    visible_counts = (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 777).contiguous()
    thresholds = scan_select.sample_thresholds(index_queries, bucket_centroids, visible_counts, top_p)
    shifted = thresholds - triton_scan_select.MARGIN * index_queries.float().norm(dim=1)
    capacity = round(triton_scan_select.CAPACITY * top_p * capacity_scale)
    listed, survivor_counts = bucket_scan.scan_buckets(
        index_queries, bucket_centroids, visible_counts, shifted, capacity
    )
    expected = scored_by_triton(index_queries, bucket_centroids, listed, survivor_counts, thresholds)
    keys, reached = scoring.score_candidates(
        index_queries, bucket_centroids, listed, survivor_counts, thresholds
    )
    assert torch.equal(keys, expected[0])
    assert torch.equal(reached, expected[1])
    assert (survivor_counts > capacity).all() == (capacity_scale < 1)


def test_negative_scores_and_an_empty_row_pack_as_triton_packs_them():
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    bucket_centroids = unit_rows(500, generator)
    index_queries = -unit_rows(4, generator)
    listed = torch.randint(0, 500, (4, 96), device=DEVICE, dtype=torch.int32, generator=generator)
    survivor_counts = torch.tensor([96, 0, 17, 200], device=DEVICE, dtype=torch.int32)
    thresholds = torch.tensor([-0.1, 0.0, -float("inf"), float("inf")], device=DEVICE)
    expected = scored_by_triton(index_queries, bucket_centroids, listed, survivor_counts, thresholds)
    keys, reached = scoring.score_candidates(
        index_queries, bucket_centroids, listed, survivor_counts, thresholds
    )
    assert torch.equal(keys, expected[0])
    assert torch.equal(reached, expected[1])


def test_rows_the_kernel_does_not_take_are_refused():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    bucket_centroids = unit_rows(100, generator, index_dim=128)
    index_queries = unit_rows(10, generator, index_dim=128)
    listed = torch.zeros(10, 8, device=DEVICE, dtype=torch.int32)
    counts = torch.zeros(10, device=DEVICE, dtype=torch.int32)
    assert not scoring.supports(index_queries, bucket_centroids)
    with pytest.raises(ValueError, match="64 index dims"):
        scoring.score_candidates(
            index_queries, bucket_centroids, listed, counts, torch.zeros(10, device=DEVICE)
        )


def test_the_listing_scores_in_cutedsl_when_asked(monkeypatch):
    calls = []
    score = scoring.score_candidates
    monkeypatch.setattr(scoring, "score_candidates", lambda *arguments: calls.append(1) or score(*arguments))
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    bucket_centroids = unit_rows(65536, generator)
    index_queries = unit_rows(128, generator)
    visible_counts = torch.full((128,), 65536, device=DEVICE, dtype=torch.int32)
    thresholds = scan_select.sample_thresholds(index_queries, bucket_centroids, visible_counts, 512)
    scan_select.listed_keys(index_queries, bucket_centroids, visible_counts, thresholds, 1024)
    assert not calls
    scan_select.listed_keys(
        index_queries, bucket_centroids, visible_counts, thresholds, 1024, kernels="cutedsl"
    )
    assert calls


@pytest.mark.parametrize(
    ("rows", "buckets", "width"),
    [
        (256, 70001, None),
        # Columns past the last bucket stay empty
        (300, 5000, 6000),
        (17, 100, 2048),
    ],
)
def test_dense_keys_are_the_triton_dense_keys(rows, buckets, width):
    generator = torch.Generator(device=DEVICE).manual_seed(rows + buckets)
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = (unit_rows(rows, generator).float() * 3).bfloat16()
    visible_counts = (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) * 97 % buckets).int()
    # A row that sees nothing, one bucket and every bucket
    visible_counts[:3] = torch.tensor([0, 1, buckets], device=DEVICE, dtype=torch.int32)
    expected = scan_select.dense_keys(index_queries, bucket_centroids, visible_counts, width)
    assert torch.equal(scoring.score_dense(index_queries, bucket_centroids, visible_counts, width), expected)


def test_the_select_scores_short_rows_densely_in_cutedsl_when_asked(monkeypatch):
    calls = []
    score = scoring.score_dense
    monkeypatch.setattr(scoring, "score_dense", lambda *arguments: calls.append(1) or score(*arguments))
    generator = torch.Generator(device=DEVICE).manual_seed(8)
    bucket_centroids = unit_rows(4096, generator)
    index_queries = unit_rows(128, generator)
    # Rows that see fewer buckets than the top-p reach too few at any threshold
    visible_counts = torch.arange(1, 129, device=DEVICE, dtype=torch.int32) * 8
    by_triton = scan_select.ScanSelect(128, 512)(index_queries, bucket_centroids, visible_counts)
    assert not calls
    by_cutedsl = scan_select.ScanSelect(128, 512, kernels="cutedsl")(
        index_queries, bucket_centroids, visible_counts
    )
    assert calls
    assert torch.equal(by_triton, by_cutedsl)
