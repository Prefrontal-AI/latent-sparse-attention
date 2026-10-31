"""The CuTeDSL listing against the survivor words it replaces.

One GPU. Both list the buckets at or above each row's threshold from the same tensor-core scores, so a row lists
the same set of buckets and the same count either way; only the order within a row differs, and nothing reads it.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import dispatch, scan_select
from lsa.core.kernels.cutedsl import bucket_scan
from lsa.core.kernels.triton import scan_select as triton_scan_select

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# A consumer Blackwell, which the dispatch table names for no step
UNNAMED_GPU = (12, 0)


def unit_rows(count, generator, index_dim=64):
    return F.normalize(torch.randn(count, index_dim, device=DEVICE, generator=generator), dim=-1).bfloat16()


def anchored_queries(bucket_centroids, rows, generator):
    """Queries near a few centroids, so a row's best buckets stand out of the noise"""
    anchors = torch.randint(0, bucket_centroids.shape[0], (rows,), device=DEVICE, generator=generator)
    noise = torch.randn(rows, bucket_centroids.shape[1], device=DEVICE, generator=generator)
    return (3 * bucket_centroids[anchors].float() + noise).bfloat16()


def shifted_thresholds(index_queries, bucket_centroids, visible_counts, top_p):
    thresholds = scan_select.sample_thresholds(index_queries, bucket_centroids, visible_counts, top_p)
    return thresholds - triton_scan_select.MARGIN * index_queries.float().norm(dim=1)


def assert_same_listing(
    index_queries, bucket_centroids, visible_counts, thresholds, capacity, span_tiles=None
):
    expected, expected_counts = triton_scan_select.word_listing(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity
    )
    listed, counts = bucket_scan.scan_buckets(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity, span_tiles
    )
    assert torch.equal(counts, expected_counts)
    for row in (expected_counts <= capacity).nonzero().flatten().tolist():
        held = expected_counts[row].item()
        assert torch.equal(listed[row, :held].sort().values, expected[row, :held].sort().values), row
    return counts


@pytest.mark.parametrize(
    ("rows", "buckets", "top_p"), [(128, 4096, 64), (300, 70001, 512), (1000, 131071, 1024)]
)
def test_each_row_lists_the_buckets_the_survivor_words_list(rows, buckets, top_p):
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = anchored_queries(bucket_centroids, rows, generator)
    # Each row sees a little less than the last, so tiles end inside a row's visible buckets
    visible_counts = (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 777).contiguous()
    thresholds = shifted_thresholds(index_queries, bucket_centroids, visible_counts, top_p)
    counts = assert_same_listing(
        index_queries,
        bucket_centroids,
        visible_counts,
        thresholds,
        round(triton_scan_select.CAPACITY * top_p),
    )
    assert counts.float().mean() > top_p


@pytest.mark.parametrize("span_tiles", [1, 3, 5, 64, bucket_scan.MAX_SPAN_TILES])
def test_a_span_of_any_length_lists_the_same_buckets(span_tiles):
    generator = torch.Generator(device=DEVICE).manual_seed(span_tiles)
    rows, buckets = 200, 70001
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = anchored_queries(bucket_centroids, rows, generator)
    # Each row sees a little less than the last, so the last tile of a span ends inside a row's visible buckets
    visible_counts = (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 777).contiguous()
    thresholds = shifted_thresholds(index_queries, bucket_centroids, visible_counts, 256)
    counts = assert_same_listing(
        index_queries, bucket_centroids, visible_counts, thresholds, 4096, span_tiles
    )
    assert counts.float().mean() > 256


def test_a_row_past_its_shared_slots_appends_the_rest_to_the_global_listing():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    rows, buckets = 256, 20000
    # A repeated text - a run of one centroid that every row ties on, far more than a span's slots
    bucket_centroids = unit_rows(buckets, generator)
    bucket_centroids[1000 : 1000 + 40 * bucket_scan.ROW_SLOTS] = bucket_centroids[999]
    index_queries = bucket_centroids[999].expand(rows, -1).contiguous()
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    thresholds = torch.full((rows,), 0.99, device=DEVICE)
    counts = assert_same_listing(index_queries, bucket_centroids, visible_counts, thresholds, 4096)
    assert (counts > bucket_scan.ROW_SLOTS).all()


def test_an_overflowed_row_counts_every_survivor_and_holds_its_capacity():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    rows, buckets, capacity = 130, 9000, 100
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = unit_rows(rows, generator)
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    # Every bucket a row sees survives
    thresholds = torch.full((rows,), -float("inf"), device=DEVICE)
    listed, counts = bucket_scan.scan_buckets(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity
    )
    assert (counts == buckets).all()
    assert ((listed >= 0) & (listed < buckets)).all()
    assert all(row.unique().numel() == capacity for row in listed)


@pytest.mark.parametrize("norm", [1.0, 300.0])
def test_the_select_picks_the_same_buckets_in_either_language(norm):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    rows, buckets, top_p = 512, 131072, 1024
    bucket_centroids = unit_rows(buckets, generator)
    index_queries = (unit_rows(rows, generator).float() * norm).bfloat16()
    visible_counts = (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 512).contiguous()
    by_triton = scan_select.ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    by_cutedsl = scan_select.ScanSelect(rows, top_p, kernels="cutedsl")(
        index_queries, bucket_centroids, visible_counts
    )
    assert torch.equal(by_triton, by_cutedsl)


def test_the_select_lists_by_survivor_words_unless_asked_for_cutedsl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bucket_scan, "scan_buckets", lambda *arguments: calls.append(1) or listing_by_words(*arguments)
    )
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    bucket_centroids = unit_rows(65536, generator)
    index_queries = unit_rows(128, generator)
    visible_counts = torch.full((128,), 65536, device=DEVICE, dtype=torch.int32)
    scan_select.ScanSelect(128, 512)(index_queries, bucket_centroids, visible_counts)
    assert not calls
    scan_select.ScanSelect(128, 512, kernels="cutedsl")(index_queries, bucket_centroids, visible_counts)
    assert calls


def listing_by_words(index_queries, bucket_centroids, visible_counts, thresholds, capacity):
    return triton_scan_select.word_listing(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity
    )


@pytest.mark.parametrize("span_tiles", [0, -1, bucket_scan.MAX_SPAN_TILES + 1])
def test_a_span_the_kernel_does_not_hold_is_refused(span_tiles):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    bucket_centroids = unit_rows(100, generator)
    index_queries = unit_rows(10, generator)
    visible_counts = torch.full((10,), 100, device=DEVICE, dtype=torch.int32)
    thresholds = torch.zeros(10, device=DEVICE)
    with pytest.raises(ValueError, match=f"got {span_tiles}"):
        bucket_scan.scan_buckets(index_queries, bucket_centroids, visible_counts, thresholds, 64, span_tiles)


def test_rows_the_kernel_does_not_take_are_refused():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    bucket_centroids = unit_rows(100, generator, index_dim=48)
    index_queries = unit_rows(10, generator, index_dim=48)
    visible_counts = torch.full((10,), 100, device=DEVICE, dtype=torch.int32)
    thresholds = torch.zeros(10, device=DEVICE)
    assert not bucket_scan.supports(index_queries, bucket_centroids)
    with pytest.raises(ValueError, match="64 index dims on a (Hopper|Blackwell) GPU"):
        bucket_scan.scan_buckets(index_queries, bucket_centroids, visible_counts, thresholds, 64)


def test_a_gpu_with_neither_mma_lists_by_survivor_words(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "capability", lambda device: UNNAMED_GPU)
    monkeypatch.setattr(
        bucket_scan, "scan_buckets", lambda *arguments: calls.append(1) or listing_by_words(*arguments)
    )
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    bucket_centroids = unit_rows(4096, generator)
    index_queries = unit_rows(128, generator)
    visible_counts = torch.full((128,), 4096, device=DEVICE, dtype=torch.int32)
    assert not bucket_scan.supports(index_queries, bucket_centroids)
    scan_select.ScanSelect(128, 64, kernels="cutedsl")(index_queries, bucket_centroids, visible_counts)
    assert not calls
