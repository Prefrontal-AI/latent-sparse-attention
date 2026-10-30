"""The fused scan-and-select against the dense scores it claims to pick the exact best of.

One GPU. Every case compares with the top-p of the same scores computed for every bucket, so the check is
exact equality; the rows are built to take each of the select's paths - listed, listed again wider, and scored
in full.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels import scan_select
from lsa.core.kernels.scan_select import ScanSelect, dense_keys, listed_keys, sample_thresholds
from lsa.core.kernels.triton import scan_select as triton_scan_select
from lsa.core.kernels.triton.keys import EMPTY, pack_keys, unpack_ids
from lsa.core.kernels.triton.scan_select import dense_scores
from lsa.core.kernels.triton.shape import ShapeTable

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


def unit_rows(count, generator, index_dim=64):
    return F.normalize(torch.randn(count, index_dim, device=DEVICE, generator=generator), dim=-1).bfloat16()


def clustered_queries(count, generator, index_dim=64):
    """Queries around one direction, and that direction - a bucket on it outscores every random one for all"""
    direction = F.normalize(torch.randn(1, index_dim, device=DEVICE, generator=generator), dim=-1)
    noise = 0.2 * torch.randn(count, index_dim, device=DEVICE, generator=generator) / index_dim**0.5
    return F.normalize(direction + noise, dim=-1).bfloat16(), direction


def ragged_lengths(rows, buckets):
    """Each row sees a little less than the last - a prefill chunk's sub-blocks"""
    return (buckets - torch.arange(rows, device=DEVICE, dtype=torch.int32) % 512).contiguous()


def reference_picks(index_queries, bucket_centroids, visible_counts, top_p):
    width = max(bucket_centroids.shape[0], top_p)
    return unpack_ids(
        dense_keys(index_queries, bucket_centroids, visible_counts, width).topk(top_p, dim=1).values
    )


def paths_taken(index_queries, bucket_centroids, visible_counts, top_p):
    """Rows that overflowed their buffer and rows short of top_p at their threshold, on the first listing"""
    thresholds = sample_thresholds(index_queries, bucket_centroids, visible_counts, top_p)
    capacity = round(triton_scan_select.CAPACITY * top_p)
    _, survivor_counts, reached = listed_keys(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity
    )
    return int((survivor_counts > capacity).sum()), int((reached < top_p).sum())


@pytest.mark.parametrize(
    ("rows", "buckets", "top_p", "norm"),
    [(512, 65536, 1024, 1.0), (1024, 200000, 512, 1.0), (300, 131071, 1024, 1.0), (512, 131072, 1024, 300.0)],
)
def test_the_picks_are_the_exact_best_of_the_scores(rows, buckets, top_p, norm):
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    index_queries, bucket_centroids = (
        (unit_rows(rows, generator).float() * norm).bfloat16(),
        unit_rows(buckets, generator),
    )
    visible_counts = ragged_lengths(rows, buckets)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    assert torch.equal(picked, reference_picks(index_queries, bucket_centroids, visible_counts, top_p))


def test_the_picks_hold_the_matrix_top_k_by_value():
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    rows, buckets, top_p = 256, 100000, 1024
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    visible_counts = ragged_lengths(rows, buckets)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts).long()
    matrix = torch.mm(index_queries, bucket_centroids.t(), out_dtype=torch.float32)
    matrix = matrix.masked_fill(
        torch.arange(buckets, device=DEVICE)[None] >= visible_counts[:, None], -torch.inf
    )
    cutoff = matrix.topk(top_p, dim=1).values[:, -1:]
    # Two fp32 sums of the same bf16 products differ in the last bits
    assert (matrix.gather(1, picked) >= cutoff - 1e-5).all()


def test_the_scores_are_an_exact_dot_of_the_bf16_rows():
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    index_queries, bucket_centroids = unit_rows(8, generator), unit_rows(5000, generator)
    visible_counts = torch.full((8,), 5000, device=DEVICE, dtype=torch.int32)
    visible_counts[3] = 1234
    scores = dense_scores(index_queries, bucket_centroids, visible_counts)
    exact = index_queries.double() @ bucket_centroids.double().t()
    visible = torch.arange(5000, device=DEVICE)[None] < visible_counts[:, None]
    assert torch.allclose(scores[visible].double(), exact[visible], rtol=0, atol=1e-6)
    assert torch.isinf(scores[~visible]).all()


def test_buckets_tied_at_the_threshold_are_listed_again_wider():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    rows, buckets, top_p = 512, 131072, 1024
    index_queries, direction = clustered_queries(rows, generator)
    bucket_centroids = unit_rows(buckets, generator)
    # A repeated text - thousands of identical buckets near every query
    bucket_centroids[20000:28000] = direction.bfloat16()
    visible_counts = ragged_lengths(rows, buckets)
    overflowed, _ = paths_taken(index_queries, bucket_centroids, visible_counts, top_p)
    assert overflowed > 0
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    assert torch.equal(picked, reference_picks(index_queries, bucket_centroids, visible_counts, top_p))


def test_a_row_tied_past_the_widest_listing_is_scored_in_full():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    rows, buckets, top_p = 64, 131072, 512
    index_queries, direction = clustered_queries(rows, generator)
    bucket_centroids = unit_rows(buckets, generator)
    # More identical buckets than WIDE top_p, each scoring above everything else
    tied = triton_scan_select.WIDE * top_p + 1000
    bucket_centroids[:tied] = direction.bfloat16()
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    reference = reference_picks(index_queries, bucket_centroids, visible_counts, top_p)
    assert torch.equal(picked, reference)
    # Ties go to the lower bucket
    assert torch.equal(
        picked.sort(1).values, torch.arange(top_p, device=DEVICE, dtype=torch.int32).expand(rows, -1)
    )


def test_a_row_short_of_picks_at_its_threshold_is_scored_in_full():
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    rows, buckets, top_p = 128, 131072, 1024
    index_queries, direction = clustered_queries(rows, generator)
    bucket_centroids = unit_rows(buckets, generator)
    # The sample sees only buckets far above the rest, so its threshold leaves too few above it
    stride = max(1, round(triton_scan_select.TARGET * top_p / triton_scan_select.SAMPLE_RANK))
    sampled = torch.arange(0, buckets, stride, device=DEVICE)[:300]
    noise = 0.05 * torch.randn(300, 64, device=DEVICE, generator=generator) / 8
    bucket_centroids[sampled] = F.normalize(direction + noise, dim=-1).bfloat16()
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    _, short = paths_taken(index_queries, bucket_centroids, visible_counts, top_p)
    assert short > 0
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    assert torch.equal(picked, reference_picks(index_queries, bucket_centroids, visible_counts, top_p))


def test_a_row_seeing_fewer_than_picks_takes_all_it_sees_and_pads():
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    rows, buckets, top_p = 16, 70000, 1024
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    visible_counts[:4] = torch.tensor([0, 1, 300, 1023], device=DEVICE, dtype=torch.int32)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    assert torch.equal(picked, reference_picks(index_queries, bucket_centroids, visible_counts, top_p))
    for row, length in enumerate([0, 1, 300, 1023]):
        assert torch.equal(
            picked[row, length:], torch.full((top_p - length,), -1, device=DEVICE, dtype=torch.int32)
        )
        assert torch.equal(
            picked[row, :length].sort().values, torch.arange(length, device=DEVICE, dtype=torch.int32)
        )


def test_the_picks_come_best_first():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    rows, buckets, top_p = 64, 80000, 512
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts).long()
    scores = dense_scores(index_queries, bucket_centroids, visible_counts).gather(1, picked)
    assert (scores[:, 1:] <= scores[:, :-1]).all()


def test_the_scan_packs_keys_as_the_torch_side_does():
    generator = torch.Generator(device=DEVICE).manual_seed(10)
    index_queries, bucket_centroids = unit_rows(8, generator), unit_rows(3000, generator)
    visible_counts = torch.full((8,), 3000, device=DEVICE, dtype=torch.int32)
    visible_counts[2] = 17
    keys = dense_keys(index_queries, bucket_centroids, visible_counts)
    scores = dense_scores(index_queries, bucket_centroids, visible_counts)
    buckets = torch.arange(3000, device=DEVICE, dtype=torch.int32).expand(8, -1)
    visible = buckets < visible_counts[:, None]
    assert torch.equal(keys, pack_keys(scores.masked_fill(~visible, 0.0), buckets.masked_fill(~visible, -1)))
    assert torch.equal(unpack_ids(keys), buckets.masked_fill(~visible, -1))
    assert (keys[~visible] == EMPTY).all()


def test_queries_of_any_norm_list_about_as_many_as_unit_ones():
    generator = torch.Generator(device=DEVICE).manual_seed(8)
    rows, buckets, top_p = 512, 131072, 1024
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    # The served centroid read never normalizes its index_queries - a head sum through the map is hundreds long
    for norm in (1.0, 0.01, 300.0):
        scaled = (index_queries.float() * norm).bfloat16()
        assert paths_taken(scaled, bucket_centroids, visible_counts, top_p) == (0, 0)
        picked = ScanSelect(rows, top_p)(scaled, bucket_centroids, visible_counts)
        assert torch.equal(picked, reference_picks(scaled, bucket_centroids, visible_counts, top_p))


def test_a_tie_mass_near_the_threshold_is_listed_again_rather_than_scored_in_full(monkeypatch):
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    rows, buckets, top_p = 1024, 131072, 1024
    index_queries, direction = clustered_queries(rows, generator)
    index_queries = (index_queries.float() * 500).bfloat16()
    bucket_centroids = unit_rows(buckets, generator)
    # 6,000 identical buckets, blended toward the index_queries until they score about where the threshold falls
    target = (
        torch.mm(index_queries[:64].float(), bucket_centroids.float().t())
        .topk(1400, dim=1)
        .values[:, -1]
        .mean()
    )
    other = F.normalize(torch.randn(1, 64, device=DEVICE, generator=generator), dim=-1)
    for share in torch.linspace(0, 1, 101, device=DEVICE):
        tied = F.normalize(share * direction + (1 - share) * other, dim=-1)
        if (index_queries[:64].float() @ tied.t()).mean() >= target:
            break
    bucket_centroids[30000:36000] = tied.bfloat16()
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    dense_rows = []
    full = scan_select.dense_keys

    def counted(index_queries, bucket_centroids, visible_counts, width=None, **options):
        dense_rows.append(index_queries.shape[0])
        return full(index_queries, bucket_centroids, visible_counts, width, **options)

    monkeypatch.setattr(scan_select, "dense_keys", counted)
    picked = ScanSelect(rows, top_p)(index_queries, bucket_centroids, visible_counts)
    assert paths_taken(index_queries, bucket_centroids, visible_counts, top_p)[0] > rows // 2
    assert sum(dense_rows) <= rows // 100
    assert torch.equal(picked, reference_picks(index_queries, bucket_centroids, visible_counts, top_p))


def random_shapes(buckets, generator, rank=2, deviations=2.0):
    """Shapes of the spread Ward's buckets of 8 keys have - variances of a few hundredths, a smaller floor"""
    basis = torch.randint(0, 256, (buckets, rank, 8), device=DEVICE, generator=generator, dtype=torch.uint8)
    variances = (
        (0.03 * torch.rand(buckets, rank, device=DEVICE, generator=generator)).sort(1, descending=True).values
    )
    floor = 0.002 * torch.rand(buckets, device=DEVICE, generator=generator)
    return ShapeTable(basis, variances, floor, deviations)


@pytest.mark.parametrize(("rows", "buckets", "top_p", "rank"), [(512, 65536, 1024, 2), (300, 131071, 512, 1)])
def test_shaped_picks_are_the_exact_best_of_the_shaped_scores(rows, buckets, top_p, rank):
    generator = torch.Generator(device=DEVICE).manual_seed(rows + rank)
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    shapes = random_shapes(buckets, generator, rank)
    visible_counts = ragged_lengths(rows, buckets)
    select = ScanSelect(rows, top_p)
    listing = select.listed(index_queries, bucket_centroids, visible_counts, shapes)
    width = max(buckets, top_p)
    want = unpack_ids(
        dense_keys(index_queries, bucket_centroids, visible_counts, width, shapes).topk(top_p, dim=1).values
    )
    assert torch.equal(select.pick(listing), want)


def test_shaped_scores_differ_from_the_centroids_and_rows_short_of_picks_score_in_full():
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    rows, buckets, top_p = 64, 40000, 1024
    index_queries, bucket_centroids = unit_rows(rows, generator), unit_rows(buckets, generator)
    shapes = random_shapes(buckets, generator)
    visible_counts = torch.full((rows,), buckets, device=DEVICE, dtype=torch.int32)
    shaped = dense_scores(index_queries, bucket_centroids, visible_counts, shapes)
    plain = dense_scores(index_queries, bucket_centroids, visible_counts)
    assert (shaped >= plain).all() and (shaped > plain).float().mean() > 0.9
    # A threshold no shaped score reaches - every row scored in full
    select = ScanSelect(rows, top_p)
    listing = select.listed(index_queries, bucket_centroids, visible_counts, shapes)
    starved = listing._replace(reached=torch.zeros_like(listing.reached))
    want = unpack_ids(
        dense_keys(index_queries, bucket_centroids, visible_counts, buckets, shapes).topk(top_p).values
    )
    assert torch.equal(select.pick(starved), want)
