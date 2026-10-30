"""The shape kernels against the torch reference - the build, the two spread paths and the decode scan.

One GPU. The planes hold a query to 8 bits a chunk, so a shaped score is the reference's to a tolerance; the kernels
that score against the planes share one helper, so they agree with each other exactly.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.shape import (
    ShapeTable,
    add_planes_spread,
    add_spread,
    build_bucket_shapes,
    build_shapes,
    plane_gaps,
    scan_paged_shaped,
    score_shaped_rows,
    top_directions,
)
from lsa.core.shape import Shapes, leading_directions, shaped_scores, spread, unpack_shape_basis
from lsa.core.shape import build_shapes as reference_build_shapes

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
WIDTH = 64


def anisotropic_buckets(count, size, generator, width=WIDTH):
    """Buckets of `size` unit keys spread along two directions of their own - keys, centroids, members"""
    centres = F.normalize(torch.randn(count, 1, width, device=DEVICE, generator=generator), dim=-1)
    axes = F.normalize(torch.randn(count, 2, width, device=DEVICE, generator=generator), dim=-1)
    spreads = torch.randn(count, size, 2, device=DEVICE, generator=generator) * torch.tensor(
        [0.4, 0.15], device=DEVICE
    )
    noise = 0.02 * torch.randn(count, size, width, device=DEVICE, generator=generator)
    keys = F.normalize(centres + (spreads[..., None] * axes[:, None]).sum(2) + noise, dim=-1)
    centroids = F.normalize(keys.sum(1), dim=-1)
    members = torch.arange(count * size, device=DEVICE).view(count, size)
    return keys.view(-1, width).bfloat16(), centroids, members


def shape_table(count, rank, generator, deviations=2.0):
    """Shapes of buckets of 16 anisotropic keys, as the cache stores them, and their keys, centroids, members"""
    keys, centroids, members = anisotropic_buckets(count, 16, generator)
    basis, variances, floor = build_shapes(lambda positions: keys[positions], centroids, members, rank)
    table = ShapeTable(basis, variances, floor, deviations)
    return table, keys, centroids, members


def as_reference(table):
    return Shapes(unpack_shape_basis(table.basis), table.variances, table.floor)


def unit_queries(count, generator):
    return F.normalize(torch.randn(count, WIDTH, device=DEVICE, generator=generator), dim=-1).bfloat16()


def test_power_iteration_finds_the_leading_directions_of_a_spread_covariance():
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    keys, centroids, members = anisotropic_buckets(64, 64, generator)
    residuals = keys.float().view(64, 64, WIDTH) - centroids[:, None]
    covariance = residuals.transpose(1, 2) @ residuals / 64
    directions, variances, trace = top_directions(covariance, 2)
    exact, exact_variances = leading_directions(covariance, 2)
    torch.testing.assert_close(variances, exact_variances, rtol=1e-3, atol=1e-6)
    assert ((directions * exact).sum(-1).abs() > 0.999).all()
    torch.testing.assert_close(trace, covariance.diagonal(dim1=1, dim2=2).sum(1))


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_the_node_build_is_the_reference_build_to_bf16(rank):
    # Power iteration over bf16 residuals - a bucket whose leading variances nearly tie may take another direction
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    keys, centroids, members = anisotropic_buckets(256, 64, generator)
    basis, variances, floor = build_shapes(lambda positions: keys[positions], centroids, members, rank)
    reference = reference_build_shapes(keys, centroids, members, rank)
    close = torch.isclose(variances, reference.variances, rtol=3e-2, atol=1e-4).all(1)
    assert close.float().mean() > 0.98
    torch.testing.assert_close(floor[close], reference.floor[close], rtol=3e-2, atol=1e-4)
    queries = unit_queries(64, generator).float()
    built = Shapes(unpack_shape_basis(basis), variances, floor)
    near = torch.isclose(spread(queries, built), spread(queries, reference), rtol=3e-2, atol=3e-3)
    assert near.float().mean() > 0.98


@pytest.mark.parametrize("rank", [0, 1, 2, 4])
def test_the_bucket_build_is_the_reference_build(rank):
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    keys, centroids, members = anisotropic_buckets(512, 16, generator)
    # Buckets of 8 to 16 keys, padded to the cap
    sizes = torch.randint(8, 17, (512,), device=DEVICE, generator=generator)
    members = torch.where(torch.arange(16, device=DEVICE)[None] < sizes[:, None], members, -1)
    held = members >= 0
    centroids = F.normalize((keys.float()[members.clamp_min(0)] * held[..., None]).sum(1), dim=-1)
    basis, variances, floor = build_bucket_shapes(lambda positions: keys[positions], centroids, members, rank)
    reference = reference_build_shapes(keys, centroids, members, rank)
    torch.testing.assert_close(variances, reference.variances, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(floor, reference.floor, rtol=1e-3, atol=1e-6)
    queries = unit_queries(64, generator).float()
    built = Shapes(unpack_shape_basis(basis), variances, floor)
    near = torch.isclose(spread(queries, built), spread(queries, reference), rtol=1e-3, atol=1e-5)
    # A direction is the same up to its sign, but for a bucket whose variances tie
    assert near.float().mean() > 0.99


@pytest.mark.parametrize("rank", [0, 2, 3])
def test_a_shaped_row_scores_the_references_estimated_best_key(rank):
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    table, _, centroids, _ = shape_table(512, rank, generator)
    queries = unit_queries(32, generator)
    rows = torch.randint(-1, 512, (32, 200), device=DEVICE, generator=generator, dtype=torch.int32)
    got = score_shaped_rows(
        queries, centroids.bfloat16(), table.basis, table.variances, table.floor, rows, table.deviations
    )
    want = shaped_scores(queries.float(), centroids.bfloat16().float(), as_reference(table), table.deviations)
    want = torch.where(rows >= 0, want.gather(1, rows.clamp_min(0).long()), -torch.inf)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=0)


def test_the_dense_spread_and_the_row_scoring_agree_on_the_spread_bit_for_bit():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    table, _, centroids, _ = shape_table(300, 2, generator)
    queries = unit_queries(16, generator)
    zero = torch.zeros(300, WIDTH, device=DEVICE, dtype=torch.bfloat16)
    rows = torch.arange(300, device=DEVICE, dtype=torch.int32).expand(16, -1).contiguous()
    from_rows = score_shaped_rows(queries, zero, table.basis, table.variances, table.floor, rows, 2.0)
    dense = add_planes_spread(torch.zeros(16, 300, device=DEVICE), queries, table)
    assert torch.equal(dense, from_rows)


def test_the_decode_scan_scores_each_page_row_as_the_row_scoring_does():
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    page_size, pages = 128, 4
    table, _, centroids, _ = shape_table(page_size * pages, 2, generator)
    queries = unit_queries(2, generator)
    page_table = torch.tensor([[2, 0, 3, -1], [1, 3, -1, -1]], device=DEVICE, dtype=torch.int32)
    counts = torch.tensor([3 * page_size - 7, page_size + 5], device=DEVICE, dtype=torch.int32)
    centroids = centroids.bfloat16()
    scores = scan_paged_shaped(queries, centroids, table, page_table, counts, page_size, 4 * page_size)
    logical = torch.arange(4 * page_size, device=DEVICE)
    for request in range(2):
        page = page_table[request, logical // page_size]
        held = (logical < counts[request]) & (page >= 0)
        rows = torch.where(held, page * page_size + logical % page_size, -1).int()[None]
        want = score_shaped_rows(
            queries[request : request + 1], centroids, table.basis, table.variances, table.floor, rows, 2.0
        )
        torch.testing.assert_close(scores[request : request + 1], want, atol=1e-6, rtol=0)


def test_the_tensor_core_spread_is_the_references_to_bf16():
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    table, _, centroids, _ = shape_table(700, 2, generator)
    queries = unit_queries(100, generator)
    got = add_spread(
        torch.zeros(100, 700, device=DEVICE), queries, table.basis, table.variances, table.floor, 2.0
    )
    want = 2.0 * spread(queries.float(), as_reference(table))
    torch.testing.assert_close(got, want, atol=2e-3, rtol=0)


def test_a_queries_planes_project_within_its_gap_of_the_query():
    # A shape of one direction, variance 1 and no floor has the planes' projection |q . s| as its spread
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    count = 4096
    queries = unit_queries(64, generator)
    signs = torch.randint(
        0, 256, (count, 1, WIDTH // 8), device=DEVICE, generator=generator, dtype=torch.uint8
    )
    table = ShapeTable(signs, torch.ones(count, 1, device=DEVICE), torch.zeros(count, device=DEVICE), 1.0)
    from_planes = add_planes_spread(torch.zeros(64, count, device=DEVICE), queries, table)
    exact = (queries.float() @ unpack_shape_basis(signs)[:, 0].T).abs()
    assert ((from_planes - exact).abs() <= plane_gaps(queries)[:, None]).all()
    assert plane_gaps(queries).max() < 0.02
