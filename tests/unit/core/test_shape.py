"""Bucket shapes in torch - the spread they estimate, their packed basis, the bound the fused select lists by.

On a CPU. A shape of full rank is the covariance itself, so its spread is the keys' own standard deviation of scores;
a lower rank keeps the trace, and the sign basis keeps the shape's order of buckets close to the exact one.
"""

import pytest
import torch
import torch.nn.functional as F

from lsa.core import Buckets, build_buckets, read, shape_buckets
from lsa.core.shape import (
    Shapes,
    build_shapes,
    covariances,
    pack_shape_basis,
    shaped_scores,
    sign_directions,
    spread,
    unpack_shape_basis,
)

WIDTH = 32


def anisotropic_buckets(count, size, generator, width=WIDTH):
    """Buckets of `size` unit keys spread along two directions of their own, and their keys and members"""
    centres = F.normalize(torch.randn(count, 1, width, generator=generator), dim=-1)
    axes = F.normalize(torch.randn(count, 2, width, generator=generator), dim=-1)
    spreads = torch.randn(count, size, 2, generator=generator) * torch.tensor([0.4, 0.2])
    noise = 0.02 * torch.randn(count, size, width, generator=generator)
    keys = F.normalize(centres + (spreads[..., None] * axes[:, None]).sum(2) + noise, dim=-1)
    members = torch.arange(count * size).view(count, size)
    centroids = F.normalize(keys.sum(1), dim=-1)
    return keys.view(-1, width), centroids, members


def test_a_full_rank_shape_with_its_exact_directions_is_the_keys_own_spread():
    generator = torch.Generator().manual_seed(0)
    keys, centroids, members = anisotropic_buckets(6, 12, generator)
    covariance, _ = covariances(keys, centroids, members)
    values, vectors = torch.linalg.eigh(covariance.double())
    exact = Shapes(vectors.transpose(1, 2).flip(1).float(), values.flip(1).float(), torch.zeros(6))
    queries = F.normalize(torch.randn(5, WIDTH, generator=generator), dim=-1)
    residual_scores = queries @ (keys.view(6, 12, WIDTH) - centroids[:, None]).transpose(1, 2)
    want = residual_scores.square().mean(-1).transpose(0, 1).sqrt()
    torch.testing.assert_close(spread(queries, exact), want, atol=1e-5, rtol=1e-4)


def test_a_shape_keeps_the_trace_whatever_its_rank():
    generator = torch.Generator().manual_seed(1)
    keys, centroids, members = anisotropic_buckets(4, 16, generator)
    covariance, _ = covariances(keys, centroids, members)
    trace = covariance.diagonal(dim1=1, dim2=2).sum(1)
    for rank in (0, 1, 2, 8):
        shapes = build_shapes(keys, centroids, members, rank)
        kept = shapes.variances.sum(1) + shapes.floor * (WIDTH - rank)
        torch.testing.assert_close(kept, trace, atol=1e-6, rtol=1e-4)


def test_rank_zero_is_an_even_spread_of_the_trace():
    generator = torch.Generator().manual_seed(2)
    keys, centroids, members = anisotropic_buckets(3, 8, generator)
    shapes = build_shapes(keys, centroids, members, 0)
    assert shapes.basis.shape == (3, 0, WIDTH)
    queries = F.normalize(torch.randn(4, WIDTH, generator=generator), dim=-1)
    torch.testing.assert_close(spread(queries, shapes), shapes.floor.sqrt()[None].expand(4, -1))


def test_padded_members_add_nothing_and_divide_by_the_buckets_own_size():
    generator = torch.Generator().manual_seed(3)
    keys, centroids, members = anisotropic_buckets(2, 8, generator)
    padded = torch.cat([members, torch.full((2, 8), -1)], 1)
    alone, with_padding = build_shapes(keys, centroids, members, 2), build_shapes(keys, centroids, padded, 2)
    torch.testing.assert_close(with_padding.variances, alone.variances)
    torch.testing.assert_close(with_padding.floor, alone.floor)


def test_the_basis_packs_to_a_bit_a_dimension_and_back():
    generator = torch.Generator().manual_seed(4)
    basis = sign_directions(torch.randn(5, 3, 64, generator=generator))
    packed = pack_shape_basis(basis)
    assert packed.shape == (5, 3, 8) and packed.dtype == torch.uint8
    assert torch.equal(unpack_shape_basis(packed), basis)
    assert torch.allclose(basis.norm(dim=-1), torch.ones(5, 3))


def test_a_direction_and_its_negation_score_alike():
    generator = torch.Generator().manual_seed(5)
    keys, centroids, members = anisotropic_buckets(4, 16, generator)
    shapes = build_shapes(keys, centroids, members, 2)
    flipped = Shapes(-shapes.basis, shapes.variances, shapes.floor)
    queries = F.normalize(torch.randn(8, WIDTH, generator=generator), dim=-1)
    torch.testing.assert_close(spread(queries, flipped), spread(queries, shapes))


def test_the_reference_read_scores_a_shaped_bucket_by_its_estimated_best_key():
    generator = torch.Generator().manual_seed(7)
    keys = F.normalize(torch.randn(4 * 64, WIDTH, generator=generator), dim=-1)
    buckets = build_buckets(keys, 4, 8, 16, 64)
    shapes = shape_buckets(keys, buckets, 2)
    queries = F.normalize(torch.randn(16, WIDTH, generator=generator), dim=-1)
    positions = torch.full((16,), 4 * 64 - 1)
    plain, _ = read(queries, positions, buckets, 4, 16, 0, 64)
    unshaped, _ = read(queries, positions, buckets, 4, 16, 0, 64, shapes, 0.0)
    assert torch.equal(plain, unshaped)
    shaped, _ = read(queries, positions, buckets, 4, 16, 0, 64, shapes, 2.0)
    # The three blocks closed before the queries' own
    scores = shaped_scores(queries, buckets.bucket_centroids, shapes, 2.0)[:, :24]
    best = scores.argmax(1)
    first = buckets.bucket_members[best]
    assert all(set(first[row][first[row] >= 0].tolist()) <= set(shaped[row].tolist()) for row in range(16))


def test_shape_buckets_refuses_a_rank_past_the_width():
    keys = F.normalize(torch.randn(64, WIDTH), dim=-1)
    buckets: Buckets = build_buckets(keys, 4, 8, 16, 64)
    with pytest.raises(ValueError, match="rank must be from 0 to the keys' width 32 - got 33"):
        shape_buckets(keys, buckets, 33)
