import logging

import pytest

from fusion_lsa.core.reference import build_buckets, read
from fusion_lsa.core.ward import ward_rounds

from helpers import to_mlx, to_numpy

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def torch_ward(upstream_src):
    from lsa.core import ward as torch_ward_mod

    return torch_ward_mod


@pytest.fixture(scope="module")
def torch_reference(upstream_src):
    from lsa.core import reference as torch_ref

    return torch_ref


def test_ward_rounds_exact_cost_ties(torch_ward):
    # Duplicated pool keys make merge costs exactly equal - pins the tie-to-lower-index rule
    import torch

    base = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(3))
    pk = torch.cat([base, base, base, base], dim=1)  # 16 pools, 4 distinct vectors
    ps = torch.full((1, 16), 4.0)
    labels_t, reached_t = torch_ward.ward_rounds(pk, ps, 4, 64)
    labels_m, reached_m = ward_rounds(to_mlx(pk), to_mlx(ps), 4, 64)
    assert to_numpy(labels_m).tolist() == labels_t.tolist()
    assert to_numpy(reached_m).tolist() == reached_t.tolist()


def test_read_identical_bucket_scores(torch_reference):
    # Two buckets with identical centroids make bucket scores exactly equal - the selected key
    # set must match the oracle's, order allowed to differ (SDPA is permutation-invariant)
    import torch

    gen = torch.Generator().manual_seed(5)
    keys = torch.randn(64, 16, generator=gen)
    keys = keys / keys.norm(dim=1, keepdim=True)
    buckets_t = torch_reference.build_buckets(keys, 4, 8, 16, 32)
    buckets_m = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    # Force two centroids equal in both implementations
    dup = buckets_t.bucket_centroids[3].clone()
    centroids_t = buckets_t.bucket_centroids.clone()
    centroids_t[5] = dup
    buckets_t = buckets_t._replace(bucket_centroids=centroids_t)
    centroids_m = buckets_m.bucket_centroids
    centroids_m = mx_replace_row(centroids_m, 5, to_mlx(dup))
    buckets_m = buckets_m._replace(bucket_centroids=centroids_m)
    queries = torch.randn(4, 16, generator=gen)
    queries = queries / queries.norm(dim=1, keepdim=True)
    positions = torch.tensor([63, 63, 63, 63])
    args = (4, 32, 2, 8)
    want_sel, want_len = torch_reference.read(queries, positions, buckets_t, *args)
    got_sel, got_len = read(to_mlx(queries), to_mlx(positions), buckets_m, *args)
    assert to_numpy(got_len).tolist() == want_len.tolist()
    for row in range(4):
        want_set = set(want_sel[row].tolist())
        got_set = set(to_numpy(got_sel[row]).tolist())
        assert got_set == want_set, f"row {row} selected set differs under exact score tie"


def mx_replace_row(a, row, value):
    import mlx.core as mx

    idx = mx.array([[row]])
    return mx.put_along_axis(a, idx, value[None, :], axis=0)
