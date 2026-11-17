import logging

import pytest

from fusion_lsa.budget import EarlyBudget
from fusion_lsa.core.reference import build_buckets, check_sizes, check_unit_rows, expand_buckets, read

from helpers import to_mlx, to_numpy

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def torch_reference(upstream_src):
    from lsa.core import reference as torch_ref

    return torch_ref


def make_keys(torch, t, dim, seed=3):
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(t, dim, generator=gen)
    return keys / keys.norm(dim=1, keepdim=True)


def test_check_unit_rows_parity(torch_reference):
    import torch

    bad = torch.randn(4, 8) * 3
    with pytest.raises(ValueError, match="unit rows"):
        check_unit_rows("keys", to_mlx(bad))
    with pytest.raises(ValueError, match="unit rows"):
        torch_reference.check_unit_rows("keys", bad)
    good = make_keys(torch, 4, 8)
    check_unit_rows("keys", to_mlx(good))
    with pytest.raises(ValueError, match="must be"):
        check_unit_rows("keys", to_mlx(torch.randn(2, 2, 8)))


def test_check_sizes_errors(torch_reference):
    with pytest.raises(ValueError, match="divide"):
        check_sizes(4, 6, 16, 64)
    with pytest.raises(ValueError, match="twice"):
        check_sizes(4, 8, 12, 64)
    check_sizes(4, 8, 16, 64)


def test_build_buckets_parity_ward(torch_reference):
    import torch

    keys = make_keys(torch, 128, 16, seed=5)
    args = (4, 8, 16, 32)
    want = torch_reference.build_buckets(keys, *args)
    got = build_buckets(to_mlx(keys), *args)
    assert to_numpy(got.bucket_centroids).shape == tuple(want.bucket_centroids.shape)
    assert (to_numpy(got.bucket_centroids) - want.bucket_centroids.numpy()).max() < 1e-3
    assert to_numpy(got.bucket_members).tolist() == want.bucket_members.tolist()
    assert to_numpy(got.bucket_sizes).tolist() == want.bucket_sizes.tolist()
    assert got.block_bucket_count == want.block_bucket_count


def test_build_buckets_parity_position(torch_reference):
    import torch

    keys = make_keys(torch, 128, 16, seed=6)
    args = (4, 8, 16, 32, "position")
    want = torch_reference.build_buckets(keys, *args)
    got = build_buckets(to_mlx(keys), *args)
    assert to_numpy(got.bucket_members).tolist() == want.bucket_members.tolist()
    assert (to_numpy(got.bucket_centroids) - want.bucket_centroids.numpy()).max() < 1e-3


def test_build_buckets_partial_block_open(torch_reference):
    import torch

    keys = make_keys(torch, 100, 16, seed=7)
    want = torch_reference.build_buckets(keys, 4, 8, 16, 32)
    got = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    assert got.bucket_centroids.shape[0] == want.bucket_centroids.shape[0] == 12


def test_read_parity_centroid(torch_reference):
    import torch

    keys = make_keys(torch, 128, 16, seed=5)
    buckets_t = torch_reference.build_buckets(keys, 4, 8, 16, 32)
    buckets_m = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    gen = torch.Generator().manual_seed(9)
    queries = torch.randn(12, 16, generator=gen)
    queries = queries / queries.norm(dim=1, keepdim=True)
    positions = torch.tensor([0, 5, 31, 32, 40, 63, 64, 90, 100, 127, 200, 300])
    args = (8, 64, 2, 16)
    want_sel, want_len = torch_reference.read(queries, positions, buckets_t, *args)
    got_sel, got_len = read(to_mlx(queries), to_mlx(positions), buckets_m, *args)
    assert to_numpy(got_sel).tolist() == want_sel.tolist()
    assert to_numpy(got_len).tolist() == want_len.tolist()


def test_read_parity_early_budget(torch_reference):
    import torch

    keys = make_keys(torch, 128, 16, seed=5)
    buckets_t = torch_reference.build_buckets(keys, 4, 8, 16, 32)
    buckets_m = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    gen = torch.Generator().manual_seed(9)
    queries = torch.randn(6, 16, generator=gen)
    queries = queries / queries.norm(dim=1, keepdim=True)
    positions = torch.tensor([0, 10, 50, 100, 127, 40])
    args = (8, 64, 2, 16)
    budget = {"position": 60, "top_k": 16, "top_p": 4}
    want_sel, _ = torch_reference.read(queries, positions, buckets_t, *args, early_budget=torch_reference.EarlyBudget.read(budget))
    got_sel, _ = read(to_mlx(queries), to_mlx(positions), buckets_m, *args, early_budget=EarlyBudget.read(budget))
    assert to_numpy(got_sel).tolist() == want_sel.tolist()


def test_read_rejects_bad_inputs(torch_reference):
    import torch

    keys = make_keys(torch, 64, 16, seed=5)
    buckets_m = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    bad_q = torch.randn(4, 16) * 5
    with pytest.raises(ValueError, match="unit rows"):
        read(to_mlx(bad_q), to_mlx(torch.arange(4)), buckets_m, 8, 64, 2, 16)
    with pytest.raises(ValueError, match="dims"):
        read(to_mlx(make_keys(torch, 4, 8, seed=1)), to_mlx(torch.arange(4)), buckets_m, 8, 64, 2, 16)


def test_expand_buckets_parity(torch_reference):
    import torch

    keys = make_keys(torch, 128, 16, seed=5)
    buckets_t = torch_reference.build_buckets(keys, 4, 8, 16, 32)
    buckets_m = build_buckets(to_mlx(keys), 4, 8, 16, 32)
    picked_t = torch.tensor([5, 2])
    want = torch_reference.expand_buckets(picked_t, 100, buckets_t, 32, 2, 8)
    got = expand_buckets(to_mlx(picked_t), 100, buckets_m, 32, 2, 8)
    assert to_numpy(got).tolist() == want.tolist()
