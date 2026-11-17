import logging

import pytest

from fusion_lsa.core.index_space import LearnedProjection, regroup, rownorm

from helpers import to_mlx, to_numpy

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def torch_index_space(upstream_src):
    from lsa.core import index_space as torch_is

    return torch_is


def test_regroup_parity(torch_index_space):
    import torch

    gen = torch.Generator().manual_seed(3)
    b, hq, hkv, tq, tk, dh = 2, 8, 2, 5, 7, 16
    q = torch.randn(b, hq, tq, dh, generator=gen)
    k = torch.randn(b, hkv, tk, dh, generator=gen)
    q_idx_t, k_idx_t = torch_index_space.regroup(q, k)
    q_idx_m, k_idx_m = regroup(to_mlx(q), to_mlx(k))
    assert (to_numpy(q_idx_m) - q_idx_t.numpy()).max() < 1e-3
    assert (to_numpy(k_idx_m) - k_idx_t.numpy()).max() < 1e-3
    # unit rows, zero rows stay zero
    qz = q_idx_t[0, :, 0] * 0
    q0_t, k0_t = torch_index_space.regroup(q * 0, k)
    q0_m, k0_m = regroup(to_mlx(q * 0), to_mlx(k))
    assert abs(to_numpy(q0_m) - q0_t.numpy()).max() < 1e-6
    assert abs(to_numpy(k0_m) - k0_t.numpy()).max() < 1e-6


def test_regroup_rejects_mismatched_heads(torch_index_space):
    import torch

    q = torch.randn(1, 7, 2, 16)
    k = torch.randn(1, 2, 2, 16)
    with pytest.raises(ValueError, match="group"):
        regroup(to_mlx(q), to_mlx(k))
    with pytest.raises(ValueError, match="head dim"):
        regroup(to_mlx(q), to_mlx(torch.randn(1, 2, 2, 8)))


def test_rownorm_zero_row_stays_zero():
    import mlx.core as mx

    a = mx.array([[3.0, 4.0], [0.0, 0.0]])
    out = rownorm(a)
    assert abs(out[0][0].item() - 0.6) < 1e-6 and abs(out[0][1].item() - 0.8) < 1e-6
    assert out[1].tolist() == [0.0, 0.0]


def test_learned_projection_validation():
    import mlx.core as mx

    wq = mx.random.normal((16, 8), key=mx.random.key(1))
    wk = mx.random.normal((16, 8), key=mx.random.key(2))
    LearnedProjection(query_projection=wq, key_projection=wk)
    with pytest.raises(ValueError, match="share their shape"):
        LearnedProjection(query_projection=wq, key_projection=mx.random.normal((16, 4), key=mx.random.key(3)))
    with pytest.raises(ValueError, match="must be"):
        LearnedProjection(query_projection=wq[0], key_projection=wk[0])


def test_project_parity(torch_index_space):
    import torch

    gen = torch.Generator().manual_seed(9)
    rows = torch.randn(4, 16, generator=gen)
    index_map = torch.randn(16, 8, generator=gen)
    want = torch_index_space.project(rows, index_map)
    got = LearnedProjection.project_rows(to_mlx(rows), to_mlx(index_map))
    assert (to_numpy(got) - want.numpy()).max() < 1e-3
