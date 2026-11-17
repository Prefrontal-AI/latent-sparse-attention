import logging

import mlx.core as mx
import pytest

from fusion_lsa.core.ward import members_from_labels, pool_keys, ward_merge, ward_rounds

from helpers import to_mlx, to_numpy

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def torch_ward(upstream_src):
    from lsa.core import ward as torch_ward_mod

    return torch_ward_mod


def test_pool_keys_parity(torch_ward):
    import torch

    keys = torch.randn(2, 16, 8)
    got = pool_keys(to_mlx(keys), 4)
    want = torch_ward.pool_keys(keys, 4)
    assert to_numpy(got).shape == tuple(want.shape)
    assert (to_numpy(got) - want.numpy()).max() < 1e-5


def test_ward_rounds_parity(torch_ward):
    import torch

    gen = torch.Generator().manual_seed(7)
    for trial in range(8):
        blocks, pools, dim = 3, 16, 8
        pk = torch.randn(blocks, pools, dim, generator=gen)
        ps = torch.full((blocks, pools), 4.0)
        labels_t, reached_t = torch_ward.ward_rounds(pk, ps, 4, 32)
        labels_m, reached_m = ward_rounds(to_mlx(pk), to_mlx(ps), 4, 32)
        assert to_numpy(labels_m).tolist() == labels_t.tolist(), f"trial {trial} labels differ"
        assert to_numpy(reached_m).tolist() == reached_t.tolist(), f"trial {trial} reached differs"


def test_ward_rounds_stops_short_under_tight_cap(torch_ward):
    import torch

    gen = torch.Generator().manual_seed(11)
    pk = torch.randn(2, 8, 4, generator=gen)
    ps = torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]] * 2)
    labels_t, reached_t = torch_ward.ward_rounds(pk, ps, 2, 2)
    labels_m, reached_m = ward_rounds(to_mlx(pk), to_mlx(ps), 2, 2)
    assert to_numpy(labels_m).tolist() == labels_t.tolist()
    assert not reached_t.all(), "expected oracle to stop short under the tight cap"
    assert not to_numpy(reached_m).all()


def test_ward_merge_parity_and_errors(torch_ward):
    import torch

    gen = torch.Generator().manual_seed(13)
    pk = torch.randn(2, 8, 4, generator=gen)
    ps = torch.full((2, 8), 1.0)
    labels_t = torch_ward.ward_merge(pk, ps, 2, 8)
    labels_m = ward_merge(to_mlx(pk), to_mlx(ps), 2, 8)
    assert to_numpy(labels_m).tolist() == labels_t.tolist()
    with pytest.raises(ValueError, match="cap"):
        ward_merge(to_mlx(pk), to_mlx(ps), 2, 1)
    with pytest.raises(ValueError, match="cap"):
        torch_ward.ward_merge(pk, ps, 2, 1)
    tight_ps = torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]] * 2)
    with pytest.raises(ValueError, match="cap"):
        torch_ward.ward_merge(pk, tight_ps, 2, 2)
    with pytest.raises(ValueError, match="cap"):
        ward_merge(to_mlx(pk), to_mlx(tight_ps), 2, 2)


def test_members_from_labels_parity(torch_ward):
    import torch

    labels_t = torch.tensor([1, 0, 1, 2, 0, 1])
    got = members_from_labels(to_mlx(labels_t), 3, 4)
    want = torch_ward.members_from_labels(labels_t, 3, 4)
    assert to_numpy(got).tolist() == want.tolist()


def test_members_from_labels_rejects_oversized_bucket():
    labels = mx.zeros((5,), dtype=mx.int32)
    with pytest.raises(ValueError, match="cap"):
        members_from_labels(labels, 1, 4)
