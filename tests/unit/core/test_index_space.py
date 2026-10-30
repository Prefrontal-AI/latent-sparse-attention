"""Invariant 4 - the regrouped score reproduces the head sum, and rows are unit norm.

The whole shared selection rests on this identity: one query vector per position scores exactly what the
attention heads sum to.
"""

import pytest
import torch

from lsa.core.index_space import (
    LatentMetric,
    LearnedProjection,
    RotaryLayout,
    position_free,
    project,
    regroup,
    regroup_latent,
    rownorm,
)

torch.manual_seed(0)
B, HQ, HKV, T, DH = 1, 8, 2, 64, 16


def heads():
    return torch.randn(B, HQ, T, DH), torch.randn(B, HKV, T, DH)


def head_sum_score(q, k):
    """The head-summed score [B, Tq, Tk] regroup must reproduce - the O(T^2) object the index avoids"""
    broadcast_key_heads = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    return torch.einsum("bhqd,bhsd->bqs", q, broadcast_key_heads)


def test_regroup_reproduces_the_head_sum():
    q, k = heads()
    direct = head_sum_score(q, k)
    grouped = q.view(B, HKV, HQ // HKV, T, DH).sum(2).transpose(1, 2).reshape(B, T, HKV * DH)
    flat = k.transpose(1, 2).reshape(B, T, HKV * DH)
    err = (direct - grouped @ flat.transpose(1, 2)).abs().max() / direct.abs().max()
    assert float(err) < 1e-5


def test_regroup_rows_are_unit_norm():
    q_idx, k_idx = regroup(*heads())
    assert torch.allclose(q_idx.norm(dim=-1), torch.ones(B, T), atol=1e-5)
    assert torch.allclose(k_idx.norm(dim=-1), torch.ones(B, T), atol=1e-5)


def test_normalising_equal_norm_keys_keeps_the_ranking():
    """Key normalisation is only a no-op where the key rows already share a norm.

    QK-norm leaves them at cv 0.02, which is why `lsa.md` calls it near a no-op rather than one, and it
    stays because k-means++ seeding assumes cosines in [-1, 1].
    """
    q, k = heads()
    k = torch.nn.functional.normalize(k, dim=-1) * 3.0
    q_idx, k_idx = regroup(q, k)
    direct = head_sum_score(q, k)[0]
    assert torch.equal(direct.argsort(-1), (q_idx[0] @ k_idx[0].t()).argsort(-1))


def test_regroup_rejects_uneven_groups():
    with pytest.raises(ValueError, match="group evenly"):
        regroup(torch.randn(B, 7, T, DH), torch.randn(B, 2, T, DH))


def test_zero_rows_stay_zero():
    x = torch.zeros(1, 3, 8)
    x[0, 1] = 1.0
    normed = rownorm(x)
    assert float(normed[0, 0].norm()) == 0.0
    assert float(normed[0, 2].norm()) == 0.0
    assert abs(float(normed[0, 1].norm()) - 1.0) < 1e-6


def test_no_rotation_keeps_every_dim():
    q, k = heads()
    q_pf, k_pf = position_free(q, k, RotaryLayout("none"))
    assert torch.equal(q_pf, q)
    assert torch.equal(k_pf, k)


def test_prefix_rotation_drops_the_front():
    q, k = heads()
    q_pf, k_pf = position_free(q, k, RotaryLayout("prefix", 4))
    assert torch.equal(q_pf, q[..., 4:])
    assert torch.equal(k_pf, k[..., 4:])


def test_suffix_rotation_drops_the_back():
    q, k = heads()
    q_pf, k_pf = position_free(q, k, RotaryLayout("suffix", 4))
    assert torch.equal(q_pf, q[..., :-4])
    assert torch.equal(k_pf, k[..., :-4])


def test_the_two_ends_select_different_slices():
    """A naive 'drop a prefix' on a suffix-rotated model corrupts the sub-term"""
    q, k = heads()
    front, _ = position_free(q, k, RotaryLayout("prefix", 4))
    back, _ = position_free(q, k, RotaryLayout("suffix", 4))
    assert front.shape == back.shape
    assert not torch.equal(front, back)


def test_a_fully_rotated_head_has_no_sub_term():
    q, k = heads()
    with pytest.raises(ValueError, match="nothing is position-free"):
        position_free(q, k, RotaryLayout("prefix", DH))


def test_spec_end_and_dim_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        RotaryLayout("none", 4)
    with pytest.raises(ValueError, match="disagree"):
        RotaryLayout("prefix", 0)
    with pytest.raises(ValueError, match="end must be one of"):
        RotaryLayout("middle", 4)


def maps(dim, learned_dim):
    return LearnedProjection(
        query_projection=torch.randn(dim, learned_dim), key_projection=torch.randn(dim, learned_dim)
    )


def test_a_learned_map_produces_unit_rows_in_its_own_dim():
    q_idx, k_idx = regroup(*heads())
    learned = maps(HKV * DH, 24)
    learned_queries, learned_keys = (
        project(q_idx, learned.query_projection),
        project(k_idx, learned.key_projection),
    )
    assert learned_queries.shape == (B, T, 24)
    assert learned_keys.shape == (B, T, 24)
    assert torch.allclose(learned_keys.norm(dim=-1), torch.ones(B, T), atol=1e-5)
    assert learned.index_dim == 24


def test_a_learned_map_of_the_identity_leaves_the_space_alone():
    _, k_idx = regroup(*heads())
    dim = HKV * DH
    assert torch.allclose(project(k_idx, torch.eye(dim)), k_idx, atol=1e-5)


def test_a_zero_row_stays_zero_through_a_learned_map():
    rows = torch.randn(4, HKV * DH)
    rows[2] = 0
    projected = project(rows, torch.randn(HKV * DH, 16))
    assert float(projected[2].abs().max()) == 0.0


def test_a_learned_map_that_does_not_read_this_space_raises():
    with pytest.raises(ValueError, match="the map reads"):
        project(torch.randn(4, HKV * DH), torch.randn(HKV * DH + 1, 16))


def test_two_maps_of_different_shapes_are_not_one_space():
    with pytest.raises(ValueError, match="share their shape"):
        LearnedProjection(query_projection=torch.randn(8, 4), key_projection=torch.randn(8, 5))


def test_a_map_that_is_not_a_matrix_raises():
    with pytest.raises(ValueError, match=r"\[Hkv \* dh, index_dim\]"):
        LearnedProjection(query_projection=torch.randn(8), key_projection=torch.randn(8))


# The MLA shape - one latent every key is expanded from, and one key map per query head
R = 12


def latent_layer():
    """Return query heads, a latent, and the maps that expand it into per-head keys."""
    q = torch.randn(B, HQ, T, DH, dtype=torch.float64)
    latent = torch.randn(B, T, R, dtype=torch.float64)
    key_maps = torch.randn(HQ, DH, R, dtype=torch.float64) / DH**0.5
    return q, latent, key_maps


def expanded(latent, key_maps):
    """Return the per-head keys an MLA layer would have materialised [B, Hq, T, dh]."""
    return torch.einsum("btr,hdr->bhtd", latent, key_maps)


def row_scale(narrow, wide):
    """Return the per-query factor that best takes one score matrix onto the other."""
    return (narrow * wide).sum(-1) / (wide * wide).sum(-1)


def test_the_latent_keys_are_an_isometry_of_the_expanded_ones():
    q, latent, key_maps = latent_layer()
    _, narrow_keys = regroup_latent(q, latent, LatentMetric.from_key_maps(key_maps))
    _, wide_keys = regroup(q, expanded(latent, key_maps))

    # Every key-key cosine agrees, so k-means sees the same geometry and builds the same partition
    gap = (narrow_keys @ narrow_keys.transpose(1, 2)) - (wide_keys @ wide_keys.transpose(1, 2))
    assert float(gap.abs().max()) < 1e-5
    assert narrow_keys.shape == (B, T, R)
    assert wide_keys.shape == (B, T, HQ * DH)


def test_the_latent_scores_rank_the_keys_the_expanded_ones_do():
    q, latent, key_maps = latent_layer()
    narrow_queries, narrow_keys = regroup_latent(q, latent, LatentMetric.from_key_maps(key_maps))
    wide_queries, wide_keys = regroup(q, expanded(latent, key_maps))
    narrow = narrow_queries @ narrow_keys.transpose(1, 2)
    wide = wide_queries @ wide_keys.transpose(1, 2)

    # One positive factor per query, which a top-k and a renormalising loss are blind to
    # => not 1, since the wide query's component outside the keys' subspace inflates only its norm
    scale = row_scale(narrow, wide)
    assert float(scale.min()) > 0
    assert float((narrow - scale[..., None] * wide).abs().max() / narrow.abs().max()) < 1e-5
    assert torch.equal(narrow.argsort(dim=-1), wide.argsort(dim=-1))


def test_the_unfactored_latent_is_a_different_geometry():
    q, latent, key_maps = latent_layer()
    _, wide_keys = regroup(q, expanded(latent, key_maps))
    plain = rownorm(latent)

    # The raw logits agree either way, which is the temptation - but the cosines a partition reads do not
    gap = (plain @ plain.transpose(1, 2)) - (wide_keys @ wide_keys.transpose(1, 2))
    assert float(gap.abs().max()) > 0.01


def test_the_latent_rows_are_unit_norm():
    q, latent, key_maps = latent_layer()
    index_queries, index_keys = regroup_latent(q, latent, LatentMetric.from_key_maps(key_maps))
    assert torch.allclose(index_queries.norm(dim=-1), torch.ones(B, T), atol=1e-5)
    assert torch.allclose(index_keys.norm(dim=-1), torch.ones(B, T), atol=1e-5)


def test_the_latent_spec_reports_its_dim():
    _, _, key_maps = latent_layer()
    assert LatentMetric.from_key_maps(key_maps).dim == R


def test_key_maps_that_do_not_span_the_latent_raise():
    key_maps = torch.randn(HQ, DH, R)
    key_maps[..., -1] = 0
    with pytest.raises(ValueError, match="do not span"):
        LatentMetric.from_key_maps(key_maps)


def test_key_maps_that_are_not_a_stack_raise():
    with pytest.raises(ValueError, match=r"\[Hq, dh, r\]"):
        LatentMetric.from_key_maps(torch.randn(DH, R))


def test_a_metric_factor_that_does_not_match_the_maps_raises():
    _, _, key_maps = latent_layer()
    with pytest.raises(ValueError, match="metric_factor must be"):
        LatentMetric(key_maps=key_maps, metric_factor=torch.eye(R + 1))


def test_queries_that_do_not_match_the_key_maps_raise():
    q, latent, key_maps = latent_layer()
    with pytest.raises(ValueError, match="to match the key maps"):
        regroup_latent(q[:, :2], latent, LatentMetric.from_key_maps(key_maps))


def test_a_latent_that_does_not_match_the_key_maps_raises():
    q, latent, key_maps = latent_layer()
    with pytest.raises(ValueError, match="latent must be"):
        regroup_latent(q, latent[..., :-1], LatentMetric.from_key_maps(key_maps))
