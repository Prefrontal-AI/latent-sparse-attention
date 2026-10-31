"""The per-key divergence - what it bounds, what truncating the target does, and what calibration is for."""

import pytest
import torch

from lsa.training.distillation.config import ObjectiveConfig
from lsa.training.distillation.constants import Cache, Divergence, Tail, Target, Term
from lsa.training.distillation.data.cache import Document, layer_path, write
from lsa.training.distillation.losses import (
    calibrate,
    calibration_grid,
    calibration_losses,
    key_divergence,
    key_scores,
    rest_bucket_cross_entropy,
    step_terms,
    teacher_entropy,
)
from lsa.training.distillation.model import IndexMap, init_rows
from tests.helpers.training.distillation.data import fake_cache


def key_loss(index_map, one, rows, **objective):
    """The per-key term a step pays for, as the trainer computes it"""
    return step_terms(index_map, one, rows, ObjectiveConfig(**objective))[Term.KEY]


def document(tmp_path):
    fake_cache.collection(tmp_path)
    return Document(layer_path(tmp_path, "tokens", "train", "source0", 0), fake_cache.config())


@torch.no_grad()
def test_the_loss_never_goes_below_the_teachers_own_entropy(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    index_map.set_tau(0.05)
    assert float(key_loss(index_map, one, one.train_rows())) >= teacher_entropy([one]) - 1e-4


def test_a_map_that_points_at_the_teachers_keys_loses_less(tmp_path):
    one = document(tmp_path)
    rows = one.train_rows()
    index_map = IndexMap(one.model_dim, one.model_dim, init=init_rows(one.x))
    index_map.set_tau(0.05)
    before = float(key_loss(index_map, one, rows).detach())

    optimiser = torch.optim.Adam([index_map.key_projection, index_map.query_projection], lr=1e-2)
    for _ in range(20):
        optimiser.zero_grad()
        key_loss(index_map, one, rows).backward()
        optimiser.step()
    assert float(key_loss(index_map, one, rows).detach()) < before


@torch.no_grad()
def test_calibration_picks_a_temperature_that_loses_less_than_a_flat_one(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, one.model_dim, init=init_rows(one.x))
    index_map.set_tau(1.0)
    flat = float(key_loss(index_map, one, one.train_rows()))
    chosen = calibrate(index_map, [one])
    assert chosen < 1.0
    assert float(key_loss(index_map, one, one.train_rows())) <= flat


def test_the_loss_carries_a_gradient_to_both_maps(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    index_map.set_tau(0.05)
    key_loss(index_map, one, one.train_rows()).backward()
    assert float(index_map.key_projection.grad.abs().sum()) > 0
    assert float(index_map.query_projection.grad.abs().sum()) > 0


@torch.no_grad()
def test_a_depth_past_what_the_cache_holds_is_the_untruncated_target(tmp_path):
    one = document(tmp_path)
    rows = one.train_rows()
    every = one.key_target(rows)
    assert torch.equal(one.key_target(rows, fake_cache.TOP), every)
    assert torch.equal(one.key_target(rows, fake_cache.TOP * 4), every)


@torch.no_grad()
def test_truncating_the_target_concentrates_it_and_leaves_the_rest_alone(tmp_path):
    one = document(tmp_path)
    rows = one.train_rows()
    every = one.key_target(rows)
    few = one.key_target(rows, 4)
    # Both are distributions over the readable keys
    assert torch.allclose(few.sum(-1), torch.ones(len(rows)), atol=1e-5)
    # Truncation only ever removes support, and what survives carries more of the mass
    assert bool(((few > 0) <= (every > 0)).all())
    assert float(few.max(-1).values.mean()) > float(every.max(-1).values.mean())


@torch.no_grad()
def test_the_symmetric_form_is_finite_and_bounded_where_reverse_kl_would_not_be(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    index_map.set_tau(0.05)
    # The target is zero almost everywhere over a document, which is exactly where reverse KL diverges
    value = float(key_loss(index_map, one, one.train_rows(), divergence=Divergence.JENSEN_SHANNON))
    assert 0.0 <= value <= torch.log(torch.tensor(2.0))


def test_the_symmetric_form_carries_a_gradient_too(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    index_map.set_tau(0.05)
    key_loss(index_map, one, one.train_rows(), divergence=Divergence.JENSEN_SHANNON).backward()
    assert float(index_map.key_projection.grad.abs().sum()) > 0
    assert float(index_map.query_projection.grad.abs().sum()) > 0


def test_an_objective_the_loss_cannot_compute_is_refused(tmp_path):
    one = document(tmp_path)
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    with pytest.raises(ValueError, match="divergence"):
        key_divergence(
            key_scores(index_map, one, one.train_rows()), one, one.train_rows(), divergence="reverse_kl"
        )
    with pytest.raises(ValueError, match="divergence"):
        ObjectiveConfig(divergence="reverse_kl")
    with pytest.raises(ValueError, match="target_depth"):
        ObjectiveConfig(target_depth=0)


def test_a_dense_on_native_loss_trains_on_what_the_selector_picked(tmp_path):
    blob = fake_cache.document()
    # The selector picks only each row's strongest teacher key
    blob[Cache.NATIVE] = blob["top_key_ids"][:, :1].int()
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    write(path, blob)
    one = Document(path, fake_cache.config())
    rows = one.train_rows()
    index_map = IndexMap(one.model_dim, 8, init=init_rows(one.x))
    index_map.set_tau(0.05)
    # One key holds all the mass => the loss is its negative log-probability
    log_predicted = torch.log_softmax(
        (key_scores(index_map, one, rows) / index_map.tau).masked_fill(
            ~one.visible_keys(rows), float("-inf")
        ),
        -1,
    )
    expected = -log_predicted.gather(1, one.native_key_ids[rows].long()).mean()
    assert torch.allclose(key_loss(index_map, one, rows, target=Target.DENSE_ON_NATIVE), expected, atol=1e-5)
    assert teacher_entropy([one], target=Target.DENSE_ON_NATIVE) < 1e-6


REST = dict(target_depth=4, tail=Tail.REST_BUCKET)


def rest_bucket_inputs(tmp_path):
    """A document, some random index scores over it, and the rows a loss reads"""
    one = document(tmp_path)
    rows = one.train_rows()
    scores = torch.randn(len(rows), one.key_count, generator=torch.Generator().manual_seed(3))
    return one, rows, scores


def test_the_rest_bucket_loss_matches_a_row_by_row_reference(tmp_path):
    one, rows, scores = rest_bucket_inputs(tmp_path)
    depth = REST["target_depth"]
    losses = []
    for i, row in enumerate(rows.tolist()):
        visible = one.visible_keys(rows)[i]
        log_p = torch.log_softmax(scores[i][visible].double(), dim=-1)
        position = torch.full((one.key_count,), -1, dtype=torch.long)
        position[visible] = torch.arange(int(visible.sum()))
        kept_ids = one.top_key_ids[row, :depth].long()
        kept_share = one.top_key_mass[row, :depth].double() / one.head_count
        rho = kept_share.sum()
        sigma = log_p[position[kept_ids]].exp().sum()
        losses.append(-(kept_share * log_p[position[kept_ids]]).sum() - (1 - rho) * torch.log(1 - sigma))
    expected = torch.stack(losses).mean()
    assert not visible.all()
    assert float(rho) < 1
    loss = key_divergence(scores, one, rows, depth, tail=Tail.REST_BUCKET)
    assert torch.allclose(loss.double(), expected, atol=1e-5)


def test_the_rest_bucket_gradient_is_the_shares_gap_on_kept_keys_and_a_rescaled_mass_elsewhere(tmp_path):
    one, rows, scores = rest_bucket_inputs(tmp_path)
    depth = REST["target_depth"]
    scores = scores.requires_grad_()
    key_divergence(scores, one, rows, depth, tail=Tail.REST_BUCKET).backward()

    visible = one.visible_keys(rows)
    share = one.key_share(rows, depth)
    kept = share > 0
    rho = share.sum(-1, keepdim=True)
    p_hat = torch.softmax(scores.detach().masked_fill(~visible, float("-inf")), dim=-1)
    sigma = (p_hat * kept).sum(-1, keepdim=True)
    expected = torch.where(kept, p_hat - share, p_hat * (1 - (1 - rho) / (1 - sigma)))
    expected = expected.masked_fill(~visible, 0.0) / len(rows)
    assert torch.allclose(scores.grad, expected, atol=1e-6)


def test_a_depth_that_holds_all_the_mass_makes_the_rest_bucket_the_truncated_loss(tmp_path):
    one, rows, scores = rest_bucket_inputs(tmp_path)
    # The fake teacher's cached keys hold all of every head's mass
    depth = fake_cache.TOP
    truncated = key_divergence(scores, one, rows, depth)
    assert torch.allclose(
        key_divergence(scores, one, rows, depth, tail=Tail.REST_BUCKET), truncated, atol=1e-3
    )


def test_a_row_with_nothing_outside_the_kept_keys_pays_nothing_for_the_rest():
    log_predicted = torch.log_softmax(torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True), dim=-1)
    readable = torch.ones(1, 3, dtype=torch.bool)
    share = torch.tensor([[0.2, 0.3, 0.5 - 1e-7]])
    loss = rest_bucket_cross_entropy(log_predicted, readable, share, torch.tensor([1e-7]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, -(share * log_predicted).sum())


def test_the_rest_bucket_stays_finite_where_the_kept_keys_hold_nearly_all_the_student_mass():
    scores = torch.tensor([[40.0, 39.0, 0.0, -5.0]], requires_grad=True)
    log_predicted = torch.log_softmax(scores, dim=-1)
    readable = torch.ones(1, 4, dtype=torch.bool)
    share = torch.tensor([[0.5, 0.4, 0.0, 0.0]])
    loss = rest_bucket_cross_entropy(log_predicted, readable, share, torch.tensor([0.1]))
    loss.backward()
    # The student's outside mass is e^-40.3 => its log is -40.3066, which 1 - sigma in fp32 would round to zero
    assert float(loss.detach()) == pytest.approx(0.5 * 0.3133 + 0.4 * 1.3133 + 0.1 * 40.3066, rel=1e-4)
    assert torch.isfinite(scores.grad).all()


@torch.no_grad()
def test_the_rest_bucket_loss_stays_above_its_targets_entropy_and_calibration_scores_it(tmp_path):
    one = document(tmp_path)
    rows = one.train_rows()
    index_map = IndexMap(one.model_dim, one.model_dim, init=init_rows(one.x))
    index_map.set_tau(0.05)
    entropy = teacher_entropy([one], depth=4, tail=Tail.REST_BUCKET)
    assert float(key_loss(index_map, one, rows, **REST)) >= entropy - 1e-4
    # Calibration reads the cross-entropy the run descends
    grid = calibration_grid(index_map)
    losses = calibration_losses(index_map, one, grid, depth=4, tail=Tail.REST_BUCKET)
    for tau, loss in zip(grid[::7], losses[::7], strict=True):
        index_map.set_tau(float(tau))
        assert float(key_loss(index_map, one, rows, **REST)) == pytest.approx(float(loss), rel=1e-4)


def test_a_rest_bucket_objective_the_loss_cannot_compute_is_refused(tmp_path):
    one, rows, scores = rest_bucket_inputs(tmp_path)
    with pytest.raises(ValueError, match="objective.tail must be one of"):
        ObjectiveConfig(tail="remainder")
    with pytest.raises(ValueError, match="needs an objective.target_depth"):
        ObjectiveConfig(tail=Tail.REST_BUCKET)
    with pytest.raises(ValueError, match="forward cross-entropy"):
        ObjectiveConfig(tail=Tail.REST_BUCKET, target_depth=4, divergence=Divergence.JENSEN_SHANNON)
    with pytest.raises(ValueError, match="which 'native_selection' is not read off"):
        ObjectiveConfig(tail=Tail.REST_BUCKET, target=Target.NATIVE_SELECTION)
    with pytest.raises(ValueError, match="tail must be one of"):
        key_divergence(scores, one, rows, 4, tail="remainder")
    with pytest.raises(ValueError, match="beyond a depth"):
        key_divergence(scores, one, rows, None, tail=Tail.REST_BUCKET)
    with pytest.raises(ValueError, match="beyond a depth"):
        key_divergence(scores, one, rows, 4, Divergence.JENSEN_SHANNON, tail=Tail.REST_BUCKET)
    with pytest.raises(ValueError, match="key_share reads the dense ranking"):
        one.key_share(rows, 4, Target.NATIVE_ATTENTION)
    assert ObjectiveConfig().tail == Tail.TRUNCATE


def test_the_rest_bucket_under_a_closed_block_counts_only_the_keys_the_block_leaves_readable(tmp_path):
    one = document(tmp_path).with_closed_block(128)
    rows = one.eval_rows()
    readable = one.visible_keys(rows)
    share = one.key_share(rows, REST["target_depth"])
    assert not bool((share > 0)[~readable].any())
    assert bool((share > 0).any())
    scores = torch.randn(len(rows), one.key_count, generator=torch.Generator().manual_seed(3))
    loss = key_divergence(scores, one, rows, REST["target_depth"], tail=Tail.REST_BUCKET)
    assert torch.isfinite(loss)
