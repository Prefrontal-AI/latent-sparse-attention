import pytest

from lsa.training.native.constants import DecayShape
from lsa.training.native.schedule import decay_start_step, decay_step_count, learning_rate_factor
from tests.helpers.training.schedule import reference_decay_steps, reference_factor


@pytest.mark.parametrize("shape", DecayShape.ALL)
@pytest.mark.parametrize(
    ("steps", "warmup_steps", "decay_ratio", "min_lr_factor"),
    [
        (17167, 859, 0.2, 0.1),
        (1000, 0, 0.01, 0.1),
        (95368, 954, 0.01, 0.0),
        (50, 5, 0.5, 0.25),
        (8000, 100, 1.0, 0.1),
        (20, 20, 0.5, 0.0),
        (100, 20, 0.0, 0.1),
    ],
)
def test_every_update_follows_the_reference_training_codes_formula(
    shape, steps, warmup_steps, decay_ratio, min_lr_factor
):
    decay_steps = decay_step_count(steps, decay_ratio)
    assert decay_steps == reference_decay_steps(decay_ratio, steps)
    for step in range(steps + 1):
        ours = learning_rate_factor(
            step,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_factor=min_lr_factor,
            decay_shape=shape,
        )
        assert ours == reference_factor(step, warmup_steps, steps, decay_steps, min_lr_factor, shape)


def test_the_decay_start_is_the_last_step_at_the_peak():
    steps, warmup_steps = 1000, 50
    start = decay_start_step(steps, 0.2)
    assert start == 800

    def peak(step):
        return learning_rate_factor(
            step,
            steps=steps,
            warmup_steps=warmup_steps,
            decay_steps=200,
            min_lr_factor=0.1,
            decay_shape="sqrt",
        )

    assert peak(start) == 1.0
    assert peak(start + 1) < 1.0


def test_a_full_decay_uses_the_whole_run_even_when_warmup_overlaps():
    assert decay_step_count(8000, 1.0) == 8000
    assert decay_start_step(8000, 1.0) == 0


def test_a_run_with_no_decay_never_leaves_the_peak():
    assert decay_start_step(15000, 0.0) == 15000
    assert (
        learning_rate_factor(
            14999, steps=15000, warmup_steps=500, decay_steps=0, min_lr_factor=0.1, decay_shape="cosine"
        )
        == 1.0
    )


def test_a_shape_that_is_not_one_is_refused():
    with pytest.raises(ValueError, match="decay_shape"):
        learning_rate_factor(
            900, steps=1000, warmup_steps=0, decay_steps=500, min_lr_factor=0.1, decay_shape="exp"
        )


@pytest.mark.parametrize("warmup", [0, 1, 401])
def test_cosine_matches_torch_linear_warmup_and_cosine_at_every_update(warmup):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    steps = 19074
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=3e-4)
    cosine = CosineAnnealingLR(optimizer, T_max=steps - warmup, eta_min=3e-5)
    if warmup:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, total_iters=warmup)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine], milestones=[warmup])
    else:
        scheduler = cosine
    for step in range(steps + 1):
        actual = learning_rate_factor(
            step,
            steps=steps,
            warmup_steps=warmup,
            decay_steps=1,
            min_lr_factor=0.1,
            decay_shape="sqrt",
            schedule="cosine",
        )
        assert actual * 3e-4 == pytest.approx(optimizer.param_groups[0]["lr"], rel=1e-11, abs=1e-16)
        optimizer.step()
        scheduler.step()


def test_cosine_warmup_preserves_linear_lrs_tiny_intercept():
    assert (
        learning_rate_factor(
            1,
            steps=100,
            warmup_steps=10,
            decay_steps=1,
            min_lr_factor=0.1,
            decay_shape="sqrt",
            schedule="cosine",
        )
        == 1e-8 + (1 - 1e-8) / 10
    )


def test_an_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="schedule"):
        learning_rate_factor(
            0,
            steps=100,
            warmup_steps=10,
            decay_steps=20,
            min_lr_factor=0.1,
            decay_shape="sqrt",
            schedule="plateau",
        )
