"""The learning-rate schedules - warmup-stable-decay and a cosine after warmup.

torchtitan's own WSD differs from the paper's trainer's by one step in the warmup and by a rounding of the
decay length, and the tests hold this schedule to the trainer's WSD factors, transcribed, so the schedule is
the one function here and [the trainer](trainer.py)'s container only steps it.
It imports nothing from torchtitan, so the config can say where the decay starts.
"""

import math

from lsa.training.native.constants import DecayShape, Schedule


def decay_step_count(steps: int, decay_ratio: float) -> int:
    """Return the decay share of the run rounded up, as the paper's trainer does.

    Args:
        steps: Optimiser steps of the whole run.
        decay_ratio: Share of the run that decays, 0 for none.

    Returns:
        The decay's length in steps.
    """
    return math.ceil(decay_ratio * steps)


def decay_start_step(steps: int, decay_ratio: float) -> int:
    """Return the completed update count at the end of the stable stage.

    Args:
        steps: Optimiser steps of the whole run.
        decay_ratio: Share of the run that decays.

    Returns:
        The step whose checkpoint is the decay start.
    """
    return steps - decay_step_count(steps, decay_ratio)


def learning_rate_factor(
    step: int,
    *,
    steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_factor: float,
    decay_shape: str,
    schedule: str = Schedule.WSD,
) -> float:
    """Return the learning rate of update `step` as a share of the peak.

    WSD warms up from zero, holds the peak until the last `decay_steps`, then decays to `min_lr_factor` shaped linear,
    as one minus the square root of its progress, or as a cosine.
    A cosine schedule warms up with LinearLR's tiny intercept, then decays over every remaining step.
    It is a function of the step alone, so a resume replays it exactly.

    Args:
        step: Index of the update, from 0.
        steps: Optimiser steps of the whole run.
        warmup_steps: Steps of the warmup.
        decay_steps: Steps that decay, at the end of the run.
        min_lr_factor: The share of the peak the decay ends at.
        decay_shape: One of `DecayShape.ALL`.
        schedule: WSD or a cosine over every step after warmup.

    Returns:
        The factor the peak learning rate is scaled by.

    Raises:
        ValueError: `schedule` or `decay_shape` is not one of its options.
    """
    if schedule == Schedule.COSINE:
        if step < warmup_steps:
            return 1e-8 + (1.0 - 1e-8) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, steps - warmup_steps))
        return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule != Schedule.WSD:
        raise ValueError(f"schedule must be one of {Schedule.ALL} - got {schedule!r}")
    if step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    start = steps - decay_steps
    if step < start:
        return 1.0
    progress = min(1.0, (step - start) / max(1, decay_steps))
    if decay_shape == DecayShape.LINEAR:
        remaining = 1.0 - progress
    elif decay_shape == DecayShape.SQRT:
        remaining = 1.0 - math.sqrt(progress)
    elif decay_shape == DecayShape.COSINE:
        remaining = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"decay_shape must be one of {DecayShape.ALL} - got {decay_shape!r}")
    return min_lr_factor + (1.0 - min_lr_factor) * remaining
