"""The tracker - what it flattens, what it derives, and that it stays out of the way when off."""

import sys

import pytest

from lsa.training.distillation.config import DistillConfig, RunConfig, WandbConfig
from lsa.training.distillation.tracking import Tracker, scalars


def a_report():
    """A report shaped the way `run.report` builds one, with numbers small enough to check by hand"""
    return {
        "step": 2000,
        "tau": {"3": 0.05, "7": 0.06},
        "natural": {"mass": 0.6, "recall16": 0.7},
    }


def test_every_leaf_of_a_report_becomes_one_series():
    flat = scalars({"a": {"b": 1.0, "c": {"d": 2.0}}, "e": 3.0}, "eval")
    assert flat == {"eval/a/b": 1.0, "eval/a/c/d": 2.0, "eval/e": 3.0}


def test_a_run_with_wandb_off_never_reaches_wandb(tmp_path):
    """Off is the default, and a run that is off must not need wandb installed at all"""
    sys.modules.pop("wandb", None)

    tracker = Tracker(DistillConfig(run=RunConfig(output=str(tmp_path), name="quiet")), tmp_path)
    tracker.report(0, a_report())
    tracker.step(100, 3, 1.0, 0.05, 0.5)
    tracker.close()

    assert tracker.writer is None
    assert "wandb" not in sys.modules


def test_a_cadence_that_would_log_nothing_is_rejected():
    with pytest.raises(ValueError, match="wandb.log_every must be positive"):
        WandbConfig(log_every=0)
