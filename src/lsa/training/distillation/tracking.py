"""Where a distillation run's scalars go, for the runs that send them anywhere.

A run already writes every number it measures to json and prints the headline ones, which is enough to read
one run. A sweep is what json is bad at - the dim ladder is several runs whose whole point is comparison,
and comparing them means opening a directory by hand.

Off by default, and a run with it off never imports wandb.

The distillation metrics themselves are [metrics.py](metrics.py), so this module is named for what it does with them.

The step loss is logged per layer rather than as one series. A step trains one layer, chosen round-robin, so
a single curve is a sawtooth of which layer the step landed on rather than anything about learning.
"""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig


def scalars(tree: Any, prefix: str) -> dict[str, float]:
    """Flatten a metric tree into one `prefix/path` per leaf.

    Args:
        tree: Nested dicts with numbers at the leaves, as `metrics.mean_reports` returns.
        prefix: What to name the root.

    Returns:
        One entry per leaf, keyed by its path.
    """
    if not isinstance(tree, dict):
        return {prefix: float(tree)}

    flat: dict[str, float] = {}
    for key, value in tree.items():
        flat.update(scalars(value, f"{prefix}/{key}"))
    return flat


class Tracker:
    """One run on Weights & Biases, or nothing at all when it is switched off."""

    def __init__(
        self, experiment: "DistillConfig", out: Path, label: str | None = None, resume_id: str | None = None
    ) -> None:
        """Start a run on Weights & Biases, or stay silent.

        wandb names the run itself and `prefix` goes in front of that name rather than replacing it, so the
        runs of one sweep sort together and still keep the two words that tell them apart.

        A run trained in shards is one wandb group named for the run, with a member per shard for its layers'
        training scalars and one more for the joined reports, which are the ones a sweep compares.

        Args:
            experiment: The experiment, for its wandb settings and the config recorded beside the scalars.
            out: The run directory, so an offline run's files land beside its reports.
            label: What this member of a sharded run is, or None for a run trained whole.
            resume_id: The wandb run a resumed run goes on logging into, or None to start one.
        """
        self.writer: Any = None
        if not experiment.run.wandb.enabled:
            return

        import wandb

        out.mkdir(parents=True, exist_ok=True)
        self.writer = wandb.init(
            project=experiment.run.wandb.project,
            entity=experiment.run.wandb.entity,
            dir=str(out),
            tags=experiment.run.wandb.tags or None,
            config=asdict(experiment),
            group=None if label is None else experiment.run.name,
            id=resume_id,
            resume="must" if resume_id is not None else None,
        )
        if resume_id is not None:
            return
        if experiment.run.wandb.prefix:
            self.writer.name = f"{experiment.run.wandb.prefix}-{self.writer.name}"
        if label is not None:
            self.writer.name = f"{self.writer.name}-{label}"

    @property
    def run_id(self) -> str | None:
        """The wandb run's id, which a resumed run logs into again - None when off"""
        return None if self.writer is None else str(self.writer.id)

    def step(
        self,
        step: int,
        layer: int,
        loss: float,
        tau: float,
        grad_norm: float,
        terms: dict[str, float] | None = None,
    ) -> None:
        """Log what one optimiser step did.

        Args:
            step: The step.
            layer: The layer this step trained.
            loss: What it descended, every term at its weight.
            tau: That layer's temperature after the step.
            grad_norm: The gradient norm before clipping.
            terms: Each term unweighted, and the per-key KL above the target's entropy, by name.
        """
        if self.writer is None:
            return
        self.writer.log(
            {
                f"train/loss/layer_{layer}": loss,
                f"train/tau/layer_{layer}": tau,
                "train/grad_norm": grad_norm,
            }
            | {f"train/{name}/layer_{layer}": value for name, value in (terms or {}).items()},
            step=step,
        )

    def report(self, step: int, report: dict[str, Any]) -> None:
        """Log an evaluation, every leaf of it.

        Args:
            step: The step being reported.
            report: The report, as `run.report` builds it.
        """
        if self.writer is None:
            return
        measured = {key: value for key, value in report.items() if key != "step"}
        self.writer.log(scalars(measured, "eval"), step=step)

    def close(self) -> None:
        """Finish the run, so an interrupted job does not leave it open."""
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
