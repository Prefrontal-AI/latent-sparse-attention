"""Submit distillation training jobs from a model/data/run configuration."""

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import env, slurm
from lsa.training import require_extra
from lsa.training.distillation.constants import Job

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig


from lsa.training.distillation.checkpoint import resolve_layers
from lsa.training.distillation.trainer import train

# What a job asks for when the command line says nothing - one GPU is what a run uses, and the host memory is
# what the pool of documents waiting for the card is sized from
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": Job.GPUS,
    "cpus_per_task": Job.CPUS,
    "mem_gb": Job.MEM_GB,
    "timeout_min": Job.HOURS * 60,
}


def submit(
    experiments: list["DistillConfig"],
    parameters: Mapping[str, Any] = PARAMETERS,
    folder: str | Path | None = None,
    force: bool = False,
    stage: Callable[..., Path] = train,
    shards: int | None = None,
    nodes: int = 1,
) -> list[Any]:
    """Submit one job per run, print each one's id, and return them without waiting.

    The stage is submitted directly - `train` and `rescore_maps` are both module-level functions taking the run
    and the force flag, which is exactly what an array maps over.

    Args:
        experiments: The experiments, as `config.load` returns them.
        parameters: What each job asks for, as submitit parameters.
        folder: Where submitit writes each job's output, or None for the shared log root.
        force: Redo a run whose output is already written.
        stage: What to run per config - `train`, or `rescore_maps` to score a checkpoint at a different read.
        shards: Train each run's layers in this many array tasks and join them after, or None to train
            each run in one.
        nodes: Nodes each run's job takes - past one, the online teacher's copies on the others read a share of
            the documents for the trainer on the first.

    Returns:
        One submitit job per run, in the order they were given, or per shard followed by one join per run.

    Raises:
        ValueError: Shards were asked of a stage other than `train`, or nodes of a run without an online teacher
            or together with shards.
    """
    # The shard module reads this one's run directory
    # => imported here, where the cycle has closed
    from lsa.training.distillation.shard import join, split

    if shards is not None and stage is not train:
        raise ValueError("only training splits into shards - a rescore scores every layer in one job")
    if nodes > 1:
        if shards is not None or stage is not train:
            raise ValueError(
                "a run of several nodes splits its documents, not its layers - give --nodes alone"
            )
        alone = [experiment.run.name for experiment in experiments if experiment.data.teacher is None]
        if alone:
            raise ValueError(f"{alone} read a cache - only the online teacher reads across nodes")
        parameters = {**parameters, "nodes": nodes, "tasks_per_node": 1}

    executor = slurm.executor(Job.NAME, env.DISTILL, folder, **parameters)
    for experiment in experiments:
        experiment.run.output = str(Path(experiment.run.output).resolve())
        experiment.data.root = str(Path(experiment.data.root).resolve())
        if experiment.data.teacher is not None:
            experiment.data.teacher.extraction = str(Path(experiment.data.teacher.extraction).resolve())
        for document_set in [*experiment.data.train, *experiment.data.evaluate]:
            if document_set.root:
                document_set.root = str(Path(document_set.root).resolve())
        if experiment.model.space.init:
            experiment.model.space.init = str(Path(experiment.model.space.init).resolve())

    if shards is None:
        jobs: list[Any] = executor.map_array(stage, experiments, [force] * len(experiments))
        for job, experiment in zip(jobs, experiments, strict=True):
            print(f"submitted {job.job_id}  {experiment.run.name}", flush=True)
        return jobs

    layers = {experiment.run.name: resolve_layers(experiment) for experiment in experiments}
    tasks = [
        (experiment, shard)
        for experiment in experiments
        for shard in split(layers[experiment.run.name], shards)
    ]
    shard_jobs: list[Any] = executor.map_array(
        train,
        [experiment for experiment, _ in tasks],
        [force] * len(tasks),
        [Job.DEVICE] * len(tasks),
        [shard for _, shard in tasks],
    )
    for job, (experiment, shard) in zip(shard_jobs, tasks, strict=True):
        print(f"submitted {job.job_id}  {experiment.run.name} layers {shard}", flush=True)

    array_id = str(shard_jobs[0].job_id).split("_")[0]
    join_parameters = {
        **parameters,
        "gpus_per_node": 0,
        "mem_gb": Job.JOIN_MEM_GB,
        "timeout_min": round(Job.JOIN_HOURS * 60),
        "slurm_additional_parameters": {
            **parameters.get("slurm_additional_parameters", {}),
            "dependency": f"afterok:{array_id}",
        },
    }
    join_executor = slurm.executor(Job.NAME, env.DISTILL, folder, **join_parameters)
    join_jobs = [
        join_executor.submit(join, experiment, layers[experiment.run.name], force)
        for experiment in experiments
    ]
    for job, experiment in zip(join_jobs, experiments, strict=True):
        print(f"submitted {job.job_id}  {experiment.run.name} join", flush=True)
    return shard_jobs + join_jobs


def main() -> None:
    """Parse the command line and submit every run the file describes."""
    parser = argparse.ArgumentParser(description="Train the learned index space, one slurm job per run")
    parser.add_argument("config", help="path to the training yaml")
    slurm.add_flags(parser)
    parser.add_argument("--force", action="store_true", help="retrain a run whose maps are already written")
    parser.add_argument("--shards", type=int, default=None, help="train each run's layers in this many tasks")
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="nodes a run taught online takes, the teacher's copies on all of them",
    )
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")

    from lsa.training.distillation.config import load

    experiments = load(args.config, args.overrides)
    for experiment in experiments:
        teacher = experiment.data.teacher
        targets = f"online from {teacher.extraction}" if teacher is not None else experiment.data.root
        print(
            f"{experiment.run.name}: index_dim {experiment.model.indexer.index_dim}, "
            f"steps {experiment.run.optim.steps}, targets {targets}",
            flush=True,
        )
    if args.dry_run:
        return
    submit(
        experiments,
        {**PARAMETERS, **slurm.overrides(args)},
        args.folder,
        args.force,
        shards=args.shards,
        nodes=args.nodes,
    )
