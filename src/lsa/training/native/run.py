"""Train a native model under the attention one file describes, or score a checkpoint of one, one slurm job per run.

    lsa native train configs/training/native/qwen35_0_8b/multires.yaml --gpus 8 --hours 4
    lsa native train configs/training/native/qwen35_0_8b/dense.yaml --dry-run run.optim.steps=20
    lsa native evaluate configs/training/native/qwen35_0_8b/multires.yaml \\
        --checkpoint ${LSA_ROOT}/training/native/qwen35_0_8b/b16k-multires/checkpoint/step-8000 attention.coarse=false
    lsa native evaluate configs/training/native/qwen35_0_8b/multires.yaml --suite benchmark \\
        --checkpoint ${LSA_ROOT}/training/native/qwen35_0_8b/b16k-multires/checkpoint/step-8000 --tasks core

A run is many processes, one per GPU. submitit submits `run_task` rather than a script, so a job runs this
interpreter and this checkout with nothing to activate, and each task fills in the variables `torchrun` would have
set before it trains.

The trainer is torchtitan's. What this file does is hand it the config [the trainer module](trainer.py) built from
the yaml, and write beside what torchtitan writes a record of the run that produced it.
An evaluation builds the same model, loads a checkpoint's weights and runs [the held-out evaluation](evaluation.py) once,
writing the same records under the run's evaluation folder at the checkpoint's step.
"""

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import env, slurm
from lsa.config_loader import run_provenance
from lsa.io import write_json
from lsa.training import require_extra
from lsa.training.native.constants import Job, Results

# The sets a checkpoint is scored on - the held-out set of the run, or the standard benchmarks through lm-eval
SUITES = ("held-out", "benchmark")

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
if TYPE_CHECKING:
    from lsa.training.native.config import NativeConfig


def train(experiment: "NativeConfig") -> Path:
    """Train one run and return the directory it wrote into.

    Args:
        experiment: The experiment.

    Returns:
        The run directory.
    """
    require_extra("omegaconf", "torchtitan")

    from torchtitan.tools.logging import init_logger

    from lsa.training.native.trainer import check_label_contract, trainer_config

    # The trainer logs through this and torchtitan's own entry point is what normally installs it
    init_logger()
    check_label_contract()

    config = trainer_config(experiment)
    out = experiment.out
    if _is_leader():
        write_json(out / Results.CONFIG, meta(experiment))
    if experiment.run.runtime.wandb:
        out.mkdir(parents=True, exist_ok=True)
        for name, value in env.wandb_run(experiment.run.name, str(out)).items():
            os.environ.setdefault(name, value)

    trainer = config.build()
    try:
        trainer.train()
    finally:
        trainer.close()
    return out


def evaluate(experiment: "NativeConfig", checkpoint: str) -> Path:
    """Score one checkpoint on the held-out set and return the directory the records went to.

    Args:
        experiment: The experiment, whose attention and held-out block the checkpoint is scored under.
        checkpoint: A torchtitan step folder, `.../checkpoint/step-N`.

    Returns:
        The run directory.
    """
    require_extra("omegaconf", "torchtitan")

    import torch.distributed as dist
    from torchtitan.tools.logging import init_logger

    from lsa.training.native.evaluation import (
        HeldOutEvaluator,
        build_model,
        checkpoint_step,
        evaluator_config,
        load_model_weights,
    )
    from lsa.training.native.trainer import check_label_contract, trainer_config

    init_logger()
    check_label_contract()

    step = checkpoint_step(checkpoint)
    config = trainer_config(experiment)
    out = experiment.out
    if _is_leader():
        write_json(out / Results.CONFIG, {**meta(experiment), "checkpoint": checkpoint, "step": step})

    built = build_model(config)
    load_model_weights([built.model], checkpoint)
    evaluator = HeldOutEvaluator(
        evaluator_config(experiment),
        parallelism=config.parallelism,
        dp_world_size=built.dp_world_size,
        dp_rank=built.dp_rank,
        parallel_dims=built.parallel_dims,
        validation_context=built.forward_context,
        metrics_processor=None,
        seq_len=experiment.data.seq_len,
        num_tokens_per_batch=experiment.run.optim.tokens_per_rank,
    )
    evaluator.validate([built.model], step)
    dist.barrier()
    dist.destroy_process_group()
    return out


def meta(experiment: "NativeConfig") -> dict[str, Any]:
    """Record what produced the numbers next to them.

    Args:
        experiment: The experiment.

    Returns:
        The metadata block, ready to write.
    """
    import torch

    return {
        **run_provenance(experiment),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _is_leader() -> bool:
    """Return whether this process is the one that writes shared files."""
    return int(os.environ.get("RANK", 0)) == 0


def describe(experiment: "NativeConfig") -> str:
    """Return the summary of a run, naming what it reads.

    Args:
        experiment: The experiment.

    Returns:
        The summary, as the submitter and the trainer both print it.
    """
    attention = experiment.attention
    detail = (
        f"{attention.kind}, coarse {attention.coarse}, variance {attention.variance}, partition "
        f"{attention.partition}, rank {attention.rank}, window {attention.window_size}, sink {attention.sink_size}, "
        f"top_k {attention.top_k}, top_p {attention.selection_width}, bucket {attention.bucket_size}/"
        f"{attention.bucket_cap}, block {attention.block_size}"
        if attention.kind == "multires"
        else f"lsa, window {attention.window_size}, sink {attention.sink_size}, top_k {attention.top_k}, top_p "
        f"{attention.selection_width}, bucket {attention.bucket_size}/{attention.bucket_cap}, pool "
        f"{attention.pool_size}, block {attention.block_size}"
        if attention.kind == "lsa"
        else f"{attention.kind}, window {attention.window_size}, sink {attention.sink_size}"
    )
    if attention.kind == "nsa":
        detail = (
            f"nsa, window {attention.window_size}, compression {attention.compress_block_size}/"
            f"{attention.compress_stride}, compress_hidden_dim {attention.compress_hidden_dim}, "
            f"selection {attention.select_block_count} blocks of {attention.select_block_size}, "
            f"key_budget {attention.key_budget} plus compressed tokens"
        )
    if attention.kind == "pisa":
        detail = (
            f"pisa, leaf_block_size {attention.leaf_block_size}, branching {attention.branching}, "
            f"level_block_count {attention.level_block_count}, local_block_count {attention.local_block_count}, "
            f"key_budget {attention.key_budget}"
        )
    evaluation = (
        f"{experiment.run.evaluation.sequences} held-out sequences every {experiment.run.evaluation.every} steps"
        if experiment.run.evaluation.enabled
        else "none"
    )
    return (
        f"{experiment.run.name}: qwen3_5/{experiment.model.flavor} from {experiment.run.init_from or 'random init'} "
        f"at {experiment.data.seq_len}, "
        f"steps {experiment.run.optim.steps}\n"
        f"  attention: {detail}\n"
        f"  schedule: {experiment.run.optim.schedule}, lr {experiment.run.optim.lr}, "
        f"warmup {experiment.run.optim.warmup_steps}, global tokens {experiment.run.optim.tokens_per_step}\n"
        f"  seeds: model {experiment.run.runtime.seed}, data {experiment.data.seed}\n"
        f"  tokenizer: {experiment.tokenizer_path}\n"
        f"  corpus:    {describe_corpora(experiment)}\n"
        f"  evaluate:  {evaluation}"
    )


def describe_corpora(experiment: "NativeConfig") -> str:
    """Return what each corpus a run trains on is and how it is weighted.

    Args:
        experiment: The experiment.

    Returns:
        One phrase a corpus, joined by commas.
    """
    from lsa import assets

    phrases = []
    for card, weight in experiment.data.cards:
        if isinstance(card, assets.TokenizedCard):
            kind = f"tokenized by {card.tokenizer}, {card.sample_length} tokens a row"
        elif isinstance(card, assets.ShardCard):
            kind = f"token shards of {card.tokenizer}"
        else:
            kind = "text, tokenized as it is read"
        phrases.append(f"{card.name} x{weight:g} -> {kind}")
    return ", ".join(phrases)


# What a job asks for when the command line says nothing
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": Job.GPUS,
    "cpus_per_task": Job.CPUS,
    "mem_gb": Job.MEM_GB,
    "timeout_min": Job.HOURS * 60,
    "nodes": Job.NODES,
}


def run_task(experiment: "NativeConfig") -> Path:
    """Train one run inside a slurm task, having first become a rank.

    Args:
        experiment: The experiment.

    Returns:
        The directory the run wrote into.
    """
    env.adopt_job_environment()
    return train(experiment)


def evaluate_task(experiment: "NativeConfig", checkpoint: str) -> Path:
    """Score one checkpoint inside a slurm task, having first become a rank.

    Args:
        experiment: The experiment.
        checkpoint: A torchtitan step folder.

    Returns:
        The directory the records were written into.
    """
    env.adopt_job_environment()
    return evaluate(experiment, checkpoint)


def submit(
    experiments: list["NativeConfig"],
    parameters: Mapping[str, Any] = PARAMETERS,
    folder: str | Path | None = None,
    checkpoints: list[str] | None = None,
) -> list[Any]:
    """Submit one job per run, print each one's id, and return them without waiting.

    Args:
        experiments: The experiments, as `config.load` returns them.
        parameters: What each job asks for, as submitit parameters. Its GPUs per node are also its tasks per node.
        folder: Where submitit writes each job's output, or None for the shared log root.
        checkpoints: The step folder each run scores instead of training, in the order of the runs, or None to train.

    Returns:
        One submitit job per run, in the order they were given.
    """
    executor = slurm.executor(
        Job.NAME,
        env.exports(env.DISTRIBUTED_TORCH),
        folder,
        **{**parameters, "tasks_per_node": parameters["gpus_per_node"]},
    )
    # A compute node does not start where the submission did, so a relative path has to be resolved here
    for experiment in experiments:
        experiment.run.output = str(Path(experiment.run.output).resolve())

    if checkpoints is None:
        jobs: list[Any] = executor.map_array(run_task, experiments)
    else:
        jobs = executor.map_array(evaluate_task, experiments, checkpoints)
    for job, experiment in zip(jobs, experiments, strict=True):
        print(f"submitted {job.job_id}  {experiment.run.name}", flush=True)
    return jobs


def main(command: str) -> None:
    """Parse the command line and submit every run the file describes.

    Args:
        command: `train` to train each run, `evaluate` to score a checkpoint under each - on the held-out set,
            or on the standard benchmarks when `--suite benchmark` says so.
    """
    evaluating = command == "evaluate"
    arguments = sys.argv[1:]
    if evaluating:
        # The suite decides which options exist
        # => read it before the parser that holds them is built
        named = argparse.ArgumentParser(add_help=False)
        named.add_argument("--suite", choices=SUITES, default=SUITES[0])
        suite, arguments = named.parse_known_args(arguments)
        if suite.suite == "benchmark":
            from lsa.training.native.benchmark import run as benchmark

            parser = argparse.ArgumentParser(
                prog=sys.argv[0], description="Benchmark a native checkpoint on standard tasks, one slurm job"
            )
            parser.add_argument(
                "--suite", choices=SUITES, default=SUITES[0], help="the held-out set or the benchmarks"
            )
            benchmark.add_arguments(parser)
            benchmark.command(parser.parse_args(arguments))
            return
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Score a native checkpoint on the held-out set, one slurm job per run"
        if evaluating
        else "Train Qwen3.5 under a native attention, one slurm job per run",
    )
    parser.add_argument("config", help="path to the native yaml")
    if evaluating:
        parser.add_argument(
            "--suite",
            choices=SUITES,
            default=SUITES[0],
            help="the set to score - `benchmark` takes the options of `--suite benchmark --help`",
        )
        parser.add_argument(
            "--checkpoint",
            default=None,
            help="a torchtitan step folder, .../checkpoint/step-N - the config's run.evaluation.checkpoint by default",
        )
    slurm.add_flags(parser)
    parser.add_argument("--nodes", type=int, default=Job.NODES, help="nodes per job")
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args(arguments)

    require_extra("omegaconf", "hydra")

    from lsa.training.native.config import load

    experiments = load(args.config, args.overrides)
    checkpoints = (
        [experiment.scored_checkpoint(args.checkpoint) for experiment in experiments] if evaluating else None
    )
    parameters = {**PARAMETERS, **slurm.overrides(args), "nodes": args.nodes}
    for experiment in experiments:
        experiment.check_ranks(parameters["gpus_per_node"] * parameters["nodes"])
        print(describe(experiment), flush=True)
    if args.dry_run:
        return

    submit(experiments, parameters, args.folder, checkpoints)
