"""Run one native checkpoint through lm-eval and leave its records, one slurm job a checkpoint with a rank on every GPU.

    lsa native evaluate --suite benchmark configs/training/native/qwen35_0_8b/multires.yaml --checkpoint <step dir> --tasks core
    lsa native evaluate --suite benchmark configs/training/native/qwen35_0_8b/multires.yaml --checkpoint <step dir> --tasks ruler \
        attention.coarse=false --label b16k-multires-fine-only-read

Every rank holds a full replica and lm-eval hands it a share of each task's requests, so the ranks never talk
during a forward. Their results meet on rank 0 through a CPU group, which writes the records.
"""

import argparse
import os
import re
import socket
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import env, paths, slurm
from lsa.config_loader import run_provenance
from lsa.training import require_extra
from lsa.training.native.benchmark import records, tasks
from lsa.training.native.benchmark.lm import BATCH_TOKENS, Backbone, NativeLM, Tokenizer

if TYPE_CHECKING:
    from lsa.training.native.config import NativeConfig
    from lsa.training.native.evaluation import EvaluationModel

# What a job asks for when the command line says nothing
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": 8,
    "cpus_per_task": 8,
    "mem_gb": 400,
    "timeout_min": 8 * 60,
    "nodes": 1,
}

# Seconds a rank waits for the slowest one - ranks finish a task at different times
COLLECTIVE_TIMEOUT = 4 * 3600
# Where records go when the command line names none, under the shared training root
RECORDS = "native/benchmark"
# The seed lm-eval's own CLI defaults to
SEED = 1234
STEP = re.compile(r"step-(\d+)$")


def step_of(checkpoint: str) -> int:
    """Return the step a torchtitan checkpoint folder holds.

    Args:
        checkpoint: The `.../checkpoint/step-N` folder.

    Returns:
        N.

    Raises:
        ValueError: The folder is not named `step-N`.
    """
    match = STEP.search(checkpoint.rstrip("/"))
    if match is None:
        raise ValueError(f"{checkpoint} is not a step folder - expected .../checkpoint/step-<N>")
    return int(match.group(1))


def default_label(checkpoint: str) -> str:
    """Return the name a checkpoint's records are filed under - its run directory's.

    Args:
        checkpoint: The `.../<run>/checkpoint/step-N` folder.

    Returns:
        The run directory's name.
    """
    return Path(checkpoint.rstrip("/")).parent.parent.name


def preset_name(tasks_argument: str) -> str:
    """Return the directory name a `--tasks` value is filed under - a preset's own name, or a list joined by plus.

    Args:
        tasks_argument: The `--tasks` value.

    Returns:
        The name.
    """
    return "+".join(item.strip() for item in tasks_argument.split(","))


def titan_backbone(built: "EvaluationModel", parallelism: Any) -> Backbone:
    """Split a model built for evaluation into the hidden-state forward and the output projection.

    The sparse arms read a sequence as the blocks their layers were built with, so each forward sets the length of
    the batch it is given on every such layer first. The forward is the evaluation's own - torchtitan's input
    preparation and the context it runs under - so a score here is a score of the model as it was trained.

    Args:
        built: The model, its mesh and its forward context, from `build_model`.
        parallelism: The config's parallelism block.

    Returns:
        The backbone.
    """
    import spmd_types as spmd
    import torch
    from torchtitan.tools import utils

    from lsa.training.native.layers import MultiresInnerAttention

    model: Any = built.model
    attentions = [module for module in model.modules() if isinstance(module, MultiresInnerAttention)]

    def hidden(tokens: torch.Tensor) -> torch.Tensor:
        """Run a batch [rows, length] and return its hidden states [rows, length, model_dim]."""
        rows, length = tokens.shape
        for attention in attentions:
            attention.seq_len = length
        flat = tokens.reshape(-1)
        batch = {
            "input": flat,
            "labels": flat,
            "positions": torch.arange(rows * length, device=tokens.device) % length,
            "padding_mask": torch.zeros_like(flat, dtype=torch.bool),
        }
        inputs, _, extra = model.preprocess_inputs(
            batch, parallel_dims=built.parallel_dims, parallelism=parallelism
        )
        with built.forward_context(), spmd.local():
            return model(inputs, **extra).reshape(rows, length, -1)

    def project(states: torch.Tensor) -> torch.Tensor:
        """Project hidden states [n, model_dim] to logits [n, vocab]."""
        with spmd.local():
            return model.lm_head(states)

    return Backbone(hidden=hidden, project=project, device=torch.device(utils.device_type))


def load_backbone(experiment: "NativeConfig", checkpoint: str) -> Backbone:
    """Build the model a config names with the evaluation's own builder, load a checkpoint's weights and split it.

    The config has to name a replica on every rank - `run.dp_shard=1 run.dp_replicate=<ranks>` - because each rank
    runs batches of its own shapes, and a sharded model gathers its parameters in lockstep.

    Args:
        experiment: The experiment, whose attention is the arm.
        checkpoint: The torchtitan step folder.

    Returns:
        The backbone.
    """
    from lsa.training.native.evaluation import build_model, load_model_weights
    from lsa.training.native.trainer import trainer_config

    config = trainer_config(experiment)
    built = build_model(config)
    load_model_weights([built.model], checkpoint)
    return titan_backbone(built, config.parallelism)


def load_tokenizer(experiment: "NativeConfig") -> Tokenizer:
    """Return the tokenizer the model was trained with, as lm-eval needs it.

    Args:
        experiment: The experiment.

    Returns:
        The tokenizer, whose end id is the end-of-text token.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(experiment.tokenizer_path)
    return Tokenizer(
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        decode=lambda ids: str(tokenizer.decode(ids, skip_special_tokens=False)),
        end_id=tokenizer.eos_token_id,
    )


def options_of(
    limit: int | None, max_length: int, batch_tokens: int, max_gen_toks: int | None
) -> dict[str, Any]:
    """Return the options a run was given, for the record.

    Args:
        limit: Examples per task, or None for all.
        max_length: The longest context fed to the model.
        batch_tokens: Tokens one forward holds.
        max_gen_toks: The cap on generated tokens, or None.

    Returns:
        The options by name.
    """
    return {
        "limit": limit,
        "max_length": max_length,
        "batch_tokens": batch_tokens,
        "max_gen_toks": max_gen_toks,
        "seed": SEED,
        "dtype": "bfloat16",
        "generation": "greedy, whole prefix recomputed at every token, no decode cache",
    }


def benchmark(
    experiment: "NativeConfig",
    checkpoint: str,
    names: list[str],
    preset: str,
    label: str,
    records_root: str,
    limit: int | None = None,
    max_length: int | None = None,
    batch_tokens: int = BATCH_TOKENS,
    max_gen_toks: int | None = None,
) -> Path:
    """Run a checkpoint through tasks and write its records, from every rank of a job.

    Args:
        experiment: The experiment, whose attention is the arm.
        checkpoint: The torchtitan step folder.
        names: The tasks, as `tasks.expand` returns them.
        preset: What the tasks are called in the records.
        label: What the checkpoint is called in the records.
        records_root: The records root.
        limit: Examples per task, or None for all.
        max_length: The longest context fed to the model, or None for the longer of its training length and the longest RULER cell.
        batch_tokens: Tokens one forward holds.
        max_gen_toks: A cap on the tokens any task generates, or None.

    Returns:
        The directory the records went into.
    """
    require_extra("lm_eval", "torchtitan")

    import lm_eval
    import torch
    import torch.distributed as dist
    from lm_eval.tasks import TaskManager

    start = time.perf_counter()
    max_length = max_length or max(experiment.data.seq_len, *tasks.lengths_of(names))
    backbone = load_backbone(experiment, checkpoint)
    # The builder started NCCL, which a collective of python objects does not run on
    group = (
        dist.new_group(backend="gloo", timeout=timedelta(seconds=COLLECTIVE_TIMEOUT))
        if dist.is_initialized()
        else None
    )
    lm = NativeLM(backbone, load_tokenizer(experiment), max_length, batch_tokens, max_gen_toks, group)
    loaded = time.perf_counter() - start
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks.build(names),
        task_manager=TaskManager(),
        limit=limit,
        log_samples=True,
        bootstrap_iters=10000,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )
    seconds = gather_seconds(lm)
    folder = records.run_folder(records_root, label, preset)
    if results is not None:
        records.write_records(
            folder,
            results,
            {
                **run_provenance(experiment),
                "checkpoint": checkpoint,
                "step": step_of(checkpoint),
                "label": label,
                "preset": preset,
                "tasks": names,
                "attention": run_provenance(experiment)["run"]["attention"],
                "options": options_of(limit, max_length, batch_tokens, max_gen_toks),
                "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                "lm_eval": lm_eval.__version__,
                "torch": torch.__version__,
                "host": socket.gethostname(),
                "job": os.environ.get("SLURM_JOB_ID"),
                "seconds": {
                    "load": round(loaded, 1),
                    "total": round(time.perf_counter() - start, 1),
                    **seconds,
                },
            },
        )
    lm.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return folder


def gather_seconds(lm: NativeLM) -> dict[str, float]:
    """Return the seconds each request type took, on the slowest rank, since the job waits for it.

    Args:
        lm: The model, whose ranks have all finished their requests.

    Returns:
        Request type to seconds.
    """
    gathered = lm.gather_object(lm.seconds) or [lm.seconds]
    names = {name for part in gathered for name in part}
    return {name: round(max(part.get(name, 0.0) for part in gathered), 1) for name in sorted(names)}


def run_task(
    experiment: "NativeConfig",
    checkpoint: str,
    tasks_argument: str,
    label: str,
    records_root: str,
    extra: Mapping[str, Any],
) -> Path:
    """Benchmark one checkpoint inside a slurm task, having first become a rank.

    Args:
        experiment: The experiment.
        checkpoint: The torchtitan step folder.
        tasks_argument: The `--tasks` value.
        label: What the checkpoint is called in the records.
        records_root: The records root.
        extra: The remaining options of `benchmark`.

    Returns:
        The records directory.
    """
    env.adopt_job_environment()
    return benchmark(
        experiment,
        checkpoint,
        tasks.expand(tasks_argument),
        preset_name(tasks_argument),
        label,
        records_root,
        **extra,
    )


def submit(
    experiment: "NativeConfig",
    checkpoint: str,
    tasks_argument: str,
    label: str,
    records_root: str,
    extra: Mapping[str, Any],
    parameters: Mapping[str, Any] = PARAMETERS,
    folder: str | Path | None = None,
) -> Any:
    """Submit one checkpoint's benchmark as one slurm job and print its id.

    Args:
        experiment: The experiment.
        checkpoint: The torchtitan step folder.
        tasks_argument: The `--tasks` value.
        label: What the checkpoint is called in the records.
        records_root: The records root.
        extra: The remaining options of `benchmark`.
        parameters: What the job asks for, as submitit parameters. Its GPUs per node are also its tasks per node.
        folder: Where submitit writes the job's output, or None for the shared log root.

    Returns:
        The submitit job.
    """
    executor = slurm.executor(
        "lsa-native-benchmark",
        (*env.OFFLINE_HF, *env.exports(env.DISTRIBUTED_TORCH)),
        folder,
        **{**parameters, "tasks_per_node": parameters["gpus_per_node"]},
    )
    job = executor.submit(run_task, experiment, checkpoint, tasks_argument, label, records_root, extra)
    print(f"submitted {job.job_id}  {label} {tasks_argument}", flush=True)
    return job


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add what `lsa native evaluate --suite benchmark` takes to its parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("config", help="path to the native yaml that names the checkpoint's attention")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="the torchtitan step folder, .../checkpoint/step-N - the config's run.evaluation.checkpoint by default",
    )
    parser.add_argument(
        "--tasks", default="core", help=f"a preset in {sorted(tasks.PRESETS)} or tasks, comma-separated"
    )
    parser.add_argument(
        "--records", default=None, help="the records root - beside the checkpoint's run by default"
    )
    parser.add_argument(
        "--label", default=None, help="what the checkpoint is called in the records - its run's name"
    )
    parser.add_argument("--limit", type=int, default=None, help="examples per task, for a smoke test")
    parser.add_argument(
        "--max-length", type=int, default=None, help="longest context fed to the model - the training length"
    )
    parser.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS, help="tokens one forward holds")
    parser.add_argument("--max-gen-toks", type=int, default=None, help="cap on the tokens any task generates")
    parser.add_argument(
        "overrides", nargs="*", help="dotted key=value assignments to the config, applied last"
    )
    slurm.add_flags(parser)


def command(args: argparse.Namespace) -> None:
    """Submit the benchmark a parsed command line describes.

    Args:
        args: The parsed arguments of `add_arguments`.
    """
    require_extra("omegaconf", "hydra")

    from lsa.training.native.config import load

    parameters = {**PARAMETERS, **slurm.overrides(args)}
    ranks = parameters["gpus_per_node"] * parameters["nodes"]
    # A replica on every rank, then the user's own overrides
    [experiment] = load(args.config, ["run.dp_shard=1", f"run.dp_replicate={ranks}", *args.overrides])
    tasks.expand(args.tasks)
    checkpoint = experiment.scored_checkpoint(args.checkpoint)
    label = args.label or default_label(checkpoint)
    records_root = args.records or str(paths.training() / RECORDS)
    extra = {
        "limit": args.limit,
        "max_length": args.max_length,
        "batch_tokens": args.batch_tokens,
        "max_gen_toks": args.max_gen_toks,
    }
    print(
        f"{label}: step {step_of(checkpoint)}, {experiment.attention.kind}, tasks {args.tasks}, records {records_root}",
        flush=True,
    )
    if args.dry_run:
        return
    submit(experiment, checkpoint, args.tasks, label, records_root, extra, parameters, args.folder)
