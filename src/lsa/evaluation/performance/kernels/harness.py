"""Drive the attention-layer harness, one process per measured context.

The harness is [measure.layer](measure/layer.py), which is run as a module rather than imported: it loads
a checkpoint into sglang's serving path and needs the container the server uses. So this builds its command
line, runs it in that container and rebuilds the record after every context - an interrupted ladder still
has a readable record of the points it reached.

A context whose samples already exist is kept. That is what makes a rerun finish a ladder rather than
repeat it, and the ladder is ordered so the cheap points land first.
"""

import json
import shlex
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import io
from lsa.evaluation.constants import Attention, Records
from lsa.evaluation.performance.kernels import record
from lsa.sglang.constants import Server
from lsa.sglang.launch import cache_variables, checkout_root, container_command

if TYPE_CHECKING:
    from lsa.evaluation.config import KernelsConfig

# The score buffer is the one large allocation and the default pool fragments at 10M
# => one expanding segment. The DeepGEMM precompile of every shape is six minutes the layer never needs
MEASURE_VARIABLES: dict[str, str] = {
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}


# What the container runs, from the checkout installed beside sglang
HARNESS_MODULE = "lsa.evaluation.performance.kernels.measure.layer"


def install_line() -> str:
    """Return the shell line that enters the checkout and installs it beside sglang, without its dependencies.

    Returns:
        One shell line, for the measurement to follow with `&&`.
    """
    root = shlex.quote(str(checkout_root()))
    return f"cd {root} && pip install --no-deps --quiet ."


def harness_args(experiment: "KernelsConfig", settings: Path | None, context: int, output: Path) -> list[str]:
    """Build the harness command line for one context.

    The phase decides the last flag and only the last flag: prefill measures a window of chunks up to the
    context, decode measures one token and compares it against its CUDA graph replay. The harness refuses
    a graph comparison over a window, which is why they are exclusive rather than both.

    Args:
        experiment: The kernel run - the variant to time from `model.attention`, the truncated checkpoint and
            what to measure from `data`, the devices it is sharded over from `run.tp`.
        settings: The written design settings, which an LSA or PISA run is enabled with.
        context: The context to measure at.
        output: Where the raw file goes - the harness suffixes it per chunk or token.

    Returns:
        The argv, starting at the interpreter.

    Raises:
        ValueError: The run is enabled with a selector and has no settings written for it.
    """
    kernels = experiment.data
    attention = experiment.model.attention
    selects = attention in (Attention.LSA, Attention.PISA)
    if selects and settings is None:
        raise ValueError(f"a {attention} kernel run is enabled with written settings and was handed none")
    argv = [
        "python",
        "-m",
        HARNESS_MODULE,
        "--model-path",
        kernels.checkpoint,
        "--prompt-file",
        kernels.prompt,
        "--indexer",
        attention,
        "--phase",
        kernels.phase,
        "--context",
        str(context),
        "--chunk",
        str(kernels.chunk),
        "--layer",
        str(kernels.layer),
        "--tp-size",
        str(experiment.run.tp),
        "--warmup",
        str(kernels.warmup),
        "--repeats",
        str(kernels.trials),
        "--output",
        str(output),
        "--repeat-prompt",
        "--diagnose-invalid-topk",
        "--dsa-prefill-backend",
        kernels.prefill_backend,
        "--dsa-decode-backend",
        kernels.decode_backend,
    ]
    if selects:
        argv += ["--indexer-config", str(settings)]
    # The card's rotation, which a server reads through the same override
    if experiment.model.rope_parameters:
        override = {Server.TEXT_CONFIG: {"rope_parameters": experiment.model.rope_parameters}}
        argv += ["--model-override", json.dumps(override)]
    if kernels.phase == "decode":
        # A descending decode reads a mirror at fixed addresses, which a graph follows - unless the node buckets are shaped
        # => only that design runs eagerly, with no graph to compare
        indexer = experiment.model.indexer
        shaped = indexer.top_level < 1 or indexer.node_bucket_shape is not None
        if not (experiment.model.serving.decode == "descend" and attention == Attention.LSA and shaped):
            argv += ["--graph-compare"]
        if kernels.offsets:
            argv += ["--decode-offsets", *(str(offset) for offset in kernels.offsets)]
    else:
        argv += ["--window-chunks", str(kernels.window_chunks)]
    return argv + list(experiment.run.args)


def measure_command(
    experiment: "KernelsConfig", settings: Path | None, context: int, output: Path
) -> list[str]:
    """Wrap the harness so it runs inside the pinned image, with the checkout installed beside sglang.

    It runs from the checkout root, which is where the harness reads its commit and diff from - the
    installed copy has neither.

    The variables ride in the shell line rather than in the child's environment, because a container
    started by `docker run` does not inherit one and a measurement that silently loses its allocator
    setting fragments at the long contexts rather than failing.

    Args:
        experiment: The kernel run, whose launch names the image and the runner.
        settings: The written design settings, or None.
        context: The context to measure at.
        output: Where the raw file goes.

    Returns:
        The argv for `subprocess.run`.
    """
    backend = experiment.server()
    argv = harness_args(experiment, settings, context, output)
    exports = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in {**cache_variables(), **MEASURE_VARIABLES}.items()
    )
    measure = " ".join(shlex.quote(part) for part in argv)
    return container_command(backend, f"{install_line()} && exec env {exports} {measure}")


def measure(
    experiment: "KernelsConfig",
    settings: Path | None,
    context: int,
    run_dir: Path,
    log: Callable[[str], None],
) -> bool:
    """Measure one context, unless it is already measured.

    A harness process that exits non-zero has its samples dropped, because a partial window is a mean over
    whichever chunks happened to finish rather than over the window that was asked for.

    Args:
        experiment: The kernel run.
        settings: The written design settings, or None.
        context: The context to measure at.
        run_dir: The run directory the samples land under.
        log: Where progress goes.

    Returns:
        Whether the context now has samples.
    """
    if record.sample_paths(run_dir, context):
        log(f"  {context}: samples exist - kept")
        return True
    output = run_dir / "samples" / f"{context}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"  {context}: {io.utc_now()}")
    argv = measure_command(experiment, settings, context, output)
    status = subprocess.run(argv, cwd=checkout_root()).returncode
    if status:
        log(f"  {context}: harness exit {status} - its samples are dropped")
        for path in record.sample_paths(run_dir, context):
            path.unlink()
        return False
    if not record.sample_paths(run_dir, context):
        log(f"  {context}: the harness left no samples in {run_dir}")
        return False
    return True


def write(run_dir: Path, run_id: str, meta: dict[str, Any], contexts: list[int], failed: list[int]) -> Path:
    """Write one run's record, atomically.

    The ladder rewrites it after every context, so a reader can hit it mid-write - and a records reader that
    parsed half a record would drop a run from its listing rather than show it climbing.

    Args:
        run_dir: The run directory.
        run_id: Its name.
        meta: What the launcher knows about the run.
        contexts: The contexts the run asked for.
        failed: The contexts whose harness process did not finish.

    Returns:
        The record's path.
    """
    path = run_dir / Records.RECORD
    io.write_json(path, record.build(run_id, meta, run_dir, contexts, failed), indent=1)
    return path


def rebuild(run_dir: Path) -> Path:
    """Rebuild a run's record from its samples, keeping the facts only the launcher knew.

    Needs no GPU and no container, which is what makes it the way to pick up a record whose writer was
    killed between a harness process and the rewrite that follows it.

    Args:
        run_dir: The run directory.

    Returns:
        The record's path.
    """
    path = run_dir / Records.RECORD
    old = io.read_json(path, {})
    run = old.get("run", {})
    meta = {key: run.get(key) for key in ("note", "tag", "started", "date", "seconds")}
    meta["model"] = old.get("model", {}).get("name", "")
    meta["name"] = run.get("name", "")
    meta["image"] = old.get("setup", {}).get("image", "")
    meta["approximate_selection"] = old.get("setup", {}).get("approximate_selection")
    return write(run_dir, run_dir.name, meta, record.contexts_of(run_dir), old.get("failed", []))


def run_kernels(experiment: "KernelsConfig", settings: Path | None, label: str, run_dir: Path) -> Path:
    """Measure every context of one ladder, rewriting the record after each.

    The record names the model by the hub id `model.name` resolves to, which the truncated checkpoint's own
    name does not carry.

    Args:
        experiment: The kernel run.
        settings: The written design settings, or None.
        label: The configuration label the record is grouped by.
        run_dir: The run directory.

    Returns:
        The record's path.

    Raises:
        RuntimeError: No context produced samples, so there is no record to write.
    """
    kernels = experiment.data
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = io.read_json(run_dir / Records.RECORD, {})
    started = existing.get("run", {}).get("started") or io.utc_now()
    # A resumed ladder carries the wall clock it already spent - counting only this attempt would report
    # a three-hour run as the twenty minutes of it that ran after the job was requeued
    spent = existing.get("run", {}).get("seconds") or 0.0
    clock = time.perf_counter()
    print(
        f"{run_dir.name}  {label}  {kernels.phase}  contexts {kernels.contexts}  repeats {kernels.trials}",
        flush=True,
    )

    failed = []
    for context in kernels.contexts:
        if not measure(experiment, settings, context, run_dir, print):
            failed.append(context)
        if record.contexts_of(run_dir):
            meta = {
                "note": experiment.run.note,
                "tag": experiment.run.tag,
                "started": started,
                "date": io.utc_now(),
                "seconds": round(spent + time.perf_counter() - clock, 1),
                "model": experiment.model.repo,
                "name": label,
                "approximate_selection": kernels.approximate_selection,
                "image": experiment.run.image,
            }
            write(run_dir, run_dir.name, meta, kernels.contexts, failed)
    if not record.contexts_of(run_dir):
        raise RuntimeError(f"{run_dir}: every context failed - nothing was measured")
    print(
        f"done  {len(kernels.contexts) - len(failed)}/{len(kernels.contexts)} contexts  {run_dir}", flush=True
    )
    return run_dir / Records.RECORD
