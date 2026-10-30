"""Time the modules a layer runs besides its attention, inside the image the server uses, as a one-GPU job.

    lsa kernels modules configs/evaluation/kernels/glm53_flash/native.yaml --phase decode

The measurement is [measure.modules](measure/modules.py), which loads the same truncated checkpoint the
attention-layer harness loads and times its feed-forward, its mHC boundaries, the embedding, the final norm
and the head. This builds its command line from a kernel config, runs it in that config's container with the
checkout installed the way the harness installs it, and puts the raw file under a run directory of its own.

The job holds one GPU, and the container is started inside it as an overlapping step.
"""

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from lsa import env, slurm
from lsa.evaluation import config as experiment_config
from lsa.evaluation.accuracy.records import build_run_id, model_label, records_dir
from lsa.evaluation.constants import Evaluation
from lsa.evaluation.performance.kernels.harness import MEASURE_VARIABLES, install_line
from lsa.sglang.constants import Server
from lsa.sglang.launch import SglangConfig, cache_variables, checkout_root, container_command

# What the run id names the measurement by, where an attention run names its configuration
LABEL = "modules"

# What the container runs, from the checkout installed beside sglang
TIMER_MODULE = "lsa.evaluation.performance.kernels.measure.modules"

# What the job asks for - one GPU, the threads the server's step needs for it, and an hour for one phase
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": 1,
    "cpus_per_task": Server.CPUS_PER_GPU,
    "mem_gb": Server.MEM_GB,
    "timeout_min": 60,
}


def modules_args(checkpoint: str, phase: str, output: Path, repeats: int) -> list[str]:
    """Build the timer's command line.

    Args:
        checkpoint: The truncated checkpoint - the one the attention-layer harness loads.
        phase: `prefill` for one chunk eagerly, `decode` for one token replayed as a graph.
        output: Where the raw file goes.
        repeats: Timed calls per module.

    Returns:
        The argv, starting at the interpreter.
    """
    return [
        "python",
        "-m",
        TIMER_MODULE,
        "--model-path",
        checkpoint,
        "--phase",
        phase,
        "--repeats",
        str(repeats),
        "--output",
        str(output),
    ]


def modules_command(
    backend: SglangConfig, checkpoint: str, phase: str, output: Path, repeats: int
) -> list[str]:
    """Wrap the timer so it runs inside the pinned image with the harness's environment.

    Args:
        backend: The launch block, which names the image and the runner.
        checkpoint: The truncated checkpoint.
        phase: Which half of serving to time.
        output: Where the raw file goes.
        repeats: Timed calls per module.

    Returns:
        The argv for `subprocess.run`.
    """
    # The allocator and precompile settings the harness runs under, so a module and a layer are timed alike
    exports = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in {**cache_variables(), **MEASURE_VARIABLES}.items()
    )
    measure = " ".join(shlex.quote(part) for part in modules_args(checkpoint, phase, output, repeats))
    return container_command(backend, f"{install_line()} && exec env {exports} {measure}")


def time_modules(experiment: experiment_config.KernelsConfig, phase: str, repeats: int, output: Path) -> Path:
    """Run the module timer in the config's container and return the raw file it wrote.

    Args:
        experiment: The kernel config - its image, checkpoint and model.
        phase: `prefill` or `decode`.
        repeats: Timed calls per module.
        output: Where the raw file goes.

    Returns:
        The raw file.

    Raises:
        SystemExit: The timer failed, or wrote nothing.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = modules_command(experiment.server(), experiment.data.checkpoint, phase, output, repeats)
    print(" ".join(shlex.quote(part) for part in argv), flush=True)
    status = subprocess.run(argv, cwd=checkout_root()).returncode
    if status or not output.is_file():
        raise SystemExit(
            f"the module timer exited {status} and left {'a' if output.is_file() else 'no'} {output}"
        )
    print(output, flush=True)
    return output


def main() -> None:
    """Submit the timing of one phase's modules for the model a kernel config names."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="a kernel config - its image, checkpoint and model are used")
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="the results root, run lands under performance/kernels - default the shared one",
    )
    parser.add_argument("--run-id", default="", help="the run directory to write - default a new one")
    slurm.add_flags(parser)
    args = parser.parse_args()

    experiment = experiment_config.load(args.config, kind=Evaluation.KERNELS)[0]
    if not isinstance(experiment, experiment_config.KernelsConfig) or not experiment.data.checkpoint:
        raise ValueError(f"{args.config}: data.checkpoint names no truncated checkpoint")
    run_id = args.run_id or build_run_id(
        f"{model_label(experiment.model.repo or experiment.model.name)}__{LABEL}__{args.phase}"
    )
    output = (Path(records_dir(Evaluation.KERNELS, args.results)) / run_id / "modules.json").resolve()
    print(f"{experiment.name}: {args.phase} modules into {output}", flush=True)
    slurm.submit(
        args,
        "lsa-modules",
        env.OFFLINE_HF,
        PARAMETERS,
        time_modules,
        experiment,
        args.phase,
        args.repeats,
        output,
    )
