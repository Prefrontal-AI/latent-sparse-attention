"""Profile every kernel, CuTeDSL and Triton, with Nsight Compute on one GPU and write the roofline JSON, as a one-GPU job.

    lsa kernels roofline --output $LSA_ROOT/roofline
    lsa kernels roofline --output DIR --contexts 262144 --steps "bucket scan" "expansion by threshold"
    lsa kernels roofline --output DIR --kernels triton
    lsa kernels roofline --output DIR --steps "decode select" --graph --batch 1 4 8

The job runs [profile](profile.py) twice - once for the timings with no tool attached, since `ncu` slows every launch,
then under `ncu` with the full section set and the counters the roofline needs. It then reads the report into `roofline.json` and renders `roofline.html` from it, beside it. Every step runs under each arm of `--kernels` - both by default, so a step reads side by side in `triton` and `cutedsl` on the same inputs.
The run holds one GPU of the site's open partition, which is the one outside its reservation.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from lsa import env, site, slurm
from lsa.evaluation.performance.roofline import page, report, workloads

# Where the CUDA toolkit keeps ncu, where it is not on the path
NCU = "/usr/local/cuda/bin/ncu"

PROFILE_MODULE = "lsa.evaluation.performance.roofline.profile"

DEFAULT_CONTEXTS = (131072, 262144, 1048576, 2097152, 10485760)

# What the job asks for - one GPU, never the reservation
PARAMETERS: dict[str, Any] = {
    "gpus_per_node": 1,
    "cpus_per_task": 16,
    "mem_gb": 128,
    "timeout_min": 120,
}


def parameters() -> dict[str, Any]:
    """Return what the job asks for, in the site's open partition if it has one.

    Returns:
        `PARAMETERS`, plus `slurm_partition` when the site names an `open_partition`.
    """
    partition = site.slurm_values().get("open_partition")
    return {**PARAMETERS, **({"slurm_partition": partition} if partition else {})}


def ncu_command(output: Path, profile_argv: list[str]) -> list[str]:
    """Build the Nsight Compute command around the profile.

    Args:
        output: The report's path without its suffix.
        profile_argv: The profile's command line, starting at the interpreter.

    Returns:
        The argv for `subprocess.run`.
    """
    ncu = NCU if Path(NCU).is_file() else shutil.which("ncu")
    if ncu is None:
        raise FileNotFoundError(f"no ncu at {NCU} or on the path")
    return [
        ncu,
        "--profile-from-start",
        "off",
        "--nvtx",
        "--set",
        "full",
        "--clock-control",
        "none",
        "--metrics",
        ",".join(report.EXTRA_METRICS),
        "--import-source",
        "no",
        "-f",
        "-o",
        str(output),
        *profile_argv,
    ]


def profile_kernels(
    output: Path,
    contexts: list[int],
    steps: list[str],
    kernels: list[str],
    repeats: int,
    batches: list[int] | None = None,
) -> Path:
    """Time the steps outside ncu, profile them under it and write the roofline JSON.

    Args:
        output: The directory that gets the report, the timings and `roofline.json`.
        contexts: Tokens in the request, one profile pass each.
        steps: Step names, or `all`.
        kernels: The arms to profile each step in.
        repeats: Timed calls per step and arm.
        batches: Batches to time a graph replay at, or None for no graph timing.

    Returns:
        The roofline JSON.

    Raises:
        SystemExit: The profile failed, or wrote no report.
    """
    output.mkdir(parents=True, exist_ok=True)
    timings = output / "timings.json"
    profile_argv = [
        sys.executable,
        "-m",
        PROFILE_MODULE,
        "--contexts",
        *map(str, contexts),
        "--steps",
        *steps,
        "--kernels",
        *kernels,
        "--timings",
        str(timings),
        "--repeats",
        str(repeats),
    ]
    if batches is not None:
        profile_argv += ["--graph", "--batch", *map(str, batches)]
    # ncu slows every launch, profiled or not => time outside it, profile inside it
    timed = subprocess.run([*profile_argv, "--mode", "time"]).returncode
    if timed:
        raise SystemExit(f"the timing pass exited {timed}")
    argv = ncu_command(output / "roofline", [*profile_argv, "--mode", "profile"])
    print(" ".join(shlex.quote(part) for part in argv), flush=True)
    status = subprocess.run(argv).returncode
    if status or not (output / "roofline.ncu-rep").is_file():
        raise SystemExit(f"ncu exited {status} and left no {output / 'roofline.ncu-rep'}")
    roofline = report.write_roofline(output / "roofline.ncu-rep", timings, output / "roofline.json")
    print(roofline, flush=True)
    print(page.render(roofline, output / "roofline.html"), flush=True)
    return roofline


def main() -> None:
    """Submit the profile of every kernel."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="where the report and the JSON go")
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--steps", nargs="+", default=["all"], help="step names, or all")
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=workloads.KERNELS,
        default=list(workloads.KERNELS),
        help="the arms to profile each step in",
    )
    parser.add_argument("--repeats", type=int, default=20, help="timed calls per step")
    parser.add_argument("--graph", action="store_true", help="also time each step replayed from a CUDA graph")
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1], help="requests, for the steps that batch"
    )
    slurm.add_flags(parser)
    args = parser.parse_args()

    output = args.output.resolve()
    print(f"profiling {' '.join(args.steps)} at {args.contexts} into {output}", flush=True)
    slurm.submit(
        args,
        "lsa-roofline",
        env.OFFLINE_HF,
        parameters(),
        profile_kernels,
        output,
        args.contexts,
        args.steps,
        args.kernels,
        args.repeats,
        args.batch if args.graph else None,
    )
