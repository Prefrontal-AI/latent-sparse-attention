"""Launch each step once inside a profiler window, so Nsight Compute sees one call per step and context.

    ncu --profile-from-start off --nvtx python -m lsa.evaluation.performance.roofline.profile --contexts 262144

Each step is built for each `kernels` arm, called once to compile and warm, timed with the profiler stopped - and with `--graph` replayed from a captured CUDA graph, at each `--batch` for a step that batches, as `graph_median_us` by batch - then
called once more between a profiler start and stop and inside an NVTX range named `step@context@kernels`. The range
is how the report tells which step and which arm a kernel belongs to.

Under `ncu` every launch pays for the tool's hook, profiled or not, so a timing taken there reads hundreds of
microseconds too long. `--mode time` runs the timings alone, outside `ncu`, and `--mode profile` runs the profiled calls alone, inside it.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from lsa.evaluation.performance.roofline import workloads

DEFAULT_REPEATS = 20
MODES = ("both", "time", "profile")
# Replays one graph timing averages over, so a launch's gaps are in it and the event's resolution is not
REPLAYS = 200


def time_launch(launch: workloads.Launch, repeats: int) -> tuple[float, float]:
    """Time a launcher with CUDA events.

    Args:
        launch: The step's launcher, already called once.
        repeats: Timed calls.

    Returns:
        The median and the minimum, in milliseconds.
    """
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times), min(times)


def time_graph(launch: workloads.Launch, repeats: int) -> float:
    """Time a launcher replayed from a captured CUDA graph, the way a served decode step runs it.

    Args:
        launch: The step's launcher, already called once.
        repeats: Timings, each the mean of `REPLAYS` back-to-back replays.

    Returns:
        The median timing, in microseconds a replay.
    """
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(REPLAYS // 10):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        stop.record()
        stop.synchronize()
        times.append(1000 * start.elapsed_time(stop) / REPLAYS)
    return statistics.median(times)


def graph_timings(
    step: workloads.Step, context: int, kernels: str, repeats: int, batches: list[int]
) -> dict[str, float]:
    """Time a step's graph replay at each batch it runs in.

    Args:
        step: The step.
        context: Tokens in each request.
        kernels: `cutedsl` or `triton`.
        repeats: Timings per batch.
        batches: Batches asked for - a step that does not run in a batch takes the first request only.

    Returns:
        Microseconds a replay by batch size, a key a batch, none for a step whose launches cannot be captured.
    """
    timings = {}
    for batch in batches if step.build_batch is not None else [1]:
        launch = (
            step.build(context, kernels)
            if step.build_batch is None
            else step.build_batch(context, batch, kernels)
        )
        if launch is None:
            continue
        launch()
        torch.cuda.synchronize()
        try:
            timings[str(batch)] = time_graph(launch, repeats)
        except RuntimeError as error:
            # A step that reads a value back to the host cannot be captured
            # => no graph timing for it
            print(f"{step.name}@{context}@{kernels} x{batch}: not capturable ({error})", flush=True)
            torch.cuda.synchronize()
            continue
        print(f"{step.name}@{context}@{kernels} x{batch}: graph {timings[str(batch)]:.2f} us", flush=True)
    return timings


def profile_launch(launch: workloads.Launch, label: str) -> None:
    """Call a launcher once inside the profiler window and an NVTX range.

    Args:
        launch: The step's launcher.
        label: The range's name, `step@context@kernels`.
    """
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(label)
    launch()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def profile_context(
    steps: list[workloads.Step],
    context: int,
    kernels: list[str],
    repeats: int,
    final: bool,
    batches: list[int] | None = None,
    mode: str = "both",
) -> list[dict[str, Any]]:
    """Time and profile every step that serves a context.

    Args:
        steps: The steps asked for.
        context: Tokens in the request.
        kernels: The arms to profile each step in, `cutedsl` and `triton`.
        repeats: Timed calls per step.
        final: Whether this is the last context, where the steps that do not follow it run.
        batches: Batches to time a graph replay at, or None for no graph timing.
        mode: `time` for the timings alone, `profile` for the profiled calls alone, `both` for each.

    Returns:
        One timing row per step and arm that ran, none in `profile` mode.
    """
    rows = []
    for step in steps:
        if not step.follows_context and not final:
            continue
        for arm in kernels:
            launch = workloads.launcher(step, context, arm)
            name = f"{step.name}@{context}@{arm}"
            if launch is None:
                print(f"{name}: not served", flush=True)
                continue
            launch()
            torch.cuda.synchronize()
            if mode == "profile":
                profile_launch(launch, name)
                del launch
                torch.cuda.empty_cache()
                continue
            median_ms, min_ms = time_launch(launch, repeats)
            if mode == "both":
                profile_launch(launch, name)
            print(f"{name}: median {median_ms:.4f} ms, min {min_ms:.4f} ms", flush=True)
            row: dict[str, Any] = {
                "step": step.name,
                "stage": step.stage,
                "context": context,
                "kernels": arm,
                "follows_context": step.follows_context,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "repeats": repeats,
            }
            if batches is not None:
                row["graph_median_us"] = graph_timings(step, context, arm, repeats, batches)
            rows.append(row)
            del launch
            torch.cuda.empty_cache()
    return rows


def free_context() -> None:
    """Drop the cached state of a context and return its memory to the driver."""
    workloads.bucket_cache.cache_clear()
    workloads.flat_listing.cache_clear()
    torch.cuda.empty_cache()


def main() -> None:
    """Profile the asked steps at each context and write the timings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--contexts", type=int, nargs="+", required=True, help="tokens in the request")
    parser.add_argument("--steps", nargs="+", default=["all"], help="step names, or all")
    parser.add_argument("--timings", type=Path, required=True, help="where the timings JSON goes")
    parser.add_argument(
        "--kernels", nargs="+", choices=workloads.KERNELS, default=list(workloads.KERNELS), help="the arms"
    )
    parser.add_argument("--mode", choices=MODES, default="both", help="what to run - see the module doc")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--graph", action="store_true", help="also time each step replayed from a CUDA graph")
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1], help="requests, for the steps that batch"
    )
    args = parser.parse_args()

    steps = (
        list(workloads.STEPS)
        if args.steps == ["all"]
        else [workloads.step_named(name) for name in args.steps]
    )
    rows = []
    for context in args.contexts:
        final = context == max(args.contexts)
        rows += profile_context(
            steps, context, args.kernels, args.repeats, final, args.batch if args.graph else None, args.mode
        )
        free_context()
    if args.mode == "profile":
        return
    args.timings.parent.mkdir(parents=True, exist_ok=True)
    args.timings.write_text(json.dumps(rows, indent=2))
    print(args.timings, flush=True)


if __name__ == "__main__":
    main()
