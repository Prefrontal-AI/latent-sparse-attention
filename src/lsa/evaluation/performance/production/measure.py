"""Drive sglang's serving benchmark against a live server, one client process per length.

The server is the one `launch.serve` starts for a scored run, with the same image, the same plugin and the
same arming check - only what is asked of it differs. The client is sglang's own, already in that image, so
it runs there too: a step of its own on the node that answers http, holding no GPU, pointed at the server.

A length whose samples already exist is kept, and the ladder stops at the first length that fails. That is
what lets a pinned run id continue on a larger deployment - the one-node lengths stay, the two-node ones add
theirs beside them - and what keeps a length the pool cannot hold from being followed by longer ones.
"""

import json
import math
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsa import assets, io, slurm
from lsa.evaluation.constants import Production
from lsa.evaluation.performance import telemetry
from lsa.evaluation.performance.production import record
from lsa.sglang.constants import Server
from lsa.sglang.launch import cache_variables, container_command, mount_list

if TYPE_CHECKING:
    from lsa.evaluation.config import ProductionConfig, ProductionDataConfig
    from lsa.sglang.launch import SglangConfig


def prompt_file(production: "ProductionDataConfig", length: int) -> Path:
    """Return the prompt file one length is drawn from, as the run's workload card lays them out."""
    return Path(assets.workload(production.workload).file_path(length))


def benchmark_args(experiment: "ProductionConfig", url: str, length: int, output: Path) -> list[str]:
    """Build the client's command line for one length.

    `--tokenize-prompt` with `--random-range-ratio 1` is what makes a prompt exactly `length` token ids:
    the client tokenizes an entry, cuts it there and sends the ids, so the server tokenizes nothing.

    Args:
        experiment: The production run - what to send from `data`, anything else for the client from `run.args`.
        url: The server's http root.
        length: The prompt length in tokens.
        output: Where the client appends its result.

    Returns:
        The argv, starting at the interpreter.
    """
    production = experiment.data
    return [
        "python",
        "-m",
        Production.CLIENT,
        "--backend",
        "sglang",
        "--base-url",
        url,
        "--dataset-name",
        "random",
        "--dataset-path",
        str(prompt_file(production, length)),
        "--random-input-len",
        str(length),
        "--random-output-len",
        str(production.output_tokens),
        "--random-range-ratio",
        "1",
        "--tokenize-prompt",
        "--num-prompts",
        str(production.requests),
        "--max-concurrency",
        str(production.concurrency),
        "--warmup-requests",
        str(production.warmup),
        "--flush-cache",
        "--output-file",
        str(output),
        "--output-details",
        *profile_args(production, output, length),
        *experiment.run.args,
    ]


def profile_args(production: "ProductionDataConfig", output: Path, length: int) -> list[str]:
    """Build the client's flags that have the server profile a length's decode steps, or none when not asked.

    The client posts `/start_profile` once the warmup is done, and the server writes one trace a rank under a
    directory of its own inside `directory`. Only decode is profiled, since prefill is not what is being asked.

    Args:
        production: The run's data block, whose `profile_steps` says how many decode steps to record.
        output: Where the client appends its result, `<run>/samples/<deployment>/<length>.jsonl` - the traces land
            under the run directory three levels up.
        length: The prompt length in tokens.

    Returns:
        The flags, empty when `profile_steps` is 0.
    """
    if not production.profile_steps:
        return []
    return [
        "--profile",
        "--profile-by-stage",
        "--profile-stages",
        "decode",
        "--profile-num-steps",
        str(production.profile_steps),
        "--profile-output-dir",
        str(output.parents[2] / Production.PROFILES / str(length)),
    ]


def prefill_args(experiment: "ProductionConfig", url: str, length: int, output: Path) -> list[str]:
    """Build the client's command line for the one request whose prefill is profiled.

    It is the timed request's line with nothing to warm, one request and two output tokens, and no profile flags -
    `watch_prefill` starts the profiler from outside, once the request is far enough through its prompt.
    """
    production = experiment.data
    shrunk = replace(
        production,
        requests=1,
        concurrency=1,
        warmup=0,
        output_tokens=2,
        profile_steps=0,
        profile_prefill_chunks=0,
    )
    return benchmark_args(replace(experiment, data=shrunk), url, length, output)


def watch_prefill(
    url: str,
    log: Path,
    chunk: int,
    chunks: int,
    production: "ProductionDataConfig",
    directory: Path,
    stop: threading.Event,
    start: int,
) -> None:
    """Start the server's profiler once its log shows a share of the request's prefill chunks done, in a thread.

    The count is rank 0's `Prefill batch` lines of the chunk's size appended after `start`, which is how the
    server reports every chunk. The profiler is the server's own, asked for `profile_prefill_chunks` forward steps
    from the next batch on - not by stage, because by stage it would start with the request's first chunk.

    Args:
        url: The server's http root.
        log: The server's log, which grows while the request runs.
        chunk: The server's chunk size in tokens.
        chunks: How many chunks the request's prompt makes.
        production: The run's data block.
        directory: Where the traces are written.
        stop: Set when the client is done, which ends the watch.
        start: The log's size before the request was sent, so lines the server writes before the thread runs count.
    """
    target = max(1, int(production.profile_prefill_at * chunks))
    pattern = re.compile(rf"TP0 EP0\] Prefill batch.*#new-token: {chunk}\b")
    done = 0
    with log.open("r", errors="replace") as handle:
        handle.seek(start)
        while not stop.is_set():
            line = handle.readline()
            if not line:
                time.sleep(0.2)
                continue
            done += bool(pattern.search(line))
            if done >= target:
                break
        else:
            return
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "activities": ["CPU", "GPU"],
        "num_steps": production.profile_prefill_chunks,
        "profile_by_stage": False,
        "with_stack": False,
        "record_shapes": False,
        "output_dir": str(directory),
    }
    request = urllib.request.Request(
        f"{url}/start_profile", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=Server.POLL_TIMEOUT_SECONDS) as answer:
        print(f"  profiler started after {done}/{chunks} chunks: {answer.status}", flush=True)


def profile_prefill(
    experiment: "ProductionConfig", url: str, info: dict[str, Any], length: int, run_dir: Path
) -> bool:
    """Send one request whose prefill, late in the prompt, the server's profiler records.

    Returns:
        Whether the client finished.
    """
    production = experiment.data
    chunk = int(info.get("chunked_prefill_size") or 32768)
    chunks = math.ceil(length / chunk)
    directory = run_dir / Production.PROFILES / f"{length}-prefill"
    scratch = run_dir / "profile-prefill" / f"{length}.jsonl"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    log = run_dir / "server.log"
    watcher = threading.Thread(
        target=watch_prefill,
        args=(url, log, chunk, chunks, production, directory, stop, log.stat().st_size),
        daemon=True,
    )
    watcher.start()
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in cache_variables().items())
    client = " ".join(shlex.quote(part) for part in prefill_args(experiment, url, length, scratch))
    status = subprocess.run(client_command(experiment.server(), f"exec env {exports} {client}")).returncode
    stop.set()
    watcher.join(timeout=120)
    return not status


def client_command(backend: "SglangConfig", inner: str) -> list[str]:
    """Wrap a shell line so it runs in the server's image, on this node, beside the server.

    Not `served.container_command`, whose step asks for the server's GPUs on every node the server spans.
    The client needs one task on the node whose loopback the server answers on, and no device at all.

    Args:
        backend: The launch block, which names the image and the runner.
        inner: The shell line to run inside.

    Returns:
        The argv for `subprocess.run`.
    """
    if backend.runner != "srun":
        return container_command(backend, inner)
    return [
        "srun",
        # The launcher holds the allocation as a step already, so this one overlaps it or never starts
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={socket.gethostname()}",
        "--mem=0",
        "--cpu-bind=none",
        f"--container-image={backend.image}",
        f"--container-mounts={','.join(mount_list())}",
        "bash",
        "-lc",
        inner,
    ]


def measure_command(experiment: "ProductionConfig", url: str, length: int, output: Path) -> list[str]:
    """Wrap the client so it runs inside the pinned image.

    Args:
        experiment: The production run, whose launch names the image and the runner.
        url: The server's http root.
        length: The prompt length in tokens.
        output: Where the client appends its result.

    Returns:
        The argv for `subprocess.run`.
    """
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in cache_variables().items())
    client = " ".join(shlex.quote(part) for part in benchmark_args(experiment, url, length, output))
    return client_command(experiment.server(), f"exec env {exports} {client}")


def server_info(url: str) -> dict[str, Any]:
    """Return what the server says it is - its mesh, its context and every setting it started with."""
    with urllib.request.urlopen(f"{url}/server_info", timeout=Server.POLL_TIMEOUT_SECONDS) as answer:
        info: dict[str, Any] = json.loads(answer.read())
    return info


def job_hosts() -> list[str]:
    """Return the nodes of this job - every node its servers span, which the launcher's allocation holds.

    Returns:
        The expanded allocation, or this node alone outside slurm.
    """
    return slurm.hostnames(os.environ.get("SLURM_JOB_NODELIST", "")) or [socket.gethostname()]


def measure(
    experiment: "ProductionConfig",
    url: str,
    info: dict[str, Any],
    length: int,
    run_dir: Path,
    log: Callable[[str], None],
) -> bool:
    """Measure one length, unless it is already measured or longer than the server can hold.

    A result that is not finished is set aside rather than appended to, because the record reads the last
    line of the file and a failed attempt's partial line would otherwise be what it reads.

    Args:
        experiment: The production run.
        url: The server's http root.
        info: The server's `/server_info`.
        length: The prompt length in tokens.
        run_dir: The run directory the samples land under.
        log: Where progress goes.

    Returns:
        Whether the ladder may climb past this length.
    """
    production = experiment.data
    output = record.sample_path(run_dir, record.deployment(info), length)
    if record.finished(output, production.requests):
        log(f"  {length}: samples exist - kept")
        return True
    if length + production.output_tokens > info["context_length"]:
        log(f"  {length}: over the server's context {info['context_length']} - skipped")
        return True
    if output.exists():
        output.replace(output.with_suffix(".failed"))
    output.parent.mkdir(parents=True, exist_ok=True)
    log(f"  {length}: {io.utc_now()}")
    # From the first request to the last response, on every node the server spans
    with telemetry.sample_nodes(
        output.with_suffix(""), job_hosts(), socket.gethostname(), experiment.server().tp
    ):
        status = subprocess.run(measure_command(experiment, url, length, output)).returncode
    if status or not record.finished(output, production.requests):
        log(f"  {length}: client exit {status} - not finished")
        return False
    if production.profile_prefill_chunks:
        log(f"  {length}: prefill profile {io.utc_now()}")
        if not profile_prefill(experiment, url, info, length, run_dir):
            log(f"  {length}: the prefill profile's client did not finish")
    return True


def run_ladder(experiment: "ProductionConfig", url: str, label: str, run_dir: Path) -> list[int]:
    """Measure every length of one ladder against a live server, rewriting the record after each.

    The record names the model by the hub id `model.name` resolves to.

    Args:
        experiment: The production run.
        url: The server's http root.
        label: The configuration label the record is grouped by.
        run_dir: The run directory.

    Returns:
        The lengths the ladder did not reach, from the one that failed on - empty when it finished.

    Raises:
        ValueError: The run directory already holds a record of another configuration.
        RuntimeError: No length produced samples, so there is no record to write.
    """
    production = experiment.data
    existing = record.load(run_dir)
    existing_name = existing.get("run", {}).get("name")
    if existing_name and existing_name != label:
        raise ValueError(f"{run_dir} is {existing_name} and this server runs {label}")
    info = server_info(url)
    started = existing.get("run", {}).get("started") or io.utc_now()
    # A continued run carries the time it already spent measuring - the server start is in neither
    spent = existing.get("run", {}).get("seconds") or 0.0
    clock = time.perf_counter()
    print(
        f"{run_dir.name}  {label}  {record.deployment(info)}  {url}  lengths {production.lengths}", flush=True
    )

    done = 0
    for length in production.lengths:
        if not measure(experiment, url, info, length, run_dir, print):
            break
        done += 1
        meta = {
            "note": experiment.run.note,
            "tag": experiment.run.tag,
            "started": started,
            "date": io.utc_now(),
            "seconds": round(spent + time.perf_counter() - clock, 1),
            "image": experiment.run.image,
            "gpu": experiment.run.gpu,
            "model": experiment.model.repo,
            "name": label,
            "warmup": production.warmup,
        }
        record.write(run_dir, meta)
    if not any(run_dir.glob("samples/*/*.jsonl")):
        raise RuntimeError(f"{run_dir}: no length finished - nothing was measured")
    print(f"done  {done}/{len(production.lengths)} lengths  {run_dir}", flush=True)
    return production.lengths[done:]
