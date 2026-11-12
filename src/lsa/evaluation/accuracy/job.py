"""The accuracy step of a served experiment - score a live server with the suite, as a subprocess.

`lsa.sglang.launch` starts the server and waits until LSA is enabled; this runs `lsa.evaluation.accuracy`
against it, into the run directory the launcher names. The suite reshapes nothing, and a scoring run is long,
so its teardown is the subprocess's problem rather than the launcher's.
"""

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lsa.evaluation import gpu

if TYPE_CHECKING:
    from lsa.evaluation.config import AccuracyConfig


def suite_command(
    experiment: "AccuracyConfig",
    records: Path,
    run_id: str,
    force: bool = False,
    generation: Path | None = None,
    worker: str | None = None,
) -> list[str]:
    """Build the command that scores a live server.

    The suite runs as a module of this package rather than a script in another tree. It is still a
    subprocess, because a scoring run is long and its teardown should not be this process's problem.

    The run id is dictated, and the suite reads a run id it already has a directory for as "continue this
    one" - so a rerun of a finished experiment would score nothing at all. `force` is `--no-resume`, which
    is the suite's own name for scoring every sample again into that same directory.

    Args:
        experiment: The scored run - the suite it asks from `data`, how it is driven and labelled from `run`,
            the port the server answers on, and the hub id the record names the model by.
        records: Where the run directory is written, which is the root a records reader is pointed at.
        run_id: The run id to dictate, which the launcher derives from `run.name` or takes from `run.id`. The
            name goes with it.
        force: Score every sample again rather than keeping the ones the run directory already holds.
        generation: The model's generation block as a file, which is every sampling setting the suite sends.
        worker: The worker's name when this job is one of a pool, which the suite then runs as, or None for a
            job that scores the run alone.

    Returns:
        The argv for `subprocess.run`.
    """
    data, run = experiment.data, experiment.run
    argv = [
        run.python or sys.executable,
        # Its stdout is the job's log file
        # => unbuffered, so each task's score lands as it is graded
        "-u",
        "-m",
        "lsa.evaluation.accuracy",
        "--host",
        "127.0.0.1",
        "--port",
        str(run.port),
        "--suite",
        data.suite,
        "--out",
        str(records),
        "--concurrency-scale",
        str(run.concurrency_scale),
        "--request-timeout",
        str(run.request_timeout),
    ]
    if data.tasks:
        argv += ["--tasks", *data.tasks]
    if data.lengths:
        argv += ["--lengths", *map(str, data.lengths)]
    if run.note:
        argv += ["--note", run.note]
    if run.tag:
        argv += ["--tag", run.tag]
    argv += ["--name", run.launch_name(), "--run-id", run_id]
    if run.gpu:
        argv += ["--gpu", gpu.resolve(run.gpu)]
    if experiment.model.repo:
        argv += ["--model", experiment.model.repo]
    if experiment.model.recorded_attention:
        argv += ["--attention-kind", experiment.model.recorded_attention]
    if run.image:
        argv += ["--image", run.image]
    if data.root:
        argv += ["--data-root", data.root]
    if data.longbench_context:
        argv += ["--max-context", str(data.longbench_context)]
    if data.pins:
        argv += ["--pins", data.pins]
    if generation:
        argv += ["--generation", str(generation)]
    if force:
        argv.append("--no-resume")
    if worker:
        argv += ["--pool", "--worker", worker]
    return argv + list(run.args)


def score(
    experiment: "AccuracyConfig",
    records: Path,
    run_id: str,
    force: bool = False,
    generation: Path | None = None,
    worker: str | None = None,
) -> None:
    """Run the harness suite against a live server, into the run directory the launcher names.

    Args:
        experiment: The scored run.
        records: Where the run directory is written, created if it is not there.
        run_id: The run directory to write.
        force: Score every sample again rather than resuming the run directory.
        generation: The model's generation block as a file.
        worker: The worker's name when the job is one of a pool.

    Raises:
        RuntimeError: The runner failed.
    """
    records.mkdir(parents=True, exist_ok=True)
    argv = suite_command(experiment, records, run_id, force, generation, worker)
    print(f"scoring: {' '.join(argv)}", flush=True)
    finished = subprocess.run(argv)
    if finished.returncode:
        raise RuntimeError(f"the suite runner exited with {finished.returncode}")
