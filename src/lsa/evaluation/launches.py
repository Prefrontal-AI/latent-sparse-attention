"""Each launch of a run - who started it, in which slurm job, and how it ended.

    <run dir>/launches.jsonl
    {"job": "183015", "user": "alice", "host": "node-3", "started": "2026-09-28T06:20:30+00:00"}
    {"job": "183015", "ended": "2026-09-28T09:01:12+00:00", "outcome": "completed"}

A record is rewritten as a run goes, so a run killed between two of its writes reads the same as one still
writing, and nothing in it says who started it. The launch log is the other half: [the launcher](run.py)
appends a start line before the runner touches anything and an end line when the runner returns or raises.
A launch slurm stopped - cancelled, timed out, out of memory, its node lost - never reaches its end line, and
slurm's accounting says what happened to it instead.

A run is resumed by launching it again into the same directory, so a log holds every launch, and the last one
is what [status](status.py) reads: a run that failed once and was finished by a second launch is complete.

A pooled run has several launches at once, one per worker, and each start line names its worker. An end line
closes the launch of its own job, so workers that end in any order close their own lines.

The log is append-only and one line per write, so a reader never sees a half-written launch and a rerun never
loses the history of the launches before it.
"""

import contextlib
import datetime
import getpass
import json
import os
import shutil
import socket
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.constants import Launches


def locate_log(run_dir: str | os.PathLike[str]) -> Path:
    """Return a run directory's launch log."""
    return Path(run_dir) / Launches.FILE


def start_line(worker: str | None = None) -> dict[str, Any]:
    """Return this process's start line - its slurm job, the user slurm runs it as, and where.

    `SLURM_JOB_USER` rather than the login name, because the job is the user's however the process came to
    run - and outside slurm the login name is all there is.

    Args:
        worker: The worker's name when the launch is one of a pool, which the line then carries.
    """
    line = {
        "job": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("SLURM_JOB_USER") or getpass.getuser(),
        "host": socket.gethostname(),
        "started": io.utc_now(),
    }
    if worker:
        line["worker"] = worker
    return line


def end_line(start: Mapping[str, Any], outcome: str, error: str = "") -> dict[str, Any]:
    """Return the line that closes a launch.

    Args:
        start: The launch's start line.
        outcome: `Launches.COMPLETED` or `Launches.FAILED`.
        error: What the runner raised, for a failed launch.

    Returns:
        The end line, naming the job it closes.
    """
    line = {"job": start.get("job"), "ended": io.utc_now(), "outcome": outcome}
    if error:
        line["error"] = error
    return line


def append(run_dir: str | os.PathLike[str], line: Mapping[str, Any]) -> None:
    """Append one line to a run directory's launch log, in one write."""
    with locate_log(run_dir).open("a") as log:
        log.write(json.dumps(line) + "\n")


def tag(run_dir: str | os.PathLike[str]) -> None:
    """Name the run directory as this process's slurm job's comment, so the queue leads from the job to its run.

    A job's name cannot say where its run lands - a single job's is the same for every run, and the records
    root is whatever the config named. Best effort: a launch outside slurm, or one slurm will not update, runs
    all the same, and only goes unlisted in flight.

    Args:
        run_dir: The run directory the launch writes into.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if not job or shutil.which("scontrol") is None:
        return
    command = ["scontrol", "update", f"JobId={job}", f"Comment={Path(run_dir).resolve()}"]
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(command, capture_output=True, check=False, timeout=30)


@contextlib.contextmanager
def launched(run_dir: str | os.PathLike[str], worker: str | None = None) -> Iterator[None]:
    """Log one launch into a run directory around the code that runs it.

    Only an `Exception` is a failure written here. A process stopped by a signal or by `SystemExit` was
    stopped from outside, and slurm, which stopped it, is what says how.

    Args:
        run_dir: The run directory, which exists.
        worker: The worker's name when the launch is one of a pool.
    """
    start = start_line(worker)
    append(run_dir, start)
    tag(run_dir)
    try:
        yield
    except Exception as failure:
        append(run_dir, end_line(start, Launches.FAILED, f"{type(failure).__name__}: {failure}"))
        raise
    append(run_dir, end_line(start, Launches.COMPLETED))


def fold(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold a log's lines into launches, oldest first, each start line merged with the end line closing it.

    An end line closes the open launch of its own job, and the latest one still open when no job matches - a
    run directory launched before pools was launched into one job at a time.

    Args:
        lines: The log's lines, in the order they were written.

    Returns:
        One dict per launch - `job`, `user`, `host`, `started`, and `ended`, `outcome` and `error` once it ended.
    """
    launches: list[dict[str, Any]] = []
    for line in lines:
        if "outcome" in line:
            open_ = [launch for launch in launches if "outcome" not in launch]
            of_job = [launch for launch in open_ if launch.get("job") == line.get("job")]
            open_ = of_job or open_
            if open_:
                open_[-1].update({key: value for key, value in line.items() if key != "job"})
        else:
            launches.append(dict(line))
    return launches


def read(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return a run directory's launches, oldest first - none for a run launched before there was a log."""
    path = locate_log(run_dir)
    if not path.is_file():
        return []
    with path.open() as log:
        return fold([json.loads(line) for line in log if line.strip()])


def owner(launches: list[dict[str, Any]]) -> str:
    """Return who started a run - its first launch's user, or empty when nothing says."""
    if not launches:
        return ""
    return launches[0].get("user") or ""


def unsettled(launches: list[dict[str, Any]]) -> str | None:
    """Return the slurm job of a run's last launch when only slurm can say how it ended, or None."""
    if not launches or "outcome" in launches[-1]:
        return None
    return launches[-1].get("job")


def open_jobs(launches: list[dict[str, Any]]) -> list[str]:
    """Return the slurm jobs of every launch only slurm can say how it ended - several for a pooled run."""
    return [launch["job"] for launch in launches if "outcome" not in launch and launch.get("job")]


def pooled(launches: list[dict[str, Any]]) -> bool:
    """Say whether a run was launched as a pool of workers, any one of its launches naming one."""
    return any(launch.get("worker") for launch in launches)


def accounting(jobs: Sequence[str], fields: Sequence[str] = ("State",)) -> dict[str, dict[str, str]]:
    """Ask slurm's accounting about jobs, one sacct call for all of them.

    A job is found under the id `SLURM_JOB_ID` gives it and, for an array task, under its `<array>_<task>`
    id too. A job sacct does not know is left out, and so is every job when there is no sacct to ask - a
    machine outside the cluster reads as one that knows nothing. Standard library only, because the viewer
    asks it and the viewer imports nothing else.

    Args:
        jobs: Slurm job ids.
        fields: sacct's own field names - `State`, `User`, `Start`, `End`, `NodeList`.

    Returns:
        Field to value, per job id.
    """
    wanted = sorted({job for job in jobs if job})
    if not wanted or shutil.which("sacct") is None:
        return {}
    columns = ["JobIDRaw", "JobID", *fields]
    command = ["sacct", "-X", "-n", "-P", "-o", ",".join(columns), "-j", ",".join(wanted)]
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        values = line.split("|")
        if len(values) != len(columns):
            continue
        row = dict(zip(fields, values[2:], strict=True))
        found[values[0]] = found[values[1]] = row
    return found


# What `queue` reads of each job, by squeue's own letters - the comment last, the one field free to hold a `|`
QUEUE_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "%A"),
    ("job", "%i"),
    ("name", "%j"),
    ("user", "%u"),
    ("state", "%T"),
    ("reason", "%r"),
    ("nodes", "%D"),
    ("per_node", "%b"),
    ("hosts", "%N"),
    ("start", "%S"),
    ("workdir", "%Z"),
    ("comment", "%k"),
)

# How squeue is asked to write a time, which it otherwise writes as the reader's environment says
SLURM_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def queue() -> list[dict[str, str]]:
    """Return every job slurm holds - pending, running or ending - whoever's it is, one squeue call for all.

    A pending array's tasks that have not started share one row, `255668_[1-2%3]`. A start is ISO 8601, and
    empty for a job still pending. Standard library only, for the viewer, as `accounting` is.

    Returns:
        Field to value per job, the fields as `QUEUE_FIELDS` names them - none on a machine with no squeue.

    Raises:
        RuntimeError: squeue is there and failed, which must not read as nothing running.
    """
    if shutil.which("squeue") is None:
        return []
    command = ["squeue", "-h", "-a", "-o", "|".join(code for _, code in QUEUE_FIELDS)]
    try:
        out = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env={**os.environ, "SLURM_TIME_FORMAT": SLURM_TIME_FORMAT},
        ).stdout
    except (OSError, subprocess.SubprocessError) as failure:
        raise RuntimeError(f"squeue failed: {failure}") from failure
    jobs = []
    for line in out.splitlines():
        values = line.split("|", len(QUEUE_FIELDS) - 1)
        if len(values) != len(QUEUE_FIELDS):
            continue
        job = dict(zip((name for name, _ in QUEUE_FIELDS), values, strict=True))
        job["start"] = iso_time(job["start"]) if job["state"] != "PENDING" else ""
        job["comment"] = "" if job["comment"] == "(null)" else job["comment"]
        jobs.append(job)
    return jobs


def iso_time(text: str) -> str:
    """Rewrite a time squeue wrote as `SLURM_TIME_FORMAT` says as ISO 8601, and anything else as empty."""
    try:
        return datetime.datetime.strptime(text, SLURM_TIME_FORMAT).isoformat()
    except ValueError:
        return ""
