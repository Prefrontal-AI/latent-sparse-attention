"""What a run came to - one status, from its launch log, slurm and its record, worked out where it is read.

Nothing stores it. The launch log says how each launch ended, slurm says how a launch that never wrote its end
line did, and the record says whether everything the run set out to measure is there. The viewer reads the
status for every run it lists and `lsa evaluate publish` reads it before copying a run to the shared root,
and both come here so the two never disagree.

The last launch decides, in this order:

- running or queued in slurm: in progress
- ended with the runner raising, or slurm says it failed: failed
- slurm cancelled it or ran it out of time: cancelled, timed out
- ended well, with something the record set out to measure missing: partial
- ended well: complete
- no end line and slurm has no record of the job: unknown

A run with no launch log at all - an imported one, which no launcher ran - is complete once it has written its
`meta.json`, which is written last, and partial when the record has something missing. With no `meta.json` either
it has no status.

A pooled run has a launch per worker, and the last one says nothing of the others. It is in progress while any
worker is running or queued, and complete once its record holds everything, whichever of its workers died on
the way - the survivors finished what the dead ones were holding. With work left, the last launch decides as
above.
"""

from collections.abc import Callable, Mapping
from typing import Any

from lsa.evaluation import launches as launch_log
from lsa.evaluation.constants import Evaluation, Launches, Status


def accuracy_missing(record: dict[str, Any]) -> str:
    """Say which of a scored run's planned tasks are not scored in full, or nothing when every one is.

    A task skipped for want of data is missing too - the run set out to score it and has no number for it.

    Args:
        record: An accuracy record.

    Returns:
        The tasks missing, or empty.
    """
    tasks = record["tasks"]
    planned = {plan["task"] for plan in record["run"]["plans"]}
    missing = sorted(
        task for task in planned if task not in tasks or tasks[task]["scored"] < tasks[task]["n"]
    )
    if not missing:
        return ""
    return f"{len(missing)} of {len(planned)} planned tasks not scored in full: {', '.join(missing)}"


def kernels_missing(record: dict[str, Any]) -> str:
    """Say which contexts of a kernel run did not finish, or nothing when every one did.

    Args:
        record: A kernel record.

    Returns:
        The contexts missing, or empty.
    """
    failed = record["failed"]
    if not failed:
        return ""
    return f"contexts {', '.join(str(context) for context in failed)} did not finish"


def production_missing(record: dict[str, Any]) -> str:
    """Say nothing - a ladder that stops short raises, so its launch already reads as failed."""
    return ""


# What each kind of record says is left to do
MISSING: dict[str, Callable[[dict[str, Any]], str]] = {
    Evaluation.ACCURACY: accuracy_missing,
    Evaluation.KERNELS: kernels_missing,
    Evaluation.PRODUCTION: production_missing,
}


def status(
    launches: list[dict[str, Any]], missing: str, slurm_states: Mapping[str, str], meta_written: bool = False
) -> tuple[str, str]:
    """Return a run's status, one of `Status.ALL`, and what it rests on.

    Args:
        launches: The run's launches, oldest first, as `launches.read` returns them.
        missing: What the record says is left to do, as `MISSING` gives it, or empty.
        slurm_states: Slurm's state by job id, for the job of a last launch with no end line.
        meta_written: Whether the run directory holds a `meta.json`, which a run with no launch log is finished by.

    Returns:
        The status, empty for a run with neither a launch log nor a `meta.json`, and one line saying where it
        came from.
    """
    if not launches and not meta_written:
        return "", ""
    if not launches:
        if missing:
            return Status.PARTIAL, f"no launch log, meta written; {missing}"
        return (
            Status.COMPLETE,
            "no launch log, meta written, and the record holds everything it set out to measure",
        )
    if launch_log.pooled(launches):
        pool_status = pooled_status(launches, missing, slurm_states)
        if pool_status:
            return pool_status
    last = launches[-1]
    job = last.get("job") or "outside slurm"
    if last.get("outcome") == Launches.FAILED:
        return Status.FAILED, f"job {job}: {last.get('error', '')}"
    if last.get("outcome") == Launches.COMPLETED:
        ended, why = Status.COMPLETE, f"job {job} ended {last['ended']}"
    else:
        state = slurm_states.get(last.get("job") or "")
        if not state:
            return Status.UNKNOWN, f"job {job} has no end line and slurm has no record of it"
        # `CANCELLED by 1004` names who cancelled it
        ended, why = Status.SLURM.get(state.split()[0], Status.FAILED), f"job {job}: slurm says {state}"
    if ended == Status.COMPLETE and missing:
        return Status.PARTIAL, f"{why}; {missing}"
    return ended, why


def pooled_status(
    launches: list[dict[str, Any]], missing: str, slurm_states: Mapping[str, str]
) -> tuple[str, str] | None:
    """Say what a pool of workers came to when that is not just its last launch's end, or None when it is.

    Args:
        launches: The run's launches, oldest first.
        missing: What the record says is left to do, or empty.
        slurm_states: Slurm's state by job id, for the launches with no end line.

    Returns:
        In progress while a worker with no end line is queued or running, complete when nothing is missing
        and a worker ended well, and otherwise None.
    """
    alive = [
        job
        for job in launch_log.open_jobs(launches)
        if Status.SLURM.get(slurm_states.get(job, "").split(" ")[0]) == Status.IN_PROGRESS
    ]
    workers = len({launch.get("worker") for launch in launches if launch.get("worker")})
    if alive:
        return (
            Status.IN_PROGRESS,
            f"{len(alive)} of {workers} workers running or queued: jobs {', '.join(alive)}",
        )
    ended_well = any(launch.get("outcome") == Launches.COMPLETED for launch in launches)
    if not missing and ended_well:
        return Status.COMPLETE, f"{workers} workers, and the record holds everything it set out to measure"
    return None
