"""The executor every submitter builds, and the flags every submit command overrides it with.

Every command that computes submits rather than runs, so this is how all of them reach the cluster.
Every submitter sends a function rather than a script through submitit, so a job runs this interpreter and
this checkout with nothing to activate and no second copy of the environment to keep in step.
What runs, how many jobs, and what a job asks for belong to the submitter, spelled in submitit's own
parameter names - `gpus_per_node`, `timeout_min`, `slurm_partition` - and passed through unchanged.
What is the same everywhere lives here: the shared log root, the environment contract as `slurm_setup`, the
flags a command line overrides a job with, and the [site's](site.py) Slurm values.

A job asks for, in order of precedence, the command line's flags, then its submitter's parameters, then the site's
`slurm` values - `partition`, `account`, `qos`, `constraint` and `exclude`. A value the site leaves out is not passed
at all, so no `-p`, no reservation and no `-x`. The site's `launcher` is read here and nowhere else: `slurm`
submits through sbatch, `local` runs through submitit's local executor on one node.

    slurm.add_flags(parser)
    args = parser.parse_args()
    slurm.submit(args, "lsa-x", env.OFFLINE_HF, PARAMETERS, work, config)

`submit` is the one-job case. A submitter that maps over an array builds the executor itself, the same way:

    parameters = {**PARAMETERS, **slurm.overrides(args)}
    executor = slurm.executor("lsa-x", env.OFFLINE_HF, args.folder, **parameters)
    jobs = executor.map_array(work, configs)

Every flag defaults to None, so a submitter whose parameters come from its config can load the config first
and still tell a flag given from one left out.

`slurm_additional_parameters` is one dict, and submitit replaces it whole on every update, so `overrides`
is the only place that builds it and a submitter with sbatch options of its own - a requeue, a dependency -
merges them into the dict `overrides` returned rather than passing a second one:

    parameters["slurm_additional_parameters"] = {"requeue": True, **parameters.get("slurm_additional_parameters", {})}
"""

import argparse
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

from lsa import paths, site

# What a job that needs no GPU asks for when its submitter says nothing more
CPU: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 120}


def site_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Lay the site's Slurm values under the parameters a job asks for.

    A parameter given wins over the site's. The nodes to `exclude` are the exception: they are a denylist, so the
    site's and the job's are both kept.

    Args:
        parameters: submitit's own parameters, as the submitter and the command line gave them.

    Returns:
        The same parameters, with the site's `partition`, `account`, `qos`, `constraint` and `exclude` where the job
        did not name its own. Unchanged when the site sets none of them.
    """
    values = site.slurm_values()
    resolved: dict[str, Any] = {
        f"slurm_{key}": values[key] for key in ("partition", "account", "qos", "constraint") if key in values
    }
    resolved.update(parameters)
    if "exclude" in values:
        additional = dict(parameters.get("slurm_additional_parameters") or {})
        excluded = [*values["exclude"].split(","), *str(additional.get("exclude", "")).split(",")]
        additional["exclude"] = ",".join(dict.fromkeys(node for node in excluded if node))
        resolved["slurm_additional_parameters"] = additional
    return resolved


def executor(
    name: str, environment: Sequence[str], folder: str | Path | None = None, **parameters: Any
) -> submitit.AutoExecutor:
    """Return an executor whose jobs carry `name`, start from `environment` and ask for `parameters`.

    This is the one place a job is given to an executor, and so the one place the site's `launcher` is read.
    `slurm_python` is a constructor argument of submitit's rather than an update, so it goes there, and a
    parameter left at None is unset, as submitit reads it.
    The job's shell exports the site this process reads, so a site picked with `--site` is the job's too.
    It also puts the bin directory of the interpreter the job runs under on `PATH`, because a job is not
    started from an activated venv and a build tool the venv installs, such as ninja, is found there or not at all.

    Args:
        name: Job name, as `squeue` shows it.
        environment: Shell lines the job runs before the task starts - a contract from `lsa.env`.
        folder: Where submitit writes each job's stdout, stderr and pickles, or None for the shared log
            root, so a job's output does not depend on the directory it was submitted from.
        **parameters: submitit's own parameters, as `AutoExecutor.update_parameters` takes them, plus
            `slurm_python` for the interpreter the job runs under.

    Raises:
        Unsupported: The site's launcher is `local`, and the job spans nodes or names a reservation.
    """
    launcher = site.launcher()
    asked = site_parameters(parameters)
    if launcher == "local":
        require_local(asked)
    python = asked.pop("slurm_python", None)
    scripts = shlex.quote(str(Path(python or sys.executable).parent))
    auto_executor = submitit.AutoExecutor(
        folder=folder or paths.logs(), cluster=launcher, **{f"{launcher}_python": python}
    )
    auto_executor.update_parameters(
        name=name,
        **{
            f"{launcher}_setup": [
                *environment,
                f'export PATH={scripts}:"$PATH"',
                f"export {site.SITE_VARIABLE}={site.current()}",
            ]
        },
        **asked,
    )
    return auto_executor


def require_local(parameters: Mapping[str, Any]) -> None:
    """Refuse a job the local executor cannot run.

    Args:
        parameters: What the job asks for, as submitit parameters.

    Raises:
        Unsupported: The job spans more than one node, or names a reservation.
    """
    nodes = parameters.get("nodes") or 1
    if nodes > 1:
        raise site.Unsupported(
            f"{nodes} nodes need slurm, and the launcher of site {site.current()!r} is local"
        )
    if "reservation" in (parameters.get("slurm_additional_parameters") or {}):
        raise site.Unsupported(
            f"a reservation needs slurm, and the launcher of site {site.current()!r} is local"
        )


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Add `--gpus`, `--cpus`, `--mem-gb`, `--hours`, `--partition`, `--reservation`, `--nodelist`, `--folder` and `--dry-run`.

    Args:
        parser: The submit command's parser.
    """
    parser.add_argument("--gpus", type=int, help="GPUs per node")
    parser.add_argument("--cpus", type=int, help="CPUs per task")
    parser.add_argument("--mem-gb", type=int, help="host memory per node, in GB")
    parser.add_argument("--hours", type=float, help="wall-clock limit")
    parser.add_argument("--partition", help="slurm partition (default the site's)")
    parser.add_argument("--reservation", help="slurm reservation to run inside, if one is free")
    parser.add_argument(
        "--nodelist", help="slurm nodes the job may run on, to keep one node free for another"
    )
    parser.add_argument("--folder", help="where submitit writes job output (default under $LSA_ROOT/logs)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be submitted, submit nothing"
    )


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return the flags a command line gave, as the submitit parameters they set.

    A flag left out is absent, so `executor.update_parameters(**overrides(args))`, or the same dict laid over
    a submitter's own, changes only what was given. `--folder` and `--dry-run` are not parameters and stay on
    `args`.

    Args:
        args: The parsed command line, from a parser `add_flags` added to.

    Returns:
        Parameter name to value - `gpus_per_node`, `cpus_per_task`, `mem_gb`, `timeout_min`,
        `slurm_partition`, and `slurm_additional_parameters` holding the reservation and the nodes.
    """
    additional = {
        name: value
        for name, value in (("reservation", args.reservation), ("nodelist", getattr(args, "nodelist", None)))
        if value is not None
    }
    given = {
        "gpus_per_node": args.gpus,
        "cpus_per_task": args.cpus,
        "mem_gb": args.mem_gb,
        "timeout_min": None if args.hours is None else round(args.hours * 60),
        "slurm_partition": args.partition,
        "slurm_additional_parameters": additional or None,
    }
    return {parameter: value for parameter, value in given.items() if value is not None}


def submit(
    args: argparse.Namespace,
    name: str,
    environment: Sequence[str],
    parameters: Mapping[str, Any],
    function: Callable[..., Any],
    *arguments: Any,
) -> Any:
    """Submit one job running `function(*arguments)`, the command line's flags laid over `parameters`.

    `function` is pickled, so it is a module-level function and its arguments are resolved before they go -
    a compute node does not start where the submission did.

    Args:
        args: The parsed command line, from a parser `add_flags` added to.
        name: Job name, as `squeue` shows it.
        environment: Shell lines slurm runs before the task starts - a contract from `lsa.env`.
        parameters: What the job asks for when the command line says nothing, as submitit parameters.
        function: What the job runs.
        *arguments: What it is called with.

    Returns:
        The submitit job, or None under `--dry-run`, which prints what would be asked for and submits nothing.
    """
    asked = {**parameters, **overrides(args)}
    if args.dry_run:
        print(f"{name}: would submit {function.__module__}.{function.__qualname__} with {asked}", flush=True)
        return None
    job = executor(name, environment, args.folder, **asked).submit(function, *arguments)
    print(f"submitted {job.job_id}  {name}", flush=True)
    return job


# Node states a worker cannot be started on, whatever the queue says of them
UNAVAILABLE_STATES: tuple[str, ...] = ("drain", "down", "fail", "maint", "not_responding", "inval")


@dataclass(frozen=True)
class QueuedJob:
    """One row of `squeue` - a job that is pending or running."""

    job: str
    name: str
    state: str
    # Hostlist expression, empty while pending
    nodes: str
    reservation: str
    user: str


def require_slurm(what: str) -> None:
    """Refuse a question only a slurm cluster can answer.

    Args:
        what: What was asked, as the error says it.

    Raises:
        Unsupported: The site's launcher is not slurm, so there is no queue or reservation to ask.
    """
    if site.launcher() != "slurm":
        raise site.Unsupported(f"{what} needs slurm, and the launcher of site {site.current()!r} is local")


def ask(command: Sequence[str]) -> str:
    """Run a slurm query and return what it printed.

    Args:
        command: The argv - `squeue`, `sinfo` or `scontrol`.

    Returns:
        Standard output.

    Raises:
        RuntimeError: The command is not there or failed, which a caller deciding what to submit must not read
            as an empty queue.
    """
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as failure:
        raise RuntimeError(f"{' '.join(command)} failed: {failure}") from failure
    return done.stdout


def queue(user: str | None = None) -> list[QueuedJob]:
    """Return the pending and running jobs, of one user or of everyone.

    Args:
        user: Whose jobs, or None for all of them.

    Raises:
        Unsupported: The site's launcher is not slurm.
    """
    require_slurm("the queue")
    command = ["squeue", "-h", "-t", "PENDING,RUNNING", "-o", "%i|%j|%T|%N|%v|%u"]
    if user:
        command += ["-u", user]
    jobs = []
    for line in ask(command).splitlines():
        fields = line.strip().split("|")
        if len(fields) == 6:
            jobs.append(QueuedJob(*fields))
    return jobs


def hostnames(expression: str) -> list[str]:
    """Expand a slurm hostlist such as `gpu[01-03]` into its node names."""
    if not expression or expression in ("(null)", "None"):
        return []
    return ask(["scontrol", "show", "hostnames", expression]).split()


def reservation_nodes(reservation: str) -> list[str]:
    """Return the nodes a reservation holds.

    Raises:
        Unsupported: The site's launcher is not slurm.
        RuntimeError: Slurm does not know the reservation.
    """
    require_slurm("a reservation")
    out = ask(["scontrol", "show", "reservation", reservation, "-o"])
    found = re.search(r"\bNodes=(\S+)", out)
    if not found:
        raise RuntimeError(f"slurm reports no nodes for reservation {reservation}")
    return hostnames(found.group(1))


def free_nodes(reservation: str) -> list[str]:
    """Return the nodes of a reservation nothing is running on and that can take a job.

    A node an idle reservation holds shows as reserved rather than idle in `sinfo`, so a node is free when it is
    up and no running job is on it, not when `sinfo` says idle.

    Args:
        reservation: The reservation's name.

    Returns:
        The free nodes, in name order.
    """
    nodes = reservation_nodes(reservation)
    states: dict[str, str] = {}
    for line in ask(["sinfo", "-h", "-N", "-o", "%N|%T", "-n", ",".join(nodes)]).splitlines():
        node, _, state = line.strip().partition("|")
        states[node] = state.lower()
    busy = {node for job in queue() if job.state == "RUNNING" for node in hostnames(job.nodes)}
    return sorted(
        node
        for node in nodes
        if node not in busy and not any(bad in states.get(node, "") for bad in UNAVAILABLE_STATES)
    )
