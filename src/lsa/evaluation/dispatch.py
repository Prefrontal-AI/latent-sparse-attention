"""Keep a slurm reservation full by adding workers to the pooled runs with the most left to do.

    lsa evaluate dispatch results/accuracy --dry-run
    lsa evaluate dispatch results/accuracy/<run id> [<run dir> ...]
    lsa evaluate dispatch results --publish-every 3600 --publish-root ${LSA_ROOT}/evaluations

A run launched with `lsa evaluate accuracy --workers N` is a pool: any number of jobs, each with a server of its
own, drain its requests together. What is left is on the shared filesystem, so a node that frees in the
reservation can be given to whichever run needs it most without anyone re-cutting a run by its tasks. Each look
at the reservation does the same few things.
The reservation is `--reservation`, else the site's `slurm.reservation`, and a site with neither refuses.

It counts the nodes of the reservation that are up and have no running job, less the jobs already queued into
the reservation. It measures every run it was given - the requests left, how many are claimed and so already
someone's, and how long a request takes from the rows the workers wrote. And it gives a free node to the run
that would finish last at its present workers, one worker at a time, until the nodes or the runs are used up.
A run is never given more workers than it can use, which is its unclaimed requests over what a worker holds at
once, nor more than `--max-workers`.

A run whose last launches all failed is left alone, since another worker on a broken server only fails again.
A run needs the `pool/submission.json` that `lsa evaluate accuracy --workers` or `--pool` wrote, which says
how to submit another worker to it, and one that has none is named and passed over.

With `--publish-every` it also publishes the runs as they stand and joins the configurations named with `--join`,
through `lsa evaluate publish --sync` and `lsa records join` themselves.

It is light polling of the shared filesystem and slurm, so it runs in the foreground on the login node, in
tmux, and `--dry-run` prints every decision and submits nothing. It never cancels anything.
"""

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import io, paths, site, slurm
from lsa.evaluation import launches, publish, require_extra
from lsa.evaluation.accuracy import pool
from lsa.evaluation.accuracy.data import suites
from lsa.evaluation.accuracy.records import join
from lsa.evaluation.constants import Dispatch, Launches, Pool, Records

# What a server holds at once when neither the plan's cap nor the run's server info says
SERVER_REQUESTS = 8


@dataclass
class Measured:
    """What one run has left, how it is being worked and how long it will take - read from its files."""

    run_dir: Path
    remaining: int = 0
    claimed: int = 0
    # Workers with a fresh heartbeat, and jobs queued or running for the run, of which some are still starting
    live: int = 0
    queued: int = 0
    seconds_per_unit: float = 0.0
    # Worker-slots the unclaimed requests need - each task's unclaimed requests over what one worker holds of it
    slots_needed: float = 0.0
    # Remaining seconds at one worker, summed over tasks
    work_seconds: float = 0.0
    stalled: str = ""
    added: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def workers(self) -> int:
        """Return the workers the run has or is about to have, counting those added in this look."""
        return max(self.live, self.queued) + self.added

    @property
    def starting(self) -> int:
        """Return the jobs submitted that have no heartbeat yet, which will claim before another is useful."""
        return max(0, self.queued - self.live) + self.added

    @property
    def projected_seconds(self) -> float:
        """Return how long the run takes at the workers it has, which is what the dispatcher orders runs by."""
        return self.work_seconds / max(1, self.workers)

    def room(self, max_workers: int) -> int:
        """Return how many more workers the run can use - what is unclaimed less those starting, within the cap."""
        useful = math.ceil(self.slots_needed) - self.starting
        return max(0, min(useful, max_workers - self.workers))


def load_submission(run_dir: Path) -> dict[str, Any] | None:
    """Return how a run is submitted to, or None for a run no pooled launch wrote."""
    submission: dict[str, Any] | None = io.read_json(run_dir / Pool.DIR / Pool.SUBMISSION, None)
    return submission


def load_experiment(submission: dict[str, Any]) -> Any:
    """Load the evaluation a submission came from, pinned to the run it was written for.

    Args:
        submission: What `submit_pool` wrote.

    Returns:
        The accuracy config, its `run.id` and `run.output` those of the run.
    """
    from lsa.evaluation.config import load

    experiments = load(submission["config"], submission["overrides"], submission["kind"])
    experiment = experiments[submission["index"]]
    experiment.run.output = submission["output"]
    experiment.run.id = submission["id"]
    return experiment


def server_capacity(run_dir: Path) -> int:
    """Return how many requests the run's server holds at once, from the server info a worker wrote."""
    info = io.read_json(run_dir / Records.SERVER_INFO, {})
    return int(info.get("max_running_requests") or SERVER_REQUESTS)


def plan_doc_ids(plan: suites.Plan, run_dir: Path) -> list[int] | None:
    """Return the docs a plan sends, or None when only the data can say and no worker has counted them.

    A worker writes the docs of every plan it builds, which is what the data held - a RULER file with fewer rows
    than the plan's `n` has fewer docs than the plan names.
    """
    counted = io.read_json(pool.pool_dir(run_dir) / Pool.UNITS / f"{plan.task}.json", None)
    if counted is not None:
        return [int(doc) for doc in counted["doc_ids"]]
    return plan.ids()


def stalled_by(run_dir: Path, max_failures: int) -> str:
    """Say why a run is left alone - its last launches all failed - or return nothing."""
    logged = launches.read(run_dir)
    tail = logged[-max_failures:]
    if len(tail) == max_failures and all(launch.get("outcome") == Launches.FAILED for launch in tail):
        return f"its last {max_failures} launches failed - {tail[-1].get('error', '')}"
    return ""


def measure(
    run_dir: Path,
    plans: list[suites.Plan],
    concurrency_scale: float,
    queued: int,
    ledger: pool.Ledger,
    default_unit_seconds: float,
    lease: float = Pool.LEASE_SECONDS,
    now: float | None = None,
) -> Measured:
    """Read a run's files and say what is left of it.

    Args:
        run_dir: The run directory.
        plans: The plans of its suite, as the config selects them.
        concurrency_scale: The run's `concurrency_scale`.
        queued: Jobs queued or running in slurm for the run.
        ledger: A ledger over the run directory, kept between looks so each file is parsed once.
        default_unit_seconds: What a request costs when none has been measured.
        lease: Seconds a claim holds without a touch.
        now: Seconds since the epoch, or None for the clock.

    Returns:
        The run's measurement.
    """
    now = time.time() if now is None else now
    record = io.read_json(run_dir / Records.RECORD, {})
    skipped = set(record.get("skipped", {}))
    measured = Measured(run_dir, queued=queued)
    measured.live = len(pool.live_workers(run_dir, Dispatch.DEAD_AFTER, now))
    capacity = server_capacity(run_dir)
    costs: dict[str, float] = {}
    left: dict[str, tuple[int, int, int]] = {}
    for plan in plans:
        if plan.task in skipped:
            continue
        ids = plan_doc_ids(plan, run_dir)
        if ids is None:
            measured.notes.append(f"{plan.task}: docs not counted yet")
            continue
        standing = ledger.standing(plan.task)
        wanted = {(doc, repeat) for repeat in range(plan.repeats) for doc in ids}
        remaining = {key for key in wanted if key not in standing.finished and not standing.exhausted(key)}
        if not remaining:
            continue
        touched = pool.newest_claims(pool.pool_dir(run_dir) / Pool.CLAIMS, plan.task)
        names = {key: f"{key[0]}-{key[1]}" for key in remaining}
        claimed = sum(1 for name in names.values() if name in touched and now - touched[name][1] < lease)
        cap = (
            capacity
            if plan.concurrency is None
            else min(capacity, max(1, int(plan.concurrency * concurrency_scale)))
        )
        left[plan.task] = (len(remaining), claimed, cap)
        if standing.seconds:
            costs[plan.task] = sum(standing.seconds) / len(standing.seconds)
    fallback = sum(costs.values()) / len(costs) if costs else default_unit_seconds
    measured.seconds_per_unit = fallback
    for task, (remaining_count, claimed_count, cap) in left.items():
        measured.remaining += remaining_count
        measured.claimed += claimed_count
        measured.slots_needed += (remaining_count - claimed_count) / cap
        measured.work_seconds += remaining_count * costs.get(task, fallback) / cap
    return measured


@dataclass
class Settings:
    """What one dispatcher looks at and how it acts."""

    runs: list[Path]
    reservation: str
    max_workers: int
    keep_free: int
    default_unit_seconds: float
    max_failures: int
    dry_run: bool
    parameters: dict[str, Any]
    folder: str | None


class Dispatcher:
    """The runs one dispatcher looks after, and what it remembers between two looks."""

    def __init__(self, settings: Settings) -> None:
        """Start a dispatcher that has looked at nothing yet."""
        self.settings = settings
        self.ledgers: dict[Path, pool.Ledger] = {}
        self.plans: dict[Path, tuple[list[suites.Plan], float]] = {}

    def discover(self) -> list[Path]:
        """Return the run directories under the given paths that can be submitted to, naming the ones that cannot."""
        found = []
        for path in self.settings.runs:
            candidates = [path] if (path / Records.RECORD).is_file() or pool.is_pooled(path) else []
            for parent in (path, path / "accuracy"):
                if parent.is_dir():
                    candidates += sorted(child for child in parent.iterdir() if child.is_dir())
            for run_dir in dict.fromkeys(candidates):
                if load_submission(run_dir) is not None:
                    found.append(run_dir)
                elif pool.is_pooled(run_dir):
                    print(f"skip {run_dir} - pooled but has no {Pool.SUBMISSION}", flush=True)
        return found

    def plans_of(self, run_dir: Path, submission: dict[str, Any]) -> tuple[list[suites.Plan], float]:
        """Return a run's plans and its concurrency scale, loaded from its config once."""
        if run_dir not in self.plans:
            experiment = load_experiment(submission)
            data = experiment.data
            plans = suites.select(data.suite, data.tasks or None, data.lengths or None)
            self.plans[run_dir] = (plans, experiment.run.concurrency_scale)
        return self.plans[run_dir]

    def look(self) -> list[Measured]:
        """Measure every run, counting the jobs slurm has queued for each."""
        queued = slurm.queue()
        measures = []
        for run_dir in self.discover():
            submission = load_submission(run_dir)
            if submission is None:
                continue
            plans, scale = self.plans_of(run_dir, submission)
            name = f"{Launches.POOL_JOB}{run_dir.name}"
            jobs = sum(1 for job in queued if job.name == name)
            ledger = self.ledgers.setdefault(run_dir, pool.Ledger(run_dir))
            measured = measure(run_dir, plans, scale, jobs, ledger, self.settings.default_unit_seconds)
            measured.stalled = stalled_by(run_dir, self.settings.max_failures)
            measures.append(measured)
        return measures

    def free_slots(self) -> int:
        """Return how many workers the reservation can start now - free nodes less those queued for it and the ones kept free."""
        free = len(slurm.free_nodes(self.settings.reservation))
        waiting = sum(
            1
            for job in slurm.queue()
            if job.state == "PENDING" and job.reservation == self.settings.reservation
        )
        return max(0, free - waiting - self.settings.keep_free)

    def decide(self, measures: list[Measured], slots: int) -> list[Measured]:
        """Give each free slot to the run that would finish last, and return the runs given one, once per slot.

        Args:
            measures: Every run's measurement, whose `added` this fills in.
            slots: Free workers the reservation can start.

        Returns:
            The run given each slot, in the order they were given.
        """
        given = []
        for _ in range(slots):
            wanting = [
                measured
                for measured in measures
                if not measured.stalled and measured.remaining and measured.room(self.settings.max_workers)
            ]
            if not wanting:
                break
            chosen = max(
                wanting,
                key=lambda measured: (measured.projected_seconds, -measured.workers, measured.run_dir.name),
            )
            chosen.added += 1
            given.append(chosen)
        return given

    def report(self, measures: list[Measured], slots: int) -> None:
        """Print what every run has left and what the reservation can start."""
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {slots} free slots in {self.settings.reservation}",
            flush=True,
        )
        for measured in measures:
            state = f"STALLED - {measured.stalled}" if measured.stalled else "ok"
            print(
                f"  {measured.run_dir.name}: {measured.remaining} requests left, {measured.claimed} claimed, "
                f"{measured.live} workers live of {max(measured.live, measured.queued)}, "
                f"{measured.seconds_per_unit:.0f} s a request, about {measured.projected_seconds / 3600:.1f} h  [{state}]",
                flush=True,
            )
            for note in measured.notes:
                print(f"    note: {note}", flush=True)

    def submit(self, chosen: Measured) -> None:
        """Submit one worker to a run, or say that it would be."""
        from lsa.evaluation.run import job_parameters, submit_pool

        label = f"{chosen.run_dir.name} ({chosen.remaining} left, about {chosen.projected_seconds / 3600:.1f} h at {chosen.workers - 1} workers)"
        if self.settings.dry_run:
            print(f"would add a worker to {label}", flush=True)
            return
        submission = load_submission(chosen.run_dir)
        if submission is None:
            return
        experiment = load_experiment(submission)
        print(f"adding a worker to {label}", flush=True)
        source = {key: submission[key] for key in ("config", "overrides", "index")}
        parameters = {**job_parameters([experiment]), **self.settings.parameters}
        submit_pool(experiment, 1, source, parameters, self.settings.folder)

    def step(self) -> None:
        """Look once and give the free slots away."""
        measures = self.look()
        slots = self.free_slots()
        self.report(measures, slots)
        for chosen in self.decide(measures, slots):
            self.submit(chosen)


def publish_runs(
    run_dirs: list[Path], root: Path, joins: list[list[Path]], into: Path | None, apply: bool
) -> None:
    """Publish the runs as they stand and join the configurations asked for, through the commands that do it by hand.

    Args:
        run_dirs: The run directories being dispatched.
        root: The evaluations root they are published into.
        joins: Each a base run directory and the runs that extend it.
        into: Where joined runs go.
        apply: Write, rather than print what would be written.
    """
    results = sorted({run_dir.parent.parent for run_dir in run_dirs})
    publish.sync(results, root, apply)
    if into is None:
        return
    for parts in joins:
        try:
            join.join_runs(parts, into, apply, in_progress=True, replace=True)
        except ValueError as refusal:
            print(f"skip join of {parts[0].name} - {refusal}", flush=True)


def main() -> None:
    """Parse the command line and keep the reservation full."""
    parser = argparse.ArgumentParser(
        description="Add workers to pooled runs as nodes free in a reservation",
        epilog="Every submit flag of `lsa evaluate accuracy` applies to the workers added.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="run directories, or a results root holding them")
    slurm.add_flags(parser)
    parser.add_argument(
        "--max-workers", type=int, default=Dispatch.MAX_WORKERS, help="workers one run may have"
    )
    parser.add_argument("--keep-free", type=int, default=0, help="nodes of the reservation never to fill")
    parser.add_argument("--interval", type=float, default=Dispatch.INTERVAL, help="seconds between two looks")
    parser.add_argument("--once", action="store_true", help="look once and exit")
    parser.add_argument(
        "--unit-seconds",
        type=float,
        default=Dispatch.UNIT_SECONDS,
        help="seconds a request takes in a run where none has finished",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=Dispatch.MAX_FAILURES,
        help="failed launches in a row that stop a run's workers",
    )
    parser.add_argument(
        "--publish-every",
        type=float,
        default=0.0,
        help="seconds between publishing the runs as they stand, 0 for never",
    )
    parser.add_argument(
        "--publish-root", type=Path, default=None, help="the evaluations root, the shared one by default"
    )
    parser.add_argument(
        "--join",
        nargs="+",
        action="append",
        type=Path,
        default=[],
        metavar="RUN",
        help="a base run directory and the runs that extend it, joined at each publish - repeatable",
    )
    parser.add_argument("--join-into", type=Path, default=None, help="where joined runs go")
    args = parser.parse_args()
    args.reservation = args.reservation or site.slurm_values().get("reservation")
    if not args.reservation:
        raise site.Unsupported(
            f"dispatch keeps a reservation full, and site {site.current()!r} sets none - name one on the command line"
        )
    require_extra("omegaconf", "hydra")

    parameters = slurm.overrides(args)
    settings = Settings(
        runs=[path.resolve() for path in args.runs],
        reservation=args.reservation,
        max_workers=args.max_workers,
        keep_free=args.keep_free,
        default_unit_seconds=args.unit_seconds,
        max_failures=args.max_failures,
        dry_run=args.dry_run,
        parameters=parameters,
        folder=args.folder,
    )
    dispatcher = Dispatcher(settings)
    last_publish = 0.0
    while True:
        dispatcher.step()
        if args.publish_every and time.monotonic() - last_publish >= args.publish_every:
            publish_runs(
                dispatcher.discover(),
                args.publish_root or paths.evaluations(),
                [[path.resolve() for path in parts] for parts in args.join],
                args.join_into,
                not args.dry_run,
            )
            last_publish = time.monotonic()
        if args.once:
            return
        time.sleep(args.interval)
