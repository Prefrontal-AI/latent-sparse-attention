"""One worker of a pooled run - one server, draining the run's queue of requests beside its sibling workers.

    python -m lsa.evaluation.accuracy --pool --run-id RUN ...

A `Runner` sends a whole suite to its server. A `PoolRunner` sends only the units it claims from the run's
[queue](pool.py), as many as its server holds at once, taking them from the first task that still has
unclaimed ones so tasks finish in plan order and a record that is still filling stays meaningful. It keeps one
server up and leaves when nothing is left to claim - a unit another worker holds is that worker's, and one
whose worker dies is taken by the next worker that looks, or by one the dispatcher submits.

The phases of a run hold within a worker: it does not claim from a later phase while it has a request in
flight from an earlier one, because a reasoning chain served beside a long prompt's chunked prefill decodes
only between its chunks. Another worker is another server and is in whichever phase it likes.

A shared context is in flight alone, as in a single job: a worker claims from no other context while it has a
request of one in flight, and claims one request of a context its server has not yet prefilled until that
request has. Questions over one context share its KV only once it is in the radix cache - contexts sent side
by side evict each other at a few million tokens, and every question prefills the whole context again.

A plan whose cap is the server's own keeps one request more than that in flight, which waits in the server's
queue. The server runs no more at once, but takes the waiting request's prompt in - a few million tokens to read
and tokenize, about a minute at 10M - while the one ahead of it decodes, rather than with the GPUs idle after it
ends. A row's time is its time in a slot: the wait is recorded beside it, from the claim to the reply that freed
the slot.

A worker writes sample rows to its own shard and never to the graded file. The record is rebuilt from
everything on disk by `rebuild`, under a lock that only one worker holds at a time, and every worker rebuilds
as it finishes so the last one out leaves a record that holds every row. The last one out rebuilds once more
after a pause, since the rows another node appended a moment before may not yet show on its own. A rebuild
is a function of the files, so two that overlapped would write the same record twice, which is why the lock
is a courtesy to the CPU and not what keeps the record true. A task enters the record once all its units are done or out of attempts, and
a task entry is graded again only when the files say it changed.
"""

import asyncio
import contextlib
import dataclasses
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lsa import io
from lsa.evaluation.accuracy import pool, records, tasks
from lsa.evaluation.accuracy.client import Client, user_message
from lsa.evaluation.accuracy.data import suites
from lsa.evaluation.accuracy.runner import (
    Runner,
    finished_rows,
    grade_plan,
    require_sampling,
    sample_row,
    score_line,
    targeted_items,
)
from lsa.evaluation.constants import Dispatch, Pool, Records


@dataclass
class Prepared:
    """A plan whose prompts are built - everything a worker needs to send its units and to grade the task."""

    definition: tasks.TaskDefinition
    lm_eval_task: Any
    items: dict[int, tasks.Item]
    digest: str
    # Every unit of the plan, repeat by repeat and doc by doc
    units: list[pool.Unit]


class PoolRunner(Runner):
    """One worker's share of a run, continued wherever the run's other workers and earlier launches left it."""

    def __init__(self, *args: Any, worker: str | None = None, **kwargs: Any) -> None:
        """Start a worker over the run directory the runner arguments name.

        Args:
            *args: What `Runner` takes.
            worker: This worker's name, unique among the run's workers, or None for its slurm job.
            **kwargs: What `Runner` takes.

        Raises:
            ValueError: The run is asked to score every sample again, which a pool never does.
        """
        super().__init__(*args, **kwargs)
        if not self.resume:
            raise ValueError(
                "a pooled run always continues the samples on disk - use a run of its own to start over"
            )
        self.worker = worker or pool.worker_name()
        # Workers started in one second outside slurm would share the id a launch's clock gives them
        self.server_id = self.job or self.worker
        self.ledger = pool.Ledger(self.out)
        self.graded_ledger = pool.Ledger(self.out, exact=True)
        self.queue = pool.Queue(self.out, self.worker, on_beat=self.say_alive)
        self.prepared: dict[str, Prepared | None] = {}
        self.skip_reasons: dict[str, str] = {}
        # Units this worker's server failed, which it leaves to another worker or a later launch
        self.failed_here: set[pool.Unit] = set()
        self.group_caps: dict[str, int] = {}
        # Shared contexts a request of which this worker's server has answered, so the cache holds them
        self.prefilled: set[str] = set()
        # Units claimed past the server's cap, when they were claimed, and when a reply ahead of them freed a slot
        self.queued: dict[pool.Unit, float] = {}
        self.admitted: dict[pool.Unit, float] = {}
        self.last_rebuild = 0.0
        self.running_count = 0
        self.ended = False
        self.wanted: dict[str, set[pool.Key] | None] = {}

    def say_alive(self) -> None:
        """Write what this worker is and when it last was alive, where the dispatcher and the record read it."""
        pool.write_worker(
            self.out,
            self.worker,
            {
                "worker": self.worker,
                "job": self.job,
                "host": socket.gethostname(),
                "started": self.launched,
                "heartbeat": time.time(),
                "seconds": round(time.perf_counter() - self.start, 1),
                "running": self.running_count,
                "ended": self.ended,
                "server_id": self.server_id,
                "server": self.server_entry(),
            },
        )

    def prepare(self, plan: suites.Plan) -> Prepared | None:
        """Build a plan's prompts once, or say why the task is skipped and return None.

        Args:
            plan: The plan.

        Returns:
            The built plan, or None for a task this server cannot hold or this machine has no data for.
        """
        if plan.task not in self.prepared:
            self.prepared[plan.task] = self.build(plan)
        return self.prepared[plan.task]

    def build(self, plan: suites.Plan) -> Prepared | None:
        """Build a plan's prompts, hold their hash to the pinned one and settle which units it is made of."""
        definition = tasks.resolve(plan.task, self.data_root, self.max_context)
        reason = self.over_context(definition, plan)
        if reason:
            self.skip_reasons[plan.task] = reason
            self.log(f"  {plan.key}: SKIPPED - {reason}")
            return None
        try:
            lm_eval_task = definition.build()
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.skip_reasons[plan.task] = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.log(f"  {plan.key}: SKIPPED - {self.skip_reasons[plan.task]}")
            return None
        items = targeted_items(definition, lm_eval_task, plan)
        digest = tasks.prompt_set_hash(items)
        self.check_hash(plan, digest)
        pool.publish_once(
            pool.pool_dir(self.out) / Pool.UNITS / f"{plan.task}.json",
            {"doc_ids": [item.doc_id for item in items]},
        )
        units = [
            pool.Unit(plan.task, item.doc_id, repeat) for repeat in range(plan.repeats) for item in items
        ]
        prepared = Prepared(definition, lm_eval_task, {item.doc_id: item for item in items}, digest, units)
        self.check_earlier_samples(plan)
        return prepared

    def check_earlier_samples(self, plan: suites.Plan) -> None:
        """Refuse to add to the samples an earlier launch drew under other sampling than this plan's."""
        if self.ledger.standing(plan.task).finished:
            rows = finished_rows(pool.task_rows(self.out, plan.task))
            self.check_resumable(plan, rows)

    def wanted_keys(self, plan: suites.Plan) -> set[pool.Key] | None:
        """Return the (doc, repeat) of every unit a plan names without building its prompts, or None when only they say.

        A plan that asks for the first n docs or a named subset knows its docs, so a task every unit of which is
        done never has its prompts built again by a worker that only looks. A file with fewer docs than n makes
        this a superset, which can only make a done task look not done and send the worker to build it.
        """
        if plan.task not in self.wanted:
            ids = plan.ids()
            self.wanted[plan.task] = (
                None if ids is None else {(doc, repeat) for repeat in range(plan.repeats) for doc in ids}
            )
        return self.wanted[plan.task]

    def pending(self, plan: suites.Plan) -> list[pool.Unit]:
        """Return a plan's units that are not done, not out of attempts and not given up on by this worker."""
        standing = self.ledger.standing(plan.task)
        wanted = self.wanted_keys(plan)
        if wanted is not None and all(key in standing.finished or standing.exhausted(key) for key in wanted):
            return []
        prepared = self.prepare(plan)
        if prepared is None:
            return []
        return [
            unit
            for unit in prepared.units
            if unit.key not in standing.finished
            and not standing.exhausted(unit.key)
            and unit not in self.failed_here
        ]

    def claim_more(self, running: dict[pool.Unit, suites.Plan]) -> list[tuple[suites.Plan, pool.Unit]]:
        """Claim units until the server's cap is full, from the first task in plan order that has any.

        A plan of a shared context is claimed from only while no other context is in flight, and only one at a
        time until the server has prefilled its context.

        Args:
            running: The units this worker has in flight, each with its plan.

        Returns:
            The units claimed, each with its plan.
        """
        room = self.in_flight - len(running)
        phase = min((plan.phase for plan in running.values()), default=None)
        in_group = Counter(plan.cap_group for plan in running.values())
        in_context = Counter(plan.shared_context for plan in running.values() if plan.shared_context)
        claimed: list[tuple[suites.Plan, pool.Unit]] = []
        for plan in sorted(self.plans, key=lambda plan: plan.phase):
            if phase is not None and plan.phase > phase:
                break
            extra = int(self.queues_behind(plan))
            allowed = room + extra
            if plan.concurrency is not None:
                allowed = min(allowed, self.group_caps[plan.cap_group] + extra - in_group[plan.cap_group])
            context = plan.shared_context
            if context is not None and set(in_context) - {context}:
                continue
            if context is not None and context not in self.prefilled:
                allowed = min(allowed, 1 - in_context[context])
            if allowed <= 0:
                continue
            units = self.queue.claim(self.pending(plan), allowed)
            claimed += [(plan, unit) for unit in units]
            now = time.perf_counter()
            for unit in units[max(room, 0) :]:
                self.queued[unit] = now
            room -= len(units)
            in_group[plan.cap_group] += len(units)
            if units:
                phase = plan.phase
                if context is not None:
                    in_context[context] += len(units)
        return claimed

    def queues_behind(self, plan: suites.Plan) -> bool:
        """Say whether one request of a plan more than the server runs may wait in its queue.

        Only a plan capped at the server's own cap, so the server runs no more of it at once than without, and over a
        shared context only once the context is cached, so the waiting request never prefills it a second time.

        Args:
            plan: A plan of the run.
        """
        if plan.concurrency is None or self.group_caps.get(plan.cap_group) != self.in_flight:
            return False
        return plan.shared_context is None or plan.shared_context in self.prefilled

    def admit_next(self) -> None:
        """Mark the unit waiting longest in the server's queue as taken in, a reply ahead of it having freed a slot."""
        waiting = [unit for unit in self.queued if unit not in self.admitted]
        if waiting:
            self.admitted[waiting[0]] = time.perf_counter()

    def queue_wait(self, unit: pool.Unit) -> float:
        """Return how long a unit waited in the server's queue, and forget it.

        Args:
            unit: A unit whose reply just landed.
        """
        claimed = self.queued.pop(unit, None)
        admitted = self.admitted.pop(unit, None)
        if claimed is None or admitted is None:
            return 0.0
        return max(admitted - claimed, 0.0)

    async def run_unit(
        self,
        plan: suites.Plan,
        unit: pool.Unit,
        client: Client,
        session: Any,
        limits: list[asyncio.Semaphore],
    ) -> None:
        """Send one claimed unit and keep its reply in this worker's shard the moment it lands.

        Args:
            plan: The unit's plan.
            unit: The unit.
            client: The client the worker sends through.
            session: Its open session.
            limits: The semaphores the request holds - its cap group's, if it has a cap, then the pool's.
        """
        prepared = self.prepare(plan)
        if prepared is None:
            return
        item = prepared.items[unit.doc_id]
        # A request waiting in the server's queue spends up to a whole request's time there before its own
        # => it may take two
        timeout = 2 * client.timeout if unit in self.queued else None
        replies = await client.gather(
            session, [user_message(item.prompt)], require_sampling(plan), limits=limits, timeout=timeout
        )
        row = {
            **sample_row(item, unit.repeat, replies[0]),
            "server": self.server_id,
            "concurrency": self.task_concurrency(plan),
        }
        waited = self.queue_wait(unit)
        if waited:
            row["seconds"] = round(max(row["seconds"] - waited, 0.0), 2)
            row["queued_seconds"] = round(waited, 2)
        records.append_jsonl(str(pool.shard_file(self.out, self.worker, plan.task)), row)
        if row["error"]:
            self.failed_here.add(unit)
            self.queue.release(unit)
            self.log(f"  {plan.key}: doc {unit.doc_id} repeat {unit.repeat} failed - {row['error']}")
            self.check_server_alive()
            return
        if plan.shared_context is not None:
            self.prefilled.add(plan.shared_context)
        self.queue.finish(unit)

    async def run_pool(self) -> None:
        """Claim and send units until none is left to claim, rebuilding the record as tasks come to an end."""
        # A slot past the server's cap for the one request that may wait in its queue
        shared = asyncio.Semaphore(self.in_flight + 1)
        group_limits: dict[str, asyncio.Semaphore] = {}
        for plan in self.plans:
            if plan.concurrency is not None and plan.cap_group not in self.group_caps:
                cap = self.group_caps[plan.cap_group] = self.task_concurrency(plan)
                group_limits[plan.cap_group] = asyncio.Semaphore(cap + int(cap == self.in_flight))
        client = Client(
            self.base_url, concurrency=self.in_flight, timeout=self.timeout, top_logprobs=self.top_logprobs
        )
        flying: dict[asyncio.Task[None], tuple[suites.Plan, pool.Unit]] = {}
        async with client.session() as session:
            try:
                while True:
                    running = {unit: plan for plan, unit in flying.values()}
                    for plan, unit in self.claim_more(running):
                        group = [group_limits[plan.cap_group]] if plan.concurrency is not None else []
                        future = asyncio.create_task(
                            self.run_unit(plan, unit, client, session, [*group, shared])
                        )
                        flying[future] = (plan, unit)
                    self.running_count = len(flying)
                    if not flying:
                        return
                    done, _ = await asyncio.wait(flying, return_when=asyncio.FIRST_COMPLETED)
                    for future in done:
                        flying.pop(future)
                        future.result()
                        self.admit_next()
                    if time.monotonic() - self.last_rebuild > Pool.REBUILD_SECONDS:
                        self.rebuild(wait=False)
            finally:
                for future in flying:
                    future.cancel()
                await asyncio.gather(*flying, return_exceptions=True)

    def rebuild(self, wait: bool = True) -> bool:
        """Rebuild the record from every file the run holds, under the lock that lets one worker do it at a time.

        Args:
            wait: Wait for the worker that holds the lock rather than leave the rebuild to it.

        Returns:
            True when this call rebuilt it.
        """
        lock = io.Claim(pool.pool_dir(self.out) / Pool.LOCK, Pool.LOCK_STALE_SECONDS)
        while not lock.take():
            if not wait:
                return False
            time.sleep(2.0)
        try:
            self.grade_settled()
            Runner.write(self)
        finally:
            lock.release()
        self.last_rebuild = time.monotonic()
        return True

    def write(self) -> None:
        """Write the record as the files stand, or leave it to the worker already writing it."""
        self.rebuild(wait=False)

    def grade_settled(self) -> None:
        """Take up what the record on disk holds, grade every task of this run that came to an end, and set the header.

        A task came to an end when every unit of it is done or out of attempts, whichever worker did it. One
        whose entry already says what the files say is left as it is.
        """
        existing = io.read_json(os.path.join(self.out, Records.RECORD), None) or {}
        prior_run = existing.get("run", {})
        self.results = existing.get("tasks", {})
        self.skipped = existing.get("skipped", {})
        self.prior_plans = prior_run.get("plans", [])
        self.first_server = existing.get("server")
        self.first_suite = prior_run.get("suite")
        self.started = prior_run.get("started") or self.started
        self.edits = prior_run.get("edits", self.edits)
        workers = pool.read_workers(self.out)
        self.prior_servers = {
            **existing.get("servers", {}),
            **{body.get("server_id") or name: body["server"] for name, body in workers.items()},
        }
        prior = io.read_json(pool.pool_dir(self.out) / Pool.PRIOR, {}).get("seconds", 0.0)
        self.prior_seconds = prior + sum(
            body.get("seconds", 0.0) for name, body in workers.items() if name != self.worker
        )
        for plan in self.plans:
            self.grade_task(plan)

    def entry_current(self, task: str, wanted: set[pool.Key], standing: pool.Standing) -> bool:
        """Say whether a task has come to an end and its record entry already says what the files say."""
        done = len(wanted & standing.finished)
        out_of_attempts = sum(1 for key in wanted if standing.exhausted(key))
        entry = self.results.get(task)
        return bool(
            entry
            and done + out_of_attempts == len(wanted)
            and entry["scored"] == done
            and entry["n"] == len(wanted)
            and wanted & standing.finished <= standing.graded
        )

    def grade_task(self, plan: suites.Plan) -> None:
        """Grade one task into the record if all its units have come to an end and its entry is out of date."""
        standing = self.graded_ledger.standing(plan.task)
        wanted = self.wanted_keys(plan)
        if wanted is not None and self.entry_current(plan.task, wanted, standing):
            self.skipped.pop(plan.task, None)
            return
        prepared = self.prepare(plan)
        if prepared is None:
            self.skipped[plan.task] = self.skip_reasons[plan.task]
            return
        self.skipped.pop(plan.task, None)
        wanted = {unit.key for unit in prepared.units}
        done = len(wanted & standing.finished)
        out_of_attempts = sum(1 for key in wanted if standing.exhausted(key))
        if done + out_of_attempts < len(wanted) or self.entry_current(plan.task, wanted, standing):
            return
        rows = [
            row
            for row in pool.one_per_unit(pool.task_rows(self.out, plan.task))
            if (row["doc_id"], row["repeat"]) in wanted
        ]
        enforced = records.budget_enforced(dataclasses.asdict(require_sampling(plan)), self.info)
        entry = grade_plan(
            plan,
            prepared.definition,
            prepared.lm_eval_task,
            list(prepared.items.values()),
            rows,
            self.out,
            self.data_root,
            self.manifest,
            enforced,
            self.prompt_block(plan, prepared.lm_eval_task, prepared.digest),
        )
        concurrency = self.task_concurrency(plan)
        # Workers of one run may serve a task under different caps, and rows written before rows carried one
        # => each row's time is shared by its own cap, and the entry counts the rows each cap served
        caps = Counter(row.get("concurrency") or Records.UNKNOWN_CAP for row in rows)
        seconds = round(
            sum((row.get("seconds") or 0.0) / (row.get("concurrency") or concurrency) for row in rows), 3
        )
        servers = Counter(row["server"] for row in rows if row.get("server"))
        output_tokens = entry["tokens"]["output"]["sum"] if entry["tokens"]["output"] else None
        entry.update(
            {
                "concurrency": max((cap for cap in caps if cap != Records.UNKNOWN_CAP), default=concurrency),
                "concurrency_rows": {str(cap): count for cap, count in sorted(caps.items(), key=str)},
                "seconds": seconds,
                "server": servers.most_common(1)[0][0] if servers else self.server_id,
                "output_tokens_per_second": round(output_tokens / seconds, 1)
                if output_tokens and seconds
                else None,
            }
        )
        self.results[plan.task] = entry
        self.log(f"{score_line(plan, prepared.definition, entry)}  errors {entry['errors']}")

    def last_out(self) -> bool:
        """Say whether every other worker of the run has ended or stopped beating.

        A worker says it ended before its closing rebuild and asks this after, so of two that end together
        the second to ask sees the first ended.
        """
        others = pool.live_workers(self.out, Dispatch.DEAD_AFTER)
        return all(worker == self.worker for worker in others)

    def warn_missing(self) -> None:
        """Say which tasks of this run the record holds no entry for, so a gap is not read as a smaller suite."""
        missing = [
            plan.task
            for plan in self.plans
            if plan.task not in self.results and plan.task not in self.skipped
        ]
        if missing:
            self.log(f"WARNING {len(missing)} tasks have no entry in the record: {' '.join(missing)}")

    def run(self) -> dict[str, Any]:
        """Drain the run's queue, rebuild the record once the last unit this worker sent is in, and return it."""
        os.makedirs(self.out, exist_ok=True)
        io.write_json(os.path.join(self.out, Records.SERVER_INFO), self.info, indent=1)
        self.log(
            f"{self.run_id}: worker {self.worker}, {len(self.plans)} tasks -> {self.out}, "
            f"{self.in_flight} requests in flight"
        )
        pool.publish_once(pool.pool_dir(self.out) / Pool.PRIOR, {"seconds": self.prior_seconds})
        self.say_alive()
        self.queue.start()
        try:
            asyncio.run(self.run_pool())
        except BaseException:
            with contextlib.suppress(Exception):
                self.rebuild(wait=False)
            raise
        finally:
            self.queue.stop()
            self.ended = True
            self.say_alive()
        self.rebuild(wait=True)
        if self.last_out():
            time.sleep(Pool.SETTLE_SECONDS)
            self.rebuild(wait=True)
            self.warn_missing()
        self.log(f"worker {self.worker} done in {round(time.perf_counter() - self.start, 1)} s")
        if self.skip_reasons:
            self.log(f"skipped {len(self.skip_reasons)} tasks without data: {' '.join(self.skip_reasons)}")
        return self.header()
