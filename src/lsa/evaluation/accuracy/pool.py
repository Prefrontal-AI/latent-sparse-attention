"""A run's requests as a queue on the shared filesystem, so any number of workers can drain one run.

    <run dir>/pool/claims/<task>/<doc>-<repeat>.<epoch>     one claim file per unit and epoch, its mtime the heartbeat
    <run dir>/pool/shards/<worker>/<task>.jsonl             the rows a worker sampled, written by that worker alone
    <run dir>/pool/workers/<worker>.json                    where a worker runs, its server and its last heartbeat
    <run dir>/samples/<task>.jsonl                          the graded rows of every shard and of earlier launches

A unit is one request - a doc of a task at one repeat, which is what a sample row is. A run is the same
run whether one job or ten drain it, so nothing here is a second format: a run directory a single job wrote
is a pool whose only shard is the graded file, and every unit it holds counts as done.

A worker takes a unit by creating its claim file with `O_CREAT | O_EXCL`, which the shared filesystem
decides: of two workers creating the same name, one is told it exists. The creation is the only atomic
step the queue leans on - no `flock`, which NFS and Lustre clients honour only under conditions a cluster
does not promise, no rename tricks and no lock server. A claim is a lease, kept by the mtime of its file.
A thread of the worker touches every claim it holds each minute, whatever its requests are doing, so a
sample of twenty minutes keeps its claim and a worker that died stops touching it. A lease nobody touched
for `Pool.LEASE_SECONDS` is expired, and the unit is taken again under the next epoch - a new file
`<unit>.<epoch + 1>`, created exclusively, so of several workers finding the lease expired one wins and
the rest find the epoch taken. The old claim is never deleted or rewritten, which is what keeps the
takeover free of a window in which two workers each believe they hold it.

Whether a unit is done is whatever the rows on disk say, never a flag beside them: a row with no error in
any shard or in the graded file. So a claim file is only the right to send a request, a unit finished by
a worker that then died is done, and a sample resumed from a single-job run needs no migration.
A request that came back with an error is retried up to `Pool.MAX_ATTEMPTS` times in all, counted from the
error rows of the shards, so a prompt the server refuses does not keep a pool alive.

A worker appends to its own shard only, so no file is ever appended to by two processes. The graded
`samples/<task>.jsonl` is written whole by the record rebuild alone, and a reader that wants a task's rows
takes it and every shard together.
"""

import json
import os
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.constants import Pool

Key = tuple[int, int]


@dataclass(frozen=True)
class Unit:
    """One request of a run - a doc of a task at one repeat."""

    task: str
    doc_id: int
    repeat: int

    @property
    def key(self) -> Key:
        """Return the (doc, repeat) a sample row names itself by."""
        return (self.doc_id, self.repeat)

    @property
    def name(self) -> str:
        """Return the unit's name inside its task's claim directory."""
        return f"{self.doc_id}-{self.repeat}"


def pool_dir(run_dir: str | os.PathLike[str]) -> Path:
    """Return a run directory's pool directory."""
    return Path(run_dir) / Pool.DIR


def is_pooled(run_dir: str | os.PathLike[str]) -> bool:
    """Say whether a run directory has had a pooled launch."""
    return pool_dir(run_dir).is_dir()


def worker_name() -> str:
    """Return this process's worker name - its slurm job, which is unique, or its host and pid outside slurm."""
    return os.environ.get("SLURM_JOB_ID") or f"{socket.gethostname()}-{os.getpid()}"


def samples_file(run_dir: str | os.PathLike[str], task: str) -> Path:
    """Return where a run keeps one task's graded rows."""
    return Path(run_dir) / "samples" / f"{task}.jsonl"


def shard_file(run_dir: str | os.PathLike[str], worker: str, task: str) -> Path:
    """Return where one worker keeps the rows it sampled of one task."""
    return pool_dir(run_dir) / Pool.SHARDS / worker / f"{task}.jsonl"


def shard_files(run_dir: str | os.PathLike[str], task: str) -> list[Path]:
    """Return every worker's shard of one task, in worker order."""
    return sorted((pool_dir(run_dir) / Pool.SHARDS).glob(f"*/{task}.jsonl"))


def worker_file(run_dir: str | os.PathLike[str], worker: str) -> Path:
    """Return where one worker says what it is and that it is alive."""
    return pool_dir(run_dir) / Pool.WORKERS / f"{worker}.json"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read the rows of a jsonl file another process may be appending to, dropping a line cut in the middle."""
    if not path.exists():
        return []
    rows = []
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def task_rows(run_dir: str | os.PathLike[str], task: str) -> list[dict[str, Any]]:
    """Return every row a run holds of one task - the graded file first, then each worker's shard.

    Args:
        run_dir: The run directory.
        task: The task's name.

    Returns:
        The rows, a (doc, repeat) possibly more than once - `one_per_unit` settles which stands.
    """
    rows = read_rows(samples_file(run_dir, task))
    for shard in shard_files(run_dir, task):
        rows += read_rows(shard)
    return rows


def one_per_unit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per (doc, repeat) - a row with no error over one with, the later of two alike."""
    kept: dict[Key, dict[str, Any]] = {}
    for row in rows:
        key = (row["doc_id"], row["repeat"])
        if key not in kept or row.get("error") is None or kept[key].get("error") is not None:
            kept[key] = row
    return list(kept.values())


@dataclass
class Standing:
    """What a task's files say of its units - which are done and how often each has failed."""

    finished: set[Key] = field(default_factory=set)
    # The finished units the graded file holds, which a rebuild is owed for any the shards hold and it does not
    graded: set[Key] = field(default_factory=set)
    failures: Counter[Key] = field(default_factory=Counter)
    # The seconds each finished request took, for a reader that wants to know how long a unit costs
    seconds: list[float] = field(default_factory=list)

    def exhausted(self, key: Key) -> bool:
        """Say whether a unit has failed as often as the pool allows."""
        return key not in self.finished and self.failures[key] >= Pool.MAX_ATTEMPTS


class Ledger:
    """What a run's sample files say is done, read as little as it can be.

    A shard is only ever appended to, so each file is parsed from where the last read stopped. The graded
    file is rewritten whole by a rebuild, and is parsed again from the start when it is a different file.

    A bare stat on a shared filesystem may be answered from the client's cache, a second or so behind what
    another node appended, while an open asks the server. A ledger that only picks units to claim can live
    with that - a done unit that looks pending still has its finisher's live claim. One that grades cannot,
    since a task whose last rows it misses looks unfinished and is left out of the record, so it is `exact`.
    """

    def __init__(self, run_dir: str | os.PathLike[str], exact: bool = False) -> None:
        """Start a ledger over a run directory, reading nothing yet.

        Args:
            run_dir: The run directory.
            exact: Open every file to learn its size rather than trust a stat the filesystem may have cached.
        """
        self.run_dir = Path(run_dir)
        self.exact = exact
        self.parsed: dict[Path, tuple[int, int, list[tuple[Key, bool, float]]]] = {}

    def stat_of(self, path: Path) -> os.stat_result | None:
        """Return a file's inode and size as this ledger trusts them, or None when it is not there."""
        try:
            if not self.exact:
                return path.stat()
            with path.open("rb") as source:
                return os.fstat(source.fileno())
        except FileNotFoundError:
            return None

    def keys_of(self, path: Path) -> list[tuple[Key, bool, float]]:
        """Return the (doc, repeat) of every whole row of a file, with whether the row is an error and its seconds."""
        stat = self.stat_of(path)
        if stat is None:
            self.parsed.pop(path, None)
            return []
        inode, offset, keys = self.parsed.get(path, (stat.st_ino, 0, []))
        if inode != stat.st_ino or stat.st_size < offset:
            inode, offset, keys = stat.st_ino, 0, []
        if stat.st_size > offset:
            with path.open("rb") as source:
                source.seek(offset)
                text = source.read()
            whole = text[: text.rfind(b"\n") + 1]
            for line in whole.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.append(
                    ((row["doc_id"], row["repeat"]), row.get("error") is not None, row.get("seconds") or 0.0)
                )
            offset += len(whole)
        self.parsed[path] = (inode, offset, keys)
        return keys

    def standing(self, task: str) -> Standing:
        """Return which units of a task are done and how often the others failed.

        The graded file counts for what is finished and not for failures - the rows it keeps of requests that
        failed in an earlier single job are retried afresh, as that job would have retried them.
        """
        standing = Standing()
        for key, failed, _ in self.keys_of(samples_file(self.run_dir, task)):
            if not failed:
                standing.finished.add(key)
                standing.graded.add(key)
        for shard in shard_files(self.run_dir, task):
            for key, failed, seconds in self.keys_of(shard):
                if failed:
                    standing.failures[key] += 1
                else:
                    standing.finished.add(key)
                    standing.seconds.append(seconds)
        return standing


def newest_claims(claims: Path, task: str) -> dict[str, tuple[int, float]]:
    """Return the newest claim of every unit of a task that has one - its epoch and when it was last touched.

    Args:
        claims: The run's claims directory.
        task: The task's name.

    Returns:
        Unit name to its newest epoch and that claim file's mtime.
    """
    newest: dict[str, tuple[int, float]] = {}
    try:
        entries = list(os.scandir(claims / task))
    except FileNotFoundError:
        return newest
    for entry in entries:
        name, _, epoch = entry.name.rpartition(".")
        if not epoch.isdigit():
            continue
        try:
            touched = entry.stat().st_mtime
        except FileNotFoundError:
            continue
        if name not in newest or int(epoch) > newest[name][0]:
            newest[name] = (int(epoch), touched)
    return newest


class Queue:
    """One worker's claims on a run's units, kept alive by a heartbeat thread."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        worker: str,
        lease: float = Pool.LEASE_SECONDS,
        heartbeat: float = Pool.HEARTBEAT_SECONDS,
        clock: Callable[[], float] = time.time,
        on_beat: Callable[[], None] | None = None,
    ) -> None:
        """Open a worker's side of a run's queue.

        Args:
            run_dir: The run directory.
            worker: This worker's name, unique among the run's workers.
            lease: Seconds a claim holds without a touch.
            heartbeat: Seconds between two touches of every claim held.
            clock: Seconds since the epoch, which a claim's mtime is compared with.
            on_beat: Called after each touch, from the heartbeat thread - where the worker says it is alive.
        """
        self.claims = pool_dir(run_dir) / Pool.CLAIMS
        self.worker = worker
        self.lease = lease
        self.heartbeat = heartbeat
        self.clock = clock
        self.on_beat = on_beat
        self.held: dict[Unit, Path] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def create(self, unit: Unit, epoch: int) -> bool:
        """Try to create the claim file of one epoch of a unit, which exactly one worker is told it did."""
        path = self.claims / unit.task / f"{unit.name}.{epoch}"
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        body = {"worker": self.worker, "host": socket.gethostname(), "claimed": io.utc_now()}
        os.write(handle, json.dumps(body).encode())
        os.close(handle)
        with self.lock:
            self.held[unit] = path
        return True

    def claim(self, candidates: list[Unit], limit: int) -> list[Unit]:
        """Claim up to `limit` of the candidates, in order, skipping any another worker holds.

        Args:
            candidates: Units not done, in the order they should be taken.
            limit: How many to claim.

        Returns:
            The units claimed, in candidate order.
        """
        taken: list[Unit] = []
        newest: dict[str, dict[str, tuple[int, float]]] = {}
        for unit in candidates:
            if len(taken) >= limit:
                break
            if unit in self.held:
                continue
            if unit.task not in newest:
                (self.claims / unit.task).mkdir(parents=True, exist_ok=True)
                newest[unit.task] = newest_claims(self.claims, unit.task)
            epoch = 0
            seen = newest[unit.task].get(unit.name)
            if seen is not None:
                epoch, touched = seen
                if self.clock() - touched < self.lease:
                    continue
                epoch += 1
            if self.create(unit, epoch):
                taken.append(unit)
        return taken

    def touch(self) -> None:
        """Refresh the lease of every claim held, and say the worker is alive."""
        with self.lock:
            paths = list(self.held.values())
        for path in paths:
            try:
                os.utime(path)
            except FileNotFoundError:
                continue
        if self.on_beat is not None:
            self.on_beat()

    def start(self) -> None:
        """Start the thread that touches the claims every `heartbeat` seconds."""
        self.stopping.clear()
        self.thread = threading.Thread(target=self.beat, daemon=True)
        self.thread.start()

    def beat(self) -> None:
        """Touch the claims until stopped."""
        while not self.stopping.wait(self.heartbeat):
            self.touch()

    def stop(self) -> None:
        """Stop the heartbeat and give up every claim still held, so the next worker takes them without waiting."""
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            units = list(self.held)
        for unit in units:
            self.release(unit)

    def finish(self, unit: Unit) -> None:
        """Stop keeping a unit's claim, whose row is on disk and so makes the claim moot."""
        with self.lock:
            self.held.pop(unit, None)

    def release(self, unit: Unit) -> None:
        """Give a unit up - its claim looks untouched since the epoch, so the next worker takes it at once."""
        with self.lock:
            path = self.held.pop(unit, None)
        if path is None:
            return
        try:
            os.utime(path, (0, 0))
        except FileNotFoundError:
            return


def write_worker(run_dir: str | os.PathLike[str], worker: str, body: dict[str, Any]) -> None:
    """Write what a worker says of itself, replacing what it said last."""
    io.write_json(worker_file(run_dir, worker), body, indent=1)


def read_workers(run_dir: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Return what every worker of a run said of itself last, by worker name."""
    found: dict[str, dict[str, Any]] = {}
    directory = pool_dir(run_dir) / Pool.WORKERS
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        body = io.read_json(path, None)
        if body:
            found[path.stem] = body
    return found


def live_workers(run_dir: str | os.PathLike[str], within: float, now: float | None = None) -> list[str]:
    """Return the workers of a run whose heartbeat is no older than `within` seconds."""
    now = time.time() if now is None else now
    return [
        worker
        for worker, body in read_workers(run_dir).items()
        if now - body.get("heartbeat", 0.0) <= within and not body.get("ended")
    ]


def publish_once(path: Path, body: dict[str, Any]) -> bool:
    """Write a json file unless one is there, whole and atomically - the first writer's content stands.

    The body is written under a name of its own and linked into place, and a link fails when its target
    exists, so a reader never meets a file half written and two writers never both succeed.

    Args:
        path: Where the file goes.
        body: What it holds.

    Returns:
        True when this call wrote it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    scratch.write_text(json.dumps(body, indent=1) + "\n")
    try:
        os.link(scratch, path)
    except FileExistsError:
        return False
    finally:
        scratch.unlink(missing_ok=True)
    return True
