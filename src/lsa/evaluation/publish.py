"""Copy runs from a checkout's results into the shared evaluations root, where teammates and the viewer read.

    lsa evaluate publish results/accuracy/<run id> [<run dir> ...]            # dry run
    lsa evaluate publish results/accuracy/<run id> --apply
    lsa evaluate publish --sync results [<results root> ...] --apply          # every run worth showing, kept current

A run writes into its checkout's `results/` while its name, tag and note are still being settled, and is
published once they are - so the shared root only ever holds what someone meant to show. A checkout is not
somewhere a run can stay, though: a worktree is removed with everything under it.

A complete run is published as it is - its status, as [status](status.py) works it out for the viewer, from its
launch log, slurm and its record. A partial one, with a task unscored or a context unfinished, is published
only with `--partial`.

A run still going is published only with `--in-progress`, as a snapshot: its record read once and written
whole, every other file copied as it stands, a jsonl cut at its last whole line, and the record's
`run.snapshot` saying when it was taken and what slurm said of the job. The viewer lists a snapshot as in
progress whatever slurm says by then, until a later publish replaces it.

A run is built beside its target and renamed into place, so a reader never sees half of one. One already
published is refused, unless `--replace` says to refresh it - the new copy swapped in for the old. A run
`lsa records join` wrote, under its base's id or its own, replaces each of its parts published under its own id:
those are moved, never deleted, into the archive beside the root, `<root>/../archive/<root name>/<kind dir>/`.
So is the run a renamed one was renamed from (`lsa records rename`), named in its `run.renamed_from`.
A run that a published joined run holds as a part is not published on its own again - join it again instead.

`--sync` publishes every run under the results roots it is given that is complete, partial or in progress,
refreshing what is published already and passing over a run that has not changed since, so one command keeps
the shared root current. Every command is a dry run until `--apply`.
"""

import argparse
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lsa import io, paths
from lsa.evaluation import launches
from lsa.evaluation.constants import Evaluation, Kernels, Pool, Production, Records, Status
from lsa.evaluation.runs import run_directory
from lsa.evaluation.status import MISSING, status

# What a run's record says it is, to the kind whose directory it goes in
KIND_OF_SCHEMA: dict[str, str] = {
    Records.SCHEMA: Evaluation.ACCURACY,
    Kernels.SCHEMA: Evaluation.KERNELS,
    Production.SCHEMA: Evaluation.PRODUCTION,
}

# The statuses a run is published in, and the flag each needs beyond a plain publish
PUBLISHED: tuple[str, ...] = (Status.COMPLETE, Status.PARTIAL, Status.IN_PROGRESS)


@dataclass
class Publication:
    """What publishing one run would do - the record it writes, where, and what it replaces."""

    run: Path
    record: dict[str, Any]
    kind: str
    status: str
    why: str
    target: Path
    # Published under its own id already - refreshed in place
    refresh: bool = False
    # Published runs of other ids it replaces, moved to the archive
    archived: list[Path] = field(default_factory=list)


def read_kind(run: Path) -> tuple[dict[str, Any], str]:
    """Return a run's record, read once, and the kind it is published as.

    Raises:
        FileNotFoundError: The run has no record.
        ValueError: The record is of no kind this publishes.
    """
    record_path = run / Records.RECORD
    if not record_path.is_file():
        raise FileNotFoundError(f"{run} holds no record.json")
    record = io.read_json(record_path)
    kind = KIND_OF_SCHEMA.get(record.get("schema", ""))
    if kind is None:
        raise ValueError(f"{run}: a {record.get('schema')!r} record is not one this publishes")
    return record, kind


def parts_of(record: dict[str, Any]) -> list[str]:
    """Return the ids of the runs a joined record was joined from, none for a run that was not."""
    return [str(part["id"]) for part in (record.get("run") or {}).get("parts") or []]


def snapshot_of(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return what a record says of the snapshot it is, None for a record that is not one."""
    return (record.get("run") or {}).get("snapshot") or None


def run_status(run: Path, missing: str) -> tuple[str, str]:
    """Return a run's status and what it rests on, as the viewer works it out.

    Args:
        run: The run directory.
        missing: What its record says is left to do, or empty.

    Returns:
        The status, and where it came from.
    """
    logged = launches.read(run)
    states = {job: row["State"] for job, row in launches.accounting(launches.open_jobs(logged)).items()}
    ended, why = status(logged, missing, states, (run / Records.META).is_file())
    return ended or Status.UNKNOWN, why or "no launch log"


def snapshot_marker(run: Path, why: str) -> dict[str, Any]:
    """Return what a snapshot of a run still going says of itself - when, of which job, in what state, from where."""
    job = launches.unsettled(launches.read(run))
    return {"taken": io.utc_now(), "job": job, "state": why, "source": str(run)}


def joined_holding(run_id: str, published: Path) -> Path | None:
    """Return the published joined run that holds a run as one of its parts, or None."""
    if not published.is_dir():
        return None
    for candidate in sorted(published.iterdir()):
        record = io.read_json(candidate / Records.RECORD, None)
        if record and run_id in parts_of(record):
            return candidate
    return None


def plan_publication(
    run: Path, root: Path, partial: bool = False, in_progress: bool = False, replace: bool = False
) -> Publication:
    """Work out what publishing one run would do, refusing what may not be published.

    Args:
        run: The run directory.
        root: The evaluations root it is published into.
        partial: Publish a run that is partial.
        in_progress: Publish a snapshot of a run still going - partial ones too.
        replace: Refresh a run published already, and archive the parts a joined run replaces.

    Returns:
        The publication.

    Raises:
        FileNotFoundError: The run has no record.
        ValueError: The record is of no kind this publishes, the run is in a status not asked for, or a
            published joined run holds it as a part.
        FileExistsError: The run, or a part of a joined run, is published already and `replace` was not given.
    """
    record, kind = read_kind(run)
    ended, why = run_status(run, MISSING[kind](record))
    joined_snapshot = snapshot_of(record)
    if joined_snapshot:
        ended, why = Status.IN_PROGRESS, f"joined from runs still going, as of {joined_snapshot['taken']}"
    if ended == Status.IN_PROGRESS and not in_progress:
        raise ValueError(f"{run} is {ended} - {why} - pass --in-progress to publish a snapshot of it")
    if ended == Status.PARTIAL and not (partial or in_progress):
        raise ValueError(f"{run} is {Status.PARTIAL} - {why} - pass --partial to publish it as it is")
    if ended not in PUBLISHED:
        raise ValueError(f"{run} is {ended} - {why}")
    published = root.joinpath(*Evaluation.DIRS[kind])
    holder = joined_holding(run.name, published)
    if holder and not parts_of(record):
        raise ValueError(
            f"{run.name} is published as a part of {holder.name} - join it again and publish that"
        )
    target = published / run.name
    archived = [
        published / name for name in parts_of(record) if name != run.name and (published / name).exists()
    ]
    # A joined run over its base published alone is another run under the same id - the base is archived too
    published_record = io.read_json(target / Records.RECORD, None) if target.exists() else None
    if published_record and parts_of(record) and not parts_of(published_record):
        archived.insert(0, target)
    # A renamed run replaces the run it was renamed from
    renamed_from = published / str((record.get("run") or {}).get("renamed_from") or run.name)
    if renamed_from != target and renamed_from.exists() and renamed_from not in archived:
        archived.append(renamed_from)
    taken = [path for path in [target, *archived] if path.exists()]
    if taken and not replace:
        names = ", ".join(str(path) for path in taken)
        raise FileExistsError(
            f"{names} already published - pass --replace to refresh it and archive the parts"
        )
    if ended == Status.IN_PROGRESS and not snapshot_of(record):
        record = {**record, "run": {**record["run"], "snapshot": snapshot_marker(run, why)}}
    return Publication(
        run, record, kind, ended, why, target, target.exists() and target not in archived, archived
    )


def unchanged(publication: Publication) -> bool:
    """Return whether the published copy is what publishing would write - the same record, a snapshot's time
    aside, and every other file the same size as its copy would be."""
    if not publication.refresh:
        return False
    published = io.read_json(publication.target / Records.RECORD, None)
    if not published:
        return False

    def bare(record: dict[str, Any]) -> dict[str, Any]:
        snapshot = snapshot_of(record) or {}
        moment = {key: value for key, value in snapshot.items() if key not in ("taken", "state")}
        return {**record, "run": {**record["run"], "snapshot": moment}}

    return bare(published) == bare(publication.record) and file_sizes(
        publication.run, copied=True
    ) == file_sizes(publication.target)


def run_files(run: Path) -> list[Path]:
    """Return the files of a run directory a copy of it holds - everything but its record and its worker pool.

    The pool directory is the queue several workers shared and the raw shards they wrote. The graded sample
    files under `samples/` hold every one of those rows, so a copy of the shards would be the data twice and
    a stale claim file for a reader to wonder about.
    """
    return [
        path
        for path in sorted(run.rglob("*"))
        if path.is_file()
        and path.relative_to(run) != Path(Records.RECORD)
        and path.relative_to(run).parts[0] != Pool.DIR
    ]


def file_sizes(run: Path, copied: bool = False) -> dict[Path, int]:
    """Return every file of a run directory but its record, by path, with its size - as copied, when asked."""
    sizes = {}
    for path in run_files(run):
        relative = path.relative_to(run)
        size = path.stat().st_size
        if copied and path.suffix == ".jsonl":
            content = path.read_bytes()
            size = content.rfind(b"\n") + 1
        sizes[relative] = size
    return sizes


def copy_consistent(source: Path, destination: Path) -> None:
    """Copy a run's file as it stands - a jsonl a job may be appending to cut at its last whole line."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix != ".jsonl":
        shutil.copy2(source, destination)
        return
    content = source.read_bytes()
    destination.write_bytes(content[: content.rfind(b"\n") + 1])


def build_beside(target: Path) -> Path:
    """Return a fresh scratch directory to build a run directory in and rename into place.

    It sits a level above the run's own directory, on the same filesystem, so a reader listing that directory
    never meets a run half built.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".building", dir=target.parent.parent))


def swap_in(built: Path, target: Path) -> None:
    """Rename a built directory into place, swapping out the copy of the same run that was there."""
    built.chmod(0o2775)
    if not target.exists():
        built.rename(target)
        return
    old = built.with_name(f"{built.name}.old")
    target.rename(old)
    built.rename(target)
    shutil.rmtree(old)


def write_publication(publication: Publication) -> None:
    """Copy a run into its target, its record as read once, built beside the target and swapped in."""
    built = build_beside(publication.target)
    try:
        for source in run_files(publication.run):
            copy_consistent(source, built / source.relative_to(publication.run))
        io.write_json(built / Records.RECORD, publication.record, indent=1)
        swap_in(built, publication.target)
    finally:
        if built.exists():
            shutil.rmtree(built)


def archive_dir(root: Path, kind: str) -> Path:
    """Return where a replaced run of one kind is moved - `archive/evaluations` beside the root, nested as the root is."""
    return root.parent.joinpath("archive", paths.evaluations().name, *Evaluation.DIRS[kind])


def archived_path(published: Path, archive: Path) -> Path:
    """Return where a replaced run lands in the archive - under its own name, stamped when that is taken."""
    landing = archive / published.name
    if not landing.exists():
        return landing
    return archive / f"{published.name}.replaced-{datetime.now(UTC).strftime(Records.RUN_STAMP)}"


def carry_out(publication: Publication, root: Path, apply: bool) -> None:
    """Print what one publication does, and do it when asked - archive what it replaces, then swap it in."""
    archive = archive_dir(root, publication.kind)
    for published in publication.archived:
        landing = archived_path(published, archive)
        print(f"  archive {published} -> {landing}", flush=True)
        if apply:
            landing.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(published, landing)
    verb = "refresh" if publication.refresh else "publish"
    snapshot = " (snapshot)" if publication.status == Status.IN_PROGRESS else ""
    print(f"{verb} {publication.run} -> {publication.target}  [{publication.status}{snapshot}]", flush=True)
    if apply:
        write_publication(publication)


def publish(
    runs: list[Path],
    root: Path,
    apply: bool,
    partial: bool = False,
    replace: bool = False,
    in_progress: bool = False,
) -> list[Path]:
    """Publish each run, after checking every one of them, so one refusal publishes nothing.

    Args:
        runs: The run directories.
        root: The evaluations root.
        apply: Copy, rather than print what would be copied.
        partial: Publish runs that are partial.
        replace: Refresh runs published already, and archive the parts a joined run replaces.
        in_progress: Publish snapshots of runs still going.

    Returns:
        Where each run went, or would go.
    """
    publications = [plan_publication(run, root, partial, in_progress, replace) for run in runs]
    for publication in publications:
        carry_out(publication, root, apply)
    if not apply:
        print("dry run - pass --apply to copy", flush=True)
    return [publication.target for publication in publications]


def results_runs(results_roots: list[Path]) -> list[Path]:
    """Return every run directory under the results roots, each kind's directory in turn."""
    runs = []
    for results in results_roots:
        for kind in Evaluation.KINDS:
            directory = results.joinpath(*Evaluation.DIRS[kind])
            if directory.is_dir():
                runs += [path for path in sorted(directory.iterdir()) if (path / Records.RECORD).is_file()]
    return runs


def sync(results_roots: list[Path], root: Path, apply: bool) -> list[Path]:
    """Publish or refresh every run under some results roots that is complete, partial or in progress.

    A run that cannot be published is named with the reason and passed over, and so is one whose published copy
    already holds its record.

    Args:
        results_roots: Checkouts' results roots, each holding `accuracy/` and `performance/`.
        root: The evaluations root.
        apply: Copy, rather than print what would be copied.

    Returns:
        Where each run published or refreshed went, or would go.
    """
    targets = []
    for run in results_runs(results_roots):
        try:
            publication = plan_publication(run, root, partial=True, in_progress=True, replace=True)
        except (ValueError, FileExistsError, FileNotFoundError, KeyError) as refusal:
            print(f"skip {run} - {refusal}", flush=True)
            continue
        if unchanged(publication):
            print(f"same {run} - published as it is", flush=True)
            continue
        carry_out(publication, root, apply)
        targets.append(publication.target)
    if not apply:
        print("dry run - pass --apply to copy", flush=True)
    return targets


def main() -> None:
    """Publish runs to the shared evaluations root."""
    parser = argparse.ArgumentParser(description="Copy runs into the shared evaluations root")
    parser.add_argument(
        "runs",
        nargs="*",
        type=run_directory,
        help="run directories under a checkout's results, or their record.json",
    )
    parser.add_argument(
        "--sync",
        nargs="+",
        type=Path,
        default=None,
        metavar="RESULTS",
        help="publish or refresh every complete, partial or in-progress run under these results roots",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="the evaluations root, the shared one by default"
    )
    parser.add_argument("--partial", action="store_true", help="publish runs that are partial")
    parser.add_argument(
        "--in-progress", action="store_true", help="publish snapshots of runs still going, partial ones too"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="refresh runs published already, and archive the parts a joined run replaces",
    )
    parser.add_argument("--apply", action="store_true", help="copy, rather than print what would be copied")
    args = parser.parse_args()
    root = args.root or paths.evaluations()
    if args.sync:
        sync([results.resolve() for results in args.sync], root, args.apply)
        return
    if not args.runs:
        parser.error("name run directories, or --sync results roots")
    runs = [run.resolve() for run in args.runs]
    publish(runs, root, args.apply, args.partial, args.replace, args.in_progress)
