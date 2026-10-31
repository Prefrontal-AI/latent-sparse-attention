"""Join the runs one configuration was split across into one run, which holds every task any of them scored.

    lsa records join <base run dir> <extension run dir> [...] --into results/joined/accuracy    # dry run
    lsa records join <base run dir> <extension run dir> [...] --into results/joined/accuracy --apply
    lsa records join ... --into results/joined/accuracy --in-progress --replace --apply         # refresh a snapshot

A configuration's accuracy is sometimes scored by several runs, each over the lengths its server could hold - a
`full` run to 2M, beside runs at 5M and 10M on a server that holds 10M. The joined run is the base run with the
others' tasks added. It keeps the base's id, so it lands at `<dir>/<base id>`, and it replaces the base wherever
the base was published - `lsa evaluate publish --replace` archives the parts published on their own.

Parts join only when they are one measurement split by task. They share the model, the indexer and its
settings and map, the mesh, the suite, the sampling each benchmark was generated with and every server setting
but the serving knobs in `Records.SERVING_KNOBS`, whose values each part keeps in `run.parts`. A task an
extension scored and another part planned was planned the same way in both. No two parts hold the same task,
and an extension scored at least one - the rule the viewer extends a picked run by. A part whose last launch
did not end well is not joined, and nor is one still going unless `--in-progress` asks for a snapshot: each
part's record as it stands, the joined record's `run.snapshot` saying when and which parts were still going.

A table's accuracy is assembled instead from runs that are not one measurement - a suite's sets scored on
whatever image and mesh served them, a benchmark added later in a run of its own. `--assemble <suite>` names the
suite the joined record is of, and lets the parts differ in mesh, suite and every server setting, each part's
own kept in `run.parts`, and in the indexer's serving choices that read the same keys either way. They still
share the model, the indexer's design and map, and the sampling and plan of every task they hold. `--supersede <task or benchmark>` lets a later part's tasks stand in for an
earlier part's, and `--drop <task or benchmark>` leaves tasks out of the joined record - in either case the
left-out tasks' samples stay under `parts/<part id>/` and the part's summary names them. `--id <run id>` gives
the joined run an id of its own rather than the base's, as when the parts' names predate the one the run goes by.

`--mixed-early-budget` lets the parts differ in the indexer's early budget - a later arm that reads fewer keys
before a position, beside an earlier one that did not - and in no other indexer setting. Each task keeps the
indexer settings of the server that scored it, the record's own indexer names the early budget of the last part
that sets one, and `run.parts` keeps each part's. The parts that set an early budget set the same one.

Performance records - kernels and production - join by context instead, see [the performance join](../../performance/join.py).
`--mixed-early-budget` works there too, and `--across-harness` and `--contexts` are theirs alone.

The joined directory holds the joined record, every part's samples under `samples/`, each part's other files -
its launch log, metadata, server info and log - under `parts/<part id>/`, and a launch log of every part's
launches with the join's own launch last. So the joined run's status is the record's: complete once every task
any part planned is scored, and in progress while it is a snapshot. It is built beside its target and renamed
into place, and `--replace` swaps a rebuilt one in for the last. Nothing under a part's directory is written.
"""

import argparse
import copy
import getpass
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation import gpu, launches
from lsa.evaluation.accuracy.records import mesh_label_of
from lsa.evaluation.constants import Evaluation, Launches, Records, Status
from lsa.evaluation.publish import (
    KIND_OF_SCHEMA,
    build_beside,
    copy_consistent,
    run_files,
    run_status,
    swap_in,
)
from lsa.evaluation.runs import run_directory
from lsa.evaluation.status import MISSING, accuracy_missing

# The server settings every part has to agree on - the serving knobs and the model's path left out, the model
# being checked by name
MATCHED_SETTINGS: tuple[str, ...] = (
    *(key for key in Records.SERVER_KEYS if key not in Records.SERVING_KNOBS and key != "model_path"),
    "indexer",
)

# The indexer's serving choices that read the same keys either way - an assembly lets them differ
INDEXER_SERVING_KNOBS: tuple[str, ...] = (
    "kernels",
    "decode_context_limits",
    "prefix_reuse",
    "spare_contexts",
)

# The one indexer setting `--mixed-early-budget` lets parts differ in - the early budget a later arm adds
EARLY_BUDGET = "early_budget"


@dataclass
class Part:
    """One run that goes into a join - its record as joining reads it, and what it rests on."""

    run_dir: Path
    record: dict[str, Any]
    status: str
    status_why: str
    launches: list[dict[str, Any]]
    left_out: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        """Return the part's run id."""
        return str(self.record["run"]["id"])


@dataclass
class Join:
    """What joining some runs would write, and why it would not."""

    parts: list[Part]
    target: Path
    # The suite an assembly is of, empty for a join
    assemble: str = ""
    # Whether the parts may differ in their indexer's early budget
    mixed_early_budget: bool = False
    record: dict[str, Any] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)
    # Source to destination, the destination relative to the joined directory
    copies: list[tuple[Path, Path]] = field(default_factory=list)
    launch_lines: list[dict[str, Any]] = field(default_factory=list)
    status: str = ""
    missing: str = ""


def read_part(run_dir: Path) -> Part:
    """Read one run as a part of a join.

    Args:
        run_dir: The run directory.

    Returns:
        The part.

    Raises:
        FileNotFoundError: The directory holds no record.
        ValueError: The record is of a run no server ran.
    """
    if not (run_dir / Records.RECORD).is_file():
        raise FileNotFoundError(f"{run_dir} holds no {Records.RECORD}")
    record = io.read_json(run_dir / Records.RECORD)
    if "server" in record and record["server"] is None:
        raise ValueError(f"{run_dir} was not served, so it holds no server to join on")
    ended, why = run_status(run_dir, missing_of(record))
    return Part(run_dir, record, ended, why, launches.read(run_dir))


def missing_of(record: dict[str, Any]) -> str:
    """Say what a record of any kind has left to do, or nothing for a record of no kind."""
    if "tasks" in record:
        return accuracy_missing(record)
    kind = KIND_OF_SCHEMA.get(record.get("schema", ""))
    return MISSING[kind](record) if kind else ""


def server_entries(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return every server a record names - its first under `server`, then each under `servers`."""
    entries = [("server", record.get("server") or {})]
    entries += [(f"servers.{server_id}", entry) for server_id, entry in (record.get("servers") or {}).items()]
    return entries


def benchmark_of(task: str, record: dict[str, Any]) -> str:
    """Return the benchmark a task of a record belongs to - its entry's, or the generation key it is named after."""
    entry = record["tasks"].get(task) or {}
    if entry.get("benchmark"):
        return str(entry["benchmark"])
    named = [
        key for key in record["run"].get("generation") or {} if task == key or task.startswith(key + "_")
    ]
    return max(named, key=len, default=task)


def is_selected(task: str, record: dict[str, Any], names: tuple[str, ...]) -> bool:
    """Return whether a task is named, by itself or by its benchmark."""
    return task in names or benchmark_of(task, record) in names


def trimmed(part: Part, left_out: set[str]) -> Part:
    """Return a part without some tasks - their scores, plans and skips, and the sampling of a benchmark it then
    plans no task of."""
    record = copy.deepcopy(part.record)
    run = record["run"]
    kept_plans = [plan for plan in run.get("plans") or [] if plan["task"] not in left_out]
    benchmarks = {benchmark_of(plan["task"], record) for plan in kept_plans}
    benchmarks |= {benchmark_of(task, record) for task in record["tasks"] if task not in left_out}
    run["generation"] = {
        key: value for key, value in (run.get("generation") or {}).items() if key in benchmarks
    }
    run["plans"] = kept_plans
    record["tasks"] = {task: entry for task, entry in record["tasks"].items() if task not in left_out}
    record["skipped"] = {
        task: why for task, why in (record.get("skipped") or {}).items() if task not in left_out
    }
    held = set(part.record["tasks"]) | {plan["task"] for plan in part.record["run"].get("plans") or []}
    return Part(
        part.run_dir,
        record,
        part.status,
        part.status_why,
        part.launches,
        sorted(held & left_out),
    )


def trim_parts(parts: list[Part], drop: tuple[str, ...], supersede: tuple[str, ...]) -> list[Part]:
    """Return the parts with the dropped tasks left out of each, and a superseded task left out of every part but
    the last that holds it.

    Args:
        parts: The base first, then the extensions.
        drop: Tasks or benchmarks no part keeps.
        supersede: Tasks or benchmarks the last part holding them keeps.

    Returns:
        The parts, each without what it gives up.
    """
    last_holder: dict[str, int] = {}
    for index, part in enumerate(parts):
        for task in part.record["tasks"]:
            if is_selected(task, part.record, supersede):
                last_holder[task] = index
    result = []
    for index, part in enumerate(parts):
        named = set(part.record["tasks"]) | {plan["task"] for plan in part.record["run"].get("plans") or []}
        left_out = {task for task in named if is_selected(task, part.record, drop)}
        left_out |= {task for task in named if last_holder.get(task, index) > index}
        result.append(trimmed(part, left_out) if left_out else part)
    return result


def design_of(
    config: dict[str, Any] | None, assemble: bool, mixed_early_budget: bool = False
) -> dict[str, Any] | None:
    """Return the indexer settings two parts must share - all of them, or in an assembly all but the serving
    choices that read the same keys either way, and with mixed early budgets all but the early budget."""
    if config is None:
        return config
    served = INDEXER_SERVING_KNOBS if assemble else ()
    left_out = (*served, *((EARLY_BUDGET,) if mixed_early_budget else ()))
    return {key: value for key, value in config.items() if key not in left_out}


def early_budget_of(part: Part) -> dict[str, Any] | None:
    """Return the early budget a part's indexer was set with, None for none."""
    return (part.record["attention"].get("config") or {}).get(EARLY_BUDGET)


def last_early_budget_part(parts: list[Part]) -> Part | None:
    """Return the last part whose indexer sets an early budget, None when no part does."""
    return next((part for part in reversed(parts) if early_budget_of(part)), None)


def early_budget_refusals(parts: list[Part]) -> list[str]:
    """Say where the parts that set an early budget do not set the same one."""
    budgets = {part.run_id: early_budget_of(part) for part in parts if early_budget_of(part)}
    if len({json.dumps(budget, sort_keys=True) for budget in budgets.values()}) < 2:
        return []
    return [f"early budgets differ between parts: {budgets}"]


def plan_protocol(plan: dict[str, Any]) -> dict[str, Any]:
    """Return what two plans of one task must share to be one measurement - all but how many ran at once.

    A sampling key that is unset reads the same whether it was written as null, an empty dict or not at all.
    """
    sampling = {
        key: value for key, value in (plan.get("sampling") or {}).items() if value not in (None, {}, [])
    }
    kept = {key: value for key, value in plan.items() if key not in ("task", "concurrency", "sampling")}
    return {**kept, "sampling": sampling}


def unset_dropped(sampling: dict[str, Any] | None) -> dict[str, Any]:
    """Return a sampling with its unset keys dropped."""
    return {key: value for key, value in (sampling or {}).items() if value not in (None, {}, [])}


def identity_refusals(
    base: Part, part: Part, assemble: bool = False, mixed_early_budget: bool = False
) -> list[str]:
    """Say where a part was not served as the base was - card, model, indexer, settings, map, mesh, suite, generation.

    An assembly lets the mesh and the suite differ, and never the card.
    Mixed early budgets let the early budget differ, and no other indexer setting.
    """
    refusals = gpu.refusals(base.record, part.record)
    mine, theirs = base.record, part.record
    for key in ("name", "label"):
        if mine["model"].get(key) != theirs["model"].get(key):
            refusals.append(f"model {key} {mine['model'].get(key)!r} vs {theirs['model'].get(key)!r}")
    kinds = [record["attention"].get("kind") for record in (mine, theirs)]
    if all(kinds) and kinds[0] != kinds[1]:
        refusals.append(f"attention kind {kinds[0]} vs {kinds[1]}")
    if mine["attention"].get("indexer") != theirs["attention"].get("indexer"):
        refusals.append(f"indexer {mine['attention'].get('indexer')} vs {theirs['attention'].get('indexer')}")
    if design_of(mine["attention"].get("config"), assemble, mixed_early_budget) != design_of(
        theirs["attention"].get("config"), assemble, mixed_early_budget
    ):
        refusals.append("indexer settings differ")
    map_sha = [((record["attention"].get("map") or {}).get("sha256")) for record in (mine, theirs)]
    if map_sha[0] != map_sha[1]:
        refusals.append(f"map {map_sha[0]} vs {map_sha[1]}")
    labels = {mesh_label_of(entry) for _, entry in server_entries(theirs)}
    if not assemble and labels != {mesh_label_of(mine["server"])}:
        refusals.append(f"mesh {mesh_label_of(mine['server'])} vs {', '.join(sorted(labels))}")
    if not assemble and mine["run"].get("suite") != theirs["run"].get("suite"):
        refusals.append(f"suite {mine['run'].get('suite')} vs {theirs['run'].get('suite')}")
    generation = [record["run"].get("generation") or {} for record in (mine, theirs)]
    for benchmark in sorted(set(generation[0]) & set(generation[1])):
        if unset_dropped(generation[0][benchmark]) != unset_dropped(generation[1][benchmark]):
            refusals.append(f"{benchmark} generated under other sampling")
    return refusals


def setting_refusals(
    base: Part, part: Part, assemble: bool = False, mixed_early_budget: bool = False
) -> list[str]:
    """Say which server setting a part's servers set otherwise than the base's first server did.

    A setting only one of the two recorded is passed over - an older record kept fewer of them.
    An assembly checks the indexer settings alone.
    """
    first = base.record["server"]
    refusals = []
    for where, entry in server_entries(part.record):
        if entry.get("config") is not None and design_of(
            entry["config"], assemble, mixed_early_budget
        ) != design_of(first.get("config"), assemble, mixed_early_budget):
            refusals.append(f"{where}: indexer settings differ from the base's")
        for key in () if assemble else MATCHED_SETTINGS:
            if key in first and key in entry and first[key] != entry[key]:
                refusals.append(f"{where}: {key} {entry[key]!r}, the base's {first[key]!r}")
    return refusals


def task_refusals(parts: list[Part]) -> list[str]:
    """Say where the parts are not one run split by task - a task held twice, an extension with nothing scored,
    or a task planned two ways."""
    refusals = []
    held = set(parts[0].record["tasks"])
    for part in parts[1:]:
        tasks = part.record["tasks"]
        twice = sorted(held & set(tasks))
        if twice:
            refusals.append(f"{part.run_id} holds tasks an earlier part does: {', '.join(twice)}")
        if not any(entry.get("score") is not None for entry in tasks.values()):
            refusals.append(f"{part.run_id} scored no task")
        held |= set(tasks)
    planned: dict[str, tuple[str, dict[str, Any]]] = {}
    for part in parts:
        for plan in part.record["run"].get("plans") or []:
            protocol = plan_protocol(plan)
            earlier = planned.setdefault(plan["task"], (part.run_id, protocol))
            if earlier[1] != protocol:
                refusals.append(
                    f"{plan['task']} planned as {earlier[1]} in {earlier[0]} and {protocol} in {part.run_id}"
                )
    return refusals


def target_refusals(join: Join, replace: bool, run_id: str) -> list[str]:
    """Say where the joined directory cannot go - somewhere taken, a part's own, or under an id that is no name."""
    refusals = []
    if join.target.exists() and not replace:
        refusals.append(f"{join.target} already exists - pass --replace to rebuild it")
    if join.target.resolve() in {part.run_dir.resolve() for part in join.parts}:
        refusals.append(f"{join.target} is a part's own directory")
    if run_id and Path(run_id).name != run_id:
        refusals.append(f"--id {run_id!r} is not a plain directory name")
    return refusals


def status_refusals(parts: list[Part], in_progress: bool = False) -> list[str]:
    """Say which part did not end well - a complete or a partial run is joined, and one still going if asked."""
    joined = (Status.COMPLETE, Status.PARTIAL, *([Status.IN_PROGRESS] if in_progress else []))
    hint = "" if in_progress else " - pass --in-progress to join a snapshot of it"
    return [
        f"{part.run_id} is {part.status or 'without a launch log'} - {part.status_why}"
        + (hint if part.status == Status.IN_PROGRESS else "")
        for part in parts
        if part.status not in joined
    ]


def join_snapshot(parts: list[Part]) -> dict[str, Any] | None:
    """Return what a join of parts still going says of itself - when, and each such part's job state - or None."""
    going = {part.run_id: part.status_why for part in parts if part.status == Status.IN_PROGRESS}
    if not going:
        return None
    return {"taken": io.utc_now(), "state": "; ".join(f"{run_id}: {why}" for run_id, why in going.items())}


def serving_knobs(
    base: Part, part: Part, assemble: bool = False, mixed_early_budget: bool = False
) -> dict[str, dict[str, Any]]:
    """Return each server of a part with the serving knobs it set otherwise than the base's first server - every
    server setting, in an assembly, and the early budget when budgets are mixed."""
    first = base.record["server"]
    differing = {}
    # The first server is one of `servers` too, in a record that names them
    entries = server_entries(part.record)
    for where, entry in entries[1:] or entries:
        keys = (*Records.SERVING_KNOBS, *MATCHED_SETTINGS) if assemble else Records.SERVING_KNOBS
        knobs = {key: entry[key] for key in keys if key in entry and entry[key] != first.get(key)}
        mine, theirs = first.get("config") or {}, entry.get("config") or {}
        config_keys = (
            *(INDEXER_SERVING_KNOBS if assemble else ()),
            *((EARLY_BUDGET,) if mixed_early_budget else ()),
        )
        knobs |= {
            f"config.{key}": theirs[key]
            for key in config_keys
            if key in theirs and theirs[key] != mine.get(key)
        }
        if knobs:
            differing[where] = knobs
    return differing


def part_summary(
    base: Part, part: Part, assemble: bool = False, mixed_early_budget: bool = False
) -> dict[str, Any]:
    """Return what the joined record keeps of one part - where it came from, what it scored and how it was served."""
    run = part.record["run"]
    return {
        "id": part.run_id,
        "suite": run.get("suite"),
        "note": run.get("note"),
        "tag": run.get("tag"),
        "started": run.get("started"),
        "date": run.get("date"),
        "seconds": run.get("seconds"),
        "host": run.get("host"),
        "commits": run.get("commits"),
        "versions": run.get("versions"),
        "status": part.status,
        "mesh": {"label": mesh_label_of(part.record["server"])},
        "servers": sorted(part.record.get("servers") or {}),
        "jobs": [launch.get("job") for launch in part.launches],
        "tasks": sorted(part.record["tasks"]),
        "left_out": part.left_out,
        "serving": serving_knobs(base, part, assemble, mixed_early_budget),
        "parts": run.get("parts"),
    }


def joined_record(
    parts: list[Part], assemble: str = "", run_id: str = "", mixed_early_budget: bool = False
) -> dict[str, Any]:
    """Return the record of the joined run - the base's, with every part's tasks, plans, servers and history.

    Each task names the server that scored it and each server keeps its own indexer settings, so a task's
    indexer is its part's even where the parts' early budgets differ.
    The record's own indexer names the early budget of the last part that sets one.

    Args:
        parts: The base first, then the extensions.
        assemble: The suite an assembly is of, or empty for a join.
        run_id: The joined run's id, or empty for the base's.
        mixed_early_budget: The parts may differ in their early budget.

    Returns:
        The record.
    """
    base = parts[0]
    record = copy.deepcopy(base.record)
    newest = last_early_budget_part(parts)
    if mixed_early_budget and newest:
        record["attention"]["config"] = {
            **record["attention"]["config"],
            EARLY_BUDGET: early_budget_of(newest),
        }
    run = record["run"]
    plans = {plan["task"]: plan for plan in run.get("plans") or []}
    tasks = dict(record["tasks"])
    generation: dict[str, Any] = {}
    servers: dict[str, Any] = {}
    for part in parts:
        generation.update(part.record["run"].get("generation") or {})
        for plan in part.record["run"].get("plans") or []:
            if plan["task"] in part.record["tasks"] or plan["task"] not in plans:
                plans[plan["task"]] = plan
        tasks.update(part.record["tasks"])
        for server_id, entry in (part.record.get("servers") or {}).items():
            servers[server_id] = {**entry, "run": part.run_id}
    # A task one part skipped is still to come when another part plans it without skipping it
    pending = {
        plan["task"]
        for part in parts
        for plan in part.record["run"].get("plans") or []
        if plan["task"] not in (part.record.get("skipped") or {})
    }
    skipped = {
        task: why
        for part in parts
        for task, why in (part.record.get("skipped") or {}).items()
        if task not in tasks and task not in pending
    }
    run["id"] = run_id or base.run_id
    run["plans"] = list(plans.values())
    run["generation"] = generation
    if assemble:
        run["suite"] = assemble
    run["seconds"] = round(sum(part.record["run"].get("seconds") or 0.0 for part in parts), 1)
    run["date"] = max(str(part.record["run"].get("date") or "") for part in parts)
    run["parts"] = [part_summary(base, part, bool(assemble), mixed_early_budget) for part in parts]
    verb = f"assembled as {assemble}" if assemble else "joined"
    run["edits"] = [
        *(run.get("edits") or []),
        {
            "date": io.utc_now(),
            "who": getpass.getuser(),
            "what": f"{verb} {', '.join(part.run_id for part in parts[1:])} into {base.run_id}"
            + (f" as {run_id}" if run_id else "")
            + "".join(
                f"; {part.run_id} leaves out {', '.join(part.left_out)}" for part in parts if part.left_out
            ),
        },
    ]
    record["servers"] = servers
    record["tasks"] = tasks
    record["skipped"] = skipped
    return record


def planned_copies(parts: list[Part], mixed_early_budget: bool = False) -> list[tuple[Path, Path]]:
    """Return every file the joined directory takes from the parts, each with where it lands in it.

    A part's samples of a task its record holds go under `samples/`, and every other file of its directory -
    a samples file of a task it never finished included - under `parts/<part id>/`. The base's server info,
    generation and indexer settings are also kept at the top, where a reader of one run looks for them - the
    indexer settings of the last part that sets an early budget, when budgets are mixed.
    """
    copies: list[tuple[Path, Path]] = []
    for part in parts:
        own = {Path(entry["samples"]) for entry in part.record["tasks"].values() if entry.get("samples")}
        for source in run_files(part.run_dir):
            relative = source.relative_to(part.run_dir)
            landing = relative if relative in own else Path(Records.PARTS, part.run_id, relative)
            copies.append((source, landing))
    base = parts[0].run_dir
    newest = last_early_budget_part(parts)
    indexer_dir = newest.run_dir if mixed_early_budget and newest else base
    for name, directory in (
        (Records.SERVER_INFO, base),
        ("generation.json", base),
        ("indexer.json", indexer_dir),
    ):
        if (directory / name).is_file():
            copies.append((directory / name, Path(name)))
    return copies


def joined_launches(parts: list[Part]) -> list[dict[str, Any]]:
    """Return the joined run's launch log - every part's lines, each naming its part, then the join's own launch.

    The join's launch is last, so the joined run's status is what its record says: complete once every planned
    task is scored. The first line is the base's first launch, so the owner stays whoever started the base.
    """
    lines = []
    for part in parts:
        path = launches.locate_log(part.run_dir)
        if not path.is_file():
            continue
        # A log a job may be appending to - whole lines only
        text = path.read_text()
        whole = text[: text.rfind("\n") + 1].splitlines()
        lines += [{**json.loads(line), "run": part.run_id} for line in whole if line.strip()]
    start = {**launches.start_line(), "joined": [part.run_id for part in parts]}
    return [*lines, start, launches.end_line(start, Launches.COMPLETED)]


def plan_join(
    run_dirs: list[Path],
    into: Path,
    in_progress: bool = False,
    replace: bool = False,
    assemble: str = "",
    drop: tuple[str, ...] = (),
    supersede: tuple[str, ...] = (),
    run_id: str = "",
    mixed_early_budget: bool = False,
) -> Join:
    """Work out what joining runs would write and whether it may, writing nothing.

    Args:
        run_dirs: The base run directory first, then the runs that extend it.
        into: The directory the joined run directory goes in.
        in_progress: Join parts still going, as a snapshot of their records as they stand.
        replace: Rebuild a joined run written before.
        assemble: The suite to assemble the parts as, letting their mesh, suite and server settings differ.
        drop: Tasks or benchmarks no part keeps.
        supersede: Tasks or benchmarks the last part holding them keeps.
        run_id: The joined run's id, or empty for the base's.
        mixed_early_budget: Let the parts differ in the indexer's early budget and in no other indexer setting.

    Returns:
        The join - the record it would write, the files it would copy, and each reason it refuses.
    """
    parts = [read_part(run_dir) for run_dir in run_dirs]
    join = Join(
        parts=parts,
        target=into / (run_id or parts[0].run_id),
        assemble=assemble,
        mixed_early_budget=mixed_early_budget,
    )
    if any(part.record.get("schema") != Records.SCHEMA for part in parts):
        join.refusals = [
            f"{part.run_id}: not an accuracy record"
            for part in parts
            if part.record.get("schema") != Records.SCHEMA
        ]
        return join
    held = {
        name
        for part in parts
        for task in [*part.record["tasks"], *(plan["task"] for plan in part.record["run"].get("plans") or [])]
        for name in (task, benchmark_of(task, part.record))
    }
    join.refusals += [
        f"--drop or --supersede names {name}, which no part holds"
        for name in (*drop, *supersede)
        if name not in held
    ]
    parts = trim_parts(parts, drop, supersede)
    join.parts = parts
    if len({part.run_dir.resolve() for part in parts}) < len(parts):
        join.refusals.append("a run is named twice")
    for part in parts[1:]:
        join.refusals += [
            f"{part.run_id}: {refusal}"
            for refusal in identity_refusals(parts[0], part, bool(assemble), mixed_early_budget)
        ]
    for part in parts:
        join.refusals += [
            f"{part.run_id}: {refusal}"
            for refusal in setting_refusals(parts[0], part, bool(assemble), mixed_early_budget)
        ]
    if mixed_early_budget:
        join.refusals += early_budget_refusals(parts)
    join.refusals += task_refusals(parts)
    join.refusals += status_refusals(parts, in_progress)
    join.refusals += target_refusals(join, replace, run_id)
    join.record = joined_record(parts, assemble, run_id, mixed_early_budget)
    join.copies = planned_copies(parts, mixed_early_budget)
    join.launch_lines = joined_launches(parts)
    join.missing = accuracy_missing(join.record)
    join.status = Status.PARTIAL if join.missing else Status.COMPLETE
    snapshot = join_snapshot(parts)
    if snapshot:
        join.record["run"]["snapshot"] = snapshot
        join.status = Status.IN_PROGRESS
    return join


def describe(join: Join) -> str:
    """Return the lines a join prints - each part, the joined run, and every reason it refuses."""
    lines = []
    for index, part in enumerate(join.parts):
        role = "base" if index == 0 else "extends"
        scored = sum(1 for entry in part.record.get("tasks", {}).values() if entry.get("score") is not None)
        lines.append(f"{role:<8}{part.run_id}  [{part.status or 'no launch log'}]  {scored} tasks scored")
        if part.record.get("schema") != Records.SCHEMA:
            continue
        lines.append(f"          mesh {mesh_label_of(part.record['server'])}")
        jobs = [str(launch.get("job")) for launch in part.launches]
        lines.append(
            f"          jobs {', '.join(jobs) or '-'}  servers {', '.join(sorted(part.record.get('servers') or {})) or '-'}"
        )
        if part.left_out:
            lines.append(f"          leaves out {', '.join(part.left_out)}")
        if index:
            for where, knobs in serving_knobs(
                join.parts[0], part, bool(join.assemble), join.mixed_early_budget
            ).items():
                lines.append(
                    f"          {where} serves with {', '.join(f'{k} {v}' for k, v in knobs.items())}"
                )
    if join.record:
        planned = {plan["task"] for plan in join.record["run"]["plans"]}
        size = sum(source.stat().st_size if source.is_file() else 0 for source, _ in join.copies)
        lines.append(
            f"{'assembled' if join.assemble else 'joined'}  {join.target}  [{join.status}]  {len(join.record['tasks'])} tasks of {len(planned)} planned, "
            f"{len(join.copies)} files ({size / 2**20:.0f} MiB), {len(join.record['servers'])} servers"
        )
        if join.missing:
            lines.append(f"          {join.missing}")
    lines += [f"refused  {refusal}" for refusal in join.refusals]
    return "\n".join(lines)


def write_join(join: Join) -> None:
    """Write the joined run directory, built beside its target and swapped into place whole."""
    built = build_beside(join.target)
    try:
        for source, landing in join.copies:
            copy_consistent(source, built / landing)
        io.write_json(built / Records.RECORD, join.record, indent=1)
        with launches.locate_log(built).open("w") as log:
            log.writelines(json.dumps(line) + "\n" for line in join.launch_lines)
        swap_in(built, join.target)
    finally:
        if built.exists():
            shutil.rmtree(built)


def is_performance(run_dir: Path) -> bool:
    """Return whether a run directory holds a kernel or production record."""
    schema = (io.read_json(run_dir / Records.RECORD, None) or {}).get("schema", "")
    return KIND_OF_SCHEMA.get(schema, Evaluation.ACCURACY) != Evaluation.ACCURACY


def join_runs(
    run_dirs: list[Path],
    into: Path,
    apply: bool = False,
    in_progress: bool = False,
    replace: bool = False,
    assemble: str = "",
    drop: tuple[str, ...] = (),
    supersede: tuple[str, ...] = (),
    run_id: str = "",
    across_hosts: bool = False,
    mixed_early_budget: bool = False,
    across_harness: bool = False,
    contexts: tuple[str, ...] = (),
) -> Join:
    """Join runs into one, printing what it would write - nothing is written unless asked, or when it refuses.

    Args:
        run_dirs: The base run directory first, then the runs that extend it.
        into: The directory the joined run directory goes in.
        apply: Write it, rather than print what would be written.
        in_progress: Join parts still going, as a snapshot.
        replace: Rebuild a joined run written before.
        assemble: The suite to assemble the parts as, letting their mesh, suite and server settings differ.
        drop: Tasks or benchmarks no part keeps.
        supersede: Tasks or benchmarks the last part holding them keeps.
        run_id: The joined run's id, or empty for the base's.
        across_hosts: For performance records, let parts differ in host when their card model is the same.
        mixed_early_budget: Let the parts differ in the indexer's early budget and in no other indexer setting.
        across_harness: For performance records, let parts differ in the hashes of the harness files.
        contexts: For performance records, one `<run id>=<first>-<last>` per part, the contexts it supplies.

    Returns:
        The join.

    Raises:
        ValueError: Asked to write a join that refuses, or for accuracy records given a performance flag.
    """
    if not any(is_performance(run_dir) for run_dir in run_dirs) and (across_harness or contexts):
        raise ValueError("--across-harness and --contexts are for performance records")
    if any(is_performance(run_dir) for run_dir in run_dirs):
        # Imported here - the performance join builds on this module
        from lsa.evaluation.performance.join import join_performance

        return join_performance(
            run_dirs,
            into,
            apply,
            in_progress,
            replace,
            assemble,
            tuple(drop),
            tuple(supersede),
            run_id,
            across_hosts,
            mixed_early_budget,
            across_harness,
            tuple(contexts),
        )
    join = plan_join(
        run_dirs,
        into,
        in_progress,
        replace,
        assemble,
        tuple(drop),
        tuple(supersede),
        run_id,
        mixed_early_budget,
    )
    print(describe(join), flush=True)
    if not apply:
        print("dry run - pass --apply to write", flush=True)
        return join
    if join.refusals:
        raise ValueError(f"not joined - {len(join.refusals)} refusals above")
    write_join(join)
    print(f"wrote {join.target}", flush=True)
    return join


def main() -> None:
    """Join the runs named on the command line."""
    parser = argparse.ArgumentParser(
        description="Join the runs one configuration was split across into one run"
    )
    parser.add_argument(
        "base", type=run_directory, help="the run the others extend - the joined run keeps its id"
    )
    parser.add_argument(
        "extensions", nargs="+", type=run_directory, help="runs that scored tasks the base did not"
    )
    parser.add_argument("--into", type=Path, required=True, help="where the joined run directory goes")
    parser.add_argument("--in-progress", action="store_true", help="join parts still going, as a snapshot")
    parser.add_argument("--replace", action="store_true", help="rebuild a joined run written before")
    parser.add_argument(
        "--assemble",
        default="",
        metavar="SUITE",
        help="assemble parts served otherwise - mesh, suite, server settings - as one run of this suite",
    )
    parser.add_argument(
        "--drop", action="append", default=[], metavar="TASK", help="a task or benchmark no part keeps"
    )
    parser.add_argument(
        "--supersede",
        action="append",
        default=[],
        metavar="TASK",
        help="a task or benchmark - for performance records a context - the last part holding it keeps",
    )
    parser.add_argument(
        "--across-hosts",
        action="store_true",
        help="for performance records, join parts measured on different nodes of one card model",
    )
    parser.add_argument("--id", default="", help="the joined run's id, the base's by default")
    parser.add_argument(
        "--mixed-early-budget",
        action="store_true",
        help="let the parts differ in the indexer's early budget - every other indexer setting still has to match",
    )
    parser.add_argument(
        "--across-harness",
        action="store_true",
        help="for performance records, join parts whose harness files differ - each part's hashes are kept",
    )
    parser.add_argument(
        "--contexts",
        action="append",
        default=[],
        metavar="RUN=FIRST-LAST",
        help="for performance records, the contexts a part supplies, given for every part - either end may be empty",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write it, rather than print what would be written"
    )
    args = parser.parse_args()
    runs = [path.resolve() for path in [args.base, *args.extensions]]
    join_runs(
        runs,
        args.into.resolve(),
        args.apply,
        args.in_progress,
        args.replace,
        args.assemble,
        tuple(args.drop),
        tuple(args.supersede),
        args.id,
        args.across_hosts,
        args.mixed_early_budget,
        args.across_harness,
        tuple(args.contexts),
    )
