"""Join the performance runs one ladder was split across into one run, which holds every context any of them measured.

    lsa records join <base run dir> <extension run dir> [...] --into results/joined/production    # dry run
    lsa records join <base run dir> <extension run dir> [...] --into results/joined/production --apply
    lsa records join ... --into results/joined/kernels --supersede 524288 --apply                  # the last part wins
lsa records join ... --into results/joined/kernels --across-hosts --apply                      # nodes of one card model
    lsa records join <early run dir> <plain run dir> --into results/joined/kernels --mixed-early-budget \\
        --contexts <early run id>=-1048576 --contexts <plain run id>=2097152- --across-harness --apply

A method's ladder is measured by several runs, each over the contexts its server or its time limit allowed - 16k to
128k, then 512k to 1M, then 2M to 10M. The joined run is the base run with the others' points added. It keeps the
base's id, so it lands at `<dir>/<base id>`, and `lsa evaluate publish --replace` archives the parts published
on their own, found by the ids in `run.parts` - the one mechanism the accuracy join writes too.

Parts join only when they are one measurement split by context. They are of one kind, kernels or production, and
agree on every other field of the record - the card model (`run.gpu`), the model, the method and its settings, the servers and the mesh, the
image, the harness and, for kernels, the phase. Only the run's own id, dates and duration, the launch's label and
whether the prompt had to repeat to reach a context may differ, and each part keeps its own in `run.parts`. A
refusal names the field and both values. The host differs only under `--across-hosts`, and then only between nodes
of one card model - a differing `run.gpu` still refuses - with each part's host kept in `run.parts`. No two parts hold the same context unless `--supersede <context>` says
the last part holding it wins, its earlier points staying under `parts/<part id>/`. `--assemble` and `--drop` are
accuracy flags and refuse here.

`--mixed-early-budget` lets the parts differ in the indexer's early budget and in no other setting, as it does for
accuracy records. The record's own indexer names the early budget of the last part that sets one, and `run.parts`
keeps each part's. `--across-harness` lets the parts differ in the hashes of the harness files and the commit they ran at, and
`run.parts` keeps each part's hashes and commits beside the contexts it supplied. A part's note and the name it was renamed from may differ
always, and `run.parts` keeps them.

`--contexts <run id>=<first>-<last>` says which contexts a part supplies, either end left empty for no limit. It is
given for every part or none, and refuses when two ranges overlap, when a part holds a context no range covers, or
when a context a part holds lies in the range of a part that does not hold it. A point outside its part's range is
left out like a superseded one, and `run.parts` keeps each range. It does not combine with `--supersede`.

The joined directory holds the joined record, every part's samples of a context it keeps under `samples/`, each
part's other files - its launch log, metadata, server info and log - under `parts/<part id>/`, and a launch log of
every part's launches with the join's own launch last. The joined run's status is the record's: partial while
a context failed in every part, complete otherwise. Nothing under a part's directory is written.
"""

import copy
import getpass
import json
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation import gpu
from lsa.evaluation.accuracy.records.join import (
    Join,
    Part,
    join_snapshot,
    joined_launches,
    read_part,
    status_refusals,
    target_refusals,
    write_join,
)
from lsa.evaluation.constants import Records, Status
from lsa.evaluation.publish import KIND_OF_SCHEMA, run_files
from lsa.evaluation.status import MISSING

# What two parts of one ladder may differ in - the run's own identity and timing, the launch's label, and whether
# the prompt repeated, which depends on the longest context the part reached
LADDER_FIELDS: tuple[str, ...] = (
    "run.id",
    "run.started",
    "run.date",
    "run.seconds",
    "run.parts",
    "run.edits",
    "run.snapshot",
    "run.name",
    "run.note",
    "run.renamed_from",
    "setup.prompt.repeated",
)

# What the record keeps of the harness code, as the hashes of each file and the commit it ran at - `--across-harness`
# lets them differ
HARNESS_FIELDS: tuple[str, ...] = ("run.harness.", "run.commits.")

# Where a record names the indexer's early budget - a production record's indexer, a kernels record's setup
EARLY_BUDGET = "early_budget"
EARLY_BUDGET_HOMES: tuple[tuple[str, ...], ...] = (("indexer",), ("setup", "config"))

# What a record holds beside its description - the measured contexts and the ones that failed
MEASURED_FIELDS: tuple[str, ...] = ("points", "failed")

# Files kept at the joined run's top, from the base
BASE_FILES: tuple[str, ...] = (Records.META, Records.SERVER_INFO, "indexer.json")


def flatten(value: dict[str, Any], prefix: str = "", left_out: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return a record's description as dotted keys to values, without its measurements or what may differ.

    Args:
        value: The record, or a part of it.
        prefix: The dotted path to value.
        left_out: Keys at any depth, and dotted prefixes, that are left out as well.

    Returns:
        The keys to values.
    """
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = prefix + key
        if name in MEASURED_FIELDS or name in LADDER_FIELDS:
            continue
        if key in left_out or any(name.startswith(left) for left in left_out if left.endswith(".")):
            continue
        if isinstance(item, dict):
            flat |= flatten(item, name + ".", left_out)
        else:
            flat[name] = item
    return flat


def field_refusals(
    base: Part,
    part: Part,
    across_hosts: bool = False,
    mixed_early_budget: bool = False,
    across_harness: bool = False,
) -> list[str]:
    """Say which field of a part's record differs from the base's, with both values.

    Args:
        base: The base part.
        part: The part checked against it.
        across_hosts: Let the host differ, the card model still agreeing.
        mixed_early_budget: Let the indexer's early budget differ.
        across_harness: Let the hashes of the harness files and the commit differ.

    Returns:
        One line per differing field, the card model first.
    """
    left_out = (
        *((EARLY_BUDGET,) if mixed_early_budget else ()),
        *(HARNESS_FIELDS if across_harness else ()),
    )
    mine, theirs = flatten(base.record, left_out=left_out), flatten(part.record, left_out=left_out)
    if across_hosts:
        mine.pop("run.host", None)
        theirs.pop("run.host", None)
    # The card is compared by `gpu.refusals`
    mine.pop("run.gpu", None)
    theirs.pop("run.gpu", None)
    return [
        *gpu.refusals(base.record, part.record),
        *(
            f"{key} {mine.get(key)!r} vs {theirs.get(key)!r}"
            for key in sorted(set(mine) | set(theirs))
            if mine.get(key) != theirs.get(key)
        ),
    ]


def contexts_of(record: dict[str, Any]) -> list[str]:
    """Return the contexts a record holds a point of, as the strings its points are keyed by, ascending."""
    return sorted(record["points"], key=int)


def early_budget_home(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the dict of a record that holds its indexer's early budget when it has one, None for neither home."""
    for path in EARLY_BUDGET_HOMES:
        home: Any = record
        for key in path:
            home = home.get(key) if isinstance(home, dict) else None
        if isinstance(home, dict):
            return home
    return None


def early_budget_of(part: Part) -> dict[str, Any] | None:
    """Return the early budget a part's indexer was set with, None for none."""
    return (early_budget_home(part.record) or {}).get(EARLY_BUDGET)


def early_budget_refusals(parts: list[Part]) -> list[str]:
    """Say where the parts that set an early budget do not set the same one."""
    budgets = {part.run_id: early_budget_of(part) for part in parts if early_budget_of(part)}
    if len({json.dumps(budget, sort_keys=True) for budget in budgets.values()}) < 2:
        return []
    return [f"early budgets differ between parts: {budgets}"]


Span = tuple[int | None, int | None]


def parse_spans(specs: tuple[str, ...], parts: list[Part]) -> tuple[dict[str, Span], list[str]]:
    """Read the `--contexts <run id>=<first>-<last>` flags into each part's span, and say what is wrong with them.

    Args:
        specs: The flags' values.
        parts: The parts the spans are of.

    Returns:
        Each named part's first and last context, None for no limit, and one line per flag that is wrong.
    """
    spans: dict[str, Span] = {}
    refusals = []
    run_ids = [part.run_id for part in parts]
    for spec in specs:
        run_id, _, text = spec.rpartition("=")
        first, dash, last = text.partition("-")
        if run_id not in run_ids or not dash or not all(end.isdigit() or not end for end in (first, last)):
            refusals.append(f"--contexts {spec!r} is not <run id>=<first>-<last> of one of {run_ids}")
        elif run_id in spans:
            refusals.append(f"--contexts names {run_id} twice")
        else:
            spans[run_id] = (int(first) if first else None, int(last) if last else None)
    missing = [run_id for run_id in run_ids if run_id not in spans]
    if spans and missing and not refusals:
        refusals.append(f"--contexts is given for every part or none - not for {', '.join(missing)}")
    return spans, refusals


def in_span(context: str, span: Span) -> bool:
    """Say whether a context lies in a span."""
    first, last = span
    return (first is None or int(context) >= first) and (last is None or int(context) <= last)


def span_refusals(parts: list[Part], spans: dict[str, Span]) -> list[str]:
    """Say where the spans overlap, or where a context a part holds is not supplied by the span that covers it."""
    refusals = []
    for index, part in enumerate(parts):
        first, last = spans[part.run_id]
        for other in parts[index + 1 :]:
            other_first, other_last = spans[other.run_id]
            if (last is None or other_first is None or other_first <= last) and (
                other_last is None or first is None or first <= other_last
            ):
                refusals.append(f"--contexts of {part.run_id} and {other.run_id} overlap")
    held = {context for part in parts for context in part.record["points"]}
    for context in sorted(held, key=int):
        covering = [part for part in parts if in_span(context, spans[part.run_id])]
        holders = [part.run_id for part in parts if context in part.record["points"]]
        if not covering:
            refusals.append(f"context {context} is held by {', '.join(holders)} and no --contexts covers it")
        elif len(covering) == 1 and context not in covering[0].record["points"]:
            refusals.append(
                f"context {context} is held by {', '.join(holders)} but --contexts gives it to "
                f"{covering[0].run_id}, which does not hold it"
            )
    return refusals


def trim_to_spans(parts: list[Part], spans: dict[str, Span]) -> list[Part]:
    """Return the parts, each without the points of the contexts outside its span."""
    result = []
    for part in parts:
        left_out = {context for context in part.record["points"] if not in_span(context, spans[part.run_id])}
        result.append(trimmed(part, left_out) if left_out else part)
    return result


def trimmed(part: Part, left_out: set[str]) -> Part:
    """Return a part without the points of some contexts, which its directory still holds."""
    record = copy.deepcopy(part.record)
    record["points"] = {
        context: point for context, point in record["points"].items() if context not in left_out
    }
    return Part(
        part.run_dir,
        record,
        part.status,
        part.status_why,
        part.launches,
        sorted(left_out & set(part.record["points"]), key=int),
    )


def trim_parts(parts: list[Part], supersede: tuple[str, ...]) -> list[Part]:
    """Return the parts, a superseded context left out of every part but the last that holds it.

    Args:
        parts: The base first, then the extensions.
        supersede: Contexts the last part holding them keeps.

    Returns:
        The parts, each without what it gives up.
    """
    last_holder = {
        context: index
        for index, part in enumerate(parts)
        for context in part.record["points"]
        if context in supersede
    }
    result = []
    for index, part in enumerate(parts):
        left_out = {context for context in part.record["points"] if last_holder.get(context, index) > index}
        result.append(trimmed(part, left_out) if left_out else part)
    return result


def context_refusals(parts: list[Part]) -> list[str]:
    """Say which context two parts hold, once superseded contexts are left out."""
    holder: dict[str, str] = {}
    refusals = []
    for part in parts:
        for context in contexts_of(part.record):
            if context in holder:
                refusals.append(
                    f"context {context} is held by {holder[context]} and {part.run_id} - "
                    f"pass --supersede {context} to keep the later"
                )
            holder[context] = part.run_id
    return refusals


def part_summary(
    part: Part,
    mixed_early_budget: bool = False,
    across_harness: bool = False,
    span: Span | None = None,
) -> dict[str, Any]:
    """Return what the joined record keeps of one part - where it came from, what it measured and when.

    Args:
        part: The part, as trimmed.
        mixed_early_budget: Keep the part's early budget.
        across_harness: Keep the part's harness hashes.
        span: The first and last context the part was asked to supply, None for no limit or no span.

    Returns:
        The summary.
    """
    run = part.record["run"]
    kept = {
        **({"early_budget": early_budget_of(part)} if mixed_early_budget else {}),
        **({"harness": run.get("harness")} if across_harness else {}),
        **({"span": {"first": span[0], "last": span[1]}} if span else {}),
    }
    return {
        "id": part.run_id,
        "name": run.get("name"),
        "note": run.get("note"),
        "renamed_from": run.get("renamed_from"),
        "tag": run.get("tag"),
        "started": run.get("started"),
        "date": run.get("date"),
        "seconds": run.get("seconds"),
        "host": run.get("host"),
        "commits": run.get("commits"),
        "status": part.status,
        "jobs": [launch.get("job") for launch in part.launches],
        "contexts": contexts_of(part.record),
        "left_out": part.left_out,
        "prompt_repeated": (part.record.get("setup") or {}).get("prompt", {}).get("repeated"),
        **kept,
    }


def joined_record(
    parts: list[Part],
    run_id: str = "",
    mixed_early_budget: bool = False,
    across_harness: bool = False,
    spans: dict[str, Span] | None = None,
) -> dict[str, Any]:
    """Return the record of the joined run - the base's, with every part's points and history.

    With mixed early budgets the record's own indexer names the early budget of the last part that sets one.

    Args:
        parts: The base first, then the extensions.
        run_id: The joined run's id, or empty for the base's.
        mixed_early_budget: The parts may differ in their early budget.
        across_harness: The parts may differ in their harness hashes.
        spans: Each part's span by run id, empty for none.

    Returns:
        The record.
    """
    base = parts[0]
    spans = spans or {}
    record = copy.deepcopy(base.record)
    newest = next((part for part in reversed(parts) if early_budget_of(part)), None)
    home = early_budget_home(record)
    if mixed_early_budget and newest and home is not None:
        home[EARLY_BUDGET] = early_budget_of(newest)
    run = record["run"]
    points: dict[str, Any] = {}
    for part in parts:
        points |= part.record["points"]
    record["points"] = {context: points[context] for context in sorted(points, key=int)}
    if "failed" in record:
        failed = {context for part in parts for context in part.record["failed"]}
        record["failed"] = sorted((context for context in failed if str(context) not in points), key=int)
    run["id"] = run_id or base.run_id
    run["started"] = min(str(part.record["run"].get("started") or "") for part in parts)
    run["seconds"] = round(sum(part.record["run"].get("seconds") or 0.0 for part in parts), 1)
    run["date"] = max(str(part.record["run"].get("date") or "") for part in parts)
    run["parts"] = [
        part_summary(part, mixed_early_budget, across_harness, spans.get(part.run_id)) for part in parts
    ]
    run["edits"] = [
        *(run.get("edits") or []),
        {
            "date": io.utc_now(),
            "who": getpass.getuser(),
            "what": f"joined {', '.join(part.run_id for part in parts[1:])} into {base.run_id}"
            + (f" as {run_id}" if run_id else "")
            + "".join(
                f"; {part.run_id} leaves out {', '.join(part.left_out)}" for part in parts if part.left_out
            )
            + "".join(
                f"; {supplier} supplies {'' if span[0] is None else span[0]}-{'' if span[1] is None else span[1]}"
                for supplier, span in spans.items()
            ),
        },
    ]
    return record


def sample_context(relative: Path) -> str:
    """Return the context a samples file is of - the number its name starts with, or empty for any other file."""
    if relative.parts[0] != "samples":
        return ""
    head = relative.name.split(".")[0].split("-")[0]
    return head if head.isdigit() else ""


def planned_copies(parts: list[Part]) -> list[tuple[Path, Path]]:
    """Return every file the joined directory takes from the parts, each with where it lands in it.

    A part's samples of a context its record holds go under `samples/`, and every other file of its directory -
    the samples of a context it left out or never finished included - under `parts/<part id>/`. The base's
    metadata, server info and indexer settings are also kept at the top, where a reader of one run looks for them.
    """
    copies: list[tuple[Path, Path]] = []
    for part in parts:
        for source in run_files(part.run_dir):
            relative = source.relative_to(part.run_dir)
            kept = sample_context(relative) in part.record["points"]
            copies.append((source, relative if kept else Path(Records.PARTS, part.run_id, relative)))
    base = parts[0].run_dir
    copies += [(base / name, Path(name)) for name in BASE_FILES if (base / name).is_file()]
    return copies


def plan_performance_join(
    run_dirs: list[Path],
    into: Path,
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
    """Work out what joining performance runs would write and whether it may, writing nothing.

    Args:
        run_dirs: The base run directory first, then the runs that extend it.
        into: The directory the joined run directory goes in.
        in_progress: Join parts still going, as a snapshot of their records as they stand.
        replace: Rebuild a joined run written before.
        assemble: Refused - an accuracy flag.
        drop: Refused - an accuracy flag.
        supersede: Contexts the last part holding them keeps.
        run_id: The joined run's id, or empty for the base's.
        across_hosts: Let parts differ in host when their card model is the same.
        mixed_early_budget: Let parts differ in the indexer's early budget and in no other setting.
        across_harness: Let parts differ in the hashes of the harness files.
        contexts: One `<run id>=<first>-<last>` per part, the contexts it supplies.

    Returns:
        The join - the record it would write, the files it would copy, and each reason it refuses.
    """
    parts = [read_part(run_dir) for run_dir in run_dirs]
    join = Join(parts=parts, target=into / (run_id or parts[0].run_id))
    join.refusals = [
        f"{flag} is for accuracy records"
        for flag, value in (("--assemble", assemble), ("--drop", drop))
        if value
    ]
    join.refusals += [
        f"{part.run_id}: a {part.record.get('schema')!r} record is not a kernel or production record"
        for part in parts
        if part.record.get("schema") not in KIND_OF_SCHEMA or part.record["schema"] == Records.SCHEMA
    ]
    if join.refusals:
        return join
    held = {context for part in parts for context in part.record["points"]}
    join.refusals += [
        f"--supersede names {context}, which no part holds" for context in supersede if context not in held
    ]
    schemas = {part.record["schema"] for part in parts}
    if len(schemas) > 1:
        join.refusals.append(f"parts are of different kinds - {', '.join(sorted(schemas))}")
        return join
    spans, span_flags = parse_spans(contexts, parts)
    join.refusals += span_flags
    if spans and supersede:
        join.refusals.append("--contexts and --supersede both choose contexts - pass one")
    if spans and not join.refusals:
        join.refusals += span_refusals(parts, spans)
    if join.refusals:
        return join
    parts = trim_to_spans(parts, spans) if spans else trim_parts(parts, supersede)
    join.parts = parts
    if len({part.run_dir.resolve() for part in parts}) < len(parts):
        join.refusals.append("a run is named twice")
    for part in parts[1:]:
        join.refusals += [
            f"{part.run_id}: {refusal}"
            for refusal in field_refusals(parts[0], part, across_hosts, mixed_early_budget, across_harness)
        ]
    if mixed_early_budget:
        join.refusals += early_budget_refusals(parts)
    join.refusals += context_refusals(parts)
    join.refusals += status_refusals(parts, in_progress)
    join.refusals += target_refusals(join, replace, run_id)
    join.record = joined_record(parts, run_id, mixed_early_budget, across_harness, spans)
    join.copies = planned_copies(parts)
    join.launch_lines = joined_launches(parts)
    join.missing = MISSING[KIND_OF_SCHEMA[join.record["schema"]]](join.record)
    join.status = Status.PARTIAL if join.missing else Status.COMPLETE
    snapshot = join_snapshot(parts)
    if snapshot:
        join.record["run"]["snapshot"] = snapshot
        join.status = Status.IN_PROGRESS
    return join


def describe_performance(join: Join) -> str:
    """Return the lines a performance join prints - each part, the joined run, and every reason it refuses."""
    lines = []
    for index, part in enumerate(join.parts):
        role = "base" if index == 0 else "extends"
        contexts = ", ".join(contexts_of(part.record)) if "points" in part.record else "-"
        lines.append(f"{role:<8}{part.run_id}  [{part.status or 'no launch log'}]  contexts {contexts}")
        jobs = [str(launch.get("job")) for launch in part.launches]
        lines.append(f"          jobs {', '.join(jobs) or '-'}")
        if part.left_out:
            lines.append(f"          leaves out {', '.join(part.left_out)}")
    if join.record:
        size = sum(source.stat().st_size if source.is_file() else 0 for source, _ in join.copies)
        lines.append(
            f"joined  {join.target}  [{join.status}]  contexts {', '.join(contexts_of(join.record))}, "
            f"{len(join.copies)} files ({size / 2**20:.0f} MiB)"
        )
        if join.missing:
            lines.append(f"          {join.missing}")
    lines += [f"refused  {refusal}" for refusal in join.refusals]
    return "\n".join(lines)


def join_performance(
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
    """Join performance runs into one, printing what it would write - nothing is written unless asked.

    Args:
        run_dirs: The base run directory first, then the runs that extend it.
        into: The directory the joined run directory goes in.
        apply: Write it, rather than print what would be written.
        in_progress: Join parts still going, as a snapshot.
        replace: Rebuild a joined run written before.
        assemble: Refused - an accuracy flag.
        drop: Refused - an accuracy flag.
        supersede: Contexts the last part holding them keeps.
        run_id: The joined run's id, or empty for the base's.
        across_hosts: Let parts differ in host when their card model is the same.
        mixed_early_budget: Let parts differ in the indexer's early budget and in no other setting.
        across_harness: Let parts differ in the hashes of the harness files.
        contexts: One `<run id>=<first>-<last>` per part, the contexts it supplies.

    Returns:
        The join.

    Raises:
        ValueError: Asked to write a join that refuses.
    """
    join = plan_performance_join(
        run_dirs,
        into,
        in_progress,
        replace,
        assemble,
        drop,
        supersede,
        run_id,
        across_hosts,
        mixed_early_budget,
        across_harness,
        contexts,
    )
    print(describe_performance(join), flush=True)
    if not apply:
        print("dry run - pass --apply to write", flush=True)
        return join
    if join.refusals:
        raise ValueError(f"not joined - {len(join.refusals)} refusals above")
    write_join(join)
    print(f"wrote {join.target}", flush=True)
    return join
