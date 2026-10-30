"""Print records as a table, whichever kind they are - accuracy, kernels or production.

    lsa records show <run dir or record.json> [more...]

The kind is read from the record's schema, as `lsa evaluate publish` reads it, and the records of one call are all of one kind.
One accuracy record is a line per task, several align the tasks by name so configurations can be read side by side.
One kernels record is a line per context, several align the contexts with each configuration's layer time and its speedup over the first.
Production records are a row per length and deployment, and two get a speedup column, the first over the second.
"""

import argparse
from pathlib import Path
from typing import Any

from lsa.evaluation.accuracy.records import mesh_label_of
from lsa.evaluation.constants import Evaluation
from lsa.evaluation.publish import read_kind

Record = dict[str, Any]


def mean_of(entry: dict[str, Any], count: str) -> int | str:
    """Return a task's mean of one token count - 0 where no row had it, a dash where the task has no tokens."""
    if "tokens" not in entry:
        return "-"
    return int((entry["tokens"][count] or {}).get("mean") or 0)


def cell(entry: dict[str, Any], key: str, spec: str) -> str:
    """Return one of a task's counts under a format spec, or a dash of the same width where it has none.

    An import of aggregate scores has no counts for most tasks.
    """
    value = entry.get(key)
    return format("-" if value is None else value, spec if value is not None else spec.split(".")[0])


def line(task_name: str, entry: dict[str, Any]) -> str:
    """Return one task's line - its score and spread, its counts and its mean tokens."""
    score = "-" if entry["score"] is None else f"{entry['score']:.3f}"
    spread = f" ±{entry['repeat_std']:.3f}" if entry.get("repeat_std") is not None else ""
    return (
        f"{task_name:<32} {score:>6}{spread:<8} n {cell(entry, 'scored', '>4')}/{cell(entry, 'n', '<4')} "
        f"empty {cell(entry, 'empty', '>3')} trunc {cell(entry, 'truncated', '>3')} err {cell(entry, 'errors', '>2')}  "
        f"prompt {mean_of(entry, 'prompt'):>8} reason {mean_of(entry, 'reasoning'):>6} "
        f"answer {mean_of(entry, 'answer'):>5}  {cell(entry, 'seconds', '>8.0f')} s"
    )


def mesh_of(record: Record) -> str:
    """Return a record's mesh label, or a dash for a record no server ran."""
    return mesh_label_of(record["server"]) if record["server"] else "-"


def show(record: Record) -> None:
    """Print one record - a header line, then a line per task."""
    run = record["run"]
    print(
        f"{run['id']}  {record['model']['name']}  {run['name']}  {mesh_of(record)}  {run['suite']}  "
        f"{run['date']}  {run['seconds']} s"
    )
    if run["note"]:
        print(f"  note: {run['note']}")
    print(
        f"  {'task':<32} {'score':>6}{'':<8} {'':>13} {'':>9} {'':>9} {'':>6}  {'tokens: mean prompt / reasoning / answer':<40}"
    )
    for task_name, entry in record["tasks"].items():
        print("  " + line(task_name, entry))


def compare(records: list[Record]) -> None:
    """Print several records side by side, one line per task any of them holds."""
    names: list[str] = []
    for record in records:
        for task_name in record["tasks"]:
            if task_name not in names:
                names.append(task_name)
    labels = [f"{r['run']['name']} {mesh_of(r)}" for r in records]
    print(f"{'task':<32} " + " ".join(f"{label:>22}" for label in labels))
    for task_name in names:
        cells = []
        for record in records:
            entry = record["tasks"].get(task_name)
            if entry is None or entry["score"] is None:
                cells.append(f"{'-':>22}")
            else:
                counts = (
                    f" (e{entry['empty']} t{entry['truncated']} n{entry['scored']})"
                    if "empty" in entry
                    else ""
                )
                cells.append(f"{entry['score']:.3f}{counts}".rjust(22))
        print(f"{task_name:<32} " + " ".join(cells))


def print_records(records: list[Record], kind: str) -> None:
    """Print records of one kind as the table that kind reads as.

    Args:
        records: The records, baseline first.
        kind: Their kind, one of the `Evaluation` kinds.
    """
    if kind == Evaluation.PRODUCTION:
        from lsa.evaluation.performance.production import record

        print(record.table(records))
        return
    if kind == Evaluation.KERNELS:
        from lsa.evaluation.performance.kernels import show as kernels

        if len(records) == 1:
            kernels.show(records[0])
        else:
            kernels.compare(records)
        return
    if len(records) == 1:
        show(records[0])
    else:
        compare(records)


def main() -> None:
    """Print the records named on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("records", nargs="+", metavar="RECORD", help="a run directory or its record.json")
    paths = [Path(path) for path in parser.parse_args().records]
    read = [read_kind(path.parent if path.is_file() else path) for path in paths]
    kinds = sorted({kind for _, kind in read})
    if len(kinds) > 1:
        parser.error(f"records of one kind at a time - got {', '.join(kinds)}")
    print_records([record for record, _ in read], kinds[0])
