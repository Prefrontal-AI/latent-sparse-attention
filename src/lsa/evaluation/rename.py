"""Give a record of any kind a descriptive name, and a note where the name cannot say it all.

    lsa records rename <run dir> --name <name> --into <dir>                  # dry run
    lsa records rename <run dir> --name <name> --note "<sentence>" --into <dir> --apply
    lsa evaluate publish <dir>/<new id> --replace --apply                    # swap it in, archive the old

A run is launched under a label that says what was being tried - the batch it belonged to - and keeps it as its
config in the viewer long after the batch is forgotten. A record is renamed once it is worth showing, to what it
is: model, method, mesh and card - see `lsa.evaluation.gpu` for the card's token, last in every name.

The renamed run keeps the old id's timestamp, so its id is `<name>__<timestamp>`, and lands at
`<into>/<new id>`. Its record carries the new id and name, the note when one is given, and
`run.renamed_from`, the old id, which `lsa evaluate publish --replace` archives the way it archives a joined
run's parts. A joined run's `run.parts` keep the labels their launches ran under - they are what each launch
was called. Every file of the run is copied as it is, its launch log and metadata included, and the source is
never written.

A kernel record's `run.name` follows the rename like any other kind's. Records written before the name was the
run's own carry `native` or `full` there, and renaming one to its own id's name, which keeps the id, corrects it.
"""

import argparse
import copy
import getpass
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.constants import Records
from lsa.evaluation.publish import build_beside, copy_consistent, read_kind, run_files
from lsa.evaluation.runs import run_directory

# What a name is made of - lowercase words joined by underscores, as every record's config label is
NAME_PATTERN = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")


@dataclass
class Rename:
    """What renaming one record would write."""

    source: Path
    record: dict[str, Any]
    target: Path
    old_id: str
    refusals: list[str] = field(default_factory=list)


def plan_rename(run_dir: Path, name: str, into: Path, note: str = "") -> Rename:
    """Work out what renaming a record of any kind would write, writing nothing.

    Args:
        run_dir: The run directory.
        name: The new config label.
        into: The directory the renamed run directory goes in.
        note: A sentence for the record's note, or empty to keep the one it has.

    Returns:
        The rename - the record it would write, where, and each reason it refuses.

    Raises:
        FileNotFoundError: The directory holds no record.
        ValueError: The record is of no kind that can be renamed, or the name is not lowercase words joined by
            underscores.
    """
    record, _ = read_kind(run_dir)
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"name {name!r} is not lowercase words joined by underscores")
    old_id = str(record["run"]["id"])
    if "__" not in old_id:
        raise ValueError(f"run id {old_id!r} holds no timestamp after '__' to keep")
    new_id = f"{name}__{old_id.split('__', 1)[1]}"
    result = copy.deepcopy(record)
    run = result["run"]
    run["id"] = new_id
    run["renamed_from"] = old_id
    if note:
        run["note"] = note
    run["edits"] = [
        *(run.get("edits") or []),
        {
            "date": io.utc_now(),
            "who": getpass.getuser(),
            "what": f"renamed {old_id} to {new_id}" + (f"; note {note!r}" if note else ""),
        },
    ]
    run["name"] = name
    plan = Rename(run_dir, result, into / new_id, old_id)
    if new_id == old_id and record["run"].get("name") == name:
        plan.refusals.append(f"{old_id} is named {name} already")
    if plan.target.exists():
        plan.refusals.append(f"{plan.target} exists")
    return plan


def write_rename(plan: Rename) -> None:
    """Copy the run into its new directory, the record rewritten, built beside the target and renamed in."""
    built = build_beside(plan.target)
    try:
        for source in run_files(plan.source):
            copy_consistent(source, built / source.relative_to(plan.source))
        io.write_json(built / Records.RECORD, plan.record, indent=1)
        built.chmod(0o2775)
        built.rename(plan.target)
    finally:
        if built.exists():
            shutil.rmtree(built)


def rename_run(run_dir: Path, name: str, into: Path, note: str = "", apply: bool = False) -> Rename:
    """Rename a record, printing what it would write - nothing is written unless asked.

    Args:
        run_dir: The run directory.
        name: The new config label.
        into: The directory the renamed run directory goes in.
        note: A sentence for the record's note, or empty to keep the one it has.
        apply: Write it, rather than print what would be written.

    Returns:
        The rename.

    Raises:
        ValueError: Asked to write a rename that refuses.
    """
    plan = plan_rename(run_dir, name, into, note)
    print(f"{plan.old_id} -> {plan.record['run']['id']}  note {plan.record['run'].get('note') or '-'!r}")
    for refusal in plan.refusals:
        print(f"refused  {refusal}", flush=True)
    if not apply:
        print("dry run - pass --apply to write", flush=True)
        return plan
    if plan.refusals:
        raise ValueError(f"not renamed - {len(plan.refusals)} refusals above")
    write_rename(plan)
    print(f"wrote {plan.target}", flush=True)
    return plan


def main() -> None:
    """Rename the record named on the command line."""
    parser = argparse.ArgumentParser(description="Give a record a descriptive name and a note")
    parser.add_argument("run", type=run_directory, help="the run directory to rename, or its record.json")
    parser.add_argument("--name", required=True, help="the new config label, the run id's name part")
    parser.add_argument("--note", default="", help="one sentence of what the name cannot say")
    parser.add_argument("--into", type=Path, required=True, help="where the renamed run directory goes")
    parser.add_argument(
        "--apply", action="store_true", help="write it, rather than print what would be written"
    )
    args = parser.parse_args()
    rename_run(args.run.resolve(), args.name, args.into.resolve(), args.note, args.apply)
