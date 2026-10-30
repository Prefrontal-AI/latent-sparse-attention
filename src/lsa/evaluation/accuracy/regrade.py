"""Rebuild a record from its sample files, with no server - for a grader change, or a record a stopped run left inconsistent."""

import dataclasses
import os
from typing import Any

from lsa import io
from lsa.evaluation.accuracy import records, tasks
from lsa.evaluation.accuracy.data import DATA_ROOT, load_manifest, suites
from lsa.evaluation.accuracy.runner import (
    Log,
    check_schema,
    finished_rows,
    grade_plan,
    require_sampling,
    samples_path,
    score_line,
    targeted_items,
)
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Records


def regrade(
    run_dir: str, data_root: str = DATA_ROOT, max_context: int | None = None, log: Log = print
) -> dict[str, Any]:
    """Rebuild a record from its sample files, with no server - for a grader change, or a record a stopped run left inconsistent.

    Args:
        run_dir: The run directory.
        data_root: The fixed prompt files.
        max_context: Which LongBench v2 file the run read.
        log: Where progress lines go.

    Returns:
        The rebuilt record, with the timing fields the old one had.

    Raises:
        FileNotFoundError: The run directory holds no record.
    """
    old = io.read_json(os.path.join(run_dir, Records.RECORD), None)
    if old is None:
        raise FileNotFoundError(os.path.join(run_dir, Records.RECORD))
    check_schema(old, os.path.join(run_dir, Records.RECORD))
    manifest = load_manifest(data_root)
    results = {}
    for plan_dict in old["run"]["plans"]:
        plan = suites.Plan(**{**plan_dict, "sampling": Sampling(**plan_dict["sampling"])})
        if not os.path.exists(samples_path(run_dir, plan)):
            continue
        definition = tasks.resolve(plan.task, data_root, max_context)
        lm_eval_task = definition.build()
        items = targeted_items(definition, lm_eval_task, plan)
        rows = finished_rows(io.read_jsonl(samples_path(run_dir, plan)))
        previous = old["tasks"].get(plan.task, {})
        server = old["servers"].get(previous.get("server"), old["server"])
        enforced = records.budget_enforced(dataclasses.asdict(require_sampling(plan)), server)
        prompt = {**previous.get("prompt", {}), "hash": tasks.prompt_set_hash(items)}
        entry = grade_plan(
            plan, definition, lm_eval_task, items, rows, run_dir, data_root, manifest, enforced, prompt
        )
        for key in ("concurrency", "concurrency_rows", "seconds", "output_tokens_per_second", "server"):
            entry[key] = previous.get(key)
        results[plan.task] = entry
        log(score_line(plan, definition, entry))
    old["tasks"] = results
    old["run"]["regraded"] = io.utc_now()
    io.write_json(os.path.join(run_dir, Records.RECORD), old, indent=1)
    return dict(old)
