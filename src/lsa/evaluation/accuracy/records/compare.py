"""Compare runs doc by doc: lsa records compare <baseline run> <run> [<run> ...] [--tasks cwe* aime25]

Every design scores the same docs, so two runs pair by doc id - a doc's repeats are averaged first, and the gap
(run minus baseline) is the mean of the docs' gaps. Its standard error and its bootstrap 95% interval are taken
over those gaps, so the difficulty both runs share cancels, and a gap whose interval holds no zero is marked `*`.
Wins, losses and ties count the docs the run scored above, below and level with the baseline, and `same` is the
share of samples both finished whose answer text is identical - the parity signal finer than the score, where a run
that differs from the baseline in a budget alone should read 1.000 below the budget's position.
Under the tasks a rollup gives each RULER length its docs-weighted scores, gap and `same`.

A run is its directory or its `record.json`, a joined run included - the record names where its samples are.
One table is printed per run, a row per task both scored, and `--json` prints the numbers instead.
"""

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.accuracy.records import compare_runs
from lsa.evaluation.constants import Records
from lsa.evaluation.runs import run_directory


def excludes_zero(interval: list[float] | None) -> bool:
    """Return whether a bootstrap interval lies wholly on one side of zero."""
    if not interval:
        return False
    low, high = interval
    return low > 0 or high < 0


def task_order(task: str) -> tuple[str, int]:
    """Order tasks by name and then by the length that ends a RULER cell's name, so a task's lengths read in order."""
    stem, _, tail = task.rpartition("_")
    return (stem, int(tail)) if tail.isdigit() else (task, 0)


def line(task: str, comparison: dict[str, Any]) -> str:
    """Return one task's row - its gap and the interval, the docs matched and the wins, losses and ties."""
    if comparison["gap"] is None:
        return f"{task:<34} no doc scored by both"
    low, high = comparison["ci95"] or (float("nan"), float("nan"))
    error = comparison["stderr"]
    mark = "*" if excludes_zero(comparison["ci95"]) else " "
    return (
        f"{task:<34} {comparison['docs']:>5} {comparison['score_a']:>7.3f} {comparison['score_b']:>7.3f} "
        f"{comparison['gap']:>+8.3f} {'-' if error is None else f'{error:.3f}':>6}  [{low:>+7.3f}, {high:>+7.3f}]{mark} "
        f"{comparison['wins']:>4} {comparison['losses']:>4} {comparison['ties']:>4} {rate(comparison['identical']):>6}"
    )


def rate(share: float | None) -> str:
    """Return a share as it is printed, or a dash for none."""
    return "-" if share is None else f"{share:.3f}"


def weighted(group: list[dict[str, Any]], key: str) -> float:
    """Return the mean of one field of paired comparisons, each weighted by its docs, or nan for none."""
    docs = sum(comparison["docs"] for comparison in group)
    return sum(comparison[key] * comparison["docs"] for comparison in group) / docs if docs else float("nan")


def rollup(comparisons: dict[str, dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Return each RULER length's scores over every task that ends in it, weighted by the docs each task matched.

    Args:
        comparisons: Task name to its paired comparison.

    Returns:
        Length to the docs matched, both runs' mean score, the gap and the share of identical answers, over the
        tasks of that length that matched a doc. A task whose name ends in no length is left out.
    """
    by_length: dict[int, list[dict[str, Any]]] = {}
    for task, comparison in comparisons.items():
        length = task_order(task)[1]
        if length and comparison["docs"]:
            by_length.setdefault(length, []).append(comparison)
    rolled: dict[int, dict[str, float]] = {}
    for length, group in sorted(by_length.items()):
        base = weighted(group, "score_a")
        score = weighted(group, "score_b")
        rolled[length] = {
            "docs": sum(comparison["docs"] for comparison in group),
            "base": base,
            "run": score,
            "gap": score - base,
            "same": weighted([each for each in group if each["identical"] is not None], "identical"),
        }
    return rolled


def table(base: Path, other: Path, comparisons: dict[str, dict[str, Any]]) -> str:
    """Return the table of one run against the baseline."""
    head = (
        f"{other.name}  minus  {base.name}\n"
        f"{'task':<34} {'docs':>5} {'base':>7} {'run':>7} {'gap':>8} {'se':>6}  {'95% ci':<19} {'win':>4} {'loss':>4} {'tie':>4} {'same':>6}"
    )
    rows = [line(task, comparisons[task]) for task in sorted(comparisons, key=task_order)]
    lengths = [
        f"{'RULER ' + str(length):<34} {t['docs']:>5} {t['base']:>7.3f} {t['run']:>7.3f} {t['gap']:>+8.3f} "
        f"{'':>6}  {'':<19} {'':>4} {'':>4} {'':>4} {t['same']:>6.3f}"
        for length, t in rollup(comparisons).items()
    ]
    return "\n".join([head, *rows, *lengths])


def chosen(patterns: list[str] | None, names: list[str]) -> list[str] | None:
    """Return the task names the patterns select, or None for every task.

    Raises:
        ValueError: A pattern matches no task of the baseline.
    """
    if not patterns:
        return None
    selected: list[str] = []
    for pattern in patterns:
        matched = fnmatch.filter(names, pattern)
        if not matched:
            raise ValueError(f"no task matches {pattern!r}")
        selected.extend(name for name in matched if name not in selected)
    return selected


def main() -> None:
    """Print each named run's paired comparison with the first."""
    parser = argparse.ArgumentParser(description="Compare runs against a baseline, doc by doc")
    parser.add_argument(
        "runs", nargs="+", type=run_directory, help="the baseline run, then the runs compared with it"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="task names or shell patterns over the tasks both runs hold, every shared task otherwise",
    )
    parser.add_argument("--json", action="store_true", help="print the numbers as json")
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("name the baseline and at least one run to compare with it")
    base = args.runs[0]
    baseline = io.read_json(base / Records.RECORD)["tasks"]
    try:
        results = {}
        for run in args.runs[1:]:
            # A pattern reads the tasks both runs hold, so one grid's `ruler_cwe_*` meets another's two lengths
            names = [name for name in baseline if name in io.read_json(run / Records.RECORD)["tasks"]]
            results[run] = compare_runs(base, run, chosen(args.tasks, names))
    except ValueError as failure:
        parser.error(str(failure))
    if args.json:
        print(json.dumps({str(run): comparisons for run, comparisons in results.items()}, indent=1))
        return
    print("\n\n".join(table(base, run, comparisons) for run, comparisons in results.items()))
    print("\n* the 95% bootstrap interval of the gap excludes 0")
