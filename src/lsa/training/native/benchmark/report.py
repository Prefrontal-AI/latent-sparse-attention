"""Compare arms from the records a benchmark left, paired by example.

    lsa native report <records> --baseline b16k-dense

Every arm answers the same fixed examples, so a difference is taken example by example and resampled over examples
rather than over two independent means: the bootstrap draws example ids with replacement, recomputes both arms'
statistic on the same draw and keeps the percentile interval of the difference. A perplexity is a ratio of sums, so
its statistic is recomputed on each draw rather than averaged.

RULER is read by length and by task, and the window arm is read apart: its attention reaches 516 keys, so a cell it
solves past that reach is a cell that does not test long range.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from lsa.training.native.benchmark import records, tasks

# The metric each task is compared on, and whether a higher value is better
PRIMARY: dict[str, str] = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "piqa": "acc",
    "winogrande": "acc",
    "lambada_openai": "acc",
    "wikitext": "word_perplexity",
    "sciq": "acc",
}
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 0
# Draws resampled at once, so the index matrix stays small for a 10k-example task
BOOTSTRAP_CHUNK = 250


def statistic(values: np.ndarray) -> float:
    """Reduce per-example values to the task's number.

    Args:
        values: Per-example scores [n], or per-example (log-likelihood, count) pairs [n, 2] for a perplexity.

    Returns:
        The mean score, or the perplexity of the pooled pairs.
    """
    if values.ndim == 1:
        return float(values.mean())
    return float(np.exp(-values[:, 0].sum() / values[:, 1].sum()))


def resampled(values: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Return the statistic of each draw.

    Args:
        values: Per-example values, as `statistic` takes them.
        draws: Example indices [r, n] int64, one row a resample.

    Returns:
        The statistic of each resample [r].
    """
    picked = values[draws]
    if values.ndim == 1:
        return np.asarray(picked.mean(axis=1))
    return np.asarray(np.exp(-picked[:, :, 0].sum(axis=1) / picked[:, :, 1].sum(axis=1)))


def paired_bootstrap(
    base: np.ndarray, other: np.ndarray, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """Bootstrap the difference of a statistic between two arms over the same examples.

    Args:
        base: The reference arm's per-example values, as `statistic` takes them.
        other: The compared arm's, in the same example order.
        iterations: Resamples drawn.
        seed: Seed of the draws, so a table is the same every time it is made.

    Returns:
        A tuple of:
            The difference of the statistic, other less base.
            The lower end of its 95% interval.
            The upper end of its 95% interval.
    """
    generator = np.random.default_rng(seed)
    count = len(base)
    differences = []
    for start in range(0, iterations, BOOTSTRAP_CHUNK):
        draws = generator.integers(0, count, size=(min(BOOTSTRAP_CHUNK, iterations - start), count))
        differences.append(resampled(other, draws) - resampled(base, draws))
    spread = np.concatenate(differences)
    low, high = np.percentile(spread, [2.5, 97.5])
    return statistic(other) - statistic(base), float(low), float(high)


def values_of(rows: dict[int, dict[str, Any]], ids: list[int], metric: str) -> np.ndarray:
    """Collect one metric of the rows with these ids.

    Args:
        rows: A task's sample rows, by id.
        ids: The ids, in the order the values are wanted.
        metric: The metric's name.

    Returns:
        The values [n], or [n, 2] when the metric is a pair.
    """
    return np.array([rows[i]["metrics"][metric] for i in ids], dtype=float)


def shared_ids(*tables: dict[int, dict[str, Any]]) -> list[int]:
    """Return the example ids every table holds with the same prompt, ascending.

    Args:
        *tables: Sample rows by id, one per arm.

    Returns:
        The ids a pair can be made from.
    """
    first, *rest = tables
    return sorted(
        i
        for i in first
        if all(i in table and table[i]["prompt_hash"] == first[i]["prompt_hash"] for table in rest)
    )


def collect(folder: Path, names: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    """Read the sample tables of the tasks a run holds.

    Args:
        folder: The run's directory.
        names: The tasks.

    Returns:
        The rows by id, for every task whose file is there.
    """
    found = {}
    for name in names:
        if (folder / records.SAMPLES / f"{name}.jsonl").exists():
            found[name] = records.read_samples(folder, name)
    return found


def format_cell(value: float, interval: tuple[float, float] | None = None, digits: int = 3) -> str:
    """Format a number, with its interval when it has one."""
    if interval is None:
        return f"{value:.{digits}f}"
    return f"{value:+.{digits}f} [{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def core_tables(arms: dict[str, Path], baseline: str, names: list[str]) -> str:
    """Render the core tasks - each arm's score, and its paired difference from the baseline.

    Args:
        arms: Each arm's run directory.
        baseline: The arm the others are compared with.
        names: The tasks to report.

    Returns:
        Two markdown tables.
    """
    tables = {arm: collect(folder, names) for arm, folder in arms.items()}
    present = [name for name in names if all(name in found for found in tables.values())]
    scores, differences = [], []
    for name in present:
        metric = PRIMARY[name]
        ids = shared_ids(*(found[name] for found in tables.values()))
        base = values_of(tables[baseline][name], ids, metric)
        row, difference = [f"{name} ({metric})"], [f"{name} ({metric})"]
        for arm in arms:
            values = values_of(tables[arm][name], ids, metric)
            row.append(format_cell(statistic(values)))
            if arm != baseline:
                delta, low, high = paired_bootstrap(base, values)
                difference.append(format_cell(delta, (low, high)))
        scores.append(row)
        differences.append(difference)
    others = [arm for arm in arms if arm != baseline]
    return "\n\n".join(
        [
            table(["task", *arms], scores),
            f"Paired difference from {baseline}, 95% interval over examples:",
            table(["task", *others], differences),
        ]
    )


def ruler_tables(arms: dict[str, Path], baseline: str, names: list[str]) -> str:
    """Render RULER by length and by task, with the window arm's reach marked.

    Args:
        arms: Each arm's run directory.
        baseline: The arm the others are compared with.
        names: The RULER cells to report.

    Returns:
        Markdown tables: the score by length, its paired difference from the baseline, and the score by task.
    """
    tables = {arm: collect(folder, names) for arm, folder in arms.items()}
    cells = [name for name in names if all(name in found for found in tables.values())]
    lengths = tasks.lengths_of(cells)
    by_length, difference_by_length = [], []
    for length in lengths:
        members = [name for name in cells if tasks.parse_ruler(name)[1] == length]
        pooled = {
            arm: np.concatenate(
                [
                    values_of(tables[arm][name], shared_ids(*(tables[a][name] for a in arms)), "score")
                    for name in members
                ]
            )
            for arm in arms
        }
        row, difference = [str(length)], [str(length)]
        for arm in arms:
            row.append(format_cell(statistic(pooled[arm])))
            if arm != baseline:
                delta, low, high = paired_bootstrap(pooled[baseline], pooled[arm])
                difference.append(format_cell(delta, (low, high)))
        by_length.append(row)
        difference_by_length.append(difference)
    by_task = []
    for name in cells:
        ids = shared_ids(*(tables[arm][name] for arm in arms))
        by_task.append(
            [
                name[len(tasks.RULER_PREFIX) :],
                *(
                    format_cell(statistic(values_of(tables[arm][name], ids, "score")), digits=2)
                    for arm in arms
                ),
            ]
        )
    others = [arm for arm in arms if arm != baseline]
    return "\n\n".join(
        [
            table(["length", *arms], by_length),
            f"Paired difference from {baseline}, 95% interval over examples:",
            table(["length", *others], difference_by_length),
            "By task:",
            table(["task_length", *arms], by_task),
        ]
    )


def leakage_table(arm: Path, reference: Path | None, names: list[str]) -> str:
    """Render the window arm's RULER score by task and length, beside a reference arm's.

    The window arm reads 516 keys per attention layer, so a prompt longer than the key budget (1,540 tokens) is not
    within its reach, and a cell it solves is a cell its recurrent layers or the prompt's own regularity solve - not
    a test of long-range attention. Every RULER length here is past that budget.

    Args:
        arm: The window arm's run directory.
        reference: Another arm's, shown beside it, or None.
        names: The RULER cells to report.

    Returns:
        A markdown table of mean score per task and length, and the share of cells the window arm scores above zero.
    """
    window = collect(arm, names)
    other = collect(reference, names) if reference else {}
    cells = [name for name in names if name in window]
    rows = []
    for name in cells:
        ids = sorted(window[name])
        row = [
            name[len(tasks.RULER_PREFIX) :],
            format_cell(statistic(values_of(window[name], ids, "score")), digits=2),
        ]
        if name in other:
            row.append(format_cell(statistic(values_of(other[name], sorted(other[name]), "score")), digits=2))
        rows.append(row)
    header = ["task_length", arm.parent.name] + ([reference.parent.name] if reference else [])
    solved = sum(statistic(values_of(window[name], sorted(window[name]), "score")) > 0 for name in cells)
    return "\n\n".join(
        [
            table(header, rows),
            f"The window arm scores above zero on {solved} of {len(cells)} cells, every one past 1,540 tokens.",
        ]
    )


def report(
    records_root: str | Path, baseline: str, preset_names: list[str], window: str | None = None
) -> str:
    """Render the tables of every preset the records hold.

    Args:
        records_root: The records root.
        baseline: The label of the arm the others are compared with.
        preset_names: The presets to report, each a directory beside the arms' checkpoints.
        window: The label of the window arm, whose RULER scores are also shown as the leakage check.

    Returns:
        Markdown.
    """
    sections = []
    for preset in preset_names:
        arms = {
            path.parent.name: path
            for path in sorted(Path(records_root).glob(f"*/{preset}"))
            if (path / records.RESULTS).exists()
        }
        if baseline not in arms:
            continue
        names = tasks.expand(preset)
        body = (
            ruler_tables(arms, baseline, names)
            if tasks.is_ruler(names[0])
            else core_tables(arms, baseline, names)
        )
        sections.append(f"## {preset}\n\n{body}")
        if window is not None and window in arms and tasks.is_ruler(names[0]):
            sections.append(
                f"### Leakage check - {window}\n\n{leakage_table(arms[window], arms[baseline], names)}"
            )
    return "\n\n".join(sections)


def main() -> None:
    """Print the comparison tables of a records root."""
    parser = argparse.ArgumentParser(description="Paired comparison of benchmark records")
    parser.add_argument("records", help="the records root")
    parser.add_argument("--baseline", default="b16k-dense", help="the arm the others are compared with")
    parser.add_argument("--window", default="b16k-window", help="the window arm, for the RULER leakage check")
    parser.add_argument("--presets", default="core,ruler", help="presets to report, comma-separated")
    args = parser.parse_args()
    print(report(args.records, args.baseline, args.presets.split(","), args.window))
