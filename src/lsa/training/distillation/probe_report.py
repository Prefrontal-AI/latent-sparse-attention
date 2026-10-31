"""The tables of a served read over probe prompts - an arm by a task and a length, with the error taken over prompts.

A probe record keeps each scored row's kind, how many long-range keys it looks for and how far back the nearest is, and
every read's metrics a row (`lsa distill served-read`).
A cell is made the way the prompt is the unit: the rows of a (prompt, layer) are averaged first, a prompt's layers next,
and the prompts of a task and length last, as a mean and its standard error.
A row with nothing to find leaves a metric out of its prompt's mean, so a prompt with none at all leaves the cell.

| table | rows | columns |
|---|---|---|
| a metric, one kind of query | the reads | each task at each length |
| `target_held` on the list task | the reads | the rows split by the number of targets, or by the distance back to the nearest |
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from lsa.training.distillation.constants import Probe, Served

# The metrics a table of tasks and lengths is made of, in order
TASK_METRICS: tuple[str, ...] = (
    "target_held",
    "target_any",
    "line_held",
    "mass",
    "recall",
    "native_overlap",
    "agreement",
    *Served.COST_METRICS,
)
# The metric the list task's tables split by the number of targets and by distance
SPLIT_METRIC = "target_held"

# Picks the indices of the rows of a record a cell reads
Selector = Callable[[dict[str, Any]], list[int]]


def length_label(length: int) -> str:
    """Name a length as the extraction does - 16k up to 1M."""
    return f"{length >> 20}M" if length >= 1 << 20 else f"{length >> 10}k"


def mean(values: Iterable[float]) -> float:
    """Average what is a number, NaN when nothing is."""
    kept = [value for value in values if not math.isnan(value)]
    return sum(kept) / len(kept) if kept else math.nan


def standard_error(values: Sequence[float]) -> float:
    """Return the standard error of the mean, NaN for fewer than two values."""
    if len(values) < 2:
        return math.nan
    centre = sum(values) / len(values)
    variance = sum((value - centre) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def edge_label(value: int) -> str:
    """Name a bucket's edge - 4096 is `4k`."""
    return f"{value >> 10}k" if value >= 1024 else str(value)


def bucket_labels(edges: Sequence[int], count_edges: bool) -> list[str]:
    """Name the buckets a tuple of lower edges makes, the last one open.

    Args:
        edges: Each bucket's lower edge, ascending.
        count_edges: Whether the edges are counts - `2-3` - or distances - `4k-16k`.

    Returns:
        One label a bucket.
    """
    labels = []
    for i, low in enumerate(edges):
        if i + 1 == len(edges):
            labels.append(f"{edge_label(low)}+")
        elif count_edges:
            high = edges[i + 1] - 1
            labels.append(str(low) if high == low else f"{low}-{high}")
        else:
            labels.append(f"{edge_label(low)}-{edge_label(edges[i + 1])}")
    return labels


def bucket_of(value: int, edges: Sequence[int]) -> int:
    """Return the bucket a value falls in - the last whose lower edge it reaches, -1 under the first."""
    return max((i for i, low in enumerate(edges) if value >= low), default=-1)


def kind_rows(record: dict[str, Any], kind: str) -> list[int]:
    """Return the indices of the rows of a record that are one kind of query."""
    names = record["probe"]["query_kinds"]
    return [i for i, index in enumerate(record["probe"]["rows"]["query_kind"]) if names[index] == kind]


def split_rows(record: dict[str, Any], kind: str, field: str, edges: Sequence[int], bucket: int) -> list[int]:
    """Return the indices of the rows of one kind of a record whose `field` falls in a bucket, and which have a target."""
    values = record["probe"]["rows"][field]
    counts = record["probe"]["rows"]["target_count"]
    return [i for i in kind_rows(record, kind) if counts[i] > 0 and bucket_of(values[i], edges) == bucket]


def kind_column(kind: str) -> Callable[[str], Selector]:
    """Make the selector of a column that reads every row of one kind of query."""
    return lambda _: lambda record: kind_rows(record, kind)


def bucket_column(
    kind: str, field: str, edges: Sequence[int], labels: Sequence[str]
) -> Callable[[str], Selector]:
    """Make the selector of a column that is a bucket - the rows of one kind whose `field` falls in the bucket it is named for."""
    return lambda label: lambda record: split_rows(record, kind, field, edges, labels.index(label))


def cell(
    records: Sequence[dict[str, Any]], read: str, metric: str, select: Selector
) -> tuple[float, float, int]:
    """Make one cell - a metric of one read over the prompts of some records, the prompt the unit.

    Args:
        records: The probe records of one task and length.
        read: The read.
        metric: The metric.
        select: Picks the rows of a record the cell reads.

    Returns:
        The mean over prompts, its standard error, and the prompts it holds - NaN and 0 when none has a value.
    """
    layers: dict[str, list[float]] = {}
    for record in records:
        values = record["probe"]["reads"].get(read, {}).get(metric)
        if values is None:
            continue
        layers.setdefault(record["document"], []).append(mean(values[i] for i in select(record)))
    prompts = [value for value in (mean(each) for each in layers.values()) if not math.isnan(value)]
    if not prompts:
        return math.nan, math.nan, 0
    return sum(prompts) / len(prompts), standard_error(prompts), len(prompts)


def render(mean: float, error: float) -> str:
    """Print a cell as `mean ± error`, `-` where there is no mean."""
    if math.isnan(mean):
        return "-"
    return f"{mean:.3f}" if math.isnan(error) else f"{mean:.3f} ± {error:.3f}"


def table(
    columns: Mapping[str, Sequence[dict[str, Any]]],
    reads: Sequence[str],
    metric: str,
    selector_of: Callable[[str], Selector],
    form: Callable[[float, float], str] = render,
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    """Render one table of cells, a read a row, and leave out a read that has no value in any column.

    Args:
        columns: The records of each column, by its heading.
        reads: The rows, in order.
        metric: The metric.
        selector_of: Makes the row selector of a column, from its heading.
        form: How a cell is printed.

    Returns:
        The table's lines, empty when no cell has a value, and each cell's mean, error and prompts by read and column.
    """
    numbers: dict[str, dict[str, dict[str, float]]] = {}
    for read in reads:
        cells = {name: cell(records, read, metric, selector_of(name)) for name, records in columns.items()}
        if all(math.isnan(mean) for mean, _, _ in cells.values()):
            continue
        numbers[read] = {
            name: {"mean": mean, "se": error, "prompts": count}
            for name, (mean, error, count) in cells.items()
        }
    if not numbers:
        return [], {}
    lines = ["| read | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for read, cells_of in numbers.items():
        lines.append(f"| {read} | " + " | ".join(form(c["mean"], c["se"]) for c in cells_of.values()) + " |")
    return lines, numbers


def probe_tables(records: Sequence[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Build the tables of every probe record - by task and length for each kind of query, and the list task split.

    Args:
        records: Records, of which those with a `probe` block are read.

    Returns:
        The report's lines, empty when no record is a probe's, and the numbers behind them.
    """
    probes = [record for record in records if "probe" in record]
    if not probes:
        return [], {}
    reads = list(dict.fromkeys(read for record in probes for read in record["probe"]["reads"]))
    kinds = list(dict.fromkeys(name for record in probes for name in record["probe"]["query_kinds"]))
    columns: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(probes, key=lambda record: (record["length"], record["task"])):
        columns.setdefault(f"{record['task']} {length_label(record['length'])}", []).append(record)
    counts = ", ".join(
        f"{name}: {len({r['document'] for r in records_of})}" for name, records_of in columns.items()
    )
    lines = [
        "",
        "# Probes",
        "",
        "A cell is the mean over prompts of a prompt's mean - its rows first, then its layers - with its standard error.",
        f"Prompts a column: {counts}.",
    ]
    summary: dict[str, Any] = {}
    for kind in kinds:
        for metric in TASK_METRICS:
            body, numbers = table(columns, reads, metric, kind_column(kind))
            if body:
                summary.setdefault(kind, {})[metric] = numbers
                lines += ["", f"## {metric} by task and length - {kind} queries", "", *body]
    lines += split_tables(columns, reads, kinds, summary)
    return lines, summary


def split_tables(
    columns: Mapping[str, Sequence[dict[str, Any]]],
    reads: Sequence[str],
    kinds: Sequence[str],
    summary: dict[str, Any],
) -> list[str]:
    """Build the list task's `target_held` tables, the rows split by the number of targets and by the distance back.

    Args:
        columns: The records of each task and length.
        reads: The rows, in order.
        kinds: The kinds of query.
        summary: The numbers, added to.

    Returns:
        The tables' lines.
    """
    lines: list[str] = []
    splits = (
        ("target count", "target_count", Served.TARGET_COUNT_EDGES, True),
        ("distance back to the nearest target", "target_distance", Served.DISTANCE_EDGES, False),
    )
    for heading, field, edges, count_edges in splits:
        labels = bucket_labels(edges, count_edges)
        for name, records in columns.items():
            if not name.startswith(f"{Probe.LIST_TASK} "):
                continue
            for kind in kinds:
                by_bucket = {label: records for label in labels}
                body, numbers = table(
                    by_bucket,
                    reads,
                    SPLIT_METRIC,
                    bucket_column(kind, field, edges, labels),
                )
                if body:
                    summary.setdefault("splits", {}).setdefault(field, {}).setdefault(name, {})[kind] = (
                        numbers
                    )
                    lines += ["", f"## {SPLIT_METRIC} by {heading} - {name}, {kind} queries", "", *body]
    return lines
