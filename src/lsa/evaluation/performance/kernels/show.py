"""Lay out kernel records as a table - what `lsa records show` prints for them.

One record: every context with the layer time, its parts, memory and the checks.
Several records: the contexts aligned, each configuration's layer time and its speedup over the first.
"""

from typing import Any

Record = dict[str, Any]
# What a decode point says of its graph replay - None for a decode that runs eagerly, as a descending decode of shaped node buckets does
GRAPH_VERDICTS = {True: "ok", False: "unvalidated", None: "none (eager)"}
Point = dict[str, Any]


def context_label(context: int | str) -> str:
    """Name a context the way the page does - 2M, 512K."""
    tokens = int(context)
    return f"{tokens // 2**20}M" if tokens % 2**20 == 0 else f"{tokens // 1024}K"


def layer_gib(memory: dict[str, int]) -> float:
    """Return the layer's cache and index at the context plus its forward's working memory, in GiB."""
    return (memory["kv_bytes"] + memory["index_bytes"] + memory["working_bytes"]) / 2**30


def topk(checks: dict[str, Any], count: int) -> str:
    """Say how many measured points had an exact top-k, or that there was no selection to check."""
    return "topk n/a" if checks["topk_exact"] is None else f"topk exact {checks['topk_exact']}/{count}"


def selection(checks: dict[str, Any]) -> str:
    """Say whether every rerun of a chunk chose the selection its preparation did, and how many rows differed."""
    exact, differing = checks.get("selection_exact"), checks.get("rows_differing")
    if exact is None:
        return "selection n/a"
    return f"selection {'exact' if exact else 'DIFFERS'} ({differing} rows differ)"


def primary(point: Point, phase: str) -> float:
    """Return the time a user pays per call at this context.

    Prefill is the mean chunk of the window. Decode is an ordinary token replayed as a graph, which is
    how it is served - falling back to eager for a run that recorded no graph, as a shaped node bucket design does.

    Args:
        point: One context's point.
        phase: Which half of serving this record measured.

    Returns:
        Milliseconds.
    """
    if phase == "prefill":
        return point["module_ms"]["mean"]
    ordinary = point["cases"].get("ordinary") or {}
    return ordinary.get("graph_ms") or ordinary.get("eager_ms") or point["eager_ms"]["mean"]


def prefill_line(context: str, point: Point) -> str:
    """Return one context's prefill line."""
    parts, checks, module = point["components_ms"], point["checks"], point["module_ms"]
    chunks = len(point["chunks"])
    growth = f"{point['growth_ms']:8.3f}" if point["growth_ms"] is not None else f"{'-':>8}"
    return (
        f"{context_label(context):>6} {module['mean']:9.3f} ms ±{module['std']:<7.3f} chunks {chunks:>2}  "
        f"ordinary {point['ordinary_ms'] or 0:8.3f}  growth {growth}  "
        f"scan {parts['index q scan'] or 0:8.3f} select {parts['index k select'] or 0:7.3f} "
        f"attention {parts['topk attention'] or 0:7.3f}  "
        f"memory {layer_gib(point['memory']):6.1f} GiB  {topk(checks, chunks)}  {selection(checks)}"
    )


def decode_line(context: str, point: Point) -> str:
    """Return one context's decode line."""
    cases, checks = point["cases"], point["checks"]
    ordinary = cases.get("ordinary") or {}
    graph = f"{ordinary['graph_ms']:8.4f}" if ordinary.get("graph_ms") is not None else f"{'-':>8}"
    amortized = f"{point['amortized_ms']:8.4f}" if point.get("amortized_ms") is not None else f"{'-':>8}"
    tokens = len(point["tokens"])
    return (
        f"{context_label(context):>6} graph {graph} ms  amortized {amortized}  "
        f"eager {ordinary.get('eager_ms') or 0:8.4f}  metadata {ordinary.get('metadata_ms') or 0:8.4f}  "
        f"tokens {tokens:>2}  "
        f"memory {layer_gib(point['memory']):6.1f} GiB  {topk(checks, tokens)}  {selection(checks)}  "
        f"graph {GRAPH_VERDICTS[checks['graph_validated']]}  "
        f"graph slots differing {checks.get('graph_selection_differing_slots', '-')}"
    )


def show(record: Record) -> None:
    """Print one record - a header line, then a line per context."""
    run = record["run"]
    print(
        f"{run['id']}  {record['model']['name']}  {run['name']}  {run['phase']}  "
        f"{run['date']}  {run['seconds']} s  {run['gpu']}"
    )
    if run.get("note"):
        print(f"  note: {run['note']}")
    line = prefill_line if run["phase"] == "prefill" else decode_line
    for context, point in record["points"].items():
        print("  " + line(context, point))
    if record.get("failed"):
        print(f"  failed: {record['failed']}")


def compare(records: list[Record]) -> None:
    """Print several records side by side, each context on one line.

    The first record is the baseline every other column is read against.

    Args:
        records: The records to align, baseline first.
    """
    phase = records[0]["run"]["phase"]
    contexts = sorted({int(context) for record in records for context in record["points"]})
    labels = [record["run"]["name"] for record in records]
    print(f"{'context':>8} " + " ".join(f"{label:>24}" for label in labels))
    for context in contexts:
        cells: list[str] = []
        base = None
        for record in records:
            point = record["points"].get(str(context))
            if point is None:
                cells.append(f"{'-':>24}")
                continue
            value = primary(point, phase)
            base = value if base is None else base
            speedup = f" x{base / value:5.2f}" if value else ""
            cells.append(f"{value:10.4f} ms{speedup}".rjust(24))
        print(f"{context_label(context):>8} " + " ".join(cells))
