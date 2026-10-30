"""Compose a hybrid model end to end from one layer of each kind and the modules between them.

    lsa kernels compose --config <checkpoint>/config.json \
        --modules <prefill run>/modules.json <decode run>/modules.json \
        --records <run>/record.json [more records...] --out composition.json
    lsa kernels compose --show composition.json

The sparse attention layer and the linear attention layer come from the attention-layer harness's records,
each one real layer inside sglang prefilled to the context. Everything else a layer runs - the feed-forward,
the mHC boundaries, the embedding, the final norm and the head - does not depend on the context and comes
from the module timer. The model is their sum, each weighted by how many of it the checkpoint config holds.

Prefill is cumulative - the time to prefill a prompt of the context, chunk by chunk from token zero. A
harness prefill point is the mean chunk over a window ending at the context, and a chunk's cost is close to
linear in its prefix, so each window mean is read as the cost of a chunk at the window's centre. The chunks
between two centres are interpolated linearly, the chunks before the first centre take the first mean, and
the chunks past the last centre follow the last segment. Only the points up to the context are used, so a
prefill to 1M is read off measurements no longer than 1M. The shared modules cost the same every chunk, and
the head runs once, on the last position.

Decode is one token at the context. Each layer takes its amortized time - an ordinary token's graph replay
plus its maintenance spread over the tokens it serves - and the shared modules take their graph replay.

What this leaves out is everything between layers that is not a kernel: host gaps, communication,
scheduling, and whether the whole model fits on one card at all.
"""

import argparse
import statistics
from collections.abc import Mapping
from typing import Any

from lsa import io
from lsa.config import text_config
from lsa.evaluation.constants import Attention, Glm, Kernels
from lsa.evaluation.performance.kernels.show import context_label

# The shared modules, in the order a table lists them
MODULES: tuple[str, ...] = ("moe", "dense", "boundary", "embedding", "final_norm", "head")
# What a composed step is split into, in the order a table lists them
PARTS: tuple[str, ...] = ("sparse attention", "linear attention", *MODULES)

# The checkpoint's name for its linear attention layers, beside `Glm.SPARSE_LAYER`
LINEAR_LAYER = "linear_attention"

Record = dict[str, Any]
Raw = dict[str, Any]


def layer_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Count how many of each composed part one forward runs.

    A decoder layer holds one attention - sparse or linear - one feed-forward - dense for the first
    `first_k_dense_replace` layers, experts after - and two boundaries, one around each of them. The
    prediction layers past `num_hidden_layers` are not counted, because a plain forward does not run them.

    Args:
        config: The checkpoint's config, or its text config.

    Returns:
        Part name to how many of it one forward runs.

    Raises:
        ValueError: A layer is neither of the two attention types this composes.
    """
    text = text_config(config)
    layers = text["num_hidden_layers"]
    types = text["layer_types"][:layers]
    unknown = sorted(set(types) - {Glm.SPARSE_LAYER, LINEAR_LAYER})
    if unknown:
        raise ValueError(f"layer types {unknown} are neither {Glm.SPARSE_LAYER} nor {LINEAR_LAYER}")
    dense = text["first_k_dense_replace"]
    return {
        "sparse attention": types.count(Glm.SPARSE_LAYER),
        "linear attention": types.count(LINEAR_LAYER),
        "moe": layers - dense,
        "dense": dense,
        "boundary": 2 * layers,
        "embedding": 1,
        "final_norm": 1,
        "head": 1,
    }


def module_costs(raw: Raw) -> dict[str, float]:
    """Return what each shared module costs per call in the phase the raw file timed.

    Prefill is the median eager call over CUDA events, the way the harness times a chunk. Decode is the
    median graph replay, the way the harness times a token.

    Args:
        raw: One module timer file.

    Returns:
        Module name to milliseconds per call - per chunk at prefill, per token at decode.

    Raises:
        ValueError: The file is not the module timer's, or lacks a module the composition needs.
    """
    if raw.get("schema") != Kernels.MODULES_SCHEMA:
        raise ValueError(f"schema {raw.get('schema')!r} - expected {Kernels.MODULES_SCHEMA!r}")
    missing = sorted(set(MODULES) - set(raw["modules"]))
    if missing:
        raise ValueError(f"{raw['phase']} module file lacks {missing}")
    key = "event_ms" if raw["phase"] == "prefill" else "graph_ms"
    return {name: statistics.median(raw["modules"][name][key]) for name in MODULES}


def window_curve(record: Record, context: int) -> list[tuple[float, float]]:
    """Read a prefill record as a chunk's cost against its prefix, up to one context.

    Args:
        record: A prefill record from the attention-layer harness.
        context: The longest context whose point is used.

    Returns:
        (window centre, mean chunk milliseconds) per point, ascending.
    """
    curve = []
    for name, point in record["points"].items():
        if int(name) <= context:
            first_end, last_end = point["window"]
            curve.append(((first_end + last_end) / 2, point["module_ms"]["mean"]))
    return sorted(curve)


def chunk_cost(curve: list[tuple[float, float]], end: int) -> float:
    """Return the cost of the chunk ending at a position, off the window curve.

    Args:
        curve: (window centre, mean chunk milliseconds), ascending.
        end: The position the chunk ends at.

    Returns:
        Milliseconds - flat before the first centre, linear between centres, and along the last segment
        after the last one.
    """
    if len(curve) == 1 or end <= curve[0][0]:
        return curve[0][1]
    for (left, left_ms), (right, right_ms) in zip(curve, curve[1:], strict=False):
        if end <= right:
            return left_ms + (right_ms - left_ms) * (end - left) / (right - left)
    (left, left_ms), (right, right_ms) = curve[-2], curve[-1]
    return right_ms + (right_ms - left_ms) * (end - right) / (right - left)


def cumulative_prefill_ms(record: Record, context: int, chunk: int) -> float:
    """Return the time one layer takes to prefill a prompt of the context from token zero.

    Args:
        record: A prefill record from the attention-layer harness, holding a point at the context.
        context: The prompt length.
        chunk: The prefill chunk.

    Returns:
        Milliseconds, summed over every chunk up to the context.
    """
    curve = window_curve(record, context)
    return sum(chunk_cost(curve, end) for end in range(chunk, context + 1, chunk))


def decode_step_ms(record: Record, context: int) -> float:
    """Return one layer's amortized decode token at a context.

    Raises:
        ValueError: The point has no amortized time, which needs an ordinary token replayed as a graph.
    """
    amortized = record["points"][str(context)].get("amortized_ms")
    if amortized is None:
        raise ValueError(f"{record['run']['id']}: no amortized decode time at {context}")
    return amortized


def prefill_parts(
    attention: Record,
    linear: Record,
    modules: dict[str, float],
    counts: dict[str, int],
    context: int,
    chunk: int,
) -> dict[str, float]:
    """Split one configuration's composed prefill at one context into its parts.

    Args:
        attention: The configuration's sparse attention prefill record.
        linear: The linear attention prefill record.
        modules: Each shared module's milliseconds per chunk.
        counts: What `layer_counts` returned.
        context: The prompt length.
        chunk: The prefill chunk.

    Returns:
        Part name to milliseconds over the whole prefill.
    """
    chunks = context // chunk
    parts = {
        "sparse attention": counts["sparse attention"] * cumulative_prefill_ms(attention, context, chunk),
        "linear attention": counts["linear attention"] * cumulative_prefill_ms(linear, context, chunk),
    }
    for name in MODULES:
        # The logits of the last position only
        calls = 1 if name == "head" else chunks
        parts[name] = counts[name] * modules[name] * calls
    return parts


def decode_parts(
    attention: Record, linear: Record, modules: dict[str, float], counts: dict[str, int], context: int
) -> dict[str, float]:
    """Split one configuration's composed decode token at one context into its parts.

    Args:
        attention: The configuration's sparse attention decode record.
        linear: The linear attention decode record.
        modules: Each shared module's milliseconds per token.
        counts: What `layer_counts` returned.
        context: The context the token is generated at.

    Returns:
        Part name to milliseconds for the token.
    """
    parts = {
        "sparse attention": counts["sparse attention"] * decode_step_ms(attention, context),
        "linear attention": counts["linear attention"] * decode_step_ms(linear, context),
    }
    for name in MODULES:
        parts[name] = counts[name] * modules[name]
    return parts


def configuration_of(record: Record) -> str:
    """Name the configuration a record times - `native` or `full` for a path with no indexer to describe.

    Args:
        record: A harness record.

    Returns:
        The setup's indexer for `native` and `full`, which the baseline is read by, else the run's name.
    """
    indexer = record["setup"]["indexer"]
    return indexer if indexer in (Attention.NATIVE, Attention.FULL) else str(record["run"]["name"])


def split_records(records: Mapping[str, Record]) -> tuple[dict[str, Record], dict[tuple[str, str], Record]]:
    """Separate the linear attention records from the configurations', each keyed by what it is.

    Args:
        records: Path to record.

    Returns:
        The linear attention record of each phase, and each configuration's record keyed by (label, phase).

    Raises:
        ValueError: A record is not the harness's, or two records claim one slot.
    """
    linear: dict[str, Record] = {}
    configurations: dict[tuple[str, str], Record] = {}
    for path, record in records.items():
        if record.get("schema") != Kernels.SCHEMA:
            raise ValueError(f"{path}: schema {record.get('schema')!r} - expected {Kernels.SCHEMA!r}")
        phase = record["run"]["phase"]
        if record["setup"]["indexer"] == Attention.KDA:
            if phase in linear:
                raise ValueError(f"{path}: a second linear attention {phase} record")
            linear[phase] = record
            continue
        key = (configuration_of(record), phase)
        if key in configurations:
            raise ValueError(f"{path}: a second {key[0]} {phase} record")
        configurations[key] = record
    return linear, configurations


def speedups(totals: dict[str, dict[str, float]], baseline: str) -> dict[str, dict[str, float]]:
    """Read every configuration's totals against the baseline's, at the contexts both have.

    Args:
        totals: Configuration label to context to total.
        baseline: The label every configuration is read against.

    Returns:
        Configuration label to context to baseline time over configuration time - above one is faster than
        the baseline.
    """
    base = totals.get(baseline, {})
    return {
        label: {
            context: round(base[context] / value, 4)
            for context, value in by_context.items()
            if context in base
        }
        for label, by_context in totals.items()
    }


def file_digest(path: str) -> str:
    """Return the first twelve hex digits of a file's sha256.

    Raises:
        FileNotFoundError: There is no file at the path.
    """
    digest = io.file_sha256(path)
    if digest is None:
        raise FileNotFoundError(path)
    return digest[:12]


def card_uuids(record: Record) -> list[str]:
    """Return the cards a record's points were measured on."""
    return sorted({point["gpu_uuid"] for point in record["points"].values() if point.get("gpu_uuid")})


def compose(
    config: Mapping[str, Any], modules: Mapping[str, Raw], records: Mapping[str, Record], baseline: str
) -> dict[str, Any]:
    """Compose every configuration the records hold into a whole-model prefill and decode.

    Args:
        config: The full checkpoint's config - the layer counts come from it, not from the truncated cut.
        modules: Module timer file per phase.
        records: Path to harness record, the configurations' and the linear attention's.
        baseline: The configuration label every speedup is read against.

    Returns:
        The composition, with its parts, totals, speedups and the cards behind them.

    Raises:
        ValueError: A phase has a configuration but no linear attention record or no module file.
    """
    counts = layer_counts(config)
    linear, configurations = split_records(records)
    body: dict[str, Any] = {"counts": counts, "baseline": baseline, "phases": {}}
    for phase in ("prefill", "decode"):
        labels = sorted(label for label, record_phase in configurations if record_phase == phase)
        if not labels:
            continue
        if phase not in linear or phase not in modules:
            raise ValueError(
                f"{phase} has configurations {labels} but no linear attention record or no module file"
            )
        costs = module_costs(modules[phase])
        linear_record = linear[phase]
        chunk = linear_record["setup"]["chunk"]
        parts: dict[str, dict[str, dict[str, float]]] = {}
        totals: dict[str, dict[str, float]] = {}
        cards = set(card_uuids(linear_record)) | {modules[phase]["gpu_uuid"]}
        for label in labels:
            record = configurations[(label, phase)]
            cards |= set(card_uuids(record))
            contexts = sorted(set(map(int, record["points"])) & set(map(int, linear_record["points"])))
            parts[label] = {}
            totals[label] = {}
            for context in contexts:
                split = (
                    prefill_parts(record, linear_record, costs, counts, context, chunk)
                    if phase == "prefill"
                    else decode_parts(record, linear_record, costs, counts, context)
                )
                parts[label][str(context)] = {name: round(value, 4) for name, value in split.items()}
                totals[label][str(context)] = round(sum(split.values()), 4)
        body["phases"][phase] = {
            "measure": "the whole prompt from token zero"
            if phase == "prefill"
            else "one token at the context",
            "chunk": chunk,
            "modules_ms": {name: round(value, 5) for name, value in costs.items()},
            "parts_ms": parts,
            "total_ms": totals,
            "speedup": speedups(totals, baseline),
            "cards": sorted(cards),
            "same_card": len(cards) == 1,
        }
    return body


def build(
    config_path: str, module_paths: list[str], record_paths: list[str], baseline: str
) -> dict[str, Any]:
    """Compose from files on disk and name every input by path and digest.

    Args:
        config_path: The full checkpoint's `config.json`.
        module_paths: Module timer files, one per phase.
        record_paths: Harness records, the configurations' and the linear attention's.
        baseline: The configuration every speedup is read against.

    Returns:
        The composition record.
    """
    modules = {raw["phase"]: raw for raw in (io.read_json(path) for path in module_paths)}
    records = {path: io.read_json(path) for path in record_paths}
    return {
        "schema": Kernels.COMPOSITION_SCHEMA,
        "date": io.utc_now(),
        "sources": {
            "config": {"path": config_path, "sha256": file_digest(config_path)},
            "modules": [{"path": path, "sha256": file_digest(path)} for path in module_paths],
            "records": [
                {
                    "path": path,
                    "sha256": file_digest(path),
                    "label": configuration_of(record),
                    "phase": record["run"]["phase"],
                    "commits": record["run"].get("commits"),
                }
                for path, record in records.items()
            ],
        },
        **compose(io.read_json(config_path), modules, records, baseline),
    }


def show(composition: dict[str, Any]) -> None:
    """Print each phase's totals with their speedups, then each configuration's parts."""
    baseline = composition["baseline"]
    for phase, block in composition["phases"].items():
        scale, unit = (1000.0, "s") if phase == "prefill" else (1.0, "ms")
        labels = [baseline, *sorted(set(block["total_ms"]) - {baseline})]
        labels = [label for label in labels if label in block["total_ms"]]
        same = "one card" if block["same_card"] else f"{len(block['cards'])} cards"
        print(f"{phase} - {block['measure']}, in {unit}, measured on {same}")
        print(f"{'context':>8} " + " ".join(f"{label:>26}" for label in labels))
        contexts = sorted({int(c) for label in labels for c in block["total_ms"][label]})
        for context in contexts:
            cells = []
            for label in labels:
                total = block["total_ms"][label].get(str(context))
                speedup = block["speedup"][label].get(str(context))
                if total is None:
                    cells.append(f"{'-':>26}")
                    continue
                ratio = f" x{speedup:5.2f}" if speedup is not None else ""
                cells.append(f"{total / scale:12.4f} {unit}{ratio}".rjust(26))
            print(f"{context_label(context):>8} " + " ".join(cells))
        for label in labels:
            print(f"  {label} parts, share of the total")
            print(f"  {'context':>8} " + " ".join(f"{part:>16}" for part in PARTS))
            for context, parts in block["parts_ms"][label].items():
                total = block["total_ms"][label][context]
                shares = " ".join(f"{100 * parts[part] / total:15.1f}%" for part in PARTS)
                print(f"  {context_label(context):>8} {shares}")
        print()


def main() -> None:
    """Build a composition from files, or print one."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", help="print this composition and do nothing else")
    parser.add_argument("--config", help="the full checkpoint's config.json")
    parser.add_argument("--modules", nargs="+", default=[], help="module timer files, one per phase")
    parser.add_argument(
        "--records", nargs="+", default=[], help="harness records, configurations and linear attention"
    )
    parser.add_argument("--baseline", default=Attention.NATIVE)
    parser.add_argument("--out", help="where the composition goes")
    args = parser.parse_args()
    if args.show:
        show(io.read_json(args.show))
        return
    if not (args.config and args.modules and args.records and args.out):
        parser.error("building needs --config, --modules, --records and --out")
    composition = build(args.config, args.modules, args.records, args.baseline)
    io.write_json(args.out, composition, indent=1)
    show(composition)
    print(args.out)
