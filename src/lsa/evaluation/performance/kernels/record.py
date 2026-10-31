"""The record of a kernel run - one JSON per run, the harness's raw output beside it.

    <results>/performance/kernels/<run_id>/record.json              everything below
    <results>/performance/kernels/<run_id>/samples/<context>*.json  one raw harness file per measured chunk or token

The raw files are written by [the attention-layer harness](measure/layer.py), one
process per measured point, and this is the only thing that reads them. It aggregates rather than
measures: every number here is a mean or a max over trials the harness timed.

record.json
  schema, run {id, name, phase, note, tag, started, date, seconds, host, gpu, commits, diff, harness, versions}
  model {name, label, path}
  setup {indexer, approximate_selection, config, layer, tp_size, chunk, prefill_backend,
         decode_backend, backend_classes, backend_source, cuda_graph, warmup, repeats, window_chunks | offsets, prompt, image,
         mesh - one GPU's, the harness being one process}
  points {context: prefill - window, growth_chunks, module_ms, step_ms, ordinary_ms, growth_ms,
                             chunks [{end, growth, module_ms, instrumented_ms, components_ms}]
                   decode  - graph_ms, free_indexer_graph_ms, indexer_only_graph_ms, graph_indexer_ms, eager_ms,
                             metadata_ms, step_ms, cases {case: tokens, graph_ms, free_indexer_graph_ms,
                             indexer_only_graph_ms, graph_indexer_ms, eager_ms, metadata_ms}, amortized_ms,
                             maintenance_ms, metadata_components_ms, tokens [{seq, offset, case, graph_ms,
                             free_indexer_graph_ms, indexer_only_graph_ms, graph_indexer_ms, eager_ms, metadata_ms,
                             instrumented_ms, components_ms, metadata_components_ms}]
                   both    - instrumented_ms, components_ms, memory, checks, gpu_uuid, hardware, samples}
  failed  contexts whose harness process did not finish

`hardware` is {host, driver, telemetry {hosts, gpus {uuid: {host, index, samples, sm_mhz_min, sm_mhz_median,
mem_mhz_median, power_w_max, power_limit_w, temperature_c_max, power_capped_fraction, slowdown_fraction, slowdown_reasons, throttle_reasons}}, files}}.
The telemetry is `nvidia-smi` sampled every 100 ms over the timed trials, its rows in `samples/` beside the raw
files as `<name>.gpu-<host>.csv`. It is a fact of the point and not of the setup, so parts measured on different
hosts still join.

The chunk and token rows are what a click in the viewer opens, and every trial behind them stays in
`samples/`.
"""

import glob
import hashlib
import os
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.accuracy.records import mesh
from lsa.evaluation.constants import Kernels, Mesh
from lsa.evaluation.performance import telemetry
from lsa.evaluation.performance.kernels.measure.recorder import COMPONENTS as MEASURED
from lsa.evaluation.performance.kernels.measure.selection import weigh

# What the recorder charges a module's time to - its eight ranges plus the remainder
COMPONENTS: tuple[str, ...] = (*MEASURED, "other")

Raw = dict[str, Any]
Sample = dict[str, Any]


def stats(values: Iterable[float]) -> dict[str, float] | None:
    """Summarise the trials behind one number.

    Args:
        values: One value per trial.

    Returns:
        Mean, population standard deviation, extremes and the trial count, or None for no trials.
    """
    trials = list(values)
    if not trials:
        return None
    return {
        "mean": round(statistics.fmean(trials), 4),
        "std": round(statistics.pstdev(trials), 4) if len(trials) > 1 else 0.0,
        "min": round(min(trials), 4),
        "max": round(max(trials), 4),
        "n": len(trials),
    }


def mean(values: Iterable[float]) -> float | None:
    """Return the mean of some trials, or None when there are none."""
    trials = list(values)
    return round(statistics.fmean(trials), 4) if trials else None


def sample_paths(run_dir: Path, context: int) -> list[Path]:
    """Return the raw harness files of one context, in name order.

    A prefill window and a decode probe both write `<context>-<position>.json`, and a single measured
    point writes `<context>.json`.

    Args:
        run_dir: The run directory.
        context: The context the point is reported at.

    Returns:
        Every raw file belonging to that context.
    """
    stem = str(run_dir / "samples" / str(context))
    return [Path(path) for path in sorted(glob.glob(f"{stem}.json") + glob.glob(f"{stem}-*.json"))]


def contexts_of(run_dir: Path) -> list[int]:
    """Return the contexts a run directory holds samples for, ascending."""
    folder = run_dir / "samples"
    names = os.listdir(folder) if folder.is_dir() else []
    return sorted({int(name.split("-")[0].split(".")[0]) for name in names if name.endswith(".json")})


def read_samples(run_dir: Path, context: int) -> list[Raw]:
    """Read one context's raw harness files, ordered by the position each measured.

    Args:
        run_dir: The run directory.
        context: The context the point is reported at.

    Returns:
        The parsed files, ascending by the position they measured.

    Raises:
        ValueError: A file is not what this harness writes.
    """
    raws = []
    for path in sample_paths(run_dir, context):
        raw = io.read_json(path)
        if raw.get("schema") != Kernels.RAW_SCHEMA:
            raise ValueError(f"{path}: schema {raw.get('schema')!r} - expected {Kernels.RAW_SCHEMA!r}")
        raws.append(raw)
    return sorted(raws, key=lambda raw: raw["args"]["context"])


def trials(raws: Iterable[Raw], mode: str) -> list[Sample]:
    """Return every trial of one timing mode across some raw files."""
    return [sample for raw in raws for sample in raw["samples"] if sample["mode"] == mode]


def graph_trials(raws: Iterable[Raw], mode: str = "graph") -> list[Sample]:
    """Return every graph-replay trial of one mode, which the harness records beside the eager ones.

    Args:
        raws: The raw files.
        mode: `graph` for the layer's own step, `free_indexer` for the step with its indexer doing no work,
            `indexer_only` for its indexer's call alone.
    """
    return [
        sample
        for raw in raws
        for sample in (raw["graph_comparison"] or {}).get("samples", [])
        if sample["mode"] == mode
    ]


def graph_indexer_ms(raws: Iterable[Raw]) -> float | None:
    """Return what the indexer adds to a graph-replayed step - the full replay less the free indexer's.

    Each repeat times both replays, so the difference is taken per repeat and its median kept.

    Args:
        raws: The raw files.

    Returns:
        Milliseconds, or None where no free-indexer replay was timed.
    """
    differences = []
    for raw in raws:
        samples = (raw["graph_comparison"] or {}).get("samples", [])
        full = {sample["repeat"]: sample["module_ms"] for sample in samples if sample["mode"] == "graph"}
        free = {
            sample["repeat"]: sample["module_ms"] for sample in samples if sample["mode"] == "free_indexer"
        }
        differences += [full[repeat] - free[repeat] for repeat in full.keys() & free.keys()]
    return round(statistics.median(differences), 5) if differences else None


def components(samples: Iterable[Sample], key: str = "components") -> dict[str, float | None]:
    """Average each component's exclusive time over some instrumented trials.

    Args:
        samples: Trials timed with component events.
        key: Which range set to read - the module's own, or the metadata the step opens outside it.

    Returns:
        One mean per component, in the recorder's order.
    """
    rows = list(samples)
    return {name: mean(sample[key]["exclusive_ms"][name] for sample in rows) for name in COMPONENTS}


def memory(raws: list[Raw], samples: Iterable[Sample]) -> dict[str, int]:
    """Describe what the layer holds at a context and what its forward allocates on top.

    `index_bytes` is the index the variant actually reads - GLM's own for `native`, and the design's index keys,
    bucket cache and node buckets for `lsa`, and the leaf means for `pisa`.

    Args:
        raws: The context's raw files.
        samples: The trials to take the peaks from.

    Returns:
        The peaks, the working memory, and the two resident buffers.
    """
    rows = list(samples)
    last, indexer = raws[-1]["memory"], raws[-1]["args"]["indexer"]
    return {
        "peak_allocated_bytes": max(sample["peak_allocated_bytes"] for sample in rows),
        "peak_reserved_bytes": max(sample["peak_reserved_bytes"] for sample in rows),
        # What the forward allocates over what was resident - the score workspace, kernel scratch
        "working_bytes": max(
            sample["peak_allocated_bytes"] - sample["allocated_before_bytes"] for sample in rows
        ),
        "kv_bytes": last["kv_capacity_bytes"],
        "index_bytes": {
            "native": last["native_index_capacity_bytes"],
            "lsa": last.get("lsa_index_bytes", 0),
            "pisa": last.get("pisa_index_bytes", 0),
        }.get(indexer, 0),
    }


def exact(outputs: Any) -> bool:
    """Return whether every output tensor matched the reference - an absent one counts as matched."""
    parts = outputs if isinstance(outputs, list) else [outputs]
    return all(part is None or part["exact"] for part in parts)


def passed(values: Iterable[bool | None]) -> bool | None:
    """Reduce checks to one verdict, staying None when a check did not apply.

    A layer without an index has no scores to check, and calling that a pass would report a check that
    never ran as a check that succeeded.

    Args:
        values: One verdict per raw file, or None where the check did not apply.

    Returns:
        None if any did not apply, otherwise whether they all passed.
    """
    verdicts = list(values)
    return None if any(value is None for value in verdicts) else all(verdicts)


def checks(raws: list[Raw], samples: Iterable[Sample]) -> dict[str, Any]:
    """Collect what the harness validated at one context.

    Args:
        raws: The context's raw files.
        samples: The trials to count allocator trouble over.

    Returns:
        The output, state, score and selection verdicts, None for a run timed without its checks, and the
        allocator retries behind them. `selection_exact` is whether every rerun of a chunk chose the prepared
        selection on every row - `rows_differing` counts the rows that did not.
    """
    rows = list(samples)
    checked = [raw["validation"] for raw in raws]
    retries = sum(sample["allocator_retries"] + sample["allocator_ooms"] for sample in rows)
    # A timed-only run validated nothing, so it has no verdict to give
    if any(check.get("time_only") for check in checked):
        return {
            "output_exact": None,
            "state_exact": None,
            "scores_exact": None,
            "topk_exact": None,
            "selection_exact": None,
            "rows_differing": None,
            "allocator_retries": retries,
        }
    topk = [check["topk_cutoff_valid"] for check in checked]
    # Weighed here and not as the harness wrote them, so the bar can move without measuring again
    reruns = [
        weigh(raw["validation"]["selection_rows"]) for raw in raws if "selection_rows" in raw["validation"]
    ]
    return {
        "output_exact": all(exact(check["reference_order_output"]) for check in checked),
        "state_exact": all(check["final_cache_and_index_state_exact"] for check in checked),
        "scores_exact": passed(
            check["checked_score_rows_exact"] and check["independent_score_oracle"]["passed"]
            for check in checked
        ),
        "topk_exact": None if any(value is None for value in topk) else sum(bool(v) for v in topk),
        "selection_exact": all(rerun["exact"] for rerun in reruns) if reruns else None,
        "rows_differing": sum(rerun["differing"] for rerun in reruns) if reruns else None,
        "allocator_retries": retries,
    }


def chunk_row(raw: Raw) -> dict[str, Any]:
    """Summarise one prefill chunk - the row a click in the viewer opens."""
    plain, detailed = trials([raw], "plain"), trials([raw], "components")
    return {
        "end": raw["args"]["context"],
        "growth": bool(raw.get("hierarchy_growth")),
        "module_ms": mean(sample["module_ms"] for sample in plain),
        "instrumented_ms": mean(sample["module_ms"] for sample in detailed),
        "components_ms": components(detailed),
    }


def prefill_point(raws: list[Raw]) -> dict[str, Any]:
    """Reduce a prefill window to the point one context reports.

    The window is the last chunks up to the context, so the mean is what a user pays per chunk there, and the
    chunks that grow the hierarchy are called out because they are the ones that pair buckets into nodes.

    Args:
        raws: The context's raw files, ascending by the chunk each measured.

    Returns:
        The point.
    """
    plain, detailed = trials(raws, "plain"), trials(raws, "components")
    growth = [raw for raw in raws if raw.get("hierarchy_growth")]
    ordinary = [raw for raw in raws if not raw.get("hierarchy_growth")]
    return {
        "window": [raws[0]["args"]["context"], raws[-1]["args"]["context"]],
        "growth_chunks": len(growth),
        "module_ms": stats(sample["module_ms"] for sample in plain),
        "step_ms": stats(sample["step_gpu_ms"] for sample in plain),
        "ordinary_ms": mean(sample["module_ms"] for sample in trials(ordinary, "plain")),
        "growth_ms": mean(sample["module_ms"] for sample in trials(growth, "plain")),
        "instrumented_ms": mean(sample["module_ms"] for sample in detailed),
        "components_ms": components(detailed),
        "memory": memory(raws, plain),
        "checks": checks(raws, plain),
        "chunks": [chunk_row(raw) for raw in raws],
    }


def case_of(raw: Raw) -> str:
    """Name a decode token by the heaviest maintenance it triggered."""
    for name in Kernels.CASES:
        if raw["decode_case"][name]:
            return name
    return "ordinary"


def decode_case(raws: list[Raw]) -> dict[str, Any]:
    """Average the tokens of one case - graph replay, eager, and the metadata outside the graph."""
    plain, graph = trials(raws, "plain"), graph_trials(raws)
    return {
        "tokens": len(raws),
        "graph_ms": mean(sample["module_ms"] for sample in graph),
        "free_indexer_graph_ms": mean(sample["module_ms"] for sample in graph_trials(raws, "free_indexer")),
        "indexer_only_graph_ms": mean(sample["module_ms"] for sample in graph_trials(raws, "indexer_only")),
        "graph_indexer_ms": graph_indexer_ms(raws),
        "eager_ms": mean(sample["module_ms"] for sample in plain),
        "metadata_ms": mean(sample["metadata_ms"] for sample in plain),
    }


def periods(raw: Raw) -> dict[str, int]:
    """Return how often each maintenance case recurs, in tokens.

    The harness writes them from the indexer it ran - the design closes a cluster block every `window_size`
    keys, takes a page every `PAGE_SIZE` and grows its hierarchy a page that fills past the growth gate. The
    model's own indexer closes a pool of four keys.

    Args:
        raw: One decode token's raw file.

    Returns:
        One period per case the indexer maintains - empty for a layer that maintains nothing.
    """
    written = raw["decode_case"].get("periods")
    if written is not None:
        return dict(written)
    return {"pool_close": 4} if raw["args"]["indexer"] == "native" else {}


def maintenance(cases: dict[str, Any], periods: dict[str, int]) -> dict[str, float] | None:
    """Spread each case's extra cost over the tokens it serves.

    A case token pays for the maintenance it triggers on top of what an ordinary token pays, and it
    recurs once per period - so this is what maintenance adds to the average token.

    Args:
        cases: The per-case averages.
        periods: How often each case recurs, in tokens.

    Returns:
        Milliseconds per token per case, or None when no ordinary token was measured to read against.
    """
    ordinary = cases.get("ordinary")
    basis = timed(ordinary)
    if not ordinary or basis is None:
        return None
    spread = {}
    for name, period in periods.items():
        case = cases.get(name)
        if case and case[basis] is not None:
            extra = case[basis] - ordinary[basis] + case["metadata_ms"] - ordinary["metadata_ms"]
            spread[name] = round(extra / period, 5)
    return spread


def amortized(cases: dict[str, Any], spread: dict[str, float] | None) -> float | None:
    """Return an ordinary token's time plus this layer's maintenance spread over the tokens it serves.

    Args:
        cases: The per-case averages.
        spread: Milliseconds per token per case, or None when no ordinary token was measured.

    Returns:
        Milliseconds, or None where there is no ordinary token to read against.
    """
    basis = timed(cases.get("ordinary"))
    if spread is None or basis is None:
        return None
    return round(cases["ordinary"][basis] + sum(spread.values()), 4)


def timed(case: dict[str, Any] | None) -> str | None:
    """Name the time a case is read by - a graph replay, or eager where the run recorded no graph.

    A descending decode of shaped node buckets runs eagerly in serving, since its shapes follow the hierarchy, so eager is its time.

    Args:
        case: One case's averages.

    Returns:
        The key of the time, or None when the case has neither.
    """
    if not case:
        return None
    for key in ("graph_ms", "eager_ms"):
        if case[key] is not None:
            return key
    return None


def token_row(raw: Raw) -> dict[str, Any]:
    """Summarise one decode token - the row a click in the viewer opens."""
    plain, detailed, graph = trials([raw], "plain"), trials([raw], "components"), graph_trials([raw])
    return {
        "seq": raw["args"]["context"],
        "offset": raw["decode_case"]["offset"],
        "case": case_of(raw),
        "graph_ms": mean(sample["module_ms"] for sample in graph),
        "free_indexer_graph_ms": mean(sample["module_ms"] for sample in graph_trials([raw], "free_indexer")),
        "indexer_only_graph_ms": mean(sample["module_ms"] for sample in graph_trials([raw], "indexer_only")),
        "graph_indexer_ms": graph_indexer_ms([raw]),
        "eager_ms": mean(sample["module_ms"] for sample in plain),
        "metadata_ms": mean(sample["metadata_ms"] for sample in plain),
        "instrumented_ms": mean(sample["module_ms"] for sample in detailed),
        "components_ms": components(detailed),
        "metadata_components_ms": components(detailed, "metadata_components"),
    }


def graph_differing_slots(compared: list[dict[str, Any]]) -> int | None:
    """Return the most KV slots a graph replay selected that an unpinned eager step did not, over a point's tokens.

    Args:
        compared: The graph comparisons of the point's tokens.

    Returns:
        The max, or None where no token recorded one.
    """
    counts = [
        c["graph_selection_differing_slots"] for c in compared if "graph_selection_differing_slots" in c
    ]
    return max(counts) if counts else None


def decode_point(raws: list[Raw]) -> dict[str, Any]:
    """Reduce the tokens probed around a context to the point it reports.

    Args:
        raws: The context's raw files, ascending by the token each measured.

    Returns:
        The point.
    """
    plain, detailed, graph = trials(raws, "plain"), trials(raws, "components"), graph_trials(raws)
    groups = {name: [raw for raw in raws if case_of(raw) == name] for name in ("ordinary", *Kernels.CASES)}
    # A decode with no graph to replay, and a time-only replay, check no output
    # => nothing to validate
    compared = [
        raw["graph_comparison"]
        for raw in raws
        if raw["graph_comparison"] and "reference_order_output_exact" in raw["graph_comparison"]
    ]
    validated = (
        None
        if not compared
        else all(
            len(compared) == len(raws)
            and comparison.get("status") == "validated"
            and comparison["reference_order_output_exact"]
            for comparison in compared
        )
    )
    cases = {name: decode_case(group) for name, group in groups.items() if group}
    spread = maintenance(cases, periods(raws[0]))
    return {
        "graph_ms": stats(sample["module_ms"] for sample in graph),
        "free_indexer_graph_ms": stats(sample["module_ms"] for sample in graph_trials(raws, "free_indexer")),
        "indexer_only_graph_ms": stats(sample["module_ms"] for sample in graph_trials(raws, "indexer_only")),
        "graph_indexer_ms": graph_indexer_ms(raws),
        "eager_ms": stats(sample["module_ms"] for sample in plain),
        "metadata_ms": stats(sample["metadata_ms"] for sample in plain),
        "step_ms": stats(sample["step_gpu_ms"] for sample in plain),
        "cases": cases,
        # An ordinary token's graph replay plus this layer's maintenance spread over the tokens it serves
        "amortized_ms": amortized(cases, spread),
        "maintenance_ms": spread,
        "instrumented_ms": mean(sample["module_ms"] for sample in detailed),
        "components_ms": components(detailed),
        "metadata_components_ms": components(detailed, "metadata_components"),
        "memory": memory(raws, plain + graph),
        "checks": {
            **checks(raws, plain),
            "graph_validated": validated,
            "graph_selection_differing_slots": graph_differing_slots(compared),
        },
        "tokens": [token_row(raw) for raw in raws],
    }


def gpu_uuid(raw: Raw) -> str:
    """Return the uuid of the card a point was measured on.

    Args:
        raw: One raw file.

    Returns:
        The uuid.
    """
    return str(raw["gpu_uuid"])


def hardware(raw: Raw, stems: list[Path]) -> dict[str, Any]:
    """Say where a point was measured and what its card did meanwhile.

    A fact of the point rather than of the setup, so parts measured on different hosts still join.

    Args:
        raw: The point's first raw file.
        stems: Its raw files' paths without their suffix, beside which the sampler wrote its rows.

    Returns:
        The host, the driver version, and the sampler's summary - None for a point measured before the sampler.
    """
    return {
        "host": raw["hostname"],
        "driver": raw.get("driver"),
        "telemetry": telemetry.read_point(stems),
    }


def points(run_dir: Path, contexts: Iterable[int]) -> dict[str, dict[str, Any]]:
    """Build one point per context that has samples.

    Args:
        run_dir: The run directory.
        contexts: The contexts the run asked for.

    Returns:
        The points, keyed by context as a string so the record stays valid json.
    """
    built = {}
    for context in contexts:
        raws = read_samples(run_dir, context)
        if not raws:
            continue
        point = decode_point(raws) if raws[0].get("phase") == "decode" else prefill_point(raws)
        point["gpu_uuid"] = gpu_uuid(raws[0])
        point["hardware"] = hardware(
            raws[0], [path.with_suffix("") for path in sample_paths(run_dir, context)]
        )
        point["samples"] = [str(path.relative_to(run_dir)) for path in sample_paths(run_dir, context)]
        built[str(context)] = point
    return built


def header(run_id: str, meta: dict[str, Any], raw: Raw) -> dict[str, Any]:
    """Build the run, model and setup blocks.

    The run's own facts come from the launcher and the rest from the first raw file, because the harness
    is what knows the host, the card and the backends it ended up with.

    Args:
        run_id: The run directory's name.
        meta: What the launcher knows - note, tag, times, model name and image.
        raw: The run's first raw file.

    Returns:
        The record without its points.
    """
    args = raw["args"]
    decode = raw.get("phase") == "decode"
    model = meta.get("model") or os.path.basename(args["model_path"].rstrip("/"))
    setup = {
        "indexer": args["indexer"],
        "approximate_selection": meta.get("approximate_selection"),
        "config": raw.get("indexer_config"),
        "layer": args["layer"],
        "tp_size": args["tp_size"],
        "chunk": args["chunk"],
        "prefill_backend": raw["requested_prefill_backend"],
        "decode_backend": raw.get("requested_decode_backend"),
        # The classes that ran behind the names, and whether the names are upstream's, an override, our design's or the fork's
        "backend_classes": raw.get("backend_classes"),
        "backend_source": raw.get("backend_source"),
        "cuda_graph": bool(args.get("graph_compare")),
        "warmup": args["warmup"],
        "repeats": args["repeats"],
        "image": meta.get("image", ""),
        Mesh.FIELD: mesh({"tp_size": args["tp_size"]}, meta.get("image") or None),
        "prompt": {
            "file": args["prompt_file"],
            "tokens": raw["prompt_source_tokens"],
            "repeated": raw["prompt_repeated"],
        },
    }
    if decode:
        setup["offsets"] = args["decode_offsets"]
    else:
        setup["window_chunks"] = args["window_chunks"]
    return {
        "schema": Kernels.SCHEMA,
        "run": {
            "id": run_id,
            "name": meta.get("name") or args["indexer"],
            "phase": "decode" if decode else "prefill",
            "note": meta.get("note", ""),
            "tag": meta.get("tag", ""),
            "started": meta.get("started"),
            "date": meta.get("date"),
            "seconds": meta.get("seconds"),
            "host": raw["hostname"],
            "gpu": raw["gpu"],
            # A worktree's `.git` is a file the container cannot follow, so the harness records provenance
            # best effort and a measurement is not lost over it
            "commits": {"lsa": (raw["lsa_commit"] or "")[:7]},
            # A measurement off a dirty tree is not the commit it claims - this is what says so
            "diff": hashlib.sha256((raw["lsa_diff"] or "").encode()).hexdigest()[:12],
            "harness": {name: digest[:12] for name, digest in raw["source_sha256"].items()},
            "versions": {"torch": raw["torch"], "cuda": raw["cuda"]},
        },
        "model": {
            "name": model,
            "label": os.path.basename(model).lower(),
            "path": args["model_path"],
        },
        "setup": setup,
    }


def build(
    run_id: str,
    meta: dict[str, Any],
    run_dir: Path,
    contexts: Iterable[int],
    failed: Iterable[int] = (),
) -> dict[str, Any]:
    """Build the whole record of one run from the raw files under it.

    Args:
        run_id: The run directory's name.
        meta: What the launcher knows about the run.
        run_dir: The run directory.
        contexts: The contexts the run asked for.
        failed: The contexts whose harness process did not finish.

    Returns:
        The record.

    Raises:
        FileNotFoundError: No context has samples, so there is nothing to describe.
    """
    wanted = list(contexts)
    measured = [context for context in wanted if sample_paths(run_dir, context)]
    if not measured:
        raise FileNotFoundError(f"{run_dir}: no samples")
    first = read_samples(run_dir, measured[0])[0]
    return {
        **header(run_id, meta, first),
        "points": points(run_dir, wanted),
        "failed": sorted(failed),
    }
