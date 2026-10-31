"""The record of a production run - the client's result files folded into one JSON - and its table.

    <results>/performance/production/<run_id>/record.json                          everything below
    <results>/performance/production/<run_id>/samples/<deployment>/<length>.jsonl  the client's own output, every request in it

record.json
  schema, run {id, name, note, tag, started, date, seconds, client, image, gpu}
  model {name, label, path}
  setup {client, warmup, requests, output_tokens, concurrency, max_running_requests}
  indexer
  servers {deployment: {nodes, the server's settings, cuda_graph, token_capacity, mesh}}
  points {length: {deployment: {ttft_s, tpot_ms, e2e_s {median, min, max}, itl_p99_ms, itl_max_ms,
                                requests [{ttft_s, tpot_ms, e2e_s, input_tokens, output_tokens}], duration_s,
                                output_tokens_per_s, peak_output_tokens_per_s, wave_in_flight, wave_complete,
                                decode_window {start_s, seconds, output_tokens_per_s, tpot_ms} | null}}}

`hardware` is what the sampler saw of every GPU of every node the server spans, from the first request to the last
response: {hosts, gpus {uuid: {host, index, samples, sm_mhz_min, sm_mhz_median, mem_mhz_median, power_w_max,
power_limit_w, temperature_c_max, power_capped_fraction, slowdown_fraction, slowdown_reasons, throttle_reasons}}, files} - None for a length measured
before it. The rows are `samples/<deployment>/<length>.gpu-<host>.csv`. It is a fact of the point and not of the
setup, so parts measured on different hosts still join.

`wave_in_flight` is the most requests the server decoded together at that length, counted from the server's own
log (see `waves`), and `wave_complete` is whether it equals `setup.concurrency` - both None where the log cannot say.

A deployment is the server's mesh label, `tp<n>-cp<n>-dcp<n>-pp<n>-ep<n>` (see `Mesh`), so a length measured
on one node and on two, or under tensor and under context parallel, keeps both. A run measured before the
mesh was kept wrote its samples under `tp<n>[-pp<n>]`, and they stay there whatever a rebuild keys them by.
The layout is the one the colleague's records of 2026-09-23 use, so the two read side by side.

`lsa records show <run dir> [<run dir>]` prints the table of one or two runs, the second beside the first with the speedup.
"""

import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.accuracy.records import mesh, model_label
from lsa.evaluation.constants import Mesh, Production, Records
from lsa.evaluation.performance import telemetry
from lsa.evaluation.performance.production.waves import wave_peaks

Result = dict[str, Any]


def deployment(info: dict[str, Any]) -> str:
    """Name the mesh a server runs on, as `tp8-cp1-dcp1-pp1-ep8`.

    Args:
        info: The server's `/server_info`.

    Returns:
        The deployment name.
    """
    return str(mesh(info, None)["label"])


def sample_path(run_dir: Path, name: str, length: int) -> Path:
    """Return where the client writes one length's result on one deployment."""
    return run_dir / "samples" / name / f"{length}.jsonl"


def read_result(path: Path) -> Result | None:
    """Read the client's result for one length.

    The client appends one line per call, so a length that was retried holds several and the last is the one
    that counts.

    Args:
        path: The result file.

    Returns:
        The last result, or None for an empty file.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def finished(path: Path, requests: int) -> bool:
    """Return whether a result file holds every request asked for, with no error among them.

    Args:
        path: The result file.
        requests: The measured requests the length was asked for.

    Returns:
        True when the length needs no more measuring.
    """
    if not path.is_file():
        return False
    result = read_result(path)
    if not result:
        return False
    return bool(result.get("completed") == requests and not any(result.get("errors") or []))


def stats(values: list[float]) -> dict[str, float]:
    """Summarise a few requests as their median and extremes."""
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def request_rows(result: Result) -> list[dict[str, float]]:
    """Return one row per measured request.

    TPOT is the mean gap between streamed tokens, which is `(E2E - TTFT) / (output tokens - 1)` - the
    definition sglang's, vLLM's and NVIDIA's serving benchmarks share.

    Args:
        result: The client's result for one length.

    Returns:
        Each request's TTFT, TPOT, end-to-end time and token counts.

    Raises:
        ValueError: A request streamed no tokens - the server refused or failed it, which the message carries.
    """
    rows = []
    for ttft, gaps, input_tokens, output_tokens in zip(
        result["ttfts"], result["itls"], result["input_lens"], result["output_lens"], strict=True
    ):
        if not gaps:
            errors = sorted({str(error) for error in result.get("errors") or [] if error})
            raise ValueError(
                f"a request of {input_tokens} input tokens streamed no tokens - the server refused or failed it: "
                f"{'; '.join(errors) or 'no error recorded'}"
            )
        rows.append(
            {
                "ttft_s": ttft,
                "tpot_ms": 1000 * sum(gaps) / len(gaps),
                "e2e_s": ttft + sum(gaps),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
    return rows


def decode_window(result: Result) -> dict[str, float] | None:
    """Measure the stretch where every request of one wave decodes at once.

    One wave is every request in flight together, sent at once at an unbounded rate, so all start at the same
    instant and a token's time is its request's TTFT plus the gaps before it. The window opens at the last first
    token and closes at the first request to finish. The server runs every queued prefill before a decode step,
    so an early request's tokens stall until the last prompt is in and the window is where the wave decodes.

    Args:
        result: The client's result for one length.

    Returns:
        When the window opens after the requests were sent, how long it lasts, the output tokens a second the node
        served inside it and the median gap between two tokens there - or None when the requests came in more
        than one wave or never all decoded at once.
    """
    if result["completed"] > result["max_concurrency"]:
        return None
    token_times = [
        list(itertools.accumulate(gaps, initial=ttft))[1:]
        for ttft, gaps in zip(result["ttfts"], result["itls"], strict=True)
    ]
    start = max(result["ttfts"])
    end = min(times[-1] for times in token_times if times)
    if end <= start:
        return None
    inside = [
        gap
        for times, gaps in zip(token_times, result["itls"], strict=True)
        for time, gap in zip(times, gaps, strict=True)
        if start < time <= end
    ]
    return {
        "start_s": start,
        "seconds": end - start,
        "output_tokens_per_s": len(inside) / (end - start),
        "tpot_ms": 1000 * statistics.median(inside),
    }


def wave_complete(wave_in_flight: int | None, concurrency: int) -> bool | None:
    """Say whether the requests the client asked to run together all did.

    Args:
        wave_in_flight: The most requests the server decoded together, None when the client did not count them.
        concurrency: The requests the client asked to keep in flight.

    Returns:
        Whether they are equal, or None when the peak was not counted.
    """
    if wave_in_flight is None:
        return None
    return wave_in_flight == concurrency


def add_waves(
    points: dict[str, dict[str, Any]], concurrency: int, log: Path, lengths: list[str] | None = None
) -> None:
    """Count each context's wave from a server log and set it on the points, in place.

    The contexts are matched to the log's flushes in ascending order, so they must all be one deployment's, one
    server's - anything else is left None.

    Args:
        points: Each context's points by deployment, as the record keeps them.
        concurrency: The requests the client was asked to keep in flight.
        log: The server's log.
        lengths: The contexts this log served, all of them when omitted.
    """
    chosen = sorted((length for length in points if lengths is None or length in lengths), key=int)
    deployments = {name for length in chosen for name in points[length]}
    if len(deployments) != 1:
        return
    name = deployments.pop()
    entries = [points[length][name] for length in chosen]
    peaks = wave_peaks(log, [entry["duration_s"] for entry in entries])
    for entry, peak in zip(entries, peaks, strict=True):
        entry["wave_in_flight"] = peak
        entry["wave_complete"] = wave_complete(peak, concurrency)


def point(result: Result) -> dict[str, Any]:
    """Summarise one length on one deployment.

    Args:
        result: The client's result for that length.

    Returns:
        Medians with extremes, the tail of the token gaps, every request, and what the node served across them -
        output tokens a second over the whole length, its peak second, and the stretch where every request decoded
        together. The wave fields are None until `add_waves` counts them from the server's log.
    """
    rows = request_rows(result)
    gaps = sorted(1000 * gap for request in result["itls"] for gap in request)
    return {
        "ttft_s": stats([row["ttft_s"] for row in rows]),
        "tpot_ms": stats([row["tpot_ms"] for row in rows]),
        "e2e_s": stats([row["e2e_s"] for row in rows]),
        # The jitter a reader of the stream sees
        "itl_p99_ms": statistics.quantiles(gaps, n=100)[98],
        "itl_max_ms": gaps[-1],
        "requests": rows,
        "duration_s": result["duration"],
        # Absent from a result the client wrote before it counted them
        # => None, so an old run still rebuilds
        "output_tokens_per_s": result.get("output_throughput"),
        "peak_output_tokens_per_s": result.get("max_output_tokens_per_s"),
        # Counted from the server's log once the run's points are together
        # => None here, because the client's count bins requests into whole seconds
        "wave_in_flight": None,
        "wave_complete": None,
        "decode_window": decode_window(result),
    }


def setup(result: Result, warmup: int) -> dict[str, Any]:
    """Describe how the client measured.

    Args:
        result: Any of the run's results.
        warmup: The warmup requests, which the client's result does not name.

    Returns:
        The client, its warmup, the request shape, the concurrency asked for and the request limit the server
        ran under - None when its settings did not say.
    """
    return {
        "client": Production.CLIENT,
        "warmup": warmup,
        "requests": result["completed"],
        "output_tokens": result["random_output_len"],
        "concurrency": result["max_concurrency"],
        "max_running_requests": (result.get("server_info") or {}).get("max_running_requests"),
    }


def server(result: Result, image: str | None) -> dict[str, Any]:
    """Describe the server a result was measured against, from the settings it reported.

    Args:
        result: The client's result, which embeds the server's `/server_info`.
        image: The container the server ran in, which it cannot report itself.

    Returns:
        Its node count, the settings a number depends on, which graph backend each phase ran, its KV
        capacity in tokens and its mesh.
    """
    info = result.get("server_info") or {}
    states = (info.get("internal_states") or [{}])[0]
    graphs = info.get("cuda_graph_config") or {}
    return {
        "nodes": info.get("nnodes"),
        **{key: info.get(key) for key in Production.SERVER_KEYS},
        # The backend per phase is what a token's cost depends on - the captured batch sizes are fifty rows
        "cuda_graph": {phase: block.get("backend") for phase, block in graphs.items()},
        "token_capacity": states.get("memory_usage", {}).get("token_capacity"),
        Mesh.FIELD: mesh(info, image),
    }


def build(run_id: str, meta: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Fold every result file of a run into its record.

    The indexer is read from the settings file the run was served with, because a stock server's
    `/server_info` cannot say whether the plugin was armed. A deployment is named by the mesh its result
    reports rather than by its directory.

    Args:
        run_id: The run directory's name.
        meta: What the launcher knows - note, tag, times, the label, the model's hub id, the image.
        run_dir: The run directory.

    Returns:
        The record body.
    """
    points: dict[str, dict[str, Any]] = {}
    servers: dict[str, Any] = {}
    how: dict[str, Any] = {}
    model_path = ""
    results = [(path, read_result(path)) for path in sorted(run_dir.glob("samples/*/*.jsonl"))]
    named = [
        (path, result, deployment(result.get("server_info") or {})) for path, result in results if result
    ]
    for path, result, name in named:
        points.setdefault(path.stem, {})[name] = {
            **point(result),
            "hardware": telemetry.read_point([path.with_suffix("")]),
        }
        servers[name] = server(result, meta.get("image"))
        how = setup(result, meta.get("warmup", 1))
        model_path = (result.get("server_info") or {}).get("model_path", "")
    points = dict(sorted(points.items(), key=lambda item: int(item[0])))
    if how:
        add_waves(points, how["concurrency"], run_dir / "server.log")
    settings = run_dir / "indexer.json"
    name = meta.get("model") or model_path
    return {
        "schema": Production.SCHEMA,
        "run": {
            **{key: meta.get(key) for key in ("note", "tag", "started", "date", "seconds", "image", "gpu")},
            "id": run_id,
            "name": meta.get("name", ""),
            "client": Production.CLIENT,
        },
        "model": {"name": name, "label": model_label(name), "path": model_path},
        "setup": how,
        "indexer": io.read_json(settings, None),
        "servers": servers,
        "points": points,
    }


def write(run_dir: Path, meta: dict[str, Any]) -> Path:
    """Write one run's record, atomically.

    The ladder rewrites it after every length, so a reader can hit it mid-write.

    Args:
        run_dir: The run directory.
        meta: What the launcher knows about the run.

    Returns:
        The record's path.
    """
    path = run_dir / Records.RECORD
    io.write_json(path, build(run_dir.name, meta, run_dir), indent=1)
    return path


def load(run_dir: Path) -> dict[str, Any]:
    """Return a run's record, or an empty one before its first length lands."""
    path = run_dir / Records.RECORD
    return io.read_json(path, {})


def cell(entry: dict[str, Any] | None, key: str) -> str:
    """Format one median with its extremes, or a dash where there is no point."""
    if not entry:
        return "-"
    value = entry[key]
    return f"{value['median']:.2f} ({value['min']:.2f} - {value['max']:.2f})"


def table(records: list[dict[str, Any]]) -> str:
    """Lay out records as one row per length and deployment.

    Args:
        records: One or more records. Two get a speedup column each, the first over the second - so native
            first reads as how much faster the second one is.

    Returns:
        The aligned table.
    """
    rows = sorted(
        {
            (int(length), name)
            for record in records
            for length, by_name in record["points"].items()
            for name in by_name
        }
    )
    header = ["length", "deployment"]
    for record in records:
        header += [f"{record['run']['name']} TTFT s", "TPOT ms"]
    if len(records) == 2:
        header += ["TTFT x", "TPOT x"]
    lines = [header]
    for length, name in rows:
        entries = [record["points"].get(str(length), {}).get(name) for record in records]
        line = [str(length), name]
        for entry in entries:
            line += [cell(entry, "ttft_s"), cell(entry, "tpot_ms")]
        if len(records) == 2 and all(entries):
            for key in ("ttft_s", "tpot_ms"):
                line.append(f"{entries[0][key]['median'] / entries[1][key]['median']:.2f}")
        lines.append(line)
    widths = [max(len(line[index]) for line in lines if index < len(line)) for index in range(len(header))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(line, widths, strict=False)).rstrip()
        for line in lines
    )
