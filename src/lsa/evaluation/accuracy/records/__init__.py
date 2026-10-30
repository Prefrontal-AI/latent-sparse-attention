"""The record of a run - one JSON per run, one JSONL of samples per task.

results/<run_id>/record.json        everything below
results/<run_id>/samples/<task>.jsonl  one row per (doc, repeat): answer, reasoning, tokens, metrics
results/<run_id>/server_info.json   the server's full /server_info

record.json
  schema, run {id, name, suite, note, tag, edits [{date, who, what}],
               date, seconds, host, gpu, commits, versions, plans}
  model {name, path}
  attention {indexer, config, map {path, sha256}}  - what a comparison varies
  server {indexer, config (the settings block the server reports), the serving flags that matter,
          mesh {label, tp_size, attn_tp_size, attn_cp_size, dcp_size, enable_prefill_cp, cp_strategy, pp_size,
                ep_size, dp_size, enable_dp_attention, nnodes, moe_runner_backend, image}}
  servers {id: the same, one per server that answered a task - a slurm job id, or the host and launch}
  A natively trained model no server ran has `server` null and `servers` empty, and its tasks keep only a score,
  a `lower_is_better` flag where a lower one is better, and what the source gave.
  tasks {task: {benchmark, category, length, length_actual, key, n_docs, repeats, n, scored, errors, empty, truncated,
                finished, metric, score, score_finished, stderr, stderr_doc, ci95, repeat_scores, repeat_std,
                metrics (mean of every metric),
                tokens {prompt, reasoning, answer, output: {mean, std, median, p90, max, sum}},
                sampling, concurrency, seconds, output_tokens_per_second, prompt_set_hash, samples,
                data {root, file, corpus, and the file's manifest entry}, prompt {delimiter, gen_prefix,
                hash, pinned, strict}, observed {reasoned, cut, budget_enforced}, server, protocol}}

Whether a run finished is not in the record - see [status](../../status.py), which reads it beside the run's
launch log.
"""

import hashlib
import json
import logging
import math
import os
import statistics
from datetime import UTC, datetime
from typing import Any

from lsa import io, paths
from lsa.evaluation.constants import Evaluation, Mesh, Records

logger = logging.getLogger(__name__)

# A think block the server closed at once under a zero budget still counts its closing token
# => a reply reasoned only past this
FORCED_CLOSE_TOKENS = 1

# Bootstrap indices held at once, draws x docs
BOOTSTRAP_BLOCK_CELLS = 4_000_000

# A gap this small between two docs' means is no gap
TIE_TOLERANCE = 1e-9


def records_root() -> str:
    """Return the root every run directory lands in.

    The shared evaluations root when it exists, and `results` under the working directory otherwise - runs land
    there while they are being named, so it is a workflow and not a guess, and the fallback is logged with the
    shared path that was missing. Resolved per call rather than at import, so it reads the `LSA_ROOT` the
    caller actually has.
    """
    shared = paths.evaluations()
    if shared.is_dir():
        return str(shared)
    local = os.path.join(os.getcwd(), *Records.LOCAL_ROOT)
    logger.warning("No shared evaluations root at %s, so records are read from %s", shared, local)
    return local


def records_dir(kind: str, root: str | os.PathLike[str] | None = None) -> str:
    """Return the directory one kind's run directories land in.

    Args:
        kind: One of `Evaluation.KINDS`.
        root: The results root, `records_root()` when None.

    Returns:
        `<root>/accuracy`, `<root>/performance/kernels` or `<root>/performance/production`.
    """
    return os.path.join(root or records_root(), *Evaluation.DIRS[kind])


def model_label(model: str) -> str:
    """Name a checkpoint the way a record's `model.label` does.

    Args:
        model: The checkpoint path or repo id the server was given.

    Returns:
        The last path segment, lowercased.
    """
    return model.rstrip("/").rsplit("/", 1)[-1].lower()


def mesh(info: dict[str, Any], image: str | None) -> dict[str, Any]:
    """Describe the GPUs a server was laid out on, every axis with its size even where it is 1.

    A stock image reports none of the context-parallel settings, which is the same as their being off.

    Args:
        info: The server's `/server_info`, or the `server_args` its log prints - both name the settings.
        image: The container the server ran in, or None where nothing recorded it.

    Returns:
        The mesh - its label, each axis's size, the prefill split, the MoE runner and the image.
    """
    tp_size = int(info.get("tp_size") or 1)
    attn_cp_size = int(info.get("attn_cp_size") or 1)
    dp_size = int(info.get("dp_size") or 1)
    enable_dp_attention = bool(info.get("enable_dp_attention"))
    attention_dp = dp_size if enable_dp_attention else 1
    fields: dict[str, Any] = {
        "tp_size": tp_size,
        "attn_tp_size": max(tp_size // attention_dp // attn_cp_size, 1),
        "attn_cp_size": attn_cp_size,
        "dcp_size": int(info.get("dcp_size") or 1),
        "enable_prefill_cp": bool(info.get("enable_prefill_cp")),
        "cp_strategy": info.get("cp_strategy"),
        "pp_size": int(info.get("pp_size") or 1),
        "ep_size": int(info.get("ep_size") or 1),
        "dp_size": dp_size,
        "enable_dp_attention": enable_dp_attention,
        "nnodes": int(info.get("nnodes") or 1),
        "moe_runner_backend": info.get("moe_runner_backend"),
        "image": image,
    }
    return {"label": mesh_label(fields), **fields}


def mesh_label(fields: dict[str, Any]) -> str:
    """Name a mesh by every axis, as `tp1-cp8-dcp8-pp1-ep8` - see `Mesh`."""
    return "-".join(f"{axis}{fields[key]}" for axis, key in Mesh.AXES)


def mesh_label_of(server: dict[str, Any]) -> str:
    """Return a server entry's mesh label."""
    return str(server[Mesh.FIELD]["label"])


def configuration_key(
    model_label: str,
    indexer: str,
    config: dict[str, Any] | None,
    mesh_labels: list[str],
    gpu: str = "",
) -> str:
    """Name what a run served with a short hash - the model, the indexer and its settings, the mesh and the card.

    Two runs with one key differ only in what they measured and in the serving knobs a key leaves out, so the
    viewer lists them as one configuration, extends a picked run with another, and `lsa records join` reads
    it as the first thing two parts of one run share.

    Args:
        model_label: The record's `model.label`.
        indexer: `native`, `lsa` or another attention path.
        config: The indexer's settings, None for none.
        mesh_labels: Each server's mesh label, as `mesh_label_of` names it.
        gpu: The card model the run ran on, empty where none is known - which leaves the key as it was before
            the card was part of it, so a record of unknown card keeps its key.

    Returns:
        Twelve hex characters, the same whatever order the settings' keys were written in.
    """
    fields = [model_label, indexer, config, mesh_labels, *([gpu] if gpu else [])]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]


def build_run_id(name: str, tag: str = "") -> str:
    """Build the id of a run, which names its directory and is what the viewer groups by.

    Args:
        name: What the config calls the run, `run.name`.
        tag: A short label appended to the id, or empty.

    Returns:
        `<name>__<stamp>[-<tag>]`, the stamp being the UTC start as `2026-09-23__15-37-53`.
    """
    stamp = datetime.now(UTC).strftime(Records.RUN_STAMP)
    return f"{name}__{stamp}" + (f"-{tag}" if tag else "")


def stats(values: list[float | None]) -> dict[str, float] | None:
    """Summarise one token count over a task's rows, or return None when no row has it."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    ordered = sorted(present)
    return {
        "mean": round(statistics.fmean(present), 1),
        "std": round(statistics.pstdev(present), 1) if len(present) > 1 else 0.0,
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "max": ordered[-1],
        "sum": sum(present),
    }


def standard_error(score: float | None, count: int) -> float | None:
    """Return the binomial standard error of a score over its samples, or None for a score that is no rate."""
    if score is None or not 0 <= score <= 1:
        return None
    return round(math.sqrt(score * (1 - score) / count), 4)


def doc_means(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    """Average each doc's repeats of one metric, over the rows that were scored.

    Args:
        rows: A task's sample rows.
        metric: The metric key.

    Returns:
        Doc id to the mean of its scored repeats - a doc with no scored row is left out.
    """
    by_doc: dict[int, list[float]] = {}
    for row in rows:
        if row.get("error") is None:
            by_doc.setdefault(row["doc_id"], []).append(row["metrics"][metric])
    return {doc_id: statistics.fmean(values) for doc_id, values in by_doc.items()}


def clustered_standard_error(values: list[float]) -> float | None:
    """Return the standard error of the mean over independent units, None for fewer than two."""
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def bootstrap_interval(values: list[float]) -> list[float] | None:
    """Return the percentile bootstrap interval of the mean of values, None for fewer than two.

    The values are resampled with replacement `Records.BOOTSTRAP_DRAWS` times from a generator seeded with
    `Records.BOOTSTRAP_SEED`, so the same values always give the same interval.

    Args:
        values: One value per independent unit - a doc's mean, or a doc's gap between two runs.

    Returns:
        The lower and upper bound at `Records.BOOTSTRAP_LEVEL`.
    """
    # Numpy comes with the evaluation extra, and a bare install reads records without it
    import numpy as np

    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(Records.BOOTSTRAP_SEED)
    means = np.empty(Records.BOOTSTRAP_DRAWS)
    # Resample in blocks, so a task of thousands of docs never holds draws x docs indices at once
    block = max(1, BOOTSTRAP_BLOCK_CELLS // len(data))
    for start in range(0, Records.BOOTSTRAP_DRAWS, block):
        stop = min(start + block, Records.BOOTSTRAP_DRAWS)
        means[start:stop] = data[generator.integers(0, len(data), size=(stop - start, len(data)))].mean(
            axis=1
        )
    tail = 100 * (1 - Records.BOOTSTRAP_LEVEL) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return [round(float(low), 4), round(float(high), 4)]


def doc_statistics(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Return the spread of a task's score with each doc counted once, however many repeats it has.

    Args:
        rows: The task's sample rows.
        metric: The task's primary metric key.

    Returns:
        `stderr_doc`, the standard error over the docs' means, and `ci95`, the bootstrap interval of their mean.
        Both are None for fewer than two docs scored.
    """
    means = list(doc_means(rows, metric).values())
    error = clustered_standard_error(means)
    return {"stderr_doc": None if error is None else round(error, 4), "ci95": bootstrap_interval(means)}


def identical_answers(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> tuple[int, int]:
    """Count the samples both runs answered with the same text.

    Args:
        rows_a: The baseline's sample rows.
        rows_b: The other run's sample rows.

    Returns:
        The samples whose answer text is identical, and the (doc, repeat) samples both runs finished.
    """
    answers_a = {
        (row["doc_id"], row["repeat"]): row.get("answer") for row in rows_a if row.get("error") is None
    }
    answers_b = {
        (row["doc_id"], row["repeat"]): row.get("answer") for row in rows_b if row.get("error") is None
    }
    shared = answers_a.keys() & answers_b.keys()
    return sum(1 for key in shared if answers_a[key] == answers_b[key]), len(shared)


def paired_comparison(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    """Compare two runs of one task over the docs both scored, each doc's repeats averaged first.

    Every design draws the same docs, so a doc is one pair and the gap is measured within it - the docs' own
    difficulty, which both runs share, drops out of the error.

    Args:
        rows_a: The baseline's sample rows.
        rows_b: The other run's sample rows.
        metric: The metric key both score.

    Returns:
        `docs` matched, both runs' mean over them, `gap` (b minus a), its `stderr` and `ci95`, the docs b scored
        above (`wins`), below (`losses`) and level on (`ties`), the docs only one run scored, and `identical`,
        the share of samples both finished whose answer text is the same - None when none were finished by both.
    """
    same, shared = identical_answers(rows_a, rows_b)
    means_a = doc_means(rows_a, metric)
    means_b = doc_means(rows_b, metric)
    matched = sorted(means_a.keys() & means_b.keys())
    gaps = [means_b[doc_id] - means_a[doc_id] for doc_id in matched]
    error = clustered_standard_error(gaps)
    return {
        "docs": len(matched),
        "score_a": round(statistics.fmean(means_a[doc_id] for doc_id in matched), 4) if matched else None,
        "score_b": round(statistics.fmean(means_b[doc_id] for doc_id in matched), 4) if matched else None,
        "gap": round(statistics.fmean(gaps), 4) if gaps else None,
        "stderr": None if error is None else round(error, 4),
        "ci95": bootstrap_interval(gaps),
        "wins": sum(1 for gap in gaps if gap > TIE_TOLERANCE),
        "losses": sum(1 for gap in gaps if gap < -TIE_TOLERANCE),
        "ties": sum(1 for gap in gaps if abs(gap) <= TIE_TOLERANCE),
        "only_a": len(means_a.keys() - means_b.keys()),
        "only_b": len(means_b.keys() - means_a.keys()),
        "identical": round(same / shared, 4) if shared else None,
    }


def sample_rows_of(run_dir: str | os.PathLike[str], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a task's sample rows from a run directory, joined or not - the record names the file.

    Returns:
        The rows, empty when the record names no samples or the file is gone.
    """
    if not entry.get("samples"):
        return []
    rows: list[dict[str, Any]] = io.read_jsonl(os.path.join(run_dir, entry["samples"]))
    return rows


def compare_runs(
    run_dir_a: str | os.PathLike[str], run_dir_b: str | os.PathLike[str], tasks: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Pair two runs task by task, over the tasks both scored.

    Args:
        run_dir_a: The baseline's run directory.
        run_dir_b: The other run's directory.
        tasks: The tasks to compare, every one both runs hold when None.

    Returns:
        Task name to its metric and `paired_comparison`, in the baseline's task order.

    Raises:
        ValueError: A named task is missing from a run, or the two runs score it by different metrics.
    """
    entries_a = io.read_json(os.path.join(run_dir_a, Records.RECORD))["tasks"]
    entries_b = io.read_json(os.path.join(run_dir_b, Records.RECORD))["tasks"]
    missing = [task for task in tasks or [] if task not in entries_a or task not in entries_b]
    if missing:
        raise ValueError(f"not in both runs: {', '.join(missing)}")
    compared = {}
    for task in tasks or [name for name in entries_a if name in entries_b]:
        metric = entries_a[task]["metric"]
        if metric != entries_b[task]["metric"]:
            raise ValueError(f"{task}: {metric} against {entries_b[task]['metric']}")
        rows_a = sample_rows_of(run_dir_a, entries_a[task])
        rows_b = sample_rows_of(run_dir_b, entries_b[task])
        compared[task] = {"metric": metric, **paired_comparison(rows_a, rows_b, metric)}
    return compared


def breakdown(rows: list[dict[str, Any]], metric: str, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Score a task's rows split by fields of their docs' metadata, as PaperTrail's by tier.

    Args:
        rows: The task's sample rows, graded, each with its doc's `meta`.
        metric: The task's primary metric key.
        keys: The metadata fields to split by.

    Returns:
        Field to value to the counts, the score, the score over finished samples and the score of each repeat.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        by_value: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_value.setdefault(str(row.get("meta", {}).get(key)), []).append(row)
        out[key] = {}
        for value in sorted(by_value):
            summary = summarize(by_value[value], metric)
            out[key][value] = {
                name: summary[name]
                for name in ("n", "scored", "truncated", "score", "score_finished", "repeat_scores")
            }
    return out


def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Aggregate the sample rows of one task, counting the rows with an error and scoring none of them.

    Args:
        rows: The task's sample rows, graded.
        metric: The task's primary metric key.

    Returns:
        The counts, the score and its spread, and the token statistics a record entry carries.
    """
    scored = [row for row in rows if row.get("error") is None]
    values = [row["metrics"][metric] for row in scored]
    finished = [row for row in scored if row.get("finish_reason") != "length"]
    finished_values = [row["metrics"][metric] for row in finished]
    repeats = sorted({row["repeat"] for row in rows})
    repeat_scores: list[float | None] = []
    for repeat in repeats:
        part = [row["metrics"][metric] for row in scored if row["repeat"] == repeat]
        repeat_scores.append(round(statistics.fmean(part), 4) if part else None)
    score = round(statistics.fmean(values), 4) if values else None
    repeat_values = [s for s in repeat_scores if s is not None]
    metric_names = sorted({name for row in scored for name in row["metrics"]})
    out = {
        "n": len(rows),
        "scored": len(scored),
        "errors": len(rows) - len(scored),
        "empty": sum(1 for row in scored if not (row["answer"] or "").strip()),
        "truncated": sum(1 for row in scored if row.get("finish_reason") == "length"),
        "finished": len(finished),
        "metric": metric,
        "score": score,
        # The same metric over the samples that were not cut at the completion budget.
        # What a gap between this and `score` means depends on whether the run bounded its thinking.
        # Unbounded, a cut sample had not closed the think block and so scored zero whatever the model
        # knew - 1067 of 1068 cut samples emitted no answer at all - and the score was largely a
        # termination rate. Under `thinking_budget` a cut sample answers first and runs on afterwards, so
        # it is gradeable and usually right, and the two numbers converge.
        "score_finished": round(statistics.fmean(finished_values), 4) if finished_values else None,
        # The binomial standard error over all scored samples, every repeat counted as an independent draw
        # => too small once a problem is asked more than once, kept so older records and readers still agree
        "stderr": standard_error(score, len(values)),
        # The standard error over the docs' means, a doc's repeats averaged first, and the bootstrap 95% interval
        # of that mean over docs - the score's error when the repeats of one problem are correlated
        **doc_statistics(rows, metric),
        "repeat_scores": repeat_scores,
        "repeat_std": round(statistics.pstdev(repeat_values), 4)
        if len(repeats) > 1 and repeat_values
        else None,
        "metrics": {
            name: round(statistics.fmean(row["metrics"][name] for row in scored), 4) for name in metric_names
        }
        if scored
        else {},
        "tokens": {
            "prompt": stats([row.get("prompt_tokens") for row in scored]),
            "reasoning": stats([row.get("reasoning_tokens") for row in scored]),
            "answer": stats([row.get("answer_tokens") for row in scored]),
            "output": stats([row.get("completion_tokens") for row in scored]),
        },
    }
    return out


def protocol_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """Return what two runs of one task must share for their scores to be one measurement.

    The attention and the serving setup are what a comparison is for, so they are left out. A sampling key that is
    unset reads the same whether a record wrote it as null, as an empty dict or not at all.

    Args:
        entry: One task of a record.

    Returns:
        The prompt set, how many docs it holds, and every sampling setting that was set.
    """
    sampling = {
        key: value for key, value in (entry.get("sampling") or {}).items() if value not in (None, {}, [])
    }
    return {"prompts": entry.get("prompt_set_hash"), "docs": entry.get("n_docs"), **sampling}


def protocol_key(entry: dict[str, Any]) -> str | None:
    """Name a task's protocol with a short hash, or None when the record holds no prompt-set hash to check."""
    fields = protocol_fields(entry)
    if fields["prompts"] is None:
        return None
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]


def corpus_name(entry: dict[str, Any] | None) -> str | None:
    """Name the corpus a fixed sample file came from, off its manifest entry.

    Returns:
        `ruler-nvidia@<commit>` for NVIDIA's generator, `lm-eval-<version>` for lm-eval's port, or None when
        the entry names neither.
    """
    if not entry:
        return None
    if entry.get("generator"):
        return f"ruler-nvidia@{entry['generator'][:8]}"
    if entry.get("lm_eval"):
        return f"lm-eval-{entry['lm_eval']}"
    return None


def budget_enforced(sampling: dict[str, Any] | None, server: dict[str, Any] | None) -> bool | None:
    """Say whether a task's thinking budget was held to, or None when it set none.

    A server launched without strict thinking ignores the budget rather than refusing it, so the sampling
    alone does not say whether the model was stopped.
    """
    if (sampling or {}).get("thinking_budget") is None:
        return None
    return bool((server or {}).get("enable_strict_thinking"))


def observed(rows: list[dict[str, Any]], enforced: bool | None) -> dict[str, Any]:
    """Return what the replies of one task did, which the sampling alone does not say.

    Args:
        rows: The task's sample rows.
        enforced: Whether the server held the thinking budget, from `budget_enforced`.

    Returns:
        The share of scored rows that reasoned at all and the share cut at the completion budget, None when
        nothing was scored, and whether the thinking budget was enforced.
    """
    scored = [row for row in rows if row.get("error") is None]
    if not scored:
        return {"reasoned": None, "cut": None, "budget_enforced": enforced}
    reasoned = sum(1 for row in scored if (row.get("reasoning_tokens") or 0) > FORCED_CLOSE_TOKENS)
    cut = sum(1 for row in scored if row.get("finish_reason") == "length")
    return {
        "reasoned": round(reasoned / len(scored), 4),
        "cut": round(cut / len(scored), 4),
        "budget_enforced": enforced,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    """Add one row to a jsonl file as it arrives, so a stopped run keeps every reply that came back."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as target:
        target.write(json.dumps(row, ensure_ascii=False) + "\n")
        target.flush()
