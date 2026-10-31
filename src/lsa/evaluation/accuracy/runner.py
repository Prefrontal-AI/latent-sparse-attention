"""Run a suite against a served model and write the records.

For every task: build the fixed prompts, check their hash, send what is not done yet (a stopped run
continues where it was), grade every reply through the task's own filters and scorer, write the
samples and the records. The server is a black box behind a URL; everything about it that the record
needs comes from /server_info, the LSA settings included.

A run goes in phases - the reasoning tasks, then LongBench v2 and GraphWalks, then PaperTrail, then RULER (`Suite.PHASES`) -
because a reasoning chain served beside a long prompt's chunked prefill decodes only between its chunks.
Within a phase every task and every repeat is in flight at once, drawing from one pool as large as the
server's own request cap, so the server never drains between two tasks. The one exception is a context that
several plans' prompts open with - a PaperTrail condition's - which goes one at a time, prefilled by a single
request before its questions are sent. A task's `seconds` runs from when its
prompts are ready to its last reply, waits for a slot included. Tasks overlap, so the run's own `seconds` is the
wall clock of the whole.

One record can span several servers - a run continued on another job, or a length only a server spanning
nodes holds. `servers` keeps each one by its job id and every task names the one that answered it.
"""

import asyncio
import contextlib
import dataclasses
import json
import os
import platform
import socket
import time
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp

from lsa import io
from lsa.evaluation.accuracy import pool, records, tasks
from lsa.evaluation.accuracy.client import Client, Messages, user_message
from lsa.evaluation.accuracy.data import (
    DATA_ROOT,
    GRAPHWALKS_PROMPT_MAX,
    load_manifest,
    longbench_card,
    ruler_path,
    suites,
)
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Attention, Evaluation, Mesh, Records, Suite
from lsa.sglang import settings
from lsa.sglang.constants import Plugin
from lsa.sglang.settings.design import SELECTOR

HERE = os.path.dirname(os.path.abspath(__file__))

Log = Callable[[str], Any]


def server_info(base_url: str) -> dict[str, Any]:
    """Return a server's `/server_info`.

    Args:
        base_url: The server's http root.

    Returns:
        The parsed answer.
    """
    with urllib.request.urlopen(base_url.rstrip("/") + "/server_info", timeout=120) as response:
        info: dict[str, Any] = json.load(response)
    return info


def request_cap(info: dict[str, Any]) -> int:
    """Return how many requests a server runs at once, which sglang may hold under its `--max-running-requests`.

    sglang lowers the cap at start where another cache runs out first - a hybrid model's state cache holds a few
    requests - and reports the lowered one per data-parallel rank among its internal states.

    Args:
        info: The server's `/server_info`.
    """
    asked: int = info.get("max_running_requests") or 0
    effective = [
        state["effective_max_running_requests_per_dp"]
        for state in info.get("internal_states") or []
        if state.get("effective_max_running_requests_per_dp")
    ]
    if not asked or not effective:
        return asked
    return min(asked, min(effective) * (info.get("dp_size") or 1))


def served_settings(info: dict[str, Any]) -> dict[str, Any] | None:
    """Return the LSA settings a server runs under, read off its `/server_info`.

    The plugin's block rides in the model config override, which sglang reports as the server's
    `json_model_override_args`. The record quotes the block without its `debug` aids.

    Args:
        info: The server's `/server_info`.

    Returns:
        The settings, or None where the server runs the checkpoint's own indexer.
    """
    override = info.get("json_model_override_args") or {}
    if isinstance(override, str):
        override = json.loads(override)
    block = settings.block_of(override)
    if block is None:
        return None
    body = {name: value for name, value in block.items() if name != Plugin.DEBUG}
    return body or None


def unused_run_id(base: str, out_root: str) -> str:
    """Return a run id no directory under the root holds yet.

    Args:
        base: The id `build_run_id` derives.
        out_root: Where run directories land.

    Returns:
        `base`, or `base` with `-2`, `-3` and on for a second run of the same kind in one second.
    """
    run_id, counter = base, 1
    while os.path.exists(os.path.join(out_root, run_id)):
        counter += 1
        run_id = f"{base}-{counter}"
    return run_id


def check_schema(existing: dict[str, Any], path: str) -> None:
    """Refuse to add to a record of another schema, which would leave one record in two shapes.

    Args:
        existing: The record already at the path.
        path: Where it is, for the message.

    Raises:
        ValueError: The record at the path is not of the schema this runner writes.
    """
    if existing.get("schema") != Records.SCHEMA:
        raise ValueError(
            f"{path} is a {existing.get('schema')} record and this runner writes {Records.SCHEMA}"
        )


def finished_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the successful rows, one per (doc, repeat) - the last one wins and failed rows are dropped."""
    kept = {}
    for row in rows:
        if row.get("error") is None:
            kept[(row["doc_id"], row["repeat"])] = row
    return list(kept.values())


def sample_row(item: tasks.Item, repeat: int, reply: Any) -> dict[str, Any]:
    """Return the sample row one reply to one prompt is written as.

    Args:
        item: The prompt.
        repeat: Which repeat of the prompt this was.
        reply: The client's `Reply`.

    Returns:
        The row, without its grade.
    """
    answer_tokens = None
    if reply.completion_tokens is not None:
        answer_tokens = reply.completion_tokens - (reply.reasoning_tokens or 0)
    first_token = {"first_token": reply.first_token} if reply.first_token is not None else {}
    return {
        "doc_id": item.doc_id,
        "repeat": repeat,
        "prompt_hash": item.prompt_hash,
        "target": item.target,
        "answer": reply.content,
        "reasoning": reply.reasoning,
        "prompt_tokens": reply.prompt_tokens,
        "cached_tokens": reply.cached_tokens,
        "completion_tokens": reply.completion_tokens,
        "reasoning_tokens": reply.reasoning_tokens,
        "answer_tokens": answer_tokens,
        "finish_reason": reply.finish_reason,
        "seconds": round(reply.seconds, 2),
        "error": reply.error,
        **first_token,
    }


def rebudget(plan: suites.Plan, max_tokens: int) -> suites.Plan:
    """Give a plan a different completion budget.

    A reasoning model can spend a whole budget inside its thinking block and emit nothing, which the record
    scores as a wrong answer rather than as a budget miss - so a task whose answer needs enumeration reads
    0.00 at every rung and says nothing about any of them. The suite's budgets were calibrated against
    GLM-5.3-Flash, and a model that reasons at a different length needs its own ceiling.

    The prompt set is untouched, because sampling does not enter a prompt - so the recorded prompt-set hash
    still holds and a rebudgeted run stays comparable on its prompts.

    Args:
        plan: The plan to rebudget.
        max_tokens: Completion budget to give it.

    Returns:
        The plan with that budget.
    """
    return dataclasses.replace(
        plan, sampling=dataclasses.replace(require_sampling(plan), max_tokens=max_tokens)
    )


def resized(plan: suites.Plan, samples: int) -> suites.Plan:
    """Ask a RULER plan for its file's first `samples` rows rather than the grid's count.

    The plan's key names the count, so a resized prompt set is held to no hash pinned for the grid's.

    Args:
        plan: A plan of the suite.
        samples: Rows to take.

    Returns:
        The plan with that count, or the plan itself when it is not RULER.
    """
    if plan.dataset != Suite.RULER:
        return plan
    return dataclasses.replace(plan, n=samples)


def sampled(plan: suites.Plan, generation: dict[str, Any] | None) -> suites.Plan:
    """Give a plan the sampling of its dataset, unless it states its own.

    Args:
        plan: A plan of the suite.
        generation: The run's generation block, one request-settings mapping per dataset, or None.

    Returns:
        The plan with its sampling set, and its own `max_tokens` over the dataset's where it has one.

    Raises:
        ValueError: The plan states no sampling and the run has no entry for its dataset.
    """
    if plan.sampling is not None:
        return plan
    if not generation or plan.dataset not in generation:
        raise ValueError(
            f"{plan.task} is sampled by the {plan.dataset!r} entry of a generation block, and this run has "
            f"{sorted(generation or {})} - the experiment yaml says how its model is sampled"
        )
    sampling = Sampling(**generation[plan.dataset])
    if plan.max_tokens:
        sampling = dataclasses.replace(sampling, max_tokens=plan.max_tokens)
    return dataclasses.replace(plan, sampling=sampling)


def require_sampling(plan: suites.Plan) -> Sampling:
    """Return a plan's sampling, which `sampled` sets on every plan a run holds.

    Args:
        plan: A plan of the run.

    Returns:
        What its requests send.

    Raises:
        ValueError: The plan was never given its sampling.
    """
    if plan.sampling is None:
        raise ValueError(f"{plan.task} has no sampling - `sampled` gives every plan of a run its own")
    return plan.sampling


def targeted_items(
    definition: tasks.TaskDefinition, lm_eval_task: Any, plan: suites.Plan
) -> list[tasks.Item]:
    """Build the prompts a plan asks of a task, each with its target.

    Args:
        definition: The task.
        lm_eval_task: Its lm-eval task, built.
        plan: Which docs of it.

    Returns:
        The items, in doc id order.
    """
    items = tasks.build_items(lm_eval_task, plan.ids())
    for item in items:
        item.target = definition.target(item.doc)
    return items


def data_block(
    definition: tasks.TaskDefinition, lm_eval_task: Any, data_root: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Return which file a task read and what the data root's manifest says of it, or the hub dataset.

    Args:
        definition: The task.
        lm_eval_task: Its lm-eval task.
        data_root: The fixed prompt files' root.
        manifest: What generation recorded about every file under it.

    Returns:
        The task's `data` block.
    """
    if definition.path is None:
        return {"hub": lm_eval_task.config.dataset_path, "name": lm_eval_task.config.dataset_name}
    file = os.path.relpath(definition.path, data_root)
    entry = manifest.get(file)
    return {"root": data_root, "file": file, "corpus": records.corpus_name(entry), **(entry or {})}


def grade_rows(
    plan: suites.Plan,
    definition: tasks.TaskDefinition,
    lm_eval_task: Any,
    items: list[tasks.Item],
    rows: list[dict[str, Any]],
    run_dir: str,
    data: dict[str, Any],
    enforced: bool | None,
) -> dict[str, Any]:
    """Grade one task's rows through its own filters and scorer, and rewrite its sample file in order.

    Args:
        plan: The task's plan.
        definition: The task.
        lm_eval_task: Its lm-eval task.
        items: Its prompts.
        rows: Its finished sample rows.
        run_dir: The run directory, which holds `samples/<task>.jsonl`.
        data: What `data_block` says the task read.
        enforced: Whether the server held the task's thinking budget.

    Returns:
        The task's record entry, without the timing, prompt and server fields.
    """
    # Repeat by repeat, because the task filters work on the whole instance list
    by_id = {item.doc_id: item for item in items}
    rows = [row for row in rows if row["doc_id"] in by_id]
    for repeat in sorted({row["repeat"] for row in rows}):
        part = [row for row in rows if row["repeat"] == repeat and row.get("error") is None]
        metrics = tasks.grade(
            lm_eval_task, [by_id[row["doc_id"]] for row in part], [row["answer"] for row in part]
        )
        for row, values in zip(part, metrics, strict=False):
            row["metrics"] = values
            row["extracted"] = definition.extract(row["answer"])
    # What the viewer shows of a doc, kept beside its reply so a reader of the file can split by it
    for row in rows:
        meta = definition.meta(by_id[row["doc_id"]].doc)
        if meta:
            row["meta"] = meta
    rows.sort(key=lambda row: (row["repeat"], row["doc_id"]))
    path = samples_path(run_dir, plan)
    io.write_jsonl(path, rows)
    return {
        "benchmark": definition.benchmark,
        "category": definition.category,
        "length": definition.length,
        # RULER: the median prompt length of the fixed samples in tokens
        "length_actual": data.get("prompt_median") if definition.benchmark == Suite.RULER else None,
        "key": plan.key,
        "n_docs": len(items),
        "repeats": plan.repeats,
        **records.summarize(rows, definition.metric),
        **(
            {"breakdown": records.breakdown(rows, definition.metric, definition.breakdown)}
            if definition.breakdown
            else {}
        ),
        "sampling": dataclasses.asdict(require_sampling(plan)),
        "samples": os.path.relpath(path, run_dir),
        "data": data,
        "observed": records.observed(rows, enforced),
    }


def grade_plan(
    plan: suites.Plan,
    definition: tasks.TaskDefinition,
    lm_eval_task: Any,
    items: list[tasks.Item],
    rows: list[dict[str, Any]],
    run_dir: str,
    data_root: str,
    manifest: dict[str, Any],
    enforced: bool | None,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    """Grade one task into the record entry both a run and a regrade write, minus the timing and the server.

    Args:
        plan: The task's plan.
        definition: The task.
        lm_eval_task: Its lm-eval task.
        items: Its prompts.
        rows: Its finished sample rows.
        run_dir: The run directory, which holds `samples/<task>.jsonl`.
        data_root: The fixed prompt files' root.
        manifest: What generation recorded about every file under it.
        enforced: Whether the server held the task's thinking budget.
        prompt: How its prompts were assembled, which carries their hash.

    Returns:
        The task's record entry, with its prompt-set hash and protocol key.
    """
    data = data_block(definition, lm_eval_task, data_root, manifest)
    entry = grade_rows(plan, definition, lm_eval_task, items, rows, run_dir, data, enforced)
    entry["prompt_set_hash"] = prompt["hash"]
    entry["prompt"] = prompt
    entry["protocol"] = records.protocol_key(entry)
    return entry


def samples_path(run_dir: str, plan: suites.Plan) -> str:
    """Return where a run keeps one task's sample rows."""
    return os.path.join(run_dir, "samples", f"{plan.task}.jsonl")


def score_line(plan: suites.Plan, definition: tasks.TaskDefinition, entry: dict[str, Any]) -> str:
    """Return the progress line a graded task is logged with."""
    return (
        f"  {plan.key}: {definition.metric} {entry['score']}  n {entry['scored']}/{entry['n']}  empty {entry['empty']}"
        f"  truncated {entry['truncated']} (finished {entry['score_finished']})"
    )


class ContextGate:
    """Let the plans of one shared context send at a time, once a first request has prefilled it.

    Questions over one context share its KV only once it is in the radix cache. Sent together cold, each prefills
    the whole context - at 512k that ran native's indexer and LSA's bucket cache out of memory. So one context is in
    flight at a time, and a request of one token puts it in the cache before its questions go.
    """

    def __init__(self) -> None:
        """Set the changed condition, the context in flight, its holders and the prefills."""
        self.changed = asyncio.Condition()
        self.current: str | None = None
        self.holders = 0
        self.prefills: dict[str, asyncio.Future[Any]] = {}

    @contextlib.asynccontextmanager
    async def admitted(self, context: str, prefill: Callable[[], Awaitable[Any]]) -> AsyncIterator[None]:
        """Wait for the context's turn and its prefill, then hold it in flight until the block ends.

        Args:
            context: The shared context's name.
            prefill: Sends the request that prefills it - the first plan of the context to arrive sends it, and
                every other one waits on that request.
        """
        async with self.changed:
            await self.changed.wait_for(lambda: self.current in (None, context))
            self.current = context
            self.holders += 1
        try:
            if context not in self.prefills:
                self.prefills[context] = asyncio.ensure_future(prefill())
            await self.prefills[context]
            yield
        finally:
            async with self.changed:
                self.holders -= 1
                if not self.holders:
                    self.current = None
                    self.changed.notify_all()


class Runner:
    """One suite against one server into one record, continued where a run of the same id stopped."""

    def __init__(
        self,
        base_url: str,
        suite_name: str,
        out_root: str | None = None,
        note: str = "",
        tag: str = "",
        name: str = "",
        only: list[str] | None = None,
        lengths: list[int] | None = None,
        samples: int | None = None,
        top_logprobs: int | None = None,
        data_root: str = DATA_ROOT,
        max_context: int | None = None,
        resume: bool = True,
        strict_hashes: bool = True,
        concurrency_scale: float = 1.0,
        timeout: float = 14400.0,
        log: Log = print,
        run_id: str | None = None,
        max_tokens: int | None = None,
        generation: dict[str, Any] | None = None,
        model: str | None = None,
        attention_kind: str | None = None,
        image: str | None = None,
        gpu: str | None = None,
        pins: str | None = None,
    ) -> None:
        """Ask the server what it is, pick the plans, and take up whatever a run of the same id left.

        Args:
            base_url: The server's http root.
            suite_name: One of `suites.SUITES`.
            out_root: Where run directories land, or None for the records root's accuracy directory.
            note: Free text for the record.
            tag: A short label appended to the run id.
            name: What the config calls the run, `run.name` - the record is grouped by it and a derived run id
                starts with it. Defaults to `run_id`.
            only: Task names or plan keys to run, or None for the whole suite.
            lengths: RULER lengths to run, or None for every length.
            samples: Rows of every RULER task, or None for the grid's count.
            top_logprobs: Alternatives each reply's first token is kept with, or None to keep no logprobs.
            data_root: The fixed RULER and LongBench files.
            max_context: Which LongBench v2 file to read.
            resume: Keep the samples a run directory already holds.
            strict_hashes: Stop on a prompt set whose hash is not the pinned one, rather than warn.
            concurrency_scale: Scales every capped plan's concurrency.
            timeout: Seconds one request may take.
            log: Where progress lines go.
            run_id: The run directory to write, or None to derive one from `name`.
            max_tokens: A completion budget for every plan, or None for each dataset's own.
            generation: One request-settings mapping per dataset.
            model: The hub id the record names the model by, or None to name it after the served path.
            attention_kind: The method the record files the run under, one of `Attention.RECORDED`, or None
                where the caller does not know it.
            image: The container the server runs in, or None where the caller does not know it.
            gpu: The card model the server runs on, as the device reports it, or None where the caller does not know it.
            pins: The set of prompt-set pins `data_root`'s files are held to (`suites.pins_path`), None for the
                default corpus's.

        Raises:
            ValueError: Neither `name` nor `run_id` is given, the server reports no request cap, a LongBench
                budget runs past its context, or the run directory holds a record of another schema.
        """
        self.base_url = base_url
        self.suite_name = suite_name
        self.plans = suites.select(suite_name, only, lengths)
        if samples:
            self.plans = [resized(plan, samples) for plan in self.plans]
        self.top_logprobs = top_logprobs
        self.generation = generation
        self.plans = [sampled(plan, generation) for plan in self.plans]
        if max_tokens:
            self.plans = [rebudget(plan, max_tokens) for plan in self.plans]
        out_root = out_root or records.records_dir(Evaluation.ACCURACY)
        self.info = server_info(base_url)
        self.in_flight = request_cap(self.info)
        if not self.in_flight:
            raise ValueError(
                f"the server at {base_url} reports no max_running_requests, which sizes the pool every task "
                "draws from - launch it with --max-running-requests"
            )
        self.config = served_settings(self.info)
        self.indexer = Attention.NATIVE if self.config is None else self.config.get(SELECTOR, Attention.LSA)
        self.image = image
        self.gpu = gpu
        self.label = name or run_id or ""
        if not self.label:
            raise ValueError(
                "a run is named by its config - give the runner a name or the run id to continue"
            )
        # A given run id continues that run's directory - samples already there are kept
        self.run_id = run_id or unused_run_id(records.build_run_id(self.label, tag), out_root)
        self.out = os.path.join(out_root, self.run_id)
        self.note = note
        self.tag = tag
        self.data_root = data_root
        self.max_context = max_context or longbench_card().context
        self.resume = resume
        self.strict = strict_hashes
        self.concurrency_scale = concurrency_scale
        self.timeout = timeout
        self.log = log
        self.contexts = ContextGate()
        self.expected_hashes: dict[str, str] = io.read_json(suites.pins_path(pins), {})
        self.manifest = load_manifest(data_root)
        self.check_budgets()
        existing: dict[str, Any] = {}
        if resume:
            existing = io.read_json(os.path.join(self.out, Records.RECORD), None) or {}
        if existing:
            check_schema(existing, os.path.join(self.out, Records.RECORD))
        # A continued run adds to the record it finds - the tasks run earlier, the plans they came from, the
        # time the earlier launches spent and every server that answered them
        prior_run = existing.get("run", {})
        self.results: dict[str, Any] = existing.get("tasks", {})
        self.skipped: dict[str, str] = existing.get("skipped", {})
        self.prior_plans: list[dict[str, Any]] = prior_run.get("plans", [])
        self.first_server: dict[str, Any] | None = existing.get("server")
        self.first_suite: str | None = prior_run.get("suite")
        self.prior_seconds: float = prior_run.get("seconds") or 0.0
        self.prior_servers: dict[str, Any] = existing.get("servers", {})
        # A resumed run launched without a hub id keeps the one its first launch recorded
        prior_model = existing.get("model", {}).get("name")
        self.model = model or prior_model or os.path.basename(self.info["model_path"].rstrip("/"))
        self.attention_kind = attention_kind or existing.get("attention", {}).get("kind")
        # Pinned here rather than left to `header`, which is called once before the first task and again
        # after every one of them. Deriving it from the clock each time made `started` the moment of the
        # last write, so a six-hour run recorded a start a second before its end and the field said nothing
        self.started: str = prior_run.get("started") or io.utc_now()
        self.start = time.perf_counter()
        # Only what a label said survives a continued run - complete or partial is worked out at every write
        self.edits = prior_run.get("edits", [])
        self.launched = io.utc_now()
        self.job = os.environ.get("SLURM_JOB_ID")
        self.nodes = int(os.environ.get("SLURM_JOB_NUM_NODES") or self.info.get("nnodes") or 1)
        self.server_id = self.job or f"{socket.gethostname()}-{self.launched}"
        map_path = (self.config or {}).get("projections")
        self.map = {"path": map_path, "sha256": io.file_sha256(map_path)} if map_path else None

    def check_budgets(self) -> None:
        """Refuse a LongBench budget the server cannot hold over the prompts prepared for it.

        The files are truncated to `max_context`, so where that fits the server every prompt fits too - and
        then a prompt plus its completion budget has to fit as well, or the longest prompts come back as
        http 400 and the task is scored on the rest. A server shorter than `max_context` loses those
        prompts whatever the budget, so it is not checked here.

        Raises:
            ValueError: The budget over the prepared prompts runs past the server's context.
        """
        context = self.info.get("context_length")
        if not context or self.max_context > context:
            return
        for plan in self.plans:
            budget = require_sampling(plan).max_tokens
            if plan.dataset == Suite.LONGBENCH and self.max_context + budget > context:
                raise ValueError(
                    f"longbench2 prompts of up to {self.max_context} tokens and a budget of {budget} run past "
                    f"the server's {context} - the budget is what the context leaves over the longest prompt"
                )

    def check_resumable(self, plan: suites.Plan, rows: list[dict[str, Any]]) -> None:
        """Refuse to add samples to a task that an earlier run of this id sampled differently.

        A finished task carries its sampling in its entry. An unfinished one has only the plan the earlier
        run wrote into the header, which is what it was sampling when it stopped.

        Args:
            plan: The plan about to run.
            rows: The finished samples already on disk for it.

        Raises:
            ValueError: The earlier samples were drawn under sampling other than this plan's.
        """
        planned = {prior["task"]: prior.get("sampling") for prior in self.prior_plans}
        earlier = self.results.get(plan.task, {}).get("sampling") or (
            planned.get(plan.task) if rows else None
        )
        sampling = dataclasses.asdict(require_sampling(plan))
        if earlier is not None and earlier != sampling:
            raise ValueError(
                f"{plan.task} was scored in {self.out} under {earlier} and this run samples it as "
                f"{sampling} - rerun with --no-resume or give the run its own tag"
            )

    def all_plans(self) -> list[dict[str, Any]]:
        """Return the plans of this run and of the earlier runs of the same id, one per task, the newest winning."""
        merged = {plan["task"]: plan for plan in self.prior_plans}
        for plan in self.plans:
            merged[plan.task] = dataclasses.asdict(plan)
        return list(merged.values())

    def header(self) -> dict[str, Any]:
        """Return the whole record as it stands."""
        import lm_eval

        now = io.utc_now()
        return {
            "schema": Records.SCHEMA,
            "run": {
                "id": self.run_id,
                "name": self.label,
                # A continued run keeps the suite it started as - run.plans says what every task came from
                "suite": self.first_suite or self.suite_name,
                "note": self.note,
                "tag": self.tag,
                "edits": self.edits,
                "generation": self.generation,
                "started": self.started,
                "date": now,
                "seconds": round(self.prior_seconds + time.perf_counter() - self.start, 1),
                "host": socket.gethostname(),
                "gpu": self.gpu,
                "server_url": self.base_url,
                "commits": {"lsa": io.git_sha(HERE)},
                "versions": {
                    "lm_eval": lm_eval.__version__,
                    "python": platform.python_version(),
                    "sglang": self.info.get("version"),
                },
                "plans": self.all_plans(),
            },
            "model": {
                "name": self.model,
                "label": os.path.basename(self.model).lower(),
                "path": self.info["model_path"],
            },
            "attention": {
                "kind": self.attention_kind,
                "indexer": self.indexer,
                "config": self.config,
                "map": self.map,
            },
            # The server a run started on stays where it was
            "server": self.first_server or self.server_block(),
            "servers": {**self.prior_servers, self.server_id: self.server_entry()},
            "tasks": self.results,
            "skipped": self.skipped,
        }

    def server_block(self) -> dict[str, Any]:
        """Return the serving settings that matter, with what this run worked out about the indexer."""
        return {
            **{key: self.info.get(key) for key in Records.SERVER_KEYS},
            "indexer": self.indexer,
            "config": self.config,
            Mesh.FIELD: records.mesh(self.info, self.image),
        }

    def server_entry(self) -> dict[str, Any]:
        """Return this launch's server as the record's `servers` holds it - where it ran, on what, and when."""
        return {
            "job": self.job,
            "host": socket.gethostname(),
            "nodes": self.nodes,
            "url": self.base_url,
            "date": self.launched,
            "lsa": io.git_sha(HERE),
            **self.server_block(),
        }

    def prompt_block(self, plan: suites.Plan, lm_eval_task: Any, digest: str) -> dict[str, Any]:
        """Return how a task's prompts were assembled, and whether their hash was held to a pinned one."""
        return {
            "delimiter": lm_eval_task.config.target_delimiter,
            "gen_prefix": lm_eval_task.config.gen_prefix,
            "hash": digest,
            "pinned": self.expected_hashes.get(plan.key),
            "strict": self.strict,
        }

    def write(self) -> None:
        """Write the record as it stands."""
        io.write_json(os.path.join(self.out, Records.RECORD), self.header(), indent=1)

    def check_hash(self, plan: suites.Plan, digest: str) -> None:
        """Hold a plan's prompt-set hash to the pinned one.

        Args:
            plan: The plan.
            digest: The hash of the prompts it built.

        Raises:
            RuntimeError: The hash differs from the pinned one and the run is strict about it.
        """
        expected = self.expected_hashes.get(plan.key)
        if expected is None:
            self.log(f"  {plan.key}: prompt-set hash {digest} (no expected value recorded)")
        elif expected != digest:
            message = f"{plan.key}: prompt-set hash {digest} differs from the recorded {expected} - the prompts are not the fixed ones"
            if self.strict:
                raise RuntimeError(message)
            self.log("  WARNING " + message)

    def manifest_entry(self, definition: tasks.TaskDefinition) -> dict[str, Any] | None:
        """Return what generation recorded about this RULER or PaperTrail task's fixed samples, or None."""
        if definition.benchmark == Suite.RULER:
            ruler_task, length = definition.name[len(Suite.RULER_PREFIX) :].rsplit("_", 1)
            path = ruler_path(ruler_task, int(length), self.data_root)
        else:
            path = definition.path or ""
        entry: dict[str, Any] | None = self.manifest.get(os.path.relpath(path, self.data_root))
        return entry

    def over_context(self, definition: tasks.TaskDefinition, plan: suites.Plan) -> str | None:
        """Return why a RULER length, a GraphWalks tier or a PaperTrail condition does not fit this server, or None.

        So one suite serves every deployment.

        Args:
            definition: The task.
            plan: Its plan, whose budget has to fit beside the longest prompt.

        Returns:
            The reason, or None when it fits or the server or the manifest does not say.

        Raises:
            ValueError: The file's manifest entry has no prompt sizes - lm-eval's port wrote it, not NVIDIA's
                generators.
        """
        context = self.info.get("context_length")
        if definition.benchmark == Suite.GRAPHWALKS and context:
            needed = GRAPHWALKS_PROMPT_MAX[definition.name] + require_sampling(plan).max_tokens
            return (
                None
                if needed <= context
                else f"{needed} tokens of prompt and answer over the server's {context}"
            )
        if definition.benchmark not in (Suite.RULER, Suite.PAPERTRAIL):
            return None
        entry = self.manifest_entry(definition)
        if not entry or not context:
            return None
        if "prompt_max" not in entry:
            raise ValueError(
                f"{definition.name}: the manifest under {self.data_root} records no prompt sizes, so its files predate "
                "NVIDIA's generators - point --data-root at the NVIDIA corpus"
            )
        needed = entry["prompt_max"] + require_sampling(plan).max_tokens
        if needed <= context:
            return None
        return f"{needed} tokens of prompt and answer over the server's {context}"

    # A question past its shared context is a few hundred tokens
    # => this bounds what each further question adds beside its answer budget
    QUESTION_TOKENS: int = 4096

    def task_concurrency(self, plan: suites.Plan) -> int:
        """Return how many requests of a plan's cap group may be in flight at once, never above the server's cap.

        Questions over a shared context take what the KV pool holds instead of the plan's cap where they share what
        the attention keeps of the context. Each then reads only its own selection, so a batch shares a decode
        step's fixed cost, while full attention streams the whole context once a request.
        """
        if plan.concurrency is None:
            return self.in_flight
        if plan.shared_context is not None and self.shares_context_index():
            return min(self.in_flight, self.questions_in_pool(plan))
        return min(self.in_flight, max(1, int(plan.concurrency * self.concurrency_scale)))

    def shares_context_index(self) -> bool:
        """Say whether requests over one cached context share the index the attention reads it through.

        Block sparse attention's index is the cached keys' own. LSA's buckets are shared under its prefix reuse,
        which takes effect only where the server caches prefixes - without it each request builds buckets over the
        whole context, and a second one at millions of tokens does not fit beside the first.
        """
        if self.attention_kind == Attention.BLOCK_SPARSE:
            return True
        reuse = bool((self.config or {}).get("prefix_reuse"))
        return self.attention_kind == Attention.LSA and reuse and not self.info.get("disable_radix_cache")

    def questions_in_pool(self, plan: suites.Plan) -> int:
        """Return how many questions over one shared context the server's KV pool holds at once.

        The first is bounded by the server's context length, its prompt and answer together. Each further one
        adds only its question and its answer budget, since the context sits in the radix cache once.

        Args:
            plan: A plan of a shared context.
        """
        pool_tokens = self.info.get("max_total_num_tokens")
        context = self.info.get("context_length")
        if not pool_tokens or not context:
            return 1
        each = require_sampling(plan).max_tokens + self.QUESTION_TOKENS
        return 1 + max(0, (pool_tokens - context) // each)

    @contextlib.asynccontextmanager
    async def context_turn(
        self, plan: suites.Plan, client: Client, session: aiohttp.ClientSession, first: Messages
    ) -> AsyncIterator[None]:
        """Hold a plan's shared context in flight, if it has one, once a first request has prefilled it.

        Args:
            plan: The plan about to send.
            client: The plan's client.
            session: The session from `client.session`.
            first: A conversation of the plan, whose context the prefill puts in the cache.
        """
        context = plan.shared_context
        if context is None:
            yield
            return

        async def prefill() -> None:
            """Send one token's worth of the first conversation, and log what the cache served of it."""
            sampling = dataclasses.replace(require_sampling(plan), max_tokens=1)
            reply = (await client.gather(session, [first], sampling))[0]
            outcome = f"error {reply.error}" if reply.error else f"{reply.cached_tokens} of them cached"
            self.log(
                f"  {context}: prefilled {reply.prompt_tokens} prompt tokens in {round(reply.seconds, 1)} s, {outcome}"
            )

        async with self.contexts.admitted(context, prefill):
            yield

    async def run_plan(self, plan: suites.Plan, limits: list[asyncio.Semaphore]) -> None:
        """Send what a plan still needs, every repeat at once, then grade it into the results.

        Args:
            plan: The plan to run.
            limits: The semaphores each of its requests holds - its cap group's, if it has a cap, then the
                pool every plan of the run draws from.
        """
        definition = tasks.resolve(plan.task, self.data_root, self.max_context)
        # An earlier run's skip is retried
        self.skipped.pop(plan.task, None)
        reason = self.over_context(definition, plan)
        if reason:
            self.skipped[plan.task] = reason
            self.log(f"  {plan.key}: SKIPPED - {reason}")
            return
        try:
            lm_eval_task = definition.build()
        except (FileNotFoundError, ValueError, OSError) as exc:
            # No fixed samples for this task yet - the record says so and the run goes on
            self.skipped[plan.task] = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.log(f"  {plan.key}: SKIPPED - {self.skipped[plan.task]}")
            return
        items = targeted_items(definition, lm_eval_task, plan)
        digest = tasks.prompt_set_hash(items)
        self.check_hash(plan, digest)

        path = samples_path(self.out, plan)
        rows = finished_rows(pool.task_rows(self.out, plan.task)) if self.resume else []
        if rows:
            self.check_resumable(plan, rows)
        done = {(row["doc_id"], row["repeat"]) for row in rows}
        # Failed rows are retried, so they leave the file first
        if rows or os.path.exists(path):
            io.write_jsonl(path, rows)
        client = Client(
            self.base_url,
            concurrency=self.task_concurrency(plan),
            timeout=self.timeout,
            top_logprobs=self.top_logprobs,
        )
        # From when this task could first send, so a wait for a slot of the pool counts
        started = time.perf_counter()
        todo = [
            (item, repeat)
            for repeat in range(plan.repeats)
            for item in items
            if (item.doc_id, repeat) not in done
        ]
        if todo:
            self.log(
                f"  {plan.key}: {len(todo)} requests over {plan.repeats} repeats, {len(done)} already done"
            )

            def on_reply(index: int, reply: Any) -> None:
                """Keep a reply the moment it lands, so a stopped run loses none."""
                item, repeat = todo[index]
                records.append_jsonl(path, sample_row(item, repeat, reply))

            async with client.session() as session:
                messages = [user_message(item.prompt) for item, _ in todo]
                async with self.context_turn(plan, client, session, messages[0]):
                    await client.gather(
                        session, messages, require_sampling(plan), on_reply=on_reply, limits=limits
                    )
            rows = io.read_jsonl(path)
        seconds = round(time.perf_counter() - started, 3)
        enforced = records.budget_enforced(dataclasses.asdict(require_sampling(plan)), self.info)
        entry = grade_plan(
            plan,
            definition,
            lm_eval_task,
            items,
            rows,
            self.out,
            self.data_root,
            self.manifest,
            enforced,
            self.prompt_block(plan, lm_eval_task, digest),
        )
        entry.update({"concurrency": client.concurrency, "seconds": seconds, "server": self.server_id})
        output_tokens = entry["tokens"]["output"]["sum"] if entry["tokens"]["output"] else None
        entry["output_tokens_per_second"] = (
            round(output_tokens / seconds, 1) if output_tokens and seconds else None
        )
        self.results[plan.task] = entry
        self.log(f"{score_line(plan, definition, entry)}  errors {entry['errors']}  {seconds} s")
        if entry["errors"]:
            self.check_server_alive()

    def check_server_alive(self) -> None:
        """After failed requests, stop if the server is gone rather than fail every remaining task.

        Raises:
            RuntimeError: The server no longer answers.
        """
        try:
            server_info(self.base_url)
        # Broad on purpose - any failure at all means the server is unreachable
        except Exception as exc:
            self.write()
            raise RuntimeError(
                f"server at {self.base_url} is not answering ({type(exc).__name__}) - run stopped; "
                f"continue it with --run-id {self.run_id}"
            ) from exc

    async def run_plans(self) -> None:
        """Run the plans phase by phase, every plan of a phase at once, writing the record as each finishes.

        A phase starts when the last reply of the one before it is in, so a long prompt's prefill never holds
        up a reasoning chain's decode. Every plan draws from one pool of the server's size, and a capped plan
        also holds a slot of its cap group, one semaphore for every plan of the group - so the RULER tasks of
        one length are bounded together, at the cap one of them had when tasks ran in turn. The loop is one
        thread, so the record has one writer and no write sees a task half graded.
        """
        shared = asyncio.Semaphore(self.in_flight)
        group_limits: dict[str, asyncio.Semaphore] = {}
        for plan in self.plans:
            if plan.concurrency is not None:
                group_limits.setdefault(plan.cap_group, asyncio.Semaphore(self.task_concurrency(plan)))

        async def run_and_write(plan: suites.Plan) -> None:
            """Run one plan under its cap group's limit and the pool's, then write the record."""
            group = [group_limits[plan.cap_group]] if plan.concurrency is not None else []
            await self.run_plan(plan, [*group, shared])
            self.write()

        for phase in sorted({plan.phase for plan in self.plans}):
            plans = [plan for plan in self.plans if plan.phase == phase]
            self.log(
                f"phase {phase + 1} of {len(Suite.PHASES)}: {len(plans)} tasks, {', '.join(Suite.PHASES[phase])}"
            )
            await asyncio.gather(*(run_and_write(plan) for plan in plans))

    def run(self) -> dict[str, Any]:
        """Run every plan and return the record it wrote."""
        os.makedirs(self.out, exist_ok=True)
        io.write_json(os.path.join(self.out, Records.SERVER_INFO), self.info, indent=1)
        self.log(f"{self.run_id}: {len(self.plans)} tasks -> {self.out}, {self.in_flight} requests in flight")
        self.write()
        asyncio.run(self.run_plans())
        self.log(
            f"done in {round(time.perf_counter() - self.start, 1)} s: {os.path.join(self.out, Records.RECORD)}"
        )
        if self.skipped:
            self.log(f"skipped {len(self.skipped)} tasks without data: {' '.join(self.skipped)}")
        return self.header()
