"""Workers of one pooled run against fake servers - no duplicates, a lease that expires, and the record a single job writes."""

import importlib
import os
import tempfile
import threading
from collections import Counter
from unittest import mock

import pytest

from lsa import io
from lsa.evaluation.accuracy import pool, runner, worker
from lsa.evaluation.accuracy.data import suites
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Pool
from tests.helpers.evaluation.fakes import FakeServer, make_data, respond

INFO = {"model_path": "zai-org/Fake-Model", "max_running_requests": 4, "version": "fake"}


def slow(messages):
    return {**respond(messages), "delay": 0.15}


def plans():
    sampling = Sampling(max_tokens=64)
    return [
        suites.Plan("ruler_niah_single_1_4096", n=4, repeats=2, sampling=sampling, concurrency=2),
        suites.Plan("ruler_cwe_4096", n=6, sampling=sampling, concurrency=2),
    ]


@pytest.fixture(scope="module")
def data():
    root = tempfile.mkdtemp()
    make_data(root)
    return root


@pytest.fixture
def servers():
    started = []

    def start(count):
        for _ in range(count):
            started.append(FakeServer(slow, info=INFO).start())
        return started[-count:]

    yield start
    for server in started:
        server.stop()


def make(server, data, out, worker_name=None, pooled=True, run_id="pooled", **kwargs):
    cls = worker.PoolRunner if pooled else runner.Runner
    extra = {"worker": worker_name, **kwargs} if pooled else {}
    return cls(
        server.base_url,
        "pool",
        out_root=out,
        data_root=data,
        log=lambda *_: None,
        name="test",
        run_id=run_id,
        **extra,
    )


def meeting(*servers):
    """Make each server hold its first answer until every server has been asked, so all of them are in the run."""
    asked = [threading.Event() for _ in servers]

    def hold(own):
        def answer(messages):
            asked[own].set()
            for event in asked:
                assert event.wait(60), "a worker never reached the run"
            return slow(messages)

        return answer

    for i, server in enumerate(servers):
        server.respond = hold(i)


def run_all(*runs):
    # tqdm builds its class lock on first use, which two threads building tasks at once can race
    importlib.import_module("tqdm").tqdm.get_lock()
    threads = [threading.Thread(target=run.run) for run in runs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def rows_sampled(run_dir):
    sent = []
    for shard in pool.pool_dir(run_dir).glob(f"{Pool.SHARDS}/*/*.jsonl"):
        sent += [(shard.stem, row["doc_id"], row["repeat"]) for row in io.read_jsonl(shard)]
    return sent


@pytest.fixture(autouse=True)
def suite():
    with mock.patch.dict(suites.SUITES, {"pool": plans}):
        yield


def test_two_workers_share_a_run_and_no_request_is_sent_twice(data, servers, monkeypatch):
    # A worker names its server by the slurm job it runs in, and both run in the one job of a test run there
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    first, second = servers(2)
    meeting(first, second)
    out = tempfile.mkdtemp()
    run_all(make(first, data, out, "w1"), make(second, data, out, "w2"))
    run_dir = os.path.join(out, "pooled")
    sent = rows_sampled(run_dir)
    # Every (task, doc, repeat) exactly once, split between the two shards and so the two servers
    assert len(sent) == 14
    assert max(Counter(sent).values()) == 1
    assert len(first.requests) + len(second.requests) == 14
    assert first.requests and second.requests
    rec = io.read_json(os.path.join(out, "pooled", "record.json"))
    assert {task: entry["scored"] for task, entry in rec["tasks"].items()} == {
        "ruler_niah_single_1_4096": 8,
        "ruler_cwe_4096": 6,
    }
    assert set(pool.read_workers(os.path.join(out, "pooled"))) == {"w1", "w2"}
    # Every worker is a server of the one record
    assert len(rec["servers"]) == 2


def test_the_pooled_record_is_the_single_job_record(data, servers):
    (alone,) = servers(1)
    single_out = tempfile.mkdtemp()
    make(alone, data, single_out, pooled=False, run_id="single").run()
    first, second = servers(2)
    pooled_out = tempfile.mkdtemp()
    run_all(make(first, data, pooled_out, "w1"), make(second, data, pooled_out, "w2"))
    expected = io.read_json(os.path.join(single_out, "single", "record.json"))["tasks"]
    found = io.read_json(os.path.join(pooled_out, "pooled", "record.json"))["tasks"]
    assert set(found) == set(expected)
    for task, entry in expected.items():
        for field in (
            "n",
            "scored",
            "errors",
            "empty",
            "truncated",
            "score",
            "repeat_scores",
            "metrics",
            "tokens",
        ):
            assert found[task][field] == entry[field], (task, field)
    # The graded rows are the same too, in the same order
    for task in expected:
        single = io.read_jsonl(os.path.join(single_out, "single", "samples", f"{task}.jsonl"))
        pooled = io.read_jsonl(os.path.join(pooled_out, "pooled", "samples", f"{task}.jsonl"))
        assert [(row["doc_id"], row["repeat"], row["metrics"]) for row in pooled] == [
            (row["doc_id"], row["repeat"], row["metrics"]) for row in single
        ]


def test_workers_of_one_run_under_different_caps_record_the_rows_each_served(data, servers):
    first, second = servers(2)
    meeting(first, second)
    out = tempfile.mkdtemp()
    run_all(make(first, data, out, "w1"), make(second, data, out, "w2", concurrency_scale=0.5))
    entry = io.read_json(os.path.join(out, "pooled", "record.json"))["tasks"]["ruler_cwe_4096"]
    served = {int(cap): count for cap, count in entry["concurrency_rows"].items()}
    # Every row is counted under the cap of the worker that sent it, and the entry names the largest
    assert set(served) <= {1, 2} and sum(served.values()) == 6
    assert entry["concurrency"] == max(served)


def test_samples_a_single_job_left_count_as_done(data, servers):
    (alone,) = servers(1)
    out = tempfile.mkdtemp()
    make(alone, data, out, pooled=False).run()
    path = os.path.join(out, "pooled", "samples", "ruler_cwe_4096.jsonl")
    rows = io.read_jsonl(path)
    io.write_jsonl(path, rows[:2])
    (helper,) = servers(1)
    make(helper, data, out, "late").run()
    # Only the four rows missing were sent, and the task is whole again
    assert len(helper.requests) == 4
    rec = io.read_json(os.path.join(out, "pooled", "record.json"))
    assert rec["tasks"]["ruler_cwe_4096"]["scored"] == 6
    assert len(io.read_jsonl(path)) == 6
    # The single job's rows carry no cap, the late worker's carry its own
    assert rec["tasks"]["ruler_cwe_4096"]["concurrency_rows"] == {"2": 4, "unknown": 2}
    assert len(pool.task_rows(os.path.join(out, "pooled"), "ruler_cwe_4096")) == 10


def test_a_worker_that_died_holding_units_has_them_taken_once_its_lease_expires(data, servers):
    (helper,) = servers(1)
    out = tempfile.mkdtemp()
    run_dir = os.path.join(out, "pooled")
    dead = pool.Queue(run_dir, "dead")
    units = [pool.Unit("ruler_cwe_4096", doc, 0) for doc in range(6)]
    taken = dead.claim(units, 3)
    assert len(taken) == 3
    # The worker was killed, so nothing touches its claims any more
    for path in dead.held.values():
        os.utime(path, (1, 1))
    make(helper, data, out, "survivor").run()
    # Its three units were taken over, and all of ruler_niah's too
    assert len(helper.requests) == 14
    claims = os.listdir(os.path.join(run_dir, Pool.DIR, Pool.CLAIMS, "ruler_cwe_4096"))
    assert sorted(name for name in claims if name.startswith("0-0")) == ["0-0.0", "0-0.1"]
    rec = io.read_json(os.path.join(run_dir, "record.json"))
    assert rec["tasks"]["ruler_cwe_4096"]["scored"] == 6


def test_a_live_claim_is_left_to_the_worker_that_holds_it(data, servers):
    (helper,) = servers(1)
    out = tempfile.mkdtemp()
    run_dir = os.path.join(out, "pooled")
    holder = pool.Queue(run_dir, "holder")
    holder.claim([pool.Unit("ruler_cwe_4096", 0, 0), pool.Unit("ruler_cwe_4096", 1, 0)], 2)
    make(helper, data, out, "other").run()
    assert len(helper.requests) == 12
    rec = io.read_json(os.path.join(run_dir, "record.json"))
    # The task is not whole until the holder's units are, so the record does not claim it
    assert "ruler_cwe_4096" not in rec["tasks"]
    assert rec["tasks"]["ruler_niah_single_1_4096"]["scored"] == 8


def test_a_worker_keeps_one_shared_context_in_flight_and_sends_its_first_question_alone(data):
    # niah_single_2 opens with a second context, which the fake tells apart by its text
    rows = io.read_jsonl(os.path.join(data, "ruler", "niah_single_1_4096.jsonl"))
    second = [{**row, "input": row["input"].replace("Text", "Second text")} for row in rows]
    io.write_jsonl(os.path.join(data, "ruler", "niah_single_2_4096.jsonl"), second)
    contexts = {
        "ruler_cwe_4096": "first",
        "ruler_niah_single_1_4096": "first",
        "ruler_niah_single_2_4096": "second",
    }

    def grouped(messages):
        group = "second" if "Second" in messages[-1]["content"] else "first"
        return {**slow(messages), "group": group}

    server = FakeServer(grouped, info=INFO).start()
    plans = [suites.Plan(task, n=3, sampling=Sampling(max_tokens=64)) for task in contexts]
    try:
        with (
            mock.patch.dict(suites.SUITES, {"pool": lambda: plans}),
            mock.patch.object(suites.Plan, "shared_context", property(lambda plan: contexts[plan.task])),
        ):
            make(server, data, tempfile.mkdtemp(), "w").run()
    finally:
        server.stop()
    events = server.events
    # Each context's first question is alone on the server, and the second context waits for every reply of the first
    assert events[:2] == [("in", "first"), ("out", "first")]
    second_starts = events.index(("in", "second"))
    assert events[second_starts + 1] == ("out", "second")
    assert max(i for i, event in enumerate(events) if event == ("out", "first")) < second_starts
    # Once prefilled, a context fills the server's cap
    assert server.peak_of == {"first": 4, "second": 2}


def run_one_slot(data, plans, contexts, slots, cap, extra_info=None, **kwargs):
    info = {**INFO, "max_running_requests": cap, **(extra_info or {})}
    server = FakeServer(slow, info=info, slots=slots).start()
    out = tempfile.mkdtemp()
    try:
        with (
            mock.patch.dict(suites.SUITES, {"pool": lambda: plans}),
            mock.patch.object(suites.Plan, "shared_context", property(lambda plan: contexts.get(plan.task))),
        ):
            make(server, data, out, "w", **kwargs).run()
    finally:
        server.stop()
    rows = [
        row
        for shard in pool.pool_dir(os.path.join(out, "pooled")).glob(f"{Pool.SHARDS}/*/*.jsonl")
        for row in io.read_jsonl(shard)
    ]
    return server, rows


def test_a_plan_at_the_server_s_cap_queues_one_request_behind_it_once_its_context_is_cached(data):
    plans = [suites.Plan("ruler_cwe_4096", n=6, sampling=Sampling(max_tokens=64), concurrency=1)]
    server, rows = run_one_slot(data, plans, {"ruler_cwe_4096": "one"}, slots=1, cap=1)
    # The first question prefills the context alone, then one more always waits behind the one running
    assert server.events[:2] == [("in", None), ("out", None)]
    assert server.peak == 1 and server.peak_waiting == 1
    # A row's time is its time in the slot, and the wait is recorded beside it
    queued = [row for row in rows if "queued_seconds" in row]
    assert queued and all(row["queued_seconds"] > 0.05 for row in queued)
    assert all(row["seconds"] < 0.3 for row in rows)
    assert len(rows) == 6


def test_a_queued_request_may_wait_a_whole_request_beyond_its_own_timeout(data):
    # Each answer takes 0.15 s against a 0.25 s timeout, so a request queued behind another would time out at 0.3 s
    plans = [suites.Plan("ruler_cwe_4096", n=6, sampling=Sampling(max_tokens=64), concurrency=1)]
    server, rows = run_one_slot(data, plans, {"ruler_cwe_4096": "one"}, slots=1, cap=1, timeout=0.25)
    assert server.peak_waiting == 1
    assert not any(row["error"] for row in rows)
    assert len(rows) == 6


def test_a_plan_capped_below_the_server_queues_nothing(data):
    plans = [suites.Plan("ruler_cwe_4096", n=6, sampling=Sampling(max_tokens=64), concurrency=2)]
    server, rows = run_one_slot(data, plans, {}, slots=4, cap=4)
    assert server.peak == 2 and server.peak_waiting == 0
    assert not any("queued_seconds" in row for row in rows)


@pytest.mark.parametrize(("questions_in_pool", "peak", "waiting"), [(3, 3, 1), (2, 2, 0)])
def test_block_sparse_questions_fill_the_pool_and_queue_one_more_only_at_the_server_s_cap(
    data, questions_in_pool, peak, waiting
):
    # A pool of one context and the questions beside it - each a 64-token budget and the question bound
    each = 64 + runner.Runner.QUESTION_TOKENS
    pool_room = {"context_length": 100_000, "max_total_num_tokens": 100_000 + (questions_in_pool - 1) * each}
    plans = [suites.Plan("ruler_cwe_4096", n=6, sampling=Sampling(max_tokens=64), concurrency=1)]
    server, rows = run_one_slot(
        data,
        plans,
        {"ruler_cwe_4096": "one"},
        slots=3,
        cap=3,
        extra_info=pool_room,
        attention_kind="block_sparse",
    )
    # The pool's cap replaces the plan's, and only a cap at the server's own keeps one more waiting behind it
    assert server.peak == peak and server.peak_waiting == waiting
    assert all(row["concurrency"] == peak for row in rows)
    assert any("queued_seconds" in row for row in rows) == bool(waiting)
    assert len(rows) == 6


def test_a_prompt_the_server_refuses_stops_being_sent_after_the_attempts(data):
    def refuse(messages):
        return {"status": 400, "error": "the prompt is longer than the context length"}

    server = FakeServer(refuse, info=INFO).start()
    try:
        out = tempfile.mkdtemp()
        with mock.patch.dict(suites.SUITES, {"pool": lambda: plans()[1:]}):
            # A failed request ends a launch's try at a unit and the server still answers, so each launch is one more
            with mock.patch.object(worker.PoolRunner, "check_server_alive", lambda self: None):
                for attempt in range(Pool.MAX_ATTEMPTS):
                    make(server, data, out, f"w{attempt}").run()
            before = len(server.requests)
            make(server, data, out, "last").run()
        assert len(server.requests) == before
        rec = io.read_json(os.path.join(out, "pooled", "record.json"))
        entry = rec["tasks"]["ruler_cwe_4096"]
        assert (entry["scored"], entry["errors"]) == (0, 6)
    finally:
        server.stop()


def test_the_runner_timeout_is_the_pool_clients(data, servers, tmp_path, monkeypatch):
    sent = []
    real = worker.Client

    def spy(*args, **kwargs):
        sent.append(kwargs["timeout"])
        return real(*args, **kwargs)

    monkeypatch.setattr(worker, "Client", spy)
    (server,) = servers(1)
    run_all(make(server, data, str(tmp_path), "w1", timeout=28800.0))
    assert sent == [28800.0]
