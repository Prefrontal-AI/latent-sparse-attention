"""The dispatcher against a fake cluster - which run gets a free node, and when it stops."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsa import io, site, slurm
from lsa.evaluation import dispatch
from lsa.evaluation.accuracy import pool
from lsa.evaluation.accuracy.data import suites
from lsa.evaluation.constants import Dispatch, Launches, Pool, Records
from tests.helpers.sites import use_site

RESERVATION = "lsa"


class Cluster:
    """What squeue, sinfo and scontrol say, as one object a test sets up and `slurm.ask` reads from."""

    def __init__(self):
        self.nodes = ["n1", "n2", "n3", "n4"]
        self.states = {}
        # (job id, name, state, nodes, reservation)
        self.jobs = []
        self.asked = []

    def ask(self, command):
        self.asked.append(list(command))
        tool = command[0]
        if tool == "squeue":
            return "".join(
                f"{i}|{name}|{state}|{nodes}|{reservation}|bob\n"
                for i, name, state, nodes, reservation in self.jobs
            )
        if tool == "sinfo":
            return "".join(f"{node}|{self.states.get(node, 'reserved')}\n" for node in self.nodes)
        if command[:3] == ["scontrol", "show", "reservation"]:
            return f"ReservationName={RESERVATION} Nodes=gpu[1-4] Users=bob\n"
        if command[:3] == ["scontrol", "show", "hostnames"]:
            expression = command[3]
            return "\n".join(self.nodes if expression == "gpu[1-4]" else expression.split(",")) + "\n"
        raise AssertionError(command)


@pytest.fixture(autouse=True)
def on_slurm(monkeypatch, tmp_path_factory):
    use_site(monkeypatch, tmp_path_factory.mktemp("site"), "root: /test/root\nlauncher: slurm\n")


@pytest.fixture
def cluster(monkeypatch):
    fake = Cluster()
    monkeypatch.setattr(slurm, "ask", fake.ask)
    return fake


def test_a_node_is_free_when_it_is_up_and_nothing_runs_on_it(cluster):
    cluster.states = {"n3": "drained", "n4": "down*"}
    cluster.jobs = [("1", "lsa-battery", "RUNNING", "n1", RESERVATION)]
    assert slurm.free_nodes(RESERVATION) == ["n2"]


def test_a_pending_job_in_the_reservation_takes_a_free_node_before_it_runs(cluster):
    cluster.jobs = [
        ("1", "x", "RUNNING", "n1", RESERVATION),
        ("2", "lsa-pool-a", "PENDING", "", RESERVATION),
        ("3", "elsewhere", "PENDING", "", "other"),
    ]
    dispatcher = dispatch.Dispatcher(settings([]))
    assert dispatcher.free_slots() == 2


def test_a_node_kept_free_is_never_given_away(cluster):
    dispatcher = dispatch.Dispatcher(settings([], keep_free=1))
    assert dispatcher.free_slots() == 3


def test_a_squeue_that_fails_stops_the_dispatcher_rather_than_reading_an_empty_queue(monkeypatch):
    def broken(command):
        raise RuntimeError("squeue failed")

    monkeypatch.setattr(slurm, "ask", broken)
    with pytest.raises(RuntimeError, match="squeue failed"):
        dispatch.Dispatcher(settings([])).free_slots()


def settings(runs, **flags):
    values = {
        "runs": runs,
        "reservation": RESERVATION,
        "max_workers": 3,
        "keep_free": 0,
        "default_unit_seconds": 1200.0,
        "max_failures": Dispatch.MAX_FAILURES,
        "dry_run": True,
        "parameters": {},
        "folder": None,
    }
    return dispatch.Settings(**{**values, **flags})


def measured(name, remaining, work_seconds, live=0, queued=0, slots=None, stalled=""):
    return dispatch.Measured(
        Path(name),
        remaining=remaining,
        live=live,
        queued=queued,
        work_seconds=work_seconds,
        slots_needed=remaining if slots is None else slots,
        stalled=stalled,
    )


def test_the_run_that_would_finish_last_gets_the_first_node_and_the_next_the_one_after(cluster):
    slow = measured("slow", 20, 20 * 1200.0)
    quick = measured("quick", 5, 5 * 1200.0)
    given = dispatch.Dispatcher(settings([], max_workers=9)).decide([quick, slow], 4)
    # slow is 24000 s alone, 12000 with two, 8000 with three - still above quick's 6000 - so it takes all four
    assert [m.run_dir.name for m in given] == ["slow"] * 4
    assert quick.added == 0


def test_once_the_slow_run_is_under_the_next_the_next_gets_one(cluster):
    slow = measured("slow", 20, 20 * 1200.0)
    quick = measured("quick", 15, 15 * 1200.0)
    given = dispatch.Dispatcher(settings([], max_workers=9)).decide([quick, slow], 5)
    # slow is 24000 s with none or one worker and 12000 with two, under quick's 18000, which two workers halve in turn
    assert [m.run_dir.name for m in given] == ["slow", "slow", "quick", "quick", "slow"]


def test_a_run_stops_at_its_maximum_workers(cluster):
    slow = measured("slow", 100, 100 * 1200.0)
    other = measured("other", 2, 2 * 1200.0)
    given = dispatch.Dispatcher(settings([], max_workers=2)).decide([slow, other], 6)
    assert slow.added == 2
    assert other.added == 2
    assert len(given) == 4


def test_a_run_gets_no_more_workers_than_its_unclaimed_requests_can_use(cluster):
    # Two slots of work left, one worker already starting for them
    run = measured("run", 3, 3 * 1200.0, live=1, queued=2, slots=1.4)
    assert run.room(8) == 1
    assert measured("empty", 0, 0.0).room(8) == 0
    claimed_only = measured("claimed", 4, 4 * 1200.0, live=4, queued=4, slots=0.0)
    assert claimed_only.room(8) == 0


def test_a_run_whose_launches_all_failed_is_left_alone(cluster):
    broken = measured("broken", 50, 50 * 1200.0, stalled="its last 3 launches failed")
    fine = measured("fine", 1, 1200.0)
    given = dispatch.Dispatcher(settings([])).decide([broken, fine], 3)
    assert [m.run_dir.name for m in given] == ["fine"]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def a_run(tmp_path, name="run-a", done=0, seconds=600.0, claimed=0, logged=()):
    run = tmp_path / "results" / "accuracy" / name
    io.write_json(
        run / Pool.DIR / Pool.SUBMISSION,
        {"config": "x.yaml", "overrides": [], "index": 0, "kind": "accuracy", "output": "o", "id": name},
    )
    io.write_json(run / Records.SERVER_INFO, {"max_running_requests": 1})
    write_rows(
        pool.shard_file(run, "w1", "ruler_cwe_10485760"),
        [{"doc_id": doc, "repeat": 0, "error": None, "seconds": seconds} for doc in range(done)],
    )
    queue = pool.Queue(run, "w1")
    queue.claim([pool.Unit("ruler_cwe_10485760", doc, 0) for doc in range(done, done + claimed)], claimed)
    for line in logged:
        (run / "launches.jsonl").open("a").write(json.dumps(line) + "\n")
    return run


PLAN = suites.Plan("ruler_cwe_10485760", n=20, concurrency=1)


def look(run, queued=0, now=None):
    return dispatch.measure(run, [PLAN], 1.0, queued, pool.Ledger(run), 1200.0, now=now)


def test_what_is_left_is_the_plan_less_what_the_shards_hold_and_what_is_claimed(tmp_path):
    run = a_run(tmp_path, done=5, claimed=2)
    found = look(run)
    assert (found.remaining, found.claimed) == (15, 2)
    # Fifteen requests left, one at a time per worker, so thirteen need a worker of their own
    assert found.slots_needed == 13
    assert found.seconds_per_unit == 600.0
    assert found.work_seconds == 15 * 600.0


def test_a_request_nothing_has_measured_costs_the_default(tmp_path):
    found = look(a_run(tmp_path, done=0))
    assert (found.seconds_per_unit, found.work_seconds) == (1200.0, 20 * 1200.0)


def test_a_claim_nobody_touched_in_the_lease_is_not_counted_as_held(tmp_path):
    run = a_run(tmp_path, done=0, claimed=3)
    assert look(run).claimed == 3
    assert look(run, now=time.time() + Pool.LEASE_SECONDS + 60).claimed == 0


def test_workers_the_run_has_divide_its_time(tmp_path):
    run = a_run(tmp_path, done=0)
    alone = look(run)
    pool.write_worker(run, "w1", {"heartbeat": time.time()})
    pool.write_worker(run, "w2", {"heartbeat": time.time()})
    with_two = look(run, queued=2)
    assert with_two.live == 2
    assert with_two.projected_seconds == alone.projected_seconds / 2


def test_a_run_whose_last_launches_failed_is_stalled(tmp_path):
    failed = {"job": "1", "ended": "x", "outcome": Launches.FAILED, "error": "server died"}
    start = {"job": "1", "started": "x"}
    run = a_run(tmp_path, logged=[start, failed] * 3)
    assert "server died" in dispatch.stalled_by(run, 3)
    assert dispatch.stalled_by(run, 4) == ""


def test_dispatch_takes_the_sites_reservation_and_refuses_a_site_with_none(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        dispatch, "Dispatcher", lambda settings: seen.append(settings) or SimpleNamespace(step=lambda: None)
    )
    monkeypatch.setattr(sys, "argv", ["dispatch", str(tmp_path), "--once"])
    use_site(monkeypatch, tmp_path, "root: /test/root\n" + "launcher: slurm\nslurm:\n  reservation: pool\n")
    dispatch.main()
    assert seen[0].reservation == "pool"
    assert seen[0].parameters == {"slurm_additional_parameters": {"reservation": "pool"}}
    use_site(monkeypatch, tmp_path, "root: /test/root\n" + "launcher: local\n")
    with pytest.raises(site.Unsupported, match="sets none"):
        dispatch.main()
