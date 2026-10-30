"""The launch log beside a record, and the status and owner a run list reads from it."""

import subprocess

import pytest

from lsa.evaluation import launches

STARTED = {"job": "200", "user": "alice", "host": "node-3", "started": "2026-10-01T10:00:00+00:00"}


def ended(outcome, **extra):
    return {"job": STARTED["job"], "ended": "2026-10-01T11:00:00+00:00", "outcome": outcome, **extra}


def test_an_end_line_closes_the_launch_it_follows():
    second = {**STARTED, "job": "201", "user": "bob"}
    folded = launches.fold([STARTED, ended("failed", error="boom"), second])
    assert [launch.get("outcome") for launch in folded] == ["failed", None]
    assert folded[0]["error"] == "boom"
    # The run is whoever started it first, however many launches finished it
    assert launches.owner(folded) == "alice"
    assert launches.unsettled(folded) == "201"


def test_launched_writes_a_start_and_an_end_line(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "300")
    monkeypatch.setenv("SLURM_JOB_USER", "alice")
    with launches.launched(tmp_path):
        assert launches.unsettled(launches.read(tmp_path)) == "300"
    (launch,) = launches.read(tmp_path)
    assert (launch["job"], launch["user"], launch["outcome"]) == ("300", "alice", "completed")


def test_launched_writes_what_the_runner_raised_and_raises_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "301")
    with pytest.raises(RuntimeError, match="server died"), launches.launched(tmp_path):
        raise RuntimeError("server died")
    with launches.launched(tmp_path):
        pass
    first, second = launches.read(tmp_path)
    assert (first["outcome"], first["error"]) == ("failed", "RuntimeError: server died")
    # A rerun into the same directory adds a launch and keeps the history
    assert second["outcome"] == "completed"


def test_a_stop_from_outside_leaves_the_launch_to_slurm(tmp_path):
    with pytest.raises(KeyboardInterrupt), launches.launched(tmp_path):
        raise KeyboardInterrupt
    assert "outcome" not in launches.read(tmp_path)[0]


def test_accounting_reads_one_row_per_job_under_both_its_ids(monkeypatch):
    out = "1001|900_3|alice|RUNNING\n1002|1002|bob|CANCELLED by 1004\n"
    calls = []

    def run(command, **_):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=out)

    monkeypatch.setattr(launches.shutil, "which", lambda name: "/usr/bin/sacct")
    monkeypatch.setattr(launches.subprocess, "run", run)
    found = launches.accounting(["1001", "1002"], ("User", "State"))
    assert found["1001"] == found["900_3"] == {"User": "alice", "State": "RUNNING"}
    assert found["1002"]["State"] == "CANCELLED by 1004"
    assert len(calls) == 1


def test_accounting_with_no_sacct_knows_nothing(monkeypatch):
    monkeypatch.setattr(launches.shutil, "which", lambda name: None)
    assert launches.accounting(["1001"]) == {}


def test_workers_that_end_out_of_order_each_close_their_own_launch():
    first = {**STARTED, "job": "201", "worker": "201"}
    second = {**STARTED, "job": "202", "worker": "202"}
    folded = launches.fold([STARTED, first, second, ended("failed", error="x") | {"job": "201"}])
    assert [launch.get("outcome") for launch in folded] == [None, "failed", None]
    assert launches.open_jobs(folded) == ["200", "202"]
    folded = launches.fold([first, second, {"job": "202", "ended": "t", "outcome": "completed"}])
    assert [launch.get("outcome") for launch in folded] == [None, "completed"]
    assert launches.pooled(folded)
    assert not launches.pooled([STARTED])


def test_a_pooled_launch_names_its_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "300")
    with launches.launched(tmp_path, worker="300"):
        pass
    (launch,) = launches.read(tmp_path)
    assert launch["worker"] == "300"
