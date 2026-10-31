"""The one status a run list shows and publish gates on, from the launch log, slurm and the record."""

import pytest

from lsa.evaluation import launches
from lsa.evaluation.status import accuracy_missing, kernels_missing, status

STARTED = {"job": "200", "user": "alice", "host": "node-3", "started": "2026-10-01T10:00:00+00:00"}


def ended(outcome, **extra):
    return {"job": STARTED["job"], "ended": "2026-10-01T11:00:00+00:00", "outcome": outcome, **extra}


def test_a_run_with_no_launch_log_has_no_status():
    assert status([], "", {}) == ("", "")


def test_a_run_with_a_meta_and_no_launch_log_reads_as_complete():
    """An imported run has no launcher to log, so its `meta.json`, written last, is its end"""
    assert status([], "", {}, meta_written=True)[0] == "complete"


def test_a_run_with_a_meta_and_no_launch_log_is_partial_with_work_left():
    result, why = status([], "1 of 3 tasks not scored", {}, meta_written=True)
    assert result == "partial"
    assert why.endswith("1 of 3 tasks not scored")


def test_the_end_line_says_complete_or_failed():
    assert status(launches.fold([STARTED, ended("completed")]), "", {})[0] == "complete"
    assert status(launches.fold([STARTED, ended("failed", error="ValueError: x")]), "", {}) == (
        "failed",
        "job 200: ValueError: x",
    )


def test_a_launch_that_ended_well_with_work_left_is_partial():
    result, why = status(launches.fold([STARTED, ended("completed")]), "1 of 3 tasks not scored", {})
    assert result == "partial"
    assert why.endswith("1 of 3 tasks not scored")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("RUNNING", "in progress"),
        ("PENDING", "in progress"),
        ("CANCELLED by 1004", "cancelled"),
        ("TIMEOUT", "timed out"),
        ("OUT_OF_MEMORY", "failed"),
        ("NODE_FAIL", "failed"),
        ("COMPLETED", "complete"),
    ],
)
def test_a_launch_with_no_end_line_is_what_slurm_says(state, expected):
    assert status([STARTED], "", {"200": state})[0] == expected


def test_slurm_saying_completed_still_answers_to_the_record():
    """A job that exited without the launcher's end line is only as finished as its record"""
    assert status([STARTED], "2 contexts did not finish", {"200": "COMPLETED"})[0] == "partial"


def test_a_launch_slurm_cannot_place_is_unknown():
    assert status([STARTED], "", {})[0] == "unknown"
    assert status([{**STARTED, "job": None}], "", {})[0] == "unknown"


def test_an_accuracy_run_misses_a_planned_task_unscored_or_short():
    record = {
        "run": {"plans": [{"task": "a"}, {"task": "b"}, {"task": "c"}]},
        "tasks": {"a": {"n": 25, "scored": 25}, "b": {"n": 25, "scored": 20}},
    }
    assert accuracy_missing(record) == "2 of 3 planned tasks not scored in full: b, c"
    record["tasks"] |= {"b": {"n": 25, "scored": 25}, "c": {"n": 5, "scored": 5}}
    assert accuracy_missing(record) == ""


def test_a_kernel_run_misses_the_contexts_that_did_not_finish():
    assert kernels_missing({"failed": []}) == ""
    assert kernels_missing({"failed": [1048576, 2097152]}) == "contexts 1048576, 2097152 did not finish"


def worker(job, **extra):
    return {**STARTED, "job": job, "worker": job, **extra}


def test_a_pool_is_in_progress_while_any_worker_is_alive():
    logged = launches.fold([worker("1"), worker("2"), ended("completed") | {"job": "1"}])
    result, why = status(logged, "3 of 5 planned tasks not scored in full: a", {"2": "RUNNING"})
    assert result == "in progress"
    assert "1 of 2 workers" in why


def test_a_pool_is_complete_once_its_record_is_whole_whichever_worker_died_on_the_way():
    logged = launches.fold([worker("1"), worker("2"), ended("completed") | {"job": "1"}])
    assert status(logged, "", {"2": "NODE_FAIL"})[0] == "complete"


def test_a_pool_with_work_left_and_no_worker_alive_is_what_its_last_launch_says():
    logged = launches.fold([worker("1"), ended("completed") | {"job": "1"}, worker("2")])
    assert status(logged, "1 of 2 planned tasks not scored in full: a", {"2": "TIMEOUT"})[0] == "timed out"
