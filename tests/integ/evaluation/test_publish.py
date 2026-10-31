"""That a run reaches the shared root whole, in its kind's directory, as it is when it is finished, as a snapshot
when it is still going, and over a published copy only when asked - the copy swapped, a joined run's parts
archived."""

import json

import pytest

from lsa.evaluation import launches, publish
from lsa.evaluation.constants import Kernels, Records

STARTED = {"job": None, "user": "alice", "host": "node-1", "started": "2026-10-01T10:00:00+00:00"}
ENDED = {"job": None, "ended": "2026-10-01T11:00:00+00:00", "outcome": "completed"}
RUNNING = {"job": "4242", "user": "alice", "host": "node-1", "started": "2026-10-01T10:00:00+00:00"}


@pytest.fixture(autouse=True)
def slurm(monkeypatch):
    """Slurm as a dict - job 4242 runs, every other job is unknown to it."""
    states = {"4242": "RUNNING"}
    monkeypatch.setattr(
        launches,
        "accounting",
        lambda jobs, fields=("State",): {j: {"State": states[j]} for j in jobs if j in states},
    )
    return states


def a_run(results, run_id, scored=25, schema=Records.SCHEMA, logged=(STARTED, ENDED), meta=True, parts=None):
    run = results / run_id
    (run / "samples").mkdir(parents=True)
    (run / "samples" / "ruler_vt_4096.jsonl").write_text("{}\n")
    if schema == Kernels.SCHEMA:
        record = {"schema": schema, "run": {"id": run_id}, "failed": [] if scored == 25 else [1048576]}
    else:
        plans = [{"task": "ruler_vt_4096"}]
        record = {
            "schema": schema,
            "run": {"id": run_id, "plans": plans, **({"parts": [{"id": p} for p in parts]} if parts else {})},
            "tasks": {"ruler_vt_4096": {"n": 25, "scored": scored}},
        }
    (run / "record.json").write_text(json.dumps(record))
    for line in logged:
        launches.append(run, line)
    if meta:
        (run / "meta.json").write_text("{}")
    return run


def target(run, root, **flags):
    return publish.plan_publication(run, root, **flags).target


def read(path):
    return json.loads((path / "record.json").read_text())


def test_a_complete_run_is_copied_whole_into_its_kinds_directory(tmp_path):
    run = a_run(tmp_path / "results" / "accuracy", "m__native__ruler__2026-10-01__10-00-00")
    root = tmp_path / "evaluations"

    [published] = publish.publish([run], root, apply=True)

    assert published == root / "accuracy" / run.name
    assert (published / "samples" / "ruler_vt_4096.jsonl").read_text() == "{}\n"
    assert (published / "launches.jsonl").is_file()
    assert read(published) == read(run)
    assert run.is_dir()
    # Built a level up and renamed in - nothing left beside it
    assert sorted(path.name for path in root.iterdir()) == ["accuracy"]


def test_a_kernels_run_goes_under_performance(tmp_path):
    run = a_run(tmp_path / "results", "m__native__prefill__2026-10-01__10-00-00", schema=Kernels.SCHEMA)
    assert (
        target(run, tmp_path / "evaluations").parent == tmp_path / "evaluations" / "performance" / "kernels"
    )


def test_a_dry_run_copies_nothing(tmp_path):
    run = a_run(tmp_path / "results", "a-run")
    publish.publish([run], tmp_path / "evaluations", apply=False)
    assert not (tmp_path / "evaluations").exists()


def test_a_run_whose_launch_failed_or_never_ended_is_refused(tmp_path):
    failed = a_run(tmp_path / "results", "failed-run", logged=(STARTED, {**ENDED, "outcome": "failed"}))
    with pytest.raises(ValueError, match="failed"):
        target(failed, tmp_path / "evaluations", partial=True, in_progress=True)
    # A launch with no end line slurm knows nothing of
    open_ = a_run(tmp_path / "results", "open-run", logged=(STARTED,))
    with pytest.raises(ValueError, match="unknown"):
        target(open_, tmp_path / "evaluations", partial=True, in_progress=True)


def test_a_partial_run_needs_saying_so(tmp_path):
    partial = a_run(tmp_path / "results", "partial-run", scored=20)
    with pytest.raises(ValueError, match="--partial"):
        target(partial, tmp_path / "evaluations")
    assert target(partial, tmp_path / "evaluations", partial=True).name == "partial-run"

    kernels = a_run(tmp_path / "results", "kernels-run", scored=0, schema=Kernels.SCHEMA)
    with pytest.raises(ValueError, match="1048576 did not finish"):
        target(kernels, tmp_path / "evaluations")


def test_a_run_with_no_launch_log_is_published_once_its_meta_is_written(tmp_path):
    imported = a_run(tmp_path / "results", "imported-run", logged=())
    assert target(imported, tmp_path / "evaluations").name == "imported-run"


def test_a_run_with_no_launch_log_and_no_meta_is_not_published(tmp_path):
    unfinished = a_run(tmp_path / "results", "unfinished-run", logged=(), meta=False)
    with pytest.raises(ValueError, match="no launch log"):
        target(unfinished, tmp_path / "evaluations", partial=True)


def test_a_published_run_is_never_overwritten_and_one_refusal_publishes_nothing(tmp_path):
    root = tmp_path / "evaluations"
    done = a_run(tmp_path / "results", "done-run")
    publish.publish([done], root, apply=True)
    fresh = a_run(tmp_path / "other", "fresh-run")
    again = a_run(tmp_path / "again", "done-run")

    with pytest.raises(FileExistsError, match="--replace"):
        publish.publish([fresh, again], root, apply=True)
    assert not (root / "accuracy" / "fresh-run").exists()


def test_a_running_run_is_published_only_as_a_snapshot(tmp_path):
    root = tmp_path / "evaluations"
    running = a_run(tmp_path / "results", "running-run", scored=10, logged=(RUNNING,), meta=False)
    # A half-written sample line at the end of a file the job appends to
    (running / "samples" / "ruler_vt_4096.jsonl").write_text('{"doc_id": 0}\n{"doc_id": 1, "ans')
    with pytest.raises(ValueError, match="--in-progress"):
        target(running, root, partial=True)

    [published] = publish.publish([running], root, apply=True, in_progress=True)

    snapshot = read(published)["run"]["snapshot"]
    assert snapshot["job"] == "4242"
    assert "RUNNING" in snapshot["state"]
    assert snapshot["source"] == str(running)
    assert (published / "samples" / "ruler_vt_4096.jsonl").read_text() == '{"doc_id": 0}\n'
    # The source keeps its record as it was
    assert "snapshot" not in read(running)["run"]


def test_replace_swaps_a_fresh_copy_in_for_the_published_one(tmp_path, slurm):
    root = tmp_path / "evaluations"
    run = a_run(tmp_path / "results", "growing-run", scored=10, logged=(RUNNING,), meta=False)
    publish.publish([run], root, apply=True, in_progress=True)
    (run / "samples" / "late.jsonl").write_text("{}\n")
    record = read(run)
    record["tasks"]["ruler_vt_4096"]["scored"] = 25
    (run / "record.json").write_text(json.dumps(record))
    launches.append(run, {**ENDED, "job": "4242"})

    with pytest.raises(FileExistsError, match="--replace"):
        publish.publish([run], root, apply=True)
    [published] = publish.publish([run], root, apply=True, replace=True)

    assert "snapshot" not in read(published)["run"]
    assert (published / "samples" / "late.jsonl").is_file()
    assert sorted(path.name for path in root.iterdir()) == ["accuracy"]
    assert sorted(path.name for path in (root / "accuracy").iterdir()) == ["growing-run"]


def test_a_joined_run_archives_its_parts_and_its_parts_are_not_published_alone_again(tmp_path):
    root = tmp_path / "evaluations"
    base = a_run(tmp_path / "results", "base-run", scored=20)
    extension = a_run(tmp_path / "results", "base-run-5m")
    publish.publish([base, extension], root, apply=True, partial=True)
    joined = a_run(tmp_path / "joined", "base-run", parts=["base-run", "base-run-5m"])

    with pytest.raises(FileExistsError, match="--replace"):
        publish.publish([joined], root, apply=False)
    publish.publish([joined], root, apply=True, replace=True)

    archive = tmp_path / "archive" / "evaluations" / "accuracy"
    assert sorted(path.name for path in archive.iterdir()) == ["base-run", "base-run-5m"]
    assert read(archive / "base-run")["run"].get("parts") is None
    assert [path.name for path in (root / "accuracy").iterdir()] == ["base-run"]
    assert read(root / "accuracy" / "base-run")["run"]["parts"]
    for part in (base, extension):
        with pytest.raises(ValueError, match="published as a part of base-run"):
            target(part, root, partial=True, replace=True)


def test_sync_publishes_what_is_worth_showing_and_passes_over_the_rest(tmp_path):
    root = tmp_path / "evaluations"
    results = tmp_path / "results"
    a_run(results / "accuracy", "done-run")
    a_run(results / "accuracy", "short-run", scored=20)
    a_run(results / "accuracy", "running-run", scored=10, logged=(RUNNING,), meta=False)
    a_run(results / "accuracy", "failed-run", logged=(STARTED, {**ENDED, "outcome": "failed"}))
    a_run(results / "performance" / "kernels", "kernels-run", schema=Kernels.SCHEMA)

    assert publish.sync([results], root, apply=False) == [
        root / "accuracy" / name for name in ("done-run", "running-run", "short-run")
    ] + [root / "performance" / "kernels" / "kernels-run"]
    assert not root.exists()
    publish.sync([results], root, apply=True)
    assert sorted(path.name for path in (root / "accuracy").iterdir()) == [
        "done-run",
        "running-run",
        "short-run",
    ]

    # Unchanged runs are passed over, the snapshot is refreshed once its record moves
    assert publish.sync([results], root, apply=False) == []
    record = read(results / "accuracy" / "running-run")
    record["tasks"]["ruler_vt_4096"]["scored"] = 15
    (results / "accuracy" / "running-run" / "record.json").write_text(json.dumps(record))
    assert publish.sync([results], root, apply=True) == [root / "accuracy" / "running-run"]
    assert read(root / "accuracy" / "running-run")["tasks"]["ruler_vt_4096"]["scored"] == 15

    # A launch log or a sample that moved with the record unchanged is refreshed too
    launches.append(results / "accuracy" / "done-run", STARTED)
    launches.append(results / "accuracy" / "done-run", ENDED)
    (results / "accuracy" / "short-run" / "samples" / "ruler_vt_4096.jsonl").write_text('{}\n{"doc_id": 1}\n')
    assert publish.sync([results], root, apply=False) == [
        root / "accuracy" / "done-run",
        root / "accuracy" / "short-run",
    ]


def test_a_pooled_run_is_published_without_its_queue(tmp_path):
    run = a_run(tmp_path / "results" / "accuracy", "m__native__ruler__2026-10-01__10-00-00")
    claims = run / "pool" / "claims" / "ruler_vt_4096"
    claims.mkdir(parents=True)
    (claims / "0-0.0").write_text("{}")
    shard = run / "pool" / "shards" / "w1"
    shard.mkdir(parents=True)
    (shard / "ruler_vt_4096.jsonl").write_text("{}\n")
    root = tmp_path / "evaluations"
    (published,) = publish.publish([run], root, apply=True)
    assert (published / "samples" / "ruler_vt_4096.jsonl").is_file()
    assert not (published / "pool").exists()


def test_the_archive_is_named_evaluations_however_the_root_is_spelled(tmp_path):
    assert (
        publish.archive_dir(tmp_path / "evaluation", "accuracy")
        == tmp_path / "archive" / "evaluations" / "accuracy"
    )
    assert (
        publish.archive_dir(tmp_path / "anything", "accuracy")
        == tmp_path / "archive" / "evaluations" / "accuracy"
    )
