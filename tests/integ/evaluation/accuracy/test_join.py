"""Joining the runs one configuration was split across - what joins, what it writes, and what refuses."""

import json

import pytest

from lsa.evaluation import launches, publish
from lsa.evaluation.accuracy.records import join
from lsa.evaluation.constants import Records, Status

CP_INFO = {
    "tp_size": 8,
    "attn_cp_size": 8,
    "dcp_size": 8,
    "enable_prefill_cp": True,
    "cp_strategy": "contiguous",
    "pp_size": 1,
    "ep_size": 8,
    "nnodes": 1,
    "moe_runner_backend": "deep_gemm",
}
RULER = {"temperature": 0.0, "max_tokens": 4096, "top_k": None, "think": False}
SHORT = ["ruler_vt_4096", "ruler_cwe_4096"]
LONG = ["ruler_vt_5000000", "ruler_cwe_5000000"]


@pytest.fixture(autouse=True)
def slurm(monkeypatch):
    """Slurm as a dict - job 99 runs, every other job is unknown to it."""
    states = {"99": "RUNNING"}
    monkeypatch.setattr(
        launches,
        "accounting",
        lambda jobs, fields=("State",): {j: {"State": states[j]} for j in jobs if j in states},
    )
    return states


def plan(task, **sampling):
    return {
        "task": task,
        "n": 20,
        "subset": None,
        "repeats": 1,
        "sampling": {**RULER, **sampling},
        "concurrency": 4,
    }


def server(**settings):
    return {
        "indexer": "lsa",
        "config": {"top_level": 0, "top_k": 4096},
        "tp_size": 8,
        "pp_size": 1,
        "nnodes": 1,
        "attention_backend": "fa3",
        "version": "fork",
        "context_length": 2100000,
        "chunked_prefill_size": 32768,
        **settings,
    }


def a_part(results, run_id, scored, planned, job, ended=True, served=None, plans=None, mesh=False, gpu=None):
    run_dir = results / "accuracy" / run_id
    (run_dir / "samples").mkdir(parents=True)
    tasks = {}
    for task in scored:
        (run_dir / "samples" / f"{task}.jsonl").write_text(json.dumps({"doc_id": 0}) + "\n")
        tasks[task] = {"n": 20, "scored": 20, "score": 0.5, "samples": f"samples/{task}.jsonl"}
    first = server(**(served or {}))
    first["mesh"] = {"label": "tp8-cp1-dcp1-pp1-ep8" if mesh else "tp1-cp8-dcp8-pp1-ep8"}
    record = {
        "schema": Records.SCHEMA,
        "run": {
            "id": run_id,
            "suite": "full",
            "note": "",
            "tag": "",
            "edits": [],
            "generation": {"ruler": RULER},
            "started": "2026-10-04T01:00:00+00:00",
            "date": f"2026-10-04T0{len(scored)}:00:00+00:00",
            "seconds": 100.0,
            "host": "node-1",
            "commits": {"lsa": "e0f8abc5"},
            "versions": {"sglang": "fork"},
            "plans": plans if plans is not None else [plan(task) for task in planned],
        },
        "model": {"name": "zai-org/GLM-5.3-Flash", "label": "glm-5.3-flash", "path": "/models/glm"},
        "attention": {
            "indexer": "lsa",
            "config": first["config"],
            "map": {"sha256": "abc"},
        },
        "server": first,
        "servers": {job: {**first, "job": job}},
        "tasks": tasks,
        "skipped": {task: "over the server's context" for task in planned if task not in scored},
    }
    if gpu:
        record["run"]["gpu"] = gpu
    (run_dir / Records.RECORD).write_text(json.dumps(record))
    (run_dir / Records.SERVER_INFO).write_text(json.dumps(CP_INFO))
    (run_dir / "server.log").write_text("log\n")
    launches.append(
        run_dir, {"job": job, "user": "bob", "host": "node-1", "started": "2026-10-04T01:00:00+00:00"}
    )
    if ended:
        launches.append(run_dir, {"job": job, "ended": "2026-10-04T09:00:00+00:00", "outcome": "completed"})
    return run_dir


@pytest.fixture
def battery(tmp_path):
    """A base that stopped at 4k and planned 5M, and a run at 5M on a longer server chunking 8k."""
    results = tmp_path / "results"
    base = a_part(results, "lsa-flat-cp__1", SHORT, SHORT + LONG, "11")
    extension = a_part(
        results,
        "lsa-flat-cp-ultralong__2",
        LONG,
        LONG,
        "12",
        served={"context_length": 10500000, "chunked_prefill_size": 8192},
    )
    return results, base, extension


def test_a_dry_run_writes_nothing_and_says_what_it_would(tmp_path, battery, capsys):
    results, base, extension = battery
    before = (base / Records.RECORD).read_text()

    planned = join.join_runs([base, extension], tmp_path / "joined" / "accuracy")

    assert planned.refusals == []
    assert planned.status == Status.COMPLETE
    assert not (tmp_path / "joined").exists()
    assert (base / Records.RECORD).read_text() == before
    out = capsys.readouterr().out
    assert "chunked_prefill_size 8192" in out
    assert "mesh tp1-cp8-dcp8-pp1-ep8" in out


def test_the_joined_run_holds_every_part_and_reads_complete(tmp_path, battery):
    results, base, extension = battery
    into = tmp_path / "joined" / "accuracy"

    join.join_runs([base, extension], into, apply=True)

    joined = into / base.name
    record = json.loads((joined / Records.RECORD).read_text())
    assert record["run"]["id"] == base.name
    assert sorted(record["tasks"]) == sorted(SHORT + LONG)
    assert record["skipped"] == {}
    assert sorted(path.name for path in (joined / "samples").iterdir()) == sorted(
        f"{t}.jsonl" for t in SHORT + LONG
    )
    assert (joined / "parts" / extension.name / "server.log").is_file()
    assert (joined / "parts" / base.name / "launches.jsonl").is_file()
    assert record["server"]["mesh"]["label"] == "tp1-cp8-dcp8-pp1-ep8"
    assert {entry["run"] for entry in record["servers"].values()} == {base.name, extension.name}
    assert [part["id"] for part in record["run"]["parts"]] == [base.name, extension.name]
    assert record["run"]["parts"][1]["serving"] == {
        "servers.12": {"context_length": 10500000, "chunked_prefill_size": 8192}
    }
    assert record["run"]["edits"][-1]["what"] == f"joined {extension.name} into {base.name}"
    # The base's owner, the join's own launch last - so the status is the record's
    logged = launches.read(joined)
    assert launches.owner(logged) == "bob"
    assert logged[-1]["joined"] == [base.name, extension.name]
    assert publish.run_status(joined, "")[0] == Status.COMPLETE
    assert publish.plan_publication(joined, tmp_path / "evaluations").status == Status.COMPLETE
    # No part was touched, and nothing was left beside the joined run
    assert sorted(path.name for path in (tmp_path / "joined").iterdir()) == ["accuracy"]
    assert "parts" not in json.loads((base / Records.RECORD).read_text())["run"]


def test_a_join_written_before_is_rebuilt_only_when_asked(tmp_path, battery):
    results, base, extension = battery
    into = tmp_path / "joined" / "accuracy"
    join.join_runs([base, extension], into, apply=True)
    with pytest.raises(ValueError, match="not joined"):
        join.join_runs([base, extension], into, apply=True)
    join.join_runs([base, extension], into, apply=True, replace=True)
    assert [path.name for path in into.iterdir()] == [base.name]


def test_parts_still_going_join_only_as_a_snapshot(tmp_path):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT + LONG, "11")
    running = a_part(results, "ultralong", LONG[:1], LONG, "99", ended=False)
    # A job still going has skipped nothing - its unscored tasks are still to come
    record = json.loads((running / Records.RECORD).read_text())
    (running / Records.RECORD).write_text(json.dumps({**record, "skipped": {}}))
    (running / "samples" / "partial.jsonl").write_text('{"doc_id": 0}\n{"doc')

    refused = join.plan_join([base, running], tmp_path / "joined")
    assert any("in progress" in refusal and "--in-progress" in refusal for refusal in refused.refusals)

    snapshot = join.join_runs([base, running], tmp_path / "joined", apply=True, in_progress=True)
    assert snapshot.status == Status.IN_PROGRESS
    record = json.loads((tmp_path / "joined" / "base" / Records.RECORD).read_text())
    assert "ultralong: job 99: slurm says RUNNING" in record["run"]["snapshot"]["state"]
    assert record["skipped"] == {}
    assert (
        tmp_path / "joined" / "base" / "parts" / "ultralong" / "samples" / "partial.jsonl"
    ).read_text() == ('{"doc_id": 0}\n')
    publication = publish.plan_publication(
        tmp_path / "joined" / "base", tmp_path / "evaluations", in_progress=True
    )
    assert publication.status == Status.IN_PROGRESS


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"scored": SHORT[:1] + LONG}, "holds tasks an earlier part does: ruler_vt_4096"),
        ({"served": {"config": {"top_level": 1}}}, "indexer settings differ"),
        ({"served": {"attention_backend": "flashinfer"}}, "attention_backend 'flashinfer', the base's 'fa3'"),
        ({"mesh": True}, "mesh tp1-cp8-dcp8-pp1-ep8 vs tp8-cp1-dcp1-pp1-ep8"),
        ({"plans": [plan(task, temperature=1.0) for task in LONG]}, "ruler_vt_5000000 planned as"),
        ({"scored": []}, "scored no task"),
        ({"ended": False, "job": "77"}, "is unknown"),
    ],
)
def test_parts_that_are_not_one_measurement_refuse(tmp_path, change, refusal):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT + LONG, "11")
    arguments = {"scored": LONG, "planned": LONG, "job": "12", **change}
    extension = a_part(results, "extension", **arguments)

    planned = join.plan_join([base, extension], tmp_path / "joined")

    assert any(refusal in line for line in planned.refusals), planned.refusals
    with pytest.raises(ValueError, match="not joined"):
        join.join_runs([base, extension], tmp_path / "joined", apply=True)
    assert not (tmp_path / "joined" / "base").exists()


@pytest.mark.parametrize("assemble", ["", "full"])
def test_parts_that_ran_on_other_cards_never_join(tmp_path, assemble):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT + LONG, "11", gpu="NVIDIA H200")
    extension = a_part(results, "extension", LONG, LONG, "12", gpu="NVIDIA B300 SXM6 AC")

    planned = join.plan_join([base, extension], tmp_path / "joined", assemble=assemble)

    assert "run.gpu 'NVIDIA H200' vs 'NVIDIA B300 SXM6 AC'" in planned.refusals[0]


def test_a_pooled_part_joins_without_its_queue_and_its_workers_read_as_one_run(tmp_path, battery):
    results, base, extension = battery
    claims = extension / "pool" / "claims" / LONG[0]
    claims.mkdir(parents=True)
    (claims / "0-0.0").write_text("{}")
    shard = extension / "pool" / "shards" / "13"
    shard.mkdir(parents=True)
    (shard / f"{LONG[0]}.jsonl").write_text('{"doc_id": 0}\n')
    for job in ("13", "14"):
        launches.append(extension, {"job": job, "worker": job, "started": "2026-10-04T02:00:00+00:00"})
    launches.append(extension, {"job": "13", "ended": "2026-10-04T08:00:00+00:00", "outcome": "completed"})
    into = tmp_path / "joined" / "accuracy"

    join.join_runs([base, extension], into, apply=True, in_progress=True)

    combined = into / base.name
    assert not (combined / "pool").exists()
    assert not (combined / "parts" / extension.name / "pool").exists()
    record = json.loads((combined / Records.RECORD).read_text())
    assert sorted(record["tasks"]) == sorted(SHORT + LONG)
    logged = launches.read(combined)
    assert {launch["job"] for launch in logged if launch.get("worker")} == {"13", "14"}


REASONING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 163840, "think": True}


def edit_record(run_dir, **run):
    record = json.loads((run_dir / Records.RECORD).read_text())
    record["run"].update(run)
    (run_dir / Records.RECORD).write_text(json.dumps(record))
    return run_dir


@pytest.fixture
def sets(tmp_path):
    """A core run on the context-parallel mesh, and a benchmark scored later on tensor parallel in its own suite."""
    results = tmp_path / "results"
    base = a_part(results, "native-cp__1", SHORT, SHORT, "11")
    later = a_part(
        results,
        "papertrail-native__2",
        ["papertrail_qec_L1000000"],
        ["papertrail_qec_L1000000"],
        "12",
        served={"attention_backend": "flashmla_sparse", "version": "fork+g4c18"},
        plans=[{**plan("papertrail_qec_L1000000"), "sampling": REASONING}],
        mesh=True,
    )
    edit_record(later, suite="papertrail", generation={"ruler": RULER, "papertrail": REASONING})
    return results, base, later


def test_an_assembly_takes_sets_served_otherwise_and_keeps_how_each_was(tmp_path, sets):
    results, base, later = sets
    into = tmp_path / "joined" / "accuracy"
    assert join.plan_join([base, later], into).refusals

    join.join_runs([base, later], into, apply=True, assemble="full")

    record = json.loads((into / base.name / Records.RECORD).read_text())
    assert record["run"]["suite"] == "full"
    assert sorted(record["tasks"]) == sorted([*SHORT, "papertrail_qec_L1000000"])
    assert record["run"]["generation"] == {"ruler": RULER, "papertrail": REASONING}
    summary = record["run"]["parts"][1]
    assert summary["suite"] == "papertrail"
    assert summary["mesh"]["label"] == "tp8-cp1-dcp1-pp1-ep8"
    assert summary["serving"]["servers.12"] == {
        "attention_backend": "flashmla_sparse",
        "version": "fork+g4c18",
    }
    assert record["run"]["edits"][-1]["what"] == f"assembled as full {later.name} into {base.name}"


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"served": {"config": {"top_level": 1}}}, "indexer settings differ"),
        (
            {"plans": [plan("ruler_vt_4096", temperature=1.0), *[plan(task) for task in LONG]]},
            "ruler_vt_4096 planned as",
        ),
        ({"scored": SHORT}, "holds tasks an earlier part does: ruler_cwe_4096, ruler_vt_4096"),
    ],
)
def test_an_assembly_still_refuses_another_indexer_or_protocol(tmp_path, change, refusal):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT, "11")
    arguments = {"scored": LONG, "planned": LONG, "job": "12", "mesh": True, **change}

    planned = join.plan_join(
        [base, a_part(results, "extension", **arguments)], tmp_path / "joined", assemble="full"
    )

    assert any(refusal in line for line in planned.refusals), planned.refusals


def test_a_superseded_benchmark_is_kept_from_the_last_part_holding_it(tmp_path):
    results = tmp_path / "results"
    short_budget = {**REASONING, "max_tokens": 32768}
    base = a_part(
        results,
        "base",
        [*SHORT, "longbench2"],
        [],
        "11",
        plans=[*[plan(task) for task in SHORT], {**plan("longbench2"), "sampling": short_budget}],
    )
    edit_record(base, generation={"ruler": RULER, "longbench2": short_budget})
    rerun = a_part(
        results, "rerun", ["longbench2"], [], "12", plans=[{**plan("longbench2"), "sampling": REASONING}]
    )
    edit_record(rerun, generation={"ruler": RULER, "longbench2": REASONING})
    into = tmp_path / "joined"
    refused = join.plan_join([base, rerun], into)
    assert any("holds tasks an earlier part does: longbench2" in line for line in refused.refusals)
    assert any("longbench2 generated under other sampling" in line for line in refused.refusals)

    join.join_runs([base, rerun], into, apply=True, supersede=("longbench2",))

    record = json.loads((into / "base" / Records.RECORD).read_text())
    assert sorted(record["tasks"]) == sorted([*SHORT, "longbench2"])
    assert record["run"]["generation"]["longbench2"] == REASONING
    assert [plan["sampling"] for plan in record["run"]["plans"] if plan["task"] == "longbench2"] == [
        REASONING
    ]
    assert record["run"]["parts"][0]["left_out"] == ["longbench2"]
    assert (into / "base" / "parts" / "base" / "samples" / "longbench2.jsonl").is_file()
    assert (into / "base" / "samples" / "longbench2.jsonl").is_file()


def test_a_dropped_benchmark_leaves_every_part_and_the_record_reads_complete(tmp_path, battery):
    results, base, extension = battery
    into = tmp_path / "joined"

    planned = join.join_runs([base, extension], into, apply=True, drop=("ruler_vt_5000000", "ruler_cwe_4096"))

    assert planned.status == Status.COMPLETE
    record = json.loads((into / base.name / Records.RECORD).read_text())
    assert sorted(record["tasks"]) == ["ruler_cwe_5000000", "ruler_vt_4096"]
    assert sorted(plan["task"] for plan in record["run"]["plans"]) == ["ruler_cwe_5000000", "ruler_vt_4096"]
    assert record["run"]["parts"][1]["left_out"] == ["ruler_vt_5000000"]


def test_a_name_no_part_holds_refuses(tmp_path, battery):
    results, base, extension = battery

    planned = join.plan_join([base, extension], tmp_path / "joined", drop=("graphwalks",))

    assert planned.refusals == ["--drop or --supersede names graphwalks, which no part holds"]


def test_a_benchmark_is_named_by_its_tasks(tmp_path, sets):
    results, base, later = sets

    planned = join.plan_join([base, later], tmp_path / "joined", assemble="full", drop=("papertrail",))

    assert planned.refusals == [f"{later.name} scored no task"]


def test_an_assembly_lets_the_indexer_serve_otherwise_but_not_select_otherwise(tmp_path):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT, "11")
    serving = {"top_level": 0, "top_k": 4096, "kernels": "cutedsl", "prefix_reuse": False}
    served = a_part(results, "served", LONG, LONG, "12", served={"config": serving})
    selecting = a_part(
        results, "selecting", LONG, LONG, "13", served={"config": {**serving, "decode": "tree"}}
    )

    assert join.plan_join([base, served], tmp_path / "joined").refusals
    assembled = join.plan_join([base, served], tmp_path / "joined", assemble="full")
    assert assembled.refusals == []
    assert assembled.record["run"]["parts"][1]["serving"]["servers.12"] == {
        "config.kernels": "cutedsl",
        "config.prefix_reuse": False,
    }
    refused = join.plan_join([base, selecting], tmp_path / "joined", assemble="full")
    assert any("indexer settings differ" in line for line in refused.refusals)


def test_a_joined_run_takes_an_id_of_its_own_when_asked(tmp_path, sets):
    results, base, later = sets
    into = tmp_path / "joined"

    join.join_runs([base, later], into, apply=True, assemble="full", run_id="glm53_flash_native_cp__1")

    record = json.loads((into / "glm53_flash_native_cp__1" / Records.RECORD).read_text())
    assert record["run"]["id"] == "glm53_flash_native_cp__1"
    assert record["run"]["parts"][0]["id"] == base.name
    assert record["run"]["edits"][-1]["what"].endswith(f"into {base.name} as glm53_flash_native_cp__1")
    assert not (into / base.name).exists()
    refused = join.plan_join([base, later], into, assemble="full", run_id="a/b")
    assert "--id 'a/b' is not a plain directory name" in refused.refusals


def test_parts_of_another_attention_kind_refuse_even_in_an_assembly(tmp_path):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT, "11")
    extension = a_part(results, "extension", LONG, LONG, "12")
    for run_dir, kind in ((base, "full"), (extension, "block_sparse")):
        record = json.loads((run_dir / Records.RECORD).read_text())
        record["attention"]["kind"] = kind
        (run_dir / Records.RECORD).write_text(json.dumps(record))

    planned = join.plan_join([base, extension], tmp_path / "joined", assemble="full")

    assert "extension: attention kind full vs block_sparse" in planned.refusals


EARLY = {"position": 1048576, "top_k": 2048, "top_p": 512}


def served_with(**settings):
    return {"config": {"top_level": 0, "top_k": 4096, **settings}}


def scored_by_its_server(run_dir, job):
    record = json.loads((run_dir / Records.RECORD).read_text())
    for entry in record["tasks"].values():
        entry["server"] = job
    (run_dir / Records.RECORD).write_text(json.dumps(record))
    return run_dir


@pytest.mark.parametrize("assemble", ["", "full"])
def test_parts_with_other_early_budgets_refuse_unless_asked(tmp_path, assemble):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT, "11")
    early = a_part(results, "early", LONG, LONG, "12", served=served_with(early_budget=EARLY))

    refused = join.plan_join([base, early], tmp_path / "joined", assemble=assemble)
    allowed = join.plan_join([base, early], tmp_path / "joined", assemble=assemble, mixed_early_budget=True)

    assert any("indexer settings differ" in line for line in refused.refusals)
    assert allowed.refusals == []


def test_mixed_early_budgets_relax_no_other_indexer_setting(tmp_path):
    results = tmp_path / "results"
    base = a_part(results, "base", SHORT, SHORT, "11")
    other = a_part(results, "other", LONG, LONG, "12", served=served_with(early_budget=EARLY, top_level=1))
    smaller = a_part(
        results, "smaller", LONG, LONG, "13", served=served_with(early_budget={**EARLY, "top_k": 1024})
    )

    refused = join.plan_join([base, other], tmp_path / "joined", assemble="full", mixed_early_budget=True)
    differing = join.plan_join(
        [base, a_part(results, "early", LONG, LONG, "14", served=served_with(early_budget=EARLY)), smaller],
        tmp_path / "joined",
        assemble="full",
        mixed_early_budget=True,
    )

    assert any("indexer settings differ" in line for line in refused.refusals)
    assert any("early budgets differ between parts" in line for line in differing.refusals)


def test_mixed_early_budgets_keep_each_tasks_indexer_and_name_the_early_budget(tmp_path):
    results = tmp_path / "results"
    base = scored_by_its_server(a_part(results, "base", SHORT, SHORT, "11"), "11")
    early = scored_by_its_server(
        a_part(results, "early", LONG, LONG, "12", served=served_with(early_budget=EARLY)), "12"
    )
    into = tmp_path / "joined"
    (base / "indexer.json").write_text(json.dumps({"early_budget": None}))
    (early / "indexer.json").write_text(json.dumps({"early_budget": EARLY}))

    join.join_runs([base, early], into, apply=True, assemble="full", mixed_early_budget=True)

    record = json.loads((into / base.name / Records.RECORD).read_text())
    assert record["attention"]["config"]["early_budget"] == EARLY
    for task, job, budget in [(SHORT[0], "11", None), (LONG[0], "12", EARLY)]:
        scoring = record["servers"][record["tasks"][task]["server"]]
        assert scoring["config"].get("early_budget") == budget
        assert scoring["run"] == {"11": "base", "12": "early"}[job]
    assert record["run"]["parts"][1]["serving"]["servers.12"] == {"config.early_budget": EARLY}
    assert record["run"]["parts"][0]["serving"] == {}
    assert json.loads((into / base.name / "indexer.json").read_text()) == {"early_budget": EARLY}
