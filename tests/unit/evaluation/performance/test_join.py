"""Joining the performance runs one ladder was split across - what joins, what it writes, and what refuses."""

import json
from pathlib import Path
from typing import Any

import pytest

from lsa import io
from lsa.evaluation import launches, publish
from lsa.evaluation.accuracy.records import join
from lsa.evaluation.constants import Kernels, Production, Records, Status


@pytest.fixture(autouse=True)
def slurm(monkeypatch):
    """Slurm knows no job, so a launch that ended says how it ended."""
    monkeypatch.setattr(launches, "accounting", lambda jobs, fields=("State",): {})


def point(context):
    return {"ttft_s": {"median": context / 1e6}}


def make_run(
    root: Path, run_id: str, contexts: list[int], kind: str = "production", date: str = "2026-10-16", **setup
) -> Path:
    run_dir = root / run_id
    schema = Production.SCHEMA if kind == "production" else Kernels.SCHEMA
    record: dict[str, Any] = {
        "schema": schema,
        "run": {
            "id": run_id,
            "name": f"label_{run_id}",
            "started": f"{run_id}-start",
            "date": date,
            "seconds": 10.0,
        },
        "model": {"name": "glm"},
        "setup": {"warmup": 1, "requests": 3, **setup},
        "points": {str(context): point(context) for context in contexts},
    }
    if kind == "kernels":
        record["run"]["phase"] = "prefill"
        record["failed"] = []
    sample_dir = run_dir / "samples" / "server"
    sample_dir.mkdir(parents=True)
    for context in contexts:
        (sample_dir / f"{context}.jsonl").write_text(f"{context}\n")
    io.write_json(run_dir / Records.RECORD, record)
    (run_dir / Records.META).write_text("{}")
    start = {"job": run_id, "user": "bob", "host": "node-1", "started": "2026-10-16T18:44:36+00:00"}
    lines = [start, launches.end_line(start, "completed")]
    (run_dir / "launches.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return run_dir


@pytest.fixture
def ladder(tmp_path):
    root = tmp_path / "runs"
    return [
        make_run(root, "short", [16384, 32768]),
        make_run(root, "long", [1048576, 2097152], date="2026-10-17"),
    ]


def test_joins_a_ladder_by_context(ladder, tmp_path):
    result = join.join_runs(ladder, tmp_path / "joined", apply=True)

    assert result.refusals == []
    target = tmp_path / "joined" / "short"
    record = io.read_json(target / Records.RECORD)
    assert list(record["points"]) == ["16384", "32768", "1048576", "2097152"]
    assert record["run"]["id"] == "short"
    assert record["run"]["seconds"] == 20.0
    assert record["run"]["date"] == "2026-10-17"
    assert sorted(path.name for path in (target / "samples" / "server").iterdir()) == [
        "1048576.jsonl",
        "16384.jsonl",
        "2097152.jsonl",
        "32768.jsonl",
    ]
    assert (target / "parts" / "long" / "meta.json").is_file()
    logged = launches.read(target)
    assert [launch.get("run") for launch in logged[:2]] == ["short", "long"]
    assert logged[-1]["joined"] == ["short", "long"]


def test_names_its_parts_so_publish_archives_them(ladder, tmp_path):
    result = join.join_runs(ladder, tmp_path / "joined")

    assert publish.parts_of(result.record) == ["short", "long"]
    assert [part["contexts"] for part in result.record["run"]["parts"]] == [
        ["16384", "32768"],
        ["1048576", "2097152"],
    ]
    assert not (tmp_path / "joined").exists()


def test_refuses_parts_served_otherwise(tmp_path, capsys):
    root = tmp_path / "runs"
    base = make_run(root, "short", [16384], mesh={"label": "cp8"})
    other = make_run(root, "long", [1048576], mesh={"label": "tp8"})

    result = join.join_runs([base, other], tmp_path / "joined")

    assert result.refusals == ["long: setup.mesh.label 'cp8' vs 'tp8'"]
    assert "refused  long: setup.mesh.label 'cp8' vs 'tp8'" in capsys.readouterr().out


def test_refuses_a_context_two_parts_hold(tmp_path):
    root = tmp_path / "runs"
    parts = [make_run(root, "a", [16384, 32768]), make_run(root, "b", [32768, 65536])]

    result = join.join_runs(parts, tmp_path / "joined")

    assert result.refusals == ["context 32768 is held by a and b - pass --supersede 32768 to keep the later"]
    with pytest.raises(ValueError, match="1 refusals"):
        join.join_runs(parts, tmp_path / "joined", apply=True)


def test_supersede_keeps_the_last_part_holding_a_context(tmp_path):
    root = tmp_path / "runs"
    parts = [make_run(root, "a", [16384, 32768]), make_run(root, "b", [32768, 65536])]

    result = join.join_runs(parts, tmp_path / "joined", apply=True, supersede=("32768",))

    assert result.refusals == []
    target = tmp_path / "joined" / "a"
    assert result.record["run"]["parts"][0]["left_out"] == ["32768"]
    assert (target / "parts" / "a" / "samples" / "server" / "32768.jsonl").is_file()
    assert (target / "samples" / "server" / "32768.jsonl").read_text() == "32768\n"
    assert result.record["run"]["parts"][1]["contexts"] == ["32768", "65536"]


def test_refuses_mixed_kinds_and_accuracy_flags(tmp_path):
    root = tmp_path / "runs"
    parts = [make_run(root, "a", [16384]), make_run(root, "b", [32768], kind="kernels")]

    assert join.join_runs(parts, tmp_path / "joined").refusals == [
        f"parts are of different kinds - {Kernels.SCHEMA}, {Production.SCHEMA}"
    ]
    assert join.join_runs(parts[:1] + parts[:1], tmp_path / "joined", drop=("x",)).refusals == [
        "--drop is for accuracy records"
    ]


def test_kernel_parts_must_share_a_phase_and_may_differ_in_prompt_repeat(tmp_path):
    root = tmp_path / "runs"
    base = make_run(root, "a", [131072], kind="kernels", prompt={"tokens": 5, "repeated": False})
    same = make_run(root, "b", [5242880], kind="kernels", prompt={"tokens": 5, "repeated": True})
    other = make_run(root, "c", [10485760], kind="kernels", prompt={"tokens": 5, "repeated": True})
    record = io.read_json(other / Records.RECORD)
    record["run"]["phase"] = "decode"
    io.write_json(other / Records.RECORD, record)

    assert join.join_runs([base, same], tmp_path / "joined").refusals == []
    assert join.join_runs([base, other], tmp_path / "joined").refusals == [
        "c: run.phase 'prefill' vs 'decode'"
    ]


def test_a_failed_context_stays_failed_unless_another_part_measured_it(tmp_path):
    root = tmp_path / "runs"
    base = make_run(root, "a", [131072], kind="kernels")
    extension = make_run(root, "b", [262144], kind="kernels")
    for run_dir, failed in ((base, [262144, 524288]), (extension, [])):
        record = io.read_json(run_dir / Records.RECORD)
        record["failed"] = failed
        io.write_json(run_dir / Records.RECORD, record)

    result = join.join_runs([base, extension], tmp_path / "joined")

    assert result.record["failed"] == [524288]
    assert result.status == Status.PARTIAL


def on_host(run_dir: Path, host: str, gpu: str = "NVIDIA H200") -> Path:
    record = io.read_json(run_dir / Records.RECORD)
    record["run"] |= {"host": host, "gpu": gpu}
    io.write_json(run_dir / Records.RECORD, record)
    return run_dir


def test_refuses_parts_measured_on_other_hosts_without_the_flag(tmp_path):
    root = tmp_path / "runs"
    base = on_host(make_run(root, "a", [131072], kind="kernels"), "node-1")
    extension = on_host(make_run(root, "b", [262144], kind="kernels"), "node-17")

    result = join.join_runs([base, extension], tmp_path / "joined")

    assert result.refusals == ["b: run.host 'node-1' vs 'node-17'"]


def test_joins_parts_of_one_card_model_across_hosts_and_keeps_each_host(tmp_path):
    root = tmp_path / "runs"
    base = on_host(make_run(root, "a", [131072], kind="kernels"), "node-1")
    extension = on_host(make_run(root, "b", [262144], kind="kernels"), "node-17")

    result = join.join_runs([base, extension], tmp_path / "joined", across_hosts=True)

    assert result.refusals == []
    assert [part["host"] for part in result.record["run"]["parts"]] == ["node-1", "node-17"]


def test_refuses_parts_of_other_card_models_even_across_hosts(tmp_path):
    root = tmp_path / "runs"
    base = on_host(make_run(root, "a", [131072], kind="kernels"), "node-1")
    extension = on_host(make_run(root, "b", [262144], kind="kernels"), "node-17", "NVIDIA B300")

    result = join.join_runs([base, extension], tmp_path / "joined", across_hosts=True)

    assert result.refusals == ["b: run.gpu 'NVIDIA H200' vs 'NVIDIA B300'"]


def edited(run_dir: Path, **record_keys: Any) -> Path:
    """Merge a dict into each top-level key of a run's record."""
    record = io.read_json(run_dir / Records.RECORD)
    for key, value in record_keys.items():
        record[key] = {**record.get(key, {}), **value}
    io.write_json(run_dir / Records.RECORD, record)
    return run_dir


EARLY = {"position": 1048576, "top_k": 2048, "top_p": 512}


def test_refuses_parts_of_other_early_budgets_without_the_flag(tmp_path):
    root = tmp_path / "runs"
    base = edited(make_run(root, "a", [131072]), indexer={"top_k": 4096, "early_budget": EARLY})
    plain = edited(make_run(root, "b", [2097152]), indexer={"top_k": 4096})

    result = join.join_runs([base, plain], tmp_path / "joined")

    assert result.refusals == [
        "b: indexer.early_budget.position 1048576 vs None",
        "b: indexer.early_budget.top_k 2048 vs None",
        "b: indexer.early_budget.top_p 512 vs None",
    ]


@pytest.mark.parametrize(
    ("kind", "where"), [("production", "indexer"), ("kernels", "setup")], ids=["production", "kernels"]
)
def test_mixed_early_budget_joins_a_plain_part_and_keeps_each_budget(tmp_path, kind, where):
    root = tmp_path / "runs"

    def part(run_id, contexts, **budget):
        run_dir = make_run(root, run_id, contexts, kind=kind)
        if where == "indexer":
            return edited(run_dir, indexer={"top_k": 4096, **budget})
        return edited(run_dir, setup={"config": {"top_k": 4096, **budget}})

    plain = part("plain", [2097152])
    early = part("early", [131072], early_budget=EARLY)

    result = join.join_runs([plain, early], tmp_path / "joined", mixed_early_budget=True)

    assert result.refusals == []
    home = result.record["indexer"] if where == "indexer" else result.record["setup"]["config"]
    assert home["early_budget"] == EARLY
    assert [part["early_budget"] for part in result.record["run"]["parts"]] == [None, EARLY]


def test_mixed_early_budget_relaxes_no_other_indexer_setting(tmp_path):
    root = tmp_path / "runs"
    base = edited(make_run(root, "a", [131072]), indexer={"top_k": 4096, "early_budget": EARLY})
    other = edited(make_run(root, "b", [2097152]), indexer={"top_k": 2048})

    result = join.join_runs([base, other], tmp_path / "joined", mixed_early_budget=True)

    assert result.refusals == ["b: indexer.top_k 4096 vs 2048"]


def test_mixed_early_budget_refuses_two_different_budgets(tmp_path):
    root = tmp_path / "runs"
    base = edited(make_run(root, "a", [131072]), indexer={"early_budget": EARLY})
    other = edited(make_run(root, "b", [2097152]), indexer={"early_budget": {**EARLY, "top_k": 1024}})

    result = join.join_runs([base, other], tmp_path / "joined", mixed_early_budget=True)

    assert len(result.refusals) == 1
    assert result.refusals[0].startswith("early budgets differ between parts")


@pytest.fixture
def split(tmp_path):
    """An early run that holds every context and a plain one that holds the same, as the cp8 records do."""
    root = tmp_path / "runs"
    contexts = [131072, 1048576, 2097152]
    early = edited(make_run(root, "early", contexts), indexer={"early_budget": EARLY})
    plain = make_run(root, "plain", contexts)
    return [early, plain]


def test_contexts_pick_what_each_part_supplies_and_the_record_says_so(split, tmp_path):
    result = join.join_runs(
        split,
        tmp_path / "joined",
        apply=True,
        mixed_early_budget=True,
        contexts=("early=-1048576", "plain=2097152-"),
    )

    assert result.refusals == []
    record = io.read_json(tmp_path / "joined" / "early" / Records.RECORD)
    assert list(record["points"]) == ["131072", "1048576", "2097152"]
    early, plain = record["run"]["parts"]
    assert early["span"] == {"first": None, "last": 1048576}
    assert plain["span"] == {"first": 2097152, "last": None}
    assert plain["left_out"] == ["131072", "1048576"]
    assert plain["contexts"] == ["2097152"]
    assert early["left_out"] == ["2097152"]
    assert "plain supplies 2097152-" in record["run"]["edits"][-1]["what"]
    assert (
        tmp_path / "joined" / "early" / "parts" / "plain" / "samples" / "server" / "131072.jsonl"
    ).is_file()


def test_contexts_refuse_overlapping_ranges(split, tmp_path):
    result = join.join_runs(
        split, tmp_path / "joined", mixed_early_budget=True, contexts=("early=-1048576", "plain=1048576-")
    )

    assert result.refusals == ["--contexts of early and plain overlap"]


def test_contexts_refuse_a_held_context_no_range_covers(split, tmp_path):
    result = join.join_runs(
        split, tmp_path / "joined", mixed_early_budget=True, contexts=("early=-524288", "plain=2097152-")
    )

    assert result.refusals == ["context 1048576 is held by early, plain and no --contexts covers it"]


def test_contexts_refuse_naming_a_part_the_context_is_not_in(tmp_path):
    root = tmp_path / "runs"
    early = make_run(root, "early", [262144])
    plain = make_run(root, "plain", [131072, 262144])

    result = join.join_runs([early, plain], tmp_path / "joined", contexts=("early=-131072", "plain=131073-"))

    assert result.refusals == [
        "context 131072 is held by plain but --contexts gives it to early, which does not hold it"
    ]


def test_contexts_must_be_given_for_every_part_and_well_formed(split, tmp_path):
    some = join.join_runs(split, tmp_path / "joined", mixed_early_budget=True, contexts=("early=-1048576",))
    bad = join.join_runs(
        split, tmp_path / "joined", mixed_early_budget=True, contexts=("early=-x", "plain=2097152-")
    )
    unknown = join.join_runs(split, tmp_path / "joined", mixed_early_budget=True, contexts=("nope=1-2",))
    twice = join.join_runs(
        split, tmp_path / "joined", mixed_early_budget=True, contexts=("early=-1", "early=-2")
    )

    assert some.refusals == ["--contexts is given for every part or none - not for plain"]
    assert bad.refusals[0].startswith("--contexts 'early=-x' is not <run id>=<first>-<last>")
    assert unknown.refusals[0].startswith("--contexts 'nope=1-2' is not <run id>=<first>-<last>")
    assert twice.refusals == ["--contexts names early twice"]


def test_contexts_do_not_combine_with_supersede(split, tmp_path):
    result = join.join_runs(
        split,
        tmp_path / "joined",
        mixed_early_budget=True,
        supersede=("131072",),
        contexts=("early=-1048576", "plain=2097152-"),
    )

    assert result.refusals == ["--contexts and --supersede both choose contexts - pass one"]


def with_harness(run_dir: Path, **hashes: str) -> Path:
    return edited(run_dir, run={"harness": hashes})


def test_refuses_parts_of_other_harness_files_without_the_flag(tmp_path):
    root = tmp_path / "runs"
    base = with_harness(make_run(root, "a", [131072], kind="kernels"), layer="1")
    extension = with_harness(make_run(root, "b", [262144], kind="kernels"), layer="2")

    result = join.join_runs([base, extension], tmp_path / "joined")

    assert result.refusals == ["b: run.harness.layer '1' vs '2'"]


def test_across_harness_keeps_each_parts_hashes_beside_its_contexts(tmp_path):
    root = tmp_path / "runs"
    base = with_harness(make_run(root, "a", [131072], kind="kernels"), layer="1")
    extension = with_harness(make_run(root, "b", [262144], kind="kernels"), layer="2", extra="3")

    result = join.join_runs([base, extension], tmp_path / "joined", across_harness=True)

    assert result.refusals == []
    assert [(part["contexts"], part["harness"]) for part in result.record["run"]["parts"]] == [
        (["131072"], {"layer": "1"}),
        (["262144"], {"layer": "2", "extra": "3"}),
    ]


def test_across_harness_lets_the_commit_differ_and_keeps_each(tmp_path):
    root = tmp_path / "runs"
    base = edited(make_run(root, "a", [131072], kind="kernels"), run={"commits": {"lsa": "4ba5e74"}})
    extension = edited(make_run(root, "b", [262144], kind="kernels"), run={"commits": {"lsa": "20aff8a"}})

    refused = join.join_runs([base, extension], tmp_path / "joined")
    result = join.join_runs([base, extension], tmp_path / "joined", across_harness=True)

    assert refused.refusals == ["b: run.commits.lsa '4ba5e74' vs '20aff8a'"]
    assert result.refusals == []
    assert [part["commits"] for part in result.record["run"]["parts"]] == [
        {"lsa": "4ba5e74"},
        {"lsa": "20aff8a"},
    ]


def test_across_harness_relaxes_no_other_field(tmp_path):
    root = tmp_path / "runs"
    base = with_harness(make_run(root, "a", [131072], mesh={"label": "cp8"}), layer="1")
    extension = with_harness(make_run(root, "b", [262144], mesh={"label": "tp8"}), layer="2")

    result = join.join_runs([base, extension], tmp_path / "joined", across_harness=True)

    assert result.refusals == ["b: setup.mesh.label 'cp8' vs 'tp8'"]


def test_parts_may_differ_in_note_and_name_renamed_from_and_keep_both(tmp_path):
    root = tmp_path / "runs"
    base = edited(make_run(root, "a", [131072]), run={"note": "first", "renamed_from": "old_a"})
    extension = edited(make_run(root, "b", [262144]), run={"note": "second"})

    result = join.join_runs([base, extension], tmp_path / "joined")

    assert result.refusals == []
    assert [(part["note"], part["renamed_from"]) for part in result.record["run"]["parts"]] == [
        ("first", "old_a"),
        ("second", None),
    ]
