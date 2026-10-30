"""That renaming a record of any kind writes a copy under its new id, with the old id kept and archived by a
replacing publish, and refuses what is not a lowercase name or not a record."""

import json

import pytest

from lsa.evaluation import launches, publish, rename
from lsa.evaluation.constants import Kernels, Production, Records

OLD = "e2e_p1_qwen_dcp_flat__2026-10-16__19-40-01"
NEW = "qwen38_27b_dcp8_lsa_flat"
LOG = (
    {"job": None, "user": "bob", "host": "node-1", "started": "2026-10-16T10:00:00+00:00"},
    {"job": None, "ended": "2026-10-16T11:00:00+00:00", "outcome": "completed"},
)


def a_run(root, run_id=OLD, schema=Production.SCHEMA):
    run = root / run_id
    (run / "samples").mkdir(parents=True)
    (run / "samples" / "16384.jsonl").write_text("{}\n")
    (run / "meta.json").write_text("{}")
    for line in LOG:
        launches.append(run, line)
    record = {
        "schema": schema,
        "run": {"id": run_id, "name": run_id.split("__")[0], "note": "", "parts": [{"id": run_id}]},
        "points": {},
    }
    (run / "record.json").write_text(json.dumps(record))
    return run


def read(path):
    return json.loads((path / "record.json").read_text())


def test_a_rename_writes_a_copy_under_the_new_id_and_leaves_the_source(tmp_path):
    run = a_run(tmp_path / "results")
    before = read(run)

    plan = rename.rename_run(run, NEW, tmp_path / "renamed", "Joined from two launches.", apply=True)

    record = read(plan.target)
    assert plan.target == tmp_path / "renamed" / f"{NEW}__2026-10-16__19-40-01"
    assert (record["run"]["id"], record["run"]["name"]) == (plan.target.name, NEW)
    assert (record["run"]["renamed_from"], record["run"]["note"]) == (OLD, "Joined from two launches.")
    assert record["run"]["parts"] == before["run"]["parts"]
    assert (plan.target / "samples" / "16384.jsonl").is_file()
    assert (plan.target / "launches.jsonl").is_file()
    assert read(run) == before


def test_no_note_keeps_the_one_the_record_has(tmp_path):
    run = a_run(tmp_path / "results")
    plan = rename.plan_rename(run, NEW, tmp_path / "renamed")
    assert plan.record["run"]["note"] == ""


def test_a_dry_run_writes_nothing(tmp_path):
    run = a_run(tmp_path / "results")
    rename.rename_run(run, NEW, tmp_path / "renamed")
    assert not (tmp_path / "renamed").exists()


@pytest.mark.parametrize("name", ["Qwen", "a-b", "a__b", "", "_a"])
def test_a_name_that_is_not_lowercase_words_is_refused(tmp_path, name):
    run = a_run(tmp_path / "results")
    with pytest.raises(ValueError, match="lowercase words"):
        rename.plan_rename(run, name, tmp_path / "renamed")


def test_a_record_of_no_kind_is_refused(tmp_path):
    run = a_run(tmp_path / "results", schema="lsa-other/1")
    with pytest.raises(ValueError, match="not one this publishes"):
        rename.plan_rename(run, NEW, tmp_path / "renamed")


def test_an_accuracy_record_is_renamed_like_a_production_one(tmp_path):
    run = a_run(tmp_path / "results", "glm53_flash_lsa_flat__2026-10-03__20-57-43", Records.SCHEMA)

    plan = rename.rename_run(run, "glm53_flash_lsa_flat_h200", tmp_path / "renamed", apply=True)

    record = read(plan.target)
    assert record["run"]["id"] == "glm53_flash_lsa_flat_h200__2026-10-03__20-57-43"
    assert record["run"]["name"] == "glm53_flash_lsa_flat_h200"
    assert record["run"]["renamed_from"] == "glm53_flash_lsa_flat__2026-10-03__20-57-43"


def a_kernel_run(root, label):
    run = a_run(root, "glm53_flash_kernels_native_phasedecode__2026-10-16__17-44-39", Kernels.SCHEMA)
    record = read(run)
    record["run"]["name"] = label
    (run / "record.json").write_text(json.dumps(record))
    return run


def test_a_kernel_record_named_by_its_config_follows_the_rename(tmp_path):
    run = a_kernel_run(tmp_path / "results", "glm53_flash_kernels_native_phasedecode")

    plan = rename.plan_rename(run, "glm53_flash_kernels_native_phasedecode_h200", tmp_path / "renamed")

    assert plan.record["run"]["name"] == "glm53_flash_kernels_native_phasedecode_h200"


def test_a_kernel_record_named_for_the_path_it_timed_takes_the_new_name(tmp_path):
    run = a_kernel_run(tmp_path / "results", "native")

    plan = rename.plan_rename(run, "glm53_flash_kernels_native_phasedecode_h200", tmp_path / "renamed")

    assert plan.record["run"]["name"] == "glm53_flash_kernels_native_phasedecode_h200"
    assert plan.record["run"]["id"] == "glm53_flash_kernels_native_phasedecode_h200__2026-10-16__17-44-39"


def test_a_kernel_record_named_for_its_path_is_corrected_under_its_own_id(tmp_path):
    run = a_kernel_run(tmp_path / "results", "native")

    plan = rename.plan_rename(run, "glm53_flash_kernels_native_phasedecode", tmp_path / "renamed")

    assert not plan.refusals
    assert plan.record["run"]["id"] == "glm53_flash_kernels_native_phasedecode__2026-10-16__17-44-39"
    assert plan.record["run"]["name"] == "glm53_flash_kernels_native_phasedecode"


def test_a_rename_to_the_name_a_record_has_is_refused(tmp_path):
    run = a_kernel_run(tmp_path / "results", "glm53_flash_kernels_native_phasedecode")

    plan = rename.plan_rename(run, "glm53_flash_kernels_native_phasedecode", tmp_path / "renamed")

    assert plan.refusals


def test_a_rename_over_an_existing_directory_is_refused(tmp_path):
    run = a_run(tmp_path / "results")
    rename.rename_run(run, NEW, tmp_path / "renamed", apply=True)
    with pytest.raises(ValueError, match="not renamed"):
        rename.rename_run(run, NEW, tmp_path / "renamed", apply=True)


def test_publishing_a_renamed_run_with_replace_archives_the_old_one(tmp_path):
    root = tmp_path / "evaluations"
    old = a_run(tmp_path / "results")
    publish.publish([old], root, apply=True)
    plan = rename.rename_run(old, NEW, tmp_path / "renamed", apply=True)

    with pytest.raises(FileExistsError, match="--replace"):
        publish.publish([plan.target], root, apply=True)
    [published] = publish.publish([plan.target], root, apply=True, replace=True)

    directory = root / "performance" / "production"
    assert sorted(path.name for path in directory.iterdir()) == [published.name]
    archive = tmp_path / "archive" / "evaluations" / "performance" / "production"
    assert [path.name for path in archive.iterdir()] == [OLD]
