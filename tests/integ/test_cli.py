"""The public command routes arguments intact, prints help and handles dry runs without work."""

import os
import subprocess
import sys

import pytest

from lsa import cli, site


def test_an_unknown_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as finished:
        cli.main(["nope"])
    assert finished.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_a_group_without_a_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as finished:
        cli.main(["records"])
    assert finished.value.code == 2


def test_the_rest_of_the_line_reaches_the_module_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr("lsa.env.main", lambda: seen.append(list(sys.argv)))
    cli.main(["env", "first", "--flag", "value", "--help"])
    assert seen == [["lsa env", "first", "--flag", "value", "--help"]]


def test_a_site_before_the_subcommand_is_selected_and_not_passed_on(monkeypatch):
    seen = []
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.setattr("lsa.env.main", lambda: seen.append((site.current(), list(sys.argv))))
    cli.main(["--site", "local", "env", "first"])
    assert seen == [("local", ["lsa env", "first"])]


def test_a_value_the_site_leaves_empty_prints_one_line_and_exits_non_zero(monkeypatch, capsys):
    def refuse():
        raise site.NotNamed("run.image is not named - fill it in configs/site/<site>.yaml")

    monkeypatch.setattr("lsa.env.main", refuse)
    with pytest.raises(SystemExit) as finished:
        cli.main(["env"])
    assert finished.value.code == "lsa: run.image is not named - fill it in configs/site/<site>.yaml"
    assert capsys.readouterr().err == ""


def test_any_other_error_still_surfaces_with_its_traceback(monkeypatch):
    def refuse():
        raise ValueError("a bug")

    monkeypatch.setattr("lsa.env.main", refuse)
    with pytest.raises(ValueError, match="a bug"):
        cli.main(["env"])


def test_a_nested_subcommand_is_named_in_the_program_its_parser_prints(monkeypatch):
    seen = []
    monkeypatch.setattr("lsa.evaluation.accuracy.records.show.main", lambda: seen.append(list(sys.argv)))
    cli.main(["records", "show", "a/record.json", "b/record.json"])
    assert seen == [["lsa records show", "a/record.json", "b/record.json"]]


def test_an_evaluation_kind_is_handed_to_the_one_main_that_reads_all_three(monkeypatch):
    seen = []
    monkeypatch.setattr("lsa.evaluation.run.main", lambda kind: seen.append((kind, list(sys.argv))))
    for kind in ("accuracy", "kernels", "production"):
        cli.main(["evaluate", kind, "config.yaml", "--dry-run"])
    assert seen == [
        (kind, [f"lsa evaluate {kind}", "config.yaml", "--dry-run"])
        for kind in ("accuracy", "kernels", "production")
    ]


def test_python_dash_m_lsa_is_the_same_command():
    finished = subprocess.run([sys.executable, "-m", "lsa", "--help"], capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    assert "evaluate" in finished.stdout


def test_a_dry_run_prepares_nothing_and_submits_nothing(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run built an executor")

    monkeypatch.setattr("lsa.slurm.executor", refuse)
    cli.main(["data", "prepare", "production", "--dry-run"])
    assert (
        "would submit lsa.evaluation.performance.production.data.prompts.build_files"
        in capsys.readouterr().out
    )


def test_a_target_extraction_dry_run_needs_no_selection_files_or_data(monkeypatch, tmp_path, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run planned documents or submitted extraction")

    monkeypatch.setattr("lsa.slurm.executor", refuse)
    monkeypatch.setattr("lsa.training.distillation.data.teacher.plan", refuse)
    output = tmp_path / "targets"
    path = tmp_path / "extraction.yaml"
    path.write_text(
        f"""model:
  name: glm53_flash
data:
  selection:
    datasets: [glm53_flash]
    lengths: [16384]
    tokens_per_bucket: 32768
    domains: {{academic: [arxiv]}}
run:
  name: unit
  collection: selected
  output: {output}
"""
    )

    cli.main(["distill", "targets", str(path), "--dry-run"])

    shown = capsys.readouterr().out
    assert "unit" in shown
    assert str(output) in shown
    assert not output.exists()


def test_a_relative_path_is_read_from_where_the_command_was_typed(tmp_path, monkeypatch):
    run = tmp_path / "results" / "accuracy" / "run"
    run.mkdir(parents=True)
    (run / "record.json").write_text('{"schema": "not-a-record"}')
    monkeypatch.chdir(tmp_path)
    # The record is found, and refused for what it is rather than for being absent
    with pytest.raises(ValueError, match="not one this publishes"):
        cli.main(["records", "show", "results/accuracy/run/record.json"])
    with pytest.raises(ValueError, match="not one this publishes"):
        cli.main(["records", "show", "results/accuracy/run"])
    assert os.getcwd() == str(tmp_path)
