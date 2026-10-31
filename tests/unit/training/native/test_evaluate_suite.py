import sys

import pytest

from lsa.training.native import run
from lsa.training.native.benchmark import run as benchmark


def test_the_benchmark_suite_hands_the_command_line_to_the_benchmark(monkeypatch):
    seen = []
    monkeypatch.setattr(benchmark, "command", seen.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lsa native evaluate", "configs/a.yaml", "--suite", "benchmark", "--checkpoint", "/x/step-1"]
        + ["--tasks", "ruler", "--label", "arm"],
    )

    run.main("evaluate")

    [args] = seen
    assert (args.config, args.checkpoint, args.tasks, args.label) == (
        "configs/a.yaml",
        "/x/step-1",
        "ruler",
        "arm",
    )


def test_the_held_out_suite_is_the_default_and_takes_no_benchmark_option(monkeypatch):
    monkeypatch.setattr(benchmark, "command", lambda args: pytest.fail("the benchmark ran"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["lsa native evaluate", "configs/a.yaml", "--checkpoint", "/x/step-1", "--tasks", "core"],
    )

    with pytest.raises(SystemExit):
        run.main("evaluate")


def test_an_unknown_suite_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lsa native evaluate", "configs/a.yaml", "--suite", "mmlu"])

    with pytest.raises(SystemExit):
        run.main("evaluate")
