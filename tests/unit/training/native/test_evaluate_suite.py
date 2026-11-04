import os
import subprocess
import sys
from pathlib import Path

import pytest

from lsa.training.native import run
from lsa.training.native.benchmark import run as benchmark
from lsa.training.native.benchmark import tasks

ROOT = Path(__file__).resolve().parents[4]


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


def test_the_default_context_is_the_training_length_when_no_ruler_task_is_selected():
    assert benchmark.context_length(4096, tasks.expand("hellaswag,arc_easy")) == 4096
    assert benchmark.context_length(4096, []) == 4096


def test_the_default_context_grows_to_the_longest_ruler_cell():
    assert benchmark.context_length(4096, tasks.expand("ruler")) == 16384
    assert benchmark.context_length(65536, tasks.expand("ruler")) == 65536


@pytest.mark.parametrize("suite", ["held-out", "benchmark"])
def test_a_dry_run_says_what_it_would_run_and_loads_neither_torch_nor_lm_eval(suite, tmp_path):
    config = ROOT / "configs/training/native/qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml"
    script = (
        "import sys; from lsa.training.native import run; "
        f"sys.argv = ['lsa', {str(config)!r}, '--suite', {suite!r}, '--dry-run']; "
        "run.main('evaluate'); "
        "print('LOADED', sorted({'torch', 'lm_eval', 'transformers'} & set(sys.modules)))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LSA_ROOT": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    *printed, loaded = done.stdout.splitlines()
    assert loaded == "LOADED []"
    assert any("decay128k-lsa-2048" in line for line in printed)
