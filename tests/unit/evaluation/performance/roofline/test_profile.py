"""The profile's passes - timings are taken with no profiler attached, the profiled calls alone under it."""

from pathlib import Path
from typing import Any

import pytest

from lsa.evaluation.performance.roofline import profile, run, workloads


def run_context(monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    calls = {"launch": 0, "time": 0, "graph": 0, "profile": 0}

    def launch() -> None:
        calls["launch"] += 1

    def count(name: str, result: Any) -> Any:
        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return result

        return counted

    monkeypatch.setattr(profile.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profile.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(workloads, "launcher", lambda step, context, arm: launch)
    monkeypatch.setattr(profile, "time_launch", count("time", (1.0, 1.0)))
    monkeypatch.setattr(profile, "graph_timings", count("graph", {"1": 2.0}))
    monkeypatch.setattr(profile, "profile_launch", count("profile", None))
    step = workloads.Step("fake", "decode", True, lambda context, kernels: launch)
    rows = profile.profile_context([step], 1024, ["cutedsl"], 1, True, [1], mode)
    return rows, calls


def test_time_mode_times_and_never_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, calls = run_context(monkeypatch, "time")
    assert len(rows) == 1
    assert rows[0]["graph_median_us"] == {"1": 2.0}
    assert (calls["time"], calls["graph"], calls["profile"]) == (1, 1, 0)


def test_profile_mode_profiles_and_never_times(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, calls = run_context(monkeypatch, "profile")
    assert rows == []
    assert (calls["time"], calls["graph"], calls["profile"]) == (0, 0, 1)


def test_both_mode_does_each(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, calls = run_context(monkeypatch, "both")
    assert len(rows) == 1
    assert (calls["time"], calls["graph"], calls["profile"]) == (1, 1, 1)


def test_the_job_times_outside_ncu_and_profiles_inside_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    class Done:
        returncode = 0

    def fake_run(argv: list[str]) -> Done:
        commands.append(argv)
        if "ncu" in argv[0]:
            (tmp_path / "roofline.ncu-rep").write_text("")
        return Done()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(run, "NCU", "/nonexistent/ncu")
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/ncu")
    monkeypatch.setattr(run.report, "write_roofline", lambda *args: tmp_path / "roofline.json")
    monkeypatch.setattr(run.page, "render", lambda *args: tmp_path / "roofline.html")
    run.profile_kernels(tmp_path, [1024], ["all"], ["cutedsl"], 1, [1])
    timed, profiled = commands
    assert "ncu" not in timed[0]
    assert timed[timed.index("--mode") + 1] == "time"
    assert "--graph" in timed
    assert "ncu" in profiled[0]
    assert profiled[profiled.index("--mode") + 1] == "profile"


@pytest.mark.gpu
def test_decode_select_replays_from_a_graph_well_under_its_eager_call() -> None:
    step = workloads.step_named("decode select")
    launch = workloads.launcher(step, 262144, "cutedsl")
    assert launch is not None
    launch()
    eager_ms, _ = profile.time_launch(launch, 10)
    graph_us = profile.time_graph(launch, 5)
    assert graph_us < 1000 * eager_ms / 2


def test_graph_timings_skips_a_step_that_cannot_be_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(launch: Any, repeats: int) -> float:
        raise RuntimeError("Cannot copy between CPU and CUDA tensors during CUDA graph capture")

    monkeypatch.setattr(profile.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profile, "time_graph", refuse)
    step = workloads.Step("fake", "decode", True, lambda context, kernels: lambda: None)
    assert profile.graph_timings(step, 1024, "triton", 1, [1]) == {}
