import sys
from types import SimpleNamespace

import pytest

from lsa import cli


@pytest.fixture
def called(monkeypatch):
    """Replace the module a subcommand imports with one that records its `main` call and the command line it saw."""
    calls = []

    def fake_import(name):
        return SimpleNamespace(main=lambda *arguments: calls.append((name, arguments, list(sys.argv))))

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    return calls


def test_native_train_routes_to_the_run_module_with_its_command(called):
    cli.main(["native", "train", "configs/a.yaml", "run.name=x"])
    assert called == [
        ("lsa.training.native.run", ("train",), ["lsa native train", "configs/a.yaml", "run.name=x"])
    ]


def test_native_evaluate_routes_to_the_same_module_with_its_own_command(called):
    cli.main(["native", "evaluate", "configs/a.yaml", "--checkpoint", "/x/step-1", "attention.coarse=false"])
    assert called == [
        (
            "lsa.training.native.run",
            ("evaluate",),
            [
                "lsa native evaluate",
                "configs/a.yaml",
                "--checkpoint",
                "/x/step-1",
                "attention.coarse=false",
            ],
        )
    ]


def test_native_alone_is_a_group_and_asks_for_a_subcommand(called):
    with pytest.raises(SystemExit):
        cli.main(["native"])
    assert called == []


def test_the_evaluate_group_is_not_the_native_evaluate_command(called):
    cli.main(["evaluate", "accuracy", "configs/a.yaml"])
    assert called[0][1] == ("accuracy",)


def test_native_evaluate_hands_the_suite_flag_to_the_run_module_untouched(called):
    cli.main(["native", "evaluate", "configs/a.yaml", "--suite", "benchmark", "--checkpoint", "/x/step-1"])
    assert called == [
        (
            "lsa.training.native.run",
            ("evaluate",),
            ["lsa native evaluate", "configs/a.yaml", "--suite", "benchmark", "--checkpoint", "/x/step-1"],
        )
    ]


def test_native_report_routes_to_the_benchmark_report(called):
    cli.main(["native", "report", "/x/records", "--baseline", "b16k-dense"])
    assert called == [
        (
            "lsa.training.native.benchmark.report",
            (),
            ["lsa native report", "/x/records", "--baseline", "b16k-dense"],
        )
    ]


@pytest.mark.parametrize(
    "words,module,arguments",
    [
        (["distill", "train"], "lsa.training.distillation.run", ()),
        (["distill", "rescore"], "lsa.training.distillation.rescore_maps", ()),
        (["distill", "served-read"], "lsa.training.distillation.served_read", ()),
        (["distill", "export"], "lsa.training.distillation.export", ()),
        (["distill", "inject"], "lsa.training.distillation.inject", ()),
        (["distill", "targets"], "lsa.training.distillation.data.teacher", ()),
        (["distill", "native-picks"], "lsa.training.distillation.data.native", ()),
        (["kernels", "cut"], "lsa.evaluation.performance.kernels.data.cut", ()),
        (["evaluate", "isolated"], "lsa.evaluation.performance.kernels.isolated", ()),
        (["records", "rebuild"], "lsa.evaluation.performance.kernels.rebuild", ()),
    ],
)
def test_a_regrouped_command_routes_to_the_module_that_owns_it(called, words, module, arguments):
    cli.main([*words, "x"])
    assert called == [(module, arguments, ["lsa " + " ".join(words), "x"])]


@pytest.mark.parametrize(
    "words",
    [
        ["rescore-maps"],
        ["served-read"],
        ["export"],
        ["inject"],
        ["cut"],
        ["kernels", "isolated"],
        ["kernels", "rebuild"],
        ["data", "ruler"],
        ["data", "prepare-distillation-targets"],
        ["data", "native-picks"],
        ["data", "compare-online-targets"],
        ["distill", "check-online-teacher"],
        ["native", "benchmark"],
        ["kernels", "show"],
        ["production", "record"],
        ["records", "import-native-decay"],
        ["records", "samples"],
        ["data", "suites"],
        ["data", "teacher-loss"],
    ],
)
def test_an_old_spelling_is_refused(called, words):
    with pytest.raises(SystemExit):
        cli.main([*words, "x"])
    assert called == []


def test_distill_alone_is_a_group_and_asks_for_a_subcommand(called):
    with pytest.raises(SystemExit):
        cli.main(["distill"])
    assert called == []
