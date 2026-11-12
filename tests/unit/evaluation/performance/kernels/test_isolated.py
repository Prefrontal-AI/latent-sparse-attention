"""The isolated layer run - every method of a model, one after another, in one job on one GPU."""

import argparse

import pytest

from lsa import site, slurm
from lsa.evaluation.constants import Kernels
from lsa.evaluation.performance.kernels import isolated


def parsed(*flags):
    parser = argparse.ArgumentParser()
    slurm.add_flags(parser)
    return parser.parse_args(list(flags))


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_every_method_runs_both_phases_in_chunks_of_32k_over_the_published_ladder(model):
    experiments = isolated.load_methods(model, None)

    assert [e.model.attention for e in experiments][::2] == [e.model.attention for e in experiments][1::2]
    assert [e.data.phase for e in experiments] == ["prefill", "decode"] * len(isolated.METHODS[model])
    assert {e.data.chunk for e in experiments} == {32768}
    assert {tuple(e.data.contexts) for e in experiments} == {
        (131072, 262144, 524288, 1048576, 2097152, 5242880, 10485760)
    }
    assert len({e.name for e in experiments}) == len(experiments)


@pytest.mark.parametrize("phase", isolated.PHASES)
@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_a_chosen_phase_is_the_only_one_run_for_every_method(model, phase):
    experiments = isolated.load_methods(model, None, (phase,))

    assert [e.data.phase for e in experiments] == [phase] * len(isolated.METHODS[model])
    assert {e.model.attention for e in experiments} == {
        e.model.attention for e in isolated.load_methods(model, None)
    }


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_only_the_design_methods_run_cutedsl_kernels(model):
    for experiment in isolated.load_methods(model, None):
        design = experiment.model.attention == "lsa"
        assert (experiment.model.serving.kernels == "cutedsl") is design


def test_glm_runs_the_sparse_kernels_its_records_were_timed_with():
    experiments = isolated.load_methods("glm53_flash", None)

    assert {(e.data.prefill_backend, e.data.decode_backend) for e in experiments} == {
        ("flashmla_sparse", "fa3")
    }


def test_the_output_override_reaches_every_method(tmp_path):
    experiments = isolated.load_methods("qwen38_27b", str(tmp_path))

    assert {e.run.output for e in experiments} == {str(tmp_path)}


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_trailing_overrides_reach_every_experiment_and_beat_the_methods_own(model):
    experiments = isolated.load_methods(
        model, None, overrides=("data.contexts=[10485760]", "data.chunk=8192", "run.hours=3")
    )

    assert {tuple(e.data.contexts) for e in experiments} == {(10485760,)}
    assert {e.data.chunk for e in experiments} == {8192}
    assert {e.hours for e in experiments} == {3}


def test_overrides_after_the_flags_reach_the_printout_and_the_job_hours(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["isolated", "glm53_flash", "--method", "full", "--method", "lsa_flat", "data.contexts=[10485760]",
         "run.hours=2", "--dry-run"],
    )  # fmt: skip
    isolated.main()

    out = capsys.readouterr().out
    assert out.count("contexts [10485760]") == 4
    assert "'timeout_min': 480" in out


def test_a_chosen_method_is_the_only_one_run_and_may_be_a_fair_ablation():
    experiments = isolated.load_methods("glm53_flash", None, methods=("pisa_c8",))

    assert [e.data.phase for e in experiments] == ["prefill", "decode"]
    assert {e.model.pisa.block_size for e in experiments} == {8}
    assert {e.model.pisa.keep_count for e in experiments} == {576}


def test_the_16_key_set_is_a_method_with_half_the_keep_count():
    for model in ("glm53_flash", "qwen38_27b"):
        experiments = isolated.load_methods(model, None, methods=("pisa_c16",))

        assert [e.data.phase for e in experiments] == ["prefill", "decode"]
        assert {(e.model.pisa.block_size, e.model.pisa.keep_count) for e in experiments} == {(16, 288)}


def test_an_unknown_method_names_the_ones_there_are():
    with pytest.raises(ValueError, match="pisa_c8"):
        isolated.load_methods("glm53_flash", None, methods=("nope",))


def test_an_unknown_model_names_the_ones_there_are():
    with pytest.raises(ValueError, match="glm53_flash"):
        isolated.load_methods("nope", None)


def test_the_job_holds_one_gpu_of_a_shared_node_inside_the_reservation():
    experiments = isolated.load_methods("qwen38_27b", None)

    parameters = isolated.job_parameters(experiments, parsed("--reservation", "res"))

    assert parameters["gpus_per_node"] == 1
    # Not the whole node - an exclusive rerun moved H200 layer prefill by at most 1.5%
    assert "slurm_exclusive" not in parameters
    assert parameters["slurm_additional_parameters"]["reservation"] == "res"
    # A one-GPU job is left unbound, as `bound_to_gpus` leaves every one-GPU job
    assert "gres_flags" not in parameters["slurm_additional_parameters"]
    assert parameters["timeout_min"] == sum(e.hours for e in experiments) * 60


def test_a_method_that_fails_does_not_stop_the_ones_after_it(monkeypatch):
    experiments = isolated.load_methods("qwen38_27b", None)
    ran = []

    def run_experiment(experiment):
        ran.append(experiment.name)
        if experiment is experiments[0]:
            raise RuntimeError("boom")

    monkeypatch.setattr("lsa.evaluation.run.run_experiment", run_experiment)
    monkeypatch.delenv("SLURM_PROCID", raising=False)

    with pytest.raises(RuntimeError, match="failed 1 of"):
        isolated.run_methods(experiments)

    assert ran == [e.name for e in experiments]


def test_only_the_first_task_runs_the_methods(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")

    assert isolated.run_methods(isolated.load_methods("qwen38_27b", None)) is None


def test_qwen_times_both_pisa_sets_as_glm_does():
    stock = isolated.load_methods("qwen38_27b", None, methods=("pisa",))
    fair = isolated.load_methods("qwen38_27b", None, methods=("pisa_c8",))

    assert {e.model.attention for e in stock + fair} == {"pisa"}
    assert {(e.model.pisa.block_size, e.model.pisa.keep_count) for e in fair} == {(8, 576)}
    assert "--time-only" in stock[0].run.args
    assert [e.data.phase for e in fair] == ["prefill", "decode"]


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_b300_loads_each_methods_b300_file_in_chunks_of_32k(model):
    experiments = isolated.load_methods(model, None, card="b300")

    assert {e.model.attention for e in experiments} == {
        e.model.attention for e in isolated.load_methods(model, None)
    }
    assert {e.data.chunk for e in experiments} == {32768}
    assert all(e.run.image == site.load()["images"]["b300"] for e in experiments)
    assert all(
        isolated.method_files(model, method, "b300")
        == [str(isolated.configs_dir() / model / f"b300_{method}.yaml")]
        for method in isolated.METHODS[model]
    )


def test_b300_glm_keeps_the_configs_own_backends_and_not_fa3():
    experiments = isolated.load_methods("glm53_flash", None, card="b300")

    assert {e.data.decode_backend for e in experiments} == {"flashmla_sparse"}
    assert "data.decode_backend=fa3" not in isolated.method_overrides("glm53_flash", "pisa", None, "b300")
    assert "data.decode_backend=fa3" in isolated.method_overrides("glm53_flash", "pisa", None, "h200")


def test_b300_early_budget_method_maps_to_its_per_phase_files():
    experiments = isolated.load_methods("glm53_flash", None, methods=("lsa_flat_early_budget",), card="b300")

    assert [e.data.phase for e in experiments] == ["prefill", "decode"]
    assert {e.data.chunk for e in experiments} == {32768}
    assert {e.model.serving.kernels for e in experiments} == {"cutedsl"}
    assert {e.model.indexer.early_budget.position for e in experiments} == {1048576}


def test_the_early_budget_method_is_not_an_h200_method():
    with pytest.raises(ValueError, match="no method"):
        isolated.load_methods("glm53_flash", None, methods=("lsa_flat_early_budget",))


def test_a_method_with_no_b300_file_is_an_error(monkeypatch):
    monkeypatch.setitem(isolated.METHODS, "qwen38_27b", (*isolated.METHODS["qwen38_27b"], "gdn"))

    with pytest.raises(ValueError, match="no b300 config for the method 'gdn'.*b300_gdn.yaml"):
        isolated.load_methods("qwen38_27b", None, card="b300")


def test_an_unknown_card_is_refused():
    with pytest.raises(ValueError, match="no card"):
        isolated.load_methods("qwen38_27b", None, card="a100")


def test_b300_glm_dense_read_runs_both_phases_like_the_other_methods():
    experiments = isolated.load_methods("glm53_flash", None, card="b300")

    assert [e.data.phase for e in experiments if "_full_" in e.name] == ["prefill", "decode"]


def h200_configs(model):
    return {
        path.stem
        for path in (isolated.configs_dir() / model).glob("*.yaml")
        if not path.name.startswith(("_", "b300_"))
    }


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_every_method_a_job_can_run_has_a_config(model):
    for method in (*isolated.METHODS[model], *isolated.EXTRA_METHODS[model]):
        assert (isolated.configs_dir() / model / f"{method}.yaml").is_file()


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_every_config_left_out_of_the_default_methods_is_listed_with_why(model):
    default = set(isolated.METHODS[model])
    extra = set(isolated.EXTRA_METHODS[model])
    untimed = isolated.UNTIMED_METHODS[model]

    # Each config is in exactly one of the three, so a new one cannot be left out without a word
    assert not default & extra and not default & set(untimed) and not extra & set(untimed)
    assert h200_configs(model) == default | extra | set(untimed)
    assert all(why.strip() for why in untimed.values())


@pytest.mark.parametrize("model", sorted(isolated.METHODS))
def test_an_untimed_config_is_refused_as_a_method(model):
    for method in isolated.UNTIMED_METHODS[model]:
        with pytest.raises(ValueError, match=f"no method \\['{method}'\\]"):
            isolated.load_methods(model, None, methods=(method,))


def test_the_methods_prefill_in_the_chunk_the_roofline_models():
    assert isolated.method_overrides("qwen38_27b", "full", None)[0] == f"data.chunk={Kernels.CHUNK}"


def test_a_dry_run_says_whether_the_container_sees_the_records_root(monkeypatch, capsys, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["isolated", "qwen38_27b", "--method", "full", "--output", str(elsewhere), "--dry-run"]
    )
    isolated.main()

    assert "output: refused at submission - qwen38_27b_kernels_full" in capsys.readouterr().out


def test_a_submission_outside_the_mounts_is_refused_before_the_job(monkeypatch, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["isolated", "qwen38_27b", "--method", "full", "--output", str(elsewhere)]
    )
    monkeypatch.setattr(slurm, "executor", lambda *_, **__: pytest.fail("a job was made"))

    with pytest.raises(ValueError, match="outside every directory the container mounts"):
        isolated.main()


def test_the_old_arm_flag_still_names_a_method_and_stays_out_of_the_help(monkeypatch, capsys):
    """`--method` was `--arm` before the rename, and a command written then keeps working"""
    monkeypatch.setattr(
        "sys.argv", ["isolated", "glm53_flash", "--arm", "full", "--method", "lsa_flat", "--dry-run"]
    )
    isolated.main()
    out = capsys.readouterr().out
    assert "method full:" in out
    assert "method lsa_flat:" in out

    monkeypatch.setattr("sys.argv", ["isolated", "--help"])
    with pytest.raises(SystemExit):
        isolated.main()
    assert "--arm" not in capsys.readouterr().out


def test_a_job_pickled_before_the_rename_still_finds_its_function():
    assert isolated.run_arms is isolated.run_methods
