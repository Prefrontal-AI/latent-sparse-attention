"""The isolated layer run - every arm of a model, one after another, in one job on one GPU."""

import argparse

import pytest

from lsa import site, slurm
from lsa.evaluation.constants import Kernels
from lsa.evaluation.performance.kernels import isolated


def parsed(*flags):
    parser = argparse.ArgumentParser()
    slurm.add_flags(parser)
    return parser.parse_args(list(flags))


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_every_arm_runs_both_phases_in_chunks_of_32k_over_the_published_ladder(model):
    experiments = isolated.load_arms(model, None)

    assert [e.model.attention for e in experiments][::2] == [e.model.attention for e in experiments][1::2]
    assert [e.data.phase for e in experiments] == ["prefill", "decode"] * len(isolated.ARMS[model])
    assert {e.data.chunk for e in experiments} == {32768}
    assert {tuple(e.data.contexts) for e in experiments} == {
        (131072, 262144, 524288, 1048576, 2097152, 5242880, 10485760)
    }
    assert len({e.name for e in experiments}) == len(experiments)


@pytest.mark.parametrize("phase", isolated.PHASES)
@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_a_chosen_phase_is_the_only_one_run_for_every_arm(model, phase):
    experiments = isolated.load_arms(model, None, (phase,))

    assert [e.data.phase for e in experiments] == [phase] * len(isolated.ARMS[model])
    assert {e.model.attention for e in experiments} == {
        e.model.attention for e in isolated.load_arms(model, None)
    }


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_only_the_design_arms_run_cutedsl_kernels(model):
    for experiment in isolated.load_arms(model, None):
        design = experiment.model.attention == "lsa"
        assert (experiment.model.serving.kernels == "cutedsl") is design


def test_glm_runs_the_sparse_kernels_its_records_were_timed_with():
    experiments = isolated.load_arms("glm53_flash", None)

    assert {(e.data.prefill_backend, e.data.decode_backend) for e in experiments} == {
        ("flashmla_sparse", "fa3")
    }


def test_the_output_override_reaches_every_arm(tmp_path):
    experiments = isolated.load_arms("qwen38_27b", str(tmp_path))

    assert {e.run.output for e in experiments} == {str(tmp_path)}


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_trailing_overrides_reach_every_experiment_and_beat_the_arms_own(model):
    experiments = isolated.load_arms(
        model, None, overrides=("data.contexts=[10485760]", "data.chunk=8192", "run.hours=3")
    )

    assert {tuple(e.data.contexts) for e in experiments} == {(10485760,)}
    assert {e.data.chunk for e in experiments} == {8192}
    assert {e.hours for e in experiments} == {3}


def test_overrides_after_the_flags_reach_the_printout_and_the_job_hours(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["isolated", "glm53_flash", "--arm", "full", "--arm", "lsa_flat", "data.contexts=[10485760]",
         "run.hours=2", "--dry-run"],
    )  # fmt: skip
    isolated.main()

    out = capsys.readouterr().out
    assert out.count("contexts [10485760]") == 4
    assert "'timeout_min': 480" in out


def test_a_chosen_arm_is_the_only_one_run_and_may_be_a_fair_ablation():
    experiments = isolated.load_arms("glm53_flash", None, arms=("pisa_c8",))

    assert [e.data.phase for e in experiments] == ["prefill", "decode"]
    assert {e.model.pisa.block_size for e in experiments} == {8}
    assert {e.model.pisa.keep_count for e in experiments} == {576}


def test_the_16_key_set_is_an_arm_with_half_the_keep_count():
    for model in ("glm53_flash", "qwen38_27b"):
        experiments = isolated.load_arms(model, None, arms=("pisa_c16",))

        assert [e.data.phase for e in experiments] == ["prefill", "decode"]
        assert {(e.model.pisa.block_size, e.model.pisa.keep_count) for e in experiments} == {(16, 288)}


def test_an_unknown_arm_names_the_ones_there_are():
    with pytest.raises(ValueError, match="pisa_c8"):
        isolated.load_arms("glm53_flash", None, arms=("nope",))


def test_an_unknown_model_names_the_ones_there_are():
    with pytest.raises(ValueError, match="glm53_flash"):
        isolated.load_arms("nope", None)


def test_the_job_holds_one_gpu_of_a_shared_node_inside_the_reservation():
    experiments = isolated.load_arms("qwen38_27b", None)

    parameters = isolated.job_parameters(experiments, parsed("--reservation", "res"))

    assert parameters["gpus_per_node"] == 1
    # Not the whole node - an exclusive rerun moved H200 layer prefill by at most 1.5%
    assert "slurm_exclusive" not in parameters
    assert parameters["slurm_additional_parameters"]["reservation"] == "res"
    # A one-GPU job is left unbound, as `bound_to_gpus` leaves every one-GPU job
    assert "gres_flags" not in parameters["slurm_additional_parameters"]
    assert parameters["timeout_min"] == sum(e.hours for e in experiments) * 60


def test_an_arm_that_fails_does_not_stop_the_ones_after_it(monkeypatch):
    experiments = isolated.load_arms("qwen38_27b", None)
    ran = []

    def run_experiment(experiment):
        ran.append(experiment.name)
        if experiment is experiments[0]:
            raise RuntimeError("boom")

    monkeypatch.setattr("lsa.evaluation.run.run_experiment", run_experiment)
    monkeypatch.delenv("SLURM_PROCID", raising=False)

    with pytest.raises(RuntimeError, match="failed 1 of"):
        isolated.run_arms(experiments)

    assert ran == [e.name for e in experiments]


def test_only_the_first_task_runs_the_arms(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")

    assert isolated.run_arms(isolated.load_arms("qwen38_27b", None)) is None


def test_qwen_times_both_pisa_sets_as_glm_does():
    stock = isolated.load_arms("qwen38_27b", None, arms=("pisa",))
    fair = isolated.load_arms("qwen38_27b", None, arms=("pisa_c8",))

    assert {e.model.attention for e in stock + fair} == {"pisa"}
    assert {(e.model.pisa.block_size, e.model.pisa.keep_count) for e in fair} == {(8, 576)}
    assert "--time-only" in stock[0].run.args
    assert [e.data.phase for e in fair] == ["prefill", "decode"]


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_b300_loads_each_arms_b300_file_in_chunks_of_32k(model):
    experiments = isolated.load_arms(model, None, card="b300")

    assert {e.model.attention for e in experiments} == {
        e.model.attention for e in isolated.load_arms(model, None)
    }
    assert {e.data.chunk for e in experiments} == {32768}
    assert all(e.run.image == site.load()["images"]["b300"] for e in experiments)
    assert all(
        isolated.arm_files(model, arm, "b300") == [str(isolated.CONFIGS / model / f"b300_{arm}.yaml")]
        for arm in isolated.ARMS[model]
    )


def test_b300_glm_keeps_the_configs_own_backends_and_not_fa3():
    experiments = isolated.load_arms("glm53_flash", None, card="b300")

    assert {e.data.decode_backend for e in experiments} == {"flashmla_sparse"}
    assert "data.decode_backend=fa3" not in isolated.arm_overrides("glm53_flash", "pisa", None, "b300")
    assert "data.decode_backend=fa3" in isolated.arm_overrides("glm53_flash", "pisa", None, "h200")


def test_b300_early_budget_arm_maps_to_its_per_phase_files():
    experiments = isolated.load_arms("glm53_flash", None, arms=("lsa_flat_early_budget",), card="b300")

    assert [e.data.phase for e in experiments] == ["prefill", "decode"]
    assert {e.data.chunk for e in experiments} == {32768}
    assert {e.model.serving.kernels for e in experiments} == {"cutedsl"}
    assert {e.model.indexer.early_budget.position for e in experiments} == {1048576}


def test_the_early_budget_arm_is_not_an_h200_arm():
    with pytest.raises(ValueError, match="no arm"):
        isolated.load_arms("glm53_flash", None, arms=("lsa_flat_early_budget",))


def test_an_arm_with_no_b300_file_is_an_error(monkeypatch):
    monkeypatch.setitem(isolated.ARMS, "qwen38_27b", (*isolated.ARMS["qwen38_27b"], "gdn"))

    with pytest.raises(ValueError, match="no b300 config for the arm 'gdn'.*b300_gdn.yaml"):
        isolated.load_arms("qwen38_27b", None, card="b300")


def test_an_unknown_card_is_refused():
    with pytest.raises(ValueError, match="no card"):
        isolated.load_arms("qwen38_27b", None, card="a100")


def test_b300_glm_dense_read_runs_both_phases_like_the_other_arms():
    experiments = isolated.load_arms("glm53_flash", None, card="b300")

    assert [e.data.phase for e in experiments if "_full_" in e.name] == ["prefill", "decode"]


def h200_configs(model):
    return {
        path.stem
        for path in (isolated.CONFIGS / model).glob("*.yaml")
        if not path.name.startswith(("_", "b300_"))
    }


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_every_arm_a_job_can_run_has_a_config(model):
    for arm in (*isolated.ARMS[model], *isolated.EXTRA_ARMS[model]):
        assert (isolated.CONFIGS / model / f"{arm}.yaml").is_file()


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_every_config_left_out_of_the_default_arms_is_listed_with_why(model):
    default = set(isolated.ARMS[model])
    extra = set(isolated.EXTRA_ARMS[model])
    untimed = isolated.UNTIMED_ARMS[model]

    # Each config is in exactly one of the three, so a new one cannot be left out without a word
    assert not default & extra and not default & set(untimed) and not extra & set(untimed)
    assert h200_configs(model) == default | extra | set(untimed)
    assert all(why.strip() for why in untimed.values())


@pytest.mark.parametrize("model", sorted(isolated.ARMS))
def test_an_untimed_config_is_refused_as_an_arm(model):
    for arm in isolated.UNTIMED_ARMS[model]:
        with pytest.raises(ValueError, match=f"no arm \\['{arm}'\\]"):
            isolated.load_arms(model, None, arms=(arm,))


def test_the_arms_prefill_in_the_chunk_the_roofline_models():
    assert isolated.arm_overrides("qwen38_27b", "full", None)[0] == f"data.chunk={Kernels.CHUNK}"


def test_a_dry_run_says_whether_the_container_sees_the_records_root(monkeypatch, capsys, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["isolated", "qwen38_27b", "--arm", "full", "--output", str(elsewhere), "--dry-run"]
    )
    isolated.main()

    assert "output: refused at submission - qwen38_27b_kernels_full" in capsys.readouterr().out


def test_a_submission_outside_the_mounts_is_refused_before_the_job(monkeypatch, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS", raising=False)
    monkeypatch.setattr("sys.argv", ["isolated", "qwen38_27b", "--arm", "full", "--output", str(elsewhere)])
    monkeypatch.setattr(slurm, "executor", lambda *_, **__: pytest.fail("a job was made"))

    with pytest.raises(ValueError, match="outside every directory the container mounts"):
        isolated.main()
