"""The isolated layer run - every arm of a model, one after another, in one job on a node held whole."""

import argparse

import pytest

from lsa import site, slurm
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


def test_the_job_holds_one_gpu_of_a_node_exclusively_inside_the_reservation():
    experiments = isolated.load_arms("qwen38_27b", None)

    parameters = isolated.job_parameters(experiments, parsed("--reservation", "res"))

    assert parameters["gpus_per_node"] == 1
    assert parameters["slurm_exclusive"] is True
    assert parameters["slurm_additional_parameters"]["reservation"] == "res"
    # A one-GPU job is left unbound - the node is held whole, so no other job shares its sockets
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

    failed = isolated.run_arms(experiments)

    assert failed == [experiments[0].name]
    assert ran == [e.name for e in experiments]


def test_only_the_first_task_runs_the_arms(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")

    assert isolated.run_arms(isolated.load_arms("qwen38_27b", None)) == []


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
