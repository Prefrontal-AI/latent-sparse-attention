"""The module timer's submission - one kernel config, its overrides laid over it, its phase pinned."""

import pytest

from lsa.evaluation.performance.kernels import isolated, modules

NATIVE = str(isolated.CONFIGS / "glm53_flash" / "native.yaml")

SWEPT = """
model:
  name: zai-org/GLM-5.3-Flash
  attention: native
data:
  checkpoint: /models/cut
  prompt: /prompts/p.txt
run:
  name: swept
  image: /images/sglang.img
  tp: 1
sweep:
  data.phase: [prefill, decode]
  data.checkpoint: [/models/cut_a, /models/cut_b]
"""


@pytest.fixture
def submitted(monkeypatch, tmp_path):
    """Catch what `main` hands the submission, with the test's directory mounted."""
    monkeypatch.setenv("LSA_EXTRA_MOUNTS", str(tmp_path))
    caught = []
    monkeypatch.setattr(
        modules.slurm, "submit", lambda args, name, env, parameters, *call: caught.append(call)
    )
    return caught


def main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["modules", *argv])
    modules.main()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_the_phase_sweep_every_kernel_config_has_narrows_to_the_phase_timed(phase):
    experiment = modules.load_experiment(NATIVE, phase)

    assert experiment.data.phase == phase
    assert experiment.data.checkpoint


def test_overrides_after_the_flags_reach_the_config_the_timer_reads(monkeypatch, tmp_path, submitted):
    main(monkeypatch, NATIVE, "--phase", "decode", "--results", str(tmp_path), "data.checkpoint=/cuts/other")

    ((function, experiment, phase, repeats, output),) = submitted
    assert function is modules.time_modules
    assert experiment.data.checkpoint == "/cuts/other"
    assert (experiment.data.phase, phase, repeats) == ("decode", "decode", 30)
    assert output.is_relative_to(tmp_path)


def test_overrides_may_come_before_the_flags_too(monkeypatch, tmp_path, submitted):
    main(monkeypatch, NATIVE, "data.checkpoint=/cuts/other", "--phase", "prefill", "--results", str(tmp_path))

    ((_, experiment, *_),) = submitted
    assert experiment.data.checkpoint == "/cuts/other"


def test_a_config_yielding_several_runs_is_refused_naming_the_count_and_the_points(tmp_path):
    config = tmp_path / "swept.yaml"
    config.write_text(SWEPT)

    with pytest.raises(ValueError, match="yields 2 runs") as refused:
        modules.load_experiment(str(config), "decode")
    assert "data.checkpoint=/models/cut_a" in str(refused.value)
    assert "data.checkpoint=/models/cut_b" in str(refused.value)


def test_pinning_the_swept_key_picks_one_run(tmp_path):
    config = tmp_path / "swept.yaml"
    config.write_text(SWEPT)

    experiment = modules.load_experiment(str(config), "decode", ("data.checkpoint=/models/cut_b",))

    assert experiment.data.checkpoint == "/models/cut_b"


def test_a_config_naming_no_checkpoint_is_refused(tmp_path):
    config = tmp_path / "bare.yaml"
    config.write_text(SWEPT.split("sweep:")[0].replace("  checkpoint: /models/cut\n", ""))

    with pytest.raises(ValueError, match="names no truncated checkpoint"):
        modules.load_experiment(str(config), "prefill")


def test_a_results_root_the_container_does_not_mount_is_refused(monkeypatch, tmp_path_factory, submitted):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS")

    with pytest.raises(ValueError, match="the modules run directory .* is outside every directory"):
        main(monkeypatch, NATIVE, "--phase", "decode", "--results", str(elsewhere))
    assert submitted == []


def test_a_dry_run_prints_the_refusal_instead(monkeypatch, capsys, tmp_path_factory, submitted):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.delenv("LSA_EXTRA_MOUNTS")

    main(monkeypatch, NATIVE, "--phase", "decode", "--results", str(elsewhere), "--dry-run")

    assert "refused at submission - the modules run directory" in capsys.readouterr().out
