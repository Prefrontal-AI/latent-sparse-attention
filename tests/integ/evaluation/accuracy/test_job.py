"""The accuracy step's handover to the suite - the command line that scores a live server.

Nothing here starts a server or runs the suite. What is worth testing is that each setting of the data and run
blocks reaches the suite's own command line, and that what was not asked for stays off it.
"""

import sys
from pathlib import Path

import pytest

from lsa.evaluation.accuracy.job import suite_command
from lsa.evaluation.config import (
    AccuracyConfig,
    AccuracyDataConfig,
    AccuracyModelConfig,
    AccuracyRunConfig,
)

# A docker tag rather than a built squashfs - enroot caches the pull
IMAGE = "lmsysorg/sglang:latest"


def scored(model="", data=None, **run):
    return AccuracyConfig(
        model=AccuracyModelConfig(name=model, attention="native"),
        data=data or AccuracyDataConfig(),
        # A launch has read the card from the node by the time the suite is handed its command
        run=AccuracyRunConfig(image=IMAGE, **{"gpu": "NVIDIA H200", **run}),
    )


def test_the_suite_command_names_the_port_the_suite_and_the_records_root(tmp_path):
    argv = suite_command(scored(port=61300), tmp_path / "records", "a-run")
    # The suite is a module of this package now, not a script in another tree
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-u", "-m", "lsa.evaluation.accuracy"]
    assert argv[argv.index("--port") + 1] == "61300"
    assert argv[argv.index("--suite") + 1] == "full"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "records")


def test_the_suite_command_names_the_run_and_dictates_its_id(tmp_path):
    argv = suite_command(scored(name="a-config"), tmp_path, "a-config_h200__2026-10-01__12-00-00")
    assert argv[argv.index("--name") + 1] == "a-config_h200"
    assert argv[argv.index("--gpu") + 1] == "NVIDIA H200"
    assert argv[argv.index("--run-id") + 1] == "a-config_h200__2026-10-01__12-00-00"


def test_the_suite_command_leaves_out_what_was_not_asked_for(tmp_path):
    argv = suite_command(scored(python="python"), tmp_path, "a-run")
    assert "--tasks" not in argv
    assert "--note" not in argv
    assert "--no-resume" not in argv
    assert "--model" not in argv
    assert "--attention-kind" not in argv


def test_the_image_reaches_the_suite_so_the_record_keeps_it_with_the_mesh(tmp_path):
    argv = suite_command(scored(), tmp_path, "a-run")
    assert argv[argv.index("--image") + 1] == IMAGE


def test_the_hub_id_reaches_the_suite_so_the_record_names_the_model():
    argv = suite_command(scored(model="Qwen/Qwen3.5-4B", python="python"), Path("records"), "a-run")
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.5-4B"


def test_a_dictated_run_id_alone_resumes_that_run():
    # The run id is derived from the config, so a second run of one experiment continues the first
    argv = suite_command(scored(python="python"), Path("records"), "a-run")
    assert argv[argv.index("--run-id") + 1] == "a-run"
    assert "--no-resume" not in argv


def test_force_scores_every_sample_again_into_the_same_run():
    argv = suite_command(scored(python="python"), Path("records"), "a-run", force=True)
    assert argv[argv.index("--run-id") + 1] == "a-run"
    assert "--no-resume" in argv


def test_the_suite_command_passes_the_task_list_and_the_extras(tmp_path):
    data = AccuracyDataConfig(tasks=["gpqa_diamond", "ruler_cwe_131072"])
    extras = {"python": "python", "note": "a baseline", "tag": "debug", "args": ["--allow-hash-mismatch"]}
    argv = suite_command(scored(data=data, **extras), tmp_path, "a-run")
    assert argv[argv.index("--tasks") + 1 : argv.index("--tasks") + 3] == ["gpqa_diamond", "ruler_cwe_131072"]
    assert argv[argv.index("--note") + 1] == "a baseline"
    assert argv[argv.index("--tag") + 1] == "debug"
    assert argv[-1] == "--allow-hash-mismatch"


def test_the_data_block_refuses_a_suite_the_runner_does_not_have():
    with pytest.raises(ValueError, match="data.suite"):
        AccuracyDataConfig(suite="medium")


def test_the_run_block_refuses_what_the_runner_would():
    with pytest.raises(ValueError):
        AccuracyRunConfig(image=IMAGE, python="python", concurrency_scale=0.0)


def test_a_declared_data_root_reaches_the_suite(tmp_path):
    data = AccuracyDataConfig(root=str(tmp_path / "corpus"))
    argv = suite_command(scored(data=data, python="python"), tmp_path, "a-run")
    assert argv[argv.index("--data-root") + 1] == str(tmp_path / "corpus")


def test_a_corpus_in_another_tokenizer_names_its_longbench_file_and_its_pins(tmp_path):
    data = AccuracyDataConfig(root=str(tmp_path / "corpus"), longbench_context=935136, pins="qwen38_27b")
    argv = suite_command(scored(data=data, python="python"), tmp_path, "a-run")
    assert argv[argv.index("--max-context") + 1] == "935136"
    assert argv[argv.index("--pins") + 1] == "qwen38_27b"


def test_the_default_corpus_leaves_the_suite_its_own_longbench_file_and_pins(tmp_path):
    argv = suite_command(scored(python="python"), tmp_path, "a-run")
    assert "--max-context" not in argv
    assert "--pins" not in argv


def test_no_data_root_leaves_the_suite_its_own_default(tmp_path):
    argv = suite_command(scored(python="python"), tmp_path, "a-run")
    assert "--data-root" not in argv


def test_the_generation_file_reaches_the_suite(tmp_path):
    generation = tmp_path / "generation.json"
    argv = suite_command(scored(), tmp_path, "a-run", generation=generation)
    assert argv[argv.index("--generation") + 1] == str(generation)
    assert "--generation" not in suite_command(scored(), tmp_path, "a-run")


@pytest.mark.parametrize(
    ("model", "attention", "kind"),
    [
        ("qwen38_27b", "native", "full"),
        ("qwen38_27b_qsa", "native", "block_sparse"),
        ("glm53_flash", "native", "block_sparse"),
    ],
)
def test_the_attention_kind_reaches_the_suite_so_the_record_tells_qsa_from_full(
    tmp_path, model, attention, kind
):
    # Both checkpoints serve `native`, so only the card can say which method the run is
    experiment = AccuracyConfig(
        model=AccuracyModelConfig(name=model, attention=attention),
        data=AccuracyDataConfig(),
        run=AccuracyRunConfig(image=IMAGE, gpu="NVIDIA H200"),
    )
    argv = suite_command(experiment, tmp_path, "a-run")
    assert argv[argv.index("--attention-kind") + 1] == kind


def test_the_request_timeout_reaches_the_suite_and_defaults_to_four_hours(tmp_path):
    argv = suite_command(scored(python="python"), tmp_path, "a-run")
    assert float(argv[argv.index("--request-timeout") + 1]) == 14400
    argv = suite_command(scored(python="python", request_timeout=28800), tmp_path, "a-run")
    assert float(argv[argv.index("--request-timeout") + 1]) == 28800


def test_the_run_block_refuses_a_request_timeout_that_is_not_positive():
    with pytest.raises(ValueError, match="request_timeout"):
        AccuracyRunConfig(image=IMAGE, python="python", request_timeout=0)
