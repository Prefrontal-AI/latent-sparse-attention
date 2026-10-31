import pytest

from lsa.training.native.config import NativeConfig, RunConfig
from lsa.training.native.evaluation import checkpoint_step, evaluator_config


def test_a_checkpoint_step_is_read_from_the_folder_name():
    assert checkpoint_step("/srv/run/checkpoint/step-8000") == 8000
    assert checkpoint_step("/srv/run/checkpoint/step-8000/") == 8000


@pytest.mark.parametrize(
    "name", ["/srv/run/checkpoint", "/srv/run/checkpoint/latest", "/srv/run/step-x", "step-"]
)
def test_a_folder_that_is_not_a_step_is_refused(name):
    with pytest.raises(ValueError, match="step-N"):
        checkpoint_step(name)


def test_the_evaluator_writes_under_the_runs_evaluation_folder_and_carries_the_evaluation_block():
    from lsa.training.native.config import DataConfig, EvaluationConfig

    experiment = NativeConfig(
        data=DataConfig(name="qwen3_5", seq_len=4096),
        run=RunConfig(
            output="/tmp/native-test", name="scored", evaluation=EvaluationConfig(sequences=8, every=100)
        ),
    )
    config = evaluator_config(experiment)
    assert config.records == "/tmp/native-test/scored/evaluation"
    assert config.sequences == 8 and config.freq == 100 and config.enable
