import pytest

from lsa.training.native.config import DataConfig, NativeConfig, RunConfig, RuntimeConfig
from lsa.training.native.evaluation import evaluator_config
from lsa.training.native.trainer import trainer_config
from tests.helpers.training.data.fake_corpus import write_corpus


@pytest.mark.parametrize("enabled", [False, True])
def test_training_and_scoring_share_the_requested_precision(tmp_path, enabled):
    card = write_corpus(
        tmp_path / "corpus", sources=("arxiv",), documents=2, sample_length=512, vocab_size=4096
    )
    experiment = NativeConfig(
        data=DataConfig(path=card.path, seq_len=512),
        run=RunConfig(output=str(tmp_path / "run"), runtime=RuntimeConfig(autocast_bf16=enabled)),
    )
    trainer = trainer_config(experiment)
    assert trainer.autocast_bf16 is enabled
    assert trainer.training.dtype == "float32"
    assert trainer.training.mixed_precision_param == ("float32" if enabled else "bfloat16")
    assert trainer.training.mixed_precision_reduce == "float32"
    assert evaluator_config(experiment).autocast_bf16 is enabled
