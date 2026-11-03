from pathlib import Path

import pytest
import torch

from lsa.training.native.config import (
    CheckpointConfig,
    DataConfig,
    NativeConfig,
    OptimConfig,
    RunConfig,
    load,
)
from lsa.training.native.trainer import checkpoint_config, trainer_config
from tests.helpers.training.data.fake_corpus import write_corpus
from tests.helpers.training.schedule import reference_factor

CONFIGS = Path(__file__).resolve().parents[4] / "configs/training/native"


def test_a_save_is_staged_to_pinned_memory_and_written_in_the_background_by_default():
    assert (
        checkpoint_config(RunConfig(output="/tmp/native", checkpoint_every=100)).async_mode
        == "async_with_pinned_mem"
    )


def test_a_run_can_ask_for_the_blocking_save():
    run = RunConfig(output="/tmp/native", checkpoint_every=100, checkpoint_async=False)
    assert checkpoint_config(run).async_mode == "disabled"


def test_a_branch_loads_the_trunks_weights_alone_and_saves_nothing_unless_asked():
    checkpoint = checkpoint_config(RunConfig(output="/tmp/native", init_from="/trunk/checkpoint/step-1500"))
    assert checkpoint.enable
    assert checkpoint.initial_load_path == "/trunk/checkpoint/step-1500"
    assert checkpoint.initial_load_model_only


def test_a_run_from_random_init_that_saves_nothing_has_no_checkpointer():
    assert not checkpoint_config(RunConfig(output="/tmp/native")).enable


def test_the_trainers_scheduler_steps_the_reference_training_codes_learning_rate(tmp_path):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=5e-4)
    run = RunConfig(
        output="/tmp/native",
        optim=OptimConfig(
            steps=200, lr=5e-4, warmup_steps=10, decay_ratio=0.25, decay_shape="sqrt", min_lr_factor=0.1
        ),
    )
    card = write_corpus(
        tmp_path / "corpus", sources=("arxiv",), documents=2, sample_length=512, vocab_size=4096
    )
    experiment = NativeConfig(data=DataConfig(path=card.path, seq_len=512), run=run)
    schedulers = trainer_config(experiment).lr_scheduler.build(optimizers=[optimizer], training_steps=200)

    for step in range(200):
        assert optimizer.param_groups[0]["lr"] == 5e-4 * reference_factor(step, 10, 200, 50, 0.1, "sqrt")
        optimizer.step()
        schedulers.step()


def test_a_run_saves_its_decay_start_when_asked_and_not_otherwise():
    optim = OptimConfig(steps=1000, warmup_steps=50, decay_ratio=0.2)
    saving = checkpoint_config(
        RunConfig(output="/tmp/native", optim=optim, checkpoint=CheckpointConfig(decay_start=True))
    )
    assert saving.enable
    assert saving.decay_start_step == 800
    assert checkpoint_config(RunConfig(output="/tmp/native", optim=optim)).decay_start_step is None


def test_attention_schedules_receive_the_restored_update_before_any_microbatch(monkeypatch):
    from torchtitan.trainer import Trainer

    from lsa.training.native.trainer import NativeTrainer

    calls = []

    class Attention(torch.nn.Module):
        def set_step(self, step):
            calls.append(("attention", step))

    monkeypatch.setattr(Trainer, "train_step", lambda self, iterator: calls.append(("update", self.step)))
    trainer = object.__new__(NativeTrainer)
    trainer.model_parts = [torch.nn.Sequential(Attention(), Attention())]
    trainer.step = 9
    trainer.train_step(iter(()))
    assert calls == [("attention", 8), ("attention", 8), ("update", 9)]
    assert NativeTrainer.Config._owner is NativeTrainer


def test_trainer_config_preserves_cosine_schedule_and_its_warmup(tmp_path):
    from lsa.training.native.schedule import learning_rate_factor

    card = write_corpus(
        tmp_path / "corpus", sources=("arxiv",), documents=2, sample_length=512, vocab_size=4096
    )
    experiment = NativeConfig(
        data=DataConfig(path=card.path, seq_len=512),
        run=RunConfig(output="/tmp/native", optim=OptimConfig(schedule="cosine", steps=100, warmup_steps=5)),
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.001)
    schedulers = trainer_config(experiment).lr_scheduler.build(optimizers=[optimizer], training_steps=100)
    for step in range(101):
        expected = learning_rate_factor(
            step,
            steps=100,
            warmup_steps=5,
            decay_steps=80,
            min_lr_factor=0.1,
            decay_shape="cosine",
            schedule="cosine",
        )
        assert optimizer.param_groups[0]["lr"] == expected * 0.001
        optimizer.step()
        schedulers.step()


def test_a_scorer_of_released_weights_reads_no_corpus_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("LSA_ROOT", str(tmp_path))
    [experiment] = load(CONFIGS / "qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml")
    with pytest.raises(FileNotFoundError, match="corpus.json"):
        trainer_config(experiment)
    assert trainer_config(experiment, check_corpora=False).model_spec is not None
