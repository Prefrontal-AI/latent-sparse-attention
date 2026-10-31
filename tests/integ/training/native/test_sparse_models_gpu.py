import pytest
import torch

from lsa import env
from lsa.training.native import config as native
from lsa.training.native.trainer import check_label_contract, trainer_config

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]


@pytest.mark.parametrize("kind", ["nsa", "pisa"])
def test_sparse_arm_trains_three_accumulated_steps(tmp_path, kind):
    experiment = native.NativeConfig(
        attention=native.AttentionConfig(kind=kind, window_size=512),
        model=native.ModelConfig(
            flavor="0.8B-gpt2", query_head_count=8, kv_head_count=1, head_dim=128, rotary_dim=0
        ),
        data=native.DataConfig(name="fineweb_edu_100bt", seq_len=4096),
        run=native.RunConfig(
            output=str(tmp_path / "out"),
            optim=native.OptimConfig(steps=3, warmup_steps=1, tokens_per_rank=4096, tokens_per_step=8192),
            evaluation=native.EvaluationConfig(every=0),
        ),
    )
    env.adopt_job_environment()
    check_label_contract()
    trainer = trainer_config(experiment).build()
    try:
        trainer.train()
        for parameter in trainer.model_parts[0].parameters():
            weight = parameter.detach()
            full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
            assert torch.isfinite(full).all()
    finally:
        trainer.close()
