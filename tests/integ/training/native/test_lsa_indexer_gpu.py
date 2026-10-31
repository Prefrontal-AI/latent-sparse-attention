"""Train learned LSA through torchtitan's model, optimizer and real corpus stream."""

import pytest
import torch

from lsa import env
from lsa.training.native import config as native
from lsa.training.native.trainer import check_label_contract, trainer_config

pytestmark = [pytest.mark.gpu, pytest.mark.cluster]


@pytest.mark.parametrize(
    ("training", "summaries"), [("none", False), ("kl", False), ("ste", False), ("kl", True), ("ste", True)]
)
def test_three_native_updates_keep_lsa_parameters_and_projections_finite(tmp_path, training, summaries):
    experiment = native.NativeConfig(
        attention=native.AttentionConfig(
            kind="lsa",
            coarse=False,
            fast_read=True,
            window_size=128,
            block_size=128,
            top_k=32,
            pool_size=4,
            bucket_size=8,
            bucket_cap=16,
            index_dim=64,
            index_input="none" if training == "none" else "key_and_query",
            index_training=training,
            summaries=summaries,
        ),
        model=native.ModelConfig(
            flavor="0.8B-gpt2", query_head_count=8, kv_head_count=1, head_dim=128, rotary_dim=0
        ),
        data=native.DataConfig(name="fineweb_edu_100bt", seq_len=512),
        run=native.RunConfig(
            output=str(tmp_path / "out"),
            optim=native.OptimConfig(steps=3, warmup_steps=1, tokens_per_rank=512),
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
