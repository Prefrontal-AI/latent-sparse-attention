"""The two GPT-2 flavours train a few steps on one card under the window arm, through the real data streams.

A microbatch holds one sequence or two: a lone sequence once took GatedDeltaNet's kernel down its no-boundary path,
which gave non-finite outputs from random initialisation in the first layer, at 4,096 tokens as at 16,384.
"""

import pytest
import torch

from lsa import env
from lsa.training.native import config as native
from lsa.training.native.trainer import check_label_contract, trainer_config

pytestmark = pytest.mark.gpu

MULTI_QUERY = {"query_head_count": 8, "kv_head_count": 1, "head_dim": 128, "rotary_dim": 0}


@pytest.mark.parametrize(
    ("flavor", "head_shape", "corpus", "seq_len", "checkpointing", "sequences"),
    [
        ("0.8B-gpt2", {}, "fineweb_edu_100bt", 4096, "none", 2),
        ("0.8B-gpt2", {}, "fineweb_edu_100bt", 4096, "none", 1),
        ("0.8B-gpt2", {}, "gpt2", 16384, "none", 1),
        ("0.8B-gpt2", MULTI_QUERY, "fineweb_edu_100bt", 4096, "none", 2),
        ("340M-gpt2", {}, "fineweb_edu_100bt", 2048, "none", 2),
        ("0.8B-gpt2", {}, "gpt2", 16384, "none", 2),
        ("0.8B-gpt2", {}, "gpt2", 32768, "full", 2),
    ],
    ids=[
        "hybrid-fineweb",
        "hybrid-one-sequence-fineweb",
        "hybrid-one-sequence-long-documents-16k",
        "hybrid-multi-query-fineweb",
        "340m-fineweb",
        "hybrid-long-documents-16k",
        "hybrid-long-documents-32k",
    ],
)
def test_a_few_accumulated_steps_of_the_window_arm_leave_every_weight_finite(
    tmp_path, flavor, head_shape, corpus, seq_len, checkpointing, sequences
):
    experiment = native.NativeConfig(
        attention=native.AttentionConfig(kind="window", window_size=1536, sink_size=4),
        model=native.ModelConfig(flavor=flavor, **head_shape),
        data=native.DataConfig(name=corpus, seq_len=seq_len),
        run=native.RunConfig(
            output=str(tmp_path / "out"),
            optim=native.OptimConfig(
                steps=3,
                warmup_steps=1,
                weight_decay_scope="matrices",
                tokens_per_rank=sequences * seq_len,
                tokens_per_step=2 * sequences * seq_len,
            ),
            runtime=native.RuntimeConfig(activation_checkpoint=checkpointing, autocast_bf16=True),
            evaluation=native.EvaluationConfig(every=0),
        ),
    )
    env.adopt_job_environment()
    check_label_contract()
    trainer = trainer_config(experiment).build()
    try:
        assert trainer.gradient_accumulation_steps == 2
        trainer.train()
        for parameter in trainer.model_parts[0].parameters():
            weight = parameter.detach()
            full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
            assert torch.isfinite(full).all()
    finally:
        trainer.close()
