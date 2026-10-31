"""Exercise DSA's KL parity and train the 340M arm through torchtitan."""

import pytest
import torch

from lsa import env
from lsa.training.native import config as native
from lsa.training.native.dsa import masked_kl
from lsa.training.native.trainer import check_label_contract, trainer_config
from tests.helpers.training.kernels.inputs import run_reference


@pytest.mark.gpu
@pytest.mark.cluster
def test_sampled_kl_forward_backward_matches_reference_training_code():
    torch.manual_seed(71)
    teacher = torch.randn(2, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    student = torch.randn_like(teacher, requires_grad=True)
    mask = torch.arange(128, device="cuda")[None, :] <= torch.arange(16, device="cuda")[:, None] + 32
    outputs, gradients = run_reference(
        "dsa_distill.self_distillation_kl",
        [teacher, student, mask],
        leaves=(1,),
        grad_output=torch.ones((), device="cuda"),
    )
    loss, _ = masked_kl(teacher, student, mask)
    (gradient,) = torch.autograd.grad(loss, student)
    torch.testing.assert_close(loss, outputs[0].cuda(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gradient, gradients[0].cuda(), rtol=1e-2, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.cluster
@pytest.mark.parametrize(
    ("selection", "source", "bias", "detached", "projection_bias"),
    [
        (False, "key_and_query", False, True, False),
        (True, "key_and_query", False, True, False),
        (False, "attention_input", False, False, True),
        (False, "attention_input", True, False, False),
        (False, "attention_input", True, True, False),
        (False, "sliced_key_and_query", False, True, False),
    ],
)
def test_three_steps_of_dsa_train_dense_then_sparse(
    tmp_path, selection, source, bias, detached, projection_bias
):
    experiment = native.NativeConfig(
        attention=native.AttentionConfig(
            kind="dsa",
            index_input=source,
            index_dim=64,
            index_loss_weight=0.1,
            index_detached=detached,
            index_bias=bias,
            window_size=128,
            sink_size=4,
            top_k=32,
            sparse_start_step=1,
            kl_on_selection=selection,
        ),
        model=native.ModelConfig(flavor="340M-gpt2", attention_bias=projection_bias),
        data=native.DataConfig(name="fineweb_edu_100bt", seq_len=256),
        run=native.RunConfig(
            output=str(tmp_path / "out"),
            optim=native.OptimConfig(steps=3, warmup_steps=1, tokens_per_rank=256, tokens_per_step=512),
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
