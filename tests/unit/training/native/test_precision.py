from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torchtitan.distributed import utils as dist_utils

from lsa.training.native import precision
from lsa.training.native.evaluation import HeldOutEvaluator
from lsa.training.native.precision import allow_exact_fp32_matmul, autocast_context
from lsa.training.native.trainer import NativeTrainer


@pytest.mark.parametrize("enabled", [False, True])
def test_compute_autocasts_without_rounding_master_parameters_or_backward(enabled):
    model = torch.nn.Linear(8, 4)
    observations = []
    model.register_forward_hook(
        lambda module, inputs, output: observations.append(
            (torch.is_autocast_enabled("cpu"), output.dtype, module.weight.dtype)
        )
    )
    model.weight.register_hook(
        lambda gradient: observations.append(("backward", torch.is_autocast_enabled("cpu"), gradient.dtype))
    )
    trainer = object.__new__(NativeTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(autocast_bf16=enabled)
    trainer.train_context = nullcontext
    trainer.model_parts = [model]
    trainer.loss_fn = lambda predictions, labels, count, **kwargs: (
        (predictions.float() - labels).square().sum() / count,
        None,
    )
    trainer._forward_backward_body(torch.randn(2, 8), torch.zeros(2, 4), torch.tensor(8), {})
    assert observations == [
        (enabled, torch.bfloat16 if enabled else torch.float32, torch.float32),
        ("backward", False, torch.float32),
    ]
    assert model.weight.grad.dtype == torch.float32


def test_held_out_scoring_uses_training_compute_precision_and_preserves_random_state(monkeypatch):
    # An earlier test in the process may have initialised CUDA
    # => score on the CPU path the model is on
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    observed = []

    class Evaluator(HeldOutEvaluator):
        def _validate(self, model_parts, step):
            observed.append(model_parts[0](torch.randn(2, 8)).dtype)

    evaluator = object.__new__(Evaluator)
    evaluator.autocast_bf16 = True
    model = torch.nn.Linear(8, 4)
    initial = torch.get_rng_state().clone()
    evaluator.validate([model], 1)
    assert observed == [torch.bfloat16]
    assert torch.equal(initial, torch.get_rng_state())
    assert not torch.is_autocast_enabled("cpu")


def test_standalone_forward_context_restores_both_scopes():
    with autocast_context(context=nullcontext, device_type="cpu", enabled=True):
        assert torch.is_autocast_enabled("cpu")
        assert (torch.randn(2, 8) @ torch.randn(8, 4)).dtype == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")


@pytest.fixture
def torchtitan_setup(monkeypatch):
    # The installer replaces a module attribute => restore it after each test
    monkeypatch.setattr(
        dist_utils,
        "enable_fp32_matmul_emulation_with_bf16x9",
        dist_utils.enable_fp32_matmul_emulation_with_bf16x9,
    )
    monkeypatch.setattr(dist_utils, "device_type", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 3))
    monkeypatch.setattr(torch.version, "hip", None)


class Matmul:
    def __init__(self, error):
        self.error = error

    def __setattr__(self, name, value):
        if name == "fp32_precision" and self.error is not None:
            raise self.error
        object.__setattr__(self, name, value)


def test_a_torch_without_bfx9_falls_back_to_exact_fp32_with_a_warning(monkeypatch, torchtitan_setup):
    monkeypatch.setattr(torch.backends.cuda, "matmul", Matmul(RuntimeError("Unknown precision: bfx9")))
    messages = []
    monkeypatch.setattr(precision.logger, "warning", messages.append)
    allow_exact_fp32_matmul()
    allow_exact_fp32_matmul()
    dist_utils.enable_fp32_matmul_emulation_with_bf16x9()
    assert len(messages) == 1


def test_a_torch_with_bfx9_is_set_up_as_torchtitan_does(monkeypatch, torchtitan_setup):
    matmul = Matmul(None)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)
    allow_exact_fp32_matmul()
    dist_utils.enable_fp32_matmul_emulation_with_bf16x9()
    assert matmul.fp32_precision == "bfx9"


def test_a_failure_that_is_not_the_missing_precision_still_raises(monkeypatch, torchtitan_setup):
    monkeypatch.setattr(torch.backends.cuda, "matmul", Matmul(ValueError("bad")))
    allow_exact_fp32_matmul()
    with pytest.raises(ValueError, match="requires PyTorch"):
        dist_utils.enable_fp32_matmul_emulation_with_bf16x9()
