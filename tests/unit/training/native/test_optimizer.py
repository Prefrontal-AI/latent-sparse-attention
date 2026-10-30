"""Hold native decay policy, flag preservation and AdamW updates to explicit contracts."""

from copy import deepcopy

import pytest
import torch

from lsa.training.native.config import OptimConfig, load
from lsa.training.native.optimizer import parameter_name, preserve_parameter_flags
from tests.helpers.training.native.optimizer import Parameters, build, replace_parameters


def test_matrix_scope_decays_embeddings_and_linear_matrices_but_honors_marks():
    model = Parameters()
    optimizer = build(model, "matrices").optimizers[0]
    decayed, excluded = optimizer.param_groups
    assert decayed["param_names"] == ["embedding.weight", "linear.weight"]
    assert excluded["param_names"] == ["excluded", "scale", "norm.weight", "norm.bias", "linear.bias"]
    assert decayed["weight_decay"] == 0.2
    assert excluded["weight_decay"] == 0.0
    assert decayed["betas"] == excluded["betas"] == (0.9, 0.95)
    assert decayed["eps"] == excluded["eps"] == 1e-8


def test_all_scope_retains_the_original_parameter_order_and_decays_everything():
    model = Parameters()
    optimizer = build(model, "all").optimizers[0]
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["param_names"] == [name for name, _ in model.named_parameters()]
    assert optimizer.param_groups[0]["weight_decay"] == 0.2


def test_constructor_exclusions_survive_parameter_replacement_and_checkpoint_wrapping():
    model = preserve_parameter_flags(Parameters(), parallelize=replace_parameters)
    assert not hasattr(model._checkpoint_wrapped_module.excluded, "_no_weight_decay")
    optimizer = build(model, "matrices").optimizers[0]
    assert optimizer.param_groups[0]["param_names"] == ["embedding.weight", "linear.weight"]
    assert "excluded" in optimizer.param_groups[1]["param_names"]
    assert parameter_name("_orig_mod.layers.0._checkpoint_wrapped_module.weight") == "layers.0.weight"


@pytest.mark.parametrize("scope", ["all", "matrices"])
def test_optimizer_updates_match_explicit_adamw_groups(scope):
    model = Parameters().double()
    reference = deepcopy(model)
    decayed = [reference.embedding.weight, reference.linear.weight]
    excluded = [
        reference.excluded,
        reference.scale,
        reference.norm.weight,
        reference.norm.bias,
        reference.linear.bias,
    ]
    groups = (
        [{"params": list(reference.parameters()), "weight_decay": 0.2}]
        if scope == "all"
        else [{"params": decayed, "weight_decay": 0.2}, {"params": excluded, "weight_decay": 0.0}]
    )
    native = build(model, scope)
    expected = torch.optim.AdamW(groups, lr=0.01, betas=(0.9, 0.95), eps=1e-8, fused=False, foreach=False)
    generator = torch.Generator().manual_seed(31)
    for _ in range(5):
        for (_, parameter), (_, other) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            gradient = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.grad = gradient.clone()
            other.grad = gradient.clone()
        native.step()
        expected.step()
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)


def test_gpt2_recipes_select_matrix_decay_explicitly():
    for family in ("dense_340m_gpt2", "qwen35_0_8b_gpt2"):
        (config,) = load(f"configs/training/native/{family}/dense.yaml")
        assert config.run.optim.weight_decay_scope == "matrices"
    assert OptimConfig().weight_decay_scope == "all"
    with pytest.raises(ValueError, match="weight_decay_scope"):
        OptimConfig(weight_decay_scope="unknown")


@pytest.mark.parametrize(
    "model, decay", [(torch.nn.Linear(3, 3, bias=False), 0.2), (torch.nn.LayerNorm(3), 0.0)]
)
def test_empty_decay_categories_are_omitted(model, decay):
    groups = build(model, "matrices").optimizers[0].param_groups
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == decay


def test_matrix_group_state_round_trips_by_parameter_name():
    model = Parameters().double()
    optimizer = build(model, "matrices")
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.3)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    copied = preserve_parameter_flags(model, parallelize=lambda current: deepcopy(current))
    resumed = build(copied, "matrices")
    resumed.load_state_dict(saved)
    assert [group["weight_decay"] for group in resumed.optimizers[0].param_groups] == [0.2, 0.0]
    for original, parameter in zip(model.parameters(), copied.parameters(), strict=True):
        original.grad, parameter.grad = torch.full_like(original, 0.7), torch.full_like(parameter, 0.7)
    optimizer.step()
    resumed.step()
    for original, parameter in zip(model.parameters(), copied.parameters(), strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_a_native_optimizer_config_requires_a_single_adamw_setup():
    from lsa.training.native.optimizer import NativeOptimizers

    with pytest.raises(ValueError, match="one AdamW setup.*got"):
        NativeOptimizers.Config(param_groups=[]).build(model_parts=[Parameters()])
