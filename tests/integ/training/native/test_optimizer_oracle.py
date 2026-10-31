"""Compare grouping and updates with the actual reference training code source, including recurrent states."""

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from lsa.training.native.config import AttentionConfig, DataConfig, ModelConfig, NativeConfig, RunConfig
from lsa.training.native.models import model_spec
from lsa.training.native.optimizer import preserve_parameter_flags
from tests.helpers.training.native.optimizer import Parameters, build, replace_parameters, upstream_optimizer

pytestmark = pytest.mark.cluster


def oracle():
    clone = os.environ.get("REFERENCE_CLONE")
    if not clone:
        pytest.skip("set REFERENCE_CLONE to the reference training code main")
    return upstream_optimizer(clone)


@pytest.mark.parametrize("scope", ["all", "matrices"])
def test_actual_upstream_groups_and_optimizer_updates_match(scope):
    upstream = oracle()
    model = Parameters().double()
    reference = deepcopy(model)
    reference.excluded._no_weight_decay = True
    flags = upstream.parameter_flags(reference)
    native = build(preserve_parameter_flags(model, parallelize=replace_parameters), scope)
    reference = replace_parameters(reference)
    groups = upstream.build_param_groups(reference, 0.2, scope=scope, flags=flags)
    expected = upstream._adamw(
        upstream.torch_param_groups(groups, 0.01),
        SimpleNamespace(lr=0.01, adam_betas=[0.9, 0.95], adam_eps=1e-8, weight_decay=0.2),
    )
    for got, want in zip(native.optimizers[0].param_groups, groups, strict=True):
        assert got["param_names"] == want.names
        assert got["weight_decay"] == want.weight_decay
    generator = torch.Generator().manual_seed(9)
    for _ in range(5):
        for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
            gradient = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.grad, other.grad = gradient.clone(), gradient.clone()
        native.step()
        expected.step()
    # The two optimizers' kernels may round one element an ulp apart over five steps
    # => compare within an ulp, not bitwise
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, other, rtol=1e-15, atol=0)


def test_real_hybrid_recurrent_parameters_and_every_group_match_upstream():
    upstream = oracle()
    experiment = NativeConfig(
        attention=AttentionConfig(kind="dense"),
        model=ModelConfig(flavor="0.8B-gpt2"),
        data=DataConfig(seq_len=8),
        run=RunConfig(output="/tmp/group-contract"),
    )
    with torch.device("meta"):
        model = model_spec(experiment).model.build()
    optimizer = build(model, "matrices").optimizers[0]
    expected = upstream.build_param_groups(model, 0.2, scope="matrices")
    for got, want in zip(optimizer.param_groups, expected, strict=True):
        assert got["param_names"] == want.names
        assert got["weight_decay"] == want.weight_decay
    excluded = set(optimizer.param_groups[1]["param_names"])
    recurrent = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith(("A_log", "dt_bias"))
    ]
    assert len(recurrent) == 36
    assert all(parameter.ndim == 1 and name in excluded for name, parameter in recurrent)
    assert "tok_embeddings.weight" in optimizer.param_groups[0]["param_names"]
