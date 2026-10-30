"""Hold the GPT-2 hybrid's recurrent draws to FLA's actual initializer formulas."""

import pytest
import torch

from lsa.training.native.config import AttentionConfig, DataConfig, ModelConfig, NativeConfig, RunConfig
from lsa.training.native.initialization import initialize_gated_decay, initialize_gated_step_bias
from lsa.training.native.models import model_spec
from tests.helpers.training.native.initialization import fla_recurrent_parameters


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_seeded_recurrent_draws_match_fla_initializer_block_before_storage_cast(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        expected_decay, expected_bias = fla_recurrent_parameters(37)
        torch.manual_seed(23)
        decay = torch.nn.Parameter(torch.empty(37, dtype=dtype))
        bias = torch.nn.Parameter(torch.empty(37, dtype=dtype))
        initialize_gated_decay(decay)
        initialize_gated_step_bias(bias)
    torch.testing.assert_close(decay, expected_decay.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(bias, expected_bias.to(dtype), rtol=0, atol=0)
    assert decay.requires_grad and bias.requires_grad


def configuration(flavor):
    return model_spec(
        NativeConfig(
            attention=AttentionConfig(kind="dense"),
            model=ModelConfig(flavor=flavor),
            data=DataConfig(seq_len=8),
            run=RunConfig(output="/tmp/initialization-contract"),
        )
    ).model


def test_the_gpt2_hybrid_materializes_recurrent_parameters_in_the_upstream_ranges():
    config = configuration("0.8B-gpt2")
    recurrent = [layer.delta_net for layer in config.layers if layer.delta_net is not None]
    assert len(recurrent) == 18
    assert all(layer.param_init["A_log"] is initialize_gated_decay for layer in recurrent)
    assert all(layer.param_init["dt_bias"] is initialize_gated_step_bias for layer in recurrent)
    with torch.device("meta"):
        layer = recurrent[0].build()
    layer.to_empty(device="cpu")
    layer.init_states(buffer_device=None)
    decays = layer.A_log.exp()
    steps = torch.nn.functional.softplus(layer.dt_bias)
    assert torch.isfinite(decays).all() and torch.all((decays >= 0) & (decays <= 16))
    assert torch.all((steps >= 0.001 - 1e-7) & (steps <= 0.1 + 1e-7))
    assert not torch.equal(layer.dt_bias, torch.ones_like(layer.dt_bias))


def test_the_published_qwen_model_keeps_its_existing_initializers():
    recurrent = [layer.delta_net for layer in configuration("0.8B").layers if layer.delta_net is not None]
    assert all(layer.param_init["dt_bias"] is torch.nn.init.ones_ for layer in recurrent)
    assert all(layer.param_init["A_log"] is not initialize_gated_decay for layer in recurrent)
