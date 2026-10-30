import pytest
import torch

from lsa.training.native.config import AttentionConfig
from lsa.training.native.nsa import attention_config
from tests.unit.training.native.test_heads import DIM, initialised, shaped


def module(length=24):
    config = AttentionConfig(
        kind="nsa",
        compress_block_size=8,
        compress_stride=4,
        select_block_size=8,
        select_block_count=3,
        window_size=6,
    )
    return initialised(attention_config(shaped(rotary_dim=0), config, length))


def test_three_branch_attention_is_causal_and_independent_between_sequences():
    torch.manual_seed(2)
    attention = module()
    tokens = torch.randn(48, DIM)
    changed = tokens.clone()
    changed[12:24] += torch.randn_like(changed[12:24])
    changed[24:] += torch.randn_like(changed[24:])
    with torch.no_grad():
        expected = attention(tokens, None)
        actual = attention(changed, None)
    torch.testing.assert_close(actual[:12], expected[:12], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(actual[24:], expected[24:])
    separate = attention(tokens[:24], None)
    torch.testing.assert_close(separate, expected[:24])


def test_loss_reaches_every_branch_and_compression_parameter():
    attention = module()
    output = attention(torch.randn(24, DIM), None)
    output.square().sum().backward()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
    "changes",
    [
        {"compress_stride": 3},
        {"select_block_count": 2},
        {"compress_hidden_dim": 0},
        {"compress_block_size": 0},
    ],
)
def test_invalid_compression_and_selection_sizes_are_rejected(changes):
    with pytest.raises(ValueError):
        AttentionConfig(kind="nsa", **changes)


def test_rotary_queries_are_rejected():
    config = attention_config(shaped(rotary_dim=8), AttentionConfig(kind="nsa"), 24)
    with pytest.raises(ValueError, match="rotary_dim 0"):
        config.build()


@pytest.mark.parametrize("count,budget", [(16, 1536), (32, 2560), (64, 4608)])
def test_key_budget_counts_selected_blocks_and_window_without_extra_sinks(count, budget):
    config = AttentionConfig(kind="nsa", select_block_count=count, select_block_size=64, window_size=512)
    assert config.key_budget == budget
