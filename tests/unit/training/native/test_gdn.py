"""Hold GPT-2 convolution precision and document boundaries to mathematical contracts."""

import pytest
import torch
from torch.nn import functional as functional
from torchtitan.models.qwen3_5.gdn import InnerGatedDeltaNet, RMSNormGated

from lsa.training.native.gdn import Gpt2InnerGatedDeltaNet, Gpt2RMSNormGated, convolve_tokens
from tests.unit.training.native.test_initialization import configuration


def convolution_reference(tokens, weight, offsets):
    values = []
    float_tokens = tokens.float()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        for position in range(start, end):
            terms = [
                float_tokens[position - lag] * weight[:, 0, -lag - 1]
                for lag in range(min(weight.shape[-1], position - start + 1))
            ]
            values.append(functional.silu(torch.stack(terms).sum(0)))
    return torch.stack(values).to(tokens.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_convolution_matches_causal_polynomial_and_its_gradients(dtype):
    torch.manual_seed(31)
    tokens = torch.randn(11, 3).to(dtype).requires_grad_(True)
    weight = torch.randn(3, 1, 4).requires_grad_(True)
    offsets = torch.tensor([0, 3, 11], dtype=torch.int32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = convolve_tokens(tokens, weight, offsets)
    reference = convolution_reference(tokens, weight, offsets.tolist())
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (tokens, weight), cotangent, retain_graph=True)
    expected_gradients = torch.autograd.grad(reference, (tokens, weight), cotangent)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-6, atol=2e-6)


def test_future_tokens_and_the_previous_document_cannot_affect_a_convolution():
    tokens = torch.randn(10, 2, requires_grad=True)
    weight = torch.randn(2, 1, 4)
    offsets = torch.tensor([0, 5, 10], dtype=torch.int32)
    output = convolve_tokens(tokens, weight, offsets)
    derivative = torch.autograd.grad(output[6].sum(), tokens)[0]
    assert torch.equal(derivative[:5], torch.zeros_like(derivative[:5]))
    assert torch.equal(derivative[7:], torch.zeros_like(derivative[7:]))
    changed = tokens.detach().clone()
    changed[:5] += 100
    changed[7:] -= 100
    torch.testing.assert_close(convolve_tokens(changed, weight, offsets)[5:7], output[5:7])


def test_bf16_convolution_keeps_the_fp32_weight_and_rounds_after_activation():
    torch.manual_seed(31)
    tokens = torch.randn(17, 3).bfloat16()
    weight = torch.randn(3, 1, 4, requires_grad=True)
    expected = convolution_reference(tokens, weight, [0, 17])
    rounded_weight = convolution_reference(tokens, weight.bfloat16().float(), [0, 17])
    actual = convolve_tokens(tokens, weight, torch.tensor([0, 17], dtype=torch.int32))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, rounded_weight)
    actual.sum().backward()
    assert weight.grad.dtype == torch.float32 and torch.isfinite(weight.grad).all()


def test_gated_norm_matches_a_high_precision_reference_before_the_single_output_rounding():
    torch.manual_seed(17)
    values = torch.randn(7, 3, 128, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(values, requires_grad=True)
    norm = Gpt2RMSNormGated.Config(dim=128).build()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.7, 1.3, 128))
    reference = values.double() * torch.rsqrt(values.double().square().mean(-1, keepdim=True) + norm.eps)
    reference = (reference * norm.weight.double() * functional.silu(gate.double())).bfloat16()
    actual = norm(values, gate)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    old = RMSNormGated.Config(dim=128).build()
    with torch.no_grad():
        old.weight.copy_(norm.weight)
    assert not torch.equal(actual, old(values, gate))
    actual.square().float().mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in (values, gate, norm.weight))


def test_only_the_gpt2_hybrid_replaces_the_inner_computation_and_norm():
    gpt2 = [layer.delta_net for layer in configuration("0.8B-gpt2").layers if layer.delta_net]
    published = [layer.delta_net for layer in configuration("0.8B").layers if layer.delta_net]
    assert all(type(layer.inner_gated_delta_net) is Gpt2InnerGatedDeltaNet.Config for layer in gpt2)
    assert all(type(layer.norm) is Gpt2RMSNormGated.Config for layer in gpt2)
    assert all(type(layer.inner_gated_delta_net) is InnerGatedDeltaNet.Config for layer in published)
    assert all(type(layer.norm) is RMSNormGated.Config for layer in published)
