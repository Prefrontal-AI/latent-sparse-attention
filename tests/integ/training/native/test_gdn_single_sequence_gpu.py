"""A lone sequence through GatedDeltaNet's kernel is finite and the same as that sequence inside a batch of two.

attn_gym's chunked kernel takes a dense path when it is given no document offsets, which torchtitan does for one
sequence a microbatch. That path is non-finite from 256 tokens on, whatever the gate, so the kernel the model builds
hands it the offsets that span the sequence.
"""

import pytest
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from lsa.training.native.models import OffsetGatedDeltaKernel

pytestmark = pytest.mark.gpu

HEADS = 16
HEAD_DIM = 128


def packed_inputs(tokens: int, sequences: int):
    generator = torch.Generator("cuda").manual_seed(0)
    total = sequences * tokens
    q, k, v = (
        torch.randn(total, HEADS, HEAD_DIM, device="cuda", generator=generator).bfloat16() for _ in range(3)
    )
    q, k = (F.normalize(x.float(), dim=-1).bfloat16() for x in (q, k))
    gate = -torch.rand(total, HEADS, device="cuda", generator=generator) * 0.1
    beta = torch.rand(total, HEADS, device="cuda", generator=generator).bfloat16()
    return q, k, v, gate, beta


@pytest.mark.parametrize("tokens", [256, 4096])
def test_a_lone_sequence_is_finite_and_equals_the_same_sequence_in_a_batch_of_two(tokens):
    q, k, v, gate, beta = packed_inputs(tokens, 1)
    kernel = OffsetGatedDeltaKernel(OffsetGatedDeltaKernel.Config())
    alone = kernel(q, k, v, gate, beta)

    twice = [torch.cat([x, x]) for x in (q, k, v, gate, beta)]
    offsets = torch.tensor([0, tokens, 2 * tokens], dtype=torch.int32, device="cuda")
    batched = kernel(*twice, cu_seqlens=offsets)

    assert torch.isfinite(alone).all()
    torch.testing.assert_close(alone, batched[:tokens], rtol=0, atol=0)
    torch.testing.assert_close(alone, batched[tokens:], rtol=0, atol=0)


def test_a_lone_sequence_matches_fla_called_as_the_reference_training_code_calls_it():
    tokens = 4096
    q, k, v, gate, beta = packed_inputs(tokens, 1)
    ours = OffsetGatedDeltaKernel(OffsetGatedDeltaKernel.Config())(q.clone(), k.clone(), v, gate, beta)
    expected, _ = chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=gate[None], beta=beta[None], scale=HEAD_DIM**-0.5
    )
    torch.testing.assert_close(ours.float(), expected[0].float(), rtol=2e-2, atol=2e-2)
