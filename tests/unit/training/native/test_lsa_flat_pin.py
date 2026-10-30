"""The flat LSA arm - no indexer - answers as it did before an indexer could be asked for.

The golden file holds the read, the output and the gradients of the arm on small seeded inputs, taken from the code as it
was when `index_input` did not exist.
Every tensor must match the original read bitwise in the same process.
The saved cross-machine fixture retains a tolerance for the CPU's summation order.
"""

from pathlib import Path

import pytest
import torch

from lsa.training.native.config import AttentionConfig
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.lsa import index_vectors, read_keys
from lsa.training.native.lsa_indexer import LsaInnerAttention

GOLDEN = Path(__file__).parent / "data" / "lsa_flat_pin.pt"
LENGTH = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
ROTARY_DIM = 2
CASES = {
    "narrow": {"top_k": 8, "top_p": 2},
    "wide": {"top_k": 24},
}


def small(**changes):
    knobs = {
        "kind": "lsa",
        "coarse": False,
        "window_size": 8,
        "sink_size": 2,
        "top_k": 24,
        "bucket_size": 4,
        "pool_size": 2,
        "bucket_cap": 8,
        "block_size": 16,
    }
    return AttentionConfig(**{**knobs, **changes})


def compute(name: str, legacy: bool = False) -> dict[str, torch.Tensor]:
    """Run the arm on seeded inputs and return what it read, produced and handed back."""
    config = small(**CASES[name])
    generator = torch.Generator().manual_seed(11)
    q = torch.randn(2 * LENGTH, HEADS, HEAD_DIM, generator=generator).requires_grad_()
    k = torch.randn(2 * LENGTH, KV_HEADS, HEAD_DIM, generator=generator).requires_grad_()
    v = torch.randn(2 * LENGTH, KV_HEADS, HEAD_DIM, generator=generator).requires_grad_()
    weights = torch.randn(2 * LENGTH, HEADS, HEAD_DIM, generator=generator)
    module_type = MultiresInnerAttention if legacy else LsaInnerAttention
    module = module_type.Config(attention=config, seq_len=LENGTH, rotary_dim=ROTARY_DIM).build()
    if legacy:
        index_queries, index_keys = index_vectors(
            q.reshape(2, LENGTH, HEADS, HEAD_DIM).transpose(1, 2),
            k.reshape(2, LENGTH, KV_HEADS, HEAD_DIM).transpose(1, 2),
            ROTARY_DIM,
        )
    else:
        index_queries, index_keys = module._index_vectors(q, k, None, None)
    out = module(q, k, v, scale=HEAD_DIM**-0.5)
    (out * weights).sum().backward()
    return {
        "index_queries": index_queries,
        "index_keys": index_keys,
        "read": read_keys(index_queries, index_keys, config),
        "out": out.detach(),
        "dq": q.grad,
        "dk": k.grad,
        "dv": v.grad,
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_flat_arm_reads_and_answers_as_it_did(name):
    got = compute(name)
    legacy = compute(name, legacy=True)
    want = torch.load(GOLDEN)[name]
    for field in got:
        assert torch.equal(got[field], legacy[field]), field
    assert torch.equal(got["read"], want["read"])
    for field in ("index_queries", "index_keys", "out", "dq", "dk", "dv"):
        assert torch.allclose(got[field], want[field], rtol=1e-5, atol=1e-6), field
