"""The TileLang sparse read on lists cut to the early width, against the same read on the whole lists.

The cut drops only `-1` columns, which the kernel masks, so the two reads must agree bit for bit. Runs on the fork's
image, where `tilelang` and the backend are.
"""

import pytest
import torch

from lsa.constants import Sentinel
from lsa.sglang import plugin

pytestmark = pytest.mark.gpu

WIDTH = 4612
EARLY_WIDTH = 2624
# GLM's latent dim, which the kernel reads with no rope tail, a head count it pads to 16, and the pool's rows
LATENT_DIM = 512
HEADS = 16
POOL_ROWS = 8192
QUERIES = 96


def lists_of(counts: list[int], generator: torch.Generator) -> torch.Tensor:
    """Build lists [len(counts), WIDTH] int32 of pool slots, the first `count` of each row real and `-1` after."""
    lists = torch.full((len(counts), WIDTH), Sentinel.PAD, dtype=torch.int32, device="cuda")
    for i, count in enumerate(counts):
        slots = torch.randperm(POOL_ROWS, device="cuda", generator=generator)[:count]
        lists[i, :count] = slots.to(torch.int32)
    return lists


def read(counts: list[int]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the read of whole lists, the read through the hook, and the width the hook handed the kernel."""
    backend = pytest.importorskip(
        "sglang.srt.layers.attention.dsa_backend", reason="the TileLang read is on the fork's image"
    )
    pytest.importorskip("tilelang", reason="the TileLang read is on the fork's image")
    original = backend.DeepseekSparseAttnBackend._forward_tilelang
    generator = torch.Generator(device="cuda").manual_seed(11)
    q_all = torch.randn(
        len(counts), HEADS, LATENT_DIM, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv_cache = torch.randn(POOL_ROWS, 1, LATENT_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    lists = lists_of(counts, generator)
    widths = []

    def recording(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, return_lse=False):
        widths.append(page_table_1.shape[-1])
        return original(self, q_all, kv_cache, v_head_dim, page_table_1, sm_scale, return_lse)

    whole = original(None, q_all, kv_cache, LATENT_DIM, lists, 0.0625)
    cut = plugin.tilelang(
        recording, None, q_all, kv_cache, LATENT_DIM, lists, 0.0625, early_width=EARLY_WIDTH
    )
    return whole, cut, widths[0]


def test_an_all_early_chunk_reads_the_same_at_the_early_width():
    counts = [2564] * 40 + [1000] * 40 + [5] * 16
    whole, cut, width = read(counts)
    assert width == EARLY_WIDTH
    assert torch.equal(whole, cut)


def test_a_chunk_that_straddles_the_early_position_reads_the_same_at_full_width():
    counts = [2564] * 40 + [1000] * 16 + [WIDTH] * 40
    whole, cut, width = read(counts)
    assert width == WIDTH
    assert torch.equal(whole, cut)
