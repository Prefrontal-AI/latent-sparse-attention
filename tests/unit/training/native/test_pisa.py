import pytest
import torch

from lsa.baselines.pisa import select
from lsa.training.native.config import AttentionConfig
from lsa.training.native.kernels import pisa
from lsa.training.native.pisa import PisaInnerAttention


@pytest.mark.parametrize("length,block,branching", [(97, 4, 2), (81, 8, 3), (3, 4, 2)])
@pytest.mark.parametrize("keep_count", [3, 5])
def test_selection_matches_independent_positional_pyramid_reference(length, block, branching, keep_count):
    torch.manual_seed(11)
    keys = torch.randn(length, 2, 16)
    queries = torch.randn(length, 4, 16)
    got = pisa.select_blocks(queries[None], keys[None], block, branching, keep_count, 0.25)[0]
    expected = select(
        keys, queries, torch.arange(length), block, keep_count, branching=branching, logit_scale=0.25
    ).selected_key_ids
    for position in range(length):
        for head in range(2):
            blocks = got[position, head]
            indices = (blocks[blocks >= 0, None] * block + torch.arange(block)).flatten()
            indices = indices[indices <= position]
            want = expected[position, head]
            assert torch.equal(indices.to(want.dtype), want[want >= 0])


def test_read_is_causal_and_loss_reaches_queries_keys_and_values():
    config = AttentionConfig(kind="pisa", leaf_block_size=4, level_block_count=3)
    module = PisaInnerAttention.Config(attention=config, seq_len=24).build()
    leaves = [torch.randn(48, heads, 16, requires_grad=True) for heads in (4, 2, 2)]
    context = module(*leaves)
    context.square().sum().backward()
    for tensor in leaves:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    changed = [tensor.detach().clone() for tensor in leaves]
    for tensor in changed:
        tensor[12:] += 1
    torch.testing.assert_close(module(*changed)[:12], context[:12])


@pytest.mark.parametrize(
    "changes", [{"leaf_block_size": 0}, {"branching": 1}, {"local_block_count": 1}, {"level_block_count": 2}]
)
def test_invalid_pyramid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        AttentionConfig(kind="pisa", **changes)


@pytest.mark.parametrize("count,budget", [(24, 1536), (40, 2560), (72, 4608), (41, 2624), (73, 4672)])
def test_key_budget_counts_leaf_blocks_without_extra_window_or_sinks(count, budget):
    config = AttentionConfig(kind="pisa", level_block_count=count, leaf_block_size=64)
    assert config.key_budget == budget
