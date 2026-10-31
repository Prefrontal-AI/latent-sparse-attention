"""PISA's block-score kernel against the torch scores it replaces, and the search it runs against the reference.

The kernel reads a leaf block's keys where they sit in the pool, so the leaf cases scatter positions over slots.
"""

import pytest
import torch

from lsa.baselines import pisa as reference
from lsa.core.kernels.triton import block_scores, descend
from lsa.sglang.index import pisa
from lsa.sglang.settings.design import PisaConfig

pytestmark = pytest.mark.gpu

LOGIT_SCALE = 0.05


def torch_scores(queries, children, valid):
    return pisa.score_blocks(queries.float().cpu(), children.float().cpu(), valid.cpu(), LOGIT_SCALE)


@pytest.mark.parametrize(
    ("head_count", "key_dim", "block_size"), [(64, 512, 64), (6, 256, 64), (8, 512, 64), (6, 256, 8)]
)
def test_leaf_scores_read_from_the_pool_match_the_gathered_torch_scores(head_count, key_dim, block_size):
    generator = torch.Generator().manual_seed(head_count)
    context, slot_count, rows, width = 2048, 5000, 40, 37
    table = torch.randperm(slot_count, generator=generator)[: 2 * context].view(2, context).int().cuda()
    pool = torch.randn(slot_count, 3, key_dim + 16, generator=generator).bfloat16().cuda()
    keys = pool[:, 1, :key_dim]
    queries = torch.randn(rows, head_count, key_dim, generator=generator).bfloat16().cuda()
    requests = torch.randint(0, 2, (rows,), generator=generator).cuda()
    row_index = torch.tensor([1, 0]).cuda()
    positions = torch.randint(0, context, (rows,), generator=generator).cuda()
    # Candidates up to the query's block and a few past it, whose keys are all masked
    candidates = (
        torch.rand(rows, width, generator=generator).cuda() * (positions[:, None] // block_size + 3)
    ).long()
    candidates = candidates.clamp(max=context // block_size - 1)

    found = block_scores.block_scores(
        queries, candidates, requests, positions, keys, block_size, LOGIT_SCALE, table, row_index
    )

    children = candidates[:, :, None] * block_size + torch.arange(block_size).cuda()
    valid = children <= positions[:, None, None]
    slots = table[row_index[requests][:, None, None], children.minimum(positions[:, None, None])].long()
    expected = torch_scores(queries, keys[slots], valid)
    assert torch.equal(found.isinf().cpu(), expected.isinf())
    finite = ~expected.isinf()
    torch.testing.assert_close(found.cpu()[finite], expected[finite], rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize("head_count", [64, 6])
def test_upper_scores_of_fp32_summaries_match_the_torch_scores(head_count):
    generator = torch.Generator().manual_seed(7)
    batch_size, block_count, key_dim, rows, width = 3, 301, 512, 25, 70
    summaries = torch.randn(batch_size, block_count, key_dim, generator=generator).cuda()
    queries = torch.randn(rows, head_count, key_dim, generator=generator).bfloat16().cuda()
    requests = torch.randint(0, batch_size, (rows,), generator=generator).cuda()
    candidates = torch.randint(0, (block_count + 1) // 2, (rows, width), generator=generator).cuda()
    positions = torch.zeros(rows, dtype=torch.long).cuda()

    found = block_scores.block_scores(queries, candidates, requests, positions, summaries, 2, LOGIT_SCALE)

    children = candidates[:, :, None] * 2 + torch.arange(2).cuda()
    valid = children < block_count
    expected = torch_scores(
        queries, summaries[requests[:, None, None], children.clamp(max=block_count - 1)], valid
    )
    torch.testing.assert_close(found.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_upper_scores_of_split_summaries_match_the_fp32_dot_to_its_rounding():
    generator = torch.Generator().manual_seed(8)
    rows, width, block_count, key_dim = 30, 64, 129, 512
    # Summaries with every fp32 mantissa bit set somewhere, and magnitudes far apart
    summaries = torch.randn(1, block_count, key_dim, generator=generator) * torch.logspace(-3, 3, key_dim)
    queries = torch.randn(rows, 64, key_dim, generator=generator).bfloat16().cuda()
    candidates = torch.randint(0, block_count // 2, (rows, width), generator=generator).cuda()
    requests = torch.zeros(rows, dtype=torch.long).cuda()
    positions = torch.zeros(rows, dtype=torch.long).cuda()

    split = block_scores.block_scores(
        queries, candidates, requests, positions, summaries.cuda(), 2, LOGIT_SCALE
    )
    children = summaries.double()[0, candidates.cpu()[:, :, None] * 2 + torch.arange(2)]
    logits = torch.einsum("rhd,rbcd->rhbc", queries.cpu().double(), children) * LOGIT_SCALE
    exact = logits.logsumexp(-1).sum(1)
    torch.testing.assert_close(split.cpu().double(), exact, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    ("config", "key_count", "positions"),
    [
        (PisaConfig(), 20_000, [19_999, 12_345, 4_700, 4_607, 4_608, 70]),
        (PisaConfig(block_size=4, keep_count=8, branching=2, beam_width=6), 900, list(range(850, 900))),
    ],
)
def test_a_search_scored_by_the_kernel_selects_what_the_reference_selects(config, key_count, positions):
    generator = torch.Generator().manual_seed(key_count)
    keys = torch.randn(key_count, 32, generator=generator)
    queries = torch.randn(len(positions), 4, 32, generator=generator)
    positions = torch.tensor(positions)
    expected = reference.select(
        keys[:, None],
        queries,
        positions,
        config.block_size,
        config.keep_count,
        config.branching,
        config.beam_width,
        LOGIT_SCALE,
        config.forced_blocks,
    ).selected_key_ids[:, 0]
    means = reference.build_pyramid(keys[:, None], config.block_size, config.branching)[1][:, 0]
    levels = pisa.build_levels(means[None].cuda(), config.branching)
    leaf_keys = pisa.LeafKeys(
        torch.arange(key_count, dtype=torch.int32).cuda()[None],
        torch.zeros(1, dtype=torch.long).cuda(),
        keys.cuda(),
    )
    found = pisa.select(
        levels,
        means.shape[0],
        leaf_keys,
        queries.cuda(),
        torch.zeros(len(positions), dtype=torch.long).cuda(),
        positions.cuda(),
        config,
        LOGIT_SCALE,
    )
    assert [set(row[row >= 0].tolist()) for row in found.cpu()] == [
        set(row[row >= 0].tolist()) for row in expected
    ]


@pytest.mark.parametrize(("dtype", "nudge"), [(torch.bfloat16, 0.0), (torch.float32, 2**-22)])
def test_blocks_an_ulp_apart_at_the_keep_are_kept_within_the_tie_the_reference_reports(dtype, nudge):
    config = PisaConfig(block_size=64, keep_count=8, branching=2, beam_width=8)
    generator = torch.Generator().manual_seed(11)
    copies, key_dim, head_count = 300, 64, 4
    key_count = 64 * copies
    block = torch.randn(64, key_dim, generator=generator)
    # Copies of one block, each scaled by 1 or by an fp32 ulp or two more - ties or scores an ulp apart, at every keep
    scales = 1 + nudge * torch.randint(0, 3, (copies, 1, 1), generator=generator).float()
    keys = (block[None] * scales).reshape(key_count, key_dim)
    table = torch.arange(key_count, dtype=torch.int32).cuda()[None]
    buffer = keys.to(dtype).cuda()[:, None]
    index = pisa.PisaIndex(
        config, key_dim, 1, key_count, lambda: table, lambda: buffer, "cuda", mean_dtype=dtype
    )
    positions = torch.tensor([key_count - 1, key_count - 40, 12_345]).cuda()
    queries = torch.randn(3, head_count, key_dim, generator=generator).to(dtype).cuda()
    index.maintain(["r"], [0], [key_count - 64])
    served = index.select(queries, positions, [0], [key_count], [3], LOGIT_SCALE)
    checked = index.check(queries, positions, served, 0, key_count, LOGIT_SCALE)
    assert all(row.served == row.wanted for row in checked)
    assert all(row.verdict() in {"exact", "near-tie"} for row in checked), checked
    # The keep boundary is a tie or within an ulp of one at some level of every row
    assert all(row.gap < pisa.NEAR_TIE_GAP for row in checked), checked


@pytest.mark.parametrize(
    ("branching", "dtype"), [(2, torch.float32), (2, torch.bfloat16), (3, torch.float32)]
)
def test_the_pyramid_kept_up_on_the_gpu_is_the_one_pooled_from_scratch(branching, dtype):
    generator = torch.Generator().manual_seed(branching)
    block_size, context, key_dim = 4, 4 * 1000 + 3, 96
    keys = torch.randn(3, context, key_dim, generator=generator).to(dtype).cuda()
    means = pisa.LeafMeans(3, context, key_dim, block_size, branching, "cuda", dtype)
    # Closed in uneven steps, a row reopened partway, and a row closed to the end
    for row, key_count in [(1, 37), (1, 1200), (1, 1201), (2, context), (1, 2900)]:
        means.advance(row, key_count, lambda positions, row=row: keys[row, positions])
    means.reopen(1, 450)
    keys[1, 450:] += 1
    means.advance(1, 3333, lambda positions: keys[1, positions])
    for row, closed in ((1, 3333 // block_size), (2, context // block_size)):
        built = pisa.build_levels(means.means[row : row + 1, :closed], branching)
        for level, (kept, scratch) in enumerate(zip(means.levels, built, strict=False)):
            # Blocks wholly over closed leaves - the last of a narrower pyramid averages fewer
            whole = closed // branching**level
            assert torch.equal(kept[row, :whole].float(), scratch[0, :whole]), level


def torch_keep(scores, candidates, valid, positions, keep, blocks_below, block_size, branching):
    eligible = valid & (candidates * block_size * blocks_below <= positions[:, None])
    priority = scores.masked_fill(~eligible, -torch.inf)
    forced = pisa.forced_blocks(candidates, positions // block_size, blocks_below)
    priority = priority.masked_fill(eligible & forced, torch.inf)
    top = priority.topk(min(keep, candidates.shape[1]), dim=1)
    kept = candidates.gather(1, top.indices)
    return [set(row[ok].tolist()) for row, ok in zip(kept, top.values > -torch.inf, strict=True)]


@pytest.mark.parametrize(
    ("width", "keep", "blocks_below", "block_size"),
    [
        (1, 72, 512, 64),
        (144, 72, 1, 64),
        (144, 72, 4, 64),
        (37, 5, 2, 64),
        (1152, 576, 1, 8),
        (1152, 576, 4, 8),
    ],
)
def test_a_level_kept_in_one_launch_keeps_what_the_torch_keep_does(width, keep, blocks_below, block_size):
    generator = torch.Generator().manual_seed(width + keep)
    rows, branching = 300, 2
    # Far enough in that every candidate is a distinct block, the first among them
    start = block_size * blocks_below * 4 * width
    positions = torch.randint(start, start + 64 * 4096, (rows,), generator=generator)
    current = positions // block_size // blocks_below
    offsets = torch.stack([torch.randperm(4 * width, generator=generator)[:width] for _ in range(rows)])
    candidates = current[:, None] - 3 * width + offsets
    candidates[:, 0] = 0
    scores = torch.randn(rows, candidates.shape[1], generator=generator)
    valid = torch.rand(rows, candidates.shape[1], generator=generator) > 0.1
    expected = torch_keep(scores, candidates, valid, positions, keep, blocks_below, block_size, branching)

    kept, kept_valid = descend.keep(
        scores.cuda(),
        candidates.cuda(),
        valid.cuda(),
        positions.cuda(),
        keep,
        blocks_below,
        block_size,
        branching,
        True,
    )
    kept, kept_valid = kept.cpu(), kept_valid.cpu()
    if blocks_below == 1:
        found = [set(row[ok].tolist()) for row, ok in zip(kept, kept_valid, strict=True)]
        ascending = [row[ok].tolist() for row, ok in zip(kept, kept_valid, strict=True)]
        assert all(blocks == sorted(blocks) for blocks in ascending)
        assert all(not ok[len(row) :].any() for row, ok in zip(ascending, kept_valid, strict=True))
    else:
        found = [set((row[ok] // branching).tolist()) for row, ok in zip(kept, kept_valid, strict=True)]
        assert torch.equal(kept % branching, torch.arange(kept.shape[1]) % branching * torch.ones_like(kept))
    assert found == expected
