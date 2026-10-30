"""PISA's selection against brute force - the pyramid, the exact top blocks, the forced blocks, causality, grouped heads and the cost."""

import math

import pytest
import torch

from lsa.baselines import pisa

BLOCK_SIZE = 4
HEAD_DIM = 8


def selected_blocks(selection, row, block_size=BLOCK_SIZE, kv_head=0):
    ids = selection.selected_key_ids[row, kv_head]
    return sorted({int(i) // block_size for i in ids if i >= 0})


def exact_leaf_scores(keys, queries, position, kv_head=0):
    # Summed over the heads of the group => each leaf block's exact LSE over its keys up to the position
    scale = 1.0 / math.sqrt(keys.shape[-1])
    scores = {}
    for block in range(position // BLOCK_SIZE + 1):
        ids = torch.arange(block * BLOCK_SIZE, min((block + 1) * BLOCK_SIZE, position + 1, keys.shape[0]))
        logits = queries @ keys[ids, kv_head].T * scale
        scores[block] = float(logits.logsumexp(-1).sum())
    return scores


def forced_leaves(position):
    current = position // BLOCK_SIZE
    return {0, current - 1, current} - {-1}


def test_keeping_at_least_every_leaf_block_selects_every_key_up_to_the_position():
    torch.manual_seed(0)
    keys = torch.randn(30, 1, HEAD_DIM)
    queries = torch.randn(3, 2, HEAD_DIM)
    positions = torch.tensor([0, 13, 29])
    selection = pisa.select(keys, queries, positions, BLOCK_SIZE, keep_count=8)
    assert selection.scored_counts.sum() == 0
    assert selection.boundary_gaps.isinf().all()
    for row, position in enumerate(positions.tolist()):
        ids = selection.selected_key_ids[row, 0]
        assert ids[ids >= 0].tolist() == list(range(position + 1))


def test_a_beam_as_wide_as_every_level_keeps_the_forced_blocks_and_the_exact_top_leaf_blocks():
    torch.manual_seed(1)
    keys = torch.randn(100, 1, HEAD_DIM)
    queries = torch.randn(4, 2, HEAD_DIM)
    positions = torch.tensor([99, 70, 41, 20])
    keep_count = 6
    selection = pisa.select(keys, queries, positions, BLOCK_SIZE, keep_count, beam_width=100)
    for row, position in enumerate(positions.tolist()):
        scores = exact_leaf_scores(keys, queries[row], position)
        forced = forced_leaves(position)
        rest = sorted((b for b in scores if b not in forced), key=lambda b: -scores[b])
        expected = sorted(forced | set(rest[: keep_count - len(forced)]))
        assert selected_blocks(selection, row) == expected


def test_no_selected_key_comes_after_its_query_and_ids_are_ascending_and_padded_with_minus_one():
    torch.manual_seed(2)
    keys = torch.randn(64, 1, HEAD_DIM)
    queries = torch.randn(10, 2, HEAD_DIM)
    positions = torch.arange(0, 64, 7)[:10]
    selection = pisa.select(keys, queries[: len(positions)], positions, BLOCK_SIZE, keep_count=3)
    assert selection.selected_key_ids.dtype == torch.int32
    assert selection.selected_key_ids.shape == (len(positions), 1, 3 * BLOCK_SIZE)
    for row, position in enumerate(positions.tolist()):
        ids = selection.selected_key_ids[row, 0].tolist()
        real = [i for i in ids if i >= 0]
        assert max(real) <= position
        assert real == sorted(set(real))
        assert ids[len(real) :] == [-1] * (len(ids) - len(real))


def test_future_keys_do_not_change_the_selection_of_a_block_off_the_current_path():
    torch.manual_seed(3)
    keys = torch.randn(64, 1, HEAD_DIM)
    queries = torch.randn(1, 2, HEAD_DIM)
    position = torch.tensor([37])
    changed = keys.clone()
    changed[40:] = torch.randn(24, 1, HEAD_DIM) * 5
    first = pisa.select(keys, queries, position, BLOCK_SIZE, keep_count=4)
    second = pisa.select(changed, queries, position, BLOCK_SIZE, keep_count=4)
    assert torch.equal(first.selected_key_ids, second.selected_key_ids)


@pytest.mark.parametrize("position", [3, 4, 5, 27, 63])
def test_the_first_previous_and_current_blocks_are_kept_even_when_their_scores_are_the_worst(position):
    keys = torch.randn(64, 1, HEAD_DIM)
    # Keys the query dislikes everywhere in the forced blocks, likes elsewhere
    direction = torch.ones(1, HEAD_DIM)
    keys = keys * 0.1 + direction
    current = position // BLOCK_SIZE
    for block in forced_leaves(position):
        keys[block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE] = -direction
    queries = direction[None].repeat(1, 2, 1)
    selection = pisa.select(keys, queries, torch.tensor([position]), BLOCK_SIZE, keep_count=3)
    assert set(forced_leaves(position)) <= set(selected_blocks(selection, 0))
    assert current in selected_blocks(selection, 0)


def test_without_forced_blocks_the_current_block_is_scored_on_its_past_keys_alone():
    keys = torch.zeros(16, 1, HEAD_DIM)
    keys[4:8] = 1.0
    keys[10:12] = 50.0
    queries = torch.ones(1, 1, HEAD_DIM)
    selection = pisa.select(
        keys, queries, torch.tensor([9]), BLOCK_SIZE, keep_count=1, forced_blocks=False, beam_width=16
    )
    # Block 2 holds a huge key at 10, after the query at 9, which must not count
    assert selected_blocks(selection, 0) == [1]


def test_grouped_heads_are_summed_not_averaged_per_head_or_taken_alone():
    torch.manual_seed(4)
    keys = torch.randn(48, 1, HEAD_DIM)
    queries = torch.randn(1, 3, HEAD_DIM)
    position = 47
    scores = exact_leaf_scores(keys, queries[0], position)
    per_head = [exact_leaf_scores(keys, queries[0, h : h + 1], position) for h in range(3)]
    for block in scores:
        assert scores[block] == pytest.approx(sum(p[block] for p in per_head), rel=1e-5)
    selection = pisa.select(keys, queries, torch.tensor([position]), BLOCK_SIZE, keep_count=5, beam_width=48)
    rest = sorted((b for b in scores if b not in forced_leaves(position)), key=lambda b: -scores[b])
    assert selected_blocks(selection, 0) == sorted(forced_leaves(position) | set(rest[:2]))


def test_a_head_swap_within_a_group_changes_nothing_and_a_single_head_group_matches_two_equal_heads_scaled():
    torch.manual_seed(5)
    keys = torch.randn(64, 1, HEAD_DIM)
    queries = torch.randn(2, 2, HEAD_DIM)
    positions = torch.tensor([63, 50])
    first = pisa.select(keys, queries, positions, BLOCK_SIZE, keep_count=5)
    swapped = pisa.select(keys, queries.flip(1), positions, BLOCK_SIZE, keep_count=5)
    assert torch.equal(first.selected_key_ids, swapped.selected_key_ids)
    single = queries[:, :1]
    doubled = single.repeat(1, 2, 1)
    # Two equal heads double every score, and with the default scale that is the same ranking as one head
    assert torch.equal(
        pisa.select(keys, single, positions, BLOCK_SIZE, keep_count=5).selected_key_ids,
        pisa.select(keys, doubled, positions, BLOCK_SIZE, keep_count=5).selected_key_ids,
    )


def test_each_kv_head_gets_its_own_selection_from_its_own_group_of_query_heads():
    torch.manual_seed(6)
    keys = torch.randn(64, 2, HEAD_DIM)
    queries = torch.randn(2, 4, HEAD_DIM)
    positions = torch.tensor([63, 40])
    both = pisa.select(keys, queries, positions, BLOCK_SIZE, keep_count=4)
    for kv_head in range(2):
        alone = pisa.select(
            keys[:, kv_head : kv_head + 1],
            queries[:, 2 * kv_head : 2 * kv_head + 2],
            positions,
            BLOCK_SIZE,
            keep_count=4,
        )
        assert torch.equal(both.selected_key_ids[:, kv_head], alone.selected_key_ids[:, 0])
        assert torch.equal(both.scored_counts[:, kv_head], alone.scored_counts[:, 0])


def test_summaries_are_means_of_the_valid_children_for_a_ragged_last_block():
    torch.manual_seed(7)
    keys = torch.randn(10, 2, HEAD_DIM)
    levels = pisa.build_pyramid(keys, 4, 2)
    assert [level.shape[0] for level in levels] == [10, 3, 2, 1]
    assert torch.allclose(levels[1][0], keys[0:4].mean(0))
    assert torch.allclose(levels[1][2], keys[8:10].mean(0))
    assert torch.allclose(levels[2][0], (levels[1][0] + levels[1][1]) / 2)
    assert torch.allclose(levels[2][1], levels[1][2])
    assert torch.allclose(levels[3][0], (levels[2][0] + levels[2][1]) / 2)


def test_the_blocks_scored_at_each_level_stay_within_branching_times_the_beam():
    torch.manual_seed(8)
    keys = torch.randn(1024, 1, HEAD_DIM)
    queries = torch.randn(5, 2, HEAD_DIM)
    positions = torch.tensor([1023, 900, 600, 300, 100])
    keep_count, beam_width, branching = 4, 6, 2
    selection = pisa.select(keys, queries, positions, BLOCK_SIZE, keep_count, branching, beam_width)
    scored = selection.scored_counts[:, 0]
    assert scored.shape == (5, 9)
    assert scored[:, 0].max() <= branching * beam_width
    assert scored[:, 1:].max() <= branching * beam_width
    assert scored[0].sum() > 0
    # A wider beam only above the leaves still returns the leaf budget
    assert selection.selected_key_ids.shape[-1] == keep_count * BLOCK_SIZE


def test_a_key_budget_converts_to_a_whole_number_of_blocks():
    assert pisa.keep_count_for_budget(4096, 64) == 64
    with pytest.raises(ValueError, match="multiple"):
        pisa.keep_count_for_budget(4000, 64)


def test_bad_input_raises_at_the_boundary():
    keys = torch.randn(16, 1, HEAD_DIM)
    queries = torch.randn(1, 2, HEAD_DIM)
    with pytest.raises(ValueError, match="positions"):
        pisa.select(keys, queries, torch.tensor([16]), BLOCK_SIZE, keep_count=3)
    with pytest.raises(ValueError, match="multiple"):
        pisa.select(
            torch.randn(16, 2, HEAD_DIM), torch.randn(1, 3, HEAD_DIM), torch.tensor([3]), BLOCK_SIZE, 3
        )
    with pytest.raises(ValueError, match="at least"):
        pisa.select(keys, queries, torch.tensor([3]), BLOCK_SIZE, keep_count=2)


def test_a_level_that_drops_blocks_records_how_near_its_keep_came_to_a_tie():
    torch.manual_seed(5)
    keys = torch.randn(200, 1, HEAD_DIM)
    queries = torch.randn(2, 2, HEAD_DIM)
    selection = pisa.select(keys, queries, torch.tensor([199, 150]), BLOCK_SIZE, keep_count=5)
    scored = selection.scored_counts > 0
    gaps = selection.boundary_gaps
    assert scored.any() and (gaps[scored] >= 0).all() and gaps[scored].isfinite().all()
    assert gaps[~scored].isinf().all()
    # Copies of one block score alike, so a keep that splits them sits on a tie
    tied = keys.clone()
    tied[BLOCK_SIZE:190] = keys[BLOCK_SIZE : 2 * BLOCK_SIZE].repeat(190 // BLOCK_SIZE, 1, 1)[
        : 190 - BLOCK_SIZE
    ]
    selection = pisa.select(tied, queries[:1], torch.tensor([199]), BLOCK_SIZE, keep_count=5)
    assert selection.boundary_gaps.min() == 0
