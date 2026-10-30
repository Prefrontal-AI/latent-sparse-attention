"""The batched PISA search against the reference - the same leaf blocks for every row, in every shape a served chunk takes.

Random keys and queries have no tied scores, so the two searches agree on the set of selected keys of every row.
"""

from types import SimpleNamespace

import pytest
import torch

from lsa.baselines import pisa as reference
from lsa.sglang.index import pisa
from lsa.sglang.settings.design import PisaConfig

HEAD_DIM = 8
HEAD_COUNT = 4
LOGIT_SCALE = 0.7


def reference_ids(keys, queries, positions, config, logit_scale=LOGIT_SCALE):
    selection = reference.select(
        keys[:, None],
        queries,
        positions,
        config.block_size,
        config.keep_count,
        config.branching,
        config.beam_width,
        logit_scale,
        config.forced_blocks,
    )
    return selection.selected_key_ids[:, 0]


def direct_keys(keys):
    """One request whose positions are its slots"""
    table = torch.arange(keys.shape[0], dtype=torch.int32, device=keys.device)[None]
    return pisa.LeafKeys(table, torch.zeros(1, dtype=torch.long, device=keys.device), keys)


def batched_ids(keys, queries, positions, config, logit_scale=LOGIT_SCALE):
    means = reference.build_pyramid(keys[:, None], config.block_size, config.branching)[1][:, 0]
    levels = pisa.build_levels(means[None], config.branching)
    requests = torch.zeros(positions.shape[0], dtype=torch.long)
    return pisa.select(
        levels, means.shape[0], direct_keys(keys), queries, requests, positions, config, logit_scale
    )


def real_sets(ids):
    return [set(row[row >= 0].tolist()) for row in ids]


def assert_same_selection(keys, queries, positions, config):
    expected = reference_ids(keys, queries, positions, config)
    found = batched_ids(keys, queries, positions, config)
    assert found.dtype == torch.int32
    assert found.shape == expected.shape == (positions.shape[0], config.width)
    assert real_sets(found) == real_sets(expected)
    for row, position in zip(found, positions.tolist(), strict=True):
        real = row[row >= 0].tolist()
        # Ascending, valid first, then -1, and no key after the query
        assert real == sorted(real)
        assert row[len(real) :].tolist() == [-1] * (config.width - len(real))
        assert not real or max(real) <= position


def random_case(key_count, query_count=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(key_count, HEAD_DIM, generator=generator)
    query_count = key_count if query_count is None else query_count
    queries = torch.randn(query_count, HEAD_COUNT, HEAD_DIM, generator=generator)
    return keys, queries


@pytest.mark.parametrize("key_count", [1, 3, 4, 5, 17, 64, 100, 151])
@pytest.mark.parametrize(
    "config",
    [
        PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5),
        PisaConfig(block_size=4, keep_count=5, branching=3),
        PisaConfig(block_size=2, keep_count=4, branching=2, beam_width=3),
        PisaConfig(block_size=4, keep_count=2, branching=2, beam_width=1, forced_blocks=False),
    ],
    ids=["beam5", "g3", "block2", "unforced"],
)
def test_every_position_of_a_prompt_selects_what_the_reference_selects(key_count, config):
    keys, queries = random_case(key_count, seed=key_count)
    assert_same_selection(keys, queries, torch.arange(key_count), config)


def test_rows_at_scattered_positions_of_a_long_prompt_select_what_the_reference_selects():
    config = PisaConfig(block_size=4, keep_count=8, branching=2, beam_width=6)
    keys, queries = random_case(900, 40, seed=3)
    positions = torch.randint(0, 900, (40,), generator=torch.Generator().manual_seed(4))
    assert_same_selection(keys, queries, positions, config)


def test_the_paper_defaults_select_what_the_reference_selects_at_a_depth_the_beam_has_to_prune():
    config = PisaConfig()
    keys, queries = random_case(20_000, 6, seed=5)
    positions = torch.tensor([19_999, 12_345, 4_700, 4_607, 4_608, 70])
    assert_same_selection(keys, queries, positions, config)


def test_a_ragged_last_block_and_a_query_inside_it_select_what_the_reference_selects():
    config = PisaConfig(block_size=4, keep_count=5, branching=2)
    keys, queries = random_case(203, 6, seed=6)
    assert_same_selection(keys, queries, torch.tensor([202, 201, 200, 199, 198, 100]), config)


def test_a_tile_of_one_row_selects_what_a_whole_chunk_does(monkeypatch):
    config = PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5)
    keys, queries = random_case(120, seed=7)
    positions = torch.arange(120)
    whole = batched_ids(keys, queries, positions, config)
    monkeypatch.setattr(pisa, "TILE_BYTES", 1)
    tiled = batched_ids(keys, queries, positions, config)
    assert torch.equal(whole, tiled)


def test_a_query_matches_its_prefix_whatever_comes_after_it():
    config = PisaConfig(block_size=4, keep_count=5, branching=2)
    keys, queries = random_case(150, 1, seed=8)
    short = batched_ids(keys[:101], queries, torch.tensor([100]), config)
    long = batched_ids(keys, queries, torch.tensor([100]), config)
    assert torch.equal(short, long)


def test_the_pyramid_of_a_ragged_prompt_averages_the_children_it_has():
    keys, _ = random_case(37)
    means = reference.build_pyramid(keys[:, None], 4, 2)
    levels = pisa.build_levels(means[1][None, :, 0], 2)
    assert len(levels) == len(means) - 1
    for built, expected in zip(levels, means[1:], strict=True):
        torch.testing.assert_close(built[0], expected[:, 0])


def test_the_means_of_whole_blocks_pool_their_keys():
    keys, _ = random_case(12)
    torch.testing.assert_close(pisa.pool_means(keys, 4), keys.view(3, 4, -1).mean(1))


def test_an_empty_batch_selects_nothing():
    config = PisaConfig(block_size=4, keep_count=5)
    keys, queries = random_case(10, 0)
    assert batched_ids(keys, queries, torch.zeros(0, dtype=torch.long), config).shape == (0, config.width)


class FakePool:
    """A KV pool and request table with positions scattered over slots, so a read that skips the table reads wrong keys"""

    def __init__(self, request_count, context, slot_count, seed=0):
        generator = torch.Generator().manual_seed(seed)
        slots = torch.randperm(slot_count, generator=generator)[: request_count * context]
        self.table = slots.view(request_count, context).int()
        self.buffer = torch.zeros(slot_count, 1, HEAD_DIM)

    def write(self, row, start, keys):
        self.buffer[self.table[row, start : start + keys.shape[0]].long(), 0] = keys


def make_index(config, request_count, context, pool):
    return pisa.PisaIndex(
        config,
        HEAD_DIM,
        request_count,
        context,
        lambda: pool.table,
        lambda: pool.buffer,
        "cpu",
        mean_dtype=torch.float32,
    )


def positions_of(pool, row, slots):
    inverse = {int(slot): position for position, slot in enumerate(pool.table[row].tolist())}
    return [{inverse[int(slot)] for slot in line if slot >= 0} for line in slots]


CONFIG = PisaConfig(block_size=4, keep_count=6, branching=2, beam_width=5)


def prefill_in_chunks(index, pool, row, rid, keys, queries, chunk):
    for start in range(0, keys.shape[0], chunk):
        stop = min(start + chunk, keys.shape[0])
        pool.write(row, start, keys[start:stop])
        index.maintain([rid], [row], [start])
        positions = torch.arange(start, stop)
        slots = index.select(queries[start:stop], positions, [row], [stop], [stop - start], LOGIT_SCALE)
        expected = reference_ids(keys[:stop], queries[start:stop], positions, CONFIG)
        assert positions_of(pool, row, slots) == real_sets(expected)


def test_a_chunked_prefill_then_decode_selects_what_the_reference_selects_through_the_pool():
    pool = FakePool(3, 120, 400)
    index = make_index(CONFIG, 3, 120, pool)
    keys, queries = random_case(100, seed=9)
    prefill_in_chunks(index, pool, 1, "a", keys[:70], queries[:70], 23)
    # Decode one token at a time from the same request
    for position in range(70, 100):
        pool.write(1, position, keys[position : position + 1])
        index.maintain(["a"], [1], [position])
        slots = index.decode(
            queries[position : position + 1], torch.tensor([position]), torch.tensor([1]), LOGIT_SCALE
        )
        expected = reference_ids(
            keys[: position + 1], queries[position : position + 1], torch.tensor([position]), CONFIG
        )
        assert positions_of(pool, 1, slots) == real_sets(expected)


def test_a_new_request_in_a_row_starts_from_its_own_blocks_and_a_cached_prefix_is_read_from_the_pool():
    pool = FakePool(2, 90, 200)
    index = make_index(CONFIG, 2, 90, pool)
    first, queries = random_case(80, seed=10)
    prefill_in_chunks(index, pool, 0, "old", first, queries, 80)
    second, queries = random_case(60, seed=11)
    # The prefix is in the pool before the request is seen, as a radix cache hit leaves it
    pool.write(0, 0, second[:40])
    index.maintain(["new"], [0], [40])
    pool.write(0, 40, second[40:])
    slots = index.select(queries[40:], torch.arange(40, 60), [0], [60], [20], LOGIT_SCALE)
    expected = reference_ids(second, queries[40:], torch.arange(40, 60), CONFIG)
    assert positions_of(pool, 0, slots) == real_sets(expected)


def test_a_batch_of_requests_of_different_lengths_selects_for_each_what_the_reference_does():
    pool = FakePool(3, 100, 400)
    index = make_index(CONFIG, 3, 100, pool)
    first, first_queries = random_case(90, seed=12)
    second, second_queries = random_case(33, seed=13)
    pool.write(2, 0, first)
    pool.write(0, 0, second)
    index.maintain(["x", "y"], [2, 0], [89, 32])
    queries = torch.cat((first_queries[-1:], second_queries[-1:]))
    slots = index.select(queries, torch.tensor([89, 32]), [2, 0], [90, 33], [1, 1], LOGIT_SCALE)
    assert positions_of(pool, 2, slots[:1]) == real_sets(
        reference_ids(first, first_queries[-1:], torch.tensor([89]), CONFIG)
    )
    assert positions_of(pool, 0, slots[1:]) == real_sets(
        reference_ids(second, second_queries[-1:], torch.tensor([32]), CONFIG)
    )
    assert slots.dtype == torch.int32
    assert slots.shape == (2, CONFIG.width)


def test_a_decode_batch_over_a_reused_row_selects_for_each_what_the_reference_does():
    pool = FakePool(3, 100, 400)
    index = make_index(CONFIG, 3, 100, pool)
    # A longer request leaves its means in row 0 past where the next one's keys end
    old, old_queries = random_case(96, seed=16)
    prefill_in_chunks(index, pool, 0, "old", old, old_queries, 50)
    new, new_queries = random_case(60, seed=17)
    long, long_queries = random_case(95, seed=18)
    pool.write(0, 0, new[:30])
    pool.write(2, 0, long[:80])
    index.maintain(["new", "long"], [0, 2], [0, 0])
    for step in range(15):
        new_position, long_position = 30 + step, 80 + step
        pool.write(0, new_position, new[new_position : new_position + 1])
        pool.write(2, long_position, long[long_position : long_position + 1])
        index.maintain(["new", "long"], [0, 2], [new_position, long_position])
        queries = torch.cat(
            (new_queries[new_position : new_position + 1], long_queries[long_position : long_position + 1])
        )
        slots = index.decode(
            queries, torch.tensor([new_position, long_position]), torch.tensor([0, 2]), LOGIT_SCALE
        )
        assert positions_of(pool, 0, slots[:1]) == real_sets(
            reference_ids(
                new[: new_position + 1],
                new_queries[new_position : new_position + 1],
                torch.tensor([new_position]),
                CONFIG,
            )
        )
        assert positions_of(pool, 2, slots[1:]) == real_sets(
            reference_ids(
                long[: long_position + 1],
                long_queries[long_position : long_position + 1],
                torch.tensor([long_position]),
                CONFIG,
            )
        )


def test_an_index_that_would_score_the_open_block_is_refused():
    with pytest.raises(ValueError, match="forced_blocks"):
        make_index(PisaConfig(keep_count=1, beam_width=1, forced_blocks=False), 1, 10, FakePool(1, 10, 20))


def test_queries_gathered_from_a_group_are_scored_as_one_set_of_heads():
    keys, queries = random_case(60, 6, seed=14)
    positions = torch.arange(54, 60)
    grouped = queries.view(6, 2, 2, HEAD_DIM)
    config = PisaConfig(block_size=4, keep_count=5, branching=2)
    means = reference.build_pyramid(keys[:, None], 4, 2)[1][:, 0]
    levels = pisa.build_levels(means[None], 2)
    requests = torch.zeros(6, dtype=torch.long)
    split = pisa.select(
        levels, means.shape[0], direct_keys(keys), grouped, requests, positions, config, LOGIT_SCALE
    )
    assert real_sets(split) == real_sets(reference_ids(keys, queries, positions, config))


def test_a_pyramid_the_search_is_not_defined_for_is_refused():
    with pytest.raises(ValueError, match="keep_count"):
        PisaConfig(keep_count=2)
    with pytest.raises(ValueError, match="branching"):
        PisaConfig(branching=1)
    with pytest.raises(ValueError, match="block_size"):
        PisaConfig(block_size=0)
    with pytest.raises(ValueError, match="beam_width"):
        PisaConfig(beam_width=2)
    with pytest.raises(ValueError, match="forced_blocks"):
        PisaConfig(forced_blocks=1)
    assert PisaConfig(keep_count=1, beam_width=1, forced_blocks=False).width == 64


def test_the_debugging_check_agrees_with_a_served_list_and_sees_a_wrong_one():
    pool = FakePool(1, 100, 200)
    index = make_index(CONFIG, 1, 100, pool)
    keys, queries = random_case(90, seed=15)
    pool.write(0, 0, keys)
    index.maintain(["a"], [0], [89])
    positions = torch.tensor([89, 50])
    queries = queries[positions]
    # Two rows of one request at its end, the first a decode step's and the second a middle row of a chunk
    first = index.select(queries[:1], positions[:1], [0], [90], [1], LOGIT_SCALE)
    second = index.select(queries[1:], positions[1:], [0], [90], [1], LOGIT_SCALE)
    slots = torch.cat((first, second))
    checked = index.check(queries, positions, slots, 0, 90, LOGIT_SCALE)
    assert [(row.position, row.verdict()) for row in checked] == [(89, "exact"), (50, "exact")]
    # A list that holds a slot the search did not pick is short of the reference's, and the search did not nearly tie
    wrong = slots.clone()
    wrong[0, 0] = pool.table[0, 5]
    found = index.check(queries, positions, wrong, 0, 90, LOGIT_SCALE)[0]
    assert found.both < found.wanted and found.verdict() == "MISMATCH"


def test_the_tally_fails_on_a_mismatch_or_ranks_that_differ_and_passes_a_near_tie():
    tally = pisa.CheckTally()
    assert tally.add(pisa.CheckedRow(7, 64, 64, 64, 0.5)) == "exact"
    assert tally.add(pisa.CheckedRow(8, 60, 64, 64, pisa.NEAR_TIE_GAP / 2)) == "near-tie"
    tally.add_ranks(True)
    assert tally.line().endswith("verdict PASS")
    assert "rows 2 exact 1 near-tie 1 mismatch 0 rank-checks 1 rank-differences 0" in tally.line()
    tally.add_ranks(False)
    assert tally.line().endswith("verdict FAIL")
    clean = pisa.CheckTally()
    assert clean.add(pisa.CheckedRow(9, 60, 64, 64, 0.5)) == "MISMATCH"
    assert clean.line().endswith("verdict FAIL")


def test_the_rows_a_check_holds_spread_over_the_first_requests_that_hold_enough_keys():
    picked = pisa.check_rows_of([10, 1, 5, 3, 2, 4], [100, 20, 105, 50, 60, 70], 3, 50)
    # The second request is short of keys, the fifth past the first four
    assert picked == [0, 4, 9, 11, 13, 15, 16, 17, 18]


def test_a_served_index_keeps_its_leaf_means_in_fp32_as_the_reference_pools_them():
    pool = FakePool(1, 10, 20)
    index = pisa.PisaIndex(CONFIG, HEAD_DIM, 1, 10, lambda: pool.table, lambda: pool.buffer, "cpu")
    assert index.leaf_means.means.dtype == torch.float32


def test_a_prefix_repointed_at_another_requests_slots_reopens_its_means_and_selects_from_the_new_keys():
    pool = FakePool(2, 60, 200, seed=3)
    index = make_index(CONFIG, 2, 60, pool)
    keys, queries = random_case(60, seed=12)
    prefill_in_chunks(index, pool, 0, "a", keys[:40], queries[:40], 40)
    # The prefix cache points positions 8 to 39 at slots another forward wrote for the same tokens
    spare = torch.tensor(sorted(set(range(200)) - set(pool.table.flatten().tolist())))[:32].int()
    other_keys = keys[8:40] + 0.5 * torch.randn(32, HEAD_DIM, generator=torch.Generator().manual_seed(4))
    pool.buffer[spare.long(), 0] = other_keys
    table_pool = SimpleNamespace(req_to_token=pool.table)
    pisa.reopen_repointed(table_pool, (0, slice(8, 40)), spare)
    pool.table[0, 8:40] = spare
    assert index.leaf_means.closed[0] == 2

    repointed = torch.cat((keys[:8], other_keys, keys[40:]))
    pool.write(0, 40, repointed[40:])
    index.maintain(["a"], [0], [40])
    slots = index.select(queries[40:], torch.arange(40, 60), [0], [60], [20], LOGIT_SCALE)
    expected = reference_ids(repointed, queries[40:], torch.arange(40, 60), CONFIG)
    assert positions_of(pool, 0, slots) == real_sets(expected)
    torch.testing.assert_close(index.leaf_means.means[0, :15], pisa.pool_means(repointed, 4), rtol=0, atol=0)


def test_a_table_write_that_keeps_the_slots_or_lies_past_the_closed_blocks_reopens_nothing():
    pool = FakePool(1, 40, 80, seed=5)
    index = make_index(CONFIG, 1, 40, pool)
    keys, queries = random_case(40, seed=13)
    prefill_in_chunks(index, pool, 0, "a", keys[:24], queries[:24], 24)
    table_pool = SimpleNamespace(req_to_token=pool.table)
    pisa.reopen_repointed(table_pool, (0, slice(0, 24)), pool.table[0, :24].clone())
    pisa.reopen_repointed(table_pool, (0, slice(24, 40)), torch.zeros(16, dtype=torch.int32))
    pisa.reopen_repointed(
        table_pool, (torch.tensor([0]), torch.tensor([30])), torch.zeros(1, dtype=torch.int32)
    )
    assert index.leaf_means.closed[0] == 6


def test_the_pyramid_kept_up_as_blocks_close_and_reopen_is_the_one_pooled_from_scratch():
    keys = torch.randn(2, 403, HEAD_DIM, generator=torch.Generator().manual_seed(21))
    means = pisa.LeafMeans(2, 403, HEAD_DIM, 4, 2, "cpu", torch.float32)
    for key_count in (9, 130, 131, 380):
        means.advance(1, key_count, lambda positions: keys[1, positions])
    means.reopen(1, 50)
    keys[1, 50:] += 1
    means.advance(1, 403, lambda positions: keys[1, positions])
    built = pisa.build_levels(means.means[1:2, :100], 2)
    for level, (kept, scratch) in enumerate(zip(means.levels, built, strict=False)):
        whole = 100 // 2**level
        assert torch.equal(kept[1, :whole], scratch[0, :whole]), level
