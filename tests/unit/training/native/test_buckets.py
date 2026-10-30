import pytest
import torch

from lsa.training.native.buckets import (
    bucket_statistics,
    closed_block_count,
    partition_keys,
    select_buckets,
)
from lsa.training.native.config import AttentionConfig

HEAD_DIM = 8
SCALE = HEAD_DIM**-0.5


def small(**changes):
    knobs = {
        "window_size": 8,
        "sink_size": 2,
        "top_k": 24,
        "bucket_size": 4,
        "pool_size": 2,
        "bucket_cap": 8,
        "block_size": 16,
    }
    return AttentionConfig(**{**knobs, **changes})


def keys(streams=2, length=64, seed=0):
    return torch.randn(streams, length, HEAD_DIM, generator=torch.Generator().manual_seed(seed))


def test_the_last_block_is_never_closed_for_anyone():
    assert closed_block_count(64, 16) == 3
    assert closed_block_count(65, 16) == 4
    assert closed_block_count(16, 16) == 0


@pytest.mark.parametrize("partition", ["ward", "chunks", "random"])
def test_every_closed_key_is_in_exactly_one_bucket_of_its_block(partition):
    config = small(partition=partition)
    table = partition_keys(keys(), config)
    assert table.bucket_count == 3 * 4
    covered = table.bucket_of_key[:, : table.covered_count]
    assert (covered // config.buckets_per_block == torch.arange(48) // 16).all()
    assert (table.bucket_of_key[:, 48:] == table.bucket_count).all()
    assert table.bucket_sizes.sum().item() == 2 * 48


def test_ward_buckets_hold_the_cap_and_chunks_hold_consecutive_keys():
    ward = partition_keys(keys(), small())
    assert ward.bucket_sizes.max() <= 8
    chunks = partition_keys(keys(), small(partition="chunks"))
    assert (chunks.bucket_sizes == 4).all()
    assert (chunks.last_position - chunks.first_position == 3).all()


def test_a_random_partition_is_the_same_every_time_and_not_the_chunks():
    first = partition_keys(keys(), small(partition="random"))
    second = partition_keys(keys(), small(partition="random"))
    assert (first.bucket_of_key == second.bucket_of_key).all()
    assert (first.bucket_sizes == 4).all()
    assert (first.last_position - first.first_position > 3).any()


def test_ward_puts_near_duplicate_keys_in_one_bucket():
    base = torch.randn(1, 4, HEAD_DIM)
    block = (base[:, torch.arange(16) % 4] + 1e-3 * torch.randn(1, 16, HEAD_DIM)).repeat(1, 2, 1)
    table = partition_keys(block, small(bucket_size=4, pool_size=1, bucket_cap=8, block_size=16))
    first_block = table.bucket_of_key[0, :16]
    for kind in range(4):
        assert first_block[torch.arange(16) % 4 == kind].unique().numel() == 1


def test_statistics_are_the_plain_mean_and_variance_of_each_buckets_keys():
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, small())
    statistics = bucket_statistics(k, v, table)
    members = (table.bucket_of_key[1] == 5).nonzero().flatten()
    assert torch.allclose(statistics.key_means[1, 5], k[1, members].mean(0), atol=1e-6)
    assert torch.allclose(statistics.value_means[1, 5], v[1, members].mean(0), atol=1e-6)
    expected = (k[1, members] - k[1, members].mean(0)).square().mean()
    assert torch.allclose(statistics.variances[1, 5], expected, atol=1e-6)


def test_a_singleton_bucket_has_no_variance():
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, small(bucket_size=1, pool_size=1, bucket_cap=2, partition="chunks"))
    assert bucket_statistics(k, v, table).variances.abs().max() == 0


def test_statistics_carry_the_gradient_to_keys_and_values():
    k, v = keys().requires_grad_(), keys(seed=1).requires_grad_()
    table = partition_keys(k, small())
    statistics = bucket_statistics(k, v, table)
    (statistics.key_means.sum() + statistics.variances.sum() + statistics.value_means.sum()).backward()
    assert k.grad[:, :48].abs().sum() > 0
    assert k.grad[:, 48:].abs().sum() == 0
    assert v.grad[:, :48].abs().sum() > 0


def test_the_partition_itself_has_no_gradient_path():
    k = keys().requires_grad_()
    assert not partition_keys(k, small()).bucket_of_key.requires_grad


def queries(streams=2, group=2, length=64):
    generator = torch.Generator().manual_seed(2)
    return torch.randn(streams, group, length, HEAD_DIM, generator=generator)


def test_a_query_selects_no_bucket_of_a_block_that_is_not_closed_before_its_own():
    config = small()
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, config)
    reads = select_buckets(queries(), bucket_statistics(k, v, table), table, config, SCALE)
    block_of_query = torch.arange(64) // 16
    visible = table.block_of_bucket[None, None, :] < block_of_query[None, :, None]
    assert not (reads.read[..., :-1] & ~visible).any()
    assert not reads.read[..., -1].any()
    assert reads.read[:, :16].sum() == 0


def test_the_read_keys_never_pass_the_budget_beyond_the_buckets_the_window_and_sinks_touch():
    config = small(top_k=12)
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, config)
    statistics = bucket_statistics(k, v, table)
    reads = select_buckets(queries(), statistics, table, config, SCALE)
    chosen_only = reads.read[..., :-1].clone()
    positions = torch.arange(64)
    closed = table.block_of_bucket[None, :] < (positions // 16)[:, None]
    pinned = closed[None] & (
        (table.first_position < 2)[:, None, :]
        | (table.last_position[:, None, :] > (positions - 8)[None, :, None])
    )
    selected_keys = ((chosen_only & ~pinned) * table.bucket_sizes[:, None, :]).sum(-1)
    assert selected_keys.max() <= 12


def test_every_bucket_is_read_when_the_budget_allows_it():
    config = small(top_k=10**6, top_p=10**6)
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, config)
    reads = select_buckets(queries(), bucket_statistics(k, v, table), table, config, SCALE)
    closed = table.block_of_bucket[None, :] < (torch.arange(64) // 16)[:, None]
    assert (reads.read[..., :-1] == closed[None]).all()


def test_nothing_is_selected_without_a_budget_but_the_pinned_buckets_are_still_read():
    config = small(top_k=0)
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, config)
    reads = select_buckets(queries(), bucket_statistics(k, v, table), table, config, SCALE)
    # A query just past a block boundary has the previous block in its window
    assert reads.read[:, 16, :4].any()
    assert reads.read[:, 63, 4:8].sum() <= reads.read[:, 63, :].sum()
    assert reads.key_counts.max() < 2 * 16 + 4


@pytest.mark.parametrize("rank", ["mass", "error", "cosine"])
def test_every_rank_picks_something_for_a_query_with_candidates(rank):
    config = small(rank=rank)
    k, v = keys(), keys(seed=1)
    table = partition_keys(k, config)
    reads = select_buckets(queries(), bucket_statistics(k, v, table), table, config, SCALE)
    assert reads.key_counts[:, 48:].min() > 0
