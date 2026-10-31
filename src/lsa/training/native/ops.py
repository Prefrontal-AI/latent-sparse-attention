"""The steps of the reads that have no derivative, as custom ops the compiler leaves alone.

Ward's merge and the selection's top-p decide by the values of the keys, so tracing them hits a data-dependent
guard, and a graph break is no way out when torchtitan compiles each block with `fullgraph`. A custom op is opaque
to the tracer: it sees the shapes the fake kernel promises and the eager kernel runs the real function.
Both steps are `no_grad` already - an assignment and a top-p index have no derivative - so the ops register none, and
the differentiable path between them is traced as usual.
"""

import torch

from lsa.training.native.buckets import (
    BucketStatistics,
    BucketTable,
    Reads,
    closed_block_count,
    partition_keys,
    select_buckets,
)
from lsa.training.native.config import AttentionConfig
from lsa.training.native.lsa import read_keys


@torch.library.custom_op("lsa_native::partition_keys", mutates_args=())
def _partition_keys_op(
    keys: torch.Tensor, encoded_config: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `partition_keys` and return the table's tensors."""
    table = partition_keys(keys, AttentionConfig.decode(encoded_config))
    # Contiguous, because the fake promises that layout and the table's own slices are not
    return (
        table.bucket_of_key.contiguous(),
        table.bucket_sizes.contiguous(),
        table.first_position.contiguous(),
        table.last_position.contiguous(),
        table.block_of_bucket.contiguous(),
    )


@_partition_keys_op.register_fake
def _(
    keys: torch.Tensor, encoded_config: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty tensors shaped as the table's, which `partition_keys` returns."""
    config = AttentionConfig.decode(encoded_config)
    stream_count, key_count, _ = keys.shape
    bucket_count = closed_block_count(key_count, config.block_size) * config.buckets_per_block
    return (
        keys.new_empty(stream_count, key_count, dtype=torch.long),
        keys.new_empty(stream_count, bucket_count, dtype=torch.float32),
        keys.new_empty(stream_count, bucket_count, dtype=torch.long),
        keys.new_empty(stream_count, bucket_count, dtype=torch.long),
        keys.new_empty(bucket_count, dtype=torch.long),
    )


@torch.library.custom_op("lsa_native::select_buckets", mutates_args=())
def _select_buckets_op(
    queries: torch.Tensor,
    key_means: torch.Tensor,
    value_means: torch.Tensor,
    variances: torch.Tensor,
    bucket_of_key: torch.Tensor,
    bucket_sizes: torch.Tensor,
    first_position: torch.Tensor,
    last_position: torch.Tensor,
    block_of_bucket: torch.Tensor,
    encoded_config: str,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `select_buckets` and return the reads' tensors."""
    config = AttentionConfig.decode(encoded_config)
    table = _table(bucket_of_key, bucket_sizes, first_position, last_position, block_of_bucket, config)
    reads = select_buckets(queries, BucketStatistics(key_means, value_means, variances), table, config, scale)
    return reads.read, reads.key_counts


@_select_buckets_op.register_fake
def _(
    queries: torch.Tensor,
    key_means: torch.Tensor,
    value_means: torch.Tensor,
    variances: torch.Tensor,
    bucket_of_key: torch.Tensor,
    bucket_sizes: torch.Tensor,
    first_position: torch.Tensor,
    last_position: torch.Tensor,
    block_of_bucket: torch.Tensor,
    encoded_config: str,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return empty tensors shaped as the reads' tensors, which `select_buckets` returns."""
    stream_count, _, query_count, _ = queries.shape
    return (
        queries.new_empty(stream_count, query_count, bucket_sizes.shape[1] + 1, dtype=torch.bool),
        queries.new_empty(stream_count, query_count, dtype=torch.float32),
    )


def partition_keys_opaque(keys: torch.Tensor, config: AttentionConfig) -> BucketTable:
    """Cut the closed blocks into buckets, as `partition_keys` does, behind a custom op.

    Args:
        keys: Post-RoPE, post-norm keys [S, key_count, head_dim].
        config: The attention's knobs.

    Returns:
        The table, as `partition_keys` returns it.
    """
    bucket_of_key, bucket_sizes, first_position, last_position, block_of_bucket = _partition_keys_op(
        keys.detach(), config.encoded
    )
    return _table(bucket_of_key, bucket_sizes, first_position, last_position, block_of_bucket, config)


def select_buckets_opaque(
    queries: torch.Tensor,
    statistics: BucketStatistics,
    table: BucketTable,
    config: AttentionConfig,
    scale: float,
) -> Reads:
    """Pick each query's reads, as `select_buckets` does, behind a custom op.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        statistics: The buckets' statistics.
        table: The buckets.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The reads, as `select_buckets` returns them.
    """
    read, key_counts = _select_buckets_op(
        queries.detach(),
        statistics.key_means.detach(),
        statistics.value_means.detach(),
        statistics.variances.detach(),
        table.bucket_of_key,
        table.bucket_sizes,
        table.first_position,
        table.last_position,
        table.block_of_bucket,
        config.encoded,
        scale,
    )
    return Reads(read, key_counts)


def _table(
    bucket_of_key: torch.Tensor,
    bucket_sizes: torch.Tensor,
    first_position: torch.Tensor,
    last_position: torch.Tensor,
    block_of_bucket: torch.Tensor,
    config: AttentionConfig,
) -> BucketTable:
    """Put a table's tensors back together with the two counts its shapes give."""
    bucket_count = bucket_sizes.shape[1]
    return BucketTable(
        bucket_of_key=bucket_of_key,
        bucket_sizes=bucket_sizes,
        first_position=first_position,
        last_position=last_position,
        block_of_bucket=block_of_bucket,
        bucket_count=bucket_count,
        covered_count=bucket_count // config.buckets_per_block * config.block_size,
    )


@torch.library.custom_op("lsa_native::lsa_read_keys", mutates_args=())
def _lsa_read_keys_op(
    index_queries: torch.Tensor, index_keys: torch.Tensor, encoded_config: str
) -> torch.Tensor:
    """Run `read_keys` and return which keys each query reads."""
    return read_keys(index_queries, index_keys, AttentionConfig.decode(encoded_config))


@_lsa_read_keys_op.register_fake
def _(index_queries: torch.Tensor, index_keys: torch.Tensor, encoded_config: str) -> torch.Tensor:
    """Return an empty read shaped as `read_keys` returns."""
    sequence_count, key_count, _ = index_keys.shape
    return index_keys.new_empty(sequence_count, key_count, key_count, dtype=torch.bool)


def read_keys_opaque(
    index_queries: torch.Tensor, index_keys: torch.Tensor, config: AttentionConfig
) -> torch.Tensor:
    """Mark the keys each query reads, as `read_keys` does, behind a custom op.

    Args:
        index_queries: Index queries [N, key_count, index_dim] fp32 unit rows.
        index_keys: Index keys [N, key_count, index_dim] fp32 unit rows.
        config: The attention's knobs.

    Returns:
        Whether query reads key [N, key_count, key_count] bool.
    """
    return _lsa_read_keys_op(index_queries.detach(), index_keys.detach(), config.encoded)
