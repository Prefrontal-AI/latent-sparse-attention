"""Multi-resolution attention - every closed key enters the softmax once, itself or through its bucket's summary.

A query reads its sinks, its window and the keys of the buckets it selects at fine resolution, with logits
`s_ij = q_i . k_j / sqrt(d)` and values `v_j`. Every other closed bucket enters the same softmax as one coarse
term, with the logit `shat_ib = log |b| + q_i . mean_b / sqrt(d) + ||q_i||^2 variance_b / (2 d)` and the
bucket's mean value. Selection only decides which buckets are read at fine resolution, so next-token cross
entropy is the only loss and its gradient reaches every bucket, the unselected ones through their summaries.

Three limits hold and are tests: every bucket selected is dense attention, singleton buckets are dense attention
whatever is selected, and nothing selected is pure coarse attention beyond the window and the sinks.

Two forms of it, which a test holds to each other. `reference_attention` is a loop over queries that does one
thing at a time and is the oracle. `multires_attention` is the training path: the buckets, their statistics and
the reads are built in [buckets](buckets.py), and one `flex_attention` runs over the keys followed by the bucket
means, with a mask that keeps a key iff its bucket is read and a summary iff its bucket is closed and not read,
and a score modification that adds the coarse bias from tensors it captures - so its gradient reaches the
statistics. The cost is dense attention's plus the summaries: this path measures what the model learns, and
makes no claim to be a sparse kernel.

`MultiresInnerAttention` puts the three attentions - dense, window and this one - behind torchtitan's inner attention
interface, which is the whole of what a native run changes about Qwen3.5.

Native here is the training regime and has nothing to do with GLM's own indexer, which the flat design's
`native-picks` reads as the model's native sparse selection.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torchtitan.models.common.attention import FlexInnerAttention

from lsa.training.native.buckets import (
    BucketStatistics,
    BucketTable,
    Reads,
    at_least_float32,
    bucket_statistics,
    estimated_log_mass,
)
from lsa.training.native.config import AttentionConfig
from lsa.training.native.constants import Kind
from lsa.training.native.lsa import index_vectors
from lsa.training.native.ops import partition_keys_opaque, read_keys_opaque, select_buckets_opaque

# Rows and columns of a flex tile. A head wider than 128 leaves no room in shared memory for the default 128 x 128
TILE = 64
NARROW_HEAD_TILE = 128
NARROW_HEAD_DIM = 128

# The logit of a term that is not read, finite so that a softmax with none of them left has no NaN gradient
NO_MASS = -1e30

# Blackwell's Triton lowers an `ieee` fp32 dot at a 64-wide tile to a single-pass tf32 MMA, ~1e-3 off; three passes
# restore fp32. It reaches flex through `kernel_options`, and bf16 inputs never read it
BLACKWELL = 10
FP32_FLEX_OPTIONS = {"FLOAT32_PRECISION": "'tf32x3'"}

compiled_flex: Callable[..., Any] = torch.compile(flex_attention, dynamic=False)
compiled_block_mask: Callable[..., Any] = torch.compile(create_block_mask, dynamic=False)


def window_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, config: AttentionConfig, scale: float
) -> torch.Tensor:
    """Attend over each query's window and the sinks alone, the multi-resolution read with nothing else.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim].
    """
    window, sink = config.window_size, config.sink_size

    def mask_mod(
        stream: torch.Tensor, head: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        return (key <= query) & ((query - key < window) | (key < sink))

    return _flex(queries, keys, values, mask_mod, scale)[0]


def lsa_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    read: torch.Tensor,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Attend over each query's sinks, window, open block and the keys the flat LSA read picked, nothing else.

    Args:
        queries: Queries [S, G, key_count, head_dim], S = sequences * kv heads.
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        read: Whether query reads key, one selection a sequence for all its heads [sequences, key_count, key_count] bool.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim].
    """
    block, sink, window = config.block_size, config.sink_size, config.window_size
    kv_heads = queries.shape[0] // read.shape[0]

    def mask_mod(
        stream: torch.Tensor, head: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """Mark the keys a query reads: causal keys in its open block, the sinks, window or the LSA read."""
        in_open_block = key // block == query // block
        selected = read[stream // kv_heads, query, key]
        return (key <= query) & (in_open_block | (key < sink) | (query - key < window) | selected)

    return _flex(queries, keys, values, mask_mod, scale)[0]


class Multires(NamedTuple):
    """A multi-resolution read and the buckets it was built from, which the diagnostics read back."""

    # The context [S, G, key_count, head_dim]
    context: torch.Tensor
    table: BucketTable
    statistics: BucketStatistics
    reads: Reads


def multires_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, config: AttentionConfig, scale: float
) -> Multires:
    """Attend over the keys of the selected buckets and the summaries of the rest, in one softmax.

    Args:
        queries: Queries [S, G, key_count, head_dim], G query heads to a key-value head.
        keys: Post-RoPE, post-norm keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context and the buckets, their statistics and the reads it was computed from.
    """
    table = partition_keys_opaque(keys, config)
    statistics = bucket_statistics(keys, values, table)
    reads = select_buckets_opaque(queries, statistics, table, config, scale)
    context = read_attention(queries, keys, values, table, statistics, reads, config, scale)
    return Multires(context, table, statistics, reads)


def read_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    table: BucketTable,
    statistics: BucketStatistics,
    reads: Reads,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Run the joint softmax once the buckets, their statistics and the reads are known.

    The fine terms are one flex attention over the keys, with the keys of every bucket the query reads kept and
    its sinks, its window and its open block. The coarse terms are every other closed bucket's summary, taken
    densely, and the two are joined by their log-sum-exps - which is the one softmax over both, and what lets the
    gradient reach the statistics without flex accumulating into them from every query.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        table: The buckets.
        statistics: The buckets' statistics, which the gradient flows through.
        reads: The buckets each query reads at fine resolution.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim].
    """
    block, sink, window = config.block_size, config.sink_size, config.window_size
    bucket_of_key, read = table.bucket_of_key, reads.read

    def mask_mod(
        stream: torch.Tensor, head: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """Mark the keys a query reads: causal keys in its open block, the sinks, window or a pick."""
        in_open_block = key // block == query // block
        selected = read[stream, query, bucket_of_key[stream, key]]
        return (key <= query) & (in_open_block | (key < sink) | (query - key < window) | selected)

    fine, log_fine = _flex(queries, keys, values, mask_mod, scale)
    if not config.coarse or table.bucket_count == 0:
        return fine
    return _join_coarse(queries, fine, log_fine, statistics, table, reads, config, scale)


def _join_coarse(
    queries: torch.Tensor,
    fine: torch.Tensor,
    log_fine: torch.Tensor,
    statistics: BucketStatistics,
    table: BucketTable,
    reads: Reads,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Add the coarse terms to a context that has the fine ones, as one softmax over both.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        fine: The context of the fine terms alone [S, G, key_count, head_dim].
        log_fine: The log of the fine terms' mass [S, G, key_count] fp32.
        statistics: The buckets' statistics.
        table: The buckets.
        reads: The buckets each query reads at fine resolution.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context of every term [S, G, key_count, head_dim].
    """
    query_count = queries.shape[2]
    block_of_query = torch.arange(query_count, device=queries.device) // config.block_size
    summarised = (table.block_of_bucket[None, None, :] < block_of_query[None, :, None]) & ~reads.read[
        ..., :-1
    ]
    logits = estimated_log_mass(queries, statistics, table, scale, config.variance)
    # A large finite floor rather than -inf, so a query with nothing summarised has a gradient of zero rather than NaN
    logits = logits.masked_fill(~summarised[:, None], NO_MASS)
    log_total = torch.logaddexp(log_fine, logits.logsumexp(-1))
    shares = (logits - log_total[..., None]).exp()
    coarse = torch.einsum("sgqb,sbd->sgqd", shares, statistics.value_means)
    return (torch.exp(log_fine - log_total)[..., None] * at_least_float32(fine) + coarse).to(fine.dtype)


def _flex(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask_mod: Any,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one flex attention of a key-value head's query group over its own keys and values.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        mask_mod: Which (stream, head, query, key) are read.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim], and its log-sum-exp [S, G, key_count] fp32.
    """
    if not queries.is_cuda:
        return _masked_attention(queries, keys, values, mask_mod, scale)
    stream_count, group, query_count, _ = queries.shape
    block_mask: BlockMask = compiled_block_mask(
        mask_mod,
        stream_count,
        None,
        query_count,
        keys.shape[1],
        device=queries.device,
        BLOCK_SIZE=NARROW_HEAD_TILE if queries.shape[-1] <= NARROW_HEAD_DIM else TILE,
    )
    context, log_mass = compiled_flex(
        queries,
        keys[:, None],
        values[:, None],
        block_mask=block_mask,
        scale=scale,
        enable_gqa=group > 1,
        return_lse=True,
        kernel_options=_kernel_options(queries),
    )
    return context, log_mass.float()


def _kernel_options(queries: torch.Tensor) -> dict[str, Any] | None:
    """Pick the flex kernel's options, which keep fp32 inputs at fp32 on Blackwell.

    Args:
        queries: The queries flex runs over.

    Returns:
        The options, or None to leave the kernel as it is.
    """
    if queries.dtype == torch.float32 and torch.cuda.get_device_capability(queries.device)[0] >= BLACKWELL:
        return FP32_FLEX_OPTIONS
    return None


def _masked_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask_mod: Any,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same mask as `_flex` with the scores in memory, for a device flex has no backward on.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        mask_mod: Which (stream, head, query, key) are read.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim], and its log-sum-exp [S, G, key_count].
    """
    stream_count, group, query_count, _ = queries.shape
    key_count = keys.shape[1]
    device = queries.device
    stream = torch.arange(stream_count, device=device)[:, None, None, None]
    head = torch.arange(group, device=device)[None, :, None, None]
    query = torch.arange(query_count, device=device)[None, None, :, None]
    key = torch.arange(key_count, device=device)[None, None, None, :]
    scores = torch.einsum("sgqd,skd->sgqk", queries, keys) * scale
    scores = scores.masked_fill(~mask_mod(stream, head, query, key), float("-inf"))
    return torch.einsum("sgqk,skd->sgqd", scores.softmax(-1), values), scores.logsumexp(-1)


class MultiresInnerAttention(FlexInnerAttention):
    """Qwen3.5's full-attention kernel, swapped for the arm a run names.

    The dense arm is torchtitan's own flex attention. The window and multi-resolution arms read `[T, H, K]`
    queries and `[T, H, K]` keys of a batch of whole sequences, regroup them as one stream a sequence and key-value
    head, and put the context back.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        """Settings of the attention: the arm's knobs, tokens per sequence and the head dims rotated."""

        attention: AttentionConfig = field(default_factory=AttentionConfig)
        # Tokens per sequence, which splits the packed tokens back into the sequences they were laid end to end
        seq_len: int = 16384
        # Leading head dims the model rotates, which the lsa arm's index space drops
        rotary_dim: int = 64

    def __init__(self, config: Config) -> None:
        """Keep the arm's knobs and set the read's measurement off."""
        super().__init__(config)
        self.attention = config.attention
        self.seq_len = config.seq_len
        self.rotary_dim = config.rotary_dim
        # Whether to record what the read did, which a held-out pass switches on
        self.measure = False
        self.probes = 128
        self.measurements: dict[str, float] = {}

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Attend causally within each sequence, under the arm this layer was built with.

        Args:
            q_THK: Queries [T, H, K], sequences laid end to end.
            k_THK: Keys [T, Hkv, K].
            v_THV: Values [T, Hkv, V].
            attention_masks: torchtitan's mask, which the dense arm reads and the others ignore.
            scale: The factor logits are scaled by.
            enable_gqa: Whether the heads are grouped.
            **kwargs: The rest of what the outer attention passes, forwarded to the dense arm.

        Returns:
            The context [T, H, V].
        """
        if self.attention.kind == Kind.DENSE:
            return super().forward(
                q_THK,
                k_THK,
                v_THV,
                attention_masks=attention_masks,
                scale=scale,
                enable_gqa=enable_gqa,
                **kwargs,
            )
        scale = q_THK.shape[-1] ** -0.5 if scale is None else scale
        queries, keys, values = self._regroup(q_THK, k_THK, v_THV)
        if self.attention.kind == Kind.WINDOW:
            context = window_attention(queries, keys, values, self.attention, scale)
        elif self.attention.kind == Kind.LSA:
            context = lsa_attention(queries, keys, values, self._read(q_THK, k_THK), self.attention, scale)
        else:
            result = multires_attention(queries, keys, values, self.attention, scale)
            context = result.context
            if self.measure:
                self._measure(queries, keys, result, scale)
        return self._restore(context, q_THK)

    def _regroup(
        self, q_THK: torch.Tensor, k_THK: torch.Tensor, v_THV: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lay tokens out as one stream a sequence and key-value head.

        Args:
            q_THK: Queries [T, H, K].
            k_THK: Keys [T, Hkv, K].
            v_THV: Values [T, Hkv, V].

        Returns:
            Queries [S, G, L, K], keys [S, L, K] and values [S, L, V], S = sequences * Hkv.
        """
        tokens, heads, width = q_THK.shape
        kv_heads = k_THK.shape[1]
        sequences = tokens // self.seq_len
        queries = q_THK.view(sequences, self.seq_len, kv_heads, heads // kv_heads, width)
        queries = queries.permute(0, 2, 3, 1, 4).reshape(
            sequences * kv_heads, heads // kv_heads, self.seq_len, width
        )
        keys = k_THK.view(sequences, self.seq_len, kv_heads, width).permute(0, 2, 1, 3)
        values = v_THV.view(sequences, self.seq_len, kv_heads, v_THV.shape[-1]).permute(0, 2, 1, 3)
        return (
            queries,
            keys.reshape(-1, self.seq_len, width),
            values.reshape(-1, self.seq_len, values.shape[-1]),
        )

    def _read(self, q_THK: torch.Tensor, k_THK: torch.Tensor) -> torch.Tensor:
        """Pick the keys each query reads, once for all the heads of a sequence.

        Args:
            q_THK: Queries [T, H, K].
            k_THK: Keys [T, Hkv, K].

        Returns:
            Whether query reads key [sequences, seq_len, seq_len] bool.
        """
        tokens, heads, width = q_THK.shape
        sequences = tokens // self.seq_len
        queries = q_THK.reshape(sequences, self.seq_len, heads, width).transpose(1, 2)
        keys = k_THK.reshape(sequences, self.seq_len, -1, width).transpose(1, 2)
        index_queries, index_keys = index_vectors(queries, keys, self.rotary_dim)
        return read_keys_opaque(index_queries, index_keys, self.attention)

    def _restore(self, context: torch.Tensor, q_THK: torch.Tensor) -> torch.Tensor:
        """Undo `_regroup` on the context.

        Args:
            context: The context [S, G, L, V].
            q_THK: The queries, for the layout to return to.

        Returns:
            The context [T, H, V].
        """
        tokens, heads, _ = q_THK.shape
        sequences = tokens // self.seq_len
        kv_heads = context.shape[0] // sequences
        context = context.view(sequences, kv_heads, heads // kv_heads, self.seq_len, -1)
        return context.permute(0, 3, 1, 2, 4).reshape(tokens, heads, -1).contiguous()

    @torch.no_grad()
    def _measure(self, queries: torch.Tensor, keys: torch.Tensor, result: Multires, scale: float) -> None:
        """Add this layer's diagnostics to the running sums a held-out pass reads.

        Args:
            queries: Queries [S, G, L, K].
            keys: Keys [S, L, K].
            result: The read being measured.
            scale: The factor logits are scaled by.
        """
        shares = coarse_mass_share(
            queries, keys, result.statistics, result.table, result.reads, self.attention, scale, self.probes
        )
        shares["read_keys"] = result.reads.key_counts.mean().item()
        shares["passes"] = 1.0
        for name, value in shares.items():
            self.measurements[name] = self.measurements.get(name, 0.0) + value


@torch.no_grad()
def coarse_mass_share(
    queries: torch.Tensor,
    keys: torch.Tensor,
    statistics: BucketStatistics,
    table: BucketTable,
    reads: Reads,
    config: AttentionConfig,
    scale: float,
    probes: int,
) -> dict[str, float]:
    """Measure the share of the softmax a query puts on coarse terms, at queries spread over the sequence.

    The share is exact at the probed queries - their scores are taken against every key and every summary - and
    the probes are the only cost, so a held-out pass can afford it where a training step cannot.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        statistics: The buckets' statistics.
        table: The buckets.
        reads: The buckets each query read.
        config: The attention's knobs.
        scale: The factor logits are scaled by.
        probes: Queries probed per stream.

    Returns:
        The mean coarse share over every probe, and over the probes past the fine read's key budget, where
        some bucket is always summarised. Both are 0 when the run reads no summaries.
    """
    if not config.coarse or table.bucket_count == 0:
        return {"coarse_mass_share": 0.0, "coarse_mass_share_far": 0.0}
    key_count = keys.shape[1]
    positions = torch.linspace(0, key_count - 1, probes, device=keys.device).long().unique()
    probe_queries = queries[:, :, positions]
    key_ids = torch.arange(key_count, device=keys.device)
    block, sink, window = config.block_size, config.sink_size, config.window_size
    read_rows = reads.read[:, positions]

    fine_logits = (
        torch.einsum("sgpd,skd->sgpk", at_least_float32(probe_queries), at_least_float32(keys)) * scale
    )
    selected = read_rows.gather(2, table.bucket_of_key[:, None, :].expand(-1, positions.numel(), -1))
    fine_visible = (key_ids[None, :] <= positions[:, None])[None] & (
        ((key_ids[None, :] // block) == (positions[:, None] // block))[None]
        | (key_ids < sink)[None, None]
        | ((positions[:, None] - key_ids[None, :]) < window)[None]
        | selected
    )
    coarse_logits = estimated_log_mass(probe_queries, statistics, table, scale, config.variance)
    coarse_visible = (table.block_of_bucket[None, :] < (positions // block)[:, None])[None] & ~read_rows[
        ..., :-1
    ]
    log_fine = fine_logits.masked_fill(~fine_visible[:, None], float("-inf")).logsumexp(-1)
    log_coarse = coarse_logits.masked_fill(~coarse_visible[:, None], float("-inf")).logsumexp(-1)
    share = (log_coarse - torch.logaddexp(log_fine, log_coarse)).exp()
    far = positions >= config.key_budget
    return {
        "coarse_mass_share": share.mean().item(),
        "coarse_mass_share_far": share[:, :, far].mean().item() if far.any() else 0.0,
    }


def reference_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    table: BucketTable,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Attend one query at a time, from the definition - the oracle the training path answers to.

    It takes the partition as given and rebuilds everything else the slow way: each bucket's statistics from
    its own keys, each query's candidates, ranking and cut from a plain sort, and one softmax over the terms the
    query reads. Differentiable through the statistics, so a test can hold the training path's gradients to it.

    Args:
        queries: Queries [S, G, key_count, head_dim].
        keys: Keys [S, key_count, head_dim].
        values: Values [S, key_count, head_dim].
        table: The buckets, from `partition_keys`.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context [S, G, key_count, head_dim].
    """
    stream_count, group, key_count, head_dim = queries.shape
    context = torch.zeros_like(at_least_float32(queries))
    for stream in range(stream_count):
        members = [
            (table.bucket_of_key[stream] == bucket).nonzero().flatten()
            for bucket in range(table.bucket_count)
        ]
        means = [at_least_float32(keys[stream, ids]).mean(0) for ids in members]
        value_means = [at_least_float32(values[stream, ids]).mean(0) for ids in members]
        variances = [
            (at_least_float32(keys[stream, ids]) - mean).square().sum() / (len(ids) * head_dim)
            for ids, mean in zip(members, means, strict=True)
        ]
        for query in range(key_count):
            context[stream, :, query] = _reference_query(
                at_least_float32(queries[stream, :, query]), at_least_float32(keys[stream]), at_least_float32(values[stream]),
                members, means, value_means, variances, query, config, scale,
            )  # fmt: skip
    return context


def _reference_query(
    heads: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    members: list[torch.Tensor],
    means: list[torch.Tensor],
    value_means: list[torch.Tensor],
    variances: list[torch.Tensor],
    query: int,
    config: AttentionConfig,
    scale: float,
) -> torch.Tensor:
    """Attend one position's query heads, the reference's inner step.

    Args:
        heads: The position's queries [G, head_dim].
        keys: The stream's keys [key_count, head_dim].
        values: The stream's values [key_count, head_dim].
        members: Each bucket's key positions.
        means: Each bucket's mean key.
        value_means: Each bucket's mean value.
        variances: Each bucket's key variance per dimension.
        query: The position.
        config: The attention's knobs.
        scale: The factor logits are scaled by.

    Returns:
        The context of every head [G, head_dim].
    """
    block, sink, window = config.block_size, config.sink_size, config.window_size
    closed = [b for b in range(len(members)) if b // config.buckets_per_block < query // block]
    pinned = [b for b in closed if members[b].min() < sink or members[b].max() > query - window]
    candidates = [b for b in closed if b not in pinned]

    def log_mass(head: torch.Tensor, bucket: int) -> torch.Tensor:
        estimate = math.log(len(members[bucket])) + head @ means[bucket] * scale
        if config.variance:
            estimate = estimate + 0.5 * scale**2 * head.square().sum() * variances[bucket]
        return estimate

    ranked: list[int] = []
    if candidates and config.top_k > 0:
        logs = torch.stack([torch.stack([log_mass(head, b) for b in candidates]) for head in heads])
        if config.rank == "cosine":
            unit_means = torch.stack([means[b] / means[b].norm() for b in candidates])
            score = (heads / heads.norm(dim=-1, keepdim=True) @ unit_means.T).sum(0)
        else:
            if config.rank == "error":
                squared = heads.square().sum(-1, keepdim=True)
                logs = logs + torch.log(
                    squared * torch.stack([variances[b] for b in candidates])[None] + 1e-30
                )
            score = torch.softmax(logs, dim=-1).sum(0)
        order = sorted(range(len(candidates)), key=lambda i: (-float(score[i].detach()), candidates[i]))
        taken = 0
        for i in order[: config.selection_width]:
            size = len(members[candidates[i]])
            if taken + size > config.top_k:
                break
            taken += size
            ranked.append(candidates[i])
    read = set(ranked) | set(pinned)

    fine = [
        j
        for j in range(query + 1)
        if j // block == query // block
        or j < sink
        or query - j < window
        or any(j in members[b].tolist() for b in read)
    ]
    fine_ids = torch.tensor(fine)
    coarse = [b for b in closed if b not in read] if config.coarse else []
    out = []
    for head in heads:
        logits = (
            torch.cat([keys[fine_ids] @ head * scale, torch.stack([log_mass(head, b) for b in coarse])])
            if coarse
            else keys[fine_ids] @ head * scale
        )
        terms = (
            torch.cat([values[fine_ids], torch.stack([value_means[b] for b in coarse])])
            if coarse
            else values[fine_ids]
        )
        out.append(torch.softmax(logits, dim=0) @ terms)
    return torch.stack(out)
