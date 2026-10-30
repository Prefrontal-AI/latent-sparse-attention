"""Train a flat LSA indexer while retaining the position-free arm's exact read."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torchtitan.models.common import Linear
from torchtitan.models.common.aux_loss import AuxLoss

from lsa.training.native import lsa_summaries
from lsa.training.native.kernels import (
    cuda_sparse,
    gather_sparse,
    lsa_rank,
    lsa_select,
    reference,
)
from lsa.training.native.layers import MultiresInnerAttention, lsa_attention
from lsa.training.native.lsa import build_flat_buckets, index_vectors
from lsa.training.native.lsa_gradients import LogMass, StraightThrough


class IndexerKl(AuxLoss):
    """Inject the mean index KL under the trainer's token normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Settings of the loss - the base's, with none added."""

        pass

    def forward(self, loss: torch.Tensor, context: torch.Tensor, tokens: int) -> torch.Tensor:
        """Attach token-weighted index loss to the unchanged context."""
        return self.inject(loss * tokens, carrier=context)


def bucket_scores(
    index_queries: torch.Tensor, index_keys: torch.Tensor, members: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild differentiable centroids from fixed bucket members.

    Args:
        index_queries: Index queries [B, T, index_dim].
        index_keys: Index keys [B, T, index_dim].
        members: Bucket members [B, M, bucket_cap], padded with -1.

    Returns:
        Bucket owners [B, T] and centroid scores [B, T, M].
    """
    batch, tokens, _ = index_keys.shape
    held = members >= 0
    rows = functional.normalize(index_keys.float(), dim=-1)
    batch_ids = torch.arange(batch, device=rows.device)[:, None, None]
    sums = (rows[batch_ids, members.clamp_min(0)] * held[..., None]).sum(2)
    centroids = functional.normalize(sums, dim=-1)
    scores = index_queries.float() @ centroids.transpose(1, 2)
    owner = torch.zeros(batch, tokens + 1, dtype=torch.long, device=rows.device)
    ids = torch.arange(members.shape[1], device=rows.device).repeat_interleave(members.shape[2])
    owner.scatter_(1, torch.where(held, members, tokens).flatten(1), ids.expand(batch, -1))
    return owner[:, :tokens], scores


def window_scores(index_queries: torch.Tensor, index_keys: torch.Tensor, window: int) -> torch.Tensor:
    """Score each query's window, with offset zero denoting the query itself."""
    batch, tokens, dim = index_keys.shape
    keys = functional.normalize(index_keys.float(), dim=-1)
    keys = torch.cat([keys.new_zeros(batch, window - 1, dim), keys], dim=1)
    offsets = torch.arange(window, device=keys.device)
    parts = []
    for start in range(0, tokens, window):
        end = min(start + window, tokens)
        scores = index_queries[:, start:end].float() @ keys[:, start : end + window - 1].transpose(1, 2)
        column = torch.arange(end - start, device=keys.device)[:, None] - offsets + window - 1
        parts.append(scores.gather(2, column.expand(batch, -1, -1)))
    return torch.cat(parts, dim=1)


def selection_table(selected: torch.Tensor, tokens: int) -> torch.Tensor:
    """Expand padded selected positions into a boolean read table."""
    table = torch.zeros(*selected.shape[:2], tokens + 1, device=selected.device, dtype=torch.bool)
    table.scatter_(2, torch.where(selected >= 0, selected.long(), tokens), True)
    return table[..., :tokens].contiguous()


def read_counts(selected: torch.Tensor, members: torch.Tensor, tokens: int) -> torch.Tensor:
    """Count each query's selected keys in each bucket."""
    batch, buckets, cap = members.shape
    owner = torch.full((batch, tokens + 1), buckets, device=members.device, dtype=torch.long)
    ids = torch.arange(buckets, device=members.device).repeat_interleave(cap)
    owner.scatter_(1, torch.where(members >= 0, members, tokens).flatten(1), ids.expand(batch, -1))
    owner[:, tokens] = buckets
    owned = owner.gather(1, torch.where(selected >= 0, selected.long(), tokens).flatten(1)).view_as(selected)
    counts = torch.zeros(*selected.shape[:2], buckets + 1, device=members.device, dtype=torch.long)
    counts.scatter_add_(2, owned, torch.ones_like(owned))
    return counts[..., :buckets]


class LsaInnerAttention(MultiresInnerAttention):
    """Read flat LSA with optional learned projections, distillation and summaries."""

    @dataclass(kw_only=True, slots=True)
    class Config(MultiresInnerAttention.Config):
        """Settings of the attention: the model's dim and heads, and the summary decay steps."""

        model_dim: int = 1024
        query_head_count: int = 8
        kv_head_count: int = 1
        head_dim: int = 128
        steps: int = 100
        decay_ratio: float = 0.8

    def __init__(self, config: Config) -> None:
        """Build optional projections and retain the summary decay schedule."""
        super().__init__(config)
        self.steps = config.steps
        self.decay_steps = math.ceil(config.steps * config.decay_ratio)
        self.summary_weight = 1.0
        self.query_projection: nn.Linear | None = None
        self.key_projection: nn.Linear | None = None
        if self.attention.index_input != "none":
            query_dim = key_dim = config.model_dim
            if self.attention.index_input == "key_and_query":
                query_dim = config.head_dim * (
                    config.query_head_count if self.attention.index_query_heads == "concat" else 1
                )
                key_dim = config.kv_head_count * config.head_dim
            self.query_projection = Linear.Config(
                in_features=query_dim,
                out_features=self.attention.index_dim,
                param_init={"weight": partial(nn.init.normal_, std=0.02)},
            ).build()
            self.key_projection = Linear.Config(
                in_features=key_dim,
                out_features=self.attention.index_dim,
                param_init={"weight": partial(nn.init.normal_, std=0.02)},
            ).build()
            nn.init.normal_(self.query_projection.weight, std=0.02)
            nn.init.normal_(self.key_projection.weight, std=0.02)
        self.query_norm_weight: nn.Parameter | None = None
        self.key_norm_weight: nn.Parameter | None = None
        if self.attention.index_input != "none" and self.attention.index_norm == "rms":
            self.query_norm_weight = nn.Parameter(torch.ones(self.attention.index_dim))
            self.key_norm_weight = nn.Parameter(torch.ones(self.attention.index_dim))
        self.indexer_kl = IndexerKl.Config(coeff=self.attention.index_loss_weight).build()

    def reset_parameters(self) -> None:
        """Initialize learned RMS normalization weights after model materialization."""
        if self.query_norm_weight is not None and self.key_norm_weight is not None:
            nn.init.ones_(self.query_norm_weight)
            nn.init.ones_(self.key_norm_weight)

    def _normalize(self, inputs: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """Normalize projected index vectors and apply optional index rotation."""
        if weight is None:
            normalized = functional.normalize(inputs, dim=-1)
        else:
            normalized = functional.rms_norm(
                inputs.float(), (inputs.shape[-1],), weight.float(), eps=1e-6
            ).to(inputs.dtype)
        rotary_dim = self.attention.index_rotary_dim
        if not rotary_dim:
            return normalized
        frequencies = 10000.0 ** (-torch.arange(0, rotary_dim, 2, device=inputs.device).float() / rotary_dim)
        angles = torch.arange(self.seq_len, device=inputs.device)[:, None] * frequencies[None]
        angles = torch.cat([angles, angles], dim=-1)
        first, second = normalized[..., :rotary_dim].chunk(2, dim=-1)
        rotated = (
            normalized[..., :rotary_dim].float() * angles.cos()
            + torch.cat([-second, first], dim=-1).float() * angles.sin()
        )
        return torch.cat([rotated.to(inputs.dtype), normalized[..., rotary_dim:]], dim=-1)

    def set_step(self, step: int) -> None:
        """Anneal summaries to zero halfway through the decay from completed updates."""
        start = self.steps - self.decay_steps
        self.summary_weight = (
            1.0
            if self.decay_steps == 0 or step < start
            else max(0.0, (self.steps - self.decay_steps / 2 - step) / (self.decay_steps / 2))
        )

    def _index_vectors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_input: torch.Tensor | None,
        index_heads: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build shared index vectors from the configured source."""
        batch = q.shape[0] // self.seq_len
        q, k = (q, k) if index_heads is None else index_heads
        queries = q.reshape(batch, self.seq_len, q.shape[1], q.shape[2])
        keys = k.reshape(batch, self.seq_len, k.shape[1], k.shape[2])
        if self.attention.index_input == "none":
            if self.attention.query_head_reduction == "max":
                if keys.shape[2] != 1 or self.rotary_dim:
                    raise ValueError("query_head_reduction max requires one key head and rotary_dim 0")
                return functional.normalize(queries.detach().float(), dim=-1), functional.normalize(
                    keys[:, :, 0].detach().float(), dim=-1
                )
            return index_vectors(queries.transpose(1, 2), keys.transpose(1, 2), self.rotary_dim)
        if self.query_projection is None or self.key_projection is None:
            raise RuntimeError("learned index inputs require initialized projections")
        if self.attention.index_input == "attention_input":
            if attention_input is None:
                raise ValueError("index_input attention_input requires attention_input [T, model_dim]")
            query_source = key_source = attention_input.reshape(batch, self.seq_len, -1)
        else:
            query_source = (
                queries.flatten(2) if self.attention.index_query_heads == "concat" else queries.sum(2)
            )
            key_source = keys.flatten(2)
        if self.attention.index_detached:
            query_source, key_source = query_source.detach(), key_source.detach()
        return (
            self._normalize(self.query_projection(query_source), self.query_norm_weight),
            self._normalize(self.key_projection(key_source), self.key_norm_weight),
        )

    @torch.no_grad()
    def _selected(
        self, index_queries: torch.Tensor, index_keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select long-range positions with fused ranking where its shape is supported."""
        config = self.attention
        if self.seq_len <= config.block_size:
            selected = torch.full(
                (*index_queries.shape[:2], config.top_k), -1, dtype=torch.long, device=index_queries.device
            )
            members = torch.empty(
                index_keys.shape[0], 0, config.bucket_cap, dtype=torch.long, device=index_keys.device
            )
            return selected, members
        buckets = build_flat_buckets(index_keys, config)
        buckets = buckets[0], buckets[1].long()
        positions = torch.arange(self.seq_len, device=index_queries.device)
        args = (
            config.block_size,
            config.bucket_size,
            config.selection_width,
            config.top_k,
            config.window_size,
            config.sink_size,
        )
        ranked = (
            index_queries.to(torch.get_autocast_dtype("cuda"))
            if index_queries.is_cuda and torch.is_autocast_enabled("cuda")
            else index_queries
        )
        if config.top_k == 0 or config.selection_width == 0 or buckets[1].shape[1] == 0:
            return torch.full(
                (*index_queries.shape[:2], config.top_k), -1, dtype=torch.long, device=index_queries.device
            ), buckets[1]
        if lsa_rank.supported(ranked, buckets, config.block_size, config.selection_width, config.pool_size):
            selected = lsa_rank.select_indices(ranked, buckets, *args, pool=config.pool_size)
        elif lsa_select.supported(ranked, buckets):
            selected = lsa_select.select_indices(ranked, positions, buckets, *args)
        else:
            selected = reference.select_indices(index_queries, positions, buckets, *args)
        return selected, buckets[1]

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_input: torch.Tensor | None = None,
        index_heads: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Attend to selected keys and train the configured indexer."""
        config = self.attention
        if (
            config.index_input == "none"
            and config.query_head_reduction == "sum"
            and not config.summaries
            and not config.fast_read
        ):
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
        index_queries, index_keys = self._index_vectors(q_THK, k_THK, attention_input, index_heads)
        selected, members = self._selected(index_queries, index_keys)
        batch = q_THK.shape[0] // self.seq_len
        q = q_THK.reshape(batch, self.seq_len, q_THK.shape[1], q_THK.shape[2])
        k = k_THK.reshape(batch, self.seq_len, k_THK.shape[1], k_THK.shape[2])
        v = v_THV.reshape(batch, self.seq_len, v_THV.shape[1], v_THV.shape[2])
        grouped_q, grouped_k, grouped_v = self._regroup(q_THK, k_THK, v_THV)
        fast = (
            config.fast_read
            and q.is_cuda
            and q.dtype in gather_sparse.DTYPES
            and q.shape[-1] in gather_sparse.HEAD_DIMS
            and config.block_size <= config.window_size
        )
        kept = (
            selection_table(selected, self.seq_len)
            if not fast
            else selected.new_empty((0,), dtype=torch.bool)
        )
        if fast:
            reader_members = members if config.bucket_cap <= 32 else None
            if cuda_sparse.supported(q, k):
                context, lse = cuda_sparse.attention(
                    q, k, v, selected, config.window_size, config.sink_size, scale, reader_members, v2=True
                )
            else:
                context, lse = gather_sparse.attention(
                    q, k, v, selected, config.window_size, config.sink_size, scale, reader_members
                )
            context = context.transpose(1, 2)
        else:
            context = (
                self._restore(lsa_attention(grouped_q, grouped_k, grouped_v, kept, config, scale), q_THK)
                .reshape_as(q)
                .transpose(1, 2)
            )
            lse = self._log_mass(q, k, kept, scale)
        queries, keys, values = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ste = self.training and config.index_training == "ste"
        joint = None
        share = None
        if self.training and config.summaries and self.summary_weight > 0 and members.shape[1]:
            if fast:
                lse = LogMass.apply(
                    queries,
                    keys,
                    selected,
                    lse,
                    config.window_size,
                    config.sink_size,
                    scale,
                    members if config.bucket_cap <= 32 else None,
                )
            share, coarse = lsa_summaries.terms(
                queries,
                keys,
                values,
                lse,
                members,
                selected.new_empty((0,)),
                config.block_size,
                config.window_size,
                config.sink_size,
                scale,
                self.summary_weight,
                index_sources=(index_queries, index_keys) if ste else None,
                selected_positions=selected,
                buckets_per_block=config.buckets_per_block,
            )
            joint = (share[..., None] * context + coarse).to(context.dtype)
        if ste:
            context = StraightThrough.apply(
                context,
                index_queries,
                index_keys,
                members,
                selected,
                (queries.detach(), keys.detach(), values.detach(), lse.detach()),
                (config.window_size, config.sink_size, scale, config.ste_window),
                None if joint is None else joint.detach(),
            )
        if joint is not None and share is not None:
            context = (share[..., None] * context + coarse).to(context.dtype) if ste else joint
        if self.training and config.index_training == "kl":
            loss = self._kl(queries, keys, lse, index_queries, index_keys, selected, scale)
            context = self.indexer_kl(loss, context, q_THK.shape[0])
        return context.transpose(1, 2).reshape(q_THK.shape[0], q_THK.shape[1], -1).contiguous()

    def _log_mass(self, q: torch.Tensor, k: torch.Tensor, kept: torch.Tensor, scale: float) -> torch.Tensor:
        """Compute the reference read's log mass for the KL and straight-through paths."""
        heads = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
        logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), heads.float()) * scale
        positions = torch.arange(self.seq_len, device=q.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (
            (distance < self.attention.window_size)
            | (positions[None] < self.attention.sink_size)
            | (
                positions[:, None] // self.attention.block_size
                == positions[None] // self.attention.block_size
            )
        )
        return logits.masked_fill(~(mask[None] | kept)[:, None], float("-inf")).logsumexp(-1)

    def _kl(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        lse: torch.Tensor,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        kept: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Distill head-summed attention mass on sampled nonempty long-range selections."""
        first = self.attention.window_size + self.attention.sink_size
        eligible = max(0, self.seq_len - first)
        if eligible == 0:
            return (index_queries.sum() + index_keys.sum()) * 0.0
        positions = first + torch.randperm(eligible, device=queries.device)[: self.attention.kl_query_count]
        support = selection_table(kept[:, positions], self.seq_len)
        counted = support.any(-1)
        support = support.clone()
        support[..., 0] |= ~counted
        with torch.no_grad():
            heads = keys.repeat_interleave(queries.shape[1] // keys.shape[1], dim=1)
            logits = (queries[:, :, positions] @ heads.transpose(-1, -2)).float() * scale
            mass = (logits - lse[:, :, positions, None]).exp().sum(1).masked_fill(~support, 0.0)
            teacher = mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)
        student = (index_queries[:, positions] @ index_keys.transpose(1, 2)) * self.attention.index_dim**-0.5
        log_student = student.float().masked_fill(~support, float("-inf")).log_softmax(-1)
        difference = torch.where(support, teacher.clamp_min(1e-30).log() - log_student, 0.0)
        loss = (teacher * difference).sum(-1)
        loss = (loss * counted).sum() / counted.sum().clamp_min(1)
        if self.measure:
            self.measurements["kl"] = float(loss.detach())
        return loss
