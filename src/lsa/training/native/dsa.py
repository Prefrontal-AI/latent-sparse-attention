"""Learn a per-head long-range selector against detached attention probabilities."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn
from torch.nn import functional as functional
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.module import Module

from lsa.training.native.config import AttentionConfig
from lsa.training.native.heads import HeadShapeAttention
from lsa.training.native.kernels import dsa_sparse


def masked_kl(
    teacher: torch.Tensor, student: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill detached teacher logits over nonempty masked rows.

    Args:
        teacher: Teacher logits [B, H, Q, K].
        student: Index logits [B, H, Q, K].
        mask: Keys included in each row, broadcastable to the logits.

    Returns:
        Mean KL and detached agreement and entropy statistics.
    """
    dtype = torch.promote_types(torch.promote_types(teacher.dtype, student.dtype), torch.float32)
    teacher_logits = teacher.detach().to(dtype).masked_fill(~mask, float("-inf"))
    student_logits = student.to(dtype).masked_fill(~mask, float("-inf"))
    teacher_log = teacher_logits.log_softmax(-1)
    student_log = student_logits.log_softmax(-1)
    probability = teacher_log.exp()
    difference = torch.where(mask, teacher_log - student_log, 0.0)
    loss = (probability * difference).sum(-1).mean()
    with torch.no_grad():
        entropy = -torch.where(mask, probability * teacher_log, 0.0).sum(-1)
        support = mask.sum(-1).to(dtype)
        stats = {
            "kl": loss.detach(),
            "entropy": entropy.mean(),
            "support_size": support.mean(),
            "uniform_kl": (support.log() - entropy).mean(),
            "top1_agree": (teacher_logits.argmax(-1) == student_logits.argmax(-1)).to(dtype).mean(),
        }
    return loss, stats


class IndexerKl(AuxLoss):
    """Inject an indexer's mean KL with the trainer's token normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Settings of the loss - the base's, with none added."""

        pass

    def forward(self, loss: torch.Tensor, carrier: torch.Tensor, token_count: int) -> torch.Tensor:
        """Attach a token-weighted mean KL to the context's backward path.

        Args:
            loss: Mean per-query KL scalar.
            carrier: Attention context.
            token_count: Tokens this microbatch contributes to the step.

        Returns:
            The unchanged attention context.
        """
        return self.inject(loss * token_count, carrier=carrier)


class IndexProjection(Module):
    """Project each head separately into the learned index space."""

    def __init__(self, head_count: int, input_dim: int, index_dim: int) -> None:
        """Build one independent projection for each index head.

        Args:
            head_count: Index heads.
            input_dim: Source dims per head.
            index_dim: Index dims per head.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(head_count, input_dim, index_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each head's projection with a normal distribution."""
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project head-major input vectors.

        Args:
            inputs: Input vectors [B, H, T, D].

        Returns:
            Index vectors [B, H, T, index_dim].
        """
        return torch.einsum("bhtd,hdi->bhti", inputs, self.weight)


class DsaAttention(HeadShapeAttention):
    """Attend to learned top-k keys and train their index scores by self-distillation."""

    @dataclass(kw_only=True, slots=True)
    class Config(HeadShapeAttention.Config):
        """Settings of the attention: its projections, head shape, attention knobs and sequence length."""

        attention: AttentionConfig
        seq_len: int

    def __init__(self, config: Config) -> None:
        """Build the attention and its learned self-distilled indexer.

        Args:
            config: Attention projections, head shape and indexer knobs.

        Raises:
            ValueError: The index heads do not fit the attention or a slice exceeds its head.
        """
        super().__init__(config)
        self.attention = config.attention
        self.seq_len = config.seq_len
        self.step = 0
        heads = config.attention.index_head_count
        if heads not in (1, config.n_kv_heads, config.n_heads):
            raise ValueError(
                f"attention.index_head_count must be 1, {config.n_kv_heads} or {config.n_heads} - got {heads}"
            )
        input_dim = (
            config.wq.in_features if config.attention.index_input == "attention_input" else config.head_dim
        )
        if config.attention.index_input == "sliced_key_and_query":
            if config.attention.index_dim > config.head_dim:
                raise ValueError(
                    f"attention.index_dim must be at most head_dim {config.head_dim} for a slice - got {config.attention.index_dim}"
                )
            input_dim = config.attention.index_dim
        self.query_projection = IndexProjection(heads, input_dim, config.attention.index_dim)
        self.key_projection = IndexProjection(heads, input_dim, config.attention.index_dim)
        self.indexer_kl = IndexerKl.Config(coeff=config.attention.index_loss_weight).build()
        if config.attention.index_bias:
            if config.sharding_config is not None:
                config.sharding_config.state_shardings["index_bias_scale"] = dense_param_placement(tp=spmd.R)
            self.index_bias_scale = nn.Parameter(torch.full((1,), config.attention.index_bias_scale_init))
            if config.attention.index_detached:
                self.kl_query_projection = IndexProjection(
                    heads, config.attention.index_dim, config.attention.index_dim
                )
                self.kl_key_projection = IndexProjection(
                    heads, config.attention.index_dim, config.attention.index_dim
                )
        self.probes = config.attention.kl_query_count
        self.measure = False
        self.measurements: dict[str, float] = {}

    def reset_parameters(self) -> None:
        """Initialize the learned bias scale when the model materializes."""
        if self.attention.index_bias:
            nn.init.constant_(self.index_bias_scale, self.attention.index_bias_scale_init)

    def set_step(self, step: int) -> None:
        """Set the number of completed optimizer updates before this forward.

        Args:
            step: Completed optimizer updates, including a resumed run's updates.
        """
        self.step = step

    def index_vectors(
        self, attention_input: torch.Tensor, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the selected source into per-head index vectors.

        Args:
            attention_input: Layer inputs [T, model_dim].
            queries: Attention queries [T, H, D].
            keys: Attention keys [T, Hkv, D].

        Returns:
            Index queries and keys [B, index_head_count, seq_len, index_dim].
        """
        heads = self.attention.index_head_count
        if self.attention.index_input == "attention_input":
            source = attention_input.view(-1, self.seq_len, attention_input.shape[-1])[:, None]
            query_source = key_source = source.expand(-1, heads, -1, -1)
        else:
            query_source = self._source_heads(queries, heads)
            key_source = self._source_heads(keys, heads)
        if self.attention.index_input == "sliced_key_and_query":
            query_source = query_source[..., -self.attention.index_dim :]
            key_source = key_source[..., -self.attention.index_dim :]
        if self.attention.index_detached and not self.attention.index_bias:
            query_source, key_source = query_source.detach(), key_source.detach()
        return self.query_projection(query_source), self.key_projection(key_source)

    def _source_heads(self, inputs: torch.Tensor, heads: int) -> torch.Tensor:
        """Match attention heads to the indexer's heads.

        Args:
            inputs: Attention vectors [T, H, D].
            heads: Index head count.

        Returns:
            Input vectors [B, heads, seq_len, D].
        """
        source = inputs.view(-1, self.seq_len, inputs.shape[1], inputs.shape[2]).transpose(1, 2)
        if source.shape[1] > heads:
            return source.reshape(source.shape[0], heads, -1, self.seq_len, source.shape[-1]).mean(2)
        return source.repeat_interleave(heads // source.shape[1], dim=1)

    def _scores(self, index_queries: torch.Tensor, index_keys: torch.Tensor) -> torch.Tensor:
        """Score index query-key pairs in their projection dtype.

        Args:
            index_queries: Index queries [B, H, Q, index_dim].
            index_keys: Index keys [B, H, K, index_dim].

        Returns:
            Scaled index scores [B, H, Q, K].
        """
        scores = index_queries @ index_keys.transpose(-1, -2)
        if self.attention.index_relu:
            scores = scores.relu()
        return scores * self.attention.index_dim**-0.5

    def rank_vectors(
        self, index_queries: torch.Tensor, index_keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reproject detached bias vectors when KL learns a separate indexer.

        Args:
            index_queries: Live bias queries [B, H, T, index_dim].
            index_keys: Live bias keys [B, H, T, index_dim].

        Returns:
            The KL-trained queries and keys at the same shape.
        """
        if self.attention.index_bias and self.attention.index_detached:
            return self.kl_query_projection(index_queries.detach()), self.kl_key_projection(
                index_keys.detach()
            )
        return index_queries, index_keys

    def attend(
        self,
        x_TD: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        *,
        index_heads: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Read selected keys and inject sampled long-range self-distillation.

        Args:
            x_TD: Layer inputs [T, model_dim].
            queries: Rotated queries [T, H, D].
            keys: Rotated keys [T, Hkv, D].
            values: Values [T, Hkv, D].
            attention_masks: Dense arm masks, unused for fixed-length sequences.
            index_heads: Position-free vectors, unused by DSA's post-rotation indexer.

        Returns:
            Context [T, H, D].
        """
        index_queries, index_keys = self.index_vectors(x_TD, queries, keys)
        config = self.attention
        sparse = not self.training or self.step >= config.sparse_start_step
        rank_queries, rank_keys = self.rank_vectors(index_queries, index_keys)
        batch = queries.shape[0] // self.seq_len
        q, k, v = [
            tensor.view(batch, self.seq_len, tensor.shape[1], tensor.shape[2])
            for tensor in (queries, keys, values)
        ]
        selection = None
        scores = None
        keep = None
        if (
            sparse
            and not config.index_bias
            and not config.index_relu
            and config.index_head_count in (self.n_heads, self.n_kv_heads)
            and dsa_sparse.supported(q, index_queries)
        ):
            selection = dsa_sparse.select(
                rank_queries.detach(),
                rank_keys.detach(),
                config.window_size,
                config.sink_size,
                config.top_k,
                config.index_dim**-0.5,
            )
            context, _ = dsa_sparse.attention(
                q, k, v, selection, config.window_size, config.sink_size, self.scaling
            )
        else:
            scores = (
                self._scores(index_queries, index_keys) if sparse or config.index_bias else q.new_empty(0)
            )
            ranking = (
                self._scores(rank_queries, rank_keys)
                if config.index_bias and config.index_detached
                else scores
            )
            keep = selected_keys(ranking.detach(), config) if sparse else None
            bias = (
                (scores * self.index_bias_scale).repeat_interleave(self.n_heads // scores.shape[1], dim=1)
                if config.index_bias
                else None
            )
            if keep is not None:
                keep = keep.repeat_interleave(self.n_heads // keep.shape[1], dim=1)
                bias = torch.zeros_like(keep, dtype=q.dtype) if bias is None else bias
                bias = bias.masked_fill(~keep, float("-inf"))
            if bias is not None and keep is None:
                positions = torch.arange(self.seq_len, device=q.device)
                bias = bias.masked_fill(positions[None, :] > positions[:, None], float("-inf"))
            context = functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=bias,
                is_causal=bias is None,
                enable_gqa=self.enable_gqa,
                scale=self.scaling,
            ).transpose(1, 2)
        if self.measure or (self.training and torch.is_grad_enabled()):
            context = self._distill(
                q, k, index_queries, index_keys, context, sparse, selection, keep, rank_queries, rank_keys
            )
        return context.reshape_as(queries)

    def _distill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        index_queries: torch.Tensor,
        index_keys: torch.Tensor,
        context: torch.Tensor,
        sparse: bool,
        selection: dsa_sparse.Selection | None,
        keep: torch.Tensor | None,
        rank_queries: torch.Tensor,
        rank_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Train sampled index rows against detached long-range teacher logits.

        Args:
            q: Attention queries [B, T, H, D].
            k: Attention keys [B, T, Hkv, D].
            index_queries: Live index queries [B, Hidx, T, index_dim].
            index_keys: Live index keys [B, Hidx, T, index_dim].
            context: Attention context [B, T, H, D].
            sparse: Whether this forward uses a sparse read.
            selection: Packed GPU selection, or None for the reference.
            keep: Reference visibility mask [B, H, T, T], or None.
            rank_queries: KL-trained index queries [B, Hidx, T, index_dim].
            rank_keys: KL-trained index keys [B, Hidx, T, index_dim].

        Returns:
            The context carrying the normalized indexer gradient during training.
        """
        config = self.attention
        first = config.window_size + config.sink_size
        eligible = max(self.seq_len - first, 0)
        if eligible == 0 or (config.kl_on_selection and sparse and self.training and config.top_k == 0):
            loss = (rank_queries.sum() + rank_keys.sum()) * 0
            return (
                self.indexer_kl(loss, context, q.shape[0] * self.seq_len)
                if self.training and torch.is_grad_enabled()
                else context
            )
        rows = torch.randperm(eligible, device=q.device)[: config.kl_query_count] + first
        positions = torch.arange(self.seq_len, device=q.device)
        mask = (positions[None, :] >= config.sink_size) & (
            positions[None, :] <= rows[:, None] - config.window_size
        )
        if config.kl_on_selection and sparse and self.training:
            if selection is not None:
                support = dsa_sparse.kept_rows(selection, rows, config.window_size, config.sink_size)
                support = support.repeat_interleave(self.n_heads // support.shape[1], dim=1)
            elif keep is not None:
                support = keep[:, :, rows]
            mask = mask & support
        student = self._scores(rank_queries[:, :, rows], rank_keys)
        student = student.repeat_interleave(self.n_heads // student.shape[1], dim=1)
        teacher = (
            q.detach().transpose(1, 2)[:, :, rows]
            @ k.detach()
            .transpose(1, 2)
            .repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            .transpose(-1, -2)
            * self.scaling
        )
        if config.index_bias:
            teacher = (
                teacher
                + (self._scores(index_queries[:, :, rows], index_keys) * self.index_bias_scale)
                .repeat_interleave(self.n_heads // index_queries.shape[1], dim=1)
                .detach()
            )
        loss, stats = masked_kl(teacher, student, mask)
        if self.measure:
            for name, value in stats.items():
                key = "dsa/" + name
                self.measurements[key] = self.measurements.get(key, 0.0) + float(value)
            self.measurements["passes"] = self.measurements.get("passes", 0.0) + 1
        return (
            self.indexer_kl(loss, context, q.shape[0] * self.seq_len)
            if self.training and torch.is_grad_enabled()
            else context
        )


def selected_keys(scores: torch.Tensor, config: AttentionConfig) -> torch.Tensor:
    """Select causal sinks, window and exact top-k long-range keys.

    Args:
        scores: Index scores [B, H, T, T].
        config: Attention knobs.

    Returns:
        Key visibility [B, H, T, T] bool.
    """
    length = scores.shape[-1]
    positions = torch.arange(length, device=scores.device)
    delta = positions[:, None] - positions[None, :]
    far = (delta >= config.window_size) & (positions[None, :] >= config.sink_size)
    ranking = scores.masked_fill(~far, float("-inf"))
    # Stable descending order keeps later keys first on exact ties
    order = ranking.flip(-1).argsort(dim=-1, descending=True, stable=True)
    chosen = length - 1 - order[..., : min(config.top_k, length)]
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, chosen, True) & far
    return keep | ((delta >= 0) & ((delta < config.window_size) | (positions[None, :] < config.sink_size)))
