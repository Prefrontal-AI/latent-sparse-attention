"""Qwen3.5's full-attention layer with the head shape, the output gate and the key and query norm as settings.

Qwen3.5 fixes eight query heads over two key-value heads of 256 dims, a rotation on the leading quarter of each
head, an output gate and a norm on each query and key head.
The sparse-attention comparison reads other shapes: one key-value head of 128 dims and no rotation, and a plain
layer with none of the three extras.
This layer is torchtitan's own with each of them switched by its config, so a shape is a number in a flavour and
not a second layer class.
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import AttentionMasksType, BaseAttention
from torchtitan.models.qwen3_5.model import Qwen35Attention


class HeadShapeAttention(Qwen35Attention):
    """Grouped-query attention whose gate, norm and rotation are optional.

    The queries, keys and values are `[T, heads, head_dim]` tokens laid end to end, as torchtitan's inner attention
    takes them.
    A config's `rope`, `q_norm` and `k_norm` are carried even when switched off, because torchtitan's sharding
    configuration reaches for them on every layer, and are built only when used.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Attention.Config):
        """Settings of the attention - Qwen3.5's, with the output gate and the QK norm switchable."""

        # Whether the query projection also writes a sigmoid gate on the context
        output_gate: bool = True
        # Whether each query and key head is normalised before the rotation
        qk_norm: bool = True

    def __init__(self, config: Config) -> None:
        """Build the projections, and the rope and the QK norms only when used.

        Qwen35Attention's `__init__` builds the rope and both norms unconditionally, so it is skipped.
        """
        # Not Qwen35Attention's, which builds the rope and both norms unconditionally
        BaseAttention.__init__(self)
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.output_gate = config.output_gate
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.scaling = self.head_dim**-0.5

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()
        if self.rotary_dim:
            self.rope = config.rope.build()
        if config.qk_norm:
            self.q_norm = config.q_norm.build()
            self.k_norm = config.k_norm.build()
        self.qk_norm = config.qk_norm
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over each token's sequence and project back to the model dim.

        Args:
            x_TD: Tokens [T, model_dim].
            attention_masks: torchtitan's masks, which the dense arm reads.
            positions: Each token's position in its sequence [T], for the rotation.

        Returns:
            The layer's output [T, model_dim].
        """
        tokens = x_TD.shape[0]
        queries = self.wq(x_TD).view(tokens, -1, self.head_dim * (2 if self.output_gate else 1))
        gate = None
        if self.output_gate:
            queries, gate = queries.chunk(2, dim=-1)
        keys = self.wk(x_TD).view(tokens, -1, self.head_dim)
        values = self.wv(x_TD).view(tokens, -1, self.head_dim)
        if self.qk_norm:
            queries, keys = self.q_norm(queries), self.k_norm(keys)
        index_heads = (queries, keys)
        if self.rotary_dim:
            rotated_queries, rotated_keys = self.rope(
                queries[..., : self.rotary_dim], keys[..., : self.rotary_dim], positions
            )
            queries = torch.cat([rotated_queries, queries[..., self.rotary_dim :]], dim=-1)
            keys = torch.cat([rotated_keys, keys[..., self.rotary_dim :]], dim=-1)
        context = self.attend(
            x_TD, queries, keys, values, attention_masks, index_heads=index_heads
        ).contiguous()
        if gate is not None:
            context = context * torch.sigmoid(gate)
        return self.wo(context.view(tokens, -1))

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
        """Read the attention context with the layer input and position-free heads available.

        Args:
            x_TD: Layer input [T, model_dim].
            queries: Rotated queries [T, Hq, head_dim].
            keys: Rotated keys [T, Hkv, head_dim].
            values: Values [T, Hkv, head_dim].
            attention_masks: The model's masks.
            index_heads: Queries and keys before rotation, or None.

        Returns:
            Context [T, Hq, head_dim].
        """
        return self.inner_attention(
            queries,
            keys,
            values,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
            attention_input=x_TD,
            index_heads=index_heads,
        )
