"""Read PISA-selected blocks within each packed sequence."""

from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.models.common.attention import FlexInnerAttention

from lsa.training.native.config import AttentionConfig
from lsa.training.native.kernels import pisa
from lsa.training.native.kernels.nsa import autograd, reference


class PisaInnerAttention(FlexInnerAttention):
    """Select a positional pyramid and differentiate the selected read."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        """Settings of the attention: the attention knobs and the packed sequence length."""

        attention: AttentionConfig
        seq_len: int

    def __init__(self, config: Config) -> None:
        """Keep the pyramid settings and the packed sequence length."""
        super().__init__(config)
        self.attention = config.attention
        self.seq_len = config.seq_len

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
        """Read the selected blocks independently within each sequence.

        Args:
            q_THK: Queries [T, H, K].
            k_THK: Keys [T, Hkv, K].
            v_THV: Values [T, Hkv, V].
            attention_masks: Trainer masks, superseded by causal selection.
            scale: Attention scale, or inverse square root of head dim.
            enable_gqa: Whether query heads share keys.
            **kwargs: Additional trainer attention options.

        Returns:
            Selected context [T, H, V].
        """
        queries, keys, values = (
            tensor.reshape(-1, self.seq_len, *tensor.shape[1:]) for tensor in (q_THK, k_THK, v_THV)
        )
        scale = queries.shape[-1] ** -0.5 if scale is None else scale
        config = self.attention
        indices = pisa.select_blocks(
            queries,
            keys,
            config.leaf_block_size,
            config.branching,
            config.level_block_count,
            scale,
            0,
            config.local_block_count,
        )
        block, dim = config.leaf_block_size, queries.shape[-1]
        group = queries.shape[2] // keys.shape[2]
        kernel_shapes = (
            block >= 16
            and block & (block - 1) == 0
            and 16 <= dim <= 256
            and dim & (dim - 1) == 0
            and group <= 32
            and group & (group - 1) == 0
        )
        read = (
            autograd.selected_attention if queries.is_cuda and kernel_shapes else reference.selected_attention
        )
        context, _ = read(queries, keys, values, indices, block, scale)
        return context.reshape(q_THK.shape[0], q_THK.shape[1], values.shape[-1])
