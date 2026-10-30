"""Train NSA with separate compressed, selected and sliding branches on torchtitan's attention layer."""

from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import spmd_types as spmd
import torch
from torch import nn
from torch.nn import functional as functional
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder_sharding import dense_activation_placement, dense_param_placement
from torchtitan.protocols.module import Module

from lsa.training.native.config import AttentionConfig
from lsa.training.native.heads import HeadShapeAttention
from lsa.training.native.kernels.nsa import autograd as kernels
from lsa.training.native.kernels.nsa.reference import compressed_count


class Compression(Module):
    """Compress an overlapping block with a two-layer SiLU network."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Settings of the compression: its input and output projections."""

        input_projection: Linear.Config
        output_projection: Linear.Config

    def __init__(self, config: Config) -> None:
        """Build the block and head projections under torchtitan's module protocol."""
        super().__init__()
        self.input_projection = config.input_projection.build()
        self.output_projection = config.output_projection.build()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project concatenated block vectors through SiLU to one head vector.

        Args:
            values: Block vectors [..., compress_block_size * head_dim].

        Returns:
            Compressed vectors [..., head_dim].
        """
        return self.output_projection(functional.silu(self.input_projection(values)))


class NsaAttention(HeadShapeAttention):
    """Mix three independently projected reads with a learned gate for every query head."""

    @dataclass(kw_only=True, slots=True)
    class Config(HeadShapeAttention.Config):
        """Settings of the attention: its projections, head shape, attention knobs and sequence length."""

        attention: AttentionConfig
        seq_len: int

    def __init__(self, config: Config) -> None:
        """Build the projections and compression networks of the three branches."""
        if config.rotary_dim:
            raise ValueError(f"NSA requires model.rotary_dim 0 - got {config.rotary_dim}")
        super().__init__(config)
        self.attention = config.attention
        self.seq_len = config.seq_len
        self.branch_kv = replace(config.wk, out_features=4 * self.n_kv_heads * self.head_dim).build()
        self.branch_projection = replace(config.wq, out_features=3 * self.n_heads).build()
        hidden_dim = config.attention.compress_hidden_dim or self.head_dim
        block_size = config.attention.compress_block_size
        self.key_positions = nn.Parameter(torch.zeros(block_size, self.head_dim))
        self.value_positions = nn.Parameter(torch.zeros(block_size, self.head_dim))
        self.key_compression = self._compression(config.wk, block_size, hidden_dim)
        self.value_compression = self._compression(config.wv, block_size, hidden_dim)
        if self.qk_norm:
            self.compressed_key_norm = config.k_norm.build()
            self.window_key_norm = config.k_norm.build()
        self.read_local: Callable[[torch.Tensor], torch.Tensor] = self._read

    def reset_parameters(self) -> None:
        """Initialise the within-block position embeddings after materialising the model."""
        nn.init.zeros_(self.key_positions)
        nn.init.zeros_(self.value_positions)

    def _compression(self, template: Linear.Config, block_size: int, hidden_dim: int) -> Compression:
        """Build a two-layer compression network with the model's projection initialisation."""
        return Compression.Config(
            input_projection=replace(
                template, in_features=block_size * self.head_dim, out_features=hidden_dim
            ),
            output_projection=replace(template, in_features=hidden_dim, out_features=self.head_dim),
        ).build()

    def compress(self, values: torch.Tensor, positions: torch.Tensor, network: nn.Module) -> torch.Tensor:
        """Compress every complete overlapping block into one head vector.

        Args:
            values: Branch keys or values [B, T, Hkv, head_dim].
            positions: Within-block position embeddings [compress_block_size, head_dim].
            network: The branch's compression network.

        Returns:
            Compressed vectors [B, compressed_count, Hkv, head_dim].
        """
        block_size, stride = self.attention.compress_block_size, self.attention.compress_stride
        count = compressed_count(values.shape[1], block_size, stride)
        if not count:
            return values[:, :0]
        blocks = values.unfold(1, block_size, stride).transpose(-1, -2)
        return network((blocks + positions.to(values.dtype)).flatten(-2))

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Replicate block positions and put the three reads inside one local SPMD region.

        Args:
            parallel_dims: The trainer's data-parallel mesh degrees.
        """
        if self._sharding_config is not None:
            placement = dense_param_placement(tp=spmd.R)
            self._sharding_config.state_shardings.update(key_positions=placement, value_positions=placement)
        super().parallelize(parallel_dims)
        layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        contract = (layout.local_type, layout.partition_spec)
        self.read_local = spmd.no_typecheck(in_types=(contract,), out_types=contract)(self._read)

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend independently within each packed sequence and project the gated mixture.

        Args:
            x_TD: Layer input [T, model_dim].
            attention_masks: Trainer masks, superseded by causal branch reads.
            positions: Token positions, unused for position-free queries.

        Returns:
            Layer output [T, model_dim].
        """
        return self.read_local(x_TD)

    def _read(self, x_TD: torch.Tensor) -> torch.Tensor:
        """Compute the compressed, selected and sliding branches on local tensors.

        Args:
            x_TD: Layer input [T, model_dim].

        Returns:
            Layer output [T, model_dim].
        """
        tokens = x_TD.shape[0]
        queries = self.wq(x_TD).view(
            -1, self.seq_len, self.n_heads, self.head_dim * (2 if self.output_gate else 1)
        )
        gate = None
        if self.output_gate:
            queries, gate = queries.chunk(2, dim=-1)
        keys = self.wk(x_TD).view(-1, self.seq_len, self.n_kv_heads, self.head_dim)
        values = self.wv(x_TD).view_as(keys)
        branches = self.branch_kv(x_TD).view(-1, self.seq_len, 4, self.n_kv_heads, self.head_dim)
        compressed_keys, compressed_values, window_keys, window_values = branches.unbind(2)
        if self.qk_norm:
            queries, keys = self.q_norm(queries), self.k_norm(keys)
            compressed_keys = self.compressed_key_norm(compressed_keys)
            window_keys = self.window_key_norm(window_keys)
        compressed_keys = self.compress(compressed_keys, self.key_positions, self.key_compression)
        compressed_values = self.compress(compressed_values, self.value_positions, self.value_compression)
        config = self.attention
        compressed, logsumexp = kernels.compressed_attention(
            queries,
            compressed_keys,
            compressed_values,
            config.compress_block_size,
            config.compress_stride,
            self.scaling,
        )
        selected_blocks = kernels.select_blocks(
            queries,
            compressed_keys,
            logsumexp,
            config.compress_block_size,
            config.compress_stride,
            config.select_block_size,
            config.select_block_count,
            self.scaling,
        )
        selected, _ = kernels.selected_attention(
            queries,
            keys,
            values,
            selected_blocks,
            config.select_block_size,
            self.scaling,
        )
        window = sliding_attention(queries, window_keys, window_values, config.window_size, self.scaling)
        gates = self.branch_projection(x_TD).view(-1, self.seq_len, self.n_heads, 3).sigmoid()
        context = compressed * gates[..., :1] + selected * gates[..., 1:2] + window * gates[..., 2:]
        if gate is not None:
            context = context * gate.sigmoid()
        return self.wo(context.reshape(tokens, -1))


def sliding_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, window_size: int, scale: float
) -> torch.Tensor:
    """Read the causal window, using flash attention on CUDA and a dense reference on CPU.

    FlashAttention-2 reads it on Hopper and earlier, FlashAttention-4 on Blackwell.

    Args:
        queries: Queries [B, T, H, head_dim].
        keys: Window keys [B, T, Hkv, head_dim].
        values: Window values [B, T, Hkv, head_dim].
        window_size: Keys in the causal window.
        scale: Attention scale.

    Returns:
        Window context [B, T, H, head_dim].
    """
    if queries.is_cuda and torch.cuda.get_device_capability(queries.device)[0] >= 10:
        from flash_attn.cute import flash_attn_func as flash_attn_4_func

        # Returns the output and the log-sum-exp
        return flash_attn_4_func(
            queries, keys, values, causal=True, window_size=(window_size - 1, 0), softmax_scale=scale
        )[0]
    if queries.is_cuda:
        from flash_attn import flash_attn_func

        return flash_attn_func(
            queries, keys, values, causal=True, window_size=(window_size - 1, 0), softmax_scale=scale
        )
    positions = torch.arange(queries.shape[1], device=queries.device)
    distance = positions[:, None] - positions[None]
    mask = (distance >= 0) & (distance < window_size)
    return functional.scaled_dot_product_attention(
        queries.transpose(1, 2),
        keys.transpose(1, 2),
        values.transpose(1, 2),
        attn_mask=mask,
        scale=scale,
        enable_gqa=True,
    ).transpose(1, 2)


def attention_config(
    template: HeadShapeAttention.Config, attention: AttentionConfig, seq_len: int
) -> NsaAttention.Config:
    """Carry the model's attention shape and initialisation into the NSA layer config."""
    return NsaAttention.Config(
        **{
            field.name: getattr(template, field.name)
            for field in fields(template)
            if not field.name.startswith("_")
        },
        attention=attention,
        seq_len=seq_len,
    )
