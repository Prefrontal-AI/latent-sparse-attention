"""Qwen3.5 with its full-attention layers swapped for the arm a run names.

Qwen3.5 is a torchtitan model whose full-attention layers - one in four, the rest are GatedDeltaNet - take their
kernel from a config field. A native run changes that field and nothing else: the projections, the norms, the
partial RoPE, the output gate and the recurrent layers are the model's own, so the arms differ in the attention
and in nothing a reader has to hunt for.

Two flavours are the same model family at the GPT-2 vocabulary: the hybrid at Qwen3.5-0.8B's shape with tied
embeddings, and a 340M model of nothing but plain attention and SwiGLU. A run may also name the full-attention
layers' head shape, which is how the multi-query shape the sparse arms were compared on is asked for.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any
from unittest import mock

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.distributed import fsdp as titan_fsdp
from torchtitan.models.common import Linear
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

from lsa.training.native.config import ModelConfig, NativeConfig
from lsa.training.native.constants import Backend, Flavor, Kind
from lsa.training.native.dsa import DsaAttention
from lsa.training.native.gdn import Gpt2InnerGatedDeltaNet, Gpt2RMSNormGated
from lsa.training.native.heads import HeadShapeAttention
from lsa.training.native.initialization import initialize_gated_decay, initialize_gated_step_bias
from lsa.training.native.kernels import fa4
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.lsa_indexer import LsaInnerAttention
from lsa.training.native.nsa import attention_config as nsa_config
from lsa.training.native.optimizer import preserve_parameter_flags
from lsa.training.native.pisa import PisaInnerAttention

# What the 340M flavour is, which the 0.8B hybrid's own config does not say
DENSE_LAYER_COUNT = 19
DENSE_HIDDEN_DIM = 2736
DENSE_HEAD_COUNT = 8


class OffsetGatedDeltaKernel(GatedDeltaKernel):
    """GatedDeltaNet's kernel, given document offsets even when its microbatch holds one sequence.

    torchtitan passes no offsets for one sequence, and attn_gym's chunked kernel then takes its dense path, which is
    non-finite from 256 tokens on. The packed path is right at every length, so a lone sequence gets the one offset pair
    that spans it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        """Settings of the kernel - the base's, with none added."""

        pass

    def forward(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the recurrence over packed tokens, with offsets spanning them all when none are given.

        Args:
            xq_THK: Queries [T, H, K] bf16.
            xk_THK: Keys [T, H, K] bf16.
            xv_THV: Values [T, H, V] bf16.
            g_TH: Log decay [T, H] fp32.
            beta_TH: Write gate [T, H] bf16.
            cu_seqlens: Document offsets [N + 1] int32, or None for one sequence.

        Returns:
            The recurrence's output [T, H, V] bf16.
        """
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, xq_THK.shape[0]], dtype=torch.int32, device=xq_THK.device)
        return super().forward(xq_THK, xk_THK, xv_THV, g_TH, beta_TH, cu_seqlens=cu_seqlens)


def register_contiguous_gdn_fakes() -> None:
    """Make the fakes of attn_gym's chunked GatedDeltaNet forward and backward return contiguous outputs, as the kernels do.

    The library's fakes allocate each output like the input it matches, so when the compiler hands the op a tensor
    laid out heads-first the fake promises a layout the kernel never returns, and the compiled step fails its stride check.
    """

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cumulative_gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an empty output like the values and an empty inverse, each contiguous."""
        inverse = q.new_empty(q.shape[0], q.shape[1], v.shape[2], 64)
        return q.new_empty(v.shape), inverse

    def backward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cumulative_gate: torch.Tensor,
        beta: torch.Tensor,
        inverse: torch.Tensor,
        d_output: torch.Tensor,
        d_final_state: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        chunk_offsets: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, ...]:
        """Return an empty contiguous gradient for each of q, k, v, the cumulative gate and beta."""
        return tuple(
            torch.empty_like(tensor, memory_format=torch.contiguous_format)
            for tensor in (q, k, v, cumulative_gate, beta)
        )

    torch.library.register_fake("attn_gym::gdn_chunk_fwd", forward, allow_override=True)
    torch.library.register_fake("attn_gym::gdn_chunk_bwd", backward, allow_override=True)


@contextmanager
def plain_parameter_fsdp() -> Iterator[None]:
    """Shard data-parallel parameters with FSDP2's plain-tensor path inside the block.

    Torchtitan annotates each parameter with its SPMD layout and hands `fully_shard` the data-parallel axes
    as `dp_mesh_dims`, which this torch (2.13) accepts only for parameters that are already DTensors on the full
    mesh. Torchtitan leaves them as plain local tensors, so every multi-rank `fully_shard` fails with "Got plain
    tensor for parameter". A data-parallel mesh holds the whole parameter on each rank before sharding, so FSDP2's
    own plain path shards and reduces the same bytes and returns plain tensors to the forward, which the SPMD
    local regions expect. Tensor and context parallelism need the SPMD mesh, so a mesh with those axes passes
    through unchanged.

    Yields:
        Control while `fully_shard` ignores `dp_mesh_dims` on a data-parallel mesh.
    """
    fully_shard = titan_fsdp.fully_shard

    def shard(module: Any, *, mesh: DeviceMesh, dp_mesh_dims: Any = None, **kwargs: Any) -> Any:
        names = set(mesh.mesh_dim_names or ())
        if names <= {"dp_replicate", "dp_shard"}:
            dp_mesh_dims = None
        return fully_shard(module, mesh=mesh, dp_mesh_dims=dp_mesh_dims, **kwargs)

    with mock.patch.object(titan_fsdp, "fully_shard", shard):
        yield


def parallelize_plain(model: nn.Module, *, parallelize: Callable[..., nn.Module], **kwargs: Any) -> nn.Module:
    """Parallelize a model with data-parallel parameters sharded as plain tensors.

    Args:
        model: The model before parallelism.
        parallelize: Torchtitan's model parallelization function.
        **kwargs: The trainer's parallelization arguments.

    Returns:
        The parallelized model.
    """
    with plain_parameter_fsdp():
        return parallelize(model, **kwargs)


def model_spec(experiment: NativeConfig) -> ModelSpec:
    """Build the model this run trains, text-only when it asks for that, with its attention swapped in.

    Dense attention keeps torchtitan's own attention, on the kernel the run names, so that arm is the model as
    published.

    Args:
        experiment: The experiment.

    Returns:
        The torchtitan model spec.

    Raises:
        ValueError: The head shape asked for is not one the flavour's layers can have.
    """
    register_contiguous_gdn_fakes()
    flavor = experiment.model.flavor
    # torchtitan's varlen attention activates FA4 on Blackwell, whose shim needs the trimmed interface
    if (
        experiment.attention.backend == Backend.VARLEN
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10
    ):
        fa4.register()
    built = model_registry(
        Flavor.QWEN35_0_8B if flavor in Flavor.GPT2 else flavor,
        seq_len=experiment.data.seq_len,
        attn_backend=experiment.attention.backend,
    )
    built.parallelize_fn = partial(
        preserve_parameter_flags, parallelize=partial(parallelize_plain, parallelize=built.parallelize_fn)
    )
    if experiment.model.text_only:
        built.model.vision_encoder = None
    if flavor in Flavor.GPT2:
        gpt2_model(built.model, flavor)
    shape_heads(built.model, experiment.model)
    for layer in built.model.layers:
        if layer.delta_net is not None:
            layer.delta_net.inner_gated_delta_net.kernel = OffsetGatedDeltaKernel.Config()
    if experiment.attention.kind == Kind.DENSE:
        return built
    for layer in built.model.layers:
        if layer.attention is not None:
            if experiment.attention.kind == Kind.DSA:
                layer.attention = DsaAttention.Config(
                    **{
                        field.name: getattr(layer.attention, field.name)
                        for field in fields(HeadShapeAttention.Config)
                    },
                    attention=experiment.attention,
                    seq_len=experiment.data.seq_len,
                )
                continue
            if experiment.attention.kind == Kind.NSA:
                layer.attention = nsa_config(layer.attention, experiment.attention, experiment.data.seq_len)
                continue
            if experiment.attention.kind == Kind.PISA:
                layer.attention.inner_attention = PisaInnerAttention.Config(
                    attention=experiment.attention, seq_len=experiment.data.seq_len
                )
                continue
            if experiment.attention.kind == Kind.LSA:
                template = layer.attention
                layer.attention = shaped_attention(
                    template,
                    query_head_count=template.n_heads,
                    kv_head_count=template.n_kv_heads,
                    head_dim=template.head_dim,
                    rotary_dim=template.rotary_dim,
                    output_gate=getattr(template, "output_gate", True),
                    qk_norm=getattr(template, "qk_norm", True),
                )
                layer.attention.inner_attention = LsaInnerAttention.Config(
                    attention=experiment.attention,
                    seq_len=experiment.data.seq_len,
                    rotary_dim=template.rotary_dim,
                    model_dim=template.wq.in_features,
                    query_head_count=template.n_heads,
                    kv_head_count=template.n_kv_heads,
                    head_dim=template.head_dim,
                    steps=experiment.run.optim.steps,
                    decay_ratio=experiment.run.optim.decay_ratio,
                )
                continue
            layer.attention.inner_attention = MultiresInnerAttention.Config(
                attention=experiment.attention,
                seq_len=experiment.data.seq_len,
                rotary_dim=layer.attention.rotary_dim,
            )
    return built


def gpt2_model(config: Any, flavor: str) -> None:
    """Turn the 0.8B hybrid's config into a flavour at the GPT-2 vocabulary, as the reference training code trained it.

    The hybrid keeps Qwen3.5-0.8B's layers and ties its embeddings. The 340M flavour is nineteen blocks of plain
    attention and a SwiGLU, untied. Both draw their linear weights from N(0, 0.02), with the layers that write into the
    residual stream scaled by the depth, and rotate by half-split RoPE at its original base where the 340M has any.

    Args:
        config: The Qwen3.5 model config, edited in place.
        flavor: One of `Flavor.GPT2`.
    """
    vocab_size = Flavor.GPT2_VOCAB_SIZE
    normal = {"weight": partial(nn.init.normal_, std=Flavor.GPT2_INIT_STD), "bias": nn.init.zeros_}
    hybrid = flavor == Flavor.HYBRID_GPT2
    config.vocab_size = vocab_size
    config.enable_weight_tying = hybrid
    config.tok_embeddings = replace(config.tok_embeddings, num_embeddings=vocab_size, param_init=normal)
    config.lm_head = replace(config.lm_head, out_features=vocab_size, param_init=normal)
    if hybrid:
        for layer in config.layers:
            if layer.delta_net is not None:
                layer.delta_net.inner_gated_delta_net = Gpt2InnerGatedDeltaNet.Config(
                    **{
                        field.name: getattr(layer.delta_net.inner_gated_delta_net, field.name)
                        for field in fields(layer.delta_net.inner_gated_delta_net)
                    }
                )
                layer.delta_net.norm = Gpt2RMSNormGated.Config(
                    **{
                        field.name: getattr(layer.delta_net.norm, field.name)
                        for field in fields(layer.delta_net.norm)
                    }
                )
                layer.delta_net.param_init = {
                    **(layer.delta_net.param_init or {}),
                    "A_log": initialize_gated_decay,
                    "dt_bias": initialize_gated_step_bias,
                }
    else:
        config.layers = dense_layers(config)
    depth_std = Flavor.GPT2_INIT_STD / math.sqrt(2 * len(config.layers))
    residual = {"weight": partial(nn.init.normal_, std=depth_std), "bias": nn.init.zeros_}
    for name, linear, parent, field in config.traverse(Linear.Config):
        if not name.startswith("layers."):
            continue
        setattr(
            parent,
            field,
            replace(
                linear, param_init=residual if name.rsplit(".", 1)[-1] in ("wo", "w2", "out_proj") else normal
            ),
        )


def dense_layers(config: Any) -> list[Qwen35TransformerBlock.Config]:
    """Build the 340M flavour's blocks from the hybrid's attention and feed-forward as templates.

    Args:
        config: The hybrid's model config.

    Returns:
        Nineteen blocks of attention with no gate and no norm on its heads, rotating every head dim, and a SwiGLU.
    """
    block = next(layer for layer in config.layers if layer.attention is not None)
    template = block.attention
    dim = config.dim
    layers = []
    for _ in range(DENSE_LAYER_COUNT):
        attention = shaped_attention(
            template,
            query_head_count=DENSE_HEAD_COUNT,
            kv_head_count=DENSE_HEAD_COUNT,
            head_dim=dim // DENSE_HEAD_COUNT,
            rotary_dim=dim // DENSE_HEAD_COUNT,
            output_gate=False,
            qk_norm=False,
        )
        attention.rope = replace(attention.rope, theta=Flavor.GPT2_ROPE_THETA)
        feed_forward = make_ffn_config(
            dim=dim, hidden_dim=DENSE_HIDDEN_DIM, w1_param_init=None, w2w3_param_init=None
        )
        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention,
                feed_forward=feed_forward,
                attention_norm=replace(block.attention_norm),
                ffn_norm=replace(block.ffn_norm),
            )
        )
    return layers


def shape_heads(config: Any, model: ModelConfig) -> None:
    """Give the full-attention layers the head shape a run names, keeping what it leaves unnamed.

    Args:
        config: The Qwen3.5 model config, edited in place.
        model: The run's model block.

    Raises:
        ValueError: The query heads are not a whole number of groups over the key-value heads, or the rotation is
            wider than a head.
    """
    asked = (model.query_head_count, model.kv_head_count, model.head_dim, model.rotary_dim)
    if all(knob is None for knob in asked) and not model.attention_bias:
        return
    for layer in config.layers:
        attention = layer.attention
        if attention is None:
            continue
        if model.attention_bias:
            for name in ("wq", "wk", "wv", "wo"):
                projection = getattr(attention, name)
                initializers = dict(projection.param_init or {})
                initializers.setdefault("weight", partial(nn.init.normal_, std=0.02))
                initializers["bias"] = nn.init.zeros_
                setattr(attention, name, replace(projection, bias=True, param_init=initializers or None))
        query_head_count = model.query_head_count or attention.n_heads
        kv_head_count = model.kv_head_count or attention.n_kv_heads
        head_dim = model.head_dim or attention.head_dim
        rotary_dim = attention.rotary_dim if model.rotary_dim is None else model.rotary_dim
        if query_head_count % kv_head_count or rotary_dim > head_dim:
            raise ValueError(
                f"{query_head_count} query heads over {kv_head_count} key-value heads of {head_dim} dims, rotating "
                f"{rotary_dim}, is not a head shape - the groups must be whole and the rotation within the head"
            )
        layer.attention = shaped_attention(
            attention,
            query_head_count=query_head_count,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            output_gate=getattr(attention, "output_gate", True),
            qk_norm=getattr(attention, "qk_norm", True),
        )


def shaped_attention(
    template: Any,
    *,
    query_head_count: int,
    kv_head_count: int,
    head_dim: int,
    rotary_dim: int,
    output_gate: bool,
    qk_norm: bool,
) -> HeadShapeAttention.Config:
    """Build one layer's attention config at a head shape, from another layer's as a template.

    Args:
        template: A full-attention layer's config, whose inner attention, rotation and initialisation carry over.
        query_head_count: Query heads.
        kv_head_count: Key-value heads.
        head_dim: A head's dim.
        rotary_dim: The leading head dims rotated, 0 for none.
        output_gate: Whether the query projection also writes a sigmoid gate on the context.
        qk_norm: Whether each query and key head is normalised.

    Returns:
        The layer's config.
    """
    query_dim = query_head_count * head_dim
    half = rotary_dim // 2
    # Three sections of the rotated dims, as Qwen3.5 splits its multimodal positions
    sections = [half // 3 + (i < half % 3) for i in range(3)]
    return HeadShapeAttention.Config(
        n_heads=query_head_count,
        n_kv_heads=kv_head_count,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=replace(template.rope, dim=rotary_dim, mrope_section=sections),
        wq=replace(template.wq, out_features=query_dim * (2 if output_gate else 1)),
        wk=replace(template.wk, out_features=kv_head_count * head_dim),
        wv=replace(template.wv, out_features=kv_head_count * head_dim),
        wo=replace(template.wo, in_features=query_dim),
        q_norm=replace(template.q_norm, dim=head_dim),
        k_norm=replace(template.k_norm, dim=head_dim),
        inner_attention=template.inner_attention,
        output_gate=output_gate,
        qk_norm=qk_norm,
    )
