"""Qwen3.5 - the research testbed, and the worked example of a model that needs code.

Two things differ from a standard HF decoder and nothing else does.
The query projection emits `2 * head_dim` per head and the second half is an output gate, so `project` is
overridden. The checkpoint is multimodal and the vision tower is dead weight, so `load` drops it.

The hybrid layer stack - 32 layers, three linear-attention layers then one full-attention layer - needs no
code at all: it declares `layer_types`, and the generic adapter reads it.

A document prefilled a chunk at a time - a teacher reading a million tokens - reads its full-attention layers
through a causal kernel aligned to the bottom right, because transformers' `sdpa` path builds a dense mask of the
chunk's queries by every cached key, 64 GiB for a 64k chunk at 1M. On a card that kernel is FlashAttention-3 off the
kernel hub, which reads the 4 key heads by their groups of query heads as they are rather than one expanded per group,
at some 2.5 times the expanded read's rate.
"""

from functools import cache
from typing import Any

import torch

from lsa.evaluation.adapter.layer import Projection, SparseAttention, heads
from lsa.evaluation.adapter.model import ModelAdapter, torch_dtype
from lsa.evaluation.config import ModelConfig
from lsa.evaluation.constants import Layer


@cache
def flash_kernel() -> Any:
    """Return FlashAttention-3 off the kernel hub, at the revision pinned - loaded once a process."""
    from kernels import get_kernel

    return get_kernel(Layer.FLASH_KERNEL, revision=Layer.FLASH_KERNEL_REVISION)


def lower_right_causal_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attend a chunk of queries to every key up to its own position, each key head read by its group of query heads.

    The chunk is the last `query_count` positions of the keys, so the causal rule is aligned to the bottom right and
    no mask is built. On a card FlashAttention-3 reads the grouped heads natively. Off one - a test - torch's own
    kernel reads one key-value group at a time, its key head expanded to the group.

    Args:
        module: The attention module, unused.
        query: Query heads [B, Hq, Tq, dh].
        key: Key heads [B, Hkv, Tk, dh], the chunk's own last.
        value: Value heads [B, Hkv, Tk, dh].
        attention_mask: None - a chunk with padding is refused.
        dropout: Unused - the teacher runs in eval mode.
        scaling: The model's attention scale.
        kwargs: What transformers passes besides, unused.

    Returns:
        attention_output: The read [B, Tq, Hq, dh].
        attention_weights: None - no weights are kept.

    Raises:
        ValueError: A padding mask arrived, which a bottom-right causal read cannot honour.
    """
    if attention_mask is not None:
        raise ValueError("a lower-right causal read takes no padding mask")
    if query.is_cuda:
        # The kernel takes [B, T, H, dh], and its causal rule is already the bottom right's when Tq < Tk
        output = flash_kernel().flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            softmax_scale=scaling,
            causal=True,
        )
        return (output[0] if isinstance(output, tuple) else output), None
    from torch.nn.attention.bias import causal_lower_right

    kv_head_count = key.shape[1]
    groups = query.shape[1] // kv_head_count
    causal = causal_lower_right(query.shape[2], key.shape[2])
    outputs = [
        torch.nn.functional.scaled_dot_product_attention(
            query[:, head * groups : (head + 1) * groups],
            key[:, head : head + 1].expand(-1, groups, -1, -1),
            value[:, head : head + 1].expand(-1, groups, -1, -1),
            attn_mask=causal,
            scale=scaling,
        )
        for head in range(kv_head_count)
    ]
    return torch.cat(outputs, dim=1).transpose(1, 2).contiguous(), None


class Qwen35SparseAttention(SparseAttention):
    """Qwen3.5's attention, whose query projection carries the output gate."""

    def project(self, hidden_states: torch.Tensor) -> Projection:
        """Unpack the query projection into queries and a gate, then the usual keys and values.

        Args:
            hidden_states: Layer input [B, T, model_dim].

        Returns:
            The layer's pre-RoPE query, key and value heads, and its output gate.
        """
        attention = self.attention
        batch_size, query_count = hidden_states.shape[:2]
        head_dim = attention.head_dim

        # [B, T, Hq, 2 * dh] - the gate is the second half of every head's slice
        query_and_gate = attention.q_proj(hidden_states).view(batch_size, query_count, -1, head_dim * 2)
        # [B, T, Hq * dh] gate, applied after the read
        gate = query_and_gate[..., head_dim:].reshape(batch_size, query_count, -1)

        q = attention.q_norm(query_and_gate[..., :head_dim].transpose(1, 2))
        k = attention.k_norm(heads(attention.k_proj(hidden_states), head_dim))
        v = heads(attention.v_proj(hidden_states), head_dim)
        return Projection(q=q, k=k, v=v, gate=gate)


class Qwen35Adapter(ModelAdapter):
    """Qwen3.5 - a multimodal checkpoint whose vision tower this harness never reaches."""

    attention_class = Qwen35SparseAttention

    def load(
        self,
        model_config: ModelConfig,
        device: str,
        attention: str = "eager",
        span: tuple[int, int] | None = None,
    ) -> tuple[Any, Any]:
        """Load the checkpoint and its tokenizer, with the vision tower dropped.

        Args:
            model_config: Which checkpoint, and in what dtype.
            device: Torch device string the weights land on.
            attention: The native attention implementation.
            span: Must be None - this stack is not run a span at a time.

        Returns:
            model: The loaded model in eval mode.
            tokenizer: Its tokenizer.

        Raises:
            NotImplementedError: A span is asked for.
        """
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        if span is not None:
            raise NotImplementedError(f"{type(self).__name__} cannot run its stack a span at a time")

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_config.checkpoint,
            config=self.pretrained_config(model_config),
            dtype=torch_dtype(model_config.dtype),
            attn_implementation=attention,
            device_map=device,
        )
        model.eval()
        model.model.visual = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, AutoTokenizer.from_pretrained(model_config.checkpoint)

    def prepare_long_prefill(self, model: Any) -> None:
        """Read the full-attention layers through a bottom-right causal kernel, so a chunk builds no dense mask.

        Args:
            model: The loaded model, whose decoder stack is switched in place.
        """
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

        AttentionInterface.register(Layer.LOWER_RIGHT_CAUSAL, lower_right_causal_attention)
        # The mask a flash kernel takes - None unless a token is padding
        AttentionMaskInterface.register(Layer.LOWER_RIGHT_CAUSAL, flash_attention_mask)
        self.text_model(model).config._attn_implementation = Layer.LOWER_RIGHT_CAUSAL
