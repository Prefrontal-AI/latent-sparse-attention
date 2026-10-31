"""NSA's compressed read, block selection and selected read as differentiable torch ops.

On CUDA each read is a torch.library custom op over forward.py's kernel, with backward.py's as its gradient, so
torch.compile keeps it whole.
Off CUDA each is reference.py's, differentiated by autograd.
The selection has no gradient.
"""

from typing import Any

import torch

from lsa.training.native.kernels.nsa import backward, forward, reference


@torch.library.custom_op("lsa::nsa_compressed", mutates_args=())
def compressed_op(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the compressed read, or give zeros when no compressed token exists yet."""
    if compressed_keys.shape[1] == 0:
        batch_size, query_count, query_head_count, _ = q.shape
        context = q.new_zeros((batch_size, query_count, query_head_count, compressed_values.shape[-1]))
        return context, q.new_zeros((batch_size, query_count, query_head_count), dtype=torch.float32)
    q, compressed_keys, compressed_values = (x.contiguous() for x in (q, compressed_keys, compressed_values))
    return forward.compressed_attention_fwd(
        q, compressed_keys, compressed_values, compress_block_size, compress_stride, scale
    )


@compressed_op.register_fake
def _(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe tensor metadata for tracing."""
    batch_size, query_count, query_head_count, _ = q.shape
    context = q.new_empty((batch_size, query_count, query_head_count, compressed_values.shape[-1]))
    return context, q.new_empty((batch_size, query_count, query_head_count), dtype=torch.float32)


@torch.library.custom_op("lsa::nsa_compressed_backward", mutates_args=())
def compressed_backward_op(
    do: torch.Tensor,
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk and dv of the compressed keys and values of `compressed_op`."""
    if compressed_keys.shape[1] == 0:
        return torch.zeros_like(q), torch.zeros_like(compressed_keys), torch.zeros_like(compressed_values)
    q, compressed_keys, compressed_values, o, lse, do = (
        x.contiguous() for x in (q, compressed_keys, compressed_values, o, lse, do)
    )
    return backward.compressed_attention_bwd(
        q, compressed_keys, compressed_values, o, lse, do, compress_block_size, compress_stride, scale
    )


@compressed_backward_op.register_fake
def _(
    do: torch.Tensor,
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe tensor metadata for tracing."""
    return (
        q.new_empty(q.shape),
        compressed_keys.new_empty(compressed_keys.shape),
        compressed_values.new_empty(compressed_values.shape),
    )


def _compressed_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Compute the compressed setup operation."""
    q, compressed_keys, compressed_values, compress_block_size, compress_stride, scale = inputs
    o, lse = output
    ctx.save_for_backward(q, compressed_keys, compressed_values, o, lse)
    ctx.compress_block_size, ctx.compress_stride, ctx.scale = compress_block_size, compress_stride, scale


def _compressed_backward(ctx: Any, do: torch.Tensor, _dlse: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Compute the compressed backward operation."""
    q, compressed_keys, compressed_values, o, lse = ctx.saved_tensors
    dq, dk, dv = compressed_backward_op(
        do,
        q,
        compressed_keys,
        compressed_values,
        o,
        lse,
        ctx.compress_block_size,
        ctx.compress_stride,
        ctx.scale,
    )
    return dq, dk, dv, None, None, None


compressed_op.register_autograd(_compressed_backward, setup_context=_compressed_setup)


@torch.library.custom_op("lsa::nsa_select", mutates_args=())
def select_op(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    select_block_count: int,
    scale: float,
) -> torch.Tensor:
    """Select blocks by the kernel, or the forced blocks alone when no compressed token exists yet."""
    if compressed_keys.shape[1] == 0:
        return reference.select_blocks(
            q,
            compressed_keys,
            lse,
            compress_block_size,
            compress_stride,
            select_block_size,
            select_block_count,
            scale,
        )
    q, compressed_keys, lse = (x.contiguous() for x in (q, compressed_keys, lse))
    return forward.select_blocks(
        q,
        compressed_keys,
        lse,
        compress_block_size,
        compress_stride,
        select_block_size,
        select_block_count,
        scale,
    )


@select_op.register_fake
def _(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    select_block_count: int,
    scale: float,
) -> torch.Tensor:
    """Describe tensor metadata for tracing."""
    batch_size, query_count = q.shape[:2]
    return q.new_empty(
        (batch_size, query_count, compressed_keys.shape[2], select_block_count), dtype=torch.int32
    )


@torch.library.custom_op("lsa::nsa_selected", mutates_args=())
def selected_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the selected read."""
    q, k, v, indices = (x.contiguous() for x in (q, k, v, indices))
    return forward.selected_attention_fwd(q, k, v, indices, select_block_size, scale)


@selected_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe tensor metadata for tracing."""
    batch_size, query_count, query_head_count, _ = q.shape
    context = q.new_empty((batch_size, query_count, query_head_count, v.shape[-1]))
    return context, q.new_empty((batch_size, query_count, query_head_count), dtype=torch.float32)


@torch.library.custom_op("lsa::nsa_selected_backward", mutates_args=())
def selected_backward_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk and dv of `selected_op`."""
    q, k, v, o, lse, do, indices = (x.contiguous() for x in (q, k, v, o, lse, do, indices))
    return backward.selected_attention_bwd(q, k, v, o, lse, do, indices, select_block_size, scale)


@selected_backward_op.register_fake
def _(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe tensor metadata for tracing."""
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _selected_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Compute the selected setup operation."""
    q, k, v, indices, select_block_size, scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, indices)
    ctx.select_block_size, ctx.scale = select_block_size, scale


def _selected_backward(ctx: Any, do: torch.Tensor, _dlse: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Compute the selected backward operation."""
    q, k, v, o, lse, indices = ctx.saved_tensors
    dq, dk, dv = selected_backward_op(do, q, k, v, o, lse, indices, ctx.select_block_size, ctx.scale)
    return dq, dk, dv, None, None, None


selected_op.register_autograd(_selected_backward, setup_context=_selected_setup)


def compressed_attention(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the compressed tokens each query sees, as `reference.compressed_attention`.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        compressed_values: Compressed values [batch_size, compressed_key_count, kv_head_count, head_dim].
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    if not q.is_cuda:
        return reference.compressed_attention(
            q, compressed_keys, compressed_values, compress_block_size, compress_stride, scale
        )
    return compressed_op(q, compressed_keys, compressed_values, compress_block_size, compress_stride, scale)


def select_blocks(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    select_block_count: int,
    scale: float,
) -> torch.Tensor:
    """Pick the blocks each key-value head reads, as `reference.select_blocks`, with no gradient.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        lse: The compressed read's log-sum-exp [batch_size, query_count, query_head_count] fp32.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        select_block_size: Keys a selection block holds.
        select_block_count: Blocks a key-value head reads, the forced ones included.
        scale: The factor logits are scaled by.

    Returns:
        The blocks [batch_size, query_count, kv_head_count, select_block_count] int32, ascending, -1 padded.
    """
    q, compressed_keys, lse = q.detach(), compressed_keys.detach(), lse.detach()
    arguments = (compress_block_size, compress_stride, select_block_size, select_block_count, scale)
    if not q.is_cuda:
        return reference.select_blocks(q, compressed_keys, lse, *arguments)
    return select_op(q, compressed_keys, lse, *arguments)


def selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the keys of the selected blocks, as `reference.selected_attention`.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        k: Keys [batch_size, key_count, kv_head_count, head_dim].
        v: Values [batch_size, key_count, kv_head_count, head_dim].
        indices: The selected blocks [batch_size, query_count, kv_head_count, select_block_count] int32, -1 unused.
        select_block_size: Keys a selection block holds.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    if not q.is_cuda:
        return reference.selected_attention(q, k, v, indices, select_block_size, scale)
    return selected_op(q, k, v, indices, select_block_size, scale)
