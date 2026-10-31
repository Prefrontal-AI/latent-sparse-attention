"""Gather_sparse.attention's read by the CUDA kernels of the sparse-attention campaigns (`vendor`).

The forward is fwd_r7's planned, persistent kernel, or up to 4,096 tokens fwd_hq16_hkv2's, the
same design tuned by its campaign and there up to twice as fast; the backward bwd_r1's key-major
kernel where it is faster than gather_sparse's key-side backward, else gather_sparse's, which
needs only the forward's output and log-sum-exp. The CUDA kernels read the union of a 128-row
query tile's selected keys a 128-key block at a time, so their cost grows with the keys a tile's
union covers, near the whole causal prefix for LSA's 1,024 picks a query on random data, while
gather_sparse's grows with t * top_k. On an H200
the CUDA backward wins up to 16,384 tokens a sequence for eight query heads a key head and up to
its fast path's 51,200 for one; past 51,200 its fallback is far slower than gather_sparse's.

Under v2 (the `cuda_v2` read, LSA's default), eight query heads a 128-wide key
head read by the later local campaigns' kernels instead: fwd_hard's forward, and bwd_v14's
backward, or from HYBRID_TOKENS bwd_hybrid's (v14 or a key-major hybrid, chosen on the GPU each
call) where t is a multiple of 64. Elsewhere, and past V2_TOKENS, v2 reads as without it.

The positions are sanitised first, as the kernels require: a slot outside the read
gather_sparse.attention documents, a window_size, sink_size, future or PAD position, becomes -1. A
position listed twice is not removed (lsa_select lists none twice).
"""

import importlib
import os
import shutil
import sys
from types import ModuleType
from typing import Any

import torch
import triton
import triton.language as tl

from lsa.training.native.kernels import gather_sparse

HEAD_DIMS = (64, 128)
# Most query heads a key head shares one selection over; the backward needs a divisor of 64
GROUPS = (1, 2, 4, 8, 16, 32, 64)
# The CUDA backward's fast path's longest t, and the t times query heads a group up to which it
# Beats gather_sparse's backward (H200, top_k 1024)
CUDA_BACKWARD_TOKENS = 51200
CUDA_BACKWARD_ROWS = 131072
# The longest t fwd_hq16_hkv2 reads, at d 128 and the groups it was checked at; fwd_r7 past it
SHORT_FORWARD_TOKENS = 4096
SHORT_FORWARD_GROUPS = (1, 8)
SANITIZE_ROWS = 4
# fwd_r7's plan kernel holds 9 words a 128-key block of the sequence in shared memory
# An H200 gives a block 227 KiB, so it launches up to 826,368 tokens
PLAN_SHARED_BYTES = 227 * 1024
# The v2 kernels' head dim and group, and the longest t they were checked at
V2_HEAD_DIM, V2_GROUP, V2_TOKENS = 128, 8, 65536
# The longest t the v2 backward reads; past it the backward is the current one's
V2_BACKWARD_TOKENS = 65536
# bwd_hybrid's t must be a multiple of it; bwd_v14 takes the rest
HYBRID_MULTIPLE = 64
# bwd_hybrid from this t: below it its dispatcher runs v14 anyway and holds both pipelines'
# Buffers, 1.5% slower at 16,384; at 32,768 it is 1.19x faster
HYBRID_TOKENS = 32768
_VENDOR: dict[str, ModuleType] = {}


def _vendor(name: str) -> ModuleType:
    """The vendored module, built on first use."""
    if name not in _VENDOR:
        if shutil.which("ninja") is None:
            # Load_inline builds with ninja, which a venv holds beside its python
            os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        _VENDOR[name] = importlib.import_module(f"lsa.training.native.kernels.vendor.{name}")
    return _VENDOR[name]


def plan_fits(t: int) -> bool:
    """Check whether fwd_r7's plan of a t-token sequence fits a block's shared memory."""
    blocks = -(-t // 128)
    return (2 * blocks * 4 + blocks + 3) // 4 * 4 * 4 <= PLAN_SHARED_BYTES


def supported(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Check whether the CUDA kernels take these queries and keys.

    They take sm_90, bf16, d 64 or 128, and a sequence short enough for the forward's plan (plan_fits).
    """
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        return False
    group = q.shape[2] // k.shape[2]
    return (
        q.dtype == torch.bfloat16 and q.shape[-1] in HEAD_DIMS and group in GROUPS and plan_fits(q.shape[1])
    )


def v2_runs(t: int, group: int, d: int) -> bool:
    """Check whether the v2 kernels take this shape."""
    return d == V2_HEAD_DIM and group == V2_GROUP and t <= V2_TOKENS


def forward_kernel(t: int, group: int, d: int, v2: bool = False) -> str:
    """Return the vendored forward for this shape, by the faster on an H200, or fwd_hard's under v2."""
    if v2 and v2_runs(t, group, d):
        return "fwd_hard"
    short = t <= SHORT_FORWARD_TOKENS and group in SHORT_FORWARD_GROUPS and d == 128
    return "fwd_hq16_hkv2" if short else "fwd_r7"


def cuda_backward(t: int, group: int, grouped: bool = True) -> bool:
    """Check whether the CUDA backward or gather_sparse's runs, by the faster on an H200.

    gather_sparse's key-side backward beats the CUDA one past CUDA_BACKWARD_ROWS only with buckets to group a query's keys by (grouped).
    Without, as for a top-k selection, its runs are single keys and the CUDA backward is faster up to its fast path's length (15.5 against 31.3 ms at 1 x 32768, eight heads over one).

    Args:
        grouped: Whether there are buckets to group a query's keys by.

    Returns:
        Whether the CUDA backward runs.
    """
    return t <= CUDA_BACKWARD_TOKENS and (not grouped or t * group <= CUDA_BACKWARD_ROWS)


def backward_kernel(t: int, group: int, d: int, grouped: bool = True, v2: bool = False) -> str | None:
    """Return the vendored backward for this shape, or None where gather_sparse's runs."""
    if v2 and v2_runs(t, group, d) and t <= V2_BACKWARD_TOKENS:
        hybrid = t >= HYBRID_TOKENS and t % HYBRID_MULTIPLE == 0
        return "bwd_hybrid" if hybrid else "bwd_v14"
    return "bwd_r1" if cuda_backward(t, group, grouped) else None


@triton.jit
def _sanitize_kernel(
    IND: tl.tensor,
    OUT: tl.tensor,
    stride_ib: tl.tensor,
    stride_it: tl.tensor,
    T: tl.tensor,
    WINDOW: tl.tensor,
    SINK: tl.tensor,
    TOP_K: tl.tensor,
    K_PAD: tl.constexpr,
    R: tl.constexpr,
) -> None:
    """The positions gather_sparse reads, int32, -1 elsewhere."""
    b = tl.program_id(1).to(tl.int64)
    query = tl.program_id(0) * R + tl.arange(0, R)[:, None]
    slot = tl.arange(0, K_PAD)[None, :]
    idx, valid = gather_sparse._listed(IND + b * stride_ib, stride_it, query, slot, T, TOP_K, WINDOW, SINK)
    at = (b * T + query) * TOP_K + slot
    tl.store(OUT + at, tl.where(valid, idx, -1), mask=(query < T) & (slot < TOP_K))


def sanitize(indices: torch.Tensor, window_size: int, sink_size: int) -> torch.Tensor:
    """Sanitise gather_sparse's positions into the ones the kernels take.

    Args:
        indices: Positions [b, t, top_k], as gather_sparse.attention documents.

    Returns:
        Positions [b, 1, t, top_k] int32.
    """
    b, t, top_k = indices.shape
    indices = indices if indices.stride(-1) == 1 else indices.contiguous()
    out = torch.empty(b, 1, t, top_k, dtype=torch.int32, device=indices.device)
    _sanitize_kernel[(triton.cdiv(t, SANITIZE_ROWS), b)](
        indices, out, indices.stride(0), indices.stride(1), t, window_size, sink_size, top_k,
        K_PAD=triton.next_power_of_2(top_k), R=SANITIZE_ROWS, num_warps=4,
    )  # fmt: skip
    return out


def _selected(sel: torch.Tensor, h_kv: int) -> torch.Tensor:
    """The one selection, (b, 1, t, top_k), as every key head's."""
    return sel if h_kv == 1 else sel.expand(-1, h_kv, -1, -1).contiguous()


@torch.library.custom_op("lsa::cuda_sparse_attention", mutates_args=())
def cuda_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
    v2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the CUDA forward over sanitised positions; see cuda_sparse.attention."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    kernel = _vendor(forward_kernel(q.shape[1], q.shape[2] // k.shape[2], q.shape[3], v2))
    return kernel.attention(q, k, v, _selected(sel, k.shape[2]), window_size, sink_size, scale)


@cuda_attention_op.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
    v2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an empty output like q and an empty log-sum-exp [B, H_q, T] fp32."""
    b, t, h_q, _ = q.shape
    return q.new_empty(q.shape), q.new_empty((b, h_q, t), dtype=torch.float32)


@torch.library.custom_op("lsa::cuda_sparse_attention_backward", mutates_args=())
def cuda_attention_backward_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    sel: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
    v2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk and dv of cuda_attention_op, by the CUDA backward or gather_sparse's."""
    t, group, d = q.shape[1], q.shape[2] // k.shape[2], q.shape[3]
    kernel = backward_kernel(t, group, d, members is not None, v2)
    if kernel is not None:
        q, k, v, o, do = (x.contiguous() for x in (q, k, v, o, do))
        return _vendor(kernel).attention_backward(
            q, k, v, _selected(sel, k.shape[2]), o, lse, do, window_size, sink_size, scale
        )  # fmt: skip
    return gather_sparse._backward(do, q, k, v, o, lse, sel[:, 0], members, window_size, sink_size, scale)


@cuda_attention_backward_op.register_fake
def _(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    sel: torch.Tensor,
    members: torch.Tensor | None,
    window_size: int,
    sink_size: int,
    scale: float,
    v2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty gradients like q, k and v."""
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    """Save the tensors and settings the backward needs."""
    q, k, v, sel, members, window_size, sink_size, scale, v2 = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, sel, members)
    ctx.window_size, ctx.sink_size, ctx.scale, ctx.v2 = window_size, sink_size, scale, v2


def _autograd_backward(ctx: Any, do: torch.Tensor, _dlse: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Return the gradients of q, k and v, and none for the other inputs."""
    q, k, v, o, lse, sel, members = ctx.saved_tensors
    dq, dk, dv = cuda_attention_backward_op(
        do, q, k, v, o, lse, sel, members, ctx.window_size, ctx.sink_size, ctx.scale, ctx.v2
    )
    return dq, dk, dv, None, None, None, None, None, None


cuda_attention_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
    members: torch.Tensor | None = None,
    sanitized: bool = False,
    v2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run gather_sparse.attention by the CUDA kernels, which supported must take.

    Members serve its backward where it runs.

    Args:
        sanitized: Whether the indices already are what sanitize makes of them, int32 and -1 in every slot the read leaves, as lsa_select's and lsa_rank's are.
            Each query's distinct keys in [sink_size, i - window_size] are in its first slots, -1 after.
            They are then read as they are.
        v2: Whether to read by the v2 kernels where they run (v2_runs).
        Others: gather_sparse.attention's.

    Returns:
        gather_sparse.attention's.
    """
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if not supported(q, k) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("cuda_sparse.attention takes bf16 d 64 or 128 heads on sm_90 only")
    b, t, h_q, _ = q.shape
    if h_q % k.shape[2] or k.shape != v.shape or k.shape[:2] != (b, t):
        raise ValueError(f"q {tuple(q.shape)} does not group over k and v {tuple(k.shape)}")
    if indices.shape[:2] != (b, t) or indices.dim() != 3:
        raise ValueError(f"indices {tuple(indices.shape)} are not (b, t, top_k) for q")
    if sanitized:
        if indices.dtype != torch.int32:
            raise ValueError(f"sanitized indices are int32, got {indices.dtype}")
        sel = indices.contiguous()[:, None]
    else:
        sel = sanitize(indices, window_size, sink_size)
    with torch.autocast("cuda", enabled=False):
        o, lse = cuda_attention_op(q, k, v, sel, members, window_size, sink_size, scale, v2)
    return o, lse.detach()
