"""A transformers-shaped model that is not transformers.

The adapter's generic path reads a checkpoint through a handful of attribute names - `q_proj`, `head_dim`,
`layer_idx`, a cache with `update`, an `apply_rotary_pos_emb` in the attention class's own module.
Those names are the contract, so a fake that honours them exercises the whole layer on CPU, with no
checkpoint and nothing from the evaluation extra installed.
"""

import torch
import torch.nn as nn


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


def rotate_dims(x, dims, cos, sin):
    """Apply a real rotation to an arbitrary set of head dims, leaving the rest alone."""
    if dims.numel() == 0:
        return x
    part = x[..., dims]
    rotated = x.clone()
    rotated[..., dims] = part * cos + rotate_half(part) * sin
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """The prefix-partial rotation, resolved out of this module by `rotary_function`."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    dims = torch.arange(cos.shape[-1], device=q.device)
    return rotate_dims(q, dims, cos, sin), rotate_dims(k, dims, cos, sin)


def rotation_over(dims):
    """Build a `SparseAttention.rotate` that rotates exactly `dims`, for the layout probe to find."""

    def rotate(attention, q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        placed = dims.to(q.device)
        return rotate_dims(q, placed, cos, sin), rotate_dims(k, placed, cos, sin)

    return rotate


class FakeRotaryEmbedding(nn.Module):
    """The `(cos, sin)` tables, in the shape and the call signature transformers uses."""

    def __init__(self, rotary_dim, theta=10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.register_buffer(
            "inv_freq", 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        )

    def forward(self, x, position_ids):
        frequencies = position_ids.float()[..., None] * self.inv_freq
        angles = torch.cat([frequencies, frequencies], dim=-1)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


class FakeAttention(nn.Module):
    """A standard HF decoder's attention module, down to the attribute names."""

    def __init__(self, model_dim, query_head_count, kv_head_count, head_dim, layer_idx=0, qk_norm=False):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(model_dim, query_head_count * head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.o_proj = nn.Linear(query_head_count * head_dim, model_dim, bias=False)
        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)


class MropeRotaryEmbedding(FakeRotaryEmbedding):
    """A rotary embedding that takes one row of positions per mrope section, as Qwen3.5 does."""

    def forward(self, x, position_ids):
        sections = position_ids[:, :, None, :]
        return super().forward(x, sections[0, :, 0, :])
