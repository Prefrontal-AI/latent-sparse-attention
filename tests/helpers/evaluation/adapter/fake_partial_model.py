"""A model shaped like the partial-rotary families the generic path has to cover.

stablelm, phi, persimmon and lfm2 differ from a Qwen-shaped decoder in three ways that all look like
nothing and all change the numbers.
Their `apply_rotary_pos_emb` does not slice - the attention forward slices before calling it, so the module
function gets a whole-head `cos` and a rotary-dim one broadcasts into an error.
Their output projection is `dense`, not `o_proj`.
Their QK norm is `q_layernorm`, not `q_norm` - and a lookup that knows only one name silently skips it
rather than failing, which is the one of the three that scores instead of raising.
"""

import torch
import torch.nn as nn

from tests.helpers.evaluation.adapter.fake_model import rotate_half


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Rotate the whole of what it is given, as stablelm and phi do - the caller owns the slicing."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class PartialAttention(nn.Module):
    """A partial-rotary decoder's attention, under the names that family uses."""

    def __init__(self, model_dim, query_head_count, kv_head_count, head_dim, layer_idx=0, qk_norm=False):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(model_dim, query_head_count * head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.dense = nn.Linear(query_head_count * head_dim, model_dim, bias=False)
        if qk_norm:
            self.q_layernorm = nn.LayerNorm(head_dim)
            self.k_layernorm = nn.LayerNorm(head_dim)


def dense_output(attention, hidden_states, cos, sin, rotary_dim):
    """What the unmodified layer produces, with every module named outright rather than looked up."""
    batch_size, query_count = hidden_states.shape[:2]
    head_dim = attention.head_dim

    def heads(projected):
        return projected.view(batch_size, query_count, -1, head_dim).transpose(1, 2)

    q, k, v = (
        heads(projection(hidden_states))
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    if hasattr(attention, "q_layernorm"):
        q, k = attention.q_layernorm(q), attention.k_layernorm(k)

    # The slice the model's own attention forward does before it calls the rotary function
    q_rotated, k_rotated = apply_rotary_pos_emb(q[..., :rotary_dim], k[..., :rotary_dim], cos, sin)
    q = torch.cat([q_rotated, q[..., rotary_dim:]], dim=-1)
    k = torch.cat([k_rotated, k[..., rotary_dim:]], dim=-1)

    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)

    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * attention.scaling
    positions = torch.arange(query_count)
    logits = logits.masked_fill(positions[:, None] < positions[None, :], float("-inf"))
    output = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(logits, dim=-1), v.float()).to(q.dtype)
    return attention.dense(output.transpose(1, 2).reshape(batch_size, query_count, -1))


class NopeAttention(nn.Module):
    """A decoder that does not rotate at all, as GLM-5.3-Flash's DSA layers do not."""

    def __init__(self, model_dim, query_head_count, kv_head_count, head_dim, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(model_dim, query_head_count * head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, kv_head_count * head_dim, bias=False)
        self.o_proj = nn.Linear(query_head_count * head_dim, model_dim, bias=False)


def nope_dense_output(attention, hidden_states):
    """What a NoPE layer produces - the same read with no rotation anywhere in it."""
    batch_size, query_count = hidden_states.shape[:2]
    head_dim = attention.head_dim

    def heads(projected):
        return projected.view(batch_size, query_count, -1, head_dim).transpose(1, 2)

    q, k, v = (
        heads(projection(hidden_states))
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)

    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * attention.scaling
    positions = torch.arange(query_count)
    logits = logits.masked_fill(positions[:, None] < positions[None, :], float("-inf"))
    output = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(logits, dim=-1), v.float()).to(q.dtype)
    return attention.o_proj(output.transpose(1, 2).reshape(batch_size, query_count, -1))
