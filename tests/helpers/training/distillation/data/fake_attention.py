"""An MLA attention of the names the key and query input is read with - GLM's, small and random."""

from types import SimpleNamespace

import torch

from tests.helpers.training.distillation.data import fake_cache

HEADS, KEY_DIM, VALUE_DIM, RANK, Q_RANK = 4, 8, 4, 16, 12


def attention(model_dim=fake_cache.MODEL_DIM, heads=HEADS, rank=RANK, seed=0, dtype=torch.bfloat16):
    """One layer's `self_attn`, as `load_attention` hands it over - bf16, as the checkpoint's modules are"""
    torch.manual_seed(seed)
    return SimpleNamespace(
        kv_lora_rank=rank,
        qk_head_dim=KEY_DIM,
        qk_nope_head_dim=KEY_DIM,
        scaling=KEY_DIM**-0.5,
        v_head_dim=VALUE_DIM,
        kv_a_proj_with_mqa=torch.nn.Linear(model_dim, rank, bias=False, dtype=dtype),
        kv_a_layernorm=torch.nn.RMSNorm(rank, dtype=dtype),
        q_a_proj=torch.nn.Linear(model_dim, Q_RANK, bias=False, dtype=dtype),
        q_a_layernorm=torch.nn.RMSNorm(Q_RANK, dtype=dtype),
        q_b_proj=torch.nn.Linear(Q_RANK, heads * KEY_DIM, bias=False, dtype=dtype),
        kv_b_proj=torch.nn.Linear(rank, heads * (KEY_DIM + VALUE_DIM), bias=False, dtype=dtype),
    )


def load(model, layers, device):
    """`load_attention`'s stand-in - one fake layer each, seeded by the layer"""
    return {layer: attention(seed=layer) for layer in layers}
