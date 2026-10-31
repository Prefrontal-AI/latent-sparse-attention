import dataclasses

import torch
import torch.nn.functional as F
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.model import Qwen35Attention

from lsa.training.native.config import AttentionConfig
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.models import shaped_attention

LENGTH = 24
HEAD_DIM = 16
DIM = 1024


def template():
    built = model_registry("0.8B", seq_len=2 * LENGTH)
    return next(layer.attention for layer in built.model.layers if layer.attention is not None)


def shaped(**changes):
    shape = {
        "query_head_count": 4,
        "kv_head_count": 2,
        "head_dim": HEAD_DIM,
        "rotary_dim": 8,
        "output_gate": True,
        "qk_norm": True,
    }
    config = shaped_attention(template(), **{**shape, **changes})
    window = AttentionConfig(kind="window", window_size=LENGTH, sink_size=0)
    config.inner_attention = MultiresInnerAttention.Config(
        attention=window, seq_len=LENGTH, rotary_dim=config.rotary_dim
    )
    return config


def initialised(config):
    module = config.build()
    module.init_states(buffer_device=None)
    return module


def tokens():
    return torch.randn(LENGTH, DIM, generator=torch.Generator().manual_seed(2)), torch.arange(LENGTH)


def test_with_the_gate_and_the_norm_on_it_is_qwen3_5s_own_layer():
    config = shaped()
    ours = initialised(config)
    fields = [
        field.name for field in dataclasses.fields(Qwen35Attention.Config) if not field.name.startswith("_")
    ]
    reference = Qwen35Attention.Config(**{name: getattr(config, name) for name in fields}).build()
    reference.load_state_dict(ours.state_dict())
    x, positions = tokens()
    with torch.no_grad():
        assert torch.equal(ours(x, None, positions), reference(x, None, positions))


def test_switched_off_the_gate_the_norms_and_the_rotation_have_no_parameters():
    module = shaped(output_gate=False, qk_norm=False, rotary_dim=0).build()
    assert {name for name, _ in module.named_parameters()} == {
        "wq.weight",
        "wk.weight",
        "wv.weight",
        "wo.weight",
    }
    assert module.wq.out_features == 4 * HEAD_DIM


def test_a_plain_layer_is_causal_attention_over_its_grouped_heads():
    module = initialised(shaped(output_gate=False, qk_norm=False, rotary_dim=0, kv_head_count=1))
    x, positions = tokens()
    with torch.no_grad():
        got = module(x, None, positions)
        q = module.wq(x).view(LENGTH, 4, HEAD_DIM).transpose(0, 1)
        k = module.wk(x).view(LENGTH, 1, HEAD_DIM).transpose(0, 1)
        v = module.wv(x).view(LENGTH, 1, HEAD_DIM).transpose(0, 1)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        expected = module.wo(context.transpose(0, 1).reshape(LENGTH, -1))
    assert torch.allclose(got, expected, atol=1e-5)


def test_the_rotation_makes_the_layer_read_distances_and_none_makes_it_blind_to_them():
    x, positions = tokens()
    for rotary_dim, positional in ((8, True), (0, False)):
        module = initialised(shaped(rotary_dim=rotary_dim))
        with torch.no_grad():
            stretched = module(x, None, positions * 2)
            same = module(x, None, positions)
        assert torch.allclose(stretched, same, atol=1e-5) != positional
