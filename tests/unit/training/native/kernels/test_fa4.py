import sys
import types
from functools import partial

from torch.nn.attention import _registry

from lsa.training.native.kernels import fa4


def fake_interface(monkeypatch):
    interface = types.ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = lambda *args, **kwargs: ("out", "lse", "p", "row_max", "o_lo")
    interface._flash_attn_bwd = lambda *args, **kwargs: ("dq", "dk", "dv", "dqv", "dsink")
    for name in ("flash_attn", "flash_attn.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setattr(sys.modules["flash_attn.cute"], "interface", interface, raising=False)


def test_the_forward_keeps_the_output_and_the_log_sum_exp(monkeypatch):
    fake_interface(monkeypatch)
    assert fa4._flash_attn_fwd(1, causal=True) == ("out", "lse")


def test_the_backward_keeps_the_three_gradients(monkeypatch):
    fake_interface(monkeypatch)
    assert fa4._flash_attn_bwd(1, causal=True) == ("dq", "dk", "dv")


def test_register_points_torchs_fa4_at_this_module(monkeypatch):
    monkeypatch.setattr(_registry, "_FLASH_ATTENTION_IMPLS", {})
    fa4.register()
    register_fn = _registry._FLASH_ATTENTION_IMPLS["FA4"]
    assert isinstance(register_fn, partial)
    assert register_fn.keywords == {"module_path": "lsa.training.native.kernels.fa4"}
