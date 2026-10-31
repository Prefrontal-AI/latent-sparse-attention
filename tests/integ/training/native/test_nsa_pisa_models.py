import pytest
import torch

from lsa.training.native.models import model_spec
from lsa.training.native.nsa import NsaAttention
from lsa.training.native.pisa import PisaInnerAttention
from tests.integ.training.native.test_models import attentions, experiment


@pytest.mark.parametrize("kind", ["nsa", "pisa"])
def test_native_model_routes_every_full_attention_layer_to_the_sparse_arm(kind):
    spec = model_spec(
        experiment(
            kind,
            flavor="0.8B-gpt2",
            model={
                "query_head_count": 8,
                "kv_head_count": 1,
                "head_dim": 128,
                "rotary_dim": 0,
            },
        )
    )
    layers = attentions(spec)
    assert layers
    if kind == "nsa":
        assert all(isinstance(layer, NsaAttention.Config) for layer in layers)
    else:
        assert all(isinstance(layer.inner_attention, PisaInnerAttention.Config) for layer in layers)
    with torch.device("meta"):
        model = spec.model.build()
    model.verify_module_protocol()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_sparse_arm_sweeps_preserve_current_and_historical_budgets():
    from pathlib import Path

    from lsa import config_loader
    from lsa.training.native.config import NativeConfig

    root = Path(__file__).parents[4] / "configs/training/native/qwen35_0_8b_gpt2"
    cases = [
        ("nsam8_selection.yaml", "select_block_count", [32, 64]),
        ("pisam8_window_budgets.yaml", "level_block_count", [25, 41, 73]),
        ("pisam8_historical_budgets.yaml", "level_block_count", [40, 72]),
    ]
    for filename, field, counts in cases:
        runs = config_loader.load(NativeConfig, root / filename, name_path="run.name")
        assert [getattr(run.attention, field) for run in runs] == counts
        assert len({run.run.name for run in runs}) == len(counts)
