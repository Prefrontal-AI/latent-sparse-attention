"""Compose every DSA family overlay and its learning-rate and seed grid."""

from pathlib import Path

from lsa.training.native.config import load
from lsa.training.native.models import model_spec

ROOT = Path("configs/training/native/dense_340m_gpt2")


def test_dsa_overlays_compose_into_the_native_340m_model():
    for path in sorted(ROOT.glob("dsa*.yaml")):
        runs = load(path)
        assert len(runs) == (6 if path.stem == "dsa_lr" else 1)
        for run in runs:
            assert run.attention.kind == "dsa"
            assert run.model.flavor == "340M-gpt2"
            layers = model_spec(run).model.layers
            assert len(layers) == 19
            assert all(layer.attention.attention == run.attention for layer in layers)
            assert all(layer.attention.wq.bias == run.model.attention_bias for layer in layers)
