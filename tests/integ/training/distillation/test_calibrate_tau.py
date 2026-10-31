"""Recording the temperature of an exported map that has none, from the held-out documents of a cache."""

import json

import torch

from lsa.training.distillation import calibrate_tau
from lsa.training.distillation.config import load
from lsa.training.distillation.constants import Calibration
from lsa.training.distillation.export import map_tau
from tests.helpers.training.distillation.data import fake_cache

RUN = """
model:
  name: glm53_flash
  indexer: {{window_size: {window}, sink_size: 4, top_k: 64, index_dim: {index_dim}}}
data:
  root: {root}
  layers: [0]
  train:
    - {{name: natural, source: tokens, split: train}}
  evaluate:
    - {{name: natural, source: tokens, split: eval}}
run:
  name: unit
  output: {results}
"""


def a_run(tmp_path):
    cache = tmp_path / "cache"
    fake_cache.collection(cache, documents=("source0",), split="train")
    fake_cache.collection(cache, documents=("source2", "source3", "source4"), split="eval", seed=5)
    path = tmp_path / "run.yaml"
    path.write_text(
        RUN.format(
            root=cache, results=tmp_path / "results", window=fake_cache.WINDOW, index_dim=fake_cache.INDEX_DIM
        )
    )
    (experiment,) = load(path, [])
    return experiment


def test_a_map_is_calibrated_a_layer_at_a_time_and_its_sidecar_keeps_what_it_said(tmp_path):
    experiment = a_run(tmp_path)
    maps = tmp_path / "maps.pt"
    torch.manual_seed(0)
    torch.save(
        {
            "0.W_Q_idx": torch.randn(fake_cache.MODEL_DIM, fake_cache.INDEX_DIM),
            "0.W_K_idx": torch.randn(fake_cache.MODEL_DIM, fake_cache.INDEX_DIM),
        },
        maps,
    )
    maps.with_suffix(".json").write_text(json.dumps({"run": "kept"}))

    tau = calibrate_tau.calibrate_tau(experiment, maps, per_set=2, device="cpu")

    low, high, _ = Calibration.GRID
    assert 10**low <= tau["0"] <= 10**high
    assert map_tau(maps, [0]) == {0: tau["0"]}
    sidecar = json.loads(maps.with_suffix(".json").read_text())
    assert sidecar["run"] == "kept"
    assert sidecar["tau_calibrated"]["documents"] == 2
