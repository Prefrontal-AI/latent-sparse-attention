"""Scoring a checkpoint at a read it was not trained for."""

import json

import pytest
import torch

from lsa.training.distillation import export
from lsa.training.distillation.config import load
from lsa.training.distillation.rescore_maps import rescore_maps
from lsa.training.distillation.trainer import train
from tests.helpers.training.distillation.data import fake_cache

RUN = """
model:
  name: glm53_flash
  space:
    init: {init}
  indexer:
    input: attention_input
    window_size: 32
    sink_size: 4
    top_k: 64
    index_dim: 8
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
  optim:
    steps: {steps}
    batch: 4
  report:
    every: 4
"""


def a_run(tmp_path, init="null", steps=4, **overrides):
    cache = tmp_path / "cache"
    if not cache.exists():
        fake_cache.collection(cache, documents=("source0", "source1"), split="train")
        fake_cache.collection(cache, documents=("source2",), split="eval", seed=5)
    path = tmp_path / "run.yaml"
    path.write_text(RUN.format(root=cache, results=tmp_path / "results", init=init, steps=steps))
    (experiment,) = load(path, [f"{key}={value}" for key, value in overrides.items()])
    return experiment


def trained(tmp_path):
    return train(a_run(tmp_path), device="cpu") / "maps.pt"


def test_a_rescore_writes_one_report_and_no_maps(tmp_path):
    maps = trained(tmp_path)
    out = rescore_maps(a_run(tmp_path, init=maps, steps=0, **{"run.name": "scored"}), device="cpu")
    assert (out / "report.json").exists()
    assert (out / "meta.json").exists()
    assert not (out / "maps.pt").exists()
    assert list(out.glob("report_step*.json")) == []


def test_a_rescore_at_the_trained_read_reproduces_the_last_report(tmp_path):
    maps = trained(tmp_path)
    trained_report = json.loads((maps.parent / "report_step4.json").read_text())
    out = rescore_maps(a_run(tmp_path, init=maps, steps=0, **{"run.name": "scored"}), device="cpu")
    scored = json.loads((out / "report.json").read_text())
    # The same maps read the same way, so this is equality rather than a tolerance
    assert scored["natural"] == trained_report["natural"]


def test_a_rescore_needs_a_checkpoint_to_score(tmp_path):
    with pytest.raises(ValueError, match="model.space.init"):
        rescore_maps(a_run(tmp_path, steps=0), device="cpu")


def test_a_narrower_budget_is_a_different_report_off_the_same_maps(tmp_path):
    maps = trained(tmp_path)
    wide = rescore_maps(a_run(tmp_path, init=maps, steps=0, **{"run.name": "wide"}), device="cpu")
    narrow = rescore_maps(
        a_run(tmp_path, init=maps, steps=0, **{"run.name": "narrow", "model.indexer.top_k": 8}),
        device="cpu",
    )
    scored = [json.loads((path / "report.json").read_text())["natural"] for path in (wide, narrow)]
    assert scored[0] != scored[1]


def test_a_rescore_requires_maps_from_the_same_teacher(tmp_path):
    maps = trained(tmp_path)
    artifact = torch.load(maps, weights_only=False)
    artifact["teacher"]["model"] = "qwen35_4b"
    torch.save(artifact, maps)
    experiment = a_run(tmp_path, init=maps, steps=0, **{"run.name": "scored"})

    with pytest.raises(ValueError, match="was distilled for"):
        rescore_maps(experiment, device="cpu")


def test_a_projection_file_scores_as_the_checkpoint_it_was_exported_from(tmp_path):
    maps = trained(tmp_path)
    exported = tmp_path / "projections.pt"
    torch.save(export.projections(torch.load(maps, weights_only=False)), exported)
    from_checkpoint = rescore_maps(a_run(tmp_path, init=maps, steps=0, **{"run.name": "ckpt"}), device="cpu")
    from_file = rescore_maps(a_run(tmp_path, init=exported, steps=0, **{"run.name": "file"}), device="cpu")
    scored = [
        json.loads((path / "report.json").read_text())["natural"] for path in (from_checkpoint, from_file)
    ]
    assert scored[0] == scored[1]


def test_an_identity_projection_file_scores_the_input_itself(tmp_path):
    maps = trained(tmp_path)
    layer = torch.load(maps, weights_only=False)["maps"][0]["key_projection"]
    identity = torch.eye(layer.shape[0])
    exported = tmp_path / "identity.pt"
    torch.save({"0.W_Q_idx": identity, "0.W_K_idx": identity}, exported)
    out = rescore_maps(a_run(tmp_path, init=exported, steps=0, **{"run.name": "identity"}), device="cpu")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["index_dim"] == meta["model_dim"]


def test_a_projection_file_without_a_layer_is_refused(tmp_path):
    exported = tmp_path / "projections.pt"
    torch.save({"5.W_Q_idx": torch.eye(4), "5.W_K_idx": torch.eye(4)}, exported)
    with pytest.raises(ValueError, match="holds no projection for layers"):
        rescore_maps(a_run(tmp_path, init=exported, steps=0), device="cpu")


def test_depths_add_an_exact_read_each_and_a_mean_per_layer(tmp_path):
    maps = trained(tmp_path)
    out = rescore_maps(
        a_run(tmp_path, init=maps, steps=0, **{"run.name": "deep", "run.report.depths": "[8,64]"}),
        device="cpu",
    )
    scored = json.loads((out / "report.json").read_text())["natural"]
    assert set(scored["depths"]) == {"8", "64"}
    assert set(scored["by_layer"]) == {"0"}
    assert scored["by_layer"]["0"]["depths"]["8"] == scored["depths"]["8"]


def test_a_depth_that_is_not_positive_is_refused(tmp_path):
    with pytest.raises(ValueError, match="report.depths"):
        a_run(tmp_path, **{"run.report.depths": "[0]"})
