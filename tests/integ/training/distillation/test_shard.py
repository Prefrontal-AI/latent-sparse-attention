"""A run trained in shards and joined is the run trained whole."""

import json

import pytest
import torch

from lsa.training.distillation.config import load
from lsa.training.distillation.shard import join, split
from lsa.training.distillation.tracking import scalars
from lsa.training.distillation.trainer import train
from tests.helpers.training.distillation.data import fake_cache

RUN = """
model:
  name: glm53_flash
  indexer:
    input: attention_input
    window_size: 32
    sink_size: 4
    top_k: 64
    index_dim: 8
data:
  root: {root}
  layers: [0, 1, 2]
  train:
    - {{name: natural, source: tokens, split: train}}
  evaluate:
    - {{name: natural, source: tokens, split: eval}}
run:
  name: unit
  output: {results}
  optim:
    steps: 7
    batch: 4
    draw: token
  report:
    every: 2
"""
LAYERS = [0, 1, 2]


def a_run(tmp_path, results):
    cache = tmp_path / "cache"
    fake_cache.collection(cache, documents=("source0", "source1"), layers=LAYERS, split="train")
    fake_cache.collection(cache, documents=("source2",), layers=LAYERS, split="eval", seed=5)
    path = tmp_path / "run.yaml"
    path.write_text(RUN.format(root=cache, results=tmp_path / results))
    (experiment,) = load(path, [])
    return experiment


def sharded(tmp_path, shards):
    experiment = a_run(tmp_path, "sharded")
    for shard in shards:
        train(experiment, device="cpu", shard=shard)
    return join(experiment, LAYERS)


def test_a_run_trained_in_shards_learns_the_maps_it_learns_whole(tmp_path):
    whole = train(a_run(tmp_path, "whole"), device="cpu")
    joined = sharded(tmp_path, [[0, 2], [1]])

    for name in ("maps.pt", "maps_step4.pt"):
        held = torch.load(whole / name, weights_only=False)["maps"]
        written = torch.load(joined / name, weights_only=False)["maps"]
        assert list(written) == LAYERS
        for layer, state in held.items():
            for key, value in state.items():
                assert torch.equal(written[layer][key], value), f"{name} layer {layer} {key}"


def test_a_joined_report_is_the_report_of_the_run_trained_whole(tmp_path):
    whole = train(a_run(tmp_path, "whole"), device="cpu")
    joined = sharded(tmp_path, [[0, 2], [1]])

    names = sorted(path.name for path in whole.glob("report_step*.json"))
    assert names == sorted(path.name for path in joined.glob("report_step*.json"))
    for name in names:
        expected = scalars(json.loads((whole / name).read_text()), "report")
        assert scalars(json.loads((joined / name).read_text()), "report") == pytest.approx(expected)


def test_the_joined_meta_names_every_layer_and_every_shard(tmp_path):
    meta = json.loads((sharded(tmp_path, split(LAYERS, 2)) / "meta.json").read_text())
    assert meta["layers"] == LAYERS
    assert meta["shards"] == ["layers-0-2", "layers-1"]


def test_a_run_whose_shards_have_not_all_finished_is_not_joined(tmp_path):
    experiment = a_run(tmp_path, "sharded")
    train(experiment, device="cpu", shard=[0])
    with pytest.raises(ValueError, match="the shards train layers"):
        join(experiment, LAYERS)


def test_a_shard_of_a_layer_the_run_does_not_train_says_so(tmp_path):
    with pytest.raises(ValueError, match="does not"):
        train(a_run(tmp_path, "sharded"), device="cpu", shard=[5])


def test_layers_are_dealt_round_robin_so_every_shard_spans_the_depth():
    assert split([3, 7, 11, 15, 19], 2) == [[3, 11, 19], [7, 15]]
    assert split([3, 7], 4) == [[3], [7]]
