"""Reading a distilled map artifact, and refusing one that was trained for something else.

The artifacts here are written by the index-space trainer rather than assembled by hand, so what the injector
reads is what distillation actually saves.
"""

import pytest
import torch

from lsa.training import distilled_maps as maps
from lsa.training.distillation.config import load as load_distill
from lsa.training.distillation.trainer import train
from tests.helpers.training.distillation.data import fake_cache

RUN = """
model:
  name: {model}
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
    steps: 2
    batch: 4
  report:
    every: 2
"""

MODEL = fake_cache.MODEL


def an_artifact(tmp_path, model=MODEL):
    """Train a map at test size and return the artifact it wrote."""
    cache = tmp_path / "cache"
    extraction = {"model": model}
    fake_cache.collection(cache, documents=("source0", "source1"), split="train", **extraction)
    fake_cache.collection(cache, documents=("source2",), split="eval", seed=5, **extraction)
    path = tmp_path / "run.yaml"
    path.write_text(RUN.format(root=cache, results=tmp_path / "results", model=model))
    (experiment,) = load_distill(path)
    return train(experiment, device="cpu") / "maps.pt"


def test_an_artifact_carries_a_spec_per_layer_and_what_produced_it(tmp_path):
    artifact = maps.load(an_artifact(tmp_path))
    assert artifact.layers == [0]
    assert artifact.model == MODEL
    assert artifact.step == 2
    assert artifact.index_dim == 8


def test_the_maps_come_back_as_the_index_space_takes_them(tmp_path):
    """The indexer reads two fp32 matrices of one shape, `[model_dim, index_dim]`"""
    projection = maps.load(an_artifact(tmp_path)).projections[0]
    assert projection.query_projection.dtype is torch.float32
    assert projection.key_projection.dtype is torch.float32
    assert projection.query_projection.shape == projection.key_projection.shape
    assert projection.query_projection.shape == (fake_cache.MODEL_DIM, 8)
    assert (projection.model_dim, projection.index_dim) == (fake_cache.MODEL_DIM, 8)
    assert not projection.query_projection.requires_grad


def test_a_snapshot_reads_the_same_way_and_says_which_step_it_is(tmp_path):
    written = an_artifact(tmp_path)
    assert maps.load(written.with_name("maps_step0.pt")).step == 0
    assert maps.load(written).step == 2


def test_the_layers_a_run_indexes_come_back_in_the_order_it_asked(tmp_path):
    artifact = maps.load(an_artifact(tmp_path))
    assert list(artifact.checked_for(MODEL, [0])) == [0]


def test_a_map_distilled_for_another_model_is_refused(tmp_path):
    """Two models of one dim share the shape of every projection, so only the name catches this"""
    artifact = maps.load(an_artifact(tmp_path, model="Qwen/Qwen3.5-9B"))
    with pytest.raises(ValueError, match="was distilled for 'Qwen/Qwen3.5-9B'"):
        artifact.checked_for(MODEL, [0])


def test_model_identity_is_exact(tmp_path):
    artifact = maps.load(an_artifact(tmp_path))
    with pytest.raises(ValueError, match="was distilled for"):
        artifact.checked_for("Qwen/Qwen3.5-4B-Base", [0])


def test_a_layer_the_artifact_does_not_hold_is_refused(tmp_path):
    artifact = maps.load(an_artifact(tmp_path))
    with pytest.raises(ValueError, match=r"no map for layers \[3, 7\]"):
        artifact.checked_for(MODEL, [0, 3, 7])


def test_a_missing_artifact_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no distilled maps at"):
        maps.load(tmp_path / "nothing.pt")


def test_a_file_that_is_not_one_of_these_says_so(tmp_path):
    other = tmp_path / "other.pt"
    torch.save({"model": {}}, other)
    with pytest.raises(ValueError, match="holds no 'maps'"):
        maps.load(other)


def test_an_artifact_with_no_maps_in_it_says_so(tmp_path):
    empty = tmp_path / "empty.pt"
    torch.save({"maps": {}}, empty)
    with pytest.raises(ValueError, match="holds no maps"):
        maps.load(empty)


def test_a_layer_entry_missing_a_map_says_which(tmp_path):
    broken = tmp_path / "broken.pt"
    torch.save({"maps": {0: {"key_projection": torch.zeros(4, 2)}}}, broken)
    with pytest.raises(ValueError, match="layer 0 holds"):
        maps.load(broken)


def test_the_summary_names_the_artifact_and_its_teacher(tmp_path):
    line = maps.describe(maps.load(an_artifact(tmp_path)))
    assert MODEL in line
    assert "step 2" in line
    assert "index_dim 8" in line


def test_an_exported_run_reads_back_as_the_projections_it_trained(tmp_path):
    """The file serving loads holds `<layer>.W_K_idx` and `<layer>.W_Q_idx`, each `[model_dim, index_dim]`, as trained"""
    from lsa.sglang.index import weights
    from lsa.training.distillation.export import export

    trained = an_artifact(tmp_path)
    out = export(trained, tmp_path / "served" / "projections.pt")

    saved = torch.load(trained, weights_only=False)["maps"][0]
    query, key = weights.read(out, fake_cache.MODEL_DIM, 8, [0])[0]
    assert key.shape == query.shape == (fake_cache.MODEL_DIM, 8)
    assert torch.equal(key, saved["key_projection"].to(key.dtype))
    assert torch.equal(query, saved["query_projection"].to(query.dtype))
    flat = torch.load(out, weights_only=True)
    assert sorted(flat) == ["0.W_K_idx", "0.W_Q_idx"]
    assert torch.equal(flat["0.W_K_idx"], saved["key_projection"])
