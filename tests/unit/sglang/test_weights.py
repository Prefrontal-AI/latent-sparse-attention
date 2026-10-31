"""The projections a server reads off `x` - refused unless they hold every layer the model indexes."""

from types import SimpleNamespace

import pytest
import torch

from lsa.sglang.index import weights

MODEL_DIM, INDEX_DIM = 16, 8


def write(tmp_path, layers=(3, 7), model_dim=MODEL_DIM, index_dim=INDEX_DIM, extra=None):
    blob = {}
    for layer in layers:
        blob[f"{layer}.W_Q_idx"] = torch.randn(model_dim, index_dim)
        blob[f"{layer}.W_K_idx"] = torch.randn(model_dim, index_dim)
    blob.update(extra or {})
    path = tmp_path / "maps.pt"
    torch.save(blob, path)
    return path


def test_a_file_holding_exactly_the_layers_reads_back_per_layer(tmp_path):
    path = write(tmp_path)
    maps = weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])
    assert sorted(maps) == [3, 7]
    query_projection, key_projection = maps[3]
    assert query_projection.shape == key_projection.shape == (MODEL_DIM, INDEX_DIM)
    assert query_projection.dtype == torch.float32
    saved = torch.load(path, weights_only=True)
    assert torch.equal(query_projection, saved["3.W_Q_idx"])
    assert torch.equal(key_projection, saved["3.W_K_idx"])


@pytest.mark.parametrize(("model_dim", "index_dim"), [(MODEL_DIM + 1, INDEX_DIM), (MODEL_DIM, INDEX_DIM * 2)])
def test_a_projection_that_is_not_model_dim_by_index_dim_is_refused(tmp_path, model_dim, index_dim):
    path = write(tmp_path, model_dim=model_dim, index_dim=index_dim)
    with pytest.raises(ValueError, match=r"expected \[input_dim, index_dim\] = \[16, 8\]"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])


def test_a_map_trained_on_the_latent_is_refused(tmp_path):
    path = write(tmp_path, model_dim=512)
    with pytest.raises(ValueError, match=r"W_Q_idx is \(512, 8\)"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])


def test_a_layer_the_file_lacks_is_named(tmp_path):
    path = write(tmp_path, layers=(3,))
    with pytest.raises(ValueError, match=r"missing \[7\]$"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])


def test_a_cut_of_the_model_reads_only_its_indexed_layers_from_the_full_map(tmp_path):
    """The kernels harness serves a five-layer cut whose layer 3 alone runs an indexer, on the full model's map"""
    path = write(tmp_path, layers=(3, 7, 11))
    saved = torch.load(path, weights_only=True)
    maps = weights.read(path, MODEL_DIM, INDEX_DIM, [3])
    assert sorted(maps) == [3]
    assert torch.equal(maps[3][0], saved["3.W_Q_idx"])
    assert torch.equal(maps[3][1], saved["3.W_K_idx"])
    # A superset still has to hold every indexed layer
    with pytest.raises(ValueError, match=r"missing \[5\]$"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 5])


def test_half_a_pair_is_refused(tmp_path):
    path = write(tmp_path)
    blob = torch.load(path, weights_only=True)
    del blob["7.W_K_idx"]
    torch.save(blob, path)
    with pytest.raises(ValueError, match=r"only one of W_Q_idx and W_K_idx for layers \[7\]"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])


@pytest.mark.parametrize("name", ["3.W_V_idx", "layer3.W_Q_idx", "W_Q_idx"])
def test_a_key_that_is_not_a_projection_is_refused(tmp_path, name):
    path = write(tmp_path, extra={name: torch.zeros(MODEL_DIM, INDEX_DIM)})
    with pytest.raises(ValueError, match="a projection file holds"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3, 7])


def test_a_distillation_checkpoint_is_not_a_projection_file(tmp_path):
    path = tmp_path / "maps.pt"
    torch.save({"maps": {3: {}}, "step": 10}, path)
    with pytest.raises(ValueError, match="a projection file holds"):
        weights.read(path, MODEL_DIM, INDEX_DIM, [3])


def test_the_layers_that_run_an_indexer_are_the_sparse_ones_that_do_not_reuse_a_top_k():
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ] * 2
    config = SimpleNamespace(num_hidden_layers=8, layer_types=layer_types)
    assert weights.indexer_layers(config, lambda config, layer: False) == [3, 7]
    assert weights.indexer_layers(config, lambda config, layer: layer == 7) == [3]
    # A model with no layer types indexes wherever the engine says it does
    plain = SimpleNamespace(num_hidden_layers=4)
    assert weights.indexer_layers(plain, lambda config, layer: layer % 2 == 1) == [0, 2]
