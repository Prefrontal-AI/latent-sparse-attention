"""What an injected checkpoint carries, and every way of pairing a map with the wrong weights.

The pairing is the part worth testing. A map is a pair of matrices with no model in it, and two models of one
dim share every shape - so a map crossed between them makes a checkpoint that loads and selects the wrong
keys, and nothing downstream can tell.
"""

import json

import pytest
import torch

from lsa.io import file_sha256
from lsa.training.constants import Snapshot
from lsa.training.distillation import inject
from lsa.training.distillation.constants import Inject

LAYERS = [3, 7]
MODEL_DIM = 768
INDEX_DIM = 64


def a_checkpoint(tmp_path, layers=LAYERS, name="source", linear=()):
    """A sharded checkpoint shaped like a published one, with nothing real in the shard."""
    where = tmp_path / name
    where.mkdir()
    weight_map = {"model.language_model.embed_tokens.weight": "model-00001-of-00001.safetensors"}
    for layer in layers:
        for leaf in ("q_proj.weight", "k_proj.weight"):
            key = f"model.language_model.layers.{layer}.self_attn.{leaf}"
            weight_map[key] = "model-00001-of-00001.safetensors"
    for layer in linear:
        key = f"model.language_model.layers.{layer}.linear_attn.A_log"
        weight_map[key] = "model-00001-of-00001.safetensors"
    (where / Snapshot.INDEX).write_text(
        json.dumps({"metadata": {"total_size": 1024}, "weight_map": weight_map})
    )
    (where / Snapshot.CONFIG).write_text(json.dumps({"model_type": "qwen3_5"}))
    (where / "model-00001-of-00001.safetensors").write_bytes(b"not a real shard")
    (where / "tokenizer.json").write_text("{}")
    return where


def a_maps_file(tmp_path, model="source", layers=LAYERS, name="maps.pt", index_dim=INDEX_DIM):
    """A run's maps as `save_maps` writes them."""
    path = tmp_path / name
    torch.save(
        {
            "step": 6000,
            "maps": {
                layer: {
                    "query_projection": torch.randn(MODEL_DIM, index_dim),
                    "key_projection": torch.randn(MODEL_DIM, index_dim),
                    "log_tau": torch.zeros(()),
                }
                for layer in layers
            },
            "teacher": {
                "collection": "glm53_flash",
                "model": model,
                "layers": layers,
            },
        },
        path,
    )
    return path


def test_the_injected_checkpoint_carries_a_projection_for_every_indexed_layer(tmp_path):
    from safetensors.torch import load_file

    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    assert written.layers == LAYERS
    assert (written.model_dim, written.index_dim) == (MODEL_DIM, INDEX_DIM)
    shard = load_file(written.out / Inject.SHARD)
    for layer in LAYERS:
        assert inject.QUERY.format(layer=layer) in shard
        assert inject.KEY.format(layer=layer) in shard


def test_the_projections_are_the_maps_unchanged(tmp_path):
    from safetensors.torch import load_file

    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    saved = torch.load(maps, map_location="cpu", weights_only=False)["maps"]
    written = inject.inject(str(source), [maps], tmp_path / "out")

    shard = load_file(written.out / Inject.SHARD)
    for layer, item in saved.items():
        assert torch.equal(shard[inject.QUERY.format(layer=layer)], item["query_projection"])
        assert torch.equal(shard[inject.KEY.format(layer=layer)], item["key_projection"])


def test_the_weight_index_names_the_new_shard_and_keeps_the_old(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    index = json.loads((written.out / Snapshot.INDEX).read_text())
    assert index["weight_map"]["model.language_model.embed_tokens.weight"].startswith("model-00001")
    assert index["weight_map"][inject.QUERY.format(layer=3)] == Inject.SHARD
    assert index["metadata"]["total_size"] > 1024


def test_the_config_records_what_the_tensors_are(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    block = json.loads((written.out / Snapshot.CONFIG).read_text())[Inject.BLOCK]
    assert (block["model_dim"], block["index_dim"]) == (MODEL_DIM, INDEX_DIM)
    assert block["layers"] == LAYERS
    assert block["teacher"] == str(source)


def test_the_original_config_survives_the_added_block(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    assert json.loads((written.out / Snapshot.CONFIG).read_text())["model_type"] == "qwen3_5"


def test_the_weights_are_copied_rather_than_linked(tmp_path):
    """A link would point into the injecting user's own HuggingFace cache, which nobody else can read"""
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    for name in ("model-00001-of-00001.safetensors", "tokenizer.json", Snapshot.CONFIG, Snapshot.INDEX):
        assert not (written.out / name).is_symlink()
    shard = "model-00001-of-00001.safetensors"
    assert (written.out / shard).read_bytes() == (source / shard).read_bytes()


def test_the_config_block_names_the_maps_it_was_read_from(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    block = json.loads((written.out / Snapshot.CONFIG).read_text())[Inject.BLOCK]
    assert block["maps"] == [{"path": str(maps.resolve()), "sha256": file_sha256(maps)}]


def test_the_plugin_projection_file_is_written_beside_it(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    written = inject.inject(str(source), [maps], tmp_path / "out")

    flat = torch.load(written.out / Inject.PROJECTIONS, map_location="cpu", weights_only=True)
    assert set(flat) == {f"{layer}.W_{side}_idx" for layer in LAYERS for side in ("Q", "K")}


def test_two_halves_of_one_rung_become_one_checkpoint(tmp_path):
    """The 27B distils in two runs because sixteen layers will not fit, and an injected checkpoint wants one artifact"""
    source = a_checkpoint(tmp_path, layers=[3, 7, 11, 15])
    lower = a_maps_file(tmp_path, model=str(source), layers=[3, 7], name="lower.pt")
    upper = a_maps_file(tmp_path, model=str(source), layers=[11, 15], name="upper.pt")

    written = inject.inject(str(source), [lower, upper], tmp_path / "out")
    assert written.layers == [3, 7, 11, 15]
    assert written.tensors == 8


def test_a_map_for_another_model_is_refused(tmp_path):
    """Two models of one dim share every shape, so only the recorded name catches this"""
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model="a_different_model")

    with pytest.raises(ValueError, match="was distilled for"):
        inject.inject(str(source), [maps], tmp_path / "out")


def test_a_layer_the_checkpoint_does_not_have_is_refused(tmp_path):
    source = a_checkpoint(tmp_path, layers=[3])
    maps = a_maps_file(tmp_path, model=str(source), layers=[3, 7])

    with pytest.raises(ValueError, match="does not have"):
        inject.inject(str(source), [maps], tmp_path / "out")


def test_a_linear_attention_layer_is_refused(tmp_path):
    """Qwen3.5 runs gated delta net on three layers in four and an index applies to neither of them"""
    source = a_checkpoint(tmp_path, layers=[3], linear=[0])
    maps = a_maps_file(tmp_path, model=str(source), layers=[0])

    with pytest.raises(ValueError, match="runs linear attention on"):
        inject.inject(str(source), [maps], tmp_path / "out")


def test_a_layer_that_is_both_is_allowed(tmp_path):
    """GLM-5.3-Flash puts Kimi delta attention under `self_attn` too, so `self_attn` alone cannot refuse"""
    source = a_checkpoint(tmp_path, layers=[3], linear=[3])
    maps = a_maps_file(tmp_path, model=str(source), layers=[3])

    assert inject.inject(str(source), [maps], tmp_path / "out").layers == [3]


def test_two_files_covering_one_layer_are_refused(tmp_path):
    source = a_checkpoint(tmp_path)
    first = a_maps_file(tmp_path, model=str(source), name="first.pt")
    second = a_maps_file(tmp_path, model=str(source), name="second.pt")

    with pytest.raises(ValueError, match="already covered"):
        inject.inject(str(source), [first, second], tmp_path / "out")


def test_an_existing_output_is_kept_unless_forced(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_maps_file(tmp_path, model=str(source))
    out = tmp_path / "out"
    inject.inject(str(source), [maps], out)

    with pytest.raises(FileExistsError, match="--force"):
        inject.inject(str(source), [maps], out)
    assert inject.inject(str(source), [maps], out, force=True).layers == LAYERS


def test_a_checkpoint_with_no_weight_index_is_refused(tmp_path):
    source = tmp_path / "bare"
    source.mkdir()
    (source / Snapshot.CONFIG).write_text("{}")
    maps = a_maps_file(tmp_path, model=str(source))

    with pytest.raises(FileNotFoundError, match="has no model.safetensors.index.json"):
        inject.inject(str(source), [maps], tmp_path / "out")


def test_a_model_that_is_neither_a_card_nor_a_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither a checkpoint directory nor a registered model card"):
        inject.checkpoint_directory("not_a_model_anywhere")


def test_a_run_directory_resolves_its_finished_maps(tmp_path):
    source = a_checkpoint(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    a_maps_file(run, model=str(source))

    assert inject.inject(str(source), [run], tmp_path / "out").layers == LAYERS


def a_qsa_maps_file(tmp_path, model="source", layers=LAYERS, name="qsa_maps.pt"):
    """A run's QSA indexers as `save_maps` writes them, with the run's indexer block."""
    path = tmp_path / name
    qsa = {"query_head_count": 4, "head_dim": 128, "rotary_dim": 64, "block_size": 4, "top_k": 2048}
    torch.save(
        {
            "step": 48000,
            "maps": {
                layer: {
                    "index_qk_proj": torch.randn(5 * 128, MODEL_DIM),
                    "q_layernorm": torch.randn(128),
                    "k_layernorm": torch.randn(128),
                    "log_tau": torch.zeros(()),
                    "inverse_frequencies": torch.ones(32),
                    "attention_scaling": torch.tensor(1.0),
                }
                for layer in layers
            },
            "teacher": {"collection": "qwen38_27b", "model": model, "layers": layers},
            "run": {"model": {"qsa": qsa}},
        },
        path,
    )
    return path


def test_a_qsa_run_is_injected_as_the_server_loads_its_indexer(tmp_path):
    from safetensors.torch import load_file

    source = a_checkpoint(tmp_path)
    (source / Snapshot.CONFIG).write_text(json.dumps({"text_config": {"dtype": "bfloat16"}}))
    maps = a_qsa_maps_file(tmp_path, model=str(source))
    saved = torch.load(maps, map_location="cpu", weights_only=False)["maps"]
    written = inject.inject(str(source), [maps], tmp_path / "out")

    assert written.layers == LAYERS
    shard = load_file(written.out / Inject.SHARD)
    for layer in LAYERS:
        for name in inject.QSA_WEIGHTS:
            tensor = shard[inject.QSA_INDEXER.format(layer=layer, name=f"{name}.weight")]
            assert tensor.dtype == torch.bfloat16
            assert torch.equal(tensor, saved[layer][name].bfloat16())
    config = json.loads((written.out / Snapshot.CONFIG).read_text())
    fields = {
        name: config["text_config"][name] for name in config["text_config"] if name.startswith("indexer_")
    }
    assert fields == {
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
    assert config[Inject.BLOCK]["kind"] == "qsa"
    assert not (written.out / Inject.PROJECTIONS).exists()


def test_a_qsa_run_for_another_model_is_refused(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_qsa_maps_file(tmp_path, model="another")
    with pytest.raises(ValueError, match="distilled for"):
        inject.inject(str(source), [maps], tmp_path / "out")


def test_a_qsa_run_on_the_key_and_query_is_refused_until_the_server_reads_it(tmp_path):
    source = a_checkpoint(tmp_path)
    maps = a_qsa_maps_file(tmp_path, model=str(source))
    blob = torch.load(maps, weights_only=False)
    blob["run"]["model"]["qsa"]["input"] = "key_and_query"
    torch.save(blob, maps)
    with pytest.raises(ValueError, match="projects `x` alone"):
        inject.inject(str(source), [maps], tmp_path / "out")
