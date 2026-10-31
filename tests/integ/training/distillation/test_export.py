"""That a trained space reaches the server unchanged, and that the plugin's loader accepts it."""

import json

import pytest
import torch

from lsa.io import file_sha256
from lsa.sglang.index import weights
from lsa.training.distillation.checkpoint import checkpoint
from lsa.training.distillation.export import export, map_tau, projections, write
from lsa.training.distillation.model import IndexMap


def a_checkpoint(layers=(3, 7), model_dim=512, index_dim=64):
    maps = {layer: IndexMap(model_dim, index_dim) for layer in layers}
    return checkpoint(
        maps,
        {
            "run": {"name": "unit"},
            "step": 0,
            "teacher": {"model": "glm53_flash"},
        },
    ), maps


def test_every_layer_arrives_under_the_name_the_plugin_looks_it_up_by():
    blob, _ = a_checkpoint(layers=(3, 7, 11))

    flat = projections(blob)

    assert sorted(flat) == ["11.W_K_idx", "11.W_Q_idx", "3.W_K_idx", "3.W_Q_idx", "7.W_K_idx", "7.W_Q_idx"]


def test_the_maps_are_written_as_they_were_trained(tmp_path):
    """A transform on the way out would make the served numbers a different experiment from the scored ones"""
    blob, maps = a_checkpoint(layers=(3,))

    flat = projections(blob)

    assert torch.equal(flat["3.W_Q_idx"], maps[3].query_projection.detach())
    assert torch.equal(flat["3.W_K_idx"], maps[3].key_projection.detach())


def test_the_plugins_own_loader_reads_what_was_written(tmp_path):
    blob, maps = a_checkpoint(layers=(3,), index_dim=64)
    out = tmp_path / "projections.pt"
    torch.save(projections(blob), out)

    query, key = weights.read(out, 512, 64, [3])[3]

    assert query.shape == (512, 64)
    assert torch.equal(key, maps[3].key_projection.detach().to(key.dtype))


def test_a_layer_the_run_never_trained_is_refused_by_name(tmp_path):
    blob, _ = a_checkpoint(layers=(3,))
    out = tmp_path / "projections.pt"
    torch.save(projections(blob), out)

    with pytest.raises(ValueError, match=r"missing \[7\]"):
        weights.read(out, 512, 64, [3, 7])


def test_an_exported_map_names_the_run_that_trained_it(tmp_path):
    """A projection file is a flat tensor dict, so the sidecar beside it is the only link back to its run"""
    run = tmp_path / "long_16k_1m_dense_seed1"
    run.mkdir()
    blob, _ = a_checkpoint(layers=(3,))
    torch.save(blob, run / "maps.pt")
    source = {
        "path": "configs/training/distillation/glm53_flash/long_16k_1m_dense.yaml",
        "sweep": ["run.optim.seed=1"],
        "overrides": [],
    }
    commit = {"sha": "0" * 40, "dirty": False}
    (run / "meta.json").write_text(
        json.dumps({"config": source, "git": commit, "date": "2026-10-01T10:00:00+00:00"})
    )
    out = tmp_path / "maps" / "glm-5.3-flash__long-16k-1M-dense-seed1.pt"

    export(run, out)

    written = json.loads(out.with_suffix(".json").read_text())
    assert written["run"] == str(run.resolve())
    assert written["checkpoint"] == "maps.pt"
    assert written["config"] == source
    assert written["git"] == commit
    assert written["date"] == "2026-10-01"
    assert written["sha256"] == file_sha256(out)


def test_a_directory_is_read_as_the_run_it_holds(tmp_path):
    blob, _ = a_checkpoint(layers=(3,))
    torch.save(blob, tmp_path / "maps.pt")

    flat = write(tmp_path, tmp_path / "out" / "projections.pt")

    assert (tmp_path / "out" / "projections.pt").exists()
    assert sorted(flat) == ["3.W_K_idx", "3.W_Q_idx"]


def test_a_checkpoint_with_no_maps_says_so():
    with pytest.raises(ValueError, match="holds no 'maps'"):
        projections({"run": {}})


def test_a_missing_checkpoint_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="maps.pt"):
        write(tmp_path, tmp_path / "out.pt")


def test_a_projection_file_is_re_exported_as_it_is_with_its_input_and_source(tmp_path):
    """The maps distilled on the key latent are served under key_and_query, their tensors untouched"""
    flat = {"3.W_Q_idx": torch.randn(512, 64), "3.W_K_idx": torch.randn(512, 64)}
    source = tmp_path / "glm53_flash__long-16k-1M.pt"
    torch.save(flat, source)
    provenance = {
        "run": "long_16k_1m_kappa64",
        "checkpoint": "maps.pt",
        "config": {"path": "a.yaml"},
        "git": {},
    }
    source.with_suffix(".json").write_text(json.dumps(provenance))
    out = tmp_path / "glm53_flash__long-16k-1M-latent.pt"

    export(source, out, "key_and_query")

    written = torch.load(out, weights_only=True)
    assert sorted(written) == sorted(flat)
    assert all(torch.equal(written[name], flat[name]) for name in flat)
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["input"] == "key_and_query"
    assert sidecar["source"] == str(source.resolve())
    assert sidecar["source_sha256"] == file_sha256(source)
    assert sidecar["run"] == "long_16k_1m_kappa64"
    assert sidecar["sha256"] == file_sha256(out)


def test_an_export_never_overwrites_the_file_it_reads(tmp_path):
    source = tmp_path / "maps.pt"
    torch.save({"3.W_Q_idx": torch.randn(8, 4), "3.W_K_idx": torch.randn(8, 4)}, source)
    with pytest.raises(ValueError, match="beside it"):
        export(source, source, "key_and_query")


def test_an_input_the_design_lacks_is_refused(tmp_path):
    with pytest.raises(ValueError, match="input must be one of"):
        export(tmp_path, tmp_path / "out.pt", "latent")


def test_a_run_on_the_key_and_query_is_written_with_its_bases_folded_in_and_its_input_recorded(tmp_path):
    """What the server reads off the plain key latent is the map the run trained, so no re-export follows"""
    bases = (torch.randn(512, 512), torch.randn(512, 512))
    maps = {3: IndexMap(512, 64, bases=bases)}
    run = {"model": {"indexer": {"input": "key_and_query", "query_heads": "all"}}}
    torch.save(
        checkpoint(maps, {"run": run, "step": 0, "teacher": {"model": "glm53_flash"}}), tmp_path / "maps.pt"
    )
    out = tmp_path / "projections.pt"

    export(tmp_path, out)

    query, key = weights.read(out, 512, 64, [3])[3]
    served_query, served_key = maps[3].served_projections()
    assert torch.allclose(key, served_key.detach()) and torch.allclose(query, served_query.detach())
    assert json.loads(out.with_suffix(".json").read_text())["input"] == "key_and_query"
    with pytest.raises(ValueError, match="trained on input 'key_and_query'"):
        export(tmp_path, out, "attention_input")


def test_an_exported_run_records_the_temperature_each_layer_was_trained_at(tmp_path):
    maps = {3: IndexMap(512, 64), 7: IndexMap(512, 64)}
    maps[3].set_tau(0.05)
    maps[7].set_tau(0.2)
    torch.save(
        checkpoint(maps, {"run": {}, "step": 0, "teacher": {"model": "glm53_flash"}}), tmp_path / "maps.pt"
    )
    out = tmp_path / "projections.pt"

    export(tmp_path, out)

    assert json.loads(out.with_suffix(".json").read_text())["tau"] == pytest.approx({"3": 0.05, "7": 0.2})
    assert map_tau(out, [7, 3]) == pytest.approx({7: 0.2, 3: 0.05})


def test_a_projection_file_re_exported_keeps_the_temperature_its_own_sidecar_had(tmp_path):
    source = tmp_path / "maps.pt"
    torch.save({"3.W_Q_idx": torch.randn(8, 4), "3.W_K_idx": torch.randn(8, 4)}, source)
    source.with_suffix(".json").write_text(json.dumps({"tau": {"3": 0.1}}))
    out = tmp_path / "again.pt"

    export(source, out, "key_and_query")

    assert map_tau(out, [3]) == {3: 0.1}


def test_a_map_with_no_temperature_or_a_layer_short_of_one_says_how_to_calibrate(tmp_path):
    out = tmp_path / "maps.pt"
    torch.save({"3.W_Q_idx": torch.randn(8, 4), "3.W_K_idx": torch.randn(8, 4)}, out)
    with pytest.raises(ValueError, match="lsa distill calibrate-tau"):
        map_tau(out, [3])
    out.with_suffix(".json").write_text(json.dumps({"tau": {"3": 0.1}}))
    with pytest.raises(ValueError, match=r"layers \[7\].*lsa distill calibrate-tau"):
        map_tau(out, [3, 7])
