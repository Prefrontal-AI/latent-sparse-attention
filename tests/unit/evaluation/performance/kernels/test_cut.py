import pytest

from lsa.evaluation.performance.kernels.data import cut


def test_the_default_snapshot_is_read_from_the_hub_cache_hf_home_names(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "models--zai-org--GLM-5.3-Flash" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert cut.default_snapshot() == f"{snapshot}/"


def test_a_cache_without_the_model_names_the_model_and_the_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    with pytest.raises(
        FileNotFoundError, match=r"zai-org/GLM-5\.3-Flash.*hf download zai-org/GLM-5\.3-Flash"
    ):
        cut.default_snapshot()
