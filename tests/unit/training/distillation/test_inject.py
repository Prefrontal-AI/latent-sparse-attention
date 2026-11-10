"""An injection that cannot import safetensors fails before it touches the output directory."""

import sys

import pytest

from lsa.training.distillation import inject


@pytest.fixture
def without_safetensors(monkeypatch):
    # A None entry makes the import raise ImportError, as a sync without the training extra does
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)


def test_inject_leaves_no_output_directory_without_safetensors(tmp_path, without_safetensors):
    out = tmp_path / "injected"
    with pytest.raises(ImportError):
        inject.inject("anything", [], out, force=False)
    assert not out.exists()


def test_inject_qsa_leaves_no_output_directory_without_safetensors(tmp_path, without_safetensors):
    out = tmp_path / "injected"
    with pytest.raises(ImportError):
        inject.inject_qsa(tmp_path, {}, "anything", [], out, force=False)
    assert not out.exists()
