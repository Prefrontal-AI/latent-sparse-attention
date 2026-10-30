"""That a record's card is read from the device, named by one token everywhere, inferred only for a record that
predates the field, and never spanned by a join."""

import subprocess

import pytest

from lsa.evaluation import gpu

H200 = "NVIDIA H200"
B300 = "NVIDIA B300 SXM6 AC"
# Captured at import, before the autouse fixture replaces it for every test
DETECT = gpu.detect


def smi(monkeypatch, stdout="", error=None):
    def run(*args, **kwargs):
        if error:
            raise error
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    monkeypatch.setattr(gpu.subprocess, "run", run)


@pytest.mark.parametrize(("device", "short"), [(H200, "h200"), (B300, "b300")])
def test_a_device_has_one_token(device, short):
    assert gpu.token(device) == short
    assert gpu.resolve(short) == gpu.resolve(device) == device


def test_a_device_without_a_token_is_refused():
    with pytest.raises(ValueError, match="gpu.TOKENS"):
        gpu.token("NVIDIA A100")
    with pytest.raises(ValueError, match="neither a GPU token"):
        gpu.resolve("a100")


def test_a_name_ends_in_its_card_once():
    assert gpu.named("glm53_flash_lsa_flat_tp8", H200) == "glm53_flash_lsa_flat_tp8_h200"
    assert gpu.named("glm53_flash_lsa_flat_tp8_h200", H200) == "glm53_flash_lsa_flat_tp8_h200"


def test_the_card_is_the_one_every_gpu_of_the_node_reports(monkeypatch):
    smi(monkeypatch, f"{B300}\n{B300}\n")
    assert DETECT() == B300


@pytest.mark.parametrize("stdout", ["", f"{H200}\n{B300}\n"])
def test_a_node_with_no_gpu_or_two_models_is_refused(monkeypatch, stdout):
    smi(monkeypatch, stdout)
    with pytest.raises(RuntimeError, match="expected one GPU model"):
        DETECT()


def test_a_node_without_nvidia_smi_is_refused(monkeypatch):
    smi(monkeypatch, error=FileNotFoundError("nvidia-smi"))
    with pytest.raises(RuntimeError, match="cannot read the GPU model"):
        DETECT()


def test_a_launch_refuses_a_node_of_another_card():
    assert gpu.settle("") == H200
    assert gpu.settle("h200") == H200
    with pytest.raises(ValueError, match="submitted for"):
        gpu.settle("b300")


def record(**run):
    return {"schema": "x", "run": {"id": "r__1", **run}}


def test_two_known_cards_are_refused():
    first = record(gpu=H200)
    assert gpu.refusals(first, record(gpu=H200)) == []
    assert gpu.refusals(first, record(gpu=B300)) == [f"run.gpu '{H200}' vs '{B300}'"]
    # A record nothing names the card of is no evidence of another card
    assert gpu.refusals(record(), record(gpu=B300)) == []
