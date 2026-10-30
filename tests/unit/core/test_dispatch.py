import pytest
import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import bucket_scan, node_bucket_clustering, ward

CUDA = torch.device("cuda", 0)
# A consumer Blackwell, which the dispatch table names for no step
UNNAMED_GPU = (12, 0)


@pytest.mark.parametrize("step", sorted(dispatch.CUTEDSL))
def test_every_step_runs_cutedsl_on_hopper_when_asked(monkeypatch, step):
    monkeypatch.setattr(dispatch, "capability", lambda device: dispatch.HOPPER)
    assert dispatch.runs_cutedsl(step, "cutedsl", CUDA)
    assert not dispatch.runs_cutedsl(step, "triton", CUDA)


@pytest.mark.parametrize("step", sorted(dispatch.CUTEDSL))
def test_a_gpu_no_kernel_was_checked_on_runs_triton(monkeypatch, step):
    monkeypatch.setattr(dispatch, "capability", lambda device: UNNAMED_GPU)
    assert not dispatch.runs_cutedsl(step, "cutedsl", CUDA)


def test_a_cpu_tensor_runs_triton():
    assert not dispatch.runs_cutedsl("bucket scan", "cutedsl", torch.device("cpu"))


def test_an_unknown_step_is_refused():
    with pytest.raises(KeyError):
        dispatch.runs_cutedsl("merging", "cutedsl", torch.device("cpu"))


@pytest.mark.parametrize("step", sorted(dispatch.CUTEDSL))
def test_every_step_runs_cutedsl_on_a_b300_when_asked(monkeypatch, step):
    monkeypatch.setattr(dispatch, "capability", lambda device: dispatch.B300)
    assert dispatch.runs_cutedsl(step, "cutedsl", CUDA)
    assert not dispatch.runs_cutedsl(step, "triton", CUDA)


def test_a_b200_waits_for_its_tests(monkeypatch):
    monkeypatch.setattr(dispatch, "capability", lambda device: dispatch.B200)
    assert dispatch.B200 in dispatch.BLACKWELL
    assert not any(dispatch.runs_cutedsl(step, "cutedsl", CUDA) for step in dispatch.CUTEDSL)


@pytest.mark.parametrize(
    ("capability", "built_for"), [((9, 0), "hopper"), ((10, 0), "blackwell"), ((10, 3), "blackwell")]
)
def test_a_step_with_two_kernels_takes_the_one_built_for_the_gpu(monkeypatch, capability, built_for):
    monkeypatch.setattr(dispatch, "capability", lambda device: capability)
    for module in (bucket_scan, node_bucket_clustering, ward):
        assert f"cutedsl.{built_for}." in module.kernel_for(CUDA).__name__


def test_a_cpu_tensor_takes_the_hopper_module():
    assert bucket_scan.kernel_for(torch.device("cpu")).__name__.endswith("hopper.bucket_scan")
