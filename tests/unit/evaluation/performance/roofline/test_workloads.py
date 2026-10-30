"""Every step of the roofline's table builds and launches on a GPU, and the table matches dispatch's."""

import pytest
import torch

from lsa.core.kernels.dispatch import CUTEDSL
from lsa.evaluation.performance.roofline import workloads

LONG_CONTEXT = 262144
SHORT_CONTEXT = 131072


def test_steps_name_every_dispatch_step_and_the_anchor() -> None:
    names = [step.name for step in workloads.STEPS]
    assert sorted(names) == sorted([*CUTEDSL, "reference gemm"])
    assert len(set(names)) == len(names)


def test_step_named_refuses_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="no step named"):
        workloads.step_named("nothing")


def test_launcher_refuses_an_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown kernels"):
        workloads.launcher(workloads.STEPS[0], LONG_CONTEXT, "cuda")


def test_the_arms_are_both_languages() -> None:
    assert workloads.KERNELS == ("cutedsl", "triton")


@pytest.mark.gpu
@pytest.mark.parametrize("kernels", workloads.KERNELS)
@pytest.mark.parametrize("step", workloads.STEPS, ids=lambda step: step.name)
def test_step_launches_under_each_arm(step: workloads.Step, kernels: str) -> None:
    context = SHORT_CONTEXT if step.name == "short-context listing" else LONG_CONTEXT
    launch = workloads.launcher(step, context, kernels)
    assert launch is not None
    launch()
    torch.cuda.synchronize()


def test_batch_counts_mix_whole_short_and_shorter_rows() -> None:
    counts = workloads.batch_counts(1048576, 6)
    assert counts[0] == 1048576
    assert counts[1] % workloads.PAGE_ROWS == 5
    assert counts[2:4] == [1500, 700]
    assert counts[4:] == [1048576, 1048576]


def test_batch_counts_never_exceed_the_buckets() -> None:
    assert max(workloads.batch_counts(1000, 8)) <= 1000


@pytest.mark.gpu
@pytest.mark.parametrize("kernels", workloads.KERNELS)
def test_decode_select_batch_launches_and_replays_from_a_graph(kernels: str) -> None:
    from lsa.evaluation.performance.roofline import profile

    step = workloads.step_named("decode select")
    timings = profile.graph_timings(step, SHORT_CONTEXT, kernels, 2, [1, 4])
    assert sorted(timings) == ["1", "4"]
    assert all(value > 0 for value in timings.values())
