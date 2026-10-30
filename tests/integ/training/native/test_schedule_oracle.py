"""Compare every schedule update to the functions from the reference training code checkout."""

import ast
import math
import os
from types import SimpleNamespace

import pytest

from lsa.training.native.schedule import decay_step_count, learning_rate_factor
from tests.helpers.training.reference import reference_package

pytestmark = pytest.mark.cluster


@pytest.mark.parametrize("shape", ["linear", "sqrt", "cosine"])
@pytest.mark.parametrize(
    ("steps", "warmup", "ratio"),
    [
        (17167, 859, 0.2),
        (95368, 954, 0.01),
        (3434, 96, 0.97),
        (8000, 100, 1.0),
    ],
)
def test_each_update_matches_the_actual_reference_training_code_schedule(shape, steps, warmup, ratio):
    clone = os.environ.get("REFERENCE_CLONE")
    if not clone:
        pytest.skip("set REFERENCE_CLONE to the reference training code main")
    source = reference_package(clone) / "training/optim.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("wsd_decay_steps", "wsd_lr_factor")
    ]
    namespace = {"math": math, "TrainingConfig": SimpleNamespace}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    decay_steps = namespace["wsd_decay_steps"](SimpleNamespace(wsd_decay_fraction=ratio), steps)
    assert decay_step_count(steps, ratio) == decay_steps
    for step in range(steps + 1):
        assert learning_rate_factor(
            step,
            steps=steps,
            warmup_steps=warmup,
            decay_steps=decay_steps,
            min_lr_factor=0.1,
            decay_shape=shape,
        ) == namespace["wsd_lr_factor"](step, warmup, steps, decay_steps, 0.1, shape)


@pytest.mark.parametrize("warmup", [0, 401])
def test_cosine_matches_the_actual_reference_training_code_builder_at_every_update(warmup):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    clone = os.environ.get("REFERENCE_CLONE")
    if not clone:
        pytest.skip("set REFERENCE_CLONE to the reference training code main")
    source = reference_package(clone) / "training/optim.py"
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("GroupCosineAnnealingLR", "_cosine")
    ]
    namespace = {
        "torch": torch,
        "Sequence": list,
        "CosineAnnealingLR": CosineAnnealingLR,
        "LinearLR": LinearLR,
        "SequentialLR": SequentialLR,
        "TrainingConfig": SimpleNamespace,
        "ScheduleSpan": SimpleNamespace,
        "Schedule": tuple,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    steps = 19074
    optimizer = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 3e-4},
            {"params": [torch.nn.Parameter(torch.ones(()))], "lr": 6e-4},
        ]
    )
    scheduler, per_step = namespace["_cosine"](
        optimizer,
        SimpleNamespace(warmup_steps=warmup),
        SimpleNamespace(total_steps=steps, step_based=True, floors=[3e-5, 6e-5]),
    )
    assert per_step
    for step in range(steps + 1):
        factor = learning_rate_factor(
            step,
            steps=steps,
            warmup_steps=warmup,
            decay_steps=1,
            min_lr_factor=0.1,
            decay_shape="sqrt",
            schedule="cosine",
        )
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            [3e-4 * factor, 6e-4 * factor], rel=1e-11, abs=1e-16
        )
        optimizer.step()
        scheduler.step()
