"""Run one reference training code kernel call in its own process, so its CUDA extensions never share one with lsa's.

Two builds of one vendored CUDA source in one process fail to launch, so the equivalence tests ask this worker for
the reference training code's answer and compare it with lsa's in the test's own process.

    python -m tests.helpers.training.kernels.reference_worker <request.pt> <answer.pt>
"""

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import torch

from tests.helpers.training.reference import reference_package

REFERENCE_ENV = "REFERENCE_CLONE"


def load(clone: str) -> str:
    """Make the reference training code's kernel modules importable without its layers package, and return their prefix."""
    root = reference_package(clone)
    sys.path.insert(0, str(root.parent))
    for relative in (Path(), Path("model"), Path("model/layers")):
        name = ".".join((root.name, *relative.parts))
        package = types.ModuleType(name)
        package.__path__ = [str(root / relative)]
        sys.modules.setdefault(name, package)
    return f"{root.name}.model.layers.ops."


def to_device(value: Any, ops: str) -> Any:
    """Move tensors to the GPU, and rebuild a selection from its two tensors."""
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict) and "selection" in value:
        selection = importlib.import_module(ops + "dsa_sparse.topk").Selection
        return selection(*(to_device(x, ops) for x in value["selection"]))
    if isinstance(value, tuple):
        return tuple(to_device(x, ops) for x in value)
    return value


def run(request: dict[str, Any], clone: str) -> dict[str, Any]:
    """Call the requested function, and return its tensors and the gradients asked for."""
    ops = load(clone)
    for module_name, name, value in request["patches"]:
        setattr(importlib.import_module(ops + module_name), name, value)
    module_name, function_name = request["function"].rsplit(".", 1)
    function = getattr(importlib.import_module(ops + module_name), function_name)
    args = [to_device(x, ops) for x in request["args"]]
    kwargs = {key: to_device(x, ops) for key, x in request["kwargs"].items()}
    for index in request["leaves"]:
        args[index] = args[index].detach().requires_grad_()
    result = function(*args, **kwargs)
    outputs = (
        [result] if isinstance(result, torch.Tensor) else [x for x in result if isinstance(x, torch.Tensor)]
    )
    answer: dict[str, Any] = {"outputs": [x.detach().cpu() for x in outputs]}
    if request["leaves"]:
        leaves = [args[index] for index in request["leaves"]]
        grads = torch.autograd.grad(outputs[0], leaves, request["grad_output"].cuda())
        answer["grads"] = [g.cpu() for g in grads]
    return answer


if __name__ == "__main__":
    request = torch.load(sys.argv[1], weights_only=False)
    torch.save(run(request, request["clone"]), sys.argv[2])
