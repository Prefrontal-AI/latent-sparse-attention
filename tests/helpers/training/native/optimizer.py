"""Load upstream optimizer grouping without importing its trainer, and build small parameter fixtures."""

import ast
import dataclasses
import fnmatch
from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import nn

from lsa.training.native.config import OptimConfig
from lsa.training.native.optimizer import optimizer_config
from tests.helpers.training.reference import reference_package


def upstream_optimizer(clone):
    source = reference_package(clone) / "training/optim.py"
    wanted = {
        "canonical_name",
        "parse_lr_scale_rules",
        "ParamGroup",
        "parameter_flags",
        "build_param_groups",
        "torch_param_groups",
        "_adamw",
    }
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    namespace = dict(
        torch=torch,
        nn=nn,
        dataclasses=dataclasses,
        fnmatch=fnmatch,
        Sequence=Sequence,
        TrainingConfig=SimpleNamespace,
        WEIGHT_DECAY_SCOPES=("all", "matrices"),
        _WRAPPER_PARTS={"_checkpoint_wrapped_module", "_orig_mod"},
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


class Parameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 3)
        self.norm = nn.LayerNorm(3)
        self.linear = nn.Linear(3, 3)
        self.excluded = nn.Parameter(torch.randn(2, 3))
        self.excluded._no_weight_decay = True
        self.scale = nn.Parameter(torch.ones(()))


def replace_parameters(model, **kwargs):
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            module.register_parameter(name, nn.Parameter(parameter.detach().clone()))
    wrapped = nn.Module()
    wrapped.add_module("_checkpoint_wrapped_module", model)
    return wrapped


def build(model, scope):
    config = optimizer_config(OptimConfig(lr=0.01, weight_decay=0.2, weight_decay_scope=scope))
    config.implementation = "for-loop"
    return config.build(model_parts=[model])
