"""Execute the recurrent initialization block from the installed FLA constructor."""

import ast
import inspect
import math
import textwrap
from types import SimpleNamespace

import torch
from fla.layers.gated_deltanet import GatedDeltaNet
from torch import nn


def fla_recurrent_parameters(head_count):
    source = inspect.getsource(GatedDeltaNet.__init__)
    body = ast.parse(textwrap.dedent(source)).body[0].body
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "A"
    )
    stop = next(i for i in range(start + 1, len(body)) if isinstance(body[i], ast.If))
    holder = SimpleNamespace(num_v_heads=head_count)
    namespace = dict(self=holder, torch=torch, nn=nn, math=math)
    exec(
        compile(ast.Module(body=body[start:stop], type_ignores=[]), inspect.getfile(GatedDeltaNet), "exec"),
        namespace,
    )
    return holder.A_log.detach(), holder.dt_bias.detach()
