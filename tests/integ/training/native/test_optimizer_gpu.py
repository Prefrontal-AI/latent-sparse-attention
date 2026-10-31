"""Preserve a marked matrix through real FSDP2 and compare its AdamW updates."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from lsa.training.native.config import OptimConfig
from lsa.training.native.optimizer import optimizer_config, preserve_parameter_flags
from tests.helpers.training.native.optimizer import Parameters

pytestmark = pytest.mark.gpu


def test_decay_exclusions_and_updates_survive_real_fsdp_parameter_replacement(tmp_path):
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        model = Parameters()
        model.scale = nn.Parameter(torch.ones(1))
        expected = {name: parameter.detach().cuda().clone() for name, parameter in model.named_parameters()}
        mesh = init_device_mesh("cuda", (1,))

        def parallelize(model):
            model.cuda()
            return fully_shard(model, mesh=mesh)

        model = preserve_parameter_flags(model, parallelize=parallelize)
        config = optimizer_config(OptimConfig(lr=0.01, weight_decay=0.2, weight_decay_scope="matrices"))
        optimizer = config.build(model_parts=[model])
        groups = optimizer.optimizers[0].param_groups
        assert groups[0]["param_names"] == ["embedding.weight", "linear.weight"]
        assert "excluded" in groups[1]["param_names"]
        reference_groups = [
            {
                "params": [
                    value for name, value in expected.items() if name in ("embedding.weight", "linear.weight")
                ],
                "weight_decay": 0.2,
            },
            {
                "params": [
                    value
                    for name, value in expected.items()
                    if name not in ("embedding.weight", "linear.weight")
                ],
                "weight_decay": 0.0,
            },
        ]
        for value in expected.values():
            value.requires_grad_()
        reference = torch.optim.AdamW(
            reference_groups, lr=0.01, betas=(0.9, 0.95), eps=1e-8, fused=True, foreach=False
        )
        for _ in range(5):
            for name, parameter in model.named_parameters():
                parameter.grad = torch.ones_like(parameter)
                expected[name].grad = torch.ones_like(expected[name])
            optimizer.step()
            reference.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.full_tensor(), expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
