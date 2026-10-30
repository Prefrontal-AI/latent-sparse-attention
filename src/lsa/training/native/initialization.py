"""Initialize the GPT-2 hybrid's recurrent decays with the reference training code's FLA recipe."""

import math

import torch


@torch.no_grad()
def initialize_gated_decay(parameter: torch.Tensor) -> None:
    """Draw log decays from FLA's uniform positive decay distribution.

    Args:
        parameter: Per-value-head log decays [H], including a distributed parameter shard.
    """
    values = torch.empty_like(parameter, dtype=torch.float32).uniform_(0, 16).log_()
    parameter.copy_(values)


@torch.no_grad()
def initialize_gated_step_bias(parameter: torch.Tensor) -> None:
    """Draw FLA's inverse-softplus biases from log-uniform step sizes.

    Args:
        parameter: Per-value-head step biases [H], including a distributed parameter shard.
    """
    steps = torch.rand_like(parameter, dtype=torch.float32)
    steps = torch.exp(steps * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp_min_(1e-4)
    parameter.copy_(steps + torch.log(-torch.expm1(-steps)))
