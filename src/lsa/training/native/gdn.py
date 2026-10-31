"""Preserve the GPT-2 hybrid's convolution and gated-normalization precision."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from fla.modules.convolution import causal_conv1d
from torch.nn import functional as functional
from torchtitan.models.qwen3_5.gdn import InnerGatedDeltaNet, RMSNormGated


def convolve_tokens(tokens: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Convolve packed documents with FP32 weights before rounding the SiLU output.

    Args:
        tokens: Projected tokens [T, C] bf16 or fp32.
        weight: Depthwise weights [C, 1, W] fp32.
        cu_seqlens: Document offsets [N + 1] int32.

    Returns:
        Activated convolutions [T, C] in the tokens' dtype.
    """
    if tokens.is_cuda:
        output, _ = cast(Callable[..., tuple[torch.Tensor, torch.Tensor | None]], causal_conv1d)(
            x=tokens.unsqueeze(0),
            weight=weight.squeeze(1),
            activation="silu",
            cu_seqlens=cu_seqlens,
            backend="triton",
        )
        return output.squeeze(0)
    offsets = cu_seqlens.tolist()
    documents = []
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            document = tokens[start:end].float().transpose(0, 1).unsqueeze(0)
            padded = functional.pad(document, (weight.shape[-1] - 1, 0))
            convolved = functional.conv1d(padded, weight.float(), groups=weight.shape[0])
            documents.append(functional.silu(convolved).squeeze(0).transpose(0, 1))
    return torch.cat(documents).to(tokens.dtype)


class Gpt2InnerGatedDeltaNet(InnerGatedDeltaNet):
    """Retain FP32 convolution weights inside torchtitan's local GDN boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerGatedDeltaNet.Config):
        """Settings of the kernel - the base's, with none added."""

        pass

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Convolve each document and run the existing recurrence with BF16 write gates.

        Args:
            query_TC: Projected queries [T, Ck] bf16.
            key_TC: Projected keys [T, Ck] bf16.
            value_TC: Projected values [T, Cv] bf16.
            a_TH: Raw decay gates [T, H] bf16.
            b_TH: Raw write gates [T, H] bf16.
            conv_q_weight_C1W: Query convolution weights [Ck, 1, W] fp32.
            conv_k_weight_C1W: Key convolution weights [Ck, 1, W] fp32.
            conv_v_weight_C1W: Value convolution weights [Cv, 1, W] fp32.
            A_log_H: Log decay rates [H] fp32.
            dt_bias_H: Step biases [H] fp32.
            cu_seqlens: Document offsets [N + 1] int32.
            key_head_dim: Query and key head dim.
            value_head_dim: Value head dim.

        Returns:
            Recurrent values [T, H, value_head_dim] bf16.
        """
        tokens = query_TC.shape[0]
        queries = convolve_tokens(query_TC, conv_q_weight_C1W, cu_seqlens).reshape(tokens, -1, key_head_dim)
        keys = convolve_tokens(key_TC, conv_k_weight_C1W, cu_seqlens).reshape(tokens, -1, key_head_dim)
        values = convolve_tokens(value_TC, conv_v_weight_C1W, cu_seqlens).reshape(tokens, -1, value_head_dim)
        decay = -torch.exp(A_log_H.float()) * functional.softplus(a_TH.float() + dt_bias_H)
        beta = torch.sigmoid(b_TH)
        return self.kernel(queries, keys, values, decay, beta, cu_seqlens=cu_seqlens)


class Gpt2RMSNormGated(RMSNormGated):
    """Round the gated normalized value after multiplying the FP32 output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNormGated.Config):
        """Settings of the norm - the base's, with none added."""

        pass

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize and gate in FP32 before restoring the input dtype.

        Args:
            x: Recurrent values [T, H, V] bf16 or fp32.
            gate: Output gates [T, H, V] in the input dtype.

        Returns:
            Gated normalized values [T, H, V] in the input dtype.
        """
        values = x.float()
        variance = values.square().mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps) * self.weight.float()
        return (normalized * functional.silu(gate.float())).to(x.dtype)
