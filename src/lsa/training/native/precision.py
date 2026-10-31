"""Keep native scoring in the same compute precision as its training run."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import torch
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger


@contextmanager
def autocast_context(
    *, context: Callable[[], AbstractContextManager[None]], device_type: str, enabled: bool
) -> Iterator[None]:
    """Combine the distributed context with optional BF16 autocast.

    Args:
        context: The model's distributed forward context.
        device_type: The device kind executing the forward.
        enabled: Whether compute uses BF16 with full-precision master parameters.

    Yields:
        The distributed and mixed-precision forward scope.
    """
    with context(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=enabled):
        yield


def allow_exact_fp32_matmul() -> None:
    """Let torchtitan's distributed setup proceed on a torch without BF16x9 matmul emulation.

    On compute capability 10.0 and later, `init_distributed` asks torch for BF16x9 emulation of FP32 matmuls and
    fails when torch does not know the precision.
    BF16x9 only speeds FP32 matmuls up, and training matmuls run under BF16 autocast anyway,
    so the fallback leaves FP32 matmuls at torch's default, which is exact FP32.
    Any other failure still raises, and a torch that supports the precision is set up as torchtitan does it.
    Calling this again changes nothing.
    """
    original = dist_utils.enable_fp32_matmul_emulation_with_bf16x9
    if getattr(original, "falls_back_to_exact_fp32", False):
        return

    def enable_or_fall_back() -> None:
        """Enable BF16x9 matmul emulation, leaving FP32 matmuls exact if torch does not know the precision."""
        try:
            original()
        except ValueError as exc:
            # Torchtitan wraps torch's rejection of the precision, and nothing else it raises has this cause
            if not isinstance(exc.__cause__, (AttributeError, RuntimeError)):
                raise
            logger.warning("torch has no BF16x9 matmul emulation, so FP32 matmuls stay at exact FP32")

    enable_or_fall_back.falls_back_to_exact_fp32 = True  # type: ignore[attr-defined]
    dist_utils.enable_fp32_matmul_emulation_with_bf16x9 = enable_or_fall_back
