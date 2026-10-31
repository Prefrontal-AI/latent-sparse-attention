"""Measure the machine's own anchors for the roofline - a plain copy's bandwidth and cuBLAS's dense GEMM rates.

    python -m lsa.evaluation.performance.roofline.anchors --output anchors.json

A copy reads and writes a buffer far larger than the L2, so its bytes over its time is what HBM gives a stream. A GEMM is
square and large, so its FLOPs over its time is what the tensor cores give a kernel that is all MMA. Each is the best of
a few timed calls after a warm-up, with the clocks left free as the profile leaves them.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

# Bytes a copy moves each way - far past any L2
COPY_BYTES = 4 << 30
# The GEMMs' dimension, and the calls each is timed over
GEMM_DIM = 8192
REPEATS = 20


def best_seconds(launch: Callable[[], Any], repeats: int = REPEATS) -> float:
    """Return a launch's fastest time over some calls, after one to warm.

    Args:
        launch: The call to time.
        repeats: Timed calls.
    """
    launch()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1e-3)
    return min(times)


def copy_bytes_per_second() -> float:
    """Return the bytes a plain device copy reads and writes a second."""
    source = torch.empty(COPY_BYTES, device="cuda", dtype=torch.uint8)
    target = torch.empty_like(source)
    seconds = best_seconds(lambda: target.copy_(source))
    return 2 * COPY_BYTES / seconds


def gemm_flops(dtype: torch.dtype) -> float:
    """Return the FLOP/s of a square GEMM through cuBLAS.

    Args:
        dtype: `torch.bfloat16`, or `torch.float8_e4m3fn` for the scaled fp8 GEMM.
    """
    a = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda").to(dtype)
    b = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda").to(dtype)
    if dtype == torch.float8_e4m3fn:
        scale = torch.ones((), device="cuda")
        b = b.t().contiguous().t()
        seconds = best_seconds(lambda: torch._scaled_mm(a, b, scale, scale, out_dtype=torch.bfloat16))
    else:
        seconds = best_seconds(lambda: torch.mm(a, b))
    return 2 * GEMM_DIM**3 / seconds


def measure() -> dict[str, Any]:
    """Measure every anchor on the current GPU."""
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": properties.multi_processor_count,
        "copy_bytes_per_second": copy_bytes_per_second(),
        "bf16_gemm_flops": gemm_flops(torch.bfloat16),
        "fp8_gemm_flops": gemm_flops(torch.float8_e4m3fn),
        "torch": torch.__version__,
    }


def main() -> None:
    """Measure the anchors and write them as JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="the JSON to write")
    args = parser.parse_args()
    anchors = measure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anchors, indent=2))
    print(json.dumps(anchors, indent=2))


if __name__ == "__main__":
    main()
