"""The Triton code generation the CuTeDSL thresholds copy, read off the compiled kernel.

One GPU. `cutedsl/thresholds.py` sums a row's samples in the order Triton's `_histogram_thresholds` does, and that order
depends on facts no Triton source states - which sample counts Triton loads four wide, and how it divides.
These tests fail on those facts, so a Triton change is named here and not found as a threshold mismatch.
"""

import re

import pytest
import torch
import triton

from lsa.core.kernels.constants import ThresholdEstimate
from lsa.core.kernels.triton import scan_select as triton_scan_select

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
# Triton specializes an integer argument divisible by 16, which is what lets it load four fp32 words at once
# => the CuTeDSL thresholds take `four_wide = sample_count % 16 == 0`
DIVISIBILITY = 16
FOUR_WIDE = [DIVISIBILITY, 3 * DIVISIBILITY, 18432, 36864]
NOT_FOUR_WIDE = [4, 8, 20, 40, 20001, 20004, 20008, 24570]


def compiled_histogram(sample_count):
    rows = 1
    scores = torch.zeros(rows, sample_count, device=DEVICE)
    visible_counts = torch.full((rows,), sample_count * 3, device=DEVICE, dtype=torch.int32)
    norms = torch.ones(rows, device=DEVICE)
    thresholds = torch.empty(rows, device=DEVICE)
    return triton_scan_select._histogram_thresholds[(rows,)](
        scores,
        visible_counts,
        norms,
        thresholds,
        sample_count,
        3,
        10,
        BINS=ThresholdEstimate.BINS,
        BLOCK=ThresholdEstimate.BLOCK,
        num_warps=4,
    )


def is_four_wide(kernel):
    # A blocked layout of four elements a thread is Triton's four-wide load of the samples
    return "sizePerThread = [4]" in kernel.asm["ttgir"]


def sample_count_is_specialized(kernel):
    signature = re.search(r"%sample_count: i32 (\{tt\.divisibility = (\d+) : i32\})?", kernel.asm["ttgir"])
    return signature is not None and signature.group(2) == str(DIVISIBILITY)


@pytest.mark.parametrize("sample_count", FOUR_WIDE)
def test_a_multiple_of_16_samples_is_loaded_four_wide(sample_count):
    kernel = compiled_histogram(sample_count)
    assert sample_count_is_specialized(kernel), f"Triton {triton.__version__} no longer specializes on 16"
    assert is_four_wide(kernel), (
        f"Triton {triton.__version__} no longer loads {sample_count} samples four wide"
    )


@pytest.mark.parametrize("sample_count", NOT_FOUR_WIDE)
def test_any_other_sample_count_is_loaded_one_at_a_time(sample_count):
    kernel = compiled_histogram(sample_count)
    assert not sample_count_is_specialized(kernel), f"Triton {triton.__version__} specializes {sample_count}"
    assert not is_four_wide(kernel), f"Triton {triton.__version__} loads {sample_count} samples four wide"


def test_a_warp_adds_by_a_butterfly_from_the_widest_step_down():
    ptx = compiled_histogram(20001).asm["ptx"]
    offsets = [int(offset) for offset in re.findall(r"shfl\.sync\.bfly\.b32\s+%\w+,\s*%\w+,\s*(\d+),", ptx)]
    assert offsets[:5] == [16, 8, 4, 2, 1], (
        f"Triton {triton.__version__} adds a warp in another order: {offsets}"
    )


def test_a_mean_is_divided_with_the_approximate_division():
    ptx = compiled_histogram(20001).asm["ptx"]
    assert "div.full.f32" in ptx, f"Triton {triton.__version__} no longer divides with div.full"
    assert "div.rn.f32" not in ptx
