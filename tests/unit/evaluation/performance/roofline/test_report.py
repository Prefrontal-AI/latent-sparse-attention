"""The roofline report's arithmetic, over counters written by hand."""

import pytest

from lsa.core.kernels import dispatch
from lsa.evaluation.performance.roofline import report


def test_derive_counts_fp32_and_tensor_flops() -> None:
    metrics = {
        report.FFMA: 10.0,
        report.FADD: 3.0,
        report.FMUL: 2.0,
        report.TENSOR_BF16: 100.0,
        report.DURATION: 1000.0,
        report.DRAM_READ: 300.0,
        report.DRAM_WRITE: 100.0,
        report.L2_BYTES: 800.0,
    }
    record = report.derive(metrics)
    assert record["fp32_flops"] == 25.0
    assert record["tensor_flops"] == 100.0 * report.TENSOR_FLOPS_PER_OP
    flops = 25.0 + 100.0 * report.TENSOR_FLOPS_PER_OP
    assert record["flops"] == flops
    assert record["duration_us"] == pytest.approx(1.0)
    assert record["dram_bytes"] == 400.0
    assert record["dram_intensity"] == flops / 400.0
    assert record["l2_intensity"] == flops / 800.0
    assert record["flops_per_second"] == pytest.approx(flops / 1e-6)
    assert record["dram_bytes_per_second"] == pytest.approx(400.0 / 1e-6)


def test_derive_leaves_rates_empty_without_time_or_bytes() -> None:
    record = report.derive({})
    assert record["flops"] == 0.0
    assert record["flops_per_second"] is None
    assert record["dram_intensity"] is None
    assert record["l2_intensity"] is None
    assert record["grid_size"] is None


def test_derive_treats_a_missing_counter_as_zero() -> None:
    record = report.derive({report.FFMA: None, report.FADD: 4.0, report.DURATION: 1.0})
    assert record["fp32_flops"] == 4.0


def test_machine_peaks_multiply_per_cycle_peaks_by_their_clock() -> None:
    metrics = {
        "dram__bytes.sum.peak_sustained": 2.0,
        "dram__cycles_elapsed.avg.per_second": 10.0,
        f"{report.FFMA}.peak_sustained": 3.0,
        report.CLOCK: 5.0,
    }
    peaks = report.machine_peaks(metrics)
    assert peaks["dram_bytes_per_second"] == 20.0
    assert peaks["fp32_flops"] == 30.0
    assert report.machine_peaks({})["fp32_flops"] is None


def test_classify_names_cutedsl_kernels() -> None:
    assert (
        report.classify(f"kernel_cutlass_kernel_{report.CUTEDSL_MARK}hopperlistingSurvivor_object_at_")
        == "cutedsl"
    )
    assert report.classify("cutlass_80_tensorop_s16816gemm") == "other"
    assert report.classify("_score_buckets") == "triton"
    assert report.classify("_score_buckets(...)") == "triton"
    assert report.classify("void at::native::vectorized_elementwise_kernel") == "other"


def test_derive_splits_tensor_flops_by_pipe() -> None:
    record = report.derive({report.TENSOR_BF16: 1.0, report.TENSOR_TF32: 2.0, report.TENSOR_FP8: 4.0})
    assert record["tensor_bf16_flops"] == 1.0
    assert record["tensor_tf32_flops"] == 2.0
    assert record["tensor_fp8_flops"] == 4.0
    assert record["tensor_flops"] == 7.0


def test_build_roofline_carries_stage_and_machine() -> None:
    kernel = {
        "label": "expansion by threshold@1024@cutedsl",
        "name": f"kernel_cutlass_kernel_{report.CUTEDSL_MARK}expansionFill",
        "metrics": {
            "device__attribute_multiprocessor_count": 132.0,
            "lts__t_bytes.sum.peak_sustained": 2.0,
            "lts__cycles_elapsed.avg.per_second": 3.0,
        },
        "device": "NVIDIA H200",
    }
    timings = [
        {"step": "expansion by threshold", "stage": "expansion", "context": 1024, "kernels": "cutedsl"}
    ]
    roofline = report.build_roofline([kernel], timings, "2025.2.1.0", "15c43801")
    assert roofline["machine"]["commit"] == "15c43801"
    assert roofline["kernels"][0]["stage"] == "expansion"
    assert roofline["kernels"][0]["kernels"] == "cutedsl"
    assert roofline["steps"][0]["kernels"] == "cutedsl"
    assert roofline["steps"][0]["launches"] == [0]
    assert roofline["machine"]["name"] == "NVIDIA H200"
    assert roofline["machine"]["sms"] == 132.0
    assert roofline["machine"]["ncu_version"] == "2025.2.1.0"
    assert roofline["machine"]["reported"]["l2_bytes_per_second"] == 6.0


def test_build_roofline_keeps_each_arms_launches_apart() -> None:
    def kernel(label: str, name: str) -> dict[str, object]:
        return {"label": label, "name": name, "metrics": {}, "device": "NVIDIA H200"}

    kernels = [
        kernel("expansion by threshold@1024@triton", "_score_buckets"),
        kernel(
            "expansion by threshold@1024@cutedsl",
            f"kernel_cutlass_kernel_{report.CUTEDSL_MARK}expansionFill",
        ),
        kernel("expansion by threshold@1024@triton", "void at::native::elementwise_kernel"),
    ]
    timings = [
        {"step": "expansion by threshold", "stage": "expansion", "context": 1024, "kernels": arm}
        for arm in ("triton", "cutedsl")
    ]
    roofline = report.build_roofline(kernels, timings)
    assert [record["kernels"] for record in roofline["kernels"]] == ["triton", "cutedsl", "triton"]
    assert [step["launches"] for step in roofline["steps"]] == [[0, 2], [1]]
    assert roofline["kernels"][2]["kind"] == "other"


def test_build_roofline_carries_the_h200_specification() -> None:
    roofline = report.build_roofline(
        [{"label": "fill@1024@cutedsl", "name": "a", "metrics": {}, "device": "NVIDIA H200"}], [], None
    )
    machine = roofline["machine"]
    assert machine["label"] == "H200"
    assert machine["specification"]["tensor_bf16_flops"] == 989.4e12
    assert machine["anchors"]["copy_bytes_per_second"] == 4224e9
    assert "bucket scan" in machine["cutedsl_steps"]


def test_build_roofline_names_the_b300_and_the_steps_it_runs_in_cutedsl() -> None:
    roofline = report.build_roofline(
        [{"label": "fill@1024@cutedsl", "name": "a", "metrics": {}, "device": "NVIDIA B300 SXM6 AC"}],
        [],
        None,
    )
    machine = roofline["machine"]
    assert machine["label"] == "B300"
    assert machine["capability"] == [10, 3]
    assert machine["specification"]["tensor_bf16_flops"] == 2250e12
    assert machine["specification"]["hbm_bytes_per_second"] == 8.0e12
    assert machine["specification_source"]
    assert set(machine["cutedsl_steps"]) == {
        step for step, gpus in dispatch.CUTEDSL.items() if dispatch.B300 in gpus
    }


def test_a_gpu_the_table_lacks_has_no_specification() -> None:
    assert report.machine_named("NVIDIA L40S") == {"label": "NVIDIA L40S", "specification": {}, "anchors": {}}
