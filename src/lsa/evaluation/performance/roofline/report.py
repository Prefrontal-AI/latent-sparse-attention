"""Read a profile's Nsight Compute report into one record per kernel and write the roofline JSON.

    lsa kernels roofline --output DIR      # the job writes DIR/roofline.json through this

A kernel's record holds its counters and what they imply - FLOPs, bytes at DRAM and L2, the intensities that place
it on the roofline - beside the machine's peaks, its specification and the anchors measured on it.
"""

import importlib
import json
import pkgutil
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from triton.runtime.jit import JITFunction

from lsa.core.kernels import cutedsl, dispatch
from lsa.io import git_sha

# Where Nsight Compute installs keep their Python report module
NCU_INSTALLS = Path("/opt/nvidia")

# The reference GEMM's bf16 counter reads 1.0156 * 2 * 8192^3, the 1.56% being its 320-row tiles covering 8192 in 26
# => a tensor counter already counts FLOPs, one each
TENSOR_FLOPS_PER_OP = 1.0

# Each GPU's dense specification in FLOP/s and bytes/s, and the anchors measured on it, by the name ncu reports
MACHINES: dict[str, dict[str, Any]] = {
    "H200": {
        "capability": (9, 0),
        "specification": {
            "tensor_bf16_flops": 989.4e12,
            "tensor_tf32_flops": 494.7e12,
            "tensor_fp8_flops": 1979e12,
            "fp32_flops": 66.9e12,
            "hbm_bytes_per_second": 4.8e12,
            "sm_count": 132,
        },
        "specification_source": "NVIDIA H200 datasheet, dense",
        "anchors": {"copy_bytes_per_second": 4224e9, "bf16_gemm_flops": 778e12},
        "anchors_source": "the roofline-cost-model experiment's H200 anchors",
    },
    "B300": {
        "capability": (10, 3),
        "specification": {
            "tensor_bf16_flops": 2250e12,
            "tensor_tf32_flops": 1125e12,
            "tensor_fp8_flops": 4500e12,
            "fp32_flops": 75e12,
            "hbm_bytes_per_second": 8.0e12,
            "sm_count": 148,
        },
        "specification_source": (
            "NVIDIA HGX B300 - 36, 18 and 72 PFLOPS bf16, tf32 and fp8 for 8 GPUs with sparsity, dense half that; "
            "8 TB/s HBM3e and 75 TFLOPS FP32 a GPU from the Blackwell Ultra datasheet"
        ),
        "anchors": {"copy_bytes_per_second": 6.58e12, "bf16_gemm_flops": 1787e12, "fp8_gemm_flops": 3558e12},
        "anchors_source": "the roofline anchors measured on a B300",
    },
}

# A CuTeDSL kernel is named after its Python module with the dots squeezed out
CUTEDSL_MARK = cutedsl.__name__.replace(".", "")

# A Triton kernel is named after its `triton.jit` function, read from the modules under lsa.core.kernels.triton
TRITON_PACKAGE = "lsa.core.kernels.triton"

FFMA = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
FADD = "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
FMUL = "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
TENSOR_BF16 = "sm__ops_path_tensor_src_bf16_dst_fp32.sum"
TENSOR_TF32 = "sm__ops_path_tensor_src_tf32_dst_fp32.sum"
TENSOR_FP8 = "sm__ops_path_tensor_src_fp8.sum"
INTEGER = "sm__sass_thread_inst_executed_op_integer_pred_on.sum"
DURATION = "gpu__time_duration.sum"
CLOCK = "sm__cycles_elapsed.avg.per_second"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
L2_BYTES = "lts__t_bytes.sum"
L1_BYTES = "l1tex__t_bytes.sum"

# The metrics a record keeps, by the name it keeps them under
SPEED_OF_LIGHT = {
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
}
LAUNCH = {
    "grid_size": "launch__grid_size",
    "block_size": "launch__block_size",
    "registers_per_thread": "launch__registers_per_thread",
    "dynamic_shared_bytes": "launch__shared_mem_per_block_dynamic",
    "static_shared_bytes": "launch__shared_mem_per_block_static",
    "cluster_size": "launch__cluster_dim_x",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
INSTRUCTIONS = {
    "instructions": "sm__inst_executed.sum",
    "integer_thread_instructions": INTEGER,
}
BYTES = {
    "dram_read_bytes": DRAM_READ,
    "dram_write_bytes": DRAM_WRITE,
    "l2_bytes": L2_BYTES,
    "l1_bytes": L1_BYTES,
}

# Peaks ncu reports for the machine, read from the first kernel
PEAKS = {
    "dram_bytes_per_cycle": "dram__bytes.sum.peak_sustained",
    "dram_cycles_per_second": "dram__cycles_elapsed.avg.per_second",
    "l2_bytes_per_cycle": f"{L2_BYTES}.peak_sustained",
    "l2_cycles_per_second": "lts__cycles_elapsed.avg.per_second",
    "ffma_per_cycle": f"{FFMA}.peak_sustained",
    "sm_count": "device__attribute_multiprocessor_count",
    "sm_clock": CLOCK,
}

# The device's name, a string metric
DEVICE_NAME = "device__attribute_display_name"

# Metrics ncu's full set does not collect
EXTRA_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            DURATION,
            CLOCK,
            DRAM_READ,
            DRAM_WRITE,
            L2_BYTES,
            L1_BYTES,
            FFMA,
            FADD,
            FMUL,
            TENSOR_BF16,
            TENSOR_TF32,
            TENSOR_FP8,
            INTEGER,
            "sm__inst_executed.sum",
            *SPEED_OF_LIGHT.values(),
            *LAUNCH.values(),
            *PEAKS.values(),
        ]
    )
)


def value_or_zero(metrics: Mapping[str, float | None], name: str) -> float:
    """Return a metric, or zero where the profile did not report it."""
    value = metrics.get(name)
    return 0.0 if value is None else value


def derive(metrics: Mapping[str, float | None]) -> dict[str, Any]:
    """Turn one kernel's counters into its record's numbers.

    Args:
        metrics: Metric name to value, None where the report lacks the metric.

    Returns:
        The counters kept by name, the FLOPs, and the derived rates and intensities - None where a rate divides
        by nothing.
    """
    fp32_flops = (
        2 * value_or_zero(metrics, FFMA) + value_or_zero(metrics, FADD) + value_or_zero(metrics, FMUL)
    )
    tensor_bf16_flops = TENSOR_FLOPS_PER_OP * value_or_zero(metrics, TENSOR_BF16)
    tensor_tf32_flops = TENSOR_FLOPS_PER_OP * value_or_zero(metrics, TENSOR_TF32)
    tensor_fp8_flops = TENSOR_FLOPS_PER_OP * value_or_zero(metrics, TENSOR_FP8)
    tensor_flops = tensor_bf16_flops + tensor_tf32_flops + tensor_fp8_flops
    flops = fp32_flops + tensor_flops
    seconds = value_or_zero(metrics, DURATION) * 1e-9
    dram_bytes = value_or_zero(metrics, DRAM_READ) + value_or_zero(metrics, DRAM_WRITE)
    l2_bytes = value_or_zero(metrics, L2_BYTES)
    record: dict[str, Any] = {
        "duration_us": value_or_zero(metrics, DURATION) * 1e-3,
        "clock_hz": metrics.get(CLOCK),
        "fp32_flops": fp32_flops,
        "tensor_bf16_flops": tensor_bf16_flops,
        "tensor_tf32_flops": tensor_tf32_flops,
        "tensor_fp8_flops": tensor_fp8_flops,
        "tensor_flops": tensor_flops,
        "flops": flops,
        "dram_bytes": dram_bytes,
        "flops_per_second": flops / seconds if seconds else None,
        "dram_intensity": flops / dram_bytes if dram_bytes else None,
        "l2_intensity": flops / l2_bytes if l2_bytes else None,
        "dram_bytes_per_second": dram_bytes / seconds if seconds else None,
    }
    for group in (BYTES, INSTRUCTIONS, SPEED_OF_LIGHT, LAUNCH):
        record.update({key: metrics.get(name) for key, name in group.items()})
    return record


def machine_peaks(metrics: Mapping[str, float | None]) -> dict[str, Any]:
    """Compute the machine's peaks ncu reports from one kernel's metrics.

    Args:
        metrics: Metric name to value for any one kernel.

    Returns:
        The raw peak metrics, and the DRAM and L2 bytes per second and FP32 FLOP/s they give.
    """
    raw = {key: metrics.get(name) for key, name in PEAKS.items()}
    dram_per_cycle, dram_clock = raw["dram_bytes_per_cycle"], raw["dram_cycles_per_second"]
    l2_per_cycle, l2_clock = raw["l2_bytes_per_cycle"], raw["l2_cycles_per_second"]
    ffma_per_cycle, clock = raw["ffma_per_cycle"], raw["sm_clock"]
    return {
        **raw,
        "l2_bytes_per_second": l2_per_cycle * l2_clock if l2_per_cycle and l2_clock else None,
        "dram_bytes_per_second": dram_per_cycle * dram_clock if dram_per_cycle and dram_clock else None,
        "fp32_flops": 2 * ffma_per_cycle * clock if ffma_per_cycle and clock else None,
    }


@cache
def triton_kernel_names() -> frozenset[str]:
    """Return the names of the `triton.jit` functions under lsa.core.kernels.triton."""
    package = importlib.import_module(TRITON_PACKAGE)
    names = set()
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{TRITON_PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            function = getattr(value, "fn", None)
            if isinstance(value, JITFunction) and function is not None:
                names.add(function.__name__)
    return frozenset(names)


def classify(name: str) -> str:
    """Tell a CuTeDSL or Triton kernel from the torch and cuBLAS kernels around it.

    Args:
        name: The kernel's name in the report.

    Returns:
        `cutedsl`, `triton` or `other`.
    """
    if CUTEDSL_MARK in name:
        return "cutedsl"
    return "triton" if name.split("(")[0] in triton_kernel_names() else "other"


def ncu_report_module() -> Any:
    """Import Nsight Compute's report module from its install, which is not on any path."""
    ncu = shutil.which("ncu")
    # The newest install first, as the newest ncu wrote the report
    roots = sorted(NCU_INSTALLS.glob("nsight-compute*/*/extras/python"), reverse=True)
    if ncu:
        install = Path(ncu).resolve().parent
        roots += sorted(install.glob("../../nsight-compute*/extras/python")) + sorted(
            install.glob("extras/python")
        )
    for root in roots:
        if (root / "ncu_report.py").is_file():
            sys.path.insert(0, str(root))
            return importlib.import_module("ncu_report")
    raise FileNotFoundError(f"no ncu_report.py under {', '.join(map(str, roots))}")


def range_label(action: Any) -> str | None:
    """Return the innermost NVTX push/pop range a kernel launched in.

    Args:
        action: A profiled kernel.
    """
    state = action.nvtx_state()
    for domain_id in state.domains():
        names = state.domain_by_id(domain_id).push_pop_ranges()
        if names:
            return names[-1]
    return None


def read_kernels(report_path: Path) -> list[dict[str, Any]]:
    """Read every profiled kernel of a report.

    Args:
        report_path: The `.ncu-rep` file.

    Returns:
        One dict per kernel - its `label`, `name` and metrics by name.
    """
    ncu_report = ncu_report_module()
    context = ncu_report.load_report(str(report_path))
    kernels = []
    for range_index in range(context.num_ranges()):
        profiled = context.range_by_idx(range_index)
        for action_index in range(profiled.num_actions()):
            action = profiled.action_by_idx(action_index)
            metrics: dict[str, float | None] = {}
            for name in action.metric_names():
                metric = action.metric_by_name(name)
                metrics[name] = metric.as_double() if metric.has_value() else None
            device = action.metric_by_name(DEVICE_NAME)
            kernels.append(
                {
                    "label": range_label(action),
                    "name": action.name(),
                    "metrics": metrics,
                    "device": device.as_string() if device is not None else None,
                }
            )
    return kernels


def ncu_version() -> str | None:
    """Return the installed Nsight Compute's version, or None where there is none."""
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if not Path(ncu).is_file():
        return None
    output = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout
    versions = [line.split(maxsplit=1)[1] for line in output.splitlines() if line.startswith("Version")]
    return versions[0] if versions else None


def machine_named(device: str | None) -> dict[str, Any]:
    """Return the specification and anchors of the GPU ncu names, and its short label.

    Args:
        device: The device's display name, as `NVIDIA B300 SXM6 AC`.

    Returns:
        The label, compute capability, specification and anchors with their sources, and the steps that run in
        CuTeDSL there - an empty specification for a GPU the table lacks.
    """
    for label, machine in MACHINES.items():
        if device and label in device.split():
            capability = machine["capability"]
            # The steps whose CuTeDSL kernel this GPU runs - the rest run in Triton under either setting
            cutedsl_steps = [step for step, gpus in dispatch.CUTEDSL.items() if capability in gpus]
            return {"label": label, **machine, "capability": list(capability), "cutedsl_steps": cutedsl_steps}
    return {"label": device, "specification": {}, "anchors": {}}


def build_roofline(
    kernels: Sequence[dict[str, Any]],
    timings: Sequence[dict[str, Any]],
    version: str | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Assemble the roofline JSON.

    Args:
        kernels: What `read_kernels` returned.
        timings: The profile's timing rows.
        version: Nsight Compute's version.
        commit: The short sha of the checkout the profile ran on.

    Returns:
        The machine, a record per kernel, and the steps - their timings with the indices of the kernels each launched.
    """
    stages = {row["step"]: row["stage"] for row in timings}
    records = []
    for kernel in kernels:
        label = kernel["label"] or ""
        step, _, rest = label.partition("@")
        context, _, arm = rest.partition("@")
        records.append(
            {
                "step": step,
                "stage": stages.get(step),
                "context": int(context) if context else None,
                "kernels": arm or None,
                "kernel": kernel["name"],
                "kind": classify(kernel["name"]),
                **derive(kernel["metrics"]),
            }
        )
    steps = [
        {
            **row,
            "launches": [
                index
                for index, record in enumerate(records)
                if record["step"] == row["step"]
                and record["context"] == row["context"]
                and record["kernels"] == row.get("kernels")
            ],
        }
        for row in timings
    ]
    metrics = kernels[0]["metrics"] if kernels else {}
    reported = machine_peaks(metrics)
    machine = {
        "name": kernels[0]["device"] if kernels else None,
        "sms": reported["sm_count"],
        "ncu_version": version,
        "commit": commit,
        **machine_named(kernels[0]["device"] if kernels else None),
        "reported": reported,
        "tensor_flops_per_op": TENSOR_FLOPS_PER_OP,
    }
    return {"machine": machine, "kernels": records, "steps": steps}


def write_roofline(report_path: Path, timings_path: Path, output: Path) -> Path:
    """Write the roofline JSON of a profile.

    Args:
        report_path: The `.ncu-rep` file.
        timings_path: The timings the profile wrote.
        output: The JSON to write.

    Returns:
        The JSON's path.
    """
    roofline = build_roofline(
        read_kernels(report_path),
        json.loads(timings_path.read_text()),
        ncu_version(),
        git_sha(Path(__file__).parent),
    )
    output.write_text(json.dumps(roofline, indent=2))
    return output
