"""Which language runs each step of a read on which GPU - the one place a `kernels` setting meets the hardware.

Every step has a Triton kernel that runs on any CUDA GPU, and `kernels: triton` runs them all. Under
`kernels: cutedsl` a step runs its CuTeDSL kernel when the table below names the GPU, and the kernel's own
`supports` takes the rows; anything else runs the Triton kernel, which reads the same.
The choice is made in the front doors beside this file - [ward](ward.py), [expansion](expansion.py) and
[scan_select](scan_select.py) - and nowhere in the Triton modules.

| step | CuTeDSL kernel | GPUs |
|---|---|---|
| threshold estimate | [thresholds](cutedsl/thresholds.py) | H200, B300 |
| bucket scan | [hopper/bucket_scan](cutedsl/hopper/bucket_scan.py), [blackwell/bucket_scan](cutedsl/blackwell/bucket_scan.py) | H200, B300 |
| candidate scoring | [scoring](cutedsl/scoring.py) | H200, B300 |
| dense scoring | [scoring](cutedsl/scoring.py) | H200, B300 |
| expansion by threshold | [expansion](cutedsl/expansion.py) | H200, B300 |
| expansion in rank order | [expansion](cutedsl/expansion.py) | H200, B300 |
| beam step | [beam](cutedsl/beam.py) | H200, B300 |
| start-level keep | [beam](cutedsl/beam.py), and [keep](cutedsl/keep.py) for a row too wide to stage | H200, B300 |
| top-p pick | [beam](cutedsl/beam.py) | H200, B300 |
| node-bucket clustering | [hopper/node_bucket_clustering](cutedsl/hopper/node_bucket_clustering.py), its tiles on [blackwell/tile](cutedsl/blackwell/tile.py) | H200, B300 |
| ward clustering | [hopper/ward](cutedsl/hopper/ward.py), its tiles on [blackwell/tile](cutedsl/blackwell/tile.py) | H200, B300 |
| decode select | [decode](cutedsl/decode.py) | H200, B300 |
| short-context listing | [top_k](cutedsl/top_k.py) | H200, B300 |

A kernel under [cutedsl/hopper](cutedsl/hopper/__init__.py) uses an instruction only Hopper has, one under
[cutedsl/blackwell](cutedsl/blackwell/__init__.py) one only Blackwell has, and a step with both picks the one its GPU
has through [bucket_scan](cutedsl/bucket_scan.py), [node_bucket_clustering](cutedsl/node_bucket_clustering.py) or [ward](cutedsl/ward.py). The rest use
neither, so a GPU joins their rows once their tests pass on one. A B200 has every instruction a B300 has, and joins
the table once the tests have run on one.

A node-shaped level that carries children runs its beam step in CuTeDSL when its shapes are of rank 2.
Every other shaped score runs in Triton under either setting.
"""

from functools import cache

import torch

HOPPER = (9, 0)
B200 = (10, 0)
B300 = (10, 3)
# The GPUs that have `tcgen05` MMA and tensor memory, which a kernel under cutedsl/blackwell needs
BLACKWELL = frozenset({B200, B300})

# The GPUs each step's CuTeDSL kernel has been checked on, equal to the Triton kernel
CUTEDSL: dict[str, frozenset[tuple[int, int]]] = {
    "threshold estimate": frozenset({HOPPER, B300}),
    "bucket scan": frozenset({HOPPER, B300}),
    "candidate scoring": frozenset({HOPPER, B300}),
    "dense scoring": frozenset({HOPPER, B300}),
    "expansion by threshold": frozenset({HOPPER, B300}),
    "expansion in rank order": frozenset({HOPPER, B300}),
    "beam step": frozenset({HOPPER, B300}),
    "start-level keep": frozenset({HOPPER, B300}),
    "top-p pick": frozenset({HOPPER, B300}),
    "node-bucket clustering": frozenset({HOPPER, B300}),
    "ward clustering": frozenset({HOPPER, B300}),
    "decode select": frozenset({HOPPER, B300}),
    "short-context listing": frozenset({HOPPER, B300}),
}


@cache
def capability(device: torch.device) -> tuple[int, int]:
    """Return a CUDA device's compute capability, read once.

    Args:
        device: A CUDA device.
    """
    return torch.cuda.get_device_capability(device)


def runs_cutedsl(step: str, kernels: str, device: torch.device) -> bool:
    """Tell whether a step runs its CuTeDSL kernel - asked for, and checked on this GPU.

    Args:
        step: A step of the table.
        kernels: The `kernels` setting.
        device: The device the step's tensors are on.

    Raises:
        KeyError: If the table has no such step.
    """
    gpus = CUTEDSL[step]
    return kernels == "cutedsl" and device.type == "cuda" and capability(device) in gpus
