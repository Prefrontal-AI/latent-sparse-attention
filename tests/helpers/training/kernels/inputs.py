"""Inputs, a dense reference and the reference training code loader the kernel tests share."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

WINDOW_SIZE = 512
SINK_SIZE = 4
POOL_SIZE = 4
BUCKET_SIZE = 8
BUCKET_CAP = 16
PAD = -1
REFERENCE_ENV = "REFERENCE_CLONE"


def require_hopper() -> None:
    """Skip a test whose kernel is built for sm_90a on any other device."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the kernel is built for sm_90a")


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """Return the norm of the difference over the norm of the reference."""
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def max_abs_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """Return the largest absolute difference."""
    return (got.detach().float().cpu() - want.detach().float().cpu()).abs().max().item()


def reference_clone() -> str:
    """Return the reference training code clone the environment names, or skip a test that needs it."""
    clone = os.environ.get(REFERENCE_ENV)
    if clone is None:
        pytest.skip(f"{REFERENCE_ENV} names no reference training code clone")
    return clone


def to_cpu(value: Any) -> Any:
    """Detach tensors and bring them to the CPU, inside a tuple or a selection too."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(to_cpu(x) for x in value)
    if isinstance(value, dict):
        return {key: to_cpu(x) for key, x in value.items()}
    if isinstance(value, list):
        return [to_cpu(x) for x in value]
    return value


def run_reference(
    function: str,
    args: list[Any],
    kwargs: dict[str, Any] | None = None,
    leaves: tuple[int, ...] = (),
    grad_output: torch.Tensor | None = None,
    patches: tuple[tuple[str, str, Any], ...] = (),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run a reference training code kernel function in a worker process.

    Args:
        function: Module and function under the reference training code's ops, as `cuda_sparse.attention`.
        args: Positional arguments; a Selection is passed as {"selection": [bits, occupancy]}.
        kwargs: Keyword arguments.
        leaves: Positions of the arguments to differentiate with respect to.
        grad_output: The gradient of the first output, when leaves is not empty.
        patches: Module, attribute and value to set on the reference training code's modules before the call.

    Returns:
        The function's tensor outputs and the gradients of the first with respect to the leaves, on the CPU.
    """
    clone = reference_clone()
    request = {
        "function": function,
        "args": [to_cpu(x) for x in args],
        "kwargs": kwargs or {},
        "leaves": list(leaves),
        "patches": list(patches),
        "grad_output": None if grad_output is None else grad_output.detach().cpu(),
        "clone": clone,
    }
    with tempfile.TemporaryDirectory() as directory:
        request_path, answer_path = Path(directory) / "request.pt", Path(directory) / "answer.pt"
        torch.save(request, request_path)
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.helpers.training.kernels.reference_worker",
                str(request_path),
                str(answer_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parents[4],
        )
        if done.returncode:
            raise RuntimeError(f"the reference training code worker failed:\n{done.stderr[-3000:]}")
        answer = torch.load(answer_path, weights_only=False)
    return answer["outputs"], answer.get("grads", [])


def unit_rows(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Draw unit-norm rows on the generator's device."""
    return F.normalize(torch.randn(*shape, device=generator.device, generator=generator), dim=-1)


def make_buckets(
    keys: torch.Tensor,
    block_size: int,
    seed: int,
    pool_size: int = POOL_SIZE,
    bucket_size: int = BUCKET_SIZE,
    bucket_cap: int = BUCKET_CAP,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Cut each closed block of unit keys into buckets of whole pools, of uneven sizes, in random order.

    Args:
        keys: Unit index keys [b, t, index_dim].
        block_size: Keys per block, a multiple of bucket_size and of pool_size.
        seed: Seed of the assignment.

    Returns:
        None when no block is closed, else unit centroids [b, m, index_dim] fp32 and members [b, m, bucket_cap] int64, each bucket's key positions in
        order and -1 past its size, buckets numbered block after block.
    """
    rng = np.random.default_rng(seed)
    batch, length, dim = keys.shape
    block_count = length // block_size
    if not block_count:
        return None
    buckets_per_block = block_size // bucket_size
    pools_per_bucket = bucket_size // pool_size
    pool_limit = bucket_cap // pool_size
    members = np.full((batch, block_count * buckets_per_block, bucket_cap), PAD, dtype=np.int64)
    for b in range(batch):
        for block in range(block_count):
            pools = rng.permutation(block_size // pool_size)
            sizes = np.full(buckets_per_block, pools_per_bucket)
            for _ in range(buckets_per_block):
                donor, taker = rng.integers(buckets_per_block, size=2)
                if sizes[donor] > 1 and sizes[taker] < pool_limit and donor != taker:
                    sizes[donor] -= 1
                    sizes[taker] += 1
            first = 0
            for bucket, size in enumerate(sizes):
                chosen = np.sort(pools[first : first + size])
                first += size
                positions = (chosen[:, None] * pool_size + np.arange(pool_size)).ravel() + block * block_size
                members[b, block * buckets_per_block + bucket, : len(positions)] = positions
    held = torch.from_numpy(members).to(keys.device)
    gathered = keys.float()[torch.arange(batch, device=keys.device)[:, None, None], held.clamp_min(0)]
    summed = (gathered * (held >= 0)[..., None]).sum(2)
    return F.normalize(summed, dim=-1), held


def random_selection(
    batch: int,
    length: int,
    top_k: int,
    generator: torch.Generator,
    window_size: int = WINDOW_SIZE,
    sink_size: int = SINK_SIZE,
) -> torch.Tensor:
    """Draw distinct positions in [sink_size, i - window_size] for each query, with holes and a short tail.

    Returns:
        Positions [b, t, top_k] int64, -1 where a query holds no key.
    """
    device = generator.device
    position = torch.arange(length, device=device)
    allowed = (position[None, :] >= sink_size) & (position[None, :] <= position[:, None] - window_size)
    scores = torch.rand(batch, length, length, device=device, generator=generator).masked_fill(~allowed, -1)
    values, ids = scores.topk(min(top_k, length), dim=-1)
    ids = torch.where(values >= 0, ids, PAD)
    ids = F.pad(ids, (0, top_k - ids.shape[-1]), value=PAD)
    holes = torch.rand(ids.shape, device=device, generator=generator) < 0.1
    return torch.where(holes, PAD, ids)


def keep_table(ids: torch.Tensor, length: int) -> torch.Tensor:
    """Return the bool table [b, t, length] of the positions a selection lists."""
    spread = torch.zeros(*ids.shape[:2], length + 1, dtype=torch.bool, device=ids.device)
    spread.scatter_(-1, torch.where(ids >= 0, ids, length).long(), True)
    return spread[..., :length]


def dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ids: torch.Tensor,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked fp32 attention over the window, the sink and the listed long-range keys, and its log-sum-exp."""
    length = q.shape[1]
    group = q.shape[2] // k.shape[2]
    position = torch.arange(length, device=q.device)
    delta = position[:, None] - position[None, :]
    near = (delta >= 0) & ((delta < window_size) | (position[None, :] < sink_size))
    long_range = (position[None, :] >= sink_size) & (delta >= window_size)
    keep = near[None] | (keep_table(ids, length) & long_range)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    kf, vf = kf.repeat_interleave(group, 1), vf.repeat_interleave(group, 1)
    scores = (qf @ kf.transpose(-1, -2) * scale).masked_fill(~keep[:, None], float("-inf"))
    return (scores.softmax(-1) @ vf).transpose(1, 2), scores.logsumexp(-1)


def attention_inputs(
    batch: int, length: int, query_heads: int, kv_heads: int, head_dim: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Generator]:
    """Draw bf16 q, k and v that require grad, and the generator that drew them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=generator).bfloat16().requires_grad_()

    return (
        draw(batch, length, query_heads, head_dim),
        draw(batch, length, kv_heads, head_dim),
        draw(batch, length, kv_heads, head_dim),
        generator,
    )
