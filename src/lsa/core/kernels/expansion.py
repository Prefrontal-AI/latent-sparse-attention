"""Expansion in the language a step runs in - the one place `expand_in_order` and `expand_by_threshold` choose it.

Under `kernels: cutedsl` a step runs its CuTeDSL kernel when [dispatch](dispatch.py)'s table names the GPU and the
kernel's own `supports` takes the tensors; anything else runs the [Triton kernel](triton/expansion.py), which reads
the same.
"""

import torch

from lsa.budget import EarlyBudget
from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import expansion as cutedsl_expansion
from lsa.core.kernels.triton import expansion as triton_expansion


def expand_in_order(
    selected: torch.Tensor,
    bucket_members: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    slot_rows: torch.Tensor | None = None,
    kernels: str = "triton",
    early: EarlyBudget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read lists from buckets of any size, walking the picks best first until the top-k is full.

    Args:
        selected: Picked cache rows [R, top_p] int32, best score first, -1 padded.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        positions: Each row's position [R] int32.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.
        slot_table: Each request's positions to KV slots [requests, context] int32 - when given, every position
            is stored as its slot instead, and `slot_rows` names each row's request.
        slot_rows: Each row's request [R] int64, read with `slot_table`.
        kernels: `cutedsl` to read a row in one CTA where the picks allow it, `triton` in one program - the same
            read either way.
        early: The budget of rows before its position, or None for `top_k` and every pick throughout.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded, the expansion first.
        Keys each list holds [R] int32.
    """
    if dispatch.runs_cutedsl(
        "expansion in rank order", kernels, selected.device
    ) and cutedsl_expansion.supports_rank_order(selected, bucket_members):
        return cutedsl_expansion.expand_in_order(
            selected, bucket_members, positions, window_size, sink_size, top_k, slot_table, slot_rows, early
        )
    return triton_expansion.expand_in_order(
        selected, bucket_members, positions, window_size, sink_size, top_k, slot_table, slot_rows, early
    )


def expand_by_threshold(
    keys: torch.Tensor,
    listed: torch.Tensor,
    bucket_members: torch.Tensor,
    bucket_sizes: torch.Tensor,
    request_blocks: torch.Tensor,
    request_rows: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    cluster_block_size: int,
    top_p: int,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    block_positions: torch.Tensor | None = None,
    kernels: str = "triton",
    early: EarlyBudget | None = None,
) -> torch.Tensor:
    """Fill each row's read from its listed buckets by the key a best-first walk of its top-p stops at.

    Args:
        keys: Listed packed keys [R, C] int64, the empty key after a row's listed count.
        listed: Keys each row listed [R] int32, at most C.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        bucket_sizes: Members of each cache row [rows] int32.
        request_blocks: Pages of each request [requests, pages] int32, read through a request's logical buckets.
        request_rows: Each row's request [R] int64.
        positions: Each row's position [R] int32.
        page_size: Cache rows a page holds.
        cluster_block_size: Positions a cluster block covers.
        top_p: Best buckets the walk reads at most.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.
        slot_table: Each request's positions to KV slots [requests, context] int32 - positions when None.
        block_positions: Each page's first position [pages in the cache] int32 - when given, the keys hold cache
            rows rather than a request's logical buckets, and `request_blocks` is not read.
        kernels: `cutedsl` to fill a row in one CTA, `triton` in one program - the same read either way.
        early: The budget of rows before its position, or None for `top_p` and `top_k` throughout. The listing holds
            the design's own top-p, so it holds every early row's.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded.
    """
    arguments = (
        keys,
        listed,
        bucket_members,
        bucket_sizes,
        request_blocks,
        request_rows,
        positions,
        page_size,
        cluster_block_size,
        top_p,
        window_size,
        sink_size,
        top_k,
        slot_table,
        block_positions,
        early,
    )
    if dispatch.runs_cutedsl("expansion by threshold", kernels, keys.device) and cutedsl_expansion.supports(
        keys, bucket_members, request_blocks
    ):
        return cutedsl_expansion.expand_by_threshold(*arguments)
    return triton_expansion.expand_by_threshold(*arguments)
