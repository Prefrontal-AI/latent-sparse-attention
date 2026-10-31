"""Expansion - the picked buckets read as their keys, and the window and the sink folded in after.

Two forms choose the same keys. `expand_in_order` walks the picks best first and stops once the top-k is full.
`expand_by_threshold` finds the lowest packed key at which the listed buckets fill the top-k and emits every
bucket above it in listing order, so it needs no sort. A key inside the window or the sink is not counted
toward the top-k, because attention reads it anyway, and both forms close with the same window-and-sink tail.

Both write a read list per row, `[top_k + window_size + sink_size]` int32 and -1 padded, and its length.
Under an early budget a row before its position reads the early `top_k` and `top_p` instead, chosen on the device from the
row's position, and its list stays -1 padded to the same width.
"""

import torch
import triton
import triton.language as tl

from lsa.budget import EarlyBudget
from lsa.core.kernels.triton.keys import key_bucket


@triton.jit
def _append_window_and_sink(
    out_row_ptr: tl.tensor,
    slot_row_ptr: tl.tensor,
    position: tl.tensor,
    window_start: tl.tensor,
    filled: tl.tensor,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
) -> tl.tensor:
    """Append the window's last positions then the sinks after `filled` keys, the one tail of a read.

    Args:
        out_row_ptr: The row's read list int32.
        slot_row_ptr: The row's positions to KV slots int32 - read only if MAP_SLOTS.
        position: The row's position.
        window_start: First position of the window.
        filled: Keys already in the read list.

    Returns:
        Keys the read list holds.
    """
    local = tl.arange(0, LOCAL_BLOCK)
    in_window = local < WINDOW_SIZE
    window_id = position + 1 - WINDOW_SIZE + local
    in_sinks = (local >= WINDOW_SIZE) & (local < WINDOW_SIZE + SINK_SIZE)
    sink_id = local - WINDOW_SIZE
    ids = tl.where(in_window, window_id, sink_id)
    valid = (in_window & (window_id >= 0)) | (in_sinks & (sink_id < window_start) & (sink_id <= position))
    at = filled + tl.cumsum(valid.to(tl.int32), 0) - 1
    if MAP_SLOTS:
        ids = tl.load(slot_row_ptr + ids, mask=valid, other=-1)
    tl.store(out_row_ptr + at, ids, mask=valid)
    return filled + tl.sum(valid.to(tl.int32), 0)


@triton.jit
def _row_budget(
    position: tl.tensor,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """Return a row's top-k and top-p - the early budget's before its position, the design's from it.

    No early budget (EARLY_POSITION 0) leaves both constants.
    """
    top_k = TOP_K
    top_p = TOP_P
    if EARLY_POSITION > 0:
        early = position < EARLY_POSITION
        top_k = tl.where(early, EARLY_TOP_K, TOP_K)
        top_p = tl.where(early, EARLY_TOP_P, TOP_P)
    return top_k, top_p


@triton.jit
def _expand_in_order(
    selected_ptr: tl.tensor,
    bucket_members_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    slot_table_ptr: tl.tensor,
    slot_rows_ptr: tl.tensor,
    selected_stride: tl.tensor,
    out_stride: tl.tensor,
    slot_stride: tl.tensor,
    TOP_P: tl.constexpr,
    BUCKET_CAP: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> None:
    """Expand one row's picks best first into its read list, stopping once the top-k is full.

    Args:
        selected_ptr: Picked cache rows [R, TOP_P] int32, best score first, -1 padded.
        bucket_members_ptr: Members of each cache row [rows, BUCKET_CAP] int32, -1 padded.
        positions_ptr: Each row's position [R] int32.
        out_ptr: Read lists [R, WIDTH] int32, -1 padded.
        lengths_ptr: Keys each read list holds [R] int32.
        slot_table_ptr: Each request's positions to KV slots [requests, context] int32 - read if MAP_SLOTS.
        slot_rows_ptr: Each row's request [R] int64 - read only if MAP_SLOTS.
    """
    row = tl.program_id(0).to(tl.int64)
    position = tl.load(positions_ptr + row)
    top_k, top_p = _row_budget(position, TOP_K, TOP_P, EARLY_POSITION, EARLY_TOP_K, EARLY_TOP_P)
    window_start = tl.maximum(0, position + 1 - WINDOW_SIZE)
    # The request's positions to KV slots, read as they are stored rather than in a pass of their own
    slot_row = 0
    if MAP_SLOTS:
        slot_row = tl.load(slot_rows_ptr + row).to(tl.int64)
    pad = tl.arange(0, PAD_BLOCK)
    tl.store(out_ptr + row * out_stride + pad, -1, mask=pad < WIDTH)
    tl.debug_barrier()
    # TILE picks at a time, best first, each a row of BUCKET_CAP members - the running count says where they land
    slot = tl.arange(0, TILE * BUCKET_CAP)
    filled = 0
    for first in range(0, TOP_P, TILE):
        pick = first + slot // BUCKET_CAP
        bucket = tl.load(
            selected_ptr + row * selected_stride + pick, mask=(pick < top_p) & (filled < top_k), other=-1
        )
        ids = tl.load(
            bucket_members_ptr + bucket.to(tl.int64) * BUCKET_CAP + slot % BUCKET_CAP,
            mask=bucket >= 0,
            other=-1,
        )
        valid = (bucket >= 0) & (ids >= SINK_SIZE) & (ids < window_start)
        at = filled + tl.cumsum(valid.to(tl.int32)) - 1
        stored = valid & (at < top_k)
        if MAP_SLOTS:
            ids = tl.load(slot_table_ptr + slot_row * slot_stride + ids, mask=stored, other=-1)
        tl.store(out_ptr + row * out_stride + at, ids, mask=stored)
        filled += tl.sum(valid.to(tl.int32))
    filled = tl.minimum(filled, top_k)
    length = _append_window_and_sink(
        out_ptr + row * out_stride,
        slot_table_ptr + slot_row * slot_stride,
        position,
        window_start,
        filled,
        WINDOW_SIZE,
        SINK_SIZE,
        LOCAL_BLOCK,
        MAP_SLOTS,
    )
    tl.store(lengths_ptr + row, length)


def expand_in_order(
    selected: torch.Tensor,
    bucket_members: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    slot_rows: torch.Tensor | None = None,
    early: EarlyBudget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read lists from buckets of any size, walking the picks best first until the top-k is full.

    The walk takes a few picks at a time, so a wide member row costs what its members cost and the walk stops
    loading once the budget is full. A key below the sinks or inside the window is not counted, and the last
    bucket is cut short. The window then the sinks follow.

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
        early: The budget of rows before its position - they walk its first `top_p` picks and stop at its `top_k` -
            or None to walk every pick to `top_k` throughout.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded, the expansion first.
        Keys each list holds [R] int32.
    """
    top_p = selected.size(1)
    bucket_cap = bucket_members.size(1)
    width = top_k + window_size + sink_size
    out = torch.empty(selected.size(0), width, device=selected.device, dtype=torch.int32)
    lengths = torch.empty(selected.size(0), device=selected.device, dtype=torch.int32)
    # A decode batch is a few rows, each a latency-bound walk - wide tiles take fewer steps; a prefill chunk's rows
    # fill the device, where narrow tiles keep more of them resident
    decoding = selected.size(0) <= 32
    tile = max(1, (4096 if decoding else 1024) // bucket_cap)
    mapped = slot_table is not None
    _expand_in_order[(selected.size(0),)](
        selected,
        bucket_members,
        positions,
        out,
        lengths,
        slot_table if slot_table is not None else bucket_members,
        slot_rows if slot_rows is not None else positions,
        selected.stride(0),
        width,
        slot_table.stride(0) if slot_table is not None else 0,
        TOP_P=top_p,
        BUCKET_CAP=bucket_cap,
        TILE=tile,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        TOP_K=top_k,
        WIDTH=width,
        PAD_BLOCK=triton.next_power_of_2(width),
        LOCAL_BLOCK=triton.next_power_of_2(window_size + sink_size),
        MAP_SLOTS=mapped,
        EARLY_POSITION=0 if early is None else early.position,
        EARLY_TOP_K=0 if early is None else early.top_k,
        EARLY_TOP_P=0 if early is None else early.top_p,
        num_warps=16 if decoding else 4,
    )
    return out, lengths


@triton.jit
def _expand_by_threshold(
    keys_ptr: tl.tensor,
    listed_ptr: tl.tensor,
    bucket_members_ptr: tl.tensor,
    bucket_sizes_ptr: tl.tensor,
    request_blocks_ptr: tl.tensor,
    block_positions_ptr: tl.tensor,
    request_rows_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    scratch_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    slot_table_ptr: tl.tensor,
    key_stride: tl.tensor,
    block_stride: tl.tensor,
    out_stride: tl.tensor,
    slot_stride: tl.tensor,
    page_size: tl.tensor,
    cluster_block_size: tl.tensor,
    TOP_P: tl.constexpr,
    BUCKET_CAP: tl.constexpr,
    LISTED: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    PHYSICAL: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> None:
    """Expand one row's listed buckets in listing order, up to the key its best-first walk stops at.

    Args:
        keys_ptr: Listed packed keys [R, key_stride] int64, the empty key after a row's listed count.
        listed_ptr: Keys each row listed [R] int32.
        bucket_members_ptr: Members of each cache row [rows, BUCKET_CAP] int32, -1 padded.
        bucket_sizes_ptr: Members of each cache row [rows] int32.
        request_blocks_ptr: Pages of each request [requests, pages] int32 - not read if PHYSICAL.
        block_positions_ptr: Each page's first position int32 - read only if PHYSICAL.
        request_rows_ptr: Each row's request [R] int64.
        positions_ptr: Each row's position [R] int32.
        scratch_ptr: Buckets above the stop key, listed to walk [R, TOP_P] int32.
        out_ptr: Read lists [R, WIDTH] int32, -1 padded.
        lengths_ptr: Keys each read list holds [R] int32.
        slot_table_ptr: Each request's positions to KV slots [requests, context] int32 - read if MAP_SLOTS.
        page_size: Cache rows a page holds.
        cluster_block_size: Positions a cluster block covers.
    """
    row = tl.program_id(0).to(tl.int64)
    position = tl.load(positions_ptr + row)
    top_k, top_p = _row_budget(position, TOP_K, TOP_P, EARLY_POSITION, EARLY_TOP_K, EARLY_TOP_P)
    window_start = tl.maximum(0, position + 1 - WINDOW_SIZE)
    count = tl.minimum(tl.load(listed_ptr + row), key_stride)
    request = tl.load(request_rows_ptr + row).to(tl.int64)
    # The row's listed keys in unsigned order, the empty key 0, and each one's bucket and cache row
    index = tl.arange(0, LISTED)
    key = tl.load(keys_ptr + row * key_stride + index, mask=index < count, other=-(2**63))
    order = (key ^ -(2**63)).to(tl.uint64, bitcast=True)
    bucket = key_bucket(key)
    if PHYSICAL:
        # The keys hold cache rows, and a page knows its first position
        cache_row = tl.where(order > 0, bucket, -1).to(tl.int64)
        first_position = tl.load(block_positions_ptr + cache_row // page_size, mask=cache_row >= 0, other=0)
    else:
        block = bucket // page_size
        page = tl.load(request_blocks_ptr + request * block_stride + block, mask=order > 0, other=-1)
        cache_row = tl.where((order > 0) & (page >= 0), page * page_size + bucket % page_size, -1).to(
            tl.int64
        )
        first_position = block * cluster_block_size
    live = cache_row >= 0
    # Members a bucket gives the read - all it has, unless its block reaches the sinks or the window
    members = tl.load(bucket_sizes_ptr + cache_row, mask=live, other=0)
    near = live & ((first_position < SINK_SIZE) | (first_position + cluster_block_size > window_start))
    if tl.sum(near.to(tl.int32), 0) > 0:
        counted = tl.zeros([LISTED], dtype=tl.int32)
        for s in tl.static_range(BUCKET_CAP):
            member = tl.load(bucket_members_ptr + cache_row * BUCKET_CAP + s, mask=near, other=-1)
            counted += ((member >= SINK_SIZE) & (member < window_start)).to(tl.int32)
        members = tl.where(near, counted, members)
    # The last bucket a best-first walk reaches - the largest key at or above which the buckets hold the budget
    # or number the picks - its score word first and its bucket word only on a tie, members and buckets in one sum
    base = 2 * LISTED
    weight = members * base + (order > 0).to(tl.int32)
    high = (order >> 32).to(tl.uint32)
    low = order.to(tl.uint32)
    one = tl.full([], 1, tl.uint32)
    top = tl.zeros([], dtype=tl.uint32)
    for i in range(32):
        candidate = top | (one << tl.cast(31 - i, tl.uint32))
        total = tl.sum(tl.where(high >= candidate, weight, 0), 0)
        top = tl.where((total // base >= top_k) | (total % base >= top_p), candidate, top)
    above = tl.sum(tl.where(high > top, weight, 0), 0)
    tied = high == top
    bottom = tl.zeros([], dtype=tl.uint32)
    if tl.sum(tied.to(tl.int32), 0) > 1:
        for i in range(32):
            candidate = bottom | (one << tl.cast(31 - i, tl.uint32))
            total = above + tl.sum(tl.where(tied & (low >= candidate), weight, 0), 0)
            bottom = tl.where((total // base >= top_k) | (total % base >= top_p), candidate, bottom)
    else:
        bottom = tl.max(tl.where(tied, low, 0), 0)
    last = (top.to(tl.uint64) << 32) | bottom.to(tl.uint64)
    room = top_k - tl.sum(tl.where(order > last, members, 0), 0)
    # Every bucket above the last, listed to walk, and the last cut to the room left
    whole = (order > last) & live
    tl.store(
        scratch_ptr + row * TOP_P + tl.cumsum(whole.to(tl.int32), 0) - 1, cache_row.to(tl.int32), mask=whole
    )
    whole_count = tl.sum(whole.to(tl.int32), 0)
    cut_row = tl.max(tl.where((order == last) & live, cache_row, -1), 0)
    # The request's token table row is its request number
    slot_row = request
    pad = tl.arange(0, PAD_BLOCK)
    tl.store(out_ptr + row * out_stride + pad, -1, mask=pad < WIDTH)
    tl.debug_barrier()
    slot = tl.arange(0, TILE * BUCKET_CAP)
    filled = 0
    for first in range(0, whole_count, TILE):
        pick = first + slot // BUCKET_CAP
        walked = tl.load(scratch_ptr + row * TOP_P + pick, mask=pick < whole_count, other=-1).to(tl.int64)
        ids = tl.load(
            bucket_members_ptr + walked * BUCKET_CAP + slot % BUCKET_CAP, mask=walked >= 0, other=-1
        )
        kept = (walked >= 0) & (ids >= SINK_SIZE) & (ids < window_start)
        at = filled + tl.cumsum(kept.to(tl.int32), 0) - 1
        if MAP_SLOTS:
            ids = tl.load(slot_table_ptr + slot_row * slot_stride + ids, mask=kept, other=-1)
        tl.store(out_ptr + row * out_stride + at, ids, mask=kept)
        filled += tl.sum(kept.to(tl.int32), 0)
    cut_slot = tl.arange(0, BUCKET_CAP)
    ids = tl.load(bucket_members_ptr + cut_row * BUCKET_CAP + cut_slot, mask=cut_row >= 0, other=-1)
    valid = (cut_row >= 0) & (ids >= SINK_SIZE) & (ids < window_start)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    kept = valid & (rank < room)
    if MAP_SLOTS:
        ids = tl.load(slot_table_ptr + slot_row * slot_stride + ids, mask=kept, other=-1)
    tl.store(out_ptr + row * out_stride + filled + rank, ids, mask=kept)
    filled += tl.sum(kept.to(tl.int32), 0)
    length = _append_window_and_sink(
        out_ptr + row * out_stride,
        slot_table_ptr + slot_row * slot_stride,
        position,
        window_start,
        filled,
        WINDOW_SIZE,
        SINK_SIZE,
        LOCAL_BLOCK,
        MAP_SLOTS,
    )
    tl.store(lengths_ptr + row, length)


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
    early: EarlyBudget | None = None,
) -> torch.Tensor:
    """Fill each row's read from its listed buckets by the key a best-first walk of its top-p stops at.

    The walk `expand_in_order` takes over a row's top-p stops at one key - the largest at or above which the
    buckets hold the top-k's members or number the top-p - so every member of a bucket above it is read, and
    the first members of the one at it until the top-k is full. Finding that key bit by bit from the listed keys
    stands in for their top-k, its sort and the walk, and reads the same positions in listing order rather than
    by score.

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
        early: The budget of rows before its position - the walk of each stops at the key its `top_p` and `top_k`
            find in the listing, which holds every such row's top-p - or None for `top_p` and `top_k` throughout.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded.
    """
    rows = keys.shape[0]
    width = top_k + window_size + sink_size
    out = torch.empty(rows, width, device=keys.device, dtype=torch.int32)
    lengths = torch.empty(rows, device=keys.device, dtype=torch.int32)
    scratch = torch.empty(rows, top_p, device=keys.device, dtype=torch.int32)
    _expand_by_threshold[(rows,)](
        keys,
        listed,
        bucket_members,
        bucket_sizes,
        request_blocks,
        request_blocks if block_positions is None else block_positions,
        request_rows,
        positions,
        scratch,
        out,
        lengths,
        bucket_members if slot_table is None else slot_table,
        keys.stride(0),
        request_blocks.stride(0),
        width,
        0 if slot_table is None else slot_table.stride(0),
        page_size,
        cluster_block_size,
        TOP_P=top_p,
        BUCKET_CAP=bucket_members.shape[1],
        LISTED=triton.next_power_of_2(keys.shape[1]),
        TILE=128,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        TOP_K=top_k,
        WIDTH=width,
        PAD_BLOCK=triton.next_power_of_2(width),
        LOCAL_BLOCK=triton.next_power_of_2(window_size + sink_size),
        MAP_SLOTS=slot_table is not None,
        PHYSICAL=block_positions is not None,
        EARLY_POSITION=0 if early is None else early.position,
        EARLY_TOP_K=0 if early is None else early.top_k,
        EARLY_TOP_P=0 if early is None else early.top_p,
        num_warps=4,
    )
    return out
