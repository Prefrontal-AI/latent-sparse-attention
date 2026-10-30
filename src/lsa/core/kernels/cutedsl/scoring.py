"""Each row's listed buckets scored and packed with their bucket into keys, in one CuTeDSL kernel.

The scan lists a row's survivors as buckets; this scores each one against the row's index query and packs score and
bucket into the int64 key the select orders by, the empty key after the row's listed count. A CTA takes a row, and
its sixteen groups of eight lanes score every sixteenth listed bucket, several in flight.

The dense form scores every bucket a row sees instead, each at its own column, for a row the listing may have missed
a best bucket of: a CTA takes a row and a span of 8,192 buckets, as the Triton dense scoring spans them.

A score is added in the order the Triton scoring adds a 64-dim dot - each lane its eight products in turn, then the
lanes by halves - so the keys are the Triton keys bit for bit, and so is the count of listed buckets each row scored
at or above its threshold.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

# Index dims the kernel takes, and the lanes and dims of each that add one score, as Triton lays a 64-dim dot out
INDEX_DIM = 64
GROUP_LANES = 8
DIMS_PER_LANE = INDEX_DIM // GROUP_LANES
THREADS = 128
GROUPS = THREADS // GROUP_LANES
# Listed buckets a group of lanes has in flight
IN_FLIGHT = 8
# The least int64, the key past a row's listed count
EMPTY = -(2**63)
# Buckets a CTA of the dense scoring takes, as the Triton dense scoring spans them
DENSE_SPAN = 8192


def pack_key(score: Any, bucket: Any) -> Any:
    """Return a score and its bucket as one int64 key - the score's order-preserving bits high, the bucket reversed low.

    Args:
        score: The fp32 score.
        bucket: The int32 bucket.
    """
    cell = cute.make_rmem_tensor(1, cutlass.Float32)
    cell[0] = score
    bits = cute.recast_tensor(cell, cutlass.Int32)[0]
    # A negative score's magnitude bits flip, so the int32 order is the float order
    ordered = bits ^ ((bits >> 31) & cutlass.Int32(0x7FFFFFFF))
    return (cutlass.Int64(ordered) << 32) | cutlass.Int64(cutlass.Int32(0x7FFFFFFF) - bucket)


class CandidateScoring:
    """The CuTeDSL scoring of listed buckets."""

    def __init__(self, dense: bool) -> None:
        """Fix the form and the CTA's barrier.

        Args:
            dense: Score every bucket a row sees, at its own column, rather than the listed ones.
        """
        self.dense = dense
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        thresholds: cute.Tensor,
        keys: cute.Tensor,
        reached: cute.Tensor,
        stream: Any,
    ) -> None:
        """Launch a CTA a row.

        Args:
            index_queries: Index queries [R, index_dim] bf16.
            bucket_centroids: Bucket centroids [N, index_dim] bf16.
            buckets: The listing [R, capacity] int32.
            survivor_counts: Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.
            thresholds: Score each row's top-p is expected at or above [R] fp32.
            keys: Packed keys [R, capacity] int64.
            reached: Listed buckets each row scored at or above its threshold [R] int32.
            stream: The stream to launch on.
        """

        spans = 1
        if cutlass.const_expr(self.dense):
            spans = cute.ceil_div(keys.shape[1], DENSE_SPAN)

        @cute.struct
        class Storage:
            """Shared memory of a block: the count of buckets reached."""

            reached_count: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 4]

        self.kernel(
            index_queries, bucket_centroids, buckets, survivor_counts, thresholds, keys, reached, Storage
        ).launch(grid=(index_queries.shape[0], spans, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        thresholds: cute.Tensor,
        keys: cute.Tensor,
        reached: cute.Tensor,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Score a row's listed buckets into keys, fill the rest of its keys empty and count those reached.

        Args:
            index_queries: Index queries [R, index_dim] bf16.
            bucket_centroids: Bucket centroids [N, index_dim] bf16.
            buckets: The listing [R, capacity] int32.
            survivor_counts: Buckets each row listed [R] int32.
            thresholds: Score each row's top-p is expected at or above [R] fp32.
            keys: Packed keys [R, capacity] int64.
            reached: Listed buckets each row scored at or above its threshold [R] int32.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, span, _ = cute.arch.block_idx()
        group = thread // GROUP_LANES
        member = thread % GROUP_LANES
        # A dense CTA scores its span of the buckets the row sees, each at its own column
        begin = cutlass.Int32(0)
        end = keys.shape[1]
        if cutlass.const_expr(self.dense):
            begin = span * DENSE_SPAN
            end = cutlass.min(begin + DENSE_SPAN, keys.shape[1])
        count = cutlass.max(cutlass.min(survivor_counts[row], end), begin)
        threshold = thresholds[row]

        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        reached_count = storage.reached_count.get_tensor(cute.make_layout(1))
        if thread == 0:
            reached_count[0] = cutlass.Int32(0)
        for slot in range(count + thread, end, THREADS):
            keys[row, slot] = cutlass.Int64(EMPTY)

        query_bf16 = cute.make_rmem_tensor(DIMS_PER_LANE, index_queries.element_type)
        cute.autovec_copy(
            cute.local_tile(index_queries, (1, DIMS_PER_LANE), (row, member))[0, None], query_bf16
        )
        query = cute.make_rmem_tensor(DIMS_PER_LANE, cutlass.Float32)
        for e in cutlass.range_constexpr(DIMS_PER_LANE):
            query[e] = query_bf16[e].to(cutlass.Float32)
        listed = cute.make_rmem_tensor(IN_FLIGHT, cutlass.Int32)
        loaded = cute.make_rmem_tensor(
            cute.make_layout((IN_FLIGHT, DIMS_PER_LANE), stride=(DIMS_PER_LANE, 1)),
            bucket_centroids.element_type,
        )
        high = cutlass.Int32(0)
        self.barrier.arrive_and_wait()

        for first in range(begin, count, GROUPS * IN_FLIGHT):
            for f in cutlass.range_constexpr(IN_FLIGHT):
                slot = first + f * GROUPS + group
                listed[f] = cutlass.Int32(-1)
                if slot < count:
                    if cutlass.const_expr(self.dense):
                        listed[f] = slot
                    else:
                        listed[f] = buckets[row, slot]
            for f in cutlass.range_constexpr(IN_FLIGHT):
                cute.autovec_copy(
                    cute.local_tile(
                        bucket_centroids, (1, DIMS_PER_LANE), (cutlass.max(listed[f], 0), member)
                    )[0, None],
                    loaded[f, None],
                )
            for f in cutlass.range_constexpr(IN_FLIGHT):
                # Triton's lane adds its eight products in turn, then the lanes add by halves
                score = query[0] * loaded[f, 0].to(cutlass.Float32) + query[1] * loaded[f, 1].to(
                    cutlass.Float32
                )
                for e in cutlass.range_constexpr(2, DIMS_PER_LANE):
                    score = score + query[e] * loaded[f, e].to(cutlass.Float32)
                for offset in cutlass.range_constexpr(3):
                    score = score + cute.arch.shuffle_sync_bfly(score, GROUP_LANES // 2 >> offset)
                if (member == 0) & (listed[f] >= 0):
                    keys[row, first + f * GROUPS + group] = pack_key(score, listed[f])
                    high = high + cutlass.Int32(score >= threshold)

        if high > 0:
            cute.arch.atomic_add((reached_count.iterator).llvm_ptr, high, scope="cta")
        self.barrier.arrive_and_wait()
        if thread == 0:
            reached[row] = reached_count[0]


COMPILED: dict[str, Any] = {}


def compiled(dense: bool) -> Any:
    """Return the scoring compiled once a form, taking torch tensors as they are.

    Args:
        dense: Score every bucket a row sees rather than the listed ones.
    """
    key = "dense" if dense else "listed"
    if key in COMPILED:
        return COMPILED[key]
    rows, bucket_count, capacity, listing_rows, listing_columns = (cute.sym_int32() for _ in range(5))
    bucket_stride, key_stride = cute.sym_int64(), cute.sym_int64()
    arguments = (
        make_fake_compact_tensor(cutlass.BFloat16, (rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(
            cutlass.BFloat16, (bucket_count, INDEX_DIM), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32, (listing_rows, listing_columns), stride=(bucket_stride, 1), assumed_align=4
        ),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int64, (rows, capacity), stride=(key_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED[key] = cute.compile(CandidateScoring(dense), *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


def supports(index_queries: torch.Tensor, bucket_centroids: torch.Tensor) -> bool:
    """Tell whether the CuTeDSL scoring takes these rows - bf16, 64 index dims, rows contiguous and 16-byte aligned.

    Args:
        index_queries: Index queries [R, index_dim].
        bucket_centroids: Bucket centroids [N, index_dim].
    """
    return (
        index_queries.dtype == bucket_centroids.dtype == torch.bfloat16
        and index_queries.shape[1] == bucket_centroids.shape[1] == INDEX_DIM
        and index_queries.is_contiguous()
        and bucket_centroids.is_contiguous()
        and index_queries.data_ptr() % 16 == 0
        and bucket_centroids.data_ptr() % 16 == 0
    )


def score_candidates(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    buckets: torch.Tensor,
    survivor_counts: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each row's listed buckets and pack each with its bucket, as the Triton scoring does bit for bit.

    Args:
        index_queries: Index queries [R, index_dim] bf16.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        buckets: The listing [R, capacity] int32, a row's first `capacity` survivors.
        survivor_counts: Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.
        thresholds: Score each row's top-p is expected at or above [R] fp32.

    Returns:
        Packed keys [R, capacity] int64 of the listed buckets, the empty key after.
        Listed buckets each row scored at or above its threshold [R] int32.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    if (
        not supports(index_queries, bucket_centroids)
        or buckets.dtype != torch.int32
        or buckets.stride(1) != 1
    ):
        raise ValueError(
            "the CuTeDSL scoring takes contiguous, 16-byte aligned bf16 rows of 64 index dims and an int32 listing "
            f"with contiguous rows - got {index_queries.dtype} {tuple(index_queries.shape)}, {bucket_centroids.dtype} "
            f"{tuple(bucket_centroids.shape)} and {buckets.dtype} strides {buckets.stride()}"
        )
    rows, capacity = buckets.shape
    keys = torch.empty(rows, capacity, device=buckets.device, dtype=torch.int64)
    reached = torch.zeros(rows, device=buckets.device, dtype=torch.int32)
    if rows == 0 or capacity == 0:
        return keys, reached
    compiled(False)(
        index_queries,
        bucket_centroids,
        buckets,
        survivor_counts.int().contiguous(),
        thresholds.float().contiguous(),
        keys,
        reached,
    )
    return keys, reached


STAND_INS: dict[torch.device, torch.Tensor] = {}


def score_dense(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    width: int | None = None,
) -> torch.Tensor:
    """Score every bucket each row sees and pack it with its bucket at its own column, as the Triton scoring does.

    Args:
        index_queries: Index queries [R, index_dim] bf16.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        width: Columns of the result, at least N - N when None.

    Returns:
        Packed keys [R, width] int64, the empty key past a row's visible count.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    if not supports(index_queries, bucket_centroids):
        raise ValueError(
            "the CuTeDSL scoring takes contiguous, 16-byte aligned bf16 rows of 64 index dims - got "
            f"{index_queries.dtype} {tuple(index_queries.shape)} and {bucket_centroids.dtype} "
            f"{tuple(bucket_centroids.shape)}"
        )
    rows = index_queries.shape[0]
    device = index_queries.device
    keys = torch.empty(rows, width or bucket_centroids.shape[0], device=device, dtype=torch.int64)
    if rows == 0 or keys.shape[1] == 0:
        return keys
    if device not in STAND_INS:
        STAND_INS[device] = torch.zeros(1, 1, device=device, dtype=torch.int32)
    compiled(True)(
        index_queries,
        bucket_centroids,
        STAND_INS[device],
        visible_counts.int().contiguous(),
        torch.full((rows,), torch.inf, device=device),
        keys,
        torch.empty(rows, device=device, dtype=torch.int32),
    )
    return keys
