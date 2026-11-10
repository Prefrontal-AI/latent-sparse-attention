"""The buckets each index query scores at or above its threshold, listed in one CuTeDSL kernel on Blackwell.

The scan's listing of survivors, as [the Hopper bucket scan](../hopper/bucket_scan.py) lists them, on Blackwell's
fifth-generation tensor cores. A CTA holds 128 index queries and streams a span of bucket centroids past them: one
warp copies tiles of centroids into shared memory with TMA, one warp scores each tile into tensor memory with
`tcgen05.mma`, and four warps test the scores while the next tile is scored into the other accumulator.

Tensor memory has a lane a row, and a thread of the four testing warps reads its lane - so a thread holds one row's
scores, its threshold and its visible count, and a row's list in shared memory is the thread's alone. A tile of 32 of
a row's scores is tested by its maximum, and only a tile whose maximum reaches the threshold is walked score by score.
Survivors are rare - some 1,700 of 1.27M buckets a row at 10M - so almost every tile is a few three-way maxima.

The scores are the tensor-core bf16 dot with an fp32 accumulator, as the Triton scan's are - tested equal to Triton's.
The lowered threshold (MARGIN) and the rescoring make the select exact regardless. The order a row lists them in is
not defined and not read.

Scores and thresholds are finite, bar a threshold of -inf or inf. The fast path tests the sign bit of a score less its
threshold, which counts a NaN as a hit where `>=` does not. Scores are internal, so nothing checks.

A span is capped at the last tile rather than padded to a whole pass as Hopper's is, because the pipelines keep their
stage and phase as state at run time, so no tile past the buckets is ever scored.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.blackwell_helpers as blackwell
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from lsa.core.kernels import dispatch
from lsa.core.kernels.constants import Arch, BlackwellBucketScan, Index, SharedBucketScan
from lsa.core.kernels.cutedsl.hopper.bucket_scan import push_sign


@dsl_user_op
def max3(a: Any, b: Any, c: Any, *, loc: Any = None, ip: Any = None) -> Any:
    """Return the largest of three fp32 values in one instruction, as Blackwell has it.

    Args:
        a: An fp32 value.
        b: An fp32 value.
        c: An fp32 value.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip) for value in (a, b, c)],
            "max.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


class BucketScan:
    """The CuTeDSL bucket scan on Blackwell."""

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        visible_counts: cute.Tensor,
        thresholds: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        span_tiles: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Lay out shared and tensor memory, the copies and the MMA, and launch a CTA per 128 rows and span of buckets.

        Args:
            index_queries: Index queries [R, index_dim] bf16.
            bucket_centroids: Bucket centroids [N, index_dim] bf16.
            visible_counts: Buckets each row sees [R] int32, at most N.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32, zero on entry.
            buckets: The listing [R, capacity] int32.
            span_tiles: Tiles a CTA scans.
            stream: The stream to launch on.
        """
        dtype = index_queries.element_type
        mma_tiler = (BlackwellBucketScan.BLOCK_Q, BlackwellBucketScan.BLOCK_N, Index.DIM)
        tiled_mma = blackwell.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            mma_tiler[:2],
        )
        query_layout = blackwell.make_smem_layout_a(tiled_mma, mma_tiler, dtype, 1)
        centroid_layout = blackwell.make_smem_layout_b(
            tiled_mma, mma_tiler, dtype, BlackwellBucketScan.STAGES
        )
        cluster = cute.make_layout((1, 1, 1, 1))
        self.query_bytes = cute.size_in_bytes(dtype, cute.slice_(query_layout, (None, None, None, 0)))
        self.tile_bytes = cute.size_in_bytes(dtype, cute.slice_(centroid_layout, (None, None, None, 0)))
        query_copy, query_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            index_queries,
            cute.slice_(query_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster.shape,
        )
        centroid_copy, centroid_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            bucket_centroids,
            cute.slice_(centroid_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster.shape,
        )

        @cute.struct
        class Storage:
            """Shared memory of a block: barriers, tensor memory address, row buckets, operand tiles."""

            query_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            centroid_barriers: cute.struct.MemRange[cutlass.Int64, 2 * BlackwellBucketScan.STAGES]
            score_barriers: cute.struct.MemRange[cutlass.Int64, 2 * BlackwellBucketScan.ACCUMULATORS]
            tmem_address: cutlass.Int32
            row_buckets: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    BlackwellBucketScan.TEST_GROUPS
                    * BlackwellBucketScan.BLOCK_Q
                    * SharedBucketScan.ROW_SLOTS,
                ],
                16,
            ]
            queries: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(query_layout.outer)], 1024]
            centroids: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(centroid_layout.outer)], 1024
            ]

        grid = (
            cute.ceil_div(index_queries.shape[0], BlackwellBucketScan.BLOCK_Q),
            cute.ceil_div(cute.ceil_div(bucket_centroids.shape[0], BlackwellBucketScan.BLOCK_N), span_tiles),
            1,
        )
        self.kernel(
            tiled_mma,
            query_copy,
            query_tensor,
            centroid_copy,
            centroid_tensor,
            visible_counts,
            thresholds,
            survivor_counts,
            buckets,
            span_tiles,
            bucket_centroids.shape[0],
            query_layout,
            centroid_layout,
            cluster,
            Storage,
        ).launch(
            grid=grid,
            block=[BlackwellBucketScan.THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=BlackwellBucketScan.MIN_BLOCKS,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        query_copy: cute.CopyAtom,
        index_queries: cute.Tensor,
        centroid_copy: cute.CopyAtom,
        bucket_centroids: cute.Tensor,
        visible_counts: cute.Tensor,
        thresholds: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        span_tiles: cutlass.Int32,
        bucket_count: cutlass.Int32,
        query_layout: cute.ComposedLayout,
        centroid_layout: cute.ComposedLayout,
        cluster: cute.Layout,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Score a span of buckets for 128 rows, list each row's survivors in shared memory, then flush them.

        Args:
            tiled_mma: The MMA over a tile.
            query_copy: The TMA copy of the CTA's index queries.
            index_queries: Index queries [R, index_dim] bf16, as the TMA reads them.
            centroid_copy: The TMA copy of a tile of bucket centroids.
            bucket_centroids: Bucket centroids [N, index_dim] bf16, as the TMA reads them.
            visible_counts: Buckets each row sees [R] int32.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
            span_tiles: Tiles a CTA scans.
            bucket_count: Buckets there are.
            query_layout: Shared memory of the CTA's index queries, swizzled as the MMA reads it.
            centroid_layout: Shared memory of the tiles of bucket centroids in flight.
            cluster: The one-CTA cluster.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        query_block, span, _ = cute.arch.block_idx()
        rows = visible_counts.shape[0]
        first_tile = span * span_tiles
        stop_tile = cutlass.min(
            first_tile + span_tiles, cute.ceil_div(bucket_count, BlackwellBucketScan.BLOCK_N)
        )

        if warp == BlackwellBucketScan.COPY_WARP:
            cpasync.prefetch_descriptor(query_copy)
            cpasync.prefetch_descriptor(centroid_copy)

        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        shared_queries = storage.queries.get_tensor(query_layout.outer, swizzle=query_layout.inner)
        shared_centroids = storage.centroids.get_tensor(centroid_layout.outer, swizzle=centroid_layout.inner)
        row_buckets = storage.row_buckets.get_tensor(
            # A slot's rows side by side, so a warp's threads append to and flush from distinct banks
            cute.make_layout(
                (BlackwellBucketScan.BLOCK_Q, SharedBucketScan.ROW_SLOTS, BlackwellBucketScan.TEST_GROUPS),
                stride=(
                    1,
                    BlackwellBucketScan.BLOCK_Q,
                    BlackwellBucketScan.BLOCK_Q * SharedBucketScan.ROW_SLOTS,
                ),
            )
        )

        single = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        query_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.query_barriers.data_ptr(),
            num_stages=1,
            producer_group=single,
            consumer_group=single,
            tx_count=self.query_bytes,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        centroid_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.centroid_barriers.data_ptr(),
            num_stages=BlackwellBucketScan.STAGES,
            producer_group=single,
            consumer_group=single,
            tx_count=self.tile_bytes,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        score_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.score_barriers.data_ptr(),
            num_stages=BlackwellBucketScan.ACCUMULATORS,
            producer_group=single,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, BlackwellBucketScan.BLOCK_Q),
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_address.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=BlackwellBucketScan.TMEM_BARRIER, num_threads=BlackwellBucketScan.TESTERS + 32
            ),
            allocator_warp_id=0,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster, is_relaxed=True)

        # The MMA's views of the operands and of the accumulators in tensor memory
        mma_slice = tiled_mma.get_slice(0)
        tCrA = tiled_mma.make_fragment_A(shared_queries)
        tCrB = tiled_mma.make_fragment_B(shared_centroids)
        accumulator_shape = tiled_mma.partition_shape_C(
            (BlackwellBucketScan.BLOCK_Q, BlackwellBucketScan.BLOCK_N)
        )
        fake_accumulators = tiled_mma.make_fragment_C(
            cute.append(accumulator_shape, BlackwellBucketScan.ACCUMULATORS)
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster)

        if warp == BlackwellBucketScan.COPY_WARP:
            self.produce(
                query_pipeline,
                centroid_pipeline,
                query_copy,
                centroid_copy,
                index_queries,
                bucket_centroids,
                shared_queries,
                shared_centroids,
                mma_slice,
                query_block,
                first_tile,
                stop_tile,
            )

        if warp == BlackwellBucketScan.MMA_WARP:
            tmem.wait_for_alloc()
            accumulators = cute.make_tensor(tmem.retrieve_ptr(cutlass.Float32), fake_accumulators.layout)
            self.score(
                query_pipeline,
                centroid_pipeline,
                score_pipeline,
                tiled_mma,
                accumulators,
                tCrA,
                tCrB,
                first_tile,
                stop_tile,
            )

        if warp < BlackwellBucketScan.TEST_WARPS:
            tmem.allocate(BlackwellBucketScan.ACCUMULATORS * BlackwellBucketScan.BLOCK_N)
            tmem.wait_for_alloc()
            tmem_pointer = tmem.retrieve_ptr(cutlass.Float32)
            accumulators = cute.make_tensor(tmem_pointer, fake_accumulators.layout)
            self.consume(
                score_pipeline,
                thread % BlackwellBucketScan.BLOCK_Q,
                warp // 4,
                query_block,
                rows,
                first_tile,
                stop_tile,
                accumulators,
                visible_counts,
                thresholds,
                survivor_counts,
                buckets,
                row_buckets[None, None, warp // 4],
            )
            # Another CTA on the SM may take the columns once they are freed
            # => every tester is past its last read first
            pipeline.NamedBarrier(
                barrier_id=BlackwellBucketScan.FREE_BARRIER, num_threads=BlackwellBucketScan.TESTERS
            ).arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_pointer)

    @cute.jit
    def produce(
        self,
        query_pipeline: Any,
        centroid_pipeline: Any,
        query_copy: cute.CopyAtom,
        centroid_copy: cute.CopyAtom,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        shared_queries: cute.Tensor,
        shared_centroids: cute.Tensor,
        mma_slice: Any,
        query_block: cutlass.Int32,
        first_tile: cutlass.Int32,
        stop_tile: cutlass.Int32,
    ) -> None:
        """Copy the CTA's queries in once, then each tile of bucket centroids into the stage the MMA has freed.

        Args:
            query_pipeline: The pipeline of the queries' stage.
            centroid_pipeline: The pipeline of full and empty stages.
            query_copy: The TMA copy of the CTA's index queries.
            centroid_copy: The TMA copy of a tile of bucket centroids.
            index_queries: Index queries [R, index_dim] bf16, as the TMA reads them.
            bucket_centroids: Bucket centroids [N, index_dim] bf16, as the TMA reads them.
            shared_queries: The CTA's index queries in shared memory.
            shared_centroids: The tile stages in shared memory.
            mma_slice: The MMA's slice, which partitions the operands.
            query_block: The CTA's block of rows.
            first_tile: The first tile of the span.
            stop_tile: One past the last.
        """
        query_tiles = cute.local_tile(index_queries, (BlackwellBucketScan.BLOCK_Q, Index.DIM), (None, None))
        centroid_tiles = cute.local_tile(
            bucket_centroids, (BlackwellBucketScan.BLOCK_N, Index.DIM), (None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            query_copy,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_queries, 0, 3),
            cute.group_modes(mma_slice.partition_A(query_tiles), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            centroid_copy,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_centroids, 0, 3),
            cute.group_modes(mma_slice.partition_B(centroid_tiles), 0, 3),
        )
        query_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        query_pipeline.producer_acquire(query_state)
        cute.copy(
            query_copy,
            tAgA[(None, query_block, 0)],
            tAsA[(None, 0)],
            tma_bar_ptr=query_pipeline.producer_get_barrier(query_state),
        )
        centroid_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, BlackwellBucketScan.STAGES
        )
        for tile in cutlass.range(first_tile, stop_tile, 1, unroll=1):
            centroid_pipeline.producer_acquire(centroid_state)
            cute.copy(
                centroid_copy,
                tBgB[(None, tile, 0)],
                tBsB[(None, centroid_state.index)],
                tma_bar_ptr=centroid_pipeline.producer_get_barrier(centroid_state),
            )
            centroid_state.advance()
        centroid_pipeline.producer_tail(centroid_state)

    @cute.jit
    def score(
        self,
        query_pipeline: Any,
        centroid_pipeline: Any,
        score_pipeline: Any,
        tiled_mma: cute.TiledMma,
        accumulators: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        first_tile: cutlass.Int32,
        stop_tile: cutlass.Int32,
    ) -> None:
        """Score each tile of the span for the CTA's 128 rows into the next accumulator the testers have freed.

        Args:
            query_pipeline: The pipeline of the queries' stage.
            centroid_pipeline: The pipeline of full and empty stages.
            score_pipeline: The pipeline from the MMA to the testers.
            tiled_mma: The MMA over a tile.
            accumulators: The accumulators in tensor memory.
            tCrA: The MMA's view of the queries in shared memory.
            tCrB: The MMA's view of the tiles in shared memory.
            first_tile: The first tile of the span.
            stop_tile: One past the last.
        """
        query_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        centroid_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, BlackwellBucketScan.STAGES
        )
        score_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, BlackwellBucketScan.ACCUMULATORS
        )
        query_pipeline.consumer_wait(query_state)
        for _ in cutlass.range(first_tile, stop_tile, 1, unroll=1):
            score_pipeline.producer_acquire(score_state)
            centroid_pipeline.consumer_wait(centroid_state)
            self.score_tile(
                tiled_mma,
                accumulators[(None, None, None, score_state.index)],
                tCrA,
                tCrB,
                centroid_state.index,
            )
            centroid_pipeline.consumer_release(centroid_state)
            score_pipeline.producer_commit(score_state)
            centroid_state.advance()
            score_state.advance()
        score_pipeline.producer_tail(score_state)

    @cute.jit
    def score_tile(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        stage: cutlass.Int32,
    ) -> None:
        """Start the MMA of the CTA's 128 rows against a tile into an accumulator, to be waited on by the testers.

        Args:
            tiled_mma: The MMA over a tile.
            accumulator: The accumulator it overwrites.
            tCrA: The MMA's view of the queries in shared memory.
            tCrB: The MMA's view of the tiles in shared memory.
            stage: The tile's stage.
        """
        # The first product overwrites the accumulator
        # => no zeroing pass
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(
                tiled_mma, accumulator, tCrA[(None, None, k, 0)], tCrB[(None, None, k, stage)], accumulator
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def consume(
        self,
        score_pipeline: Any,
        thread: cutlass.Int32,
        group: cutlass.Int32,
        query_block: cutlass.Int32,
        rows: cutlass.Int32,
        first_tile: cutlass.Int32,
        stop_tile: cutlass.Int32,
        accumulators: cute.Tensor,
        visible_counts: cute.Tensor,
        thresholds: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        row_buckets: cute.Tensor,
    ) -> None:
        """Read each tile's scores of the thread's row from tensor memory, list its survivors and flush the list.

        Args:
            score_pipeline: The pipeline from the MMA to the testers.
            thread: The thread's lane, and the row of the CTA it tests.
            group: The thread's group of testers, which tests every TEST_GROUPS-th tile.
            query_block: The CTA's block of rows.
            rows: Rows there are.
            first_tile: The span's first tile.
            stop_tile: The tile after the span's last.
            accumulators: The accumulators in tensor memory.
            visible_counts: Buckets each row sees [R] int32.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
            row_buckets: The group's lists of its rows in shared memory.
        """
        row = query_block * BlackwellBucketScan.BLOCK_Q + thread
        threshold = cutlass.Float32.inf
        visible_count = cutlass.Int32(0)
        if row < rows:
            threshold = thresholds[row]
            visible_count = visible_counts[row]

        load = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(getattr(tcgen05.Repetition, f"x{BlackwellBucketScan.CHUNK}")), cutlass.Float32
        )
        chunks = cute.flat_divide(
            accumulators[((None, None), 0, 0, None)], (BlackwellBucketScan.BLOCK_Q, BlackwellBucketScan.CHUNK)
        )
        tiled_load = tcgen05.make_tmem_copy(load, chunks[(None, None, 0, 0, 0)])
        load_slice = tiled_load.get_slice(thread)
        tTR_tAcc = load_slice.partition_S(chunks)
        # A thread's chunk is CHUNK consecutive buckets of its row, the first at bit 0
        coordinates = cute.flat_divide(
            cute.make_identity_tensor((BlackwellBucketScan.BLOCK_Q, BlackwellBucketScan.BLOCK_N)),
            (BlackwellBucketScan.BLOCK_Q, BlackwellBucketScan.CHUNK),
        )
        chunk_count = BlackwellBucketScan.BLOCK_N // BlackwellBucketScan.CHUNK
        chunk_shape = load_slice.partition_D(coordinates)[(None, None, None, 0, 0)].shape
        tile_scores = [cute.make_rmem_tensor(chunk_shape, cutlass.Float32) for _ in range(chunk_count)]

        held = cutlass.Int32(0)
        # Group g tests tiles g, g + 2, ... and the MMA scores tile t into accumulator t % ACCUMULATORS
        # => group g reads accumulators g and g + 2, and its state starts at g and advances by TEST_GROUPS
        score_state = pipeline.PipelineState(
            BlackwellBucketScan.ACCUMULATORS, cutlass.Int32(0), group, cutlass.Int32(0)
        )
        for tile in cutlass.range(first_tile + group, stop_tile, BlackwellBucketScan.TEST_GROUPS, unroll=1):
            score_pipeline.consumer_wait(score_state)
            for chunk in cutlass.range_constexpr(chunk_count):
                cute.copy(
                    tiled_load, tTR_tAcc[(None, None, None, 0, chunk, score_state.index)], tile_scores[chunk]
                )
            # A load from tensor memory lands in its registers only at the wait
            cute.arch.fence_view_async_tmem_load()
            score_pipeline.consumer_release(score_state)
            for _ in cutlass.range_constexpr(BlackwellBucketScan.TEST_GROUPS):
                score_state.advance()

            for chunk in cutlass.range_constexpr(chunk_count):
                held = self.test_scores(
                    tile_scores[chunk],
                    tile * BlackwellBucketScan.BLOCK_N + chunk * BlackwellBucketScan.CHUNK,
                    held,
                    thread,
                    row,
                    threshold,
                    visible_count,
                    row_buckets,
                    survivor_counts,
                    buckets,
                )

        if held > 0:
            self.flush(thread, row, held, row_buckets, survivor_counts, buckets)

    @cute.jit
    def test_scores(
        self,
        scores: cute.Tensor,
        start: cutlass.Int32,
        held: cutlass.Int32,
        thread: cutlass.Int32,
        row: cutlass.Int32,
        threshold: cutlass.Float32,
        visible_count: cutlass.Int32,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
    ) -> cutlass.Int32:
        """Append the buckets of a row's chunk of scores that reach its threshold, if its maximum does.

        Args:
            scores: The row's CHUNK scores against consecutive buckets, in registers.
            start: The chunk's first bucket.
            held: Buckets the row's list holds.
            thread: The thread, and the row of the CTA it tests.
            row: The row.
            threshold: Score the row lists at or above.
            visible_count: Buckets the row sees.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.

        Returns:
            Buckets the row's list holds.
        """
        if self.chunk_max(scores, cute.size(scores)) >= threshold:
            # A score at or above its threshold leaves a difference of sign 0, and the last pushed is bit 0
            # A NaN difference has sign 0, the sign of a hit
            # => scores and thresholds must be finite
            misses = cutlass.Uint32(0)
            for i in cutlass.range_constexpr(cute.size(scores) - 2, -1, -2):
                low, high = cute.arch.sub_packed_f32x2((scores[i], scores[i + 1]), (threshold, threshold))
                misses = push_sign(push_sign(misses, high), low)
            hits = ~misses
            # Past a row's visible buckets a score cannot list
            if start + BlackwellBucketScan.CHUNK > visible_count:
                visible = cutlass.Uint32(cutlass.max(visible_count - start, 0))
                hits = hits & ((cutlass.Uint32(1) << visible) - cutlass.Uint32(1))
            if hits != 0:
                held = self.append(hits, start, held, thread, row, row_buckets, survivor_counts, buckets)
        return held

    @cute.jit
    def append(
        self,
        hits: cutlass.Uint32,
        start: cutlass.Int32,
        held: cutlass.Int32,
        thread: cutlass.Int32,
        row: cutlass.Int32,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
    ) -> cutlass.Int32:
        """Append the bucket of each set bit to the row's list in shared memory, flushing the list first if it is full.

        Args:
            hits: A bit a score of the chunk, set for a survivor.
            start: The chunk's first bucket.
            held: Buckets the row's list holds.
            thread: The thread, and the row of the CTA it tests.
            row: The row.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.

        Returns:
            Buckets the row's list holds.
        """
        # A list too full for the chunk's survivors moves to the global listing whole, with one atomic
        if held + cutlass.Int32(cute.arch.popc(hits)) > SharedBucketScan.ROW_SLOTS:
            self.flush(thread, row, held, row_buckets, survivor_counts, buckets)
            held = cutlass.Int32(0)
        while hits != 0:
            bit = cutlass.Uint32(31) - cute.arch.clz(hits)
            hits = hits ^ (cutlass.Uint32(1) << bit)
            row_buckets[thread, held] = start + cutlass.Int32(bit)
            held = held + 1
        return held

    @cute.jit
    def flush(
        self,
        thread: cutlass.Int32,
        row: cutlass.Int32,
        held: cutlass.Int32,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
    ) -> None:
        """Reserve a row's place in the global listing with one atomic and copy its list in shared memory there.

        A row's list is its thread's alone, so the flush needs no barrier.

        Args:
            thread: The thread, and the row of the CTA it tests.
            row: The row.
            held: Buckets the row's list holds.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
        """
        base = cute.arch.atomic_add(
            (survivor_counts.iterator + row).llvm_ptr, held, sem="relaxed", scope="gpu"
        )
        count = cutlass.min(held, buckets.shape[1] - base)
        listed = cute.make_tensor(
            buckets[row, None].iterator + base, cute.make_layout(SharedBucketScan.ROW_SLOTS)
        )
        for i in cutlass.range_constexpr(SharedBucketScan.ROW_SLOTS):
            if i < count:
                listed[i] = row_buckets[thread, i]

    @cute.jit
    def chunk_max(self, scores: cute.Tensor, size: cutlass.Constexpr) -> cutlass.Float32:
        """Return the largest of a chunk's scores, three at a time.

        Args:
            scores: The chunk's scores in registers.
            size: Scores in the chunk - after the first, three at a time, and an even size leaves one for a two-way maximum.
        """
        best = scores[0]
        for i in cutlass.range_constexpr(1, size - 1, 2):
            best = max3(best, scores[i], scores[i + 1])
        if cutlass.const_expr(size % 2 == 0):
            best = cute.arch.fmax(best, scores[size - 1])
        return best


COMPILED: dict[str, Any] = {}


def compiled() -> Any:
    """Return the listing compiled once, taking torch tensors as they are."""
    if "bucket scan" in COMPILED:
        return COMPILED["bucket scan"]
    rows, bucket_count, capacity = (cute.sym_int32() for _ in range(3))
    bucket_stride = cute.sym_int64()
    arguments = (
        make_fake_compact_tensor(cutlass.BFloat16, (rows, Index.DIM), stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(
            cutlass.BFloat16, (bucket_count, Index.DIM), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (rows, capacity), stride=(bucket_stride, 1), assumed_align=4),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED["bucket scan"] = cute.compile(BucketScan(), *arguments, options="--enable-tvm-ffi")
    return COMPILED["bucket scan"]


def supports(index_queries: torch.Tensor, bucket_centroids: torch.Tensor) -> bool:
    """Tell whether the CuTeDSL listing takes these rows - on Blackwell, bf16, 64 index dims, contiguous, 16-byte aligned.

    Args:
        index_queries: Index queries [R, index_dim].
        bucket_centroids: Bucket centroids [N, index_dim].
    """
    return (
        index_queries.is_cuda
        and dispatch.capability(index_queries.device) in Arch.BLACKWELL
        and index_queries.dtype == bucket_centroids.dtype == torch.bfloat16
        and index_queries.shape[1] == bucket_centroids.shape[1] == Index.DIM
        and index_queries.is_contiguous()
        and bucket_centroids.is_contiguous()
        and index_queries.data_ptr() % 16 == 0
        and bucket_centroids.data_ptr() % 16 == 0
    )


def span_for(rows: int, bucket_count: int, device: torch.device) -> int:
    """Return the tiles a CTA scans, so the CTAs fill the GPU's slots about WAVES times over.

    Args:
        rows: Rows there are.
        bucket_count: Buckets there are.
        device: The GPU.
    """
    tiles = -(-bucket_count // BlackwellBucketScan.BLOCK_N)
    query_blocks = -(-rows // BlackwellBucketScan.BLOCK_Q)
    slots = torch.cuda.get_device_properties(device).multi_processor_count * BlackwellBucketScan.MIN_BLOCKS
    shortest, longest = SharedBucketScan.SPAN_TILES
    return min(longest, max(shortest, -(-tiles * query_blocks // (BlackwellBucketScan.WAVES * slots))))


def scan_buckets(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    thresholds: torch.Tensor,
    capacity: int,
    span_tiles: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """List the buckets each row sees that score at or above its threshold.

    Args:
        index_queries: Index queries [R, index_dim] bf16.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        thresholds: Score each row lists at or above [R] fp32.
        capacity: Buckets a row's listing holds.
        span_tiles: Tiles a CTA scans, up to MAX_SPAN_TILES - None to fill the GPU about WAVES times over.

    Returns:
        The listing [R, capacity] int32 - a row's first `capacity` survivors in no defined order, unset after.
        Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.

    Raises:
        ValueError: The rows are not ones the kernel takes, or the span is not one it holds.
    """
    if span_tiles is not None and not 1 <= span_tiles <= SharedBucketScan.MAX_SPAN_TILES:
        raise ValueError(
            f"the CuTeDSL listing scans 1 to {SharedBucketScan.MAX_SPAN_TILES} tiles a span - got {span_tiles}"
        )
    if not supports(index_queries, bucket_centroids):
        raise ValueError(
            "the CuTeDSL listing takes contiguous, 16-byte aligned bf16 rows of 64 index dims on a Blackwell GPU - "
            f"got {index_queries.dtype} {tuple(index_queries.shape)} and {bucket_centroids.dtype} "
            f"{tuple(bucket_centroids.shape)} on {index_queries.device}"
        )
    rows = index_queries.shape[0]
    device = index_queries.device
    survivor_counts = torch.zeros(rows, device=device, dtype=torch.int32)
    buckets = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    if rows == 0 or bucket_centroids.shape[0] == 0 or capacity == 0:
        return buckets, survivor_counts
    compiled()(
        index_queries,
        bucket_centroids,
        visible_counts.int().contiguous(),
        thresholds.float().contiguous(),
        survivor_counts,
        buckets,
        span_tiles or span_for(rows, bucket_centroids.shape[0], device),
    )
    return buckets, survivor_counts
