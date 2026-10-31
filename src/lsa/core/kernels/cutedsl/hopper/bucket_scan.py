"""The buckets each index query scores at or above its threshold, listed in one CuTeDSL kernel on Hopper.

The scan's listing of survivors, compacted in shared memory: a CTA holds 128 index queries and streams a span of
bucket centroids past them, scoring each tile of 64 buckets on the tensor cores. A score at or above its row's
threshold appends its bucket to the row's list in shared memory, and when the span is done each row reserves its
place in the global listing with one atomic and writes its buckets there. A row that fills its list in shared
memory appends the rest to the global listing one at a time.

The CTA is a warpgroup of consumers and a producer warp. The producer copies each tile of centroids into shared
memory with the tensor memory accelerator, behind a full and an empty barrier a stage, so a consumer waits on its
tile and never on its peers. The tensor cores read both the queries and the centroids from shared memory, so no
register holds an operand. A tile is scored 64 rows at a time into two accumulators, so the tensor cores score one
while the threads test the other.

A pass of STAGES tiles is unrolled, so a tile's stage is a constant of the code and its MMA operands are addressed
with constants. The span is padded to a whole pass, not capped at the last tile as Blackwell's is, because a stage
rolling over from one pass to the next must stay a constant - a tile past the buckets arrives as zeros, which the
rows' visible counts mask, and the span stops at the buckets' last pass.

The list holds a row's survivors, not their bits, so no second pass expands them. Survivors are rare - some 1,700
of 1.27M buckets a row at 10M - so a thread first sets one bit a score with no branch and queues the word, and at
the end of a pass appends the set bits of its queued words only.

The scores are the tensor-core bf16 dot with an fp32 accumulator, as the Triton scan's are - tested equal to Triton's.
The lowered threshold (MARGIN) and the rescoring make the select exact regardless. The order a row lists them in is
not defined and not read.

Scores and thresholds are finite, bar a threshold of -inf or inf. The fast path tests the sign bit of a score less its
threshold, which counts a NaN as a hit where `>=` does not. Scores are internal, so nothing checks.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as hopper
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.hopper.tile import accumulator_rows_by_columns

# Index queries a CTA holds, a warpgroup's threads, buckets a tile scores, and rows an accumulator holds
BLOCK_Q = 128
BLOCK_N = 64
MMA_ROWS = 64
# Tiles a CTA scans before its rows reserve their place in the global listing, at most and at least - within
# them a span is sized so the CTAs fill the GPU's slots about WAVES times over
SPAN_TILES = (4, 64)
WAVES = 64
# Survivors a row holds in shared memory a span - some 11 arrive at 10M, so few rows ever append past it
ROW_SLOTS = 32
# Tiles of bucket centroids in flight, which is also the tiles of a pass, and CTAs an SM holds
STAGES = 2
MIN_BLOCKS = 4
# Words of hits a thread queues in shared memory a pass, one for each of a pass's tiles and row halves
QUEUE_SLOTS = 2 * STAGES
# Tiles a span holds at most, which a queued word's tag in an int16 bounds
MAX_SPAN_TILES = 2**14
# Columns of a thread's accumulator fragment, the bits of a queued word a row
FRAGMENT_COLUMNS = BLOCK_N // 4
# Index dims the kernel takes - a row of 128 bytes, the widest swizzle
INDEX_DIM = 64


@dsl_user_op
def push_sign(word: Any, value: Any, *, loc: Any = None, ip: Any = None) -> Any:
    """Return a word shifted left one bit with the sign of an fp32 value brought in.

    `shf.l.wrap d, a, b, 1` gives d = (b << 1) | (a >>> 31), with a the value and b the word,
    so the float's sign bit becomes bit 0.

    Args:
        word: The uint32 word.
        value: The fp32 value.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip), cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
            "shf.l.wrap.b32 $0, $1, $2, 1;",
            "=r,f,r",
            loc=loc,
            ip=ip,
        )
    )


class BucketScan:
    """The CuTeDSL bucket scan."""

    def __init__(self) -> None:
        """Fix the CTA's consumer threads, its producer warp and its barrier."""
        self.threads = BLOCK_Q
        self.cta_threads = BLOCK_Q + 32
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)

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
        """Lay out shared memory, the copies and the MMA, and launch a CTA per 128 rows and span of buckets.

        Args:
            index_queries: Index queries [R, index_dim] bf16.
            bucket_centroids: Bucket centroids [N, index_dim] bf16.
            visible_counts: Buckets each row sees [R] int32, at most N.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32, zero on entry.
            buckets: The listing [R, capacity] int32.
            span_tiles: Tiles a CTA scans, a multiple of STAGES.
            stream: The stream to launch on.
        """
        dtype = index_queries.element_type
        mma_tile = (BLOCK_Q, BLOCK_N, INDEX_DIM)
        query_layout = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, mma_tile, dtype, 1)
        centroid_layout = hopper.make_smem_layout_b(LayoutEnum.ROW_MAJOR, mma_tile, dtype, STAGES)

        @cute.struct
        class Storage:
            """Shared memory of a block: query and centroid tiles, row buckets, queue and stage barriers."""

            queries: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(query_layout)], 1024]
            centroids: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(centroid_layout)], 1024]
            row_counts: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, BLOCK_Q], 16]
            row_buckets: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, BLOCK_Q * ROW_SLOTS], 16]
            queue_words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, BLOCK_Q * QUEUE_SLOTS], 16]
            queue_tags: cute.struct.Align[cute.struct.MemRange[cutlass.Int16, BLOCK_Q * QUEUE_SLOTS], 16]
            full_empty: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2 * STAGES], 8]

        elements = 128 // dtype.width
        per_row = INDEX_DIM // elements
        global_to_shared = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype, num_bits_per_copy=128
            ),
            cute.make_layout((self.threads // per_row, per_row), stride=(per_row, 1)),
            cute.make_layout((1, elements)),
        )
        tiled_mma = hopper.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            (1, 1, 1),
            (MMA_ROWS, BLOCK_N),
        )
        tma_atom, tma_centroids = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            bucket_centroids,
            cute.slice_(centroid_layout, (None, None, 0)),
            (BLOCK_N, INDEX_DIM),
        )
        grid = (
            cute.ceil_div(index_queries.shape[0], BLOCK_Q),
            cute.ceil_div(cute.ceil_div(bucket_centroids.shape[0], BLOCK_N), span_tiles),
            1,
        )
        self.kernel(
            index_queries,
            bucket_centroids.shape[0],
            visible_counts,
            thresholds,
            survivor_counts,
            buckets,
            span_tiles,
            query_layout,
            centroid_layout,
            global_to_shared,
            tiled_mma,
            tma_atom,
            tma_centroids,
            Storage,
        ).launch(grid=grid, block=[self.cta_threads, 1, 1], stream=stream, min_blocks_per_mp=MIN_BLOCKS)

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        bucket_count: cutlass.Int32,
        visible_counts: cute.Tensor,
        thresholds: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        span_tiles: cutlass.Int32,
        query_layout: cute.ComposedLayout,
        centroid_layout: cute.ComposedLayout,
        global_to_shared: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tma_atom: cute.CopyAtom,
        tma_centroids: cute.Tensor,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Copy a span of buckets in with a producer warp and score them for 128 rows with the consumers.

        Args:
            index_queries: Index queries [R, index_dim] bf16.
            bucket_count: Buckets there are.
            visible_counts: Buckets each row sees [R] int32.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
            span_tiles: Tiles a CTA scans, a multiple of STAGES.
            query_layout: Shared memory of the CTA's index queries, swizzled as the MMA reads it.
            centroid_layout: Shared memory of the tiles of bucket centroids in flight.
            global_to_shared: The asynchronous copy of the queries into shared memory.
            tiled_mma: The warpgroup's MMA over 64 rows and a tile.
            tma_atom: The bulk copy of a tile of centroids into shared memory.
            tma_centroids: The centroids as the bulk copy addresses them.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        warp_group = cute.arch.make_warp_uniform(thread // self.threads)
        query_block, span, _ = cute.arch.block_idx()
        rows = index_queries.shape[0]
        first_tile = span * span_tiles
        tile_count = cute.ceil_div(bucket_count, BLOCK_N)
        # The span is padded to whole passes, so its last tiles may be past the buckets, but never a whole pass past
        # => a tile past the buckets arrives as zeros and the visible counts mask it
        stop_tile = first_tile + cutlass.min(
            span_tiles, (tile_count - first_tile + STAGES - 1) // STAGES * STAGES
        )

        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        shared_queries = storage.queries.get_tensor(query_layout.outer, swizzle=query_layout.inner)
        shared_centroids = storage.centroids.get_tensor(centroid_layout.outer, swizzle=centroid_layout.inner)
        row_counts = storage.row_counts.get_tensor(cute.make_layout(BLOCK_Q))
        row_buckets = storage.row_buckets.get_tensor(
            cute.make_layout((BLOCK_Q, ROW_SLOTS), stride=(ROW_SLOTS, 1))
        )
        # A slot's words are the threads' in a row
        # => a warp's stores fall in different banks
        queue_layout = cute.make_layout((QUEUE_SLOTS, BLOCK_Q), stride=(BLOCK_Q, 1))
        queue_words = storage.queue_words.get_tensor(queue_layout)
        queue_tags = storage.queue_tags.get_tensor(queue_layout)
        tile_bytes = cute.size_in_bytes(
            index_queries.element_type, cute.slice_(centroid_layout, (None, None, 0))
        )
        centroid_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.full_empty.data_ptr(),
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads // 32),
            tx_count=tile_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        centroid_tiles = cute.local_tile(tma_centroids, (BLOCK_N, INDEX_DIM), (None, 0))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_centroids, 0, 2),
            cute.group_modes(centroid_tiles, 0, 2),
        )

        # The consumers copy the queries in once
        if warp_group == 0:
            row_counts[thread] = cutlass.Int32(0)
            copy_slice = global_to_shared.get_slice(thread)
            query_tile = cute.local_tile(index_queries, (BLOCK_Q, INDEX_DIM), (query_block, 0))
            query_coords = cute.local_tile(
                cute.make_identity_tensor(index_queries.shape), (BLOCK_Q, INDEX_DIM), (query_block, 0)
            )
            tQgQ = copy_slice.partition_S(query_tile)
            tQsQ = copy_slice.partition_D(shared_queries[None, None, 0])
            tQcQ = copy_slice.partition_S(query_coords)
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][0], rows):
                    cute.copy(global_to_shared, tQgQ[None, m, None], tQsQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        if warp_group == 1:
            self.produce(centroid_pipeline, tma_atom, tBgB, tBsB, first_tile, stop_tile)
        else:
            self.consume(
                centroid_pipeline,
                thread,
                query_block,
                rows,
                bucket_count,
                first_tile,
                stop_tile,
                visible_counts,
                thresholds,
                survivor_counts,
                buckets,
                shared_queries,
                shared_centroids,
                queue_words,
                queue_tags,
                row_counts,
                row_buckets,
                tiled_mma,
            )

    @cute.jit
    def produce(
        self,
        centroid_pipeline: Any,
        tma_atom: cute.CopyAtom,
        tBgB: cute.Tensor,
        tBsB: cute.Tensor,
        first_tile: cutlass.Int32,
        stop_tile: cutlass.Int32,
    ) -> None:
        """Copy each tile of bucket centroids into the stage the consumers have freed.

        Args:
            centroid_pipeline: The pipeline of full and empty stages.
            tma_atom: The bulk copy of a tile.
            tBgB: The tiles in global memory.
            tBsB: The tile stages in shared memory.
            first_tile: The first tile of the span.
            stop_tile: One past the last.
        """
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
        for tile in range(first_tile, stop_tile, 1):
            centroid_pipeline.producer_acquire(state)
            cute.copy(
                tma_atom,
                tBgB[None, tile],
                tBsB[None, state.index],
                tma_bar_ptr=centroid_pipeline.producer_get_barrier(state),
            )
            centroid_pipeline.producer_commit(state)
            state.advance()

    @cute.jit
    def consume(
        self,
        centroid_pipeline: Any,
        thread: cutlass.Int32,
        query_block: cutlass.Int32,
        rows: cutlass.Int32,
        bucket_count: cutlass.Int32,
        first_tile: cutlass.Int32,
        stop_tile: cutlass.Int32,
        visible_counts: cute.Tensor,
        thresholds: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        shared_queries: cute.Tensor,
        shared_centroids: cute.Tensor,
        queue_words: cute.Tensor,
        queue_tags: cute.Tensor,
        row_counts: cute.Tensor,
        row_buckets: cute.Tensor,
        tiled_mma: cute.TiledMma,
    ) -> None:
        """Score the span's tiles for 128 rows as they arrive, list each row's survivors, then flush them.

        Args:
            centroid_pipeline: The pipeline of full and empty stages.
            thread: The thread, and the row of the CTA it flushes.
            query_block: The CTA's block of rows.
            rows: Rows there are.
            bucket_count: Buckets there are.
            first_tile: The first tile of the span.
            stop_tile: One past the last.
            visible_counts: Buckets each row sees [R] int32.
            thresholds: Score each row lists at or above [R] fp32.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
            shared_queries: The CTA's index queries in shared memory.
            shared_centroids: The tile stages in shared memory.
            queue_words: The threads' queues of words of hits [QUEUE_SLOTS, BLOCK_Q].
            queue_tags: The row half and tile of each queued word, from the span's first.
            row_counts: Survivors each row of the CTA has had this span.
            row_buckets: The rows' lists in shared memory.
            tiled_mma: The warpgroup's MMA over 64 rows and a tile.
        """
        mma_slice = tiled_mma.get_slice(thread)
        tCrA = [
            tiled_mma.make_fragment_A(
                mma_slice.partition_A(
                    cute.local_tile(shared_queries[None, None, 0], (MMA_ROWS, INDEX_DIM), (row_half, 0))
                )
            )
            for row_half in range(2)
        ]
        tCrB = tiled_mma.make_fragment_B(mma_slice.partition_B(shared_centroids))
        accumulator_shape = mma_slice.partition_shape_C((MMA_ROWS, BLOCK_N))
        accumulators = [cute.make_rmem_tensor(accumulator_shape, cutlass.Float32) for _ in range(2)]

        # A thread's rows are the same on every tile
        # => their thresholds and visible counts load once, into registers
        row_thresholds = []
        row_visible_counts = []
        least_visible = []
        for row_half in cutlass.range_constexpr(2):
            first_coords = accumulator_rows_by_columns(
                mma_slice.partition_C(
                    cute.local_tile(
                        cute.make_identity_tensor((rows, bucket_count)),
                        (MMA_ROWS, BLOCK_N),
                        (2 * query_block + row_half, 0),
                    )
                )
            )
            rows_held = cute.size(first_coords, mode=[0])
            half_thresholds = cute.make_rmem_tensor(rows_held, cutlass.Float32)
            half_visible_counts = cute.make_rmem_tensor(rows_held, cutlass.Int32)
            for r in cutlass.range_constexpr(rows_held):
                row = first_coords[r, 0][0]
                if row < rows:
                    half_thresholds[r] = thresholds[row]
                    half_visible_counts[r] = visible_counts[row]
                else:
                    half_thresholds[r] = cutlass.Float32.inf
                    half_visible_counts[r] = cutlass.Int32(0)
            half_least_visible = half_visible_counts[0]
            for r in cutlass.range_constexpr(1, rows_held):
                half_least_visible = cutlass.min(half_least_visible, half_visible_counts[r])
            row_thresholds.append(half_thresholds)
            row_visible_counts.append(half_visible_counts)
            least_visible.append(half_least_visible)

        # Two accumulators alternate, A0 for a tile's rows 0-63 and A1 for its rows 64-127
        # Timeline of a tile t in stage s, with MMA(a) the tile scored into a
        #   MMA(A1) issued       in flight A0, A1   wait_group(1) => A0 done   test A0 while A1 is scored
        #   MMA(A0, t + 1) issued  in flight A1, A0   wait_group(1) => A1 done   release stage s, test A1 while A0 is scored
        # The last tile of the span issues no next MMA and waits with wait_group(0)
        # A stage is released only once A1 is done, because both halves of tile t read it
        # => the producer may refill it with tile t + STAGES only then
        waited = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        released = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        centroid_pipeline.consumer_wait(waited)
        waited.advance()
        self.score_tile(accumulators[0], tCrA[0], tCrB, 0)

        # A pass takes STAGES tiles, so a tile's stage is known when the kernel is built
        # => the MMA's operands are addressed with constants
        for first_of_pass in range(first_tile, stop_tile, STAGES):
            queued = cutlass.Int32(0)
            for stage in cutlass.range_constexpr(STAGES):
                tile = first_of_pass + stage
                self.score_tile(accumulators[1], tCrA[1], tCrB, stage)
                warpgroup.wait_group(1)
                queued = self.test_scores(
                    accumulators[0],
                    tile,
                    0,
                    first_tile,
                    thread,
                    queued,
                    mma_slice,
                    query_block,
                    rows,
                    bucket_count,
                    least_visible[0],
                    row_thresholds[0],
                    row_visible_counts[0],
                    queue_words,
                    queue_tags,
                )
                if stage + 1 < STAGES:
                    centroid_pipeline.consumer_wait(waited)
                    waited.advance()
                    self.score_tile(accumulators[0], tCrA[0], tCrB, stage + 1)
                    warpgroup.wait_group(1)
                else:
                    if first_of_pass + STAGES < stop_tile:
                        centroid_pipeline.consumer_wait(waited)
                        waited.advance()
                        self.score_tile(accumulators[0], tCrA[0], tCrB, 0)
                        warpgroup.wait_group(1)
                    else:
                        warpgroup.wait_group(0)
                # Both row halves of the tile are scored
                # => its stage takes the tile STAGES ahead
                centroid_pipeline.consumer_release(released)
                released.advance()
                queued = self.test_scores(
                    accumulators[1],
                    tile,
                    1,
                    first_tile,
                    thread,
                    queued,
                    mma_slice,
                    query_block,
                    rows,
                    bucket_count,
                    least_visible[1],
                    row_thresholds[1],
                    row_visible_counts[1],
                    queue_words,
                    queue_tags,
                )
            self.drain(
                thread,
                queued,
                first_tile,
                query_block,
                queue_words,
                queue_tags,
                row_counts,
                row_buckets,
                survivor_counts,
                buckets,
            )

        self.flush(thread, query_block, rows, row_counts, row_buckets, survivor_counts, buckets)

    @cute.jit
    def score_tile(
        self,
        accumulator: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        stage: int,
    ) -> None:
        """Start the MMA of 64 of the CTA's rows against a tile into an accumulator, to be waited on later.

        Args:
            accumulator: The accumulator it overwrites.
            tCrA: The MMA's view of the 64 rows' queries in shared memory.
            tCrB: The MMA's view of the tiles in shared memory.
            stage: The tile's stage.
        """
        # The kernel's MMA is built in another region of the consume loop, which this one may not use
        # => an MMA of its own, the same one, rebuilt here
        mma = hopper.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            (1, 1, 1),
            (MMA_ROWS, BLOCK_N),
        )
        # The first product overwrites the accumulator
        # => no zeroing pass
        mma.set(warpgroup.Field.ACCUMULATE, False)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(mma, accumulator, tCrA[None, None, k], tCrB[None, None, k, stage], accumulator)
            mma.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def test_scores(
        self,
        accumulator: cute.Tensor,
        tile: cutlass.Int32,
        row_half: int,
        first_tile: cutlass.Int32,
        thread: cutlass.Int32,
        queued: cutlass.Int32,
        mma_slice: Any,
        query_block: cutlass.Int32,
        rows: cutlass.Int32,
        bucket_count: cutlass.Int32,
        least_visible: cutlass.Int32,
        row_thresholds: cute.Tensor,
        row_visible_counts: cute.Tensor,
        queue_words: cute.Tensor,
        queue_tags: cute.Tensor,
    ) -> cutlass.Int32:
        """Set a bit for each score of 64 rows against a tile at or above its row's threshold and queue the word.

        Args:
            accumulator: The scores of the 64 rows against the tile.
            tile: The tile.
            row_half: The 64 rows, the first or the second of the CTA's.
            first_tile: The first tile of the span.
            thread: The thread, and the queue it holds.
            queued: Words the thread has queued this pass.
            mma_slice: This thread's slice of the MMA.
            query_block: The CTA's block of rows.
            rows: Rows there are.
            bucket_count: Buckets there are.
            least_visible: Buckets the thread's row seeing fewest sees.
            row_thresholds: The thread's rows' thresholds.
            row_visible_counts: Buckets each of the thread's rows sees.
            queue_words: The threads' queues of words of hits [QUEUE_SLOTS, BLOCK_Q].
            queue_tags: The row half and tile of each queued word, from the span's first.

        Returns:
            Words the thread has queued this pass.
        """
        scores = accumulator_rows_by_columns(accumulator)
        coords = accumulator_rows_by_columns(
            mma_slice.partition_C(
                cute.local_tile(
                    cute.make_identity_tensor((rows, bucket_count)),
                    (MMA_ROWS, BLOCK_N),
                    (2 * query_block + row_half, tile),
                )
            )
        )
        # A thread holds 32 scores
        # => a word has a bit for each
        rows_held = cute.size(scores, mode=[0])
        columns = cute.size(scores, mode=[1])
        hits = cutlass.Uint32(0)
        if (tile + 1) * BLOCK_N <= least_visible:
            # A score at or above its threshold leaves a difference of sign 0, exactly, and the last pushed is bit 0
            # A NaN difference has sign 0, the sign of a hit
            # => scores and thresholds must be finite
            misses = cutlass.Uint32(0)
            for i in cutlass.range_constexpr(rows_held * columns - 1, -1, -1):
                r = i // columns
                misses = push_sign(misses, scores[r, i % columns] - row_thresholds[r])
            hits = ~misses
        else:
            for r in cutlass.range_constexpr(rows_held):
                for c in cutlass.range_constexpr(columns):
                    hit = cutlass.Uint32(
                        (scores[r, c] >= row_thresholds[r]) & (coords[r, c][1] < row_visible_counts[r])
                    )
                    hits = hits | (hit << (r * columns + c))
        # Words are queued and drained at the end of a pass, not appended here
        # => the divergent clz loop of a drain stays out of the scoring and testing overlap
        # A word is stored whether or not it has a hit, and kept only if it does
        # => no branch, and a word with no hit is written over by the next
        queue_words[queued, thread] = hits
        queue_tags[queued, thread] = cutlass.Int16(2 * (tile - first_tile) + row_half)
        return queued + cutlass.Int32(hits != 0)

    @cute.jit
    def drain(
        self,
        thread: cutlass.Int32,
        queued: cutlass.Int32,
        first_tile: cutlass.Int32,
        query_block: cutlass.Int32,
        queue_words: cute.Tensor,
        queue_tags: cute.Tensor,
        row_counts: cute.Tensor,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
    ) -> None:
        """Append the survivors of a thread's queued words to their rows' lists.

        Args:
            thread: The thread, and the queue it holds.
            queued: Words the thread has queued.
            first_tile: The first tile of the span.
            query_block: The CTA's block of rows.
            queue_words: The threads' queues of words of hits [QUEUE_SLOTS, BLOCK_Q].
            queue_tags: The row half and tile of each queued word, from the span's first.
            row_counts: Survivors each row of the CTA has had this span.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
        """
        # A lane l of warp w holds, in the m64n64 accumulator
        #   rows     16 w + l // 4, and the row 8 below it
        #   columns  2 (l % 4) and 2 (l % 4) + 1, then the same pair 8 further, for 8 pairs
        # => a lane quad shares its rows, and bit i of a word is row i // 16, column i % 16 of the fragment
        lane_row = query_block * BLOCK_Q + thread // 32 * 16 + thread % 32 // 4
        lane_bucket = thread % 4 * 2
        for slot in range(queued):
            tag = cutlass.Int32(queue_tags[slot, thread])
            self.append(
                queue_words[slot, thread],
                lane_row + (tag & 1) * MMA_ROWS,
                (first_tile + tag // 2) * BLOCK_N + lane_bucket,
                row_counts,
                row_buckets,
                survivor_counts,
                buckets,
                query_block,
            )

    @cute.jit
    def append(
        self,
        hits: cutlass.Uint32,
        first_row: cutlass.Int32,
        first_bucket: cutlass.Int32,
        row_counts: cute.Tensor,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
        query_block: cutlass.Int32,
    ) -> None:
        """Append the bucket of each set bit to its row's list in shared memory, or past it to the global listing.

        Args:
            hits: A bit a score, set for a survivor.
            first_row: The row of the thread's first score.
            first_bucket: The bucket of the thread's first score.
            row_counts: Survivors each row of the CTA has had this span.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
            query_block: The CTA's block of rows.
        """
        while hits != 0:
            top = cutlass.Uint32(31) - cute.arch.clz(hits)
            hits = hits ^ (cutlass.Uint32(1) << top)
            # A bit is r * 16 + c, and c is 2 * pair + (c & 1)
            # => the row is 8 r more, the bucket (c & 1) + 8 * (c >> 1) more
            r = top // FRAGMENT_COLUMNS
            c = top % FRAGMENT_COLUMNS
            row = first_row + cutlass.Int32(r << 3)
            bucket = first_bucket + cutlass.Int32((c & 1) + ((c >> 1) << 3))
            local = row - query_block * BLOCK_Q
            # A row's 32 slots never reset, so after ROW_SLOTS survivors in a span each is a GPU atomic
            # A reset needs the four lanes of a row to agree, and in this divergent loop they drain different words
            # => no flush-and-refill as Blackwell has, where a thread owns its row
            slot = cute.arch.atomic_add((row_counts.iterator + local).llvm_ptr, cutlass.Int32(1), scope="cta")
            if slot < ROW_SLOTS:
                row_buckets[local, slot] = bucket
            else:
                at = cute.arch.atomic_add(
                    (survivor_counts.iterator + row).llvm_ptr, cutlass.Int32(1), sem="relaxed", scope="gpu"
                )
                if at < buckets.shape[1]:
                    buckets[row, at] = bucket

    @cute.jit
    def flush(
        self,
        thread: cutlass.Int32,
        query_block: cutlass.Int32,
        rows: cutlass.Int32,
        row_counts: cute.Tensor,
        row_buckets: cute.Tensor,
        survivor_counts: cute.Tensor,
        buckets: cute.Tensor,
    ) -> None:
        """Reserve each row's place in the global listing with one atomic and copy its list there.

        Args:
            thread: The thread, and the row of the CTA it flushes.
            query_block: The CTA's block of rows.
            rows: Rows there are.
            row_counts: Survivors each row of the CTA had this span.
            row_buckets: The rows' lists in shared memory.
            survivor_counts: Buckets each row listed [R] int32.
            buckets: The listing [R, capacity] int32.
        """
        self.barrier.arrive_and_wait()
        row = query_block * BLOCK_Q + thread
        if row < rows:
            held = cutlass.min(row_counts[thread], ROW_SLOTS)
            if held > 0:
                base = cute.arch.atomic_add(
                    (survivor_counts.iterator + row).llvm_ptr, held, sem="relaxed", scope="gpu"
                )
                for i in range(cutlass.min(held, buckets.shape[1] - base)):
                    buckets[row, base + i] = row_buckets[thread, i]


COMPILED: dict[str, Any] = {}


def compiled() -> Any:
    """Return the listing compiled once, taking torch tensors as they are."""
    if "bucket scan" in COMPILED:
        return COMPILED["bucket scan"]
    rows, bucket_count, capacity = (cute.sym_int32() for _ in range(3))
    bucket_stride = cute.sym_int64()
    arguments = (
        make_fake_compact_tensor(cutlass.BFloat16, (rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(
            cutlass.BFloat16, (bucket_count, INDEX_DIM), stride_order=(1, 0), assumed_align=16
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
    """Tell whether the CuTeDSL listing takes these rows - on Hopper, bf16, 64 index dims, contiguous, 16-byte aligned.

    Args:
        index_queries: Index queries [R, index_dim].
        bucket_centroids: Bucket centroids [N, index_dim].
    """
    return (
        index_queries.is_cuda
        and dispatch.capability(index_queries.device) == dispatch.HOPPER
        and index_queries.dtype == bucket_centroids.dtype == torch.bfloat16
        and index_queries.shape[1] == bucket_centroids.shape[1] == INDEX_DIM
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
    tiles = -(-bucket_count // BLOCK_N)
    query_blocks = -(-rows // BLOCK_Q)
    slots = torch.cuda.get_device_properties(device).multi_processor_count * MIN_BLOCKS
    shortest, longest = SPAN_TILES
    return min(longest, max(shortest, -(-tiles * query_blocks // (WAVES * slots))))


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
        span_tiles: Tiles a CTA scans, up to MAX_SPAN_TILES, rounded up to whole passes - None to fill the GPU about
            WAVES times over.

    Returns:
        The listing [R, capacity] int32 - a row's first `capacity` survivors in no defined order, unset after.
        Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.

    Raises:
        ValueError: The rows are not ones the kernel takes, or the span is not one it holds.
    """
    if span_tiles is not None and not 1 <= span_tiles <= MAX_SPAN_TILES:
        raise ValueError(f"the CuTeDSL listing scans 1 to {MAX_SPAN_TILES} tiles a span - got {span_tiles}")
    if not supports(index_queries, bucket_centroids):
        raise ValueError(
            "the CuTeDSL listing takes contiguous, 16-byte aligned bf16 rows of 64 index dims on a Hopper GPU - got "
            f"{index_queries.dtype} {tuple(index_queries.shape)} and {bucket_centroids.dtype} "
            f"{tuple(bucket_centroids.shape)} on {index_queries.device}"
        )
    rows = index_queries.shape[0]
    device = index_queries.device
    survivor_counts = torch.zeros(rows, device=device, dtype=torch.int32)
    buckets = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    if rows == 0 or bucket_centroids.shape[0] == 0 or capacity == 0:
        return buckets, survivor_counts
    span_tiles = span_tiles or span_for(rows, bucket_centroids.shape[0], device)
    compiled()(
        index_queries,
        bucket_centroids,
        visible_counts.int().contiguous(),
        thresholds.float().contiguous(),
        survivor_counts,
        buckets,
        -(-span_tiles // STAGES) * STAGES,
    )
    return buckets, survivor_counts
