"""Flat's sampled thresholds in CuTeDSL - each row's threshold from every stride-th bucket, without the sample's scores.

The Triton path writes the sample's scores, a row by every stride-th bucket - 1.8 GB at 10M - then reads each row
twice: once for its mean and best, once for a histogram of 64 bins between them. Here the scores never leave the
chip. A CTA holds 128 rows and a part of one block of 4096 samples, reads the samples straight from the centroids,
scores them a tile at a time on the tensor cores, and works from the MMA's accumulators: a first pass adds each
row's samples, a second bins them, and a last launch takes each row's threshold from its bins.

The thresholds are the Triton ones bit for bit. The scores are the same bf16 dot with an fp32 accumulator. The mean
is the fp32 sum Triton forms: a block of 4096 is 128 threads, each adding its own columns in order - every 128th,
or runs of four when the sample count is a multiple of 16, which Triton loads four wide - then each warp's 32 by a
butterfly, the warps as (0 + 2) + (1 + 3), and the blocks in order. A Triton warp's columns are every other tile,
or every other pair of tiles four wide, so a block splits by Triton warp into parts summed apart. A thread keeps the
running sum of each Triton thread whose columns it holds, the four-wide tiles read permuted so a thread holds each
run of four whole. The divisions are Triton's approximate `div.full` and the threshold its two fused multiply-adds.

Only the top of a row's histogram decides its threshold. The first pass also keeps each row's greatest score in 128
disjoint groups, and the rank-th greatest of those maxima is at most the row's rank-th best - so the threshold's bin
is at least that floor's, and the second pass counts no bin under it. A launch between the passes takes each row's
mean, bin width, floor and the least score that bins at or above it, so the second pass tests a score against one
edge and bins only those that reach it.

The maxima and the bins live in per-row words the later launches read and reset, so a call starts on clean ones.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

# Rows a CTA holds, samples a tile scores, and the CTA's threads - a warp's 16 rows a thread's two
ROWS = 128
COLUMNS = 64
THREADS = 256
WARPS = THREADS // 32
INDEX_DIM = 64
SWIZZLE_DIM = 64
# Slots a thread tests together against its row's edge
HIT_SLOTS = 4
# CTAs an SM holds at once, which caps a thread's registers - the sums pass's, the histogram pass's
MIN_BLOCKS = {False: 2, True: 3}
# Samples a Triton program of the histogram adds as one block, and the tiles of it
BLOCK = 4096
TILES = BLOCK // COLUMNS
BINS = 64
# Disjoint groups of a row's samples, whose maxima's rank-th greatest bounds its rank-th best - each block of an
# odd or even place in a set of groups of its own
GROUPS = 128
GROUP_SETS = 2
# Triton's slack under a bin's edge, so a sampled bucket scored at it again still counts
EDGE_SLACK = 2**-14
# Steps an estimate of a bin's edge moves, in ulps, to meet it
EDGE_STEPS = 8


def accumulator_rows_by_columns(accumulator: cute.Tensor) -> cute.Tensor:
    """View an MMA accumulator fragment as its rows by its columns.

    Args:
        accumulator: A thread's fragment of the MMA's C, ((2, 2), M, N).

    Returns:
        The same registers as ((2, M), (2, N)).
    """
    layout = cute.make_layout(accumulator.layout.shape)
    rows_by_columns = cute.make_layout(
        ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[2])),
        stride=((layout.stride[0][1], layout.stride[1]), (layout.stride[0][0], layout.stride[2])),
    )
    return cute.make_tensor(accumulator.iterator, cute.composition(accumulator.layout, rows_by_columns))


def supports(index_queries: torch.Tensor, bucket_centroids: torch.Tensor) -> bool:
    """Tell whether the CuTeDSL thresholds take these rows - 64 index dims in bf16.

    Args:
        index_queries: Index queries [R, index_dim].
        bucket_centroids: Bucket centroids [N, index_dim].
    """
    return (
        index_queries.shape[1] == INDEX_DIM
        and index_queries.dtype == torch.bfloat16
        and bucket_centroids.dtype == torch.bfloat16
        and bucket_centroids.shape[0] > 0
    )


@cute.jit
def row_limit(visible_counts: cute.Tensor, row: Any, stride: Any, sample_count: Any) -> Any:
    """Return the samples a row sees - every stride-th bucket it sees, none past the last row.

    Args:
        visible_counts: Buckets each row sees [R] int32.
        row: The row.
        stride: Buckets between two samples.
        sample_count: Samples in all.
    """
    limit = cutlass.Int32(0)
    if row < visible_counts.shape[0]:
        limit = cutlass.min((visible_counts[row] + stride - 1) // stride, sample_count)
    return limit


@cute.jit
def row_mean_width(warp_sums: cute.Tensor, bests: cute.Tensor, row: Any, limit: Any) -> Any:
    """Return a row's mean and bin width, its Triton warps' sums added in order as Triton adds them.

    Args:
        warp_sums: Each row's sum a block by Triton warp [R, blocks, 4] fp32.
        bests: Each row's best score [R] fp32.
        row: The row.
        limit: Samples the row sees.
    """
    total = cutlass.Float32(0.0)
    for b in range(warp_sums.shape[1]):
        total = total + (
            (warp_sums[row, b, 0] + warp_sums[row, b, 2]) + (warp_sums[row, b, 1] + warp_sums[row, b, 3])
        )
    mean = cute.math.div(total, cutlass.Float32(cutlass.max(limit, 1)), full=True)
    width = cutlass.max(
        cute.math.div(bests[row] - mean, cutlass.Float32(BINS), full=True), cutlass.Float32(1e-30)
    )
    return mean, width


@cute.jit
def row_bin(score: Any, mean: Any, reciprocal: Any) -> Any:
    """Return a score's bin of its row as Triton bins it, or a negative number under the mean.

    Args:
        score: The score.
        mean: The row's mean.
        reciprocal: The reciprocal of the row's bin width.
    """
    # div.full is the dividend times the divisor's reciprocal, for a divisor as far from 0 and inf as a width,
    # and Triton's clamp at 0 only lifts what is under every counted bin
    return cutlass.Int32(cutlass.min((score - mean) * reciprocal, cutlass.Float32(BINS - 1)))


def greatest_of(values: list[Any]) -> Any:
    """Return the greatest of a power of two of values, taken as a tree so no max waits on the one before.

    Args:
        values: The values.
    """
    while len(values) > 1:
        values = [cute.arch.fmax(values[2 * j], values[2 * j + 1]) for j in range(len(values) // 2)]
    return values[0]


@cute.jit
def float_key(value: Any) -> Any:
    """Return an fp32's bits as an int32 that orders as it does.

    Args:
        value: The fp32.
    """
    bits = cutlass.Int32(llvm.bitcast(T.i32(), cutlass.Float32(value).ir_value()))
    key = bits
    if bits < 0:
        key = bits ^ cutlass.Int32(0x7FFFFFFF)
    return key


@cute.jit
def key_float(key: Any) -> Any:
    """Return the fp32 whose `float_key` is key.

    Args:
        key: The int32 key.
    """
    bits = key
    if key < 0:
        bits = key ^ cutlass.Int32(0x7FFFFFFF)
    return cutlass.Float32(llvm.bitcast(T.f32(), cutlass.Int32(bits).ir_value()))


@cute.jit
def count_keys_from(keys: list[Any], least: Any) -> Any:
    """Return how many of a warp's keys are at least a key.

    Args:
        keys: Each lane's keys, int32.
        least: The key.
    """
    count = cutlass.Int32(0)
    for key in keys:
        count = count + cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(key >= least)))
    return count


@cute.jit
def rank_th_greatest(keys: list[Any], rank: Any) -> Any:
    """Return the rank-th greatest key of a warp's - the greatest key with rank keys at or above it.

    Args:
        keys: Each lane's keys, int32.
        rank: The rank, from 1 to the keys in all.
    """
    # The sign first, then each lower bit from the top, kept where rank keys still reach it
    base = cutlass.Int32(0)
    if count_keys_from(keys, cutlass.Int32(0)) < rank:
        base = cutlass.Int32(-(2**31))
    below = cutlass.Int32(0)
    for i in range(31):
        candidate = below | (cutlass.Int32(1) << (30 - i))
        if count_keys_from(keys, base + candidate) >= rank:
            below = candidate
    return base + below


@cute.jit
def bin_edge(lowest_bin: Any, mean: Any, reciprocal: Any, width: Any) -> Any:
    """Return the least score a row bins at or above its lowest counted bin.

    Args:
        lowest_bin: The row's lowest counted bin, 1 to 63.
        mean: The row's mean.
        reciprocal: The reciprocal of the row's bin width.
        width: The row's bin width.
    """
    # The bin is the truncated (score - mean) * reciprocal, which never falls as the score rises
    # => the scores binned at or above a bin are those from one edge, found from the estimate by whole steps
    target = cutlass.Float32(lowest_bin)
    key = float_key(cute.math.fma(target, width, mean))
    for _ in range(EDGE_STEPS):
        if (key_float(key - 1) - mean) * reciprocal >= target:
            key = key - 1
    for _ in range(EDGE_STEPS):
        if (key_float(key) - mean) * reciprocal < target:
            key = key + 1
    return key_float(key)


class SamplePass:
    """One pass over a part of a block of 4096 samples for 128 rows - its Triton warps' sums, or its histogram."""

    def __init__(self, histogram: bool, four_wide: bool) -> None:
        """Fix which pass, and which order Triton adds a thread's columns in.

        Args:
            histogram: The second pass, binning the samples - the first adds them.
            four_wide: The sample count is a multiple of 16, so Triton adds runs of four - and the tiles are permuted.
        """
        self.histogram = histogram
        # The sums pass takes two tiles a barrier, so each tile's parity is known at compile time
        self.unroll = 1 if histogram else 2
        self.stages = 2 * self.unroll
        self.four_wide = four_wide
        # A Triton warp's columns are every other tile one at a time, every other pair of tiles four at a time
        # => a block splits by Triton warp into parts, each summed in Triton's order on its own
        self.parts = 4 if four_wide else 2
        self.part_tiles = TILES // self.parts
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)

    def tile_of(self, block: Any, part: Any, i: Any) -> Any:
        """Return the i-th tile of a block's part.

        Args:
            block: The block.
            part: The part - its Triton warps.
            i: The tile's place in the part.
        """
        if self.four_wide:
            return block * TILES + 8 * (i // 2) + 2 * part + i % 2
        return block * TILES + 2 * i + part

    def slot_column(self, slot: int) -> int:
        """Return a slot's column in its tile, less its thread's own offset.

        Args:
            slot: The slot, c = 2 n + e of the accumulator's n-th column pair.
        """
        if self.four_wide:
            return 16 * (slot // 4) + slot % 4
        return 8 * (slot // 2) + slot % 2

    def tile_sample(self, row: Any) -> Any:
        """Return the sample a tile's row of shared memory holds, from the tile's first.

        Args:
            row: The row, 8 n + 2 q + e - the MMA's column.
        """
        if self.four_wide:
            return 16 * (row // 16) + 4 * ((row // 2) % 4) + 2 * ((row // 8) % 2) + row % 2
        return row

    @cute.jit
    def tile_source(self, centroids: cute.Tensor, tiles: Any, tile: Any, stride: Any) -> cute.Tensor:
        """Return a tile of samples where they lie among the centroids, its rows in the MMA's column order.

        Args:
            centroids: Bucket centroids [N, 64] bf16.
            tiles: Tiles of samples in all, the last past the last sample.
            tile: The tile.
            stride: Buckets between two samples.
        """
        # The samples as one tensor from the centroids' first, so a tile's start keeps their alignment
        sample_stride = cute.assume(stride * INDEX_DIM, divby=INDEX_DIM)
        if cutlass.const_expr(self.four_wide):
            # MMA column 8 n + 2 q + e holds sample 16 (n // 2) + 4 q + 2 (n % 2) + e of its tile
            tile_rows = cute.make_layout(
                (2, 4, 2, 4), stride=(sample_stride, 4 * sample_stride, 2 * sample_stride, 16 * sample_stride)
            )
        else:
            tile_rows = cute.make_layout(COLUMNS, stride=sample_stride)
        samples = cute.make_tensor(
            centroids.iterator,
            cute.make_layout(
                ((tile_rows.shape, tiles), INDEX_DIM), stride=((tile_rows.stride, COLUMNS * sample_stride), 1)
            ),
        )
        return samples[(None, tile), None]

    @cute.jit
    def __call__(
        self,
        queries: cute.Tensor,
        centroids: cute.Tensor,
        visible_counts: cute.Tensor,
        warp_sums: cute.Tensor,
        bests: cute.Tensor,
        group_maxima: cute.Tensor,
        counts: cute.Tensor,
        prepared: cute.Tensor,
        stride: cutlass.Int32,
        sample_count: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Lay out shared memory, the copies and the MMA, and launch a CTA per 128 rows and block part.

        Args:
            queries: Index queries [R, 64] bf16.
            centroids: Bucket centroids [N, 64] bf16.
            visible_counts: Buckets each row sees [R] int32.
            warp_sums: Each row's sum a block by Triton warp [R', blocks, 4] fp32, R' the rows rounded up to 128.
            bests: Each row's best score [R'] fp32, -inf on entry to the first pass.
            group_maxima: Each row's 64 group maxima [R', 64] fp32, -inf on entry to the first pass.
            counts: Each row's histogram [R', 64] int32, zero on entry to the second pass.
            prepared: Each row's mean, bin width reciprocal, least counted score and bin width [R', 4] fp32.
            stride: Buckets between two samples.
            sample_count: Samples in all.
            stream: The stream to launch on.
        """
        dtype = queries.element_type
        swizzled = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, SWIZZLE_DIM), stride=(SWIZZLE_DIM, 1))
        )
        query_layout = cute.tile_to_shape(swizzled, (ROWS, INDEX_DIM), (0, 1))
        sample_layout = cute.tile_to_shape(swizzled, (COLUMNS, INDEX_DIM, self.stages), (0, 1, 2))
        histogram_words = ROWS * BINS if self.histogram else 1

        @cute.struct
        class Storage:
            """Shared memory of a block: query tile, sample stages, a histogram and per-warp limits."""

            queries: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(query_layout)], 1024]
            samples: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sample_layout)], 1024]
            histogram: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, histogram_words], 16]
            warp_limits: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, WARPS], 16]

        elements = 128 // dtype.width
        per_row = SWIZZLE_DIM // elements
        global_to_shared = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dtype, num_bits_per_copy=128
            ),
            cute.make_layout((THREADS // per_row, per_row), stride=(per_row, 1)),
            cute.make_layout((1, elements)),
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16)),
            (WARPS, 1, 1),
            permutation_mnk=(WARPS * 16, 16, 16),
        )
        self.kernel(
            queries,
            centroids,
            visible_counts,
            warp_sums,
            bests,
            group_maxima,
            counts,
            prepared,
            stride,
            sample_count,
            query_layout,
            sample_layout,
            global_to_shared,
            tiled_mma,
            Storage,
        ).launch(
            grid=(cute.ceil_div(queries.shape[0], ROWS), warp_sums.shape[1] * self.parts, 1),
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=MIN_BLOCKS[self.histogram],
        )

    @cute.kernel
    def kernel(
        self,
        queries: cute.Tensor,
        centroids: cute.Tensor,
        visible_counts: cute.Tensor,
        warp_sums: cute.Tensor,
        bests: cute.Tensor,
        group_maxima: cute.Tensor,
        counts: cute.Tensor,
        prepared: cute.Tensor,
        stride: cutlass.Int32,
        sample_count: cutlass.Int32,
        query_layout: cute.ComposedLayout,
        sample_layout: cute.ComposedLayout,
        global_to_shared: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Score a block part's samples for 128 rows a tile at a time, and add or bin each tile from the accumulators.

        Args:
            queries: Index queries [R, 64] bf16.
            centroids: Bucket centroids [N, 64] bf16.
            visible_counts: Buckets each row sees [R] int32.
            warp_sums: Each row's sum a block by Triton warp [R', blocks, 4] fp32.
            bests: Each row's best score [R'] fp32.
            group_maxima: Each row's 64 group maxima [R', 64] fp32.
            counts: Each row's histogram [R', 64] int32.
            prepared: Each row's mean, bin width reciprocal, least counted score and bin width [R', 4] fp32.
            stride: Buckets between two samples.
            sample_count: Samples in all.
            query_layout: Shared memory of the CTA's index queries.
            sample_layout: Shared memory of the tiles of samples in flight.
            global_to_shared: The asynchronous copy of a tile into shared memory.
            tiled_mma: The warps' MMA over a tile.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row_block, block_part, _ = cute.arch.block_idx()
        block = block_part // self.parts
        part = block_part % self.parts
        lane = cute.arch.lane_idx()
        quad = lane % 4
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        shared_queries = storage.queries.get_tensor(query_layout)
        shared_samples = storage.samples.get_tensor(sample_layout)
        histogram = storage.histogram.get_tensor(cute.make_layout(ROWS * BINS if self.histogram else 1))
        warp_limits = storage.warp_limits.get_tensor(cute.make_layout(WARPS))

        # A row past the last is never read, and scores as a dead row
        copy_slice = global_to_shared.get_slice(thread)
        tQgQ = copy_slice.partition_S(cute.local_tile(queries, (ROWS, INDEX_DIM), (row_block, 0)))
        tQcQ = copy_slice.partition_S(
            cute.local_tile(cute.make_identity_tensor(queries.shape), (ROWS, INDEX_DIM), (row_block, 0))
        )
        query_rows = cute.make_rmem_tensor((1, *tQcQ.shape[1:]), cutlass.Boolean)
        for m in cutlass.range_constexpr(cute.size(tQcQ, mode=[1])):
            for k in cutlass.range_constexpr(cute.size(tQcQ, mode=[2])):
                query_rows[0, m, k] = cute.elem_less(tQcQ[0, m, k], queries.shape)
        cute.copy(global_to_shared, tQgQ, copy_slice.partition_D(shared_queries), pred=query_rows)
        cute.arch.cp_async_commit_group()
        # A sample past the last is never read, and scores as a dead sample
        tScS = copy_slice.partition_S(cute.make_identity_tensor((COLUMNS, INDEX_DIM)))
        tiles = warp_sums.shape[1] * TILES
        for first in cutlass.range_constexpr(self.unroll):
            self.load_tile(
                centroids,
                tiles,
                self.tile_of(block, part, first),
                stride,
                sample_count,
                global_to_shared,
                copy_slice,
                tScS,
                shared_samples,
                first,
            )
        cute.arch.cp_async_commit_group()

        mma_slice = tiled_mma.get_slice(thread)
        tSrQ = mma_slice.make_fragment_A(mma_slice.partition_A(shared_queries))
        tSrS = mma_slice.make_fragment_B(mma_slice.partition_B(shared_samples[None, None, 0]))
        load_queries = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), queries.element_type
            ),
            tiled_mma,
        ).get_slice(thread)
        load_samples = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), queries.element_type
            ),
            tiled_mma,
        ).get_slice(thread)
        tSrQ_view = load_queries.retile(tSrQ)
        tSrS_view = load_samples.retile(tSrS)

        # A thread's two rows and the slots it holds of each
        coords = accumulator_rows_by_columns(
            mma_slice.partition_C(
                cute.local_tile(
                    cute.make_identity_tensor((cute.ceil_div(queries.shape[0], ROWS) * ROWS, COLUMNS)),
                    (ROWS, COLUMNS),
                    (row_block, 0),
                )
            )
        )
        row_count = cute.size(coords, mode=[0])
        slots = cute.size(coords, mode=[1])
        row_limits = cute.make_rmem_tensor(row_count, cutlass.Int32)
        means = cute.make_rmem_tensor(row_count, cutlass.Float32)
        reciprocals = cute.make_rmem_tensor(row_count, cutlass.Float32)
        edges = cute.make_rmem_tensor(row_count, cutlass.Float32)
        for r in cutlass.range_constexpr(row_count):
            row = coords[r, 0][0]
            row_limits[r] = row_limit(visible_counts, row, stride, sample_count)
            if cutlass.const_expr(self.histogram):
                means[r] = prepared[row, 0]
                reciprocals[r] = prepared[row, 1]
                edges[r] = prepared[row, 2]
        # The CTA's least limit - a tile under it is whole for every row
        least = row_limits[0]
        for r in cutlass.range_constexpr(1, row_count):
            least = cutlass.min(least, row_limits[r])
        for offset in cutlass.range_constexpr(5):
            least = cutlass.min(least, cute.arch.shuffle_sync_bfly(least, 16 >> offset))
        if lane == 0:
            warp_limits[thread // 32] = least
        if cutlass.const_expr(self.histogram):
            for i in range(thread, ROWS * BINS, THREADS):
                histogram[i] = cutlass.Int32(0)
        # The running sums of the Triton threads whose columns this thread holds - a run of four's by tile parity
        # -0.0 adds as nothing, so a sum begun from it is Triton's begun from its first column
        partials = cute.make_rmem_tensor(
            2 * row_count * (slots // 4) if self.four_wide else row_count * slots, cutlass.Float32
        )
        partials.fill(-0.0)
        maxima = cute.make_rmem_tensor((row_count, slots // self.parts), cutlass.Float32)
        maxima.fill(-cutlass.Float32.inf)

        cute.arch.cp_async_wait_group(1)
        self.barrier.arrive_and_wait()
        whole_limit = warp_limits[0]
        for w in cutlass.range_constexpr(1, WARPS):
            whole_limit = cutlass.min(whole_limit, warp_limits[w])
        tSsQ = load_queries.partition_S(shared_queries)
        for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(load_queries, tSsQ[None, None, k], tSrQ_view[None, None, k])

        accumulator_shape = mma_slice.partition_shape_C((ROWS, COLUMNS))
        # Two tiles a step, so each tile's parity is known at compile time
        unroll = self.unroll
        steps = self.part_tiles // unroll
        # The tiles every row of the CTA sees whole are a prefix of the part
        # => they take a loop with no slot checks, the rest one that checks each slot
        whole_tiles = cutlass.Int32(0)
        for i in range(self.part_tiles):
            if (self.tile_of(block, part, i) + 1) * COLUMNS <= whole_limit:
                whole_tiles = whole_tiles + 1
        whole_steps = whole_tiles // unroll
        for checked in (False, True):
            first_step, last_step = (whole_steps, steps) if checked else (0, whole_steps)
            for step in range(first_step, last_step):
                for parity in cutlass.range_constexpr(unroll):
                    i = step * unroll + parity
                    tile = self.tile_of(block, part, i)
                    tSsS = load_samples.partition_S(shared_samples[None, None, i % self.stages])
                    if cutlass.const_expr(parity == 0):
                        cute.arch.cp_async_wait_group(0)
                        self.barrier.arrive_and_wait()
                        # Every thread is past the last step's tiles
                        # => their stages take the next step's
                        if i + unroll < self.part_tiles:
                            for ahead in cutlass.range_constexpr(unroll):
                                self.load_tile(
                                    centroids,
                                    tiles,
                                    self.tile_of(block, part, i + unroll + ahead),
                                    stride,
                                    sample_count,
                                    global_to_shared,
                                    copy_slice,
                                    tScS,
                                    shared_samples,
                                    (i + unroll + ahead) % self.stages,
                                )
                        cute.arch.cp_async_commit_group()

                    accumulator = cute.make_rmem_tensor(accumulator_shape, cutlass.Float32)
                    accumulator.fill(0.0)
                    for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
                        cute.copy(load_samples, tSsS[None, None, k], tSrS_view[None, None, k])
                        cute.gemm(
                            tiled_mma, accumulator, tSrQ[None, None, k], tSrS[None, None, k], accumulator
                        )
                    scores = accumulator_rows_by_columns(accumulator)
                    live = cute.make_rmem_tensor((row_count, slots), cutlass.Boolean)
                    if cutlass.const_expr(checked):
                        # A dead sample scores zero, as in Triton, below every bin and no group's maximum
                        first_column = tile * COLUMNS + (4 * quad if self.four_wide else 2 * quad)
                        for r in cutlass.range_constexpr(row_count):
                            for c in cutlass.range_constexpr(slots):
                                live[r, c] = first_column + self.slot_column(c) < row_limits[r]
                                if cutlass.const_expr(not self.histogram):
                                    scores[r, c] = cutlass.Float32(
                                        cutlass.select_(live[r, c], scores[r, c], cutlass.Float32(0.0))
                                    )
                    if cutlass.const_expr(self.histogram):
                        self.bin_tile(
                            scores,
                            live,
                            checked,
                            coords,
                            row_block,
                            means,
                            reciprocals,
                            edges,
                            histogram,
                        )
                    else:
                        for r in cutlass.range_constexpr(row_count):
                            for c in cutlass.range_constexpr(slots):
                                group = c // self.parts
                                grown = cute.arch.fmax(maxima[r, group], scores[r, c])
                                if cutlass.const_expr(checked):
                                    grown = cutlass.Float32(
                                        cutlass.select_(live[r, c], grown, maxima[r, group])
                                    )
                                maxima[r, group] = grown
                        self.add_tile(scores, parity, row_count, slots, partials)

        if cutlass.const_expr(self.histogram):
            self.barrier.arrive_and_wait()
            for i in range(thread, ROWS * BINS, THREADS):
                count = histogram[i]
                if count > 0:
                    cute.arch.atomic_add((counts.iterator + (row_block * ROWS * BINS + i)).llvm_ptr, count)
        else:
            self.write_part(
                coords, block, part, quad, row_count, slots, partials, maxima, warp_sums, bests, group_maxima
            )

    @cute.jit
    def load_tile(
        self,
        centroids: cute.Tensor,
        tiles: Any,
        tile: Any,
        stride: Any,
        sample_count: Any,
        global_to_shared: cute.TiledCopy,
        copy_slice: Any,
        tScS: cute.Tensor,
        shared_samples: cute.Tensor,
        stage: Any,
    ) -> None:
        """Start copying a tile of samples into a stage, none past the last.

        Args:
            centroids: Bucket centroids [N, 64] bf16.
            tiles: Tiles of samples in all.
            tile: The tile.
            stride: Buckets between two samples.
            sample_count: Samples in all.
            global_to_shared: The asynchronous copy of a tile into shared memory.
            copy_slice: This thread's part of the copy.
            tScS: This thread's rows and dims of a tile.
            shared_samples: The stages.
            stage: The stage.
        """
        present = cute.make_rmem_tensor((1, *tScS.shape[1:]), cutlass.Boolean)
        for m in cutlass.range_constexpr(cute.size(tScS, mode=[1])):
            for k in cutlass.range_constexpr(cute.size(tScS, mode=[2])):
                present[0, m, k] = tile * COLUMNS + self.tile_sample(tScS[0, m, k][0]) < sample_count
        cute.copy(
            global_to_shared,
            copy_slice.partition_S(self.tile_source(centroids, tiles, tile, stride)),
            copy_slice.partition_D(shared_samples[None, None, stage]),
            pred=present,
        )

    @cute.jit
    def add_tile(
        self, values: cute.Tensor, parity: int, row_count: int, slots: int, partials: cute.Tensor
    ) -> None:
        """Add a tile's live scores to the running sums of the Triton threads they feed, each in its order.

        Args:
            values: The tile's scores, zero where dead [2, 16] fp32.
            parity: The tile's parity in its part.
            row_count: Rows a thread holds.
            slots: Slots a thread holds of a row.
            partials: The running sums.
        """
        if cutlass.const_expr(not self.four_wide):
            # Slot c of every tile of the part feeds the same Triton thread
            for r in cutlass.range_constexpr(row_count):
                for c in cutlass.range_constexpr(slots):
                    partials[r * slots + c] = partials[r * slots + c] + values[r, c]
        else:
            # Slots 4 k to 4 k + 3 are one Triton thread's run of four, of the eighth of its block by the tile's parity
            runs = slots // 4
            for r in cutlass.range_constexpr(row_count):
                for k in cutlass.range_constexpr(runs):
                    p = (parity * row_count + r) * runs + k
                    partials[p] = (
                        ((partials[p] + values[r, 4 * k]) + values[r, 4 * k + 1]) + values[r, 4 * k + 2]
                    ) + values[r, 4 * k + 3]

    @cute.jit
    def bin_tile(
        self,
        scores: cute.Tensor,
        live: cute.Tensor,
        checked: cutlass.Constexpr,
        coords: cute.Tensor,
        row_block: Any,
        means: cute.Tensor,
        reciprocals: cute.Tensor,
        edges: cute.Tensor,
        histogram: cute.Tensor,
    ) -> None:
        """Count a tile's live scores into their rows' bins, none under a row's lowest counted bin.

        Args:
            scores: The tile's scores, rows by slots.
            live: Which slots are live, read when checked.
            checked: Whether any slot may be dead.
            coords: Each slot's row and column.
            row_block: The CTA's rows.
            means: Each row's mean.
            reciprocals: The reciprocal of each row's bin width.
            edges: The least score each row counts.
            histogram: The CTA's bins [128 * 64] int32.
        """
        # A score is counted rarely, so the greatest of a few slots says whether any of them counts
        for r in cutlass.range_constexpr(cute.size(scores, mode=[0])):
            local_row = coords[r, 0][0] - row_block * ROWS
            for first in cutlass.range_constexpr(0, cute.size(scores, mode=[1]), HIT_SLOTS):
                candidates = []
                for c in cutlass.range_constexpr(first, first + HIT_SLOTS):
                    score = scores[r, c]
                    if cutlass.const_expr(checked):
                        score = cutlass.Float32(cutlass.select_(live[r, c], score, -cutlass.Float32.inf))
                    candidates.append(score)
                greatest = greatest_of(candidates)
                if greatest >= edges[r]:
                    for c in cutlass.range_constexpr(first, first + HIT_SLOTS):
                        counted = scores[r, c] >= edges[r]
                        if cutlass.const_expr(checked):
                            counted = counted & live[r, c]
                        if counted:
                            bin_index = row_bin(scores[r, c], means[r], reciprocals[r])
                            cute.arch.atomic_add(
                                (histogram.iterator + (local_row * BINS + bin_index)).llvm_ptr,
                                cutlass.Int32(1),
                                scope="cta",
                            )

    @cute.jit
    def write_part(
        self,
        coords: cute.Tensor,
        block: Any,
        part: Any,
        quad: Any,
        row_count: int,
        slots: int,
        partials: cute.Tensor,
        maxima: cute.Tensor,
        warp_sums: cute.Tensor,
        bests: cute.Tensor,
        group_maxima: cute.Tensor,
    ) -> None:
        """Add the part's Triton threads' sums of each row as each Triton warp adds them, and raise the row's maxima.

        Args:
            coords: Each slot's row and column.
            block: The block.
            part: The part - its Triton warps.
            quad: This lane's place in its quad.
            row_count: Rows a thread holds.
            slots: Slots a thread holds of a row.
            partials: The running sums.
            maxima: This thread's group maxima [2, 16] fp32.
            warp_sums: Each row's sum a block by Triton warp.
            bests: Each row's best score.
            group_maxima: Each row's 64 group maxima.
        """
        for r in cutlass.range_constexpr(row_count):
            row = coords[r, 0][0]
            if cutlass.const_expr(not self.four_wide):
                # Triton thread 64 part + 8 n + 2 quad + e: lane bits e, quad and n % 4, warp 2 part + n // 4
                for high in cutlass.range_constexpr(2):
                    pair = cute.make_rmem_tensor(2, cutlass.Float32)
                    for e in cutlass.range_constexpr(2):
                        lanes = cute.make_rmem_tensor(4, cutlass.Float32)
                        for low in cutlass.range_constexpr(4):
                            lanes[low] = partials[r * slots + 2 * (4 * high + low) + e]
                        lanes[0] = lanes[0] + lanes[2]
                        lanes[1] = lanes[1] + lanes[3]
                        lanes[0] = lanes[0] + lanes[1]
                        folded = lanes[0] + cute.arch.shuffle_sync_bfly(lanes[0], 2)
                        pair[e] = folded + cute.arch.shuffle_sync_bfly(folded, 1)
                    if quad == 0:
                        warp_sums[row, block, 2 * part + high] = pair[0] + pair[1]
            else:
                # Triton thread 16 eighth + 4 k + quad: lane bits quad, k and eighth % 2, warp eighth // 2 - the part
                runs = slots // 4
                lanes = cute.make_rmem_tensor(runs, cutlass.Float32)
                for k in cutlass.range_constexpr(runs):
                    lanes[k] = partials[r * runs + k] + partials[(row_count + r) * runs + k]
                lanes[0] = lanes[0] + lanes[2]
                lanes[1] = lanes[1] + lanes[3]
                lanes[0] = lanes[0] + lanes[1]
                folded = lanes[0] + cute.arch.shuffle_sync_bfly(lanes[0], 2)
                folded = folded + cute.arch.shuffle_sync_bfly(folded, 1)
                if quad == 0:
                    warp_sums[row, block, part] = folded
            # A group is one slot of one quad lane over every block and part - 64 disjoint groups a row
            groups = slots // self.parts
            first_group = (block % GROUP_SETS) * (GROUPS // GROUP_SETS) + (part * 4 + quad) * groups
            best = maxima[r, 0]
            for c in cutlass.range_constexpr(groups):
                best = cute.arch.fmax(best, maxima[r, c])
                cute.arch.atomic_fmax(
                    (group_maxima.iterator + (row * GROUPS + first_group + c)).llvm_ptr, maxima[r, c]
                )
            for offset in cutlass.range_constexpr(2):
                best = cute.arch.fmax(best, cute.arch.shuffle_sync_bfly(best, 1 << offset))
            if quad == 0:
                cute.arch.atomic_fmax((bests.iterator + row).llvm_ptr, best)


class Prepare:
    """Each row's mean, bin width reciprocal and least counted score, from its sums and group maxima - a warp a row."""

    @cute.jit
    def __call__(
        self,
        warp_sums: cute.Tensor,
        bests: cute.Tensor,
        group_maxima: cute.Tensor,
        visible_counts: cute.Tensor,
        prepared: cute.Tensor,
        stride: cutlass.Int32,
        sample_count: cutlass.Int32,
        rank: cutlass.Int32,
        floored: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a warp a row.

        Args:
            warp_sums: Each row's sum a block by Triton warp [R', blocks, 4] fp32.
            bests: Each row's best score [R'] fp32.
            group_maxima: Each row's group maxima [R', 128] fp32, reset to -inf.
            visible_counts: Buckets each row sees [R] int32.
            prepared: Each row's mean, bin width reciprocal, least counted score and bin width [R', 4] fp32.
            stride: Buckets between two samples.
            sample_count: Samples in all.
            rank: Samples at or above the threshold's bin a row wants.
            floored: 1 to count no bin under the floor's, 0 to count every bin.
            stream: The stream to launch on.
        """
        self.kernel(
            warp_sums, bests, group_maxima, visible_counts, prepared, stride, sample_count, rank, floored
        ).launch(grid=(cute.ceil_div(prepared.shape[0], WARPS), 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        warp_sums: cute.Tensor,
        bests: cute.Tensor,
        group_maxima: cute.Tensor,
        visible_counts: cute.Tensor,
        prepared: cute.Tensor,
        stride: cutlass.Int32,
        sample_count: cutlass.Int32,
        rank: cutlass.Int32,
        floored: cutlass.Int32,
    ) -> None:
        """Take a row's mean and bin width, the bin of its floor, and the least score that bins there or above.

        Args:
            warp_sums: Each row's sum a block by Triton warp.
            bests: Each row's best score.
            group_maxima: Each row's group maxima.
            visible_counts: Buckets each row sees.
            prepared: Each row's mean, bin width reciprocal and least counted score.
            stride: Buckets between two samples.
            sample_count: Samples in all.
            rank: Samples at or above the threshold's bin a row wants.
            floored: 1 to count no bin under the floor's.
        """
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = block * WARPS + thread // 32
        if row < prepared.shape[0]:
            mean, width = row_mean_width(
                warp_sums, bests, row, row_limit(visible_counts, row, stride, sample_count)
            )
            reciprocal = cute.math.div(cutlass.Float32(1.0), width, full=True)
            keys = []
            for j in cutlass.range_constexpr(GROUPS // 32):
                keys.append(float_key(group_maxima[row, lane + 32 * j]))
                group_maxima[row, lane + 32 * j] = -cutlass.Float32.inf
            # The floor is a score of the row, and at most its rank-th best - binned the same way, its bin is at
            # most the threshold's
            # => the bins under it, and the first, never decide the threshold, and go uncounted
            lowest_bin = cutlass.Int32(1)
            if floored != 0:
                floor = key_float(rank_th_greatest(keys, rank))
                lowest_bin = cutlass.max(row_bin(floor, mean, reciprocal), 1)
            edge = bin_edge(lowest_bin, mean, reciprocal, width)
            if lane == 0:
                prepared[row, 0] = mean
                prepared[row, 1] = reciprocal
                prepared[row, 2] = edge
                prepared[row, 3] = width


class Threshold:
    """Each row's threshold from its histogram - the lower edge of the top bin the rank reaches, as Triton forms it."""

    @cute.jit
    def __call__(
        self,
        bests: cute.Tensor,
        counts: cute.Tensor,
        prepared: cute.Tensor,
        norms: cute.Tensor,
        thresholds: cute.Tensor,
        rank: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a warp a row.

        Args:
            bests: Each row's best score [R'] fp32, reset to -inf.
            counts: Each row's histogram [R', 64] int32, reset to zero.
            prepared: Each row's mean, bin width reciprocal, least counted score and bin width [R', 4] fp32.
            norms: Each index query's norm [R] fp32.
            thresholds: Each row's threshold [R] fp32.
            rank: Samples at or above the threshold's bin a row wants.
            stream: The stream to launch on.
        """
        self.kernel(bests, counts, prepared, norms, thresholds, rank).launch(
            grid=(cute.ceil_div(thresholds.shape[0], WARPS), 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        bests: cute.Tensor,
        counts: cute.Tensor,
        prepared: cute.Tensor,
        norms: cute.Tensor,
        thresholds: cute.Tensor,
        rank: cutlass.Int32,
    ) -> None:
        """Take a row's top bin the rank reaches and its edge less Triton's slack, then reset the row's words.

        Args:
            bests: Each row's best score.
            counts: Each row's histogram.
            prepared: Each row's mean, bin width reciprocal, least counted score and bin width.
            norms: Each index query's norm.
            thresholds: Each row's threshold.
            rank: Samples at or above the threshold's bin a row wants.
        """
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = block * WARPS + thread // 32
        if row < thresholds.shape[0]:
            # Lane l holds bins 63 - 2 l and 62 - 2 l, so the lanes in order count down from the top bin
            high_bin = BINS - 1 - 2 * lane
            high = counts[row, high_bin]
            low = counts[row, high_bin - 1]
            counts[row, high_bin] = cutlass.Int32(0)
            counts[row, high_bin - 1] = cutlass.Int32(0)
            inclusive = high + low
            for i in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(inclusive, 1 << i, mask_and_clamp=0)
                if lane >= (1 << i):
                    inclusive = inclusive + other
            # The first bin is never counted
            reached_high = cute.arch.vote_ballot_sync(inclusive - low >= rank)
            reached_low = cute.arch.vote_ballot_sync((inclusive >= rank) & (lane < 31))
            reached = cutlass.Int32(reached_high | reached_low)
            top = cutlass.Int32(0)
            if reached != 0:
                first = cutlass.Int32(cute.arch.popc((reached & -reached) - 1))
                top = BINS - 1 - 2 * first
                if (cutlass.Int32(reached_high) >> first) & 1 == 0:
                    top = top - 1
            # -inf where no bin above the first reaches the rank
            threshold = -cutlass.Float32.inf
            if top > 0:
                edge = cute.math.fma(prepared[row, 3], cutlass.Float32(top), prepared[row, 0])
                threshold = cute.math.fma(norms[row], cutlass.Float32(-EDGE_SLACK), edge)
            if lane == 0:
                thresholds[row] = threshold
                bests[row] = -cutlass.Float32.inf


class Workspace:
    """A device's per-row words - the maxima and bins each call leaves as it found them."""

    def __init__(self, rows: int, device: torch.device) -> None:
        """Allocate the words for rows rows, clean.

        Args:
            rows: Rows the words cover, a multiple of 128.
            device: The device.
        """
        self.bests = torch.full((rows,), -torch.inf, device=device)
        self.group_maxima = torch.full((rows, GROUPS), -torch.inf, device=device)
        self.counts = torch.zeros(rows, BINS, device=device, dtype=torch.int32)


WORKSPACES: dict[torch.device, Workspace] = {}
COMPILED: dict[tuple[str, bool], Any] = {}


def workspace_for(rows: int, device: torch.device) -> Workspace:
    """Return the device's workspace, grown to cover rows rows.

    Args:
        rows: Rows the call covers, a multiple of 128.
        device: The device.
    """
    if device not in WORKSPACES or WORKSPACES[device].bests.shape[0] < rows:
        WORKSPACES[device] = Workspace(rows, device)
    return WORKSPACES[device]


def compiled(name: str, four_wide: bool) -> Any:
    """Return a launch compiled once for any rows, centroids and blocks, taking torch tensors as they are.

    Args:
        name: The launch - `sums`, `histogram` or `threshold`.
        four_wide: The order Triton adds a thread's columns in.
    """
    key = (name, four_wide)
    if key in COMPILED:
        return COMPILED[key]
    rows, centroid_count, padded, blocks = (
        cute.sym_int32(),
        cute.sym_int32(),
        cute.sym_int32(),
        cute.sym_int32(),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    warp_sums = make_fake_compact_tensor(
        cutlass.Float32, (padded, blocks, 4), stride_order=(2, 1, 0), assumed_align=4
    )
    bests = make_fake_compact_tensor(cutlass.Float32, (padded,), assumed_align=4)
    group_maxima = make_fake_compact_tensor(
        cutlass.Float32, (padded, GROUPS), stride_order=(1, 0), assumed_align=4
    )
    counts = make_fake_compact_tensor(cutlass.Int32, (padded, BINS), stride_order=(1, 0), assumed_align=4)
    visible_counts = make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4)
    prepared = make_fake_compact_tensor(cutlass.Float32, (padded, 4), stride_order=(1, 0), assumed_align=16)
    scalars = (cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(1))
    arguments: tuple[Any, ...]
    kernel: Any
    if name == "threshold":
        norms = make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4)
        thresholds = make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4)
        arguments = (bests, counts, prepared, norms, thresholds, scalars[0], stream)
        kernel = Threshold()
    elif name == "prepare":
        arguments = (
            warp_sums,
            bests,
            group_maxima,
            visible_counts,
            prepared,
            *scalars,
            cutlass.Int32(1),
            stream,
        )
        kernel = Prepare()
    else:
        queries = make_fake_compact_tensor(
            cutlass.BFloat16, (rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16
        )
        centroids = make_fake_compact_tensor(
            cutlass.BFloat16, (centroid_count, INDEX_DIM), stride_order=(1, 0), assumed_align=16
        )
        arguments = (
            queries,
            centroids,
            visible_counts,
            warp_sums,
            bests,
            group_maxima,
            counts,
            prepared,
            *scalars[:2],
            stream,
        )
        kernel = SamplePass(name == "histogram", four_wide)
    COMPILED[key] = cute.compile(kernel, *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


def sample_thresholds(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    stride: int,
    rank: int,
) -> torch.Tensor:
    """Estimate each row's score at its rank among every stride-th bucket it sees - the Triton thresholds, bit for bit.

    Args:
        index_queries: Index queries [R, 64] bf16.
        bucket_centroids: Bucket centroids [N, 64] bf16 of unit rows.
        visible_counts: Buckets each row sees [R] int32, at most N.
        stride: Buckets between two samples.
        rank: Samples at or above the threshold's bin a row wants.

    Returns:
        Estimated scores [R] fp32, -inf for a row seeing too few to rank.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    if not supports(index_queries, bucket_centroids):
        raise ValueError(
            "the CuTeDSL thresholds take bf16 index queries and centroids of 64 dims, and a centroid at least - got "
            f"{index_queries.dtype} {tuple(index_queries.shape)} and {bucket_centroids.dtype} "
            f"{tuple(bucket_centroids.shape)}"
        )
    rows = index_queries.shape[0]
    device = index_queries.device
    sample_count = -(-bucket_centroids.shape[0] // stride)
    four_wide = sample_count % 16 == 0
    padded = -(-rows // ROWS) * ROWS
    words = workspace_for(padded, device)
    bests, group_maxima, counts = words.bests[:padded], words.group_maxima[:padded], words.counts[:padded]
    warp_sums = torch.empty(padded, -(-sample_count // BLOCK), 4, device=device)
    thresholds = torch.empty(rows, device=device)
    norms = index_queries.float().norm(dim=1)
    queries, centroids = index_queries.contiguous(), bucket_centroids.contiguous()
    # The rank-th greatest of the group maxima has rank scores at or above it, so it bounds no rank past the groups
    floored = int(rank <= GROUPS)
    prepared = torch.empty(padded, 4, device=device)
    passes = (
        queries,
        centroids,
        visible_counts,
        warp_sums,
        bests,
        group_maxima,
        counts,
        prepared,
        stride,
        sample_count,
    )
    compiled("sums", four_wide)(*passes)
    compiled("prepare", False)(
        warp_sums, bests, group_maxima, visible_counts, prepared, stride, sample_count, rank, floored
    )
    compiled("histogram", four_wide)(*passes)
    compiled("threshold", False)(bests, counts, prepared, norms, thresholds, rank)
    return thresholds
