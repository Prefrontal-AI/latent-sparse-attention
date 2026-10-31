"""A tree's levels in CuTeDSL, a kernel each - the carried children scored, the best kept, their children named.

A level scores its open node buckets and the children the level above kept, keeps the best `beam_width` columns of
each query's row, and names the children of the kept ones for the level below. The Triton path writes the row of
scores, picks from it in four launches and gathers the children in a fifth. Here a CTA holds one query's row in shared
memory: it scores the carried children, finds the row's k-th greatest score a byte at a time, and writes the kept
columns' children straight out, so the row never leaves the SM.

The same kernel keeps a start level, whose scores are in hand and only read, and a second writes the last level: the
buckets no node bucket covers and the last beam's children, the best `top_p` written as the keys the expansion by
threshold reads, so the bucket level's scores, pick and cache rows are one launch too.

A row many times wider than its keep is not searched whole. A strided sample of its words sets a threshold expected to
leave about 1.3 times the keep above it, the words at or above it are listed, and the search runs over the listing -
or across the row when the listing misses the keep or overflows its room, so the sample sets only the cost.

The scores are the Triton path's bit for bit. A child's score is the fp32 sum of its centroid's products with the
query - exact, since an fp8 or a bf16 centroid times a bf16 query fits fp32 - added as Triton's layout adds them: each
thread's 16 bytes of the centroid in order, then the threads of a child by xor from the widest step down. A thread's
order is Triton's choice and is the pinned Triton's - 3.8 adds them as a tree of adjacent pairs instead, which the
parity tests catch. So the kept columns are the same: the greatest `beam_width` by score, a tie to the lower column,
in column order.

A shaped level's carried child adds z times its shape's spread, as the shaped scorer computes it: the centroid's sum
contracted - its second product first, the rest fused onto it - and the spread split over four lanes, a direction and
a basis word each, a neighbour's term fused in by xor. Only shapes of SHAPE_RANK directions are taken, since another
rank changes how the shaped scorer splits and adds them.
"""

import math
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from lsa.core.kernels.cutedsl import keep as cutedsl_keep
from lsa.core.kernels.triton.shape import basis_words, sign_query

# Threads of a beam step that scores children, and of every other launch - the more loads in flight, the better, until
# a scoring level's gathers crowd each other out. A scoring level whose shared memory passes SCORING_SHARED_BYTES
# falls from four CTAs an SM to three and takes the larger CTA too: over 8,192 children and 3,024 open node buckets it
# ran 1.77 ms at 256 threads and 1.56 at 512, and 1.52 and 1.55 over 2,524. The pick gains as much and never loses
SCORING_THREADS = 256
READING_THREADS = 512
SCORING_SHARED_BYTES = 49 * 1024
# A row already scored is only read, and twice the threads halve each one's run of it
# => 1,024 threads
SCORED_THREADS = 1024
MOST_WARPS = SCORED_THREADS // 32
# Bins of a radix pass, and the bins each lane of the warp that finds a pass's digit holds
BINS = 256
BINS_PER_LANE = BINS // 32
# The ordered word of -inf - a kept column at it keeps no children
NEG_INF_WORD = 0x007FFFFF
# The key of no bucket
EMPTY_KEY = -(2**63)
# Words of shared memory before the row - the histogram, each warp's two counts and the search's state, to 16 bytes
STATE_WORDS = 8
FIXED_WORDS = (BINS + 2 * MOST_WARPS + STATE_WORDS + 3) // 4 * 4
# Shared memory a CTA may take on Hopper - the fixed words, the row and the kept columns
SHARED_BYTES = 227 * 1024
SHARED_WORDS = SHARED_BYTES // 4 - FIXED_WORDS
# Scores a thread loads before it stores any
COPY_IN_FLIGHT = 16
# Words a row samples, at a fixed stride where they lie, the listing it expects above the sample's threshold and the
# most it holds, in the keep - a row narrower than SAMPLED_FROM times its keep, or than the sample, is searched whole.
# Every word of room past the row costs occupancy: a pick of 1,024 over 9,232 columns ran 2.05 ms with room for twice
# its keep and 1.72 with room for 1.5 times
SAMPLES = 4096
TARGET = 1.15
HOLD = 1.5
SAMPLED_FROM = 8
# Children a thread gathers before it adds any - enough loads in flight to cover a gather's latency
IN_FLIGHT = 8
# Bytes a thread of a child holds, as Triton's layout of the child scores holds them
THREAD_BYTES = 16
INDEX_DIM = 64
# Sign directions a shaped level's kernel takes, and the basis words of one
SHAPE_RANK = 2
SHAPE_WORDS = INDEX_DIM // 32
# Bits of the query levels a sign basis is scored against
QUERY_BITS = 8


class BeamStep:
    """One beam step, compiled once a centroid precision, node bucket size and thread count, its row sized at launch."""

    def __init__(
        self,
        element_bytes: int,
        node_bucket_size: int,
        threads: int,
        shaped: bool = False,
        scored: bool = False,
    ) -> None:
        """Fix the order the scores are added in, and the CTA's threads.

        Args:
            element_bytes: Bytes of a centroid's element - 1 for fp8, 2 for bf16.
            node_bucket_size: Children a node bucket holds.
            threads: Threads of a CTA, SCORING_THREADS or READING_THREADS.
            shaped: Whether a carried row's score adds its shape's spread.
            scored: Whether the whole row arrives scored in `open_scores`, its columns named by `carried`.
        """
        self.element_bytes = element_bytes
        self.shaped = shaped
        self.scored = scored
        self.dims_per_thread = THREAD_BYTES // element_bytes
        self.lanes = INDEX_DIM // self.dims_per_thread
        # The xor offsets from the widest down
        self.offsets = [self.lanes >> (i + 1) for i in range(self.lanes.bit_length() - 1)]
        self.node_bucket_size = node_bucket_size
        self.threads = threads
        self.warps = threads // 32
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads)

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        open_scores: cute.Tensor,
        open_children: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        open_count: cutlass.Int32,
        keep: cutlass.Int32,
        listing_room: cutlass.Int32,
        bounded: cutlass.Int32,
        rescale: cutlass.Float32,
        basis: cute.Tensor,
        variances: cute.Tensor,
        floor: cute.Tensor,
        planes: cute.Tensor,
        query_terms: cute.Tensor,
        deviations: cutlass.Float32,
        shared_bytes: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a query.

        Args:
            index_queries: Index queries [S, 64] bf16.
            open_scores: The open node buckets' scores [S, open] fp32.
            open_children: Their children [open, node_bucket_size] int32.
            carried: The rows the level scores after its open node buckets [S, C] int32, -1 for none.
            centroids: The closed node buckets' centroids [N, 64] the carried rows name, fp8 or bf16.
            closed_children: Their children [N, node_bucket_size] int32.
            out: The kept node buckets' children [S, keep * node_bucket_size] int32.
            lengths: Carried columns each query scores [S] int32, read when `bounded`.
            open_count: Open node buckets.
            keep: Columns a query keeps, the beam's width.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
            bounded: Nonzero for a query to score only its first `lengths` carried columns, 0 for all of them.
            rescale: One over what the centroids were multiplied by on storing.
            basis: The closed node buckets' sign words [N, SHAPE_RANK * 2] int32, read when shaped.
            variances: Their variances [N, SHAPE_RANK] fp32.
            floor: Their floors [N] fp32.
            planes: Each query's bit-planes [S, QUERY_BITS * 2] int32.
            query_terms: Each query's level step and total [S, 2] fp32.
            deviations: Standard deviations a shaped score adds, z.
            shared_bytes: Shared memory a CTA takes - FIXED_WORDS, the row and the kept columns.
            stream: The stream to launch on.
        """
        self.kernel(
            index_queries,
            open_scores,
            open_children,
            carried,
            centroids,
            closed_children,
            out,
            lengths,
            open_count,
            keep,
            listing_room,
            bounded,
            rescale,
            (basis, variances, floor, planes, query_terms, deviations),
        ).launch(
            grid=(index_queries.shape[0], 1, 1), block=[self.threads, 1, 1], smem=shared_bytes, stream=stream
        )

    @cute.jit
    def warp_sum(self, value: cutlass.Int32) -> cutlass.Int32:
        """Return the sum of a value over the warp, in every lane.

        Args:
            value: This lane's value.
        """
        for i in cutlass.range_constexpr(5):
            value = value + cute.arch.shuffle_sync_bfly(value, 16 >> i)
        return value

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        open_scores: cute.Tensor,
        open_children: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        open_count: cutlass.Int32,
        keep: cutlass.Int32,
        listing_room: cutlass.Int32,
        bounded: cutlass.Int32,
        rescale: cutlass.Float32,
        shapes: tuple,
    ) -> None:
        """Score one query's row into shared memory, find its k-th greatest score, and write the kept children.

        Args:
            index_queries: Index queries [S, 64] bf16.
            open_scores: The open node buckets' scores [S, open] fp32.
            open_children: Their children [open, node_bucket_size] int32.
            carried: The rows the level scores after its open node buckets [S, C] int32.
            centroids: The closed node buckets' centroids [N, 64].
            closed_children: Their children [N, node_bucket_size] int32.
            out: The kept node buckets' children [S, keep * node_bucket_size] int32.
            lengths: Carried columns each query scores [S] int32, read when `bounded`.
            open_count: Open node buckets.
            keep: Columns a query keeps.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
            bounded: Nonzero for a query to score only its first `lengths` carried columns, 0 for all of them.
            rescale: One over what the centroids were multiplied by on storing.
            shapes: The closed node buckets' basis, variances and floor, the queries' planes and terms, and z.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        carried_count = cutlass.Int32(carried.shape[1])
        if cutlass.const_expr(self.scored):
            # The scored row is read to its length, and no column is scored here
            carried_count = cutlass.Int32(0)
            if bounded != 0:
                open_count = cutlass.min(open_count, cutlass.max(lengths[row], 0))
        elif bounded != 0:
            carried_count = cutlass.min(carried_count, cutlass.max(lengths[row], 0))
        width = open_count + carried_count
        words, kept = self.keep_columns(
            index_queries,
            open_scores,
            carried,
            centroids,
            open_count,
            carried_count,
            keep,
            listing_room,
            rescale,
            shapes,
        )
        want = cutlass.min(width, keep)
        if cutlass.const_expr(self.scored):
            self.write_scored_children(row, want, kept, words, carried, closed_children, out)
        else:
            for slot in range(thread, want, self.threads):
                column = cutlass.Int32(kept[slot])
                self.write_children(
                    row, column, slot, words[column], open_count, carried, open_children, closed_children, out
                )
        for slot in range(want + thread, keep, self.threads):
            for c in cutlass.range_constexpr(self.node_bucket_size):
                out[row, slot * self.node_bucket_size + c] = cutlass.Int32(-1)

    @cute.jit
    def keep_columns(
        self,
        index_queries: cute.Tensor,
        open_scores: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        open_count: cutlass.Int32,
        carried_count: cutlass.Int32,
        keep: cutlass.Int32,
        listing_room: cutlass.Int32,
        rescale: cutlass.Float32,
        shapes: Any = None,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Score this CTA's query's row into shared memory and stage its kept columns in column order.

        Args:
            index_queries: Index queries [S, 64] bf16.
            open_scores: The open columns' scores [S, open] fp32.
            carried: The rows the level scores after its open columns [S, C] int32.
            centroids: The centroids [N, 64] the carried rows name.
            open_count: Open columns.
            carried_count: Carried columns this query scores, at most the carried rows.
            keep: Columns a query keeps.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
            rescale: One over what the centroids were multiplied by on storing.
            shapes: What a shaped level's spread reads - None for an unshaped one.

        Returns:
            The row's words [width] and its kept columns [keep] in shared memory, the lesser of the width and the keep
            of them staged.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = thread // 32
        width = open_count + carried_count

        # One block of shared memory sized at launch - the fixed words, the row's words, and the room past them
        counts = cute.arch.get_dyn_smem(cutlass.Int32, 16)
        histogram = cute.make_tensor(counts, cute.make_layout(BINS))
        warp_above = cute.make_tensor(counts + BINS, cute.make_layout(MOST_WARPS))
        warp_tied = cute.make_tensor(counts + BINS + MOST_WARPS, cute.make_layout(MOST_WARPS))
        shared = cute.make_tensor(counts + BINS + 2 * MOST_WARPS, cute.make_layout(STATE_WORDS))
        words = cute.recast_tensor(
            cute.make_tensor(counts + FIXED_WORDS, cute.make_layout(width)), cutlass.Uint32
        )
        # Past the row, the listing above the sample's threshold, then the kept columns
        spare_count = cutlass.max(listing_room, kept_words(keep))
        spare = cute.recast_tensor(
            cute.make_tensor(counts + FIXED_WORDS + width, cute.make_layout(spare_count)), cutlass.Uint32
        )
        # A column fits 16 bits, and half the room keeps a wide level at five CTAs an SM
        kept = cute.recast_tensor(
            cute.make_tensor(counts + FIXED_WORDS + width, cute.make_layout(kept_words(keep))), cutlass.Uint16
        )

        self.load_open(open_scores, row, open_count, words)
        if cutlass.const_expr(not self.scored):
            self.score_carried(
                index_queries, carried, centroids, row, open_count, carried_count, rescale, words, shapes
            )
        self.barrier.arrive_and_wait()

        # The want-th greatest word and the ties at it the keep still takes - searched among the words at or above a
        # sampled threshold when they hold the want and fit, else across the row
        want = cutlass.min(width, keep)
        listed = cutlass.Int32(-1)
        # A row narrower than the host sized the room for is searched whole
        if listing_room != 0 and width >= cutlass.max(SAMPLED_FROM * keep, SAMPLES):
            sample = cute.make_tensor(words.iterator, cute.make_layout(SAMPLES, stride=width // SAMPLES))
            rank = cutlass.Int32(cutlass.Float32(want * SAMPLES) * TARGET / cutlass.Float32(width)) + 1
            self.search(sample, SAMPLES, cutlass.min(rank, SAMPLES), 2, histogram, shared, warp, lane)
            listed = self.list_above(
                words, width, cutlass.Uint32(shared[0]), spare, listing_room, shared, lane
            )
        if listed >= want and listed <= listing_room:
            self.search(spare, listed, want, 4, histogram, shared, warp, lane)
        else:
            self.search(words, width, want, 4, histogram, shared, warp, lane)
        level = cutlass.Uint32(shared[0])
        ties = shared[1]

        # Each warp walks its own run of columns; one count of every run places them all
        run = ((width + self.warps - 1) // self.warps + 31) // 32 * 32
        first = warp * run
        last = cutlass.min(first + run, width)
        above_count = cutlass.Int32(0)
        tied_count = cutlass.Int32(0)
        for i in range(first + lane, last, 32):
            word = cutlass.Uint32(words[i])
            above_count = above_count + cutlass.Int32(word > level)
            tied_count = tied_count + cutlass.Int32(word == level)
        above_count = self.warp_sum(above_count)
        tied_count = self.warp_sum(tied_count)
        if lane == 0:
            warp_above[warp] = above_count
            warp_tied[warp] = tied_count
        self.barrier.arrive_and_wait()
        tied_before = cutlass.Int32(0)
        filled = cutlass.Int32(0)
        for w in cutlass.range_constexpr(self.warps):
            if w < warp:
                filled = (
                    filled + warp_above[w] + cutlass.min(cutlass.max(ties - tied_before, 0), warp_tied[w])
                )
                tied_before = tied_before + warp_tied[w]
        for start in range(first, last, 32):
            i = start + lane
            inside = i < last
            word = cutlass.Uint32(0)
            if inside:
                word = cutlass.Uint32(words[i])
            tied = inside and word == level
            tied_mask = cute.arch.vote_ballot_sync(tied)
            tie_rank = tied_before + cutlass.Int32(cute.arch.popc(tied_mask & cute.arch.lanemask_lt()))
            chosen = inside and (word > level or (tied and tie_rank < ties))
            chosen_mask = cute.arch.vote_ballot_sync(chosen)
            slot = filled + cutlass.Int32(cute.arch.popc(chosen_mask & cute.arch.lanemask_lt()))
            if chosen:
                kept[slot] = cutlass.Uint16(i)
            tied_before = tied_before + cutlass.Int32(cute.arch.popc(tied_mask))
            filled = filled + cutlass.Int32(cute.arch.popc(chosen_mask))
        self.barrier.arrive_and_wait()
        return words, kept

    @cute.jit
    def load_open(
        self, open_scores: cute.Tensor, row: cutlass.Int32, open_count: cutlass.Int32, words: cute.Tensor
    ) -> None:
        """Copy a query's open scores into its row as words, every thread's loads in flight before it stores any.

        Args:
            open_scores: The open columns' scores [S, open] fp32.
            row: The query.
            open_count: Open columns.
            words: The row's words in shared memory.
        """
        thread, _, _ = cute.arch.thread_idx()
        loaded_scores = cute.make_rmem_tensor(COPY_IN_FLIGHT, cutlass.Float32)
        for base in range(0, open_count, self.threads * COPY_IN_FLIGHT):
            for f in cutlass.range_constexpr(COPY_IN_FLIGHT):
                i = base + f * self.threads + thread
                loaded_scores[f] = cutlass.Float32(0)
                if i < open_count:
                    loaded_scores[f] = open_scores[row, i]
            for f in cutlass.range_constexpr(COPY_IN_FLIGHT):
                i = base + f * self.threads + thread
                if i < open_count:
                    words[i] = self.word_of(loaded_scores[f])

    @cute.jit
    def score_carried(
        self,
        index_queries: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        row: cutlass.Int32,
        open_count: cutlass.Int32,
        carried_count: cutlass.Int32,
        rescale: cutlass.Float32,
        words: cute.Tensor,
        shapes: Any,
    ) -> None:
        """Score a query's carried rows into its row as words, after its open columns, -inf under a -1.

        Args:
            index_queries: Index queries [S, 64] bf16.
            carried: The rows the level scores after its open columns [S, C] int32.
            centroids: The centroids [N, 64] the carried rows name.
            row: The query.
            open_count: Open columns.
            carried_count: Carried columns to score, at most the carried rows.
            rescale: One over what the centroids were multiplied by on storing.
            words: The row's words in shared memory.
            shapes: What a shaped level's spread reads - None for an unshaped one.
        """
        thread, _, _ = cute.arch.thread_idx()
        # A child is `lanes` threads of 16 bytes each, added as Triton adds them
        member = thread % self.lanes
        group = thread // self.lanes
        groups = self.threads // self.lanes
        query = cute.make_rmem_tensor(self.dims_per_thread, cutlass.Float32)
        for e in cutlass.range_constexpr(self.dims_per_thread):
            query[e] = index_queries[row, member * self.dims_per_thread + e].to(cutlass.Float32)
        loaded = cute.make_rmem_tensor(
            cute.make_layout((IN_FLIGHT, self.dims_per_thread), stride=(self.dims_per_thread, 1)),
            centroids.element_type,
        )
        kids = cute.make_rmem_tensor(IN_FLIGHT, cutlass.Int32)
        products = cute.make_rmem_tensor(self.dims_per_thread, cutlass.Float32)
        # A shaped child's spread is four lanes', one a direction and basis word, as the shaped scorer splits it
        query_planes = cute.make_rmem_tensor(QUERY_BITS, cutlass.Int32)
        if cutlass.const_expr(self.shaped):
            planes = shapes[3]
            for p in cutlass.range_constexpr(QUERY_BITS):
                query_planes[p] = planes[row, p * SHAPE_WORDS + member % SHAPE_WORDS]
        for base in range(0, carried_count, groups * IN_FLIGHT):
            for f in cutlass.range_constexpr(IN_FLIGHT):
                j = base + f * groups + group
                kids[f] = cutlass.Int32(-1)
                if j < carried_count:
                    kids[f] = carried[row, j]
            for f in cutlass.range_constexpr(IN_FLIGHT):
                cute.autovec_copy(
                    cute.local_tile(centroids, (1, self.dims_per_thread), (cutlass.max(kids[f], 0), member))[
                        0, None
                    ],
                    loaded[f, None],
                )
            for f in cutlass.range_constexpr(IN_FLIGHT):
                j = base + f * groups + group
                total = cutlass.Float32(0)
                if cutlass.const_expr(self.shaped):
                    # The shaped scorer's sum is contracted - the second product first, the rest fused onto it
                    total = loaded[f, 1].to(cutlass.Float32) * query[1]
                    total = cute.math.fma(loaded[f, 0].to(cutlass.Float32), query[0], total)
                    for e in cutlass.range_constexpr(self.dims_per_thread - 2):
                        total = cute.math.fma(loaded[f, e + 2].to(cutlass.Float32), query[e + 2], total)
                else:
                    for e in cutlass.range_constexpr(self.dims_per_thread):
                        products[e] = loaded[f, e].to(cutlass.Float32) * query[e]
                    total = products[0]
                    for e in cutlass.range_constexpr(self.dims_per_thread - 1):
                        total = total + products[e + 1]
                for offset in cutlass.range_constexpr(len(self.offsets)):
                    total = total + cute.arch.shuffle_sync_bfly(total, self.offsets[offset])
                score = total * rescale
                if cutlass.const_expr(self.shaped):
                    score = self.shaped_score(
                        total, rescale, cutlass.max(kids[f], 0), row, member, query_planes, shapes
                    )
                if member == 0 and j < carried_count:
                    if kids[f] < 0:
                        score = -cutlass.Float32.inf
                    words[open_count + j] = self.word_of(score)

    @cute.jit
    def shaped_score(
        self,
        total: cutlass.Float32,
        rescale: cutlass.Float32,
        node: cutlass.Int32,
        row: cutlass.Int32,
        member: cutlass.Int32,
        query_planes: cute.Tensor,
        shapes: Any,
    ) -> cutlass.Float32:
        """Return a carried row's shaped score in its first lane - its centroid's product plus z times its spread.

        The child's first four lanes each take a direction and a basis word and fold in a neighbour's as the shaped
        scorer does, fused where it fuses, so the first lane's score is the same bit for bit.

        Args:
            total: The centroid's product with the query, unscaled.
            rescale: One over what the centroids were multiplied by on storing.
            node: The carried row.
            row: The query.
            member: This thread's lane within the child.
            query_planes: The query's bit-planes of this lane's basis word [QUERY_BITS] int32.
            shapes: The basis, variances and floor, the queries' planes and terms, and z.
        """
        basis, variances, floor, _, query_terms, deviations = shapes
        word_index = member % SHAPE_WORDS
        direction = (member // SHAPE_WORDS) % SHAPE_RANK
        step = query_terms[row, 0]
        word = basis[node, direction * SHAPE_WORDS + word_index]
        level = cute.arch.popc(word) * cutlass.Int32(1 - (1 << (QUERY_BITS - 1)))
        for p in cutlass.range_constexpr(QUERY_BITS):
            level = level + (cute.arch.popc(word & query_planes[p]) << p)
        levels = cutlass.Float32(level)
        chosen = cute.math.fma(step, levels, cute.arch.shuffle_sync_bfly(step * levels, 1))
        projection = cute.math.fma(chosen, cutlass.Float32(2.0), -query_terms[row, 1]) * cute.math.rsqrt(
            cutlass.Float32(INDEX_DIM), fastmath=True
        )
        squared = projection * projection
        spread_variance = variances[node, direction]
        along = cute.math.fma(
            squared, spread_variance, cute.arch.shuffle_sync_bfly(squared * spread_variance, SHAPE_WORDS)
        )
        kept = cute.math.fma(projection, projection, cute.arch.shuffle_sync_bfly(squared, SHAPE_WORDS))
        variance = cute.math.fma(cutlass.Float32(1.0) - kept, floor[node], along)
        spread = cute.math.sqrt(cutlass.max(variance, cutlass.Float32(0.0)), fastmath=True)
        if cutlass.const_expr(self.element_bytes == 1):
            return cute.math.fma(rescale, total, deviations * spread)
        return cute.math.fma(deviations, spread, total * rescale)

    @cute.jit
    def word_of(self, score: cutlass.Float32) -> cutlass.Uint32:
        """Return a score as the unsigned word of the same order.

        Args:
            score: The fp32 score.
        """
        cell = cute.make_rmem_tensor(1, cutlass.Float32)
        cell[0] = score
        bits = cute.recast_tensor(cell, cutlass.Uint32)[0]
        word = bits | cutlass.Uint32(0x80000000)
        if (bits >> 31) != 0:
            word = ~bits
        return word

    @cute.jit
    def search(
        self,
        values: cute.Tensor,
        count: cutlass.Int32,
        rank: cutlass.Int32,
        passes: cutlass.Constexpr,
        histogram: cute.Tensor,
        shared: cute.Tensor,
        warp: cutlass.Int32,
        lane: cutlass.Int32,
    ) -> None:
        """Find the `rank`-th greatest value's top `passes` bytes, and the rank left at them, into the search's state.

        Args:
            values: Words [count] in shared memory.
            count: Words held.
            rank: The rank sought, from 1, at most the count.
            passes: Bytes settled from the top.
            histogram: Shared memory for a byte's counts.
            shared: The search's state.
            warp: This thread's warp.
            lane: This thread's lane.
        """
        thread, _, _ = cute.arch.thread_idx()
        if thread == 0:
            shared[0] = cutlass.Int32(0)
            shared[1] = rank
        for p in cutlass.range_constexpr(passes):
            shift = 24 - 8 * p
            histogram[thread % BINS] = cutlass.Int32(0)
            self.barrier.arrive_and_wait()
            prefix = cutlass.Uint32(shared[0])
            high = cutlass.Uint32(0)
            if cutlass.const_expr(p > 0):
                high = cutlass.Uint32(0xFFFFFFFF) << (shift + 8)
            for i in range(thread, count, self.threads):
                word = cutlass.Uint32(values[i])
                if (word & high) == (prefix & high):
                    cute.arch.atomic_add(
                        (histogram.iterator + cutlass.Int32((word >> shift) & 255)).llvm_ptr,
                        cutlass.Int32(1),
                        scope="cta",
                    )
            self.barrier.arrive_and_wait()
            if warp == 0:
                self.find_digit(lane, histogram, shared, prefix, shift)
            self.barrier.arrive_and_wait()
            if thread == 0:
                shared[0] = shared[2]
                shared[1] = shared[3]
            self.barrier.arrive_and_wait()

    @cute.jit
    def list_above(
        self,
        words: cute.Tensor,
        width: cutlass.Int32,
        threshold: cutlass.Uint32,
        listing: cute.Tensor,
        capacity: cutlass.Int32,
        shared: cute.Tensor,
        lane: cutlass.Int32,
    ) -> cutlass.Int32:
        """List the words at or above a threshold in no order, and return their count, those past the capacity unlisted.

        Args:
            words: The row's words [width].
            width: Columns the row holds.
            threshold: The least word listed.
            listing: Where the words go [capacity].
            capacity: Words the listing holds.
            shared: The search's state - its fifth word counts the listing.
            lane: This thread's lane.
        """
        thread, _, _ = cute.arch.thread_idx()
        if thread == 0:
            shared[4] = cutlass.Int32(0)
        self.barrier.arrive_and_wait()
        for start in range(thread - lane, width, self.threads):
            i = start + lane
            word = cutlass.Uint32(0)
            if i < width:
                word = cutlass.Uint32(words[i])
            above = i < width and word >= threshold
            mask = cute.arch.vote_ballot_sync(above)
            base = cutlass.Int32(0)
            if lane == 0 and mask != 0:
                base = cute.arch.atomic_add(
                    (shared.iterator + 4).llvm_ptr, cutlass.Int32(cute.arch.popc(mask)), scope="cta"
                )
            base = cute.arch.shuffle_sync(base, 0)
            place = base + cutlass.Int32(cute.arch.popc(mask & cute.arch.lanemask_lt()))
            if above and place < capacity:
                listing[place] = word
        self.barrier.arrive_and_wait()
        listed = shared[4]
        self.barrier.arrive_and_wait()
        return listed

    @cute.jit
    def find_digit(
        self,
        lane: cutlass.Int32,
        histogram: cute.Tensor,
        shared: cute.Tensor,
        prefix: cutlass.Uint32,
        shift: int,
    ) -> None:
        """Find the pass's digit of the want-th greatest word in one warp, and how many of the want are below it.

        Args:
            lane: This lane.
            histogram: The pass's counts a digit.
            shared: The search's state - the prefix and the want left in, the next prefix and want out.
            prefix: The digits found so far.
            shift: The pass's bit offset.
        """
        remaining = shared[1]
        # Lane l holds the bins 8l to 8l + 7 counted from the top
        lane_total = cutlass.Int32(0)
        for b in cutlass.range_constexpr(BINS_PER_LANE):
            lane_total = lane_total + histogram[BINS - 1 - (lane * BINS_PER_LANE + b)]
        before = lane_total
        for i in cutlass.range_constexpr(5):
            # A shuffle up clamps at lane 0, which cutlass-dsl 4.6.2 does not default to
            other = cute.arch.shuffle_sync_up(before, 1 << i, mask_and_clamp=0)
            if lane >= (1 << i):
                before = before + other
        before = before - lane_total
        if before < remaining and before + lane_total >= remaining:
            above = before
            digit = cutlass.Int32(0)
            found = cutlass.Int32(0)
            for b in cutlass.range_constexpr(BINS_PER_LANE):
                bin_index = BINS - 1 - (lane * BINS_PER_LANE + b)
                count = histogram[bin_index]
                if found == 0 and above + count >= remaining:
                    digit = bin_index
                    found = 1
                if found == 0:
                    above = above + count
            shared[2] = cutlass.Int32(prefix | (cutlass.Uint32(digit) << shift))
            shared[3] = remaining - above

    @cute.jit
    def write_scored_children(
        self,
        row: cutlass.Int32,
        want: cutlass.Int32,
        kept: cute.Tensor,
        words: cute.Tensor,
        carried: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
    ) -> None:
        """Write the children of a scored row's kept columns - none for a column at -inf or with no row.

        Args:
            row: The query.
            want: Columns kept.
            kept: The kept columns in column order [keep] in shared memory.
            words: The row's words in shared memory.
            carried: The rows each column names [S, W] int32.
            closed_children: The children of the rows [N, node_bucket_size] int32.
            out: The kept columns' children [S, keep * node_bucket_size] int32.
        """
        thread, _, _ = cute.arch.thread_idx()
        children = cute.make_rmem_tensor(self.node_bucket_size, cutlass.Int32)
        for slot in range(thread, want, self.threads):
            column = cutlass.Int32(kept[slot])
            node = cutlass.Int32(-1)
            if cutlass.Uint32(words[column]) > cutlass.Uint32(NEG_INF_WORD):
                node = carried[row, column]
            # The loads are all in flight before the first store
            for c in cutlass.range_constexpr(self.node_bucket_size):
                children[c] = cutlass.Int32(-1)
                if node >= 0:
                    children[c] = closed_children[node, c]
            for c in cutlass.range_constexpr(self.node_bucket_size):
                out[row, slot * self.node_bucket_size + c] = children[c]

    @cute.jit
    def write_children(
        self,
        row: cutlass.Int32,
        column: cutlass.Int32,
        slot: cutlass.Int32,
        word: cutlass.Uint32,
        open_count: cutlass.Int32,
        carried: cute.Tensor,
        open_children: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
    ) -> None:
        """Write a kept column's children at its slot - none for a column at -inf or a carried row that is none.

        Args:
            row: The query.
            column: The kept column.
            slot: Its place among the kept, in column order.
            word: Its score's ordered word.
            open_count: Open node buckets.
            carried: The carried rows [S, C] int32.
            open_children: The open node buckets' children.
            closed_children: The closed node buckets' children.
            out: The kept node buckets' children.
        """
        valid = word > cutlass.Uint32(NEG_INF_WORD)
        node = cutlass.Int32(-1)
        if cutlass.const_expr(self.scored):
            node = carried[row, column]
            valid = valid and node >= 0
        elif column >= open_count:
            node = carried[row, column - open_count]
            valid = valid and node >= 0
        for c in cutlass.range_constexpr(self.node_bucket_size):
            kid = cutlass.Int32(-1)
            if valid:
                if cutlass.const_expr(self.scored):
                    kid = closed_children[node, c]
                else:
                    if column < open_count:
                        kid = open_children[column, c]
                    else:
                        kid = closed_children[node, c]
            out[row, slot * self.node_bucket_size + c] = kid


class TopPPick(BeamStep):
    """The last level - the buckets no node bucket covers and the last beam's children scored, the best `top_p` named.

    A kept column is written as the key the expansion by threshold reads: its score's ordered word over its cache row
    reversed, the empty key for a column scored -inf or with no row.
    """

    def __init__(self, element_bytes: int, threads: int) -> None:
        """Fix the order the scores are added in, and the CTA's threads.

        Args:
            element_bytes: Bytes of a centroid's element - 1 for fp8, 2 for bf16.
            threads: Threads of a CTA, SCORING_THREADS or READING_THREADS.
        """
        super().__init__(element_bytes, 1, threads)

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        open_scores: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        pages: cute.Tensor,
        out: cute.Tensor,
        open_count: cutlass.Int32,
        covered_buckets: cutlass.Int32,
        page_rows: cutlass.Int32,
        listing_room: cutlass.Int32,
        shared_bytes: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a query.

        Args:
            index_queries: Index queries [S, 64] bf16.
            open_scores: The uncovered buckets' scores [S, open] fp32, -inf past a query's visible count.
            carried: The cache rows the last beam kept [S, C] int32, -1 for none.
            centroids: The bucket centroids [rows, 64] the carried rows name.
            pages: The request's pages [max_pages] int32.
            out: Each query's picked keys [S, W] int64, W the lesser of its width and the top-p.
            open_count: Uncovered buckets scored, from logical bucket `covered_buckets`.
            covered_buckets: Buckets under a node bucket.
            page_rows: Cache rows a page holds.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
            shared_bytes: Shared memory a CTA takes - FIXED_WORDS, the row and the kept columns.
            stream: The stream to launch on.
        """
        self.kernel(
            index_queries,
            open_scores,
            carried,
            centroids,
            pages,
            out,
            open_count,
            covered_buckets,
            page_rows,
            listing_room,
        ).launch(
            grid=(index_queries.shape[0], 1, 1), block=[self.threads, 1, 1], smem=shared_bytes, stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        open_scores: cute.Tensor,
        carried: cute.Tensor,
        centroids: cute.Tensor,
        pages: cute.Tensor,
        out: cute.Tensor,
        open_count: cutlass.Int32,
        covered_buckets: cutlass.Int32,
        page_rows: cutlass.Int32,
        listing_room: cutlass.Int32,
    ) -> None:
        """Score one query's row into shared memory, find its top-p-th greatest score, and write the kept keys.

        Args:
            index_queries: Index queries [S, 64] bf16.
            open_scores: The uncovered buckets' scores [S, open] fp32.
            carried: The cache rows the last beam kept [S, C] int32.
            centroids: The bucket centroids [rows, 64].
            pages: The request's pages [max_pages] int32.
            out: Each query's picked keys [S, W] int64.
            open_count: Uncovered buckets scored.
            covered_buckets: Buckets under a node bucket.
            page_rows: Cache rows a page holds.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        keep = out.shape[1]
        words, kept = self.keep_columns(
            index_queries,
            open_scores,
            carried,
            centroids,
            open_count,
            carried.shape[1],
            keep,
            listing_room,
            cutlass.Float32(1.0),
        )
        for slot in range(thread, keep, self.threads):
            column = cutlass.Int32(kept[slot])
            word = cutlass.Uint32(words[column])
            valid = word > cutlass.Uint32(NEG_INF_WORD)
            cache_row = cutlass.Int32(-1)
            if column < open_count:
                logical = covered_buckets + column
                page = pages[logical // page_rows]
                cache_row = page * page_rows + logical % page_rows
                valid = valid and page >= 0
            else:
                cache_row = carried[row, column - open_count]
                valid = valid and cache_row >= 0
            key = cutlass.Int64(EMPTY_KEY)
            if valid:
                ordered = cutlass.Int32(word ^ cutlass.Uint32(0x80000000))
                key = (cutlass.Int64(ordered) << 32) | cutlass.Int64(0x7FFFFFFF - cache_row)
            out[row, slot] = key


class ScoredPick(BeamStep):
    """The pick over a row already scored - the best `top_p` columns written as the keys the expansion reads.

    A column's key is its score's ordered word over its cache row reversed, the empty key for a column scored -inf or
    with no row. A query keeps at most as many columns as its length, and its other slots hold the empty key.
    """

    def __init__(self, element_bytes: int, threads: int) -> None:
        """Fix the CTA's threads.

        Args:
            element_bytes: Bytes of a centroid's element - 1 for fp8, 2 for bf16.
            threads: Threads of a CTA.
        """
        super().__init__(element_bytes, 1, threads, scored=True)

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        listed: cute.Tensor,
        listing_room: cutlass.Int32,
        shared_bytes: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a query.

        Args:
            scores: The row's scores [S, W] fp32, read to each query's length.
            candidates: The cache row of each column [S, W] int32, -1 for none.
            lengths: Columns each query holds [S] int32.
            out: Each query's picked keys [S, keep] int64.
            listed: Keys each query lists [S] int32 - the keep, written here.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
            shared_bytes: Shared memory a CTA takes - FIXED_WORDS, the row and the kept columns.
            stream: The stream to launch on.
        """
        self.kernel(scores, candidates, lengths, out, listed, listing_room).launch(
            grid=(scores.shape[0], 1, 1), block=[self.threads, 1, 1], smem=shared_bytes, stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        listed: cute.Tensor,
        listing_room: cutlass.Int32,
    ) -> None:
        """Find the top-p-th greatest score of one query's row and write the kept keys.

        Args:
            scores: The row's scores [S, W] fp32.
            candidates: The cache row of each column [S, W] int32.
            lengths: Columns each query holds [S] int32.
            out: Each query's picked keys [S, keep] int64.
            listed: Keys each query lists [S] int32.
            listing_room: Words the listing above a sampled threshold holds, or 0 for a row searched whole.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        keep = out.shape[1]
        count = cutlass.min(cutlass.Int32(scores.shape[1]), cutlass.max(lengths[row], 0))
        # Nothing reads the queries or the centroids of a scored row
        words, kept = self.keep_columns(
            scores,
            scores,
            candidates,
            scores,
            count,
            cutlass.Int32(0),
            keep,
            listing_room,
            cutlass.Float32(1.0),
        )
        want = cutlass.min(count, keep)
        for slot in range(thread, keep, self.threads):
            key = cutlass.Int64(EMPTY_KEY)
            if slot < want:
                column = cutlass.Int32(kept[slot])
                word = cutlass.Uint32(words[column])
                cache_row = candidates[row, column]
                if word > cutlass.Uint32(NEG_INF_WORD) and cache_row >= 0:
                    ordered = cutlass.Int32(word ^ cutlass.Uint32(0x80000000))
                    key = (cutlass.Int64(ordered) << 32) | cutlass.Int64(0x7FFFFFFF - cache_row)
            out[row, slot] = key
        if thread == 0:
            listed[row] = keep


COMPILED: dict[tuple[int, int, int, bool, bool], Any] = {}
COMPILED_PICKS: dict[tuple[int, int], Any] = {}
COMPILED_SCORED_PICKS: dict[int, Any] = {}


def room(width: int, keep: int) -> tuple[int, int]:
    """Return the words a row's listing above its sampled threshold holds, 0 for a row searched whole, and the words
    past the row the CTA takes.

    Args:
        width: Columns a query scores.
        keep: Columns a query keeps.
    """
    if width < max(SAMPLED_FROM * keep, SAMPLES):
        return 0, kept_words(keep)
    listing_room = math.ceil(HOLD * keep)
    return listing_room, max(listing_room, kept_words(keep))


def threads_for(carried_count: int, shared_bytes: int) -> int:
    """Return a beam step's threads - the scoring threads for a level that scores children within its share of
    shared memory, the reading threads otherwise.

    Args:
        carried_count: Rows a query scores.
        shared_bytes: Shared memory a CTA takes.
    """
    return SCORING_THREADS if carried_count and shared_bytes <= SCORING_SHARED_BYTES else READING_THREADS


def kept_words(keep: int) -> int:
    """Return the words of shared memory a row's kept columns take, two to a word.

    Args:
        keep: Columns a query keeps.
    """
    return (keep + 1) // 2


def supports(
    index_queries: torch.Tensor,
    centroids: torch.Tensor,
    carried: torch.Tensor,
    open_count: int,
    keep: int,
    shape_rank: int | None = None,
) -> bool:
    """Tell whether the CuTeDSL beam step takes these rows.

    It takes 64 index dims, bf16 queries, fp8 or bf16 centroids, shapes of SHAPE_RANK directions if any, and a row
    and its kept columns that fit a CTA's shared memory.

    Args:
        index_queries: Index queries [S, index_dim].
        centroids: The closed node buckets' centroids [N, index_dim].
        carried: The carried rows [S, C].
        open_count: Open node buckets.
        keep: Columns a query keeps.
        shape_rank: Directions of the closed node buckets' shapes, or None when they are unshaped.
    """
    return (
        shape_rank in (None, SHAPE_RANK)
        and open_count + carried.shape[1] + room(open_count + carried.shape[1], keep)[1] <= SHARED_WORDS
        and index_queries.shape[1] == INDEX_DIM
        and index_queries.dtype == torch.bfloat16
        and centroids.dtype in (torch.float8_e4m3fn, torch.bfloat16)
        and centroids.is_contiguous()
        and centroids.data_ptr() % 16 == 0
        and carried.stride(1) == 1
    )


def compiled(
    element_bytes: int, node_bucket_size: int, threads: int, shaped: bool, scored: bool = False
) -> Any:
    """Return the beam step compiled once a precision and node bucket size, taking torch tensors as they are.

    Args:
        element_bytes: Bytes a centroid entry takes - 1 for fp8, 2 for bf16.
        node_bucket_size: Children a node bucket holds.
        threads: Threads of a CTA.
        shaped: Whether a carried row's score adds its shape's spread.
        scored: Whether the whole row arrives scored.
    """
    key = (element_bytes, node_bucket_size, threads, shaped, scored)
    if key in COMPILED:
        return COMPILED[key]
    rows, open_count, carried_count, centroid_count, closed_count, width, shape_count, open_rows = (
        cute.sym_int32() for _ in range(8)
    )
    query_stride, score_stride, open_stride, carried_stride, closed_stride, out_stride = (
        cute.sym_int64() for _ in range(6)
    )
    centroid_type = cutlass.Float8E4M3FN if element_bytes == 1 else cutlass.BFloat16
    arguments = (
        make_fake_tensor(cutlass.BFloat16, (rows, INDEX_DIM), stride=(query_stride, 1), assumed_align=16),
        make_fake_tensor(cutlass.Float32, (rows, open_count), stride=(score_stride, 1), assumed_align=4),
        make_fake_tensor(
            cutlass.Int32, (open_rows, node_bucket_size), stride=(open_stride, 1), assumed_align=4
        ),
        make_fake_tensor(cutlass.Int32, (rows, carried_count), stride=(carried_stride, 1), assumed_align=4),
        make_fake_compact_tensor(
            centroid_type, (centroid_count, INDEX_DIM), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_tensor(
            cutlass.Int32, (closed_count, node_bucket_size), stride=(closed_stride, 1), assumed_align=4
        ),
        make_fake_tensor(cutlass.Int32, (rows, width), stride=(out_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Float32(1.0),
        make_fake_compact_tensor(cutlass.Int32, (shape_count, SHAPE_RANK * SHAPE_WORDS), stride_order=(1, 0)),
        make_fake_compact_tensor(cutlass.Float32, (shape_count, SHAPE_RANK), stride_order=(1, 0)),
        make_fake_compact_tensor(cutlass.Float32, (shape_count,)),
        make_fake_compact_tensor(cutlass.Int32, (rows, QUERY_BITS * SHAPE_WORDS), stride_order=(1, 0)),
        make_fake_compact_tensor(cutlass.Float32, (rows, 2), stride_order=(1, 0)),
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kernel = BeamStep(element_bytes, node_bucket_size, threads, shaped, scored)
    COMPILED[key] = cute.compile(kernel, *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


def beam_step(
    index_queries: torch.Tensor,
    open_scores: torch.Tensor,
    open_children: torch.Tensor,
    carried: torch.Tensor,
    centroids: torch.Tensor,
    closed_children: torch.Tensor,
    keep: int,
    centroid_scale: float,
    shapes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep each query's best `keep` of its open node buckets and carried children, and name their children.

    Args:
        index_queries: Index queries [S, 64] bf16.
        open_scores: The open node buckets' scores [S, open] fp32.
        open_children: Their children [open, node_bucket_size] int32.
        carried: The children the level above kept [S, C] int32, -1 for none.
        centroids: The closed node buckets' centroids [N, 64] the carried rows name - bf16, or fp8 holding
            `centroid_scale` times the row.
        closed_children: Their children [N, node_bucket_size] int32.
        keep: Columns a query keeps, the beam's width.
        centroid_scale: What fp8 centroids were multiplied by on storing; 1 for bf16.
        shapes: The closed node buckets' packed sign basis [N, SHAPE_RANK, 8] uint8, variances [N, SHAPE_RANK] fp32
            and floors [N] fp32, and the deviations their scores add - None when they are unshaped.
        lengths: Carried columns each query scores [S] int32 - the first of its carried rows, the rest ignored - or
            None for all of them. A query keeps at most as many columns as it scores, and its other slots hold no children.

    Returns:
        The kept node buckets' children [S, keep * node_bucket_size] int32 - a kept bucket's side by side in column
        order, -1 where a query kept nothing.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    open_count = open_scores.shape[1]
    shape_rank = None if shapes is None else shapes[1].shape[1]
    if not supports(index_queries, centroids, carried, open_count, keep, shape_rank):
        raise ValueError(
            "the CuTeDSL beam step takes bf16 index queries of 64 dims over contiguous fp8 or bf16 centroids, shapes "
            f"of {SHAPE_RANK} directions, and a row and its kept within {SHARED_WORDS} words - got "
            f"{index_queries.dtype} {tuple(index_queries.shape)}, {centroids.dtype} {tuple(centroids.shape)}, shapes "
            f"of {shape_rank} directions and {open_count} + {carried.shape[1]} columns"
        )
    rows = index_queries.shape[0]
    node_bucket_size = closed_children.shape[1]
    width = open_count + carried.shape[1]
    listing_room, spare = room(width, keep)
    out = torch.empty(rows, keep * node_bucket_size, device=index_queries.device, dtype=torch.int32)
    if rows == 0:
        return out
    if open_count == 0:
        # Nothing reads them, but CuTe wants a tensor
        open_scores = torch.empty(rows, 1, device=index_queries.device)
        open_children = torch.empty(1, node_bucket_size, device=index_queries.device, dtype=torch.int32)
    threads = threads_for(carried.shape[1], (FIXED_WORDS + width + spare) * 4)
    basis, variances, floor, planes, query_terms, deviations = shape_arguments(index_queries, shapes)
    compiled(centroids.element_size(), node_bucket_size, threads, shapes is not None)(
        index_queries.contiguous(),
        open_scores,
        open_children.contiguous(),
        carried,
        centroids,
        closed_children.contiguous(),
        out,
        torch.empty(rows, device=out.device, dtype=torch.int32) if lengths is None else lengths,
        open_count,
        keep,
        listing_room,
        int(lengths is not None),
        1.0 / centroid_scale,
        basis,
        variances,
        floor,
        planes,
        query_terms,
        deviations,
        (FIXED_WORDS + width + spare) * 4,
    )
    return out


def keep_scored(
    index_queries: torch.Tensor,
    scores: torch.Tensor,
    candidates: torch.Tensor,
    centroids: torch.Tensor,
    closed_children: torch.Tensor,
    keep: int,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Keep each query's best `keep` columns of a row already scored, and name their children.

    The scores are the Triton scorer's, so the kept columns are the Triton keep's: the greatest by score, a tie to the
    lower column, in column order.

    Args:
        index_queries: Index queries [S, 64] bf16, which nothing reads.
        scores: The row's scores [S, W] fp32, read to each query's length.
        candidates: The rows of the level's pool each column names [S, W] int32, -1 for none.
        centroids: The pool's centroids [N, 64], which nothing reads - bf16, or fp8.
        closed_children: The pool's children [N, node_bucket_size] int32.
        keep: Columns a query keeps, the beam's width.
        lengths: Columns each query holds [S] int32.

    Returns:
        The kept columns' children [S, keep * node_bucket_size] int32 in column order, -1 where a query kept nothing.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    if not supports(index_queries, centroids, candidates, 0, keep) or scores.stride(1) != 1:
        raise ValueError(
            "the CuTeDSL scored keep takes bf16 index queries of 64 dims over contiguous fp8 or bf16 centroids and a "
            f"row and its kept within {SHARED_WORDS} words - got {index_queries.dtype} {tuple(index_queries.shape)}, "
            f"{centroids.dtype} {tuple(centroids.shape)} and {candidates.shape[1]} columns"
        )
    rows, width = scores.shape
    node_bucket_size = closed_children.shape[1]
    listing_room, spare = room(width, keep)
    out = torch.empty(rows, keep * node_bucket_size, device=scores.device, dtype=torch.int32)
    if rows == 0:
        return out
    basis, variances, floor, planes, query_terms, deviations = shape_arguments(index_queries, None)
    shared_bytes = (FIXED_WORDS + width + spare) * 4
    compiled(centroids.element_size(), node_bucket_size, SCORED_THREADS, False, True)(
        index_queries.contiguous(),
        scores,
        closed_children[:0],
        candidates,
        centroids,
        closed_children.contiguous(),
        out,
        lengths,
        width,
        keep,
        listing_room,
        1,
        1.0,
        basis,
        variances,
        floor,
        planes,
        query_terms,
        deviations,
        shared_bytes,
    )
    return out


def compiled_scored_pick() -> Any:
    """Return the pick over scored rows compiled once, taking torch tensors as they are.

    Returns:
        The compiled pick.
    """
    if 0 in COMPILED_SCORED_PICKS:
        return COMPILED_SCORED_PICKS[0]
    rows, width, keep = (cute.sym_int32() for _ in range(3))
    score_stride, candidate_stride, out_stride = (cute.sym_int64() for _ in range(3))
    arguments = (
        make_fake_tensor(cutlass.Float32, (rows, width), stride=(score_stride, 1), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (rows, width), stride=(candidate_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        make_fake_tensor(cutlass.Int64, (rows, keep), stride=(out_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED_SCORED_PICKS[0] = cute.compile(
        ScoredPick(1, SCORED_THREADS), *arguments, options="--enable-tvm-ffi"
    )
    return COMPILED_SCORED_PICKS[0]


def supports_scored_pick(width: int, top_p: int) -> bool:
    """Tell whether the CuTeDSL pick takes scored rows of this width - a row and its kept fit a CTA's shared memory.

    Args:
        width: Columns a query's row holds.
        top_p: Buckets a query picks.

    Returns:
        Whether the pick takes the rows.
    """
    return width + room(width, min(top_p, width))[1] <= SHARED_WORDS


def pick_scored(
    scores: torch.Tensor, candidates: torch.Tensor, lengths: torch.Tensor, top_p: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each query's best `top_p` columns of a scored row, keyed by their cache rows.

    Args:
        scores: The row's scores [S, W] fp32, read to each query's length.
        candidates: The cache row of each column [S, W] int32, -1 for none.
        lengths: Columns each query holds [S] int32.
        top_p: Buckets a query picks.

    Returns:
        The picked scores packed with their cache rows [S, min(W, top_p)] int64 in column order, the empty key for a
        column scored -inf, with no row, or past the query's length, and the keys each query lists [S] int32 - all
        of them.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    rows, width = scores.shape
    keep = min(top_p, width)
    if not supports_scored_pick(width, top_p) or scores.stride(1) != 1 or candidates.stride(1) != 1:
        raise ValueError(
            f"the CuTeDSL scored pick takes a row and its kept within {SHARED_WORDS} words of contiguous columns - "
            f"got {width} columns of strides {scores.stride(1)} and {candidates.stride(1)} and a top-p of {top_p}"
        )
    out = torch.empty(rows, keep, device=scores.device, dtype=torch.int64)
    listed = torch.empty(rows, device=scores.device, dtype=torch.int32)
    if rows == 0 or keep == 0:
        return out, listed.fill_(keep)
    listing_room, spare = room(width, keep)
    compiled_scored_pick()(
        scores, candidates, lengths, out, listed, listing_room, (FIXED_WORDS + width + spare) * 4
    )
    return out, listed


def shape_arguments(
    index_queries: torch.Tensor, shapes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return what a shaped level's spread reads, laid out as the kernel reads it, or stand-ins nothing reads, left unwritten.

    Args:
        index_queries: Index queries [S, 64] bf16.
        shapes: The closed node buckets' basis, variances and floors, and their deviations - or None.

    Returns:
        The sign words [N, SHAPE_RANK * 2] int32, variances [N, SHAPE_RANK] fp32, floors [N] fp32, the queries'
        bit-planes [S, QUERY_BITS * 2] int32, their level step and total [S, 2] fp32, and the deviations.
    """
    rows, device = index_queries.shape[0], index_queries.device
    if shapes is None:
        # Nothing reads them, and a fill of each would be five launches a level in a graph
        return (
            torch.empty(1, SHAPE_RANK * SHAPE_WORDS, device=device, dtype=torch.int32),
            torch.empty(1, SHAPE_RANK, device=device),
            torch.empty(1, device=device),
            torch.empty(rows, QUERY_BITS * SHAPE_WORDS, device=device, dtype=torch.int32),
            torch.empty(rows, 2, device=device),
            0.0,
        )
    basis, variances, floor, deviations = shapes
    planes = sign_query(index_queries)
    words = (
        basis_words(basis).view(-1, SHAPE_RANK * SHAPE_WORDS)
        if basis.numel()
        else torch.zeros(1, SHAPE_RANK * SHAPE_WORDS, device=device, dtype=torch.int32)
    )
    query_terms = torch.stack([planes.steps[:, 0], planes.totals], dim=1)
    return (
        words,
        variances.contiguous(),
        floor.contiguous(),
        planes.planes.view(rows, QUERY_BITS * SHAPE_WORDS),
        query_terms,
        deviations,
    )


def supports_keep(index_queries: torch.Tensor, width: int, keep: int) -> bool:
    """Tell whether the CuTeDSL keep takes a start level's rows - 64-dim bf16 queries, and a row and its kept that fit.

    Args:
        index_queries: Index queries [S, index_dim].
        width: Columns a query's row of scores holds.
        keep: Columns a query keeps.
    """
    return (
        index_queries.shape[1] == INDEX_DIM
        and index_queries.dtype == torch.bfloat16
        and width + room(width, keep)[1] <= SHARED_WORDS
    )


def keep_start_level(
    index_queries: torch.Tensor, scores: torch.Tensor, children: torch.Tensor, keep: int
) -> torch.Tensor:
    """Keep each query's best `keep` columns of a start level's scores, and name their children.

    The level's scores are in hand - every node bucket's, shaped or not - so the kernel only reads them.

    Args:
        index_queries: Index queries [S, 64] bf16.
        scores: The level's scores [S, N] fp32.
        children: Each column's children [N, node_bucket_size] int32.
        keep: Columns a query keeps, the beam's width.

    Returns:
        The kept node buckets' children [S, keep * node_bucket_size] int32 - a kept bucket's side by side in column
        order, -1 where a query kept nothing.
    """
    # A row this wide stages too much for a second CTA on an SM
    # => it is listed first, and only its listing is searched
    staged_bytes = 4 * (FIXED_WORDS + scores.shape[1] + room(scores.shape[1], keep)[1])
    if 2 * staged_bytes > SHARED_BYTES and cutedsl_keep.supports(scores, children, keep):
        return cutedsl_keep.keep_start_level_listed(scores, children, keep)
    carried = torch.empty(len(index_queries), 0, device=index_queries.device, dtype=torch.int32)
    # No carried row names one, but CuTe wants a tensor
    centroids = torch.zeros(1, INDEX_DIM, device=index_queries.device, dtype=torch.bfloat16)
    return beam_step(index_queries, scores, children, carried, centroids, children, keep, 1.0)


def compiled_pick(element_bytes: int, threads: int) -> Any:
    """Return the last level compiled once a centroid precision and thread count, taking torch tensors as they are.

    Args:
        element_bytes: Bytes a centroid entry takes - 1 for fp8, 2 for bf16.
        threads: Threads of a CTA.
    """
    key = (element_bytes, threads)
    if key in COMPILED_PICKS:
        return COMPILED_PICKS[key]
    rows, open_count, carried_count, centroid_count, page_count, width = (cute.sym_int32() for _ in range(6))
    query_stride, score_stride, carried_stride, out_stride = (cute.sym_int64() for _ in range(4))
    centroid_type = cutlass.Float8E4M3FN if element_bytes == 1 else cutlass.BFloat16
    arguments = (
        make_fake_tensor(cutlass.BFloat16, (rows, INDEX_DIM), stride=(query_stride, 1), assumed_align=16),
        make_fake_tensor(cutlass.Float32, (rows, open_count), stride=(score_stride, 1), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (rows, carried_count), stride=(carried_stride, 1), assumed_align=4),
        make_fake_compact_tensor(
            centroid_type, (centroid_count, INDEX_DIM), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_compact_tensor(cutlass.Int32, (page_count,), assumed_align=4),
        make_fake_tensor(cutlass.Int64, (rows, width), stride=(out_stride, 1), assumed_align=8),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED_PICKS[key] = cute.compile(
        TopPPick(element_bytes, threads), *arguments, options="--enable-tvm-ffi"
    )
    return COMPILED_PICKS[key]


def pick_top_p(
    index_queries: torch.Tensor,
    open_scores: torch.Tensor,
    carried: torch.Tensor | None,
    centroids: torch.Tensor,
    pages: torch.Tensor,
    covered_buckets: int,
    page_rows: int,
    top_p: int,
) -> torch.Tensor:
    """Pick each query's best `top_p` of its uncovered buckets and the last beam's children, keyed by their cache rows.

    Args:
        index_queries: Index queries [S, 64] bf16.
        open_scores: The uncovered buckets' scores [S, open] fp32, -inf past a query's visible count.
        carried: The cache rows the last beam kept [S, C] int32, -1 for none, or None.
        centroids: The bucket centroids [rows, 64] bf16 the carried rows name.
        pages: The request's pages [max_pages] int32.
        covered_buckets: Buckets under a node bucket - the first uncovered bucket.
        page_rows: Cache rows a page holds.
        top_p: Buckets a query picks.

    Returns:
        The picked buckets' scores packed with their cache rows [S, W] int64 in column order, W the lesser of the row's
        width and the top-p - the empty key for a column scored -inf or with no row.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    rows = index_queries.shape[0]
    if carried is None:
        carried = torch.empty(rows, 0, device=index_queries.device, dtype=torch.int32)
    open_count = open_scores.shape[1]
    width = open_count + carried.shape[1]
    keep = min(top_p, width)
    if not supports(index_queries, centroids, carried, open_count, keep):
        raise ValueError(
            "the CuTeDSL pick takes bf16 index queries of 64 dims over contiguous fp8 or bf16 centroids, and a row "
            f"and its kept within {SHARED_WORDS} words - got {index_queries.dtype} {tuple(index_queries.shape)}, "
            f"{centroids.dtype} {tuple(centroids.shape)} and {open_count} + {carried.shape[1]} columns"
        )
    out = torch.empty(rows, keep, device=index_queries.device, dtype=torch.int64)
    if rows == 0 or keep == 0:
        return out
    if open_count == 0:
        # Nothing reads them, but CuTe wants a tensor
        open_scores = torch.empty(rows, 1, device=index_queries.device)
    listing_room, spare = room(width, keep)
    compiled_pick(centroids.element_size(), READING_THREADS)(
        index_queries.contiguous(),
        open_scores,
        carried,
        centroids,
        pages,
        out,
        open_count,
        covered_buckets,
        page_rows,
        listing_room,
        (FIXED_WORDS + width + spare) * 4,
    )
    return out
