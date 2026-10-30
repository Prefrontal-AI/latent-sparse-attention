"""A flat decode step's select in CuTeDSL - each request's buckets scored through its pages and its best listed as keys.

The Triton decode scores every visible bucket into a matrix and picks each row's exact top-p from it. The expansion by
threshold needs no exact top-p - it finds its own stopping key in any listing that holds the top-p - so here the scan
itself lists the buckets that clear a threshold sampled ahead of it, and no pass over the matrix follows.

A step runs four launches. The first scores a strided sample of each row's buckets, in runs of neighbours, and the
second takes the score about `target` times the top-p of a row's buckets sit at or above, as the Triton pick plans
it; a row no wider than its listing takes every bucket. The scan scores every bucket, writes the score, and stages
each one at or above the threshold as a key - a span's keys in bucket order, the span's place taken with one atomic.
The last lists each row's staged spans in turn, so a listing is in bucket order and the same on every run, and a row
whose staging holds fewer than its top-p or overflowed is listed exactly from the scores it wrote instead. The sample
only sets how much is listed, never what is read.

Each launch starts as soon as the one before has begun, and waits for it at its first read of what it wrote. The
scan scores its span while the threshold is still being ranked and waits only to compare the scores with it, so the
scan's reads, most of a long context's step, run beside the sample and the threshold.

A score is the paged scan's bit for bit - each of eight lanes adds its eight products in turn, the lanes add by halves,
and the sum is added to zero - so a key is the key the Triton pick lists.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from lsa.core.kernels.cutedsl.scoring import EMPTY, pack_key
from lsa.core.kernels.triton.top_k import RUN, SAMPLES, keep_plan

INDEX_DIM = 64
GROUP_LANES = 8
DIMS_PER_LANE = INDEX_DIM // GROUP_LANES
THREADS = 128
GROUPS = THREADS // GROUP_LANES
IN_FLIGHT = 8
# Buckets a CTA of the scan takes
SPAN = 256
# Spans a request's listing is ordered from at most, their places held in shared memory - contexts to 16M
MOST_SPANS = 8192
SPAN_BITS = MOST_SPANS.bit_length() - 1
# Threads of the threshold and of the ordering, and the bits of a word each pass of their search settles
SELECT_THREADS = 1024
SELECT_WARPS = SELECT_THREADS // 32
# CTAs a request's ordering is split over, each taking every ORDER_PARTS-th run of a CTA's width of its keys
ORDER_PARTS = 4
DIGIT_BITS = 8
BINS = 1 << DIGIT_BITS
# Samples of a request a thread of the threshold holds, at most
SAMPLES_PER_THREAD = -(-SAMPLES[-1] // SELECT_THREADS)
# The threshold's search settles the top two digits of ten bits, a bin a thread - more than the word's kept bits
SEARCH_BITS = 10
SEARCH_BINS = 1 << SEARCH_BITS
SEARCH_PASSES = 2
# A histogram of the search holds its bins padded by one in every 32, so a lane's 32 bins sit in distinct banks
HISTOGRAM_SIZE = SEARCH_BINS + SEARCH_BINS // 32
# Low bits of the sampled threshold's word left clear - a granule of the scores far below the listing's slack
THRESHOLD_LOW = 13


class PagedScoring:
    """The paged scan's score of one bucket, shared by the sample and the scan."""

    @cute.jit
    def load_query(self, index_queries: cute.Tensor, request: Any, member: Any) -> cute.Tensor:
        """Return this lane's eight dims of a request's index query in fp32.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            request: The request.
            member: This lane within its group of eight.
        """
        query_bf16 = cute.make_rmem_tensor(DIMS_PER_LANE, index_queries.element_type)
        cute.autovec_copy(
            cute.local_tile(index_queries, (1, DIMS_PER_LANE), (request, member))[0, None], query_bf16
        )
        query = cute.make_rmem_tensor(DIMS_PER_LANE, cutlass.Float32)
        for e in cutlass.range_constexpr(DIMS_PER_LANE):
            query[e] = query_bf16[e].to(cutlass.Float32)
        return query

    @cute.jit
    def cache_row(
        self, pages: cute.Tensor, request: Any, bucket: Any, count: Any, page_rows: cutlass.Constexpr
    ) -> Any:
        """Return a request's logical bucket's row of the cache, or -1 past its count or on a page it has not got.

        Args:
            pages: Each request's pages [B, max_pages] int32, -1 for none.
            request: The request.
            bucket: The logical bucket.
            count: Buckets the request sees.
            page_rows: Cache rows a page holds.
        """
        row = cutlass.Int32(-1)
        if bucket < count:
            page = pages[request, bucket // page_rows]
            if page >= 0:
                row = page * page_rows + bucket % page_rows
        return row

    @cute.jit
    def dot(self, query: cute.Tensor, loaded: cute.Tensor) -> Any:
        """Return the score the paged scan adds - this lane's products in turn, the lanes by halves, onto zero.

        Args:
            query: This lane's dims of the index query in fp32.
            loaded: This lane's dims of the centroid in bf16.
        """
        score = query[0] * loaded[0].to(cutlass.Float32) + query[1] * loaded[1].to(cutlass.Float32)
        for e in cutlass.range_constexpr(2, DIMS_PER_LANE):
            score = score + query[e] * loaded[e].to(cutlass.Float32)
        for offset in cutlass.range_constexpr(3):
            score = score + cute.arch.shuffle_sync_bfly(score, GROUP_LANES // 2 >> offset)
        return cutlass.Float32(0.0) + score


@cute.jit
def score_word(score: Any) -> Any:
    """Return an fp32 score as the unsigned 32-bit word of the same order, held in an int32.

    Args:
        score: The fp32 score.
    """
    cell = cute.make_rmem_tensor(1, cutlass.Float32)
    cell[0] = score
    bits = cute.recast_tensor(cell, cutlass.Int32)[0]
    ordered = bits ^ ((bits >> 31) & cutlass.Int32(0x7FFFFFFF))
    return ordered ^ cutlass.Int32(-(2**31))


@cute.jit
def word_score(word: Any) -> Any:
    """Return the fp32 score an unsigned word of `score_word` came from.

    Args:
        word: The word, held in an int32.
    """
    ordered = word ^ cutlass.Int32(-(2**31))
    bits = ordered ^ ((ordered >> 31) & cutlass.Int32(0x7FFFFFFF))
    cell = cute.make_rmem_tensor(1, cutlass.Int32)
    cell[0] = bits
    return cute.recast_tensor(cell, cutlass.Float32)[0]


class WordSearch:
    """The k-th greatest of a row's words, a digit at a time from the top, by histograms in shared memory."""

    def __init__(self, threads: int = SELECT_THREADS) -> None:
        """Fix the CTA's threads and its barrier.

        Args:
            threads: Threads of the CTA, at least BINS.
        """
        self.threads = threads
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads)

    @cute.jit
    def exclusive_sum(self, value: Any, warp_totals: cute.Tensor) -> tuple[Any, Any]:
        """Return the sum of value over the threads before this one, and over all of them.

        Args:
            value: This thread's value.
            warp_totals: Shared memory for each warp's total.
        """
        return block_exclusive_sum(value, warp_totals, self.threads // 32, self.barrier)

    @cute.jit
    def find_digit(
        self, histogram: cute.Tensor, rank: Any, warp_totals: cute.Tensor, shared: cute.Tensor
    ) -> tuple[Any, Any]:
        """Return the greatest digit with at least `rank` counted at or above it, and how many sit above it.

        Args:
            histogram: Each digit's count [BINS] int32.
            rank: The rank sought, from 1.
            warp_totals: Shared memory for the scan.
            shared: Shared memory to hand the answer to every thread.
        """
        thread, _, _ = cute.arch.thread_idx()
        # Thread t holds the digit BINS - 1 - t, so the scan runs from the top digit down
        digit = BINS - 1 - thread
        held = cutlass.Int32(0)
        if thread < BINS:
            held = histogram[digit]
        above, _ = self.exclusive_sum(held, warp_totals)
        if thread < BINS and above < rank and above + held >= rank:
            shared[0] = digit
            shared[1] = above
        self.barrier.arrive_and_wait()
        found = shared[0]
        found_above = shared[1]
        self.barrier.arrive_and_wait()
        return found, found_above

    @cute.jit
    def kth(
        self,
        words: cute.Tensor,
        length: Any,
        rank: Any,
        passes: cutlass.Constexpr,
        scored: cutlass.Constexpr,
        histogram: cute.Tensor,
        warp_totals: cute.Tensor,
        shared: cute.Tensor,
    ) -> tuple[Any, Any]:
        """Return the top `passes` digits of the `rank`-th greatest word, and how many words sit above them.

        Args:
            words: A row's words [length] int32, read as unsigned - or its fp32 scores, when scored.
            length: Words the row holds.
            rank: The rank sought, from 1, at most the length.
            passes: Digits settled.
            scored: Whether the row holds scores, each read as its word.
            histogram: Shared memory for a digit's counts [BINS] int32.
            warp_totals: Shared memory for the scan.
            shared: Shared memory to hand an answer to every thread.
        """
        thread, _, _ = cute.arch.thread_idx()
        prefix = cutlass.Int32(0)
        above = cutlass.Int32(0)
        for p in cutlass.range_constexpr(passes):
            shift = 32 - DIGIT_BITS * (p + 1)
            if thread < BINS:
                histogram[thread] = cutlass.Int32(0)
            self.barrier.arrive_and_wait()
            for i in range(thread, length, self.threads):
                word = cutlass.Uint32(0)
                if cutlass.const_expr(scored):
                    word = cutlass.Uint32(score_word(words[i]))
                else:
                    word = cutlass.Uint32(words[i])
                matches = True
                if cutlass.const_expr(p > 0):
                    matches = (word >> (shift + DIGIT_BITS)) == (
                        cutlass.Uint32(prefix) >> (shift + DIGIT_BITS)
                    )
                if matches:
                    digit = cutlass.Int32((word >> shift) & cutlass.Uint32(BINS - 1))
                    cute.arch.atomic_add((histogram.iterator + digit).llvm_ptr, cutlass.Int32(1), scope="cta")
            self.barrier.arrive_and_wait()
            digit, digit_above = self.find_digit(histogram, rank - above, warp_totals, shared)
            prefix = prefix | (digit << shift)
            above = above + digit_above
        return prefix, above

    @cute.jit
    def list_exact(
        self,
        row: cute.Tensor,
        length: Any,
        want: Any,
        keys: cute.Tensor,
        histogram: cute.Tensor,
        warp_totals: cute.Tensor,
        shared: cute.Tensor,
    ) -> None:
        """List a row's exact best `want` as keys in column order - a tie at the last going to the lower column.

        Args:
            row: The row's scores [length] fp32.
            length: Columns the row holds.
            want: Columns listed, at most the length.
            keys: The row's listing [capacity] int64, its first `want` written.
            histogram: Shared memory for a digit's counts [BINS] int32.
            warp_totals: Shared memory for the scan.
            shared: Shared memory to hand an answer to every thread.
        """
        thread, _, _ = cute.arch.thread_idx()
        kth_word, above = self.kth(row, length, want, 32 // DIGIT_BITS, True, histogram, warp_totals, shared)
        # Every word above the want-th, and its ties lowest column first until the want is full, in column order
        placed = cutlass.Int32(0)
        ties_taken = cutlass.Int32(0)
        for first in range(0, length, self.threads):
            i = first + thread
            score = -cutlass.Float32.inf
            greater = False
            tied = cutlass.Int32(0)
            if i < length:
                score = row[i]
                word = cutlass.Uint32(score_word(score))
                greater = word > cutlass.Uint32(kth_word)
                tied = cutlass.Int32(word == cutlass.Uint32(kth_word))
            tie_before, tie_total = self.exclusive_sum(tied, warp_totals)
            take = cutlass.Int32(0)
            if greater or (tied != 0 and ties_taken + tie_before < want - above):
                take = cutlass.Int32(1)
            before, total = self.exclusive_sum(take, warp_totals)
            if take != 0:
                keys[placed + before] = pack_key(score, i)
            placed = placed + total
            ties_taken = ties_taken + tie_total


class Sample(PagedScoring):
    """The scores of a strided sample of each request's buckets, as words."""

    def __init__(self, page_rows: int) -> None:
        """Fix the cache rows a page holds.

        Args:
            page_rows: Cache rows a page holds.
        """
        self.page_rows = page_rows

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        pages: cute.Tensor,
        counts: cute.Tensor,
        words: cute.Tensor,
        stride: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a request and 128 samples.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16.
            pages: Each request's pages [B, max_pages] int32.
            counts: Buckets each request sees [B] int32.
            words: Each sample's score as an unsigned word [B, samples] int32, 0 past the request's count.
            stride: Buckets between two runs of samples.
            stream: The stream to launch on.
        """
        self.kernel(index_queries, bucket_centroids, pages, counts, words, stride).launch(
            grid=(cute.ceil_div(words.shape[1], THREADS), index_queries.shape[0], 1),
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        pages: cute.Tensor,
        counts: cute.Tensor,
        words: cute.Tensor,
        stride: cutlass.Int32,
    ) -> None:
        """Score this CTA's samples of a request, a group of eight lanes a sample.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16.
            pages: Each request's pages [B, max_pages] int32.
            counts: Buckets each request sees [B] int32.
            words: Each sample's score as an unsigned word [B, samples] int32.
            stride: Buckets between two runs of samples.
        """
        # The first of the step's kernels, whatever ran before it is done once every CTA has passed here
        # => the next may start, and its CTAs run ahead of this one's results as far as they read none
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        part, request, _ = cute.arch.block_idx()
        group = thread // GROUP_LANES
        member = thread % GROUP_LANES
        count = counts[request]
        query = self.load_query(index_queries, request, member)
        loaded = cute.make_rmem_tensor(
            cute.make_layout((IN_FLIGHT, DIMS_PER_LANE), stride=(DIMS_PER_LANE, 1)),
            bucket_centroids.element_type,
        )
        rows = cute.make_rmem_tensor(IN_FLIGHT, cutlass.Int32)
        for f in cutlass.range_constexpr(IN_FLIGHT):
            sample = part * THREADS + f * GROUPS + group
            column = (sample // RUN) * stride + sample % RUN
            rows[f] = self.cache_row(pages, request, column, count, self.page_rows)
        for f in cutlass.range_constexpr(IN_FLIGHT):
            cute.autovec_copy(
                cute.local_tile(bucket_centroids, (1, DIMS_PER_LANE), (cutlass.max(rows[f], 0), member))[
                    0, None
                ],
                loaded[f, None],
            )
        for f in cutlass.range_constexpr(IN_FLIGHT):
            score = self.dot(query, loaded[f, None])
            sample = part * THREADS + f * GROUPS + group
            column = (sample // RUN) * stride + sample % RUN
            if member == 0 and sample < words.shape[1]:
                word = cutlass.Int32(0)
                if column < count:
                    if rows[f] < 0:
                        score = -cutlass.Float32.inf
                    word = score_word(score)
                words[request, sample] = word


class Threshold:
    """Each request's threshold, from its sample - the score about `target` times its top-p sit at or above."""

    def __init__(self) -> None:
        """Fix the CTA's barrier."""
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=SELECT_THREADS)

    @cute.jit
    def __call__(
        self,
        counts: cute.Tensor,
        words: cute.Tensor,
        thresholds: cute.Tensor,
        listed: cute.Tensor,
        stride: cutlass.Int32,
        top_p: cutlass.Int32,
        target: cutlass.Float32,
        capacity: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a request.

        Args:
            counts: Buckets each request sees [B] int32.
            words: Each sample's score as an unsigned word [B, samples] int32, 0 past the request's count.
            thresholds: Each request's threshold [B] fp32, written.
            listed: Keys each request lists [B] int32, zeroed for the scan.
            stride: Buckets between two runs of samples.
            top_p: Buckets the read takes at most.
            target: The listing's expected size, in the top-p.
            capacity: Keys a listing holds.
            stream: The stream to launch on.
        """

        @cute.struct
        class Storage:
            """Shared memory of a block: the search passes' histograms and the block's scalars."""

            histograms: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, SEARCH_PASSES * HISTOGRAM_SIZE], 16
            ]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.kernel(counts, words, thresholds, listed, stride, top_p, target, capacity, Storage).launch(
            grid=(words.shape[0], 1, 1), block=[SELECT_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        counts: cute.Tensor,
        words: cute.Tensor,
        thresholds: cute.Tensor,
        listed: cute.Tensor,
        stride: cutlass.Int32,
        top_p: cutlass.Int32,
        target: cutlass.Float32,
        capacity: cutlass.Int32,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Rank a request's sample and write its threshold, -inf for one its listing holds whole.

        Args:
            counts: Buckets each request sees [B] int32.
            words: Each sample's score as an unsigned word [B, samples] int32.
            thresholds: Each request's threshold [B] fp32.
            listed: Keys each request lists [B] int32.
            stride: Buckets between two runs of samples.
            top_p: Buckets the read takes at most.
            target: The listing's expected size, in the top-p.
            capacity: Keys a listing holds.
            Storage: The shared memory's struct.
        """
        # The sample's CTAs have passed their own wait, so nothing before them is still running
        # => the scan may start now and score while this ranks
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        request, _, _ = cute.arch.block_idx()
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        histograms = storage.histograms.get_tensor(cute.make_layout(SEARCH_PASSES * HISTOGRAM_SIZE))
        shared = storage.shared.get_tensor(cute.make_layout(4))
        count = counts[request]
        samples = words.shape[1]
        # The samples a request sees are the first of its runs that start below its count
        whole_runs = count // stride
        seen = cutlass.min(samples, whole_runs * RUN + cutlass.min(count - whole_runs * stride, RUN))
        wanted = target * cutlass.Float32(top_p)
        threshold = -cutlass.Float32.inf
        # Both passes' histograms are cleared while the sample is still being scored
        for k in cutlass.range_constexpr(-(-SEARCH_PASSES * HISTOGRAM_SIZE // SELECT_THREADS)):
            i = k * SELECT_THREADS + thread
            if i < SEARCH_PASSES * HISTOGRAM_SIZE:
                histograms[i] = cutlass.Int32(0)
        self.barrier.arrive_and_wait()
        cute.arch.griddepcontrol_wait()
        # A thread holds its samples' words in registers, the passes run over them
        mine = cute.make_rmem_tensor(SAMPLES_PER_THREAD, cutlass.Int32)
        for w in cutlass.range_constexpr(SAMPLES_PER_THREAD):
            mine[w] = words[request, cutlass.min(w * SELECT_THREADS + thread, samples - 1)]
        if count > capacity and wanted < cutlass.Float32(count) and seen > 0:
            rank = cutlass.Int32(wanted * cutlass.Float32(seen) / cutlass.Float32(count)) + 1
            rank = cutlass.min(cutlass.max(rank, 1), seen)
            prefix = cutlass.Int32(0)
            above = cutlass.Int32(0)
            lane = thread % 32
            for p in cutlass.range_constexpr(SEARCH_PASSES):
                shift = 32 - SEARCH_BITS * (p + 1)
                base = p * HISTOGRAM_SIZE
                for w in cutlass.range_constexpr(SAMPLES_PER_THREAD):
                    # A run of the CTA's samples all past the ones seen is skipped by every thread
                    if w * SELECT_THREADS < seen:
                        word = cutlass.Uint32(mine[w])
                        counted = w * SELECT_THREADS + thread < seen
                        if cutlass.const_expr(p > 0):
                            counted = counted and (word >> (shift + SEARCH_BITS)) == (
                                cutlass.Uint32(prefix) >> (shift + SEARCH_BITS)
                            )
                        if counted:
                            digit = cutlass.Int32((word >> shift) & cutlass.Uint32(SEARCH_BINS - 1))
                            cute.arch.atomic_add(
                                (histograms.iterator + base + digit + (digit >> 5)).llvm_ptr,
                                cutlass.Int32(1),
                                scope="cta",
                            )
                self.barrier.arrive_and_wait()
                # One warp finds the digit - a lane takes 32 digits from the top, the lanes' sums are scanned
                if thread < 32:
                    top = SEARCH_BINS - 1 - 32 * lane
                    counts_here = cute.make_rmem_tensor(32, cutlass.Int32)
                    total = cutlass.Int32(0)
                    for j in cutlass.range_constexpr(32):
                        counts_here[j] = histograms[base + (top - j) + (top >> 5)]
                        total = total + counts_here[j]
                    inclusive = total
                    for i in cutlass.range_constexpr(5):
                        other = cute.arch.shuffle_sync_up(inclusive, 1 << i, mask_and_clamp=0)
                        if lane >= (1 << i):
                            inclusive = inclusive + other
                    before = inclusive - total
                    if before < rank - above and inclusive >= rank - above:
                        running = before
                        for j in cutlass.range_constexpr(32):
                            if running < rank - above and running + counts_here[j] >= rank - above:
                                shared[1] = top - j
                                shared[2] = running
                            running = running + counts_here[j]
                self.barrier.arrive_and_wait()
                prefix = prefix | (shared[1] << shift)
                above = above + shared[2]
            level = prefix & cutlass.Int32(~((1 << THRESHOLD_LOW) - 1))
            if level != 0:
                threshold = word_score(level)
        if thread == 0:
            thresholds[request] = threshold
            listed[request] = cutlass.Int32(0)


class Scan(PagedScoring):
    """Every bucket of each request scored and written, and those at or above its threshold staged a span at a time."""

    def __init__(self, page_rows: int) -> None:
        """Fix the cache rows a page holds and the CTA's barrier.

        Args:
            page_rows: Cache rows a page holds.
        """
        self.page_rows = page_rows
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)

    @cute.jit
    def __call__(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        pages: cute.Tensor,
        counts: cute.Tensor,
        thresholds: cute.Tensor,
        scores: cute.Tensor,
        staged: cute.Tensor,
        span_counts: cute.Tensor,
        span_offsets: cute.Tensor,
        listed: cute.Tensor,
        stream: Any,
    ) -> None:
        """Launch a CTA a request and span of buckets.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16.
            pages: Each request's pages [B, max_pages] int32.
            counts: Buckets each request sees [B] int32.
            thresholds: Each request's threshold [B] fp32.
            scores: Scores [B, max_rows] fp32, written, -inf past a request's count or on a page it has not got.
            staged: Each request's keys at or above its threshold [B, capacity] int64, a span's in bucket order.
            span_counts: Keys each span staged [B, spans] int32.
            span_offsets: Where each span's keys start in its request's staging [B, spans] int32.
            listed: Keys each request staged [B] int32, zero on entry - above capacity for one that overflowed.
            stream: The stream to launch on.
        """

        @cute.struct
        class Storage:
            """Shared memory of a block: one span's scores, the per-warp totals and the block's scalars."""

            span_scores: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, SPAN], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, THREADS // 32], 16]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.kernel(
            index_queries,
            bucket_centroids,
            pages,
            counts,
            thresholds,
            scores,
            staged,
            span_counts,
            span_offsets,
            listed,
            Storage,
        ).launch(
            grid=(cute.ceil_div(scores.shape[1], SPAN), index_queries.shape[0], 1),
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        index_queries: cute.Tensor,
        bucket_centroids: cute.Tensor,
        pages: cute.Tensor,
        counts: cute.Tensor,
        thresholds: cute.Tensor,
        scores: cute.Tensor,
        staged: cute.Tensor,
        span_counts: cute.Tensor,
        span_offsets: cute.Tensor,
        listed: cute.Tensor,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Score a span of a request's buckets, a group of eight lanes a bucket, then stage its keys in bucket order.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16.
            pages: Each request's pages [B, max_pages] int32.
            counts: Buckets each request sees [B] int32.
            thresholds: Each request's threshold [B] fp32.
            scores: Scores [B, max_rows] fp32.
            staged: Each request's keys at or above its threshold [B, capacity] int64.
            span_counts: Keys each span staged [B, spans] int32.
            span_offsets: Where each span's keys start in its request's staging [B, spans] int32.
            listed: Keys each request staged [B] int32.
            Storage: The shared memory's struct.
        """
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        span, request, _ = cute.arch.block_idx()
        group = thread // GROUP_LANES
        member = thread % GROUP_LANES
        begin = span * SPAN
        end = cutlass.min(begin + SPAN, scores.shape[1])
        count = counts[request]
        capacity = staged.shape[1]
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        span_scores = storage.span_scores.get_tensor(cute.make_layout(SPAN))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(THREADS // 32))
        shared = storage.shared.get_tensor(cute.make_layout(4))
        query = self.load_query(index_queries, request, member)
        loaded = cute.make_rmem_tensor(
            cute.make_layout((IN_FLIGHT, DIMS_PER_LANE), stride=(DIMS_PER_LANE, 1)),
            bucket_centroids.element_type,
        )
        rows = cute.make_rmem_tensor(IN_FLIGHT, cutlass.Int32)
        # The span is scored before the threshold is waited for, so the scoring runs beside the threshold's search
        for first in range(begin, end, GROUPS * IN_FLIGHT):
            for f in cutlass.range_constexpr(IN_FLIGHT):
                rows[f] = self.cache_row(pages, request, first + f * GROUPS + group, count, self.page_rows)
            for f in cutlass.range_constexpr(IN_FLIGHT):
                cute.autovec_copy(
                    cute.local_tile(bucket_centroids, (1, DIMS_PER_LANE), (cutlass.max(rows[f], 0), member))[
                        0, None
                    ],
                    loaded[f, None],
                )
            for f in cutlass.range_constexpr(IN_FLIGHT):
                score = self.dot(query, loaded[f, None])
                bucket = first + f * GROUPS + group
                if member == 0 and bucket < end:
                    if rows[f] < 0:
                        score = -cutlass.Float32.inf
                    scores[request, bucket] = score
                    span_scores[bucket - begin] = score
        # What the step's earlier kernels left is read from here on
        cute.arch.griddepcontrol_wait()
        threshold = thresholds[request]
        self.barrier.arrive_and_wait()

        # The span's keys staged in bucket order, its place in the staging taken with one atomic
        per_thread = SPAN // THREADS
        kept = cute.make_rmem_tensor(per_thread, cutlass.Int32)
        mine = cutlass.Int32(0)
        for j in cutlass.range_constexpr(per_thread):
            bucket = begin + thread * per_thread + j
            kept[j] = cutlass.Int32(0)
            if bucket < end and bucket < count and span_scores[thread * per_thread + j] >= threshold:
                kept[j] = cutlass.Int32(1)
            mine = mine + kept[j]
        before, total = block_exclusive_sum(mine, warp_totals, THREADS // 32, self.barrier)
        if thread == 0:
            offset = cutlass.Int32(0)
            if total > 0:
                offset = cute.arch.atomic_add((listed.iterator + request).llvm_ptr, total, scope="gpu")
            shared[0] = offset
            span_counts[request, span] = total
            span_offsets[request, span] = offset
        self.barrier.arrive_and_wait()
        place = shared[0] + before
        for j in cutlass.range_constexpr(per_thread):
            local = thread * per_thread + j
            if kept[j] != 0:
                if place < capacity:
                    staged[request, place] = pack_key(span_scores[local], begin + local)
                place = place + 1


class Order(WordSearch):
    """Each request's listing in bucket order - its staged spans in turn, or its exact top-p from its scores."""

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        counts: cute.Tensor,
        staged: cute.Tensor,
        span_counts: cute.Tensor,
        span_offsets: cute.Tensor,
        keys: cute.Tensor,
        listed: cute.Tensor,
        top_p: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a request's CTAs, each taking every few-th run of a CTA's width of its keys.

        Args:
            scores: Scores [B, max_rows] fp32.
            counts: Buckets each request sees [B] int32.
            staged: Each request's keys at or above its threshold [B, capacity] int64.
            span_counts: Keys each span staged [B, spans] int32.
            span_offsets: Where each span's keys start in its request's staging [B, spans] int32.
            keys: Each request's listing [B, capacity] int64, written in bucket order.
            listed: Keys each request staged [B] int32, rewritten for one listed exactly.
            top_p: Buckets the read takes at most.
            stream: The stream to launch on.
        """

        @cute.struct
        class Storage:
            """Shared memory of a block: the score histogram, per-warp totals, scalars and span places."""

            histogram: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, BINS], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, SELECT_WARPS], 16]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            places: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MOST_SPANS], 16]

        self.kernel(scores, counts, staged, span_counts, span_offsets, keys, listed, top_p, Storage).launch(
            grid=(ORDER_PARTS, scores.shape[0], 1), block=[SELECT_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        counts: cute.Tensor,
        staged: cute.Tensor,
        span_counts: cute.Tensor,
        span_offsets: cute.Tensor,
        keys: cute.Tensor,
        listed: cute.Tensor,
        top_p: cutlass.Int32,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Copy a request's staged spans in turn, or list its exact top-p in bucket order when the staging misses it.

        Args:
            scores: Scores [B, max_rows] fp32.
            counts: Buckets each request sees [B] int32.
            staged: Each request's keys at or above its threshold [B, capacity] int64.
            span_counts: Keys each span staged [B, spans] int32.
            span_offsets: Where each span's keys start in its request's staging [B, spans] int32.
            keys: Each request's listing [B, capacity] int64.
            listed: Keys each request staged [B] int32.
            top_p: Buckets the read takes at most.
            Storage: The shared memory's struct.
        """
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        part, request, _ = cute.arch.block_idx()
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        histogram = storage.histogram.get_tensor(cute.make_layout(BINS))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(SELECT_WARPS))
        shared = storage.shared.get_tensor(cute.make_layout(4))
        cute.arch.griddepcontrol_wait()
        places = storage.places.get_tensor(cute.make_layout(MOST_SPANS))
        length = cutlass.min(counts[request], scores.shape[1])
        want = cutlass.min(top_p, length)
        # A thread sums a run of spans, so one scan places them all - and holds what the spans staged
        spans = span_counts.shape[1]
        run = cute.ceil_div(spans, SELECT_THREADS)
        first = thread * run
        last = cutlass.min(first + run, spans)
        mine = cutlass.Int32(0)
        for span in range(first, last):
            mine = mine + span_counts[request, span]
        place, held = self.exclusive_sum(mine, warp_totals)
        if held >= want and held <= keys.shape[1]:
            for span in range(first, last):
                places[span] = place
                place = place + span_counts[request, span]
            self.barrier.arrive_and_wait()
            # The best buckets cluster, so one span may hold most of the listing
            # => a thread takes every CTA-width-th key and finds its span by the last place at or before it
            for key in range(part * SELECT_THREADS + thread, held, ORDER_PARTS * SELECT_THREADS):
                span = cutlass.Int32(0)
                for bit in cutlass.range_constexpr(SPAN_BITS):
                    step = 1 << (SPAN_BITS - 1 - bit)
                    if span + step < spans:
                        if places[span + step] <= key:
                            span = span + step
                keys[request, key] = staged[request, span_offsets[request, span] + key - places[span]]
        elif part == 0:
            self.list_exact(
                scores[request, None], length, want, keys[request, None], histogram, warp_totals, shared
            )
            if thread == 0:
                listed[request] = want


@cute.jit
def block_exclusive_sum(
    value: Any, warp_totals: cute.Tensor, warps: cutlass.Constexpr, barrier: Any
) -> tuple[Any, Any]:
    """Return the sum of value over a CTA's threads before this one, and over all of them.

    Args:
        value: This thread's value.
        warp_totals: Shared memory for each warp's total.
        warps: The CTA's warps.
        barrier: The CTA's barrier.
    """
    thread, _, _ = cute.arch.thread_idx()
    lane = thread % 32
    warp = thread // 32
    inclusive = value
    for i in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(inclusive, 1 << i, mask_and_clamp=0)
        if lane >= (1 << i):
            inclusive = inclusive + other
    if lane == 31:
        warp_totals[warp] = inclusive
    barrier.arrive_and_wait()
    before = inclusive - value
    total = cutlass.Int32(0)
    for w in cutlass.range_constexpr(warps):
        if w < warp:
            before = before + warp_totals[w]
        total = total + warp_totals[w]
    barrier.arrive_and_wait()
    return before, total


COMPILED: dict[tuple[str, int], Any] = {}


def compiled(kind: str, page_rows: int) -> Any:
    """Return one of the step's kernels compiled once a page's rows, taking torch tensors as they are.

    Args:
        kind: `sample`, `threshold`, `scan` or `order`.
        page_rows: Cache rows a page holds.
    """
    key = (kind, page_rows if kind in ("sample", "scan") else 0)
    if key in COMPILED:
        return COMPILED[key]
    batch, cache_rows, max_pages, samples, columns, capacity, spans = (cute.sym_int32() for _ in range(7))
    page_stride, score_stride, key_stride, word_stride, staged_stride = (cute.sym_int64() for _ in range(5))
    queries = make_fake_compact_tensor(
        cutlass.BFloat16, (batch, INDEX_DIM), stride_order=(1, 0), assumed_align=16
    )
    centroids = make_fake_compact_tensor(
        cutlass.BFloat16, (cache_rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16
    )
    pages = make_fake_tensor(cutlass.Int32, (batch, max_pages), stride=(page_stride, 1), assumed_align=4)
    counts = make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
    words = make_fake_tensor(cutlass.Int32, (batch, samples), stride=(word_stride, 1), assumed_align=4)
    thresholds = make_fake_compact_tensor(cutlass.Float32, (batch,), assumed_align=4)
    listed = make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
    scores = make_fake_tensor(cutlass.Float32, (batch, columns), stride=(score_stride, 1), assumed_align=4)
    keys = make_fake_tensor(cutlass.Int64, (batch, capacity), stride=(key_stride, 1), assumed_align=8)
    staged = make_fake_tensor(cutlass.Int64, (batch, capacity), stride=(staged_stride, 1), assumed_align=8)
    span_counts = make_fake_compact_tensor(
        cutlass.Int32, (batch, spans), stride_order=(1, 0), assumed_align=4
    )
    span_offsets = make_fake_compact_tensor(
        cutlass.Int32, (batch, spans), stride_order=(1, 0), assumed_align=4
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    if kind == "sample":
        kernel: Any = Sample(page_rows)
        arguments: tuple[Any, ...] = (queries, centroids, pages, counts, words, cutlass.Int32(0), stream)
    elif kind == "threshold":
        kernel = Threshold()
        arguments = (
            counts,
            words,
            thresholds,
            listed,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Float32(0),
            cutlass.Int32(0),
            stream,
        )
    elif kind == "scan":
        kernel = Scan(page_rows)
        arguments = (
            queries,
            centroids,
            pages,
            counts,
            thresholds,
            scores,
            staged,
            span_counts,
            span_offsets,
            listed,
            stream,
        )
    else:
        kernel = Order()
        arguments = (
            scores,
            counts,
            staged,
            span_counts,
            span_offsets,
            keys,
            listed,
            cutlass.Int32(0),
            stream,
        )
    COMPILED[key] = cute.compile(kernel, *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


def supports(index_queries: torch.Tensor, bucket_centroids: torch.Tensor, max_rows: int) -> bool:
    """Tell whether the CuTeDSL decode select takes these rows - bf16, 64 index dims, contiguous, 16-byte aligned.

    Args:
        index_queries: One index query a request [B, index_dim].
        bucket_centroids: The cache's bucket centroids [rows, index_dim].
        max_rows: Buckets a request may see, at most `MOST_SPANS` spans of them.
    """
    return (
        -(-max_rows // SPAN) <= MOST_SPANS
        and index_queries.dtype == bucket_centroids.dtype == torch.bfloat16
        and index_queries.shape[1] == bucket_centroids.shape[1] == INDEX_DIM
        and index_queries.is_contiguous()
        and bucket_centroids.is_contiguous()
        and index_queries.data_ptr() % 16 == 0
        and bucket_centroids.data_ptr() % 16 == 0
    )


def decode_listing(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    pages: torch.Tensor,
    counts: torch.Tensor,
    page_rows: int,
    max_rows: int,
    top_p: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """List each request's buckets that hold its top-p, scored through its pages - fixed shapes, for the CUDA graph.

    Args:
        index_queries: One index query a request [B, index_dim] bf16.
        bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16, `page_rows` rows to a page.
        pages: Each request's pages [B, max_pages] int32, -1 for none.
        counts: Buckets each request sees [B] int32, at most `max_rows`.
        page_rows: Cache rows a page holds.
        max_rows: Buckets a request may see.
        top_p: Buckets the read takes at most.

    Returns:
        Packed keys [B, capacity] int64 holding each request's top-p, in bucket order - the same on every run.
        Keys each request lists [B] int32.

    Raises:
        ValueError: If the rows are not ones `supports` takes.
    """
    if not supports(index_queries, bucket_centroids, max_rows):
        raise ValueError(
            "the CuTeDSL decode select takes contiguous, 16-byte aligned bf16 rows of 64 index dims and at most "
            f"{MOST_SPANS * SPAN} buckets a request - got {index_queries.dtype} {tuple(index_queries.shape)} and "
            f"{bucket_centroids.dtype} {tuple(bucket_centroids.shape)} for {max_rows} buckets"
        )
    batch = index_queries.shape[0]
    device = index_queries.device
    samples, stride, target, capacity = keep_plan(top_p, max_rows)
    capacity = max(capacity, top_p)
    words = torch.empty(batch, samples, device=device, dtype=torch.int32)
    thresholds = torch.empty(batch, device=device, dtype=torch.float32)
    listed = torch.empty(batch, device=device, dtype=torch.int32)
    scores = torch.empty(batch, max_rows, device=device, dtype=torch.float32)
    keys = torch.empty(batch, capacity, device=device, dtype=torch.int64)
    staged = torch.empty(batch, capacity, device=device, dtype=torch.int64)
    spans = -(-max_rows // SPAN)
    span_counts = torch.empty(batch, spans, device=device, dtype=torch.int32)
    span_offsets = torch.empty(batch, spans, device=device, dtype=torch.int32)
    compiled("sample", page_rows)(index_queries, bucket_centroids, pages, counts, words, stride)
    compiled("threshold", page_rows)(counts, words, thresholds, listed, stride, top_p, target, capacity)
    compiled("scan", page_rows)(
        index_queries,
        bucket_centroids,
        pages,
        counts,
        thresholds,
        scores,
        staged,
        span_counts,
        span_offsets,
        listed,
    )
    compiled("order", page_rows)(scores, counts, staged, span_counts, span_offsets, keys, listed, top_p)
    return keys, listed


class DecodeSelect:
    """A flat decode step's listing in CuTeDSL, and the best buckets a check picks from it."""

    def __init__(self, top_p: int) -> None:
        """Fix the buckets a read takes at most.

        Args:
            top_p: Buckets the read takes at most.
        """
        self.top_p = top_p

    def listed(
        self,
        index_queries: torch.Tensor,
        bucket_centroids: torch.Tensor,
        pages: torch.Tensor,
        counts: torch.Tensor,
        page_rows: int,
        max_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List each request's buckets that hold its top-p - `decode_listing` with this select's top-p.

        Args:
            index_queries: One index query a request [B, index_dim] bf16.
            bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16.
            pages: Each request's pages [B, max_pages] int32, -1 for none.
            counts: Buckets each request sees [B] int32.
            page_rows: Cache rows a page holds.
            max_rows: Buckets a request may see.

        Returns:
            Packed keys [B, capacity] int64 holding each request's top-p, in bucket order, and keys each lists [B] int32.
        """
        return decode_listing(index_queries, bucket_centroids, pages, counts, page_rows, max_rows, self.top_p)

    def pick(self, listing: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Pick each request's best buckets from a listing.

        Args:
            listing: Packed keys [B, capacity] int64 and keys each request lists [B] int32.

        Returns:
            Buckets [B, top_p] int32, the best score first and a tie to the lower bucket, -1 past a request's count.
        """
        keys, listed = listing
        width = keys.shape[1]
        held = torch.arange(width, device=keys.device)[None] < listed[:, None]
        best = keys.masked_fill(~held, EMPTY).topk(min(self.top_p, width), dim=1).values
        buckets = (0x7FFFFFFF - (best & 0xFFFFFFFF)).int()
        picked = torch.full((keys.shape[0], self.top_p), -1, device=keys.device, dtype=torch.int32)
        picked[:, : best.shape[1]] = torch.where(best == EMPTY, -1, buckets)
        return picked
