"""A short prefill's listing in CuTeDSL - each row of a score matrix's best k and room beside them, in column order.

Below the fused select's gate a flat prefill scores every visible bucket into a matrix and lists each row's best
top-p for the expansion by threshold, which finds its own stopping key in any listing that holds them. The Triton
top-k samples a row, takes a threshold from the sample, lists the columns at or above it in no order and searches a
row it misjudged, in three launches.

Here one CTA a row does all of it and reads the row twice - a strided sample and one streaming pass - holding only
the sample in shared memory, so a row is never staged and many rows run on an SM at once. The sample's stride and
rank and the listing's width are the Triton top-k's plan, and a block-wide scan places each pass's columns at or above
the threshold, so a listing is the same on every run. A pass reads a column of every CTA-wide slice, coalesced, and
lists them slice by slice; an ordered listing reads each thread's run of columns instead, a tenth slower at the widest
rows, and lists in column order. A row the sample misjudged
- fewer than its top-p listed, or more than the listing holds - is listed exactly from its scores instead.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from lsa.core.kernels.cutedsl.decode import BINS, WordSearch, score_word
from lsa.core.kernels.cutedsl.scoring import pack_key
from lsa.core.kernels.triton.top_k import WHOLE, listing_plan

# Samples a row's threshold is ranked from at most, in shared memory
MOST_SAMPLES = 8192
# Threads of a row's CTA - few enough that several rows share an SM
LISTING_THREADS = 512
LISTING_WARPS = LISTING_THREADS // 32
# Columns a thread reads in a pass - a CTA's width apart, or side by side in an ordered listing
COLUMNS_PER_THREAD = 8
# Digits of the sampled threshold settled - the rest left clear, so the threshold sits at or under the sample's rank
THRESHOLD_PASSES = 2

COMPILED: dict[bool, Any] = {}


class RowListing(WordSearch):
    """Each row's sampled threshold, its listing, and the exact listing of a row the sample misjudged."""

    def __init__(self, ordered: bool) -> None:
        """Fix whether a listing is in column order.

        Args:
            ordered: List in column order, each thread reading a run of columns.
        """
        super().__init__(LISTING_THREADS)
        self.ordered = ordered

    def column_of(self, first: Any, c: int, thread: Any) -> Any:
        """Return the column a thread reads `c`-th in the pass that starts at `first`.

        Args:
            first: The pass's first column.
            c: The read, from 0 to COLUMNS_PER_THREAD.
            thread: The thread.
        """
        if self.ordered:
            return first + thread * COLUMNS_PER_THREAD + c
        return first + c * LISTING_THREADS + thread

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        lengths: cute.Tensor,
        keys: cute.Tensor,
        listed: cute.Tensor,
        stride: cutlass.Int32,
        rank: cutlass.Int32,
        top_k: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a row.

        Args:
            scores: Scores [R, N] fp32.
            lengths: Columns each row holds [R] int32, at most N.
            keys: Each row's listing [R, W] int64, written.
            listed: Keys each row listed [R] int32, written.
            stride: Columns between two samples.
            rank: The sample's rank the threshold stops at.
            top_k: Columns each row keeps.
            stream: The stream to launch on.
        """

        @cute.struct
        class Storage:
            """Shared memory of a block: the listed samples, their histogram, per-warp totals and scalars."""

            samples: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MOST_SAMPLES], 16]
            histogram: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, BINS], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, LISTING_WARPS], 16]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.kernel(scores, lengths, keys, listed, stride, rank, top_k, Storage).launch(
            grid=(scores.shape[0], 1, 1), block=[LISTING_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        lengths: cute.Tensor,
        keys: cute.Tensor,
        listed: cute.Tensor,
        stride: cutlass.Int32,
        rank: cutlass.Int32,
        top_k: cutlass.Int32,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Sample a row, list its columns at or above the sample's threshold in column order, or list it exactly.

        Args:
            scores: Scores [R, N] fp32.
            lengths: Columns each row holds [R] int32.
            keys: Each row's listing [R, W] int64.
            listed: Keys each row listed [R] int32.
            stride: Columns between two samples.
            rank: The sample's rank the threshold stops at.
            top_k: Columns each row keeps.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row_index, _, _ = cute.arch.block_idx()
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        samples = storage.samples.get_tensor(cute.make_layout(MOST_SAMPLES))
        histogram = storage.histogram.get_tensor(cute.make_layout(BINS))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(LISTING_WARPS))
        shared = storage.shared.get_tensor(cute.make_layout(4))
        row = scores[row_index, None]
        out = keys[row_index, None]
        capacity = keys.shape[1]
        length = cutlass.min(lengths[row_index], scores.shape[1])
        want = cutlass.min(top_k, length)

        # A row the listing holds whole lists every column at its own place
        placed = length
        if length <= capacity:
            for column in range(thread, length, LISTING_THREADS):
                out[column] = pack_key(row[column], column)
        else:
            sampled = cutlass.min(cute.ceil_div(length, stride), MOST_SAMPLES)
            for j in range(thread, sampled, LISTING_THREADS):
                samples[j] = score_word(row[j * stride])
            self.barrier.arrive_and_wait()
            prefix, _ = self.kth(
                samples,
                sampled,
                cutlass.min(rank, sampled),
                THRESHOLD_PASSES,
                False,
                histogram,
                warp_totals,
                shared,
            )
            threshold = cutlass.Uint32(prefix)

            # Each pass one scan places every thread's columns at or above the threshold
            placed = cutlass.Int32(0)
            values = cute.make_rmem_tensor(COLUMNS_PER_THREAD, cutlass.Float32)
            for first in range(0, length, LISTING_THREADS * COLUMNS_PER_THREAD):
                hits = cutlass.Int32(0)
                for c in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                    column = self.column_of(first, c, thread)
                    values[c] = -cutlass.Float32.inf
                    if column < length:
                        values[c] = row[column]
                for c in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                    column = self.column_of(first, c, thread)
                    hits = hits + cutlass.Int32(
                        column < length and cutlass.Uint32(score_word(values[c])) >= threshold
                    )
                before, total = self.exclusive_sum(hits, warp_totals)
                place = placed + before
                for c in cutlass.range_constexpr(COLUMNS_PER_THREAD):
                    column = self.column_of(first, c, thread)
                    if column < length and cutlass.Uint32(score_word(values[c])) >= threshold:
                        if place < capacity:
                            out[place] = pack_key(values[c], column)
                        place = place + 1
                placed = placed + total

            if placed < want or placed > capacity:
                self.list_exact(row, length, want, out, histogram, warp_totals, shared)
                placed = want
        if thread == 0:
            listed[row_index] = placed


def compiled(ordered: bool) -> Any:
    """Return the listing compiled once an order, taking torch tensors as they are.

    Args:
        ordered: List in column order.
    """
    if ordered in COMPILED:
        return COMPILED[ordered]
    rows, columns, width = cute.sym_int32(), cute.sym_int32(), cute.sym_int32()
    score_stride, key_stride = cute.sym_int64(), cute.sym_int64()
    arguments = (
        make_fake_tensor(cutlass.Float32, (rows, columns), stride=(score_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int64, (rows, width), stride=(key_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED[ordered] = cute.compile(RowListing(ordered), *arguments, options="--enable-tvm-ffi")
    return COMPILED[ordered]


def plan(top_k: int, columns: int) -> tuple[int, int, int]:
    """Return a row's sample stride, the sample's rank its threshold stops at, and the listing's width - the Triton plan.

    Args:
        top_k: Columns each row keeps.
        columns: Columns of the score matrix.
    """
    if columns <= WHOLE * top_k:
        return 1, 1, columns
    stride, rank, capacity, _ = listing_plan(top_k, columns)
    return stride, rank, capacity


def supports(scores: torch.Tensor, top_k: int) -> bool:
    """Tell whether the CuTeDSL listing takes this matrix - fp32 rows of unit column stride whose sample fits.

    Args:
        scores: Scores [R, N].
        top_k: Columns each row keeps.
    """
    stride, _, _ = plan(top_k, scores.shape[1])
    return (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        and -(-scores.shape[1] // stride) <= MOST_SAMPLES
    )


def list_rows(
    scores: torch.Tensor, lengths: torch.Tensor, top_k: int, ordered: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """List the columns of each row that hold its best `top_k`, with the keys they pack to.

    Args:
        scores: Scores [R, N] fp32, as `supports` takes them.
        lengths: Columns each row holds [R] int32, at most N.
        top_k: Columns each row keeps.
        ordered: List in column order, rather than slice by slice.

    Returns:
        Packed keys [R, W] int64, each row's best `top_k` and room beside them, W being N when the
        row is narrow and two to four times `top_k` when it is wide - the Triton top-k's widths. Keys past a row's
        count are not meant to be read.
        Keys each row listed [R] int32, at most W.

    Raises:
        ValueError: If the listing does not take this matrix.
    """
    if not supports(scores, top_k):
        raise ValueError(
            f"the CuTeDSL listing takes fp32 rows of unit column stride sampled at most {MOST_SAMPLES} times - got "
            f"{scores.dtype} {tuple(scores.shape)} with strides {scores.stride()} for a top-k of {top_k}"
        )
    rows, columns = scores.shape
    stride, rank, width = plan(top_k, columns)
    keys = torch.empty(rows, width, device=scores.device, dtype=torch.int64)
    listed = torch.empty(rows, device=scores.device, dtype=torch.int32)
    if rows:
        compiled(ordered)(scores, lengths, keys, listed, stride, rank, top_k)
    return keys, listed
