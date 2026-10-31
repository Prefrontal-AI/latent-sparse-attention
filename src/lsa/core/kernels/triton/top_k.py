"""The exact top-k of a score matrix - the best columns of each row, tie-free and the same on every run.

A row's scores are packed with their columns ([keys](keys.py)), so a tie in score goes to the lower column and no two
keys are equal: the k greatest are one set whatever order a kernel visits them in.

A pick comes in two forms. A superset listing is what a prefill chunk's read wants: the expansion by threshold
([expansion](expansion.py)) finds its stopping key itself and needs neither the top-k nor its order. A narrow row is
packed whole; a wide row's strided sample estimates the score about 1.4 to 2 times k of its columns sit at or above,
and every column at or above it is listed, in no order.

An exact pick is exactly k columns in column order - a beam's keep, whose next level's ties go to the lower column, and
the listings of a decode step and of a tree's last level, which it reads faster. Its sample estimates the score about
1.2 to 3 times k sit at or above, and the columns at or above it are written in column order, the score's bits beside
each. A program selects the k greatest of that listing by the k-th greatest score word - found a bit at a time from
the top - and takes the columns above it, and as many of the tied ones, lowest first, as the k leaves, written in
column order with no sort. A narrow row is selected where it lies.

Either estimate is only a guess, and the listing is checked against it. A row that lists fewer than its k, or more
than the buffer holds, is searched across the whole row instead and written exactly, so a guess that is off costs time
and never an answer.

Every shape is fixed by the inputs' shapes and nothing reads back to the host, so a pick can be captured in a CUDA
graph.
"""

import math

import torch
import triton
import triton.language as tl

from lsa.core.kernels.triton.keys import EMPTY, ordered_bits, pack_key

# Widest a row is listed whole, in k - a sample cannot rank so large a share of a row
WHOLE = 4
# Most columns a row samples
SAMPLE_MOST = 24576
# Samples a tight listing's estimate must rank - spread is 1 / sqrt(rank)
# => a listing misses when it falls a third under its target or past its capacity
TIGHT_RANK = 192
# Listing (columns expected in k, columns held in k, 1 / share of the rank a bin holds before a split)
TIGHT = (1.4, 2, 5)
LOOSE = (2.0, 4, 3)
SAMPLE_RANK = 256
LIST_BLOCK = 2048
SEARCH_BLOCK = 4096
# Bits of a column id - a row is at most this wide
ID_BITS = 24
# Widest a row an exact pick selects where it lies, in columns
NARROW = 4096
# Samples an exact pick's wide row can be estimated from, and the listing sizes tried for them, in k
SAMPLES = (2048, 4096, 8192, 16384)
TARGETS = (1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 2.0, 2.25, 2.5, 3.0)
# Neighbours a sample takes along - a sector costs the same to read as one score
RUN = 8
# Standard deviations of the listing's size the estimate must clear its k and its buffer by
SIGMAS = 4.0
# Least columns a listing holds when no sample plans it tightly
LOOSE_CAPACITY = 8192
# Columns an exact pick's program reads at a time
BLOCK = 4096
# Rows under which a row is split across programs, each reading SPLIT_LEAST columns at least
SPLIT_BELOW = 64
SPLIT_LEAST = 8192
SPLIT_MOST = 64
SPLIT_PROGRAMS = 512
WARPS = 8
SELECT_WARPS = 4
# Columns, warps and loads ahead of a row listed in one program
# => small programs keep more loads in flight
ROW_BLOCK = 1024
ROW_WARPS = 2
ROW_STAGES = 3
# Warps an exact pick's estimate runs with
ESTIMATE_WARPS = 4
EMPTY_KEY: tl.constexpr = tl.constexpr(EMPTY)
# Bits of the score word a whole-row search settles a pass, and the columns it reads at a time
DIGIT_BITS: tl.constexpr = tl.constexpr(4)
DIGIT_BLOCK: tl.constexpr = tl.constexpr(512)


@triton.jit
def _pack_scores(
    scores: tl.tensor,
    lengths: tl.tensor,
    keys: tl.tensor,
    score_stride: tl.tensor,
    key_stride: tl.tensor,
    column_count: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    """Pack each row's scores with their columns, the empty key past its length.

    Args:
        scores: Scores [R, columns] fp32.
        lengths: Columns each row sees [R] int32.
        keys: Packed keys [R, column_count] int64.
    """
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = column < column_count
    live = inside & (column < tl.load(lengths + row))
    score = tl.load(scores + row * score_stride + column, mask=live, other=0.0)
    packed = (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - column).to(tl.int64)
    tl.store(keys + row * key_stride + column, tl.where(live, packed, EMPTY_KEY), mask=inside)


@triton.jit
def _list_scores(
    scores: tl.tensor,
    lengths: tl.tensor,
    thresholds: tl.tensor,
    totals: tl.tensor,
    keys: tl.tensor,
    score_stride: tl.tensor,
    key_stride: tl.tensor,
    capacity: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    """List the packed keys of each row's columns at or above its threshold, in no order.

    A program's columns land in the order the atomics arrive in.
    The listing is a set, and nothing reads it in an order.

    Args:
        scores: Scores [R, columns] fp32.
        lengths: Columns each row sees [R] int32.
        thresholds: Each row's threshold [R] fp32.
        totals: Columns each row listed [R] int32, added to - above `capacity` on overflow.
        keys: Listings [R, capacity] int64.
    """
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = column < tl.load(lengths + row)
    score = tl.load(scores + row * score_stride + column, mask=inside, other=0.0)
    kept = inside & (score >= tl.load(thresholds + row))
    count = tl.sum(kept.to(tl.int32), 0)
    base = tl.atomic_add(totals + row, count, sem="relaxed")
    slot = base + tl.cumsum(kept.to(tl.int32), 0) - 1
    packed = (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - column).to(tl.int64)
    tl.store(keys + row * key_stride + slot, packed, mask=kept & (slot < capacity))


@triton.jit
def _bin_of(score: tl.tensor, floor: tl.tensor, width: tl.tensor, BINS: tl.constexpr) -> tl.tensor:
    """Return the bin of a score among BINS bins of `width` above `floor`, clamped to the ends."""
    return tl.minimum(tl.maximum((score - floor) / width, 0.0), BINS - 1.0).to(tl.int32)


@triton.jit
def _sample_thresholds(
    scores: tl.tensor,
    lengths: tl.tensor,
    thresholds: tl.tensor,
    sample_count: tl.tensor,
    stride: tl.tensor,
    rank: tl.tensor,
    score_stride: tl.tensor,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    REFINE: tl.constexpr,
) -> None:
    """Read each row's threshold off a histogram of its strided sample, split again where one bin holds many.

    The bins span the upper half between the sample's best and mean score.

    Args:
        scores: Scores [R, columns] fp32.
        lengths: Columns each row sees [R] int32.
        thresholds: Each row's threshold [R] fp32, -inf for a row the sample cannot rank.
        sample_count: Samples a row takes.
        stride: Columns between two samples.
        rank: Rank in the sample the threshold stops at.
    """
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    row_scores = scores + row * score_stride
    # The sample's best and mean score - the bins span the upper half between them, and a score of -inf is a column
    # the row cannot see
    best = -float("inf")
    total = 0.0
    seen = 0
    for start in range(0, sample_count, BLOCK):
        sample = start + tl.arange(0, BLOCK)
        live = (sample < sample_count) & (sample * stride < length)
        score = tl.load(row_scores + sample * stride, mask=live, other=-float("inf"))
        live = live & (score > -float("inf"))
        best = tl.maximum(best, tl.max(score, 0))
        total += tl.sum(tl.where(live, score, 0.0), 0)
        seen += tl.sum(live.to(tl.int32), 0)
    mean = total / tl.maximum(seen, 1)
    width = tl.maximum((best - mean) / BINS, 1e-30)
    edges = tl.arange(0, BINS)
    coarse = tl.zeros([BINS], dtype=tl.int32)
    for start in range(0, sample_count, BLOCK):
        sample = start + tl.arange(0, BLOCK)
        live = (sample < sample_count) & (sample * stride < length)
        score = tl.load(row_scores + sample * stride, mask=live, other=-float("inf"))
        coarse += tl.histogram(_bin_of(score, mean, width, BINS), BINS)
    at_or_above = tl.sum(coarse, 0) - tl.cumsum(coarse, 0) + coarse
    top = tl.max(tl.where(at_or_above >= rank, edges, -1), 0)
    inside = mean + top.to(tl.float32) * width
    threshold = tl.where(top > 0, inside, -float("inf"))
    # The bin the rank falls in, split again when it holds many samples - a few outliers stretch the bins, and the
    # lower edge of a bin of thousands would list them all
    in_bin = tl.sum(tl.where(edges == top, coarse, 0), 0)
    if (top > 0) & (in_bin * REFINE > rank):
        above_bin = tl.sum(tl.where(edges == top, at_or_above - coarse, 0), 0)
        narrow = width / BINS
        fine = tl.zeros([BINS], dtype=tl.int32)
        outside = 0
        for start in range(0, sample_count, BLOCK):
            sample = start + tl.arange(0, BLOCK)
            live = (sample < sample_count) & (sample * stride < length)
            score = tl.load(row_scores + sample * stride, mask=live, other=-float("inf"))
            member = _bin_of(score, mean, width, BINS) == top
            fine += tl.histogram(tl.where(member, _bin_of(score, inside, narrow, BINS), 0), BINS)
            outside += tl.sum((~member).to(tl.int32), 0)
        fine = tl.where(edges == 0, fine - outside, fine)
        fine_at_or_above = tl.sum(fine, 0) - tl.cumsum(fine, 0) + fine
        sub = tl.max(tl.where(fine_at_or_above >= rank - above_bin, edges, -1), 0)
        threshold = inside + tl.maximum(sub, 0).to(tl.float32) * narrow
    tl.store(thresholds + row, threshold)


@triton.jit
def _count_above(
    scores: tl.tensor,
    row: tl.tensor,
    length: tl.tensor,
    score_stride: tl.tensor,
    level: tl.tensor,
    BLOCK: tl.constexpr,
) -> tl.tensor:
    """Count the columns of a row whose score word, unsigned, is at or above `level`."""
    found = 0
    for start in range(0, length, BLOCK):
        column = start + tl.arange(0, BLOCK)
        inside = column < length
        score = tl.load(scores + row * score_stride + column, mask=inside, other=0.0)
        word = ordered_bits(score).to(tl.int64) + 2147483648
        found += tl.sum((inside & (word >= level)).to(tl.int32), 0)
    return found


@triton.jit
def _search_rows(
    scores: tl.tensor,
    lengths: tl.tensor,
    totals: tl.tensor,
    keys: tl.tensor,
    score_stride: tl.tensor,
    key_stride: tl.tensor,
    capacity: tl.tensor,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ID_BITS: tl.constexpr,
) -> None:
    """Search each row whose listing fell short or overflowed across the whole row and write its keys.

    A listing that holds the row's best k - or all of a row that has fewer - needs nothing more.

    Args:
        scores: Scores [R, columns] fp32.
        lengths: Columns each row sees [R] int32.
        totals: Columns each row listed [R] int32, rewritten for a row searched.
        keys: Listings [R, capacity] int64, rewritten for a row searched.
    """
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    total = tl.load(totals + row)
    # A listing that holds the row's best k - or all of a row that has fewer - needs nothing more
    searching = (total < tl.minimum(length, K)) | (total > capacity)
    if searching:
        for start in range(0, capacity, BLOCK):
            slot = start + tl.arange(0, BLOCK)
            tl.store(keys + row * key_stride + slot, EMPTY_KEY, mask=slot < capacity)
        tl.debug_barrier()
        # The k-th greatest score word, a bit at a time from the top
        one = tl.full([], 1, tl.int64)
        level = tl.zeros([], dtype=tl.int64)
        for i in range(32):
            trial = level | (one << tl.cast(31 - i, tl.int64))
            if _count_above(scores, row, length, score_stride, trial, BLOCK) >= K:
                level = trial
        above = _count_above(scores, row, length, score_stride, level + 1, BLOCK)
        # The columns tied at that score a row takes, the lowest first - the column below which fewer than `need` sit
        need = K - above
        cut = tl.zeros([], dtype=tl.int32)
        for i in range(ID_BITS):
            trial = cut | (1 << (ID_BITS - 1 - i))
            tied = 0
            for start in range(0, length, BLOCK):
                column = start + tl.arange(0, BLOCK)
                inside = column < length
                score = tl.load(scores + row * score_stride + column, mask=inside, other=0.0)
                word = ordered_bits(score).to(tl.int64) + 2147483648
                tied += tl.sum((inside & (word == level) & (column < trial)).to(tl.int32), 0)
            if tied < need:
                cut = trial
        filled = 0
        for start in range(0, length, BLOCK):
            column = start + tl.arange(0, BLOCK)
            inside = column < length
            score = tl.load(scores + row * score_stride + column, mask=inside, other=0.0)
            word = ordered_bits(score).to(tl.int64) + 2147483648
            chosen = inside & ((word > level) | ((word == level) & (column <= cut)))
            slot = filled + tl.cumsum(chosen.to(tl.int32), 0) - 1
            packed = (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - column).to(tl.int64)
            tl.store(keys + row * key_stride + slot, packed, mask=chosen & (slot < capacity))
            filled += tl.sum(chosen.to(tl.int32), 0)
        tl.store(totals + row, filled)


@triton.jit
def _word(score: tl.tensor) -> tl.tensor:
    """Return the score as an unsigned 32-bit word of the same order."""
    return _unsigned(ordered_bits(score))


@triton.jit
def _unsigned(ordered: tl.tensor) -> tl.tensor:
    """Return an order-preserving int32 as the unsigned word of the same order."""
    return (ordered ^ -2147483648).to(tl.uint32, bitcast=True)


@triton.jit
def _kth(word: tl.tensor, live: tl.tensor, rank: tl.tensor, LOW: tl.constexpr) -> tl.tensor:
    """Return the `rank`-th greatest of the live words down to bit LOW, found a bit at a time from the top.

    The words share their top bits from the least to the greatest, so only the bits below those are searched.
    """
    word = tl.where(live, word, 0)
    top = tl.max(word, 0)
    diff = top ^ tl.min(tl.where(live, word, 0xFFFFFFFF), 0)
    searched = tl.zeros([], dtype=tl.int32)
    for i in range(32):
        searched += ((diff >> tl.cast(i, tl.uint32)) != 0).to(tl.int32)
    level = ((top.to(tl.int64) >> searched.to(tl.int64)) << searched.to(tl.int64)).to(tl.uint32)
    one = tl.full([], 1, tl.uint32)
    for step in range(0, tl.maximum(searched - LOW, 0)):
        trial = level | (one << tl.cast(searched - 1 - step, tl.uint32))
        level = tl.where(tl.sum((word >= trial).to(tl.int32), 0) >= rank, trial, level)
    return (level >> LOW) << LOW


@triton.jit
def _threshold(
    scores: tl.tensor,
    length: tl.tensor,
    stride: tl.tensor,
    target: tl.tensor,
    SAMPLES: tl.constexpr,
    RUN: tl.constexpr,
    K: tl.constexpr,
) -> tl.tensor:
    """Return the score about `target` times K of a row's columns sit at or above, from a strided sample.

    Returns -inf for a row narrower than the listing, which lists every column.
    """
    sample = tl.arange(0, SAMPLES)
    # Runs of RUN neighbours, a run a sector of the row
    column = (sample // RUN) * stride + sample % RUN
    live = column < length
    word = _unsigned(ordered_bits(tl.load(scores + column, mask=live, other=0.0)))
    held = tl.sum(live.to(tl.int32), 0)
    # The sample's share of the row, times the columns wanted
    wanted = target * K
    rank = tl.minimum(tl.maximum((wanted * held / tl.maximum(length, 1)).to(tl.int32) + 1, 1), held)
    # To their top 19 bits - a granule of the row's scores is far below the listing's slack
    level = _kth(word, live, rank, 13)
    ordered = (level ^ 0x80000000).to(tl.int32, bitcast=True)
    least = tl.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered).to(tl.float32, bitcast=True)
    # The lowest word is no float, and a row narrower than the listing lists every column
    return tl.where((level == 0) | (wanted >= length), -float("inf"), least)


@triton.jit
def _estimate(
    scores: tl.tensor,
    lengths: tl.tensor,
    thresholds: tl.tensor,
    score_stride: tl.tensor,
    columns: tl.tensor,
    stride: tl.tensor,
    target: tl.tensor,
    SAMPLES: tl.constexpr,
    RUN: tl.constexpr,
    K: tl.constexpr,
) -> None:
    """Estimate each row's threshold from its strided sample."""
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.load(lengths + row), columns)
    tl.store(
        thresholds + row, _threshold(scores + row * score_stride, length, stride, target, SAMPLES, RUN, K)
    )


@triton.jit
def _list_slice(
    scores: tl.tensor,
    listing: tl.tensor,
    first: tl.tensor,
    last: tl.tensor,
    least: tl.tensor,
    filled: tl.tensor,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    COUNTING: tl.constexpr,
    STAGES: tl.constexpr,
) -> tl.tensor:
    """Write the columns of a row's slice at or above `least` from slot `filled` on, in column order.

    The score's bits go over the column as they lie. A COUNTING pass only counts.

    Returns:
        The slot after the last, past CAPACITY on overflow.
    """
    for start in tl.range(first, last, BLOCK, num_stages=STAGES):
        column = start + tl.arange(0, BLOCK)
        inside = column < last
        score = tl.load(scores + column, mask=inside, other=0.0)
        kept = inside & (score >= least)
        if not COUNTING:
            slot = filled + tl.cumsum(kept.to(tl.int32), 0) - 1
            packed = (score.to(tl.int32, bitcast=True).to(tl.int64) << 32) | column.to(tl.int64)
            tl.store(listing + slot, packed, mask=kept & (slot < CAPACITY))
        filled += tl.sum(kept.to(tl.int32), 0)
    return filled


@triton.jit
def _list(
    scores: tl.tensor,
    lengths: tl.tensor,
    thresholds: tl.tensor,
    sums: tl.tensor,
    totals: tl.tensor,
    listing: tl.tensor,
    score_stride: tl.tensor,
    listing_stride: tl.tensor,
    columns: tl.tensor,
    slice_size: tl.tensor,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
    COUNTING: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    """Write the columns of a row at or above its threshold, in column order.

    A row split across programs counts its slices first, and a slice starts where those before it end.

    Args:
        scores: Scores [R, columns] fp32.
        lengths: Columns each row sees [R] int32.
        thresholds: Each row's threshold [R] fp32.
        sums: Columns each slice counted [R, PARTS] int32.
        totals: Columns each row listed [R] int32.
        listing: Listings [R, CAPACITY] int64, the score's bits over the column.
    """
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    length = tl.minimum(tl.load(lengths + row), columns)
    first = tl.minimum(part * slice_size, length)
    last = tl.minimum(first + slice_size, length)
    row_scores = scores + row * score_stride
    row_listing = listing + row * listing_stride
    least = tl.load(thresholds + row)
    filled = tl.zeros([], dtype=tl.int32)
    if PARTS > 1:
        if not COUNTING:
            parts = tl.arange(0, PARTS)
            filled = tl.sum(tl.load(sums + row * PARTS + parts, mask=parts < part, other=0), 0)
    filled = _list_slice(
        row_scores, row_listing, first, last, least, filled, CAPACITY, BLOCK, COUNTING, STAGES
    )
    if COUNTING:
        tl.store(sums + row * PARTS + part, filled)
    elif part == tl.num_programs(1) - 1:
        # The last slice ends the listing
        tl.store(totals + row, filled)


@triton.jit
def _count_at_least(
    scores: tl.tensor, length: tl.tensor, level: tl.tensor, BLOCK: tl.constexpr, STRICT: tl.constexpr
) -> tl.tensor:
    """Count the columns of a row whose unsigned score word is at or above `level` - above it when STRICT."""
    found = 0
    for start in range(0, length, BLOCK):
        column = start + tl.arange(0, BLOCK)
        inside = column < length
        word = _word(tl.load(scores + column, mask=inside, other=0.0))
        if STRICT:
            found += tl.sum((inside & (word > level)).to(tl.int32), 0)
        else:
            found += tl.sum((inside & (word >= level)).to(tl.int32), 0)
    return found


@triton.jit
def _next_digit(
    scores: tl.tensor,
    length: tl.tensor,
    level: tl.tensor,
    want: tl.tensor,
    shift: tl.tensor,
    BLOCK: tl.constexpr,
) -> tl.tensor:
    """Return `level` with the greatest digit at `shift` that keeps `want` columns at or above it."""
    digits = tl.arange(0, 1 << DIGIT_BITS)
    trial = level | (digits.to(tl.uint32) << shift)
    found = tl.zeros([1 << DIGIT_BITS], dtype=tl.int32)
    for start in range(0, length, BLOCK):
        column = start + tl.arange(0, BLOCK)
        inside = column < length
        word = _word(tl.load(scores + column, mask=inside, other=0.0))
        found += tl.sum((inside[:, None] & (word[:, None] >= trial[None, :])).to(tl.int32), 0)
    return level | (tl.max(tl.where(found >= want, digits, 0), 0).to(tl.uint32) << shift)


@triton.jit
def _emit(
    scores: tl.tensor,
    out: tl.tensor,
    length: tl.tensor,
    level: tl.tensor,
    ties: tl.tensor,
    BLOCK: tl.constexpr,
    COLUMNS: tl.constexpr,
) -> tl.tensor:
    """Write the columns of a row above the word `level`, or at it among its first `ties`, in column order.

    Returns:
        The slots written.
    """
    filled = tl.zeros([], dtype=tl.int32)
    tied_before = tl.zeros([], dtype=tl.int32)
    for start in range(0, length, BLOCK):
        column = start + tl.arange(0, BLOCK)
        inside = column < length
        score = tl.load(scores + column, mask=inside, other=0.0)
        word = _word(score)
        tied = inside & (word == level)
        chosen = (inside & (word > level)) | (tied & (tied_before + tl.cumsum(tied.to(tl.int32), 0) <= ties))
        slot = filled + tl.cumsum(chosen.to(tl.int32), 0) - 1
        if COLUMNS:
            tl.store(out + slot, column.to(tl.int32), mask=chosen)
        else:
            tl.store(out + slot, pack_key(score, column), mask=chosen)
        filled += tl.sum(chosen.to(tl.int32), 0)
        tied_before += tl.sum(tied.to(tl.int32), 0)
    return filled


@triton.jit
def _pad(
    out: tl.tensor, filled: tl.tensor, WIDTH: tl.constexpr, PAD: tl.constexpr, COLUMNS: tl.constexpr
) -> None:
    """Pad the slots after the last one written, past a row's length."""
    slot = filled + tl.arange(0, PAD)
    if COLUMNS:
        tl.store(out + slot, -1, mask=slot < WIDTH)
    else:
        tl.store(out + slot, EMPTY_KEY, mask=slot < WIDTH)


@triton.jit
def _search(
    scores: tl.tensor,
    out: tl.tensor,
    length: tl.tensor,
    want: tl.tensor,
    BLOCK: tl.constexpr,
    COLUMNS: tl.constexpr,
) -> tl.tensor:
    """Write a row's best `want` exactly, settling the k-th greatest score word DIGIT_BITS a pass.

    Returns:
        The slots written.
    """
    level = tl.zeros([], dtype=tl.uint32)
    for i in range(32 // DIGIT_BITS):
        level = _next_digit(
            scores, length, level, want, (32 - DIGIT_BITS * (i + 1)).to(tl.uint32), DIGIT_BLOCK
        )
    above = _count_at_least(scores, length, level, BLOCK, True)
    return _emit(scores, out, length, level, want - above, BLOCK, COLUMNS)


@triton.jit
def _choose(
    score: tl.tensor,
    column: tl.tensor,
    live: tl.tensor,
    want: tl.tensor,
    out: tl.tensor,
    K: tl.constexpr,
    PAD: tl.constexpr,
    COLUMNS: tl.constexpr,
) -> None:
    """Write the `want` greatest of the live scores of a tile in column order, padded to K."""
    key = (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - column).to(tl.int64)
    word = _unsigned((key >> 32).to(tl.int32))
    level = _kth(word, live, want, 0)
    above = live & (word > level)
    tied = live & (word == level)
    chosen = above | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= want - tl.sum(above.to(tl.int32), 0)))
    place = tl.cumsum(chosen.to(tl.int32), 0) - 1
    if COLUMNS:
        tl.store(out + place, (0x7FFFFFFF - (key & 0xFFFFFFFF)).to(tl.int32), mask=chosen)
    else:
        tl.store(out + place, key, mask=chosen)
    _pad(out, want, K, PAD, COLUMNS)


@triton.jit
def _select(
    scores: tl.tensor,
    lengths: tl.tensor,
    listing: tl.tensor,
    totals: tl.tensor,
    out: tl.tensor,
    score_stride: tl.tensor,
    listing_stride: tl.tensor,
    out_stride: tl.tensor,
    columns: tl.tensor,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    LISTED: tl.constexpr,
    COLUMNS: tl.constexpr,
) -> None:
    """Write the k greatest of a row's listing - or of a narrow row itself - in column order, exactly k.

    A row whose listing fell short of its k, or overflowed, is searched across the whole row instead.
    That is done here for columns, and by `_repair` for keys, whose longer writes slow every row's select.
    """
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.load(lengths + row), columns)
    want = tl.minimum(length, K)
    slot = tl.arange(0, TILE)
    out_row = out + row * out_stride
    if LISTED:
        total = tl.load(totals + row)
        if (total >= want) & (total <= TILE):
            live = slot < total
            listed = tl.load(listing + row * listing_stride + slot, mask=live, other=0)
            score = (listed >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            _choose(score, (listed & 0xFFFFFFFF).to(tl.int32), live, want, out_row, K, PAD, COLUMNS)
        if COLUMNS:
            if (total < want) | (total > TILE):
                written = _search(scores + row * score_stride, out_row, length, want, BLOCK, COLUMNS)
                _pad(out_row, written, K, PAD, COLUMNS)
    else:
        live = slot < length
        score = tl.load(scores + row * score_stride + slot, mask=live, other=0.0)
        _choose(score, slot, live, want, out_row, K, PAD, COLUMNS)


@triton.jit
def _repair(
    scores: tl.tensor,
    lengths: tl.tensor,
    totals: tl.tensor,
    out: tl.tensor,
    score_stride: tl.tensor,
    out_stride: tl.tensor,
    columns: tl.tensor,
    K: tl.constexpr,
    PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Search a row whose listing fell short or overflowed across the whole row and write its keys."""
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.load(lengths + row), columns)
    want = tl.minimum(length, K)
    total = tl.load(totals + row)
    if (total < want) | (total > CAPACITY):
        out_row = out + row * out_stride
        written = _search(scores + row * score_stride, out_row, length, want, BLOCK, False)
        _pad(out_row, written, K, PAD, False)


def listing_plan(k: int, columns: int) -> tuple[int, int, int, int]:
    """Choose how a wide row of `columns` is sampled and listed for its best `k`.

    A tight listing - 1.4 times k expected, twice k held - needs a sample that ranks at least TIGHT_RANK columns, which
    a wide row's cap on the sample may not allow, and then the listing is looser: twice k expected, four times held.

    Args:
        k: Columns each row keeps.
        columns: Columns of the score matrix.

    Returns:
        Columns between two samples, the sample's rank the estimate stops at, the columns the listing holds, and the
        share of the rank, as 1 over it, that a bin of the sample may hold before it is split again.
    """
    least = triton.cdiv(columns, SAMPLE_MOST)
    target, capacity, refine = TIGHT if round(TIGHT[0] * k / least) >= TIGHT_RANK else LOOSE
    stride = max(least, round(target * k / SAMPLE_RANK))
    return stride, max(1, round(target * k / stride)), capacity * k, refine


def keep_plan(k: int, columns: int) -> tuple[int, int, float, int]:
    """Choose how a wide row of `columns` is sampled and listed for its best `k`.

    The listing's size is the sample's count at or above its rank-th greatest, scaled to the row, so it varies about
    its target by one over the root of that rank. The target is the least that keeps the listing above k, and the
    buffer the least that holds it, to SIGMAS deviations. A row too wide for that within four times k takes the
    largest sample and target and a buffer of at least LOOSE_CAPACITY, and is searched more often - the select holds
    the whole buffer in one tile, so it may not grow with the row.

    Args:
        k: Columns each row keeps.
        columns: Columns of the score matrix.

    Returns:
        The samples the estimate reads, the columns between two runs of them, the listing's expected size in k, and
        the columns the listing holds - a power of two.
    """
    fallback = (
        SAMPLES[-1],
        RUN * math.ceil(columns * RUN / SAMPLES[-1] / RUN),
        TARGETS[-1],
        min(triton.next_power_of_2(columns), max(4 * triton.next_power_of_2(k), LOOSE_CAPACITY)),
    )
    for samples in SAMPLES:
        stride = max(RUN, RUN * math.ceil(columns * RUN / samples / RUN))
        held = min(samples, math.ceil(columns / stride) * RUN)
        for target in TARGETS:
            spread = (target * k * held / columns) ** -0.5
            capacity = triton.next_power_of_2(math.ceil(target * k * (1 + SIGMAS * spread)))
            if 1 - SIGMAS * spread >= 1 / target and capacity <= 4 * triton.next_power_of_2(k):
                return samples, stride, target, capacity
    return fallback


class TopK:
    """The best k columns of each row of a score matrix - listed for a read, and exactly k for a beam's keep."""

    def __init__(
        self, k: int, rows: int | None = None, device: str | torch.device = "cuda", exact: bool = False
    ) -> None:
        """Size the buffers a captured graph writes to, when asked.

        Args:
            k: Columns each row keeps.
            rows: Most rows a call lists for, to hold the listing in buffers sized once - a call allocates its own
                when None. A check that reads the listing a CUDA graph wrote needs them.
            device: Where the buffers live.
            exact: Whether a listing is exactly each row's best k in column order, rather than a superset in no order.
        """
        self.k = k
        self.exact = exact
        width = k if exact else WHOLE * k
        self.held = (
            None
            if rows is None
            else (
                torch.empty(rows, width, device=device, dtype=torch.int64),
                torch.empty(rows, device=device, dtype=torch.int32),
            )
        )
        self.keys = torch.empty(0, 0, dtype=torch.int64)
        self.counts = torch.empty(0, dtype=torch.int32)

    def release(self) -> None:
        """Drop the last listing, so its allocation goes back to the allocator for the next call to take.

        A listing in buffers sized once is kept, since a check reads it.
        """
        if self.held is None:
            self.keys = torch.empty(0, 0, dtype=torch.int64)
            self.counts = torch.empty(0, dtype=torch.int32)

    def listed(self, scores: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """List the columns of each row that hold its best k, with the keys they pack to.

        Args:
            scores: Scores [R, N] fp32, at most 2^24 wide.
            lengths: Columns each row holds [R] int32, at most N.

        Returns:
            Packed keys [R, W] int64. An exact listing is each row's best k in column order, W the lesser of k and N.
            A superset holds every row's best k and room beside them in no order, W being N when the row is narrow
            and two to four times k when it is wide. Keys past a row's count are not meant to be read.
            Keys each row listed [R] int32, at most W.
        """
        rows, columns = scores.shape
        if not self.exact:
            whole = columns <= WHOLE * self.k
            width = columns if whole else listing_plan(self.k, columns)[2]
        else:
            width = min(self.k, columns)
        if self.held is None:
            self.keys = torch.empty(rows, width, device=scores.device, dtype=torch.int64)
            self.counts = torch.empty(rows, device=scores.device, dtype=torch.int32)
        else:
            self.keys, self.counts = self.held[0][:rows, :width], self.held[1][:rows]
        if self.exact:
            torch.clamp(lengths, max=width, out=self.counts)
            self.select(scores, lengths, self.keys, columns=False)
        else:
            self.list_superset(scores, lengths)
        return self.keys, self.counts

    def list_superset(self, scores: torch.Tensor, lengths: torch.Tensor) -> None:
        """List every row's best k and room beside them, in no order, into `keys` [R, W] and `counts` [R]."""
        rows, columns = scores.shape
        whole = columns <= WHOLE * self.k
        stride, rank, capacity, refine = (1, 0, columns, 0) if whole else listing_plan(self.k, columns)
        if whole:
            _pack_scores[(rows, triton.cdiv(columns, LIST_BLOCK))](
                scores,
                lengths,
                self.keys,
                scores.stride(0),
                self.keys.stride(0),
                columns,
                BLOCK=LIST_BLOCK,
                num_warps=4,
            )
            self.counts.copy_(lengths)
            return
        thresholds = torch.empty(rows, device=scores.device)
        _sample_thresholds[(rows,)](
            scores,
            lengths,
            thresholds,
            triton.cdiv(columns, stride),
            stride,
            rank,
            scores.stride(0),
            BINS=64,
            BLOCK=2048,
            REFINE=refine,
            num_warps=4,
        )
        self.counts.zero_()
        _list_scores[(rows, triton.cdiv(columns, LIST_BLOCK))](
            scores,
            lengths,
            thresholds,
            self.counts,
            self.keys,
            scores.stride(0),
            self.keys.stride(0),
            capacity,
            BLOCK=LIST_BLOCK,
            num_warps=4,
        )
        _search_rows[(rows,)](
            scores,
            lengths,
            self.counts,
            self.keys,
            scores.stride(0),
            self.keys.stride(0),
            capacity,
            K=self.k,
            BLOCK=SEARCH_BLOCK,
            ID_BITS=ID_BITS,
            num_warps=8,
        )

    def select(self, scores: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor, columns: bool) -> None:
        """Write each row's best k in column order into `out` [R, k] - columns or packed keys, -1 or empty past it."""
        rows, width = scores.shape
        k = out.shape[1]
        pad = triton.next_power_of_2(k)
        if width <= NARROW:
            _select[(rows,)](
                scores,
                lengths,
                scores,
                lengths,
                out,
                scores.stride(0),
                0,
                out.stride(0),
                width,
                K=k,
                PAD=pad,
                TILE=triton.next_power_of_2(width),
                BLOCK=BLOCK,
                LISTED=False,
                COLUMNS=columns,
                num_warps=SELECT_WARPS,
            )
            return
        samples, stride, target, capacity = keep_plan(k, width)
        thresholds = torch.empty(rows, device=scores.device, dtype=torch.float32)
        _estimate[(rows,)](
            scores,
            lengths,
            thresholds,
            scores.stride(0),
            width,
            stride,
            target,
            SAMPLES=samples,
            RUN=RUN,
            K=k,
            num_warps=ESTIMATE_WARPS,
        )
        split = rows < SPLIT_BELOW and width >= 2 * SPLIT_LEAST
        parts = (
            min(SPLIT_MOST, max(2, SPLIT_PROGRAMS // rows), triton.cdiv(width, SPLIT_LEAST)) if split else 1
        )
        block, warps, stages = (BLOCK, WARPS, 1) if split else (ROW_BLOCK, ROW_WARPS, ROW_STAGES)
        slice_size = triton.cdiv(triton.cdiv(width, parts), block) * block
        listing = torch.empty(rows, capacity, device=scores.device, dtype=torch.int64)
        totals = torch.empty(rows, device=scores.device, dtype=torch.int32)
        sums = torch.empty(rows, triton.next_power_of_2(parts), device=scores.device, dtype=torch.int32)
        for counting in (True, False) if split else (False,):
            _list[(rows, parts)](
                scores,
                lengths,
                thresholds,
                sums,
                totals,
                listing,
                scores.stride(0),
                listing.stride(0),
                width,
                slice_size,
                CAPACITY=capacity,
                BLOCK=block,
                PARTS=triton.next_power_of_2(parts),
                COUNTING=counting,
                STAGES=stages,
                num_warps=warps,
            )
        _select[(rows,)](
            scores,
            lengths,
            listing,
            totals,
            out,
            scores.stride(0),
            listing.stride(0),
            out.stride(0),
            width,
            K=k,
            PAD=pad,
            TILE=capacity,
            BLOCK=BLOCK,
            LISTED=True,
            COLUMNS=columns,
            num_warps=SELECT_WARPS,
        )
        if not columns:
            _repair[(rows,)](
                scores,
                lengths,
                totals,
                out,
                scores.stride(0),
                out.stride(0),
                width,
                K=k,
                PAD=pad,
                CAPACITY=capacity,
                BLOCK=BLOCK,
                num_warps=WARPS,
            )

    def best(self) -> torch.Tensor:
        """Return the keys of the last listing's best k columns.

        Returns:
            Packed keys [R, k] int64, the best score first and a tie to the lower column, the empty key past a row's
            length.
        """
        rows, width = self.keys.shape
        held = min(self.k, width)
        listed = torch.arange(width, device=self.keys.device)[None] < self.counts[:, None]
        best = torch.full((rows, self.k), EMPTY, device=self.keys.device, dtype=torch.int64)
        best[:, :held] = self.keys.masked_fill(~listed, EMPTY).topk(held, dim=1).values
        return best

    def __call__(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pick each row's best columns within its length.

        Args:
            scores: Scores [R, N] fp32.
            lengths: Columns each row holds [R] int32, at most N.

        Returns:
            The k best columns of each row [R, k] int32 in ascending column order, -1 past the last - fewer than k
            when the row holds fewer.
        """
        kept = torch.empty(scores.shape[0], self.k, device=scores.device, dtype=torch.int32)
        self.select(scores, lengths, kept, columns=True)
        return kept


def top_keys(scores: torch.Tensor, lengths: torch.Tensor, k: int) -> torch.Tensor:
    """Pick the best `k` columns of each row within its length, as packed keys.

    Args:
        scores: Scores [R, N] fp32.
        lengths: Columns each row holds [R] int32, at most N.
        k: Columns each row keeps.

    Returns:
        Packed keys [R, k] int64, the best score first and a tie to the lower column, the empty key past the row's
        length.
    """
    pick = TopK(k)
    pick.listed(scores, lengths)
    return pick.best()
