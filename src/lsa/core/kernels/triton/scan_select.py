"""The top-p buckets of each index query's centroid scores, without writing the score matrix.

A long prefill chunk's bf16 centroid scan is a [queries, buckets] fp32 matrix - 43 GB for 8,192 queries over
1.31M buckets - that a top-k then reads back. Here a strided sample of the buckets estimates each query's
threshold; the tensor-core scan compares every score with it and lists the buckets at or above it - unshaped
bf16 rows in shared memory ([bucket_scan](../cutedsl/bucket_scan.py)), the rest as one bit a bucket, 32 buckets to a word, expanded
after; the listed buckets are scored again one query at a time; and an exact top-k runs over them.

The scores the top-p are chosen by are that second, per-query scoring - an fp32 dot in a fixed order, which
scores every bucket of a row the same way, bit for bit, when a check or a fallback needs all of them. The scan
keeps every bucket within MARGIN of the query's norm below the threshold, more than the two scorings ever
differ by, so a bucket scoring at or above the threshold is always listed; a row with at least `top_p` of
those has its exact top-p among them. The bucket centroids are unit rows and the index queries of any norm - a
score is at most the query's norm, which scales the margin and the threshold's histogram alike. A row that lists
more than its buffer holds - buckets that tie at the threshold, as a repeated text makes - lists again into a
buffer as wide as it needs, and a row with fewer than `top_p` at the threshold is scored in full.

A score travels packed with its bucket in one int64 ([keys](keys.py)), so one top-k orders by score and breaks a
tie toward the lower bucket, and the result does not depend on the order the survivors were listed in.

Shaped buckets are scored by their estimated best key ([shape](shape.py)). The sample is shaped too, so the threshold
estimates the shaped scores. The scan scores each bucket's spread on the tensor cores as well - the bf16 query against
the expanded sign directions - and widens each projection by how far the query's planes sit from the query, so every
bucket whose shaped score against the planes reaches the threshold is still listed; the second scoring adds the spread
against the planes, as every other shaped read does.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lsa.core.kernels.triton.keys import EMPTY, pack_key, unpack_scores
from lsa.core.kernels.triton.shape import (
    QUERY_BITS,
    ShapeTable,
    SignQuery,
    _spread,
    add_planes_spread,
    basis_words,
    plane_gaps,
    sign_query,
)

# Visible buckets from which flat's prefill read takes the fused select - under it a dense product and a top-k are
# cheaper
SCAN_SELECT_BUCKETS = 32768
# Columns of a beam step's score row from which the level takes the fused select - under it a top-k over the
# materialized scores is cheaper
SCAN_SELECT_BEAM_WIDTH = 196608
# Survivors a row aims for and holds, in top-p
TARGET = 1.5
CAPACITY = 3
# Sampled buckets a row's threshold rests on - its relative spread is about one over the root
SAMPLE_RANK = 64
# Below a bin edge the threshold sits, as a share of the query's norm - more than a sample's tensor-core score
# and the per-query score differ by
EDGE_SLACK: tl.constexpr = tl.constexpr(2**-14)
# Below the threshold the scan still keeps, as a share of the query's norm - over unit bf16 bucket_centroids of 64
# the tensor-core and the per-query fp32 dot differ by under 1e-5 of it
MARGIN = 1e-4
# A shaped score the scan widens by this share of its spread, past the fp32 rounding of the planes' variance
SPREAD_SLACK: tl.constexpr = tl.constexpr(1 + 2**-10)
# Buckets a row's scoring reads at a time - one layout for listed and dense rows keeps their scores equal
SCORE_BLOCK = 128
# Buckets one program of a dense row scores, and rows scored densely at once
DENSE_SPAN = 8192
DENSE_ROWS = 256
# Widest a row lists again, in top-p, before it is scored in full, and entries one wide listing holds
WIDE = 64
WIDE_ENTRIES = 2**26


@triton.jit
def _survivor_words(
    index_queries: tl.tensor,
    bucket_centroids: tl.tensor,
    visible_counts: tl.tensor,
    thresholds: tl.tensor,
    word_counts: tl.tensor,
    survivor_counts: tl.tensor,
    listed_words: tl.tensor,
    listed_bits: tl.tensor,
    query_count: tl.tensor,
    bucket_count: tl.tensor,
    word_count: tl.tensor,
    capacity: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    plane_gaps: tl.tensor,
    deviations: tl.tensor,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    WORDS: tl.constexpr,
    SHAPED: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
) -> None:
    """Scan a block of buckets against each query's threshold and list the words that hold a survivor.

        A survivor is a bucket scoring at or above the threshold, widened by MARGIN.
    Shaped buckets add their spread
        against the query's planes, widened by `plane_gaps`.

        Args:
            index_queries: Index queries [query_count, D] bf16.
            bucket_centroids: Bucket centroids [bucket_count, D] bf16.
            visible_counts: Buckets each query sees [query_count] int32.
            thresholds: Each query's threshold [query_count] fp32.
            word_counts: Words each query listed [query_count] int32, added to.
            survivor_counts: Survivors each query found [query_count] int32, added to.
            listed_words: Index of each listed word [query_count, capacity] int32.
            listed_bits: Survivor bits of each listed word [query_count, capacity] int32, one bit a bucket.
            word_count: Words of 32 buckets a row spans.
            capacity: Words a row lists at most.
            basis: Sign directions of the buckets, packed 32 to a word - read only if SHAPED.
            variances: Variances of the buckets' directions [bucket_count, RHO] fp32 - read only if SHAPED.
            floor: Variance outside the directions [bucket_count] fp32 - read only if SHAPED.
            plane_gaps: How far a query's planes project from it along a unit direction [query_count] fp32.
            deviations: Standard deviations a shaped score adds.
    """
    rows = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    live = rows < query_count
    dims = tl.arange(0, D)
    query = tl.load(
        index_queries + rows[:, None].to(tl.int64) * D + dims[None, :], mask=live[:, None], other=0.0
    )
    length = tl.load(visible_counts + rows, mask=live, other=0)
    threshold = tl.load(thresholds + rows, mask=live, other=float("inf"))
    # How far a query's planes may project from the query, along any unit direction
    gap = tl.load(plane_gaps + rows, mask=live, other=0.0)
    bit = tl.arange(0, 32)
    slots = tl.arange(0, WORDS)
    first = tl.program_id(1) * WORDS
    packed = tl.zeros([BLOCK_Q, WORDS], dtype=tl.int32)
    pair = tl.arange(0, 32 * RHOP)
    direction = pair % RHOP
    ranks = tl.arange(0, RHOP)
    for word in range(WORDS):
        buckets = (first + word) * 32 + bit
        centroid = tl.load(
            bucket_centroids + buckets[:, None].to(tl.int64) * D + dims[None, :],
            mask=buckets[:, None] < bucket_count,
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(centroid))
        if SHAPED:
            # One column per (bucket, direction) pair, the signs expanded once and read by every query of the tile
            pair_bucket = (first + word) * 32 + pair // RHOP
            pair_valid = (pair_bucket < bucket_count) & (direction < RHO)
            basis_row = tl.where(pair_valid, pair_bucket.to(tl.int64) * RHO + direction, 0)
            words = tl.load(
                basis + basis_row[:, None] * (D // 32) + tl.arange(0, D // 32)[None, :],
                mask=pair_valid[:, None],
                other=0,
            )
            bits = tl.reshape((words[:, :, None] >> bit[None, None, :]) & 1, [32 * RHOP, D])
            signs = tl.where(bits != 0, 1.0, -1.0).to(query.dtype)
            projection = tl.abs(tl.dot(query, tl.trans(signs)) * tl.rsqrt(float(D))) + gap[:, None]
            projection = tl.where(pair_valid[None, :], projection, 0.0)
            squared = tl.reshape(projection * projection, [BLOCK_Q, 32, RHOP])
            inside = buckets < bucket_count
            shape_floor = tl.load(floor + buckets, mask=inside, other=0.0)
            shape_variances = tl.load(
                variances + buckets[:, None].to(tl.int64) * RHO + ranks[None, :],
                mask=inside[:, None] & (ranks < RHO)[None, :],
                other=0.0,
            )
            # The planes' variance is floor + sum_j (lambda_j - floor) p_j^2, at most this with |p_j| widened by
            # the gap - and a hair over, past fp32 rounding
            excess = tl.maximum(shape_variances - shape_floor[:, None], 0.0)
            variance = shape_floor[None, :] + tl.sum(squared * excess[None, :, :], axis=2)
            scores += deviations * tl.sqrt(tl.maximum(variance, 0.0)) * SPREAD_SLACK
        kept = ((buckets[None, :] < length[:, None]) & (scores >= threshold[:, None])).to(tl.int32)
        # Distinct bits sum to their union
        mask = tl.sum(kept << bit[None, :], 1)
        packed = tl.where(slots[None, :] == word, mask[:, None], packed)
    kept_count = tl.sum(libdevice.popc(packed), 1)
    # The row's nonzero words, appended where its count says - one atomic a row
    hit = ((packed != 0) & (first + slots[None, :] < word_count)).to(tl.int32)
    count = tl.sum(hit, 1)
    base = tl.atomic_add(word_counts + rows, count, mask=count > 0, sem="relaxed")
    tl.atomic_add(survivor_counts + rows, kept_count, mask=kept_count > 0, sem="relaxed")
    slot = base[:, None] + tl.cumsum(hit, 1) - 1
    at = rows[:, None].to(tl.int64) * capacity + slot
    held = (hit > 0) & (slot < capacity)
    tl.store(listed_words + at, first + slots[None, :], mask=held)
    tl.store(listed_bits + at, packed, mask=held)


@triton.jit
def _survivor_buckets(
    word_counts: tl.tensor,
    listed_words: tl.tensor,
    listed_bits: tl.tensor,
    buckets: tl.tensor,
    capacity: tl.tensor,
    EXPAND: tl.constexpr,
) -> None:
    """Expand each row's listed survivor words into bucket numbers.

    Args:
        word_counts: Words each row listed [R] int32.
        listed_words: Index of each listed word [R, capacity] int32.
        listed_bits: Survivor bits of each listed word [R, capacity] int32.
        buckets: Survivor buckets [R, capacity] int32.
        capacity: Survivors a row holds at most.
    """
    row = tl.program_id(0).to(tl.int64)
    listed = tl.minimum(tl.load(word_counts + row), capacity)
    filled = 0
    for start in range(0, listed, EXPAND):
        slot = start + tl.arange(0, EXPAND)
        index = tl.load(listed_words + row * capacity + slot, mask=slot < listed, other=0)
        word = tl.load(listed_bits + row * capacity + slot, mask=slot < listed, other=0)
        # Nearly every word holds one survivor
        # => peel each word's lowest set bit, as often as the fullest word has bits, not test all 32
        count = libdevice.popc(word)
        at = filled + tl.cumsum(count, 0) - count
        for _ in range(tl.max(count, 0)):
            bit = libdevice.ffs(word) - 1
            tl.store(buckets + row * capacity + at, index * 32 + bit, mask=(word != 0) & (at < capacity))
            at += 1
            word = word & (word - 1)
        filled += tl.sum(count, 0)


@triton.jit
def _score_buckets(
    index_queries: tl.tensor,
    bucket_centroids: tl.tensor,
    buckets: tl.tensor,
    counts: tl.tensor,
    thresholds: tl.tensor,
    keys: tl.tensor,
    reached: tl.tensor,
    key_stride: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    planes: tl.tensor,
    steps: tl.tensor,
    totals: tl.tensor,
    deviations: tl.tensor,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    SPAN: tl.constexpr,
    DENSE: tl.constexpr,
    SHAPED: tl.constexpr,
    DKC: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
    PLANES: tl.constexpr,
) -> None:
    """Score a row's buckets with an fp32 dot in a fixed order and pack each score with its bucket.

    A listed row scores its survivors and reports how many reached the threshold.
    A dense row scores a span of its visible buckets.
    Shaped buckets add their spread against the query's planes.

    Args:
        index_queries: Index queries [R, D] bf16.
        bucket_centroids: Bucket centroids [N, D] bf16.
        buckets: Survivor buckets [R, key_stride] int32 - read only if not DENSE.
        counts: Buckets to score a row [R] int32, its visible count if DENSE, its survivor count otherwise.
        thresholds: Each row's threshold [R] fp32.
        keys: Scores packed with their buckets [R, key_stride] int64.
        reached: Buckets scoring at or above the threshold [R] int32 - written only if not DENSE.
        key_stride: Columns a row of `keys` holds.
        basis: Sign directions of the buckets, packed 32 to a word - read only if SHAPED.
        variances: Variances of the buckets' directions [N, RHO] fp32 - read only if SHAPED.
        floor: Variance outside the directions [N] fp32 - read only if SHAPED.
        planes: The queries' sign planes, packed 32 to a word - read only if SHAPED.
        steps: The queries' plane steps - read only if SHAPED.
        totals: The queries' plane totals [R] fp32 - read only if SHAPED.
        deviations: Standard deviations a shaped score adds.
    """
    row = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, D)
    query = tl.load(index_queries + row * D + dims).to(tl.float32)
    threshold = tl.load(thresholds + row)
    directions = tl.arange(0, RHOP)
    kept = directions < RHO
    words = tl.arange(0, D // 32)
    # A dense row's counts are its visible_counts, split into spans; a listed row's are its survivor_counts
    if DENSE:
        begin = tl.program_id(1) * SPAN
        count = tl.minimum(tl.load(counts + row), begin + SPAN)
    else:
        begin = 0
        count = tl.minimum(tl.load(counts + row), key_stride)
    high = 0
    for start in range(begin, count, BLOCK):
        slot = start + tl.arange(0, BLOCK)
        live = slot < count
        if DENSE:
            bucket = slot
        else:
            bucket = tl.load(buckets + row * key_stride + slot, mask=live, other=0)
        centroid = tl.load(
            bucket_centroids + bucket[:, None].to(tl.int64) * D + dims[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        score = tl.sum(centroid * query[None, :], 1)
        if SHAPED:
            shaped = bucket.to(tl.int64)
            word = tl.load(
                basis
                + ((shaped[:, None] * RHO + directions[None, :]) * (D // 32))[:, :, None]
                + words[None, None, :],
                mask=live[:, None, None] & kept[None, :, None],
                other=0,
            )
            shape_variances = tl.load(
                variances + shaped[:, None] * RHO + directions[None, :],
                mask=live[:, None] & kept[None, :],
                other=0.0,
            )
            spread = _spread(
                word,
                planes + row * PLANES * (D // 32),
                tl.load(steps + row * (D // DKC) + words * 32 // DKC),
                tl.load(totals + row),
                shape_variances,
                tl.load(floor + shaped, mask=live, other=0.0),
                kept[None, :],
                D,
                PLANES,
            )
            score = score + deviations * spread
        high += tl.sum((live & (score >= threshold)).to(tl.int32), 0)
        tl.store(keys + row * key_stride + slot, pack_key(score, bucket), mask=live)
    if not DENSE:
        tl.store(reached + row, high)


def shape_arguments(
    index_queries: torch.Tensor, shapes: ShapeTable | None, query_planes: SignQuery | None
) -> dict[str, object]:
    """The shape arguments `_score_buckets` takes - stand-ins, never read, for unshaped buckets.

    Args:
        index_queries: Index queries [R, d] bf16.
        shapes: The buckets' shapes, or None.
        query_planes: The queries' planes, split when None and the buckets are shaped.

    Returns:
        The keyword arguments.
    """
    d = index_queries.shape[1]
    if shapes is None:
        none = torch.zeros(1, device=index_queries.device, dtype=torch.int32)
        unread = torch.zeros(1, device=index_queries.device)
        return dict(
            basis=none,
            variances=unread,
            floor=unread,
            planes=none,
            steps=unread,
            totals=unread,
            deviations=0.0,
            SHAPED=False,
            DKC=min(128, d),
            RHO=0,
            RHOP=1,
            PLANES=QUERY_BITS,
        )
    planes = query_planes or sign_query(index_queries)
    rank = shapes.variances.shape[1]
    return dict(
        basis=basis_words(shapes.basis),
        variances=shapes.variances,
        floor=shapes.floor,
        planes=planes.planes,
        steps=planes.steps,
        totals=planes.totals,
        deviations=shapes.deviations,
        SHAPED=True,
        DKC=min(128, d),
        RHO=rank,
        RHOP=triton.next_power_of_2(max(rank, 1)),
        PLANES=QUERY_BITS,
    )


def scan_arguments(index_queries: torch.Tensor, shapes: ShapeTable | None) -> dict[str, object]:
    """The shape arguments `_survivor_words` takes - stand-ins, never read, for unshaped buckets.

    Args:
        index_queries: Index queries [R, d] bf16.
        shapes: The buckets' shapes, or None.

    Returns:
        The keyword arguments.
    """
    if shapes is None:
        unread = torch.zeros(1, device=index_queries.device)
        return dict(
            basis=torch.zeros(1, device=index_queries.device, dtype=torch.int32),
            variances=unread,
            floor=unread,
            plane_gaps=unread,
            deviations=0.0,
            SHAPED=False,
            RHO=0,
            RHOP=1,
        )
    rank = shapes.variances.shape[1]
    return dict(
        basis=basis_words(shapes.basis),
        variances=shapes.variances,
        floor=shapes.floor,
        plane_gaps=plane_gaps(index_queries),
        deviations=shapes.deviations,
        SHAPED=True,
        RHO=rank,
        RHOP=triton.next_power_of_2(max(rank, 1)),
    )


def dense_keys(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    width: int | None = None,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
) -> torch.Tensor:
    """Score every bucket each row sees and pack it with its bucket, at its own column.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        width: Columns of the result, at least N - N when None.
        shapes: The buckets' shapes, row for row, to score each by its estimated best key - None for the centroid.
        query_planes: The queries' planes, when shaped and already split.

    Returns:
        Packed keys [R, width] int64, the empty key past a row's length.
    """
    rows, d = index_queries.shape
    keys = torch.full(
        (rows, width or bucket_centroids.shape[0]), EMPTY, device=index_queries.device, dtype=torch.int64
    )
    unread = torch.zeros(rows, device=index_queries.device, dtype=torch.int32)
    _score_buckets[(rows, triton.cdiv(bucket_centroids.shape[0], DENSE_SPAN))](
        index_queries,
        bucket_centroids,
        unread,
        visible_counts,
        torch.full((rows,), torch.inf, device=index_queries.device),
        keys,
        unread,
        keys.stride(0),
        D=d,
        BLOCK=SCORE_BLOCK,
        SPAN=DENSE_SPAN,
        DENSE=True,
        num_warps=4,
        **shape_arguments(index_queries, shapes, query_planes),
    )
    return keys


def dense_scores(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    shapes: ShapeTable | None = None,
) -> torch.Tensor:
    """Score every bucket each row sees, as the top-p are chosen - what a check holds them to.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        shapes: The buckets' shapes, or None for the centroid.

    Returns:
        Scores [R, N] fp32, -inf past a row's length.
    """
    scores = torch.empty(index_queries.shape[0], bucket_centroids.shape[0], device=index_queries.device)
    for rows in torch.arange(index_queries.shape[0], device=index_queries.device).split(DENSE_ROWS):
        keys = dense_keys(index_queries[rows], bucket_centroids, visible_counts[rows], shapes=shapes)
        scores[rows] = unpack_scores(keys).masked_fill(keys == EMPTY, -torch.inf)
    return scores


def word_listing(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    thresholds: torch.Tensor,
    capacity: int,
    shapes: ShapeTable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """List the buckets each row scores at or above its threshold by survivor words, then expand the words.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        thresholds: Score each row lists at or above [R] fp32, the margin already taken off.
        capacity: Buckets a row's listing holds.
        shapes: The buckets' shapes, row for row, or None for the centroid.

    Returns:
        The listing [R, capacity] int32 - a row's first `capacity` survivors, unset after.
        Buckets each row listed [R] int32 - above capacity for a row whose listing overflowed.
    """
    rows, d = index_queries.shape
    bucket_count = bucket_centroids.shape[0]
    word_count = triton.cdiv(bucket_count, 32)
    device = index_queries.device
    word_counts = torch.zeros(rows, device=device, dtype=torch.int32)
    survivor_counts = torch.zeros(rows, device=device, dtype=torch.int32)
    listed_words = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    listed_bits = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    buckets = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    block_q, words_per_program = 128, 16
    _survivor_words[(triton.cdiv(rows, block_q), triton.cdiv(word_count, words_per_program))](
        index_queries,
        bucket_centroids,
        visible_counts,
        thresholds,
        word_counts,
        survivor_counts,
        listed_words,
        listed_bits,
        rows,
        bucket_count,
        word_count,
        capacity,
        D=d,
        BLOCK_Q=block_q,
        WORDS=words_per_program,
        num_warps=4,
        **scan_arguments(index_queries, shapes),
    )
    _survivor_buckets[(rows,)](
        word_counts, listed_words, listed_bits, buckets, capacity, EXPAND=128, num_warps=4
    )
    return buckets, survivor_counts


def listed_keys(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    thresholds: torch.Tensor,
    capacity: int,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """List the buckets each row scores near its threshold, score them and pack each with its bucket.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        thresholds: Score each row's top-p is expected at or above [R] fp32.
        capacity: Buckets a row's buffer holds.
        shapes: The buckets' shapes, row for row, or None for the centroid.
        query_planes: The queries' planes, when shaped and already split.

    Returns:
        Packed keys [R, capacity] int64 of the first `capacity` listed buckets, the empty key after.
        Buckets each row listed [R] int32 - above capacity for a row whose buffer overflowed.
        Listed buckets each row scored at or above its threshold [R] int32.
    """
    rows, d = index_queries.shape
    device = index_queries.device
    shifted = thresholds - MARGIN * index_queries.float().norm(dim=1)
    buckets, survivor_counts = word_listing(
        index_queries, bucket_centroids, visible_counts, shifted, capacity, shapes
    )
    keys = torch.full((rows, capacity), EMPTY, device=device, dtype=torch.int64)
    reached = torch.empty(rows, device=device, dtype=torch.int32)
    _score_buckets[(rows,)](
        index_queries,
        bucket_centroids,
        buckets,
        survivor_counts,
        thresholds,
        keys,
        reached,
        capacity,
        D=d,
        BLOCK=SCORE_BLOCK,
        SPAN=0,
        DENSE=False,
        num_warps=4,
        **shape_arguments(index_queries, shapes, query_planes),
    )
    return keys, survivor_counts, reached


@triton.jit
def _sample_scores(
    index_queries: tl.tensor,
    bucket_centroids: tl.tensor,
    scores: tl.tensor,
    query_count: tl.tensor,
    sample_count: tl.tensor,
    stride: tl.tensor,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_S: tl.constexpr,
) -> None:
    """Score every stride-th bucket against each query on the tensor cores.

    Args:
        index_queries: Index queries [query_count, D] bf16.
        bucket_centroids: Bucket centroids [N, D] bf16.
        scores: Sampled scores [query_count, sample_count] fp32.
        stride: Buckets between two sampled ones.
    """
    rows = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    columns = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    dims = tl.arange(0, D)
    live_rows = rows < query_count
    live_columns = columns < sample_count
    query = tl.load(
        index_queries + rows[:, None].to(tl.int64) * D + dims[None, :], mask=live_rows[:, None], other=0.0
    )
    centroid = tl.load(
        bucket_centroids + (columns * stride)[:, None].to(tl.int64) * D + dims[None, :],
        mask=live_columns[:, None],
        other=0.0,
    )
    score = tl.dot(query, tl.trans(centroid))
    tl.store(
        scores + rows[:, None].to(tl.int64) * sample_count + columns[None, :],
        score,
        mask=live_rows[:, None] & live_columns[None, :],
    )


@triton.jit
def _histogram_thresholds(
    scores: tl.tensor,
    visible_counts: tl.tensor,
    norms: tl.tensor,
    thresholds: tl.tensor,
    sample_count: tl.tensor,
    stride: tl.tensor,
    rank: tl.tensor,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Read each row's threshold off a histogram of its sampled scores.

    The threshold is the lower edge of the bin where the sample reaches `rank` scores at or above it.
    It is -inf for a row seeing fewer.

    Args:
        scores: Sampled scores [R, sample_count] fp32.
        visible_counts: Buckets each row sees [R] int32.
        norms: Each query's norm [R] fp32.
        thresholds: Each row's threshold [R] fp32.
        stride: Buckets between two sampled ones.
        rank: Rank in the sample the threshold stops at.
    """
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(visible_counts + row)
    norm = tl.load(norms + row)
    row_scores = scores + row * sample_count
    # The row's best and mean sampled score - the bins span the upper half between them
    # => few bins resolve the tail, and the serving image's Triton histograms at a cost in proportion to bins
    best = -float("inf")
    total = 0.0
    seen = 0
    for start in range(0, sample_count, BLOCK):
        column = start + tl.arange(0, BLOCK)
        live = (column < sample_count) & (column * stride < length)
        score = tl.load(row_scores + column, mask=live, other=-float("inf"))
        best = tl.maximum(best, tl.max(score, 0))
        total += tl.sum(tl.where(live, score, 0.0), 0)
        seen += tl.sum(live.to(tl.int32), 0)
    mean = total / tl.maximum(seen, 1)
    width = tl.maximum((best - mean) / BINS, 1e-30)
    # An unseen column, or one under the mean, lands in the lowest bin, which only a row seeing fewer than
    # rank above its mean reaches
    counts = tl.zeros([BINS], dtype=tl.int32)
    for start in range(0, sample_count, BLOCK):
        column = start + tl.arange(0, BLOCK)
        live = (column < sample_count) & (column * stride < length)
        score = tl.load(row_scores + column, mask=live, other=-float("inf"))
        counts += tl.histogram(
            tl.minimum(tl.maximum((score - mean) / width, 0.0), BINS - 1.0).to(tl.int32), BINS
        )
    edges = tl.arange(0, BINS)
    at_or_above = tl.sum(counts, 0) - tl.cumsum(counts, 0) + counts
    top = tl.max(tl.where(at_or_above >= rank, edges, -1), 0)
    # A hair under the edge, so a sampled bucket scored at it again still counts as at or above it
    threshold = tl.where(top > 0, mean + top.to(tl.float32) * width - EDGE_SLACK * norm, -float("inf"))
    tl.store(thresholds + row, threshold)


def sample_plan(top_p: int) -> tuple[int, int]:
    """Return the stride of the sampled buckets and the rank the threshold is read at.

    Args:
        top_p: Buckets each row selects.

    Returns:
        Buckets between two sampled ones.
        Rank in the sample of the score the threshold is estimated from.
    """
    stride = max(1, round(TARGET * top_p / SAMPLE_RANK))
    return stride, max(1, round(TARGET * top_p / stride))


def sample_thresholds(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    top_p: int,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
) -> torch.Tensor:
    """Estimate each row's score at TARGET times its top_p from every stride-th bucket it sees.

    The estimate is the lower edge of the histogram bin where the sample reaches its share of that count, so it
    errs toward a few more survivor_counts.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16 of unit rows.
        visible_counts: Buckets each row sees [R] int32, at most N.
        top_p: Buckets each row selects.
        shapes: The buckets' shapes, row for row, to estimate the shaped scores - None for the centroid.
        query_planes: The queries' planes, when shaped and already split.

    Returns:
        Estimated scores [R] fp32, -inf for a row seeing too few to rank.
    """
    rows, d = index_queries.shape
    stride, rank = sample_plan(top_p)
    sample_count = triton.cdiv(bucket_centroids.shape[0], stride)
    # cuBLAS takes twice as long over so short a reduction, and tiles of 256 samples spill registers
    scores = torch.empty(rows, sample_count, device=index_queries.device)
    block_q, block_s = 64, 128
    _sample_scores[(triton.cdiv(rows, block_q), triton.cdiv(sample_count, block_s))](
        index_queries,
        bucket_centroids,
        scores,
        rows,
        sample_count,
        stride,
        D=d,
        BLOCK_Q=block_q,
        BLOCK_S=block_s,
        num_warps=4,
    )
    if shapes is not None:
        add_planes_spread(scores, index_queries, shapes.rows(slice(None, None, stride)), query_planes)
    thresholds = torch.empty(rows, device=index_queries.device)
    _histogram_thresholds[(rows,)](
        scores,
        visible_counts,
        index_queries.float().norm(dim=1),
        thresholds,
        sample_count,
        stride,
        rank,
        BINS=64,
        BLOCK=4096,
        num_warps=4,
    )
    return thresholds
