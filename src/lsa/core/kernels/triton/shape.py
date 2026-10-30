"""Shapes on the device - building them from the keys beneath, and scoring a bucket by its estimated best key.

A shape is `rank` sign directions packed one bit a dimension, their variances and a floor (`lsa.core.shape` is
the torch oracle). A shaped score is `q . c + z sqrt(q^T Sigma q)`, and its spread term needs `q . s` for each sign
direction `s`. Expanding the bits costs more instructions than the centroid's whole product, so a packed direction is
never expanded: the query is split once into 8 bit-planes of its int8 levels, and `q . s` over a 32-bit word of the
basis is a few popcounts. Every kernel here that scores against the planes shares one helper, so a bucket scores
alike, bit for bit, whichever read scores it.

`add_spread` is the other way, the node buckets' as the freeze scored them: the bits expanded once a tile and
multiplied with the bf16 query on the tensor cores. A hierarchy's open node buckets are scored by it and the children
of its beam by the planes, which is what the freeze's saved reads hold.

A shape is built over every key under a bucket: the residuals about the centroid, their `[index_dim, index_dim]` covariance in bf16
products summed in fp32, and its leading directions by deflated power iteration, which the sign basis reads no worse
than an exact eigendecomposition.
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from lsa.core.shape import leading_directions, pack_shape_basis

# Bits of the query levels a packed basis is scored against
QUERY_BITS = 8
# Key positions a shape build reads at a time - a chunk of buckets' members gathered at once
SHAPE_KEYS = 1 << 17
# A plane gap's widening, past the fp32 rounding of the planes' levels and of the gap itself
GAP_SLACK = 1 + 2**-8


class SignQuery(NamedTuple):
    """Index queries split into bit-planes of their int levels, what a packed basis is scored against."""

    # Bit-planes of each query's levels offset to be nonnegative [S, QUERY_BITS, index_dim / 32] int32
    planes: torch.Tensor
    # The size of a level in each chunk of d [S, index_dim / chunk] fp32
    steps: torch.Tensor
    # Each query's levels summed, times their steps [S] fp32
    totals: torch.Tensor


class ShapeTable(NamedTuple):
    """The shapes of a table of buckets, row for row with their centroids, and the deviations they score with."""

    # Sign directions [m, rank, index_dim / 8] uint8, one bit a dimension
    basis: torch.Tensor
    # Variance along each direction [m, rank] fp32
    variances: torch.Tensor
    # Average variance of the directions left out [m] fp32
    floor: torch.Tensor
    # Standard deviations a score adds, z
    deviations: float

    def rows(self, rows: torch.Tensor | slice) -> "ShapeTable":
        """Return the shapes of some rows, aligned with them.

        Args:
            rows: Rows of the table [n] int64, or a slice.
        """
        return ShapeTable(self.basis[rows], self.variances[rows], self.floor[rows], self.deviations)


@triton.jit
def _sign_query_kernel(
    index_queries: tl.tensor,
    planes: tl.tensor,
    steps: tl.tensor,
    totals: tl.tensor,
    D: tl.constexpr,
    DKC: tl.constexpr,
    PLANES: tl.constexpr,
) -> None:
    """Split each index query into bit-planes of its int levels, one program a query.

    Args:
        index_queries: Index queries [S, D] bf16.
        planes: Bit-planes of the levels offset to be nonnegative [S, PLANES, D / 32] int32.
        steps: The size of a level in each chunk of DKC dims [S, D / DKC] fp32.
        totals: Each query's levels summed, times their steps [S] fp32.
    """
    row = tl.program_id(0).to(tl.int64)
    top_level: tl.constexpr = (1 << (PLANES - 1)) - 1
    shifts = tl.arange(0, 32)
    total = 0.0
    for k in tl.static_range(D // DKC):
        dims = k * DKC + tl.arange(0, DKC)
        query = tl.load(index_queries + row * D + dims).to(tl.float32)
        step = tl.maximum(tl.max(tl.abs(query), axis=0), 1e-30) / top_level
        levels = libdevice.rint(query / step).to(tl.int32)
        # Offset to be nonnegative, so each plane is a plain bit mask [DKC / 32, 32]
        offset = tl.reshape(levels + top_level, [DKC // 32, 32])
        words = k * (DKC // 32) + tl.arange(0, DKC // 32)
        for plane_index in tl.static_range(PLANES):
            plane = tl.sum(((offset >> plane_index) & 1) << shifts[None, :], axis=1)
            tl.store(planes + (row * PLANES + plane_index) * (D // 32) + words, plane)
        tl.store(steps + row * (D // DKC) + k, step)
        total += tl.sum(levels, axis=0).to(tl.float32) * step
    tl.store(totals + row, total)


def plane_gaps(index_queries: torch.Tensor) -> torch.Tensor:
    """How far each query's planes project from the query along any unit direction - the length of their difference.

    The planes hold a query's entries rounded to its chunk's levels, as `_sign_query_kernel` rounds them, so a sign
    direction's projection of the planes is within this of its projection of the query.

    Args:
        index_queries: Index queries [S, index_dim] bf16.

    Returns:
        Gaps [S] fp32, a hair wide.
    """
    count, index_dim = index_queries.shape
    chunk = min(128, index_dim)
    queries = index_queries.float().view(count, index_dim // chunk, chunk)
    top_level = (1 << (QUERY_BITS - 1)) - 1
    steps = queries.abs().amax(-1, keepdim=True).clamp_min(1e-30) / top_level
    rounded = torch.round(queries / steps) * steps
    return (rounded - queries).flatten(1).norm(dim=1) * GAP_SLACK + 1e-6


def sign_query(index_queries: torch.Tensor) -> SignQuery:
    """Split index queries into the bit-planes a packed basis is scored against.

    Args:
        index_queries: Index queries [S, index_dim] bf16, unit rows, index_dim a multiple of 32 up to 128.

    Returns:
        The queries' planes, steps and totals.
    """
    count, index_dim = index_queries.shape
    chunk = min(128, index_dim)
    device = index_queries.device
    planes = torch.empty(count, QUERY_BITS, index_dim // 32, device=device, dtype=torch.int32)
    steps = torch.empty(count, index_dim // chunk, device=device, dtype=torch.float32)
    totals = torch.empty(count, device=device, dtype=torch.float32)
    if count:
        _sign_query_kernel[(count,)](
            index_queries.contiguous(), planes, steps, totals, D=index_dim, DKC=chunk, PLANES=QUERY_BITS
        )
    return SignQuery(planes, steps, totals)


@triton.jit
def _spread(
    word: tl.tensor,
    query_planes: tl.tensor,
    step: tl.tensor,
    total: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    kept: tl.tensor,
    D: tl.constexpr,
    PLANES: tl.constexpr,
) -> tl.tensor:
    """Return a shape's estimated standard deviation of key scores against one query's planes.

    Args:
        word: Packed sign directions of N buckets [N, RHOP, D / 32] int32.
        query_planes: The query's planes [PLANES, D / 32] int32.
        step: The query's per-word steps [D / 32] fp32.
        total: The query's levels summed, times their steps.
        variances: Variance along each direction [N, RHOP] fp32.
        floor: Average variance of the directions left out [N] fp32.
        kept: Directions in use [1, RHOP] bool.

    Returns:
        Standard deviations [N] fp32.
    """
    words = tl.arange(0, D // 32)
    levels = -((1 << (PLANES - 1)) - 1) * libdevice.popc(word)
    for plane_index in tl.static_range(PLANES):
        plane = tl.load(query_planes + plane_index * (D // 32) + words)
        levels += libdevice.popc(word & plane[None, None, :]) << plane_index
    chosen = tl.sum(levels.to(tl.float32) * step[None, None, :], axis=2)
    # Bit j of a word is dimension 32 w + j, and s = +-1 / sqrt(d)
    # => q . s = (2 sum(q where bit) - sum(q)) / sqrt(d)
    projection = tl.where(kept, (2.0 * chosen - total) * tl.rsqrt(float(D)), 0.0)
    squared = projection * projection
    # The query is a unit row, so ||q||^2 splits between the basis and its complement
    variance = tl.sum(squared * variances, axis=1) + floor * (1.0 - tl.sum(squared, axis=1))
    return tl.sqrt(tl.maximum(variance, 0.0))


@triton.jit
def _score_rows_kernel(
    index_queries: tl.tensor,
    centroids: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    rows_ptr: tl.tensor,
    planes: tl.tensor,
    steps: tl.tensor,
    totals: tl.tensor,
    out: tl.tensor,
    deviations: tl.tensor,
    rescale: tl.tensor,
    rows_stride: tl.tensor,
    out_stride: tl.tensor,
    D: tl.constexpr,
    DKC: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
    PLANES: tl.constexpr,
) -> None:
    """Score the given centroid rows of each query by their estimated best key, one program a block of rows.

    Args:
        index_queries: Index queries [S, D] bf16, unit rows.
        centroids: Centroid rows [rows, D] bf16 or fp8.
        basis: Packed sign directions [rows, RHO, D / 32] int32.
        variances: Variance along each direction [rows, RHO] fp32.
        floor: Average variance of the directions left out [rows] fp32.
        rows_ptr: The rows of each query [S, COLUMNS] int32, -1 for none.
        planes: The queries' planes [S, PLANES, D / 32] int32.
        steps: The queries' plane steps [S, D / DKC] fp32.
        totals: The queries' plane totals [S] fp32.
        out: Scores [S, COLUMNS] fp32, -inf under a -1 row.
        deviations: Standard deviations a score adds.
        rescale: Factor undoing the fp8 centroid scale.
    """
    row = tl.program_id(0).to(tl.int64)
    slots = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bucket = tl.load(rows_ptr + row * rows_stride + slots, mask=slots < COLUMNS, other=-1).to(tl.int64)
    held = bucket >= 0
    dims = tl.arange(0, D)
    query = tl.load(index_queries + row * D + dims).to(tl.float32)
    directions = tl.arange(0, RHOP)
    kept = directions < RHO
    words = tl.arange(0, D // 32)
    centroid = tl.load(centroids + bucket[:, None] * D + dims[None, :], mask=held[:, None], other=0.0)
    mean = tl.sum(centroid.to(tl.float32) * query[None, :], axis=1) * rescale
    word = tl.load(
        basis
        + ((bucket[:, None] * RHO + directions[None, :]) * (D // 32))[:, :, None]
        + words[None, None, :],
        mask=held[:, None, None] & kept[None, :, None],
        other=0,
    )
    shape_variances = tl.load(
        variances + bucket[:, None] * RHO + directions[None, :], mask=held[:, None] & kept[None, :], other=0.0
    )
    spread = _spread(
        word,
        planes + row * PLANES * (D // 32),
        tl.load(steps + row * (D // DKC) + words * 32 // DKC),
        tl.load(totals + row),
        shape_variances,
        tl.load(floor + bucket, mask=held, other=0.0),
        kept[None, :],
        D,
        PLANES,
    )
    score = mean + deviations * spread
    tl.store(out + row * out_stride + slots, tl.where(held, score, float("-inf")), mask=slots < COLUMNS)


def basis_words(basis: torch.Tensor) -> torch.Tensor:
    """View a packed basis as the 32-bit words the kernels read, or a stand-in when it keeps no directions.

    Args:
        basis: The packed basis [m, rank, index_dim / 8] uint8.

    Returns:
        Its words [m, rank, index_dim / 32] int32 - one zero word when rank or m is 0, which no kernel then reads.
    """
    if not basis.numel():
        return torch.zeros(1, device=basis.device, dtype=torch.int32)
    return basis.contiguous().view(torch.int32)


def score_shaped_rows(
    index_queries: torch.Tensor,
    centroids: torch.Tensor,
    basis: torch.Tensor,
    variances: torch.Tensor,
    floor: torch.Tensor,
    rows: torch.Tensor,
    deviations: float,
    out: torch.Tensor | None = None,
    centroid_scale: float = 1.0,
    query_planes: SignQuery | None = None,
) -> torch.Tensor:
    """Score some rows of a table for each query by their estimated best key - children, listed buckets.

    Args:
        index_queries: One index query a row [S, index_dim] bf16, unit rows.
        centroids: The table's centroids [N, index_dim] - bf16, or fp8 holding `centroid_scale` times the row.
        basis: Their packed sign basis [N, rank, index_dim / 8] uint8.
        variances: Their variances [N, rank] fp32.
        floor: Their floors [N] fp32.
        rows: The rows each query scores [S, C] int32, -1 for none.
        deviations: Standard deviations a score adds, z.
        out: A buffer for the scores [S, C] fp32, a view of rows of any stride.
        centroid_scale: What fp8 centroids are multiplied by on storing; 1 for bf16.
        query_planes: The queries' planes when already split.

    Returns:
        Scores [S, C] fp32, -inf under a -1 row.
    """
    count, columns = rows.shape
    d = index_queries.shape[1]
    planes = query_planes or sign_query(index_queries)
    if out is None:
        out = torch.empty(count, columns, device=index_queries.device, dtype=torch.float32)
    rank = variances.shape[1]
    block = 32
    if count and columns:
        _score_rows_kernel[(count, triton.cdiv(columns, block))](
            index_queries,
            centroids,
            basis_words(basis),
            variances,
            floor,
            rows,
            planes.planes,
            planes.steps,
            planes.totals,
            out,
            deviations,
            1.0 / centroid_scale,
            rows.stride(0),
            out.stride(0),
            D=d,
            DKC=min(128, d),
            COLUMNS=columns,
            BLOCK=block,
            RHO=rank,
            RHOP=triton.next_power_of_2(max(rank, 1)),
            PLANES=QUERY_BITS,
            num_warps=4,
        )
    return out


@triton.jit
def _add_planes_spread_kernel(
    scores: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    planes: tl.tensor,
    steps: tl.tensor,
    totals: tl.tensor,
    deviations: tl.tensor,
    bucket_count: tl.tensor,
    score_stride: tl.tensor,
    D: tl.constexpr,
    DKC: tl.constexpr,
    BLOCK: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
    PLANES: tl.constexpr,
) -> None:
    """Add the spread of each bucket to its centroid score in place, scored against the query's planes.

    Args:
        scores: Centroid scores [S, bucket_count] fp32.
        basis: Packed sign directions [bucket_count, RHO, D / 32] int32.
        variances: Variance along each direction [bucket_count, RHO] fp32.
        floor: Average variance of the directions left out [bucket_count] fp32.
        planes: The queries' planes [S, PLANES, D / 32] int32.
        steps: The queries' plane steps [S, D / DKC] fp32.
        totals: The queries' plane totals [S] fp32.
        deviations: Standard deviations the spread adds.
    """
    row = tl.program_id(0).to(tl.int64)
    bucket = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    held = bucket < bucket_count
    directions = tl.arange(0, RHOP)
    kept = directions < RHO
    words = tl.arange(0, D // 32)
    word = tl.load(
        basis
        + ((bucket[:, None] * RHO + directions[None, :]) * (D // 32))[:, :, None]
        + words[None, None, :],
        mask=held[:, None, None] & kept[None, :, None],
        other=0,
    )
    shape_variances = tl.load(
        variances + bucket[:, None] * RHO + directions[None, :], mask=held[:, None] & kept[None, :], other=0.0
    )
    spread = _spread(
        word,
        planes + row * PLANES * (D // 32),
        tl.load(steps + row * (D // DKC) + words * 32 // DKC),
        tl.load(totals + row),
        shape_variances,
        tl.load(floor + bucket, mask=held, other=0.0),
        kept[None, :],
        D,
        PLANES,
    )
    at = scores + row * score_stride + bucket
    tl.store(at, tl.load(at, mask=held, other=0.0) + deviations * spread, mask=held)


def add_planes_spread(
    scores: torch.Tensor,
    index_queries: torch.Tensor,
    shapes: ShapeTable,
    query_planes: SignQuery | None = None,
) -> torch.Tensor:
    """Add z times each bucket's spread to its centroid scores, in place, scored against the query's planes.

    Args:
        scores: Centroid scores [S, m] fp32, a view of rows of any stride, a column for each of the shapes' rows.
        index_queries: Index queries [S, index_dim] bf16, unit rows.
        shapes: The shapes of the `m` columns.
        query_planes: The queries' planes when already split.

    Returns:
        The scores, shaped.
    """
    count, buckets = scores.shape
    d = index_queries.shape[1]
    planes = query_planes or sign_query(index_queries)
    rank = shapes.variances.shape[1]
    block = 64
    if count and buckets:
        _add_planes_spread_kernel[(count, triton.cdiv(buckets, block))](
            scores,
            basis_words(shapes.basis),
            shapes.variances,
            shapes.floor,
            planes.planes,
            planes.steps,
            planes.totals,
            shapes.deviations,
            buckets,
            scores.stride(0),
            D=d,
            DKC=min(128, d),
            BLOCK=block,
            RHO=rank,
            RHOP=triton.next_power_of_2(max(rank, 1)),
            PLANES=QUERY_BITS,
            num_warps=4,
        )
    return scores


@triton.jit
def _spread_kernel(
    index_queries: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    scores: tl.tensor,
    query_count: tl.tensor,
    bucket_count: tl.tensor,
    deviations: tl.tensor,
    query_stride: tl.tensor,
    score_stride: tl.tensor,
    D: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    DK: tl.constexpr,
) -> None:
    """Add the spread of each bucket to its centroid score in place, expanding the signs on the tensor cores.

    Args:
        index_queries: Index queries [query_count, D] bf16, unit rows.
        basis: Packed sign directions [bucket_count, RHO, D / 32] int32.
        variances: Variance along each direction [bucket_count, RHO] fp32.
        floor: Average variance of the directions left out [bucket_count] fp32.
        scores: Centroid scores [query_count, bucket_count] fp32.
        deviations: Standard deviations the spread adds.
    """
    queries = tl.program_id(0) * BT + tl.arange(0, BT)
    buckets = tl.program_id(1) * BM + tl.arange(0, BM)
    # One column per (bucket, direction) pair, so the projections are one GEMM
    pair = tl.arange(0, BM * RHOP)
    pair_bucket = tl.program_id(1) * BM + pair // RHOP
    direction = pair % RHOP
    pair_valid = (pair_bucket < bucket_count) & (direction < RHO)
    basis_row = tl.where(pair_valid, pair_bucket.to(tl.int64) * RHO + direction, 0)
    projection = tl.zeros([BT, BM * RHOP], dtype=tl.float32)
    for k in tl.static_range(D // DK):
        dims = k * DK + tl.arange(0, DK)
        query = tl.load(
            index_queries + queries[:, None].to(tl.int64) * query_stride + dims[None, :],
            mask=(queries < query_count)[:, None],
            other=0.0,
        )
        # Expanded once a tile and reused by all BT queries
        word_offsets = k * (DK // 32) + tl.arange(0, DK // 32)
        words = tl.load(
            basis + basis_row[:, None] * (D // 32) + word_offsets[None, :], mask=pair_valid[:, None], other=0
        )
        bits = tl.reshape((words[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1, [BM * RHOP, DK])
        signs = tl.where(bits != 0, 1.0, -1.0).to(tl.bfloat16)
        projection = tl.dot(query, tl.trans(signs), projection)
    projection = projection * tl.rsqrt(float(D))
    projection = tl.where(pair_valid[None, :], projection, 0.0)
    squared = tl.reshape(projection * projection, [BT, BM, RHOP])
    ranks = tl.arange(0, RHOP)
    shape_variances = tl.load(
        variances + buckets[:, None] * RHO + ranks[None, :],
        mask=(buckets < bucket_count)[:, None] & (ranks < RHO)[None, :],
        other=0.0,
    )
    weighted = tl.sum(squared * shape_variances[None, :, :], axis=2)
    total = tl.sum(squared, axis=2)
    shape_floor = tl.load(floor + buckets, mask=buckets < bucket_count, other=0.0)
    spread = tl.sqrt(tl.maximum(weighted + shape_floor[None, :] * (1.0 - total), 0.0))
    inside = (queries < query_count)[:, None] & (buckets < bucket_count)[None, :]
    at = scores + queries[:, None].to(tl.int64) * score_stride + buckets[None, :]
    tl.store(at, tl.load(at, mask=inside, other=0.0) + deviations * spread, mask=inside)


def add_spread(
    scores: torch.Tensor,
    index_queries: torch.Tensor,
    basis: torch.Tensor,
    variances: torch.Tensor,
    floor: torch.Tensor,
    deviations: float,
) -> torch.Tensor:
    """Add z times each bucket's spread to its centroid scores, in place, the bits expanded on the tensor cores.

    Args:
        scores: Centroid scores [S, m] fp32, a view of rows of any stride.
        index_queries: Index queries [S, index_dim] bf16, unit rows.
        basis: The columns' packed sign basis [m, rank, index_dim / 8] uint8.
        variances: Their variances [m, rank] fp32.
        floor: Their floors [m] fp32.
        deviations: Standard deviations a score adds, z.

    Returns:
        The scores, shaped.
    """
    count, buckets = scores.shape
    rank = variances.shape[1]
    rank_lanes = triton.next_power_of_2(max(rank, 1))
    bucket_tile = max(1, 128 // rank_lanes)
    queries = index_queries.to(torch.bfloat16).contiguous()
    if count and buckets:
        _spread_kernel[(triton.cdiv(count, 64), triton.cdiv(buckets, bucket_tile))](
            queries,
            basis_words(basis),
            variances,
            floor,
            scores,
            count,
            buckets,
            deviations,
            queries.stride(0),
            scores.stride(0),
            D=queries.shape[1],
            RHO=rank,
            RHOP=rank_lanes,
            BT=64,
            BM=bucket_tile,
            DK=64,
            num_warps=4,
        )
    return scores


@triton.jit
def _scan_paged_shaped_kernel(
    index_queries: tl.tensor,
    bucket_centroids: tl.tensor,
    basis: tl.tensor,
    variances: tl.tensor,
    floor: tl.tensor,
    planes: tl.tensor,
    steps: tl.tensor,
    totals: tl.tensor,
    pages_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    out_ptr: tl.tensor,
    deviations: tl.tensor,
    max_rows: tl.tensor,
    pages_stride: tl.tensor,
    out_stride: tl.tensor,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DK: tl.constexpr,
    DKC: tl.constexpr,
    ROWS: tl.constexpr,
    RHO: tl.constexpr,
    RHOP: tl.constexpr,
    PLANES: tl.constexpr,
) -> None:
    """Score a block of one request's bucket rows by their estimated best key through its pages.

    Args:
        index_queries: Index queries [B, D] bf16, unit rows.
        bucket_centroids: Centroid rows [rows, D] bf16.
        basis: Packed sign directions [rows, RHO, D / 32] int32.
        variances: Variance along each direction [rows, RHO] fp32.
        floor: Average variance of the directions left out [rows] fp32.
        planes: The queries' planes [B, PLANES, D / 32] int32.
        steps: The queries' plane steps [B, D / DKC] fp32.
        totals: The queries' plane totals [B] fp32.
        pages_ptr: Pages of each request [B, pages] int32.
        counts_ptr: Bucket rows each request holds [B] int32.
        out_ptr: Scores [B, max_rows] fp32, -inf for a row not held or on no page.
        deviations: Standard deviations the spread adds.
        max_rows: Columns scored a request.
    """
    program = tl.program_id(0)
    request = tl.program_id(1).to(tl.int64)
    rows = program * ROWS + tl.arange(0, ROWS)
    inside = rows < max_rows
    visible = rows < tl.load(counts_ptr + request)
    page = tl.load(pages_ptr + request * pages_stride + rows // PAGE_SIZE, mask=visible, other=-1)
    valid = visible & (page >= 0)
    cache_row = tl.where(valid, page.to(tl.int64) * PAGE_SIZE + rows % PAGE_SIZE, 0)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for k in tl.static_range(D // DK):
        dims = k * DK + tl.arange(0, DK)
        q = tl.load(index_queries + request * D + dims).to(tl.float32)
        c = tl.load(bucket_centroids + cache_row[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0)
        acc += tl.sum(c.to(tl.float32) * q[None, :], axis=1)
    directions = tl.arange(0, RHOP)
    kept = directions < RHO
    words = tl.arange(0, D // 32)
    word = tl.load(
        basis
        + ((cache_row[:, None] * RHO + directions[None, :]) * (D // 32))[:, :, None]
        + words[None, None, :],
        mask=valid[:, None, None] & kept[None, :, None],
        other=0,
    )
    shape_variances = tl.load(
        variances + cache_row[:, None] * RHO + directions[None, :],
        mask=valid[:, None] & kept[None, :],
        other=0.0,
    )
    spread = _spread(
        word,
        planes + request * PLANES * (D // 32),
        tl.load(steps + request * (D // DKC) + words * 32 // DKC),
        tl.load(totals + request),
        shape_variances,
        tl.load(floor + cache_row, mask=valid, other=0.0),
        kept[None, :],
        D,
        PLANES,
    )
    out = tl.where(valid, acc + deviations * spread, float("-inf"))
    tl.store(out_ptr + request * out_stride + rows, out, mask=inside)


def scan_paged_shaped(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    shapes: ShapeTable,
    pages: torch.Tensor,
    counts: torch.Tensor,
    page_size: int,
    max_rows: int,
) -> torch.Tensor:
    """Score each request's buckets by their estimated best key, reading centroids and shapes through its pages.

    The same contract as `scan_paged`, and the same shapes out, so the top-k and everything after it is unchanged.

    Args:
        index_queries: One index query a request [B, index_dim] bf16, unit rows.
        bucket_centroids: The cache's bucket centroids [rows, index_dim] bf16, `page_size` rows to a page.
        shapes: The cache's bucket shapes, row for row.
        pages: Each request's pages [B, max_pages] int32, -1 for none.
        counts: Visible buckets of each request [B] int32.
        page_size: Cache rows a page holds.
        max_rows: Columns of the result, at least the largest count.

    Returns:
        Scores [B, max_rows] fp32, -inf past a request's count or on a page it has not got.
    """
    batch_size, d = index_queries.shape
    planes = sign_query(index_queries)
    rank = shapes.variances.shape[1]
    out = torch.empty(batch_size, max_rows, device=index_queries.device, dtype=torch.float32)
    rows = 64
    _scan_paged_shaped_kernel[(triton.cdiv(max_rows, rows), batch_size)](
        index_queries,
        bucket_centroids,
        basis_words(shapes.basis),
        shapes.variances,
        shapes.floor,
        planes.planes,
        planes.steps,
        planes.totals,
        pages,
        counts,
        out,
        shapes.deviations,
        max_rows,
        pages.stride(0),
        out.stride(0),
        PAGE_SIZE=page_size,
        D=d,
        DK=min(128, d),
        DKC=min(128, d),
        ROWS=rows,
        RHO=rank,
        RHOP=triton.next_power_of_2(max(rank, 1)),
        PLANES=QUERY_BITS,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def _top_directions_kernel(
    covariance_ptr: tl.tensor,
    basis_ptr: tl.tensor,
    variances_ptr: tl.tensor,
    trace_ptr: tl.tensor,
    D: tl.constexpr,
    RANK: tl.constexpr,
    STEPS: tl.constexpr,
) -> None:
    """Find each covariance's leading eigen-directions by deflated power iteration, one program each.

    Args:
        covariance_ptr: Symmetric covariances [N, D, D] fp32.
        basis_ptr: Directions [N, RANK, D] fp32 unit rows, leading first.
        variances_ptr: Their variances [N, RANK] fp32.
        trace_ptr: Each covariance's trace [N] fp32.
    """
    node = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, D)
    matrix = tl.load(covariance_ptr + node * D * D + dims[:, None] * D + dims[None, :])
    diagonal = tl.sum(tl.where(dims[:, None] == dims[None, :], matrix, 0.0), axis=1)
    tl.store(trace_ptr + node, tl.sum(diagonal, axis=0))
    # The directions found so far, removed from each new one as it is iterated
    for direction in tl.static_range(RANK):
        # Start from the axis the covariance spreads along most
        start = tl.argmax(diagonal, axis=0)
        vector = tl.sum(tl.where(dims[None, :] == start, matrix, 0.0), axis=1)
        for _step in tl.static_range(STEPS):
            for found in tl.static_range(direction):
                earlier = tl.load(basis_ptr + (node * RANK + found) * D + dims)
                vector -= tl.sum(vector * earlier, axis=0) * earlier
            vector = vector / tl.maximum(tl.sqrt(tl.sum(vector * vector, axis=0)), 1e-12)
            vector = tl.sum(matrix * vector[None, :], axis=1)
        for found in tl.static_range(direction):
            earlier = tl.load(basis_ptr + (node * RANK + found) * D + dims)
            vector -= tl.sum(vector * earlier, axis=0) * earlier
        vector = vector / tl.maximum(tl.sqrt(tl.sum(vector * vector, axis=0)), 1e-12)
        value = tl.sum(vector * tl.sum(matrix * vector[None, :], axis=1), axis=0)
        tl.store(basis_ptr + (node * RANK + direction) * D + dims, vector)
        tl.store(variances_ptr + node * RANK + direction, tl.maximum(value, 0.0))
        # A found direction leaves the start's choice of the next one
        diagonal -= value * vector * vector


def top_directions(
    covariance: torch.Tensor, rank: int, steps: int = 24
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `rank` leading eigen-directions of each covariance - deflated power iteration on a GPU, eigh elsewhere.

    A batched eigendecomposition of 1,024 [64, 64] covariances costs some 2 ms and finds all 64 directions, where a
    shape keeps one or two and the served read only their signs. Power iteration from each covariance's widest axis,
    deflating the directions already found, reaches them in a few steps with the matrix held in registers.

    Args:
        covariance: Symmetric covariances [N, index_dim, index_dim] fp32, index_dim a power of two up to 128.
        rank: Directions kept a covariance.
        steps: Power steps a direction.

    Returns:
        The directions [N, rank, index_dim] fp32 unit rows, leading first.
        Their variances [N, rank] fp32.
        Each covariance's trace [N] fp32.
    """
    count, index_dim, _ = covariance.shape
    covariance = covariance.float().contiguous()
    trace = covariance.diagonal(dim1=1, dim2=2).sum(1)
    if not covariance.is_cuda or rank == 0:
        directions, variances = leading_directions(covariance, rank)
        return directions, variances, trace
    basis = torch.empty(count, rank, index_dim, device=covariance.device)
    variances = torch.empty(count, rank, device=covariance.device)
    if count:
        _top_directions_kernel[(count,)](
            covariance, basis, variances, trace, D=index_dim, RANK=rank, STEPS=steps, num_warps=4
        )
    return basis, variances, trace


def build_shapes(
    index_keys: Callable[[torch.Tensor], torch.Tensor],
    centroids: torch.Tensor,
    members: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape each bucket over every key under it, a chunk of buckets at a time.

    A bucket holds more keys than index_dim once it is a node bucket, so the `[index_dim, index_dim]` covariance is the smaller matrix.

    Args:
        index_keys: Positions [M] int64 to their index keys [M, index_dim] bf16, unit rows.
        centroids: The buckets' centroids [N, index_dim] fp32 unit rows, their keys' normalized sum.
        members: Every key position under each bucket [N, slots] int64, -1 past its size.
        rank: Directions kept a bucket.

    Returns:
        The packed sign basis [N, rank, index_dim / 8] uint8.
        Variance along each kept direction [N, rank] fp32, descending.
        The floor [N] fp32 - the trace left over, spread over the directions dropped.
    """
    count, slots = members.shape
    index_dim = centroids.shape[1]
    chunk = max(1, SHAPE_KEYS // max(slots, 1))
    bases, variances, floors = [], [], []
    for first in range(0, count, chunk):
        part = members[first : first + chunk]
        held = part >= 0
        keys = index_keys(part.clamp_min(0).flatten()).view(*part.shape, index_dim).float()
        residuals = ((keys - centroids[first : first + chunk, None]) * held[..., None]).to(torch.bfloat16)
        sizes = held.sum(1).clamp_min(1).float()
        if residuals.is_cuda:
            # bf16 products summed in fp32 on the tensor cores - an fp32 product of [d, index_dim] outputs runs on the CUDA
            # cores at a third the speed, for a basis kept as signs
            products = torch.bmm(residuals.transpose(1, 2), residuals, out_dtype=torch.float32)
        else:
            products = residuals.float().transpose(1, 2) @ residuals.float()
        basis, kept, trace = top_directions(products / sizes[:, None, None], rank)
        bases.append(basis)
        variances.append(kept)
        floors.append(((trace - kept.sum(1)) / max(index_dim - rank, 1)).clamp_min(0))
    if not bases:
        device = centroids.device
        return (
            torch.zeros(0, rank, index_dim // 8, device=device, dtype=torch.uint8),
            torch.zeros(0, rank, device=device),
            torch.zeros(0, device=device),
        )
    return pack_shape_basis(torch.cat(bases)), torch.cat(variances), torch.cat(floors)


def build_bucket_shapes(
    index_keys: Callable[[torch.Tensor], torch.Tensor],
    centroids: torch.Tensor,
    members: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape buckets of at most a cap of keys exactly, through each one's `[slots, slots]` Gram matrix.

    A bucket holds fewer keys than index_dim, so its Gram matrix `R R^T` is the smaller one, and it shares its nonzero
    eigenvalues with the covariance `R^T R`: for `G w = s w`, `R^T w / sqrt(s)` is the covariance's unit
    eigenvector with variance `s / n`. Its few keys spread nearly alike along several directions, where power iteration
    would stop short, so the eigendecomposition is exact. A padded slot adds a zero row, which leaves the eigenvalues
    as they are.

    Args:
        index_keys: Positions [M] int64 to their index keys [M, index_dim] bf16, unit rows.
        centroids: The buckets' centroids [N, index_dim] fp32 unit rows, their keys' normalized sum.
        members: Each bucket's key positions [N, slots] int64, -1 past its size.
        rank: Directions kept a bucket, at most `slots`.

    Returns:
        The packed sign basis [N, rank, index_dim / 8] uint8.
        Variance along each kept direction [N, rank] fp32, descending.
        The floor [N] fp32 - the trace left over, spread over the directions dropped.
    """
    index_dim = centroids.shape[1]
    held = members >= 0
    keys = index_keys(members.clamp_min(0).flatten()).view(*members.shape, index_dim).float()
    residuals = (keys - centroids[:, None]) * held[..., None]
    sizes = held.sum(1).clamp_min(1).float()
    trace = residuals.square().sum((1, 2)) / sizes
    values, vectors = torch.linalg.eigh(residuals @ residuals.transpose(1, 2))
    squared = values[:, values.shape[1] - rank :].flip(-1).clamp_min(1e-30)
    gram_basis = vectors[:, :, vectors.shape[2] - rank :].flip(-1)
    basis = torch.einsum("nsd,nsr->nrd", residuals, gram_basis) / squared.sqrt()[..., None]
    variances = squared / sizes[:, None]
    floor = ((trace - variances.sum(1)) / max(index_dim - rank, 1)).clamp_min(0)
    return pack_shape_basis(basis), variances, floor
