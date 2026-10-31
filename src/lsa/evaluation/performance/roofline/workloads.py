"""Inputs shaped as one served chunk of 32,768 queries sees them, and the call that launches each step once on them.

Every step of [dispatch](../../../core/kernels/dispatch.py)'s table has a launcher here, plus a bf16 GEMM through
cuBLAS that anchors the roofline. A launcher is built for a context and a `kernels` arm, `cutedsl` or `triton`, and
returns a callable that runs the step in that language on the same inputs, or None where the step does not serve at
that context - the short-context listing above 32,768 buckets. The CuTeDSL arm calls the kernels' modules directly. The Triton
arm calls the public front door with `kernels: triton` where there is one, and the Triton module where there is not.

The inputs are synthetic: unit centroids, queries anchored near a few of them, and one request's pages in a random
order. The flat scan's state is built through the CuTeDSL kernels themselves, so the listing holds what a real
threshold leaves and the expansion by threshold reads a real listing. Shapes are the served defaults of GLM-5.3-Flash's design.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from lsa.core.kernels import dispatch, scan_select
from lsa.core.kernels import expansion as front_expansion
from lsa.core.kernels import ward as front_ward
from lsa.core.kernels.cutedsl import (
    beam,
    bucket_scan,
    decode,
    expansion,
    node_bucket_clustering,
    scoring,
    thresholds,
    top_k,
    ward,
)
from lsa.core.kernels.triton import beam as triton_beam
from lsa.core.kernels.triton import scan_select as triton_scan_select
from lsa.core.kernels.triton.beam import NODE_CENTROID_SCALE
from lsa.core.kernels.triton.keys import EMPTY
from lsa.core.kernels.triton.paged import scan_paged
from lsa.core.kernels.triton.top_k import TopK
from lsa.core.ward import pool_keys

# Queries in a served prefill chunk
CHUNK = 32768
INDEX_DIM = 64
BUCKET_SIZE = 8
BUCKET_CAP = 16
TOP_P = 1024
TOP_K = 4096
WINDOW_SIZE = 512
SINK_SIZE = 4
# Keys a page holds, one cluster block
CLUSTER_BLOCK_SIZE = 512
PAGE_ROWS = CLUSTER_BLOCK_SIZE // BUCKET_SIZE
POOL_SIZE = 4
NODE_BUCKET_SIZE = 4
BEAM_WIDTH = 2048
START_NODES = 32768
# Open node buckets a level scores beside its carried children, and uncovered buckets beside the last beam
OPEN_NODES = 900
OPEN_BUCKETS = 16
# Rows the dense scoring takes at a time
DENSE_ROWS = 256
# Buckets a growth pairs in one group - the tree's node cluster block
PAIRED_BUCKETS = 4096
# Below this many visible buckets a chunk lists through the short-context listing
SHORT_BUCKETS = triton_scan_select.SCAN_SELECT_BUCKETS
# Rows and columns of the anchoring GEMM
GEMM_DIM = 8192
QUERY_NORM = 30.0
SEED = 0

Launch = Callable[[], Any]

# The languages a step is profiled in
KERNELS = ("cutedsl", "triton")


@dataclass(frozen=True)
class Step:
    """One step of the read and how to launch it."""

    # The step's name in dispatch's table
    name: str
    # Where it sits - prefill scan, expansion, tree, decode or growth
    stage: str
    # Whether its shapes follow the context, or are one chunk's whatever the context
    follows_context: bool
    build: Callable[[int, str], Launch | None]
    # The launcher for a batch of requests, for a step that runs in a batch - its shapes follow the batch
    build_batch: Callable[[int, int, str], Launch | None] | None = None


class Cache(NamedTuple):
    """One request's bucket cache at a context, and a chunk of queries at its end."""

    index_queries: torch.Tensor
    bucket_centroids: torch.Tensor
    bucket_members: torch.Tensor
    bucket_sizes: torch.Tensor
    pages: torch.Tensor
    positions: torch.Tensor
    visible_counts: torch.Tensor


class Listing(NamedTuple):
    """The flat scan's state for a chunk - thresholds, the listing and its keys."""

    thresholds: torch.Tensor
    shifted: torch.Tensor
    buckets: torch.Tensor
    survivor_counts: torch.Tensor
    keys: torch.Tensor


def generator() -> torch.Generator:
    """Return a CUDA generator at the fixed seed."""
    return torch.Generator(device="cuda").manual_seed(SEED)


def unit_rows(count: int, rows_generator: torch.Generator) -> torch.Tensor:
    """Draw unit rows.

    Args:
        count: Rows to draw.
        rows_generator: The generator to draw from.

    Returns:
        Unit rows [count, 64] fp32.
    """
    return F.normalize(torch.randn(count, INDEX_DIM, device="cuda", generator=rows_generator), dim=-1)


def bucket_count(context: int) -> int:
    """Return the buckets a context holds, whole pages of them.

    Args:
        context: Tokens in the request.
    """
    return context // CLUSTER_BLOCK_SIZE * PAGE_ROWS


@cache
def bucket_cache(context: int) -> Cache:
    """Build a request's bucket cache and the chunk of queries that ends it.

    Args:
        context: Tokens in the request.

    Returns:
        The cache, with logical bucket b stored at its page's cache row and holding keys 8b to 8b + 7.
    """
    rows_generator = generator()
    buckets = bucket_count(context)
    page_count = buckets // PAGE_ROWS
    bucket_centroids = unit_rows(buckets, rows_generator)
    # Queries near a few centroids, so a row's best buckets stand out of the noise
    anchors = torch.randint(0, buckets, (CHUNK,), device="cuda", generator=rows_generator)
    noise = torch.randn(CHUNK, INDEX_DIM, device="cuda", generator=rows_generator)
    index_queries = (QUERY_NORM * F.normalize(3 * bucket_centroids[anchors] + noise, dim=-1)).bfloat16()
    pages = torch.randperm(page_count, device="cuda", generator=rows_generator).int()
    logical = torch.arange(buckets, device="cuda")
    cache_rows = pages.long()[logical // PAGE_ROWS] * PAGE_ROWS + logical % PAGE_ROWS
    bucket_members = torch.full((buckets, BUCKET_CAP), -1, device="cuda", dtype=torch.int32)
    members = logical[:, None] * BUCKET_SIZE + torch.arange(BUCKET_SIZE, device="cuda")
    bucket_members[cache_rows, :BUCKET_SIZE] = members.int()
    stored = torch.empty_like(bucket_centroids)
    stored[cache_rows] = bucket_centroids
    positions = torch.arange(context - CHUNK, context, device="cuda", dtype=torch.int32)
    return Cache(
        index_queries=index_queries,
        bucket_centroids=stored.bfloat16(),
        bucket_members=bucket_members,
        bucket_sizes=torch.full((buckets,), BUCKET_SIZE, device="cuda", dtype=torch.int32),
        pages=pages,
        positions=positions,
        visible_counts=(positions // CLUSTER_BLOCK_SIZE * PAGE_ROWS).int(),
    )


def logical_centroids(context: int) -> torch.Tensor:
    """Return the cache's centroids in logical order, as the flat prefill scan reads them.

    Args:
        context: Tokens in the request.

    Returns:
        Bucket centroids [N, 64] bf16.
    """
    request = bucket_cache(context)
    logical = torch.arange(request.bucket_centroids.shape[0], device="cuda")
    cache_rows = request.pages.long()[logical // PAGE_ROWS] * PAGE_ROWS + logical % PAGE_ROWS
    return request.bucket_centroids[cache_rows].contiguous()


def sample_plan() -> tuple[int, int]:
    """Return the sample stride and the rank a threshold is read at, as the served select sets them."""
    stride = max(1, round(triton_scan_select.TARGET * TOP_P / triton_scan_select.SAMPLE_RANK))
    return stride, max(1, round(triton_scan_select.TARGET * TOP_P / stride))


@cache
def flat_listing(context: int) -> Listing:
    """Run the flat scan up to its keys, so later steps read what a real threshold leaves.

    Args:
        context: Tokens in the request.
    """
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    found = thresholds.sample_thresholds(
        request.index_queries, centroids, request.visible_counts, *sample_plan()
    )
    shifted = found - triton_scan_select.MARGIN * request.index_queries.float().norm(dim=1)
    buckets, survivor_counts = bucket_scan.scan_buckets(
        request.index_queries,
        centroids,
        request.visible_counts,
        shifted,
        triton_scan_select.CAPACITY * TOP_P,
    )
    keys, _ = scoring.score_candidates(request.index_queries, centroids, buckets, survivor_counts, found)
    return Listing(found, shifted, buckets, survivor_counts, keys)


def score_candidates_triton(
    index_queries: torch.Tensor, centroids: torch.Tensor, state: Listing
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each row's listed buckets into keys with the Triton kernel.

    Args:
        index_queries: Index queries [R, 64] bf16.
        centroids: Bucket centroids [N, 64] bf16.
        state: The flat scan's state, whose listing is scored.

    Returns:
        Packed keys [R, capacity] int64, the empty key after a row's listing.
        Listed buckets each row scored at or above its threshold [R] int32.
    """
    rows, capacity = state.buckets.shape
    keys = torch.full((rows, capacity), EMPTY, device="cuda", dtype=torch.int64)
    reached = torch.empty(rows, device="cuda", dtype=torch.int32)
    triton_scan_select._score_buckets[(rows,)](
        index_queries,
        centroids,
        state.buckets,
        state.survivor_counts,
        state.thresholds,
        keys,
        reached,
        capacity,
        D=INDEX_DIM,
        BLOCK=triton_scan_select.SCORE_BLOCK,
        SPAN=0,
        DENSE=False,
        num_warps=4,
        **triton_scan_select.shape_arguments(index_queries, None, None),
    )
    return keys, reached


def triton_beam_step(
    index_queries: torch.Tensor,
    open_scores: torch.Tensor,
    open_children: torch.Tensor,
    carried: torch.Tensor,
    centroids: torch.Tensor,
    closed_children: torch.Tensor,
) -> torch.Tensor:
    """Run one beam step as the Triton chain - score the carried children, keep the best, name their children.

    Args:
        index_queries: Index queries [R, 64] bf16.
        open_scores: Scores of the open node buckets [R, open] fp32.
        open_children: Children of each open node bucket [open, node_bucket_size] int32.
        carried: Carried children [R, carried_width] int32, -1 for none.
        centroids: Node centroids [nodes, 64] fp8.
        closed_children: Children of each node bucket [nodes, node_bucket_size] int32.

    Returns:
        The kept children [R, beam_width * node_bucket_size] int32.
    """
    rows, open_count = open_scores.shape
    width = open_count + carried.shape[1]
    scores = torch.empty(rows, width, device="cuda")
    scores[:, :open_count] = open_scores
    triton_beam.score_children(
        index_queries, centroids, carried, out=scores[:, open_count:], centroid_scale=NODE_CENTROID_SCALE
    )
    kept = TopK(BEAM_WIDTH)(scores, torch.full((rows,), width, device="cuda", dtype=torch.int32))
    return triton_beam.kept_children(
        scores, kept, open_count, open_children, carried, closed_children, NODE_BUCKET_SIZE
    )


def triton_start_level_keep(scores: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
    """Run the keep of a tree's start level as the Triton chain - the best columns, then their children.

    Args:
        scores: Scores of the start nodes [R, nodes] fp32.
        children: Children of each start node [nodes, node_bucket_size] int32.

    Returns:
        The kept children [R, beam_width * node_bucket_size] int32.
    """
    rows, width = scores.shape
    kept = TopK(BEAM_WIDTH)(scores, torch.full((rows,), width, device="cuda", dtype=torch.int32))
    return triton_beam.kept_children(scores, kept, width, children, None, None, NODE_BUCKET_SIZE)


def triton_top_p_pick(
    index_queries: torch.Tensor,
    open_scores: torch.Tensor,
    carried: torch.Tensor,
    bucket_centroids: torch.Tensor,
    pages: torch.Tensor,
    covered: int,
) -> torch.Tensor:
    """Run level 0's pick as the Triton chain - score the carried buckets, take the exact top-p, key it.

    Args:
        index_queries: Index queries [R, 64] bf16.
        open_scores: Scores of the uncovered buckets [R, open] fp32.
        carried: Carried child buckets [R, carried_width] int32, -1 for none.
        bucket_centroids: The cache's bucket centroids [rows, 64] bf16.
        pages: The request's pages [pages] int32.
        covered: Buckets the last beam's children cover.

    Returns:
        Packed keys of the picked buckets [R, top_p] int64.
    """
    rows, open_count = open_scores.shape
    width = open_count + carried.shape[1]
    scores = torch.empty(rows, width, device="cuda")
    scores[:, :open_count] = open_scores
    triton_beam.score_children(index_queries, bucket_centroids, carried, out=scores[:, open_count:])
    keys, listed = TopK(TOP_P, exact=True).listed(
        scores, torch.full((rows,), width, device="cuda", dtype=torch.int32)
    )
    return triton_beam.bucket_keys(keys, listed, carried, pages, open_count, covered, PAGE_ROWS)


def triton_decode_listing(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    pages: torch.Tensor,
    counts: torch.Tensor,
    max_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a decode step's select as the Triton chain - the paged scan, then the exact top-p listing.

    Args:
        index_queries: One index query a request [B, 64] bf16.
        bucket_centroids: The cache's bucket centroids [rows, 64] bf16.
        pages: Each request's pages [B, max_pages] int32.
        counts: Buckets each request sees [B] int32.
        max_rows: Buckets a request may see.

    Returns:
        Packed keys [B, top_p] int64 of the best buckets.
        Keys each request lists [B] int32.
    """
    scores = scan_paged(index_queries, bucket_centroids, pages, counts, PAGE_ROWS, max_rows)
    return TopK(TOP_P, index_queries.shape[0], "cuda", exact=True).listed(scores, counts)


def launch_threshold_estimate(context: int, kernels: str) -> Launch | None:
    """Launch the threshold estimate - the sums, the histogram and the threshold.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) < SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    stride, rank = sample_plan()
    if kernels == "triton":
        return lambda: scan_select.sample_thresholds(
            request.index_queries, centroids, request.visible_counts, TOP_P, kernels=kernels
        )
    return lambda: thresholds.sample_thresholds(
        request.index_queries, centroids, request.visible_counts, stride, rank
    )


def launch_bucket_scan(context: int, kernels: str) -> Launch | None:
    """Launch the bucket scan that lists each row's buckets at or above its threshold.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) < SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    shifted = flat_listing(context).shifted
    capacity = triton_scan_select.CAPACITY * TOP_P
    if kernels == "triton":
        return lambda: triton_scan_select.word_listing(
            request.index_queries, centroids, request.visible_counts, shifted, capacity
        )
    return lambda: bucket_scan.scan_buckets(
        request.index_queries, centroids, request.visible_counts, shifted, capacity
    )


def launch_candidate_scoring(context: int, kernels: str) -> Launch | None:
    """Launch the candidate scoring of each row's listed buckets into keys.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) < SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    state = flat_listing(context)
    if kernels == "triton":
        return lambda: score_candidates_triton(request.index_queries, centroids, state)
    return lambda: scoring.score_candidates(
        request.index_queries, centroids, state.buckets, state.survivor_counts, state.thresholds
    )


def launch_dense_scoring(context: int, kernels: str) -> Launch | None:
    """Launch the dense scoring of one batch of rows whose listing may miss a bucket.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) < SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    index_queries = request.index_queries[:DENSE_ROWS].contiguous()
    visible_counts = request.visible_counts[:DENSE_ROWS].contiguous()
    if kernels == "triton":
        return lambda: scan_select.dense_keys(index_queries, centroids, visible_counts, kernels=kernels)
    return lambda: scoring.score_dense(index_queries, centroids, visible_counts)


def launch_short_context_listing(context: int, kernels: str) -> Launch | None:
    """Launch the short-context listing of a short prefill over a dense score matrix.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) >= SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    centroids = logical_centroids(context)
    scores = torch.mm(request.index_queries, centroids.t(), out_dtype=torch.float32)
    if kernels == "triton":
        return lambda: TopK(TOP_P).listed(scores, request.visible_counts)
    return lambda: top_k.list_rows(scores, request.visible_counts, TOP_P)


def launch_expansion_by_threshold(context: int, kernels: str) -> Launch | None:
    """Launch the expansion by threshold over the flat scan's own listing.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    if bucket_count(context) < SHORT_BUCKETS:
        return None
    request = bucket_cache(context)
    state = flat_listing(context)
    request_rows = torch.zeros(CHUNK, device="cuda", dtype=torch.int64)
    request_pages = request.pages[None].contiguous()
    expand = front_expansion.expand_by_threshold if kernels == "triton" else expansion.expand_by_threshold
    return lambda: expand(
        state.keys,
        state.survivor_counts,
        request.bucket_members,
        request.bucket_sizes,
        request_pages,
        request_rows,
        request.positions,
        PAGE_ROWS,
        CLUSTER_BLOCK_SIZE,
        TOP_P,
        WINDOW_SIZE,
        SINK_SIZE,
        TOP_K,
    )


def launch_expansion_in_rank_order(context: int, kernels: str) -> Launch:
    """Launch the expansion in rank order of each row's picked buckets, best first.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    request = bucket_cache(context)
    selected = torch.randint(
        0, request.bucket_members.shape[0], (CHUNK, TOP_P), device="cuda", generator=generator()
    ).int()
    if kernels == "triton":
        return lambda: front_expansion.expand_in_order(
            selected,
            request.bucket_members,
            request.positions,
            WINDOW_SIZE,
            SINK_SIZE,
            TOP_K,
            kernels=kernels,
        )
    return lambda: expansion.expand_in_order(
        selected, request.bucket_members, request.positions, WINDOW_SIZE, SINK_SIZE, TOP_K
    )


def launch_beam_step(context: int, kernels: str) -> Launch:
    """Launch one level of the beam - the open node buckets and the carried children scored, the best kept.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    request = bucket_cache(context)
    node_buckets = bucket_count(context) // NODE_BUCKET_SIZE
    centroids = (unit_rows(node_buckets, rows_generator) * NODE_CENTROID_SCALE).to(torch.float8_e4m3fn)
    open_centroids = unit_rows(OPEN_NODES, rows_generator).bfloat16()
    open_scores = torch.mm(request.index_queries, open_centroids.t(), out_dtype=torch.float32)
    open_children = torch.randint(
        0, node_buckets, (OPEN_NODES, NODE_BUCKET_SIZE), device="cuda", generator=rows_generator
    ).int()
    closed_children = torch.randint(
        0, node_buckets, (node_buckets, NODE_BUCKET_SIZE), device="cuda", generator=rows_generator
    ).int()
    carried = torch.randint(
        0, node_buckets, (CHUNK, BEAM_WIDTH * NODE_BUCKET_SIZE), device="cuda", generator=rows_generator
    ).int()
    if kernels == "triton":
        return lambda: triton_beam_step(
            request.index_queries, open_scores, open_children, carried, centroids, closed_children
        )
    return lambda: beam.beam_step(
        request.index_queries,
        open_scores,
        open_children,
        carried,
        centroids,
        closed_children,
        BEAM_WIDTH,
        NODE_CENTROID_SCALE,
    )


def launch_start_level_keep(context: int, kernels: str) -> Launch:
    """Launch the start-level keep of a tree, listed first since it is wider than shared memory stages.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    request = bucket_cache(context)
    start_centroids = unit_rows(START_NODES, rows_generator).bfloat16()
    scores = torch.mm(request.index_queries, start_centroids.t(), out_dtype=torch.float32)
    children = torch.randint(
        0, 10**6, (START_NODES, NODE_BUCKET_SIZE), device="cuda", generator=rows_generator
    ).int()
    if kernels == "triton":
        return lambda: triton_start_level_keep(scores, children)
    return lambda: beam.keep_start_level(request.index_queries, scores, children, BEAM_WIDTH)


def launch_top_p_pick(context: int, kernels: str) -> Launch:
    """Launch the top-p pick of level 0 - the last beam's child buckets beside the uncovered ones, the exact top-p kept.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    request = bucket_cache(context)
    buckets = bucket_count(context)
    open_scores = torch.mm(
        request.index_queries,
        unit_rows(OPEN_BUCKETS, rows_generator).bfloat16().t(),
        out_dtype=torch.float32,
    )
    carried = torch.randint(
        0, buckets, (CHUNK, BEAM_WIDTH * NODE_BUCKET_SIZE), device="cuda", generator=rows_generator
    ).int()
    covered = buckets - OPEN_BUCKETS
    if kernels == "triton":
        return lambda: triton_top_p_pick(
            request.index_queries, open_scores, carried, request.bucket_centroids, request.pages, covered
        )
    return lambda: beam.pick_top_p(
        request.index_queries,
        open_scores,
        carried,
        request.bucket_centroids,
        request.pages,
        covered,
        PAGE_ROWS,
        TOP_P,
    )


def batch_counts(buckets: int, batch: int) -> list[int]:
    """Return the buckets each request of a decode batch sees - the first all of them, the rest shorter or short.

    A batch of equal rows would hide the paths a short or exact row takes, so the second request sees a third of
    the buckets, whole pages of them, the third 1,500 and the fourth 700, and the rest see them all.

    Args:
        buckets: Buckets the first request sees, at most.
        batch: Requests in the batch.
    """
    third = buckets // 3 // PAGE_ROWS * PAGE_ROWS + 5
    cycle = [buckets, min(third, buckets), min(1500, buckets), min(700, buckets)]
    return [cycle[i] if i < len(cycle) else buckets for i in range(batch)]


def launch_decode_select_batch(context: int, batch: int, kernels: str) -> Launch:
    """Launch a flat decode step's select for a batch of requests - sample, threshold, scan and order.

    Args:
        context: Tokens in each request.
        batch: Requests in the batch, sharing one request's pages and the last queries of its chunk.
        kernels: `cutedsl` or `triton`.
    """
    request = bucket_cache(context)
    buckets = bucket_count(context)
    index_queries = request.index_queries[-batch:].contiguous()
    pages = request.pages[None].repeat(batch, 1).contiguous()
    counts = torch.tensor(batch_counts(buckets, batch), device="cuda", dtype=torch.int32)
    if kernels == "triton":
        return lambda: triton_decode_listing(index_queries, request.bucket_centroids, pages, counts, buckets)
    return lambda: decode.decode_listing(
        index_queries, request.bucket_centroids, pages, counts, PAGE_ROWS, buckets, TOP_P
    )


def launch_decode_select(context: int, kernels: str) -> Launch:
    """Launch a flat decode step's select for one request - sample, threshold, scan and order.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    return launch_decode_select_batch(context, 1, kernels)


def launch_ward_clustering(context: int, kernels: str) -> Launch:
    """Launch the ward clustering of a chunk's cluster blocks, each one's pools merged into its buckets.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    blocks = CHUNK // CLUSTER_BLOCK_SIZE
    # A random walk, so neighbouring keys are alike as a text's are
    steps = 0.3 * torch.randn(blocks, CLUSTER_BLOCK_SIZE, INDEX_DIM, device="cuda", generator=rows_generator)
    index_keys = F.normalize(torch.cumsum(steps, 1), dim=-1)
    pools = pool_keys(index_keys, POOL_SIZE).contiguous()
    sizes = torch.full(pools.shape[:2], float(POOL_SIZE), device="cuda")
    bucket_total = CLUSTER_BLOCK_SIZE // BUCKET_SIZE
    # A GPU the CuTeDSL clustering does not run on reads it as `kernels: cutedsl` serves it, in Triton
    if kernels == "triton" or not dispatch.runs_cutedsl("ward clustering", kernels, pools.device):
        return lambda: front_ward.cluster_ward(pools, sizes, bucket_total, BUCKET_CAP, kernels=kernels)
    return lambda: ward.cluster_ward(pools, sizes, bucket_total, BUCKET_CAP)


def launch_node_bucket_clustering(context: int, kernels: str) -> Launch:
    """Launch one node-bucket clustering of a growth - every bucket of a node cluster block paired with its cheapest partner.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    steps = 0.3 * torch.randn(
        1, PAIRED_BUCKETS * BUCKET_SIZE, INDEX_DIM, device="cuda", generator=rows_generator
    )
    index_keys = F.normalize(torch.cumsum(steps, 1), dim=-1)
    key_sums = index_keys.view(1, PAIRED_BUCKETS, BUCKET_SIZE, INDEX_DIM).sum(2).contiguous()
    # A GPU the CuTeDSL clustering does not run on reads it as `kernels: cutedsl` serves it, in Triton
    if kernels == "triton" or not dispatch.runs_cutedsl("node-bucket clustering", kernels, key_sums.device):
        return lambda: front_ward.cluster_node_buckets(key_sums, kernels=kernels)
    return lambda: node_bucket_clustering.cluster_node_buckets(key_sums)


def launch_reference_gemm(context: int, kernels: str) -> Launch:
    """Launch a square bf16 GEMM through cuBLAS, the roofline's anchor at the tensor-core peak.

    Args:
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.
    """
    rows_generator = generator()
    a = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda", generator=rows_generator).bfloat16()
    b = torch.randn(GEMM_DIM, GEMM_DIM, device="cuda", generator=rows_generator).bfloat16()
    return lambda: torch.mm(a, b)


STEPS: tuple[Step, ...] = (
    Step("threshold estimate", "prefill scan", True, launch_threshold_estimate),
    Step("bucket scan", "prefill scan", True, launch_bucket_scan),
    Step("candidate scoring", "prefill scan", True, launch_candidate_scoring),
    Step("dense scoring", "prefill scan", True, launch_dense_scoring),
    Step("short-context listing", "prefill scan", True, launch_short_context_listing),
    Step("expansion by threshold", "expansion", True, launch_expansion_by_threshold),
    Step("expansion in rank order", "expansion", False, launch_expansion_in_rank_order),
    Step("start-level keep", "tree", False, launch_start_level_keep),
    Step("beam step", "tree", True, launch_beam_step),
    Step("top-p pick", "tree", True, launch_top_p_pick),
    Step("decode select", "decode", True, launch_decode_select, launch_decode_select_batch),
    Step("ward clustering", "growth", False, launch_ward_clustering),
    Step("node-bucket clustering", "growth", False, launch_node_bucket_clustering),
    Step("reference gemm", "anchor", False, launch_reference_gemm),
)


def launcher(step: Step, context: int, kernels: str) -> Launch | None:
    """Build a step's launcher in one language.

    Args:
        step: The step.
        context: Tokens in the request.
        kernels: `cutedsl` or `triton`.

    Raises:
        ValueError: The language is neither.
    """
    if kernels not in KERNELS:
        raise ValueError(f"unknown kernels {kernels!r} - the arms are {', '.join(KERNELS)}")
    return step.build(context, kernels)


def step_named(name: str) -> Step:
    """Return the step of that name.

    Args:
        name: A step's name, as dispatch's table spells it.

    Raises:
        ValueError: No step has that name.
    """
    for step in STEPS:
        if step.name == name:
            return step
    raise ValueError(f"no step named {name!r} - the steps are {', '.join(step.name for step in STEPS)}")
