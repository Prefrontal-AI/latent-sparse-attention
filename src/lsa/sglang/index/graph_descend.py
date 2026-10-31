"""The beam down every level for a whole decode batch at once, at shapes that never follow a request - for a CUDA graph.

The eager descend walks one request's hierarchy, whose node buckets grow and move. This one reads the graph tree's
pooled mirror instead: every row scores every level, from the top level the longest context grows down to level 1,
and a row scores only the levels at and below its own start level - the others hold no candidate at all.

A level's candidates are node bucket rows of the level's pool, -1 for none, in the order the eager path scores them,
so the keep breaks a tie the same way. A start level scores every node bucket of the row, the open ones then the closed
ones. Below it a row scores its open node buckets, then the children the level above kept. Level 0 scores the buckets
no node bucket covers, then the last beam's children, and keeps the best `top_p`.

The widths are set once from the design and the longest context: no level of a request holds more node buckets than the
facts of the growth allow, so a candidate row never needs more columns than its plan holds.
"""

from collections.abc import Callable
from typing import NamedTuple

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import beam as cutedsl_beam
from lsa.core.kernels.cutedsl import spread
from lsa.core.kernels.triton import beam
from lsa.core.kernels.triton.beam import NODE_CENTROID_SCALE
from lsa.core.kernels.triton.shape import score_shaped_rows
from lsa.core.kernels.triton.top_k import TopK
from lsa.sglang.index import hierarchy
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.graph_tree import NODE_BUCKET_PAGE, GraphTree
from lsa.sglang.settings.design import DesignSettings

# The widest level one CTA scores and keeps: it gathers a query's centroids alone, so a wider row runs behind a scorer
# that spreads it over the SMs
# => a wider level is scored apart, then kept by one CTA
SINGLE_CTA_WIDTH = 512


class LevelPlan(NamedTuple):
    """The fixed columns a level scores and the columns it keeps.

    `start_width` is the most node buckets a start level holds, `open_width` the most open node buckets a level below
    its start holds, `keep` the columns the beam keeps and `width` the columns a row of candidates has.
    """

    start_width: int
    open_width: int
    keep: int
    width: int


def plan_levels(settings: DesignSettings, tree: GraphTree) -> list[LevelPlan]:
    """Return every level's widths, from the design and the node buckets the longest context makes.

    Every level but the top the design grows holds fewer open node buckets than the next level's block, and a start
    level holds at most `start_nodes` of them or is the top grown level, so it holds fewer than the next block as well.

    Args:
        settings: The design.
        tree: The graph tree, whose levels the plans cover.

    Returns:
        A plan for each level, level 1 first.
    """
    config = settings.indexer
    plans = [LevelPlan(0, 0, 0, 0)] * tree.level_count
    carried = 0
    for level in range(tree.level_count, 0, -1):
        most = tree.most_nodes[level - 1]
        if level < config.top_level:
            block = config.at("node_cluster_block_size", level + 1)
            open_width = min(most, block - 1)
            start_width = min(most, max(config.start_nodes or 0, block))
        else:
            open_width, start_width = 0, most
        width = max(start_width, open_width + carried)
        keep = min(config.at("beam_width", level), width)
        plans[level - 1] = LevelPlan(start_width, open_width, keep, width)
        carried = keep * tree.node_bucket_size
    return plans


def uncovered_width(settings: DesignSettings, rows_per_page: int) -> int:
    """Return the most buckets no node bucket covers - what a growth leaves, and a page of buckets that closes after it.

    Args:
        settings: The design.
        rows_per_page: Bucket rows a page holds.
    """
    block = settings.indexer.at("node_cluster_block_size", 1)
    return max(hierarchy.GROWTH_BUCKETS, block) + rows_per_page


def keep_kernel(plan: LevelPlan, index_queries: torch.Tensor, centroids: torch.Tensor, kernels: str) -> str:
    """Name the kernels that keep a level.

    Args:
        plan: The level's widths.
        index_queries: Index queries [B, index_dim] bf16.
        centroids: The level's node centroids.
        kernels: The `kernels` setting.

    Returns:
        `beam step` for a row one CTA scores, `scored keep` for a wider one, else `triton chain`.
    """
    row_shape = torch.empty(0, plan.width, device="meta", dtype=torch.int32)
    if not (
        dispatch.runs_cutedsl("beam step", kernels, index_queries.device)
        and cutedsl_beam.supports(index_queries, centroids, row_shape, 0, plan.keep)
    ):
        return "triton chain"
    return "beam step" if plan.width <= SINGLE_CTA_WIDTH else "scored keep"


def keep_children(
    tree: GraphTree,
    level: int,
    plan: LevelPlan,
    index_queries: torch.Tensor,
    rows: torch.Tensor,
    carried: torch.Tensor,
    kernels: str,
    beam_top_k: TopK,
) -> torch.Tensor:
    """Score each row's candidates of a level, keep the best `plan.keep` and name their children.

    A level that keeps every column it scores is scored and named in one Triton launch that writes the children too.
    A level of at most `SINGLE_CTA_WIDTH` columns is scored, kept and named by one CuTeDSL kernel.
    A wider one is scored by one Triton launch that names its candidates too, then kept and named by one CuTeDSL kernel.
    That kernel is a cluster of CTAs a row at a small batch, a CTA a row otherwise.
    Where the CuTeDSL kernels are not used, or the row does not fit a CTA, the Triton chain keeps it in four more launches.

    Args:
        tree: The graph tree.
        level: The level, from 1.
        plan: The level's widths.
        index_queries: Index queries [B, index_dim] bf16.
        rows: The request table rows [B] int64.
        carried: Creation indices the level above kept the children of [B, C] int32, -1 for none.
        kernels: The `kernels` setting.
        beam_top_k: The top-k that keeps `plan.keep` columns.

    Returns:
        The kept node buckets' children [B, keep * node_bucket_size] int32 - creation indices of the level below, cache
        rows under level 1 - -1 where a row kept nothing.
    """
    centroids, children = tree.node_centroids[level - 1], tree.children[level - 1]
    scale = NODE_CENTROID_SCALE if tree.settings.serving.node_centroid_precision == "fp8" else 1.0
    kernel = keep_kernel(plan, index_queries, centroids, kernels)
    page_arguments = (
        tree.pages[level - 1],
        tree.made[level - 1],
        tree.closed[level - 1],
        tree.start,
        rows,
    )
    if (
        kernel != "triton chain"
        and plan.keep >= plan.width
        and children.shape[1] & (children.shape[1] - 1) == 0
    ):
        # Every column is kept, so nothing is chosen
        # => write the children of each where the level scores it
        return beam.level_children(
            *page_arguments,
            carried,
            index_queries,
            centroids,
            children,
            level,
            NODE_BUCKET_PAGE,
            plan.width,
            scale,
        )
    if kernel == "scored keep":
        candidates, scores, lengths = beam.level_scores(
            *page_arguments, carried, index_queries, centroids, level, NODE_BUCKET_PAGE, plan.width, scale
        )
        chosen = spread.layout(len(index_queries), plan.width, plan.keep)
        if chosen is not None:
            return spread.keep_scored(scores, candidates, children, plan.keep, lengths, *chosen)
        return cutedsl_beam.keep_scored(
            index_queries, scores, candidates, centroids, children, plan.keep, lengths
        )
    candidates, lengths = beam.level_candidates(*page_arguments, carried, level, NODE_BUCKET_PAGE, plan.width)
    if kernel == "beam step":
        return cutedsl_beam.beam_step(
            index_queries,
            torch.empty(len(index_queries), 0, device=index_queries.device),
            children[:0],
            candidates,
            centroids,
            children,
            plan.keep,
            scale,
            lengths=lengths,
        )
    scores = beam.score_children(index_queries, centroids, candidates, centroid_scale=scale, lengths=lengths)
    kept = beam_top_k(scores, lengths)
    return beam.kept_children(scores, kept, 0, children, candidates, children, tree.node_bucket_size)


def pick_scored(
    scores: torch.Tensor, candidates: torch.Tensor, lengths: torch.Tensor, top_p: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each row's best `top_p` columns of a scored row - over a cluster of CTAs where that is faster, else a CTA.

    Args:
        scores: The row's scores [B, W] fp32.
        candidates: The cache row of each column [B, W] int32, -1 for none.
        lengths: Columns each row holds [B] int32.
        top_p: Buckets a row picks.

    Returns:
        The picked scores packed with their cache rows and the keys each row lists, as `cutedsl_beam.pick_scored`.
    """
    rows, width = scores.shape
    chosen = spread.layout(rows, width, min(top_p, width))
    if chosen is None:
        return cutedsl_beam.pick_scored(scores, candidates, lengths, top_p)
    return spread.pick_scored(scores, candidates, lengths, top_p, *chosen)


def pick_buckets(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    rows: torch.Tensor,
    tree: GraphTree,
    cache: BucketCache,
    width: int,
    carried: torch.Tensor,
    kernels: str,
    pick_top_k: TopK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each row's best buckets of level 0, the uncovered ones first and the last beam's children after.

    The buckets a row sees are those of the cluster blocks before its position, from the first no node bucket covers.

    Args:
        index_queries: Index queries [B, index_dim] bf16.
        positions: Each row's position [B] int32.
        rows: The request table rows [B] int64.
        tree: The graph tree.
        cache: The bucket cache.
        width: Uncovered bucket columns.
        carried: Cache rows the level 1 beam kept [B, C] int32, -1 for none.
        kernels: The `kernels` setting.
        pick_top_k: The exact top-k over `top_p` buckets.

    Returns:
        The picked scores packed with their cache rows [B, W] int64, the empty key for none, and the keys each row
        lists [B] int32.
    """
    config = cache.config
    shapes = cache.bucket_shapes
    arguments = (
        positions,
        rows,
        tree.covered,
        cache.request_pages,
        carried,
    )
    sizes = (
        cache.rows_per_page,
        config.cluster_block_size,
        config.cluster_block_size // config.bucket_size,
        width + carried.shape[1],
    )
    scored = dispatch.runs_cutedsl(
        "beam step", kernels, index_queries.device
    ) and cutedsl_beam.supports_scored_pick(sizes[-1], pick_top_k.k)
    if scored and shapes is None:
        candidates, scores, lengths = beam.pick_scores(
            *arguments, index_queries, cache.bucket_centroids, *sizes
        )
        return pick_scored(scores, candidates, lengths, pick_top_k.k)
    candidates, lengths = beam.pick_candidates(*arguments, *sizes)
    if shapes is None:
        scores = beam.score_children(index_queries, cache.bucket_centroids, candidates, lengths=lengths)
    else:
        scores = score_shaped_rows(
            index_queries,
            cache.bucket_centroids,
            shapes.basis,
            shapes.variances,
            shapes.floor,
            candidates,
            shapes.deviations,
        )
    if scored:
        return pick_scored(scores, candidates, lengths, pick_top_k.k)
    keys, listed = pick_top_k.listed(scores, lengths)
    packed = beam.cache_row_keys(keys, listed, candidates)
    pick_top_k.release()
    return packed, torch.full((len(packed),), packed.shape[1], device=packed.device, dtype=torch.int32)


def descend(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    rows: torch.Tensor,
    tree: GraphTree,
    cache: BucketCache,
    plans: list[LevelPlan],
    uncovered: int,
    kernels: str,
    beam_top_k: Callable[[int], TopK],
    pick_top_k: TopK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each row's best `top_p` buckets by a beam down the graph tree, at shapes the plans fix.

    A row with no levels scores no candidate at any level, and its picks are the uncovered buckets alone - every
    bucket it sees, since none is covered.

    Args:
        index_queries: One index query a row [B, index_dim] fp32, unit rows.
        positions: Each row's position [B] int32.
        rows: The request table rows [B] int64.
        tree: The graph tree.
        cache: The bucket cache.
        plans: Every level's widths, from `plan_levels`.
        uncovered: Uncovered bucket columns, from `uncovered_width`.
        kernels: The `kernels` setting.
        beam_top_k: The top-k over a level's scores, by the beam width it keeps.
        pick_top_k: The exact top-k over `top_p` buckets.

    Returns:
        The picked scores packed with their cache rows [B, W] int64 in no order, the empty key for none, and the keys
        each row lists [B] int32 - what `Selector.expand_bucket_keys` reads.
    """
    queries = index_queries.to(torch.bfloat16)
    carried = torch.empty(len(queries), 0, device=queries.device, dtype=torch.int32)
    for level in range(tree.level_count, 0, -1):
        # Level 1 names cache rows, the others creation indices of the level below
        carried = keep_children(
            tree, level, plans[level - 1], queries, rows, carried, kernels, beam_top_k(plans[level - 1].keep)
        )
    return pick_buckets(queries, positions, rows, tree, cache, uncovered, carried, kernels, pick_top_k)
