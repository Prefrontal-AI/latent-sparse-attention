"""The top-p of a prefill chunk's bucket scores - the one place `dense_keys`, `listed_keys` and `sample_thresholds` choose a language.

Under `kernels: cutedsl` a step runs its CuTeDSL kernel when [dispatch](dispatch.py)'s table names the GPU and the
kernel's own `supports` takes the rows, and only for unshaped buckets; anything else runs the
[Triton kernel](triton/scan_select.py), which reads the same.
`ScanSelect` orders the steps - threshold estimate, bucket scan, candidate scoring, and a wider bucket scan or a dense scoring for a row that
may have missed - and holds the language they run in.
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import bucket_scan, scoring
from lsa.core.kernels.cutedsl import thresholds as cutedsl_thresholds
from lsa.core.kernels.triton import scan_select as triton_scan_select
from lsa.core.kernels.triton.keys import unpack_ids
from lsa.core.kernels.triton.scan_select import CAPACITY, DENSE_ROWS, MARGIN, WIDE, WIDE_ENTRIES
from lsa.core.kernels.triton.shape import ShapeTable, SignQuery, sign_query


def dense_keys(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    width: int | None = None,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
    kernels: str = "triton",
) -> torch.Tensor:
    """Score every bucket each row sees and pack it with its bucket, at its own column.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16.
        visible_counts: Buckets each row sees [R] int32, at most N.
        width: Columns of the result, at least N - N when None.
        shapes: The buckets' shapes, row for row, to score each by its estimated best key - None for the centroid.
        query_planes: The queries' planes, when shaped and already split.
        kernels: `cutedsl` to score unshaped 64-dim bf16 rows in CuTeDSL, `triton` to score every row in Triton - the
            same keys either way.

    Returns:
        Packed keys [R, width] int64, the empty key past a row's length.
    """
    if (
        dispatch.runs_cutedsl("dense scoring", kernels, index_queries.device)
        and shapes is None
        and scoring.supports(index_queries, bucket_centroids)
    ):
        return scoring.score_dense(index_queries, bucket_centroids, visible_counts, width)
    return triton_scan_select.dense_keys(
        index_queries, bucket_centroids, visible_counts, width, shapes, query_planes
    )


def listed_keys(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    thresholds: torch.Tensor,
    capacity: int,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
    kernels: str = "triton",
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
        kernels: `cutedsl` to list unshaped 64-dim bf16 rows on Hopper or Blackwell and score them in CuTeDSL, `triton` to
            list every row by survivor words and score it in Triton.

    Returns:
        Packed keys [R, capacity] int64 of the first `capacity` listed buckets, the empty key after.
        Buckets each row listed [R] int32 - above capacity for a row whose buffer overflowed.
        Listed buckets each row scored at or above its threshold [R] int32.
    """
    device = index_queries.device
    # Both list the same buckets and add each score in the same order
    # => CuTeDSL lists and scores where asked for and it takes the rows, survivor words and Triton the rest
    if (
        dispatch.runs_cutedsl("bucket scan", kernels, device)
        and dispatch.runs_cutedsl("candidate scoring", kernels, device)
        and shapes is None
        and bucket_scan.supports(index_queries, bucket_centroids)
    ):
        shifted = thresholds - MARGIN * index_queries.float().norm(dim=1)
        buckets, survivor_counts = bucket_scan.scan_buckets(
            index_queries, bucket_centroids, visible_counts, shifted, capacity
        )
        keys, reached = scoring.score_candidates(
            index_queries, bucket_centroids, buckets, survivor_counts, thresholds
        )
        return keys, survivor_counts, reached
    return triton_scan_select.listed_keys(
        index_queries, bucket_centroids, visible_counts, thresholds, capacity, shapes, query_planes
    )


def sample_thresholds(
    index_queries: torch.Tensor,
    bucket_centroids: torch.Tensor,
    visible_counts: torch.Tensor,
    top_p: int,
    shapes: ShapeTable | None = None,
    query_planes: SignQuery | None = None,
    kernels: str = "triton",
) -> torch.Tensor:
    """Estimate each row's score at TARGET times its top_p from every stride-th bucket it sees.

    Args:
        index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
        bucket_centroids: Bucket centroids [N, index_dim] bf16 of unit rows.
        visible_counts: Buckets each row sees [R] int32, at most N.
        top_p: Buckets each row selects.
        shapes: The buckets' shapes, row for row, to estimate the shaped scores - None for the centroid.
        query_planes: The queries' planes, when shaped and already split.
        kernels: `cutedsl` to keep the sampled scores on chip, `triton` to write them - the same thresholds.

    Returns:
        Estimated scores [R] fp32, -inf for a row seeing too few to rank.
    """
    if (
        dispatch.runs_cutedsl("threshold estimate", kernels, index_queries.device)
        and shapes is None
        and cutedsl_thresholds.supports(index_queries, bucket_centroids)
    ):
        stride, rank = triton_scan_select.sample_plan(top_p)
        return cutedsl_thresholds.sample_thresholds(
            index_queries, bucket_centroids, visible_counts, stride, rank
        )
    return triton_scan_select.sample_thresholds(
        index_queries, bucket_centroids, visible_counts, top_p, shapes, query_planes
    )


def rows_where(mask: torch.Tensor, count: int) -> torch.Tensor:
    """Return the indices of the `count` rows a mask holds, ascending - what `nonzero` returns, without reading the device for its size.

    Args:
        mask: A mask [R] bool.
        count: Rows it holds.
    """
    return torch.argsort(mask.int(), descending=True, stable=True)[:count]


class Listing(NamedTuple):
    """A chunk's rows listed near their thresholds, with what a fallback needs to score them again."""

    index_queries: torch.Tensor
    bucket_centroids: torch.Tensor
    visible_counts: torch.Tensor
    thresholds: torch.Tensor
    keys: torch.Tensor
    survivor_counts: torch.Tensor
    reached: torch.Tensor
    shapes: ShapeTable | None = None
    query_planes: SignQuery | None = None

    def take(self, rows: torch.Tensor) -> "Listing":
        """Return the listing of some of the rows, in the order given.

        Args:
            rows: The rows [F] int64.
        """
        planes = None if self.query_planes is None else SignQuery(*(part[rows] for part in self.query_planes))
        return Listing(
            self.index_queries[rows],
            self.bucket_centroids,
            self.visible_counts[rows],
            self.thresholds[rows],
            self.keys[rows],
            self.survivor_counts[rows],
            self.reached[rows],
            self.shapes,
            planes,
        )


class ScanSelect:
    """The top-p buckets of a prefill chunk's bf16 centroid scores, into an output buffer sized once."""

    def __init__(
        self, rows: int, top_p: int, device: str | torch.device = "cuda", kernels: str = "triton"
    ) -> None:
        """Size the output buffer.

        Args:
            rows: Most rows a call picks for.
            top_p: Buckets each row picks.
            device: Where the output lives.
            kernels: The thresholds' and listing's language, `triton` or `cutedsl` - the same picks either way.
        """
        self.top_p = top_p
        self.kernels = kernels
        self.out = torch.empty(rows, top_p, device=device, dtype=torch.int32)

    def __call__(
        self,
        index_queries: torch.Tensor,
        bucket_centroids: torch.Tensor,
        visible_counts: torch.Tensor,
        shapes: ShapeTable | None = None,
    ) -> torch.Tensor:
        """Pick each row's top-p buckets - `listed`, then `pick`.

        Args:
            index_queries: Index queries [R, index_dim] bf16, of any norm.
            bucket_centroids: Bucket centroids [N, index_dim] bf16 of unit rows.
            visible_counts: Buckets each row sees [R] int32, at most N.
            shapes: The buckets' shapes, row for row, or None for the centroid.

        Returns:
            Buckets [R, top_p] int32, the best score first and a tie to the lower bucket, -1 past a row's
            visible count.
        """
        return self.pick(self.listed(index_queries, bucket_centroids, visible_counts, shapes))

    def listed(
        self,
        index_queries: torch.Tensor,
        bucket_centroids: torch.Tensor,
        visible_counts: torch.Tensor,
        shapes: ShapeTable | None = None,
    ) -> Listing:
        """Estimate each row's threshold, then list and score the buckets near it.

        Args:
            index_queries: Index queries [R, index_dim] bf16, of any norm - unit rows when the buckets are shaped.
            bucket_centroids: Bucket centroids [N, index_dim] bf16 of unit rows.
            visible_counts: Buckets each row sees [R] int32, at most N.
            shapes: The buckets' shapes, row for row, to score each by its estimated best key - None for the
                centroid.

        Returns:
            The rows' listing.
        """
        planes = None if shapes is None else sign_query(index_queries)
        thresholds = sample_thresholds(
            index_queries, bucket_centroids, visible_counts, self.top_p, shapes, planes, self.kernels
        )
        keys, survivor_counts, reached = listed_keys(
            index_queries,
            bucket_centroids,
            visible_counts,
            thresholds,
            round(CAPACITY * self.top_p),
            shapes,
            planes,
            self.kernels,
        )
        return Listing(
            index_queries,
            bucket_centroids,
            visible_counts,
            thresholds,
            keys,
            survivor_counts,
            reached,
            shapes,
            planes,
        )

    def holds_top_p(self, listing: Listing) -> bool:
        """Tell whether every row's top-p is among its listed buckets - no row to list again or score in full.

        Args:
            listing: The rows' listing.

        Returns:
            True when no row overflowed its buffer or reached fewer than `top_p` at its threshold.
        """
        return not any(self.missed_runs(listing, [(0, len(listing.visible_counts))]))

    def missed_runs(self, listing: Listing, runs: Sequence[tuple[int, int]]) -> list[bool]:
        """Tell for each run of rows whether any row's top-p may not be among its listed buckets - one read of the device.

        Args:
            listing: The rows' listing.
            runs: Consecutive row ranges [start, stop) to answer for.

        Returns:
            For each run, True when a row of it overflowed its buffer or reached fewer than `top_p` at its threshold.
        """
        missed = (listing.survivor_counts > listing.keys.shape[1]) | (listing.reached < self.top_p)
        return torch.stack([missed[start:stop].any() for start, stop in runs]).tolist()

    def pick(self, listing: Listing) -> torch.Tensor:
        """Pick each listed row's best buckets, listing again or scoring in full a row that may have missed one.

        Args:
            listing: The rows' listing.

        Returns:
            Buckets [R, top_p] int32, the best score first and a tie to the lower bucket, -1 past a row's
            visible count.
        """
        index_queries, bucket_centroids, visible_counts = (
            listing.index_queries,
            listing.bucket_centroids,
            listing.visible_counts,
        )
        keys, survivor_counts, shapes, planes = (
            listing.keys,
            listing.survivor_counts,
            listing.shapes,
            listing.query_planes,
        )
        reached = listing.reached.clone()
        best = keys.topk(self.top_p, dim=1).values
        overflowed = survivor_counts > keys.shape[1]
        starved = reached < self.top_p
        # One read of the device for how many rows overflowed, how wide the widest is and how many are short
        overflow_count, widest, short_count = torch.stack(
            [overflowed.sum(), torch.where(overflowed, survivor_counts, 0).max(), starved.sum()]
        ).tolist()
        # Buckets tied at the threshold can overflow a buffer
        # => list those rows again into one as wide as they need
        if overflow_count:
            self.list_wide(listing, rows_where(overflowed, overflow_count), widest, best, reached)
            short = (reached < self.top_p).nonzero().flatten()
        else:
            short = rows_where(starved, short_count)
        # A row with fewer than top_p at its threshold may have its best below it
        # => score it in full
        for batch in short.split(DENSE_ROWS):
            width = max(bucket_centroids.shape[0], self.top_p)
            if shapes is None or planes is None:
                full = dense_keys(
                    index_queries[batch], bucket_centroids, visible_counts[batch], width, kernels=self.kernels
                )
            else:
                full = dense_keys(
                    index_queries[batch],
                    bucket_centroids,
                    visible_counts[batch],
                    width,
                    shapes=shapes,
                    query_planes=SignQuery(*(part[batch] for part in planes)),
                )
            best[batch] = full.topk(self.top_p, dim=1).values
        out = self.out[: index_queries.shape[0]]
        out.copy_(unpack_ids(best))
        return out

    def list_wide(
        self, listing: Listing, rows: torch.Tensor, width: int, best: torch.Tensor, reached: torch.Tensor
    ) -> None:
        """List overflowed rows again into buffers `width` wide, a batch at a time, and rank them in place.

        A row wider than WIDE times top_p is left for the dense scoring, its reached count zeroed.

        Args:
            listing: The rows' listing.
            rows: The overflowed rows [F] int64.
            width: Buckets the widest of them lists.
            best: Best packed keys [R, top_p] int64, written for the rows.
            reached: Listed buckets at or above each threshold [R] int32, written for the rows.
        """
        if width > WIDE * self.top_p:
            reached[rows] = 0
            return
        planes = listing.query_planes
        for batch in rows.split(max(1, WIDE_ENTRIES // width)):
            wide, _, wide_reached = listed_keys(
                listing.index_queries[batch],
                listing.bucket_centroids,
                listing.visible_counts[batch],
                listing.thresholds[batch],
                width,
                listing.shapes,
                None if planes is None else SignQuery(*(part[batch] for part in planes)),
                self.kernels,
            )
            best[batch] = wide.topk(self.top_p, dim=1).values
            reached[batch] = wide_reached
