"""What a query reads - the bucket cache scored, picked and expanded to key ids, for every design.

The routing is the design's. A prefill query reads flat when no node bucket exists yet: a dense product and a listing
of its best below `SCAN_SELECT_BUCKETS` visible buckets, and the fused select from there, its overflow handed to the
walk. Once a hierarchy has a level it reads by a beam down the levels. A decode step scans every bucket of its request
inside the CUDA graph and lists its best, or reads the beam as a prefill query does, eagerly, when the design says
`tree`. Every listing is read by the expansion by threshold.

Under `bucket_shape` every read scores a bucket by its estimated best key rather than its centroid - the dense product
with the spread added, the fused select scoring the spread in its scan as well, the decode scan reading each bucket's
shape beside its centroid, and the beam's level 0 - all against the query's planes, so the reads agree.

Nothing here knows an engine. A model hands in the request table, the index keys of a position and a top-k, and
gets back lists of KV slots `[tokens, width]`, `-1` padded, valid entries first.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl import decode as cutedsl_decode
from lsa.core.kernels.cutedsl import top_k as cutedsl_top_k
from lsa.core.kernels.expansion import expand_by_threshold, expand_in_order
from lsa.core.kernels.scan_select import Listing, ScanSelect
from lsa.core.kernels.triton.paged import scan_paged
from lsa.core.kernels.triton.scan_select import SCAN_SELECT_BUCKETS
from lsa.core.kernels.triton.shape import add_planes_spread, scan_paged_shaped
from lsa.core.kernels.triton.top_k import TopK
from lsa.sglang import context_buckets
from lsa.sglang.index import probe
from lsa.sglang.index.bucket_cache import BucketCache
from lsa.sglang.index.graph_descend import LevelPlan, descend, plan_levels, uncovered_width
from lsa.sglang.index.hierarchy import IndexKeys
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings


class Run(NamedTuple):
    """The queries that sit in one page's run of a prefill's keys, and the buckets the cache has closed by its end."""

    # Rows of the segment's queries, [first, stop)
    first: int
    stop: int
    # Buckets of closed cluster blocks
    visible: int


@dataclass
class Scratch:
    """The prefill selection's buffers, which every layer of a device reads through in turn."""

    # Buckets a row picks, and rows a call picks for
    top_p: int
    prefill_chunk: int
    device: torch.device | str
    # The language of a kernel both languages have
    kernels: str = "triton"
    prefill_top_k: TopK = field(init=False)
    tree_top_k: TopK = field(init=False)
    scan_select: ScanSelect = field(init=False)
    beam_top_ks: dict[int, TopK] = field(default_factory=dict)
    beam_selects: dict[int, ScanSelect] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build the picks over `top_p` buckets."""
        self.prefill_top_k = TopK(self.top_p)
        # A tree's last level is listed exactly, which is faster for it than a superset
        self.tree_top_k = TopK(self.top_p, exact=True)
        self.scan_select = ScanSelect(self.prefill_chunk, self.top_p, self.device, self.kernels)

    def beam_top_k(self, beam_width: int) -> TopK:
        """Return the top-k over a level's scores that keeps `beam_width` of them."""
        if beam_width not in self.beam_top_ks:
            self.beam_top_ks[beam_width] = TopK(beam_width)
        return self.beam_top_ks[beam_width]

    def beam_select(self, beam_width: int) -> ScanSelect:
        """Return the fused select that keeps `beam_width` of a level's nodes."""
        if beam_width not in self.beam_selects:
            self.beam_selects[beam_width] = ScanSelect(
                self.prefill_chunk, beam_width, self.device, self.kernels
            )
        return self.beam_selects[beam_width]


# Layers prefill one after another on a stream, so one scratch serves them all
SCRATCHES: dict[tuple[int, int, str, str], Scratch] = {}


def scratch_for(
    top_p: int, prefill_chunk: int, device: torch.device | str, kernels: str = "triton"
) -> Scratch:
    """Return the scratch every layer on `device` shares, built the first time it is asked for."""
    key = (top_p, prefill_chunk, str(device), kernels)
    if key not in SCRATCHES:
        SCRATCHES[key] = Scratch(top_p, prefill_chunk, device, kernels)
    return SCRATCHES[key]


def clear_scratches() -> None:
    """Drop every scratch, so the next layer built makes its own."""
    SCRATCHES.clear()


class Selector:
    """Selection for one layer - a bucket cache and the reads over it."""

    def __init__(
        self,
        settings: DesignSettings,
        kv_size: int,
        request_count: int,
        max_context: int,
        prefill_chunk: int,
        device: torch.device | str,
        slot_table: Callable[[], torch.Tensor],
        reuse: bool = False,
    ) -> None:
        """Size the cache and the selection buffers.

        Args:
            settings: The design.
            kv_size: Keys the KV pool holds.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
            prefill_chunk: Most query rows a prefill call is handed.
            device: Where everything lives.
            slot_table: Returns the request table, each request's positions to KV slots [requests, context] int32.
            reuse: Share the index of a cached prefix between the requests that hold it.
        """
        self.settings = settings
        self.config = settings.indexer
        self.device = device
        self.slot_table = slot_table
        self.prefill_chunk = prefill_chunk
        self.width = settings.width
        self.step = 0
        self.cache = BucketCache(settings, kv_size, request_count, max_context, device, reuse)
        self.max_rows = self.cache.max_pages * self.cache.rows_per_page
        top_p = self.config.top_p
        self.scratch = scratch_for(top_p, prefill_chunk, device, settings.serving.kernels)
        self.prefill_top_k = self.scratch.prefill_top_k
        self.tree_top_k = self.scratch.tree_top_k
        self.scan_select = self.scratch.scan_select
        # A check reads the listing a captured decode step wrote, so it lives in buffers sized once. A decode step's
        # few rows are listed exactly, which is faster for them than a superset
        self.decode_top_k = TopK(top_p, request_count, device, exact=True)
        self.decode_select = cutedsl_decode.DecodeSelect(top_p)
        # The batched descend's widths, built when it first runs
        self.level_plans: list[LevelPlan] | None = None

    def beam_top_k(self, beam_width: int) -> TopK:
        """Return the top-k over a level's scores that keeps `beam_width` of them."""
        return self.scratch.beam_top_k(beam_width)

    def beam_select(self, beam_width: int) -> ScanSelect:
        """Return the fused select that keeps `beam_width` of a level's nodes."""
        return self.scratch.beam_select(beam_width)

    def maintain(
        self,
        rids: Sequence[object],
        rows: Sequence[int],
        ends: Sequence[int],
        index_keys_of: Callable[[int], IndexKeys],
    ) -> int:
        """Bring each request's cache up to the keys already in the KV pool - eagerly, before every forward.

        A request just admitted with a cached prefix first takes that prefix's index where another request built it,
        and every full page whose keys are all in the pool is fingerprinted for the requests after it - under
        `reuse` alone.

        Args:
            rids: The batch's request ids.
            rows: Their request table rows.
            ends: Keys each request has in the pool before this forward's own.
            index_keys_of: A request table row to the lookup of its positions' index keys.

        Returns:
            The pages the batch's requests took from cached prefixes.
        """
        self.step += 1
        active = set(rows)
        taken = 0
        for rid, row, end in zip(rids, rows, ends, strict=True):
            self.cache.ensure(row, rid, self.step)
            index_keys = index_keys_of(row)
            taken += self.cache.adopt(row, end, index_keys)
            self.cache.advance(row, end, index_keys, active)
            self.cache.fingerprint(row, end, index_keys)
        return taken

    def prefill(
        self,
        row: int,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        index_keys_of: IndexKeys,
        active: set[int],
        new_keys: int | None = None,
        ragged_start: int | None = None,
        query_start: int | None = None,
    ) -> torch.Tensor:
        """Read a prefill segment of one request - append its keys, select for its queries, grow at page ends.

        The segment's keys must already be where `index_keys_of` finds them. The queries are the segment's own
        positions, unless `new_keys` says the cache grows by more than they cover: under prefill context parallel
        a rank queries a span of the chunk and indexes all of it.

        The keys go into the cache in one call, and each query reads the buckets its own position sees, which is
        what it saw when the cache grew a page at a time. Queries read together until the hierarchy next grows,
        since a query reads the hierarchy as it stood in its page. The one thing read off the device is whether any
        row's listing may be missing a top-p bucket.

        Args:
            row: The request table row.
            index_queries: The segment's index queries [S, index_dim] fp32, unit rows.
            positions: Their positions [S] int32, contiguous and inside the keys' positions.
            index_keys_of: The request's positions [M] int64 to their index keys [M, index_dim] bf16.
            active: The rows of the running batch.
            new_keys: Keys the cache appends, from its end - the number of queries when None.
            ragged_start: Where the request starts in the rows of [prefix, chunk] keys that decode context
                parallel's prefill reads - when given, the lists hold those rows instead of KV slots.
            query_start: Position of the first query, from the scheduler - read off `positions` when None.

        Returns:
            KV slots, or ragged rows, [S, width] int32, -1 padded.
        """
        cache = self.cache
        block = cache.config.cluster_block_size
        query_count = len(positions)
        key_count = query_count if new_keys is None else new_keys
        if query_start is None:
            query_start = int(positions[0]) if query_count else 0
        ragged = ragged_start is not None
        end, closed_before = cache.end[row], cache.closed_keys[row]
        cache.append(row, end + key_count, index_keys_of, active)
        lists: list[torch.Tensor] = []
        pending: list[Run] = []
        done = 0
        while done < key_count:
            take = min(key_count - done, PAGE_SIZE - end % PAGE_SIZE)
            # The queries whose positions fall in this run of keys
            first = max(end, query_start) - query_start
            stop = min(end + take, query_start + query_count) - query_start
            end += take
            if first < stop:
                closed = max(closed_before, end // block * block)
                pending.append(Run(first, stop, closed // cache.config.bucket_size))
            if cache.grows_at(row, end):
                lists.extend(self.read_runs(row, index_queries, positions, pending, ragged))
                pending = []
                cache.settle(row, index_keys_of, end)
            done += take
        lists.extend(self.read_runs(row, index_queries, positions, pending, ragged))
        read = torch.cat(lists) if lists else positions.new_empty((0, self.width))
        if ragged_start is None:
            return read
        return torch.where(read >= 0, read + ragged_start, -1).to(torch.int32)

    def read_runs(
        self,
        row: int,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        runs: list[Run],
        ragged: bool,
    ) -> list[torch.Tensor]:
        """Read the queries of consecutive page runs over the hierarchy as it stands - together where they read alike.

        Runs read together when they route the same way - flat below the fused select's gate, flat from it, or by the
        beam - and hold no more rows than a page, which bounds what a listing takes.

        Args:
            row: The request table row.
            index_queries: The segment's index queries [S, index_dim] fp32.
            positions: Their positions [S] int32.
            runs: Runs of queries read from the same hierarchy, in order.
            ragged: Read positions rather than KV slots.

        Returns:
            The lists of each group of runs [rows, width] int32, in order.
        """
        cache = self.cache
        hierarchy = None if cache.hierarchies is None else cache.hierarchies[row]
        beam = hierarchy is not None and hierarchy.levels > 0
        groups: list[list[Run]] = []
        for run in runs:
            if groups and self.reads_together(groups[-1], run, beam):
                groups[-1].append(run)
            else:
                groups.append([run])
        lists = []
        for group in groups:
            first, stop = group[0].first, group[-1].stop
            segment = slice(first, stop)
            spans = [(run.first - first, run.stop - first) for run in group]
            lists.append(
                self.read_segment(
                    row, index_queries[segment], positions[segment], ragged, group[-1].visible, spans
                )
            )
        return lists

    def reads_together(self, group: list[Run], run: Run, beam: bool) -> bool:
        """Tell whether a run reads in the same call as the runs before it.

        Args:
            group: The runs already together.
            run: The next run.
            beam: Whether the queries read by the beam.
        """
        if run.stop - group[0].first > PAGE_SIZE:
            return False
        if beam:
            return True
        gathered = self.cache.gathered_buckets
        return (gathered(group[-1].visible) >= SCAN_SELECT_BUCKETS) == (
            gathered(run.visible) >= SCAN_SELECT_BUCKETS
        )

    def read_segment(
        self,
        row: int,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        ragged: bool = False,
        visible: int | None = None,
        spans: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Read queries of one request over the cache as it stands - by the beam once a level exists, else flat.

        The positions need not be contiguous, but each must sit in the page the cache was last advanced into,
        before that page settles - which is where a prefill reads them.

        Args:
            row: The request table row.
            index_queries: The queries' index queries [S, index_dim] fp32, unit rows.
            positions: Their positions [S] int32.
            ragged: Read positions rather than KV slots.
            visible: Buckets of closed cluster blocks to read the cache over, at least every query's visible count - the
                cache's own when None.
            spans: Row ranges [start, stop) of the queries that sat in different page runs - one span of all when None.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        cache = self.cache
        visible_counts = cache.query_lengths(positions)
        request_rows = torch.full((len(positions),), row, device=positions.device, dtype=torch.int64)
        hierarchy = None if cache.hierarchies is None else cache.hierarchies[row]
        if hierarchy is None or not hierarchy.levels:
            return self.read_flat(
                row, index_queries, positions, visible_counts, request_rows, ragged, visible, spans
            )
        bucket_keys = hierarchy.beam_search(
            cache,
            row,
            index_queries,
            visible_counts,
            self.beam_top_k,
            self.tree_top_k,
            self.beam_select,
            visible,
        )
        return self.expand_bucket_keys(bucket_keys, positions, request_rows, ragged)

    def read_flat(
        self,
        row: int,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        visible_counts: torch.Tensor,
        request_rows: torch.Tensor,
        ragged: bool = False,
        visible: int | None = None,
        spans: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Read a prefill segment by scoring every visible bucket - dense below the gate, fused from it.

        Args:
            row: The request table row.
            index_queries: The segment's index queries [S, index_dim] fp32.
            positions: Their positions [S] int32.
            visible_counts: Buckets each query sees [S] int32.
            request_rows: The request table row, once a query [S] int64.
            ragged: Read positions rather than KV slots.
            visible: Buckets of closed cluster blocks to score - the cache's own when None.
            spans: Row ranges [start, stop) of the queries that sat in different page runs - one span of all when None.
                A span whose listing may miss a top-p bucket is picked on its own, as it was when it was read alone.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        cache = self.cache
        table = None if ragged else self.slot_table()
        visible = cache.visible(row) if visible is None else visible
        with probe.region("index q scan"):
            bucket_centroids = cache.gather(row, visible)
            shapes = cache.gather_shapes(row, visible)
            index_queries = index_queries.to(torch.bfloat16)
            fused = bucket_centroids.shape[0] >= SCAN_SELECT_BUCKETS
            if fused:
                listing = self.scan_select.listed(index_queries, bucket_centroids, visible_counts, shapes)
            else:
                # fp32 out of the GEMM itself - a bf16 product cast after costs five times as much
                bucket_scores = torch.mm(index_queries, bucket_centroids.t(), out_dtype=torch.float32)
                if shapes is not None:
                    add_planes_spread(bucket_scores, index_queries, shapes)
        # Every row's best picks listed - the read is expanded from the listing by the key its walk stops at
        if not fused:
            with probe.region("index k select"):
                keys, listed = self.short_listing(bucket_scores, visible_counts)
            read = self.expand_listing(keys, listed, positions, request_rows, table)
            self.prefill_top_k.release()
            return read
        spans = spans or [(0, len(positions))]
        missed = self.scan_select.missed_runs(listing, spans)
        if not any(missed):
            return self.expand_listing(listing.keys, listing.survivor_counts, positions, request_rows, table)
        return self.read_missed(listing, spans, missed, positions, request_rows, ragged, table)

    def short_listing(
        self, bucket_scores: torch.Tensor, visible_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List each query's best top-p of a short prefill's scores, with room beside them, in the kernels' language.

        Args:
            bucket_scores: Every visible bucket's score [S, N] fp32.
            visible_counts: Buckets each query sees [S] int32.

        Returns:
            Packed keys [S, W] int64.
            Keys each query listed [S] int32.
        """
        if dispatch.runs_cutedsl(
            "short-context listing", self.scratch.kernels, bucket_scores.device
        ) and cutedsl_top_k.supports(bucket_scores, self.config.top_p):
            return cutedsl_top_k.list_rows(bucket_scores, visible_counts, self.config.top_p)
        return self.prefill_top_k.listed(bucket_scores, visible_counts)

    def read_missed(
        self,
        listing: Listing,
        spans: list[tuple[int, int]],
        missed: list[bool],
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        ragged: bool,
        table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Expand a listing some spans of which may miss a top-p bucket - those picked in full, the rest from the listing.

        Args:
            listing: The rows' listing.
            spans: Row ranges [start, stop) read as one.
            missed: For each span, whether its listing may miss a bucket.
            positions: Each query's position [S] int32.
            request_rows: Each query's request table row [S] int64.
            ragged: Read positions rather than KV slots.
            table: The request table, or None to read positions.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        device = positions.device
        out = torch.empty(len(positions), self.width, device=device, dtype=torch.int32)
        for picked in (False, True):
            ranges = [
                torch.arange(start, stop, device=device)
                for (start, stop), span_missed in zip(spans, missed, strict=True)
                if span_missed == picked
            ]
            if not ranges:
                continue
            rows = torch.cat(ranges)
            part = listing.take(rows)
            if picked:
                with probe.region("index k select"):
                    logical = self.scan_select.pick(part)
                out[rows] = self.expand(
                    self.cache.physical(request_rows[rows], logical),
                    positions[rows],
                    request_rows[rows],
                    ragged,
                )
            else:
                out[rows] = self.expand_listing(
                    part.keys, part.survivor_counts, positions[rows], request_rows[rows], table
                )
        return out

    def decode(
        self, index_queries: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        """Read one decode step of a batch.

        Args:
            index_queries: One index query a request [B, index_dim] fp32, unit rows.
            positions: Each request's position [B] int32.
            rows: The request table rows [B] int64.

        Returns:
            KV slots [B, width] int32, -1 padded.
        """
        if self.settings.serving.decode == "descend":
            return self.decode_descend(index_queries, positions, rows)
        return self.decode_flat(index_queries, positions, rows)

    def decode_descend(
        self,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
        batched: bool = False,
    ) -> torch.Tensor:
        """Read each request's beam down its own hierarchy, as a prefill query does.

        Eagerly, a request at a time, except while a CUDA graph is captured - or when `batched` asks - when every
        request reads the graph tree's mirror in one batched descend at fixed shapes. A request too short for a node
        bucket scans its buckets as a flat decode does eagerly, and picks among them in the batched descend.

        Args:
            index_queries: One index query a request [B, index_dim] fp32, unit rows.
            positions: Each request's position [B] int32.
            rows: The request table rows [B] int64.
            batched: Read through the batched descend outside a capture too - for a check that holds it to the eager read.

        Returns:
            KV slots [B, width] int32, -1 padded.

        Raises:
            RuntimeError: Called while a CUDA graph is being captured, or with `batched`, by a design that builds no graph
                tree - its node buckets are shaped.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        if (capturing or batched) and self.cache.graph_tree is None:
            if capturing:
                raise RuntimeError(
                    "a descending decode over shaped node buckets runs eagerly - serve with --disable-cuda-graph"
                )
            raise RuntimeError("the batched descend reads the graph tree, which this design does not build")
        if capturing or batched:
            return self.decode_batched(index_queries, positions, rows)
        cache = self.cache
        lists = []
        for i, row in enumerate(rows.tolist()):
            one = slice(i, i + 1)
            hierarchy = None if cache.hierarchies is None else cache.hierarchies[row]
            if hierarchy is None or not hierarchy.levels:
                lists.append(self.decode_flat(index_queries[one], positions[one], rows[one]))
                continue
            bucket_keys = hierarchy.beam_search(
                cache,
                row,
                index_queries[one],
                cache.query_lengths(positions[one]),
                self.beam_top_k,
                self.tree_top_k,
                self.beam_select,
            )
            lists.append(self.expand_bucket_keys(bucket_keys, positions[one], rows[one]))
        return torch.cat(lists)

    def decode_batched(
        self, index_queries: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        """Read a descending decode step for every request at once - fixed shapes, for the CUDA graph.

        Every request reads the graph tree's mirror down every level. A request with no node buckets scores none at any
        level, and its level 0 pick holds all of its buckets - which it sees fewer of than the pick has columns, since
        none is covered.

        Args:
            index_queries: One index query a request [B, index_dim] fp32, unit rows.
            positions: Each request's position [B] int32.
            rows: The request table rows [B] int64.

        Returns:
            KV slots [B, width] int32, -1 padded.
        """
        cache, tree = self.cache, self.cache.graph_tree
        if tree is None or not tree.level_count:
            return self.decode_flat(index_queries, positions, rows)
        if self.level_plans is None:
            self.level_plans = plan_levels(self.settings, tree)
        bucket_keys, listed = descend(
            index_queries,
            positions,
            rows,
            tree,
            cache,
            self.level_plans,
            uncovered_width(self.settings, cache.rows_per_page),
            self.scratch.kernels,
            self.beam_top_k,
            self.tree_top_k,
        )
        return self.expand_bucket_keys(bucket_keys, positions, rows, listed=listed)

    def decode_flat(
        self,
        index_queries: torch.Tensor,
        positions: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Read a decode step by scanning every visible bucket of each request - fixed shapes, for the CUDA graph.

        Args:
            index_queries: One index query a request [B, index_dim] fp32, unit rows.
            positions: Each request's position [B] int32.
            rows: The request table rows [B] int64.

        Returns:
            KV slots [B, width] int32, -1 padded.
        """
        cache = self.cache
        counts = cache.decode_counts(rows)
        max_rows = self.max_rows
        limit = context_buckets.captured_limit()
        if limit is not None and context_buckets.pages_of(limit) < cache.max_pages:
            # A graph for contexts up to `limit` - no request of it has a bucket past its pages
            # => a padded row of the graph may name a longer one, so its count is cut to the columns
            max_rows = context_buckets.pages_of(limit) * cache.rows_per_page
            counts = counts.clamp(max=max_rows)
        pages = cache.request_pages[rows, : max_rows // cache.rows_per_page]
        index_queries = index_queries.to(torch.bfloat16)
        # The CuTeDSL select lists while it scans, so no pass over the score matrix follows
        if (
            cache.bucket_shapes is None
            and dispatch.runs_cutedsl("decode select", self.scratch.kernels, index_queries.device)
            and cutedsl_decode.supports(index_queries, cache.bucket_centroids, max_rows)
        ):
            keys, listed = self.decode_select.listed(
                index_queries, cache.bucket_centroids, pages, counts, cache.rows_per_page, max_rows
            )
            return self.expand_listing(keys, listed, positions, rows, self.slot_table())
        if cache.bucket_shapes is None:
            bucket_scores = scan_paged(
                index_queries,
                cache.bucket_centroids,
                pages,
                counts,
                cache.rows_per_page,
                max_rows,
            )
        else:
            bucket_scores = scan_paged_shaped(
                index_queries,
                cache.bucket_centroids,
                cache.bucket_shapes,
                pages,
                counts,
                cache.rows_per_page,
                max_rows,
            )
        keys, listed = self.decode_top_k.listed(bucket_scores, counts)
        return self.expand_listing(keys, listed, positions, rows, self.slot_table())

    def expand_listing(
        self,
        keys: torch.Tensor,
        listed: torch.Tensor,
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Read the buckets a listing holds, by the key a best-first walk of its top-p stops at.

        Args:
            keys: Listed scores packed with their logical buckets [S, W] int64, in no order, holding every row's top-p.
            listed: Keys each row lists [S] int32.
            positions: Each query's position [S] int32.
            request_rows: Each query's request table row [S] int64.
            table: The request table, or None to read positions.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        cache, config = self.cache, self.config
        return expand_by_threshold(
            keys,
            listed,
            cache.bucket_members,
            cache.bucket_sizes,
            cache.request_pages,
            request_rows,
            positions,
            cache.rows_per_page,
            PAGE_SIZE,
            config.top_p,
            config.window_size,
            config.sink_size,
            config.top_k,
            table,
            kernels=self.scratch.kernels,
            early=config.early_budget,
        )

    def expand(
        self,
        bucket_rows: torch.Tensor,
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        ragged: bool = False,
    ) -> torch.Tensor:
        """Read the picked buckets best first until the top-k is full, then the window and the sinks.

        Args:
            bucket_rows: Picked cache rows [S, top_p] int32, best first, -1 padded.
            positions: Each query's position [S] int32.
            request_rows: Each query's request table row [S] int64.
            ragged: Read positions rather than KV slots.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        config = self.config
        return expand_in_order(
            bucket_rows,
            self.cache.bucket_members,
            positions,
            config.window_size,
            config.sink_size,
            config.top_k,
            None if ragged else self.slot_table(),
            request_rows,
            kernels=self.scratch.kernels,
            early=config.early_budget,
        )[0]

    def expand_bucket_keys(
        self,
        bucket_keys: torch.Tensor,
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        ragged: bool = False,
        listed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read a beam's picks, scores packed with their cache rows, filled by threshold.

        Args:
            bucket_keys: Listed cache rows packed with their scores [S, W] int64, in no order, holding every query's top_p.
            positions: Each query's position [S] int32.
            request_rows: Each query's request table row [S] int64.
            ragged: Read positions rather than KV slots.
            listed: Keys each query lists [S] int32, or None for all W of them.

        Returns:
            KV slots, or positions, [S, width] int32, -1 padded.
        """
        cache, config = self.cache, self.config
        if listed is None:
            listed = torch.full(
                (bucket_keys.shape[0],), bucket_keys.shape[1], device=bucket_keys.device, dtype=torch.int32
            )
        return expand_by_threshold(
            bucket_keys,
            listed,
            cache.bucket_members,
            cache.bucket_sizes,
            cache.request_pages,
            request_rows,
            positions,
            cache.rows_per_page,
            PAGE_SIZE,
            config.top_p,
            config.window_size,
            config.sink_size,
            config.top_k,
            None if ragged else self.slot_table(),
            cache.page_positions,
            kernels=self.scratch.kernels,
            early=config.early_budget,
        )


def read_sampled(
    settings: DesignSettings, index_queries: torch.Tensor, index_keys: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Read some queries of one document as the prefill of the whole document reads them.

    The cache is built as a prefill builds it - each page's keys appended and its cluster blocks closed, the
    page's queries read, then the page settled, which is where a hierarchy grows - and only the queries asked
    for are read, so each reads the buckets and the hierarchy it would see inside the whole prefill.
    The request table is the identity, so a KV slot is a position and the lists hold key ids.

    A query may be several, one a tensor-parallel rank - the key and query input, where each rank sums its own
    heads - and each is read as its rank reads it, at the query's position.

    Args:
        settings: The design.
        index_queries: The queries' index queries [R, index_dim] fp32, unit rows, or [R, ranks, index_dim].
        index_keys: Every position's index key [T, index_dim] bf16, unit rows, where the read runs.
        positions: The queries' positions [R] int32, ascending, on the same device.

    Returns:
        Key ids [R, width] int32, -1 padded, or [R, ranks, width] for queries a rank.
    """
    ranks = index_queries.shape[1] if index_queries.ndim == 3 else 0
    if ranks:
        index_queries = index_queries.flatten(0, 1)
        positions = positions.repeat_interleave(ranks)
    device = index_keys.device
    key_count = index_keys.shape[0]
    table = torch.arange(key_count, device=device, dtype=torch.int32)[None]
    selector = Selector(settings, key_count, 1, key_count, PAGE_SIZE, device, lambda: table)
    cache = selector.cache

    def index_keys_of(key_positions: torch.Tensor) -> torch.Tensor:
        return index_keys[key_positions.long()]

    cache.ensure(0, "document", 0)
    lists = []
    for page_start in range(0, int(positions[-1]) + 1, PAGE_SIZE):
        cache.advance(0, min(page_start + PAGE_SIZE, key_count), index_keys_of, {0}, settle=False)
        inside = (positions >= page_start) & (positions < page_start + PAGE_SIZE)
        if bool(inside.any()):
            lists.append(selector.read_segment(0, index_queries[inside], positions[inside]))
        cache.settle(0, index_keys_of)
    read = torch.cat(lists)
    return read.view(-1, ranks, read.shape[-1]) if ranks else read
