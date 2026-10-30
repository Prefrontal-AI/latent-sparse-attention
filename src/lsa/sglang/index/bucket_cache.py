"""One layer's bucket cache - the pages each request holds, their bucket centroids, members and sizes.

A page holds `PAGE_SIZE` keys' buckets, one run of cache rows, so a request's buckets are its pages in order and the
decode kernels map them through the request's page table alone. A cluster block is clustered as soon as its last key
arrives, and the hierarchy grows over the buckets of a page once the page is full. The counters live on the host - the
device copy of the closed key count is what the CUDA graph needs.

Under `bucket_shape` each bucket also keeps its shape, built from its keys as its cluster block closes: a sign basis,
its variances and a floor - 28 bytes a bucket at rank 2 and an index dim of 64.

Under `reuse` - the server's prefix cache on - a request that arrives with a cached prefix takes the index of that
prefix instead of building it again. A full page's buckets are its own keys' alone, and never change once full, so a
page is shared between requests by a count of its holders. A request fingerprints each of its full pages from the
pool once its keys are all there (`KeyLookup.fingerprints`); when the scheduler gives its row to another request,
its leading fingerprinted pages and its hierarchy are kept as a `Retained` prefix rather than freed. A new request
fingerprints its own leading pages and takes the longest run that matches, from a retained prefix or a running
request: the pages by reference, the hierarchy as it stood at the run's end (`Hierarchy.truncated`). What follows
is built as before. A match compares the pool's bits, so a prefix whose slots were freed and written again by other
tokens matches nothing. Retained prefixes go first when pages run short.

Without `reuse` each request's pages are its own, so with the server's prefix cache on, requests that share a cached
prefix hold it once in the KV pool and once each here, and the pool's size no longer bounds the pages. The design's
`spare_contexts` adds that many full contexts of pages to what the pool and the request table call for.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.shape import ShapeTable, build_bucket_shapes
from lsa.sglang import memory
from lsa.sglang.index import probe
from lsa.sglang.index.clustering import cluster_blocks
from lsa.sglang.index.graph_tree import GraphTree
from lsa.sglang.index.hierarchy import Hierarchy, IndexKeys, read_span
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

# Cache rows of a bucket count are padded to this many
ROW_PADDING = 16


def fingerprints_of(index_keys: IndexKeys, first: int, count: int) -> list[int]:
    """Return the fingerprints of a request's pages [first, first + count), by the lookup's own read.

    Args:
        index_keys: The request's positions to their index keys.
        first: The first page.
        count: Pages, every one of whose keys is in the pool.

    Raises:
        TypeError: The lookup reads no fingerprints, which prefix reuse needs - `KeyLookup` reads them.
    """
    fingerprints = getattr(index_keys, "fingerprints", None)
    if fingerprints is None:
        raise TypeError("prefix reuse fingerprints pages from the pool, and this lookup reads none")
    return fingerprints(first, count)


@dataclass
class Retained:
    """A finished request's leading full pages, their fingerprints and its hierarchy, kept for a request that shares them."""

    pages: list[int]
    prints: list[int]
    hierarchy: Hierarchy | None
    seen: int


class BucketCache:
    """The buckets of every request of one layer, in pages of cache rows."""

    def __init__(
        self,
        settings: DesignSettings,
        kv_size: int,
        request_count: int,
        max_context: int,
        device: torch.device | str,
        reuse: bool = False,
    ) -> None:
        """Size the cache for a KV pool and a request table.

        Args:
            settings: The design.
            kv_size: Keys the KV pool holds - the cache holds a page of buckets for each `PAGE_SIZE` of them.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
            device: Where the cache lives.
            reuse: Share the index of a prefix between the requests that hold it - when the server caches prefixes.
        """
        self.settings = settings
        self.config = settings.indexer
        self.device = device
        self.reuse = reuse
        config = self.config
        self.rows_per_page = PAGE_SIZE // config.bucket_size
        self.max_pages = -(-max_context // PAGE_SIZE)
        pages = -(-kv_size // PAGE_SIZE) + request_count + settings.serving.spare_contexts * self.max_pages
        rows = pages * self.rows_per_page
        self.bucket_centroids = torch.zeros(rows, config.index_dim, device=device, dtype=torch.bfloat16)
        self.bucket_members = torch.full((rows, config.bucket_cap), -1, device=device, dtype=torch.int32)
        self.bucket_sizes = torch.zeros(rows, device=device, dtype=torch.int32)
        # Each page's first position in its request - where an expansion by threshold finds the window
        self.page_positions = torch.zeros(pages, device=device, dtype=torch.int32)
        self.request_pages = torch.full((request_count, self.max_pages), -1, device=device, dtype=torch.int32)
        self.closed_keys_device = torch.zeros(request_count, device=device, dtype=torch.int32)
        self.bucket_shapes: ShapeTable | None = None
        if config.bucket_shape is not None:
            rank = config.bucket_shape.rank
            self.bucket_shapes = ShapeTable(
                torch.zeros(rows, rank, config.index_dim // 8, device=device, dtype=torch.uint8),
                torch.zeros(rows, rank, device=device),
                torch.zeros(rows, device=device),
                config.bucket_shape.deviations,
            )
        # Popped from the end - a lone request's pages come out in order, consecutive rows
        self.free = list(range(pages - 1, -1, -1))
        # Requests and retained prefixes holding each page - a page is free again when none does
        self.holders = [0] * pages
        self.pages: list[list[int]] = [[] for _ in range(request_count)]
        # Fingerprints of each request's leading full pages, as far as their keys have all reached the pool
        self.prints: list[list[int]] = [[] for _ in range(request_count)]
        self.retained: list[Retained] = []
        # Each request's leading pages that are consecutive in the cache, so its buckets are one run of rows
        self.run = [0] * request_count
        # Keys each request has appended, and of those the keys in closed cluster blocks
        self.end = [0] * request_count
        self.closed_keys = [0] * request_count
        self.rid: list[object] = [None] * request_count
        self.seen = [0] * request_count
        self.hierarchies: list[Hierarchy] | None = None
        if config.top_level >= 1:
            self.hierarchies = [self.new_hierarchy() for _ in range(request_count)]
        # Descending decode reads the node buckets from a mirror at fixed addresses, which a CUDA graph can follow
        # Shaped node buckets are not mirrored => such a design descends eagerly only
        self.graph_tree: GraphTree | None = None
        if GraphTree.builds(settings):
            self.graph_tree = GraphTree(settings, rows, request_count, max_context, device)

    def new_hierarchy(self) -> Hierarchy:
        """Return an empty hierarchy under the design's settings."""
        return Hierarchy(
            self.config, self.settings.serving.node_centroid_precision, self.settings.serving.kernels
        )

    def ensure(self, row: int, rid: object, step: int) -> None:
        """Take a request table row for a request, releasing what its last request held.

        Args:
            row: The request table row.
            rid: The request occupying it.
            step: The indexer's step count, which orders requests for reclaiming.
        """
        if self.rid[row] != rid:
            self.release(row)
            self.rid[row] = rid
        self.seen[row] = step

    def held(self, row: int) -> dict[str, int]:
        """Return the bytes a request's index holds in this layer, by the structure that holds them.

        Args:
            row: The request table row.

        Returns:
            Bytes of each structure - the buckets of the row's pages and every node bucket of its hierarchy.
        """
        pages = len(self.pages[row])
        rows = pages * self.rows_per_page

        def row_bytes(tensor: torch.Tensor) -> int:
            return rows * tensor[0].numel() * tensor.element_size()

        held = {
            "bucket_centroids": row_bytes(self.bucket_centroids),
            "bucket_members": row_bytes(self.bucket_members) + row_bytes(self.bucket_sizes),
            "page_positions": pages * self.page_positions.element_size(),
        }
        if self.bucket_shapes is not None:
            shapes = self.bucket_shapes
            held["bucket_shapes"] = (
                row_bytes(shapes.basis) + row_bytes(shapes.variances) + row_bytes(shapes.floor)
            )
        if self.hierarchies is not None:
            # Node centroids are named for their node already, the rest of a node bucket is not
            nodes = self.hierarchies[row].held()
            held.update(
                {name if name.startswith("node_") else f"node_{name}": count for name, count in nodes.items()}
            )
        if self.graph_tree is not None:
            held.update(self.graph_tree.held(row))
        return held

    def release(self, row: int, retain: bool = True) -> None:
        """Let go of every page a row holds and forget its hierarchy - under `reuse`, keeping its prefix retained.

        A retained prefix is the request's fingerprinted pages and its hierarchy as it stood at their end. One that
        a retained prefix already covers is not kept again, and one that covers retained prefixes replaces them, so
        requests over one long prefix keep one hierarchy of it between them. At most a request table's worth are
        kept, the oldest dropped first.

        Args:
            row: The request table row.
            retain: Keep the prefix - False when the pages are wanted back.
        """
        if self.end[row]:
            memory.report_held(self.end[row], self.held(row))
        prints = self.prints[row] if retain and self.reuse else []
        covering = [entry for entry in self.retained if prints and entry.prints[: len(prints)] == prints]
        for entry in covering:
            entry.seen = max(entry.seen, self.seen[row])
        kept = 0 if covering else len(prints)
        if kept:
            for entry in [entry for entry in self.retained if prints[: len(entry.prints)] == entry.prints]:
                self.retained.remove(entry)
                self.let_go(entry.pages)
            hierarchy = None
            if self.hierarchies is not None:
                hierarchy = self.hierarchies[row].trimmed(kept * self.rows_per_page)
            self.retained.append(Retained(self.pages[row][:kept], prints, hierarchy, self.seen[row]))
            if len(self.retained) > len(self.pages):
                self.drop_oldest()
        self.let_go(self.pages[row][kept:])
        self.prints[row] = []
        self.pages[row] = []
        self.run[row] = 0
        self.request_pages[row].fill_(-1)
        self.end[row] = self.closed_keys[row] = 0
        self.closed_keys_device[row].fill_(0)
        self.rid[row] = None
        if self.hierarchies is not None:
            self.hierarchies[row] = self.new_hierarchy()
        if self.graph_tree is not None:
            self.graph_tree.reset(row)

    def let_go(self, pages: list[int]) -> None:
        """Drop one holder of each page, freeing those no one holds any longer.

        Args:
            pages: Pages, -1 for none.
        """
        freed = []
        for page in pages:
            if page < 0:
                continue
            self.holders[page] -= 1
            if not self.holders[page]:
                freed.append(page)
        self.free.extend(reversed(freed))

    def drop_oldest(self) -> None:
        """Forget the retained prefix seen longest ago."""
        oldest = min(range(len(self.retained)), key=lambda index: self.retained[index].seen)
        self.let_go(self.retained.pop(oldest).pages)

    def adopt(self, row: int, end: int, index_keys_of: IndexKeys) -> int:
        """Take the longest run of leading pages another request indexed from the same keys, under `reuse`.

        Only a request that holds nothing yet takes one - one just admitted with a cached prefix of a page or more.
        The candidates are the retained prefixes and the requests holding fingerprinted pages, and a page matches
        when its fingerprint does, every page before it having matched.

        Args:
            row: The request table row.
            end: Keys the request has in the pool.
            index_keys_of: The request's positions to their index keys, which fingerprints its pages.

        Returns:
            The pages taken.
        """
        if not self.reuse or self.end[row] or end < PAGE_SIZE:
            return 0
        donors = [(entry.pages, entry.prints, entry.hierarchy) for entry in self.retained] + [
            (
                self.pages[other][: len(self.prints[other])],
                self.prints[other],
                None if self.hierarchies is None else self.hierarchies[other],
            )
            for other in range(len(self.pages))
            if other != row and self.prints[other]
        ]
        if not donors:
            return 0
        prints = fingerprints_of(index_keys_of, 0, 1)
        donors = [donor for donor in donors if donor[1][0] == prints[0]]
        if not donors:
            return 0
        longest = min(end // PAGE_SIZE, max(len(donor[1]) for donor in donors))
        prints += fingerprints_of(index_keys_of, 1, longest - 1)

        def matched(donor_prints: list[int]) -> int:
            count = 0
            while count < min(len(donor_prints), longest) and donor_prints[count] == prints[count]:
                count += 1
            return count

        pages, donor_prints, hierarchy = max(donors, key=lambda donor: matched(donor[1]))
        count = matched(donor_prints)
        pages = pages[:count]
        for page in pages:
            self.holders[page] += 1
        self.pages[row] = list(pages)
        self.prints[row] = prints[:count]
        self.request_pages[row, :count] = torch.tensor(pages, dtype=torch.int32, device=self.device)
        self.run[row] = 1
        while self.run[row] < count and pages[self.run[row]] == pages[self.run[row] - 1] + 1:
            self.run[row] += 1
        self.end[row] = self.closed_keys[row] = count * PAGE_SIZE
        self.closed_keys_device[row].fill_(count * PAGE_SIZE)
        if self.hierarchies is not None and hierarchy is not None:
            self.hierarchies[row] = hierarchy.truncated(count * self.rows_per_page, self, index_keys_of)
            if self.graph_tree is not None:
                self.graph_tree.reset(row)
                self.graph_tree.sync(row, self.hierarchies[row])
        return count

    def fingerprint(self, row: int, end: int, index_keys_of: IndexKeys) -> None:
        """Fingerprint the request's full pages whose keys have all reached the pool, under `reuse`.

        Args:
            row: The request table row.
            end: Keys the request has in the pool.
            index_keys_of: The request's positions to their index keys, which fingerprints its pages.
        """
        if not self.reuse:
            return
        full = min(self.closed_keys[row], end) // PAGE_SIZE
        taken = len(self.prints[row])
        if full > taken:
            self.prints[row] += fingerprints_of(index_keys_of, taken, full - taken)

    def allocate(self, row: int, page_index: int, active: set[int]) -> None:
        """Give a request its next page, reclaiming an idle request's pages when none is free.

        Args:
            row: The request table row.
            page_index: Which page of the request this is.
            active: The rows of the running batch, which are never reclaimed.
        """
        if not self.free:
            self.reclaim(active)
        page = self.free.pop()
        self.holders[page] = 1
        pages = self.pages[row]
        if self.run[row] == page_index == len(pages) and (not pages or pages[-1] == page - 1):
            self.run[row] += 1
        while len(pages) <= page_index:
            pages.append(-1)
        pages[page_index] = page
        # Filled by a kernel - an item assignment is a host copy, which waits for the stream
        self.request_pages[row, page_index].fill_(page)
        self.page_positions[page].fill_(page_index * PAGE_SIZE)

    def reclaim(self, active: set[int]) -> None:
        """Drop retained prefixes, then release the oldest idle requests, until a page is free.

        Finished requests keep their pages until the scheduler reuses their row.

        Args:
            active: The rows of the running batch.

        Raises:
            RuntimeError: Every page belongs to a running request - raise `model.serving.spare_contexts`.
        """
        while self.retained:
            self.drop_oldest()
            if self.free:
                return
        idle = [row for row in range(len(self.pages)) if self.pages[row] and row not in active]
        for row in sorted(idle, key=lambda row: self.seen[row]):
            self.release(row, retain=False)
            if self.free:
                return
        raise RuntimeError(
            "the bucket cache is exhausted - every page belongs to a running request; "
            "requests sharing a cached prefix index it each, so raise model.serving.spare_contexts"
        )

    def visible(self, row: int) -> int:
        """Return the buckets of a request's closed cluster blocks."""
        return self.closed_keys[row] // self.config.bucket_size

    def query_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the buckets each query sees - every bucket of the cluster blocks before its own.

        Args:
            positions: Each query's position [S] int32.

        Returns:
            Visible buckets [S] int32.
        """
        block = self.config.cluster_block_size
        return (positions // block * (block // self.config.bucket_size)).to(torch.int32)

    def decode_counts(self, rows: torch.Tensor) -> torch.Tensor:
        """Return the buckets each decoding request sees [B] int32, from the counts the graph reads.

        Args:
            rows: The request table rows [B].
        """
        return self.closed_keys_device[rows] // self.config.bucket_size

    def logical_rows(self, row: int, count: int, first: int = 0) -> torch.Tensor:
        """Return the cache rows of a request's buckets from `first` on, padded to a multiple of 16.

        Args:
            row: The request table row.
            count: Buckets wanted.
            first: The first logical bucket.

        Returns:
            Cache rows [max(16, count rounded up to 16)] int64.
        """
        padded = self.gathered_buckets(count)
        logical = first + torch.arange(padded, device=self.device)
        pages = self.request_pages[row, (logical // self.rows_per_page).clamp_max(self.max_pages - 1)]
        return pages.clamp_min(0).long() * self.rows_per_page + logical % self.rows_per_page

    def run_rows(self, row: int, count: int, first: int = 0) -> slice | None:
        """Return the same rows as `logical_rows` as one slice, when the buckets lie in the request's run.

        Args:
            row: The request table row.
            count: Buckets wanted.
            first: The first logical bucket.

        Returns:
            The slice, or None when the buckets leave the run of consecutive pages.
        """
        padded = self.gathered_buckets(count)
        if not self.run[row] or (first + max(count, 1) - 1) // self.rows_per_page >= self.run[row]:
            return None
        start = self.pages[row][0] * self.rows_per_page + first
        if start + padded > self.bucket_centroids.shape[0]:
            return None
        return slice(start, start + padded)

    def gather(self, row: int, count: int, first: int = 0) -> torch.Tensor:
        """Return the centroids of a request's buckets from `first` on - a view where they are one run of rows.

        Args:
            row: The request table row.
            count: Buckets wanted.
            first: The first logical bucket.

        Returns:
            Centroids [max(16, count rounded up to 16), index_dim] bf16.
        """
        run = self.run_rows(row, count, first)
        if run is not None:
            return self.bucket_centroids[run]
        return self.bucket_centroids[self.logical_rows(row, count, first)]

    def gather_shapes(self, row: int, count: int, first: int = 0) -> ShapeTable | None:
        """Return the shapes of the buckets `gather` returns, row for row, or None when buckets are not shaped.

        Args:
            row: The request table row.
            count: Buckets wanted.
            first: The first logical bucket.
        """
        if self.bucket_shapes is None:
            return None
        run = self.run_rows(row, count, first)
        return self.bucket_shapes.rows(run if run is not None else self.logical_rows(row, count, first))

    def physical(self, rows: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
        """Map each request's logical bucket ids to cache rows, -1 kept.

        Args:
            rows: The request table rows [B].
            logical: Logical bucket ids [B, k] int32, -1 padded.

        Returns:
            Cache rows [B, k] int32, -1 where an id is -1 or its page is not held.
        """
        pages = self.request_pages[
            rows[:, None], (logical.clamp_min(0) // self.rows_per_page).clamp_max(self.max_pages - 1)
        ]
        physical = pages * self.rows_per_page + logical.clamp_min(0) % self.rows_per_page
        return torch.where((logical >= 0) & (pages >= 0), physical, -1)

    def advance(
        self, row: int, new_end: int, index_keys_of: IndexKeys, active: set[int], settle: bool = True
    ) -> None:
        """Append positions up to `new_end` - their index keys come from `index_keys_of`.

        Cluster blocks close every `window_size` positions and are clustered as they do. With `settle`, a full page
        grows the hierarchy - a caller that still has queries inside the page settles after selecting.

        Args:
            row: The request table row.
            new_end: The key count the request has after this call.
            index_keys_of: Positions [M] int64 to their index keys [M, index_dim] bf16.
            active: The rows of the running batch.
            settle: Grow the hierarchy as a page fills.
        """
        if not settle:
            self.append(row, new_end, index_keys_of, active)
            return
        while self.end[row] < new_end:
            page_end = (self.end[row] // PAGE_SIZE + 1) * PAGE_SIZE
            self.append(row, min(new_end, page_end), index_keys_of, active)
            self.settle(row, index_keys_of)

    def append(self, row: int, new_end: int, index_keys_of: IndexKeys, active: set[int]) -> None:
        """Append positions up to `new_end` across as many pages as they cover, clustering every closed block in one call.

        Args:
            row: The request table row.
            new_end: The key count the request has after this call.
            index_keys_of: Positions [M] int64 to their index keys [M, index_dim] bf16.
            active: The rows of the running batch.
        """
        while self.end[row] < new_end:
            end = self.end[row]
            if end % PAGE_SIZE == 0:
                self.allocate(row, end // PAGE_SIZE, active)
            self.end[row] = end + min(new_end - end, PAGE_SIZE - end % PAGE_SIZE)
        closed = self.end[row] // self.config.cluster_block_size * self.config.cluster_block_size
        if closed > self.closed_keys[row]:
            self.close_blocks(row, self.closed_keys[row], closed, index_keys_of)

    def gathered_buckets(self, count: int) -> int:
        """Return how many rows `gather` returns for `count` buckets - padded to a multiple of 16, 16 at least.

        Args:
            count: Buckets wanted.
        """
        return max(ROW_PADDING, -(-count // ROW_PADDING) * ROW_PADDING)

    def grows_at(self, row: int, end: int) -> bool:
        """Tell whether settling the request at `end` keys grows its hierarchy.

        Args:
            row: The request table row.
            end: A key count.
        """
        if end % PAGE_SIZE or self.hierarchies is None:
            return False
        return self.hierarchies[row].growth_blocks(end // self.config.bucket_size) > 0

    def settle(self, row: int, index_keys_of: IndexKeys, end: int | None = None) -> None:
        """Grow the hierarchy over the buckets of the pages that are full.

        Args:
            row: The request table row.
            index_keys_of: Positions [M] int64 to their index keys [M, index_dim] bf16.
            end: The key count to settle at - the request's own when None, for a caller that has appended past it.
        """
        end = self.end[row] if end is None else end
        if end % PAGE_SIZE or self.hierarchies is None:
            return
        self.hierarchies[row].grow(self, row, end // self.config.bucket_size, index_keys_of)
        if self.graph_tree is not None:
            self.graph_tree.sync(row, self.hierarchies[row])

    def close_blocks(self, row: int, start: int, stop: int, index_keys_of: IndexKeys) -> None:
        """Cluster the cluster blocks in [start, stop) into buckets, in one call.

        Shapes are built a page at a time, since their eigendecompositions and sums round by the size of the batch.

        Args:
            row: The request table row.
            start: First position, at the start of a cluster block.
            stop: One past the last position, at the end of a cluster block.
            index_keys_of: Positions [M] int64 to their index keys [M, index_dim] bf16.
        """
        block = self.config.cluster_block_size
        index_keys = read_span(index_keys_of, start, stop, self.device)
        with probe.region("index k cluster"):
            clustered = cluster_blocks(index_keys, self.config, start, self.settings.serving.kernels)
        # Each block's first cache row, on the device - a list sent from the host would wait for the stream
        block_starts = torch.arange(start, stop, block, device=self.device)
        first_rows = (
            self.request_pages[row, block_starts // PAGE_SIZE].long() * self.rows_per_page
            + block_starts % PAGE_SIZE // self.config.bucket_size
        )
        per_block = block // self.config.bucket_size
        rows = (first_rows[:, None] + torch.arange(per_block, device=self.device)).flatten()
        self.bucket_members[rows] = clustered.bucket_members
        self.bucket_sizes[rows] = clustered.bucket_sizes
        self.bucket_centroids[rows] = clustered.bucket_centroids
        if self.bucket_shapes is not None:
            with probe.region("index k cluster"):
                first = start
                while first < stop:
                    last = min(stop, (first // PAGE_SIZE + 1) * PAGE_SIZE)
                    buckets = slice(
                        (first - start) // self.config.bucket_size, (last - start) // self.config.bucket_size
                    )
                    self.shape_buckets(
                        rows[buckets],
                        clustered.bucket_members[buckets],
                        index_keys[first - start : last - start],
                        first,
                    )
                    first = last
        self.closed_keys[row] = stop
        self.closed_keys_device[row].fill_(stop)

    def shape_buckets(
        self, rows: torch.Tensor, bucket_members: torch.Tensor, index_keys: torch.Tensor, start: int
    ) -> None:
        """Build the shapes of buckets just clustered, from the keys of their blocks, into their cache rows.

        Args:
            rows: The buckets' cache rows [m] int64.
            bucket_members: Their key positions [m, bucket_cap] int32, -1 past a size.
            index_keys: The blocks' index keys [n, index_dim] bf16, unit rows, from position `start`.
            start: Position of the first key.
        """
        shapes, shape = self.bucket_shapes, self.config.bucket_shape
        if shapes is None or shape is None:
            return
        members = bucket_members.long()
        held = members >= 0
        local = (members - start).clamp_min(0)
        # fp32 centroids the shape is taken about - the bf16 ones rounded
        centroids = F.normalize((index_keys.float()[local] * held[..., None]).sum(1), dim=-1)
        basis, variances, floor = build_bucket_shapes(
            lambda positions: index_keys[(positions - start).clamp_min(0)], centroids, members, shape.rank
        )
        shapes.basis[rows] = basis
        shapes.variances[rows] = variances
        shapes.floor[rows] = floor
