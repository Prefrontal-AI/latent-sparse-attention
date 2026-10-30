"""The node buckets of every request row, mirrored into fixed-address device pools a captured decode step reads.

A hierarchy keeps its node buckets in growing tensors of its own, which move as they double and which a CUDA graph
cannot follow. The mirror holds each level in one pooled table of pages of `NODE_BUCKET_PAGE` node buckets instead, sized once
from the cache's bucket rows and the longest context, so every tensor a captured step reads keeps its address. A page
belongs to one request row, and a row's page table names its pages in creation order.

A level's node buckets are only ever appended, in creation order, and never change once made, so a sync copies only the
ones made since the last. The closed ones of a level are the first of its creation order and the open ones the rest.
"""

import torch

from lsa.sglang.index.hierarchy import Hierarchy, Rows
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings

# Node buckets a pooled page holds
NODE_BUCKET_PAGE = 256
INT32 = 4


def level_pages(
    settings: DesignSettings, total_buckets: int, request_count: int, max_context: int
) -> tuple[list[int], list[int], list[int]]:
    """Return the pages each level pools, the most pages one request holds of it and the most node buckets, for the levels the longest context grows.

    Args:
        settings: The design, with `top_level` 1 or more.
        total_buckets: Bucket rows the cache holds - every bucket the requests can hold between them.
        request_count: Rows of the request table.
        max_context: Longest context a request holds.

    Returns:
        Pooled pages a level, most pages a request holds of a level, and most node buckets a request makes of a level -
        each `level_count` long.
    """
    config = settings.indexer
    hierarchy = Hierarchy(config, settings.serving.node_centroid_precision)
    closed_buckets = max_context // PAGE_SIZE * (PAGE_SIZE // config.bucket_size)
    _, made, _ = hierarchy.counts_at(closed_buckets)
    pooled, per_request, most_nodes = [], [], []
    for level, count in enumerate(made, 1):
        if not count:
            break
        # A request wastes at most a page of a level, the rest is what the buckets can make
        pooled.append(
            -(-total_buckets // (hierarchy.node_bucket_size**level * NODE_BUCKET_PAGE)) + request_count
        )
        per_request.append(-(-count // NODE_BUCKET_PAGE))
        most_nodes.append(count)
    return pooled, per_request, most_nodes


class GraphTree:
    """Every request row's node buckets, one pooled table of pages a level, at addresses that never move.

    Level `l` is `node_centroids[l - 1]` [pooled pages * `NODE_BUCKET_PAGE`, index_dim], bf16 or fp8 as the hierarchy stores
    them, and `children[l - 1]` [pooled pages * `NODE_BUCKET_PAGE`, node_bucket_size] int32 - cache rows under level 1, creation
    indices of the level below above it. `pages[l - 1]` [request_count, most pages a request holds] int32 names a row's
    pages in creation order, -1 for none, and creation index `i` sits in page `i // NODE_BUCKET_PAGE` at slot `i % NODE_BUCKET_PAGE`.
    `made` and `closed` [level_count, request_count] int32 count each row's node buckets and its closed ones a level,
    `start` [request_count] int32 is its start level - 0 with no levels - and `covered` [request_count] int32 its
    covered buckets. `most_nodes` is the most node buckets a request makes of each level.
    """

    def __init__(
        self,
        settings: DesignSettings,
        total_buckets: int,
        request_count: int,
        max_context: int,
        device: torch.device | str,
    ) -> None:
        """Size the pools for a cache and allocate them, empty.

        Args:
            settings: The design, with `top_level` 1 or more.
            total_buckets: Bucket rows the cache holds.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
            device: Where the pools live.

        Raises:
            ValueError: The design shapes its node buckets, which the mirror does not hold.
        """
        config = settings.indexer
        if config.node_bucket_shape is not None:
            raise ValueError(
                f"the graph tree mirrors node buckets of centroids and children only - got node_bucket_shape "
                f"{config.node_bucket_shape}, wanted none"
            )
        pooled, per_request, most_nodes = level_pages(settings, total_buckets, request_count, max_context)
        self.settings = settings
        self.most_nodes = most_nodes
        self.device = device
        self.level_count = len(pooled)
        self.node_bucket_size = Hierarchy(config, settings.serving.node_centroid_precision).node_bucket_size
        centroid_dtype = (
            torch.float8_e4m3fn if settings.serving.node_centroid_precision == "fp8" else torch.bfloat16
        )
        self.node_centroids = [
            torch.zeros(pages * NODE_BUCKET_PAGE, config.index_dim, device=device, dtype=centroid_dtype)
            for pages in pooled
        ]
        self.children = [
            torch.full(
                (pages * NODE_BUCKET_PAGE, self.node_bucket_size), -1, device=device, dtype=torch.int32
            )
            for pages in pooled
        ]
        self.pages = [
            torch.full((request_count, pages), -1, device=device, dtype=torch.int32) for pages in per_request
        ]
        self.made = torch.zeros(self.level_count, request_count, device=device, dtype=torch.int32)
        self.closed = torch.zeros(self.level_count, request_count, device=device, dtype=torch.int32)
        self.start = torch.zeros(request_count, device=device, dtype=torch.int32)
        self.covered = torch.zeros(request_count, device=device, dtype=torch.int32)
        # Popped from the end - a lone request's pages come out in order
        self.free = [list(range(pages - 1, -1, -1)) for pages in pooled]
        # Pages each request holds a level, and the node buckets of it already copied
        self.held_pages: list[list[list[int]]] = [
            [[] for _ in range(self.level_count)] for _ in range(request_count)
        ]
        self.copied = [[0] * self.level_count for _ in range(request_count)]

    @staticmethod
    def pool_bytes(settings: DesignSettings, total_buckets: int, request_count: int, max_context: int) -> int:
        """Return the bytes one layer's mirror allocates, from the settings alone.

        Args:
            settings: The design, with `top_level` 1 or more.
            total_buckets: Bucket rows the cache holds.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
        """
        pooled, per_request, _ = level_pages(settings, total_buckets, request_count, max_context)
        nodes = sum(pooled) * NODE_BUCKET_PAGE * sum(GraphTree.node_bucket_bytes(settings).values())
        tables = sum(per_request) * request_count * INT32
        counts = (2 * len(pooled) + 2) * request_count * INT32
        return nodes + tables + counts

    @staticmethod
    def builds(settings: DesignSettings) -> bool:
        """Return whether a design descends through a graph tree - node levels, a descending decode, unshaped node buckets.

        Args:
            settings: The design.
        """
        config = settings.indexer
        return (
            config.top_level >= 1
            and settings.serving.decode == "descend"
            and config.node_bucket_shape is None
        )

    @staticmethod
    def bytes_per_bucket(settings: DesignSettings, max_context: int) -> float:
        """Return the bytes the pools grow by for each bucket row of the cache.

        Level `l` pools a node bucket for each `node_bucket_size ** l` bucket rows, down to the levels the longest context
        grows. What a request holds past that - a page of each level - is `fixed_bytes`.

        Args:
            settings: The design, with `top_level` 1 or more.
            max_context: Longest context a request holds.
        """
        node = sum(GraphTree.node_bucket_bytes(settings).values())
        size = Hierarchy(settings.indexer, settings.serving.node_centroid_precision).node_bucket_size
        levels = len(level_pages(settings, 0, 0, max_context)[0])
        return node * sum(1 / size**level for level in range(1, levels + 1))

    @staticmethod
    def fixed_bytes(settings: DesignSettings, request_count: int, max_context: int) -> int:
        """Return the bytes one layer's mirror allocates however many bucket rows the cache holds.

        That is a page of each level a request holds and the page tables and counts, with a page of each level more for the
        pools' rounding up. `pool_bytes` is at most this and `bytes_per_bucket` for each bucket row.

        Args:
            settings: The design, with `top_level` 1 or more.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
        """
        levels = len(level_pages(settings, 0, 0, max_context)[0])
        rounding = levels * NODE_BUCKET_PAGE * sum(GraphTree.node_bucket_bytes(settings).values())
        return GraphTree.pool_bytes(settings, 0, request_count, max_context) + rounding

    @staticmethod
    def node_bucket_bytes(settings: DesignSettings) -> dict[str, int]:
        """Return the bytes a node bucket holds in the pools, by the structure that holds them.

        Args:
            settings: The design, with `top_level` 1 or more.
        """
        config = settings.indexer
        fp8 = settings.serving.node_centroid_precision == "fp8"
        size = Hierarchy(config, settings.serving.node_centroid_precision).node_bucket_size
        return {"node_centroids": config.index_dim * (1 if fp8 else 2), "children": size * INT32}

    def held(self, row: int) -> dict[str, int]:
        """Return the bytes a request row's mirror holds, by the structure that holds them.

        Args:
            row: The request table row.
        """
        held = self.held_pages[row]
        node = GraphTree.node_bucket_bytes(self.settings)
        pages = sum(len(level) for level in held)
        return {
            "graph_node_centroids": sum(len(level) for level in held)
            * NODE_BUCKET_PAGE
            * node["node_centroids"],
            "graph_children": pages * NODE_BUCKET_PAGE * node["children"],
            "graph_pages": pages * INT32,
        }

    def sync(self, row: int, hierarchy: Hierarchy) -> None:
        """Copy the node buckets a request's hierarchy made since the last sync, and refresh the row's counts.

        Args:
            row: The request table row.
            hierarchy: The row's hierarchy, grown at least as far as the last sync saw it.

        Raises:
            ValueError: The hierarchy holds more levels than the longest context grows, or fewer node buckets of a
                level than the mirror holds - `reset` the row first.
        """
        if hierarchy.levels > self.level_count:
            raise ValueError(
                f"the graph tree holds {self.level_count} levels for the cache's longest context - "
                f"got a hierarchy of {hierarchy.levels}"
            )
        for level in range(1, hierarchy.levels + 1):
            closed, opened = hierarchy.closed_node_buckets[level - 1], hierarchy.open_node_buckets[level - 1]
            made, first = len(closed) + len(opened), self.copied[row][level - 1]
            if made < first:
                raise ValueError(
                    f"level {level} holds {made} node buckets and the mirror has copied {first} - reset the row first"
                )
            if made == first:
                continue
            self.take_pages(row, level, made)
            self.copy(row, level, first, made, closed, opened)
            self.copied[row][level - 1] = made
        self.publish(row, hierarchy)

    def reset(self, row: int) -> None:
        """Free a request row's pages and zero its counts.

        Args:
            row: The request table row.
        """
        for level, held in enumerate(self.held_pages[row]):
            self.free[level].extend(reversed(held))
            held.clear()
            self.pages[level][row].fill_(-1)
        self.copied[row] = [0] * self.level_count
        self.made[:, row].fill_(0)
        self.closed[:, row].fill_(0)
        self.start[row].fill_(0)
        self.covered[row].fill_(0)

    def take_pages(self, row: int, level: int, made: int) -> None:
        """Give a request row the pages that `made` node buckets of a level fill.

        Args:
            row: The request table row.
            level: The level, from 1.
            made: Node buckets the row holds of it.

        Raises:
            RuntimeError: The pool has no page free, or the row would hold more than the longest context grows.
        """
        held = self.held_pages[row][level - 1]
        needed = -(-made // NODE_BUCKET_PAGE)
        if needed <= len(held):
            return
        if needed > self.pages[level - 1].shape[1]:
            raise RuntimeError(
                f"level {level} needs {needed} pages for {made} node buckets and a request holds at most "
                f"{self.pages[level - 1].shape[1]} - the context outgrew the cache's maximum"
            )
        if len(self.free[level - 1]) < needed - len(held):
            raise RuntimeError(
                f"the graph tree's level {level} is out of pages - wanted {needed - len(held)}, "
                f"{len(self.free[level - 1])} free"
            )
        taken = [self.free[level - 1].pop() for _ in range(needed - len(held))]
        self.pages[level - 1][row, len(held) : needed] = torch.tensor(
            taken, dtype=torch.int32, device=self.device
        )
        held.extend(taken)

    def copy(self, row: int, level: int, first: int, stop: int, closed: Rows, opened: Rows) -> None:
        """Copy node buckets [first, stop) of a level in creation order into the row's pages.

        Args:
            row: The request table row.
            level: The level, from 1.
            first: The first creation index to copy.
            stop: One past the last.
            closed: The level's closed node buckets, the first of its creation order.
            opened: Its open node buckets, the rest.
        """
        index = torch.arange(first, stop, device=self.device)
        physical = (
            self.pages[level - 1][row][index // NODE_BUCKET_PAGE].long() * NODE_BUCKET_PAGE
            + index % NODE_BUCKET_PAGE
        )
        self.node_centroids[level - 1][physical] = created(closed, opened, "node_centroids", first, stop)
        self.children[level - 1][physical] = created(closed, opened, "children", first, stop)

    def publish(self, row: int, hierarchy: Hierarchy) -> None:
        """Write a request row's counts, start level and covered buckets to the device.

        Args:
            row: The request table row.
            hierarchy: The row's hierarchy.
        """
        levels = range(hierarchy.levels)
        made = [self.copied[row][level] for level in levels] + [0] * (self.level_count - hierarchy.levels)
        closed = [len(hierarchy.closed_node_buckets[level]) for level in levels] + [0] * (
            self.level_count - hierarchy.levels
        )
        self.made[:, row] = torch.tensor(made, dtype=torch.int32, device=self.device)
        self.closed[:, row] = torch.tensor(closed, dtype=torch.int32, device=self.device)
        self.start[row] = hierarchy.start_level() if hierarchy.levels else 0
        self.covered[row] = hierarchy.covered_buckets


def created(closed: Rows, opened: Rows, name: str, first: int, stop: int) -> torch.Tensor:
    """Return a field of the node buckets [first, stop) in creation order - the closed ones, then the open.

    Args:
        closed: A level's closed node buckets, the first of its creation order.
        opened: Its open node buckets, the rest.
        name: The field.
        first: The first creation index.
        stop: One past the last.

    Returns:
        The field's rows [stop - first, ...].
    """
    shut = len(closed)
    parts = []
    if first < shut:
        parts.append(closed[name][first : min(stop, shut)])
    if stop > shut:
        parts.append(opened[name][max(first, shut) - shut : stop - shut])
    return torch.cat(parts)
