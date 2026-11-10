"""One request's hierarchy - node bucket levels over its closed buckets, grown in full groups and read by a beam.

Level 0 is the cache's own buckets. As soon as a node cluster block's worth of closed buckets that no node bucket
covers yet exist, they are paired by the Ward merge `log2(node_bucket_size)` times into node buckets of exactly
`node_bucket_size` children each. Every level above forms the same way over its own uncovered node buckets, up
to `top_level`, so a level never holds more than a block of uncovered node buckets below the top and the frontier a query
scores stays bounded while the context grows.

A node bucket's centroid is the normalized sum of every index key beneath it. Nothing moves when a block is paired:
the cache's bucket rows stay where clustering wrote them and the closed rows of a level stay in the order they
closed. A node bucket keeps its `node_bucket_size` children as a row of row numbers - cache rows for level 1, the
level below's closed rows above it - which the beam's child kernels read through.

A node bucket is open until a node bucket above covers it, and closed after. Open node buckets are scored
directly, as the window's keys are read at level 0.

Under `node_bucket_shape` a node bucket also keeps the shape of every key beneath it, built as it opens, and the beam
scores it by its estimated best key - its open node buckets with the bits expanded on the tensor cores, its carried children
against the query's planes, as the internal snapshot scored them. Level 0 is scored by `bucket_shape` when the design sets one.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from lsa.core.kernels import dispatch
from lsa.core.kernels.constants import TritonBeam, TritonScanSelect
from lsa.core.kernels.cutedsl import beam as cutedsl_beam
from lsa.core.kernels.scan_select import ScanSelect
from lsa.core.kernels.triton import beam
from lsa.core.kernels.triton.shape import add_planes_spread, add_spread, build_shapes, score_shaped_rows
from lsa.core.kernels.triton.top_k import TopK
from lsa.core.kernels.ward import cluster_node_buckets
from lsa.core.ward import ward_rounds
from lsa.sglang.index import probe
from lsa.sglang.settings.design import PAGE_SIZE, IndexerConfig

if TYPE_CHECKING:
    from lsa.sglang.index.bucket_cache import BucketCache

# Positions [M] int64 to their index keys [M, index_dim] bf16, unit rows
IndexKeys = Callable[[torch.Tensor], torch.Tensor]
# Key positions summed a call - a chunk of buckets' members gathered at once
KEY_SUM_POSITIONS = 1 << 17
# Buckets the first level pairs at once at least - smaller groups wait until this many have closed, so a group of
# a page's buckets is not a pairing's launches every page; what waits is scored where it stands
GROWTH_BUCKETS = 4096


def read_span(index_keys: IndexKeys, start: int, stop: int, device: torch.device | str) -> torch.Tensor:
    """Return the index keys of positions [start, stop) [stop - start, index_dim] bf16 - by the lookup's own span read when it has one."""
    span = getattr(index_keys, "span", None)
    if span is None:
        return index_keys(torch.arange(start, stop, device=device))
    return span(start, stop)


class Rows:
    """Rows of named tensors appended at the end and taken from the front, each grown by doubling."""

    def __init__(self) -> None:
        """Set up empty rows."""
        self.parts: dict[str, torch.Tensor] = {}
        self.start = 0
        self.stop = 0

    def __len__(self) -> int:
        """Return the rows held."""
        return self.stop - self.start

    def __eq__(self, other: object) -> bool:
        """Return whether both hold equal rows under the same names."""
        if not isinstance(other, Rows) or len(self) != len(other) or self.parts.keys() != other.parts.keys():
            return False
        return all(torch.equal(self[name], other[name]) for name in self.parts)

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the rows held of a named tensor."""
        return self.parts[name][self.start : self.stop]

    def extend(self, **rows: torch.Tensor) -> None:
        """Append rows, every named tensor with the same number of them."""
        count = len(next(iter(rows.values())))
        if not self.parts or self.stop + count > len(next(iter(self.parts.values()))):
            self.grow(rows, count)
        for name, part in rows.items():
            self.parts[name][self.stop : self.stop + count] = part
        self.stop += count

    def held(self) -> dict[str, int]:
        """Return the bytes each named tensor holds, its room to grow included."""
        return {name: part.numel() * part.element_size() for name, part in self.parts.items()}

    def part(self, start: int, stop: int) -> "Rows":
        """Return a copy of rows [start, stop), with every name even when it holds none."""
        part = Rows()
        if self.parts:
            part.extend(**{name: self[name][start:stop] for name in self.parts})
        return part

    def take(self, count: int) -> dict[str, torch.Tensor]:
        """Remove the first `count` rows and return copies of them."""
        taken = {name: part[self.start : self.start + count].clone() for name, part in self.parts.items()}
        self.start += count
        return taken

    def grow(self, rows: dict[str, torch.Tensor], count: int) -> None:
        """Move the live rows to the front of tensors twice what they and the new ones need."""
        capacity = max(64, 2 * (len(self) + count))
        grown = {}
        for name, part in rows.items():
            fresh = part.new_empty((capacity, *part.shape[1:]))
            if name in self.parts:
                fresh[: len(self)] = self[name]
            grown[name] = fresh
        self.parts, self.stop, self.start = grown, len(self), 0


def pair_into_node_buckets(
    key_sums: torch.Tensor, node_bucket_size: int, kernels: str = "triton"
) -> torch.Tensor:
    """Each bucket's node bucket [G, P] int64 - the buckets paired by the Ward merge, then the pairs, to exactly the size.

    A merge to half the buckets with at most two a bucket is a perfect matching, which always completes, where one
    merge to a quarter with at most four stops short of exactly four.

    Args:
        key_sums: Each group's buckets' key sums [G, P, index_dim] fp32, P a power of two.
        node_bucket_size: Children a node bucket holds, a power of two.
        kernels: The language of the pairing's rounds - the same pairs either way.

    Returns:
        Each bucket's node bucket within its group, numbered from 0 in the order of its first child.
    """
    groups, buckets, index_dim = key_sums.shape
    labels = torch.arange(buckets, device=key_sums.device).expand(groups, buckets)
    for _ in range(node_bucket_size.bit_length() - 1):
        count = key_sums.shape[1]
        pairs = pair(key_sums, kernels)
        labels = pairs.gather(1, labels)
        key_sums = torch.zeros(groups, count // 2, index_dim, device=key_sums.device).scatter_add_(
            1, pairs[..., None].expand_as(key_sums), key_sums
        )
    return labels


def pair(key_sums: torch.Tensor, kernels: str = "triton") -> torch.Tensor:
    """Each bucket's pair [G, P] int64 - a Ward merge of every group to half its buckets, at most two a bucket.

    On a GPU the pairing kernel, which spreads a group's rounds over the device and searches only the buckets still
    unpaired; elsewhere the torch rounds.

    Args:
        key_sums: Each group's buckets' key sums [G, P, index_dim] fp32.
        kernels: The language of the pairing's rounds on a GPU - the same pairs either way.

    Returns:
        Each bucket's pair [G, P] int64, numbered from 0 in the order of its first bucket.
    """
    if key_sums.is_cuda:
        return cluster_node_buckets(key_sums, kernels=kernels)
    ones = torch.ones(key_sums.shape[:2], device=key_sums.device)
    return ward_rounds(key_sums, ones, key_sums.shape[1] // 2, 2)[0]


def bucket_key_sums(index_keys: IndexKeys, members: torch.Tensor) -> torch.Tensor:
    """Each bucket's index key sum [N, index_dim] fp32 - members [N, bucket_cap] positions, -1 past a size - a chunk at a time."""
    sums = []
    chunk = max(1, KEY_SUM_POSITIONS // members.shape[1])
    for first in range(0, len(members), chunk):
        part = members[first : first + chunk]
        keys = index_keys(part.clamp_min(0).flatten()).view(*part.shape, -1).float()
        sums.append((keys * (part >= 0)[..., None]).sum(1))
    return torch.cat(sums)


class Hierarchy:
    """The node bucket levels of one request - open node buckets no node bucket above covers, and closed ones.

    `open_node_buckets[level - 1]` and `closed_node_buckets[level - 1]` hold level `level`'s node buckets: `node_centroids` [N, index_dim],
    bf16 or fp8 at `NODE_CENTROID_SCALE`, and `children` [N, node_bucket_size] int32 - the rows of their
    children in the level below's closed rows, the cache's bucket rows under level 1. An open node bucket
    below the top level also keeps its `key_sums` [N, index_dim] fp32, which the pairing a level up reads.
    Under `node_bucket_shape` every node bucket keeps its shape too: `basis` [N, rank, index_dim / 8] uint8, `variances`
    [N, rank] fp32 and `floor` [N] fp32.
    """

    def __init__(self, config: IndexerConfig, node_centroid_precision: str, kernels: str = "triton") -> None:
        """Hold no levels yet.

        Args:
            config: The design, with `top_level` 1 or more.
            node_centroid_precision: `bf16` or `fp8`, what node centroids are stored as.
            kernels: `cutedsl` to run a beam step's scoring and keep in one CuTeDSL kernel where it takes the rows,
                and a growth's pairing in one launch, `triton` for every level and round - the same tree either way.

        Raises:
            ValueError: The node bucket size differs between levels, or a node cluster block is not a power of
                two from 16 node bucket sizes to 4,096.
        """
        self.config = config
        self.fp8 = node_centroid_precision == "fp8"
        self.kernels = kernels
        self.node_shape = config.node_bucket_shape
        sizes = {config.at("node_bucket_size", level) for level in range(1, config.top_level + 1)}
        if len(sizes) != 1:
            raise ValueError(f"a hierarchy holds one node bucket size at every level - got {sorted(sizes)}")
        self.node_bucket_size = sizes.pop()
        for level in range(1, config.top_level + 1):
            block = config.at("node_cluster_block_size", level)
            if block & (block - 1) or not 16 * self.node_bucket_size <= block <= 4096:
                raise ValueError(
                    f"node_cluster_block_size must be a power of two from 16 node bucket sizes to 4,096 - "
                    f"got {block} at level {level} with node_bucket_size {self.node_bucket_size}"
                )
        # Level 0 buckets a node bucket covers - the request's first ones, in logical order
        self.covered_buckets = 0
        self.open_node_buckets: list[Rows] = []
        self.closed_node_buckets: list[Rows] = []

    def __eq__(self, other: object) -> bool:
        """Return whether both cover the same buckets and hold equal node buckets."""
        if not isinstance(other, Hierarchy):
            return False
        return (self.covered_buckets, self.open_node_buckets, self.closed_node_buckets) == (
            other.covered_buckets,
            other.open_node_buckets,
            other.closed_node_buckets,
        )

    @property
    def levels(self) -> int:
        """Node bucket levels grown so far."""
        return len(self.open_node_buckets)

    def held(self) -> dict[str, int]:
        """Return the bytes each named tensor holds, over every level's open and closed node buckets."""
        held: dict[str, int] = {}
        for rows in self.open_node_buckets + self.closed_node_buckets:
            for name, count in rows.held().items():
                held[name] = held.get(name, 0) + count
        return held

    def growth_blocks(self, closed_buckets: int) -> int:
        """Return the node cluster blocks a growth over `closed_buckets` pairs - 0 while too few wait.

        Args:
            closed_buckets: Level 0 buckets of the request that are closed - its first ones, in logical order.
        """
        block = self.config.at("node_cluster_block_size", 1)
        blocks = (closed_buckets - self.covered_buckets) // block
        return blocks if blocks >= max(1, GROWTH_BUCKETS // block) else 0

    def grow(self, cache: "BucketCache", row: int, closed_buckets: int, index_keys: IndexKeys) -> None:
        """Grow every level over its full blocks of uncovered node buckets, the buckets first, `GROWTH_BUCKETS` at least.

        Args:
            cache: The bucket cache, whose bucket rows level 1 arranges.
            row: The request's row.
            closed_buckets: Level 0 buckets of the request that are closed - its first ones, in logical order.
            index_keys: Positions to their index keys.
        """
        blocks = self.growth_blocks(closed_buckets)
        if not blocks:
            return
        with probe.region("index k cluster"):
            self.grow_from_buckets(cache, row, blocks, index_keys)
            level = 1
            while level < self.config.top_level:
                block = self.config.at("node_cluster_block_size", level + 1)
                if len(self.open_node_buckets[level - 1]) < block:
                    break
                self.grow_from_level(
                    cache, level, len(self.open_node_buckets[level - 1]) // block, index_keys
                )
                level += 1

    def counts_at(self, closed_buckets: int) -> tuple[int, list[int], list[int]]:
        """Return what growing at every page end up to `closed_buckets` leaves, from the counts alone.

        The cache settles at every page end, so this replays `grow` there: the buckets covered, and each level's
        node buckets made and closed.

        Args:
            closed_buckets: Level 0 buckets closed, a page end's.

        Returns:
            Covered buckets, node buckets made a level, node buckets closed a level - the lists `top_level` long.
        """
        top, size = self.config.top_level, self.node_bucket_size
        block = self.config.at("node_cluster_block_size", 1)
        per_page = PAGE_SIZE // self.config.bucket_size
        covered, made, closed = 0, [0] * top, [0] * top
        for end in range(per_page, closed_buckets + 1, per_page):
            blocks = (end - covered) // block
            if blocks < max(1, GROWTH_BUCKETS // block):
                continue
            covered += blocks * block
            made[0] += blocks * block // size
            for level in range(1, top):
                above = self.config.at("node_cluster_block_size", level + 1)
                taken = (made[level - 1] - closed[level - 1]) // above * above
                if not taken:
                    break
                closed[level - 1] += taken
                made[level] += taken // size
        return covered, made, closed

    def trimmed(self, closed_buckets: int) -> "Hierarchy":
        """Return a copy without the node buckets made after `closed_buckets` - those closed since stay closed.

        What a retained prefix keeps, with no index key read: `truncated` opens the closed ones again when a
        request takes the prefix.

        Args:
            closed_buckets: A page end's level 0 buckets, at most this hierarchy's.
        """
        covered, made, _ = self.counts_at(closed_buckets)
        kept = Hierarchy(self.config, "fp8" if self.fp8 else "bf16", self.kernels)
        kept.covered_buckets = covered
        for level in range(1, self.levels + 1):
            count = made[level - 1]
            if not count:
                break
            held = self.closed_node_buckets[level - 1]
            kept.closed_node_buckets.append(held.part(0, min(len(held), count)) if len(held) else Rows())
            kept.open_node_buckets.append(
                self.open_node_buckets[level - 1].part(0, max(0, count - len(held)))
            )
        return kept

    def truncated(self, closed_buckets: int, cache: "BucketCache", index_keys: IndexKeys) -> "Hierarchy":
        """Return the hierarchy of a request grown to `closed_buckets`, from this one grown at least as far.

        Growth only appends, so a level's node buckets then are its first ones now. One made later is dropped, and
        one closed later is opened again with its key sums, summed from the index keys beneath it in the order its
        making summed them.

        Args:
            closed_buckets: A page end's level 0 buckets, at most this hierarchy's.
            cache: The bucket cache holding the buckets beneath, at the rows the node buckets name.
            index_keys: Positions to their index keys - the same keys this hierarchy was grown from.
        """
        covered, made, closed = self.counts_at(closed_buckets)
        truncated = Hierarchy(self.config, "fp8" if self.fp8 else "bf16", self.kernels)
        truncated.covered_buckets = covered
        for level in range(1, self.levels + 1):
            count, kept = made[level - 1], closed[level - 1]
            if not count:
                break
            held = self.closed_node_buckets[level - 1]
            truncated.closed_node_buckets.append(held.part(0, kept) if kept else Rows())
            opened = self.open_node_buckets[level - 1].part(0, max(0, count - len(held)))
            stop = min(len(held), count)
            if kept < stop:
                reopened = {name: held[name][kept:stop] for name in held.parts}
                if level < self.config.top_level:
                    nodes = torch.arange(kept, stop, device=held["children"].device)
                    reopened["key_sums"] = self.summed(level, nodes, cache, index_keys)
                rows = Rows()
                rows.extend(**reopened)
                if len(opened):
                    rows.extend(**{name: opened[name] for name in opened.parts})
                opened = rows
            truncated.open_node_buckets.append(opened)
        return truncated

    def summed(
        self, level: int, nodes: torch.Tensor, cache: "BucketCache", index_keys: IndexKeys
    ) -> torch.Tensor:
        """Return the key sums of closed node buckets [N, index_dim] fp32, each its children's in their order.

        Args:
            level: Their level, from 1.
            nodes: Their closed rows [N] int64.
            cache: The bucket cache holding the buckets beneath.
            index_keys: Positions to their index keys.
        """
        children = self.closed_node_buckets[level - 1]["children"][nodes].long().flatten()
        if level == 1:
            sums = bucket_key_sums(index_keys, cache.bucket_members[children].long())
        else:
            sums = self.summed(level - 1, children, cache, index_keys)
        return sums.view(len(nodes), self.node_bucket_size, -1).sum(1)

    def grow_from_buckets(self, cache: "BucketCache", row: int, blocks: int, index_keys: IndexKeys) -> None:
        """Cluster the next blocks of buckets into level-1 node buckets, which name their buckets' rows."""
        size = self.node_bucket_size
        block = self.config.at("node_cluster_block_size", 1)
        count = blocks * block
        rows = cache.logical_rows(row, count, first=self.covered_buckets)[:count]
        key_sums = bucket_key_sums(index_keys, cache.bucket_members[rows].long())
        labels = pair_into_node_buckets(key_sums.view(blocks, block, -1), size, self.kernels)
        order = (
            torch.argsort(labels, dim=1, stable=True)
            + block * torch.arange(blocks, device=labels.device)[:, None]
        ).flatten()
        node_bucket_sums = key_sums[order].view(count // size, size, -1).sum(1)
        children = rows[order].view(-1, size).int()
        self.stage(1, node_bucket_sums, children, cache, index_keys)
        self.covered_buckets += count

    def grow_from_level(self, cache: "BucketCache", level: int, blocks: int, index_keys: IndexKeys) -> None:
        """Close the level's first open blocks in the order they opened, and open their node buckets above."""
        size = self.node_bucket_size
        block = self.config.at("node_cluster_block_size", level + 1)
        count = blocks * block
        node_buckets = self.open_node_buckets[level - 1].take(count)
        labels = pair_into_node_buckets(node_buckets["key_sums"].view(blocks, block, -1), size, self.kernels)
        order = (
            torch.argsort(labels, dim=1, stable=True)
            + block * torch.arange(blocks, device=labels.device)[:, None]
        ).flatten()
        closed = self.closed_node_buckets[level - 1]
        first = len(closed)
        closed.extend(**{name: part for name, part in node_buckets.items() if name != "key_sums"})
        node_bucket_sums = node_buckets["key_sums"][order].view(count // size, size, -1).sum(1)
        children = (first + order).view(-1, size).int()
        self.stage(level + 1, node_bucket_sums, children, cache, index_keys)

    def buckets_under(self, level: int, children: torch.Tensor) -> torch.Tensor:
        """The cache rows of every bucket under level-`level` node buckets [N, node_bucket_size ** level] int64.

        Args:
            level: A level from 1 to `levels`.
            children: The node buckets' children [N, node_bucket_size] int32.
        """
        rows = children.long()
        for below in range(level - 1, 0, -1):
            rows = self.closed_node_buckets[below - 1]["children"][rows].long().flatten(1)
        return rows

    def stage(
        self,
        level: int,
        key_sums: torch.Tensor,
        children: torch.Tensor,
        cache: "BucketCache | None" = None,
        index_keys: IndexKeys | None = None,
    ) -> None:
        """Open new node buckets at a level, shaped over every key beneath them when node buckets are.

        Args:
            level: The level, from 1.
            key_sums: The node buckets' index key sums [N, index_dim] fp32.
            children: Their children [N, node_bucket_size] int32.
            cache: The bucket cache whose members a shape reads - only needed under `node_bucket_shape`.
            index_keys: Positions to their index keys - only needed under `node_bucket_shape`.

        Raises:
            ValueError: Node buckets are shaped and the cache or the index keys are missing.
        """
        while self.levels < level:
            self.open_node_buckets.append(Rows())
            self.closed_node_buckets.append(Rows())
        node_centroids = F.normalize(key_sums, dim=-1)
        stored = (
            (node_centroids * TritonBeam.NODE_CENTROID_SCALE).to(torch.float8_e4m3fn)
            if self.fp8
            else node_centroids.to(torch.bfloat16)
        )
        node_buckets = {"node_centroids": stored, "children": children}
        # A level at the top is never paired, so its node buckets keep no key sums
        if level < self.config.top_level:
            node_buckets["key_sums"] = key_sums
        if self.node_shape is not None:
            if cache is None or index_keys is None:
                raise ValueError(
                    "a shaped node bucket is built from the keys beneath it - pass the cache and index keys"
                )
            members = cache.bucket_members[self.buckets_under(level, children)].long().flatten(1)
            basis, variances, floor = build_shapes(index_keys, node_centroids, members, self.node_shape.rank)
            node_buckets.update(basis=basis, variances=variances, floor=floor)
        self.open_node_buckets[level - 1].extend(**node_buckets)

    def start_level(self) -> int:
        """Return the level a beam starts at - the lowest of at most `start_nodes` node buckets, else the top.

        Returns:
            A level from 1 to `levels`.
        """
        start_nodes = self.config.start_nodes
        if start_nodes:
            for level in range(1, self.levels + 1):
                if (
                    len(self.open_node_buckets[level - 1]) + len(self.closed_node_buckets[level - 1])
                    <= start_nodes
                ):
                    return level
        return self.levels

    def beam_search(
        self,
        cache: "BucketCache",
        row: int,
        index_queries: torch.Tensor,
        visible_counts: torch.Tensor,
        beam_top_k: Callable[[int], TopK],
        pick_top_k: TopK,
        beam_select: Callable[[int], ScanSelect] | None = None,
        visible: int | None = None,
    ) -> torch.Tensor:
        """The best `top_p` buckets for each query by a beam down the levels.

        Each level's open node buckets are scored with the children of the beam the level above kept, into one row of
        scores a query, and the level keeps the best `beam_width`; level 0 scores the buckets no node bucket covers
        within each query's visible count beside the last beam's children, and keeps the best `top_p`.
        Under `start_nodes` the beam starts at `start_level` and scores every node bucket of it, as though the level above
        had kept all of its own.

        Args:
            cache: The bucket cache.
            row: The request's row.
            index_queries: Index queries [S, index_dim] fp32.
            visible_counts: Buckets each query sees [S] int32.
            beam_top_k: The top-k over a level's scores, by the beam width it keeps.
            pick_top_k: The top-k over the buckets' scores, `top_p` wide - the expansion by threshold reads its listing.
            beam_select: The fused select, by beam width, a start level at least `SCAN_SELECT_BEAM_WIDTH` wide takes
                its beam by without writing its score matrix - None for the top-k.
            visible: Buckets of closed cluster blocks the cache is read over, at least every query's visible count - the
                cache's own when None.

        Returns:
            The scores of the buckets that hold each query's best `top_p`, packed with their cache rows [S, W] int64 in
            no order, W being what a listing holds - the empty key for none.
        """
        size = self.node_bucket_size
        count = len(index_queries)
        index_queries = index_queries.to(torch.bfloat16)
        start = self.start_level()
        carried = None
        for level in range(start, 0, -1):
            open_node_buckets, closed_node_buckets = (
                self.open_node_buckets[level - 1],
                self.closed_node_buckets[level - 1],
            )
            open_count = len(open_node_buckets)
            whole = level == start and self.config.start_nodes is not None and len(closed_node_buckets) > 0
            # A level whose every score is in hand before its keep - the top, or a start scored whole
            top = whole or carried is None
            children = len(closed_node_buckets) if whole else (0 if carried is None else carried.shape[1])
            width = open_count + children
            beam_width = self.config.at("beam_width", level)
            # A shaped level adds its spread to every score, which the fused select does not
            unshaped = self.node_shape is None
            if (
                beam_select is not None
                and unshaped
                and carried is None
                and width >= TritonScanSelect.SCAN_SELECT_BEAM_WIDTH
            ):
                names = ("node_centroids", "children")
                level_node_buckets = (
                    {name: torch.cat([open_node_buckets[name], closed_node_buckets[name]]) for name in names}
                    if whole
                    else open_node_buckets
                )
                with probe.region("index q scan"):
                    carried = self.fused_beam(
                        index_queries,
                        level_node_buckets["node_centroids"],
                        level_node_buckets["children"],
                        beam_select(beam_width),
                    )
                continue
            with probe.region("index q scan"):
                if whole:
                    # Every node bucket in one product - the closed ones as the children every query carries
                    level_node_buckets = {
                        name: torch.cat([open_node_buckets[name], closed_node_buckets[name]])
                        for name in closed_node_buckets.parts
                    }
                    scores = self.node_bucket_scores(index_queries, level_node_buckets)
                    top_children = level_node_buckets["children"]
                    rows = torch.arange(
                        len(closed_node_buckets), device=index_queries.device, dtype=torch.int32
                    )
                    carried = rows[None].expand(count, -1)
                elif carried is None:
                    scores = self.node_bucket_scores(index_queries, open_node_buckets)
                    top_children = open_node_buckets["children"]
                elif self.fused_level(index_queries, closed_node_buckets, carried, open_count, beam_width):
                    # Scoring, keep and children in one kernel - the row of scores is never written
                    carried = cutedsl_beam.beam_step(
                        index_queries,
                        self.node_bucket_scores(index_queries, open_node_buckets),
                        open_node_buckets["children"],
                        carried,
                        closed_node_buckets["node_centroids"],
                        closed_node_buckets["children"],
                        beam_width,
                        TritonBeam.NODE_CENTROID_SCALE if self.fp8 else 1.0,
                        None
                        if self.node_shape is None
                        else (
                            closed_node_buckets["basis"],
                            closed_node_buckets["variances"],
                            closed_node_buckets["floor"],
                            self.node_shape.deviations,
                        ),
                    )
                    continue
                else:
                    scores = torch.empty(count, width, device=index_queries.device)
                    scores[:, :open_count] = self.node_bucket_scores(index_queries, open_node_buckets)
                    self.child_scores(index_queries, closed_node_buckets, carried, scores[:, open_count:])
            with probe.region("index k select"):
                if top and self.kept_in_cutedsl(index_queries, scores, beam_width):
                    carried = cutedsl_beam.keep_start_level(index_queries, scores, top_children, beam_width)
                    continue
                kept = beam_top_k(beam_width)(
                    scores, torch.full((count,), width, device=index_queries.device, dtype=torch.int32)
                )
                carried = beam.kept_children(
                    scores,
                    kept,
                    open_count,
                    open_node_buckets["children"],
                    carried,
                    closed_node_buckets["children"] if carried is not None else None,
                    size,
                )
        return self.pick_buckets(cache, row, index_queries, visible_counts, carried, pick_top_k, visible)

    def fused_level(
        self,
        index_queries: torch.Tensor,
        closed_node_buckets: Rows,
        carried: torch.Tensor,
        open_count: int,
        beam_width: int,
    ) -> bool:
        """Tell whether a level that carries children runs as one CuTeDSL kernel.

        Args:
            index_queries: Index queries [S, index_dim] bf16.
            closed_node_buckets: The level's closed node buckets, which the carried rows name.
            carried: The children the level above kept [S, C] int32.
            open_count: The level's open node buckets, scored beside the carried children.
            beam_width: Columns a query keeps.
        """
        return dispatch.runs_cutedsl(
            "beam step", self.kernels, index_queries.device
        ) and cutedsl_beam.supports(
            index_queries,
            closed_node_buckets["node_centroids"],
            carried,
            open_count,
            beam_width,
            None if self.node_shape is None else self.node_shape.rank,
        )

    def kept_in_cutedsl(self, index_queries: torch.Tensor, scores: torch.Tensor, beam_width: int) -> bool:
        """Tell whether a start-level keep and children run as one CuTeDSL kernel over the scores in hand.

        Args:
            index_queries: Index queries [S, index_dim] bf16.
            scores: The level's scores [S, N] fp32, its node buckets' children in column order.
            beam_width: Columns a query keeps.
        """
        return dispatch.runs_cutedsl(
            "start-level keep", self.kernels, index_queries.device
        ) and cutedsl_beam.supports_keep(index_queries, scores.shape[1], beam_width)

    def picked_in_cutedsl(
        self,
        cache: "BucketCache",
        index_queries: torch.Tensor,
        carried: torch.Tensor | None,
        width: int,
        top_p: int,
    ) -> bool:
        """Tell whether the last level's scoring of the beam's children and its pick run as one CuTeDSL kernel.

        Args:
            cache: The bucket cache - its buckets unshaped, since the kernel adds no spread.
            index_queries: Index queries [S, index_dim] bf16.
            carried: The cache rows the last beam kept [S, C] int32, or None.
            width: Columns a query scores - its uncovered buckets and the carried rows.
            top_p: Buckets a query picks.
        """
        carried_rows = (
            torch.empty(len(index_queries), 0, device=index_queries.device, dtype=torch.int32)
            if carried is None
            else carried
        )
        return (
            dispatch.runs_cutedsl("top-p pick", self.kernels, index_queries.device)
            and cache.bucket_shapes is None
            and cutedsl_beam.supports(
                index_queries,
                cache.bucket_centroids,
                carried_rows,
                width - carried_rows.shape[1],
                min(top_p, width),
            )
        )

    def fused_beam(
        self,
        index_queries: torch.Tensor,
        node_centroids: torch.Tensor,
        children: torch.Tensor,
        select: ScanSelect,
    ) -> torch.Tensor:
        """Return each query's kept node buckets' children [S, beam_width * node_bucket_size] int32, -1 padded.

        Args:
            index_queries: Index queries [S, index_dim] bf16.
            node_centroids: The level's node centroids [N, index_dim], bf16 or fp8 at `NODE_CENTROID_SCALE`.
            children: Their children [N, node_bucket_size] int32.
            select: The fused select, as wide as the level's beam.
        """
        if node_centroids.dtype != torch.bfloat16:
            node_centroids = node_centroids.to(torch.bfloat16) * (1 / TritonBeam.NODE_CENTROID_SCALE)
        counts = torch.full(
            (len(index_queries),), len(node_centroids), device=index_queries.device, dtype=torch.int32
        )
        return beam.children_of(select(index_queries, node_centroids, counts), children)

    def node_bucket_scores(
        self, index_queries: torch.Tensor, node_buckets: Rows | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Every query's score of a level's node buckets [S, N] fp32 - their estimated best key when they are shaped."""
        node_centroids = node_buckets["node_centroids"]
        if node_centroids.dtype != torch.bfloat16:
            # A power of two - exact in bf16
            node_centroids = node_centroids.to(torch.bfloat16) * (1 / TritonBeam.NODE_CENTROID_SCALE)
        scores = torch.mm(index_queries, node_centroids.t(), out_dtype=torch.float32)
        if self.node_shape is not None:
            add_spread(
                scores,
                index_queries,
                node_buckets["basis"],
                node_buckets["variances"],
                node_buckets["floor"],
                self.node_shape.deviations,
            )
        return scores

    def child_scores(
        self, index_queries: torch.Tensor, closed_node_buckets: Rows, carried: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Every query's score of the children of the node buckets it kept a level up, into `out` [S, carried]."""
        scale = TritonBeam.NODE_CENTROID_SCALE if self.fp8 else 1.0
        if self.node_shape is None:
            beam.score_children(
                index_queries, closed_node_buckets["node_centroids"], carried, out=out, centroid_scale=scale
            )
            return
        score_shaped_rows(
            index_queries,
            closed_node_buckets["node_centroids"],
            closed_node_buckets["basis"],
            closed_node_buckets["variances"],
            closed_node_buckets["floor"],
            carried,
            self.node_shape.deviations,
            out=out,
            centroid_scale=scale,
        )

    def pick_buckets(
        self,
        cache: "BucketCache",
        row: int,
        index_queries: torch.Tensor,
        visible_counts: torch.Tensor,
        carried: torch.Tensor | None,
        pick_top_k: TopK,
        visible: int | None = None,
    ) -> torch.Tensor:
        """Pick the best buckets of level 0 for each query.

        The candidates are the buckets no node bucket covers within its visible count, and the last beam's
        children.

        Args:
            cache: The bucket cache.
            row: The request's table row.
            index_queries: Index queries [S, index_dim] bf16.
            visible_counts: Buckets each query sees [S] int32.
            carried: Cache rows the last beam kept, scored after the uncovered buckets [S, carried] int32.
            pick_top_k: The top-k that lists the picks.
            visible: Buckets the request shows - the cache's count when None.

        Returns:
            The picked scores packed with their cache rows [S, W] int64.
        """
        count = len(index_queries)
        uncovered = max((cache.visible(row) if visible is None else visible) - self.covered_buckets, 0)
        bucket_centroids = cache.gather(row, uncovered, first=self.covered_buckets)
        shapes = cache.gather_shapes(row, uncovered, first=self.covered_buckets)
        open_count = bucket_centroids.shape[0]
        width = open_count + (0 if carried is None else carried.shape[1])
        with probe.region("index q scan"):
            own = torch.mm(index_queries, bucket_centroids.t(), out_dtype=torch.float32)
            if shapes is not None:
                add_planes_spread(own, index_queries, shapes)
            columns = torch.arange(open_count, device=index_queries.device)
            seen = (columns[None] < (visible_counts.long() - self.covered_buckets)[:, None]) & (
                columns[None] < uncovered
            )
            own.masked_fill_(~seen, float("-inf"))
            if self.picked_in_cutedsl(cache, index_queries, carried, width, pick_top_k.k):
                # Scoring the beam's children, the pick and the cache rows in one kernel - the row is never written
                return cutedsl_beam.pick_top_p(
                    index_queries,
                    own,
                    carried,
                    cache.bucket_centroids,
                    cache.request_pages[row],
                    self.covered_buckets,
                    cache.rows_per_page,
                    pick_top_k.k,
                )
            if carried is None:
                scores = own
            else:
                scores = torch.empty(count, width, device=index_queries.device)
                scores[:, :open_count] = own
                if cache.bucket_shapes is None:
                    beam.score_children(
                        index_queries, cache.bucket_centroids, carried, out=scores[:, open_count:]
                    )
                else:
                    bucket_shapes = cache.bucket_shapes
                    score_shaped_rows(
                        index_queries,
                        cache.bucket_centroids,
                        bucket_shapes.basis,
                        bucket_shapes.variances,
                        bucket_shapes.floor,
                        carried,
                        bucket_shapes.deviations,
                        out=scores[:, open_count:],
                    )
        with probe.region("index k select"):
            keys, listed = pick_top_k.listed(
                scores, torch.full((count,), width, device=index_queries.device, dtype=torch.int32)
            )
            bucket_keys = beam.bucket_keys(
                keys,
                listed,
                carried,
                cache.request_pages[row],
                open_count,
                self.covered_buckets,
                cache.rows_per_page,
            )
            pick_top_k.release()
            return bucket_keys
