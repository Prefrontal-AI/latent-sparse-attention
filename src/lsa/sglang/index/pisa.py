"""PISA's search, batched for serving - every query row of a request's chunk at once, on the means of the closed leaf blocks the layer keeps.

[lsa.baselines.pisa](../../baselines/pisa.py) is the reference, one query and one KV head at a time, and this is the same search
over a tensor of rows: the candidates of a level are a `[rows, width]` block of ids, a level's blocks are scored together, and a
row's keep is one top-k over a priority that puts the forced blocks first.
Both pick the same leaf blocks wherever no two scores tie, which `tests/unit/sglang/test_pisa_index.py` holds them to.

A leaf block is `block_size` keys and its summary their mean.
The layer keeps the means of the blocks a request has closed, one row of them per request table row, and the blocks above
them, pooled again over the leaves that close - a sixty-fourth of the keys, halving at every level.
So a decode step reads the summaries of the blocks it scores and the leaf blocks' keys, never the whole cache.

A block is scored by the LogSumExp, over its children, of the scaled dot of each child with the head's absorbed query, summed over
the heads - GLM's MLA has one latent KV head, so the one list serves every head.
The blocks the search scores lie wholly before the query: a block that holds the query is an ancestor of its current leaf
block, which is forced, and one that starts after it is dropped.
So a summary that covers keys past the query - the open last block - is never read, and a row sees no later key.
"""

import itertools
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from lsa.baselines import pisa as reference
from lsa.core.kernels.triton import block_scores, descend, pyramid
from lsa.sglang.settings.design import PisaConfig

# Bytes of gathered leaf-block keys a tile of rows may hold - bounds the leaf level's scores, which are the search's peak
TILE_BYTES: int = 2 << 30

# Every index's leaf means in this process, which a write that repoints a request's positions reopens
OPEN_MEANS: "weakref.WeakSet[LeafMeans]" = weakref.WeakSet()

# Rows a tile holds where the kernel scores the blocks - bounds the kept keys' ids and their sort, int64 a key a row
GPU_TILE_ROWS: int = 8192

# A served list that differs from the reference's is a near-tie when the reference search, at some level, kept a block
# within this share of the score of the best one it dropped - far more than another summation order moves a score
NEAR_TIE_GAP: float = 1e-3

# What a masked position is sorted by, past every key id
NO_KEY: int = 1 << 40

# Scores a level's candidates across the ranks that hold a share of the heads, or None where a rank holds them all
ScoreReduce = Callable[[torch.Tensor], torch.Tensor] | None


@dataclass(frozen=True)
class LeafKeys:
    """Where a search reads a leaf block's keys - each request's row of the request table, and the pool it points into.

    Attributes:
        table: The request table, each request's positions to KV slots [table_rows, context] int32.
        row_index: Each request's table row [requests] int64.
        keys: The pool's keys of the head the search selects for [slots, key_dim], any slot stride.
    """

    table: torch.Tensor
    row_index: torch.Tensor
    keys: torch.Tensor

    def read(self, requests: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Read keys by position.

        Args:
            requests: Each row's request, an index into `row_index` [n] int64.
            positions: Key positions of each row [n, m] int64.

        Returns:
            The keys [n, m, key_dim].
        """
        return self.keys[self.table[self.row_index[requests][:, None], positions].long()]


def pool_means(keys: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool whole blocks of keys.

    Args:
        keys: Keys [blocks * block_size, key_dim], any float dtype.
        block_size: Keys in a block.

    Returns:
        The blocks' means [blocks, key_dim] fp32.
    """
    return keys.view(-1, block_size, keys.shape[-1]).float().mean(1)


def level_widths(leaf_count: int, branching: int) -> list[int]:
    """Return the blocks of every level of a pyramid over some leaf blocks, the leaves first, up to a single root.

    Args:
        leaf_count: Leaf blocks.
        branching: Children of a block above the leaves.
    """
    widths = [leaf_count]
    while widths[-1] > 1:
        widths.append(math.ceil(widths[-1] / branching))
    return widths


def pool_parents(
    below: torch.Tensor, first: int, stop: int, width_below: int, branching: int
) -> torch.Tensor:
    """Pool blocks `[first, stop)` of a level as the mean of the children each has.

    Args:
        below: Each request's summaries of the level below [requests, block_count, key_dim], any float dtype.
        first: The first block pooled.
        stop: One past the last.
        width_below: Blocks the level below holds, its children past that absent.
        branching: Children of a block.

    Returns:
        The blocks' summaries [requests, stop - first, key_dim] fp32.
    """
    children = below[:, first * branching : min(stop * branching, width_below)].float()
    padded = torch.zeros(below.shape[0], (stop - first) * branching, below.shape[2], device=below.device)
    padded[:, : children.shape[1]] = children
    counts = (width_below - torch.arange(first, stop, device=below.device) * branching).clamp(max=branching)
    return padded.view(below.shape[0], stop - first, branching, -1).sum(2) / counts[None, :, None]


def build_levels(leaf_means: torch.Tensor, branching: int) -> list[torch.Tensor]:
    """Pool the leaf blocks' means into the blocks above them, up to a single root.

    A parent is the mean of the children it has, so a ragged last block averages what exists.
    A block that holds no real leaf is never scored, so the zeros a request's unused leaves hold change nothing.

    Args:
        leaf_means: Each request's leaf means [batch_size, leaf_count, key_dim], any float dtype.
        branching: Children of a block above the leaves.

    Returns:
        The summaries of each level [batch_size, block_count, key_dim] fp32, the leaf blocks first, up to the root.
    """
    levels = [leaf_means.float()]
    for width in level_widths(leaf_means.shape[1], branching)[1:]:
        levels.append(pool_parents(levels[-1], 0, width, levels[-1].shape[1], branching))
    return levels


def head_dots(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Dot every head's query with every key of its row, accumulated in fp32.

    Args:
        queries: Each row's heads' queries [rows, head_count, key_dim].
        keys: Each row's keys [rows, key_count, key_dim].

    Returns:
        The dots [rows, head_count, key_count] fp32.
    """
    if queries.is_cuda and queries.dtype == keys.dtype and queries.dtype != torch.float32:
        return torch.bmm(queries, keys.transpose(1, 2), out_dtype=torch.float32)
    return torch.bmm(queries.float(), keys.float().transpose(1, 2))


def score_blocks(
    queries: torch.Tensor,
    children: torch.Tensor,
    child_valid: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """Score blocks by the LogSumExp of the query's dots with their children, summed over the heads.

    Args:
        queries: Each row's heads' queries [rows, head_count, key_dim].
        children: The children of each row's candidate blocks, summaries or keys [rows, width, child_count, key_dim].
        child_valid: Which children exist and may be read [rows, width, child_count] bool.
        logit_scale: What the dots are multiplied by.

    Returns:
        The blocks' scores [rows, width] fp32.
    """
    row_count, width, child_count, key_dim = children.shape
    logits = head_dots(queries, children.view(row_count, width * child_count, key_dim)) * logit_scale
    logits = logits.view(row_count, -1, width, child_count).masked_fill(~child_valid[:, None], -torch.inf)
    return logits.logsumexp(-1).sum(1)


def forced_blocks(cand: torch.Tensor, current: torch.Tensor, blocks_below: int) -> torch.Tensor:
    """Mark the candidates that hold the first, previous or current leaf block.

    Args:
        cand: Each row's candidate blocks of a level [rows, width] int64.
        current: Each row's current leaf block [rows] int64.
        blocks_below: Leaf blocks a block of this level holds.

    Returns:
        Which candidates are forced [rows, width] bool.
    """
    previous = (current - 1) // blocks_below
    ancestors = torch.stack((torch.zeros_like(previous), previous, current // blocks_below), dim=1)
    # A first row has no previous block, which `// ` floors to -1 and no candidate matches
    return (cand[:, :, None] == ancestors[:, None, :]).any(-1)


def search_tile(
    levels: list[torch.Tensor],
    leaf_count: int,
    leaf_keys: "LeafKeys",
    queries: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    config: PisaConfig,
    logit_scale: float,
    reduce_scores: ScoreReduce = None,
) -> torch.Tensor:
    """Search the pyramid for a tile of query rows, coarse to fine.

    Args:
        levels: The summaries of each level [table_rows, block_count, key_dim], the leaf blocks first, a request's
            row its `leaf_keys.row_index`.
        leaf_count: Leaf blocks the pyramid spans, at least every request's closed ones - a level's blocks past
            what that spans are not read.
        leaf_keys: Where each request's keys are read by position.
        queries: The rows' heads' queries [rows, head_count, key_dim].
        requests: Each row's index into the batch [rows] int64.
        positions: Each row's absolute position [rows] int64.
        config: The pyramid and the budget.
        logit_scale: What the dots are multiplied by.
        reduce_scores: Sums a level's scores over the ranks that hold the other heads, or None.

    Returns:
        Key positions of the kept leaf blocks [rows, keep_count * block_size] int32, ascending, -1 padded.
    """
    row_count, device = positions.shape[0], positions.device
    block_size, branching = config.block_size, config.branching
    current = positions // block_size
    candidates = torch.zeros(row_count, 1, dtype=torch.long, device=device)
    candidate_valid = torch.ones(row_count, 1, dtype=torch.bool, device=device)
    widths = level_widths(leaf_count, branching)
    for level in range(len(widths), 0, -1):
        keep = config.keep_count if level == 1 else config.beam
        blocks_below = branching ** (level - 1)
        # Every row is scored, though a row under its keep keeps all its eligible blocks whatever they score
        # => no host sync to pick the rows that need it, so a decode step can be captured
        scores = level_scores(
            level, levels, widths, leaf_keys, queries, requests, positions, candidates, config, logit_scale
        )
        if reduce_scores is not None:
            scores = reduce_scores(scores)
        if queries.is_cuda:
            candidates, candidate_valid = descend.keep(
                scores,
                candidates,
                candidate_valid,
                positions,
                keep,
                blocks_below,
                block_size,
                branching,
                config.forced_blocks,
            )
            continue
        # Keys a block of this level spans => candidates starting after the query are out
        eligible = candidate_valid & (candidates * block_size * blocks_below <= positions[:, None])
        # Forced blocks first, then the best free ones => one top-k keeps both, and a row under its keep keeps all
        priority = scores.masked_fill(~eligible, -torch.inf)
        if config.forced_blocks:
            priority = priority.masked_fill(
                eligible & forced_blocks(candidates, current, blocks_below), torch.inf
            )
        top = priority.topk(min(keep, candidates.shape[1]), dim=1)
        candidates = candidates.gather(1, top.indices)
        candidate_valid = top.values > -torch.inf
        if level > 1:
            children = candidates[:, :, None] * branching + torch.arange(branching, device=device)
            candidates = children.flatten(1)
            candidate_valid = candidate_valid[:, :, None].expand(-1, -1, branching).flatten(1)

    key_ids = candidates[:, :, None] * block_size + torch.arange(block_size, device=device)
    key_valid = candidate_valid[:, :, None] & (key_ids <= positions[:, None, None])
    # The kernel keeps the leaf blocks ascending, and only the last can hold keys past the query
    # => its keys are listed in order already
    ascending = key_ids.masked_fill(~key_valid, NO_KEY).flatten(1)
    if not queries.is_cuda:
        ascending = ascending.sort(1).values
    selected = ascending.masked_fill(ascending == NO_KEY, -1).int()
    return torch.nn.functional.pad(selected, (0, config.width - selected.shape[1]), value=-1)


def level_scores(
    level: int,
    levels: list[torch.Tensor],
    widths: list[int],
    leaf_keys: "LeafKeys",
    queries: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    candidates: torch.Tensor,
    config: PisaConfig,
    logit_scale: float,
) -> torch.Tensor:
    """Score every candidate block of one level by its children, for the rows that need it.

    Args:
        level: The level, 1 for the leaf blocks.
        levels: The summaries of each level [table_rows, block_count, key_dim], the leaf blocks first.
        widths: Blocks of each level the search spans, the leaf blocks first.
        leaf_keys: Where each request's keys and summaries are read.
        queries: The rows' heads' queries [rows, head_count, key_dim].
        requests: Each row's index into the batch [rows] int64.
        positions: Each row's absolute position [rows] int64.
        candidates: Each row's candidate blocks [rows, width] int64.
        config: The pyramid and the budget.
        logit_scale: What the dots are multiplied by.

    Returns:
        The candidates' scores [rows, width] fp32, meaningless for a candidate that is not eligible.
    """
    children_per_block = config.block_size if level == 1 else config.branching
    if queries.is_cuda and block_scores.COLUMNS % children_per_block == 0:
        if level == 1:
            return block_scores.block_scores(
                queries,
                candidates,
                requests,
                positions,
                leaf_keys.keys,
                children_per_block,
                logit_scale,
                leaf_keys.table,
                leaf_keys.row_index,
            )
        return block_scores.block_scores(
            queries,
            candidates,
            requests,
            positions,
            levels[level - 2],
            children_per_block,
            logit_scale,
            row_index=leaf_keys.row_index,
            child_limit=widths[level - 2],
        )
    children = candidates[:, :, None] * children_per_block + torch.arange(
        children_per_block, device=candidates.device
    )
    if level == 1:
        # A key past the query is masked, and read from the query's own position meanwhile
        valid = children <= positions[:, None, None]
        keys = leaf_keys.read(requests, children.minimum(positions[:, None, None]).flatten(1))
        keys = keys.view(*children.shape, -1)
        return score_blocks(queries, keys, valid, logit_scale)
    width = widths[level - 2]
    valid = children < width
    keys = levels[level - 2][
        leaf_keys.row_index[requests][:, None, None], children.clamp(max=width - 1)
    ].float()
    return score_blocks(queries.float(), keys, valid, logit_scale)


def select(
    levels: list[torch.Tensor],
    leaf_count: int,
    leaf_keys: "LeafKeys",
    queries: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    config: PisaConfig,
    logit_scale: float,
    reduce_scores: ScoreReduce = None,
) -> torch.Tensor:
    """Select the leaf blocks each query row reads, a tile of rows at a time.

    On a GPU the kernel reads the keys where they sit, so a tile is bounded by the kept keys alone.

    Args:
        levels: The summaries of each level [table_rows, block_count, key_dim], the leaf blocks first, a request's
            row its `leaf_keys.row_index`.
        leaf_count: Leaf blocks the pyramid spans, at least every request's closed ones.
        leaf_keys: Where each request's keys are read by position.
        queries: The rows' heads' queries [rows, ..., key_dim], every head in the middle dims.
        requests: Each row's index into the batch [rows] int64.
        positions: Each row's absolute position [rows] int64.
        config: The pyramid and the budget.
        logit_scale: What the dots are multiplied by, GLM's softmax scale.
        reduce_scores: Sums a level's scores over the ranks that hold the other heads, or None.

    Returns:
        Key positions of the kept leaf blocks [rows, keep_count * block_size] int32, ascending, -1 padded.
    """
    key_dim = queries.shape[-1]
    leaf_bytes = config.beam * config.branching * config.block_size * key_dim * queries.element_size()
    tile = GPU_TILE_ROWS if queries.is_cuda else max(1, TILE_BYTES // leaf_bytes)
    selected = [
        search_tile(
            levels,
            leaf_count,
            leaf_keys,
            queries[start : start + tile].reshape(-1, math.prod(queries.shape[1:-1]), key_dim),
            requests[start : start + tile],
            positions[start : start + tile],
            config,
            logit_scale,
            reduce_scores,
        )
        for start in range(0, positions.shape[0], tile)
    ]
    if not selected:
        return torch.empty(0, config.width, dtype=torch.int32, device=positions.device)
    return torch.cat(selected)


@dataclass(frozen=True)
class RowShare:
    """How a prefill's rows are shared across the ranks that each hold a share of the heads.

    Every rank searches a share of the rows with every head's queries and the lists are gathered, rather than every rank
    searching every row with its own heads and summing the scores - a row's key reads and its keep are then made once,
    not once a rank.

    Attributes:
        rank: This rank's place in the group, whose share of the rows it searches.
        size: Ranks in the group.
        exchange: Hands each rank its share of the rows with every rank's heads - this rank's heads of every row
            [S, heads_local, key_dim], S a multiple of `size`, to its rows' heads, rank by rank [S / size, size * heads_local, key_dim].
        gather: Every rank's lists, rank by rank - this rank's [S / size, width] to the group's [S, width].
    """

    rank: int
    size: int
    exchange: Callable[[torch.Tensor], torch.Tensor]
    gather: Callable[[torch.Tensor], torch.Tensor]


class CheckedRow(NamedTuple):
    """One served list held against the reference's.

    Attributes:
        position: The row's position.
        both: Keys both lists hold.
        served: Keys the served list holds.
        wanted: Keys the reference's holds.
        gap: The reference search's nearest tie, its smallest boundary gap over the levels.
    """

    position: int
    both: int
    served: int
    wanted: int
    gap: float

    def verdict(self) -> str:
        """Say whether the lists are the same, differ where the reference nearly tied, or differ outright."""
        if self.both == self.served == self.wanted:
            return "exact"
        return "near-tie" if self.gap < NEAR_TIE_GAP else "MISMATCH"


class CheckTally:
    """Every served list a debugging run held against the reference, and whether the ranks agreed."""

    def __init__(self) -> None:
        """Start with nothing counted."""
        self.counts = {"exact": 0, "near-tie": 0, "MISMATCH": 0}
        self.rank_checks = 0
        self.rank_differences = 0
        self.left_batch = 0

    def add(self, row: CheckedRow) -> str:
        """Count a row and return its verdict."""
        verdict = row.verdict()
        self.counts[verdict] += 1
        return verdict

    def add_ranks(self, same: bool) -> None:
        """Count one comparison of the ranks' lists."""
        self.rank_checks += 1
        self.rank_differences += int(not same)

    def line(self) -> str:
        """Return the run's tally so far and its verdict - FAIL on any mismatch or any rank that differed."""
        failed = self.counts["MISMATCH"] or self.rank_differences
        return (
            f"lsa-lsa: pisa check tally rows {sum(self.counts.values())} exact {self.counts['exact']} "
            f"near-tie {self.counts['near-tie']} mismatch {self.counts['MISMATCH']} "
            f"rank-checks {self.rank_checks} rank-differences {self.rank_differences} left-batch {self.left_batch} "
            f"near-tie-gap {NEAR_TIE_GAP:g} "
            f"verdict {'FAIL' if failed else 'PASS'}"
        )


# The served run's tally, shared by every layer's checks
TALLY = CheckTally()

# Requests of a batch a check holds, the batch's first
CHECKED_REQUESTS: int = 4


def first_rank() -> bool:
    """Say whether this process prints a check's lines - the first rank, or the only one."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def report_check(label: str, rows: Sequence[CheckedRow]) -> None:
    """Count checked rows in the run's tally, and print each and the tally so far from the first rank.

    Args:
        label: What was checked, the layer and head and whether the step was captured.
        rows: The rows held against the reference.
    """
    verdicts = [TALLY.add(row) for row in rows]
    if not first_rank():
        return
    for row, verdict in zip(rows, verdicts, strict=True):
        print(
            f"lsa-lsa: pisa check {label} position {row.position} - {row.both} of the reference's {row.wanted} keys, "
            f"served {row.served}, {verdict} at gap {row.gap:.2e}",
            flush=True,
        )
    print(TALLY.line(), flush=True)


def report_left(layer: int, positions: Sequence[int]) -> None:
    """Count captured rows left unchecked because their request left the batch, and say so from the first rank.

    Args:
        layer: The layer whose record held them.
        positions: The rows' positions.
    """
    TALLY.left_batch += len(positions)
    if positions and first_rank():
        print(
            f"lsa-lsa: pisa check layer {layer} captured decode skipped positions {list(positions)} - "
            "the request left the batch, so its table row may hold another's",
            flush=True,
        )


def rows_by_request(
    table_rows: Sequence[int], key_counts: Sequence[int]
) -> list[tuple[tuple[int, int], list[int]]]:
    """Group checked rows by their request, so the reference builds each request's pyramid once.

    Args:
        table_rows: Each checked row's request table row.
        key_counts: Keys each checked row's request holds.

    Returns:
        Each request's table row and key count, with the indices of its rows, in first-seen order.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for at, request in enumerate(zip(table_rows, key_counts, strict=True)):
        groups.setdefault(request, []).append(at)
    return list(groups.items())


def check_rows_of(
    lengths: Sequence[int], key_counts: Sequence[int], rows_each: int, start_from: int
) -> list[int]:
    """Pick the rows a check holds - spread over each of the batch's first requests that holds enough keys.

    Args:
        lengths: Query rows each request has in the batch, end to end.
        key_counts: Keys each request holds in the pool.
        rows_each: Rows to hold of each request.
        start_from: Keys a request must hold to be checked.

    Returns:
        The rows' indices into the batch.
    """
    picked, start = [], 0
    for length, key_count in list(zip(lengths, key_counts, strict=False))[:CHECKED_REQUESTS]:
        if key_count >= start_from and length:
            spread = torch.linspace(0, length - 1, min(rows_each, length)).long().unique()
            picked += (start + spread).tolist()
        start += length
    return picked


class DecodeRecord:
    """A captured decode step's first rows, copied inside the graph so the step can be held against the reference after it ran.

    A replay runs no Python, so these copies are the only way to see what a captured step selected. The buffers are
    allocated by the eager warm-up a capture runs first, and a record is taken only when its rows are the ones the
    batch that was replayed was told to hold - the warm-up's batch is no request's.
    A record is read at the next forward, and a request that finished on the recorded step may have handed its table
    row to another by then, so only the rows of requests still in the batch are held to the reference.
    """

    def __init__(self, row_count: int = 4) -> None:
        """Start with no buffers and nothing to take.

        Args:
            row_count: Rows of a step it copies, the batch's first.
        """
        self.row_count = row_count
        self.buffers: tuple[torch.Tensor, ...] | None = None
        self.expected_rows: list[int] = []
        self.expected_rids: list[object] = []
        self.taken_positions: list[int] = []

    def record(self, rows: torch.Tensor, positions: torch.Tensor, *tensors: torch.Tensor) -> None:
        """Copy the first rows of a decode step's inputs and lists, when the step is being captured.

        Args:
            rows: Each row's request table row [S].
            positions: Each row's position [S].
            *tensors: The queries, lists or anything else the check reads, each one row a request first.
        """
        if not positions.is_cuda:
            return
        firsts = tuple(tensor[: self.row_count] for tensor in (rows, positions, *tensors))
        # Allocated once at the full row count, since every captured batch size writes into the same buffers
        if self.buffers is None:
            self.buffers = tuple(
                torch.zeros(self.row_count, *first.shape[1:], dtype=first.dtype, device=first.device)
                for first in firsts
            )
        if torch.cuda.is_current_stream_capturing():
            for buffer, first in zip(self.buffers, firsts, strict=True):
                buffer[: first.shape[0]].copy_(first)

    def expect(self, rows: Sequence[int], rids: Sequence[object]) -> None:
        """Say which requests the batch about to be replayed puts first, and at which table rows - none for a batch that is not one.

        Args:
            rows: The batch's requests' table rows.
            rids: The batch's request ids, in the same order.
        """
        self.expected_rows = list(rows[: self.row_count])
        self.expected_rids = list(rids[: len(self.expected_rows)])

    def take(
        self, rows_now: Sequence[int], rids_now: Sequence[object]
    ) -> tuple[list[int], list[int], list[bool], tuple[torch.Tensor, ...]] | None:
        """Return the last replayed step's first rows - their table rows, positions, whether each request is still in the batch, and the rest - once, or None.

        Args:
            rows_now: The table rows of the batch about to be forwarded.
            rids_now: Its request ids, in the same order.

        Returns:
            The rows, the positions, whether each row's request still holds its row, and the other recorded tensors
            cut to them, or None when there is no new replayed step.
        """
        count = len(self.expected_rows)
        if self.buffers is None or not count:
            return None
        rows, positions = self.buffers[0][:count].tolist(), self.buffers[1][:count].tolist()
        if rows != self.expected_rows or positions == self.taken_positions:
            return None
        self.taken_positions = positions
        holding = dict(zip(rows_now, rids_now, strict=False))
        current = [holding.get(row) == rid for row, rid in zip(rows, self.expected_rids, strict=True)]
        return rows, positions, current, tuple(buffer[:count] for buffer in self.buffers[2:])


def reopen_repointed(table_pool: Any, indices: Any, values: Any) -> None:
    """Reopen every index's closed blocks whose positions a request table write points at other KV slots.

    The prefix cache repoints a request's positions at the slots another request holds for the same tokens, once a
    chunk is cached - keys of the same tokens from another forward, an ulp or so away.
    A mean pooled from the old slots would then summarize keys attention no longer reads.
    A write of positions past the closed blocks, as every new token's is, changes no mean and costs no comparison.

    Args:
        table_pool: sglang's request-to-token pool, whose table is about to be written.
        indices: The written entries - a request table row and a slice of its positions where a mean can change.
        values: The KV slots being written there.
    """
    if not OPEN_MEANS or not isinstance(indices, tuple) or len(indices) != 2:
        return
    row, span = indices
    if not isinstance(row, int) or not isinstance(span, slice):
        return
    holding = [means for means in OPEN_MEANS if row < len(means.closed)]
    start = span.start or 0
    closed_keys = max((means.closed[row] * means.block_size for means in holding), default=0)
    stop = min(span.stop, closed_keys, start + len(values))
    if stop <= start:
        return
    old = table_pool.req_to_token[row, start:stop]
    changed = (old != values[: stop - start].to(old.device, old.dtype)).nonzero()
    if changed.numel():
        first = start + int(changed[0, 0])
        for means in holding:
            means.reopen(row, first)


class LeafMeans:
    """The means of the leaf blocks each request has closed, one row of them per request table row, and the pyramid above them.

    The pyramid spans the longest context, and a summary is pooled again whenever a leaf under it closes or reopens.
    A summary over a leaf a request has not closed holds whatever that leaf held, which no search reads: such a
    block holds the query's current leaf block, which is forced, or starts after the query, which is dropped.
    A narrower pyramid's blocks are the same summaries but its ragged last ones, which hold the last leaf it spans.
    """

    def __init__(
        self,
        request_count: int,
        max_context: int,
        key_dim: int,
        block_size: int,
        branching: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        """Allocate the means and the pyramid.

        Args:
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
            key_dim: Dims of a key.
            block_size: Keys in a leaf block.
            branching: Children of a block above the leaves.
            device: Where the means live.
            dtype: What the leaf means are stored as - the blocks above are fp32.
        """
        self.block_size = block_size
        self.branching = branching
        widths = level_widths(math.ceil(max_context / block_size), branching)
        self.means = torch.zeros(request_count, widths[0], key_dim, device=device, dtype=dtype)
        self.upper = torch.zeros(request_count, sum(widths[1:]), key_dim, device=device)
        offsets = [0, *itertools.accumulate(widths[1:-1])] if len(widths) > 1 else []
        self.levels = [self.means] + [
            self.upper[:, offset : offset + width] for offset, width in zip(offsets, widths[1:], strict=True)
        ]
        self.widths = torch.tensor(widths, device=device)
        self.offsets = torch.tensor([0, *offsets], device=device)
        self.closed = [0] * request_count
        self.rids: list[object] = [None] * request_count
        OPEN_MEANS.add(self)

    def ensure(self, row: int, rid: object) -> None:
        """Forget a row's means when another request has taken it.

        Args:
            row: The request table row.
            rid: The request occupying it.
        """
        if self.rids[row] != rid:
            self.rids[row] = rid
            self.closed[row] = 0

    def reopen(self, row: int, position: int) -> None:
        """Reopen a row's blocks from the one holding a position, so the next call pools them again from the pool.

        Args:
            row: The request table row.
            position: The first position whose KV slot changed.
        """
        self.closed[row] = min(self.closed[row], position // self.block_size)

    def advance(self, row: int, key_count: int, keys_at: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Close the blocks the request's keys have completed since the last call.

        Args:
            row: The request table row.
            key_count: Keys the request holds in the pool.
            keys_at: Reads keys - positions [n] to keys [n, key_dim].
        """
        target = key_count // self.block_size
        first = self.closed[row]
        if target <= first:
            return
        positions = torch.arange(first * self.block_size, target * self.block_size, device=self.means.device)
        self.means[row, first:target] = pool_means(keys_at(positions), self.block_size).to(self.means.dtype)
        self.closed[row] = target
        self.lift(row, first, target)

    def lift(self, row: int, first: int, stop: int) -> None:
        """Pool again every summary above a row's leaves `[first, stop)`.

        Args:
            row: The request table row.
            first: The first leaf whose mean changed.
            stop: One past the last.
        """
        if self.means.is_cuda:
            pyramid.lift(self.means, self.upper, self.widths, self.offsets, row, first, stop, self.branching)
            return
        for below, level in itertools.pairwise(self.levels):
            first, stop = first // self.branching, math.ceil(stop / self.branching)
            level[row : row + 1, first:stop] = pool_parents(
                below[row : row + 1], first, stop, below.shape[1], self.branching
            )


class PisaIndex:
    """One layer's PISA selection for one key-value head, over the keys the KV pool holds and the means of its closed blocks -
    no engine in sight.

    A model hands in the slots its tokens were given and the request table, and gets back lists of KV slots.
    """

    def __init__(
        self,
        config: PisaConfig,
        key_dim: int,
        request_count: int,
        max_context: int,
        slot_table: Callable[[], torch.Tensor],
        key_buffer: Callable[[], torch.Tensor],
        device: torch.device | str,
        mean_dtype: torch.dtype = torch.float32,
        head: int = 0,
        reduce_scores: ScoreReduce = None,
        share_rows: RowShare | None = None,
    ) -> None:
        """Hold the way to the pool and allocate the leaf means.

        Args:
            config: The pyramid and the budget.
            key_dim: Dims of a key, which the queries share - GLM's key latent, or one of Qwen's key heads.
            request_count: Rows of the request table.
            max_context: Longest context a request holds.
            slot_table: Returns the request table, each request's positions to KV slots [requests, context] int32.
            key_buffer: Returns the layer's KV pool buffer [slots, Hkv, key_dim or more].
            device: Where everything lives.
            mean_dtype: What the leaf means are stored as - fp32 as the reference pools them, since bf16 means move
                a block's score by enough to swap a block where the search nearly ties.
            head: The key-value head of the pool this index selects for - GLM's one latent, or one of Qwen's.
            reduce_scores: Sums a level's scores over the ranks that hold the other heads, or None where the queries
                a search is handed are every head's.
            share_rows: How a prefill's rows are shared across those ranks, or None where every rank searches every
                row - a decode step's few rows always are, with `reduce_scores`.

        Raises:
            ValueError: The design does not force the first, previous and current blocks - the summaries of the open
                block and of the leaves past a request are only never read because the blocks above them are forced.
        """
        if not config.forced_blocks:
            raise ValueError(
                "served PISA forces the first, previous and current blocks - got forced_blocks false, which would score "
                "the open block's summary"
            )
        self.config = config
        self.key_dim = key_dim
        self.head = head
        self.slot_table = slot_table
        self.key_buffer = key_buffer
        self.reduce_scores = reduce_scores
        self.share_rows = share_rows
        self.leaf_means = LeafMeans(
            request_count, max_context, key_dim, config.block_size, config.branching, device, mean_dtype
        )

    def maintain(self, rids: Sequence[object], rows: Sequence[int], ends: Sequence[int]) -> None:
        """Hand a request that took a row its own means, and close the blocks of the keys already in the pool, before a forward.

        A decode step's own key closes at most its current block, whose summary and every one above it are forced,
        so closing up to the keys before the step leaves a captured decode nothing to catch up on.

        Args:
            rids: The batch's request ids.
            rows: Their request table rows.
            ends: Keys each request holds in the pool before this forward's own.
        """
        for rid, row, end in zip(rids, rows, ends, strict=True):
            self.leaf_means.ensure(row, rid)
            self.leaf_means.advance(row, end, self.keys_of(row))

    def write(self, slots: torch.Tensor, key: torch.Tensor) -> None:
        """Write keys at their KV slots, which attention writes again after the indexer.

        Args:
            slots: KV slots [S] int64.
            key: Their key latents [S, key_dim] bf16.
        """
        self.key_buffer()[slots, self.head, : self.key_dim] = key

    def keys_at_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """Read the key latents at KV slots.

        Args:
            slots: KV slots, any shape.

        Returns:
            The latents [*slots.shape, key_dim] bf16.
        """
        return self.key_buffer()[slots, self.head, : self.key_dim]

    def keys_of(self, row: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the reader of one request's keys by position.

        Args:
            row: The request table row.

        Returns:
            A function from positions [n] to key latents [n, key_dim] bf16, read through the request table.
        """
        table = self.slot_table()

        def keys_at(positions: torch.Tensor) -> torch.Tensor:
            return self.keys_at_slots(table[row, positions].long())

        return keys_at

    def select(
        self,
        queries: torch.Tensor,
        positions: torch.Tensor,
        rows: Sequence[int],
        key_counts: Sequence[int],
        lengths: Sequence[int],
        logit_scale: float,
    ) -> torch.Tensor:
        """Select for a batch's query rows, whose keys are all in the pool.

        Args:
            queries: Every row's heads' queries [S, ..., key_dim], the requests' rows end to end.
            positions: Each row's position in its request [S] int.
            rows: The requests' table rows.
            key_counts: Keys each request holds in the pool, the batch's own included.
            lengths: Query rows each request has in the batch.
            logit_scale: What the dots are multiplied by, GLM's softmax scale.

        Returns:
            KV slots [S, keep_count * block_size] int32, valid entries first, then -1.
        """
        device = self.slot_table().device
        for row, key_count in zip(rows, key_counts, strict=True):
            self.leaf_means.advance(row, key_count, self.keys_of(row))
        requests = torch.repeat_interleave(
            torch.arange(len(rows), device=device),
            torch.tensor(list(lengths), device=device),
            output_size=positions.shape[0],
        )
        leaf_count = math.ceil(max(key_counts) / self.config.block_size)
        row_index = torch.tensor(list(rows), device=device)
        share = self.share_rows
        if share is None or share.size == 1:
            return self.search(queries, positions, row_index, requests, leaf_count, logit_scale)
        # Padded to whole shares with rows at the first request's first position, whose lists are dropped
        row_count = positions.shape[0]
        each = math.ceil(row_count / share.size)
        padding = each * share.size - row_count
        queries = torch.nn.functional.pad(
            queries.reshape(row_count, -1, queries.shape[-1]), (0, 0, 0, 0, 0, padding)
        )
        mine = slice(share.rank * each, (share.rank + 1) * each)
        lists = self.search(
            share.exchange(queries),
            torch.nn.functional.pad(positions, (0, padding))[mine],
            row_index,
            torch.nn.functional.pad(requests, (0, padding))[mine],
            leaf_count,
            logit_scale,
            every_head=True,
        )
        return share.gather(lists)[:row_count]

    def decode(
        self, queries: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor, logit_scale: float
    ) -> torch.Tensor:
        """Select for a decode step, a row a request, from device tensors alone - so the step can be captured in a CUDA graph.

        The search spans the request table's whole context, so its shape is the same at every step.
        Leaves past a request's keys only reach blocks that start after its query or hold it, so they change
        nothing it selects. `maintain` has closed every block before each request's current one.

        Args:
            queries: Each row's heads' queries [S, ..., key_dim], one row a request.
            positions: Each row's position in its request [S] int.
            rows: Each request's table row [S] int.
            logit_scale: What the dots are multiplied by, GLM's softmax scale.

        Returns:
            KV slots [S, keep_count * block_size] int32, valid entries first, then -1.
        """
        table = self.slot_table()
        leaf_count = math.ceil(table.shape[1] / self.config.block_size)
        requests = torch.arange(positions.shape[0], device=table.device)
        return self.search(queries, positions, rows.long(), requests, leaf_count, logit_scale)

    def search(
        self,
        queries: torch.Tensor,
        positions: torch.Tensor,
        row_index: torch.Tensor,
        requests: torch.Tensor,
        leaf_count: int,
        logit_scale: float,
        every_head: bool = False,
    ) -> torch.Tensor:
        """Search the pyramid of the requests' closed blocks for every query row, and turn the kept keys into KV slots.

        Args:
            queries: Every row's heads' queries [S, ..., key_dim].
            positions: Each row's position in its request [S] int.
            row_index: Each request's table row [requests] int64.
            requests: Each row's request, an index into `row_index` [S] int64.
            leaf_count: Leaf blocks the search spans, at least every request's closed ones.
            logit_scale: What the dots are multiplied by.
            every_head: Whether the queries are every head's, so no rank's scores are summed in.

        Returns:
            KV slots [S, keep_count * block_size] int32, valid entries first, then -1.
        """
        table = self.slot_table()
        leaf_keys = LeafKeys(table, row_index, self.key_buffer()[:, self.head, : self.key_dim])
        key_ids = select(
            self.leaf_means.levels,
            min(leaf_count, self.leaf_means.means.shape[1]),
            leaf_keys,
            queries,
            requests,
            positions.long(),
            self.config,
            logit_scale,
            None if every_head else self.reduce_scores,
        )
        slots = table[row_index[requests][:, None], key_ids.clamp(min=0).long()]
        return slots.masked_fill(key_ids < 0, -1).int()

    def check(
        self,
        queries: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        row: int,
        key_count: int,
        logit_scale: float,
    ) -> list["CheckedRow"]:
        """Hold a few of a request's served lists against the reference search over the same keys and queries.

        The reference runs on the host, a row at a time, so this is for a debugging run.

        Args:
            queries: The checked rows' heads' queries [rows, ..., key_dim].
            positions: Their positions [rows] int.
            slots: The lists the index served them [rows, width] int32.
            row: The request's table row.
            key_count: Keys the request holds in the pool.
            logit_scale: What the dots are multiplied by.

        Returns:
            Each row's position, the keys both lists hold, the keys each holds, and the reference's nearest tie.
        """
        config = self.config
        table_row = self.slot_table()[row, :key_count].long()
        keys = self.keys_at_slots(table_row).float().cpu()
        selection = reference.select(
            keys[:, None],
            queries.reshape(positions.shape[0], -1, self.key_dim).float().cpu(),
            positions.cpu().long(),
            config.block_size,
            config.keep_count,
            config.branching,
            config.beam_width,
            logit_scale,
            config.forced_blocks,
        )
        expected = selection.selected_key_ids[:, 0]
        gaps = selection.boundary_gaps[:, 0].amin(-1)
        # A served slot back to its position in the request
        order = table_row.argsort()
        served = order[torch.searchsorted(table_row[order], slots.long().clamp(min=0))].cpu()
        served = served.masked_fill(slots.cpu() < 0, -1)
        checked = []
        for position, mine, theirs, gap in zip(
            positions.tolist(), served, expected, gaps.tolist(), strict=True
        ):
            held, wanted = set(mine[mine >= 0].tolist()), set(theirs[theirs >= 0].tolist())
            checked.append(CheckedRow(position, len(held & wanted), len(held), len(wanted), gap))
        return checked


def refuse_rope(config: Any) -> None:
    """Refuse a model whose latent carries rope dims, which the absorbed queries would then carry too.

    Args:
        config: The model's text config.

    Raises:
        ValueError: `qk_rope_head_dim` is not 0 - the search scores the latent alone.
    """
    if int(getattr(config, "qk_rope_head_dim", 0)) != 0:
        raise ValueError(
            f"PISA scores the key latent against the absorbed query and reads no rope dims - "
            f"got qk_rope_head_dim {config.qk_rope_head_dim}"
        )
