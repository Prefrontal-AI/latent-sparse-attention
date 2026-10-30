"""The growth pairing of buckets by the Ward cost in one persistent CuTeDSL kernel, round for round the Triton pairing.

A round finds each open bucket's cheapest open partner and pairs every two that are each other's cheapest; the
rounds stop once a check every eight rounds finds no bucket open, or under a 64th of them paired since the last
one, and the buckets still open pair in index order. The Triton pairing launches three kernels a round and looks
at the counts from the host at every check; here every round runs in one launch.

A block's CTAs each search a share of its open buckets, a tile of 64 against every other on the tensor cores, and
its first CTA - the leader - pairs the mutual ones and lists the rest. The CTAs of a block wait on each other
through a counter in global memory, and the leaders of all the blocks meet at each check, so every CTA must be
resident at once: a block takes at most an SM's share of the GPU.

The cost between two open buckets is the same every round, since pairing moves no key sum. Once 64 or fewer of a
block's buckets are open, its leader scores them against each other once, into shared memory, and every round after
is a scan of that matrix with no other CTA to wait on - the long tail of rounds that pair a bucket or two.

A merge cost is Triton's tf32x3 dot bit for bit: each operand split into its tf32 rounding and the rest, the three
products issued in Triton's order on the same warpgroup MMA, and the square roots approximate as Triton's are. So
every round pairs the same buckets.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as hopper
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.utils import LayoutEnum

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.hopper.tile import (
    INDEX_BITS,
    INDEX_DIM,
    MAX_KEY,
    THREADS,
    TILE,
    WARPS,
    HopperTile,
    accumulator_rows_by_columns,
    approximate_sqrt,
    ordered_cost,
    pack_cost_key,
)

# Open buckets a leader pairs alone from its matrix of costs, a row padded a word against bank conflicts
TAIL = TILE
TAIL_STRIDE = TAIL + 1
# The tail's two warps, whose open counts the last step adds
TAIL_WARPS = TAIL // 32
# Rounds between two checks, as the Triton pairing looks
CHECK = 8
# Blocks a launch takes at most
MAX_BLOCKS = 256
# The workspace's int32 parts, each G words but the last - a block's count of open buckets, whether to stop, the counts
# at the two alternating checks (two parts), the arrival counter of each block's CTAs, and one counter for the leaders
COUNTS, STOPS, CHECKS, ARRIVALS, LEADERS = 0, 1, 2, 4, 5


class NodeBucketClustering(HopperTile):
    """The CuTeDSL node-bucket clustering.

    A round, while a block holds more than a tile of open buckets. Every participating CTA searches its tiles of the
    open list, then arrives at the block's counter and waits for the other participants. The leader - slot 0 - alone
    pairs the mutual buckets, writes the count and the stop flag, and meets the other blocks' leaders every
    `CHECK`th round. Every participant then arrives again, so that none reads the count before the leader wrote it.
    The block's counter therefore rises by the participants twice a round, and `block_arrivals` is what it must reach.

    Counters only rise, and the open count only falls, so the participants of a later round are a prefix of the
    earlier one's. A CTA that finds itself past the participants leaves for good - it never arrives again, and the
    counter is never again asked to count it. The leaders alone go on to the tail, where a block's leader scores its
    last 64 buckets once and pairs them alone. A wide round's check comes between its two arrivals, after its pairing,
    and a tail round's after its pairing too, and both count a round before the test - so the k-th check of every
    block is the same round whichever phase the block is in, and the leaders' counter rises by the blocks at each.

    The workspace, in int32 words for G blocks, is [5G+1]:

    | words | part | written by |
    |---|---|---|
    | `[0, G)` | `COUNTS` - open buckets of each block | the leader, each round |
    | `[G, 2G)` | `STOPS` - whether each block stops | the leader, each round |
    | `[2G, 4G)` | `CHECKS` - open buckets at the two alternating checks | the leader, each check |
    | `[4G, 5G)` | `ARRIVALS` - each block's CTAs' arrival counter | every CTA, twice a round |
    | `5G` | `LEADERS` - the leaders' arrival counter | the leaders, each check |

    The order list is double-buffered: `parity * blocks * buckets` is where a round reads the open buckets, and the
    other half is where it writes those still open.

    Every CTA of a launch spins on these counters in global memory, so every CTA must be resident at once. The
    launch is cooperative, which fails at launch where it cannot be, instead of hanging.

    The leader's matrix of costs in `Storage` is allocated in every CTA, though only the leader uses it - dropping it
    from the others would take a dynamic split of shared memory.
    """

    @cute.jit
    def __call__(
        self,
        key_sums: cute.Tensor,
        square: cute.Tensor,
        alignments: cute.Tensor,
        partner: cute.Tensor,
        order: cute.Tensor,
        cost_keys: cute.Tensor,
        workspace: cute.Tensor,
        buckets: cutlass.Int32,
        blocks: cutlass.Int32,
        ctas_per_block: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Lay out shared memory and the MMA, and launch every block's CTAs at once.

        Args:
            key_sums: Each bucket's key sum [G * P, index_dim] fp32.
            square: Each bucket's squared length [G * P] fp32.
            alignments: Each bucket's length [G * P] fp32, written.
            partner: Each bucket's partner within its block [G * P] int64, -1 on entry.
            order: Two lists of each block's open buckets [2 * G * P] int32, the first `0, 1, ...` on entry.
            cost_keys: Each open bucket's cheapest partner's cost key [G * P] int64.
            workspace: The counts, flags and counters [5 * G + 1] int32, zero on entry.
            buckets: Buckets a block holds, P.
            blocks: Blocks, G.
            ctas_per_block: CTAs a block's search is spread over.
            stream: The stream to launch on.
        """
        mma_tile = (TILE, TILE, INDEX_DIM)
        # A stage each for an operand's tf32 rounding and its rest
        row_layout = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, mma_tile, cutlass.TFloat32, 2)
        column_layout = hopper.make_smem_layout_b(LayoutEnum.ROW_MAJOR, mma_tile, cutlass.TFloat32, 2)

        @cute.struct
        class Storage:
            """Shared memory of a block: the tile, the tail's costs and places, scalars."""

            tile: self.tile_storage(row_layout, column_layout)  # type: ignore[valid-type]
            costs: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TAIL * TAIL_STRIDE], 16]
            tail_buckets: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TAIL], 16]
            tail_open: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TAIL], 16]
            nearest_places: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, TAIL], 16]
            previous: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_BLOCKS], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, WARPS], 16]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        tiled_mma = self.make_tile_mma()
        self.kernel(
            key_sums,
            square,
            alignments,
            partner,
            order,
            cost_keys,
            workspace,
            buckets,
            blocks,
            ctas_per_block,
            row_layout,
            column_layout,
            tiled_mma,
            Storage,
        ).launch(grid=(blocks * ctas_per_block, 1, 1), block=[THREADS, 1, 1], cooperative=True, stream=stream)

    @cute.kernel
    def kernel(
        self,
        key_sums: cute.Tensor,
        square: cute.Tensor,
        alignments: cute.Tensor,
        partner: cute.Tensor,
        order: cute.Tensor,
        cost_keys: cute.Tensor,
        workspace: cute.Tensor,
        buckets: cutlass.Int32,
        blocks: cutlass.Int32,
        ctas_per_block: cutlass.Int32,
        row_layout: cute.ComposedLayout,
        column_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Run every round of a block's pairing - searched by all its CTAs while wide, by its leader after.

        Args:
            key_sums: Each bucket's key sum [G * P, index_dim] fp32.
            square: Each bucket's squared length [G * P] fp32.
            alignments: Each bucket's length [G * P] fp32, written by the block's CTAs before the first round.
            partner: Each bucket's partner within its block [G * P] int64.
            order: Two lists of each block's open buckets [2 * G * P] int32.
            cost_keys: Each open bucket's cheapest partner's cost key [G * P] int64.
            workspace: The counts, flags and counters [5 * G + 1] int32.
            buckets: Buckets a block holds.
            blocks: Blocks.
            ctas_per_block: CTAs a block's search is spread over.
            row_layout: Shared memory of a tile of rows, both splits, swizzled as the MMA reads it.
            column_layout: Shared memory of a tile of columns, both splits.
            tiled_mma: The warpgroup's MMA over a tile.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        block = cta // ctas_per_block
        slot = cta - block * ctas_per_block
        base = block * buckets
        total_rounds = buckets // 2

        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        row_splits = storage.tile.rows.get_tensor(row_layout.outer, swizzle=row_layout.inner)
        column_splits = storage.tile.columns.get_tensor(column_layout.outer, swizzle=column_layout.inner)
        costs = storage.costs.get_tensor(cute.make_layout((TAIL, TAIL), stride=(TAIL_STRIDE, 1)))
        tail_buckets = storage.tail_buckets.get_tensor(cute.make_layout(TAIL))
        tail_open = storage.tail_open.get_tensor(cute.make_layout(TAIL))
        nearest_places = storage.nearest_places.get_tensor(cute.make_layout(TAIL))
        previous = storage.previous.get_tensor(cute.make_layout(MAX_BLOCKS))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(WARPS))
        shared = storage.shared.get_tensor(cute.make_layout(4))

        tCrA, tCrB, accumulator, coords = self.tile_views(
            tiled_mma, storage, row_layout, column_layout, thread
        )

        for g in range(thread, blocks, THREADS):
            previous[g] = buckets
        for i in range(slot * THREADS + thread, buckets, ctas_per_block * THREADS):
            alignments[base + i] = approximate_sqrt(cutlass.max(square[base + i], cutlass.Float32(0.0)))
        self.wait_for(workspace, ARRIVALS * blocks + block, ctas_per_block)

        count = buckets
        parity = cutlass.Int32(0)
        check_parity = cutlass.Int32(0)
        rounds = cutlass.Int32(0)
        block_arrivals = ctas_per_block
        leader_arrivals = cutlass.Int32(0)
        running = cutlass.Int32(1)
        built = cutlass.Int32(0)
        while running != 0:
            if count > TAIL:
                tiles = cute.ceil_div(count, TILE)
                participants = cutlass.min(ctas_per_block, tiles)
                if slot < participants:
                    for row_tile in range(slot, tiles, ctas_per_block):
                        self.search(
                            key_sums,
                            square,
                            alignments,
                            order,
                            cost_keys,
                            base,
                            parity * blocks * buckets,
                            row_tile,
                            count,
                            row_splits,
                            column_splits,
                            tiled_mma,
                            tCrA,
                            tCrB,
                            accumulator,
                            coords,
                        )
                    block_arrivals = block_arrivals + participants
                    self.wait_for(workspace, ARRIVALS * blocks + block, block_arrivals)
                    if slot == 0:
                        kept = self.pair_mutual(
                            partner,
                            order,
                            cost_keys,
                            workspace,
                            base,
                            blocks,
                            buckets,
                            block,
                            parity,
                            count,
                            warp_totals,
                        )
                        stop = cutlass.Int32(0)
                        if (rounds + 1) % CHECK == 0:
                            leader_arrivals = leader_arrivals + blocks
                            stop = self.check(
                                workspace,
                                previous,
                                shared,
                                blocks,
                                block,
                                kept,
                                rounds + 1,
                                total_rounds,
                                check_parity,
                                leader_arrivals,
                            )
                            check_parity = 1 - check_parity
                        if thread == 0:
                            workspace[STOPS * blocks + block] = stop
                    block_arrivals = block_arrivals + participants
                    self.wait_for(workspace, ARRIVALS * blocks + block, block_arrivals)
                    count = workspace[COUNTS * blocks + block]
                    running = 1 - workspace[STOPS * blocks + block]
                    parity = 1 - parity
                    rounds = rounds + 1
                else:
                    running = cutlass.Int32(0)
            else:
                if slot == 0:
                    if built == 0:
                        self.build_costs(
                            key_sums,
                            square,
                            alignments,
                            order,
                            base,
                            parity * blocks * buckets,
                            count,
                            row_splits,
                            column_splits,
                            tiled_mma,
                            tCrA,
                            tCrB,
                            accumulator,
                            coords,
                            costs,
                            tail_buckets,
                            tail_open,
                        )
                        built = cutlass.Int32(1)
                    count = self.pair_tail(
                        partner, base, costs, tail_buckets, tail_open, nearest_places, shared
                    )
                    rounds = rounds + 1
                    if rounds % CHECK == 0:
                        leader_arrivals = leader_arrivals + blocks
                        stop = self.check(
                            workspace,
                            previous,
                            shared,
                            blocks,
                            block,
                            count,
                            rounds,
                            total_rounds,
                            check_parity,
                            leader_arrivals,
                        )
                        check_parity = 1 - check_parity
                        running = 1 - stop
                else:
                    running = cutlass.Int32(0)

        if slot == 0 and count > 0:
            if built != 0:
                self.list_tail(tail_buckets, tail_open, order, base, parity * blocks * buckets, shared)
            self.pair_leftovers(partner, order, base, parity * blocks * buckets, count)
        self.release_tiles()

    @cute.jit
    def wait_for(self, workspace: cute.Tensor, counter: Any, arrivals: Any) -> None:
        """Arrive at a counter in global memory and wait until it reaches the arrivals expected.

        Args:
            workspace: The counts, flags and counters.
            counter: The counter's index.
            arrivals: Its value once every CTA waited for has arrived.
        """
        thread, _, _ = cute.arch.thread_idx()
        cute.arch.barrier()
        if thread == 0:
            pointer = workspace.iterator + counter
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(pointer, cutlass.Int32(1), sem="relaxed", scope="gpu")
            seen = cute.arch.load(pointer, cutlass.Int32, sem="acquire", scope="gpu")
            while seen < arrivals:
                seen = cute.arch.load(pointer, cutlass.Int32, sem="acquire", scope="gpu")
            cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier()

    @cute.jit
    def search(
        self,
        key_sums: cute.Tensor,
        square: cute.Tensor,
        alignments: cute.Tensor,
        order: cute.Tensor,
        cost_keys: cute.Tensor,
        base: Any,
        order_base: Any,
        row_tile: Any,
        count: Any,
        row_splits: cute.Tensor,
        column_splits: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulator: cute.Tensor,
        coords: cute.Tensor,
    ) -> None:
        """Write the cost key of each open bucket of a tile's cheapest open partner, against every open bucket of the block.

        Args:
            key_sums: Each bucket's key sum.
            square: Each bucket's squared length.
            alignments: Each bucket's length.
            order: The lists of open buckets.
            cost_keys: Each open bucket's cheapest partner's cost key, written for the tile's.
            base: The block's first bucket.
            order_base: The list read's first entry.
            row_tile: The tile of the list searched for.
            count: Buckets open.
            row_splits: The rows' splits in shared memory.
            column_splits: The columns' splits in shared memory.
            tiled_mma: The warpgroup's MMA over a tile.
            tCrA: The MMA's view of the rows' splits.
            tCrB: The MMA's view of the columns' splits.
            accumulator: A tile's dots.
            coords: Each accumulator register's row and column in the tile, as rows by columns.
        """
        thread, _, _ = cute.arch.thread_idx()
        values = accumulator_rows_by_columns(accumulator)
        row_count = cute.size(values, mode=[0])
        column_count = cute.size(values, mode=[1])
        first_row = row_tile * TILE
        tiles = cute.ceil_div(count, TILE)
        staged = cute.make_rmem_tensor(INDEX_DIM // 2, cutlass.Float32)
        cute.arch.barrier()
        self.load_split(key_sums, order, base, order_base, first_row, count, staged)
        self.store_split(staged, row_splits)
        row_squares = cute.make_rmem_tensor(row_count, cutlass.Float32)
        row_alignments = cute.make_rmem_tensor(row_count, cutlass.Float32)
        best_costs = cute.make_rmem_tensor(row_count, cutlass.Int32)
        best_columns = cute.make_rmem_tensor(row_count, cutlass.Int32)
        for r in cutlass.range_constexpr(row_count):
            place = first_row + coords[r, 0][0]
            row_bucket = cutlass.Int32(0)
            if place < count:
                row_bucket = order[order_base + base + place]
            row_squares[r] = square[base + row_bucket]
            row_alignments[r] = alignments[base + row_bucket]
            best_costs[r] = cutlass.Int32(0)
            best_columns[r] = cutlass.Int32(-1)
        column_squares = cute.make_rmem_tensor(column_count, cutlass.Float32)
        column_alignments = cute.make_rmem_tensor(column_count, cutlass.Float32)
        self.load_split(key_sums, order, base, order_base, 0, count, staged)

        # The next tile's key sums load while this one is scored
        for column_tile in range(tiles):
            first_column = column_tile * TILE
            cute.arch.barrier()
            self.store_split(staged, column_splits)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if column_tile + 1 < tiles:
                self.load_split(key_sums, order, base, order_base, first_column + TILE, count, staged)
            for c in cutlass.range_constexpr(column_count):
                column = first_column + coords[0, c][1]
                column_bucket = cutlass.Int32(0)
                if column < count:
                    column_bucket = order[order_base + base + column]
                column_squares[c] = square[base + column_bucket]
                column_alignments[c] = alignments[base + column_bucket]
            self.score_tile(tiled_mma, tCrA, tCrB, accumulator)
            # A thread meets its columns in rising order
            # => a strictly lower cost replaces the best, and a tie keeps the lower column, as the cost key's low bits do
            for c in cutlass.range_constexpr(column_count):
                column = first_column + coords[0, c][1]
                for r in cutlass.range_constexpr(row_count):
                    cost = ordered_cost(
                        self.merge_cost(
                            values[r, c],
                            row_squares[r],
                            row_alignments[r],
                            column_squares[c],
                            column_alignments[c],
                        )
                    )
                    if column < count and column != first_row + coords[r, 0][0]:
                        if best_columns[r] < 0 or cost < best_costs[r]:
                            best_costs[r] = cost
                            best_columns[r] = column

        # A row's columns sit in the four threads of a quad
        for r in cutlass.range_constexpr(row_count):
            cost = best_costs[r]
            column = best_columns[r]
            for i in cutlass.range_constexpr(2):
                other_cost = cute.arch.shuffle_sync_bfly(cost, offset=1 << i)
                other_column = cute.arch.shuffle_sync_bfly(column, offset=1 << i)
                if other_column >= 0:
                    if column < 0 or other_cost < cost or (other_cost == cost and other_column < column):
                        cost = other_cost
                        column = other_column
            place = first_row + coords[r, 0][0]
            if thread % 4 == 0 and place < count:
                cost_key = cutlass.Int64(MAX_KEY)
                if column >= 0:
                    cost_key = pack_cost_key(cost, column)
                cost_keys[base + place] = cost_key

    @cute.jit
    def pair_mutual(
        self,
        partner: cute.Tensor,
        order: cute.Tensor,
        cost_keys: cute.Tensor,
        workspace: cute.Tensor,
        base: Any,
        blocks: Any,
        buckets: Any,
        block: Any,
        parity: Any,
        count: Any,
        warp_totals: cute.Tensor,
    ) -> Any:
        """Pair every two open buckets that are each other's cheapest, and list the rest in order in the other list.

        Args:
            partner: Each bucket's partner within its block.
            order: The lists of open buckets.
            cost_keys: Each open bucket's cheapest partner's cost key.
            workspace: The counts, flags and counters - the block's open count written.
            base: The block's first bucket.
            blocks: Blocks.
            buckets: Buckets a block holds.
            block: The block.
            parity: The list read.
            count: Buckets open.
            warp_totals: Shared memory for the scan.

        Returns:
            Buckets still open.
        """
        thread, _, _ = cute.arch.thread_idx()
        read_base = parity * blocks * buckets + base
        write_base = (1 - parity) * blocks * buckets + base
        index_mask = cutlass.Int64((1 << INDEX_BITS) - 1)
        kept = cutlass.Int32(0)
        for first in range(0, count, THREADS):
            place = first + thread
            keep = cutlass.Int32(0)
            bucket = cutlass.Int32(0)
            if place < count:
                bucket = order[read_base + place]
                cost_key = cost_keys[base + place]
                mutual = cutlass.Int32(0)
                if cost_key < cutlass.Int64(MAX_KEY):
                    best = cutlass.Int32(cost_key & index_mask)
                    back = cost_keys[base + best]
                    if back < cutlass.Int64(MAX_KEY) and cutlass.Int32(back & index_mask) == place:
                        mutual = cutlass.Int32(1)
                        partner[base + bucket] = cutlass.Int64(order[read_base + best])
                keep = 1 - mutual
            before, chunk_kept = self.exclusive_sum(keep, warp_totals)
            if keep != 0:
                order[write_base + kept + before] = bucket
            kept = kept + chunk_kept
        if thread == 0:
            workspace[COUNTS * blocks + block] = kept
        return kept

    @cute.jit
    def check(
        self,
        workspace: cute.Tensor,
        previous: cute.Tensor,
        shared: cute.Tensor,
        blocks: Any,
        block: Any,
        count: Any,
        rounds: Any,
        total_rounds: Any,
        check_parity: Any,
        leader_arrivals: Any,
    ) -> Any:
        """Meet the other blocks' leaders and tell whether to stop - none open, too few paired, or the last round.

        Args:
            workspace: The counts, flags and counters.
            previous: Each block's open count at the last check, updated.
            shared: Shared memory to hand the answer to every thread.
            blocks: Blocks.
            block: The block.
            count: The block's buckets open.
            rounds: Rounds done.
            total_rounds: Rounds the Triton pairing launches at most.
            check_parity: Which of the two check counts this check writes.
            leader_arrivals: The leaders' counter once every leader has arrived.

        Returns:
            1 to stop, 0 to go on.
        """
        thread, _, _ = cute.arch.thread_idx()
        checks = (CHECKS + check_parity) * blocks
        if thread == 0:
            workspace[checks + block] = count
        self.wait_for(workspace, LEADERS * blocks, leader_arrivals)
        if thread == 0:
            open_any = cutlass.Int32(0)
            stalled = cutlass.Int32(1)
            for g in range(blocks):
                now = workspace[checks + g]
                before = previous[g]
                if now > 0:
                    open_any = cutlass.Int32(1)
                if (before - now) * 64 >= before:
                    stalled = cutlass.Int32(0)
                previous[g] = now
            stop = cutlass.Int32(0)
            if open_any == 0 or stalled != 0 or rounds >= total_rounds:
                stop = cutlass.Int32(1)
            shared[0] = stop
        cute.arch.barrier()
        stop = shared[0]
        cute.arch.barrier()
        return stop

    @cute.jit
    def build_costs(
        self,
        key_sums: cute.Tensor,
        square: cute.Tensor,
        alignments: cute.Tensor,
        order: cute.Tensor,
        base: Any,
        order_base: Any,
        count: Any,
        row_splits: cute.Tensor,
        column_splits: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulator: cute.Tensor,
        coords: cute.Tensor,
        costs: cute.Tensor,
        tail_buckets: cute.Tensor,
        tail_open: cute.Tensor,
    ) -> None:
        """Score the last 64 or fewer open buckets against each other once, into shared memory, and list them.

        Args:
            key_sums: Each bucket's key sum.
            square: Each bucket's squared length.
            alignments: Each bucket's length.
            order: The lists of open buckets.
            base: The block's first bucket.
            order_base: The list read's first entry.
            count: Buckets open, at most 64.
            row_splits: The rows' splits in shared memory.
            column_splits: The columns' splits in shared memory.
            tiled_mma: The warpgroup's MMA over a tile.
            tCrA: The MMA's view of the rows' splits.
            tCrB: The MMA's view of the columns' splits.
            accumulator: A tile's dots.
            coords: Each accumulator register's row and column in the tile.
            costs: Each open bucket's ordered cost to every other, by their places, written.
            tail_buckets: Each place's bucket, written.
            tail_open: Whether each place is open, written.
        """
        thread, _, _ = cute.arch.thread_idx()
        values = accumulator_rows_by_columns(accumulator)
        if thread < TAIL:
            bucket = cutlass.Int32(0)
            if thread < count:
                bucket = order[order_base + base + thread]
            tail_buckets[thread] = bucket
            tail_open[thread] = cutlass.Int32(thread < count)
        cute.arch.barrier()
        staged = cute.make_rmem_tensor(INDEX_DIM // 2, cutlass.Float32)
        self.load_split(key_sums, order, base, order_base, 0, count, staged)
        self.store_split(staged, row_splits)
        self.store_split(staged, column_splits)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        self.score_tile(tiled_mma, tCrA, tCrB, accumulator)
        for c in cutlass.range_constexpr(cute.size(values, mode=[1])):
            column = coords[0, c][1]
            column_square = square[base + tail_buckets[column]]
            column_alignment = alignments[base + tail_buckets[column]]
            for r in cutlass.range_constexpr(cute.size(values, mode=[0])):
                row = coords[r, 0][0]
                row_square = square[base + tail_buckets[row]]
                row_alignment = alignments[base + tail_buckets[row]]
                cost = self.merge_cost(
                    values[r, c], row_square, row_alignment, column_square, column_alignment
                )
                costs[row, column] = ordered_cost(cost)
        cute.arch.barrier()

    @cute.jit
    def pair_tail(
        self,
        partner: cute.Tensor,
        base: Any,
        costs: cute.Tensor,
        tail_buckets: cute.Tensor,
        tail_open: cute.Tensor,
        nearest_places: cute.Tensor,
        shared: cute.Tensor,
    ) -> Any:
        """Run one round over the matrix of costs - each open place's cheapest open partner, mutual ones paired.

        Args:
            partner: Each bucket's partner within its block.
            base: The block's first bucket.
            costs: Each open bucket's ordered cost to every other, by their places.
            tail_buckets: Each place's bucket.
            tail_open: Whether each place is open, updated.
            nearest_places: Each place's cheapest partner's place, -1 for none.
            shared: Shared memory to hand the count to every thread.

        Returns:
            Buckets still open.
        """
        thread, _, _ = cute.arch.thread_idx()
        is_open = cutlass.Int32(0)
        if thread < TAIL:
            is_open = tail_open[thread]
            pick = cutlass.Int32(-1)
            if is_open != 0:
                cheapest = cutlass.Int64(MAX_KEY)
                for column in cutlass.range(TAIL, unroll_full=True):
                    if tail_open[column] != 0 and column != thread:
                        cheapest = cutlass.min(cheapest, pack_cost_key(costs[thread, column], column))
                if cheapest < cutlass.Int64(MAX_KEY):
                    pick = cutlass.Int32(cheapest & cutlass.Int64((1 << INDEX_BITS) - 1))
            nearest_places[thread] = pick
        cute.arch.barrier()
        still = cutlass.Int32(0)
        if thread < TAIL:
            pick = nearest_places[thread]
            still = is_open
            if is_open != 0 and pick >= 0:
                if nearest_places[pick] == thread:
                    partner[base + tail_buckets[thread]] = cutlass.Int64(tail_buckets[pick])
                    still = cutlass.Int32(0)
        cute.arch.barrier()
        open_mask = cute.arch.vote_ballot_sync(still != 0)
        if thread < TAIL:
            tail_open[thread] = still
            if thread % 32 == 0:
                shared[1 + thread // 32] = cutlass.Int32(cute.arch.popc(open_mask))
        cute.arch.barrier()
        count = cutlass.Int32(0)
        for w in cutlass.range_constexpr(TAIL_WARPS):
            count = count + shared[1 + w]
        cute.arch.barrier()
        return count

    @cute.jit
    def list_tail(
        self,
        tail_buckets: cute.Tensor,
        tail_open: cute.Tensor,
        order: cute.Tensor,
        base: Any,
        order_base: Any,
        shared: cute.Tensor,
    ) -> None:
        """List the places still open of the matrix of costs, in order, as the wide rounds list them.

        Args:
            tail_buckets: Each place's bucket.
            tail_open: Whether each place is open.
            order: The lists of open buckets, the one read written.
            base: The block's first bucket.
            order_base: The list's first entry.
            shared: Shared memory to hand warp 0's open count to warp 1.
        """
        thread, _, _ = cute.arch.thread_idx()
        is_open = cutlass.Int32(0)
        if thread < TAIL:
            is_open = tail_open[thread]
        open_mask = cute.arch.vote_ballot_sync(is_open != 0)
        cute.arch.barrier()
        if thread == 0:
            shared[0] = cutlass.Int32(cute.arch.popc(open_mask))
        cute.arch.barrier()
        warp_zero = shared[0]
        cute.arch.barrier()
        if thread < TAIL and is_open != 0:
            place = cutlass.Int32(cute.arch.popc(open_mask & cute.arch.lanemask_lt()))
            if thread >= 32:
                place = place + warp_zero
            order[order_base + base + place] = tail_buckets[thread]
        cute.arch.barrier()

    @cute.jit
    def pair_leftovers(
        self, partner: cute.Tensor, order: cute.Tensor, base: Any, order_base: Any, count: Any
    ) -> None:
        """Pair the buckets still open in index order - first with second, third with fourth.

        Args:
            partner: Each bucket's partner within its block.
            order: The lists of open buckets.
            base: The block's first bucket.
            order_base: The list read's first entry.
            count: Buckets open.
        """
        thread, _, _ = cute.arch.thread_idx()
        for place in range(thread, count, THREADS):
            mate = place ^ 1
            if mate < count:
                partner[base + order[order_base + base + place]] = cutlass.Int64(
                    order[order_base + base + mate]
                )


COMPILED: dict[type[NodeBucketClustering], Any] = {}


def compiled(kernel: type[NodeBucketClustering] = NodeBucketClustering) -> Any:
    """Return the pairing compiled once a kernel class, taking torch tensors as they are.

    Args:
        kernel: The pairing's class - Hopper's, or one that takes its tiles on another GPU's MMA.
    """
    if kernel in COMPILED:
        return COMPILED[kernel]
    rows, lists, workspace = cute.sym_int32(), cute.sym_int32(), cute.sym_int32()
    arguments = (
        make_fake_compact_tensor(cutlass.Float32, (rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (rows,), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (lists,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (rows,), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (workspace,), assumed_align=4),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED[kernel] = cute.compile(kernel(), *arguments, options="--enable-tvm-ffi")
    return COMPILED[kernel]


def takes_blocks(key_sums: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL pairing's rounds take these blocks on any GPU - tf32x3, 64 index dims, 64 to 4,096 buckets.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    blocks, buckets, index_dim = key_sums.shape
    return (
        key_sums.is_cuda
        and precision == "tf32x3"
        and index_dim == INDEX_DIM
        and TILE <= buckets <= 1 << INDEX_BITS
        and buckets & (buckets - 1) == 0
        and 0 < blocks <= min(MAX_BLOCKS, multiprocessors(key_sums.device))
    )


def supports(key_sums: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL pairing takes these blocks - tf32x3, 64 index dims, 64 to 4,096 buckets, Hopper.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return (
        key_sums.is_cuda
        and dispatch.capability(key_sums.device) == dispatch.HOPPER
        and takes_blocks(key_sums, precision)
    )


def multiprocessors(device: torch.device) -> int:
    """Return a GPU's multiprocessors.

    Args:
        device: The GPU.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


def cluster_node_buckets(key_sums: torch.Tensor) -> torch.Tensor:
    """Return each bucket's partner within its block, every round in one launch - the Triton rounds' partners.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32, as `supports` takes them.

    Returns:
        Each bucket's partner [G, P] int64.

    Raises:
        ValueError: If the pairing does not take these blocks.
    """
    if not supports(key_sums, "tf32x3"):
        raise ValueError(
            "the CuTeDSL pairing takes tf32x3 blocks of 64 to 4,096 buckets, a power of two, of 64 index dims, "
            f"at most one block an SM, on a Hopper GPU - got {tuple(key_sums.shape)} on {key_sums.device}"
        )
    return launch(key_sums, NodeBucketClustering)


def launch(key_sums: torch.Tensor, kernel: type[NodeBucketClustering]) -> torch.Tensor:
    """Run every round of the pairing in one launch of a kernel class.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32, as the class's GPU takes them.
        kernel: The pairing's class.

    Returns:
        Each bucket's partner [G, P] int64.
    """
    blocks, buckets, _ = key_sums.shape
    device = key_sums.device
    key_sums = key_sums.float().contiguous()
    square = (key_sums * key_sums).sum(-1).contiguous()
    partner = torch.full((blocks * buckets,), -1, dtype=torch.int64, device=device)
    order = torch.empty(2 * blocks * buckets, dtype=torch.int32, device=device)
    order[: blocks * buckets] = torch.arange(buckets, dtype=torch.int32, device=device).repeat(blocks)
    cost_keys = torch.empty(blocks * buckets, dtype=torch.int64, device=device)
    workspace = torch.zeros(5 * blocks + 1, dtype=torch.int32, device=device)
    ctas_per_block = min(buckets // TILE, multiprocessors(device) // blocks)
    compiled(kernel)(
        key_sums.view(blocks * buckets, -1),
        square.view(-1),
        torch.empty_like(square.view(-1)),
        partner,
        order,
        cost_keys,
        workspace,
        buckets,
        blocks,
        ctas_per_block,
    )
    return partner.view(blocks, buckets)
