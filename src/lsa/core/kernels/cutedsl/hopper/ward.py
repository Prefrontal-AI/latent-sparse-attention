"""A cluster block's Ward merge in CuTeDSL, every round on its own pools in shared memory - the Triton merge's.

A round finds each live pool's cheapest partner under the cap, merges every two that are each other's cheapest - the
cheapest pairs first, when more are mutual than the block's count leaves room for - and stops once the block reaches
its count or a round merges nothing. The Triton merge runs one program a block and reads every round's key sums from
global memory; here a CTA holds its block's key sums, lengths, sizes and links in shared memory from first round to
last, so a round is a few tiles of MMAs between barriers.

A block's rounds are serial, and a round's search is every pair of its pools, so a block of unpooled keys - 512 pools,
sixteen times a pooled block's pairs - would leave one SM working and the rest idle. A block is merged by a cluster of
CTAs instead, as many as keep every block's cluster on the GPU at once: each searches a share of the rows against
every column, hands its rows' keys to its peers through their shared memory, and then merges the round's pairs on its
own copy of the block - the same pairs in every copy. Every copy holds the same keys because nothing in a round is
atomic and every CTA takes the same loop decisions, so every cluster barrier pairs.

A merge cost is the node-bucket clustering's, Triton's tf32x3 dot bit for bit, and a merged pool's squared length is added as the
Triton merge adds it - each run of four dims second squared first, then the first, third and fourth fused in, and the
runs added in halves - so every round merges the same pools and every pool gets the same label.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as hopper
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum

from lsa.core.kernels import dispatch
from lsa.core.kernels.cutedsl.hopper.tile import (
    INDEX_DIM,
    MAX_KEY,
    THREADS,
    TILE,
    HopperTile,
    accumulator_rows_by_columns,
    approximate_sqrt,
    ordered_cost,
    pack_cost_key,
)

# Pools a block may hold - a 512-key window's keys unpooled, whose sums fill most of a CTA's shared memory
MOST_POOLS = 512
# At 512 pools the opt-in budget is 227 KB = 232,448 B, and a CTA holds
#   tiles   rows and columns, [64, 64] fp32 in two splits each                  2 * 32,768 =  65,536 B
#   sums    512 * (64 + 1 padding) fp32                                                   = 133,120 B
#   state   square, alignment, size, alive, parent, 5 * 512 * 4                           =  10,240 B
#   round   best 2 * 512 * 4, cost keys 2 * 512 * 8, mutual 512 * 8                       =  16,384 B
#   misc    warp totals, tile barrier, tensor memory address                              =      28 B
# => 225,308 B, 7,140 B under the budget - a four-word padding (8,192 B) would leave 996 B
# CTAs a cluster may hold - the portable limit
MOST_CLUSTER_CTAS = 8
# Words a row of sums is padded by, so a warp reading one dim of 16 pools meets 16 banks, not one
SUMS_PADDING = 1
# Dims a run of the squared length adds before the runs are halved
RUN = 4

COMPILED: dict[tuple[int, int, str], Any] = {}
MAX_ACTIVE_CLUSTERS: dict[tuple[int, int], int] = {}


@dsl_user_op
def store_to_peer(address: cute.Pointer, peer: Any, value: Any, *, loc: Any = None, ip: Any = None) -> None:
    """Store a 32-bit word at the same shared memory address in a peer CTA of the cluster.

    Args:
        address: The word's address in this CTA's shared memory.
        peer: The peer's index among the cluster CTAs.
        value: The int32 word.
    """
    remote = cute.arch.map_dsmem_ptr(address, peer, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            remote.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class WardClustering(HopperTile):
    """The CuTeDSL ward clustering, compiled once a pool count, cluster width and class."""

    def __init__(self, pools: int, cluster_ctas: int) -> None:
        """Fix the block's pools and the CTAs that search them.

        Args:
            pools: Pools a block holds, P - a power of two from TILE to MOST_POOLS.
            cluster_ctas: CTAs of the cluster a block is merged by, each searching P / cluster_ctas rows a round.
        """
        self.pools = pools
        self.cluster_ctas = cluster_ctas

    @cute.jit
    def __call__(
        self,
        key_sums: cute.Tensor,
        sizes: cute.Tensor,
        labels: cute.Tensor,
        reached: cute.Tensor,
        bucket_count: cutlass.Int32,
        bucket_cap: cutlass.Float32,
        stream: Any,
    ) -> None:
        """Lay out shared memory and the MMA, and launch a cluster of cluster CTAs a block.

        Args:
            key_sums: Each block's pool keys [G * P, index_dim] fp32.
            sizes: Keys each pool holds [G * P] fp32.
            labels: Each pool's bucket [G * P] int64, written.
            reached: Whether each block reached its count [G] int32, written.
            bucket_count: Buckets a block ends with.
            bucket_cap: Most keys a merged bucket may hold.
            stream: The stream to launch on.
        """
        mma_tile = (TILE, TILE, INDEX_DIM)
        row_layout = hopper.make_smem_layout_a(LayoutEnum.ROW_MAJOR, mma_tile, cutlass.TFloat32, 2)
        column_layout = hopper.make_smem_layout_b(LayoutEnum.ROW_MAJOR, mma_tile, cutlass.TFloat32, 2)
        pools = self.pools

        @cute.struct
        class Storage:
            """Shared memory of a block: operand tiles, the pools' sums and merge state, a round's keys."""

            tile: self.tile_storage(row_layout, column_layout)  # type: ignore[valid-type]
            sums: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, pools * (INDEX_DIM + SUMS_PADDING)], 16
            ]
            square: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, pools], 16]
            alignment: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, pools], 16]
            size: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, pools], 16]
            alive: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, pools], 16]
            parent: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, pools], 16]
            # A round's cheapest partners and keys, one copy a round's parity
            # A peer may still read round r - 1's copy while this CTA writes round r's
            # => round r - 2's reads all finish before the round r - 1 cluster barrier, so two copies suffice
            best: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * pools], 16]
            cost_keys: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2 * pools], 16]
            mutual: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, pools], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, THREADS // 32], 16]

        tiled_mma = self.make_tile_mma()
        self.kernel(
            key_sums,
            sizes,
            labels,
            reached,
            bucket_count,
            bucket_cap,
            row_layout,
            column_layout,
            tiled_mma,
            Storage,
        ).launch(
            grid=(reached.shape[0] * self.cluster_ctas, 1, 1),
            block=[THREADS, 1, 1],
            cluster=[self.cluster_ctas, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        key_sums: cute.Tensor,
        sizes: cute.Tensor,
        labels: cute.Tensor,
        reached: cute.Tensor,
        bucket_count: cutlass.Int32,
        bucket_cap: cutlass.Float32,
        row_layout: cute.ComposedLayout,
        column_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Merge one block's pools down to its count, then label every pool by the bucket it ended in.

        Args:
            key_sums: Each block's pool keys [G * P, index_dim] fp32.
            sizes: Keys each pool holds [G * P] fp32.
            labels: Each pool's bucket [G * P] int64.
            reached: Whether each block reached its count [G] int32.
            bucket_count: Buckets a block ends with.
            bucket_cap: Most keys a merged bucket may hold.
            row_layout: Shared memory of a tile of rows, both splits, swizzled as the MMA reads it.
            column_layout: Shared memory of a tile of columns, both splits.
            tiled_mma: The warpgroup's MMA over a tile.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        pools = self.pools
        cluster_ctas = self.cluster_ctas
        block = cta // cluster_ctas
        cluster_cta = cta % cluster_ctas
        base = block * pools
        # The rows this CTA searches and labels
        share = pools // cluster_ctas
        first_own = cluster_cta * share

        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        row_splits = storage.tile.rows.get_tensor(row_layout.outer, swizzle=row_layout.inner)
        column_splits = storage.tile.columns.get_tensor(column_layout.outer, swizzle=column_layout.inner)
        sums = storage.sums.get_tensor(
            cute.make_layout((pools, INDEX_DIM), stride=(INDEX_DIM + SUMS_PADDING, 1))
        )
        square = storage.square.get_tensor(cute.make_layout(pools))
        alignment = storage.alignment.get_tensor(cute.make_layout(pools))
        size = storage.size.get_tensor(cute.make_layout(pools))
        alive = storage.alive.get_tensor(cute.make_layout(pools))
        parent = storage.parent.get_tensor(cute.make_layout(pools))
        best_copies = storage.best.get_tensor(cute.make_layout(2 * pools))
        cost_key_copies = storage.cost_keys.get_tensor(cute.make_layout(2 * pools))
        mutual = storage.mutual.get_tensor(cute.make_layout(pools))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(THREADS // 32))

        tCrA, tCrB, accumulator, coords = self.tile_views(
            tiled_mma, storage, row_layout, column_layout, thread
        )

        for i in range(thread, pools * INDEX_DIM, THREADS):
            sums[i // INDEX_DIM, i % INDEX_DIM] = key_sums[base + i // INDEX_DIM, i % INDEX_DIM]
        cute.arch.barrier()
        for i in range(thread, pools, THREADS):
            self.measure(sums, square, alignment, i)
            size[i] = sizes[base + i]
            alive[i] = cutlass.Int32(1)
            parent[i] = i
        cute.arch.barrier()
        # Every peer is running before any writes into its shared memory
        if cutlass.const_expr(cluster_ctas > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        alive_count = cutlass.Int32(pools)
        applied = cutlass.Int32(1)
        rounds = cutlass.Int32(0)
        while alive_count > bucket_count and applied > 0 and rounds < pools:
            parity = rounds % 2
            best = cute.make_tensor(best_copies.iterator + parity * pools, cute.make_layout(pools))
            cost_keys = cute.make_tensor(cost_key_copies.iterator + parity * pools, cute.make_layout(pools))
            for own_tile in cutlass.range_constexpr(share // TILE):
                row_tile = cluster_cta * (share // TILE) + own_tile
                self.cheapest(
                    sums,
                    square,
                    alignment,
                    size,
                    alive,
                    best,
                    cost_keys,
                    row_tile,
                    bucket_cap,
                    row_splits,
                    column_splits,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    accumulator,
                    coords,
                )
            cute.arch.barrier()
            if cutlass.const_expr(cluster_ctas > 1):
                self.share_cheapest(best, cost_keys, first_own, share, cluster_cta)

            # A pair that is each other's cheapest, named by its lower pool
            mutual_count = cutlass.Int32(0)
            for first in range(0, pools, THREADS):
                i = first + thread
                is_mutual = cutlass.Int32(0)
                # A block narrower than the CTA leaves threads with no pool
                if i < pools:
                    partner = best[i]
                    if partner >= 0:
                        if best[partner] == i and i < partner:
                            is_mutual = cutlass.Int32(1)
                    mutual[i] = cutlass.Int64(MAX_KEY)
                    if is_mutual != 0:
                        mutual[i] = cost_keys[i]
                _, chunk = self.exclusive_sum(is_mutual, warp_totals)
                mutual_count = mutual_count + chunk
            cute.arch.barrier()

            # The cheapest pairs the block's count leaves room for merge, the lower pool taking the higher
            merges_left = alive_count - bucket_count
            applied = cutlass.min(mutual_count, merges_left)
            for i in range(thread, pools, THREADS):
                mine = mutual[i]
                merging = cutlass.Int32(0)
                if mine < cutlass.Int64(MAX_KEY):
                    merge_order = cutlass.Int32(0)
                    if mutual_count > merges_left:
                        for j in range(pools):
                            merge_order = merge_order + cutlass.Int32(mutual[j] < mine)
                    merging = cutlass.Int32(merge_order < merges_left)
                if merging != 0:
                    partner = best[i]
                    for d in cutlass.range(INDEX_DIM, unroll_full=True):
                        sums[i, d] = sums[i, d] + sums[partner, d]
                    self.measure(sums, square, alignment, i)
                    size[i] = size[i] + size[partner]
                    alive[partner] = cutlass.Int32(0)
                    parent[partner] = i
            cute.arch.barrier()
            alive_count = alive_count - applied
            rounds = rounds + 1

        # A surviving pool's bucket is numbered by how many survive before it, and a merged pool takes its root's
        best = cute.make_tensor(best_copies.iterator, cute.make_layout(pools))
        seen = cutlass.Int32(0)
        for first in range(0, pools, THREADS):
            i = first + thread
            survives = cutlass.Int32(0)
            if i < pools:
                survives = alive[i]
            before, chunk = self.exclusive_sum(survives, warp_totals)
            if i < pools:
                best[i] = seen + before
            seen = seen + chunk
        cute.arch.barrier()
        for i in range(first_own + thread, first_own + share, THREADS):
            root = i
            while parent[root] != root:
                root = parent[root]
            labels[base + i] = cutlass.Int64(best[root])
        if thread == 0 and cluster_cta == 0:
            reached[block] = cutlass.Int32(alive_count <= bucket_count)
        self.release_tiles()

    @cute.jit
    def share_cheapest(
        self,
        best: cute.Tensor,
        cost_keys: cute.Tensor,
        first_own: Any,
        share: cutlass.Constexpr,
        cluster_cta: Any,
    ) -> None:
        """Copy this CTA's rows' cheapest partners and keys into every peer's shared memory, then wait for theirs.

        Args:
            best: This round's cheapest partners in shared memory.
            cost_keys: This round's keys in shared memory.
            first_own: The first row this CTA searched.
            share: Rows each CTA searched.
            cluster_cta: This CTA's index among the cluster CTAs.
        """
        thread, _, _ = cute.arch.thread_idx()
        for i in range(first_own + thread, first_own + share, THREADS):
            partner = best[i]
            partner_cost_key = cost_keys[i]
            low_word = cutlass.Int32(partner_cost_key & cutlass.Int64(0xFFFFFFFF))
            high_word = cutlass.Int32(partner_cost_key >> 32)
            words = cute.recast_tensor(
                cute.make_tensor(cost_keys.iterator + i, cute.make_layout(1)), cutlass.Int32
            )
            for peer in cutlass.range_constexpr(self.cluster_ctas):
                if peer != cluster_cta:
                    store_to_peer(best.iterator + i, peer, partner)
                    store_to_peer(words.iterator, peer, low_word)
                    store_to_peer(words.iterator + 1, peer, high_word)
        # The arrive releases this CTA's stores and the wait acquires every peer's
        # The last remote store precedes the last cluster barrier
        # => no CTA exits while a peer still writes to its shared memory
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def measure(self, sums: cute.Tensor, square: cute.Tensor, alignment: cute.Tensor, pool: Any) -> None:
        """Write a pool's squared length and length, the square added as the Triton merge adds it.

        Args:
            sums: The block's key sums in shared memory.
            square: Each pool's squared length, written for the pool.
            alignment: Each pool's length, written for the pool.
            pool: The pool.
        """
        runs = cute.make_rmem_tensor(INDEX_DIM // RUN, cutlass.Float32)
        for j in cutlass.range_constexpr(INDEX_DIM // RUN):
            first = sums[pool, RUN * j]
            second = sums[pool, RUN * j + 1]
            third = sums[pool, RUN * j + 2]
            fourth = sums[pool, RUN * j + 3]
            run = cute.math.fma(first, first, second * second)
            run = cute.math.fma(third, third, run)
            runs[j] = cute.math.fma(fourth, fourth, run)
        # Each run added to the one half the runs along, until one is left
        for step in cutlass.range_constexpr((INDEX_DIM // RUN).bit_length() - 1):
            half = INDEX_DIM // RUN // 2 >> step
            for j in cutlass.range_constexpr(half):
                runs[j] = runs[j] + runs[j + half]
        square[pool] = runs[0]
        alignment[pool] = approximate_sqrt(cutlass.max(runs[0], cutlass.Float32(0.0)))

    @cute.jit
    def stage(self, sums: cute.Tensor, first: Any, splits: cute.Tensor) -> None:
        """Store a tile of pools' key sums into shared memory as the MMA reads them, split into tf32 and the rest.

        Args:
            sums: The block's key sums in shared memory.
            first: The tile's first pool.
            splits: The tile's two stages.
        """
        thread, _, _ = cute.arch.thread_idx()
        # A warp reads one dim of 16 pools, two threads a pool
        # The padded rows put the pools in 16 banks, and the two halves of a pool share one
        # => a 2-way conflict stays, from 32-way unpadded - the halves are 32 words apart, whatever the row stride
        staged = cute.make_rmem_tensor(INDEX_DIM // 2, cutlass.Float32)
        pool = first + thread // 2
        first_dim = (thread % 2) * (INDEX_DIM // 2)
        for d in cutlass.range_constexpr(INDEX_DIM // 2):
            staged[d] = sums[pool, first_dim + d]
        self.store_split(staged, splits)

    @cute.jit
    def cheapest(
        self,
        sums: cute.Tensor,
        square: cute.Tensor,
        alignment: cute.Tensor,
        size: cute.Tensor,
        alive: cute.Tensor,
        best: cute.Tensor,
        cost_keys: cute.Tensor,
        row_tile: Any,
        bucket_cap: Any,
        row_splits: cute.Tensor,
        column_splits: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        accumulator: cute.Tensor,
        coords: cute.Tensor,
    ) -> None:
        """Write each live pool of a tile's cheapest partner the cap allows, and its key, against every pool of the block.

        Args:
            sums: The block's key sums in shared memory.
            square: Each pool's squared length.
            alignment: Each pool's length.
            size: Keys each pool holds.
            alive: Whether each pool is live.
            best: Each pool's cheapest partner, -1 for none, written for the tile's.
            cost_keys: Each pool's cheapest partner's key, written for the tile's.
            row_tile: The tile searched for.
            bucket_cap: Most keys a merged bucket may hold.
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
        cute.arch.barrier()
        self.stage(sums, first_row, row_splits)
        best_costs = cute.make_rmem_tensor(row_count, cutlass.Int32)
        best_columns = cute.make_rmem_tensor(row_count, cutlass.Int32)
        row_squares = cute.make_rmem_tensor(row_count, cutlass.Float32)
        row_alignments = cute.make_rmem_tensor(row_count, cutlass.Float32)
        # A dead row's room is negative, so no column fits it
        row_room = cute.make_rmem_tensor(row_count, cutlass.Float32)
        for r in cutlass.range_constexpr(row_count):
            row = first_row + coords[r, 0][0]
            best_costs[r] = cutlass.Int32(0)
            best_columns[r] = cutlass.Int32(-1)
            row_squares[r] = square[row]
            row_alignments[r] = alignment[row]
            row_room[r] = bucket_cap - size[row]
            if alive[row] == 0:
                row_room[r] = cutlass.Float32(-1.0)
        column_squares = cute.make_rmem_tensor(column_count, cutlass.Float32)
        column_alignments = cute.make_rmem_tensor(column_count, cutlass.Float32)
        column_sizes = cute.make_rmem_tensor(column_count, cutlass.Float32)
        # A runtime loop, its state all in registers
        # => the tile's body is compiled once, not once a column tile
        for column_tile in range(self.pools // TILE):
            first_column = column_tile * TILE
            cute.arch.barrier()
            self.stage(sums, first_column, column_splits)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            for c in cutlass.range_constexpr(column_count):
                column = first_column + coords[0, c][1]
                column_squares[c] = square[column]
                column_alignments[c] = alignment[column]
                # A dead column's size passes any room
                column_sizes[c] = size[column]
                if alive[column] == 0:
                    column_sizes[c] = cutlass.Float32.inf
            self.score_tile(tiled_mma, tCrA, tCrB, accumulator)
            # A thread meets its columns in rising order
            # => a strictly lower cost replaces the best, and a tie keeps the lower column, as the key's low bits do
            for c in cutlass.range_constexpr(column_count):
                column = first_column + coords[0, c][1]
                for r in cutlass.range_constexpr(row_count):
                    row = first_row + coords[r, 0][0]
                    cost = ordered_cost(
                        self.merge_cost(
                            values[r, c],
                            row_squares[r],
                            row_alignments[r],
                            column_squares[c],
                            column_alignments[c],
                        )
                    )
                    # Sizes are whole key counts, so the room's subtraction is exact
                    # => a column fits the room exactly when the two sizes fit the cap
                    allowed = row != column and column_sizes[c] <= row_room[r]
                    if allowed:
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
            if thread % 4 == 0:
                row = first_row + coords[r, 0][0]
                cost_keys[row] = cutlass.Int64(MAX_KEY)
                best[row] = cutlass.Int32(-1)
                if column >= 0:
                    cost_keys[row] = pack_cost_key(cost, column)
                    best[row] = column


def max_active_clusters(device: torch.device, cluster_ctas: int) -> int:
    """Return the clusters of a width the GPU runs at once, each CTA holding the most shared memory it may.

    A cluster sits in one GPC, so clusters of 4 or 8 CTAs run in waves long before every SM is taken.

    Args:
        device: The GPU.
        cluster_ctas: CTAs of a cluster.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    if (index, cluster_ctas) not in MAX_ACTIVE_CLUSTERS:
        with torch.cuda.device(index):
            MAX_ACTIVE_CLUSTERS[index, cluster_ctas] = cutlass.utils.HardwareInfo(
                index
            ).get_max_active_clusters(cluster_ctas)
    return MAX_ACTIVE_CLUSTERS[index, cluster_ctas]


def cluster_ctas_for(pools: int, blocks: int, device: torch.device) -> int:
    """Return the CTAs a block is merged by - the most that keep every block's cluster on the GPU at once.

    A block's rounds are serial, so a few blocks leave most SMs idle unless each is split. Its CTAs each search a tile
    of rows or more, and the GPU runs one CTA an SM, its shared memory holding the block's key sums.
    A wider cluster is taken only while the GPU runs all the blocks' clusters of that width at once.

    Args:
        pools: Pools a block holds.
        blocks: Blocks merged at once.
        device: The GPU.
    """
    cluster_ctas = 1
    while cluster_ctas * 2 <= min(MOST_CLUSTER_CTAS, pools // TILE) and blocks <= max_active_clusters(
        device, cluster_ctas * 2
    ):
        cluster_ctas *= 2
    return cluster_ctas


def compiled(pools: int, cluster_ctas: int, kernel: type[WardClustering] = WardClustering) -> Any:
    """Return the Ward merge compiled once a pool count, cluster and kernel class, taking torch tensors as they are.

    Args:
        pools: Pools a block holds.
        cluster_ctas: CTAs a block is merged by.
        kernel: The merge's class - Hopper's, or one that takes its tiles on another GPU's MMA.
    """
    key = (pools, cluster_ctas, kernel.__name__)
    if key in COMPILED:
        return COMPILED[key]
    rows, blocks = cute.sym_int32(), cute.sym_int32()
    arguments = (
        make_fake_compact_tensor(cutlass.Float32, (rows, INDEX_DIM), stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(cutlass.Float32, (rows,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (rows,), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (blocks,), assumed_align=4),
        cutlass.Int32(0),
        cutlass.Float32(0.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED[key] = cute.compile(kernel(pools, cluster_ctas), *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


def fits_the_gpu(pool_keys: torch.Tensor) -> bool:
    """Tell whether every block's cluster of one CTA is resident at once - no more blocks than the GPU has SMs.

    A block of one CTA holds an SM's shared memory, so a wider cluster is capped by the clusters the GPU runs at once.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
    """
    return pool_keys.shape[0] <= torch.cuda.get_device_properties(pool_keys.device).multi_processor_count


def takes_blocks(pool_keys: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL Ward merge's rounds take these blocks on any GPU - tf32x3, 64 index dims, 64 to 512 pools.

    A block's CTA holds its key sums in an SM's shared memory, so blocks past the GPU's SMs queue behind one another
    where the Triton programs share an SM - 200 blocks of 512 pools merged in 3,678 us against 2,159.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    _, pools, index_dim = pool_keys.shape
    return (
        pool_keys.is_cuda
        and fits_the_gpu(pool_keys)
        and precision == "tf32x3"
        and index_dim == INDEX_DIM
        and TILE <= pools <= MOST_POOLS
        and pools & (pools - 1) == 0
    )


def supports(pool_keys: torch.Tensor, precision: str) -> bool:
    """Tell whether the CuTeDSL Ward merge takes these blocks - tf32x3, 64 index dims, 64 to 512 pools, Hopper.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        precision: The Gram matrix's dot.
    """
    return (
        pool_keys.is_cuda
        and dispatch.capability(pool_keys.device) == dispatch.HOPPER
        and takes_blocks(pool_keys, precision)
    )


def check_sizes(pool_keys: torch.Tensor, pool_sizes: torch.Tensor) -> None:
    """Raise unless the pool sizes are one to a pool of the keys, on their device.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].
        pool_sizes: Keys each pool holds [G, P], whole counts.

    Raises:
        ValueError: If the sizes are not [G, P] on the keys' device.
    """
    if pool_keys.ndim != 3 or tuple(pool_sizes.shape) != tuple(pool_keys.shape[:2]):
        raise ValueError(
            f"the pool sizes are one to a pool, [G, P] for pool keys [G, P, index_dim] - got sizes "
            f"{tuple(pool_sizes.shape)} for keys {tuple(pool_keys.shape)}"
        )
    if pool_sizes.device != pool_keys.device:
        raise ValueError(
            f"the pool sizes are on the pool keys' device - got sizes on {pool_sizes.device}, keys on {pool_keys.device}"
        )


def cluster_ward(
    pool_keys: torch.Tensor, pool_sizes: torch.Tensor, bucket_count: int, bucket_cap: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools down to `bucket_count` buckets - the Triton merge's labels.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32, as `supports` takes them.
        pool_sizes: Keys each pool holds [G, P] fp32, whole counts - the cap test is exact only for them.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.

    Raises:
        ValueError: If the sizes are not [G, P] on the keys' device, or the merge does not take these blocks.
    """
    check_sizes(pool_keys, pool_sizes)
    if not supports(pool_keys, "tf32x3"):
        raise ValueError(
            f"the CuTeDSL Ward merge takes tf32x3 blocks of {TILE} to {MOST_POOLS} pools, a power of two, of 64 "
            f"index dims, at most one block an SM, on a Hopper GPU - got {tuple(pool_keys.shape)} on {pool_keys.device}"
        )
    return launch(pool_keys, pool_sizes, bucket_count, bucket_cap, WardClustering)


def launch(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    kernel: type[WardClustering],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools with a kernel class.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32, as the class's GPU takes them.
        pool_sizes: Keys each pool holds [G, P] fp32, whole counts, as `check_sizes` takes them.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
        kernel: The merge's class.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.
    """
    blocks, pools, _ = pool_keys.shape
    labels = torch.empty(blocks * pools, dtype=torch.int64, device=pool_keys.device)
    reached = torch.empty(blocks, dtype=torch.int32, device=pool_keys.device)
    compiled(pools, cluster_ctas_for(pools, blocks, pool_keys.device), kernel)(
        pool_keys.float().contiguous().view(blocks * pools, -1),
        pool_sizes.float().contiguous().view(-1),
        labels,
        reached,
        bucket_count,
        float(bucket_cap),
    )
    return labels.view(blocks, pools), reached.bool()
