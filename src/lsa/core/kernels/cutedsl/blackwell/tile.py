"""A tile of 64 by 64 tf32x3 dots on Blackwell's `tcgen05` MMA, read back as the Hopper kernels read theirs.

The pairing and the Ward merge score a tile of 64 rows against 64 columns as Triton's tf32x3 dot does - rest by
rounding, rounding by rest, NaN to zero, rounding by rounding. On Hopper the three products run on the warpgroup
MMA into registers. Here they run on `tcgen05.mma` into tensor memory, as Triton's own Blackwell build issues them:
an M = 64, N = 64 tile, K = 8 an instruction, the first product overwriting the accumulator. The NaN-to-zero step reads
the accumulator into registers and stores it back between the second product and the third, as Triton does.

A thread's `tcgen05.ld.16x256b` registers hold the same rows and columns, in the same order, as its warpgroup MMA
accumulator would, so the rounds after the dots read them unchanged.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.blackwell_helpers as blackwell
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from lsa.core.kernels.cutedsl.hopper.tile import THREADS, TILE, HopperTile, accumulator_rows_by_columns

# The named barrier the tensor memory's address is published on - 0 is the CTA's own
TMEM_BARRIER = 1


@dsl_user_op
def fence_after_thread_sync(*, loc: Any = None, ip: Any = None) -> None:
    """Order the tcgen05 operations after a thread barrier behind everything before it."""
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def fence_before_thread_sync(*, loc: Any = None, ip: Any = None) -> None:
    """Order the tcgen05 operations before a thread barrier ahead of everything after it."""
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, loc=loc, ip=ip)


class BlackwellTile(HopperTile):
    """The tile of dots on `tcgen05`, mixed in ahead of a clustering's class - it overrides the Hopper tile's hooks."""

    def tile_storage(self, row_layout: cute.ComposedLayout, column_layout: cute.ComposedLayout) -> Any:
        """Return the struct of shared memory a tile owns - the splits, what the MMA signals on, and the tensor memory's address.

        Args:
            row_layout: Shared memory of a tile of rows, both splits.
            column_layout: Shared memory of a tile of columns, both splits.
        """

        @cute.struct
        class TileStorage:
            """Shared memory of a tile: the operand splits, the MMA's barrier and the tensor memory's address."""

            rows: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 1024]
            columns: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(column_layout)], 1024
            ]
            tile_barrier: cutlass.Int64
            tmem_address: cutlass.Int32

        return TileStorage

    def make_tile_mma(self) -> cute.TiledMma:
        """Return the MMA a tile of dots is taken on - Blackwell's tf32 MMA of one CTA, 64 by 64."""
        return blackwell.make_trivial_tiled_mma(
            cutlass.TFloat32,
            cutlass.TFloat32,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (TILE, TILE),
        )

    def tile_views(
        self,
        tiled_mma: cute.TiledMma,
        storage: Any,
        row_layout: cute.ComposedLayout,
        column_layout: cute.ComposedLayout,
        thread: Any,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """Take the tile's tensor memory and barrier, and return the MMA's views and the thread's accumulator.

        Args:
            tiled_mma: The MMA over a tile.
            storage: The CTA's shared memory.
            row_layout: Shared memory of a tile of rows, both splits.
            column_layout: Shared memory of a tile of columns, both splits.
            thread: The thread.

        Returns:
            The rows' splits as the MMA reads them, the columns' the same, the thread's accumulator registers, and
            each register's row and column in the tile as rows by columns.
        """
        mma_slice = tiled_mma.get_slice(0)
        tCrA = tiled_mma.make_fragment_A(
            mma_slice.partition_A(
                storage.tile.rows.get_tensor(
                    row_layout.outer, swizzle=row_layout.inner, dtype=cutlass.TFloat32
                )
            )
        )
        tCrB = tiled_mma.make_fragment_B(
            mma_slice.partition_B(
                storage.tile.columns.get_tensor(
                    column_layout.outer, swizzle=column_layout.inner, dtype=cutlass.TFloat32
                )
            )
        )
        self.tile_barrier = storage.tile.tile_barrier.ptr
        self.start_tiles(thread)
        self.tmem = cutlass.utils.TmemAllocator(
            storage.tile.tmem_address.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=TMEM_BARRIER, num_threads=THREADS),
            allocator_warp_id=0,
        )
        self.tmem.allocate(TILE)
        self.tmem.wait_for_alloc()
        self.tmem_pointer = self.tmem.retrieve_ptr(cutlass.Float32)
        layout = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((TILE, TILE))).layout
        self.dots = cute.make_tensor(self.tmem_pointer, layout)
        dots = self.dots[((None, None), 0, 0)]
        tiled_load = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32), dots
        )
        tiled_store = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.St16x256bOp(tcgen05.Repetition.x8), cutlass.Float32), dots
        )
        load_slice = tiled_load.get_slice(thread)
        self.tiled_load = tiled_load
        self.tiled_store = tiled_store
        self.loaded = load_slice.partition_S(dots)
        self.stored = tiled_store.get_slice(thread).partition_D(dots)
        places = load_slice.partition_D(cute.make_identity_tensor((TILE, TILE)))
        self.staged = cute.make_rmem_tensor(places.shape, cutlass.Float32)
        # The load's registers in the warpgroup accumulator's modes - the same registers in the same order
        # A warpgroup MMA m64n64 puts, in warp w, lane l, rows 16w + l // 4 and 16w + l // 4 + 8, columns
        # 8j + 2 * (l % 4) + {0, 1} for j in 0..7, in the order (column pair, row half, j)
        # tcgen05.ld.16x256b.x8 hands a lane of warp w the same rows of tensor memory's 16 lanes for warp w, and
        # per repetition j the same two columns of both rows, in the same order
        # => register i is the same cell, and `places` names it from the load's own partition
        flat = places[((None, 0), 0, 0)].layout
        coords = cute.make_tensor(
            places.iterator, cute.make_layout((flat.shape[0], 1, 1), stride=(flat.stride[0], 0, 0))
        )
        accumulator = cute.make_rmem_tensor(((2, 2, TILE // 8), 1, 1), cutlass.Float32)
        return tCrA, tCrB, accumulator, accumulator_rows_by_columns(coords)

    @cute.jit
    def start_tiles(self, thread: cutlass.Int32) -> None:
        """Initialise the barrier the tile's MMA signals on, before any thread waits on it.

        Args:
            thread: The thread.
        """
        if thread == 0:
            cute.arch.mbarrier_init(self.tile_barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

    @cute.jit
    def issue_products(
        self,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        stages: cutlass.Constexpr,
        first: cutlass.Constexpr,
    ) -> None:
        """Issue a tile's products of split pairs into tensor memory from one warp and signal their completion.

        Args:
            tiled_mma: The MMA over a tile.
            tCrA: The MMA's view of the rows' splits.
            tCrB: The MMA's view of the columns' splits.
            stages: The (row split, column split) pairs, in order.
            first: Whether the first product overwrites the accumulator.
        """
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            fence_after_thread_sync()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, not first)
            for row_stage, column_stage in stages:
                for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    cute.gemm(
                        tiled_mma,
                        self.dots,
                        tCrA[None, None, k, row_stage],
                        tCrB[None, None, k, column_stage],
                        self.dots,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(self.tile_barrier)

    @cute.jit
    def score_tile(
        self, tiled_mma: cute.TiledMma, tCrA: cute.Tensor, tCrB: cute.Tensor, accumulator: cute.Tensor
    ) -> None:
        """Take a tile's dots as Triton's tf32x3 does on Blackwell - rest by rounding, rounding by rest, NaN to zero, rounding by rounding.

        A call waits on the tile's barrier twice, so every call starts at the same phase.

        Args:
            tiled_mma: The MMA over a tile.
            tCrA: The MMA's view of the rows' splits.
            tCrB: The MMA's view of the columns' splits.
            accumulator: The tile's dots, overwritten.
        """
        staged = self.staged
        self.issue_products(tiled_mma, tCrA, tCrB, ((1, 0), (0, 1)), True)
        cute.arch.mbarrier_wait(self.tile_barrier, 0)
        cute.copy(self.tiled_load, self.loaded, staged)
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range_constexpr(cute.size(staged)):
            if staged[i] != staged[i]:
                staged[i] = cutlass.Float32(0.0)
        cute.copy(self.tiled_store, staged, self.stored)
        cute.arch.fence_view_async_tmem_store()
        fence_before_thread_sync()
        cute.arch.barrier()
        self.issue_products(tiled_mma, tCrA, tCrB, ((0, 0),), False)
        cute.arch.mbarrier_wait(self.tile_barrier, 1)
        cute.copy(self.tiled_load, self.loaded, staged)
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range_constexpr(cute.size(accumulator)):
            accumulator[i] = staged[i]
        # The accumulator is read again only after the next products, which wait on every thread's load
        fence_before_thread_sync()

    @cute.jit
    def release_tiles(self) -> None:
        """Give back the tile's tensor memory once every thread is past its last load."""
        cute.arch.barrier()
        self.tmem.relinquish_alloc_permit()
        self.tmem.free(self.tmem_pointer)
