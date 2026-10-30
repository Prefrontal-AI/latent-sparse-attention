"""A tile of 64 by 64 tf32x3 merge costs on Hopper's warpgroup MMA - what the pairing and the Ward merge share.

Both score a tile of 64 rows against 64 columns as Triton's tf32x3 dot does, and turn each dot into a merge cost
the way Triton adds it. The tile is a class both clusterings inherit, so another GPU's MMA replaces it by overriding
its hooks - [Blackwell's](../blackwell/tile.py) does.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import cute_nvgpu, llvm
from cutlass.cute.atom import make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu.warpgroup.mma import MmaF16BF16Trait, MmaOp
from cutlass.cutlass_dsl import T, dsl_user_op

# Buckets a tile holds, and a warpgroup's threads
TILE = 64
THREADS = 128
WARPS = THREADS // 32
# Index dims the kernels take
INDEX_DIM = 64
# The cost key's low bits name the column, as the Triton key does
INDEX_BITS = 12
MAX_KEY = 0x7FFFFFFFFFFFFFFF


def accumulator_rows_by_columns(accumulator: cute.Tensor) -> cute.Tensor:
    """View a warpgroup MMA accumulator fragment as its rows by its columns.

    Args:
        accumulator: A thread's fragment of the MMA's C, ((2, 2, N / 8), M, N').

    Returns:
        The same registers as ((2, M), (2, N / 8, N')).
    """
    layout = cute.make_layout(accumulator.layout.shape)
    shape, stride = layout.shape, layout.stride
    rows_by_columns = cute.make_layout(
        ((shape[0][1], shape[1]), (shape[0][0], shape[0][2], shape[2])),
        stride=((stride[0][1], stride[1]), (stride[0][0], stride[0][2], stride[2])),
    )
    return cute.make_tensor(accumulator.iterator, cute.composition(accumulator.layout, rows_by_columns))


class MmaTF32Op(MmaOp):
    """The warpgroup MMA of tf32 operands from shared memory, K-major, into fp32 - Hopper's, which cutlass-dsl lacks."""

    descriptive_name = "warpgroup TF32 MMA Operation"

    def __init__(self, shape_mn: tuple[int, int]) -> None:
        """Fix the operands' types and majors.

        Args:
            shape_mn: The instruction's M and N.
        """
        super().__init__(
            cutlass.TFloat32,
            cutlass.TFloat32,
            cutlass.Float32,
            (*shape_mn, 8),
            warpgroup.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )

    def _make_trait(self, *, loc: Any = None, ip: Any = None, **kwargs: Any) -> MmaF16BF16Trait:
        """Return the MMA atom's trait, built as the F16 one is.

        Args:
            loc: The MLIR location.
            ip: The MLIR insertion point.
            kwargs: Unused.
        """
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        atom_type = cute_nvgpu.MmaAtomSM90Type.get(
            shape_mnk.type.attribute,
            self.a_major_mode._to_ir(),
            self.b_major_mode._to_ir(),
            self.a_dtype.mlir_type,
            self.b_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.a_src._to_ir(),
        )
        return MmaF16BF16Trait(
            make_atom(atom_type, [cutlass.Boolean(False).ir_value(loc=loc, ip=ip)], loc=loc, ip=ip)
        )


@dsl_user_op
def round_tf32(value: Any, *, loc: Any = None, ip: Any = None) -> Any:
    """Return an fp32 value rounded to tf32, to nearest with ties away, as Triton splits a tf32x3 operand.

    Args:
        value: The fp32 value.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rna.tf32.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def approximate_sqrt(value: Any, *, loc: Any = None, ip: Any = None) -> Any:
    """Return the approximate square root Triton's `tl.sqrt` compiles to.

    Args:
        value: The fp32 value.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def ordered_cost(cost: Any) -> Any:
    """Return an fp32 cost's bits as an int32 that orders as the cost does.

    Args:
        cost: The fp32 cost.
    """
    bits = cutlass.Int32(llvm.bitcast(T.i32(), cutlass.Float32(cost).ir_value()))
    ordered = bits
    if bits < 0:
        ordered = bits ^ cutlass.Int32(0x7FFFFFFF)
    return ordered


@cute.jit
def pack_cost_key(ordered: Any, column: Any) -> Any:
    """Return a cost's key - its ordered bits high and the column low, as the Triton pairing packs them.

    Args:
        ordered: The cost's ordered bits.
        column: The partner's place among the open buckets.
    """
    return ((cutlass.Int64(ordered) + cutlass.Int64(1 << 31)) << INDEX_BITS) | cutlass.Int64(column)


class HopperTile:
    """A tile of 64 rows by 64 columns of tf32x3 merge costs, scored on Hopper's warpgroup MMA.

    A clustering inherits the tile and calls its methods from a kernel of one warpgroup - 128 threads - a CTA.
    A GPU with another MMA subclasses it and overrides the hooks below, and nothing else.

    The hooks, in the order a kernel calls them:

    - `tile_storage(row_layout, column_layout)` returns the struct of shared memory the tile owns - the rows' and
      columns' splits, and whatever its MMA signals on. The clustering holds it as the field `tile` of its own.
    - `make_tile_mma()` returns the MMA a tile of dots is taken on.
    - `tile_views(...)` returns the MMA's views of the splits, the thread's accumulator and `coords`, each
      register's row and column in the tile as rows by columns.
    - `score_tile(...)` overwrites the accumulator with the tile's tf32x3 dots, for every thread of the warpgroup
      at once, and leaves it readable on return.
    - `release_tiles()` gives back what the MMA held, once, from every thread at the end of the kernel.

    The accumulator is 32 fp32 registers of shape ((2, 2, TILE // 8), 1, 1) on any GPU, and `coords` names each
    one's row and column. The order of the registers is the guarantee: a thread's register i is the same cell of
    the tile whichever MMA filled it, so every step after the dots reads `accumulator_rows_by_columns(accumulator)`
    and `coords` and never names a GPU.
    """

    def tile_storage(self, row_layout: cute.ComposedLayout, column_layout: cute.ComposedLayout) -> Any:
        """Return the struct of shared memory a tile owns - the splits of its rows and of its columns.

        Args:
            row_layout: Shared memory of a tile of rows, both splits.
            column_layout: Shared memory of a tile of columns, both splits.
        """

        @cute.struct
        class TileStorage:
            """Shared memory of a tile: the operand splits."""

            rows: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 1024]
            columns: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(column_layout)], 1024
            ]

        return TileStorage

    def make_tile_mma(self) -> cute.TiledMma:
        """Return the MMA a tile of dots is taken on - Hopper's warpgroup tf32 MMA."""
        return cute.make_tiled_mma(cute.make_mma_atom(MmaTF32Op((TILE, TILE))), (1, 1, 1))

    def tile_views(
        self,
        tiled_mma: cute.TiledMma,
        storage: Any,
        row_layout: cute.ComposedLayout,
        column_layout: cute.ComposedLayout,
        thread: Any,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """Return the MMA's views of a tile's splits, the thread's accumulator and its coordinates.

        Args:
            tiled_mma: The MMA over a tile.
            storage: The CTA's shared memory, holding the tile's as `tile`.
            row_layout: Shared memory of a tile of rows, both splits.
            column_layout: Shared memory of a tile of columns, both splits.
            thread: The thread.

        Returns:
            The rows' splits as the MMA reads them, the columns' the same, the thread's accumulator registers, and
            each register's row and column in the tile as rows by columns.
        """
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        mma_slice = tiled_mma.get_slice(thread)
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
        accumulator = cute.make_rmem_tensor(mma_slice.partition_shape_C((TILE, TILE)), cutlass.Float32)
        coords = accumulator_rows_by_columns(mma_slice.partition_C(cute.make_identity_tensor((TILE, TILE))))
        return tCrA, tCrB, accumulator, coords

    def release_tiles(self) -> None:
        """Give back what the tiles' MMA held - nothing on Hopper."""

    @cute.jit
    def load_split(
        self,
        key_sums: cute.Tensor,
        order: cute.Tensor,
        base: Any,
        order_base: Any,
        first: Any,
        count: Any,
        staged: cute.Tensor,
    ) -> None:
        """Load this thread's half row of a tile of open buckets' key sums into registers.

        Args:
            key_sums: Each bucket's key sum [G * P, index_dim] fp32.
            order: The lists of open buckets.
            base: The block's first bucket.
            order_base: The list read's first entry.
            first: The tile's first place in the list.
            count: Buckets open.
            staged: The half row [index_dim / 2] fp32, written.
        """
        thread, _, _ = cute.arch.thread_idx()
        place = first + thread // 2
        # A place past the open count loads bucket 0, as Triton's masked load does - its scores are never read
        bucket = cutlass.Int32(0)
        if place < count:
            bucket = order[order_base + base + place]
        cute.autovec_copy(
            cute.local_tile(key_sums[base + bucket, None], (INDEX_DIM // 2,), (thread % 2,)), staged
        )

    @cute.jit
    def store_split(self, staged: cute.Tensor, splits: cute.Tensor) -> None:
        """Store this thread's half row into shared memory, split into its tf32 rounding and the rest.

        Args:
            staged: The half row [index_dim / 2] fp32.
            splits: The tile's two stages in shared memory.
        """
        thread, _, _ = cute.arch.thread_idx()
        row = thread // 2
        first_dim = (thread % 2) * (INDEX_DIM // 2)
        for d in cutlass.range_constexpr(INDEX_DIM // 2):
            rounded = round_tf32(staged[d])
            splits[row, first_dim + d, 0] = rounded
            splits[row, first_dim + d, 1] = staged[d] - rounded

    @cute.jit
    def score_tile(
        self, tiled_mma: cute.TiledMma, tCrA: cute.Tensor, tCrB: cute.Tensor, accumulator: cute.Tensor
    ) -> None:
        """Take a tile's dots as Triton's tf32x3 does - rest by rounding, rounding by rest, NaN to zero, rounding by rounding.

        Triton lowers a tf32x3 dot, with each operand split into its tf32 rounding "big" and the rest "small", as
        small x big, big x small, NaN to zero, big x big, and then adds the accumulator C. Small x small is dropped.
        The three products run in that order, each into the one accumulator, so every sum rounds as Triton's does.
        The hierarchy a growth builds must be identical under the kernels `cutedsl` and `triton`, and a different
        rounding in one cost can pair a different bucket - so the dots are bit for bit, not close.

        Args:
            tiled_mma: The warpgroup's MMA over a tile.
            tCrA: The MMA's view of the rows' splits.
            tCrB: The MMA's view of the columns' splits.
            accumulator: The tile's dots, overwritten.
        """
        accumulator.fill(0.0)
        # Small x big, then big x small
        for row_stage, column_stage in ((1, 0), (0, 1)):
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    accumulator,
                    tCrA[None, None, k, row_stage],
                    tCrB[None, None, k, column_stage],
                    accumulator,
                )
            warpgroup.commit_group()
            warpgroup.wait_group(0)
        for i in cutlass.range_constexpr(cute.size(accumulator)):
            if accumulator[i] != accumulator[i]:
                accumulator[i] = cutlass.Float32(0.0)
        # Big x big
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(tiled_mma, accumulator, tCrA[None, None, k, 0], tCrB[None, None, k, 0], accumulator)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

    @cute.jit
    def merge_cost(
        self, dot: Any, row_square: Any, row_alignment: Any, column_square: Any, column_alignment: Any
    ) -> Any:
        """Return the alignment a merge loses, in the order the Triton pairing adds it.

        Args:
            dot: The two key sums' dot.
            row_square: The row's squared length.
            row_alignment: The row's length.
            column_square: The column's squared length.
            column_alignment: The column's length.
        """
        # Triton adds its zero accumulator C after the third product
        # => the same add here; it can only turn a dot of -0 into +0, and the alignments are at least +0, so
        #    `row_alignment + column_alignment - merged` is the same cost, and so the same key, either way
        merged = approximate_sqrt(
            cutlass.max(
                (row_square + column_square) + (dot + cutlass.Float32(0.0)) * cutlass.Float32(2.0),
                cutlass.Float32(0.0),
            )
        )
        return (row_alignment + column_alignment) - merged

    @cute.jit
    def exclusive_sum(self, value: Any, warp_totals: cute.Tensor) -> tuple[Any, Any]:
        """Return the sum of value over the threads before this one, and over all of them.

        Args:
            value: This thread's value.
            warp_totals: Each warp's running total, written here.
        """
        thread, _, _ = cute.arch.thread_idx()
        lane = thread % 32
        warp = thread // 32
        inclusive = value
        for i in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(inclusive, 1 << i, mask_and_clamp=0)
            if lane >= (1 << i):
                inclusive = inclusive + other
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.barrier()
        before = inclusive - value
        total = cutlass.Int32(0)
        for w in cutlass.range_constexpr(WARPS):
            if w < warp:
                before = before + warp_totals[w]
            total = total + warp_totals[w]
        cute.arch.barrier()
        return before, total
