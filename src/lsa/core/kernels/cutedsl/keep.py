"""A wide start-level keep in CuTeDSL - each query's best node buckets of a row too wide to stage, and their children.

A start level's scores are in hand, a row a query. The beam step stages a whole row in shared memory, which past
about 24k node buckets leaves one CTA an SM and loses to Triton. Here the short listing streams each row instead,
holding only its sample, and lists the columns at or above a sampled threshold in column order - a few times the
keep. One CTA a row then takes the exact best of that listing, a tie going to the lower column, and names each
kept node bucket's children side by side in column order, the beam step's output.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from lsa.core.kernels.cutedsl import top_k as cutedsl_top_k
from lsa.core.kernels.cutedsl.decode import BINS, WordSearch

# Keys a row's listing holds at most, their words in shared memory
MOST_LISTED = 8192
KEEP_THREADS = 256
KEEP_WARPS = KEEP_THREADS // 32

COMPILED: dict[int, Any] = {}


class StartLevelKeep(WordSearch):
    """Each row's exact best `keep` keys of its listing, as the children of the node buckets they name."""

    def __init__(self, node_bucket_size: int) -> None:
        """Fix the children a node bucket holds.

        Args:
            node_bucket_size: Children a node bucket holds.
        """
        super().__init__(KEEP_THREADS)
        self.node_bucket_size = node_bucket_size

    @cute.jit
    def __call__(
        self,
        keys: cute.Tensor,
        listed: cute.Tensor,
        children: cute.Tensor,
        out: cute.Tensor,
        keep: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a row.

        Args:
            keys: Each row's listing [R, W] int64, in column order.
            listed: Keys each row listed [R] int32.
            children: Each column's children [N, node_bucket_size] int32.
            out: The kept node buckets' children [R, keep * node_bucket_size] int32, written.
            keep: Columns a row keeps.
            stream: The stream to launch on.
        """

        @cute.struct
        class Storage:
            """Shared memory of a block: the listed words, their histogram, per-warp totals and scalars."""

            words: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MOST_LISTED], 16]
            histogram: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, BINS], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, KEEP_WARPS], 16]
            shared: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.kernel(keys, listed, children, out, keep, Storage).launch(
            grid=(keys.shape[0], 1, 1), block=[KEEP_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        keys: cute.Tensor,
        listed: cute.Tensor,
        children: cute.Tensor,
        out: cute.Tensor,
        keep: cutlass.Int32,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Find a row's keep-th word of its listing, then name the children of every key above it and its first ties.

        Args:
            keys: Each row's listing [R, W] int64.
            listed: Keys each row listed [R] int32.
            children: Each column's children [N, node_bucket_size] int32.
            out: The kept node buckets' children [R, keep * node_bucket_size] int32.
            keep: Columns a row keeps.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row_index, _, _ = cute.arch.block_idx()
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        words = storage.words.get_tensor(cute.make_layout(MOST_LISTED))
        histogram = storage.histogram.get_tensor(cute.make_layout(BINS))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(KEEP_WARPS))
        shared = storage.shared.get_tensor(cute.make_layout(4))
        row = keys[row_index, None]
        kept = out[row_index, None]
        length = listed[row_index]
        want = cutlass.min(keep, length)

        # A key's high half is its score in signed order - flipping the sign bit gives the unsigned word
        for i in range(thread, length, KEEP_THREADS):
            words[i] = cutlass.Int32(row[i] >> 32) ^ cutlass.Int32(-(2**31))
        self.barrier.arrive_and_wait()
        kth_word, above = self.kth(words, length, want, 4, False, histogram, warp_totals, shared)

        # Every word above the keep-th, and its ties lowest column first until the keep is full, in column order
        placed = cutlass.Int32(0)
        ties_taken = cutlass.Int32(0)
        for first in range(0, length, KEEP_THREADS):
            i = first + thread
            greater = False
            tied = cutlass.Int32(0)
            if i < length:
                word = cutlass.Uint32(words[i])
                greater = word > cutlass.Uint32(kth_word)
                tied = cutlass.Int32(word == cutlass.Uint32(kth_word))
            tie_before, tie_total = self.exclusive_sum(tied, warp_totals)
            take = cutlass.Int32(0)
            if greater or (tied != 0 and ties_taken + tie_before < want - above):
                take = cutlass.Int32(1)
            before, total = self.exclusive_sum(take, warp_totals)
            if take != 0:
                column = cutlass.Int32(0x7FFFFFFF) - cutlass.Int32(row[i] & cutlass.Int64(0xFFFFFFFF))
                for c in cutlass.range_constexpr(self.node_bucket_size):
                    kept[(placed + before) * self.node_bucket_size + c] = children[column, c]
            placed = placed + total
            ties_taken = ties_taken + tie_total
        for i in range(placed * self.node_bucket_size + thread, kept.shape[0], KEEP_THREADS):
            kept[i] = cutlass.Int32(-1)


def compiled(node_bucket_size: int) -> Any:
    """Return the keep compiled once a node bucket size, taking torch tensors as they are.

    Args:
        node_bucket_size: Children a node bucket holds.
    """
    if node_bucket_size in COMPILED:
        return COMPILED[node_bucket_size]
    rows, width, columns, kept_width = (cute.sym_int32() for _ in range(4))
    key_stride, out_stride = cute.sym_int64(), cute.sym_int64()
    arguments = (
        make_fake_tensor(cutlass.Int64, (rows, width), stride=(key_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_compact_tensor(
            cutlass.Int32, (columns, node_bucket_size), stride_order=(1, 0), assumed_align=4
        ),
        make_fake_tensor(cutlass.Int32, (rows, kept_width), stride=(out_stride, 1), assumed_align=4),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED[node_bucket_size] = cute.compile(
        StartLevelKeep(node_bucket_size), *arguments, options="--enable-tvm-ffi"
    )
    return COMPILED[node_bucket_size]


def supports(scores: torch.Tensor, children: torch.Tensor, keep: int) -> bool:
    """Tell whether a start-level keep lists its rows first - rows the listing narrows, to a listing that fits.

    Args:
        scores: The level's scores [S, N].
        children: Each column's children [N, node_bucket_size].
        keep: Columns a query keeps.
    """
    _, _, width = cutedsl_top_k.plan(keep, scores.shape[1])
    return (
        width < scores.shape[1]
        and width <= MOST_LISTED
        and cutedsl_top_k.supports(scores, keep)
        and children.dtype == torch.int32
        and children.is_contiguous()
    )


def keep_start_level_listed(scores: torch.Tensor, children: torch.Tensor, keep: int) -> torch.Tensor:
    """Keep each query's best `keep` columns of a start level's scores by listing each row first, and name their children.

    Args:
        scores: The level's scores [S, N] fp32, as `supports` takes them.
        children: Each column's children [N, node_bucket_size] int32.
        keep: Columns a query keeps.

    Returns:
        The kept node buckets' children [S, keep * node_bucket_size] int32 - a kept bucket's side by side in column
        order, -1 where a query kept nothing.

    Raises:
        ValueError: If the keep does not list these rows.
    """
    if not supports(scores, children, keep):
        raise ValueError(
            f"the listed keep takes fp32 rows the listing narrows to at most {MOST_LISTED} keys - got "
            f"{scores.dtype} {tuple(scores.shape)} for a keep of {keep}"
        )
    rows, columns = scores.shape
    lengths = torch.full((rows,), columns, device=scores.device, dtype=torch.int32)
    keys, listed = cutedsl_top_k.list_rows(scores, lengths, keep, ordered=True)
    out = torch.empty(rows, keep * children.shape[1], device=scores.device, dtype=torch.int32)
    if rows:
        compiled(children.shape[1])(keys, listed, children, out, keep)
    return out
