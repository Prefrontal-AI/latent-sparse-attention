"""A scored row kept by a thread-block cluster - one query's row spread over the CTAs of a cluster, not held by one.

The keep and the pick of `beam` run one CTA a query, and a wide row's radix passes and its compaction then run on one SM while the rest of the GPU idles.
Here a cluster of CTAs shares a query.
Each holds a slice of the row's words in shared memory and counts its slice's digit of every pass.
Every CTA adds the cluster's counts by reading its peers' histograms over distributed shared memory.
Each CTA finds the same digit, so the search ends in every CTA at the same threshold word and the same count of ties to keep.
Once two passes leave a few words under the digits found, the CTAs list them and each ranks the want-th greatest among them in place of the last two passes.

A slice is a run of columns, in column order, so the kept columns of the cluster in column order are those of its first CTA, then its second, and so on.
A CTA counts the columns above the threshold and those at it, and its peers' counts place its columns among the kept.
It writes the children of the columns it holds.
The columns kept are the single CTA's - the greatest by score, a tie to the lower column.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from lsa.core.kernels.cutedsl import beam

# A cluster larger than 8 needs an opt-in and ran slower
# => 8 CTAs
CLUSTER = 8
# More threads or fewer ran slower, and below this many columns the single CTA is as fast
# => 512 threads, spread from 4,096 columns
SPREAD_THREADS = 512
SPREAD_FROM = 4096
# Past this many rows the clusters of 8 queue for SMs
# => clusters of 4
FEW_ROWS = 8
# A word is 4 bytes
# => 4 passes of a byte each
PASSES = 4
# Two passes leave a few words under the digits found on a row of scores without heavy ties
# => at most this many are listed and ranked, one barrier in place of two passes
GATHER_MAX = 256
# The slice follows a histogram a pass, their sum, each warp's two counts, the state and the words listed and gathered
# => these fixed words, to 16 bytes
STATE_WORDS = 16
FIXED_WORDS = (
    (PASSES * beam.BINS + beam.BINS + 2 * beam.MOST_WARPS + STATE_WORDS + 2 * GATHER_MAX + 3) // 4 * 4
)
# The state's words - the prefix and the rank left at it before each pass and after the last, then the slice's columns
# above the level and at it, the words under the digits found by the second pass and this CTA's share of them
# => four words from index 2 * PASSES + 2
SLICE_ABOVE, SLICE_TIED, BIN_COUNT, LISTED = (2 * PASSES + i for i in range(2, 6))


@dsl_user_op
def load_peer(pointer: cute.Pointer, peer: int, *, loc: Any = None, ip: Any = None) -> cutlass.Int32:
    """Load a word of shared memory from another CTA of the cluster.

    Args:
        pointer: The word's address in this CTA's shared memory.
        peer: The CTA, by its rank in the cluster.
        loc: The source location.
        ip: The insertion point.

    Returns:
        The word, an int32.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [pointer.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "{ .reg .u32 remote; mapa.shared::cluster.u32 remote, $1, $2; ld.shared::cluster.u32 $0, [remote]; }",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


class SpreadStep(beam.BeamStep):
    """A scored row's search and compaction over a cluster, the slice a CTA holds in shared memory."""

    def __init__(self, element_bytes: int, node_bucket_size: int, threads: int, cluster: int) -> None:
        """Fix the cluster's CTAs and a CTA's threads.

        Args:
            element_bytes: Bytes of a centroid's element - 1 for fp8, 2 for bf16.
            node_bucket_size: Children a kept column names.
            threads: Threads of a CTA.
            cluster: CTAs of a cluster.
        """
        super().__init__(element_bytes, node_bucket_size, threads, scored=True)
        self.cluster = cluster

    @cute.jit
    def sum_histograms(self, histograms: cute.Tensor, summed: cute.Tensor, p: int) -> None:
        """Add a pass's histogram over the cluster's CTAs into this CTA's `summed`, once every CTA has counted.

        Args:
            histograms: This CTA's histograms [PASSES * BINS] in shared memory.
            summed: The pass's counts over the cluster [BINS] in shared memory.
            p: The pass.
        """
        thread, _, _ = cute.arch.thread_idx()
        if thread < beam.BINS:
            # The loads are all in flight before the first add
            peers = cute.make_rmem_tensor(self.cluster, cutlass.Int32)
            for c in cutlass.range_constexpr(self.cluster):
                peers[c] = load_peer(histograms.iterator + p * beam.BINS + thread, c)
            total = cutlass.Int32(0)
            for c in cutlass.range_constexpr(self.cluster):
                total = total + peers[c]
            summed[thread] = total

    @cute.jit
    def find_digit_of_sum(
        self, lane: cutlass.Int32, summed: cute.Tensor, state: cute.Tensor, p: int, prefix: cutlass.Uint32
    ) -> None:
        """Find a pass's digit of the want-th greatest word in one warp, and how many of the want are below it.

        Lane l holds the bins 8l to 8l + 7 counted from the top.

        Args:
            lane: This lane.
            summed: The pass's counts over the cluster [BINS] in shared memory.
            state: The search's state - the prefix and the rank before each pass and after it.
            p: The pass.
            prefix: The digits found so far.
        """
        remaining = state[2 * p + 1]
        lane_total = cutlass.Int32(0)
        for b in cutlass.range_constexpr(beam.BINS_PER_LANE):
            lane_total = lane_total + summed[beam.BINS - 1 - (lane * beam.BINS_PER_LANE + b)]
        before = lane_total
        for i in cutlass.range_constexpr(5):
            # A shuffle up clamps at lane 0, which cutlass-dsl 4.6.2 does not default to
            other = cute.arch.shuffle_sync_up(before, 1 << i, mask_and_clamp=0)
            if lane >= (1 << i):
                before = before + other
        before = before - lane_total
        if before < remaining and before + lane_total >= remaining:
            above = before
            digit = cutlass.Int32(0)
            found = cutlass.Int32(0)
            in_digit = cutlass.Int32(0)
            for b in cutlass.range_constexpr(beam.BINS_PER_LANE):
                bin_index = beam.BINS - 1 - (lane * beam.BINS_PER_LANE + b)
                count = summed[bin_index]
                if found == 0 and above + count >= remaining:
                    digit = bin_index
                    in_digit = count
                    found = 1
                if found == 0:
                    above = above + count
            state[BIN_COUNT] = in_digit
            state[2 * p + 2] = cutlass.Int32(prefix | (cutlass.Uint32(digit) << (24 - 8 * p)))
            state[2 * p + 3] = remaining - above

    @cute.jit
    def load_slice(
        self,
        scores: cute.Tensor,
        row: cutlass.Int32,
        start: cutlass.Int32,
        mine: cutlass.Int32,
        words: cute.Tensor,
    ) -> None:
        """Copy this CTA's slice of a query's scores into shared memory as words, its loads in flight before its stores.

        Args:
            scores: The row's scores [S, W] fp32.
            row: The query.
            start: The slice's first column.
            mine: Columns the slice holds.
            words: The slice's words [mine] in shared memory.
        """
        thread, _, _ = cute.arch.thread_idx()
        loaded_scores = cute.make_rmem_tensor(beam.COPY_IN_FLIGHT, cutlass.Float32)
        for base in range(0, mine, self.threads * beam.COPY_IN_FLIGHT):
            for f in cutlass.range_constexpr(beam.COPY_IN_FLIGHT):
                i = base + f * self.threads + thread
                loaded_scores[f] = cutlass.Float32(0)
                if i < mine:
                    loaded_scores[f] = scores[row, start + i]
            for f in cutlass.range_constexpr(beam.COPY_IN_FLIGHT):
                i = base + f * self.threads + thread
                if i < mine:
                    words[i] = self.word_of(loaded_scores[f])

    @cute.jit
    def count_pass(
        self,
        p: int,
        words: cute.Tensor,
        mine: cutlass.Int32,
        histograms: cute.Tensor,
        summed: cute.Tensor,
        state: cute.Tensor,
    ) -> None:
        """Run one pass of the search over the cluster - each CTA counts its slice, then all find the same digit.

        Args:
            p: The pass.
            words: This CTA's words [slice] in shared memory.
            mine: Words the slice holds.
            histograms: This CTA's histograms [PASSES * BINS] in shared memory.
            summed: The pass's counts over the cluster [BINS] in shared memory.
            state: The search's state - the prefix and the rank before each pass and after it.
        """
        thread, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        shift = 24 - 8 * p
        prefix = cutlass.Uint32(state[2 * p])
        high = cutlass.Uint32(0)
        if cutlass.const_expr(p > 0):
            high = cutlass.Uint32(0xFFFFFFFF) << (shift + 8)
        for i in range(thread, mine, self.threads):
            word = cutlass.Uint32(words[i])
            if (word & high) == (prefix & high):
                cute.arch.atomic_add(
                    (histograms.iterator + p * beam.BINS + cutlass.Int32((word >> shift) & 255)).llvm_ptr,
                    cutlass.Int32(1),
                    scope="cta",
                )
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        self.sum_histograms(histograms, summed, p)
        self.barrier.arrive_and_wait()
        if thread < 32:
            self.find_digit_of_sum(lane, summed, state, p, prefix)
        self.barrier.arrive_and_wait()

    @cute.jit
    def rank_candidates(
        self,
        words: cute.Tensor,
        mine: cutlass.Int32,
        state: cute.Tensor,
        listing: cute.Tensor,
        candidates: cute.Tensor,
    ) -> None:
        """Find the want-th greatest word among the few the first two passes leave, by listing and ranking them.

        Each CTA lists its words under the two digits found, and every CTA reads all the lists.
        A word's rank among them is the words greater than it.
        The word at the want-th rank and the ties it still takes go where the last pass would have put them.

        Args:
            words: This CTA's words [slice] in shared memory.
            mine: Words the slice holds.
            state: The search's state - the prefix and rank after the second pass, and the words under it.
            listing: This CTA's listed words [GATHER_MAX] in shared memory.
            candidates: Every CTA's listed words [GATHER_MAX] in shared memory.
        """
        thread, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        top = cutlass.Uint32(state[4]) >> 16
        for begin in range(thread - lane, mine, self.threads):
            i = begin + lane
            word = cutlass.Uint32(0)
            if i < mine:
                word = cutlass.Uint32(words[i])
            under = i < mine and (word >> 16) == top
            mask = cute.arch.vote_ballot_sync(under)
            base = cutlass.Int32(0)
            if lane == 0 and mask != 0:
                base = cute.arch.atomic_add(
                    (state.iterator + LISTED).llvm_ptr, cutlass.Int32(cute.arch.popc(mask)), scope="cta"
                )
            base = cute.arch.shuffle_sync(base, 0)
            if under:
                listing[base + cutlass.Int32(cute.arch.popc(mask & cute.arch.lanemask_lt()))] = word
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        total = state[BIN_COUNT]
        if thread < total:
            # Word `thread` of the lists end to end, the CTAs' in order
            listed = cute.make_rmem_tensor(self.cluster, cutlass.Int32)
            for c in cutlass.range_constexpr(self.cluster):
                listed[c] = load_peer(state.iterator + LISTED, c)
            place = cutlass.Int32(thread)
            owner = cutlass.Int32(0)
            offset = cutlass.Int32(0)
            found = cutlass.Int32(0)
            for c in cutlass.range_constexpr(self.cluster):
                if found == 0 and place < listed[c]:
                    owner = cutlass.Int32(c)
                    offset = place
                    found = 1
                if found == 0:
                    place = place - listed[c]
            candidates[thread] = cutlass.Uint32(load_peer(listing.iterator + offset, owner))
        self.barrier.arrive_and_wait()
        if thread < total:
            word = cutlass.Uint32(candidates[thread])
            greater = cutlass.Int32(0)
            equal = cutlass.Int32(0)
            for i in range(total):
                other = cutlass.Uint32(candidates[i])
                greater = greater + cutlass.Int32(other > word)
                equal = equal + cutlass.Int32(other == word)
            remaining = state[2 * 2 + 1]
            if greater < remaining and remaining <= greater + equal:
                state[2 * PASSES] = cutlass.Int32(word)
                state[2 * PASSES + 1] = remaining - greater
        self.barrier.arrive_and_wait()

    @cute.jit
    def spread_columns(
        self, scores: cute.Tensor, row: cutlass.Int32, count: cutlass.Int32, keep: cutlass.Int32
    ) -> tuple[cute.Tensor, cute.Tensor, cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Find the keep-th greatest word of a query's row over the cluster and stage this CTA's kept columns.

        Args:
            scores: The row's scores [S, W] fp32.
            row: The query.
            count: Columns the query holds.
            keep: Columns a query keeps.

        Returns:
            This CTA's words [slice] and its kept columns [kept] in shared memory, in column order, as columns of the
            slice - the slice's first column, the columns this CTA keeps, and the columns kept by the CTAs before it.
        """
        thread, _, _ = cute.arch.thread_idx()
        rank = cute.arch.block_idx_in_cluster()
        lane = cute.arch.lane_idx()
        warp = thread // 32
        slice_size = (count + self.cluster - 1) // self.cluster
        start = cutlass.min(rank * slice_size, count)
        mine = cutlass.min(slice_size, count - start)

        counts = cute.arch.get_dyn_smem(cutlass.Int32, 16)
        histograms = cute.make_tensor(counts, cute.make_layout(PASSES * beam.BINS))
        summed = cute.make_tensor(counts + PASSES * beam.BINS, cute.make_layout(beam.BINS))
        warp_above = cute.make_tensor(summed.iterator + beam.BINS, cute.make_layout(beam.MOST_WARPS))
        warp_tied = cute.make_tensor(warp_above.iterator + beam.MOST_WARPS, cute.make_layout(beam.MOST_WARPS))
        state = cute.make_tensor(warp_tied.iterator + beam.MOST_WARPS, cute.make_layout(STATE_WORDS))
        listing = cute.recast_tensor(
            cute.make_tensor(state.iterator + STATE_WORDS, cute.make_layout(GATHER_MAX)), cutlass.Uint32
        )
        candidates = cute.recast_tensor(
            cute.make_tensor(listing.iterator + GATHER_MAX, cute.make_layout(GATHER_MAX)), cutlass.Uint32
        )
        words = cute.recast_tensor(
            cute.make_tensor(counts + FIXED_WORDS, cute.make_layout(slice_size)), cutlass.Uint32
        )
        kept = cute.recast_tensor(
            cute.make_tensor(counts + FIXED_WORDS + slice_size, cute.make_layout(beam.kept_words(keep))),
            cutlass.Uint16,
        )

        for i in range(thread, PASSES * beam.BINS, self.threads):
            histograms[i] = cutlass.Int32(0)
        want = cutlass.min(count, keep)
        if thread < STATE_WORDS:
            state[thread] = cutlass.Int32(0)
            if thread == 1:
                state[thread] = want
        self.load_slice(scores, row, start, mine, words)
        self.barrier.arrive_and_wait()

        for p in cutlass.range_constexpr(2):
            self.count_pass(p, words, mine, histograms, summed, state)
        # A few words left under the digits found are ranked, else the last two passes search them
        if state[BIN_COUNT] <= GATHER_MAX:
            self.rank_candidates(words, mine, state, listing, candidates)
        else:
            for p in cutlass.range_constexpr(2, PASSES):
                self.count_pass(p, words, mine, histograms, summed, state)
        level = cutlass.Uint32(state[2 * PASSES])
        ties = state[2 * PASSES + 1]

        # Each warp walks its own run of the slice; one count of every run places them all
        run = ((mine + self.warps - 1) // self.warps + 31) // 32 * 32
        first = warp * run
        last = cutlass.min(first + run, mine)
        above_count = cutlass.Int32(0)
        tied_count = cutlass.Int32(0)
        for i in range(first + lane, last, 32):
            word = cutlass.Uint32(words[i])
            above_count = above_count + cutlass.Int32(word > level)
            tied_count = tied_count + cutlass.Int32(word == level)
        above_count = self.warp_sum(above_count)
        tied_count = self.warp_sum(tied_count)
        if lane == 0:
            warp_above[warp] = above_count
            warp_tied[warp] = tied_count
        self.barrier.arrive_and_wait()

        # The slice's counts, shared with the CTAs after it
        slice_above = cutlass.Int32(0)
        slice_tied = cutlass.Int32(0)
        for w in cutlass.range_constexpr(self.warps):
            slice_above = slice_above + warp_above[w]
            slice_tied = slice_tied + warp_tied[w]
        if thread == 0:
            state[SLICE_ABOVE] = slice_above
            state[SLICE_TIED] = slice_tied
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # What the CTAs before this one keep: their columns above the level, and the ties the keep still takes
        peer_counts = cute.make_rmem_tensor(2 * self.cluster, cutlass.Int32)
        for c in cutlass.range_constexpr(self.cluster):
            peer_counts[2 * c] = cutlass.Int32(0)
            peer_counts[2 * c + 1] = cutlass.Int32(0)
            if c < rank:
                peer_counts[2 * c] = load_peer(state.iterator + SLICE_ABOVE, c)
                peer_counts[2 * c + 1] = load_peer(state.iterator + SLICE_TIED, c)
        tied_before_slice = cutlass.Int32(0)
        kept_before = cutlass.Int32(0)
        for c in cutlass.range_constexpr(self.cluster):
            kept_before = (
                kept_before
                + peer_counts[2 * c]
                + cutlass.min(cutlass.max(ties - tied_before_slice, 0), peer_counts[2 * c + 1])
            )
            tied_before_slice = tied_before_slice + peer_counts[2 * c + 1]
        # A peer reads this CTA's state until it arrives; the next wait is the last thing this CTA does
        cute.arch.cluster_arrive()
        tied_before = tied_before_slice
        filled = cutlass.Int32(0)
        for w in cutlass.range_constexpr(self.warps):
            if w < warp:
                filled = (
                    filled + warp_above[w] + cutlass.min(cutlass.max(ties - tied_before, 0), warp_tied[w])
                )
                tied_before = tied_before + warp_tied[w]
        for begin in range(first, last, 32):
            i = begin + lane
            inside = i < last
            word = cutlass.Uint32(0)
            if inside:
                word = cutlass.Uint32(words[i])
            tied = inside and word == level
            tied_mask = cute.arch.vote_ballot_sync(tied)
            tie_rank = tied_before + cutlass.Int32(cute.arch.popc(tied_mask & cute.arch.lanemask_lt()))
            chosen = inside and (word > level or (tied and tie_rank < ties))
            chosen_mask = cute.arch.vote_ballot_sync(chosen)
            slot = filled + cutlass.Int32(cute.arch.popc(chosen_mask & cute.arch.lanemask_lt()))
            if chosen:
                kept[slot] = cutlass.Uint16(i)
            tied_before = tied_before + cutlass.Int32(cute.arch.popc(tied_mask))
            filled = filled + cutlass.Int32(cute.arch.popc(chosen_mask))
        self.barrier.arrive_and_wait()
        mine_kept = slice_above + cutlass.min(cutlass.max(ties - tied_before_slice, 0), slice_tied)
        return words, kept, start, mine_kept, kept_before


class SpreadKeep(SpreadStep):
    """The keep of a scored row over a cluster - the children of the best columns, in column order."""

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        keep: cutlass.Int32,
        shared_bytes: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a cluster a query.

        Args:
            scores: The row's scores [S, W] fp32, read to each query's length.
            candidates: The rows of the level's pool each column names [S, W] int32, -1 for none.
            closed_children: The pool's children [N, node_bucket_size] int32.
            out: The kept columns' children [S, keep * node_bucket_size] int32.
            lengths: Columns each query holds [S] int32.
            keep: Columns a query keeps, the beam's width.
            shared_bytes: Shared memory a CTA takes - FIXED_WORDS, its slice and its kept columns.
            stream: The stream to launch on.
        """
        self.kernel(scores, candidates, closed_children, out, lengths, keep).launch(
            grid=(scores.shape[0] * self.cluster, 1, 1),
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=shared_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        closed_children: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        keep: cutlass.Int32,
    ) -> None:
        """Find the keep-th greatest score of one query's row over the cluster and write the kept columns' children.

        Args:
            scores: The row's scores [S, W] fp32.
            candidates: The rows each column names [S, W] int32.
            closed_children: The pool's children [N, node_bucket_size] int32.
            out: The kept columns' children [S, keep * node_bucket_size] int32.
            lengths: Columns each query holds [S] int32.
            keep: Columns a query keeps.
        """
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        rank = cute.arch.block_idx_in_cluster()
        row = block // self.cluster
        count = cutlass.min(cutlass.Int32(scores.shape[1]), cutlass.max(lengths[row], 0))
        words, kept, start, mine_kept, kept_before = self.spread_columns(scores, row, count, keep)
        children = cute.make_rmem_tensor(self.node_bucket_size, cutlass.Int32)
        for slot in range(thread, mine_kept, self.threads):
            column = start + cutlass.Int32(kept[slot])
            node = cutlass.Int32(-1)
            if cutlass.Uint32(words[column - start]) > cutlass.Uint32(beam.NEG_INF_WORD):
                node = candidates[row, column]
            # The loads are all in flight before the first store
            for c in cutlass.range_constexpr(self.node_bucket_size):
                children[c] = cutlass.Int32(-1)
                if node >= 0:
                    children[c] = closed_children[node, c]
            for c in cutlass.range_constexpr(self.node_bucket_size):
                out[row, (kept_before + slot) * self.node_bucket_size + c] = children[c]
        want = cutlass.min(count, keep)
        for slot in range(want + rank * self.threads + thread, keep, self.cluster * self.threads):
            for c in cutlass.range_constexpr(self.node_bucket_size):
                out[row, slot * self.node_bucket_size + c] = cutlass.Int32(-1)
        cute.arch.cluster_wait()


class SpreadPick(SpreadStep):
    """The pick of a scored row over a cluster - the best columns written as the keys the expansion reads."""

    def __init__(self, threads: int, cluster: int) -> None:
        """Fix the cluster's CTAs and a CTA's threads.

        Args:
            threads: Threads of a CTA.
            cluster: CTAs of a cluster.
        """
        super().__init__(1, 1, threads, cluster)

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        listed: cute.Tensor,
        shared_bytes: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a cluster a query.

        Args:
            scores: The row's scores [S, W] fp32, read to each query's length.
            candidates: The cache row of each column [S, W] int32, -1 for none.
            lengths: Columns each query holds [S] int32.
            out: Each query's picked keys [S, keep] int64.
            listed: Keys each query lists [S] int32 - the keep, written here.
            shared_bytes: Shared memory a CTA takes - FIXED_WORDS, its slice and its kept columns.
            stream: The stream to launch on.
        """
        self.kernel(scores, candidates, lengths, out, listed).launch(
            grid=(scores.shape[0] * self.cluster, 1, 1),
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=shared_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        candidates: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        listed: cute.Tensor,
    ) -> None:
        """Find the top-p-th greatest score of one query's row over the cluster and write the kept keys.

        Args:
            scores: The row's scores [S, W] fp32.
            candidates: The cache row of each column [S, W] int32.
            lengths: Columns each query holds [S] int32.
            out: Each query's picked keys [S, keep] int64.
            listed: Keys each query lists [S] int32.
        """
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        rank = cute.arch.block_idx_in_cluster()
        row = block // self.cluster
        keep = out.shape[1]
        count = cutlass.min(cutlass.Int32(scores.shape[1]), cutlass.max(lengths[row], 0))
        words, kept, start, mine_kept, kept_before = self.spread_columns(scores, row, count, keep)
        for slot in range(thread, mine_kept, self.threads):
            column = start + cutlass.Int32(kept[slot])
            word = cutlass.Uint32(words[column - start])
            cache_row = candidates[row, column]
            key = cutlass.Int64(beam.EMPTY_KEY)
            if word > cutlass.Uint32(beam.NEG_INF_WORD) and cache_row >= 0:
                ordered = cutlass.Int32(word ^ cutlass.Uint32(0x80000000))
                key = (cutlass.Int64(ordered) << 32) | cutlass.Int64(0x7FFFFFFF - cache_row)
            out[row, kept_before + slot] = key
        want = cutlass.min(count, keep)
        for slot in range(want + rank * self.threads + thread, keep, self.cluster * self.threads):
            out[row, slot] = cutlass.Int64(beam.EMPTY_KEY)
        if block % self.cluster == 0 and thread == 0:
            listed[row] = keep
        cute.arch.cluster_wait()


COMPILED_KEEPS: dict[tuple[int, int, int, int], Any] = {}
COMPILED_PICKS: dict[tuple[int, int], Any] = {}


def layout(rows: int, width: int, keep: int) -> tuple[int, int] | None:
    """Return the threads of a CTA and the CTAs of a cluster that keep a batch of rows faster than a CTA a row, or None.

    The single CTA of a row is faster below SPREAD_FROM columns, and a batch many clusters wide waits for room on the
    SMs - clusters of 8 until FEW_ROWS rows, of 4 after.

    Args:
        rows: Queries in the batch.
        width: Columns a row holds at most.
        keep: Columns a query keeps.

    Returns:
        The threads of a CTA and the CTAs of a cluster, or None for a single CTA a row.
    """
    cluster = CLUSTER if rows <= FEW_ROWS else CLUSTER // 2
    if width < SPREAD_FROM or not supports(width, keep, cluster):
        return None
    return SPREAD_THREADS, cluster


def shared_bytes(width: int, keep: int, cluster: int = CLUSTER) -> int:
    """Return the shared memory a CTA takes - the fixed words, its slice of the row and its kept columns.

    Args:
        width: Columns a row holds at most.
        keep: Columns a query keeps.
        cluster: CTAs of a cluster.

    Returns:
        Bytes of dynamic shared memory.
    """
    slice_size = -(-width // cluster)
    return (FIXED_WORDS + slice_size + beam.kept_words(keep)) * 4


def supports(width: int, keep: int, cluster: int = CLUSTER) -> bool:
    """Tell whether a cluster takes rows of this width - a slice and the kept columns fit a CTA's shared memory.

    Args:
        width: Columns a row holds at most.
        keep: Columns a query keeps.
        cluster: CTAs of a cluster.

    Returns:
        Whether a slice and the kept columns fit.
    """
    return shared_bytes(width, keep, cluster) <= beam.SHARED_BYTES and -(-width // cluster) < 2**16


def compiled_keep(element_bytes: int, node_bucket_size: int, threads: int, cluster: int) -> Any:
    """Return the keep compiled once a centroid precision, node bucket size and cluster, taking torch tensors as they are.

    Args:
        element_bytes: Bytes a centroid entry takes - 1 for fp8, 2 for bf16.
        node_bucket_size: Children a node bucket holds.
        threads: Threads of a CTA.
        cluster: CTAs of a cluster.

    Returns:
        The compiled keep.
    """
    key = (element_bytes, node_bucket_size, threads, cluster)
    if key in COMPILED_KEEPS:
        return COMPILED_KEEPS[key]
    rows, width, closed_count, out_width = (cute.sym_int32() for _ in range(4))
    score_stride, candidate_stride, closed_stride, out_stride = (cute.sym_int64() for _ in range(4))
    arguments = (
        make_fake_tensor(cutlass.Float32, (rows, width), stride=(score_stride, 1), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (rows, width), stride=(candidate_stride, 1), assumed_align=4),
        make_fake_tensor(
            cutlass.Int32, (closed_count, node_bucket_size), stride=(closed_stride, 1), assumed_align=4
        ),
        make_fake_tensor(cutlass.Int32, (rows, out_width), stride=(out_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED_KEEPS[key] = cute.compile(
        SpreadKeep(element_bytes, node_bucket_size, threads, cluster), *arguments, options="--enable-tvm-ffi"
    )
    return COMPILED_KEEPS[key]


def compiled_pick(threads: int, cluster: int) -> Any:
    """Return the pick compiled once a cluster, taking torch tensors as they are.

    Args:
        threads: Threads of a CTA.
        cluster: CTAs of a cluster.

    Returns:
        The compiled pick.
    """
    key = (threads, cluster)
    if key in COMPILED_PICKS:
        return COMPILED_PICKS[key]
    rows, width, keep = (cute.sym_int32() for _ in range(3))
    score_stride, candidate_stride, out_stride = (cute.sym_int64() for _ in range(3))
    arguments = (
        make_fake_tensor(cutlass.Float32, (rows, width), stride=(score_stride, 1), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (rows, width), stride=(candidate_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        make_fake_tensor(cutlass.Int64, (rows, keep), stride=(out_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,)),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    COMPILED_PICKS[key] = cute.compile(SpreadPick(threads, cluster), *arguments, options="--enable-tvm-ffi")
    return COMPILED_PICKS[key]


def keep_scored(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    closed_children: torch.Tensor,
    keep: int,
    lengths: torch.Tensor,
    threads: int = beam.SCORED_THREADS,
    cluster: int = CLUSTER,
) -> torch.Tensor:
    """Keep each query's best `keep` columns of a row already scored, and name their children, over a cluster.

    The kept columns are those of `beam.keep_scored`: the greatest by score, a tie to the lower column, in column order.

    Args:
        scores: The row's scores [S, W] fp32, read to each query's length.
        candidates: The rows of the level's pool each column names [S, W] int32, -1 for none.
        closed_children: The pool's children [N, node_bucket_size] int32.
        keep: Columns a query keeps, the beam's width.
        lengths: Columns each query holds [S] int32.
        threads: Threads of a CTA.
        cluster: CTAs of a cluster.

    Returns:
        The kept columns' children [S, keep * node_bucket_size] int32 in column order, -1 where a query kept nothing.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    rows, width = scores.shape
    if not supports(width, keep, cluster) or scores.stride(1) != 1 or candidates.stride(1) != 1:
        raise ValueError(
            f"the spread keep takes contiguous rows whose slice of a cluster of {cluster} and kept fit a CTA's shared "
            f"memory - got {width} columns of strides {scores.stride(1)} and {candidates.stride(1)} and a keep of {keep}"
        )
    node_bucket_size = closed_children.shape[1]
    out = torch.empty(rows, keep * node_bucket_size, device=scores.device, dtype=torch.int32)
    if rows == 0:
        return out
    compiled_keep(4, node_bucket_size, threads, cluster)(
        scores,
        candidates,
        closed_children.contiguous(),
        out,
        lengths,
        keep,
        shared_bytes(width, keep, cluster),
    )
    return out


def pick_scored(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    lengths: torch.Tensor,
    top_p: int,
    threads: int = beam.SCORED_THREADS,
    cluster: int = CLUSTER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each query's best `top_p` columns of a scored row, keyed by their cache rows, over a cluster.

    Args:
        scores: The row's scores [S, W] fp32, read to each query's length.
        candidates: The cache row of each column [S, W] int32, -1 for none.
        lengths: Columns each query holds [S] int32.
        top_p: Buckets a query picks.
        threads: Threads of a CTA.
        cluster: CTAs of a cluster.

    Returns:
        The picked scores packed with their cache rows [S, min(W, top_p)] int64 in column order, the empty key for a
        column scored -inf, with no row, or past the query's length, and the keys each query lists [S] int32 - all
        of them.

    Raises:
        ValueError: The rows are not ones the kernel takes.
    """
    rows, width = scores.shape
    keep = min(top_p, width)
    if not supports(width, keep, cluster) or scores.stride(1) != 1 or candidates.stride(1) != 1:
        raise ValueError(
            f"the spread pick takes contiguous rows whose slice of a cluster of {cluster} and kept fit a CTA's shared "
            f"memory - got {width} columns of strides {scores.stride(1)} and {candidates.stride(1)} and a top-p of {top_p}"
        )
    out = torch.empty(rows, keep, device=scores.device, dtype=torch.int64)
    listed = torch.empty(rows, device=scores.device, dtype=torch.int32)
    if rows == 0 or keep == 0:
        return out, listed.fill_(keep)
    compiled_pick(threads, cluster)(
        scores, candidates, lengths, out, listed, shared_bytes(width, keep, cluster)
    )
    return out, listed
