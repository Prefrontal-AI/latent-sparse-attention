"""The flat read's fill by threshold in CuTeDSL - a row's listed buckets read as their members, by the key a walk stops at.

The walk over a row's best buckets stops at one key: the largest at or above which the buckets hold the top-k's
members or number the top-p. Every member of a bucket above it is read, then the first members of the bucket at it
until the top-k is full, then the window and the sinks. A CTA holds one row's listed keys in registers and finds that
key a byte a pass from the top, each pass's members and buckets counted per digit in shared memory - the low word
only when keys tie on the high. One scan then places every bucket above it: its place in the walk and its first
entry of the read, so groups of lanes copy a bucket's members each, in order.

The read is the Triton fill's, entry for entry: the same key, the same buckets in the same order, and a bucket's
members in its slot order.

The expansion in rank order walks picked buckets best first instead, with no key to find: the same scan places every pick, and
each copies the members the top-k still has room for. It is the Triton expansion in rank order, entry for entry.

Under an early budget a row before its position reads the early top-k and top-p, taken from its position in the CTA, so
the shapes and the shared memory stay the design's own.
"""

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from lsa.budget import EarlyBudget

# Threads of the CTA a row takes: 1,024 read a listing faster while rows leave SMs idle, 512 once rows fill the GPU
# => 1,024 for few rows, 512 for many
FEW_ROWS_THREADS = 1024
MANY_ROWS_THREADS = 512
# Rows an SM holds before a launch fills the GPU
ROWS_PER_SM = 1
# Bins of a radix pass, and the bins each lane of the warp that finds a pass's digit holds
BINS = 256
BINS_PER_LANE = BINS // 32
# The empty key, and the bit that orders a signed key as an unsigned one
EMPTY = -(2**63)
SIGN = 2**63
# A bucket above the last key counts once in the walk and its members in the read, packed in one word
WALKED = 1 << 16
# Rounds of buckets a warp copies the members of with all their loads in flight
WALK_IN_FLIGHT = 4
# The search's state words
PREFIX_HIGH, PREFIX_LOW, ABOVE_MEMBERS, ABOVE_BUCKETS, NONE_REACHES, TIED, CUT_ROW, TAKEN = range(8)
STATE_WORDS = 8


class ExpansionByThreshold:
    """A row's read from its listed buckets - the members above the walk's last key, the last cut, the window and sinks."""

    def __init__(
        self,
        capacity: int,
        bucket_cap: int,
        top_p: int,
        top_k: int,
        window_size: int,
        sink_size: int,
        map_slots: bool,
        physical: bool,
        threads: int,
        early: EarlyBudget | None = None,
    ) -> None:
        """Fix the shapes and the read's form.

        Args:
            capacity: Keys a row's listing holds.
            bucket_cap: Members a cache row holds at most, a power of two up to 32.
            top_p: Best buckets the walk reads at most.
            top_k: Bucket members read at most.
            window_size: Recent positions read whole.
            sink_size: First positions read whole.
            map_slots: Write KV slots through the slot table rather than positions.
            physical: The keys hold cache rows, and a page knows its first position.
            threads: Threads of a CTA, FEW_ROWS_THREADS or MANY_ROWS_THREADS.
            early: The budget of rows before its position, or None for the design's own throughout.
        """
        self.capacity = capacity
        self.threads = threads
        self.warps = threads // 32
        self.per_thread = -(-capacity // threads)
        self.bucket_cap = bucket_cap
        # Lanes copy a bucket's members, one a lane, so a warp copies several buckets at once
        self.buckets_per_warp = 32 // bucket_cap
        self.top_p = top_p
        self.top_k = top_k
        self.window_size = window_size
        self.sink_size = sink_size
        self.map_slots = map_slots
        self.physical = physical
        self.early_position = 0 if early is None else early.position
        self.early_top_k = top_k if early is None else early.top_k
        self.early_top_p = top_p if early is None else early.top_p
        self.width = top_k + window_size + sink_size
        self.barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)

    @cute.jit
    def __call__(
        self,
        keys: cute.Tensor,
        listed: cute.Tensor,
        bucket_members: cute.Tensor,
        bucket_sizes: cute.Tensor,
        request_pages: cute.Tensor,
        page_positions: cute.Tensor,
        request_rows: cute.Tensor,
        positions: cute.Tensor,
        slot_table: cute.Tensor,
        out: cute.Tensor,
        rows_per_page: cutlass.Int32,
        cluster_block_size: cutlass.Int32,
        stream: Any,
    ) -> None:
        """Launch a CTA a row.

        Args:
            keys: Listed packed keys [R, capacity] int64, the empty key after a row's listed count.
            listed: Keys each row listed [R] int32.
            bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
            bucket_sizes: Members of each cache row [rows] int32.
            request_pages: Pages of each request [requests, pages] int32, read through a logical bucket.
            page_positions: Each page's first position [pages] int32, read through a cache row.
            request_rows: Each row's request [R] int64.
            positions: Each row's position [R] int32.
            slot_table: Each request's positions to KV slots [requests, context] int32.
            out: Read lists [R, top_k + window_size + sink_size] int32, -1 padded.
            rows_per_page: Cache rows a page holds.
            cluster_block_size: Positions a cluster block covers.
            stream: The stream to launch on.
        """
        top_p = self.top_p
        warps = self.warps

        @cute.struct
        class Storage:
            """Shared memory of a block: digit histograms, per-warp totals, and the walk's state and picks."""

            members_by_digit: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * BINS], 16]
            buckets_by_digit: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * BINS], 16]
            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, warps], 16]
            state: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, STATE_WORDS], 16]
            walk_rows: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, top_p], 16]
            walk_entries: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, top_p], 16]

        self.kernel(
            keys,
            listed,
            bucket_members,
            bucket_sizes,
            request_pages,
            page_positions,
            request_rows,
            positions,
            slot_table,
            out,
            rows_per_page,
            cluster_block_size,
            Storage,
        ).launch(grid=(keys.shape[0], 1, 1), block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        keys: cute.Tensor,
        listed: cute.Tensor,
        bucket_members: cute.Tensor,
        bucket_sizes: cute.Tensor,
        request_pages: cute.Tensor,
        page_positions: cute.Tensor,
        request_rows: cute.Tensor,
        positions: cute.Tensor,
        slot_table: cute.Tensor,
        out: cute.Tensor,
        rows_per_page: cutlass.Int32,
        cluster_block_size: cutlass.Int32,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Find the row's last key, then write the buckets above it, the last cut to the room left, and the tail.

        Args:
            keys: Listed packed keys.
            listed: Keys each row listed.
            bucket_members: Members of each cache row.
            bucket_sizes: Members of each cache row.
            request_pages: Pages of each request.
            page_positions: Each page's first position.
            request_rows: Each row's request.
            positions: Each row's position.
            slot_table: Each request's positions to KV slots.
            out: Read lists.
            rows_per_page: Cache rows a page holds.
            cluster_block_size: Positions a cluster block covers.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = thread // 32
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        members_by_digit = storage.members_by_digit.get_tensor(cute.make_layout(2 * BINS))
        buckets_by_digit = storage.buckets_by_digit.get_tensor(cute.make_layout(2 * BINS))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(self.warps))
        state = storage.state.get_tensor(cute.make_layout(STATE_WORDS))
        walk_rows = storage.walk_rows.get_tensor(cute.make_layout(self.top_p))
        walk_entries = storage.walk_entries.get_tensor(cute.make_layout(self.top_p))

        if thread < STATE_WORDS:
            state[thread] = cutlass.Int32(0)
        if thread < BINS:
            members_by_digit[thread] = cutlass.Int32(0)
            buckets_by_digit[thread] = cutlass.Int32(0)
        position = positions[row]
        top_k, top_p = self.row_budget(position)
        window_start = cutlass.max(0, position + 1 - self.window_size)
        count = cutlass.min(listed[row], self.capacity)
        request = cutlass.Int32(request_rows[row])

        # This thread's run of the row's keys - listing order is thread order, then place in the run
        high_words = cute.make_rmem_tensor(self.per_thread, cutlass.Uint32)
        low_words = cute.make_rmem_tensor(self.per_thread, cutlass.Uint32)
        cache_rows = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        members = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        for j in cutlass.range_constexpr(self.per_thread):
            index = thread * self.per_thread + j
            key = cutlass.Int64(EMPTY)
            if index < count:
                key = keys[row, index]
            # The key's unsigned order - 0 for the empty key
            order = cutlass.Uint64(key) ^ cutlass.Uint64(SIGN)
            high_words[j] = cutlass.Uint32(order >> 32)
            low_words[j] = cutlass.Uint32(order & cutlass.Uint64(0xFFFFFFFF))
            cache_row, first_position = self.cache_row_of(
                key, order, request, request_pages, page_positions, rows_per_page, cluster_block_size
            )
            cache_rows[j] = cache_row
            members[j] = self.members_read(
                cache_row, first_position, window_start, cluster_block_size, bucket_members, bucket_sizes
            )
        self.barrier.arrive_and_wait()

        last_high, last_low = self.last_key(
            high_words,
            low_words,
            members,
            thread,
            lane,
            warp,
            members_by_digit,
            buckets_by_digit,
            state,
            top_k,
            top_p,
        )

        # Each bucket above the last key, its place in the walk and its first entry of the read, in listing order
        packed = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        run_total = cutlass.Int32(0)
        for j in cutlass.range_constexpr(self.per_thread):
            packed[j] = run_total
            if self.above(high_words[j], low_words[j], last_high, last_low) and cache_rows[j] >= 0:
                run_total = run_total + WALKED + members[j]
        before = self.exclusive_sum(run_total, lane, warp, warp_totals)
        walked_total = warp_totals[self.warps - 1]
        for j in cutlass.range_constexpr(self.per_thread):
            if self.above(high_words[j], low_words[j], last_high, last_low) and cache_rows[j] >= 0:
                place = before + packed[j]
                walk_rows[place // WALKED] = cache_rows[j]
                walk_entries[place // WALKED] = place % WALKED
            if high_words[j] == last_high and low_words[j] == last_low and cache_rows[j] >= 0:
                state[CUT_ROW] = cache_rows[j] + 1
        self.barrier.arrive_and_wait()
        walked = walked_total // WALKED
        filled = walked_total % WALKED
        group = lane // self.bucket_cap
        slot = lane % self.bucket_cap
        self.copy_walk(
            walked,
            cutlass.Int32(0),
            cutlass.Int32(-1),
            warp,
            lane,
            walk_rows,
            walk_entries,
            window_start,
            row,
            request,
            bucket_members,
            slot_table,
            out,
        )
        # The bucket at the last key, cut to the room the top-k has left
        if warp == 0:
            cut_row = state[CUT_ROW] - 1
            if group > 0:
                cut_row = cutlass.Int32(-1)
            taken = self.copy_members(
                cut_row,
                slot,
                group,
                filled,
                top_k - filled,
                window_start,
                row,
                request,
                bucket_members,
                slot_table,
                out,
            )
            if lane == 0:
                state[TAKEN] = taken
        self.barrier.arrive_and_wait()
        filled = filled + state[TAKEN]
        # The window's positions then the sinks before it
        window_count = cutlass.min(self.window_size, position + 1)
        sink_count = cutlass.min(self.sink_size, window_start)
        for t in range(thread, window_count + sink_count, self.threads):
            read = window_start + t
            if t >= window_count:
                read = t - window_count
            if cutlass.const_expr(self.map_slots):
                read = slot_table[request, read]
            out[row, filled + t] = read
        for t in range(filled + window_count + sink_count + thread, self.width, self.threads):
            out[row, t] = cutlass.Int32(-1)

    @cute.jit
    def row_budget(self, position: Any) -> Any:
        """Return a row's top-k and top-p - the early budget's before its position, the design's from it.

        Args:
            position: The row's position.
        """
        top_k = cutlass.Int32(self.top_k)
        top_p = cutlass.Int32(self.top_p)
        if cutlass.const_expr(self.early_position > 0):
            if position < self.early_position:
                top_k = cutlass.Int32(self.early_top_k)
                top_p = cutlass.Int32(self.early_top_p)
        return top_k, top_p

    @cute.jit
    def cache_row_of(
        self,
        key: Any,
        order: Any,
        request: Any,
        request_pages: cute.Tensor,
        page_positions: cute.Tensor,
        rows_per_page: Any,
        cluster_block_size: Any,
    ) -> Any:
        """Return a key's cache row and its block's first position - -1 and 0 for the empty key or an unmapped page.

        Args:
            key: The packed key.
            order: Its unsigned order.
            request: The row's request.
            request_pages: Pages of each request.
            page_positions: Each page's first position.
            rows_per_page: Cache rows a page holds.
            cluster_block_size: Positions a cluster block covers.
        """
        bucket = cutlass.Int32(0x7FFFFFFF) - cutlass.Int32(key & cutlass.Int64(0xFFFFFFFF))
        cache_row = cutlass.Int32(-1)
        first_position = cutlass.Int32(0)
        if order > cutlass.Uint64(0):
            if cutlass.const_expr(self.physical):
                cache_row = bucket
                first_position = page_positions[bucket // rows_per_page]
            else:
                block = bucket // rows_per_page
                page = request_pages[request, block]
                if page >= 0:
                    cache_row = page * rows_per_page + bucket % rows_per_page
                first_position = block * cluster_block_size
        return cache_row, first_position

    @cute.jit
    def members_read(
        self,
        cache_row: Any,
        first_position: Any,
        window_start: Any,
        cluster_block_size: Any,
        bucket_members: cute.Tensor,
        bucket_sizes: cute.Tensor,
    ) -> Any:
        """Return the members a bucket gives the read - all it has, unless its block reaches the sinks or the window.

        Args:
            cache_row: The bucket's cache row, -1 for none.
            first_position: Its block's first position.
            window_start: The row's first window position.
            cluster_block_size: Positions a cluster block covers.
            bucket_members: Members of each cache row.
            bucket_sizes: Members of each cache row.
        """
        given = cutlass.Int32(0)
        if cache_row >= 0:
            given = bucket_sizes[cache_row]
            if first_position < self.sink_size or first_position + cluster_block_size > window_start:
                given = cutlass.Int32(0)
                for s in cutlass.range_constexpr(self.bucket_cap):
                    member = bucket_members[cache_row, s]
                    if member >= self.sink_size and member < window_start:
                        given = given + 1
        return given

    @cute.jit
    def above(self, high: Any, low: Any, last_high: Any, last_low: Any) -> Any:
        """Tell whether a key orders above the last key.

        Args:
            high: The key's high word.
            low: The key's low word.
            last_high: The last key's high word.
            last_low: The last key's low word.
        """
        return high > last_high or (high == last_high and low > last_low)

    @cute.jit
    def last_key(
        self,
        high_words: cute.Tensor,
        low_words: cute.Tensor,
        members: cute.Tensor,
        thread: Any,
        lane: Any,
        warp: Any,
        members_by_digit: cute.Tensor,
        buckets_by_digit: cute.Tensor,
        state: cute.Tensor,
        top_k: Any,
        top_p: Any,
    ) -> Any:
        """Return the largest key at or above which the buckets hold the top-k's members or number the top-p.

        Args:
            high_words: This thread's keys' high words.
            low_words: This thread's keys' low words.
            members: The members each gives the read.
            thread: This thread.
            lane: This lane.
            warp: This warp.
            members_by_digit: Two passes' members a digit, one counted while the other is read.
            buckets_by_digit: Two passes' buckets a digit.
            state: The search's state, zero on entry.
            top_k: The row's top-k.
            top_p: The row's top-p.

        Returns:
            The last key's high and low words - 0 and 0 where no key reaches either.
        """
        for shift in cutlass.range_constexpr(24, -1, -8):
            self.radix_pass(
                high_words,
                low_words,
                members,
                thread,
                lane,
                warp,
                members_by_digit,
                buckets_by_digit,
                state,
                0,
                shift,
                top_k,
                top_p,
            )
        # The low word decides only between keys that tie on the high - one such key is the last itself
        if state[NONE_REACHES] == 0 and state[TIED] > 1:
            for shift in cutlass.range_constexpr(24, -1, -8):
                self.radix_pass(
                    high_words,
                    low_words,
                    members,
                    thread,
                    lane,
                    warp,
                    members_by_digit,
                    buckets_by_digit,
                    state,
                    1,
                    shift,
                    top_k,
                    top_p,
                )
        else:
            if state[NONE_REACHES] == 0:
                prefix_high = cutlass.Uint32(state[PREFIX_HIGH])
                for j in cutlass.range_constexpr(self.per_thread):
                    if high_words[j] == prefix_high:
                        state[PREFIX_LOW] = cutlass.Int32(low_words[j])
            self.barrier.arrive_and_wait()
        return cutlass.Uint32(state[PREFIX_HIGH]), cutlass.Uint32(state[PREFIX_LOW])

    @cute.jit
    def radix_pass(
        self,
        high_words: cute.Tensor,
        low_words: cute.Tensor,
        members: cute.Tensor,
        thread: Any,
        lane: Any,
        warp: Any,
        members_by_digit: cute.Tensor,
        buckets_by_digit: cute.Tensor,
        state: cute.Tensor,
        word: int,
        shift: int,
        top_k: Any,
        top_p: Any,
    ) -> None:
        """Count the keys under the prefix by their next digit, and find the digit - clearing the other counts meanwhile.

        Args:
            high_words: This thread's keys' high words.
            low_words: This thread's keys' low words.
            members: The members each gives the read.
            thread: This thread.
            lane: This lane.
            warp: This warp.
            members_by_digit: Two passes' members a digit.
            buckets_by_digit: Two passes' buckets a digit.
            state: The search's state.
            word: 0 for the high word, 1 for the low.
            shift: The pass's bit offset in its word.
            top_k: The row's top-k.
            top_p: The row's top-p.
        """
        counted = (word * 4 + (24 - shift) // 8) % 2 * BINS
        cleared = BINS - counted
        prefix_high = cutlass.Uint32(state[PREFIX_HIGH])
        prefix_low = cutlass.Uint32(state[PREFIX_LOW])
        kept = cutlass.Uint32(0)
        if cutlass.const_expr(shift < 24):
            kept = cutlass.Uint32(0xFFFFFFFF) << (shift + 8)
        if state[NONE_REACHES] == 0:
            for j in cutlass.range_constexpr(self.per_thread):
                high = high_words[j]
                low = low_words[j]
                matches = (high & kept) == (prefix_high & kept)
                digit = cutlass.Int32((high >> shift) & 255)
                if cutlass.const_expr(word == 1):
                    matches = high == prefix_high and (low & kept) == (prefix_low & kept)
                    digit = cutlass.Int32((low >> shift) & 255)
                if matches and (high | low) != cutlass.Uint32(0):
                    if members[j] > 0:
                        cute.arch.atomic_add(
                            (members_by_digit.iterator + (counted + digit)).llvm_ptr, members[j], scope="cta"
                        )
                    cute.arch.atomic_add(
                        (buckets_by_digit.iterator + (counted + digit)).llvm_ptr,
                        cutlass.Int32(1),
                        scope="cta",
                    )
        self.barrier.arrive_and_wait()
        if warp == 0:
            if state[NONE_REACHES] == 0:
                self.find_digit(
                    lane, members_by_digit, buckets_by_digit, counted, state, word, shift, top_k, top_p
                )
        elif thread - 32 < BINS:
            members_by_digit[cleared + thread - 32] = cutlass.Int32(0)
            buckets_by_digit[cleared + thread - 32] = cutlass.Int32(0)
        self.barrier.arrive_and_wait()

    @cute.jit
    def find_digit(
        self,
        lane: Any,
        members_by_digit: cute.Tensor,
        buckets_by_digit: cute.Tensor,
        counted: int,
        state: cute.Tensor,
        word: int,
        shift: int,
        top_k: Any,
        top_p: Any,
    ) -> None:
        """Find a pass's digit of the last key in one warp, and add what lies above it to the state.

        Args:
            lane: This lane.
            members_by_digit: The passes' members a digit.
            buckets_by_digit: The passes' buckets a digit.
            counted: Where this pass's counts begin.
            state: The search's state.
            word: 0 for the high word, 1 for the low.
            shift: The pass's bit offset in its word.
            top_k: The row's top-k.
            top_p: The row's top-p.
        """
        above_members = state[ABOVE_MEMBERS]
        above_buckets = state[ABOVE_BUCKETS]
        # Lane l holds the bins 8l to 8l + 7 counted from the top
        lane_members = cutlass.Int32(0)
        lane_buckets = cutlass.Int32(0)
        for b in cutlass.range_constexpr(BINS_PER_LANE):
            lane_members = lane_members + members_by_digit[counted + BINS - 1 - (lane * BINS_PER_LANE + b)]
            lane_buckets = lane_buckets + buckets_by_digit[counted + BINS - 1 - (lane * BINS_PER_LANE + b)]
        members_before = lane_members
        buckets_before = lane_buckets
        for i in cutlass.range_constexpr(5):
            # A shuffle up clamps at lane 0, which cutlass-dsl 4.6.2 does not default to
            other_members = cute.arch.shuffle_sync_up(members_before, 1 << i, mask_and_clamp=0)
            other_buckets = cute.arch.shuffle_sync_up(buckets_before, 1 << i, mask_and_clamp=0)
            if lane >= (1 << i):
                members_before = members_before + other_members
                buckets_before = buckets_before + other_buckets
        members_before = members_before - lane_members
        buckets_before = buckets_before - lane_buckets
        reached_before = above_members + members_before >= top_k or above_buckets + buckets_before >= top_p
        reached_after = (
            above_members + members_before + lane_members >= top_k
            or above_buckets + buckets_before + lane_buckets >= top_p
        )
        crossing = reached_after and not reached_before
        if crossing:
            members_above = above_members + members_before
            buckets_above = above_buckets + buckets_before
            digit = cutlass.Int32(0)
            tied = cutlass.Int32(0)
            found = cutlass.Int32(0)
            for b in cutlass.range_constexpr(BINS_PER_LANE):
                bin_index = BINS - 1 - (lane * BINS_PER_LANE + b)
                bin_members = members_by_digit[counted + bin_index]
                bin_buckets = buckets_by_digit[counted + bin_index]
                reaches = members_above + bin_members >= top_k or buckets_above + bin_buckets >= top_p
                if found == 0 and reaches:
                    digit = bin_index
                    tied = bin_buckets
                    found = 1
                if found == 0:
                    members_above = members_above + bin_members
                    buckets_above = buckets_above + bin_buckets
            prefix = cutlass.Uint32(state[PREFIX_HIGH + word]) | (cutlass.Uint32(digit) << shift)
            state[PREFIX_HIGH + word] = cutlass.Int32(prefix)
            state[ABOVE_MEMBERS] = members_above
            state[ABOVE_BUCKETS] = buckets_above
            state[TIED] = tied
        # No digit reaches only when no key does - the walk then reads every bucket
        if cute.arch.vote_any_sync(crossing) == 0 and lane == 0:
            state[NONE_REACHES] = cutlass.Int32(1)

    @cute.jit
    def exclusive_sum(self, value: Any, lane: Any, warp: Any, warp_totals: cute.Tensor) -> Any:
        """Return the sum of value over the threads before this one, leaving the running warp totals in shared memory.

        Args:
            value: This thread's value.
            lane: This lane.
            warp: This warp.
            warp_totals: Each warp's total and those before it, inclusive, written here.
        """
        inclusive = value
        for i in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(inclusive, 1 << i, mask_and_clamp=0)
            if lane >= (1 << i):
                inclusive = inclusive + other
        if lane == 31:
            warp_totals[warp] = inclusive
        self.barrier.arrive_and_wait()
        if warp == 0:
            total = cutlass.Int32(0)
            if lane < self.warps:
                total = warp_totals[lane]
            for i in cutlass.range_constexpr(self.warps.bit_length() - 1):
                other = cute.arch.shuffle_sync_up(total, 1 << i, mask_and_clamp=0)
                if lane >= (1 << i):
                    total = total + other
            if lane < self.warps:
                warp_totals[lane] = total
        self.barrier.arrive_and_wait()
        before = inclusive - value
        if warp > 0:
            before = before + warp_totals[warp - 1]
        return before

    @cute.jit
    def copy_walk(
        self,
        walked: Any,
        idle_at: Any,
        top_k: Any,
        warp: Any,
        lane: Any,
        walk_rows: cute.Tensor,
        walk_entries: cute.Tensor,
        window_start: Any,
        row: Any,
        request: Any,
        bucket_members: cute.Tensor,
        slot_table: cute.Tensor,
        out: cute.Tensor,
    ) -> None:
        """Copy the walked buckets' members, a warp's lane groups a bucket each, WALK_IN_FLIGHT rounds at a time.

        A round's member loads, then its slot loads, are all in flight before any is stored.

        Args:
            walked: Buckets the walk reads.
            idle_at: The entry a lane group with no bucket writes at.
            top_k: The top-k a bucket's members are cut to, or -1 for all of a bucket's members.
            warp: This warp.
            lane: This lane.
            walk_rows: The walk's cache rows.
            walk_entries: Each walked bucket's first entry of the read.
            window_start: The row's first window position.
            row: The row.
            request: The row's request.
            bucket_members: Members of each cache row.
            slot_table: Each request's positions to KV slots.
            out: Read lists.
        """
        group = lane // self.bucket_cap
        slot = lane % self.bucket_cap
        group_mask = cutlass.Uint32((1 << self.bucket_cap) - 1) << (group * self.bucket_cap)
        caches = cute.make_rmem_tensor(WALK_IN_FLIGHT, cutlass.Int32)
        places = cute.make_rmem_tensor(WALK_IN_FLIGHT, cutlass.Int32)
        members = cute.make_rmem_tensor(WALK_IN_FLIGHT, cutlass.Int32)
        reads = cute.make_rmem_tensor(WALK_IN_FLIGHT, cutlass.Int32)
        taken = cute.make_rmem_tensor(WALK_IN_FLIGHT, cutlass.Boolean)
        round_buckets = self.warps * self.buckets_per_warp
        for first in range(warp * self.buckets_per_warp, walked, round_buckets * WALK_IN_FLIGHT):
            for r in cutlass.range_constexpr(WALK_IN_FLIGHT):
                walk = first + r * round_buckets + group
                caches[r] = cutlass.Int32(-1)
                places[r] = idle_at
                members[r] = cutlass.Int32(-1)
                if walk < walked:
                    caches[r] = walk_rows[walk]
                    places[r] = walk_entries[walk]
                if caches[r] >= 0:
                    members[r] = bucket_members[caches[r], slot]
            for r in cutlass.range_constexpr(WALK_IN_FLIGHT):
                valid = members[r] >= self.sink_size and members[r] < window_start
                valid_mask = cute.arch.vote_ballot_sync(valid)
                rank = cutlass.Int32(cute.arch.popc(valid_mask & group_mask & cute.arch.lanemask_lt()))
                room = cutlass.Int32(self.bucket_cap)
                if top_k >= 0:
                    room = top_k - places[r]
                taken[r] = valid and rank < room
                places[r] = places[r] + rank
                reads[r] = members[r]
                if taken[r]:
                    if cutlass.const_expr(self.map_slots):
                        reads[r] = slot_table[request, members[r]]
            for r in cutlass.range_constexpr(WALK_IN_FLIGHT):
                if taken[r]:
                    out[row, places[r]] = reads[r]

    @cute.jit
    def copy_members(
        self,
        cache_row: Any,
        slot: Any,
        group: Any,
        at: Any,
        room: Any,
        window_start: Any,
        row: Any,
        request: Any,
        bucket_members: cute.Tensor,
        slot_table: cute.Tensor,
        out: cute.Tensor,
    ) -> Any:
        """Copy a bucket's members outside the sinks and the window from at, room of them at most - a lane a member.

        Args:
            cache_row: The lane group's bucket, -1 for none.
            slot: This lane's member slot.
            group: This lane's group of the warp.
            at: The group's first entry of the read.
            room: Members to write at most.
            window_start: The row's first window position.
            row: The row.
            request: The row's request.
            bucket_members: Members of each cache row.
            slot_table: Each request's positions to KV slots.
            out: Read lists.

        Returns:
            Members the group wrote.
        """
        member = cutlass.Int32(-1)
        if cache_row >= 0:
            member = bucket_members[cache_row, slot]
        valid = member >= self.sink_size and member < window_start
        # The group's valid members before this lane's, in slot order
        valid_mask = cute.arch.vote_ballot_sync(valid)
        group_mask = cutlass.Uint32((1 << self.bucket_cap) - 1) << (group * self.bucket_cap)
        rank = cutlass.Int32(cute.arch.popc(valid_mask & group_mask & cute.arch.lanemask_lt()))
        if valid and rank < room:
            read = member
            if cutlass.const_expr(self.map_slots):
                read = slot_table[request, member]
            out[row, at + rank] = read
        return cutlass.min(cutlass.Int32(cute.arch.popc(valid_mask & group_mask)), room)


class ExpansionInRankOrder(ExpansionByThreshold):
    """A row's read from its picked buckets, best first until the top-k is full, then the window and sinks."""

    def __init__(
        self,
        top_p: int,
        bucket_cap: int,
        top_k: int,
        window_size: int,
        sink_size: int,
        map_slots: bool,
        threads: int,
        early: EarlyBudget | None = None,
    ) -> None:
        """Fix the shapes and the read's form.

        Args:
            top_p: Picks a row has.
            bucket_cap: Members a cache row holds at most, a power of two up to 32.
            top_k: Bucket members read at most.
            window_size: Recent positions read whole.
            sink_size: First positions read whole.
            map_slots: Write KV slots through the slot table rather than positions.
            threads: Threads of a CTA, FEW_ROWS_THREADS or MANY_ROWS_THREADS.
            early: The budget of rows before its position, or None for the design's own throughout.
        """
        super().__init__(
            top_p, bucket_cap, top_p, top_k, window_size, sink_size, map_slots, True, threads, early
        )

    @cute.jit
    def __call__(
        self,
        selected: cute.Tensor,
        bucket_members: cute.Tensor,
        positions: cute.Tensor,
        slot_table: cute.Tensor,
        slot_rows: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        stream: Any,
    ) -> None:
        """Launch a CTA a row.

        Args:
            selected: Picked cache rows [R, top_p] int32, best first, -1 padded.
            bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
            positions: Each row's position [R] int32.
            slot_table: Each request's positions to KV slots [requests, context] int32.
            slot_rows: Each row's request [R] int64.
            out: Read lists [R, top_k + window_size + sink_size] int32, -1 padded.
            lengths: Entries each read list holds [R] int32.
            stream: The stream to launch on.
        """
        top_p = self.top_p
        warps = self.warps

        @cute.struct
        class Storage:
            """Shared memory of a block: the per-warp totals and the walk's rows, entries and stop."""

            warp_totals: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, warps], 16]
            walk_rows: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, top_p], 16]
            walk_entries: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, top_p], 16]
            walk_stop: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 16]

        self.read_kernel(
            selected, bucket_members, positions, slot_table, slot_rows, out, lengths, Storage
        ).launch(grid=(selected.shape[0], 1, 1), block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def read_kernel(
        self,
        selected: cute.Tensor,
        bucket_members: cute.Tensor,
        positions: cute.Tensor,
        slot_table: cute.Tensor,
        slot_rows: cute.Tensor,
        out: cute.Tensor,
        lengths: cute.Tensor,
        Storage: cutlass.Constexpr,
    ) -> None:
        """Place each pick's members in the read best first, copy those the top-k has room for, then the tail.

        Args:
            selected: Picked cache rows.
            bucket_members: Members of each cache row.
            positions: Each row's position.
            slot_table: Each request's positions to KV slots.
            slot_rows: Each row's request.
            out: Read lists.
            lengths: Entries each read list holds.
            Storage: The shared memory's struct.
        """
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = thread // 32
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout(self.warps))
        walk_rows = storage.walk_rows.get_tensor(cute.make_layout(self.top_p))
        walk_entries = storage.walk_entries.get_tensor(cute.make_layout(self.top_p))
        walk_stop = storage.walk_stop.get_tensor(cute.make_layout(1))
        if thread == 0:
            walk_stop[0] = cutlass.Int32(self.top_p)
        position = positions[row]
        top_k, top_p = self.row_budget(position)
        window_start = cutlass.max(0, position + 1 - self.window_size)
        request = cutlass.Int32(0)
        if cutlass.const_expr(self.map_slots):
            request = cutlass.Int32(slot_rows[row])

        # This thread's run of the picks, and the members each gives outside the sinks and the window
        loaded = cute.make_rmem_tensor(self.bucket_cap, cutlass.Int32)
        cache_rows = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        givens = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        packed = cute.make_rmem_tensor(self.per_thread, cutlass.Int32)
        run_total = cutlass.Int32(0)
        for j in cutlass.range_constexpr(self.per_thread):
            index = thread * self.per_thread + j
            cache_row = cutlass.Int32(-1)
            if index < top_p:
                cache_row = selected[row, index]
            given = cutlass.Int32(0)
            if cache_row >= 0:
                cute.autovec_copy(bucket_members[cache_row, None], loaded)
                for s in cutlass.range_constexpr(self.bucket_cap):
                    given = given + cutlass.Int32((loaded[s] >= self.sink_size) & (loaded[s] < window_start))
            cache_rows[j] = cache_row
            givens[j] = given
            packed[j] = run_total
            if given > 0:
                run_total = run_total + WALKED + given
        before = self.exclusive_sum(run_total, lane, warp, warp_totals)
        walked_total = warp_totals[self.warps - 1]
        for j in cutlass.range_constexpr(self.per_thread):
            if givens[j] > 0:
                place = before + packed[j]
                walk_rows[place // WALKED] = cache_rows[j]
                walk_entries[place // WALKED] = place % WALKED
                # The first pick that starts past the top-k ends the walk
                if place % WALKED >= top_k:
                    cute.arch.atomic_min((walk_stop.iterator).llvm_ptr, place // WALKED, scope="cta")
        self.barrier.arrive_and_wait()
        walked = cutlass.min(walked_total // WALKED, walk_stop[0])
        filled = cutlass.min(walked_total % WALKED, top_k)
        self.copy_walk(
            walked,
            top_k,
            top_k,
            warp,
            lane,
            walk_rows,
            walk_entries,
            window_start,
            row,
            request,
            bucket_members,
            slot_table,
            out,
        )
        # The window's positions then the sinks before it
        window_count = cutlass.min(self.window_size, position + 1)
        sink_count = cutlass.min(self.sink_size, window_start)
        for t in range(thread, window_count + sink_count, self.threads):
            read = window_start + t
            if t >= window_count:
                read = t - window_count
            if cutlass.const_expr(self.map_slots):
                read = slot_table[request, read]
            out[row, filled + t] = read
        for t in range(filled + window_count + sink_count + thread, self.width, self.threads):
            out[row, t] = cutlass.Int32(-1)
        if thread == 0:
            lengths[row] = filled + window_count + sink_count


COMPILED: dict[tuple[int | EarlyBudget | None, ...], Any] = {}
COMPILED_IN_ORDER: dict[tuple[int | EarlyBudget | None, ...], Any] = {}
# Stand-ins for a slot table or page positions a read does not take
STAND_INS: dict[torch.device, torch.Tensor] = {}


def compiled(
    capacity: int,
    bucket_cap: int,
    top_p: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    map_slots: bool,
    physical: bool,
    threads: int,
    early: EarlyBudget | None = None,
) -> Any:
    """Return the fill compiled once for its shapes and form, taking torch tensors as they are.

    Args:
        capacity: Keys a row's listing holds.
        bucket_cap: Members a cache row holds at most.
        top_p: Best buckets the walk reads at most.
        top_k: Bucket members read at most.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        map_slots: Write KV slots through the slot table.
        physical: The keys hold cache rows.
        threads: Threads of a CTA.
        early: The budget of rows before its position, or None.
    """
    key = (
        capacity,
        bucket_cap,
        top_p,
        top_k,
        window_size,
        sink_size,
        int(map_slots),
        int(physical),
        threads,
        early,
    )
    if key in COMPILED:
        return COMPILED[key]
    rows, cache_rows, requests, pages, page_count, slot_requests, context = (
        cute.sym_int32() for _ in range(7)
    )
    key_stride, page_stride, slot_stride = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    width = top_k + window_size + sink_size
    arguments = (
        make_fake_tensor(cutlass.Int64, (rows, capacity), stride=(key_stride, 1), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_compact_tensor(
            cutlass.Int32, (cache_rows, bucket_cap), stride_order=(1, 0), assumed_align=4
        ),
        make_fake_compact_tensor(cutlass.Int32, (cache_rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (requests, pages), stride=(page_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (page_count,), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (rows,), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (slot_requests, context), stride=(slot_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows, width), stride_order=(1, 0), assumed_align=4),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kernel = ExpansionByThreshold(
        capacity, bucket_cap, top_p, top_k, window_size, sink_size, map_slots, physical, threads, early
    )
    COMPILED[key] = cute.compile(kernel, *arguments, options="--enable-tvm-ffi")
    return COMPILED[key]


@functools.cache
def multiprocessors_of(index: int) -> int:
    """Return the multiprocessors of a CUDA device, asked once.

    Args:
        index: The device's index.

    Returns:
        The device's multiprocessor count.
    """
    return torch.cuda.get_device_properties(index).multi_processor_count


def multiprocessors(device: torch.device) -> int:
    """Return the multiprocessors of a device.

    Args:
        device: A CUDA device, whose index may be None for the current one.

    Returns:
        The device's multiprocessor count.
    """
    return multiprocessors_of(torch.cuda.current_device() if device.index is None else device.index)


def threads_for(rows: int, device: torch.device) -> int:
    """Return the threads of a CTA for a launch of rows.

    Args:
        rows: Rows the launch has, a CTA each.
        device: The device the launch runs on.

    Returns:
        FEW_ROWS_THREADS when the rows leave SMs idle, MANY_ROWS_THREADS otherwise.
    """
    return FEW_ROWS_THREADS if rows <= ROWS_PER_SM * multiprocessors(device) else MANY_ROWS_THREADS


def stand_in(device: torch.device) -> torch.Tensor:
    """Return a one-entry int32 table for an argument a read does not take.

    Args:
        device: The device.
    """
    if device not in STAND_INS:
        STAND_INS[device] = torch.zeros(1, 1, device=device, dtype=torch.int32)
    return STAND_INS[device]


def supports(keys: torch.Tensor, bucket_members: torch.Tensor, request_pages: torch.Tensor) -> bool:
    """Tell whether the CuTeDSL fill takes this listing - int64 keys, int32 tables, contiguous rows and a cap of 2^n to 32.

    Args:
        keys: Listed packed keys [R, C].
        bucket_members: Members of each cache row [rows, bucket_cap].
        request_pages: Pages of each request [requests, pages].
    """
    return (
        keys.dtype == torch.int64
        and keys.stride(1) == 1
        and bucket_members.shape[1] in (1, 2, 4, 8, 16, 32)
        and bucket_members.dtype == torch.int32
        and bucket_members.is_contiguous()
        and request_pages.dtype == torch.int32
        and request_pages.stride(1) == 1
    )


def expand_by_threshold(
    keys: torch.Tensor,
    listed: torch.Tensor,
    bucket_members: torch.Tensor,
    bucket_sizes: torch.Tensor,
    request_pages: torch.Tensor,
    request_rows: torch.Tensor,
    positions: torch.Tensor,
    rows_per_page: int,
    cluster_block_size: int,
    top_p: int,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    page_positions: torch.Tensor | None = None,
    early: EarlyBudget | None = None,
    threads: int | None = None,
) -> torch.Tensor:
    """Fill each row's read from its listed buckets by the key a best-first walk of its top-p stops at.

    Args:
        keys: Listed packed keys [R, C] int64, the empty key after a row's listed count.
        listed: Keys each row listed [R] int32, at most C.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        bucket_sizes: Members of each cache row [rows] int32.
        request_pages: Pages of each request [requests, pages] int32, read through a request's logical buckets.
        request_rows: Each row's request [R] int64.
        positions: Each row's position [R] int32.
        rows_per_page: Cache rows a page holds.
        cluster_block_size: Positions a cluster block covers.
        top_p: Best buckets the walk reads at most.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.
        slot_table: Each request's positions to KV slots [requests, context] int32 - positions when None.
        page_positions: Each page's first position [pages] int32 - when given, the keys hold cache rows rather than
            a request's logical buckets, and `request_pages` is not read.
        early: The budget of rows before its position, whose walk stops at the key its `top_p` and `top_k` find in a
            listing that holds every such row's top-p - or None for `top_p` and `top_k` throughout.
        threads: Threads of a CTA, or None for the count the rows call for.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded.

    Raises:
        ValueError: The listing is not one the kernel takes.
    """
    if not supports(keys, bucket_members, request_pages):
        raise ValueError(
            "the CuTeDSL fill takes int64 keys, int32 tables with contiguous rows and a bucket cap of a power of two up "
            "to 32 - got keys "
            f"{keys.dtype} strides {keys.stride()}, members {bucket_members.dtype} and pages {request_pages.dtype} "
            f"strides {request_pages.stride()}"
        )
    device = keys.device
    threads = threads_for(keys.shape[0], device) if threads is None else threads
    out = torch.empty(keys.shape[0], top_k + window_size + sink_size, device=device, dtype=torch.int32)
    fill = compiled(
        keys.shape[1],
        bucket_members.shape[1],
        top_p,
        top_k,
        window_size,
        sink_size,
        slot_table is not None,
        page_positions is not None,
        threads,
        early,
    )
    fill(
        keys,
        listed,
        bucket_members,
        bucket_sizes,
        request_pages,
        stand_in(device)[0] if page_positions is None else page_positions,
        request_rows,
        positions,
        stand_in(device) if slot_table is None else slot_table,
        out,
        rows_per_page,
        cluster_block_size,
    )
    return out


def compiled_rank_order(
    top_p: int,
    bucket_cap: int,
    top_k: int,
    window_size: int,
    sink_size: int,
    map_slots: bool,
    threads: int,
    early: EarlyBudget | None = None,
) -> Any:
    """Return the expansion in rank order compiled once for its shapes and form, taking torch tensors as they are.

    Args:
        top_p: Picks a row has.
        bucket_cap: Members a cache row holds at most.
        top_k: Bucket members read at most.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        map_slots: Write KV slots through the slot table.
        threads: Threads of a CTA.
        early: The budget of rows before its position, or None.
    """
    key = (top_p, bucket_cap, top_k, window_size, sink_size, int(map_slots), threads, early)
    if key in COMPILED_IN_ORDER:
        return COMPILED_IN_ORDER[key]
    rows, cache_rows, slot_requests, context, slot_row_count = (cute.sym_int32() for _ in range(5))
    selected_stride, slot_stride = cute.sym_int64(), cute.sym_int64()
    width = top_k + window_size + sink_size
    arguments = (
        make_fake_tensor(cutlass.Int32, (rows, top_p), stride=(selected_stride, 1), assumed_align=4),
        # A member row is as aligned as its bytes, to 16, so a pick's row loads in vectors
        make_fake_compact_tensor(
            cutlass.Int32,
            (cache_rows, bucket_cap),
            stride_order=(1, 0),
            assumed_align=min(16, 4 * bucket_cap),
        ),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_tensor(cutlass.Int32, (slot_requests, context), stride=(slot_stride, 1), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (slot_row_count,), assumed_align=8),
        make_fake_compact_tensor(cutlass.Int32, (rows, width), stride_order=(1, 0), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (rows,), assumed_align=4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    kernel = ExpansionInRankOrder(top_p, bucket_cap, top_k, window_size, sink_size, map_slots, threads, early)
    COMPILED_IN_ORDER[key] = cute.compile(kernel, *arguments, options="--enable-tvm-ffi")
    return COMPILED_IN_ORDER[key]


def supports_rank_order(selected: torch.Tensor, bucket_members: torch.Tensor) -> bool:
    """Tell whether the CuTeDSL expansion in rank order takes these picks - int32, contiguous rows, a cap of 2^n to 32.

    Args:
        selected: Picked cache rows [R, top_p].
        bucket_members: Members of each cache row [rows, bucket_cap].
    """
    return (
        selected.dtype == torch.int32
        and selected.stride(1) == 1
        and bucket_members.shape[1] in (1, 2, 4, 8, 16, 32)
        and bucket_members.dtype == torch.int32
        and bucket_members.is_contiguous()
        # A pick's place and its first entry share one word
        and selected.shape[1] * bucket_members.shape[1] < WALKED
    )


def expand_in_order(
    selected: torch.Tensor,
    bucket_members: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    slot_rows: torch.Tensor | None = None,
    early: EarlyBudget | None = None,
    threads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read each row's picked buckets best first until the top-k is full, then the window and the sinks.

    Args:
        selected: Picked cache rows [R, top_p] int32, best score first, -1 padded.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        positions: Each row's position [R] int32.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.
        slot_table: Each request's positions to KV slots [requests, context] int32 - positions when None.
        slot_rows: Each row's request [R] int64, read with `slot_table`.
        early: The budget of rows before its position - they read its first `top_p` picks up to its `top_k` - or
            None for every pick up to `top_k` throughout.
        threads: Threads of a CTA, or None for the count the rows call for.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded, the expansion first.
        Entries each read list holds [R] int32.

    Raises:
        ValueError: The picks are not ones the kernel takes.
    """
    if not supports_rank_order(selected, bucket_members):
        raise ValueError(
            "the CuTeDSL expansion in rank order takes int32 picks and members with contiguous rows, a bucket cap of a power "
            f"of two up to 32 and fewer than {WALKED} members a row - got picks {selected.dtype} "
            f"{tuple(selected.shape)} strides {selected.stride()} and members {bucket_members.dtype} "
            f"{tuple(bucket_members.shape)}"
        )
    device = selected.device
    rows = selected.shape[0]
    out = torch.empty(rows, top_k + window_size + sink_size, device=device, dtype=torch.int32)
    lengths = torch.empty(rows, device=device, dtype=torch.int32)
    if rows == 0:
        return out, lengths
    read = compiled_rank_order(
        selected.shape[1],
        bucket_members.shape[1],
        top_k,
        window_size,
        sink_size,
        slot_table is not None,
        threads_for(rows, device) if threads is None else threads,
        early,
    )
    read(
        selected,
        bucket_members,
        positions.int().contiguous(),
        stand_in(device) if slot_table is None else slot_table,
        positions.new_zeros(1, dtype=torch.int64) if slot_rows is None else slot_rows.contiguous(),
        out,
        lengths,
    )
    return out, lengths
