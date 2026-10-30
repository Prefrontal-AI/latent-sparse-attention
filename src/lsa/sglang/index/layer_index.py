"""One layer's index read off its input - the projections, the index keys and the selector, with no engine in sight.

The indexer's input is the design's `input`. From the key and the heads' queries - the design's input, the default -
an index key is `normalize(key @ key_projection)`, the key $K^{(attn)}$ being GLM's 512-dim latent, and an index query
is `normalize(sum_h query_h @ query_projection)` over the heads' absorbed queries - every head of the layer under
`query_heads: all`, the default, or the heads a rank holds. The key is in the KV pool already, so nothing else is
kept: a key's index key is projected from the pool when it is read, as the freeze read it, and a prefill writes its
chunk's keys to the pool before it selects.

From the attention input `x` - a research option, which lost to the key and query on RULER cwe 512k -
an index key is `normalize(x @ key_projection)` and an index query is `normalize(x @ query_projection)`, each a
function of its position's `x` alone. The keys of every new token are written, as the token's KV is, to a bf16
`[slots, index_dim]` buffer at the token's KV slot, so a prefix the cache already holds keeps its index keys, and the
selector reads them back through the request table.

A model hands in its input, the slots its tokens were given and the request table, and gets back lists of KV slots.
Both inputs hand the selector the same unit index keys and index queries, and everything below them - the buckets,
the hierarchy, the read - is one.

Under context parallel the same layer runs on every rank, with two things told to it. Under decode context parallel a
`Shard` says which slots this rank owns - it writes the index keys of those and reads the others' by a sum across the
group, and its bucket cache covers the whole context, the same on every rank - which `check_index` can have it check
after every forward. Under prefill context parallel a `Span` says what a rank holds
of the chunk - its own rows of `x`, whose index keys are gathered so every rank builds the index over the whole chunk,
and whose queries it selects for alone.

A grouped-query model - Qwen3.8-27B - has no latent. Its key and query input is every key head's position-free dims
side by side, read out of the KV pool's slot at the same place in each head (`KeySlot`), and each group's queries
summed into the slice of the key head they read, so one index query a token serves every head.

Tensor parallel gives a rank some of a grouped-query model's key heads and some of its query heads, so what it holds is
part of the key and query input's dims. A `KeySplit` says which dims, and the maps are applied to them alone: the
projection is linear before the normalize, so every rank's partial index key and partial index query are summed across
the group in the index's own dims and normalized after, the same on every rank. A rank's pool no longer holds the whole
input, so each new token's index key is made once as it is written and kept in the `[slots, index_dim]` buffer of the
attention input's layer, at the token's KV slot, replicated on every rank.

The key and query input reads the same two layouts on the key side alone. Under decode context parallel the pool a
rank holds is its share of the slots, so a key another rank owns is projected there and its index key read by the
same sum, and a prefill indexes its chunk from the chunk's keys in hand rather than the pool. Under prefill context
parallel the chunk's keys arrive whole - the fork gathers them for attention - while the queries are this rank's
rows. The query side needs nothing: under context parallel attention is one GPU wide, so a rank holds every head.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lsa.sglang.index.consistency import check_same_index
from lsa.sglang.index.selector import Selector
from lsa.sglang.settings.design import KEY_AND_QUERY, PAGE_SIZE, DesignSettings

# Pages a fingerprint reads at once - its int64 copies stay a few hundred MB at the widest key input
FINGERPRINT_PAGES = 8
# Bits of a fingerprint one fp32 term carries across the decode group - 256 ranks sum them exactly
FINGERPRINT_PIECE_BITS = 16
# The fingerprint's weights, the same on every rank and every server
FINGERPRINT_SEED = 20261007
FINGERPRINT_WORDS = 4096
FINGERPRINT_WEIGHTS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class Shard:
    """Which index keys a rank holds under decode context parallel - those of the KV slots it owns.

    Slot `s` is owned by rank `s % size`, at row `s // size` of that rank's buffer, as the KV pool places it.
    """

    size: int = 1
    rank: int = 0
    # Sums a tensor across the group, the same on every rank
    reduce: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Refuse a rank outside the group, or a group with no way to sum over it.

        Raises:
            ValueError: `rank` is not in `range(size)`, or `size` is above 1 with no `reduce`.
        """
        if not 0 <= self.rank < self.size:
            raise ValueError(f"rank {self.rank} is not in a group of {self.size}")
        if self.size > 1 and self.reduce is None:
            raise ValueError("a decode context-parallel shard needs the sum across its group")


def fingerprint_weights(device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fingerprint's odd int64 weights - one a word of a key's input, one a position of a page.

    Args:
        device: Where they live.
    """
    key = str(device)
    if key not in FINGERPRINT_WEIGHTS:
        generator = torch.Generator().manual_seed(FINGERPRINT_SEED)
        words, positions = (
            torch.randint(-(1 << 62), 1 << 62, (count,), generator=generator) | 1
            for count in (FINGERPRINT_WORDS, PAGE_SIZE)
        )
        FINGERPRINT_WEIGHTS[key] = (words.to(device), positions.to(device))
    return FINGERPRINT_WEIGHTS[key]


def page_fingerprints(stored: torch.Tensor, owned: torch.Tensor) -> torch.Tensor:
    """Return a fingerprint of each page of what a rank stores for its keys [pages] int64.

    A weighted sum of the bits, by odd random weights a word and a position, that wraps instead of rounding - so it
    is exact, the order it adds in does not change it, and pages that differ in any bit match by chance alone. A
    slot another rank owns adds nothing, so the group's sum is the fingerprint of the whole page.

    Args:
        stored: The rank's key input or index keys at a run of whole pages [pages * PAGE_SIZE, dim] bf16, `dim` a
            multiple of 4.
        owned: Whether this rank owns each row [pages * PAGE_SIZE] bool.
    """
    words, positions = fingerprint_weights(stored.device)
    packed = stored.contiguous().view(torch.int64)
    rows = torch.where(owned, (packed * words[: packed.shape[1]]).sum(1), 0)
    return (rows.view(-1, PAGE_SIZE) * positions).sum(1)


def exact_group_sum(values: torch.Tensor, reduce: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Sum int64 values across a group exactly, wrapping as an int64 sum does, whatever dtype the collective adds in.

    Args:
        values: This rank's terms [n] int64.
        reduce: Sums a tensor across the group.

    Returns:
        The sums [n] int64, the same on every rank.
    """
    shifts = FINGERPRINT_PIECE_BITS * torch.arange(64 // FINGERPRINT_PIECE_BITS, device=values.device)
    pieces = ((values[:, None] >> shifts) & ((1 << FINGERPRINT_PIECE_BITS) - 1)).float()
    return (reduce(pieces).long() << shifts).sum(1)


@dataclass(frozen=True)
class Span:
    """What a prefill context-parallel rank holds of a chunk, and how to reach the rest.

    The chunk's requests sit end to end, each in position order. A rank has `x` for its own span of each, and
    indexes all of it.
    """

    # Every new token's position in the chunk [chunk tokens] int32
    positions: torch.Tensor
    # New tokens each request has in the chunk, and the rows of them this rank holds
    key_counts: Sequence[int]
    query_counts: Sequence[int]
    # Position of the first of those rows of each request
    query_starts: Sequence[int]
    # This rank's rows [local tokens, width] to the chunk's [chunk tokens, width]
    gather: Callable[[torch.Tensor], torch.Tensor]


class KeyBuffer:
    """A layer's index keys, bf16 at the KV slots a rank holds."""

    def __init__(self, slot_count: int, index_dim: int, device: torch.device | str, shard: Shard) -> None:
        """Allocate the buffer.

        Args:
            slot_count: Rows of the buffer - every KV slot this rank holds, padding included.
            index_dim: Dims of an index key.
            device: Where the buffer lives.
            shard: Which slots this rank owns.
        """
        self.shard = shard
        # Under decode context parallel, a row past the slots takes the writes of slots another rank owns
        self.discard = slot_count
        self.keys = torch.zeros(
            slot_count + int(shard.size > 1), index_dim, device=device, dtype=torch.bfloat16
        )

    def rows(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the buffer row of each KV slot on this rank, and whether this rank owns it.

        Args:
            slots: KV slots [S] int64.

        Returns:
            Rows [S] int64, the discard row where the slot is another rank's.
            Ownership [S] bool.
        """
        owned = slots % self.shard.size == self.shard.rank
        return torch.where(owned, slots // self.shard.size, self.discard), owned

    def write(self, slots: torch.Tensor, keys: torch.Tensor) -> None:
        """Write index keys at KV slots - those of the slots this rank owns, and no others.

        There is no boolean mask in it, so a decode step captured in a CUDA graph writes through it.

        Args:
            slots: KV slots [S] int64.
            keys: Their index keys [S, index_dim] bf16.
        """
        if self.shard.size == 1:
            self.keys[slots] = keys
            return
        self.keys[self.rows(slots)[0]] = keys

    def read(self, slots: torch.Tensor) -> torch.Tensor:
        """Read the index keys at KV slots, whichever rank holds them.

        Every rank of the group calls this with the same slots, so the sums line up - each row has one owner and the
        others add zeros.

        Args:
            slots: KV slots [S] int64.

        Returns:
            Their index keys [S, index_dim] bf16.
        """
        if self.shard.reduce is None:
            return self.keys[slots]
        rows, owned = self.rows(slots)
        return self.shard.reduce(self.keys[rows] * owned[:, None])

    def stored(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return what this rank stores for KV slots - their index keys - and whether it owns each, with no sum.

        Args:
            slots: KV slots [S] int64.

        Returns:
            Index keys [S, index_dim] bf16, the discard row's where another rank owns the slot.
            Ownership [S] bool.
        """
        rows, owned = self.rows(slots)
        return self.keys[rows], owned


class Projection:
    """A layer's two maps, applied to `x` in one pass."""

    def __init__(
        self,
        query_projection: torch.Tensor,
        key_projection: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        """Hold the maps on a device.

        Args:
            query_projection: The query map [model_dim, index_dim] fp32.
            key_projection: The key map [model_dim, index_dim] fp32.
            device: Where the maps live.
        """
        self.index_dim = int(key_projection.shape[1])
        # One GEMM for both - index queries are the first index_dim columns, index keys the rest
        both = torch.cat([query_projection, key_projection], dim=1)
        self.maps = both.to(device=device, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read index queries and index keys off `x`, each a function of its position's `x` alone.

        Args:
            x: The attention's input [S, model_dim] bf16.

        Returns:
            The index queries [S, index_dim] fp32 unit rows,
            and the index keys [S, index_dim] bf16 unit rows.
        """
        index_queries, index_keys = (x.float() @ self.maps).split(self.index_dim, dim=1)
        return F.normalize(index_queries, dim=-1), F.normalize(index_keys, dim=-1).to(torch.bfloat16)


@dataclass(frozen=True)
class KeyAndQuery:
    """The indexer's input under `key_and_query` - the new tokens' keys and their heads' queries."""

    # The key latent of each new token [S, key_dim] bf16, normed as attention caches it - every one of the chunk's
    # under prefill context parallel. A grouped-query model's is every key head's position-free dims side by side
    key: torch.Tensor
    # Each token's absorbed query a head, for this rank's heads [S, heads, key_dim] bf16. A grouped-query model's is
    # one row a token [S, 1, key_dim] fp32 - each group's queries summed into the slice of the key head they read
    query: torch.Tensor


@dataclass(frozen=True)
class KeySplit:
    """Which dims of the key and query input a tensor-parallel rank holds, and how to sum its share across the group."""

    # The input's dims this rank's heads are, [start, stop)
    start: int
    stop: int
    # Whether this rank's key heads are added to the sum - one of the ranks that replicate a head is
    counts_key: bool
    # Sums a tensor across the tensor-parallel group, the same on every rank
    reduce: Callable[[torch.Tensor], torch.Tensor]


class KeyAndQueryProjection:
    """A layer's two maps over the key and the heads' queries, in the frozen design's numerics.

    The maps are bf16 and the products bf16, as the freeze served them, so its index keys come out bit for bit:
    an index key is the key times the key projection, normalized in fp32 and stored in bf16, and an index query
    the heads' queries summed in fp32, cast to bf16, times the query projection, then normalized.
    """

    def __init__(
        self,
        query_projection: torch.Tensor,
        key_projection: torch.Tensor,
        device: torch.device | str,
        sum_heads: Callable[[torch.Tensor], torch.Tensor] | None = None,
        split: KeySplit | None = None,
    ) -> None:
        """Hold the maps on a device.

        Args:
            query_projection: The query map [key_dim, index_dim].
            key_projection: The key map [key_dim, index_dim].
            device: Where the maps live.
            sum_heads: Sums a rank's head sums across the tensor-parallel group, so an index query sums every head of
                the layer - None sums this rank's heads alone.
            split: The dims of the input this rank holds when tensor parallel splits the key heads, whose rows of the
                maps `split_index` reads - None where a rank holds the whole input.
        """
        self.sum_heads = sum_heads
        self.split = split
        self.key_dim = int(key_projection.shape[0])
        self.query_projection = query_projection.to(device=device, dtype=torch.bfloat16)
        self.key_projection = key_projection.to(device=device, dtype=torch.bfloat16)
        if split is not None:
            self.share_query = self.query_projection[split.start : split.stop]
            self.share_key = self.key_projection[split.start : split.stop]

    def index_queries(self, query: torch.Tensor) -> torch.Tensor:
        """Sum the heads' queries and project them.

        Args:
            query: Absorbed queries [S, heads, key_dim].

        Returns:
            The index queries [S, index_dim] fp32 unit rows.
        """
        summed = query.sum(1, dtype=torch.float32)
        if self.sum_heads is not None:
            summed = self.sum_heads(summed)
        summed = summed.to(torch.bfloat16)
        return F.normalize((summed @ self.query_projection).float(), dim=-1)

    def index_keys(self, key: torch.Tensor) -> torch.Tensor:
        """Project keys.

        Args:
            key: Key latents [S, key_dim] bf16.

        Returns:
            The index keys [S, index_dim] bf16 unit rows.
        """
        return F.normalize((key.to(torch.bfloat16) @ self.key_projection).float(), dim=-1).to(torch.bfloat16)

    def split_index(self, key: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make index queries and index keys from the heads this rank holds, and sum them across the group.

        Each map is applied to this rank's rows of it, and the two [index_dim] partials travel in one all-reduce
        before they are normalized. A rank that replicates another's key heads adds zeros for its key.

        Args:
            key: The key input of this rank's key heads [S, share] bf16.
            query: The grouped queries of this rank's heads [S, 1, share] fp32.

        Returns:
            The index queries [S, index_dim] fp32 unit rows,
            and the index keys [S, index_dim] bf16 unit rows.

        Raises:
            ValueError: The projection was built with no split.
        """
        split = self.split
        if split is None:
            raise ValueError("a split index needs the projection's KeySplit")
        queries = query.sum(1, dtype=torch.float32) @ self.share_query.float()
        keys = key.float() @ self.share_key.float() if split.counts_key else torch.zeros_like(queries)
        index_queries, index_keys = split.reduce(torch.cat([queries, keys], dim=1)).split(
            queries.shape[1], dim=1
        )
        return F.normalize(index_queries, dim=-1), F.normalize(index_keys, dim=-1).to(torch.bfloat16)


@dataclass(frozen=True)
class KeySlot:
    """Where the key input sits in a KV pool slot - the same dims of each of the slot's heads, side by side.

    GLM's slot is one head of its key latent and the rope dims after it, and the input is the latent. A grouped-query
    slot is every key-value head, and the input is each head's position-free dims, its rotated prefix skipped - the
    concatenation an index key of every head reads.
    """

    # Heads a slot holds, and the dims of each
    heads: int
    head_dim: int
    # The dims of each head the input skips at the front, and the dims it reads after them
    offset: int
    width: int

    def __post_init__(self) -> None:
        """Refuse a read that runs past a head.

        Raises:
            ValueError: `offset + width` is past `head_dim`, or `width` is not positive.
        """
        if self.width <= 0 or self.offset < 0 or self.offset + self.width > self.head_dim:
            raise ValueError(
                f"a key read of {self.width} dims at {self.offset} does not fit a head of {self.head_dim}"
            )

    @classmethod
    def latent(cls, key_dim: int, rope_dim: int = 0) -> "KeySlot":
        """Return the slot of an MLA pool - one head, the key latent then its rope dims."""
        return cls(heads=1, head_dim=key_dim + rope_dim, offset=0, width=key_dim)

    @property
    def key_dim(self) -> int:
        """Dims of the key input - every head's read side by side."""
        return self.heads * self.width

    def read(self, buffer: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Read the key input at some rows of a pool buffer.

        Args:
            buffer: The layer's KV pool buffer [slots, heads, head_dim] bf16.
            rows: Rows of it [S] int64.

        Returns:
            The key input [S, key_dim] bf16, head by head.
        """
        return buffer[rows, :, self.offset : self.offset + self.width].reshape(len(rows), self.key_dim)

    def write(self, buffer: torch.Tensor, rows: torch.Tensor, key: torch.Tensor) -> None:
        """Write the key input at some rows of a pool buffer, leaving the rest of each slot as it is.

        Args:
            buffer: The layer's KV pool buffer [slots, heads, head_dim] bf16.
            rows: Rows of it [S] int64.
            key: The key input [S, key_dim] bf16.
        """
        buffer[rows, :, self.offset : self.offset + self.width] = key.view(len(rows), self.heads, self.width)


class PoolKeys:
    """A layer's index keys under `key_and_query`, projected from the keys the KV pool holds when they are read.

    Under decode context parallel the pool holds the slots this rank owns, slot `s` at row `s // size`, as the
    `KeyBuffer` of the attention input does; a key is projected on the rank that owns it and its index key summed
    across the group.
    """

    def __init__(
        self,
        key_buffer: Callable[[], torch.Tensor],
        projection: KeyAndQueryProjection,
        shard: Shard | None = None,
        key_slot: KeySlot | None = None,
    ) -> None:
        """Hold the way to the pool.

        Args:
            key_buffer: Returns the layer's KV pool buffer [slots, heads, head_dim] bf16.
            projection: The layer's maps.
            shard: Which slots this rank owns under decode context parallel, or None for every slot.
            key_slot: Where the key input sits in a slot, or None for GLM's latent at the front of one head.

        Raises:
            ValueError: The slot's key input is not the dims the maps read.
        """
        self.key_buffer = key_buffer
        self.projection = projection
        self.shard = shard or Shard()
        self.key_slot = key_slot or KeySlot.latent(projection.key_dim)
        if self.key_slot.key_dim != projection.key_dim:
            raise ValueError(
                f"the pool's key input is {self.key_slot.key_dim} dims and the maps read {projection.key_dim}"
            )

    def write(self, slots: torch.Tensor, key: torch.Tensor) -> None:
        """Write keys at their KV slots, which attention writes again after the indexer.

        Under decode context parallel nothing is written: the keys of the batch being indexed are read from the
        chunk in hand, and attention writes each to the rank that owns it.

        Args:
            slots: KV slots [S] int64.
            key: Their key input [S, key_dim] bf16.
        """
        if self.shard.size > 1:
            return
        self.key_slot.write(self.key_buffer(), slots, key)

    def read(self, slots: torch.Tensor) -> torch.Tensor:
        """Read the index keys at KV slots, whichever rank holds them.

        Under decode context parallel every rank of the group calls this with the same slots, so the sums line up -
        each row has one owner and the others add zeros.

        Args:
            slots: KV slots [S] int64.

        Returns:
            Their index keys [S, index_dim] bf16.
        """
        if self.shard.reduce is None:
            return self.projection.index_keys(self.key_slot.read(self.key_buffer(), slots))
        size = self.shard.size
        owned = slots % size == self.shard.rank
        # Another rank's slot reads this rank's row 0, and is zeroed after the projection
        rows = torch.where(owned, slots // size, 0)
        index_keys = self.projection.index_keys(self.key_slot.read(self.key_buffer(), rows))
        return self.shard.reduce(index_keys * owned[:, None])

    def stored(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return what this rank's pool stores for KV slots - their key input - and whether it owns each, with no sum.

        Args:
            slots: KV slots [S] int64.

        Returns:
            The key input [S, key_dim] bf16, row 0's where another rank owns the slot.
            Ownership [S] bool.
        """
        size = self.shard.size
        owned = slots % size == self.shard.rank
        return self.key_slot.read(self.key_buffer(), torch.where(owned, slots // size, 0)), owned


class KeyLookup:
    """One request's positions to their index keys - the pool's, and under decode context parallel the chunk's own."""

    def __init__(
        self,
        buffer: "KeyBuffer | PoolKeys",
        slot_table: Callable[[], torch.Tensor],
        shard: Shard,
        row: int,
        chunk: tuple[int, torch.Tensor] | None,
    ) -> None:
        """Hold what a lookup reads.

        Args:
            buffer: The layer's index keys.
            slot_table: Returns the request table.
            shard: Which slots this rank owns.
            row: The request table row.
            chunk: The chunk being indexed, as its first position and its index keys, or None to read the pool alone.
        """
        self.buffer = buffer
        self.slot_table = slot_table
        self.shard = shard
        self.row = row
        self.chunk = chunk

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the index keys of positions [M] int64, as [M, index_dim] bf16."""
        positions = positions.long()
        if self.shard.size == 1 or self.chunk is None:
            return self.buffer.read(self.slot_table()[self.row, positions].long())
        first, chunk_keys = self.chunk
        inside = positions >= first
        found = chunk_keys[(positions - first).clamp(0, len(chunk_keys) - 1)]
        # The test is the same on every rank, so either all of them sum or none does
        if bool(inside.all()):
            return found
        earlier = self.buffer.read(self.slot_table()[self.row, positions].long())
        return torch.where(inside[:, None], found, earlier)

    def span(self, start: int, stop: int) -> torch.Tensor:
        """Return the index keys of positions [start, stop) [stop - start, index_dim] bf16, without reading the device.

        The span's overlap with the chunk is known on the host, so no test of the positions is needed.
        """
        device = self.slot_table().device
        if self.shard.size == 1 or self.chunk is None:
            positions = torch.arange(start, stop, device=device)
            return self.buffer.read(self.slot_table()[self.row, positions].long())
        first, chunk_keys = self.chunk
        if start >= first:
            return chunk_keys[start - first : stop - first]
        outside = torch.arange(start, min(stop, first), device=device)
        earlier = self.buffer.read(self.slot_table()[self.row, outside].long())
        return torch.cat([earlier, chunk_keys[: max(stop - first, 0)]])

    def fingerprints(self, first: int, count: int) -> list[int]:
        """Return the fingerprints of the request's pages [first, first + count), read from the pool.

        The pool's own bits, not index keys: a projection rounds by the size of its batch, the pool does not. Under
        decode context parallel every rank fingerprints the slots it owns and the group sums them, so every rank
        holds the same numbers.

        Args:
            first: The first page.
            count: Pages, every one of whose keys is in the pool.
        """
        device = self.slot_table().device
        prints = []
        for start in range(first, first + count, FINGERPRINT_PAGES):
            stop = min(first + count, start + FINGERPRINT_PAGES)
            positions = torch.arange(start * PAGE_SIZE, stop * PAGE_SIZE, device=device)
            prints.append(
                page_fingerprints(*self.buffer.stored(self.slot_table()[self.row, positions].long()))
            )
        if not prints:
            return []
        total = torch.cat(prints)
        if self.shard.reduce is not None:
            total = exact_group_sum(total, self.shard.reduce)
        return total.tolist()


class LayerIndex:
    """The index of one layer: where its index keys live and how a forward batch reads them."""

    def __init__(
        self,
        settings: DesignSettings,
        projections: tuple[torch.Tensor, torch.Tensor],
        slot_count: int,
        kv_size: int,
        slot_table: Callable[[], torch.Tensor],
        max_context: int,
        prefill_chunk: int,
        device: torch.device | str,
        shard: Shard | None = None,
        check_index: bool = False,
        key_buffer: Callable[[], torch.Tensor] | None = None,
        sum_heads: Callable[[torch.Tensor], torch.Tensor] | None = None,
        key_slot: KeySlot | None = None,
        prefix_reuse: bool = False,
        split: KeySplit | None = None,
    ) -> None:
        """Hold the projections, allocate the index-key buffer and build the selector.

        Args:
            settings: The design.
            projections: The layer's query projection and key projection, both [input_dim, index_dim] fp32.
            slot_count: Rows of the index-key buffer on this rank - every KV slot it holds, padding included.
            kv_size: Keys the KV pool holds on this rank.
            slot_table: Returns the request table, each request's positions to KV slots [requests, context] int32.
            max_context: Longest context a request holds.
            prefill_chunk: Most query rows a prefill call is handed.
            device: Where everything lives.
            shard: Which slots this rank owns under decode context parallel, or None for every slot. The bucket cache then covers
                `shard.size` pools, since every rank indexes each request's whole context.
            check_index: Under decode context parallel, check after every forward's maintenance and prefill that every
                rank holds the same buckets and node levels for the batch's requests.
            key_buffer: Returns the layer's KV pool buffer, which the key and query input reads its keys from - needed
                under `key_and_query` alone.
            sum_heads: Sums a rank's head sums across the tensor-parallel group, for `query_heads: all` - None sums
                this rank's heads alone.
            key_slot: Where the key and query input's key sits in a pool slot - None for GLM's latent, at the front
                of a slot's one head.
            prefix_reuse: Share the index of a cached prefix between the requests that hold it - when the server
                caches prefixes.
            split: The dims of the key and query input this rank holds when tensor parallel splits the key heads. The
                index keys are then made when written and held in a buffer of their own at the KV slots, `key_buffer`
                is not read, and decode context parallel is not served beside it.

        Raises:
            ValueError: The input is the key and query with no KV pool buffer, or a split beside decode context
                parallel.
        """
        shard = shard or Shard()
        self.index_dim = settings.indexer.index_dim
        self.shard = shard
        self.check_index = check_index and shard.size > 1
        self.prefix_reuse = prefix_reuse
        self.slot_table = slot_table
        self.buffer: KeyBuffer | PoolKeys
        self.projection: Projection | KeyAndQueryProjection
        if settings.indexer.input == KEY_AND_QUERY and split is not None:
            if shard.size > 1:
                raise ValueError(
                    "a rank holding some of the key heads keeps the whole context's index keys itself - "
                    "decode context parallel splits the slots as well"
                )
            self.projection = KeyAndQueryProjection(*projections, device, sum_heads, split)
            self.buffer = KeyBuffer(slot_count, self.index_dim, device, shard)
        elif settings.indexer.input == KEY_AND_QUERY:
            if key_buffer is None:
                raise ValueError(
                    "the key and query input reads its keys from the KV pool - it needs the pool's key buffer"
                )
            self.projection = KeyAndQueryProjection(*projections, device, sum_heads)
            self.buffer = PoolKeys(key_buffer, self.projection, shard, key_slot)
        else:
            self.projection = Projection(*projections, device)
            self.buffer = KeyBuffer(slot_count, self.index_dim, device, shard)
        # The index keys the layer holds itself - none under the key and query, whose keys are the pool's
        self.index_keys = (
            self.buffer.keys
            if isinstance(self.buffer, KeyBuffer)
            else torch.empty(0, self.index_dim, device=device, dtype=torch.bfloat16)
        )
        request_count = slot_table().shape[0]
        self.starts_checked = False
        self.selector = Selector(
            settings,
            kv_size * shard.size,
            request_count,
            max_context,
            prefill_chunk,
            device,
            slot_table,
            prefix_reuse,
        )

    def project(
        self, inputs: torch.Tensor | KeyAndQuery, slots: torch.Tensor, span: Span | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project new tokens' input, and write what their index keys are read from at their KV slots.

        Args:
            inputs: The attention input [S, model_dim] bf16, or the key and the heads' queries - under `span` the
                key is the whole chunk's and the queries this rank's rows.
            slots: The KV slots the tokens were given [S] int64 - all of the chunk's, under `span`.
            span: What this rank holds of the chunk under prefill context parallel, or None.

        Returns:
            The tokens' index queries [S, index_dim] fp32 unit rows, and the index keys of every one of the chunk's
            tokens [chunk tokens, index_dim] bf16 - None under the key and query outside decode context parallel,
            whose keys are read from the pool.

        Raises:
            TypeError: The input is not the one the index was built for.
            ValueError: The key and query's key does not cover the slots.
        """
        if isinstance(inputs, KeyAndQuery):
            if not isinstance(self.projection, KeyAndQueryProjection):
                raise TypeError("the key and query reached an index built for the attention input")
            if inputs.key.shape[0] != slots.shape[0]:
                raise ValueError(
                    f"the key and query input has {inputs.key.shape[0]} keys for {slots.shape[0]} slots - "
                    "under prefill context parallel the key must be the whole chunk's"
                )
            # A rank's pool holds some of the key heads, so the keys' index keys are made once, here
            if self.projection.split is not None and isinstance(self.buffer, KeyBuffer):
                index_queries, index_keys = self.projection.split_index(inputs.key, inputs.query)
                self.buffer.write(slots, index_keys)
                return index_queries, None
            if not isinstance(self.buffer, PoolKeys):
                raise TypeError("the key and query reached an index built for the attention input")
            self.buffer.write(slots, inputs.key)
            index_queries = self.projection.index_queries(inputs.query)
            # Under decode context parallel the pool holds a share of the chunk, so its keys are read in hand
            if self.shard.size > 1:
                return index_queries, self.projection.index_keys(inputs.key)
            return index_queries, None
        if not isinstance(self.projection, Projection) or not isinstance(self.buffer, KeyBuffer):
            raise TypeError("the attention input reached an index built for the key and query")
        index_queries, index_keys = self.projection(inputs)
        if span is not None:
            index_keys = span.gather(index_keys)
        self.buffer.write(slots, index_keys)
        return index_queries, index_keys

    def write(self, inputs: torch.Tensor | KeyAndQuery, slots: torch.Tensor) -> torch.Tensor:
        """Write the index keys of new tokens at their KV slots, and return their index queries.

        Args:
            inputs: The attention's input at the new tokens [S, model_dim] bf16, or their key and heads' queries.
            slots: The KV slots the tokens were given [S] int64.

        Returns:
            The tokens' index queries [S, index_dim] fp32 unit rows.
        """
        return self.project(inputs, slots)[0]

    def write_chunk(
        self, inputs: torch.Tensor | KeyAndQuery, slots: torch.Tensor, span: Span | None = None
    ) -> None:
        """Write the index keys of a prefill chunk without selecting - a short prompt the dense path prefills.

        Args:
            inputs: The chunk's attention input [S, model_dim] bf16 - this rank's rows only, under `span` - or its
                key and heads' queries.
            slots: The KV slots the chunk's tokens were given [S] int64 - all of the chunk's, under `span`.
            span: What this rank holds of the chunk under prefill context parallel, or None.
        """
        self.project(inputs, slots, span)

    def keys_of(self, row: int, chunk: tuple[int, torch.Tensor] | None = None) -> "KeyLookup":
        """Return the lookup of one request's positions to their index keys.

        Under decode context parallel a position outside the chunk being indexed is read from whichever rank owns
        it, which every rank of the group does at once. The chunk's own keys are every rank's already.

        Args:
            row: The request table row.
            chunk: The chunk being indexed, as its first position and its index keys [chunk tokens, index_dim],
                or None to read the pool alone.

        Returns:
            A function from positions [M] int64 to index keys [M, index_dim] bf16, which also reads a span.
        """
        return KeyLookup(self.buffer, self.slot_table, self.shard, row, chunk)

    def maintain(self, rids: Sequence[object], rows: Sequence[int], ends: Sequence[int]) -> int:
        """Bring each request's cache up to the keys already in the pool, eagerly, before a forward.

        Args:
            rids: The batch's request ids.
            rows: Their request table rows.
            ends: Keys each request has in the pool before this forward's own.

        Returns:
            The pages the batch's requests took from cached prefixes.
        """
        taken = self.selector.maintain(rids, rows, ends, self.keys_of)
        self.check(rows)
        return taken

    def check(self, rows: Sequence[int]) -> None:
        """Raise unless every rank of the decode group holds the same index for the requests, when asked to check.

        Args:
            rows: The batch's request table rows, the same on every rank.
        """
        if self.check_index and self.shard.reduce is not None:
            check_same_index(self.selector.cache, rows, self.shard.reduce, self.shard.size)

    def extend(
        self,
        inputs: torch.Tensor | KeyAndQuery,
        positions: torch.Tensor,
        slots: torch.Tensor,
        rows: Sequence[int],
        lengths: Sequence[int],
        starts: Sequence[int],
        span: Span | None = None,
        ragged: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Write the keys of a prefill batch and select for its queries, request by request.

        Args:
            inputs: The batch's attention input [S, model_dim] bf16, the requests' new tokens end to end - this
                rank's rows only, under prefill context parallel - or their key and heads' queries.
            positions: Each row's position in its request [S] int32.
            slots: Each new token's KV slot [S] int64 - all of the chunk's under `span`, where they are not `x`'s.
            rows: The requests' table rows.
            lengths: New tokens each request has in the batch.
            starts: Position of each request's first new token, from the scheduler - no device read.
            span: What this rank holds of the chunk under prefill context parallel, or None.
            ragged: Where each request starts in the rows of [prefix, chunk] keys decode context parallel's
                prefill reads, or None for lists of KV slots.

        Returns:
            KV slots, or ragged rows, [S, width] int32, -1 padded - a list for each row of `x`.
        """
        index_queries, index_keys = self.project(inputs, slots, span)
        key_starts = [int(start) for start in starts]
        query_starts: Sequence[int] = key_starts
        key_counts, query_counts = lengths, lengths
        if span is not None:
            key_counts, query_counts, query_starts = span.key_counts, span.query_counts, span.query_starts
        self.check_starts(positions, query_counts, query_starts)
        active = set(rows)
        lists, query_start, key_start = [], 0, 0
        for index, (row, key_count, query_count) in enumerate(
            zip(rows, key_counts, query_counts, strict=True)
        ):
            query_stop, key_stop = query_start + query_count, key_start + key_count
            lists.append(
                self.selector.prefill(
                    row,
                    index_queries[query_start:query_stop],
                    positions[query_start:query_stop],
                    self.keys_of(
                        row,
                        None if index_keys is None else (key_starts[index], index_keys[key_start:key_stop]),
                    ),
                    active,
                    key_count,
                    None if ragged is None else ragged[index],
                    query_starts[index],
                )
            )
            query_start, key_start = query_stop, key_stop
        self.check(rows)
        return torch.cat(lists)

    def check_starts(
        self, positions: torch.Tensor, query_counts: Sequence[int], query_starts: Sequence[int]
    ) -> None:
        """Raise unless the scheduler's first positions are the batch's own - once, on the first prefill, as it reads the device.

        Args:
            positions: Each row's position in its request [S] int32.
            query_counts: Query rows each request holds.
            query_starts: Position of each request's first query row, from the scheduler.

        Raises:
            ValueError: A request's first row is not at the position the scheduler gave.
        """
        if self.starts_checked:
            return
        self.starts_checked = True
        offsets = [sum(query_counts[:i]) for i, count in enumerate(query_counts) if count]
        firsts = positions[torch.tensor(offsets, device=positions.device, dtype=torch.long)].tolist()
        wanted = [start for start, count in zip(query_starts, query_counts, strict=True) if count]
        if firsts != wanted:
            raise ValueError(f"the batch's first positions {firsts} are not the scheduler's {wanted}")

    def decode(
        self,
        inputs: torch.Tensor | KeyAndQuery,
        positions: torch.Tensor,
        slots: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Write a decode step's keys and select for its queries - fixed shapes unless the design says tree.

        Every rank of a decode context-parallel group holds the same batch and selects the same lists.

        Args:
            inputs: The attention input of each request's new token [B, model_dim] bf16, or its key and heads'
                queries.
            positions: Each request's position [B] int32.
            slots: Each token's KV slot [B] int64.
            rows: The requests' table rows [B] int64.

        Returns:
            KV slots [B, width] int32, -1 padded.
        """
        index_queries = self.write(inputs, slots)
        return self.selector.decode(index_queries, positions, rows)
