"""What the indexer reads of sglang's context-parallel layouts, and the arithmetic that needs no engine.

Two layouts, which run alone or together on the fork the context-parallel serving image carries:

| layout | what a rank holds | what the indexer does |
|---|---|---|
| prefill context parallel (CP) | one contiguous span of each request in the chunk - its own `x` rows | projects its own rows, gathers the chunk's index keys, builds the index over the whole chunk and selects for its own rows |
| decode context parallel (DCP) | the KV pool rows of the slots it owns, slot `s` on rank `s % size` at row `s // size` | writes the index keys of the slots it owns, reads the others' by a sum across the group, and keeps a bucket cache that covers the whole context |

A decode batch is whole on every DCP rank, so every rank selects the same lists and no list is gathered.
The hierarchy of fixed and tree grows on every rank from that same bucket cache and the same summed index
keys, and its beam reads the cache alone, so every rank holds the same tree - a rank whose tree differed would select
other lists, and would call the sum a different number of times and hang the group.
There are no heads in an index query on `x`, so the layouts' head arithmetic - how many tensor-parallel GPUs' heads one
rank holds - does not enter: a position has one list however the heads are laid out.

The key and query input reads heads, and is served under context parallel only where attention is one GPU wide, so a
rank holds every head - its prefill rows' queries for all of them, and a decode's once decode context parallel has
gathered them - and sums them all, `query_heads: all`. Its keys are the key latents: a prefill chunk's arrive whole,
gathered by the fork for attention, and the KV pool's are split across the decode group as the index keys on `x` are.

The helpers that need sglang import it when called and answer "not active" on a stock image, where none of the
layouts exists.
"""

from collections.abc import Sequence
from typing import Any

import torch

from lsa.sglang.settings.design import KEY_AND_QUERY, QUERY_HEADS_ALL, DesignSettings


def is_active(forward_batch: Any) -> bool:
    """Return whether this forward splits its prefill across context-parallel ranks.

    Args:
        forward_batch: The batch.

    Returns:
        True under prefill context parallel with more than one rank.
    """
    try:
        from sglang.srt.layers.cp.utils import is_cp_active
    except ImportError:
        return False
    return bool(is_cp_active(forward_batch))


def is_served() -> bool:
    """Return whether this server runs prefill or decode context parallel, as sglang's own parallel state says.

    Returns:
        True once the parallel state is published and either layout is on, and False before it is - a process that
        has not published its arguments has no layout to check.
    """
    try:
        from sglang.srt.runtime_context import get_context, get_parallel
    except ImportError:
        return False
    if not get_context().is_config_namespace_published("parallel"):
        return False
    parallel = get_parallel()
    return bool(parallel.enable_prefill_cp or parallel.dcp_enabled)


def decode_cp() -> tuple[int, int]:
    """Return this rank's decode context-parallel group size and rank - (1, 0) without it.

    Returns:
        The group size, and this rank's place in it.
    """
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        return 1, 0
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return 1, 0
    return int(parallel.attn_dcp_size), int(parallel.attn_dcp_rank)


def prefill_split() -> str | None:
    """Return how prefill context parallel splits a chunk across ranks - None without it.

    Returns:
        The fork's `--cp-strategy` - `contiguous`, `interleave` or `zigzag` - under prefill context parallel with more
        than one rank.
    """
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        return None
    parallel = get_parallel()
    if not parallel.enable_prefill_cp or int(parallel.attn_cp_size) <= 1:
        return None
    return str(parallel.cp_strategy)


def refuse_split(split: str | None) -> None:
    """Refuse a prefill context-parallel split other than one contiguous span of each request a rank.

    The query counts, the chunk's positions, the gather of its index keys and the selection's match of queries to
    page runs all take a rank's queries of a request to be one unbroken run of positions.
    Another split scatters them, and the lists come out wrong without an error.

    Args:
        split: The fork's `--cp-strategy`, or None without prefill context parallel.

    Raises:
        ValueError: The split is not `contiguous`.
    """
    if split is not None and split != "contiguous":
        raise ValueError(
            f"--cp-strategy {split} scatters a request's queries across ranks, and LSA reads one contiguous span "
            "a rank - serve with --cp-strategy contiguous"
        )


def attention_tp_size() -> int:
    """Return how many GPUs an attention layer's heads are split over - 1 without the fork's layouts' runtime.

    Returns:
        The attention tensor-parallel size.
    """
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        return 1
    return int(get_parallel().attn_tp_size)


def attention_tp_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the attention tensor-parallel group.

    Args:
        values: The rank's term of the sum.

    Returns:
        The sum, the same on every rank.
    """
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().attn_tp_group.all_reduce(values)


def refuse_key_and_query(
    design: DesignSettings, decode_size: int, split: str | None, attention_tp: int
) -> None:
    """Refuse the key and query input under a context-parallel layout it does not read as one rank would.

    Its index query sums heads, and the lists of one rank holding every head are reproduced when every rank holds
    every head at both prefill and decode - attention context parallel across the whole tensor-parallel group, so
    attention is one GPU wide - and sums every one of them, `query_heads: all`. `rank` asks for each tensor-parallel
    rank's own heads, which no rank holds apart under that layout.

    Args:
        design: The design.
        decode_size: The decode context-parallel group's size.
        split: The fork's `--cp-strategy`, or None without prefill context parallel.
        attention_tp: How many GPUs attention's heads are split over.

    Raises:
        ValueError: The design reads the key and query under context parallel with attention split over more than
            one GPU, or with `query_heads` other than `all`.
    """
    if design.indexer.input != KEY_AND_QUERY or (decode_size == 1 and split is None):
        return
    if attention_tp != 1:
        raise ValueError(
            f"the key and query input sums the heads' queries, and under context parallel it reads every head on "
            f"every rank - attention is split over {attention_tp} GPUs here, so a rank holds some heads at prefill "
            f"and decode context parallel gathers all of them at decode; serve attention context parallel across "
            f"the whole tensor-parallel group (--attn-cp-size equal to --tp)"
        )
    if design.indexer.query_heads != QUERY_HEADS_ALL:
        raise ValueError(
            f"query_heads {design.indexer.query_heads!r} sums the heads one tensor-parallel rank holds, and under "
            f"context parallel attention is one GPU wide, so a rank holds every head and no rank's share exists - "
            f"set model.indexer.query_heads: {QUERY_HEADS_ALL}"
        )


def chunk_key_whole(forward_batch: Any) -> bool:
    """Return whether the key latents a prefill context-parallel chunk hands the indexer are already the whole chunk's.

    The fork's DSA attention gathers a chunk's latents across the group, in `prepare_kv_for_attention`, before the
    attention's query preparation returns them - for a bf16 KV cache, which the key and query input reads. A rank's
    padded share can number as many rows as the chunk, so the count cannot tell the two apart.

    Args:
        forward_batch: The context-parallel batch.

    Returns:
        True where the fork's DSA path has gathered the chunk's latents.
    """
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp

    return bool(dsa_use_prefill_cp(forward_batch))


def decode_group_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the decode context-parallel group.

    Args:
        values: The rank's term of the sum.

    Returns:
        The sum, the same on every rank.
    """
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().dcp_group.all_reduce(values)


def refuse_shapes(design: DesignSettings, decode_size: int) -> None:
    """Refuse a shaped design under decode context parallel, which no shape has been run against.

    A node bucket shape is built over every key beneath the node, which a rank holding a share of the cache would
    fetch across the group; a bucket shape is read by kernels never run against a split cache.

    Args:
        design: The design.
        decode_size: The decode context-parallel group's size.

    Raises:
        ValueError: The design sets `bucket_shape` or `node_bucket_shape` and the group has more than one rank.
    """
    indexer = design.indexer
    knobs = (("bucket_shape", indexer.bucket_shape), ("node_bucket_shape", indexer.node_bucket_shape))
    shaped = [name for name, shape in knobs if shape is not None]
    if decode_size > 1 and shaped:
        raise ValueError(
            f"{' and '.join(shaped)} is not served under decode context parallel {decode_size} - "
            f"serve the design without decode context parallel, or drop the shape"
        )


def chunk_positions(
    prefixes: Sequence[int], lengths: Sequence[int], device: torch.device | str
) -> torch.Tensor:
    """Return the positions of a chunk's new tokens, request by request in position order.

    A request's new tokens follow its cached prefix.

    Args:
        prefixes: Tokens each request already holds.
        lengths: New tokens each request has in the chunk, whole - not one rank's share.
        device: Where the positions go.

    Returns:
        The positions [chunk tokens] int32.
    """
    return torch.cat(
        [
            torch.arange(prefix, prefix + length, dtype=torch.int32, device=device)
            for prefix, length in zip(prefixes, lengths, strict=True)
        ]
    )


def ragged_offsets(seq_lens: Sequence[int]) -> list[int]:
    """Return where each request starts in the [prefix, chunk] rows decode context parallel's prefill reads.

    Args:
        seq_lens: Tokens each request holds after the chunk.

    Returns:
        The first row of each request.
    """
    offsets, start = [], 0
    for seq_len in seq_lens:
        offsets.append(start)
        start += int(seq_len)
    return offsets


def local_query_counts(forward_batch: Any) -> list[int]:
    """Return how many query rows this rank holds of each request, in batch order.

    Args:
        forward_batch: The context-parallel batch.
    """
    return list(forward_batch.attn_cp_metadata.actual_seq_q_prev_list)


def local_query_starts(forward_batch: Any) -> list[int]:
    """Return the position of the first query row this rank holds of each request, in batch order.

    A rank's rows of a request are the span its kv length ends, so the span starts where its rows' count back from there.

    Args:
        forward_batch: The context-parallel batch.
    """
    metadata = forward_batch.attn_cp_metadata
    return [
        int(kv_len) - int(count)
        for kv_len, count in zip(metadata.kv_len_prev_list, metadata.actual_seq_q_prev_list, strict=True)
    ]


def gather_chunk(rows: torch.Tensor, forward_batch: Any) -> torch.Tensor:
    """Return the whole chunk's rows from this rank's, request by request in position order.

    A rank's rows are its share whatever their count - padded to the group's longest share, a chunk of one token a
    rank arrives with as many rows as the chunk has tokens, and still needs the gather.

    Args:
        rows: This rank's rows [local tokens, width], padding included or not.
        forward_batch: The context-parallel batch.

    Returns:
        The chunk's rows [chunk tokens, width].
    """
    from sglang.srt.layers.cp.base import get_cp_strategy

    return get_cp_strategy().gather_kv_cache(rows.contiguous(), forward_batch)
