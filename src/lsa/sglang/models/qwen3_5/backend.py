"""Qwen3.5's attention under the design - the dense backend's quadratic layers reading an LSA selection.

Qwen3.8-27B has no indexer to replace: its sixteen full-attention layers attend densely through sglang's FlashAttention
backend, behind the hybrid wrapper that sends the forty-eight gated delta net layers elsewhere. So the hooks sit on that
backend. `forward_extend` and `forward_decode` are handed the rotated query and key heads and the layer, which is both
the capture and the dispatch: a quadratic layer writes its keys and values to the pool as the backend would, selects
from the key and query input [index_space](index_space.py) reads off those heads, and reads what it selected. Every
other call - a layer outside the design, an idle batch - goes to the backend as it was.

The read is not a new kernel. FlashAttention's paged read at a page size of 1 takes a page table row of arbitrary KV
slots and a count of the real ones, and every query token is its own sequence of length one, as sglang's DSA backend
reads GLM's lists. [read](../../index/read.py) counts a list.

The index catches up with the KV pool before every forward, from the backend's own metadata calls - the eager one and
the one a captured decode replays - as the DSA backend's hooks do for GLM.

Plain tensor parallel is served by sharding the index across the ranks. A rank holds some of the key heads - under
tensor parallel 8, one of four, which two ranks replicate - and some of the query heads, so it projects what it holds
through its rows of the two maps and the group sums the [index_dim] partials: every rank's index key and index query
are then the single GPU's, and every rank selects the same lists. A rank's pool holds its own key head alone, so an
index key can no longer be made from the pool when it is read. Each new token's is made once as it is written and kept
in a bf16 `[slots, index_dim]` buffer at its KV slot, whole on every rank. Every token of a rank's head is in its
pool, so the rank reads the selected slots from its own pool with the heads it holds, and the read needs nothing
more. A rank that holds every key head - tensor parallel 1, or decode context parallel across the group - projects
the index key from its pool as it always did.

Prefill context parallel, data-parallel attention, a decode group that spans only part of the tensor-parallel group,
a paged pool, speculative decoding and a cache other than bf16 are refused.

Decode context parallel across the whole tensor-parallel group is served, the mesh dense Qwen reaches 10M on. Every rank
keeps all four key heads for the slots it owns, slot `s` on rank `s % size` at row `s // size`, and its share of the
query heads. The index is the one a single GPU builds, held whole on every rank, as GLM's decode group holds it: a
forward gathers the group's query heads, so the summed query is every head's, the chunk's keys are every rank's
already, and a key in another rank's share is projected there and its index key summed across the group. Every rank
selects the same lists, with the budget a single GPU spends - `top_k` keys and `top_p` buckets over the whole context.
Each rank reads the selected slots it owns with every head, and the group merges the partial reads by their
log-sum-exp into the softmax over the whole list, each rank keeping its own heads. The image decodes such a mesh on the
triton backend and prefills it on FlashAttention, so both backends are armed and both read through FlashAttention's
paged kernel.

It imports sglang, so nothing but the plugin reaches it, and only after the plugin resolved every target.
"""

from collections.abc import Callable
from typing import Any

import torch

from lsa.constants import Sentinel
from lsa.core.kernels import dispatch
from lsa.sglang import context_parallel, memory, plugin, settings
from lsa.sglang.constants import Banner
from lsa.sglang.index import read, registry, weights
from lsa.sglang.index.layer_index import KeySplit, LayerIndex, Shard
from lsa.sglang.models.qwen3_5 import index_space
from lsa.sglang.settings.design import KEY_AND_QUERY, DesignSettings

# Where an armed backend keeps its layers' indexers
INDEXERS: str = "lsa_lsa_indexers"

# Every quadratic layer's indexer, built by the first backend armed - one index a layer, however many backends read it
ARMED: dict[int, "GroupedQueryIndexer"] = {}

# What an armed backend reads by name - the fork's image has them all, and a Qwen server runs on nothing else
# => resolved when a backend is armed, so a GLM server on a stock image never needs them
REQUIRED: tuple[str, ...] = (
    "sglang.srt.mem_cache.memory_pool.KVWriteLoc",
    "sglang.srt.runtime_context.get_parallel",
    "sglang.srt.runtime_context.get_spec",
)

# What a decode context-parallel rank reads besides - the merge of the group's partial reads, and FlashAttention's
# paged kernel for the triton backend, which has none of its own
REQUIRED_DECODE_GROUP: tuple[str, ...] = (
    "sglang.srt.layers.dcp.cp_lse_ag_out_rs_mha",
    "sglang.kernels.ops.attention.flash_attention.flash_attn_with_kvcache",
)


class GroupedQueryIndexer:
    """One quadratic layer's index, built on its first use once the model and the KV pool exist."""

    # What the plugin's readiness and the harness ask of an indexer, to tell an LSA selection from a native one
    is_lsa = True

    def __init__(
        self,
        layer_id: int,
        geometry: index_space.Geometry,
        design: DesignSettings,
        backend: Any,
        shard: Shard | None = None,
        heads: index_space.HeadShard | None = None,
        check_index: bool = False,
    ) -> None:
        """Hold what the index is built from.

        Args:
            layer_id: The layer's decoder index, as `RadixAttention.layer_id` carries it.
            geometry: The checkpoint's index geometry.
            design: The design.
            backend: The attention backend whose pools the layer reads.
            shard: Which slots this rank owns under decode context parallel, or None for every slot.
            heads: The heads this rank holds under tensor parallel, or None where it holds every key head.
            check_index: Whether to check after every forward that every decode context-parallel rank holds the same index.
        """
        self.layer = layer_id
        self.geometry = geometry
        self.design = design
        self.backend = backend
        self.shard = shard or Shard()
        self.heads = heads
        self.check_index = check_index
        self.index: LayerIndex | None = None
        registry.register(self)
        registry.announce_once(Banner.BUILT, layer_id)

    def request_table(self) -> torch.Tensor:
        """Return the request table, each request's positions to KV slots [requests, context] int32."""
        table: torch.Tensor = self.backend.req_to_token
        return table

    def key_buffer(self) -> torch.Tensor:
        """Return this layer's key cache [slots, Hkv, dh] bf16.

        Raises:
            NotImplementedError: The KV cache is not bf16, so its keys are not the ones the maps were distilled on.
        """
        buffer: torch.Tensor = self.backend.token_to_kv_pool.get_key_buffer(self.layer)
        if buffer.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"the key and query input reads a bf16 KV cache - this one is {buffer.dtype}, serve kv_cache_dtype auto"
            )
        return buffer.view(-1, self.geometry.kv_head_count, self.geometry.head_dim)

    def ready(self) -> LayerIndex:
        """Build the layer's index once, and return it.

        Returns:
            The layer's index.
        """
        if self.index is not None:
            return self.index
        table = self.request_table()
        pool = self.backend.token_to_kv_pool
        maps = weights.read(
            self.design.serving.projections,
            self.geometry.key_dim,
            self.design.indexer.index_dim,
            self.geometry.layers,
        )
        before = torch.cuda.memory_allocated(table.device)
        split = None
        if self.heads is not None:
            start, stop = self.heads.dims(self.geometry)
            split = KeySplit(start, stop, self.heads.counts_key, context_parallel.attention_tp_sum)
        self.index = LayerIndex(
            self.design,
            maps[self.layer],
            pool.size + pool.page_size,
            pool.size,
            self.request_table,
            table.shape[1],
            settings.prefill_chunk(),
            table.device,
            self.shard,
            check_index=self.check_index,
            key_buffer=self.key_buffer,
            key_slot=self.geometry.key_slot,
            prefix_reuse=self.design.serving.prefix_reuse and settings.caches_prefixes(),
            split=split,
        )
        memory.check_built(self.layer, torch.cuda.memory_allocated(table.device) - before, pool.size)
        if self.index.check_index:
            registry.announce_once(Banner.CHECKING, self.layer)
        if self.index.prefix_reuse:
            registry.announce_once(Banner.SHARING, self.layer)
        return self.index

    def maintain(self, forward_batch: Any) -> None:
        """Bring each request's cache up to the tokens already in the KV pool, before every forward.

        Args:
            forward_batch: The batch about to be forwarded, with `lsa_rows` on it.
        """
        rows = forward_batch.lsa_rows
        if self.ready().maintain(forward_batch.rids, rows, registry.ends_of(forward_batch, len(rows))):
            registry.announce_once(Banner.TAKEN, self.layer)

    def select(self, q: torch.Tensor, k: torch.Tensor, forward_batch: Any) -> torch.Tensor:
        """Select for every query of a forward, from the heads attention was handed.

        Args:
            q: Every query head [T, Hq * dh] or [T, Hq, dh], rotated on the prefix alone - the group's, gathered,
                under decode context parallel, and the rank's own under tensor parallel.
            k: Key heads [T, Hkv, dh], the same - already in the pool at `out_cache_loc`, on the rank that owns it,
                and the rank's own heads under tensor parallel.
            forward_batch: The batch.

        Returns:
            KV slots [T, width] int32, -1 padded, valid entries first.
        """
        index = self.ready()
        inputs = index_space.key_and_query(q, k, self.geometry, self.heads)
        positions = forward_batch.positions.to(torch.int32)
        slots = forward_batch.out_cache_loc.long()
        if forward_batch.forward_mode.is_extend():
            return index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
            )
        return index.decode(inputs, positions, slots, forward_batch.req_pool_indices.long())


def resolve_all(targets: tuple[str, ...], what: str) -> None:
    """Resolve every symbol a served layout reads, before a request reaches one.

    Args:
        targets: The dotted paths.
        what: What reads them, for the message.

    Raises:
        RuntimeError: A target does not resolve on this sglang.
    """
    missing = []
    for target in targets:
        try:
            plugin.resolve(target)
        except (AttributeError, ModuleNotFoundError, ImportError) as failure:
            missing.append(f"{target} ({failure})")
    if missing:
        raise RuntimeError(f"lsa-lsa cannot serve {what} on this sglang - " + "; ".join(missing))


def decode_shard() -> Shard:
    """Return which slots this rank owns under decode context parallel - every slot without it."""
    size, rank = context_parallel.decode_cp()
    if size == 1:
        return Shard()
    return Shard(size, rank, context_parallel.decode_group_sum)


def refuse_unservable(
    backend: Any, geometry: index_space.Geometry, design: DesignSettings
) -> index_space.HeadShard | None:
    """Refuse a server this backend cannot index, before it serves a request.

    Args:
        backend: The FlashAttention backend being built.
        geometry: The checkpoint's index geometry.
        design: The design the server runs.

    Returns:
        The heads this rank holds where tensor parallel splits the key heads, or None where it holds every one.

    Raises:
        NotImplementedError: A mesh, a paging, a cache or a decoding mode this backend cannot serve.
        ValueError: The design reads another input than the key and query.
        RuntimeError: A symbol the backend reads is missing from this sglang, or the metadata glue graph is on.
    """
    resolve_all(REQUIRED, "Qwen")
    from sglang.srt.runtime_context import get_parallel, get_spec

    if design.indexer.input != KEY_AND_QUERY:
        raise ValueError(
            f"Qwen under LSA reads the key and query - the design asks for {design.indexer.input}, which has no "
            "call site on a dense model"
        )
    parallel = get_parallel()
    if parallel.attn_dp_size > 1:
        raise NotImplementedError(
            f"Qwen under LSA serves with attention tensor parallel, and under data-parallel attention "
            f"({parallel.attn_dp_size} groups) a rank holds only its own group's tokens, so no rank's index covers "
            "a request's context - serve without --enable-dp-attention"
        )
    tp_size, decode_size = parallel.attn_tp_size, context_parallel.decode_cp()[0]
    heads = index_space.head_shard(geometry, tp_size, parallel.attn_tp_rank, decode_size)
    if parallel.attn_cp_size > 1 or parallel.enable_prefill_cp:
        raise NotImplementedError(
            "Qwen under LSA serves without prefill context parallel - decode context parallel alone splits the KV "
            "pool, and every rank prefills the whole chunk"
        )
    if decode_size > 1:
        resolve_all(REQUIRED_DECODE_GROUP, "Qwen under decode context parallel")
    if backend.page_size != 1:
        raise NotImplementedError(
            f"an LSA list is a set of single tokens and this server pages {backend.page_size} at a time - serve "
            "with --page-size 1"
        )
    if backend.kv_index_translator.is_translating:
        raise NotImplementedError(
            "LSA reads the request table's slots as they are - this pool translates them"
        )
    if get_spec().speculative_algorithm is not None:
        raise NotImplementedError("Qwen under LSA serves without speculative decoding")
    plugin.refuse_metadata_glue()
    return heads


def arm(result: None, self: Any, *args: Any, **kwargs: Any) -> None:
    """Give a Qwen server's FlashAttention or triton backend an indexer for every quadratic layer.

    A backend built for anything else - another model, a draft - is left as it was. A second backend of the same
    server - the triton decode beside a FlashAttention prefill - reads the indexers the first one built, since the
    pools they index are the runner's.

    Args:
        result: What `__init__` returned, which is nothing.
        self: The backend.
        *args: Its arguments, the model runner first.
        **kwargs: The same, by name.

    Returns:
        None - the backend is armed in place.

    Raises:
        TypeError: The backend was built without a model runner.
    """
    runner = kwargs.get("model_runner", args[0] if args else None)
    if runner is None:
        raise TypeError("the attention backend was built without a model runner")
    text = runner.model_config.hf_text_config
    if getattr(self, "is_draft_runner", False) or not index_space.is_grouped_query(text):
        return None
    geometry = index_space.geometry_of(text)
    config = runner.model_config.hf_config
    served, design = settings.of(config), settings.design_of(config)
    heads = refuse_unservable(self, geometry, design)
    if not ARMED:
        shard = decode_shard()
        ARMED.update(
            {
                layer: GroupedQueryIndexer(
                    layer, geometry, design, self, shard, heads, served.debug.check_index
                )
                for layer in geometry.layers
            }
        )
    setattr(self, INDEXERS, ARMED)
    if served.debug.dumps:
        print("lsa-lsa: the dump reads GLM's x - Qwen's layers dump nothing", flush=True)
    return None


def indexer_of(backend: Any, layer: Any, forward_batch: Any) -> GroupedQueryIndexer | None:
    """Return the indexer a layer reads on this forward, or None for one the backend serves as it was.

    Args:
        backend: The FlashAttention backend.
        layer: The attention layer.
        forward_batch: The batch.
    """
    indexers: dict[int, GroupedQueryIndexer] | None = getattr(backend, INDEXERS, None)
    if not indexers or forward_batch.forward_mode.is_idle():
        return None
    return indexers.get(layer.layer_id)


def maintain(*args: Any, **kwargs: Any) -> None:
    """Bring every index up to the tokens already in the KV pool, before an eager forward's metadata is built.

    Args:
        *args: The backend and its forward batch.
        **kwargs: The same, by name.

    Returns:
        None, which leaves the arguments as they were.
    """
    if getattr(args[0], INDEXERS, None):
        registry.maintain(kwargs.get("forward_batch") or args[1])
    return None


def maintain_out_graph(*args: Any, **kwargs: Any) -> None:
    """Do the same before a captured forward replays, and not while one is captured.

    Args:
        *args: The backend, its forward batch, and possibly the capture flag.
        **kwargs: The same, by name.

    Returns:
        None, which leaves the arguments as they were.
    """
    in_capture = kwargs.get("in_capture", args[2] if len(args) > 2 else False)
    if getattr(args[0], INDEXERS, None) and not in_capture:
        registry.maintain(kwargs.get("forward_batch") or args[1])
    return None


def save(
    backend: Any, layer: Any, k: torch.Tensor, v: torch.Tensor, forward_batch: Any, shard: Shard
) -> None:
    """Write this forward's keys and values to the pool, before the index reads them - as the dense backend would.

    Under decode context parallel a rank keeps the tokens it owns, at row `slot // size`, as the image's backends do.

    Args:
        backend: The attention backend.
        layer: The attention layer.
        k: Key heads [T, Hkv, dh].
        v: Value heads [T, Hkv, dh].
        forward_batch: The batch, carrying `out_cache_loc`.
        shard: Which slots this rank owns.
    """
    pool = backend.token_to_kv_pool
    if shard.size > 1:
        rows = forward_batch.out_cache_loc // shard.size
        positions = forward_batch.positions
        owned = (
            positions % shard.size == shard.rank
            if positions is not None and positions.numel() == rows.numel()
            else forward_batch.dcp_kv_mask
        )
        pool.set_kv_buffer(layer, rows, k, v, layer.k_scale, layer.v_scale, dcp_kv_mask=owned)
        return
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    location = KVWriteLoc(forward_batch.out_cache_loc, backend.forward_metadata.swa_out_cache_loc)
    pool.set_kv_buffer(layer, location, k, v, layer.k_scale, layer.v_scale)


def sparse_read(backend: Any, q: torch.Tensor, layer: Any, lists: torch.Tensor) -> torch.Tensor:
    """Attend each query over the KV slots its list names.

    Args:
        backend: The FlashAttention backend, whose kernel and pool this reads.
        q: Query heads [T, Hq * dh] or [T, Hq, dh].
        layer: The attention layer, carrying the scale and the logit cap.
        lists: KV slots per query [T, width] int32, -1 padded, valid entries first.

    Returns:
        The attention output [T, Hq * dh].
    """
    if not flash_reads(lists.device):
        return triton_sparse_read(backend, q, layer, lists)
    token_count = lists.shape[0]
    lengths = read.list_lengths(lists)
    cu_seqlens_k = torch.zeros(token_count + 1, dtype=torch.int32, device=lists.device)
    torch.cumsum(lengths, 0, dtype=torch.int32, out=cu_seqlens_k[1:])
    keys, values = backend.get_paged_mha_kv_cache(layer)
    out = backend.flash_attn_with_kvcache(
        q=q.contiguous().view(token_count, layer.tp_q_head_num, layer.head_dim),
        k_cache=keys,
        v_cache=values,
        page_table=lists.clamp(min=0),
        cache_seqlens=lengths,
        cu_seqlens_q=torch.arange(token_count + 1, dtype=torch.int32, device=lists.device),
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=layer.scaling,
        causal=True,
        softcap=layer.logit_cap,
        num_splits=backend.num_splits,
    )
    return out.view(token_count, layer.tp_q_head_num * layer.head_dim)


def flash_reads(device: torch.device) -> bool:
    """Tell whether FlashAttention's paged read at a page of one slot runs on this GPU.

    FA3 is Hopper's alone, and FA4 on Blackwell pages 128 slots at a time, so neither reads a list of single slots
    anywhere else. A selection off the GPU is a test's, read through whatever kernel the backend carries.

    Args:
        device: The device the selection is on.
    """
    return device.type != "cuda" or dispatch.capability(device)[0] == dispatch.HOPPER[0]


# Splits of one query's list the Triton read reduces over - a prefill chunk has rows enough to fill the GPU,
# a decode batch does not
TRITON_SPLITS_PREFILL: int = 1
TRITON_SPLITS_DECODE: int = 8

# More sequences than this is a prefill chunk's rows rather than a decode batch's
TRITON_PREFILL_ROWS: int = 4096


def triton_sparse_read(backend: Any, q: torch.Tensor, layer: Any, lists: torch.Tensor) -> torch.Tensor:
    """Attend each query over the KV slots its list names, through sglang's Triton decode kernel.

    Args:
        backend: The attention backend, whose pool this reads.
        q: Query heads [T, Hq * dh] or [T, Hq, dh].
        layer: The attention layer, carrying the scale and the logit cap.
        lists: KV slots per query [T, width] int32, -1 padded, valid entries first.

    Returns:
        The attention output [T, Hq * dh].
    """
    pool = backend.token_to_kv_pool
    token_count = lists.shape[0]
    out = triton_read(
        q.contiguous().view(token_count, layer.tp_q_head_num, layer.head_dim),
        pool.get_key_buffer(layer.layer_id),
        pool.get_value_buffer(layer.layer_id),
        lists,
        read.list_lengths(lists),
        layer,
    )
    return out.view(token_count, layer.tp_q_head_num * layer.v_head_dim)


def slot_index_dtype(keys: torch.Tensor, values: torch.Tensor) -> torch.dtype:
    """Return the dtype the Triton decode kernel's slot indices must take to address every row of the pool.

    The kernel offsets a slot by its row stride in the index's own dtype, so an int32 slot past 2**31 elements of the
    pool - 2M slots of 4 key heads of 256, a 10M pool on one card - wraps and reads out of bounds.

    Args:
        keys: The pool's keys [slots, Hkv, dh].
        values: The pool's values [slots, Hkv, dv].

    Returns:
        int64 when an offset into either buffer can pass 2**31, int32 otherwise.
    """
    largest = max(keys.shape[0] * keys.stride(0), values.shape[0] * values.stride(0))
    return torch.int64 if largest >= 2**31 else torch.int32


def triton_read(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lists: torch.Tensor,
    lengths: torch.Tensor,
    layer: Any,
) -> torch.Tensor:
    """Attend each sequence of one query over the slots its list names, through sglang's Triton decode kernel.

    Args:
        q: One query per sequence [S, H, dh], its heads grouped over the pool's key heads.
        keys: The pool's keys [slots, Hkv, dh].
        values: The pool's values [slots, Hkv, dv].
        lists: Slots per sequence [S, width] int32, the first `lengths` of each row real.
        lengths: Real entries of each row [S] int32, at least 1.
        layer: The attention layer, carrying the scale, the logit cap and the KV scales.

    Returns:
        The attention output [S, H, dv].
    """
    out, _ = triton_decode(q, keys, values, lists, lengths, layer, q.dtype)
    return out


def triton_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lists: torch.Tensor,
    lengths: torch.Tensor,
    layer: Any,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sglang's Triton decode kernel over each sequence's list, keeping each split's log-sum-exp.

    Every row is a sequence of its own whose keys are its list, so the kernel's indirection over KV slots reads
    the selection as FlashAttention's page table of one slot does. The list is packed without a host sync, so a
    decode step stays capturable.

    Args:
        q: One query per sequence [S, H, dh], its heads grouped over the pool's key heads.
        keys: The pool's keys [slots, Hkv, dh].
        values: The pool's values [slots, Hkv, dv].
        lists: Slots per sequence [S, width] int32, the first `lengths` of each row real.
        lengths: Real entries of each row [S] int32, at least 1.
        layer: The attention layer, carrying the scale, the logit cap and the KV scales.

        out_dtype: The output's dtype - the query's for a whole read, fp32 for a partial one.

    Returns:
        The attention output [S, H, dv].
        Its log-sum-exp per split [S, H, splits] fp32, minus infinity for a split that read nothing.
    """
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    sequences, width = lists.shape
    heads = q.shape[1]
    device = lists.device
    kv_indptr = torch.zeros(sequences + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths, 0, dtype=torch.int32, out=kv_indptr[1:])
    # Each real entry lands at its row's start plus its column, the padding at one spare slot past the end
    columns = torch.arange(width, dtype=torch.int32, device=device)
    spare = sequences * width
    destination = torch.where(columns[None, :] < lengths[:, None], kv_indptr[:-1, None] + columns, spare)
    kv_indices = torch.zeros(spare + 1, dtype=slot_index_dtype(keys, values), device=device)
    kv_indices.scatter_(0, destination.flatten().long(), lists.clamp(min=0).flatten().to(kv_indices.dtype))
    splits = TRITON_SPLITS_PREFILL if sequences > TRITON_PREFILL_ROWS else TRITON_SPLITS_DECODE
    num_kv_splits = torch.full((sequences,), splits, dtype=torch.int32, device=device)
    attn_logits = torch.empty(
        (sequences, heads, splits, values.shape[-1]), dtype=torch.float32, device=device
    )
    attn_lse = torch.full((sequences, heads, splits), float("-inf"), dtype=torch.float32, device=device)
    out = torch.empty((sequences, heads, values.shape[-1]), dtype=out_dtype, device=device)
    scaled = layer.k_scale is not None and layer.v_scale is not None
    decode_attention_fwd(
        q,
        keys,
        values,
        out,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        splits,
        layer.scaling,
        layer.k_scale_float if scaled else 1.0,
        layer.v_scale_float if scaled else 1.0,
        logit_cap=layer.logit_cap,
        enable_lean=False,
    )
    return out, attn_lse


def owned_lists(lists: torch.Tensor, shard: Shard) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected slots this rank owns, as rows of its pool, valid entries first.

    Args:
        lists: KV slots per query [T, width] int32, -1 padded, valid entries first - the same on every rank.
        shard: Which slots this rank owns.

    Returns:
        Pool rows per query [T, width] int32, -1 padded, the rows this rank holds first.
        How many each query has on this rank [T] int32, 0 where it has none.
    """
    owned = (lists > Sentinel.PAD) & (lists % shard.size == shard.rank)
    rows = torch.where(owned, lists // shard.size, Sentinel.PAD)
    # A row is never negative, so a descending sort puts every one before the padding
    return rows.sort(dim=1, descending=True).values, owned.sum(1, dtype=torch.int32)


def flash_kernel(backend: Any) -> Any:
    """Return FlashAttention's paged read - the backend's own, or the image's FA3 kernel beside the triton backend.

    Args:
        backend: The attention backend.
    """
    kernel = getattr(backend, "flash_attn_with_kvcache", None)
    if kernel is not None:
        return kernel
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    return flash_attn_with_kvcache


def paged_cache(backend: Any, layer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this layer's pool as pages of one slot.

    Args:
        backend: The attention backend.
        layer: The attention layer.

    Returns:
        The keys [rows, 1, Hkv, dh] and the values [rows, 1, Hkv, dh].
    """
    keys, values = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    return (
        keys.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
        values.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
    )


def partial_read(
    backend: Any, q: torch.Tensor, layer: Any, rows: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend every query head over the selected rows this rank holds, keeping the log-sum-exp the merge weighs by.

    Args:
        backend: The attention backend.
        q: Every query head [T, Hq, dh].
        layer: The attention layer.
        rows: Pool rows per query [T, width] int32, valid entries first.
        counts: How many of each row are valid [T] int32.

    Returns:
        The partial output [T, Hq, dh] fp32.
        Its log-sum-exp [T, Hq] fp32, minus infinity for a query this rank holds nothing of.
    """
    token_count = rows.shape[0]
    # An empty read is a softmax over nothing - read one row and give it no weight
    lengths = counts.clamp(min=1)
    if not flash_reads(rows.device):
        pool = backend.token_to_kv_pool
        out, split_lse = triton_decode(
            q.contiguous(),
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
            rows,
            lengths,
            layer,
            torch.float32,
        )
        lse = torch.logsumexp(split_lse, dim=-1).masked_fill(counts[:, None] == 0, float("-inf"))
        return out, lse
    cu_seqlens_k = torch.zeros(token_count + 1, dtype=torch.int32, device=rows.device)
    torch.cumsum(lengths, 0, dtype=torch.int32, out=cu_seqlens_k[1:])
    keys, values = paged_cache(backend, layer)
    out, lse, *_ = flash_kernel(backend)(
        q=q.contiguous(),
        k_cache=keys,
        v_cache=values,
        page_table=rows.clamp(min=0),
        cache_seqlens=lengths,
        cu_seqlens_q=torch.arange(token_count + 1, dtype=torch.int32, device=rows.device),
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=layer.scaling,
        causal=True,
        softcap=layer.logit_cap,
        num_splits=getattr(backend, "num_splits", 0),
        return_softmax_lse=True,
    )
    # The varlen kernel's log-sum-exp is [Hq, T]
    lse = lse.transpose(0, 1).float().masked_fill(counts[:, None] == 0, float("-inf"))
    return out.float(), lse


def merge_partials(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Merge the decode group's partial reads into this rank's own heads.

    Args:
        out: This rank's partial output [T, Hq, dh] fp32.
        lse: Its log-sum-exp [T, Hq] fp32.

    Returns:
        The output of this rank's heads [T, Hq / size, dh] fp32 - the softmax over every rank's keys.
    """
    from sglang.srt.layers.dcp import cp_lse_ag_out_rs_mha
    from sglang.srt.runtime_context import get_parallel

    merged: torch.Tensor = cp_lse_ag_out_rs_mha(out, lse, get_parallel().dcp_group)
    return merged


def gather_heads(q: torch.Tensor, layer: Any) -> torch.Tensor:
    """Gather the decode group's query heads, in the checkpoint's head order.

    Args:
        q: This rank's query heads [T, Hq / size * dh] or [T, Hq / size, dh].
        layer: The attention layer.

    Returns:
        Every query head [T, Hq, dh].
    """
    from sglang.srt.runtime_context import get_parallel

    local = q.reshape(-1, layer.tp_q_head_num, layer.head_dim).contiguous()
    gathered: torch.Tensor = get_parallel().dcp_group.all_gather(local, dim=1)
    return gathered.contiguous()


def group_read(
    backend: Any,
    q: torch.Tensor,
    layer: Any,
    lists: torch.Tensor,
    shard: Shard,
    merge: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = merge_partials,
) -> torch.Tensor:
    """Attend each query over the KV slots its list names, the list's keys spread across a decode group.

    Args:
        backend: The attention backend.
        q: Every query head [T, Hq, dh], the group's gathered.
        layer: The attention layer.
        lists: KV slots per query [T, width] int32, -1 padded, the same on every rank.
        shard: Which slots this rank owns.
        merge: Merges the group's partial outputs and log-sum-exps into this rank's heads.

    Returns:
        The attention output of this rank's heads [T, Hq / size * dh].
    """
    rows, counts = owned_lists(lists, shard)
    out, lse = partial_read(backend, q, layer, rows, counts)
    return merge(out, lse).to(q.dtype).reshape(q.shape[0], -1)


def select_and_read(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    """Serve a quadratic layer's attention from its selection, or hand any other call to the backend as it was.

    Under decode context parallel the group's query heads are gathered first, so the selection sums every head and
    each rank's share of the read attends with every one.

    Args:
        original: `forward_extend` or `forward_decode`.
        self: The FlashAttention or triton backend.
        *args: Its arguments.
        **kwargs: The same, by name.

    Returns:
        The attention output of this rank's heads [T, Hq * dh].

    Raises:
        RuntimeError: A prefill is being captured into a CUDA graph, which would replay one batch's selection.
        NotImplementedError: The forward is a speculative verify or draft - refused at arming, checked again.
    """
    bound = plugin.bind(original, self, *args, **kwargs)
    arguments = bound.arguments
    q, k, v, layer, forward_batch = (arguments[name] for name in ("q", "k", "v", "layer", "forward_batch"))
    indexer = indexer_of(self, layer, forward_batch)
    if indexer is None or k is None:
        return original(*bound.args, **bound.kwargs)
    mode = forward_batch.forward_mode
    if mode.is_extend() and (torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
        raise RuntimeError("LSA selects a prefill on the host - serve with --disable-prefill-cuda-graph")
    if mode.is_target_verify() or mode.is_draft_extend_v2():
        raise NotImplementedError(
            "Qwen under LSA reads a plain prefill or decode, without speculative decoding"
        )
    shard = indexer.shard
    if arguments["save_kv_cache"]:
        save(self, layer, k, v, forward_batch, shard)
    if shard.size == 1:
        return sparse_read(self, q, layer, indexer.select(q, k, forward_batch))
    every_head = gather_heads(q, layer)
    return group_read(self, every_head, layer, indexer.select(every_head, k, forward_batch), shard)
