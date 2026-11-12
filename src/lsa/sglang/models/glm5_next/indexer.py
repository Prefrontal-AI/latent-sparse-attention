"""GLM-5.3-Flash's indexer under the design - index keys and index queries in place of the checkpoint's own.

The design's input, and the default, is the key and the heads' queries: the key latent and the absorbed queries of
this rank's heads, which the plugin hands over from the end of `forward_absorb_prepare` - the frozen design's input,
served with maps distilled on `x` and re-exported for it. `query_heads: all`, the default, sums the ranks' head sums
across the group (the original implementation's `query: all`), so every rank selects the one list; `query_heads: rank` has each
tensor-parallel rank sum its own heads (the original implementation's `query: local`), so ranks may select different lists, each for its
own heads. Under context parallel it serves with attention one GPU wide and `query_heads: all`, as the original implementation served it
under prefill and decode context parallel 8: a rank holds every head, so the query side is its own, and the key side
reads the layouts below - a prefill chunk's key latents arrive whole, gathered by the fork for attention, and a key in
another decode rank's share of the pool is projected there and its index key summed across the group.

The attention input `x` is a research option (`input: attention_input`) - on RULER cwe 512k it lost about 0.15 against
native where the key and query matched it. `x` is the attention module's input: the four mHC streams
mixed and run through the input norm, which the model hands to `self.indexer(x=hidden_states, ...)` at the call site
its own DSA indexer reads it at. It is position-free and headless, so there is no rotary to undo, no head to regroup
and no head to sum - one projection of `x` is the index key, another is the index query.

`x` is whole on every tensor-parallel rank. A rank's attention gets the gathered `x` - mHC all-gathers a scattered
input before a DSA layer, and without `--enable-attn-tp-input-scattered` nothing scatters it - so every rank projects
the same `x` with the same maps and selects the same lists, and one selection serves its heads. The layer refuses data-parallel attention, where a
rank holds only its own group's tokens.

Under context parallel the layer runs on every rank of the fork's layouts, and [context_parallel](../../context_parallel.py)
says which. Prefill context parallel hands a rank its own span of the input: it projects those rows, gathers the chunk's
index keys so the index is built over the whole chunk on every rank, and selects for its own rows. Decode context
parallel splits the KV pool, and the index keys with it - a rank writes those of the slots it owns, reads the others' by
a sum across the group, and keeps a bucket cache for the whole context. A decode batch is the same on every rank, and so
is its selection. A prefill under decode context parallel is read by the fork as rows of [prefix, chunk] keys, so its
lists hold those rows and not KV slots.

Same call contract as the DSA indexer: int32 KV slots [tokens, width], -1 padded, valid entries first. The width is
`top_k + window_size + sink_size`, which the plugin sets as the model's `index_topk`. The index catches up with the KV
pool on the host before each forward, through the attention backend's metadata hook.
"""

from typing import Any

import torch
from sglang.srt.configs.model_config import dsa_layer_skips_topk
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import is_graph_dsa_split_op_surface
from sglang.srt.layers.dp_attention import attn_tp_all_reduce, is_dp_attention_enabled
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

from lsa.sglang import context_parallel, memory, settings
from lsa.sglang.constants import Banner
from lsa.sglang.index import registry, weights
from lsa.sglang.index.dump import Dump
from lsa.sglang.index.layer_index import KeyAndQuery, LayerIndex, Shard, Span
from lsa.sglang.settings.design import KEY_AND_QUERY, QUERY_HEADS_ALL


def request_table() -> torch.Tensor:
    """Return the request table, each request's positions to KV slots.

    Hybrid backends wrap the pool, and plain DSA backends alias the table directly.
    """
    return get_attn_backend().req_to_token_pool.req_to_token


def chunk_span(forward_batch: Any, device: torch.device | str) -> Span:
    """Return what this rank holds of a prefill chunk under prefill context parallel.

    Args:
        forward_batch: The context-parallel batch.
        device: Where the positions go.

    Returns:
        The chunk's positions and per-request counts, and the gather that completes a rank's rows.
    """
    key_counts = list(forward_batch.extend_seq_lens_cpu)
    return Span(
        context_parallel.chunk_positions(forward_batch.extend_prefix_lens_cpu, key_counts, device),
        key_counts,
        context_parallel.local_query_counts(forward_batch),
        context_parallel.local_query_starts(forward_batch),
        lambda rows: context_parallel.gather_chunk(rows, forward_batch),
    )


class LSAIndexer(Indexer):
    """One sparse layer's indexer, a `LayerIndex` behind the checkpoint's own indexer class so the weights still load."""

    is_lsa = True

    def __init__(self, **indexer_kwargs: Any) -> None:
        """Build the checkpoint's indexer modules, which load the weights, and refuse a server this cannot index.

        Args:
            **indexer_kwargs: What the model builds an indexer with.

        Raises:
            ValueError: `index_topk` is not the design's list width, the metadata glue graph is on, or the design
                has bucket shapes under decode context parallel, or prefill context parallel splits a chunk other
                than contiguously, or the design reads the key and query under context parallel with attention
                wider than one GPU or `query_heads` other than `all`.
            RuntimeError: Attention is data parallel, so `x` is not whole on a rank.
        """
        super().__init__(**indexer_kwargs)
        config = indexer_kwargs.get("config")
        self.design = settings.design_of(config)
        self.debug = settings.of(config).debug
        if getattr(config, "index_kpool", 1) != 1 or self.index_topk != self.design.width:
            raise ValueError(
                f"index_topk must be the design's list width {self.design.width} and index_kpool 1 - "
                "the plugin sets both on hf_text_config"
            )
        if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
            raise ValueError(
                "LSA maintenance runs on the host before each replay - disable the metadata glue graph"
            )
        if is_dp_attention_enabled():
            raise RuntimeError("LSA reads x whole on every rank - serve without --enable-dp-attention")
        context_parallel.refuse_shapes(self.design, context_parallel.decode_cp()[0])
        context_parallel.refuse_split(context_parallel.prefill_split())
        self.key_and_query = self.design.indexer.input == KEY_AND_QUERY
        decode_size, split = context_parallel.decode_cp()[0], context_parallel.prefill_split()
        if self.key_and_query and (decode_size > 1 or split is not None):
            context_parallel.refuse_key_and_query(
                self.design, decode_size, split, context_parallel.attention_tp_size()
            )
        self.layer = indexer_kwargs["layer_id"]
        self.input_dim = weights.input_dim(config, self.design.indexer.input)
        self.model_dim = indexer_kwargs["hidden_size"]
        self.layers = weights.indexer_layers(config, dsa_layer_skips_topk)
        self.chunk = settings.prefill_chunk()
        self.index: LayerIndex | None = None
        self.dump: Dump | None = None
        registry.register(self)
        registry.announce_once(Banner.BUILT, self.layer)

    def ready(self) -> LayerIndex:
        """Build the layer's index once the model and the KV pool exist, and return it.

        Returns:
            The layer's index.
        """
        if self.index is not None:
            return self.index
        table = request_table()
        pool = get_token_to_kv_pool()
        maps = weights.read(
            self.design.serving.projections, self.input_dim, self.design.indexer.index_dim, self.layers
        )
        # Under decode context parallel the pool holds this rank's share of the slots, while the index covers
        # every token of a request
        size, rank = context_parallel.decode_cp()
        before = torch.cuda.memory_allocated(table.device)
        self.index = LayerIndex(
            self.design,
            maps[self.layer],
            pool.size + pool.page_size,
            pool.size,
            request_table,
            table.shape[1],
            self.chunk,
            table.device,
            Shard(size, rank, context_parallel.decode_group_sum if size > 1 else None),
            check_index=self.debug.check_index,
            key_buffer=self.key_buffer if self.key_and_query else None,
            sum_heads=attn_tp_all_reduce if self.design.indexer.query_heads == QUERY_HEADS_ALL else None,
            prefix_reuse=self.design.serving.prefix_reuse and settings.caches_prefixes(),
        )
        memory.check_built(self.layer, torch.cuda.memory_allocated(table.device) - before, pool.size)
        if self.index.check_index:
            registry.announce_once(Banner.CHECKING, self.layer)
        if self.index.prefix_reuse:
            registry.announce_once(Banner.SHARING, self.layer)
        self.dump = Dump.of(self.debug.dump, self.layer, self.model_dim, self.design.width, table.device)
        return self.index

    def key_buffer(self) -> torch.Tensor:
        """Return this layer's KV pool buffer, whose first `kv_lora_rank` dims of a slot are its key latent.

        Returns:
            The buffer [slots, 1, kv_lora_rank + rope dims] bf16.

        Raises:
            NotImplementedError: The KV cache is not bf16, so its keys are not the latents the maps were distilled on.
        """
        buffer = get_token_to_kv_pool().get_key_buffer(self.layer)
        if buffer.dtype != torch.bfloat16:
            raise NotImplementedError(
                "the key and query input reads a bf16 KV cache - serve kv_cache_dtype bf16"
            )
        return buffer

    def forward_cuda(
        self,
        x: torch.Tensor | None,
        q_lora: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        layer_id: int,
        return_indices: bool = True,
        key: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Write this batch's index keys and select, or only write them when no selection is asked for.

        A short prompt is prefilled by the dense path, which calls the indexer for its keys alone.

        Args:
            x: The attention's input [S, model_dim] bf16, or None under the key and query input.
            q_lora: The latent query, which the design does not read.
            positions: Each token's position [S].
            forward_batch: The batch.
            layer_id: This layer.
            return_indices: Whether a selection is wanted.
            key: The key latent [S, kv_lora_rank] bf16, under the key and query input.
            query: The absorbed queries of this rank's heads [S, heads, kv_lora_rank] bf16, under the key and query
                input.

        Returns:
            KV slots [S, width] int32, -1 padded, or None.

        Raises:
            NotImplementedError: `x` arrived quantized, as a tuple.
            ValueError: The design's input did not arrive.
            RuntimeError: The prefill is captured in a piecewise or breakable CUDA graph, which the selection cannot be.
        """
        if forward_batch.forward_mode.is_idle():
            return None
        if is_graph_dsa_split_op_surface(forward_batch):
            raise RuntimeError("LSA selects eagerly - serve with the prefill CUDA graph disabled")
        inputs: torch.Tensor | KeyAndQuery
        if self.key_and_query:
            if key is None or query is None:
                raise ValueError("the key and query input needs both, from forward_absorb_prepare")
            inputs = KeyAndQuery(key, query)
        elif isinstance(x, tuple):
            raise NotImplementedError("LSA reads x in bf16 - the attention input arrived quantized")
        elif x is None:
            raise ValueError("the attention input needs x")
        else:
            inputs = x
        index = self.ready()
        extend = forward_batch.forward_mode.is_extend()
        span = (
            chunk_span(forward_batch, positions.device)
            if extend and context_parallel.is_active(forward_batch)
            else None
        )
        if (
            span is not None
            and isinstance(inputs, KeyAndQuery)
            and not context_parallel.chunk_key_whole(forward_batch)
        ):
            # The fork gathers a chunk's latents for its DSA attention, and where it did not they are this rank's
            inputs = KeyAndQuery(span.gather(inputs.key), inputs.query)
        slots = (
            self.chunk_slots(forward_batch, span) if span is not None else forward_batch.out_cache_loc.long()
        )
        if not return_indices:
            index.write_chunk(inputs, slots, span)
            if self.dump is not None and span is None and isinstance(x, torch.Tensor):
                self.dump.prefill(x, positions, None, forward_batch, request_table())
            return None
        positions = positions.to(torch.int32)
        if extend:
            # Decode context parallel's prefill reads rows of [prefix, chunk] keys rather than KV slots
            ragged = (
                context_parallel.ragged_offsets(forward_batch.seq_lens_cpu.tolist())
                if context_parallel.decode_cp()[0] > 1
                else None
            )
            lists = index.extend(
                inputs,
                positions,
                slots,
                forward_batch.lsa_rows,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
                span,
                ragged,
            )
            if self.dump is not None and span is None and isinstance(x, torch.Tensor):
                self.dump.prefill(x, positions, lists, forward_batch, request_table())
            return lists
        lists = index.decode(inputs, positions, slots, forward_batch.req_pool_indices.long())
        if self.dump is not None and isinstance(x, torch.Tensor):
            self.dump.decode(x, positions, lists)
        return lists

    def chunk_slots(self, forward_batch: Any, span: Span) -> torch.Tensor:
        """Return the KV slots of the whole chunk's new tokens, which a rank's `out_cache_loc` does not list.

        Args:
            forward_batch: The context-parallel batch, with `lsa_rows` on it.
            span: What this rank holds of the chunk.

        Returns:
            The slots [chunk tokens] int64, request by request in position order.
        """
        table = request_table()
        slots, start = [], 0
        for row, count in zip(forward_batch.lsa_rows, span.key_counts, strict=True):
            slots.append(table[row, span.positions[start : start + count].long()])
            start += count
        return torch.cat(slots).long()

    def maintain(self, forward_batch: Any) -> None:
        """Bring each request's cache up to the tokens already in the KV pool, eagerly, before every forward.

        Args:
            forward_batch: The batch about to be forwarded, with `lsa_rows` on it.
        """
        index = self.ready()
        rows = forward_batch.lsa_rows
        if self.dump is not None:
            self.dump.before_forward(forward_batch, rows, request_table())
        ends = registry.ends_of(forward_batch, len(rows))
        if index.maintain(forward_batch.rids, rows, ends):
            registry.announce_once(Banner.TAKEN, self.layer)
