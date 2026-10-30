"""Qwen3.5's attention under PISA - the paper's pyramid search on each key-value head, read by that head's query heads alone.

The hooks are LSA's on the same backends - see [backend](backend.py) - with the selection and the read swapped.
PISA selects one list a key-value head, as the paper does: a key head's rotated keys in the KV pool, its six query heads'
rotated queries, a block scored by the LogSumExp over its children of the softmax scale times each head's dot, summed over
the six. So each key head has its own index of leaf means, and nothing is projected or summed across heads.

The read is FlashAttention's paged read at a page size of 1, as LSA's, with every key head's list its own sequence. The
pool `[slots, Hkv, dh]` is read as one head of `slots * Hkv` slots, slot `s` of head `h` at `s * Hkv + h`, and a query
token's `Hkv` lists are `Hkv` sequences of one query row, each read by its group's query heads.

A rank holds whole key heads and every query head that reads them, so a tensor-parallel mesh up to the key head count is
served with no exchange between ranks. At twice the key head count a key head sits on two ranks, each with half its group's
query heads, and the pair sum their partial block scores at every level so both select the one list a smaller mesh selects.
A mesh that splits a group any other way - tensor parallel past twice the key heads, or decode context parallel - is
refused, as are a paged pool, speculative decoding and a cache other than bf16. A prefill's search is eager, and a decode
step's reads device tensors alone over a pyramid as wide as the request table, so the decode graph captures it.

It imports sglang, so nothing but the plugin reaches it, and only after the plugin resolved every target.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from lsa.sglang import context_parallel, memory, plugin, settings
from lsa.sglang.constants import Banner
from lsa.sglang.index import pisa, read, registry
from lsa.sglang.index.layer_index import Shard
from lsa.sglang.index.pisa import DecodeRecord, PisaIndex
from lsa.sglang.models.qwen3_5 import backend, index_space
from lsa.sglang.models.qwen3_5.index_space import HeadShard
from lsa.sglang.settings.debug import PisaCheck
from lsa.sglang.settings.design import PisaConfig

# Every quadratic layer's indexer, built by the first backend armed
ARMED: dict[int, "PisaGroupedQueryIndexer"] = {}

# What a key head's replicas read of sglang besides - the attention group's sum and gather
REQUIRED_REPLICAS: tuple[str, ...] = (
    "sglang.srt.layers.dp_attention.attn_tp_all_reduce",
    "sglang.srt.layers.dp_attention.attn_tp_all_gather_into_tensor",
)

# What a check line says of the step it held - nothing for an eager forward
CAPTURED_LABEL: dict[str, str] = {"forwards": "", "captured": " captured decode"}


@dataclass(frozen=True)
class Replicas:
    """The ranks that hold one key head, each with a share of the query heads that read it.

    The attention group is the key heads' replicas in order, so key head `h` sits on ranks `h * size` to `(h + 1) * size - 1`.

    Attributes:
        head: The key head this rank holds, its place in the model.
        head_count: Key heads of the model.
        size: Ranks that hold a key head.
        all_reduce: Sums a tensor over the attention group, the same result on every rank.
        gather: Stacks a tensor of every rank of the attention group [group, *shape].
    """

    head: int
    head_count: int
    size: int
    all_reduce: Callable[[torch.Tensor], torch.Tensor]
    gather: Callable[[torch.Tensor], torch.Tensor]

    def sum_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Sum a level's partial block scores over the ranks that hold this rank's key head.

        The scores go in this key head's slot of a tensor of zeros and the whole group sums it, so only the replicas
        add anything to a slot - and a sum of two scores and zeros is the same on both whatever order the group adds in.

        Args:
            scores: This rank's candidate blocks' scores over its own query heads [rows, width] fp32.

        Returns:
            The scores over the key head's whole group of query heads [rows, width] fp32, equal on every replica.
        """
        spread = scores.new_zeros(self.head_count, *scores.shape)
        spread[self.head] = scores
        return self.all_reduce(spread)[self.head]

    def group_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """Gather the query heads of this rank's key head from every replica, in head order.

        Args:
            queries: This rank's query heads of its key head [rows, share, dh].

        Returns:
            The key head's whole group [rows, size * share, dh].
        """
        stacked = self.gather(queries.contiguous()).view(self.head_count, self.size, *queries.shape)
        return stacked[self.head].transpose(0, 1).reshape(queries.shape[0], -1, queries.shape[-1])

    def agree(self, lists: torch.Tensor) -> bool:
        """Say whether every key head's replicas, on every rank, hold the same lists.

        Args:
            lists: This rank's lists [rows, width] int32.
        """
        stacked = self.gather(lists.contiguous()).view(self.head_count, self.size, *lists.shape)
        return bool((stacked == stacked[:, :1]).all())


class PisaGroupedQueryIndexer:
    """One quadratic layer's PISA selection, an index a key head this rank holds, built on first use."""

    # What the plugin's readiness and the harness ask of an indexer, to tell a PISA selection from LSA's
    is_pisa = True

    def __init__(
        self,
        layer_id: int,
        geometry: index_space.Geometry,
        config: PisaConfig,
        check: PisaCheck,
        backend: Any,
        kv_head_count: int,
        replicas: Replicas | None = None,
    ) -> None:
        """Hold what the indexes are built from.

        Args:
            layer_id: The layer's decoder index, as `RadixAttention.layer_id` carries it.
            geometry: The checkpoint's geometry.
            config: The pyramid and the budget.
            check: The rows, the keys and the steps of the check against the reference search, off at 0 rows.
            backend: The attention backend whose pools the layer reads.
            kv_head_count: Key heads this rank holds.
            replicas: The ranks sharing this rank's key head, or None where a rank holds every query head that reads
                its key heads.
        """
        self.layer = layer_id
        self.geometry = geometry
        self.config = config
        self.backend = backend
        self.kv_head_count = kv_head_count
        self.replicas = replicas
        self.indexes: list[PisaIndex] = []
        self.check_rows = check.rows
        self.check_from = check.from_keys
        self.check_steps = check.steps
        self.checked = {"forwards": 0, "captured": 0}
        self.record = DecodeRecord(pisa.CHECKED_REQUESTS)
        self.logit_scale: float | None = None
        registry.register(self)
        registry.announce_once(Banner.BUILT, layer_id)

    def request_table(self) -> torch.Tensor:
        """Return the request table, each request's positions to KV slots [requests, context] int32."""
        table: torch.Tensor = self.backend.req_to_token
        return table

    def key_buffer(self) -> torch.Tensor:
        """Return this layer's key cache [slots, Hkv, dh] bf16, this rank's key heads.

        Raises:
            NotImplementedError: The KV cache is not bf16, so its keys are not the model's own.
        """
        buffer: torch.Tensor = self.backend.token_to_kv_pool.get_key_buffer(self.layer)
        if buffer.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"PISA reads a bf16 KV cache - this one is {buffer.dtype}, serve kv_cache_dtype auto"
            )
        return buffer.view(-1, self.kv_head_count, self.geometry.head_dim)

    def ready(self) -> list[PisaIndex]:
        """Build the layer's indexes once, one a key head, and return them.

        Returns:
            The indexes, in key head order.
        """
        if self.indexes:
            return self.indexes
        table = self.request_table()
        before = torch.cuda.memory_allocated(table.device)
        self.indexes = [
            PisaIndex(
                self.config,
                self.geometry.head_dim,
                table.shape[0],
                table.shape[1],
                self.request_table,
                self.key_buffer,
                table.device,
                head=head,
                reduce_scores=None if self.replicas is None else self.replicas.sum_scores,
            )
            for head in range(self.kv_head_count)
        ]
        built = torch.cuda.memory_allocated(table.device) - before
        memory.check_built(self.layer, built, self.backend.token_to_kv_pool.size)
        return self.indexes

    def maintain(self, forward_batch: Any) -> None:
        """Hand each request in the batch its own means and close the blocks already in the pool, before every forward.

        Args:
            forward_batch: The batch about to be forwarded, with `lsa_rows` on it.
        """
        rows = forward_batch.lsa_rows
        ends = registry.ends_of(forward_batch, len(rows))
        self.check_captured(forward_batch)
        for index in self.ready():
            index.maintain(forward_batch.rids, rows, ends)
        if self.check_rows:
            self.record.expect([] if forward_batch.forward_mode.is_extend() else rows, forward_batch.rids)

    def check_captured(self, forward_batch: Any) -> None:
        """Check the last captured decode step's first rows against the reference while the run still wants it.

        Args:
            forward_batch: The batch about to be forwarded, whose requests still hold their table rows.
        """
        if not self.check_rows or self.logit_scale is None or self.checked["captured"] >= self.check_steps:
            return
        taken = self.record.take(forward_batch.lsa_rows, forward_batch.rids)
        if taken is None:
            return
        table_rows, step_positions, current, (queries, lists) = taken
        pisa.report_left(self.layer, [p for p, held in zip(step_positions, current, strict=True) if not held])
        due = [
            at
            for at, (position, held) in enumerate(zip(step_positions, current, strict=True))
            if held and position + 1 >= self.check_from
        ]
        if not due:
            return
        picked = torch.tensor(due, device=lists.device)
        self.check(
            queries[picked],
            torch.tensor([step_positions[at] for at in due], device=lists.device),
            lists[picked],
            [table_rows[at] for at in due],
            [step_positions[at] + 1 for at in due],
            self.logit_scale,
            "captured",
        )

    def select(self, q: torch.Tensor, forward_batch: Any, logit_scale: float) -> torch.Tensor:
        """Select for every query of a forward, a list a key head, whose keys are already in the pool.

        Args:
            q: This rank's query heads [T, Hq * dh] or [T, Hq, dh], rotated, each key head's group consecutive.
            forward_batch: The batch, with `lsa_rows` on it.
            logit_scale: The layer's softmax scale.

        Returns:
            KV slots [T, Hkv, width] int32, -1 padded, valid entries first.
        """
        positions = forward_batch.positions
        queries = q.reshape(q.shape[0], self.kv_head_count, -1, self.geometry.head_dim)
        if not forward_batch.forward_mode.is_extend():
            rows = forward_batch.req_pool_indices
            lists = torch.stack(
                [
                    index.decode(queries[:, head], positions, rows, logit_scale)
                    for head, index in enumerate(self.ready())
                ],
                dim=1,
            )
            if self.check_rows:
                self.logit_scale = logit_scale
                self.record.record(rows, positions, queries, lists)
            self.check_if_due(queries, positions, lists, forward_batch, logit_scale)
            return lists
        lengths = list(forward_batch.extend_seq_lens_cpu)
        row_list = forward_batch.lsa_rows
        key_counts = forward_batch.seq_lens_cpu.tolist()[: len(row_list)]
        lists = torch.stack(
            [
                index.select(
                    queries[:, head], positions, row_list, key_counts, lengths[: len(row_list)], logit_scale
                )
                for head, index in enumerate(self.ready())
            ],
            dim=1,
        )
        self.check_if_due(queries, positions, lists, forward_batch, logit_scale)
        return lists

    def check_if_due(
        self,
        queries: torch.Tensor,
        positions: torch.Tensor,
        lists: torch.Tensor,
        forward_batch: Any,
        logit_scale: float,
    ) -> None:
        """Check a few rows of the batch's first requests against the reference while the run still wants it.

        A captured decode step is checked after it ran, from its record. The warm-up a capture runs first is never
        checked, since its batch is no request's.

        Args:
            queries: Every row's query heads by key head [T, Hkv, group_size, dh].
            positions: Each row's position [T].
            lists: The lists just served [T, Hkv, width] int32.
            forward_batch: The batch, with `lsa_rows` on it.
            logit_scale: The layer's softmax scale.
        """
        if not self.check_rows or self.checked["forwards"] >= self.check_steps:
            return
        capturing = positions.is_cuda and torch.cuda.is_current_stream_capturing()
        if capturing or getattr(forward_batch, "lsa_rows", None) is None:
            return
        table_rows = list(forward_batch.lsa_rows)
        key_counts = forward_batch.seq_lens_cpu.tolist()[: len(table_rows)]
        if forward_batch.forward_mode.is_extend():
            lengths = list(forward_batch.extend_seq_lens_cpu)[: len(table_rows)]
        else:
            lengths = [1] * len(table_rows)
        picked = pisa.check_rows_of(lengths, key_counts, self.check_rows, self.check_from)
        if not picked:
            return
        requests = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))[picked].tolist()
        picked_rows = torch.tensor(picked, device=positions.device)
        self.check(
            queries[picked_rows],
            positions[picked_rows],
            lists[picked_rows],
            [table_rows[request] for request in requests],
            [key_counts[request] for request in requests],
            logit_scale,
            "forwards",
        )

    def check(
        self,
        queries: torch.Tensor,
        positions: torch.Tensor,
        lists: torch.Tensor,
        table_rows: list[int],
        key_counts: list[int],
        logit_scale: float,
        counter: str,
    ) -> None:
        """Hold every key head's served lists against the reference and tally the verdicts.

        A rank holds whole key heads, so ranks' lists differ by design and only a key head's replicas are compared,
        and they are scored with the whole group's queries.

        Args:
            queries: The checked rows' query heads by key head [rows, Hkv, group_size, dh].
            positions: Their positions [rows].
            lists: The lists served them [rows, Hkv, width] int32.
            table_rows: Each checked row's request table row.
            key_counts: Keys each checked row's request holds in the pool.
            logit_scale: The layer's softmax scale.
            counter: Which budget the check spends - eager forwards, or captured decode steps.
        """
        self.checked[counter] += 1
        if self.replicas is not None:
            pisa.TALLY.add_ranks(self.replicas.agree(lists[:, 0]))
            queries = self.replicas.group_queries(queries[:, 0])[:, None]
        for head, index in enumerate(self.ready()):
            checked = []
            # One reference pyramid a request, over the rows of it that were picked
            for (table_row, key_count), at in pisa.rows_by_request(table_rows, key_counts):
                picked = torch.tensor(at, device=lists.device)
                checked += index.check(
                    queries[picked, head],
                    positions[picked],
                    lists[picked, head],
                    table_row,
                    key_count,
                    logit_scale,
                )
            pisa.report_check(f"layer {self.layer} head {head}{CAPTURED_LABEL[counter]}", checked)


def head_pages(lists: torch.Tensor) -> torch.Tensor:
    """Turn each key head's list of KV slots into slots of the pool read as one head.

    Args:
        lists: KV slots [T, Hkv, width] int32, -1 padded.

    Returns:
        Page table rows [T * Hkv, width] int32, slot `s` of head `h` at `s * Hkv + h`, 0 where padded.
    """
    kv_head_count = lists.shape[1]
    heads = torch.arange(kv_head_count, device=lists.device, dtype=torch.int32)[None, :, None]
    pages = torch.where(lists >= 0, lists * kv_head_count + heads, 0)
    return pages.reshape(-1, lists.shape[2])


def head_read(attention_backend: Any, q: torch.Tensor, layer: Any, lists: torch.Tensor) -> torch.Tensor:
    """Attend each query head over the KV slots its own key head's list names.

    Args:
        attention_backend: The attention backend, whose kernel and pool this reads.
        q: This rank's query heads [T, Hq * dh] or [T, Hq, dh], each key head's group consecutive.
        layer: The attention layer, carrying the scale and the logit cap.
        lists: KV slots [T, Hkv, width] int32, -1 padded, valid entries first.

    Returns:
        The attention output [T, Hq * dh].
    """
    token_count, kv_head_count, width = lists.shape
    group_size = layer.tp_q_head_num // kv_head_count
    sequences = token_count * kv_head_count
    lengths = read.list_lengths(lists.view(sequences, width))
    cu_seqlens_k = torch.zeros(sequences + 1, dtype=torch.int32, device=lists.device)
    torch.cumsum(lengths, 0, dtype=torch.int32, out=cu_seqlens_k[1:])
    if not backend.flash_reads(lists.device):
        # Every key head read as a pool of its own, as the page table below does
        pool = attention_backend.token_to_kv_pool
        out = backend.triton_read(
            q.contiguous().view(sequences, group_size, layer.head_dim),
            pool.get_key_buffer(layer.layer_id).view(-1, 1, layer.head_dim),
            pool.get_value_buffer(layer.layer_id).view(-1, 1, layer.v_head_dim),
            head_pages(lists),
            lengths,
            layer,
        )
        return out.reshape(token_count, layer.tp_q_head_num * layer.v_head_dim)
    keys, values = backend.paged_cache(attention_backend, layer)
    out = backend.flash_kernel(attention_backend)(
        q=q.contiguous().view(sequences, group_size, layer.head_dim),
        k_cache=keys.view(-1, 1, 1, layer.head_dim),
        v_cache=values.view(-1, 1, 1, layer.v_head_dim),
        page_table=head_pages(lists),
        cache_seqlens=lengths,
        cu_seqlens_q=torch.arange(sequences + 1, dtype=torch.int32, device=lists.device),
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=layer.scaling,
        causal=True,
        softcap=layer.logit_cap,
        num_splits=getattr(attention_backend, "num_splits", 0),
    )
    return out.reshape(token_count, layer.tp_q_head_num * layer.v_head_dim)


def refuse_unservable(attention_backend: Any, geometry: index_space.Geometry) -> HeadShard:
    """Refuse a server PISA cannot select for on Qwen, and return the heads a rank holds.

    Args:
        attention_backend: The attention backend being built.
        geometry: The checkpoint's geometry.

    Returns:
        The key and query heads this rank holds.

    Raises:
        NotImplementedError: A mesh, a paging, a cache or a decoding mode PISA cannot serve.
        RuntimeError: A symbol the backend reads is missing from this sglang, or the metadata glue graph is on.
    """
    backend.resolve_all(backend.REQUIRED, "Qwen")
    from sglang.srt.runtime_context import get_parallel, get_spec

    parallel = get_parallel()
    tp_size = parallel.attn_tp_size
    if context_parallel.decode_cp()[0] > 1 or parallel.attn_cp_size > 1 or parallel.enable_prefill_cp:
        raise NotImplementedError(
            "Qwen under PISA serves tensor parallel only - serve without context parallel"
        )
    refuse_mesh(geometry, tp_size)
    if attention_backend.page_size != 1:
        raise NotImplementedError(
            f"a PISA list is a set of single tokens and this server pages {attention_backend.page_size} at a time - "
            "serve with --page-size 1"
        )
    if attention_backend.kv_index_translator.is_translating:
        raise NotImplementedError(
            "PISA reads the request table's slots as they are - this pool translates them"
        )
    if get_spec().speculative_algorithm is not None:
        raise NotImplementedError("Qwen under PISA serves without speculative decoding")
    plugin.refuse_metadata_glue()
    heads = index_space.head_shard(geometry, tp_size, parallel.attn_tp_rank)
    return heads or HeadShard(0, geometry.kv_head_count, geometry.query_head_count, True)


def refuse_mesh(geometry: index_space.Geometry, tp_size: int) -> None:
    """Refuse a tensor-parallel mesh that splits a key head's group any way but the whole, or across more than a pair.

    Args:
        geometry: The checkpoint's geometry.
        tp_size: The attention tensor-parallel size.

    Raises:
        NotImplementedError: The mesh neither divides the key heads nor holds each on two ranks, or does not split the
            query heads evenly.
    """
    kv_count = geometry.kv_head_count
    if tp_size <= kv_count and kv_count % tp_size == 0 or tp_size == 2 * kv_count:
        if geometry.query_head_count % tp_size == 0:
            return
        raise NotImplementedError(
            f"tensor parallel {tp_size} splits {geometry.query_head_count} query heads unevenly - serve Qwen under "
            f"PISA at a divisor of {kv_count} key heads or at {2 * kv_count}"
        )
    raise NotImplementedError(
        f"Qwen under PISA selects a list a key head and reads it with that head's whole group - tensor parallel "
        f"{tp_size} splits {kv_count} key heads, serve at a divisor of it or at {2 * kv_count}, where each key head "
        "sits on two ranks that sum their block scores"
    )


def replicas_of(geometry: index_space.Geometry, heads: HeadShard) -> Replicas | None:
    """Return the ranks sharing this rank's key head, or None where it holds every query head that reads its key heads.

    Args:
        geometry: The checkpoint's geometry.
        heads: The heads this rank holds - a key head's replicas each hold an equal share of its group.
    """
    group_size = geometry.query_head_count // geometry.kv_head_count
    size = group_size * heads.kv_count // heads.query_count
    if size == 1:
        return None
    tp_size = geometry.query_head_count // heads.query_count
    backend.resolve_all(REQUIRED_REPLICAS, "Qwen under PISA with a key head on two ranks")
    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor, attn_tp_all_reduce

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor from every rank of the attention tensor-parallel group, stacked on a new dim 0."""
        stacked = torch.empty(
            tp_size * tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
        )
        attn_tp_all_gather_into_tensor(stacked, tensor)
        return stacked.view(tp_size, *tensor.shape)

    return Replicas(heads.kv_first, geometry.kv_head_count, size, attn_tp_all_reduce, gather)


def arm(result: None, self: Any, *args: Any, **kwargs: Any) -> None:
    """Give a Qwen server's FlashAttention or triton backend a PISA indexer for every quadratic layer.

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
    heads = refuse_unservable(self, geometry)
    if not ARMED:
        replicas = replicas_of(geometry, heads)
        served = settings.of(runner.model_config.hf_config)
        config, check = settings.pisa_of(runner.model_config.hf_config), served.debug.check_pisa
        ARMED.update(
            {
                layer: PisaGroupedQueryIndexer(layer, geometry, config, check, self, heads.kv_count, replicas)
                for layer in geometry.layers
            }
        )
    setattr(self, backend.INDEXERS, ARMED)
    return None


def select_and_read(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    """Serve a quadratic layer's attention from its PISA lists, or hand any other call to the backend as it was.

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
    indexer = backend.indexer_of(self, layer, forward_batch)
    if indexer is None or k is None:
        return original(*bound.args, **bound.kwargs)
    mode = forward_batch.forward_mode
    if mode.is_extend() and (torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
        raise RuntimeError("a PISA prefill selects eagerly - serve with the prefill CUDA graph disabled")
    if mode.is_target_verify() or mode.is_draft_extend_v2():
        raise NotImplementedError(
            "Qwen under PISA reads a plain prefill or decode, without speculative decoding"
        )
    if arguments["save_kv_cache"]:
        backend.save(self, layer, k, v, forward_batch, Shard())
    return head_read(self, q, layer, indexer.select(q, forward_batch, layer.scaling))
