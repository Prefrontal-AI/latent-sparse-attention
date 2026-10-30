"""GLM-5.3-Flash's indexer under PISA - the paper's pyramid search in place of the checkpoint's own indexer.

It is called where the design's key and query input is, with the key latent and the absorbed queries of this rank's heads,
which the plugin hands over from the end of `forward_absorb_prepare` - see [plugin](../../plugin.py).
PISA scores a block by the LogSumExp over its children of the dot with each head's query, which is not linear in the
queries, so the heads are not summed first.
A prefill shares its rows across the attention group: each rank is handed every head's queries of its share of the rows,
searches them, and the lists are gathered - so a row's keys are read and its keep made once, not once a rank.
A decode step has too few rows to share, so each rank scores a level's blocks with its own heads and the scores are
summed across the group before any rank keeps. Either way every rank holds the same lists.

It serves tensor parallel only, with the prefill CUDA graph off - a prefill's search is eager. A decode step's search reads
device tensors alone and a pyramid as wide as the request table, so the decode graph captures it.
The list is `keep_count * block_size` wide, the model's `index_topk` that the plugin sets, KV slots int32 with the valid
entries first and -1 after, the format LSA's lists are.
The index closes a request's blocks before each forward and as a prefill selects, and forgets them when another request
takes its row.
"""

from typing import Any

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import is_graph_dsa_split_op_surface
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_all_reduce,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel

from lsa.sglang import context_parallel, memory, settings
from lsa.sglang.constants import Banner
from lsa.sglang.index import pisa, registry
from lsa.sglang.index.pisa import DecodeRecord, PisaIndex, RowShare, refuse_rope

# What a check line says of the step it held - nothing for an eager forward
CAPTURED_LABEL: dict[str, str] = {"forwards": "", "captured": " captured decode"}


def request_table() -> torch.Tensor:
    """Return the request table, each request's positions to KV slots."""
    return get_attn_backend().req_to_token_pool.req_to_token


def gather_heads(query: torch.Tensor) -> torch.Tensor:
    """Gather every rank's heads' absorbed queries, so each rank holds all of the layer's heads.

    Args:
        query: This rank's absorbed queries [S, heads_local, key_dim] bf16.

    Returns:
        The group's queries [S, group_size, heads_local, key_dim] bf16, a view of the gathered tensor.
    """
    group_size = context_parallel.attention_tp_size()
    if group_size == 1:
        return query[:, None]
    query = query.contiguous()
    gathered = torch.empty(
        group_size * query.shape[0], *query.shape[1:], dtype=query.dtype, device=query.device
    )
    attn_tp_all_gather_into_tensor(gathered, query)
    return gathered.view(group_size, *query.shape).transpose(0, 1)


def exchange_rows(query: torch.Tensor) -> torch.Tensor:
    """Hand each rank of the attention group every head's queries of its share of the rows.

    Args:
        query: This rank's heads' absorbed queries of every row [S, heads_local, key_dim] bf16, S a multiple of the group.

    Returns:
        This rank's share of the rows with every rank's heads, rank by rank [S / group, group * heads_local, key_dim] bf16.
    """
    group_size = context_parallel.attention_tp_size()
    query = query.contiguous()
    received = torch.empty_like(query)
    get_parallel().attn_tp_group.all_to_all_single(received, query)
    each = query.shape[0] // group_size
    return (
        received.view(group_size, each, *query.shape[1:]).transpose(0, 1).reshape(each, -1, query.shape[-1])
    )


def gather_rows(lists: torch.Tensor) -> torch.Tensor:
    """Gather every rank's share of a prefill's lists, rank by rank.

    Args:
        lists: This rank's share's lists [S / group, width] int32.

    Returns:
        The prefill's lists [S, width] int32.
    """
    group_size = context_parallel.attention_tp_size()
    gathered = torch.empty(
        group_size * lists.shape[0], *lists.shape[1:], dtype=lists.dtype, device=lists.device
    )
    attn_tp_all_gather_into_tensor(gathered, lists.contiguous())
    return gathered


class PisaIndexer(Indexer):
    """One sparse layer's PISA selection, behind the checkpoint's own indexer class so the weights still load."""

    is_pisa = True

    def __init__(self, **indexer_kwargs: Any) -> None:
        """Build the checkpoint's indexer modules, which load the weights, and refuse a server this cannot serve.

        Args:
            **indexer_kwargs: What the model builds an indexer with.

        Raises:
            ValueError: `index_topk` is not the design's list width, the metadata glue graph is on, the latent
                carries rope dims, or context parallel is on.
            RuntimeError: Attention is data parallel.
        """
        super().__init__(**indexer_kwargs)
        model_config = indexer_kwargs["config"]
        self.config = settings.pisa_of(model_config)
        if getattr(model_config, "index_kpool", 1) != 1 or self.index_topk != self.config.width:
            raise ValueError(
                f"index_topk must be PISA's list width {self.config.width} and index_kpool 1 - "
                "the plugin sets both on hf_text_config"
            )
        if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
            raise ValueError(
                "PISA maintains its index on the host before each forward - disable the metadata glue graph"
            )
        if is_dp_attention_enabled():
            raise RuntimeError(
                "PISA gathers every head's query on every rank - serve without --enable-dp-attention"
            )
        if context_parallel.decode_cp()[0] > 1 or context_parallel.prefill_split() is not None:
            raise ValueError(
                "PISA serves tensor parallel only - serve without prefill or decode context parallel"
            )
        refuse_rope(model_config)
        self.layer = indexer_kwargs["layer_id"]
        self.key_dim = int(model_config.kv_lora_rank)
        self.index: PisaIndex | None = None
        check = settings.of(model_config).debug.check_pisa
        self.check_rows = check.rows
        self.check_from = check.from_keys
        self.check_steps = check.steps
        self.checked = {"forwards": 0, "captured": 0}
        self.record = DecodeRecord(pisa.CHECKED_REQUESTS)
        self.logit_scale: float | None = None
        registry.register(self)
        registry.announce_once(Banner.BUILT, self.layer)

    def ready(self) -> PisaIndex:
        """Build the layer's index once the model and the KV pool exist, and return it.

        Returns:
            The layer's index.
        """
        if self.index is not None:
            return self.index
        table = request_table()
        pool = get_token_to_kv_pool()
        before = torch.cuda.memory_allocated(table.device)
        group_size = context_parallel.attention_tp_size()
        share_rows = (
            RowShare(int(get_parallel().attn_tp_rank), group_size, exchange_rows, gather_rows)
            if group_size > 1
            else None
        )
        self.index = PisaIndex(
            self.config,
            self.key_dim,
            table.shape[0],
            table.shape[1],
            request_table,
            self.key_buffer,
            table.device,
            reduce_scores=attn_tp_all_reduce if group_size > 1 else None,
            share_rows=share_rows,
        )
        memory.check_built(self.layer, torch.cuda.memory_allocated(table.device) - before, pool.size)
        return self.index

    def key_buffer(self) -> torch.Tensor:
        """Return this layer's KV pool buffer, whose first `kv_lora_rank` dims of a slot are its key latent.

        Returns:
            The buffer [slots, 1, kv_lora_rank] bf16.

        Raises:
            NotImplementedError: The KV cache is not bf16, so its keys are not the latents the queries are scored against.
        """
        buffer = get_token_to_kv_pool().get_key_buffer(self.layer)
        if buffer.dtype != torch.bfloat16:
            raise NotImplementedError("PISA reads a bf16 KV cache - serve kv_cache_dtype bf16")
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
        logit_scale: float | None = None,
    ) -> torch.Tensor | None:
        """Write this batch's keys and select, or only write them when no selection is asked for.

        A short prompt is prefilled by the dense path, which calls the indexer for its keys alone.

        Args:
            x: The attention's input, which PISA does not read.
            q_lora: The latent query, which PISA does not read.
            positions: Each token's position [S].
            forward_batch: The batch, with `lsa_rows` on it.
            layer_id: This layer.
            return_indices: Whether a selection is wanted.
            key: The key latent [S, kv_lora_rank] bf16.
            query: The absorbed queries of this rank's heads [S, heads_local, kv_lora_rank] bf16.
            logit_scale: GLM's softmax scale, the attention layer's own.

        Returns:
            KV slots [S, width] int32, -1 padded, or None.

        Raises:
            ValueError: The key did not arrive, or a selection was asked for without the queries or the scale.
            RuntimeError: The prefill is captured in a piecewise or breakable CUDA graph, which the selection cannot be.
        """
        if forward_batch.forward_mode.is_idle():
            return None
        if is_graph_dsa_split_op_surface(forward_batch):
            raise RuntimeError("PISA selects eagerly - serve with the prefill CUDA graph disabled")
        if key is None:
            raise ValueError("PISA reads the key latent, from forward_absorb_prepare")
        index = self.ready()
        index.write(forward_batch.out_cache_loc.long(), key)
        if not return_indices:
            return None
        if query is None or logit_scale is None:
            raise ValueError(
                "PISA selects from the absorbed queries and the softmax scale, from forward_absorb_prepare"
            )
        if not forward_batch.forward_mode.is_extend():
            lists = index.decode(query, positions, forward_batch.req_pool_indices, logit_scale)
            if self.check_rows:
                self.logit_scale = logit_scale
                self.record.record(forward_batch.req_pool_indices, positions, gather_heads(query), lists)
            self.check_if_due(index, query, positions, lists, forward_batch, logit_scale)
            return lists
        lengths = list(forward_batch.extend_seq_lens_cpu)
        rows = forward_batch.lsa_rows
        key_counts = forward_batch.seq_lens_cpu.tolist()[: len(rows)]
        lists = index.select(query, positions, rows, key_counts, lengths[: len(rows)], logit_scale)
        self.check_if_due(index, query, positions, lists, forward_batch, logit_scale)
        return lists

    def check_if_due(
        self,
        index: PisaIndex,
        query: torch.Tensor,
        positions: torch.Tensor,
        lists: torch.Tensor,
        forward_batch: Any,
        logit_scale: float,
    ) -> None:
        """Check a few rows of the batch's first requests against the reference while the run still wants it.

        Every rank decides alike from the same batch, so the gathers the check makes meet. A captured decode step is
        checked after it ran, from its record. The warm-up a capture runs first is never checked, since its batch is
        no request's.

        Args:
            index: The layer's index.
            query: This rank's heads' absorbed queries [S, heads_local, key_dim] bf16, gathered for the check.
            positions: Each row's position [S].
            lists: The lists just served [S, width] int32.
            forward_batch: The batch, with `lsa_rows` on it.
            logit_scale: GLM's softmax scale.
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
            index,
            gather_heads(query)[picked_rows],
            positions[picked_rows],
            lists[picked_rows],
            [table_rows[request] for request in requests],
            [key_counts[request] for request in requests],
            logit_scale,
            "forwards",
        )

    def check(
        self,
        index: PisaIndex,
        queries: torch.Tensor,
        positions: torch.Tensor,
        lists: torch.Tensor,
        table_rows: list[int],
        key_counts: list[int],
        logit_scale: float,
        counter: str,
    ) -> None:
        """Hold served lists against the reference, compare them across the ranks, and tally the verdicts.

        Args:
            index: The layer's index.
            queries: The checked rows' heads' queries [rows, group_size, heads_local, key_dim] bf16.
            positions: Their positions [rows].
            lists: The lists served them [rows, width] int32.
            table_rows: Each checked row's request table row.
            key_counts: Keys each checked row's request holds in the pool.
            logit_scale: GLM's softmax scale.
            counter: Which budget the check spends - eager forwards, or captured decode steps.
        """
        self.checked[counter] += 1
        group_size = context_parallel.attention_tp_size()
        if group_size > 1:
            lists = lists.contiguous()
            gathered = torch.empty(
                group_size * lists.shape[0], lists.shape[1], dtype=lists.dtype, device=lists.device
            )
            attn_tp_all_gather_into_tensor(gathered, lists)
            pisa.TALLY.add_ranks(bool((gathered.view(group_size, *lists.shape) == lists).all()))
        checked = []
        # One reference pyramid a request, over the rows of it that were picked
        for (table_row, key_count), at in pisa.rows_by_request(table_rows, key_counts):
            picked = torch.tensor(at, device=lists.device)
            checked += index.check(
                queries[picked], positions[picked], lists[picked], table_row, key_count, logit_scale
            )
        pisa.report_check(f"layer {self.layer}{CAPTURED_LABEL[counter]}", checked)

    def maintain(self, forward_batch: Any) -> None:
        """Hand each request in the batch its own means and close the blocks already in the pool, eagerly, before every forward.

        Args:
            forward_batch: The batch about to be forwarded, with `lsa_rows` on it.
        """
        rows = forward_batch.lsa_rows
        index = self.ready()
        self.check_captured(index, forward_batch)
        index.maintain(forward_batch.rids, rows, registry.ends_of(forward_batch, len(rows)))
        if self.check_rows:
            self.record.expect([] if forward_batch.forward_mode.is_extend() else rows, forward_batch.rids)

    def check_captured(self, index: PisaIndex, forward_batch: Any) -> None:
        """Check the last captured decode step's first rows against the reference while the run still wants it.

        Args:
            index: The layer's index.
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
        positions = torch.tensor([step_positions[at] for at in due], device=lists.device)
        picked = torch.tensor(due, device=lists.device)
        self.check(
            index,
            queries[picked],
            positions,
            lists[picked],
            [table_rows[at] for at in due],
            [step_positions[at] + 1 for at in due],
            self.logit_scale,
            "captured",
        )
