"""GLM-5.3-Flash - MLA, NoPE, and a learned sparse indexer that LSA replaces.

Five things differ from a standard HF decoder.

It is a vision-language checkpoint, so `AutoModelForCausalLM` does not know its config and `model_class`
answers with the image-text-to-text one instead. The vision tower loads and is never read - the text stack
under it is what everything here touches.

Its indexer scores every query against every pool in one fp32 tensor, which over a 64k document is 128 GiB.
`load` binds `index_in_query_chunks` in its place, which is the same selection a block of queries at a time.

The projections are MLA: queries come through a LoRA pair, and keys and values come out of one compressed
latent that `expand_kv` turns back into per-head keys and values. `project` does that and nothing else,
because the model caches the expanded keys and values rather than the latent.

The DSA layers are NoPE. `qk_rope_head_dim` is 0 and the config refuses anything else, so the decoder layer
passes `position_embeddings=None` and the whole head vector is position-free. `rotary_embedding` returns
None to say so, and nothing is measured.

The index space is the latent rather than those heads. Collapsing the expanded keys the way every other
model here does would give 64 x 256 = 16,384 dims for a selection the 512-dim latent makes identically, so
`index_space` takes `regroup_latent` instead and the layer holds the `LatentMetric` that carries it.

Its attention returns a third value, the top-k selection the next layer reuses when the stack shares one.
This model's `indexer_types` are all `full`, so nothing downstream reads it and None is the honest answer -
a stack that does share raises rather than returning a selection LSA never made.
"""

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from lsa.core import LatentMetric, RotaryLayout, regroup_latent
from lsa.evaluation.adapter.layer import Projection, SparseAttention, TeacherView
from lsa.evaluation.adapter.model import ModelAdapter
from lsa.evaluation.constants import Glm
from lsa.sglang.settings.design import IndexerConfig


def key_maps(attention: Any) -> torch.Tensor:
    """Return the key half of the latent expansion, one map per query head.

    `kv_b_proj` writes a key and a value per head from one latent, interleaved head by head, so the key
    maps are the leading `qk_nope_head_dim` rows of each head's block.
    The checkpoint stores this one projection unquantized, which is why the maps can be read straight off
    the weight.

    Args:
        attention: The layer's native MLA attention module.

    Returns:
        Key maps [Hq, qk_nope_head_dim, kv_lora_rank], the weight's own dtype.
    """
    key_dim = attention.qk_nope_head_dim
    per_head = attention.kv_b_proj.weight.view(-1, key_dim + attention.v_head_dim, attention.kv_lora_rank)
    return per_head[:, :key_dim]


@torch.no_grad()
def index_in_query_chunks(
    self: Any,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select each query's keys the way `Glm5NextTextIndexer.forward` does, a block of queries at a time.

    The upstream forward scores every query against every pool at once, in fp32, which over a whole 64k
    document is `[1, 65536, 32, 16384]` - 128 GiB, more than any card here holds. Nothing downstream of that
    tensor couples one query to another: the head weights are per token and the top-k runs along the pool
    axis, so scoring a block at a time returns what scoring all of them returns.
    The pools are the key side and are built once, outside the loop. Which keys a block's rows may see is
    built per block too, since over a 1M cache the whole `[queries, keys]` of it is a terabyte, and the block
    narrows as the pools grow so its scores stay the size they are at 64k.

    Args:
        self: The layer's own `Glm5NextTextIndexer`, since this is bound as its method.
        hidden_states: Layer input [B, S, model_dim].
        q_resid: Query residual [B, S, q_lora_rank], or [B, Q, q_lora_rank] at `positions` when given.
        attention_mask: Padding mask [B, S] bool.
        past_key_values: The transformers cache, or None.
        positions: Absolute positions [Q] int64 of the only queries to select for, or None for all S. The keys
            are still every position of `hidden_states`, so a row is what the full call returns for it.

    Returns:
        Selected key ids [B, S (or Q), index_topk (+ index_kpool - 1 with a tail)] int32, `-1` where nothing valid
        was selected.
    """
    batch_size, query_count = hidden_states.shape[:2]
    queried = hidden_states if positions is None else hidden_states[:, positions]
    asked = queried.shape[1]

    keys = self.k_norm(self.wk(hidden_states)).view(batch_size, query_count, -1, self.head_dim).squeeze(2)
    gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
    packed_states = torch.cat([keys, gate_scores, attention_mask.to(keys.dtype)[..., None]], dim=-1)

    key_count = query_count
    current_length = query_count
    if past_key_values is not None:
        cache_layer = past_key_values.layers[self.layer_idx]
        packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
        key_count = cache_layer.keys.shape[-2]
        current_length = cache_layer.get_seq_length()

    valid_keys = packed_states[..., -1].bool()
    pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)

    # [B, 1, index_head_dim, pools] - the key side of every score, built once rather than once per block
    pooled = pool_keys.transpose(-1, -2).float().unsqueeze(1)
    pool_end = pool_indices[..., -1].clamp(0, key_count - 1)
    select_k = min(self.index_topk // self.index_kpool, pool_indices.shape[-2])
    batch_index = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
    key_positions = torch.arange(valid_keys.shape[-1], device=hidden_states.device)

    # The fp32 scores are [B, block, heads, pools], so a block narrows as the pools grow
    block_rows = max(
        1, min(self.index_query_chunk, Glm.INDEXER_SCORE_CELLS // (self.n_heads * pooled.shape[-1]))
    )
    blocks = []
    for start in range(0, asked, block_rows):
        stop = min(start + block_rows, asked)
        span = stop - start

        q = self.wq_b(q_resid[:, start:stop]).view(batch_size, span, -1, self.head_dim)
        scores = F.relu(torch.matmul(q.float(), pooled) * self.softmax_scale)

        weights = self.weights_proj(queried[:, start:stop].to(self.weights_proj.weight.dtype)).float() * (
            self.n_heads**-0.5
        )
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # [B, block, keys] what `get_visible_tokens` returns for these rows, never built for all of them
        query_positions = (
            current_length - query_count + torch.arange(start, stop, device=hidden_states.device)
            if positions is None
            else positions[start:stop]
        )
        visible_rows = (key_positions[None, None, :] <= query_positions[None, :, None]) & valid_keys[
            :, None, :
        ]
        pool_visible = visible_rows.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, span, -1))
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)

        selected = index_scores.topk(select_k, dim=-1).indices
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_index, selected]

        block = selected_indices.flatten(-2).masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
        )
        if self.index_kpool_always_select_tail:
            block = self.append_visible_tail(block, visible_rows, valid_keys)
        blocks.append(block)

    topk_indices = torch.cat(blocks, dim=1)

    output_width = self.index_topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
    answered = attention_mask if positions is None else attention_mask[:, positions]
    topk_indices = topk_indices[..., :output_width].masked_fill(~answered[..., None], -1)
    return topk_indices.to(torch.int32)


def value_maps(attention: Any) -> torch.Tensor:
    """Return the value half of the latent expansion, one map per query head.

    Args:
        attention: The layer's native MLA attention module.

    Returns:
        Value maps [Hq, v_head_dim, kv_lora_rank], the weight's own dtype.
    """
    key_dim = attention.qk_nope_head_dim
    per_head = attention.kv_b_proj.weight.view(-1, key_dim + attention.v_head_dim, attention.kv_lora_rank)
    return per_head[:, key_dim:]


def distinct_keys(selected_key_ids: torch.Tensor, key_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each row's selection and mark the ids a query may read once each.

    The upstream mask is a set - a key the tail appends a second time is still read once - so a read that
    gathers by id has to drop the repeat.

    Args:
        selected_key_ids: Selected key ids [B, S, K] int, `-1` where nothing was selected.
        key_count: Keys in the cache.

    Returns:
        ids: The same ids sorted along each row [B, S, K] int64, clamped to a legal index.
        readable: Which of them to read [B, S, K] bool.
    """
    ids = selected_key_ids.long().sort(dim=-1).values
    readable = (ids >= 0) & (ids < key_count)
    readable[..., 1:] &= ids[..., 1:] != ids[..., :-1]
    return ids.clamp(0, key_count - 1), readable


@torch.no_grad()
def attend_through_selection(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None, torch.Tensor | None]:
    """Attend the way `Glm5NextTextAttention.forward` does, reading the selected keys alone.

    Upstream turns the selection into a dense `[queries, keys]` int32 mask and runs dense attention under
    it, which is 16 GiB at 64k and a terabyte at 1M. Here each query reads its own selected keys through the
    latent they were expanded from - its head folded through the key map, and the attended latent through
    the value map - which is the same attention, in fp32 rather than in the checkpoint's dtype.
    The cache holds that latent rather than 64 expanded heads, so it is 1 KiB a token rather than 64.

    Args:
        self: The layer's own `Glm5NextTextAttention`, since this is bound as its method.
        hidden_states: Layer input [B, S, model_dim].
        attention_mask: Padding mask [B, S] bool, which the indexer reads.
        past_key_values: The transformers cache, or None for a cache-free forward.
        prev_topk_indices: The selection a shared layer reuses, for a layer without its own indexer.
        **kwargs: Whatever else the decoder layer passes, ignored.

    Returns:
        Attention output [B, S, model_dim], None for the weights, and the selection when the next layer
        reuses it.
    """
    batch_size, query_count = hidden_states.shape[:2]
    q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
    compressed = self.kv_a_proj_with_mqa(hidden_states)
    latent = self.kv_a_layernorm(compressed[..., : self.kv_lora_rank])
    # [B, 1, T, kv_lora_rank] every key's latent - the cache is updated before the indexer, as upstream
    keys = latent[:, None]
    if past_key_values is not None:
        keys, _ = past_key_values.update(keys, keys[..., :0], self.layer_idx)
    key_count = keys.shape[-2]

    if self.indexer is not None:
        selected = self.indexer(
            hidden_states=hidden_states,
            q_resid=q_residual,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    else:
        selected = prev_topk_indices

    key_map, value_map = key_maps(self).float(), value_maps(self).float()
    latents = keys[:, 0]
    batch_index = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
    head_count = key_map.shape[0]
    block_rows = max(1, Glm.ATTENTION_SCORE_CELLS // (head_count * selected.shape[-1]))
    outputs = []
    for start in range(0, query_count, block_rows):
        stop = min(start + block_rows, query_count)
        q = self.q_b_proj(q_residual[:, start:stop]).view(batch_size, stop - start, head_count, -1)
        # [B, block, H, kv_lora_rank] each head's query folded through its key map
        folded = torch.einsum("bshd,hdc->bshc", q.float(), key_map)
        ids, readable = distinct_keys(selected[:, start:stop], key_count)
        # [B, block, K, kv_lora_rank] the latents each query selected
        gathered = latents[batch_index, ids].float()
        scores = torch.einsum("bshc,bskc->bshk", folded, gathered) * self.scaling
        weights = torch.softmax(scores.masked_fill(~readable[:, :, None], float("-inf")), dim=-1).nan_to_num()
        attended = torch.einsum("bshk,bskc->bshc", weights, gathered)
        values = torch.einsum("bshc,hvc->bshv", attended, value_map)
        outputs.append(values.to(hidden_states.dtype).flatten(-2))

    output = self.o_proj(torch.cat(outputs, dim=1))
    return output, None, selected if self.next_skip_topk else None


def bind_selected_attention(model: Any) -> int:
    """Bind the read through the selection onto every DSA attention, and say how many were bound.

    Bound per instance and only by a caller that asks for it - an extraction past 64k - so everything else
    runs the forward transformers shipped. A model bound this way caches latents, so it cannot share a cache
    with one that is not.

    Args:
        model: The loaded model, modified in place.

    Returns:
        How many attentions were rebound.
    """
    bound = 0
    for module in model.modules():
        if type(module).__name__ == Glm.ATTENTION_CLASS:
            module.forward = MethodType(attend_through_selection, module)
            bound += 1
    return bound


def chunk_indexer_queries(model: Any, chunk: int = Glm.INDEXER_QUERY_CHUNK) -> int:
    """Bind the blocked indexer onto every DSA layer that runs one, and say how many were bound.

    Bound per instance rather than onto the class, so a second model loaded in the same process keeps the
    forward transformers shipped.

    Args:
        model: The loaded model, modified in place.
        chunk: Query rows to score per pass.

    Returns:
        How many indexers were rebound.
    """
    bound = 0
    for module in model.modules():
        if type(module).__name__ == Glm.INDEXER_CLASS:
            module.index_query_chunk = chunk
            module.forward = MethodType(index_in_query_chunks, module)
            bound += 1
    return bound


class Glm5NextSparseAttention(SparseAttention):
    """GLM-5.3-Flash's MLA attention, reading a keep-set instead of its own top-k."""

    def __init__(self, attention: Any, config: IndexerConfig, rotary: RotaryLayout) -> None:
        """Wrap one MLA layer and factor the key maps its index space is measured in.

        The factoring is a Cholesky of a `[512, 512]` gram, done once when the layer is built rather than
        once per document.

        Args:
            attention: The layer's own attention module.
            config: The indexer knobs.
            rotary: Where the rotation sits, which for these layers is nowhere.
        """
        super().__init__(attention, config, rotary)
        self.latent_metric = LatentMetric.from_key_maps(key_maps(attention))

    def index_space(self, projection: Projection) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the index vectors in the latent, without expanding a single key.

        Args:
            projection: This step's projections, carrying the latent `project` kept.

        Returns:
            index_queries: Query index vectors [B, Tq, 512] fp32 unit rows.
            index_keys: Key index vectors [B, Tk, 512] fp32 unit rows.

        Raises:
            ValueError: The projection carries no latent, which means `project` was replaced by one that
                does not keep it.
        """
        if projection.latent is None:
            raise ValueError("an MLA projection must carry its latent - this one was built without it")
        return regroup_latent(projection.q, projection.latent, self.latent_metric)

    def teacher_view(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> TeacherView:
        """Return what an extraction reads off this layer, in the latent rather than in 64 expanded heads.

        Each head's logit is its query against its expanded key, and the key is the key map applied to the
        latent, so folding the query through the key map scores it against the latent itself. The teacher
        then reads one shared 512-dim key per token rather than 64 of 256 - at 1M the expanded keys alone
        would be 32 GiB, and in the fp32 the logits are taken in, 64.
        The latent is built a slice of the document at a time and the queries only at the sampled positions.

        Args:
            hidden_states: Layer input [1, T, model_dim], on any device - it is moved a slice at a time.
            positions: Sampled query positions [Q] int64, on the layer's device.
            position_embeddings: None - these layers are NoPE.

        Returns:
            The index vectors, the folded queries [Hq, Q, kv_lora_rank] and the latent keys [1, T,
            kv_lora_rank].
        """
        attention = self.attention
        device = positions.device
        slices = []
        for start in range(0, hidden_states.shape[1], Glm.LATENT_SLICE):
            chunk = hidden_states[:, start : start + Glm.LATENT_SLICE].to(device)
            compressed = attention.kv_a_proj_with_mqa(chunk)
            slices.append(attention.kv_a_layernorm(compressed[..., : attention.kv_lora_rank]))
        # [1, T, kv_lora_rank] the latent every key is expanded from
        latent = torch.cat(slices, dim=1)

        sampled = hidden_states[:, positions.cpu()].to(device)
        q_residual = attention.q_a_layernorm(attention.q_a_proj(sampled))
        # [1, Hq, Q, qk_head_dim]
        q = attention.q_b_proj(q_residual).view(1, len(positions), -1, attention.qk_head_dim).transpose(1, 2)
        index_queries, index_keys = regroup_latent(q, latent, self.latent_metric)
        folded = torch.einsum("hqd,hdc->hqc", q[0].float(), key_maps(attention).float())
        return TeacherView(index_queries[0], index_keys[0], folded, latent.float())

    def project(self, hidden_states: torch.Tensor) -> Projection:
        """Run the MLA projections and expand the latent into per-head keys and values.

        `expand_kv` is the model's own, so the keys and values here are exactly the ones its attention
        would have read.

        Args:
            hidden_states: Layer input [B, T, model_dim].

        Returns:
            The layer's query, key and value heads, and the latent they were expanded from. Nothing is
            rotated - these layers are NoPE.
        """
        attention = self.attention
        batch_size, query_count = hidden_states.shape[:2]

        # [B, Hq, T, qk_head_dim] queries, through the LoRA pair the checkpoint stores them as
        q_residual = attention.q_a_layernorm(attention.q_a_proj(hidden_states))
        q = (
            attention.q_b_proj(q_residual)
            .view(batch_size, query_count, -1, attention.qk_head_dim)
            .transpose(1, 2)
        )

        # [B, T, kv_lora_rank + qk_rope_head_dim] - one latent carrying both keys and values
        compressed = attention.kv_a_proj_with_mqa(hidden_states)
        latent, rotated_key = compressed.split([attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1)
        # [B, 1, T, ...] - one shared KV head, which is what MQA stores
        latent = attention.kv_a_layernorm(latent).view(batch_size, 1, query_count, attention.kv_lora_rank)
        rotated_key = rotated_key.view(batch_size, 1, query_count, attention.qk_rope_head_dim)

        k, v = attention.expand_kv(latent, rotated_key)
        # [B, T, kv_lora_rank] - the index space, kept because every key here was built from it
        return Projection(q=q, k=k, v=v, latent=latent[:, 0])


class Glm5NextAdapter(ModelAdapter):
    """GLM-5.3-Flash - a multimodal MLA checkpoint whose text layers do not rotate."""

    attention_class = Glm5NextSparseAttention
    layer_types = Glm.LAYER_TYPES

    def model_class(self) -> Any:
        """Return the auto class this checkpoint registers under.

        `glm5_next` is a vision-language checkpoint - `Glm5NextForConditionalGeneration` over a
        `Glm5NextTextModel` and a vision tower - so its generation head sits in the image-text-to-text
        mapping. `AutoModelForCausalLM` carries no `glm5_next` entry at all and refuses the config rather
        than loading a text-only view of it.

        Returns:
            The auto class to call `from_pretrained` on.
        """
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText

    def load(
        self,
        model_config: Any,
        device: str,
        attention: str = "eager",
        span: tuple[int, int] | None = None,
    ) -> tuple[Any, Any]:
        """Load the checkpoint, or one span of its stack, and block the DSA indexer's scoring.

        A teacher extraction runs this model's own attention over a whole 64k document, where the indexer's
        score tensor is 128 GiB in one piece. `chunk_indexer_queries` is what makes that document fit, and
        it is bound at load because every caller that runs the native attention needs it.

        Args:
            model_config: Which checkpoint, and in what dtype.
            device: A transformers device map.
            attention: The native attention implementation.
            span: Decoder layers `[start, stop)` to load, or None for the whole model.

        Returns:
            model: The loaded model in eval mode, its indexers scoring a block of queries at a time.
            tokenizer: Its tokenizer.
        """
        model, tokenizer = super().load(model_config, device, attention, span)
        chunk_indexer_queries(model)
        return model, tokenizer

    def prepare_long_prefill(self, model: Any) -> None:
        """Read every DSA layer through its selection, so the forward never builds the dense mask.

        Args:
            model: The loaded model, modified in place.
        """
        bind_selected_attention(model)

    def stack_input(self, text_model: Any, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids into the residual streams the first decoder layer reads.

        Args:
            text_model: The decoder stack.
            ids: Token ids [1, S] int64.

        Returns:
            The embedding copied into every hyper-connection stream [1, S, hc_mult, model_dim].
        """
        embedded = text_model.embed_tokens(ids)
        return embedded.unsqueeze(2).expand(-1, -1, text_model.config.hc_mult, -1).contiguous()

    def forward_span(
        self,
        text_model: Any,
        span: tuple[int, int],
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: Any,
    ) -> torch.Tensor:
        """Run decoder layers `[start, stop)` over one chunk, as `Glm5NextTextModel.forward` runs them.

        The stack's own forward hands every layer the same all-true padding mask when none is passed, no
        position tables since the DSA layers are NoPE, and the selection the layer below made.
        A span starts with no selection, which is right because `full_attention_layers` refuses a stack
        whose layers reuse one.

        Args:
            text_model: The decoder stack.
            span: Decoder layers `[start, stop)`.
            hidden: The residual streams layer `start` reads [1, S, hc_mult, model_dim].
            positions: Absolute positions of the chunk [1, S] int64.
            cache: The transformers cache the chunks share, or None for a document in one pass.

        Returns:
            The residual streams layer `stop` reads [1, S, hc_mult, model_dim].
        """
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        selected = None
        for index in range(*span):
            hidden, selected = text_model.layers[index](
                hidden,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=None,
                past_key_values=cache,
                prev_topk_indices=selected,
            )
        return hidden

    def native_selector(self, attention: Any) -> Any | None:
        """Return the layer's own DSA indexer, which picks the keys its attention reads.

        Args:
            attention: The layer's `Glm5NextTextAttention`.

        Returns:
            The indexer, or None for a layer that reuses the selection of the one below.
        """
        return attention.indexer

    def rotary_embedding(self, model: Any) -> None:
        """Declare that these layers do not rotate.

        Args:
            model: The loaded model, unused - the config is what says NoPE.

        Returns:
            None, which `attach` reads as a `none` layout with nothing to measure.
        """
        return None

    def full_attention_layers(self, text_config: Any, model_config: Any) -> list[int]:
        """Return the DSA layers, refusing a stack whose layers share one selection.

        A shared layer reads the top-k the layer below produced. A swapped layer makes no top-k, so a stack
        that shares one cannot be half replaced - and a stack run a span at a time could not hand one across.

        Args:
            text_config: The decoder stack's config.
            model_config: The model model_config, which may name the layer types to swap.

        Returns:
            Layer indices, in order.

        Raises:
            ValueError: The stack has layers that reuse another layer's selection.
        """
        layers = super().full_attention_layers(text_config, model_config)
        indexer_types = getattr(text_config, "indexer_types", None)
        if indexer_types is not None:
            shared = [index for index, kind in enumerate(indexer_types) if kind != Glm.OWN_INDEXER]
            if shared:
                raise ValueError(
                    f"{model_config.name} has layers that reuse another layer's top-k selection - {shared} - "
                    "and a swapped layer produces none for them to reuse"
                )
        return layers
