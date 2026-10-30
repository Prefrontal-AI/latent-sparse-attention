"""The teacher's attention as a target - what the model paid to the keys a selection could have taken."""

import functools

import torch

from lsa import constants as const
from lsa.baselines import qsa
from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.constants import Target, Teacher


def readable_keys(config: IndexerConfig, positions: torch.Tensor, length: int) -> torch.Tensor:
    """Return the keys each query may read - past the sinks and at least a window behind it.

    The cluster block is the window, so a key outside the window already sits in a block that closed before
    the query's own, and the block adds nothing to the rule.

    Args:
        config: The design, whose `sink_size` and `window_size` are the rule.
        positions: Absolute query positions [Q] int64.
        length: Tokens in the document.

    Returns:
        Readable-key mask [Q, T] bool.
    """
    keys = torch.arange(length, device=positions.device)
    return (keys[None] >= config.sink_size) & (positions[:, None] - keys[None] >= config.window_size)


def causal_keys(positions: torch.Tensor, length: int) -> torch.Tensor:
    """Return the keys each query attends to - every key up to its own position.

    Args:
        positions: Absolute query positions [Q] int64.
        length: Tokens in the document.

    Returns:
        Causal mask [Q, T] bool.
    """
    keys = torch.arange(length, device=positions.device)
    return keys[None] <= positions[:, None]


def closed_block_keys(positions: torch.Tensor, length: int, block_size: int) -> torch.Tensor:
    """Return the keys a QSA read may select for each query - those of the blocks its position has closed.

    Args:
        positions: Absolute query positions [Q] int64.
        length: Tokens in the document.
        block_size: Keys a block holds.

    Returns:
        Selectable-key mask [Q, T] bool.
    """
    keys = torch.arange(length, device=positions.device)
    return keys[None] < qsa.closed_blocks(positions, block_size)[:, None] * block_size


@torch.no_grad()
def head_logits(q: torch.Tensor, keys: torch.Tensor, scale: float) -> torch.Tensor:
    """Return the model's own logits of some query rows against every key, one set of rows per head.

    The logits are taken per KV group so the keys are never replicated.

    Args:
        q: Query vectors of some rows [Hq, R, dh].
        keys: Key vectors, one set per KV head [Hkv, T, dh] fp32.
        scale: The model's own attention scale.

    Returns:
        Scaled logits [Hq, R, T] fp32, before any mask.
    """
    kv_head_count, key_count, head_dim = keys.shape
    groups = q.shape[0] // kv_head_count
    scores = torch.einsum(
        "hgrd,hsd->hgrs", q.float().view(kv_head_count, groups, -1, head_dim), keys
    ).reshape(q.shape[0], -1, key_count)
    return scores * scale


def chunk_logits(
    queries: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    sink_size: int,
    window_size: int,
    start: int,
    chunk: int,
    row_count: int,
) -> torch.Tensor:
    """Return the logits of every row's heads against one chunk of keys, the keys a row may not read at -inf.

    Args:
        queries: Each row's heads' queries, flattened row by row [R * H, key_dim] fp32.
        keys: The key latent at every position [T, key_dim], in any float dtype.
        scale: The model's own attention scale.
        positions: The rows' absolute positions [R] int64.
        sink_size: Keys at the start of the document no row may read.
        window_size: Keys this close to a row, its own position included, which it may not read.
        start: The chunk's first key.
        chunk: Keys a chunk holds, the last holding what is left.
        row_count: Rows, `R`.

    Returns:
        Logits [R, H, keys in the chunk] fp32.
    """
    stop = min(start + chunk, keys.shape[0])
    ids = torch.arange(start, stop, device=queries.device)
    readable = (ids[None] >= sink_size) & (positions[:, None] - ids[None] >= window_size)
    scores = (queries @ keys[start:stop].float().t() * scale).view(row_count, -1, stop - start)
    return scores.masked_fill(~readable[:, None], const.Math.NINF)


@functools.cache
def latent_attention_scale(model: str) -> float:
    """Return the softmax scale of an MLA model's attention, read off the checkpoint's config.

    The layer scales its logits by `qk_head_dim ** -0.5`, the head's nope and rope dims together, and a layer that
    rotates has a rope part in the logit that the key latent and the folded query do not carry.

    Args:
        model: The model card, or a checkpoint path.

    Returns:
        The scale.

    Raises:
        ValueError: The model's heads carry rope dims, so its logit is not the latent's dot product alone.
    """
    from lsa.evaluation import adapter
    from lsa.evaluation.config import ModelConfig

    config = ModelConfig(name=model)
    text = adapter.adapter_for(config).text_config(config)
    if text.qk_rope_head_dim:
        raise ValueError(
            f"{model} has {text.qk_rope_head_dim} rope dims in each head, and the dense_full target reads logits "
            "that are the key latent and the folded query alone - it needs a model that does not rotate"
        )
    return float((text.qk_nope_head_dim + text.qk_rope_head_dim) ** -0.5)


@torch.no_grad()
def full_attention_target(
    head_queries: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    sink_size: int,
    window_size: int,
    cells: int = Target.CHUNK_CELLS,
) -> torch.Tensor:
    """Return the teacher's attention over every readable key - what `teacher_attention` ranks, untruncated.

    Every head softmaxes its own logits over the keys the row may read, and the target is their mean:

        pi_h(j) = softmax_j(l_hj)   l_hj = scale * q_h . key_j   over the readable j
        p(j)    = (1 / H) sum_h pi_h(j)

    `teacher_attention` sums the heads and the cache renormalises, so this is that ranking divided by `H` with
    nothing cut off.
    The logits are `head_queries . key` alone because GLM's layers do not rotate, which is what lets the key latent stand
    for every head's key - a model with a rotary part in its logit has no such form.

    The keys go a chunk at a time, twice: the first pass takes each head's log-sum-exp over the readable keys, the
    second writes the softmaxed mass, so the `[H, R, T]` logits never exist and one chunk's are `cells` fp32.
    A row that may read nothing is all zero.

    Args:
        head_queries: Each head's query folded into the key's space, at the rows [R, H, key_dim] bf16.
        key: The key latent at every position [T, key_dim] bf16.
        scale: The model's own attention scale.
        positions: The rows' absolute positions [R] int64.
        sink_size: Keys at the start of the document no row may read.
        window_size: Keys this close to a row, its own position included, which it may not read.
        cells: fp32 cells of one chunk's logits.

    Returns:
        The target [R, T] fp32, zero off each row's readable set.
    """
    row_count, head_count, key_dim = head_queries.shape
    key_count = key.shape[0]
    queries = head_queries.float().reshape(row_count * head_count, key_dim)
    keys = key.to(queries.device)
    chunk = max(1, cells // (row_count * head_count))

    peak = torch.full((row_count, head_count), const.Math.NINF, device=queries.device)
    total = torch.zeros_like(peak)
    for start in range(0, key_count, chunk):
        scores = chunk_logits(
            queries, keys, scale, positions, sink_size, window_size, start, chunk, row_count
        )
        # A head with nothing readable yet keeps a peak of -inf, which the exponent must not see
        raised = torch.maximum(peak, scores.amax(-1))
        reference = raised.masked_fill(raised == const.Math.NINF, 0.0)
        total = total * (peak - reference).exp() + (scores - reference[..., None]).exp().sum(-1)
        peak = raised
    # A row with nothing readable has no normaliser, and +inf zeroes every key of it
    normaliser = torch.where(total > 0, peak + total.log(), torch.full_like(total, float("inf")))

    target = torch.empty(row_count, key_count, device=queries.device)
    for start in range(0, key_count, chunk):
        mass = (
            (
                chunk_logits(queries, keys, scale, positions, sink_size, window_size, start, chunk, row_count)
                - normaliser[..., None]
            )
            .exp()
            .sum(1)
        )
        target[:, start : start + chunk] = mass / head_count
    return target


@torch.no_grad()
def teacher_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    visibility: IndexerConfig,
    top: int,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read off what the model's own attention paid to the keys a selection could have taken.

    Every head softmaxes over its own readable set and the masses are summed over heads, so the target is
    what one shared selection has to serve rather than any one head's ranking.

    For a QSA indexer, `block_size` set, every head softmaxes over every key behind the query, as the model attends,
    and the target is that attention on the keys of the blocks the query has closed - QSA's own target, which takes
    the full attention's distribution rather than one renormalised over what the read may select.

    The logits are taken a block of queries at a time, and the block narrows as the document grows so that
    one block's logits are the same size at 1M as at 64k. What a block may read is built with it, since the
    whole `[queries, keys]` mask of a 1M document is ten gigabytes.

    Args:
        q: Query vectors at the sampled positions [Hq, Q, dh].
        k: Key vectors, one set per KV head [Hkv, T, dh], whose dot with `q` is the model's logit.
        scale: The model's own attention scale.
        positions: Absolute query positions [Q] int64.
        visibility: The rule that says which keys a query may read.
        top: How many of the strongest keys to keep per query.
        block_size: The QSA block, or None for the LSA rule `visibility` states.

    Returns:
        top_key_ids: The strongest readable keys [Q, top] int32.
        top_key_mass: Their head-summed mass [Q, top] fp16.
        peak_key_id: Each head's single strongest readable key [Q, Hkv * G] int32.
    """
    key_count = k.shape[1]
    keys = k.float()
    block_rows = max(1, Teacher.SCORE_CELLS // (q.shape[0] * key_count))
    ids, mass, peaks = [], [], []
    for start in range(0, q.shape[1], block_rows):
        chunk = slice(start, start + block_rows)
        readable = readable_keys(visibility, positions[chunk], key_count)
        selectable = readable
        if block_size is not None:
            readable = causal_keys(positions[chunk], key_count)
            selectable = closed_block_keys(positions[chunk], key_count, block_size)
        scores = head_logits(q[:, chunk], keys, scale)
        attention = torch.softmax(scores.masked_fill(~readable[None], const.Math.NINF), dim=-1)
        del scores
        if block_size is not None:
            attention = attention.masked_fill(~selectable[None], 0.0)
        strongest = attention.sum(0).topk(top, dim=-1)
        ids.append(strongest.indices.int())
        mass.append(strongest.values.half())
        peaks.append(attention.argmax(-1).t().int())
        del attention
    return torch.cat(ids), torch.cat(mass), torch.cat(peaks)


@torch.no_grad()
def native_attention(q: torch.Tensor, k: torch.Tensor, scale: float, picked: torch.Tensor) -> torch.Tensor:
    """Return the attention the served model spends on each key its own sparse selector picked.

    Each head softmaxes over the picks alone, which is what a sparse layer computes, and the masses are
    summed over heads as the dense target's are. The picks' keys are gathered a block of queries at a time,
    the block sized so the gathered keys stay within the dense target's own budget.

    Args:
        q: Query vectors at the sampled positions [Hq, Q, dh].
        k: Key vectors, one set per KV head [Hkv, T, dh], whose dot with `q` is the model's logit.
        scale: The model's own attention scale.
        picked: The selector's picks at those positions [Q, W] int32, -1 where empty.

    Returns:
        Head-summed attention on each pick [Q, W] fp16, zero on an empty slot.
    """
    kv_head_count, _, head_dim = k.shape
    groups = q.shape[0] // kv_head_count
    keys = k.float()
    picked = picked.to(k.device).long()
    block_rows = max(1, Teacher.SCORE_CELLS // (kv_head_count * picked.shape[1] * head_dim))
    mass = []
    for start in range(0, q.shape[1], block_rows):
        chunk = slice(start, start + block_rows)
        empty = picked[chunk] < 0
        # [Hkv, R, W, dh] fp32 the keys each query's selector picked
        gathered = keys[:, picked[chunk].clamp_min(0)]
        scores = torch.einsum(
            "hgrd,hrwd->hgrw", q[:, chunk].float().view(kv_head_count, groups, -1, head_dim), gathered
        ).reshape(q.shape[0], -1, picked.shape[1])
        del gathered
        attention = torch.softmax((scores * scale).masked_fill(empty[None], const.Math.NINF), dim=-1)
        mass.append(attention.sum(0).masked_fill(empty, 0.0).half())
        del scores, attention
    return torch.cat(mass)
