"""Native Sparse Attention's three reads in plain PyTorch, the semantics its Triton kernels match.

NSA (Yuan et al., Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention, arXiv 2502.11089)
reads, for each query, three sets of keys and mixes them by gates:

- Compression: the keys and values are cut into blocks of compress_block_size keys every compress_stride keys
  (overlapping when compress_stride < compress_block_size), and a learned network maps each block to one compressed key
  and one compressed value.
  Compressed token i summarises keys [i * compress_stride, i * compress_stride + compress_block_size) and is visible to a
  query at position t once that block is complete, i * compress_stride + compress_block_size - 1 <= t.
  The compressed read is softmax attention over the visible compressed tokens.
  These functions take the compressed keys and values as given, the network is the layer's.
- Selection: the keys are cut into blocks of select_block_size keys.
  A block's importance for a query is the compressed read's attention probabilities on the compressed tokens overlapping
  it (equation 9), summed over the query heads that share a key-value head (equation 10), so a key-value head selects
  one set of blocks.
  The block holding the first key and the two blocks nearest the query - the one holding it and the one before - are
  always selected, the rest are the highest-scoring earlier blocks, select_block_count in all.
  The selected read is softmax attention over every key of the selected blocks that is not after the query.
- The window: attention over the window_size most recent keys, which is the window arm and not part of these functions.

Equation 9, for a stride d dividing both the compression block l and the selection block l': cut the keys into
sub-blocks of d keys.
Sub-block s lies in compressed tokens s - n for n in [0, l / d), and selection block j holds sub-blocks
(l' / d) j + m for m in [0, l' / d), so

    p_slc[j] = sum_{m < l'/d} sum_{n < l/d} p_cmp[(l'/d) j + m - n],

each compressed token weighted by how many of its sub-blocks fall in block j, with out-of-range and invisible tokens
contributing 0.
The paper prints the inner index as (l'/d) j - m - n, which counts tokens overlapping block j - 1 for the first m.
The + m form is the overlap the text describes.

Conventions every function follows:

- q is [batch_size, query_count, query_head_count, head_dim] and keys and values are
  [batch_size, key_count, kv_head_count, head_dim], the compressed ones
  [batch_size, compressed_key_count, kv_head_count, head_dim].
  The query heads form kv_head_count groups of group_size = query_head_count // kv_head_count consecutive heads, and
  group i reads key-value head i.
- Queries are every position of the sequence, so query row r sits at position r and query_count equals key_count.
- scale multiplies q . k before the softmax.
- lse is fp32, natural log, [batch_size, query_count, query_head_count].
  A query with no visible compressed token reads 0 from the compressed branch, has lse 0 and passes no gradient, and
  its blocks score 0 for selection.
- Block indices are int32 [batch_size, query_count, kv_head_count, select_block_count], each row ascending with -1
  padding when fewer than select_block_count blocks exist.
  A tie in score goes to the earlier block.
"""

import torch


def compressed_count(key_count: int, compress_block_size: int, compress_stride: int) -> int:
    """Count the compressed tokens of a sequence, the complete blocks of compress_block_size keys every compress_stride.

    Args:
        key_count: Keys in the sequence.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.

    Returns:
        The compressed tokens, 0 when the sequence is shorter than one block.
    """
    return max(0, (key_count - compress_block_size) // compress_stride + 1)


def group_heads(x: torch.Tensor, kv_head_count: int) -> torch.Tensor:
    """Split the query heads of a tensor by the key-value head they read.

    Args:
        x: A tensor [batch_size, query_count, query_head_count, ...].
        kv_head_count: Key-value heads.

    Returns:
        The tensor [batch_size, query_count, kv_head_count, group_size, ...].
    """
    return x.unflatten(2, (kv_head_count, x.shape[2] // kv_head_count))


def visible_tokens(
    query_count: int,
    compressed_key_count: int,
    compress_block_size: int,
    compress_stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Say which compressed tokens each query sees, those whose block ends at or before it.

    Args:
        query_count: Queries, one at every position.
        compressed_key_count: Compressed tokens.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        device: Where to build the mask.

    Returns:
        Whether query sees token [query_count, compressed_key_count] bool.
    """
    positions = torch.arange(query_count, device=device)
    ends = torch.arange(compressed_key_count, device=device) * compress_stride + compress_block_size - 1
    return ends[None, :] <= positions[:, None]


def compressed_attention(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    compressed_values: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the compressed tokens each query sees.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        compressed_values: Compressed values [batch_size, compressed_key_count, kv_head_count, head_dim].
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    query_count = q.shape[1]
    compressed_key_count, kv_head_count = compressed_keys.shape[1], compressed_keys.shape[2]
    grouped = group_heads(q.float(), kv_head_count)
    scores = torch.einsum("bthgd,bchd->bthgc", grouped, compressed_keys.float()) * scale
    visible = visible_tokens(
        query_count, compressed_key_count, compress_block_size, compress_stride, q.device
    )
    scores = scores.masked_fill(~visible[None, :, None, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    lse = lse.masked_fill(~visible.any(-1)[None, :, None, None], 0.0)
    probabilities = torch.exp(scores - lse.unsqueeze(-1))
    context = torch.einsum("bthgc,bchd->bthgd", probabilities, compressed_values.float())
    return context.flatten(2, 3).to(q.dtype), lse.flatten(2, 3)


def block_scores(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    scale: float,
) -> torch.Tensor:
    """Score every selection block for every key-value head, equations 9 and 10.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        lse: The compressed read's log-sum-exp [batch_size, query_count, query_head_count] fp32.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        select_block_size: Keys a selection block holds.
        scale: The factor logits are scaled by.

    Returns:
        The scores [batch_size, query_count, kv_head_count, block_count] fp32.

    Raises:
        ValueError: compress_stride does not divide compress_block_size and select_block_size.
    """
    if compress_block_size % compress_stride or select_block_size % compress_stride:
        raise ValueError(
            f"compress_stride {compress_stride} must divide compress_block_size {compress_block_size} "
            f"and select_block_size {select_block_size}"
        )
    batch_size, query_count = q.shape[:2]
    compressed_key_count, kv_head_count = compressed_keys.shape[1], compressed_keys.shape[2]
    grouped = group_heads(q.float(), kv_head_count)
    scores = torch.einsum("bthgd,bchd->bthgc", grouped, compressed_keys.float()) * scale
    visible = visible_tokens(
        query_count, compressed_key_count, compress_block_size, compress_stride, q.device
    )
    probabilities = torch.exp(scores - group_heads(lse, kv_head_count).unsqueeze(-1))
    probabilities = probabilities.masked_fill(~visible[None, :, None, None, :], 0.0).sum(3)
    block_count = -(-query_count // select_block_size)
    tokens_per_compression_block = compress_block_size // compress_stride
    sub_blocks_per_block = select_block_size // compress_stride
    # Sub-block s lies in compressed tokens s - n for n below the tokens a block spans, so its share sums those tokens
    sub_block_count = block_count * sub_blocks_per_block
    sub_blocks = torch.zeros(batch_size, query_count, kv_head_count, sub_block_count, device=q.device)
    for n in range(tokens_per_compression_block):
        end = min(sub_block_count, compressed_key_count + n)
        if end > n:
            sub_blocks[..., n:end] += probabilities[..., : end - n]
    return sub_blocks.unflatten(-1, (block_count, sub_blocks_per_block)).sum(-1)


def select_blocks(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    lse: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    select_block_size: int,
    select_block_count: int,
    scale: float,
) -> torch.Tensor:
    """Pick the blocks each key-value head reads, the first and the two nearest forced and the best scoring besides.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        compressed_keys: Compressed keys [batch_size, compressed_key_count, kv_head_count, head_dim].
        lse: The compressed read's log-sum-exp [batch_size, query_count, query_head_count] fp32.
        compress_block_size: Keys a compression block summarises.
        compress_stride: Keys between the starts of consecutive blocks.
        select_block_size: Keys a selection block holds.
        select_block_count: Blocks a key-value head reads, the forced ones included.
        scale: The factor logits are scaled by.

    Returns:
        The blocks [batch_size, query_count, kv_head_count, select_block_count] int32, ascending, -1 padded.
    """
    query_count = q.shape[1]
    scores = block_scores(
        q, compressed_keys, lse, compress_block_size, compress_stride, select_block_size, scale
    )
    block_count = scores.shape[-1]
    current = torch.arange(query_count, device=q.device) // select_block_size
    blocks = torch.arange(block_count, device=q.device)
    candidate = blocks[None, :] <= current[:, None]
    forced = (blocks[None, :] == 0) | (blocks[None, :] == current[:, None])
    forced = forced | (blocks[None, :] == current[:, None] - 1)
    scores = scores.masked_fill(forced[None, :, None, :], float("inf"))
    scores = scores.masked_fill(~candidate[None, :, None, :], float("-inf"))
    # Stable descending sort, so a tie keeps the earlier block first
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    best = order[..., :select_block_count]
    picked = scores.gather(-1, best) > float("-inf")
    best = torch.where(picked, best, torch.full_like(best, block_count))
    best = best.sort(-1).values
    best = torch.where(best == block_count, torch.full_like(best, -1), best)
    if best.shape[-1] < select_block_count:
        padding = best.new_full((*best.shape[:-1], select_block_count - best.shape[-1]), -1)
        best = torch.cat([best, padding], dim=-1)
    return best.to(torch.int32)


def selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    select_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over the keys of each key-value head's selected blocks that are not after the query.

    Args:
        q: Queries [batch_size, query_count, query_head_count, head_dim].
        k: Keys [batch_size, key_count, kv_head_count, head_dim].
        v: Values [batch_size, key_count, kv_head_count, head_dim].
        indices: The selected blocks [batch_size, query_count, kv_head_count, select_block_count] int32, -1 unused.
        select_block_size: Keys a selection block holds.
        scale: The factor logits are scaled by.

    Returns:
        The context [batch_size, query_count, query_head_count, head_dim] in q's dtype, and its log-sum-exp
        [batch_size, query_count, query_head_count] fp32.
    """
    query_count = q.shape[1]
    key_count, kv_head_count = k.shape[1], k.shape[2]
    grouped = group_heads(q.float(), kv_head_count)
    scores = torch.einsum("bthgd,bshd->bthgs", grouped, k.float()) * scale
    positions = torch.arange(query_count, device=q.device)
    block_of_key = torch.arange(key_count, device=q.device) // select_block_size
    # Whether key s lies in one of the row's selected blocks [batch_size, query_count, kv_head_count, key_count]
    chosen = (indices.long().unsqueeze(-1) == block_of_key).any(-2)
    causal = torch.arange(key_count, device=q.device)[None, :] <= positions[:, None]
    allowed = chosen & causal[None, :, None, :]
    scores = scores.masked_fill(~allowed.unsqueeze(3), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.exp(scores - lse.unsqueeze(-1))
    context = torch.einsum("bthgs,bshd->bthgd", probabilities, v.float())
    return context.flatten(2, 3).to(q.dtype), lse.flatten(2, 3)
