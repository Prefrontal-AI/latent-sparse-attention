"""QSA indexer scoring and selection through the triton kernel plane (M4).

The indexer's score chain beyond the three weight ops (pool -> index_qk_proj -> gemma_norm -> partial
rope, validated separately in scripts/qsa_indexer_parity.py) is a raw dot of each query row's heads with
each pooled key block, averaged over heads, then a top-k over blocks. Mapped onto the ported kernels:

- `block_scores` with one child per candidate computes exactly the raw dot per head (the LogSumExp of a
  single logit is that logit), summed over heads - a constant 1/H from the mean, which top-k ignores;
- `top_keys` gives the tie-free exact top-k, unpacked to block ids.

Zero-query padding heads (head_count 4 < MIN_HEADS 16) are dropped by the kernel's head_live mask.
"""

import logging

import torch

from lsa.core.kernels.triton.block_scores import block_scores
from lsa.core.kernels.triton.keys import unpack_ids
from lsa.core.kernels.triton.top_k import top_keys

_LOG = logging.getLogger(__name__)


def qsa_block_scores(queries: torch.Tensor, key_blocks: torch.Tensor) -> torch.Tensor:
    """Score every pooled key block against every query row's heads, via the block_scores kernel.

    Args:
        queries: Query rows' heads [rows, head_count, head_dim] bf16, normed and roped.
        key_blocks: Pooled key blocks [block_count, head_dim] fp32 or bf16, normed and roped.

    Returns:
        Scores [rows, block_count] fp32 - the mean over heads of the raw dots (the kernel's head-sum
        divided by head_count).
    """
    rows = queries.shape[0]
    block_count = key_blocks.shape[0]
    if rows == 0 or block_count == 0:
        return torch.zeros(rows, block_count, dtype=torch.float32, device=queries.device)

    # All rows read the same key-block summary row; candidates are the block indices themselves
    children = key_blocks.float().unsqueeze(0)
    candidates = torch.arange(block_count, device=queries.device, dtype=torch.int64).repeat(rows, 1)
    requests = torch.arange(rows, device=queries.device, dtype=torch.int64)
    row_index = torch.zeros(rows, device=queries.device, dtype=torch.int64)
    positions = torch.zeros(rows, device=queries.device, dtype=torch.int64)

    scores = block_scores(
        queries,
        candidates,
        requests,
        positions,
        children,
        children_per_block=1,
        logit_scale=1.0,
        row_index=row_index,
    )
    _LOG.debug("qsa block scores: rows=%d blocks=%d", rows, block_count)
    return scores / queries.shape[1]


def qsa_top_key_blocks(scores: torch.Tensor, top_k: int, row_chunk: int = 512) -> torch.Tensor:
    """Pick each row's best `top_k` key blocks, via the exact top-k kernel.

    Args:
        scores: [rows, block_count] fp32 from `qsa_block_scores`.
        top_k: Blocks each row keeps.
        row_chunk: Rows per kernel call. Above 512 rows a call returns garbage ids on
            triton-ascend / CANN 9.2.1 (see REPORT 20.11), so rows are tiled.

    Returns:
        Block ids [rows, top_k] int32, best first, ordered by score with ties to the lower id.
    """
    rows, width = scores.shape
    if top_k >= width:
        raise ValueError(f"top_k {top_k} must be below the block count {width}")
    ids = torch.empty(rows, top_k, dtype=torch.int32, device=scores.device)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        chunk = scores[start:end]
        lengths = torch.full((end - start,), width, dtype=torch.int32, device=scores.device)
        ids[start:end] = unpack_ids(top_keys(chunk, lengths, top_k))
    return ids
