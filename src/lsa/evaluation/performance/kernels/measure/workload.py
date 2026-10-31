"""How much memory one measurement needs, and which query rows to score when it cannot all be kept.

`memory_admission` refuses a configuration before it is attempted rather than after it has died, and
`score_rows` picks an evenly spaced subset of queries when the full score matrix will not fit.

A variant with no index scores nothing and gathers nothing: full attention and the linear kernels work in
place, so both figures are zero and there are no rows to pick.
"""

import math

import torch

# Cap on QSA's fp32 prefill score tile - the most its scores take at any context
QSA_PREFILL_SCORES_BYTES = 512 * 2**20

# Share of the card DSA's fp32 indexer logits may take - it slices a chunk's rows to stay under it
NATIVE_LOGITS_CARD_FRACTION = 0.3


def score_rows(chunk, context, indexer, budget_mib, bucket_size=None):
    """Pick the query rows whose scores a check holds, or None for all of them.

    Args:
        chunk: Query rows a chunk has.
        context: The chunk's end.
        indexer: Which indexer ran.
        budget_mib: The most the checked scores may take.
        bucket_size: The design's average keys in a bucket, for `lsa`.

    Returns:
        Evenly spaced rows, or None when every row's scores fit.
    """
    # PISA scores a pyramid, which gathers no score matrix to hold
    # => no rows to pick
    if indexer not in ("native", "lsa"):
        return None
    candidates = context // (4 if indexer == "native" else bucket_size)
    count = min(chunk, max(2, budget_mib * 2**20 // (4 * candidates)))
    if count == chunk:
        return None
    return sorted({i * (chunk - 1) // (count - 1) for i in range(count)})


def memory_admission(chunk, context, indexer, validation_mib, margin_gib, config=None):
    """Refuse a configuration whose scores and gathers will not fit the card, before it is attempted.

    Args:
        chunk: Query rows a chunk has.
        context: The chunk's end.
        indexer: Which indexer ran - `qsa` for Qwen's own.
        validation_mib: The budget of the checked scores.
        margin_gib: Memory left free.
        config: The design's `IndexerConfig`, for `lsa`.

    Returns:
        What was weighed and that the configuration was admitted.

    Raises:
        MemoryError: The configuration does not fit.
    """
    free, total = torch.cuda.mem_get_info()
    if indexer == "native":
        # The indexer reads a chunk in slices of rows whose logits fit its cap
        # => its scores take the whole matrix or the cap, whichever is less
        scores = min(chunk * math.ceil(context / 4) * 4, int(total * NATIVE_LOGITS_CARD_FRACTION))
        index_gather = math.ceil(context / 4) * 132
    elif indexer == "qsa":
        # QSA tiles its fp32 prefill scores by rows under a fixed cap and reads its compressed keys in place
        scores = QSA_PREFILL_SCORES_BYTES
        index_gather = 0
    elif indexer == "lsa":
        from lsa.core.kernels.triton.scan_select import SCAN_SELECT_BUCKETS

        # Dense scores are fp32 for the buckets of a read below the fused select, which writes none, and the
        # centroids a read gathers are bf16 rows - a lone request's centroids are a view, so this is the bound
        buckets = math.ceil(context / config.bucket_size)
        scores = chunk * min(buckets, SCAN_SELECT_BUCKETS) * 4
        index_gather = buckets * config.index_dim * 2
    else:
        # No index - FA3 and the linear kernels work in place
        # PISA's search has no bound of its own, so the margin stands in for it
        scores = index_gather = 0
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    available = free + reserved - allocated
    requirement = scores + index_gather + 3 * validation_mib * 2**20 + margin_gib * 2**30
    result = {
        "context": context,
        "score_workspace_bound_bytes": scores,
        "index_gather_bound_bytes": index_gather,
        "validation_budget_mib": validation_mib,
        "margin_gib": margin_gib,
        "available_including_allocator_cache_bytes": available,
        "required_extra_bound_bytes": requirement,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "device_total_bytes": total,
        "admitted": requirement < available,
    }
    if not result["admitted"]:
        raise MemoryError(f"Layer memory admission failed: {result}")
    return result
