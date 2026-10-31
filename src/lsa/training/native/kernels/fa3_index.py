"""The top-k selection by fa3idx, an FA3-style CUDA indexer (sm_90a).

The kernel is `vendor.fa3idx`, whose package docstring says where it comes from. A CTA
streams one key head's 64-key tiles by TMA into a 4-stage ring, two consumer warpgroups score them
by wgmma on 64 query rows each, and a row's top_k comes from v8d's two thresholds run on the
accumulators: a sample pass sets each row's band, the main pass keeps winners and band entries in
per-lane buffer segments, a final radix select picks the row's top_k, and rows whose band failed
redo by a multi-pass fallback.

Any group of query heads over key heads runs, a pair of heads a CTA for an even group and two
consecutive 64-row tiles of one head for an odd one (one head, DSA's per-head keys), up to
t - window_size = 65,536.

Its selection is the top_k within a tolerance, not exact: the final select stops once its bin is
narrower than 0.6 of tol = 0.0117 times the rms of the row's sampled scores, so a key within tol of
the k-th may take a slightly better one's place. dsa_sparse.select ranks bf16-rounded scores
exactly; this ranks fp32 accumulations of the same bf16 products.
"""

import torch

TOKENS = 65536
_indexer = None


def supported(index_queries: torch.Tensor, index_keys: torch.Tensor, window_size: int) -> bool:
    """Check whether fa3idx takes these index queries and keys.

    Args:
        index_queries: Index queries [b, h, t, d].
        index_keys: Index keys [b, h_k, t, d].
    """
    if not index_queries.is_cuda or torch.cuda.get_device_capability(index_queries.device) != (9, 0):
        return False
    b, h, t, d = index_queries.shape
    return (
        index_queries.dtype == torch.bfloat16
        and index_keys.dtype == torch.bfloat16
        and d == 64
        and index_keys.shape[-1] == d
        and h % index_keys.shape[1] == 0
        and index_keys.shape[0] == b
        and index_keys.shape[2] == t
        and t - window_size <= TOKENS
    )


def select(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    scale: float,
) -> torch.Tensor:
    """Select each query's top_k long-range keys [sink_size, i - window_size] by index_queries . index_keys * scale.

    Args:
        index_queries: Index queries [b, h, t, d].
        index_keys: Index keys [b, h_k, t, d].

    Returns:
        (b, h, t, top_k) int32 positions in no order, -1 past a row's candidates where it has at most top_k (its candidates in position order then).
    """
    global _indexer
    if not supported(index_queries, index_keys, window_size):
        raise ValueError("fa3_index.select takes bf16 d 64 heads on sm_90, t - window_size <= 65,536")
    if _indexer is None:
        from lsa.training.native.kernels.vendor.fa3idx import Indexer

        _indexer = Indexer()
    return _indexer(index_queries, index_keys, top_k, window=window_size, sink=sink_size, scale=scale)
