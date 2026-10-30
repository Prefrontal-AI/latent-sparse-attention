"""DSA sparse attention kernels: the indexer's exact top-k selection, and flash attention over it.

Layout-agnostic, for any layer that ranks keys with a small indexer and attends over each
query's window_size, sink_size and top_k long-range keys:

- select(index_queries, index_keys, ...): head-major indexer queries (b, h_sel, t, d_r) over indexer keys
  (b, h_ik, t, d_r), h_sel a multiple of h_ik, to a Selection with h_sel heads;
- attention(q, k, v, selection, ...): queries (b, t, h_q, d) over keys and values
  (b, t, h_kv, d), h_q a multiple of h_kv - multi-head, grouped or multi-query attention,
  query head h reading key/value head h // (h_q // h_kv) - and a selection of h_q heads,
  one per query head, or h_kv, shared by each group.

Both are torch.library custom ops with fake implementations, so a compiled model calls them
without a graph break, and attention has a registered autograd to q, k and v; the selection
has no gradient. They need a CUDA device and fp16 or bf16 inputs; attention needs a head_dim
of HEAD_DIMS. A whole unpadded causal sequence only: query i is at position i.
"""

import torch

from lsa.training.native.kernels.dsa_sparse.flash import HEAD_DIMS, sparse_attention_op
from lsa.training.native.kernels.dsa_sparse.topk import (
    Selection,
    occupancy_size,
    select_op,
    unpack,
    words_per_row,
)

__all__ = ["HEAD_DIMS", "Selection", "attention", "kept_rows", "select", "supported"]

DTYPES = (torch.float16, torch.bfloat16)


def supported(q: torch.Tensor, index_queries: torch.Tensor) -> bool:
    """Check whether the kernels take these queries and indexer queries, as autocast leaves them."""
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else q.dtype
    return (
        q.is_cuda
        and dtype in DTYPES
        and index_queries.dtype in DTYPES
        and q.shape[-1] in HEAD_DIMS
        and index_queries.shape[-1] <= 256
    )


def select(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    scale: float,
) -> Selection:
    """Select each query's top_k long-range keys by indexer score, in the indexer's dtype.

    The rank score is dtype(dtype(index_queries . index_keys) * scale), as an einsum in that dtype and its scaling round it, and the selection the one DsaDetachedProjectedAttention's dsa_select.sparse_selection makes from it.
    That is exactly top_k candidates per query, an exact tie to the later key, and every candidate of a query with at most top_k.
    Candidates of query i are keys [sink_size, i - window_size].
    The window_size and sink_size themselves are kept by position, not in bits.

    Args:
        index_queries: Head-major indexer queries [b, h_sel, t, d_r].
        index_keys: Indexer keys [b, h_ik, t, d_r], h_sel a multiple of h_ik.

    Returns:
        A Selection with h_sel heads.
    """
    if index_queries.dtype not in DTYPES or index_keys.dtype != index_queries.dtype:
        raise ValueError(f"select needs fp16 or bf16 indexer queries and keys, got {index_queries.dtype}")
    if index_queries.shape[1] % index_keys.shape[1] != 0 or index_queries.shape[::2] != index_keys.shape[::2]:
        raise ValueError(
            f"indexer queries {tuple(index_queries.shape)} and keys {tuple(index_keys.shape)} differ"
        )
    if top_k <= 0:
        b, h, t, _ = index_queries.shape
        n = occupancy_size(t)
        bits = index_queries.new_zeros((b, h, t, words_per_row(t)), dtype=torch.int32)
        return Selection(bits, index_queries.new_zeros((b, h, n, n), dtype=torch.int32))
    return Selection(*select_op(index_queries, index_keys, window_size, sink_size, top_k, scale))


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: Selection,
    window_size: int,
    sink_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal softmax attention over each query's window_size, sink_size and selected keys.

    Under autocast q, k and v are cast to its dtype, as SDPA casts them.

    Args:
        q: Queries [b, t, h_q, d].
        k: Keys [b, t, h_kv, d], h_q a multiple of h_kv.
        v: Values [b, t, h_kv, d].
        selection: A selection of h_q heads, one per query head, or h_kv, shared by each group.

    Returns:
        The [b, t, h_q, d] context and the [b, h_q, t] fp32 log-sum-exp of the scaled scores, which carries no gradient.
    """
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    b, t, h_q, d = q.shape
    h_kv = k.shape[2]
    if d not in HEAD_DIMS:
        raise ValueError(f"head_dim {d} is not one of {HEAD_DIMS}")
    if q.dtype not in DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"attention needs q, k and v of one dtype of {DTYPES}")
    if h_q % h_kv != 0 or k.shape != v.shape or k.shape[:2] != (b, t) or k.shape[3] != d:
        raise ValueError(f"q {tuple(q.shape)} does not group over k and v {tuple(k.shape)}")
    if selection.bits.shape[1] not in (h_q, h_kv) or selection.bits.shape[2] != t:
        raise ValueError(f"a selection of {selection.bits.shape[1]} heads fits neither {h_q} nor {h_kv}")
    with torch.autocast("cuda", enabled=False):
        o, lse = sparse_attention_op(
            q, k, v, selection.bits, selection.occupancy, window_size, sink_size, scale
        )
    return o, lse.detach()


def kept_rows(selection: Selection, rows: torch.Tensor, window_size: int, sink_size: int) -> torch.Tensor:
    """Return the bool mask of every key the queries in rows keep: window_size, sink_size and bits.

    Args:
        rows: Query positions [n].

    Returns:
        Mask [b, h_sel, n, t].
    """
    t = selection.bits.shape[2]
    keys = torch.arange(t, device=rows.device)
    delta = rows[:, None] - keys[None, :]
    base = (delta >= 0) & ((delta < window_size) | (keys[None, :] < sink_size))
    return unpack(selection.bits[:, :, rows], t) | base
