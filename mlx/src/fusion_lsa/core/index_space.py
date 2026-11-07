import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

NORM_EPS = 1e-6


def rownorm(rows: mx.array, eps: float = NORM_EPS) -> mx.array:
    # Normalize the final dim in fp32; a zero row stays zero
    rows = rows.astype(mx.float32)
    return rows / mx.maximum(mx.linalg.norm(rows, axis=-1, keepdims=True), eps)


def regroup(q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
    # Collapse grouped-query heads into the shared index space: [B, Hq, Tq, dh] + [B, Hkv, Tk, dh]
    # -> [B, Tq, Hkv * dh] and [B, Tk, Hkv * dh] fp32 unit rows (zero rows stay zero)
    batch_size, query_head_count, query_count, head_dim = q.shape
    if k.shape[0] != batch_size:
        raise ValueError(f"q and k must share the batch size - {batch_size} != {k.shape[0]}")
    if k.shape[3] != head_dim:
        raise ValueError(f"q and k must share the head dim - {head_dim} != {k.shape[3]}")
    kv_head_count = k.shape[1]
    if query_head_count % kv_head_count:
        raise ValueError(
            f"query heads must group evenly - {query_head_count} does not divide by {kv_head_count}"
        )
    query_heads_per_kv_head = query_head_count // kv_head_count
    index_queries = (
        q.astype(mx.float32)
        .reshape(batch_size, kv_head_count, query_heads_per_kv_head, query_count, head_dim)
        .sum(axis=2)
        .transpose(0, 2, 1, 3)
        .reshape(batch_size, query_count, kv_head_count * head_dim)
    )
    index_keys = (
        k.astype(mx.float32)
        .transpose(0, 2, 1, 3)
        .reshape(batch_size, k.shape[2], kv_head_count * head_dim)
    )
    return rownorm(index_queries), rownorm(index_keys)


@dataclass(frozen=True)
class LearnedProjection:
    # Two trained maps [Hkv * dh, index_dim] taking the regrouped space to the learned one
    query_projection: mx.array
    key_projection: mx.array

    def __post_init__(self) -> None:
        if self.query_projection.ndim != 2 or self.key_projection.ndim != 2:
            raise ValueError(
                f"both projections must be [Hkv * dh, index_dim] - got {self.query_projection.shape} "
                f"and {self.key_projection.shape}"
            )
        if self.query_projection.shape != self.key_projection.shape:
            raise ValueError(
                f"the projections must share their shape - {self.query_projection.shape} "
                f"!= {self.key_projection.shape}"
            )

    @property
    def index_dim(self) -> int:
        return int(self.key_projection.shape[1])

    @staticmethod
    def project_rows(rows: mx.array, index_map: mx.array) -> mx.array:
        # rownorm(rows @ W) - upstream index_space.project
        if rows.shape[-1] != index_map.shape[0]:
            raise ValueError(f"rows have {rows.shape[-1]} dims and the map reads {index_map.shape[0]}")
        return rownorm(rows.astype(mx.float32) @ index_map.astype(mx.float32))

    def project(self, rows: mx.array) -> mx.array:
        return self.project_rows(rows, self.key_projection)
