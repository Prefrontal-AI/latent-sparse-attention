"""The space the index lives in - the position-free sub-term of the attention logit, GQA-collapsed.

The logit splits as (rotary dims) + (the rest).
The rest is an exact sub-term of the true score with no position in it, so clustering groups keys by content
rather than by where they sit.
Which dims those are is model-specific, in how many and in which end carries the rotation, so it is a parameter
here and never a hardcoded prefix.

| model | rotary dims | `RotaryLayout` |
|---|---|---|
| Qwen3.5-4B-Base | first 64 of each 256-dim head | `RotaryLayout("prefix", 64)` |
| DeepSeek-V4-Flash | last 64 of the 512-dim MQA entry | `RotaryLayout("suffix", 64)` |
| GLM-5.3-Flash | none | `RotaryLayout("none")` |

An MLA model takes the other route.
Its keys are all built from one compressed latent, so the head projections move onto the query side and the
index is the latent - `regroup_latent` and `LatentMetric` below, 512 dims where collapsing GLM's expanded
heads would give 16,384 for the same selection.

Heads come in as `[B, H, T, dh]`, the layout a model's attention already has.
Index vectors go out as `[B, T, Hkv * dh]` fp32 unit rows, which is what everything downstream
of here consumes.

The learned variant carries those rows one step further, through a trained `[Hkv * dh, index_dim]` map into a narrower
space that the index lives in.
It is the same contract - unit rows - so nothing downstream knows which space it was handed.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lsa import constants as const


@dataclass(frozen=True)
class RotaryLayout:
    """Where the rotation sits in a head vector, and how many dims it covers."""

    end: str = const.Rotary.NONE
    dim: int = 0

    def __post_init__(self) -> None:
        """Validate that the end and dim fields agree with each other.

        Raises:
            ValueError: If `end` is not one of `const.Rotary.ENDS`.
            ValueError: If `dim` is negative.
            ValueError: If `end` and `dim` disagree about whether there is a rotation.
        """
        if self.end not in const.Rotary.ENDS:
            raise ValueError(f"end must be one of {const.Rotary.ENDS} - got {self.end!r}")
        if self.dim < 0:
            raise ValueError(f"dim must not be negative - got {self.dim}")
        if (self.end == const.Rotary.NONE) != (self.dim == 0):
            raise ValueError(
                f"end {self.end!r} and dim {self.dim} disagree about whether there is a rotation"
            )


def position_free(
    q: torch.Tensor, k: torch.Tensor, rotary: RotaryLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the rotated dims from pre-RoPE query and key heads.

    What remains is an exact, position-independent sub-term of the true score rather than an approximation.
    A model whose rotation covers the whole head vector has no such sub-term.

    Args:
        q: Pre-RoPE query heads [..., dh], floating dtype.
        k: Pre-RoPE key heads [..., dh], floating dtype.
        rotary: Where the rotation sits in the head vector and how many dims it covers.

    Returns:
        query_without_rotation: Query heads with the rotated dims dropped [..., dh - rotary.dim], same
            dtype and device as `q`.
        key_without_rotation: Key heads with the rotated dims dropped [..., dh - rotary.dim], same dtype
            and device as `k`.

    Raises:
        ValueError: If `q` and `k` do not share the head dim.
        ValueError: If `rotary.dim` covers the whole head, leaving nothing position-free.
    """

    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim:
        raise ValueError(f"q and k must share the head dim - {head_dim} != {k.shape[-1]}")
    if rotary.dim >= head_dim:
        raise ValueError(
            f"rotation covers the whole head - {rotary.dim} dims of {head_dim}, nothing is position-free"
        )

    if rotary.end == const.Rotary.NONE:
        return q, k

    if rotary.end == const.Rotary.PREFIX:
        # [..., dh - dim] views with the rotated prefix removed
        query_without_rotation, key_without_rotation = q[..., rotary.dim :], k[..., rotary.dim :]
        return query_without_rotation, key_without_rotation

    # [..., dh - dim] views with the rotated suffix removed
    query_without_rotation, key_without_rotation = (
        q[..., : head_dim - rotary.dim],
        k[..., : head_dim - rotary.dim],
    )
    return query_without_rotation, key_without_rotation


def regroup(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse grouped-query heads into the shared index space.

    Output rows are unit norm except that zero rows remain zero.

    `sum_h q_h . k_(h//G) == concat_kv( sum_(h in group) q_h ) . concat_kv( k_kv )`, so one shared selection
    scores exactly what the attention heads sum to.
    Exact, not an approximation - the identity is about summed logits, and it does not equate a softmax of
    summed logits with summed per-head attention.

    Args:
        q: Pre-index query heads [B, Hq, Tq, dh], floating.
        k: Pre-index key heads [B, Hkv, Tk, dh], floating.

    Returns:
        index_queries: Collapsed query index vectors [B, Tq, Hkv * dh] fp32.
        index_keys: Collapsed key index vectors [B, Tk, Hkv * dh] fp32.
    """
    index_queries, index_keys = regroup_partial(q, k)

    # [B, Tq, Hkv * dh] and [B, Tk, Hkv * dh] fp32 normalized index vectors
    return rownorm(index_queries), rownorm(index_keys)


def regroup_partial(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse grouped-query heads without normalizing the rows.

    The head sum and the concatenation are per key-value head, so a rank holding a slice of the heads can do
    this much alone. The normalization cannot be: it divides by the norm of the whole vector, and a rank
    that normalizes its own slice by its own norm scales each block of the row by a different number, which
    turns the assembled vector into a different direction rather than a rescaled one.
    So the sharded paths take this exit, combine across the group, and normalize once at the end.

    On one device it is `regroup` without its last line, which is the only reason `regroup` is still one
    call rather than two at every call site.

    Args:
        q: Pre-index query heads [B, Hq, Tq, dh], floating.
        k: Pre-index key heads [B, Hkv, Tk, dh], floating.

    Returns:
        index_queries: Collapsed query index vectors [B, Tq, Hkv * dh] fp32, not normalized.
        index_keys: Collapsed key index vectors [B, Tk, Hkv * dh] fp32, not normalized.

    Raises:
        ValueError: If `q` and `k` do not share the batch size.
        ValueError: If `q` and `k` do not share the head dim.
        ValueError: If the query head count does not divide evenly by the KV head count.
    """

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

    # [B, Tq, Hkv * dh] - sum each KV group, then concatenate groups
    index_queries = (
        q.float()
        .view(batch_size, kv_head_count, query_heads_per_kv_head, query_count, head_dim)
        .sum(2)
        .transpose(1, 2)
        .reshape(batch_size, query_count, kv_head_count * head_dim)
    )

    # [B, Tk, Hkv * dh] - concatenate KV heads at each position
    index_keys = k.float().transpose(1, 2).reshape(batch_size, k.shape[2], kv_head_count * head_dim)

    return index_queries, index_keys


@dataclass(frozen=True)
class LatentMetric:
    """An MLA layer's key maps, and the factor that makes its latent measure what its keys measure.

    MLA builds every key from one compressed latent, `k_h = W_h latent`, so the head-summed logit `regroup`
    preserves moves onto the query side exactly:

        sum_h q_h . k_h  =  sum_h q_h . (W_h latent)  =  (sum_h W_h^T q_h) . latent

    The key side is then the latent itself and the index is `r` wide rather than `Hq * dh` - on
    GLM-5.3-Flash, 512 against 16,384.

    That identity is about raw logits, and the index space is normalized.
    Two latents meet each other through the key maps rather than directly - `concat_k . concat_k' = latent^T
    M latent'` with `M = sum_h W_h^T W_h` - so `M` is the metric the concatenated keys impose on the latent.
    The latent's own coordinates are not orthonormal under it, because the maps stretch some directions more
    than others, and the lengths and angles a selection reads are therefore not the ones the plain latent
    has. Clustering it as it stands is a different geometry rather than a narrower spelling of the same one.

    `metric_factor` is the Cholesky factor `L` of `M`. Carried through it, `L^T latent` has back exactly the
    lengths and angles the concatenation has, which makes the two spaces isometric and every selection
    identical.
    """

    # [Hq, dh, r] the key half of the latent expansion, one map per query head
    key_maps: torch.Tensor
    # [r, r] fp32 lower-triangular `L` of the key metric, `L L^T = sum_h key_maps_h^T key_maps_h`
    metric_factor: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the two carry one layer's geometry.

        Raises:
            ValueError: `key_maps` is not `[Hq, dh, r]`.
            ValueError: `metric_factor` is not a square matrix of the latent's dim.
        """
        if self.key_maps.ndim != 3:
            raise ValueError(f"key_maps must be [Hq, dh, r] - got {tuple(self.key_maps.shape)}")
        rank = self.key_maps.shape[2]
        if self.metric_factor.shape != (rank, rank):
            raise ValueError(
                f"metric_factor must be [{rank}, {rank}] to match the key maps - "
                f"got {tuple(self.metric_factor.shape)}"
            )

    @property
    def dim(self) -> int:
        """Dims of the index space, `r`."""
        return int(self.key_maps.shape[2])

    @classmethod
    def from_key_maps(cls, key_maps: torch.Tensor) -> "LatentMetric":
        """Factor the metric the key maps impose on the latent.

        It is taken in fp64 because it sums `Hq * dh` outer products of weights a checkpoint stores in bf16,
        and a Cholesky is what tells a caller the maps are degenerate rather than a silent NaN further down.

        Args:
            key_maps: The key half of the latent expansion [Hq, dh, r], floating.

        Returns:
            The metric those maps impose, with the maps themselves.

        Raises:
            ValueError: `key_maps` is not `[Hq, dh, r]`.
            ValueError: The maps do not span the latent, so the metric has no factor.
        """
        if key_maps.ndim != 3:
            raise ValueError(f"key_maps must be [Hq, dh, r] - got {tuple(key_maps.shape)}")

        # [r, r] fp64 - sum_h W_h^T W_h, the metric the concatenated keys impose on the latent
        metric = torch.einsum("hdr,hde->re", key_maps.double(), key_maps.double())
        factored = torch.linalg.cholesky_ex(metric)
        if int(factored.info) != 0:
            raise ValueError(
                f"the key maps do not span the {key_maps.shape[2]}-dim latent - "
                "the metric they impose on it is singular and cannot be factored"
            )
        return cls(key_maps, factored.L.float())


def regroup_latent(
    q: torch.Tensor, latent: torch.Tensor, metric: LatentMetric
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse an MLA layer into the shared index space without ever expanding its keys.

    What `regroup` does for a model that stores per-head keys, for one that stores the latent they are built
    from. The output is the same contract - unit rows one selection is made in - and the same selection, at
    `r` dims instead of `Hq * dh`.

    Args:
        q: Pre-index query heads [B, Hq, Tq, dh], floating.
        latent: The compressed entry every key is built from [B, Tk, r], floating.
        metric: The layer's key maps and the factor of the metric they impose on the latent.

    Returns:
        index_queries: Collapsed query index vectors [B, Tq, r] fp32 unit rows.
        index_keys: Latent index vectors [B, Tk, r] fp32 unit rows.

    Raises:
        ValueError: `q` does not carry the heads and the head dim the key maps read.
        ValueError: `latent` is not `[B, Tk, r]` for the dim the key maps write.
    """
    head_count, head_dim, rank = metric.key_maps.shape
    if q.ndim != 4 or q.shape[1] != head_count or q.shape[3] != head_dim:
        raise ValueError(
            f"q must be [B, {head_count}, Tq, {head_dim}] to match the key maps - got {tuple(q.shape)}"
        )
    if latent.ndim != 3 or latent.shape[2] != rank:
        raise ValueError(f"latent must be [B, Tk, {rank}] to match the key maps - got {tuple(latent.shape)}")

    # [B, Tq, r] - every head's key map pulled onto the query side, summed over the heads
    absorbed_queries = torch.einsum("bhqd,hdr->bqr", q.float(), metric.key_maps.float())

    # [B, Tq, r] - L^-1 of that, which is the side that pairs with L^T latent
    index_queries = torch.linalg.solve_triangular(
        metric.metric_factor, absorbed_queries.transpose(1, 2), upper=False
    ).transpose(1, 2)

    # [B, Tk, r] - the latent carried into the metric the concatenated keys impose on it
    index_keys = latent.float() @ metric.metric_factor

    return rownorm(index_queries), rownorm(index_keys)


def rownorm(rows: torch.Tensor, eps: float = const.Math.NORM_EPS) -> torch.Tensor:
    """Normalize the final dimension in fp32 without changing shape or device.

    Unit rows make the dot products cosines in `[-1, 1]`.

    A zero row stays zero and scores 0 against everything, which is the only sane reading of a key with no
    direction.

    Args:
        rows: Rows to normalize [..., d], floating.
        eps: Norm floor that keeps a zero row at zero instead of producing NaN.

    Returns:
        Unit-norm rows [..., d] fp32, with zero rows unchanged.
    """

    # [..., d] fp32 unit rows, with zero rows unchanged
    normalized_rows = F.normalize(rows.float(), dim=-1, eps=eps)

    return normalized_rows


@dataclass(frozen=True)
class LearnedProjection:
    """Two trained maps that take the position-free space to a narrower one.

    The learned variant of the index space: `regroup` still produces the `Hkv * dh` position-free dims, and
    these carry them to the `index_dim` dims that the index lives in.
    Nothing downstream knows the difference - it takes unit rows either way.

    Both projections are `[Hkv * dh, index_dim]`, and a model with one map per indexer-carrying layer holds one of these per
    layer.
    """

    # [Hkv * dh, index_dim] the projection applied to index queries
    query_projection: torch.Tensor
    # [Hkv * dh, index_dim] the projection applied to index keys
    key_projection: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that the two maps describe one space.

        Raises:
            ValueError: A map is not a matrix.
            ValueError: The maps disagree on the space they read or the space they write.
        """
        if self.query_projection.ndim != 2 or self.key_projection.ndim != 2:
            raise ValueError(
                f"both projections must be [Hkv * dh, index_dim] - got {tuple(self.query_projection.shape)} "
                f"and {tuple(self.key_projection.shape)}"
            )
        if self.query_projection.shape != self.key_projection.shape:
            raise ValueError(
                f"the projections must share their shape - {tuple(self.query_projection.shape)} "
                f"!= {tuple(self.key_projection.shape)}"
            )

    @property
    def index_dim(self) -> int:
        """Dims of the learned space, the index dim."""
        return int(self.key_projection.shape[1])


def project_partial(rows: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
    """Carry index vectors through one learned map without normalizing the result.

    `W` is `[d, kappa]` and the index vector is the concatenation of per-head blocks, so the map is a sum
    over head blocks rather than one dense product, and a rank holding rows for its own heads can compute
    its own term from the matching rows of `W`.
    Summing the terms across the group and normalizing once is `project` of the whole vector, because the
    normalization `regroup` would have applied first is a positive scalar per row and passes through the
    normalization at the end unchanged.

    Args:
        rows: Index vectors [..., d_local], floating, normalized or not.
        index_map: The rows of the trained map belonging to these heads [d_local, kappa], floating.

    Returns:
        This rank's term of the learned index vectors [..., kappa] fp32, not normalized.

    Raises:
        ValueError: `rows` and `index_map` disagree on the dim of the space being left.
    """
    if rows.shape[-1] != index_map.shape[0]:
        raise ValueError(f"rows have {rows.shape[-1]} dims and the map reads {index_map.shape[0]}")

    # [..., kappa] fp32 one term of the sum over head blocks
    return rows.float() @ index_map.float()


def project(rows: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
    """Carry index vectors through one learned map and renormalize.

    Args:
        rows: Index vectors [..., d], floating.
        index_map: The trained map [d, kappa], floating.

    Returns:
        Learned index vectors [..., kappa] fp32 unit rows, with zero rows unchanged.

    Raises:
        ValueError: `rows` and `index_map` disagree on the dim of the space being left.
    """
    # [..., kappa] fp32 unit rows in the learned space
    return rownorm(project_partial(rows, index_map))
