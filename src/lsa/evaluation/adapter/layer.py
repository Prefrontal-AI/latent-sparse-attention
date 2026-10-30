"""The generic layer - one attention module's projections, read for the index vectors of a document.

Everything here is model-free but for five hooks - `project`, `index_space`, `rotate`, `teacher_view` and
`rotated` - and the defaults are the layout a standard HF decoder already has, so most models need none of
them.
See [qwen3_5](qwen3_5.py) for one that does.

The sparse read through a keep-set went with the k-means indexer, and comes back with the Ward one.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

import torch
import torch.nn as nn

from lsa import constants as const
from lsa.core import RotaryLayout, position_free, regroup
from lsa.evaluation.constants import Layer
from lsa.sglang.settings.design import IndexerConfig


class RotaryFunction(Protocol):
    """The `apply_rotary_pos_emb` every HF modeling file defines."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the queries and keys by the cosines and sines of their positions."""


@dataclass
class Projection:
    """What one layer's projections produce, in the layout the index and the read want.

    `q` and `k` are pre-RoPE, because that is the space the index vectors are taken in.
    """

    # [B, Hq, T, dh] pre-RoPE query heads, QK-normed if the model does that
    q: torch.Tensor
    # [B, Hkv, T, dh] pre-RoPE key heads
    k: torch.Tensor
    # [B, Hkv, T, dh] value heads
    v: torch.Tensor
    # [B, T, Hq * dh] output gate, for a model that has one
    gate: torch.Tensor | None = None
    # [B, T, r] the compressed entry the keys were built from, for an MLA model whose index space reads it
    latent: torch.Tensor | None = None


@dataclass
class TeacherView:
    """What a teacher extraction reads off one layer for one document.

    `q` and `k` are whatever pair of vectors dot to the model's own attention logit - rotated heads for a
    model that stores its keys per head, and queries folded into the latent for one that stores a latent.
    """

    # [Q, d] fp32 unit index vectors at the sampled query positions
    index_queries: torch.Tensor
    # [T, d] fp32 unit index vectors of every key
    index_keys: torch.Tensor
    # [Hq, Q, dq] query vectors at the sampled positions
    q: torch.Tensor
    # [Hkv, T, dq] key vectors, `Hq / Hkv` query heads to each
    k: torch.Tensor


@dataclass
class TeacherSlice:
    """One slice of a document's positions as a teacher projects it."""

    # The slice's projections, pre-RoPE
    projection: Projection
    # [Hq, S, dh] and [Hkv, S, dh] the rotated heads the model attends with
    q: torch.Tensor
    k: torch.Tensor
    # [R] int64 which of the sampled rows sit in the slice, and where in it
    rows: torch.Tensor
    offsets: torch.Tensor


def rotary_function(attention: Any) -> RotaryFunction:
    """Return the `apply_rotary_pos_emb` the model's own modeling file defines.

    Resolved from the attention class's module rather than imported by name, so a new model needs no entry
    here and the rotation applied is the model's own - partial, interleaved or otherwise.

    Args:
        attention: The layer's native attention module.

    Returns:
        The model's rotary function.

    Raises:
        ValueError: The modeling module defines no `apply_rotary_pos_emb`.
    """
    module_name = type(attention).__module__
    function = getattr(import_module(module_name), "apply_rotary_pos_emb", None)
    if function is None:
        raise ValueError(f"{module_name} defines no apply_rotary_pos_emb - override `rotate` for this model")
    return function


def qk_norms(attention: Any) -> tuple[Any, Any]:
    """Return the model's per-head query and key norms.

    Args:
        attention: The layer's native attention module.

    Returns:
        query_norm: The query norm module, or None when the model has none.
        key_norm: The key norm module, or None when the model has none.
    """
    for query_name, key_name in Layer.QK_NORMS:
        query_norm = getattr(attention, query_name, None)
        key_norm = getattr(attention, key_name, None)
        if query_norm is not None or key_norm is not None:
            return query_norm, key_norm
    return None, None


def heads(projected: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Split a projection's last dimension into heads and move them in front of the sequence.

    Args:
        projected: A projection output [B, T, H * dh].
        head_dim: Dims of one head.

    Returns:
        Head tensors [B, H, T, dh].
    """
    batch_size, query_count = projected.shape[:2]
    return projected.view(batch_size, query_count, -1, head_dim).transpose(1, 2)


class SparseAttention(nn.Module):
    """One layer's native attention, read for the index vectors a teacher extraction caches."""

    def __init__(self, attention: Any, config: IndexerConfig, rotary: RotaryLayout) -> None:
        """Wrap a native attention module.

        Args:
            attention: The layer's own attention module, kept and called for its projections. Typed
                loosely because `nn.Module.__getattr__` resolves every read on it to `Tensor | Module`.
            config: The indexer knobs.
            rotary: Where the rotation sits in a head vector for this model.
        """
        super().__init__()
        self.attention = attention
        self.config = config
        self.rotary = rotary

    def teacher_view(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> TeacherView:
        """Return what an extraction caches for one document, and what it scores the teacher with.

        The document is projected a slice of positions at a time and only the sampled rows of each slice's queries
        are kept, so a million-token document never holds every position's queries at once.

        Args:
            hidden_states: Layer input [1, T, model_dim], on any device - it is moved to the layer's.
            positions: Sampled query positions [Q] int64, on the layer's device.
            position_embeddings: The model's own `(cos, sin)` tables for the document, or None.

        Returns:
            The index vectors and the logit vectors, one document's worth.

        Raises:
            ValueError: The document has no position.
        """
        index_queries, q = None, None
        index_keys, k = [], []
        for piece in self.teacher_slices(hidden_states, positions, position_embeddings):
            slice_index_queries, slice_index_keys = self.index_space(piece.projection)
            index_keys.append(slice_index_keys[0])
            k.append(piece.k)
            if index_queries is None or q is None:
                index_queries = slice_index_queries.new_empty((len(positions), slice_index_queries.shape[-1]))
                q = piece.q.new_empty((piece.q.shape[0], len(positions), piece.q.shape[-1]))
            index_queries[piece.rows] = slice_index_queries[0, piece.offsets]
            q[:, piece.rows] = piece.q[:, piece.offsets]
        if index_queries is None or q is None:
            raise ValueError("a teacher view needs a document of at least one position")
        return TeacherView(index_queries, torch.cat(index_keys), q, torch.cat(k, dim=1))

    def teacher_slices(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Iterator["TeacherSlice"]:
        """Project a document's input a slice of `Layer.TEACHER_SLICE` positions at a time, as a teacher reads it.

        The slices start at the input's first position, so a document handed over a chunk of that many positions
        at a time is projected in exactly the slices it is handed whole.

        Args:
            hidden_states: Layer input [1, T, model_dim], on any device - each slice is moved to the layer's.
            positions: Sampled query positions [Q] int64, relative to the input's first, on the layer's device.
            position_embeddings: The model's own `(cos, sin)` tables over the input, or None.

        Yields:
            Each slice's projections, its rotated heads and the sampled rows inside it.
        """
        device = positions.device
        key_count = hidden_states.shape[1]
        for start in range(0, key_count, Layer.TEACHER_SLICE):
            stop = min(start + Layer.TEACHER_SLICE, key_count)
            projection = self.project(hidden_states[:, start:stop].to(device))
            tables = None
            if position_embeddings is not None:
                tables = (
                    position_embeddings[0][:, start:stop].to(device),
                    position_embeddings[1][:, start:stop].to(device),
                )
            q, k = self.rotated(projection, tables)
            # The sampled rows inside this slice, wherever they sit in `positions`
            rows = ((positions >= start) & (positions < stop)).nonzero().squeeze(1)
            yield TeacherSlice(projection, q[0], k[0], rows, positions[rows] - start)

    def index_space(self, projection: Projection) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the index vectors this layer's queries and keys are selected in.

        The default is the position-free sub-term of the logit, GQA-collapsed, which is the space for any
        model that stores its keys per head. An MLA model stores one latent instead and overrides this to
        read it - [glm5_next](glm5_next.py) is the one here.

        Args:
            projection: This step's projections.

        Returns:
            index_queries: Query index vectors [B, Tq, d] fp32 unit rows.
            index_keys: Key index vectors [B, Tk, d] fp32 unit rows.
        """
        return regroup(*position_free(projection.q, projection.k, self.rotary))

    def rotated(
        self, projection: Projection, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the rotation, or pass the heads through for a model that has none.

        A NoPE model - GLM-5.3-Flash's DSA layers are one - is handed `position_embeddings=None` by its own
        decoder layer, and the whole head vector is already position-free.

        Args:
            projection: This step's projections.
            position_embeddings: The model's `(cos, sin)` tables, or None when it does not rotate.

        Returns:
            The query heads and key heads to attend with.

        Raises:
            ValueError: There are no tables but the measured layout says this model rotates.
        """
        if position_embeddings is None:
            if self.rotary.end != const.Rotary.NONE:
                raise ValueError(
                    f"this layer was built for a {self.rotary.end!r} rotation of {self.rotary.dim} dims "
                    "but its decoder layer passed no rotary tables"
                )
            return projection.q, projection.k
        return self.rotate(self.attention, projection.q, projection.k, *position_embeddings)

    def project(self, hidden_states: torch.Tensor) -> Projection:
        """Run the layer's projections and return head tensors.

        The default is the standard HF decoder layout - `q_proj`, `k_proj`, `v_proj`, with QK-norm applied
        under either name transformers gives it, and no output gate.

        Args:
            hidden_states: Layer input [B, T, model_dim].

        Returns:
            The layer's pre-RoPE query, key and value heads, and its gate if it has one.
        """
        attention = self.attention
        head_dim = attention.head_dim
        q = heads(attention.q_proj(hidden_states), head_dim)
        k = heads(attention.k_proj(hidden_states), head_dim)
        v = heads(attention.v_proj(hidden_states), head_dim)

        query_norm, key_norm = qk_norms(attention)
        if query_norm is not None:
            q = query_norm(q)
        if key_norm is not None:
            k = key_norm(k)
        return Projection(q=q, k=k, v=v)

    @staticmethod
    def rotate(
        attention: Any, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the model's own rotary embedding to query and key heads.

        Static because [rotary](rotary.py) measures the layout by calling it before any layer is built, and
        the rotation it measures has to be the rotation the layer will apply.

        Args:
            attention: The layer's native attention module, read for its modeling module.
            q: Query heads [B, Hq, T, dh].
            k: Key heads [B, Hkv, T, dh].
            cos: The model's cosine table for this step.
            sin: The model's sine table for this step.

        Returns:
            The rotated query heads and key heads, in the shapes they arrived in.
        """
        rotate_heads = rotary_function(attention)

        # cos is as wide as the rotation, so a cos narrower than a head is a partial rotation.
        # Some models slice inside `apply_rotary_pos_emb` and some slice in their attention forward before
        # calling it - slicing here is correct for both, and the second kind broadcasts into an error without it
        rotary_dim = cos.shape[-1]
        if rotary_dim >= q.shape[-1]:
            return rotate_heads(q, k, cos, sin)

        q_rotated, k_rotated = rotate_heads(q[..., :rotary_dim], k[..., :rotary_dim], cos, sin)
        return (
            torch.cat((q_rotated, q[..., rotary_dim:]), dim=-1),
            torch.cat((k_rotated, k[..., rotary_dim:]), dim=-1),
        )
