"""The learned index space - one pair of projections per layer, and the temperature the loss reads it at.

The `key_projection` and the `query_projection` carry `x`, the attention's input, into a narrower space, and
that space is the whole index: the read scores keys in it.
An index key is `normalize(x @ key_projection)` and an index query is `normalize(x @ query_projection)`.
Both start as the leading PCA directions of the layer's own `x`, which is the best untrained space of that
width there is, and training moves them from there.

Under `indexer.input: key_and_query` the projections read GLM's key latent and the heads' queries summed, in the bases
the original implementation distilled them in: the key carried by `key_basis` and the query by `query_basis` before its projection, so an
index key is `normalize(key @ key_basis @ key_projection)`. The bases are fixed, the PCA is taken of the keys carried
into theirs and normalized, as the original implementation's cache held them, and `served_projections` folds them in - what a server reads.

`tau` is a parameter rather than a constant because the loss is a softmax over tens of thousands of keys and
its sharpness is not something to guess.
It is calibrated once on the untrained map - see [losses](losses.py) - and trained after that.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from lsa.baselines import qsa
from lsa.core import project, rownorm
from lsa.sglang.settings.design import QSAConfig
from lsa.training.distillation.constants import Space


class _ProjectChunked(torch.autograd.Function):
    """`x @ projection` over a document too long to hold in fp32 at once.

    `x` is bf16 and a million positions of it are 8.7 GB, so its fp32 copy is not something to keep for the
    backward pass. This casts a chunk at a time going forward, saves `x` itself, and casts it a chunk at a time
    going back.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, projection: torch.Tensor, chunk: int) -> torch.Tensor:
        """Project every position of `x`, a chunk at a time.

        Args:
            ctx: The autograd context.
            x: The attention's input [T, model_dim], any float dtype.
            projection: The map [model_dim, index_dim] fp32.
            chunk: Positions cast and multiplied at once.

        Returns:
            The projected positions [T, index_dim] fp32, not normalized.
        """
        ctx.save_for_backward(x, projection)
        ctx.chunk = chunk
        out = x.new_empty((x.shape[0], projection.shape[1]), dtype=torch.float32)
        for start in range(0, x.shape[0], chunk):
            out[start : start + chunk] = x[start : start + chunk].float() @ projection
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor, None]:
        """Return the gradient of the projection, which is the only input that learns.

        Args:
            ctx: The autograd context.
            grad: The gradient of the projected positions [T, index_dim] fp32.

        Returns:
            Nothing for `x`, the gradient of the projection [model_dim, index_dim] fp32, and nothing for the chunk.
        """
        x, projection = ctx.saved_tensors
        gradient = torch.zeros_like(projection)
        for start in range(0, x.shape[0], ctx.chunk):
            gradient += x[start : start + ctx.chunk].float().t() @ grad[start : start + ctx.chunk]
        return None, gradient, None


class IndexMap(nn.Module):
    """One layer's two projections and its loss temperature."""

    # The fixed bases the projections act in, under the key and query - None where they read what they are given
    key_basis: torch.Tensor | None
    query_basis: torch.Tensor | None

    def __init__(
        self,
        model_dim: int,
        index_dim: int,
        init: torch.Tensor | None = None,
        bases: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Build the projections, from PCA directions when given rows to take them from.

        Args:
            model_dim: Dims of what the projections read - `x`, the model's own, or the key under the key and query.
            index_dim: Dims of the index keys and index queries.
            init: Rows of what the key projection reads to take the leading directions from [n, model_dim], or None
                for a random start.
            bases: The key basis and the query basis the projections act in [model_dim, model_dim] each, or None to
                project what they read as it is.

        Raises:
            ValueError: `index_dim` is wider than what it reads.
        """
        super().__init__()
        if index_dim > model_dim:
            raise ValueError(f"an index of {index_dim} dims cannot be wider than the {model_dim} it reads")
        key_basis, query_basis = bases if bases is not None else (None, None)
        self.register_buffer("key_basis", None if key_basis is None else key_basis.float().clone())
        self.register_buffer("query_basis", None if query_basis is None else query_basis.float().clone())
        if init is not None and key_basis is not None:
            init = rownorm(init.float() @ key_basis.float().to(init.device))
        start = _leading_directions(init, index_dim) if init is not None else _random(model_dim, index_dim)
        self.key_projection = nn.Parameter(start.clone())
        self.query_projection = nn.Parameter(start.clone())
        self.log_tau = nn.Parameter(torch.tensor(math.log(Space.DEFAULT_TAU)))

    @property
    def tau(self) -> float:
        """The temperature the loss reads the index score at"""
        return float(self.log_tau.detach().exp())

    @property
    def model_dim(self) -> int:
        """Dims of `x`, which the projections read"""
        return int(self.key_projection.shape[0])

    @property
    def index_dim(self) -> int:
        """Dims of the index keys and index queries the projections write"""
        return int(self.key_projection.shape[1])

    def set_tau(self, tau: float) -> None:
        """Pin the temperature.

        Args:
            tau: The temperature to start from.
        """
        with torch.no_grad():
            self.log_tau.fill_(math.log(tau))

    def served_projections(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the projections as a server reads them - the bases folded in, when there are any.

        Returns:
            The query projection and the key projection, each [model_dim, index_dim] fp32, carrying the gradient.
        """
        if self.key_basis is None or self.query_basis is None:
            return self.query_projection, self.key_projection
        return self.query_basis @ self.query_projection, self.key_basis @ self.key_projection

    def index_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Project what the key projection reads at every position into index keys.

        Args:
            x: The attention's input, or the key latents [T, model_dim], any float dtype.

        Returns:
            Index keys [T, index_dim] fp32 unit rows.
        """
        _, key_projection = self.served_projections()
        return rownorm(_ProjectChunked.apply(x, key_projection, Space.PROJECT_CHUNK))

    def index_queries(self, x: torch.Tensor) -> torch.Tensor:
        """Project what the query projection reads at some positions into index queries.

        Args:
            x: The attention's input, or the heads' queries summed, at the query positions [..., model_dim], any float dtype.

        Returns:
            Index queries [..., index_dim] fp32 unit rows.
        """
        query_projection, _ = self.served_projections()
        return project(x, query_projection)


class QSAIndexer(nn.Module):
    """One layer's QSA indexer - the fused query and key projection of `x`, the two norms, and the loss temperature.

    The weights are laid out as Qwen's server loads them - `index_qk_proj` `[(H + 1) head_dim, model_dim]`, the
    query heads' rows first, and the Gemma norms' `q_layernorm` and `k_layernorm` - so an export is a rename.
    The keys of a block are pooled before their projection, which is the same mean since the projection is linear,
    so a document holds its blocks' mean `x` rather than every position's.

    The read is invariant to `tau` - it ranks blocks by the score - and a scale on the scores is one the query norm's
    gain already spans, so `tau` changes nothing the indexer can express. It is the start the softmax is read at,
    calibrated before step one as the LSA map's is.
    """

    inverse_frequencies: torch.Tensor
    attention_scaling: torch.Tensor

    def __init__(
        self,
        model_dim: int,
        config: QSAConfig,
        inverse_frequencies: torch.Tensor,
        attention_scaling: float,
        eps: float = Space.QSA_NORM_EPS,
    ) -> None:
        """Build the indexer at a random start.

        Args:
            model_dim: Dims of `x`.
            config: The indexer's shape.
            inverse_frequencies: The model's rotary frequencies [rotary_dim / 2] fp32.
            attention_scaling: What the model scales its rotary tables by.
            eps: The norms' epsilon, the model's own.

        Raises:
            ValueError: The frequencies do not rotate `config.rotary_dim` dims.
        """
        super().__init__()
        if 2 * inverse_frequencies.shape[0] != config.rotary_dim:
            raise ValueError(
                f"{inverse_frequencies.shape[0]} frequencies rotate {2 * inverse_frequencies.shape[0]} dims, and the "
                f"indexer rotates {config.rotary_dim}"
            )
        self.config = config
        self.eps = eps
        rows = (config.query_head_count + 1) * config.head_dim
        self.index_qk_proj = nn.Parameter(torch.randn(rows, model_dim) / math.sqrt(model_dim))
        self.q_layernorm = nn.Parameter(torch.zeros(config.head_dim))
        self.k_layernorm = nn.Parameter(torch.zeros(config.head_dim))
        self.log_tau = nn.Parameter(torch.tensor(math.log(Space.DEFAULT_TAU)))
        self.register_buffer("inverse_frequencies", inverse_frequencies.float().clone())
        self.register_buffer("attention_scaling", torch.tensor(float(attention_scaling)))

    @property
    def tau(self) -> float:
        """The temperature the loss reads the block score at"""
        return float(self.log_tau.detach().exp())

    @property
    def model_dim(self) -> int:
        """Dims of `x`, which the projection reads"""
        return int(self.index_qk_proj.shape[1])

    @property
    def index_dim(self) -> int:
        """Dims of one index head"""
        return self.config.head_dim

    def set_tau(self, tau: float) -> None:
        """Pin the temperature.

        Args:
            tau: The temperature to start from.
        """
        with torch.no_grad():
            self.log_tau.fill_(math.log(tau))

    def index_keys(self, block_x: torch.Tensor) -> torch.Tensor:
        """Project, normalize and rotate every block's mean `x` into its index key, rotated at the block's first position.

        Args:
            block_x: Each closed block's mean key input [B, model_dim], any float dtype - `x`, or the key heads.

        Returns:
            Index keys [B, head_dim] fp32.
        """
        key_rows = self.index_qk_proj[self.config.query_head_count * self.config.head_dim :]
        keys = _ProjectChunked.apply(block_x, key_rows.t(), Space.PROJECT_CHUNK)
        starts = torch.arange(block_x.shape[0], device=block_x.device) * self.config.block_size
        normed = qsa.gemma_norm(keys, self.k_layernorm, self.eps)
        return qsa.rotate(normed, starts, self.inverse_frequencies, float(self.attention_scaling))

    def index_queries(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Project, normalize and rotate `x` at some positions into each head's index query.

        Args:
            x: The attention input at the query positions [R, model_dim], any float dtype.
            positions: Their positions [R] int64.

        Returns:
            Index queries [R, H, head_dim] fp32.
        """
        query_rows = self.index_qk_proj[: self.config.query_head_count * self.config.head_dim]
        queries = (x.float() @ query_rows.t()).view(
            len(x), self.config.query_head_count, self.config.head_dim
        )
        normed = qsa.gemma_norm(queries, self.q_layernorm, self.eps)
        return qsa.rotate(normed, positions, self.inverse_frequencies, float(self.attention_scaling))


# What a run trains a layer of - an LSA map, or Qwen's own indexer as the baseline
Indexer = IndexMap | QSAIndexer


def _leading_directions(x: torch.Tensor, index_dim: int) -> torch.Tensor:
    """Return the leading right singular directions of centred `x`.

    A truncation onto these is the best linear map of that dim in squared error, so training starts from
    the best untrained space rather than from noise.

    Args:
        x: Rows of the attention's input [n, model_dim], floating.
        index_dim: How many directions to keep.

    Returns:
        The projection [model_dim, index_dim] fp32.
    """
    rows = x.float()
    _, _, directions = torch.linalg.svd(rows - rows.mean(0), full_matrices=False)
    return directions[:index_dim].t().contiguous()


def _random(model_dim: int, index_dim: int) -> torch.Tensor:
    """Return a random projection, scaled so its output rows are order one.

    Args:
        model_dim: Dims of `x`, the space read.
        index_dim: Dims of the space written.

    Returns:
        The projection [model_dim, index_dim] fp32.
    """
    return torch.randn(model_dim, index_dim) / math.sqrt(model_dim)


def init_rows(x: torch.Tensor, rows: int = Space.INIT_ROWS) -> torch.Tensor:
    """Sample rows of `x` spread evenly across a document, for the PCA init.

    Args:
        x: A document's attention input [T, model_dim].
        rows: How many rows to take.

    Returns:
        Sampled rows [min(rows, T), model_dim], in position order.
    """
    stride = max(1, x.shape[0] // rows)
    return x[::stride][:rows]
