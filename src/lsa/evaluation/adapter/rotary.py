"""What a model actually rotates, measured rather than declared.

LSA clusters the position-free sub-term of the logit, which is the head dims RoPE leaves alone.
Which dims those are is the one per-model fact that is silently catastrophic to get wrong - reading a
prefix as a suffix indexes on the rotary half, and nothing downstream fails, it just scores noise.

So it is measured. Rotate a fixed random key at several positions and diff: the dims that move are the
rotated ones, and prefix, suffix, none and interleaved all fall out of that mask. No weights and no
forward pass are involved, only the model's own rotary tables and its own `apply_rotary_pos_emb`.

A config may still pin `rotary_end` and `rotary_dim`, and then the pinned values are checked against the
measurement instead of replacing it.
"""

from typing import Any, Protocol

import torch

from lsa import constants as const
from lsa.core import RotaryLayout

# Positions the probe rotates at, spread over six orders of magnitude.
# Position 0 is the identity and says nothing, and small positions alone are not enough: with a rope_theta
# of 1e7 the slowest dims turn by ~1e-7 radians at position 3 and read as unrotated
_PROBE_POSITIONS: tuple[int, ...] = (1, 16, 256, 4096, 65536, 1048576)

# How far a dim must move to count as rotated, against unit-scale random input.
# A dim the rotation does not touch is copied through, so its movement is exactly zero
_MOVED_ATOL: float = 1e-6


class Rotate(Protocol):
    """`SparseAttention.rotate` - the rotation one layer will apply."""

    def __call__(
        self, attention: Any, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the queries and keys of an attention module by the cosines and sines of their positions."""


def detect_rotary(rotate: Rotate, attention: Any, rotary_embedding: Any, model_name: str) -> RotaryLayout:
    """Measure where the rotation sits in this model's head vector.

    Args:
        rotate: The rotation the layer will apply, which is what gets measured.
        attention: A native attention module from a layer the indexer will wrap.
        rotary_embedding: The text model's rotary embedding, which builds `(cos, sin)`.
        model_name: Checkpoint name, quoted in the errors.

    Returns:
        The rotary layout the model applies.

    Raises:
        ValueError: The rotation covers the whole head, so there is no position-free sub-term.
        ValueError: The rotated dims are not a contiguous prefix or suffix.
    """
    head_dim = int(attention.head_dim)
    moved = rotated_dims(rotate, attention, rotary_embedding, head_dim)
    rotated_count = int(moved.sum())

    if rotated_count == 0:
        return RotaryLayout(const.Rotary.NONE, 0)
    if rotated_count == head_dim:
        raise ValueError(
            f"{model_name} rotates all {head_dim} of its {head_dim} head dims - "
            "LSA's training-free index space is the part of the logit RoPE leaves alone, "
            "so it needs a model with partial rotary or none at all"
        )
    if bool(moved[:rotated_count].all()):
        return RotaryLayout(const.Rotary.PREFIX, rotated_count)
    if bool(moved[head_dim - rotated_count :].all()):
        return RotaryLayout(const.Rotary.SUFFIX, rotated_count)

    raise ValueError(
        f"{model_name} rotates {rotated_count} head dims that are neither a prefix nor a suffix - "
        "the index space drops a contiguous slice, so an interleaved rotation needs its own `position_free`"
    )


def rotated_dims(rotate: Rotate, attention: Any, rotary_embedding: Any, head_dim: int) -> torch.Tensor:
    """Return which head dims the model's rotation moves.

    Args:
        rotate: The rotation the layer will apply.
        attention: A native attention module, read for its device and passed back to `rotate`.
        rotary_embedding: The text model's rotary embedding.
        head_dim: Dims of one head.

    Returns:
        A boolean mask [dh] that is True where the rotation changed the value.
    """
    device = next(attention.parameters()).device
    position_count = len(_PROBE_POSITIONS)

    # [1, T, dh] fp32 - only its device and dtype are read, and fp32 keeps the diff above the tolerance
    reference_input = torch.zeros(1, position_count, head_dim, device=device, dtype=torch.float32)
    cos, sin = rotary_tables(rotary_embedding, reference_input, device)

    # [1, 1, T, dh] fp32 unit-scale sample, seeded so the answer does not move between runs
    generator = torch.Generator(device=device).manual_seed(0)
    sample = torch.randn(
        1, 1, position_count, head_dim, device=device, dtype=torch.float32, generator=generator
    )

    rotated, _ = rotate(attention, sample, sample, cos, sin)

    # [dh] largest movement per dim, over every probe position
    movement = (rotated.float() - sample).abs().amax(dim=(0, 1, 2))

    return movement > _MOVED_ATOL


def rotary_tables(
    rotary_embedding: Any, reference_input: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the `(cos, sin)` tables the probe rotates against.

    Position ids are `[B, T]` on most models and `[3, B, T]` on an mrope one, which is not something a
    config reliably says, so both are tried. Every section holds the same positions here - the probe is
    text, and what is being measured is which dims move rather than by how much.

    Args:
        rotary_embedding: The text model's rotary embedding.
        reference_input: A tensor whose device and dtype the tables are built in.
        device: Device the position ids land on.

    Returns:
        The cosine and sine tables for the probe positions.
    """
    # [1, T] int64 positions, none of them 0 - the rotation there is the identity and would hide every dim
    position_ids = torch.tensor([_PROBE_POSITIONS], device=device)
    try:
        cos, sin = rotary_embedding(reference_input, position_ids)
    except (IndexError, RuntimeError):
        # [3, 1, T] - one row per mrope section
        cos, sin = rotary_embedding(reference_input, position_ids[None].expand(3, -1, -1))
    return cos, sin


def resolve_rotary(
    declared_end: str, declared_dim: int | None, detected: RotaryLayout, model_name: str
) -> RotaryLayout:
    """Reconcile what a config declared with what the model does.

    `auto` takes the measurement. A declared value is checked against it and a disagreement raises, because
    the wrong one produces an index over the rotary dims and a plausible-looking number.

    Args:
        declared_end: `rotary_end` from the config, or `auto`.
        declared_dim: `rotary_dim` from the config, or None to take the measurement.
        detected: What `detect_rotary` measured.
        model_name: Checkpoint name, quoted in the error.

    Returns:
        The rotary layout the indexer will use.

    Raises:
        ValueError: The config disagrees with the model.
    """
    if declared_end == const.Rotary.AUTO:
        return detected

    if declared_end == const.Rotary.NONE:
        dim = 0
    elif declared_dim is None:
        dim = detected.dim
    else:
        dim = declared_dim

    declared = RotaryLayout(declared_end, dim)
    if declared != detected:
        raise ValueError(
            f"the config declares rotary_end {declared.end!r} rotary_dim {declared.dim}, "
            f"but {model_name} rotates {detected.dim} head dims as a {detected.end!r} - "
            "drop both fields to take the measured layout"
        )
    return declared
