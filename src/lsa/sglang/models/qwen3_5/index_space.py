"""Qwen3.5's key and query input, in the layout a server hands it over - and the geometry it is read with.

The design's input for a grouped-query model is the position-free part of the logit: each key-value head's dims past
its rotated prefix, side by side, and each group's query heads summed into the slice of the key head they read. The
dot of the two is what every head's logit sums to on those dims, so one selection serves every head - what
distillation trains on, `regroup_partial` over `position_free` heads, read here off a server's tensors.

Two facts keep it this small. The rotation is a prefix of 64 dims in a head of 256, and the server's fused norm-and-rope
kernel passes the other 192 through untouched, so the query and key a backend is handed after the rotation are the
pre-rotation ones on those dims - and so is the key the KV pool caches. The index keys are therefore already in the
pool: a `KeySlot` reads them out of each slot's four heads, and nothing is paged beside it.

An index key concatenates all four key-value heads and an index query sums every query head, so a rank that holds some
of them holds part of each. Both maps are linear before the normalize, so a rank projects what it holds through the rows
of the map those dims are and the group sums the [index_dim] partials - the index key and the index query are then every
head's on every rank, and every rank selects the same lists. Tensor parallel 8 gives a rank 3 of a group's 6 query heads
and 1 of the 4 key-value heads, which 2 ranks replicate, so only the first of them adds its key to the sum. Decode
context parallel across the whole group gathers the query heads and keeps every key head instead, and a rank holding
every head needs no sum.

Nothing here imports sglang, so a CPU reads it.
"""

from dataclasses import dataclass
from typing import Any

import torch

from lsa.sglang.index.layer_index import KeyAndQuery, KeySlot
from lsa.sglang.settings import field_of

# What `layer_types` calls a layer that attends quadratically, and so carries a KV cache and an index
FULL_ATTENTION: str = "full_attention"

# Where a VL checkpoint keeps its text geometry
TEXT_CONFIG: str = "text_config"

# How often a quadratic layer occurs - all a server's text config says of where they are
INTERVAL: str = "full_attention_interval"


@dataclass(frozen=True)
class Geometry:
    """One checkpoint's index geometry, read off its config."""

    query_head_count: int
    kv_head_count: int
    head_dim: int
    rotary_dim: int
    # The decoder indices that attend quadratically - the ids a server presents as `layer_id`
    layers: tuple[int, ...]

    @property
    def key_slot(self) -> KeySlot:
        """Return where the key input sits in a KV pool slot - every head's dims past its rotated prefix."""
        return KeySlot(
            heads=self.kv_head_count,
            head_dim=self.head_dim,
            offset=self.rotary_dim,
            width=self.head_dim - self.rotary_dim,
        )

    @property
    def key_dim(self) -> int:
        """Dims of the key and query input, which the maps read."""
        return self.key_slot.key_dim


def text_config_of(config: Any) -> Any:
    """Return whichever config holds the text geometry - a server's `hf_text_config` already, a VL config's nested one.

    Args:
        config: A text config, or a checkpoint config that nests one.
    """
    if field_of(config, "head_dim") is None and field_of(config, TEXT_CONFIG) is not None:
        return field_of(config, TEXT_CONFIG)
    return config


def is_grouped_query(config: Any) -> bool:
    """Return whether a model is a grouped-query stack this plugin indexes - quadratic layers over per-head keys.

    GLM's sparse layers carry a key latent, `kv_lora_rank`, and are indexed through the checkpoint's own indexer
    instead.

    Args:
        config: The model's text config.
    """
    config = text_config_of(config)
    if field_of(config, "kv_lora_rank") is not None:
        return False
    layer_types = field_of(config, "layer_types") or ()
    return FULL_ATTENTION in layer_types or bool(field_of(config, INTERVAL))


def rotary_dim_of(config: Any) -> int:
    """Return how many leading dims of a head the rotation covers.

    sglang lifts `partial_rotary_factor` onto the text config, and a published config nests it under
    `rope_parameters` - either is read.

    Args:
        config: The text config.

    Returns:
        The rotated dims of a head.

    Raises:
        ValueError: The config declares no partial rotation, so nothing of a head is position-free.
    """
    factor = field_of(config, "partial_rotary_factor")
    if factor is None:
        factor = field_of(field_of(config, "rope_parameters") or {}, "partial_rotary_factor")
    if factor is None or not 0 <= float(factor) < 1:
        raise ValueError(
            f"this config's rotation covers every dim of a head (partial_rotary_factor {factor}) - nothing is "
            "position-free to index"
        )
    return int(int(field_of(config, "head_dim")) * float(factor))


def quadratic_layers(config: Any) -> tuple[int, ...]:
    """Return the decoder indices that attend quadratically, and so carry a KV cache and a map.

    A published config names every layer in `layer_types`. The text config sglang builds drops that list and keeps
    the interval, which is the rule its model code places the layers by - every `full_attention_interval`-th.

    Args:
        config: The text config.

    Returns:
        The quadratic decoder indices, ascending.

    Raises:
        ValueError: The config names no quadratic layer.
    """
    layer_types = list(field_of(config, "layer_types") or ())
    layers = tuple(index for index, kind in enumerate(layer_types) if kind == FULL_ATTENTION)
    if layers:
        return layers
    interval, depth = field_of(config, INTERVAL), field_of(config, "num_hidden_layers")
    if not interval or not depth:
        raise ValueError(
            f"this config names no {FULL_ATTENTION} layer among {len(layer_types)} and no {INTERVAL} - nothing to index"
        )
    return tuple(range(int(interval) - 1, int(depth), int(interval)))


def geometry_of(config: Any) -> Geometry:
    """Read the index geometry off a checkpoint's config.

    Args:
        config: The text config a server holds, or the checkpoint config that nests one.

    Returns:
        The geometry.
    """
    config = text_config_of(config)
    return Geometry(
        query_head_count=int(field_of(config, "num_attention_heads")),
        kv_head_count=int(field_of(config, "num_key_value_heads")),
        head_dim=int(field_of(config, "head_dim")),
        rotary_dim=rotary_dim_of(config),
        layers=quadratic_layers(config),
    )


@dataclass(frozen=True)
class HeadShard:
    """The heads a tensor-parallel rank holds, and the dims of the key and query input they are."""

    # The first key-value head this rank holds, and how many
    kv_first: int
    kv_count: int
    # Query heads this rank holds - whole groups of the key-value heads, or a share of one group
    query_count: int
    # Whether this rank's key heads are added to the group's sum - one of the ranks that replicate them is
    counts_key: bool

    def dims(self, geometry: Geometry) -> tuple[int, int]:
        """Return the key and query input's dims this rank's heads are, as [start, stop).

        Args:
            geometry: The checkpoint's geometry.
        """
        position_free = geometry.head_dim - geometry.rotary_dim
        return self.kv_first * position_free, (self.kv_first + self.kv_count) * position_free


def splits_key_heads(tp_size: int, decode_size: int = 1) -> bool:
    """Return whether the mesh gives a rank a share of the key-value heads - the model's split over the ranks decode context parallel does not span.

    Args:
        tp_size: The attention tensor-parallel size.
        decode_size: The decode context-parallel group's size.
    """
    return tp_size // decode_size > 1


def kv_heads_held(geometry: Geometry, tp_size: int) -> int:
    """Return the key-value heads a tensor-parallel rank holds - one, replicated, once there are more ranks than heads.

    Args:
        geometry: The checkpoint's geometry.
        tp_size: The attention tensor-parallel size.
    """
    return max(1, geometry.kv_head_count // tp_size)


def head_shard(geometry: Geometry, tp_size: int, tp_rank: int, decode_size: int = 1) -> HeadShard | None:
    """Return the heads a rank holds when the mesh splits the key-value heads, and refuse a mesh this plugin cannot split.

    Under tensor parallel the key-value heads are dealt to ranks in order, a head replicated across `tp_size / Hkv`
    ranks once there are more ranks than heads, and the query heads in order beside them - so a rank's query heads
    all read key heads the rank holds. Decode context parallel across the whole group leaves every rank every key head
    and gathers the query heads before the index reads them, which needs no split.

    Args:
        geometry: The checkpoint's geometry.
        tp_size: The attention tensor-parallel size.
        tp_rank: This rank's place in it.
        decode_size: The decode context-parallel group's size.

    Returns:
        The rank's heads, or None where it holds every key-value head and sums every query head itself.

    Raises:
        NotImplementedError: Decode context parallel spans some of the tensor-parallel group, or the group does not
            split the query heads evenly, or neither divides the key-value heads nor shares each across whole ranks.
    """
    if decode_size not in (1, tp_size):
        raise NotImplementedError(
            f"decode context parallel {decode_size} under tensor parallel {tp_size} splits the pool by slots and "
            "the key-value heads by ranks at once, and the index reads one or the other - serve Qwen under LSA "
            "with --dcp-size 1 or equal to --tp"
        )
    if not splits_key_heads(tp_size, decode_size):
        return None
    if geometry.query_head_count % tp_size != 0:
        raise NotImplementedError(
            f"tensor parallel {tp_size} splits {geometry.query_head_count} query heads unevenly, and an index query "
            "sums them across ranks that each hold the same count - serve Qwen under LSA at a divisor of it"
        )
    kv_count = geometry.kv_head_count
    if tp_size <= kv_count and kv_count % tp_size == 0:
        held = kv_count // tp_size
        return HeadShard(tp_rank * held, held, geometry.query_head_count // tp_size, True)
    replicas = tp_size // kv_count
    if tp_size > kv_count and tp_size % kv_count == 0:
        return HeadShard(
            tp_rank // replicas, 1, geometry.query_head_count // tp_size, tp_rank % replicas == 0
        )
    raise NotImplementedError(
        f"tensor parallel {tp_size} neither divides {kv_count} key-value heads nor holds each on a whole number of "
        "ranks, so a rank's key heads are not a head of the model's own split - serve Qwen under LSA at a divisor "
        f"or a multiple of {kv_count}"
    )


def key_and_query(
    q: torch.Tensor, k: torch.Tensor, geometry: Geometry, heads: HeadShard | None = None
) -> KeyAndQuery:
    """Make the key and query input of a forward's tokens from the query and key heads attention was handed.

    Query head `h` reads key head `h // G`, the grouping the checkpoint's weights are laid out in. The sum is in fp32,
    as distillation's `regroup_partial` sums.

    Args:
        q: Query heads, flat [T, Hq * dh] or [T, Hq, dh], rotated on the prefix alone.
        k: Key heads, flat [T, Hkv * dh] or [T, Hkv, dh], the same.
        geometry: The checkpoint's geometry.
        heads: The heads this rank holds when the mesh splits them - `q` and `k` then hold those alone - or None for
            every head.

    Returns:
        The keys [T, Hkv * (dh - rotary)] bf16 and the grouped queries [T, 1, Hkv * (dh - rotary)] fp32, over the
        heads this rank holds.
    """
    token_count = q.shape[0]
    kv_count = geometry.kv_head_count if heads is None else heads.kv_count
    query_count = geometry.query_head_count if heads is None else heads.query_count
    position_free = geometry.head_dim - geometry.rotary_dim
    # [T, Hkv, G, dh - rotary] - each group's query heads beside their key head's
    queries = q.reshape(token_count, kv_count, query_count // kv_count, geometry.head_dim)[
        ..., geometry.rotary_dim :
    ]
    grouped = queries.sum(2, dtype=torch.float32).reshape(token_count, 1, kv_count * position_free)
    keys = k.reshape(token_count, kv_count, geometry.head_dim)[..., geometry.rotary_dim :]
    return KeyAndQuery(keys.reshape(token_count, -1).to(torch.bfloat16), grouped)
