"""The projections a server reads index keys and index queries with - one pair per sparse layer.

The file is the flat torch file `lsa distill export` writes, mapping `<layer>.W_Q_idx` and `<layer>.W_K_idx` to
`[input_dim, index_dim]` tensors. The input dim is the indexer's input's: `model_dim` for the attention input `x`,
and the key's dim - GLM's 512-dim latent, `kv_lora_rank` - for the key and the heads' queries, which is what the
maps distilled before the switch to `x` read. A grouped-query model's key is every key head's position-free dims
side by side, 768 for Qwen3.8-27B's four heads of 192, and its indexed layers are its full-attention ones. Such a file is read as it is, with no rewrite. There is no index
without one: a server refuses a file whose shapes are not the input's dim by the design's `index_dim`, or that
lacks a layer that runs an indexer. A file may hold more layers than the model indexes - the full model's map on a
cut of it - and only the indexed ones are read.
"""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.sglang.settings.design import KEY_AND_QUERY

if TYPE_CHECKING:
    from lsa.sglang.index.layer_index import KeySlot

QUERY: str = "W_Q_idx"
KEY: str = "W_K_idx"

# The layer type of a hybrid model's attention layers that carry an indexer
SPARSE_LAYER_TYPE: str = "deepseek_sparse_attention"


def indexer_layers(config: Any, skips_top_k: Callable[[Any, int], bool]) -> list[int]:
    """Return the layers of a model that run an indexer of their own.

    Args:
        config: The model's text config.
        skips_top_k: Whether a layer reuses the top-k before it - the engine's own answer.

    Returns:
        The sparse attention layers that do not reuse the layer before's top-k, in order.
    """
    layer_types = getattr(config, "layer_types", None)
    return [
        layer
        for layer in range(config.num_hidden_layers)
        if (layer_types is None or layer_types[layer] == SPARSE_LAYER_TYPE) and not skips_top_k(config, layer)
    ]


def indexed_layers(config: Any) -> list[int]:
    """Return the layers the design indexes on a model - GLM's that run an indexer, a grouped-query model's quadratic ones.

    Args:
        config: The model's text config, inside a server.

    Returns:
        The indexed layers, in order.
    """
    from lsa.sglang.models.qwen3_5 import index_space

    if index_space.is_grouped_query(config):
        return list(index_space.geometry_of(config).layers)
    from sglang.srt.configs.model_config import dsa_layer_skips_topk

    return indexer_layers(config, dsa_layer_skips_topk)


def input_dim(config: Any, indexer_input: str) -> int:
    """Return the dims the projections read for an indexer input.

    Args:
        config: The model's text config.
        indexer_input: The design's `input` - `attention_input` or `key_and_query`.

    Returns:
        `hidden_size`, the model dim, for the attention input, and the key's dim for the key and the heads' queries -
        `kv_lora_rank` for GLM's latent, and every key head's position-free dims for a grouped-query model.
    """
    from lsa.sglang.models.qwen3_5 import index_space

    if indexer_input != KEY_AND_QUERY:
        return int(config.hidden_size)
    if index_space.is_grouped_query(config):
        return index_space.geometry_of(config).key_dim
    return int(config.kv_lora_rank)


def key_slot(config: Any) -> "KeySlot":
    """Return where the key and query input's key sits in one of a model's KV pool slots.

    Args:
        config: The model's text config.

    Returns:
        A grouped-query model's every head past its rotated prefix, or GLM's latent before its rope dims.
    """
    from lsa.sglang.index.layer_index import KeySlot
    from lsa.sglang.models.qwen3_5 import index_space

    if index_space.is_grouped_query(config):
        return index_space.geometry_of(config).key_slot
    return KeySlot.latent(int(config.kv_lora_rank), int(getattr(config, "qk_rope_head_dim", 0)))


def query_rows(config: Any, attention_tp_size: int) -> int:
    """Return the rows a token's key and query input holds on a rank - its heads, or one for a grouped-query model.

    Args:
        config: The model's text config.
        attention_tp_size: Ranks the attention's heads are split over.
    """
    from lsa.sglang.models.qwen3_5 import index_space

    if index_space.is_grouped_query(config):
        return 1
    return max(1, int(config.num_attention_heads) // attention_tp_size)


def read(
    path: str | Path, input_dim: int, index_dim: int, layers: Collection[int]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Read the indexed layers' projections, refusing a file that is not the model's.

    Args:
        path: The torch file `lsa distill export` wrote.
        input_dim: Dims of the indexer's input - `model_dim` for `x`, the key's dim for the key and query.
        index_dim: The design's index dim - the dims the projections write.
        layers: The layers that run an indexer, which the file must hold - any other layer it holds is left out.

    Returns:
        Each indexed layer's query projection and key projection, both [input_dim, index_dim] fp32 on the host.

    Raises:
        ValueError: The file is not a flat dict of `<layer>.W_Q_idx` and `<layer>.W_K_idx` tensors.
        ValueError: A projection is not [input_dim, index_dim].
        ValueError: The file lacks a layer that runs an indexer.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not blob:
        raise ValueError(f"{path} is not a projection file - it holds no '<layer>.W_Q_idx' tensors")
    found: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in blob.items():
        layer, _, kind = str(name).partition(".")
        if not layer.isdigit() or kind not in (QUERY, KEY) or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"{path} holds {name!r} - a projection file holds '<layer>.{QUERY}' and '<layer>.{KEY}'"
            )
        if tuple(tensor.shape) != (input_dim, index_dim):
            raise ValueError(
                f"{path} {name} is {tuple(tensor.shape)} - expected [input_dim, index_dim] = "
                f"[{input_dim}, {index_dim}] for the design's indexer input"
            )
        found.setdefault(int(layer), {})[kind] = tensor.float()
    lonely = sorted(layer for layer, pair in found.items() if len(pair) != 2)
    if lonely:
        raise ValueError(f"{path} holds only one of {QUERY} and {KEY} for layers {lonely}")
    missing = sorted(set(layers) - set(found))
    if missing:
        raise ValueError(
            f"{path} holds layers {sorted(found)}, and the layers that run an indexer are {sorted(layers)} - "
            f"missing {missing}"
        )
    return {layer: (found[layer][QUERY], found[layer][KEY]) for layer in sorted(set(layers))}
