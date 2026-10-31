"""Hooks that keep what each layer's attention is handed while the model reads a document, and what its own selector picked."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data.stage import (
    Boundary,
    BoundaryWriter,
)


def capture_layer_inputs(
    text_model: Any, layers: list[int], ids: torch.Tensor, prefill: int = 0
) -> dict[int, tuple[Any, Any]]:
    """Run the model over a document and keep what each wanted layer's attention was handed.

    In one pass by default. With `prefill` the document goes through that many tokens at a time into a cache,
    the way a server prefills it, and each layer's input waits in host memory until the last chunk is in -
    one layer's input over a 1M document is 8 GiB, and eleven of them do not wait on the cards.

    Args:
        text_model: The decoder stack.
        layers: Layer indices to capture.
        ids: Token ids [1, T] int64.
        prefill: Tokens per pass, or 0 for the whole document in one.

    Returns:
        Layer index to `(hidden_states, position_embeddings)` over the whole document - on the layer's device
        after one pass, on the host after several.
    """
    chunked = 0 < prefill < ids.shape[1]
    chunks, hooks = _capture(text_model, layers, chunked)
    try:
        read_through(text_model, ids, prefill)
    finally:
        for hook in hooks:
            hook.remove()
    return {index: _joined(pieces) for index, pieces in chunks.items()}


@torch.no_grad()
def read_through(
    text_model: Any,
    ids: torch.Tensor,
    prefill: int = 0,
    each: Callable[[int, torch.Tensor], None] | None = None,
) -> None:
    """Run the decoder stack over a document for what its hooks see, in one pass or a chunk at a time into a cache.

    Args:
        text_model: The decoder stack.
        ids: Token ids [1, T] int64.
        prefill: Tokens per pass, or 0 for the whole document in one.
        each: Called with each pass's first position and its final hidden states [1, C, model_dim], if given.
    """
    if not 0 < prefill < ids.shape[1]:
        output = text_model(input_ids=ids, use_cache=False)
        if each is not None:
            each(0, output.last_hidden_state)
        return
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=text_model.config)
    for start in range(0, ids.shape[1], prefill):
        stop = min(start + prefill, ids.shape[1])
        positions = torch.arange(start, stop, device=ids.device)[None]
        output = text_model(
            input_ids=ids[:, start:stop],
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        )
        if each is not None:
            each(start, output.last_hidden_state)
        del output
    del cache


def capture_span_inputs(
    model_adapter: Any,
    text_model: Any,
    span: tuple[int, int],
    layers: list[int],
    ids: torch.Tensor,
    prefill: int,
    source: Boundary | None,
    sink: BoundaryWriter | None,
) -> dict[int, tuple[Any, Any]]:
    """Run one span of the stack over a document and keep what each wanted layer's attention was handed.

    The span prefills the way `capture_layer_inputs` prefills the whole stack, a chunk at a time into a
    cache when the document is longer than `prefill`. It reads the token ids when it starts the stack and
    the streams its input cut holds otherwise, and hands each chunk's output streams to the cut after it.

    Args:
        model_adapter: The adapter that runs the stack a span at a time.
        text_model: The decoder stack, loaded for this span alone.
        span: Decoder layers `[start, stop)`.
        layers: The captured layers inside the span.
        ids: Token ids [1, T] int64, on the span's device.
        prefill: Tokens per pass, or 0 for the whole document in one.
        source: The input cut's streams, or None for the span that starts the stack.
        sink: Where the output cut's streams go, or None for the span that ends the extraction.

    Returns:
        Layer index to `(hidden_states, position_embeddings)` over the whole document - on the card after
        one pass, on the host after several.
    """
    length = ids.shape[1]
    chunked = 0 < prefill < length
    step = prefill if chunked else length
    chunks, hooks = _capture(text_model, layers, chunked)
    try:
        with torch.no_grad():
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache(config=text_model.config) if chunked else None
            for start in range(0, length, step):
                stop = min(start + step, length)
                if source is None:
                    hidden = model_adapter.stack_input(text_model, ids[:, start:stop])
                else:
                    hidden = source.read(start, stop).to(ids.device)
                positions = torch.arange(start, stop, device=ids.device)[None]
                hidden = model_adapter.forward_span(text_model, span, hidden, positions, cache)
                if sink is not None:
                    sink.append(hidden)
            del cache
    finally:
        for hook in hooks:
            hook.remove()
    return {index: _joined(pieces) for index, pieces in chunks.items()}


def _capture(
    text_model: Any, layers: list[int], to_host: bool
) -> tuple[dict[int, list[tuple[Any, Any]]], list[Any]]:
    """Hook every wanted layer's attention to keep what it is handed, a chunk at a time.

    Args:
        text_model: The decoder stack.
        layers: Layer indices to capture.
        to_host: Move each chunk's inputs to host memory as they pass, for a document prefilled in chunks.

    Returns:
        chunks: Layer index to the `(hidden_states, position_embeddings)` of every chunk so far.
        hooks: The hook handles, to remove once the document is through.
    """
    chunks: dict[int, list[tuple[Any, Any]]] = {index: [] for index in layers}

    def keep(index: int, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Keep what one layer's attention was handed for this chunk, on the host when asked.

        Args:
            index: The layer.
            _module: The attention module, unused.
            args: Its positional arguments, the hidden states first when they are not a keyword.
            kwargs: Its keyword arguments, which carry the rotary tables when the model has them.
        """
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        tables = kwargs.get("position_embeddings")
        if to_host:
            hidden = hidden.to(Cache.HOST)
            tables = None if tables is None else tuple(table.to(Cache.HOST) for table in tables)
        chunks[index].append((hidden, tables))

    hooks = [
        text_model.layers[index].self_attn.register_forward_pre_hook(partial(keep, index), with_kwargs=True)
        for index in layers
    ]
    return chunks, hooks


def _joined(pieces: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    """Join one layer's inputs from every pass along the sequence.

    Args:
        pieces: `(hidden_states, position_embeddings)` per pass, in order.

    Returns:
        The same pair over the whole document.
    """
    if len(pieces) == 1:
        return pieces[0]
    hidden = torch.cat([hidden for hidden, _ in pieces], dim=1)
    if pieces[0][1] is None:
        return hidden, None
    tables = tuple(
        torch.cat([tables[part] for _, tables in pieces], dim=1) for part in range(len(pieces[0][1]))
    )
    return hidden, tables


class NativeSelection:
    """What each captured layer's own selector picked at the sampled positions, gathered pass by pass.

    A selector runs once a pass over that pass's positions, so the rows a pass covers are the sampled
    positions inside it, and the passes arrive in order.
    """

    def __init__(self, positions: torch.Tensor) -> None:
        """Start empty.

        Args:
            positions: Absolute query positions [Q] int64.
        """
        self.positions = positions.cpu()
        self.covered: dict[int, int] = {}
        self.picked: dict[int, torch.Tensor] = {}

    def record(self, layer: int, selected: torch.Tensor) -> None:
        """Keep the rows of one pass that sit at sampled positions.

        Args:
            layer: Which layer's selector ran.
            selected: Selected key ids [1, S, W] int, over the pass's S positions, `-1` where nothing was.

        Raises:
            ValueError: A pass selects a different number of keys than the passes before it.
        """
        start = self.covered.get(layer, 0)
        stop = start + selected.shape[1]
        self.covered[layer] = stop
        if layer not in self.picked:
            self.picked[layer] = torch.full((len(self.positions), selected.shape[-1]), -1, dtype=torch.int32)
        if self.picked[layer].shape[1] != selected.shape[-1]:
            raise ValueError(
                f"layer {layer} selected {selected.shape[-1]} keys a query where it had selected "
                f"{self.picked[layer].shape[1]}"
            )
        rows = ((self.positions >= start) & (self.positions < stop)).nonzero().squeeze(1)
        offsets = (self.positions[rows] - start).to(selected.device)
        self.picked[layer][rows] = selected[0].index_select(0, offsets).int().cpu()

    def of(self, layer: int, length: int) -> torch.Tensor:
        """Return one layer's selection at every sampled position.

        Args:
            layer: Which layer.
            length: Tokens in the document, which the passes must have covered.

        Returns:
            Selected key ids [Q, W] int32, `-1` where nothing was.

        Raises:
            ValueError: The layer's selector did not run over the whole document.
        """
        if self.covered.get(layer, 0) != length:
            raise ValueError(
                f"layer {layer}'s selector ran over {self.covered.get(layer, 0)} of {length} positions"
            )
        return self.picked[layer]


def hook_native_selection(
    model_adapter: Any, text_model: Any, layers: list[int], positions: torch.Tensor
) -> tuple[NativeSelection, list[Any]]:
    """Hook every wanted layer's own selector to keep what it picks at the sampled positions.

    Args:
        model_adapter: The adapter that says which module a layer selects with.
        text_model: The decoder stack.
        layers: Layer indices to keep.
        positions: Absolute query positions [Q] int64.

    Returns:
        selection: What the selectors pick, filled as the document goes through.
        hooks: The hook handles, to remove once the document is through.

    Raises:
        ValueError: A layer has no selector of its own.
    """
    selection = NativeSelection(positions)

    def keep(index: int, _module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        """Keep what one layer's selector picked in this pass.

        Args:
            index: The layer.
            _module: The selector, unused.
            _args: Its arguments, unused.
            output: Selected key ids [1, S, W] int, `-1` where nothing was.
        """
        selection.record(index, output)

    hooks: list[Any] = []
    for index in layers:
        selector = model_adapter.native_selector(text_model.layers[index].self_attn)
        if selector is None:
            for hook in hooks:
                hook.remove()
            raise ValueError(
                f"layer {index} has no sparse selector of its own, so there is nothing native to keep"
            )
        hooks.append(selector.register_forward_hook(partial(keep, index)))
    return selection, hooks


def input_device(model: Any) -> torch.device:
    """Return the device a forward's token ids have to land on.

    A checkpoint no single card holds is loaded across several, and the embedding is where a forward starts.

    Args:
        model: The loaded model.

    Returns:
        The embedding's device.
    """
    return model.get_input_embeddings().weight.device
