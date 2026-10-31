"""The whole-model half of an adapter - load the checkpoint, find the layers LSA replaces, measure their rotation.

The default works on a standard HF decoder and is what a supported off-the-shelf checkpoint runs on, with no
code and no entry in the registry.
A model that needs more overrides what it needs and inherits the rest.

The split is deliberate. This class knows about checkpoints and layer stacks; [layer](layer.py) knows about
one attention module. A new model usually touches only one of the two.
"""

from typing import Any

import torch

from lsa.config import text_config
from lsa.core import RotaryLayout
from lsa.evaluation.adapter.layer import SparseAttention
from lsa.evaluation.adapter.rotary import detect_rotary, resolve_rotary
from lsa.evaluation.config import ModelConfig
from lsa.evaluation.constants import Layer


def torch_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name to the torch dtype it means.

    Args:
        name: A torch dtype name, such as `bfloat16`.

    Returns:
        The dtype.

    Raises:
        ValueError: The name is not a torch dtype.
    """
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype must name a torch dtype - got {name!r}")
    return dtype


class ModelAdapter:
    """How one architecture family loads, where its full-attention layers are, and what wraps them."""

    # The per-layer module this adapter swaps in
    attention_class: type[SparseAttention] = SparseAttention

    # What this family calls the layers to swap, when the config does not name them
    layer_types: tuple[str, ...] = Layer.DEFAULT_TYPES

    def model_class(self) -> Any:
        """Return the transformers auto class this family loads through.

        A text-only decoder registers its generation head under `AutoModelForCausalLM`, which is what the
        default answers. A multimodal checkpoint registers the same head somewhere else and `AutoModelForCausalLM`
        does not know its config at all, so that family overrides this.

        Returns:
            The auto class to call `from_pretrained` on.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def load(
        self,
        model_config: ModelConfig,
        device: str,
        attention: str = "eager",
        span: tuple[int, int] | None = None,
    ) -> tuple[Any, Any]:
        """Load the checkpoint and its tokenizer, or one span of its decoder stack.

        Args:
            model_config: Which checkpoint, and in what dtype.
            device: Where the weights land, as a transformers device map - a device string for one card,
                or `auto` to spread a checkpoint no single card holds across every visible one. GLM-5.3-Flash
                is 306 GiB with its experts in fp8 and takes the second.
            attention: The native attention implementation. Eager is what a swapped model wants, since the
                swapped layers never reach it; a teacher extraction runs the model's own attention over a
                whole document and asks for `sdpa`.
            span: Decoder layers `[start, stop)` to load, the rest of the model left out before a weight is
                read - so one span of a checkpoint no card holds fits on one. None loads the whole model.

        Returns:
            model: The loaded model in eval mode.
            tokenizer: Its tokenizer.
        """
        from transformers import AutoTokenizer

        model_class = self.model_class()
        if span is not None:
            model_class = self.span_class(model_class, model_config, span)
        model = model_class.from_pretrained(
            model_config.checkpoint,
            config=self.pretrained_config(model_config),
            dtype=torch_dtype(model_config.dtype),
            attn_implementation=attention,
            device_map=device,
        )
        model.eval()
        return model, AutoTokenizer.from_pretrained(model_config.checkpoint)

    def pretrained_config(self, model_config: ModelConfig) -> Any:
        """Read the checkpoint's config with the rotation the run asks for in place of its own.

        Args:
            model_config: Which checkpoint, and the `rope_parameters` that replace its own, if any.

        Returns:
            The config to build the model from.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_config.checkpoint)
        if model_config.rope_parameters:
            text_config(config).rope_parameters = dict(model_config.rope_parameters)
        return config

    def span_class(self, auto_class: Any, model_config: ModelConfig, span: tuple[int, int]) -> Any:
        """Return the checkpoint's model class, loading only one span of its decoder stack.

        `from_pretrained` builds the model on the meta device, lets the quantizer swap its modules, and calls
        `maybe_distribute_model` before it reads a single weight. Pruning there leaves only the span's
        parameters for the loader to fill, and the quantizer and every weight conversion still run as they
        would for the whole model.

        Args:
            auto_class: The auto class the checkpoint loads through.
            model_config: Which checkpoint.
            span: Decoder layers `[start, stop)` to keep.

        Returns:
            A subclass of the checkpoint's own model class.
        """
        from transformers import AutoConfig

        model_class = auto_class._model_mapping[type(AutoConfig.from_pretrained(model_config.checkpoint))]
        adapter = self

        def keep_span(cls: Any, model: Any, distributed_config: Any, device_mesh: Any) -> Any:
            """Prune the built model to the span, in the hook transformers calls before it reads a weight."""
            adapter.prune_to_span(model, span)
            return model

        return type(
            f"{model_class.__name__}Span", (model_class,), {"maybe_distribute_model": classmethod(keep_span)}
        )

    def prune_to_span(self, model: Any, span: tuple[int, int]) -> None:
        """Replace everything outside one span of the decoder stack with a module that holds nothing.

        The embedding stays with the span that starts the stack. The final norm, the head and a vision tower
        go from every span, since an extraction reads the stack's layer inputs and never its output.

        Args:
            model: The model, built but not yet loaded, modified in place.
            span: Decoder layers `[start, stop)` to keep.

        Raises:
            ValueError: The span is empty or runs past the stack.
        """
        text_model = self.text_model(model)
        start, stop = span
        if not 0 <= start < stop <= len(text_model.layers):
            raise ValueError(
                f"a span must be non-empty and inside {len(text_model.layers)} layers - got {span}"
            )

        for index in range(len(text_model.layers)):
            if not start <= index < stop:
                text_model.layers[index] = torch.nn.Identity()
        if start > 0:
            text_model.embed_tokens = torch.nn.Identity()
        text_model.norm = torch.nn.Identity()
        if hasattr(model, "lm_head"):
            model.lm_head = torch.nn.Identity()
            # Nothing is left to tie the head to
            model.all_tied_weights_keys = {}
        inner = self.inner_model(model)
        for name in Layer.VISION_TOWERS:
            if hasattr(inner, name):
                setattr(inner, name, torch.nn.Identity())

    def text_config(self, model_config: ModelConfig) -> Any:
        """Read the decoder stack's config without loading a weight.

        Args:
            model_config: Which checkpoint.

        Returns:
            The config the decoder stack is built from, past whatever a multimodal checkpoint wraps it in.
        """
        return text_config(self.pretrained_config(model_config))

    def stack_input(self, text_model: Any, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids into what the first decoder layer reads, for a stack run a span at a time.

        Args:
            text_model: The decoder stack.
            ids: Token ids [1, S] int64.

        Raises:
            NotImplementedError: Always - this model's stack has not been taught to run a span at a time.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot run its stack a span at a time")

    def forward_span(
        self,
        text_model: Any,
        span: tuple[int, int],
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: Any,
    ) -> torch.Tensor:
        """Run decoder layers `[start, stop)` over one chunk, the way the stack's own forward runs them.

        Args:
            text_model: The decoder stack.
            span: Decoder layers `[start, stop)`.
            hidden: What layer `start` reads, for this chunk.
            positions: Absolute positions of the chunk [1, S] int64.
            cache: The transformers cache the chunks share, or None for a document in one pass.

        Raises:
            NotImplementedError: Always - this model's stack has not been taught to run a span at a time.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot run its stack a span at a time")

    def text_model(self, model: Any) -> Any:
        """Return the decoder stack, past whatever a multimodal checkpoint wraps it in.

        Args:
            model: The loaded model.

        Returns:
            The module that carries `layers` and `rotary_emb`.
        """
        inner = self.inner_model(model)
        return getattr(inner, "language_model", inner)

    def inner_model(self, model: Any) -> Any:
        """Return the module under a task head, which holds the embeddings, the stack and any vision tower.

        Args:
            model: The loaded model.

        Returns:
            The inner module, or the model itself when it has no head to look under.
        """
        # `base_model_prefix` is what transformers itself uses to find the stack under a task head -
        # `model` for a modern decoder, `gpt_neox` or `transformer` for an older one
        return getattr(model, getattr(model, "base_model_prefix", "model"), model)

    def prepare_long_prefill(self, model: Any) -> None:
        """Make the native forward fit a document prefilled a chunk at a time into a cache.

        The default does nothing - a model whose attention runs dense under a causal kernel already fits.

        Args:
            model: The loaded model, which an override modifies in place.
        """

    def native_selector(self, attention: Any) -> Any | None:
        """Return the module a layer's own sparse attention selects its keys with, if it has one.

        The default has none - a dense layer reads every key. An override returns a module whose forward
        gives the selected key ids `[B, S, W]`, `-1` where nothing was selected.

        Args:
            attention: The layer's attention module.

        Returns:
            The selector, or None.
        """
        return None

    def full_attention_layers(self, text_config: Any, model_config: ModelConfig) -> list[int]:
        """Return the indices of the layers the indexer should replace.

        A model that declares no `layer_types` is quadratic throughout and every layer qualifies.
        A hybrid one names its layers, but there is no standard vocabulary - Qwen3.5 says `full_attention`,
        Qwen3.8-Flash-Next says `qwen_sparse_attention` - so an adapter sets `layer_types` for the family it
        drives, `model_config.layer_types` overrides that, and a stack that matches nothing raises with the names it
        does declare.
        The config is all this reads, so an extraction knows its layers before it loads a weight.

        Args:
            text_config: The decoder stack's config.
            model_config: The model model_config, which may name the layer types to swap.

        Returns:
            Layer indices, in order.

        Raises:
            ValueError: The stack declares layer types and none of them were asked for.
        """
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            return list(range(text_config.num_hidden_layers))

        wanted = set(model_config.layer_types or self.layer_types)
        layers = [index for index, kind in enumerate(layer_types) if kind in wanted]
        if not layers:
            declared = ", ".join(sorted(set(layer_types)))
            raise ValueError(
                f"{model_config.name} has no layers of type {sorted(wanted)} - it declares {declared} - "
                "name the ones to swap in model.layer_types"
            )
        return layers

    def rotary_embedding(self, model: Any) -> Any:
        """Return the module that builds the model's `(cos, sin)` tables.

        An adapter for a model that does not rotate at all returns None instead, and the layout is `none`
        with nothing to measure. The default raises rather than reading a missing `rotary_emb` as NoPE,
        because guessing that wrong indexes on rotated dims and only shows up as a bad score.

        Args:
            model: The loaded model.

        Returns:
            The rotary embedding module, or None for a model with no rotation.

        Raises:
            ValueError: The decoder stack carries no `rotary_emb`, so the layout cannot be measured.
        """
        text_model = self.text_model(model)
        embedding = getattr(text_model, "rotary_emb", None)
        if embedding is None:
            raise ValueError(
                f"{type(text_model).__name__} carries no rotary_emb, so the rotary layout cannot be "
                "measured - override `rotary_embedding` for this model"
            )
        return embedding

    def rotary_layout(self, model: Any, model_config: ModelConfig, layers: list[int]) -> RotaryLayout:
        """Measure where this model's rotation sits, and check whatever the config pinned against it.

        Args:
            model: The loaded model.
            model_config: The model model_config, which may pin the layout or leave it to the measurement.
            layers: The layers the indexer covers - the first is what gets probed.

        Returns:
            The rotary layout the index space is taken against.
        """
        rotary_embedding = self.rotary_embedding(model)
        if rotary_embedding is None:
            # A NoPE model has no tables and nothing to measure - the whole head vector is position-free
            detected = RotaryLayout()
        else:
            detected = detect_rotary(
                self.attention_class.rotate,
                self.text_model(model).layers[layers[0]].self_attn,
                rotary_embedding,
                model_config.checkpoint,
            )
        return resolve_rotary(model_config.rotary_end, model_config.rotary_dim, detected, model_config.name)
