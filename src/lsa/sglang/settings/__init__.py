"""Where a server's settings come from - one block of the model config, which sglang already carries.

The plugin cannot add a flag to sglang's parser, and an environment variable reaches every process but says nothing about the model it was set for.
The model config reaches every process that builds one, so the settings ride in it: `--json-model-override-args '{"lsa": {...}}'`, or the same key in a checkpoint's own config.json.
The block is the design's flat body - `model.indexer` and `model.serving` side by side - with a `selector` key where the design is PISA's, and a `debug` sub-block for the checks and the dump.

No block is the checkpoint's own indexer, so the native baseline stays the absence of a setting and not a second build.
A block holding only `debug` is a native server with its checks, which is the one thing a debugging run of the checkpoint's own indexer needs.

Every reader takes the config it was built from and calls `of`, so a layer, a memory estimate and a backend read one parse of one block.
A nested checkpoint's block may sit on the top-level config or on its `text_config`, and `of` finds it on either.

| module | what it does |
|---|---|
| [design](design.py) | a design's settings - `model.indexer` and `model.serving` in the design's names, the one place a design is defined, and the flat body the block holds - and PISA's, which the same block names with a `selector` key |
| [debug](debug.py) | the `debug` sub-block - the index and PISA checks and the dump of what a layer reads |
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lsa.constants import Attention
from lsa.sglang.constants import Plugin
from lsa.sglang.settings.debug import DebugSettings
from lsa.sglang.settings.design import SELECTOR, DesignSettings, PisaConfig

# Prefill rows an indexer sizes its top-k for when the server leaves `chunked_prefill_size` unset
UNCHUNKED_PREFILL: int = 8192


@dataclass(frozen=True)
class PluginSettings:
    """What one block of the model config asks of a server - which selector indexes, its settings, and its debugging aids."""

    # The LSA design, or None where the block names another selector or none
    design: DesignSettings | None = None
    # PISA's pyramid, or None where the block names another selector or none
    pisa: PisaConfig | None = None
    debug: DebugSettings = field(default_factory=DebugSettings)

    @property
    def selector(self) -> str:
        """Which selector indexes - `Attention.LSA`, `Attention.PISA`, or `Attention.NATIVE` for the checkpoint's own."""
        if self.design is not None:
            return Attention.LSA
        if self.pisa is not None:
            return Attention.PISA
        return Attention.NATIVE

    @classmethod
    def parse(cls, block: object) -> "PluginSettings":
        """Build from the model config's block.

        Args:
            block: What the config holds under the plugin's key.

        Returns:
            The settings, validated - a block with no design keys is the checkpoint's own indexer.

        Raises:
            ValueError: The block is not an object, a key is neither a design's nor the selector's nor `debug`, or a value
                is refused.
        """
        if not isinstance(block, Mapping):
            raise ValueError(
                f"the model config's {Plugin.BLOCK!r} must be an object of settings - got {block!r}"
            )
        debug = DebugSettings.from_mapping(block.get(Plugin.DEBUG, {}))
        body = {name: value for name, value in block.items() if name != Plugin.DEBUG}
        if not body:
            return cls(debug=debug)
        if body.get(SELECTOR) == Attention.PISA:
            return cls(pisa=PisaConfig.parse(body), debug=debug)
        return cls(design=DesignSettings.parse(body), debug=debug)


def field_of(holder: object, name: str) -> Any:
    """Return a config's field, whether the config is an object or the dict a nested config can be.

    Args:
        holder: A model config, or a mapping of one.
        name: The field.

    Returns:
        The field's value, or None where the config has none.
    """
    if isinstance(holder, Mapping):
        return holder.get(name)
    return getattr(holder, name, None)


def block_of(config: object) -> Any:
    """Return what a model config holds under the plugin's key, wherever the override landed.

    Args:
        config: A model config - the top-level one, or a nested `text_config`.

    Returns:
        The block as it arrived, unchecked, or None where neither the config nor its `text_config` has one.
    """
    for holder in (config, field_of(config, "text_config")):
        found = field_of(holder, Plugin.BLOCK) if holder is not None else None
        if found is not None:
            return found
    return None


def of(config: object) -> PluginSettings:
    """Return the settings a server runs under, the one place any of them is read.

    Args:
        config: A model config - the top-level one, or a nested `text_config`.

    Returns:
        The block's settings, or the checkpoint's own indexer with no checks where the config has none.

    Raises:
        ValueError: The block is refused - see `PluginSettings.parse`.
    """
    block = block_of(config)
    if block is None:
        return PluginSettings()
    return PluginSettings.parse(block)


def design_of(config: object) -> DesignSettings:
    """Return the LSA design a server runs under.

    Args:
        config: A model config.

    Returns:
        The design the block holds.

    Raises:
        ValueError: The block is refused, or holds no LSA design.
    """
    design = of(config).design
    if design is None:
        raise ValueError(
            f"the model config's {Plugin.BLOCK!r} block holds no LSA design - this server runs {of(config).selector}"
        )
    return design


def pisa_of(config: object) -> PisaConfig:
    """Return the PISA design a server runs under.

    Args:
        config: A model config.

    Returns:
        The pyramid the block holds.

    Raises:
        ValueError: The block is refused, or does not name PISA.
    """
    pisa = of(config).pisa
    if pisa is None:
        raise ValueError(
            f"the model config's {Plugin.BLOCK!r} block does not name PISA - this server runs {of(config).selector}"
        )
    return pisa


def prefill_chunk() -> int:
    """Return the longest prefill segment the server hands an indexer at once.

    Returns:
        The server's `chunked_prefill_size`, or `UNCHUNKED_PREFILL` when it sets none.
    """
    from sglang.srt.runtime_context import get_schedule

    return int(get_schedule().chunked_prefill_size or UNCHUNKED_PREFILL)


def caches_prefixes() -> bool:
    """Return whether the server caches prefixes, so an index may take a cached prefix's index.

    Returns:
        True unless the server was launched with `--disable-radix-cache`.
    """
    from sglang.srt.runtime_context import get_memory

    return not get_memory().disable_radix_cache
