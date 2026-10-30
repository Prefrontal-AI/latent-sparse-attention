"""Reading a checkpoint's config, the same way in every package.

This module imports nothing else of the repo's.
"""

from collections.abc import Mapping
from typing import Any


def text_config(config: Any) -> Any:
    """Return the decoder's own config, past whatever a multimodal checkpoint wraps it in.

    Args:
        config: The checkpoint's config, as a parsed `config.json` or as an object with attributes.

    Returns:
        The nested `text_config` when the checkpoint has one, else the config itself.
    """
    if isinstance(config, Mapping):
        return config.get("text_config") or config
    return getattr(config, "text_config", config)
