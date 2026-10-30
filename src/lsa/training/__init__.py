"""Training - the distillation and native stages, and what more than one stage reads.

Behind the `training` extra, and behind one environment.

| stage | what it does |
|---|---|
| [distillation](distillation/__init__.py) | the index space is learned against a frozen teacher |
| [native](native/__init__.py) | a model trains from random initialisation under a multi-resolution attention, independent of distillation |

The distillation stage owns its own `data/` - the targets it extracts and reads back, and nothing else -
and what more than one stage reads lives here, above it, so no stage reaches into another.

| shared | what it is |
|---|---|
| [data](data/__init__.py) | the tokenized corpus every stage reads from its own half of one split |
| [distilled_maps](distilled_maps.py) | the maps distillation writes, as the injector reads them |
| [targets](data/targets.py) | what a targets extraction records about itself |
| [constants](constants.py) | the formats and the split every stage has to agree on |

[assets](../assets/__init__.py) is what a corpus is by name.
"""

from importlib.util import find_spec

__all__ = ["require_extra"]


def require_extra(*packages: str) -> None:
    """Fail with the missing extra rather than an import error from inside a vendor package.

    Args:
        *packages: The packages the caller needs, by name.

    Raises:
        ImportError: A package the caller needs is not installed.
    """
    missing = [name for name in packages if find_spec(name) is None]
    if missing:
        raise ImportError(
            f"lsa.training needs the training extra - missing {', '.join(missing)} - "
            "run `uv sync --extra training`"
        )
