"""The model card schema, and the one rule that reads every card.

A card is a yaml file whose name is its name, read into a dataclass that validates it.
The dataclasses are the schema, so a card with a field nothing reads fails at load rather than being ignored,
and the same loader the run configs use does the construction.

A dataset card's `kind:` picks its dataclass, and the card sits in the directory of that kind - a name is
unique across the four, so `dataset(name)` finds it wherever it is.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from lsa import config_loader
from lsa.assets.cards import (
    BenchmarkCard,
    DatasetCard,
    ModelCard,
    ShardCard,
    SourceCard,
    TokenizedCard,
    WorkloadCard,
)
from lsa.assets.constants import Kind
from lsa.data import manifests

# Where each kind of card lives, which is also what a name resolves within
_DATASETS: Path = Path(__file__).parent / "datasets"
_MODELS: Path = Path(__file__).parent / "models"

# The directory each dataset kind's cards sit in
_DIRECTORIES: dict[str, str] = {
    Kind.SOURCE: "sources",
    Kind.TOKENIZED: "tokenized",
    Kind.SHARDS: "shards",
    Kind.BENCHMARK: "benchmarks",
    Kind.WORKLOAD: "workloads",
}

_SUFFIX: str = ".yaml"


def dataset(name: str) -> DatasetCard:
    """Read one dataset card by name, whichever kind it is.

    Args:
        name: A card under `assets/datasets/<kind>`, without the suffix.

    Returns:
        The dataset it describes.

    Raises:
        ValueError: There is no card by that name, or two kinds hold one, or its `kind:` is not its
            directory's.
    """
    found = [
        kind for kind, folder in _DIRECTORIES.items() if (_DATASETS / folder / f"{name}{_SUFFIX}").is_file()
    ]
    if not found:
        raise ValueError(f"no dataset card named {name!r} - there are {dataset_names()}")
    if len(found) > 1:
        raise ValueError(f"{name!r} is a card of {found} - a dataset name is one card")
    readers: dict[str, Callable[[str], DatasetCard]] = {
        Kind.SOURCE: source,
        Kind.TOKENIZED: tokenized,
        Kind.SHARDS: shards,
        Kind.BENCHMARK: benchmark,
        Kind.WORKLOAD: workload,
    }
    return readers[found[0]](name)


def source(name: str) -> SourceCard:
    """Read one source card by name.

    Raises:
        ValueError: There is no source card by that name.
    """
    return config_loader.construct(SourceCard, _dataset_values(Kind.SOURCE, name))


def tokenized(name: str) -> TokenizedCard:
    """Read one tokenized dataset card by name.

    Raises:
        ValueError: There is no tokenized card by that name.
    """
    return config_loader.construct(TokenizedCard, _dataset_values(Kind.TOKENIZED, name))


def shards(name: str) -> ShardCard:
    """Read one token-shard card by name.

    Raises:
        ValueError: There is no shards card by that name.
    """
    return config_loader.construct(ShardCard, _dataset_values(Kind.SHARDS, name))


def benchmark(name: str) -> BenchmarkCard:
    """Read one benchmark card by name.

    Raises:
        ValueError: There is no benchmark card by that name.
    """
    return config_loader.construct(BenchmarkCard, _dataset_values(Kind.BENCHMARK, name))


def workload(name: str) -> WorkloadCard:
    """Read one workload card by name.

    Raises:
        ValueError: There is no workload card by that name.
    """
    return config_loader.construct(WorkloadCard, _dataset_values(Kind.WORKLOAD, name))


def model(name: str) -> ModelCard:
    """Read one checkpoint card by name.

    Args:
        name: A card under `assets/models`, without the suffix.

    Returns:
        The checkpoint it describes.

    Raises:
        ValueError: There is no card by that name.
    """
    path = _MODELS / f"{name}{_SUFFIX}"
    if not path.is_file():
        raise ValueError(f"no model card named {name!r} - there are {model_names()}")
    return config_loader.construct(ModelCard, _values(path))


def dataset_names(kind: str | None = None) -> list[str]:
    """Return every dataset card's name, sorted - of one kind, or of all four.

    Args:
        kind: One of `Kind.ALL`, or None for every kind.
    """
    kinds = [kind] if kind else list(Kind.ALL)
    return sorted(path.stem for one in kinds for path in (_DATASETS / _DIRECTORIES[one]).glob(f"*{_SUFFIX}"))


def model_names() -> list[str]:
    """Return every registered checkpoint name, sorted."""
    return sorted(path.stem for path in _MODELS.glob(f"*{_SUFFIX}"))


def unregistered(path: str | Path) -> TokenizedCard:
    """Describe a tokenized dataset that has no card, from the manifest tokenizing wrote beside it.

    A dataset under development is read by path rather than by name, and its manifest carries the card it was
    tokenized under, so the card comes back as it was declared and placed where the bytes are.

    Args:
        path: A tokenized dataset's directory.

    Returns:
        The dataset it describes, at that path.

    Raises:
        FileNotFoundError: The directory holds no manifest.
    """
    found = manifests.read_tokenized(path)
    return config_loader.construct(
        TokenizedCard, {**found.declared, "path": str(path), "about": "an unregistered dataset, read by path"}
    )


def resolve(name: str, path: str | None = None) -> TokenizedCard:
    """Return the tokenized dataset a run named, by card or by bare path.

    Args:
        name: A tokenized card's name, used when `path` is None.
        path: An unregistered tokenized dataset to read instead.

    Returns:
        The dataset it describes.
    """
    return unregistered(path) if path else tokenized(name)


def corpus(name: str, path: str | None = None) -> SourceCard | TokenizedCard | ShardCard:
    """Return the dataset a training run named - text, tokenized rows or token shards - by card or by bare path.

    Args:
        name: A source, tokenized or shards card's name, used when `path` is None.
        path: An unregistered tokenized dataset to read instead.

    Returns:
        The dataset it describes.

    Raises:
        ValueError: The name is a benchmark or a workload, which a training run does not read.
    """
    if path:
        return unregistered(path)
    found = dataset(name)
    if not isinstance(found, SourceCard | TokenizedCard | ShardCard):
        raise ValueError(
            f"{name!r} is a {type(found).__name__} - a training run reads a source, tokenized or shards card"
        )
    return found


def _dataset_values(kind: str, name: str) -> dict[str, Any]:
    """Return the fields of one dataset card of a kind, without its `kind:`.

    Raises:
        ValueError: There is no card by that name in that kind's directory, or its `kind:` is another.
    """
    path = _DATASETS / _DIRECTORIES[kind] / f"{name}{_SUFFIX}"
    if not path.is_file():
        raise ValueError(f"no {kind} card named {name!r} - there are {dataset_names(kind)}")
    values = _values(path)
    declared_kind = values.pop("kind", None)
    if declared_kind != kind:
        raise ValueError(f"{path} sits with the {kind} cards and says kind {declared_kind!r}")
    return values


def _values(path: Path) -> dict[str, Any]:
    """Return one card's fields, with every `${root:...}` resolved.

    Raises:
        ValueError: The file is not a mapping.
    """
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(values, dict):
        raise ValueError(f"{path} is not a mapping")
    return {str(key): value for key, value in values.items()}
