"""What a targets extraction records about itself, beside the targets it wrote.

Each collection of the teacher's attention targets carries one manifest naming every document it extracted,
the side of the distillation split each fell on, and the model and rotary layout it was taken under.
Distillation reads it to choose a set's documents and to record a map's teacher.
"""

from pathlib import Path
from typing import Any

from lsa.io import read_json, write_json
from lsa.training.constants import Targets


def read_manifest(root: str | Path, collection: str) -> dict[str, Any]:
    """Return what one collection's extraction recorded about itself.

    Args:
        root: The targets root.
        collection: The collection directory, which is where its own manifest sits.

    Returns:
        The manifest, parsed.

    Raises:
        FileNotFoundError: The collection has no manifest, so nothing has been extracted into it.
    """
    manifest = Path(root) / collection / Targets.MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(f"no extraction manifest at {manifest}")
    recorded: dict[str, Any] = read_json(manifest)
    return recorded


def write_manifest(root: str | Path, payload: dict[str, Any]) -> None:
    """Record what produced a collection next to it.

    Every shard of one extraction writes the same manifest, so this replaces the file rather than filling
    it in place - two processes writing it at once would otherwise leave one of them reading a torn file.

    Args:
        root: The collection directory.
        payload: Anything `json.dumps` accepts.
    """
    write_json(Path(root) / Targets.MANIFEST, payload)


def extracted_documents(root: str | Path, collection: str) -> frozenset[tuple[str, str]]:
    """Return every document one collection extracted, as the tokenized corpus names it.

    A distilled map was fit to these documents' attention, so a validation set that holds one of them
    scores that map on its own training data.

    Args:
        root: The targets root.
        collection: The collection directory.

    Returns:
        `(source, document id)` pairs, whichever side of the distillation split each one fell on.

    Raises:
        FileNotFoundError: The collection has no manifest, so nothing has been extracted into it.
    """
    entries = read_manifest(root, collection)[Targets.DOCUMENTS]
    return frozenset((entry["source"], str(entry["document"])) for entry in entries)


def extracted_from(collection: str | Path) -> frozenset[tuple[str, str]]:
    """Return every document one collection extracted, named by its own directory.

    Args:
        collection: The collection's directory, `<targets root>/<collection>`.

    Returns:
        `(source, document id)` pairs.

    Raises:
        FileNotFoundError: The collection has no manifest, so nothing has been extracted into it.
    """
    path = Path(collection)
    return extracted_documents(path.parent, path.name)
