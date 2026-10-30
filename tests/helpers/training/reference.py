"""Locate the reference training code's package inside the clone `REFERENCE_CLONE` names."""

from pathlib import Path


def reference_package(clone: str) -> Path:
    """Return the clone's one top-level package that holds `training/optim.py`."""
    packages = [path.parent.parent for path in Path(clone).glob("*/training/optim.py")]
    if len(packages) != 1:
        raise ValueError(f"wanted one package with training/optim.py under {clone}, found {packages}")
    return packages[0]
