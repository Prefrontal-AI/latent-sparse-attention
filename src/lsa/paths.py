"""Where the large artifacts live - one root, shaped like the repository that names them.

The root is the `root` key of the site, [the file of this machine's values](site.py), and `LSA_ROOT` overrides it.
That is the whole precedence: `LSA_ROOT` when it is set, then the site's `root`, and no third fallback - a site
without a root and no `LSA_ROOT` fails, saying where to set one.
A relative `root` is taken against the working directory, which is what the `local` site's `./runs` is.
A shared root is setgid and readable by everyone who runs here - a per-user path is not, so nothing shared is
named from one.
Its tree follows the code: `data/` holds a directory per dataset card kind and the targets
`configs/data/targets` extracts, `training/` a directory per trainer and model as `configs/training` has them,
`evaluations/` the scored and timed runs as `lsa.evaluation` nests them, and `models/` the checkpoints the
model cards name and the maps they are built from. So a card or a config's path says where its artifact is.

Every location is a function rather than a constant, so the root is read when a path is built and a run or a
test that sets `LSA_ROOT` moves all of them at once.

A config names a path under the root as `${root:models/maps/glm53_flash__long-16k-1M.pt}` rather than
spelling the root out, and [the config loader](config_loader.py) resolves it through `under_root` against the
site the run composes.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lsa import site

ROOT_VARIABLE = "LSA_ROOT"


def declared_root(values: Mapping[str, Any] | None = None) -> Path:
    """Return the root the site declares - the one place the site's `root` is read.

    Args:
        values: A loaded site, or None for the site this process reads.

    Returns:
        The site's `root`, made absolute against the working directory.

    Raises:
        NotNamed: The site sets no root.
    """
    values = site.load() if values is None else values
    declared = values.get("root")
    if not declared:
        raise site.NotNamed(
            f"the site sets no root - set `root:` in {site.SITES}/<site>.yaml, or export {ROOT_VARIABLE}"
        )
    return Path(os.path.abspath(declared))


def root(values: Mapping[str, Any] | None = None) -> Path:
    """Return the shared root every large artifact lives under.

    Args:
        values: A loaded site, or None for the site this process reads.

    Returns:
        `LSA_ROOT` when it is set, and the site's root otherwise.
    """
    override = os.environ.get(ROOT_VARIABLE)
    return Path(override) if override else declared_root(values)


def under_root(relative: str = "", values: Mapping[str, Any] | None = None) -> str:
    """Return a path under the root, as the string a config field holds.

    Args:
        relative: Path below the root, or empty for the root itself.
        values: A loaded site, or None for the site this process reads.
    """
    base = root(values)
    return str(base / relative) if relative else str(base)


def under_datasets(relative: str, values: Mapping[str, Any] | None = None) -> str:
    """Return a path under the site's shared datasets directory, as the string a card field holds.

    Args:
        relative: Path below the directory.
        values: A loaded site, or None for the site this process reads.

    Returns:
        The site's `datasets`, made absolute against the working directory, joined with `relative`.

    Raises:
        NotNamed: The site sets no `datasets`.
    """
    values = site.load() if values is None else values
    declared = values.get("datasets")
    if not declared:
        raise site.NotNamed(f"the site sets no datasets - set `datasets:` in {site.SITES}/<site>.yaml")
    return os.path.join(os.path.abspath(declared), relative)


def data() -> Path:
    """Return what a job reads rather than produces - one directory per dataset card kind, and the targets."""
    return root() / "data"


def benchmarks() -> Path:
    """Return the prepared benchmark files, their manifests and RULER's pinned generator - the suite's data root."""
    return data() / "benchmarks"


def targets() -> Path:
    """Return the teacher's attention targets a distillation reads, `<model card>/<extraction config>`."""
    return data() / "targets"


def goldens() -> Path:
    """Return the key ids each design selected on the freeze, which the golden replay compares against."""
    return data() / "goldens"


def training() -> Path:
    """Return what the trainers write, `<trainer>/<model card>/<run name>`."""
    return root() / "training"


def evaluations() -> Path:
    """Return the scored and timed runs, one directory per run under its kind's own nesting."""
    return root() / "evaluations"


def models() -> Path:
    """Return the checkpoints a server loads, one directory per model card."""
    return root() / "models"


def maps() -> Path:
    """Return the learned index maps, one file per fitted map, in the format serving loads."""
    return models() / "maps"


def logs() -> Path:
    """Return submitit's output, one set per slurm job, on storage the login node can read."""
    return root() / "logs"
