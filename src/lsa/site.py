"""Which machine this is - the one place a site is chosen and read.

A site is the file `configs/site/<site>.yaml`, and it holds every value that belongs to a machine rather than to
an experiment: the shared `root` the large artifacts live under, the server `images` by name, the `launcher` that
runs a job, and the `slurm` values a job is submitted with.
[The config loader](config_loader.py) composes it into every run as the `site` node, and code outside a loaded
run - a command that takes no config, a record reader, the viewer - reads it through `load`.

`DEFAULT_SITE` is the one place the default is chosen, and the line an export of the public repository changes.
`load` reads only the standard library and PyYAML, so a record reader that never composes a config needs neither omegaconf nor hydra.
`select` picks another for this process, which is what `lsa --site <name>` does, and `site=<name>` on a run's
command line picks another for that run alone.
A process that selected none reads `LSA_SITE`, which is how a job submitted under `--site` reads the same site its submitter did.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class NotNamed(ValueError):
    """A value a run needs from its site - an image, the root - is not set there.

    The command line prints it as one line and exits, since the fix is a line in the site's file.
    """


class Unsupported(ValueError):
    """A command needs something this site's launcher does not have - more than one node, a reservation.

    The command line prints it as one line and exits, since the fix is to run the command on a site that has it.
    """


# What a site's `launcher` may be, and so the executors a job can run under
LAUNCHERS: tuple[str, ...] = ("slurm", "local")

# What a site's `slurm` mapping may hold
# `partition`, `account`, `qos`, `constraint` and `exclude` are laid under every job that names none itself
# `reservation` is the pool `lsa evaluate dispatch` keeps full, and `open_partition` the partition of a job that stays out of it
SLURM_KEYS: tuple[str, ...] = (
    "partition",
    "open_partition",
    "account",
    "qos",
    "constraint",
    "reservation",
    "exclude",
)

# The site every process and run uses unless one is selected
# => the one line an export of the public repository changes
DEFAULT_SITE = "local"

# Where the sites live, beside the other configs of the checkout
SITES = Path(__file__).resolve().parents[2] / "configs" / "site"

# The variable a submitted job reads its site from, set by `lsa.slurm.executor`
SITE_VARIABLE = "LSA_SITE"

# The site this process selected, or None for the default
_selected: str | None = None


def select(name: str | None) -> None:
    """Pick the site this process reads, or go back to the default.

    Args:
        name: The site, as the file under `configs/site` is named, or None for `DEFAULT_SITE`.
    """
    global _selected
    _selected = name


def current() -> str:
    """Return the name of the site this process reads: the one selected, then `LSA_SITE`, then `DEFAULT_SITE`."""
    return _selected or os.environ.get(SITE_VARIABLE) or DEFAULT_SITE


def load(name: str | None = None) -> dict[str, Any]:
    """Read one site's values.

    Args:
        name: The site, as the file under `configs/site` is named, or None for the one this process reads.

    Returns:
        The site's mapping.

    Raises:
        ValueError: There is no such site, naming the sites there are, or the file is not a mapping.
    """
    chosen = name or current()
    path = SITES / f"{chosen}.yaml"
    if not path.is_file():
        known = sorted(found.stem for found in SITES.glob("*.yaml"))
        raise ValueError(f"no site {chosen!r} at {path} - the sites are {known}, and `site=<name>` picks one")
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"{path} must hold a mapping, not a list")
    return values


def launcher(name: str | None = None) -> str:
    """Return how this site runs a job: `slurm` submits it, `local` runs it on this machine.

    Args:
        name: The site, or None for the one this process reads.

    Raises:
        NotNamed: The site sets no launcher.
        ValueError: The launcher is neither of `LAUNCHERS`.
    """
    chosen = load(name).get("launcher")
    if not chosen:
        raise NotNamed(
            f"the site sets no launcher - set `launcher:` to one of {list(LAUNCHERS)} in {SITES}/{name or current()}.yaml"
        )
    if chosen not in LAUNCHERS:
        raise ValueError(
            f"launcher must be one of {list(LAUNCHERS)} - got {chosen!r} in site {name or current()!r}"
        )
    return str(chosen)


def slurm_values(name: str | None = None) -> dict[str, str]:
    """Read the Slurm values a site sets, and none for the ones it leaves out.

    A value the site does not set is absent, so a job is submitted without that flag at all.
    A list of `exclude` nodes comes back as the comma-separated string sbatch takes.

    Args:
        name: The site, or None for the one this process reads.

    Returns:
        Key to value, from `SLURM_KEYS`, for the keys that are set.

    Raises:
        ValueError: `slurm` is not a mapping, or holds a key that is not in `SLURM_KEYS`.
    """
    values = load(name).get("slurm") or {}
    if not isinstance(values, dict):
        raise ValueError(
            f"`slurm` in site {name or current()!r} must be a mapping, not {type(values).__name__}"
        )
    unknown = sorted(set(values) - set(SLURM_KEYS))
    if unknown:
        raise ValueError(
            f"unknown slurm keys {unknown} in site {name or current()!r} - the keys are {list(SLURM_KEYS)}"
        )
    found = {key: value for key, value in values.items() if value not in (None, "", [])}
    if isinstance(found.get("exclude"), list):
        found["exclude"] = ",".join(found["exclude"])
    return {key: str(value) for key, value in found.items()}
