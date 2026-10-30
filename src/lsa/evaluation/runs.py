"""Name a run on a command line: its directory or its `record.json`, whichever the reader was handed."""

import argparse
from pathlib import Path

from lsa.evaluation.constants import Records


def run_directory(argument: str | Path, record_required: bool = True) -> Path:
    """Return a run's directory, given the directory or its record.

    Args:
        argument: A run directory or the `record.json` inside one.
        record_required: Whether a directory must already hold its record - a rebuild and a run still going need not.

    Returns:
        The run's directory.

    Raises:
        argparse.ArgumentTypeError: The path is neither, which argparse prints as a usage error.
    """
    path = Path(argument)
    if path.is_file() and path.name == Records.RECORD:
        return path.parent
    if path.is_dir() and (not record_required or (path / Records.RECORD).is_file()):
        return path
    raise argparse.ArgumentTypeError(
        f"{path} is neither a run directory holding {Records.RECORD} nor a {Records.RECORD}"
    )


def unrequired_run_directory(argument: str | Path) -> Path:
    """Return a run's directory, given the directory or its record - a directory without a record included."""
    return run_directory(argument, record_required=False)
