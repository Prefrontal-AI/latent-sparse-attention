"""Rebuild a kernel run's record from the raw files under it.

    lsa records rebuild <results>/performance/kernels/<run_id>

Needs no GPU and no container - it reads `samples/` and rewrites `record.json`. That is what picks up a
run whose writer was killed between a harness process and the rewrite that follows it, and what
re-summarises an old run after the record's own arithmetic changes.

The facts only the launcher knew - the note, the tag, the times, the configuration label and the image -
are kept from the record already there, so a rebuild of a run that has none leaves them empty.
"""

import argparse

from lsa.evaluation.performance.kernels.harness import rebuild
from lsa.evaluation.runs import unrequired_run_directory


def main() -> None:
    """Rebuild the record of every run directory named on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=unrequired_run_directory,
        metavar="RUN_DIR",
        help="a run directory holding samples/, or its record.json",
    )
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(rebuild(run_dir.resolve()))
