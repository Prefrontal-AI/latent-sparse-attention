"""Prepare one dataset card, as one CPU slurm job - whichever generator its kind names.

    lsa data prepare glm53_flash --sources edgebench     # a tokenized card - its sources in one tokenizer's ids
    lsa data prepare edgebench --sessions <run dirs...>  # the source this repository collects
    lsa data prepare gutenberg                           # a source streamed from the hub, as the first collector did
    lsa data prepare ruler --jobs 32                    # a benchmark rendered by its generator
    lsa data prepare longbench2 --max-context 131072     # a benchmark read from the hub and truncated
    lsa data prepare papertrail                          # a benchmark copied from its own repository
    lsa data prepare production                          # the workload a production run sends
    lsa data prepare ruler --check                       # read RULER's prepared cells, submit nothing
    lsa data prepare ruler --fetch-in-job                # clone its generators in the job, where compute nodes have egress
    lsa data prepare ruler --help                        # what that card's generator takes

A source with a `collector` is collected from the hub by [collect](collect.py), the one preparer whose job reads
the hub and so is not offline. Any other source or a tokenized card is prepared by [tokenize](tokenize.py), a workload by
the production prompt builder, and a benchmark by the generator its card's `preparer` names - the kind alone
does not tell RULER from LongBench v2 or PaperTrail. Each generator declares its own options, so `lsa data prepare <card>`
takes what that card's generator does, and the slurm flags over the job it asks for.
"""

import argparse
import sys
from collections.abc import Callable
from typing import Any, Protocol

from lsa import assets, env, slurm


class CardPreparer(Protocol):
    """What a module has to declare to prepare a card - the shape `lsa data prepare` calls it through."""

    # What the slurm job asks for
    PARAMETERS: dict[str, Any]

    def add_options(self, parser: argparse.ArgumentParser) -> None:
        """Add the options this generator takes to the command's parser."""

    def job(
        self, card: assets.DatasetCard, args: argparse.Namespace
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        """Return what the job runs for one card, and what it is called with."""


def preparer_of(card: assets.DatasetCard) -> CardPreparer:
    """Return the module that prepares one card - its `add_options`, its `job` and its `PARAMETERS`.

    Imported only once the card is known, so preparing a corpus never imports the evaluation harness.

    Args:
        card: The card to prepare.

    Returns:
        The preparing module.

    Raises:
        ValueError: The card is token shards, or a benchmark that names no generator, so nothing prepares it.
    """
    if isinstance(card, assets.SourceCard) and card.collector:
        from lsa.data import collect

        return collect
    if isinstance(card, assets.SourceCard | assets.TokenizedCard):
        from lsa.data import tokenize

        return tokenize
    if isinstance(card, assets.ShardCard):
        raise ValueError(
            f"{card.name} is read where it is - another pipeline wrote it and there is nothing to prepare"
        )
    if isinstance(card, assets.WorkloadCard):
        from lsa.evaluation.performance.production.data import prompts

        return prompts
    if card.preparer == assets.Preparer.RULER:
        from lsa.evaluation.accuracy.data import ruler

        return ruler
    if card.preparer == assets.Preparer.LONGBENCH:
        from lsa.evaluation.accuracy.data import longbench

        return longbench
    if card.preparer == assets.Preparer.PAPERTRAIL:
        from lsa.evaluation.accuracy.data import papertrail

        return papertrail
    raise ValueError(f"{card.name} is read from the hub as it is - there is nothing to prepare")


def main() -> None:
    """Parse the command line and submit the job that prepares the card it names."""
    # The card decides which options exist
    # => read it before the parser that holds them is built
    named = argparse.ArgumentParser(add_help=False)
    named.add_argument("card", nargs="?")
    slurm.add_flags(named)
    card_name = named.parse_known_args()[0].card

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("card", help="the dataset card to prepare, by name")
    slurm.add_flags(parser)
    if card_name not in assets.dataset_names():
        parser.parse_args()
        parser.error(f"no dataset card named {card_name!r} - there are {assets.dataset_names()}")

    card = assets.dataset(card_name)
    try:
        preparer = preparer_of(card)
        preparer.add_options(parser)
        args = parser.parse_args()
        # Only RULER declares --check
        # => its module reads the cells and the job is never built
        if getattr(args, "check", False):
            from lsa.evaluation.accuracy.data import ruler

            sys.exit(ruler.check_prepared(args))
        function, arguments = preparer.job(card, args)
    except ValueError as refusal:
        parser.error(str(refusal))
    # A preparer that reads the hub names its own environment, and every other one runs offline
    environment = [
        *getattr(preparer, "ENVIRONMENT", (*env.OFFLINE_HF, *env.EVAL_DATA)),
        *env.exports({"PYTHONUNBUFFERED": "1"}),
    ]
    print(f"{card.name}: prepared by {function.__module__}.{function.__qualname__}", flush=True)
    slurm.submit(args, f"lsa-prepare-{card.name}", environment, preparer.PARAMETERS, function, *arguments)
