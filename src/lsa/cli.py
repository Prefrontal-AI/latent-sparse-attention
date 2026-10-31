"""The one `lsa` command - a subcommand per action, each routed to the module that owns it.

    lsa --help
    lsa --site local records show <a record directory>
    lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/lsa_flat_tp8.yaml --dry-run
    lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml --help

`--site <name>` comes first and picks the [site](site.py) every path of the command reads, the default site unless it says.

The dispatcher only routes. A subcommand's module is imported once it is chosen, so `lsa --help` and a
reader of json run on a bare install, and the rest of the command line goes to that module's own parser -
`lsa <subcommand> --help` is the module's help.

Every subcommand that computes submits a job through `lsa.slurm`, which runs it under the site's launcher - sbatch,
or submitit's local executor on one node - and `--dry-run` prints what it would submit. What stays local reads files and prints - the record tables, the
composition of a whole model - plus the shell lines `lsa env` prints and the prerequisites `lsa data prepare ruler`
fetches before submission, which a compute node with no egress cannot.
"""

import argparse
import importlib
import sys
from dataclasses import dataclass

from lsa import site


@dataclass(frozen=True)
class Command:
    """One subcommand - the module whose `main` it runs, what it does, and what `main` is called with."""

    module: str
    about: str
    # Positional arguments to `main`, for a module that serves several subcommands
    arguments: tuple[str, ...] = ()


# A group is a name to its subcommands, nested where the repository already groups
type Commands = dict[str, Command | Commands]

COMMANDS: Commands = {
    "evaluate": {
        "accuracy": Command(
            "lsa.evaluation.run", "score a served model with the accuracy suite", ("accuracy",)
        ),
        "kernels": Command(
            "lsa.evaluation.run", "time one attention layer through the harness", ("kernels",)
        ),
        "isolated": Command(
            "lsa.evaluation.performance.kernels.isolated",
            "time every arm of a model's layer in one job on a node held whole",
        ),
        "production": Command(
            "lsa.evaluation.run", "time a served model's requests end to end", ("production",)
        ),
        "publish": Command("lsa.evaluation.publish", "copy finished runs into the shared evaluation tree"),
        "dispatch": Command(
            "lsa.evaluation.dispatch", "keep a reservation full by adding workers to pooled accuracy runs"
        ),
    },
    "distill": {
        "targets": Command(
            "lsa.training.distillation.data.teacher", "select documents and extract distillation targets"
        ),
        "native-picks": Command(
            "lsa.training.distillation.data.native", "compute GLM's own sparse selection from a cache's x"
        ),
        "train": Command("lsa.training.distillation.run", "train the learned index space"),
        "rescore": Command(
            "lsa.training.distillation.rescore_maps",
            "score trained maps at a read they were not trained for",
        ),
        "served-read": Command(
            "lsa.training.distillation.served_read",
            "score an exported map's served read on held-out documents",
        ),
        "export": Command(
            "lsa.training.distillation.export", "write a run's maps as the file the server reads"
        ),
        "inject": Command(
            "lsa.training.distillation.inject", "put a distilled index space inside its checkpoint"
        ),
        "calibrate-tau": Command(
            "lsa.training.distillation.calibrate_tau",
            "record the loss temperature of an exported map that has none",
        ),
    },
    "native": {
        "train": Command(
            "lsa.training.native.run", "train a model from scratch under a native attention", ("train",)
        ),
        "evaluate": Command(
            "lsa.training.native.run",
            "score a native checkpoint on the held-out set or the standard benchmarks, by --suite",
            ("evaluate",),
        ),
        "report": Command(
            "lsa.training.native.benchmark.report", "pair the arms of benchmark records, with an interval"
        ),
    },
    "probe": Command("lsa.evaluation.adapter.probe", "screen a checkpoint before running anything on it"),
    "data": {
        "prepare": Command("lsa.data.prepare", "prepare the dataset declared by a card"),
    },
    "records": {
        "show": Command(
            "lsa.evaluation.accuracy.records.show",
            "print accuracy, kernel or production records as a table, by the kind each says it is",
        ),
        "compare": Command(
            "lsa.evaluation.accuracy.records.compare",
            "pair runs doc by doc, with the gap's error and interval",
        ),
        "join": Command(
            "lsa.evaluation.accuracy.records.join", "join the runs one configuration was split across"
        ),
        "rename": Command("lsa.evaluation.rename", "give a record a descriptive name and a note"),
        "rebuild": Command(
            "lsa.evaluation.performance.kernels.rebuild", "rebuild a run's record from its samples"
        ),
    },
    "kernels": {
        "cut": Command("lsa.evaluation.performance.kernels.data.cut", "cut a checkpoint to its first layers"),
        "modules": Command(
            "lsa.evaluation.performance.kernels.modules", "time the modules besides attention"
        ),
        "compose": Command(
            "lsa.evaluation.performance.kernels.compose", "compose a whole model from its layers"
        ),
        "roofline": Command(
            "lsa.evaluation.performance.roofline.run", "profile every CuTeDSL kernel on the roofline"
        ),
    },
    "env": Command("lsa.env", "print the offline hub contract as shell lines"),
}


def group_parser(words: list[str], group: Commands) -> argparse.ArgumentParser:
    """Build the parser that lists a group's subcommands, for its help and its usage errors.

    Args:
        words: The subcommands that led to the group, empty for the top level.
        group: The group.

    Returns:
        A parser with one subcommand per entry, each described by its `about`.
    """
    parser = argparse.ArgumentParser(prog=" ".join(["lsa", *words]), description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="subcommand", metavar="subcommand", required=True)
    for name, entry in group.items():
        about = entry.about if isinstance(entry, Command) else ", ".join(entry)
        subcommands.add_parser(name, help=about, add_help=False)
    return parser


def run(command: Command, words: list[str], rest: list[str]) -> None:
    """Import a subcommand's module and run its `main` over the rest of the command line.

    Args:
        command: The subcommand.
        words: The subcommands that named it, which become the program name its parser prints.
        rest: Everything after them.
    """
    module = importlib.import_module(command.module)
    sys.argv = [" ".join(["lsa", *words]), *rest]
    try:
        module.main(*command.arguments)
    except (site.NotNamed, site.Unsupported) as failure:
        # A value the site leaves empty is a line to fill in, and a command its launcher cannot run is a site to change, not a bug
        sys.exit(f"lsa: {failure}")


def main(argv: list[str] | None = None) -> None:
    """Route the command line to the subcommand it names.

    Args:
        argv: The arguments after the program name, or None for this process's own.
    """
    rest = list(sys.argv[1:] if argv is None else argv)
    if rest[:1] == ["--site"] and len(rest) > 1:
        site.select(rest[1])
        rest = rest[2:]
    words: list[str] = []
    entry: Command | Commands = COMMANDS
    while isinstance(entry, dict) and rest and rest[0] in entry:
        words.append(rest.pop(0))
        entry = entry[words[-1]]
    if isinstance(entry, Command):
        run(entry, words, rest)
        return
    # A group, or nothing it holds - its parser prints the help, or the usage error
    parser = group_parser(words, entry)
    parser.parse_args(rest)
    parser.error(f"no subcommand in {rest}")
