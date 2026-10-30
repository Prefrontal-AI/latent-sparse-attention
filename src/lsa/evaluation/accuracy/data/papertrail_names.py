"""How PaperTrail names its conditions and tasks - kept apart from the preparer so the suites can read them bare."""

from lsa.evaluation.accuracy.data import papertrail_card
from lsa.evaluation.constants import Suite

# The control contexts, the question alone first
QUESTION_ONLY = "C1"
CONTROLS: tuple[str, ...] = (QUESTION_ONLY, "C2_hosts", "C3_synthetic", "C4_background")

# The tier of a not-in-corpus question, which is reported apart from the score, and its tasks' suffix
NOT_IN_CORPUS = "NIC"
NIC_SUFFIX = "_nic"


def length_condition(length: int) -> str:
    """Name the condition of one length, as the benchmark's samples are named - `L400000`."""
    return f"L{length}"


def conditions(max_length: int | None = None) -> list[str]:
    """Return every condition of a sample - the controls, then the card's lengths up to `max_length`."""
    lengths = [
        length for length in papertrail_card().length_values if max_length is None or length <= max_length
    ]
    return [*CONTROLS, *(length_condition(length) for length in lengths)]


def task_name(sample: str, condition: str, not_in_corpus: bool = False) -> str:
    """Name the task of one sample's scored or not-in-corpus questions under one condition."""
    return f"{Suite.PAPERTRAIL_PREFIX}{sample}_{condition}{NIC_SUFFIX if not_in_corpus else ''}"


def parse_task(name: str) -> tuple[str, str, bool]:
    """Split a task name into its sample, its condition and whether it holds the not-in-corpus questions.

    Raises:
        KeyError: The name is no sample and condition of the card.
    """
    rest = name.removeprefix(Suite.PAPERTRAIL_PREFIX)
    not_in_corpus = rest.endswith(NIC_SUFFIX)
    rest = rest.removesuffix(NIC_SUFFIX)
    for sample in papertrail_card().tasks:
        condition = rest.removeprefix(f"{sample}_")
        if condition != rest and condition in conditions():
            return sample, condition, not_in_corpus
    raise KeyError(f"unknown PaperTrail task {name}")


def condition_length(condition: str) -> int | None:
    """Return the length a condition names, or None for a control."""
    return None if condition in CONTROLS else int(condition.removeprefix("L"))
