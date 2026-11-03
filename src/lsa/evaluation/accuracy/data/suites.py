"""What an evaluation run asks for - tasks, sample counts, repeats.

full: the recipe - a fixed 2,000 of MMLU-Pro, GPQA Diamond, a fixed 100 of MATH-500, AIME25 four times, a fixed
60 of LongBench v2, PaperTrail as `papertrail` runs it, and every RULER cell of the grid. Less MMLU-Pro, with
PaperTrail in place of GraphWalks, and with RULER at 100 rows a cell to 1M and 20 above where it held 20 and 10,
it is the earlier run of GLM-5.3-Flash, whose RULER rows are each file's first 20 and 10.
A one-node server skips the RULER lengths it cannot hold, and a two-node run restricted
to them by `data.lengths` and continued into the same run id fills them in.
ruler: RULER alone - the same RULER plans `full` runs.
papertrail: PaperTrail - every sample under its four controls and at every length, three seeds of each question.
smoke: a few prompts of every benchmark with short budgets - whether an architecture runs at all.

Sampling is not here. How a model is sampled is a property of the model, so the experiment yaml that names
it carries a `generation` block with one entry per dataset - `Suite.DATASETS` - and the runner fills every
plan from the entry of the dataset its task is drawn from. A plan states sampling of its own only where a
test needs to pin it.
"""

import dataclasses
import os

from lsa import io
from lsa.evaluation.accuracy.data import (
    ONE_NODE_LENGTH,
    papertrail_card,
    papertrail_names,
    ruler_card,
)
from lsa.evaluation.accuracy.sampling import Sampling
from lsa.evaluation.constants import Suite

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSETS_PATH = os.path.join(HERE, "subsets.json")
HASHES_PATH = os.path.join(HERE, "hashes.json")


def pins_path(pins: str | None = None) -> str:
    """Return the file a set of prompt-set pins lives in.

    The prompts of a tokenizer-sized set - RULER, LongBench v2 - depend on the tokenizer they were counted in, so
    a corpus generated for another model is pinned in a file of its own beside `hashes.json`.

    Args:
        pins: The set's name, `hashes_<pins>.json` - None or empty for the default corpus's `hashes.json`.
    """
    return os.path.join(HERE, f"hashes_{pins}.json") if pins else HASHES_PATH


@dataclasses.dataclass(frozen=True)
class Plan:
    """One task in a suite. Docs: all when n and subset are None, the first n, or a named id list."""

    task: str
    n: int | None = None
    subset: str | None = None
    repeats: int = 1
    # None takes the generation block's entry for this plan's dataset
    sampling: Sampling | None = None
    # A completion budget below the dataset's, for the smoke suite
    max_tokens: int | None = None
    # Requests of this plan's cap group in flight at once, below the server's cap - None leaves only the server's
    concurrency: int | None = None

    @property
    def cap_group(self) -> str:
        """Name the plans that share one cap - every RULER task of one length, whose prompts hold the same KV,
        and a PaperTrail condition's scored and not-in-corpus questions, which share its context."""
        if self.task.startswith(Suite.RULER_PREFIX):
            return f"{Suite.RULER_PREFIX}{self.task.rsplit('_', 1)[1]}"
        if self.task.startswith(Suite.PAPERTRAIL_PREFIX):
            return self.task.removesuffix(papertrail_names.NIC_SUFFIX)
        return self.task

    @property
    def shared_context(self) -> str | None:
        """Name the context every prompt of this plan opens with, which one request prefills for the rest.

        None for a plan whose prompts each hold their own - every task but PaperTrail.
        """
        if self.task.startswith(Suite.PAPERTRAIL_PREFIX):
            return self.cap_group
        return None

    @property
    def length(self) -> int | None:
        """Return the RULER or PaperTrail context length in the task name, or None for a control or another benchmark."""
        if self.task.startswith(Suite.RULER_PREFIX):
            return int(self.task.rsplit("_", 1)[1])
        if self.task.startswith(Suite.PAPERTRAIL_PREFIX):
            return papertrail_names.condition_length(papertrail_names.parse_task(self.task)[1])
        return None

    @property
    def dataset(self) -> str:
        """Name the dataset this plan's task is drawn from, which keys its entry of a generation block."""
        return Suite.dataset_of(self.task)

    @property
    def phase(self) -> int:
        """Return which of `Suite.PHASES` this plan runs in - the reasoning tasks before the long prompts."""
        return next(i for i, datasets in enumerate(Suite.PHASES) if self.dataset in datasets)

    @property
    def key(self) -> str:
        """Names the exact prompt set - used to store and check the prompt-set hash."""
        which = self.subset or (f"first{self.n}" if self.n else "all")
        return f"{self.task}:{which}"

    def ids(self) -> list[int] | None:
        """Return the doc ids this plan asks for - a named subset, the first n, or None for every doc."""
        if self.subset:
            return subset_ids(self.subset)
        if self.n:
            return list(range(self.n))
        return None


def ruler_concurrency(length: int) -> int:
    """Return how many requests of one RULER length run at once - past 1M the prompt fills the batch alone."""
    if length <= 131072:
        return 8
    return 4 if length <= ONE_NODE_LENGTH else 1


def ruler_plans() -> list[Plan]:
    """Return every RULER task at every length of its card's grid, every row the files hold.

    That is 100 prompts up to 1M and 20 above, where one 10M prompt is 20 minutes of prefill. A server that
    cannot hold a length skips it and the record says why, so one suite serves every deployment.
    """
    card = ruler_card()
    return [
        Plan(
            f"{Suite.RULER_PREFIX}{ruler_task}_{length}",
            n=card.rows_at(length),
            concurrency=ruler_concurrency(length),
        )
        for length in card.length_values
        for ruler_task in card.tasks
    ]


def papertrail_concurrency(length: int | None) -> int:
    """Return how many requests of one PaperTrail condition run at once.

    They share the context's KV, but an LSA request still builds its own buckets over the whole of it, from a
    bucket cache the size of the KV pool - about 2.1M tokens on one node, so three 512k questions fit and four do not.
    A control's context is up to about 430k tokens, so it runs as 512k does.
    """
    if length is None or length <= 512_000:
        return 3
    return 2 if length <= ONE_NODE_LENGTH else 1


# Independent samples of every PaperTrail question, at the model's own sampling
PAPERTRAIL_SEEDS = 3
# The one length the not-in-corpus control runs at
NOT_IN_CORPUS_LENGTH = 512_000


def papertrail() -> list[Plan]:
    """Return PaperTrail - every sample's 30 scored questions under its controls and at every length.

    Three seeds of each question, 90 answers over one context. The not-in-corpus questions check for made-up
    answers rather than score the model, so they run as a control - at one length, over the corpus they ask about.
    A server that cannot hold a length skips it and the record says why.
    """
    control_condition = papertrail_names.length_condition(NOT_IN_CORPUS_LENGTH)
    return [
        Plan(
            papertrail_names.task_name(sample, condition, not_in_corpus),
            repeats=PAPERTRAIL_SEEDS,
            concurrency=papertrail_concurrency(papertrail_names.condition_length(condition)),
        )
        for sample in papertrail_card().tasks
        for condition in papertrail_names.conditions()
        for not_in_corpus in ((False, True) if condition == control_condition else (False,))
    ]


def ruler() -> list[Plan]:
    """Return RULER alone - the long-context grid, nothing else."""
    return ruler_plans()


def full() -> list[Plan]:
    """Return the recipe - the knowledge and reasoning sets, a fixed subset of LongBench v2, PaperTrail and every RULER cell.

    A cap only where a prompt is long - a request holds its whole prompt in the KV pool, and a short one
    holds almost nothing, so the reasoning tasks are bounded by the server's cap alone. Every RULER task of a
    length shares its cap, so the KV the long prompts hold is what it was one task at a time.
    """
    return [
        Plan("mmlu_pro", subset="mmlu_pro_2000"),
        Plan("gpqa_diamond"),
        Plan("math500", subset="math500_100"),
        Plan("aime25", repeats=4),
        # Two prompts of 884,736 tokens and their 163,840 budgets fill the 2.1M pool, and a third would wait in the queue
        Plan("longbench2", subset="longbench2_60", concurrency=2),
        *papertrail(),
        *ruler_plans(),
    ]


def smoke() -> list[Plan]:
    """Return a few prompts of every benchmark with short budgets - the whole path in minutes."""
    return [
        Plan("mmlu_pro", n=8, max_tokens=2048, concurrency=8),
        Plan("gpqa_diamond", n=8, max_tokens=2048, concurrency=8),
        Plan("math500", n=8, max_tokens=2048, concurrency=8),
        Plan("aime25", n=4, repeats=2, max_tokens=2048, concurrency=8),
        Plan("longbench2", n=4, max_tokens=1024, concurrency=4),
        # A GraphWalks answer is the reply's last line, which a reasoning model at 2048 tokens rarely reaches
        Plan("graphwalks_128k", n=8, max_tokens=8192, concurrency=8),
        Plan("ruler_niah_multikey_2_4096", n=4, concurrency=8),
        Plan("ruler_cwe_4096", n=4, concurrency=8),
        Plan("ruler_niah_multikey_2_32768", n=4, concurrency=8),
        Plan("ruler_cwe_32768", n=4, concurrency=8),
    ]


SUITES = {"ruler": ruler, "full": full, "papertrail": papertrail, "smoke": smoke}

# A name `data.tasks` takes for a group of tasks, so a run need not spell each one
# => `ruler` is every RULER task at every length, which `data.lengths` narrows
TASK_GROUPS = {"ruler": lambda: [plan.task for plan in ruler_plans()]}


def expand_groups(tasks: list[str]) -> list[str]:
    """Return a task list with each group name replaced by the tasks it stands for.

    Args:
        tasks: Task names, plan keys and group names from `TASK_GROUPS`.

    Returns:
        The same list with every group spelled out, in order and without repeats.
    """
    spelled: list[str] = []
    for name in tasks:
        for task in TASK_GROUPS[name]() if name in TASK_GROUPS else [name]:
            if task not in spelled:
                spelled.append(task)
    return spelled


def select(suite_name: str, only: list[str] | None = None, lengths: list[int] | None = None) -> list[Plan]:
    """Return the plans of a suite a run asks for - only the named tasks, and only the named lengths.

    A length filter keeps RULER and PaperTrail's lengths alone, so `full` restricted to the lengths past 1M is the
    two-node half of a run.

    Args:
        suite_name: One of `SUITES`.
        only: Task names or plan keys to keep, or None for every plan.
        lengths: RULER or PaperTrail lengths to keep, or None for every length.

    Returns:
        The selected plans, in suite order.

    Raises:
        KeyError: A filter names a task, a key or a length the suite does not have, or together they match
            nothing.
    """
    plans = SUITES[suite_name]()
    if only:
        unknown = sorted(set(only) - {plan.task for plan in plans} - {plan.key for plan in plans})
        if unknown:
            offered = sorted({plan.task for plan in plans})
            raise KeyError(f"suite {suite_name} has no task {unknown}, it offers {offered}")
        plans = [plan for plan in plans if plan.task in only or plan.key in only]
    if lengths:
        unknown_lengths = sorted(set(lengths) - {plan.length for plan in SUITES[suite_name]()})
        if unknown_lengths:
            offered_lengths = sorted({plan.length for plan in SUITES[suite_name]() if plan.length})
            raise KeyError(f"suite {suite_name} has no length {unknown_lengths}, it offers {offered_lengths}")
        plans = [plan for plan in plans if plan.length in lengths]
    if (only or lengths) and not plans:
        raise KeyError(f"no task of suite {suite_name} matches {only or ''} {lengths or ''}")
    return plans


def subset_ids(name: str) -> list[int]:
    """Return the doc ids of one named subset.

    Raises:
        KeyError: `subsets.json` holds no subset of that name.
    """
    subsets = io.read_json(SUBSETS_PATH, {})
    if name not in subsets:
        raise KeyError(f"subset {name} not in {SUBSETS_PATH}")
    ids: list[int] = subsets[name]
    return ids
