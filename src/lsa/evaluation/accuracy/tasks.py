"""The tasks we evaluate, as lm-eval task objects.

What comes from lm-eval: the datasets, the GPQA prompt and its answer filters, RULER's answer
post-processing, the MATH/AIME boxed-answer grader. What is ours: the zero-shot boxed prompt for the math
tasks, a zero-shot MMLU-Pro under GPQA's template (lm-eval only has it five-shot, for a base model), the generative LongBench v2 task (the official protocol, which lm-eval only has as a log-likelihood
task), GraphWalks under OpenAI's own grader (lm-eval's port scores a correct empty answer as wrong), RULER
served from fixed sample files that NVIDIA's own generators made (`data.ruler`), so the same prompts are scored
in every run, and PaperTrail under its own grader, each condition's context joined to its questions as they are read.

Every task is a ConfigurableTask - the same class lm-eval builds from its yaml files.
"""

import dataclasses
import functools
import hashlib
import random
import string
from collections.abc import Callable
from typing import Any

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

from lsa import assets
from lsa.evaluation.accuracy.data import (
    DATA_ROOT,
    GRAPHWALKS_FILES,
    graphwalks_rows,
    longbench_card,
    longbench_path,
    papertrail_names,
    papertrail_path,
    ruler_card,
    ruler_path,
)
from lsa.evaluation.accuracy.data import longbench as longbench_prompt
from lsa.evaluation.accuracy.data import papertrail as papertrail_data
from lsa.evaluation.accuracy.grading import graphwalks, letters, longbench, mathgrade, papertrail, ruler
from lsa.evaluation.constants import Suite

# The standard instruction for reasoning models on MATH-style problems
MATH_PROMPT = "{{problem}}\n\nPlease reason step by step, and put your final answer within \\boxed{}."


@functools.cache
def manager() -> TaskManager:
    """Return lm-eval's task index, built once - it is the slow part of loading GPQA."""
    return TaskManager()


@dataclasses.dataclass(frozen=True)
class TaskDefinition:
    """One evaluated task: how to build it, what its primary metric is, what the target of a doc is."""

    benchmark: str
    name: str
    category: str
    # The primary metric key as a metric dict spells it, as `exact_match:flexible-extract`
    metric: str
    build: Callable[..., Any] = dataclasses.field(repr=False, compare=False)
    target: Callable[..., Any] = dataclasses.field(repr=False, compare=False)
    extract: Callable[..., Any] = dataclasses.field(repr=False, compare=False, default=lambda reply: None)
    # What the viewer shows of a doc - its question, and the fields a page splits samples by
    question: Callable[[dict[str, Any], str], str] = dataclasses.field(
        repr=False, compare=False, default=lambda doc, prompt: ""
    )
    meta: Callable[[dict[str, Any]], dict[str, Any]] = dataclasses.field(
        repr=False, compare=False, default=lambda doc: {}
    )
    # The fields of `meta` a record splits the score by, as PaperTrail's by tier
    breakdown: tuple[str, ...] = ()
    # Where a long prompt's closing part starts, None for the prompt's last lines
    tail_marker: str | None = None
    # The RULER or PaperTrail context length in tokens, None for a control
    length: int | None = None
    # The fixed sample file a task reads, None for a hub dataset
    path: str | None = None


def _math_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Grade one MATH-style reply against its doc's answer."""
    return mathgrade.grade(results[0], doc["answer"])


def _mmlu_pro_text(doc: dict[str, Any]) -> str:
    """Return GPQA's zero-shot chain-of-thought prompt over the question's own choices, up to ten of them."""
    choices = "".join(f"({string.ascii_uppercase[i]}) {option}\n" for i, option in enumerate(doc["options"]))
    question = f"What is the correct answer to this question: {doc['question']}"
    return f"{question}\nChoices:\n{choices}Let's think step by step: "


def _mmlu_pro_target(doc: dict[str, Any]) -> str:
    """Return the target as the choice letter in parentheses."""
    return f"({doc['answer']})"


def _mmlu_pro_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the question without its choices."""
    return doc.get("question", "")


def _mmlu_pro_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's category and source."""
    return {"category": doc.get("category"), "src": doc.get("src")}


def _gpqa_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the question without its choices."""
    return doc.get("Question", "")


def _gpqa_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's domain and subdomain."""
    return {"domain": doc.get("High-level domain"), "subdomain": doc.get("Subdomain")}


def _math_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the problem."""
    return doc.get("problem", "")


def _math500_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's subject and level."""
    return {"subject": doc.get("subject"), "level": doc.get("level")}


def _aime25_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's id."""
    return {"id": doc.get("id")}


def _longbench_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the question the prompt asks, between its question and choices markers, or empty."""
    start = prompt.rfind(longbench_prompt.QUESTION)
    end = prompt.rfind(longbench_prompt.CHOICES)
    return prompt[start + len(longbench_prompt.QUESTION) : end] if start >= 0 else ""


def _longbench_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's domain, difficulty and length."""
    keys = ("domain", "sub_domain", "difficulty", "length", "context_tokens", "truncated")
    return {key: doc.get(key) for key in keys}


def _graphwalks_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the operation the prompt asks for, or empty if it has none."""
    start = prompt.rfind(graphwalks.OPERATION)
    end = prompt.find("\n\n", start + len(graphwalks.OPERATION))
    return prompt[start + len(graphwalks.OPERATION) : end].strip() if start >= 0 else ""


def _graphwalks_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's problem type and prompt length."""
    return {"problem_type": doc.get("problem_type"), "prompt_chars": doc.get("prompt_chars")}


def _papertrail_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the question."""
    return doc.get("question") or ""


def _papertrail_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the doc's sample, condition, question, trail and answer fields, and its answer."""
    keys = (
        "sample",
        "condition",
        "qid",
        "trail",
        "tier",
        "answer_type",
        "answer_papers",
        "check_papers",
    )
    return {**{key: doc.get(key) for key in keys}, "answer": papertrail_data.answer_of(doc)}


def _ruler_question(doc: dict[str, Any], prompt: str) -> str:
    """Return the last non-empty line of the input, which holds the question."""
    lines = [line for line in doc.get("input", "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _file_task(
    name: str,
    path: str,
    doc_to_text: str,
    doc_to_target: str,
    process_results: Callable[..., Any],
    metric: str,
    gen_prefix: str | None = None,
    max_gen_toks: int = 128,
) -> ConfigurableTask:
    """Build a generative lm-eval task over one fixed sample file.

    Args:
        name: The task name.
        path: The jsonl file.
        doc_to_text: The prompt template.
        doc_to_target: The target template.
        process_results: The grader.
        metric: The metric it returns.
        gen_prefix: What the reply is prefilled with, or None.
        max_gen_toks: lm-eval's generation budget, which the suite's own sampling overrides.

    Returns:
        The task.
    """
    config: dict[str, Any] = {
        "task": name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": path}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": doc_to_text,
        "doc_to_target": doc_to_target,
        "target_delimiter": " ",
        "process_results": process_results,
        "metric_list": [{"metric": metric, "aggregation": "mean", "higher_is_better": True}],
        "generation_kwargs": {
            "max_gen_toks": max_gen_toks,
            "temperature": 0,
            "do_sample": False,
            "until": [],
        },
        "num_fewshot": 0,
        "metadata": {"version": 1.0},
    }
    if gen_prefix is not None:
        config["gen_prefix"] = gen_prefix
    return ConfigurableTask(config=config)


def _papertrail_task(name: str, path: str, not_in_corpus: bool) -> ConfigurableTask:
    """Build one PaperTrail task over a condition's question file, its context joined to every question.

    The rows are read through `papertrail_data.read_rows` rather than lm-eval's json loader, because the context
    is stored once beside the questions rather than once in each row.
    """
    import datasets

    return ConfigurableTask(
        config={
            "task": name,
            "dataset_path": "json",
            "custom_dataset": lambda **metadata: {
                "test": datasets.Dataset.from_list(
                    papertrail_data.as_docs(papertrail_data.read_rows(path, not_in_corpus))
                )
            },
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "{{prompt}}",
            "doc_to_target": "{{qid}}",
            "target_delimiter": " ",
            "process_results": papertrail.process_results,
            "metric_list": [{"metric": "score", "aggregation": "mean", "higher_is_better": True}],
            "generation_kwargs": {"max_gen_toks": 163840, "temperature": 0, "do_sample": False, "until": []},
            "num_fewshot": 0,
            "metadata": {"version": 1.0},
        }
    )


def _hub_math_task(name: str) -> ConfigurableTask:
    """Build a zero-shot boxed-answer math task over the hub dataset its card pins."""
    card = assets.benchmark(name)
    return ConfigurableTask(
        config={
            "task": name,
            "dataset_path": card.dataset,
            "dataset_kwargs": {"revision": card.revision},
            "test_split": card.split,
            "output_type": "generate_until",
            "doc_to_text": MATH_PROMPT,
            "doc_to_target": "{{answer}}",
            "process_results": _math_results,
            "metric_list": [
                {"metric": "math_verify", "aggregation": "mean", "higher_is_better": True},
                {"metric": "boxed_match", "aggregation": "mean", "higher_is_better": True},
            ],
            "generation_kwargs": {"max_gen_toks": 32768, "temperature": 0, "do_sample": False, "until": []},
            "num_fewshot": 0,
            "metadata": {"version": 1.0},
        }
    )


def _gpqa_task() -> ConfigurableTask:
    """Return lm-eval's GPQA Diamond task with our letter filter added.

    The harness filters stay as secondary metrics. Strict-match wants the exact phrase "The answer is (X)",
    and flexible-extract takes the last "(X)" in the answer, which reads a letter out of a chemical formula
    that follows the stated answer.

    lm-eval names the dataset itself, so the task is held to the card rather than built from it.

    Raises:
        ValueError: lm-eval's task reads a dataset other than the one the card pins.
    """
    task = manager().load_task_or_group(["gpqa_diamond_cot_zeroshot"])["gpqa_diamond_cot_zeroshot"]
    card = assets.benchmark(Suite.GPQA_DIAMOND)
    read = (task.config.dataset_path, [task.config.dataset_name])
    if read != (card.dataset, card.subsets):
        raise ValueError(f"lm-eval's GPQA task reads {read} and the gpqa_diamond card pins {card.dataset}")
    task._filters.append(FilterEnsemble(name="letter", filters=[letters.LetterFilter]))
    return task


def _mmlu_pro_task() -> ConfigurableTask:
    """Build zero-shot MMLU-Pro over the hub dataset its card pins, scored by the letter filter alone.

    The prompt is GPQA's, so the two knowledge tasks ask the same way, and the target is the letter in GPQA's
    "(X)" form.
    """
    card = assets.benchmark(Suite.MMLU_PRO)
    task = ConfigurableTask(
        config={
            "task": Suite.MMLU_PRO,
            "dataset_path": card.dataset,
            "dataset_kwargs": {"revision": card.revision},
            "test_split": card.split,
            "output_type": "generate_until",
            "doc_to_text": _mmlu_pro_text,
            "doc_to_target": _mmlu_pro_target,
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                }
            ],
            "generation_kwargs": {"max_gen_toks": 32768, "temperature": 0, "do_sample": False, "until": []},
            "num_fewshot": 0,
            "metadata": {"version": 1.0},
        }
    )
    letter_filter = functools.partial(letters.LetterFilter, choices=letters.MMLU_PRO)
    task._filters = [FilterEnsemble(name="letter", filters=[letter_filter])]
    return task


def _graphwalks_task(name: str) -> ConfigurableTask:
    """Build one GraphWalks task over its hub file, at the revision the card pins.

    The prompt is the row's own, which carries its instructions and its examples. The rows come from
    `graphwalks_rows` rather than lm-eval's own `load_dataset`, which cannot find a hub file offline.

    Raises:
        ValueError: The card does not list the file the task reads.
    """
    card = assets.benchmark(Suite.GRAPHWALKS)
    file = GRAPHWALKS_FILES[name]
    if file not in card.subsets:
        raise ValueError(f"{name} reads {file} and the graphwalks card lists {card.subsets}")
    return ConfigurableTask(
        config={
            "task": name,
            "dataset_path": card.dataset,
            "custom_dataset": lambda **metadata: {card.split: graphwalks_rows(name)},
            "test_split": card.split,
            "output_type": "generate_until",
            "doc_to_text": "{{prompt}}",
            "doc_to_target": "{{answer_nodes}}",
            "target_delimiter": "",
            "process_results": graphwalks.process_results,
            "metric_list": [
                {"metric": "f1", "aggregation": "mean", "higher_is_better": True},
                {"metric": "flexible_f1", "aggregation": "mean", "higher_is_better": True},
            ],
            "generation_kwargs": {"max_gen_toks": 16384, "temperature": 0, "do_sample": False, "until": []},
            "num_fewshot": 0,
            "metadata": {"version": 1.0},
        }
    )


def resolve(name: str, data_root: str = DATA_ROOT, max_context: int | None = None) -> TaskDefinition:
    """Resolve a task name - mmlu_pro, gpqa_diamond, math500, aime25, longbench2, graphwalks_<file>, ruler_<task>_<length>
    or papertrail_<sample>_<condition>, with `_nic` for its not-in-corpus questions.

    Args:
        name: The task name.
        data_root: The fixed prompt files.
        max_context: Which LongBench v2 file to read, or None for the context its card prepares.

    Returns:
        The task.

    Raises:
        KeyError: The name is no task of the suite.
    """
    if name == Suite.MMLU_PRO:
        return TaskDefinition(
            benchmark=Suite.MMLU_PRO,
            name=name,
            category="knowledge",
            metric="exact_match:letter",
            build=_mmlu_pro_task,
            target=_mmlu_pro_target,
            extract=functools.partial(letters.letter, choices=letters.MMLU_PRO),
            question=_mmlu_pro_question,
            meta=_mmlu_pro_meta,
        )
    if name == Suite.GPQA_DIAMOND:
        return TaskDefinition(
            benchmark="gpqa",
            name=name,
            category="knowledge",
            metric="exact_match:letter",
            build=_gpqa_task,
            target=lambda doc: doc["answer"],
            extract=letters.letter,
            question=_gpqa_question,
            meta=_gpqa_meta,
        )
    if name == Suite.MATH500:
        return TaskDefinition(
            benchmark=Suite.MATH500,
            name=name,
            category="reasoning",
            metric="math_verify",
            build=lambda: _hub_math_task(Suite.MATH500),
            target=lambda doc: doc["answer"],
            extract=mathgrade.boxed_answer,
            question=_math_question,
            meta=_math500_meta,
        )
    if name == Suite.AIME25:
        return TaskDefinition(
            benchmark=Suite.AIME25,
            name=name,
            category="reasoning",
            metric="math_verify",
            build=lambda: _hub_math_task(Suite.AIME25),
            target=lambda doc: doc["answer"],
            extract=mathgrade.boxed_answer,
            question=_math_question,
            meta=_aime25_meta,
        )
    if name == Suite.LONGBENCH:
        return TaskDefinition(
            benchmark=Suite.LONGBENCH,
            name=name,
            category="long_context",
            metric="acc",
            build=lambda: _file_task(
                Suite.LONGBENCH,
                longbench_path(max_context or longbench_card().context, data_root),
                "{{prompt}}",
                "{{answer}}",
                longbench.process_results,
                "acc",
                max_gen_toks=4096,
            ),
            target=lambda doc: doc["answer"],
            extract=longbench.extract_answer,
            question=_longbench_question,
            meta=_longbench_meta,
            tail_marker=longbench_prompt.CONTEXT_END,
            path=longbench_path(max_context or longbench_card().context, data_root),
        )
    if name in GRAPHWALKS_FILES:
        return TaskDefinition(
            benchmark=Suite.GRAPHWALKS,
            name=name,
            category="long_context",
            metric="f1",
            build=lambda: _graphwalks_task(name),
            target=lambda doc: doc["answer_nodes"],
            extract=graphwalks.answer_nodes,
            question=_graphwalks_question,
            meta=_graphwalks_meta,
        )
    if name.startswith(Suite.RULER_PREFIX):
        task, length_name = name[len(Suite.RULER_PREFIX) :].rsplit("_", 1)
        if task not in ruler_card().tasks:
            raise KeyError(f"unknown RULER task {task}")
        length = int(length_name)
        return TaskDefinition(
            benchmark=Suite.RULER,
            name=name,
            category="long_context",
            metric="score",
            # The generation prefix is NVIDIA's answer prefix without its leading space - joining it back
            # with one space rebuilds their prompt exactly, the common-words task included
            build=lambda: _file_task(
                name,
                ruler_path(task, length, data_root),
                "{{input}}",
                "{{outputs}}",
                ruler.results_for(task),
                "score",
                gen_prefix="{{gen_prefix}}",
            ),
            target=lambda doc: doc["outputs"],
            question=_ruler_question,
            length=length,
            path=ruler_path(task, length, data_root),
        )
    if name.startswith(Suite.PAPERTRAIL_PREFIX):
        sample, condition, not_in_corpus = papertrail_names.parse_task(name)
        path = papertrail_path(sample, condition, data_root)
        return TaskDefinition(
            benchmark=Suite.PAPERTRAIL,
            name=name,
            category="long_context",
            metric="score",
            build=lambda: _papertrail_task(name, path, not_in_corpus),
            target=papertrail_data.answer_of,
            extract=papertrail.extract_answer,
            question=_papertrail_question,
            meta=_papertrail_meta,
            breakdown=("tier", "trail"),
            length=papertrail_names.condition_length(condition),
            path=path,
        )
    raise KeyError(f"unknown task {name}")


@dataclasses.dataclass
class Item:
    """One prompt to send and score."""

    doc_id: int
    doc: dict[str, Any]
    prompt: str
    target: Any
    instance: Any = dataclasses.field(repr=False)

    @property
    def prompt_hash(self) -> str:
        """Return the short hash a sample row names its prompt by."""
        return hashlib.sha256(self.prompt.encode()).hexdigest()[:16]


def build_items(lm_eval_task: Any, ids: list[int] | None = None, seed: int = 0) -> list[Item]:
    """Build the prompts of a task for the given doc ids (all docs when None), in id order.

    lm-eval walks the docs in dataset order and keeps those whose index is in ids, so the items come
    back in ascending id order whatever the order of ids.

    The random module is seeded first: GPQA shuffles its answer choices with the global generator, and
    the same seed must give the same letters in every run. The prompt-set hash in the record is the
    check that it did.
    """
    if ids is not None:
        ids = sorted(set(ids))
        if not ids:
            return []
    random.seed(seed)
    lm_eval_task.build_all_requests(limit=None, samples=ids)
    items = []
    for inst in lm_eval_task.instances:
        # The suite scores free-form generations alone - a loglikelihood task has no reply to grade
        if inst.request_type != "generate_until":
            raise ValueError(f"the suite runs generate_until tasks - got a {inst.request_type!r} request")
        doc_id = ids[inst.doc_id] if ids is not None else inst.doc_id
        items.append(Item(doc_id=doc_id, doc=inst.doc, prompt=inst.arguments[0], target=None, instance=inst))
    return items


def prompt_set_hash(items: list[Item]) -> str:
    """Return the short hash of a prompt set, which a record holds to the pinned one."""
    joined = "".join(item.prompt_hash for item in items)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def grade(lm_eval_task: Any, items: list[Item], replies: list[str]) -> list[dict[str, float]]:
    """Score one reply per item through the task's own filters and process_results.

    Args:
        lm_eval_task: The lm-eval task the items were built from.
        items: The prompts.
        replies: One answer per item.

    Returns:
        One dict per item, metrics keyed `metric`, or `metric:filter` when the task has named filters.

    Raises:
        ValueError: There is not one reply per item.
    """
    if len(items) != len(replies):
        raise ValueError(f"one reply per item - got {len(replies)} replies for {len(items)} items")
    # The filters walk every instance of the task - instances without a reply (a failed request) get an
    # empty one so the filters do not trip over them
    for inst in lm_eval_task.instances:
        inst.resps = [""]
    for item, reply in zip(items, replies, strict=False):
        item.instance.resps = [reply]
    lm_eval_task.apply_filters()
    out = []
    for item in items:
        metrics: dict[str, float] = {}
        for key, filtered in item.instance.filtered_resps.items():
            for metric, value in lm_eval_task.process_results(item.doc, [filtered]).items():
                metrics[metric if key == "none" else f"{metric}:{key}"] = float(value)
        out.append(metrics)
    return out
