"""The tasks a benchmark runs and the presets that name them.

A preset mirrors the reference training code's `core` and RULER sets. The core tasks are lm-eval's own, read from the hub cache
the offline contract names. RULER is NVIDIA's own generators' output, and the choice is deliberate.

lm-eval ports RULER's needle-in-a-haystack tasks and the reference training code ran those, sized by a tokenizer that is not the
model's. The prompts here are the ones `lsa data prepare ruler` makes from the pinned NVIDIA clone and sizes by the
model's own tokenizer, so a prompt of "16,384 tokens" is 16,384 of this model's, and the files, their seed and their
grading are the ones the served accuracy suite uses. The one thing a Qwen3.5 run needs that the card's own files lack
is its tokenizer's rendering, so the files are prepared into a root of their own and read from there.
"""

import os
from typing import Any

from lsa import paths
from lsa.evaluation.accuracy.data import ruler_card, ruler_path
from lsa.evaluation.accuracy.grading import ruler

# The tasks the reference training code's `core` preset reads that fit a short context, and sciq beside them
CORE_TASKS: tuple[str, ...] = (
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "lambada_openai",
    "wikitext",
    "sciq",
)

# The reference training code's RULER_TASKS - the card's other needle task, multikey 3, and its six non-needle tasks are left out
RULER_TASKS: tuple[str, ...] = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multiquery",
    "niah_multivalue",
)

# The lengths the native models were trained at and below, and the one past it
TRAINED_LENGTHS: tuple[int, ...] = (4096, 8192, 16384)
EXTRAPOLATED_LENGTHS: tuple[int, ...] = (32768,)

# Where a Qwen3.5 rendering of the card's grid lives, beside the card's own root
RULER_ROOT_NAME = "benchmarks-qwen35"

RULER_PREFIX = "ruler_"


def ruler_root() -> str:
    """Return the root the Qwen3.5 rendering of RULER is read from - `LSA_RULER_ROOT`, else beside the card's own."""
    return os.environ.get("LSA_RULER_ROOT") or str(paths.data() / RULER_ROOT_NAME)


def ruler_name(task: str, length: int) -> str:
    """Return the lm-eval task name of one RULER cell."""
    return f"{RULER_PREFIX}{task}_{length}"


def ruler_cells(lengths: tuple[int, ...]) -> list[str]:
    """Return every needle task at each length, shortest length first."""
    return [ruler_name(task, length) for length in lengths for task in RULER_TASKS]


PRESETS: dict[str, list[str]] = {
    "core": list(CORE_TASKS),
    "ruler": ruler_cells(TRAINED_LENGTHS),
    # Opt-in: the models never trained at this length
    "ruler-32k": ruler_cells(EXTRAPOLATED_LENGTHS),
}


def expand(tasks: str) -> list[str]:
    """Expand a preset name or a comma-separated list, mixing presets and tasks, into task names.

    Args:
        tasks: A preset, a task name, or several of either joined by commas.

    Returns:
        The task names, each once, in the order they were first named.
    """
    names: list[str] = []
    for item in tasks.split(","):
        for name in PRESETS.get(item.strip(), [item.strip()]):
            if name and name not in names:
                names.append(name)
    return names


def is_ruler(name: str) -> bool:
    """Say whether a task name is a RULER cell."""
    return name.startswith(RULER_PREFIX)


def parse_ruler(name: str) -> tuple[str, int]:
    """Split a RULER cell's name into its task and length.

    Args:
        name: A name `ruler_name` made.

    Returns:
        The RULER task and the context length in tokens.

    Raises:
        ValueError: The name is not a cell of the card's grid.
    """
    task, _, length = name[len(RULER_PREFIX) :].rpartition("_")
    if task not in ruler_card().tasks or not length.isdigit():
        raise ValueError(f"{name} is not a RULER cell - expected ruler_<task>_<length>")
    return task, int(length)


def build_ruler(name: str, root: str) -> Any:
    """Build one RULER cell as an lm-eval task over its fixed file.

    The prompt is the file's `input` and the generation prefix its `gen_prefix`, joined by lm-eval with a space, which
    is how the NVIDIA prompt reads. The reply is graded by RULER's own string match.

    Args:
        name: The cell, as `ruler_name` spells it.
        root: The data root the file lives under.

    Returns:
        The lm-eval task.
    """
    from lm_eval.api.task import ConfigurableTask

    task, length = parse_ruler(name)
    config: dict[str, Any] = {
        "task": name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": ruler_path(task, length, root)}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{input}}",
        "doc_to_target": "{{outputs}}",
        "gen_prefix": "{{gen_prefix}}",
        "target_delimiter": " ",
        "process_results": ruler.results_for(task),
        "metric_list": [{"metric": "score", "aggregation": "mean", "higher_is_better": True}],
        "generation_kwargs": {"max_gen_toks": 128, "temperature": 0, "do_sample": False, "until": []},
        "num_fewshot": 0,
        "metadata": {"version": 1.0},
    }
    return ConfigurableTask(config=config)


def build(names: list[str], root: str | None = None) -> list[Any]:
    """Resolve task names to what lm-eval's evaluator takes - a name, or a built task for a RULER cell.

    Args:
        names: Task names, RULER cells as `ruler_<task>_<length>`.
        root: The RULER data root, or None for `ruler_root()`.

    Returns:
        The tasks, in order.
    """
    return [build_ruler(name, root or ruler_root()) if is_ruler(name) else name for name in names]


def lengths_of(names: list[str]) -> list[int]:
    """Return the distinct RULER lengths among the tasks, ascending."""
    return sorted({parse_ruler(name)[1] for name in names if is_ruler(name)})
