"""The tasks a benchmark runs and the presets that name them.

The presets follow the paper's native comparison: `core` is the short-task set, `ruler` the thirteen RULER tasks at the
lengths the native models trained at, and `ruler-4k` the same thirteen at 4,096 tokens, the paper's RULER column.
The core tasks are lm-eval's own, read from the hub cache the offline contract names.
RULER is lm-eval's port of it, rendered by `lsa native prepare-ruler` with the GPT-2 tokenizer the native models use,
so a prompt of "4,096 tokens" is 4,096 of the model's. A model with another tokenizer needs files of its own.
It is not `lsa data prepare ruler`'s output - NVIDIA's generators draw other needles and haystacks, and a checkpoint
scores about 5 points lower on their files over the seven needle tasks other than niah_multikey_3, mostly on niah_single_3.
The files are read from a root of their own, and are graded by RULER's own string match, as the served accuracy suite
grades its files.
"""

import os
from typing import Any

from lsa import paths
from lsa.evaluation.accuracy.data import ruler_card, ruler_path
from lsa.evaluation.accuracy.grading import ruler

# The short tasks the paper's core set reads that fit a short context, and sciq beside them
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

# The thirteen RULER tasks the paper's RULER column averages
RULER_TASKS: tuple[str, ...] = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_hotpot",
    "qa_squad",
)

# The lengths the native models were trained at and below, and the one past it
TRAINED_LENGTHS: tuple[int, ...] = (4096, 8192, 16384)
EXTRAPOLATED_LENGTHS: tuple[int, ...] = (32768,)

RULER_PREFIX = "ruler_"

# The tokenizer the rendered files are counted in, and the root named for it beside the card's own
RULER_TOKENIZER = "gpt2"
ROOT_NAME = "benchmarks-gpt2"


def ruler_root() -> str:
    """Return the root the GPT-2 rendering of RULER is read from - `LSA_RULER_ROOT`, else beside the card's own."""
    return os.environ.get("LSA_RULER_ROOT") or str(paths.data() / ROOT_NAME)


def check_ruler_tokenizer(names: list[str], tokenizer: str) -> None:
    """Refuse RULER cells for a model whose tokenizer is not the one the files are counted in.

    Args:
        names: Task names, as `expand` returns them.
        tokenizer: The model card whose tokenizer the experiment uses.

    Raises:
        ValueError: A RULER cell is named and the tokenizer is not GPT-2.
    """
    if tokenizer != RULER_TOKENIZER and any(is_ruler(name) for name in names):
        raise ValueError(
            f"the RULER files are counted in the {RULER_TOKENIZER} tokenizer and this experiment uses {tokenizer}, "
            f"so its prompts would be mis-sized - prepare files for its own tokenizer or score a {RULER_TOKENIZER} experiment"
        )


def ruler_name(task: str, length: int) -> str:
    """Return the lm-eval task name of one RULER cell."""
    return f"{RULER_PREFIX}{task}_{length}"


def ruler_cells(lengths: tuple[int, ...]) -> list[str]:
    """Return every RULER task at each length, shortest length first."""
    return [ruler_name(task, length) for length in lengths for task in RULER_TASKS]


PRESETS: dict[str, list[str]] = {
    "core": list(CORE_TASKS),
    "ruler": ruler_cells(TRAINED_LENGTHS),
    # The paper's RULER column, the 13 tasks at 4,096 tokens
    "ruler-4k": ruler_cells((4096,)),
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
        # The paper's native column scored the QA tasks by the fraction of references found, as every other task
        # => graded so here, where the served suite counts any one reference
        "process_results": ruler.results_for(task, any_reference=()),
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
