"""The prompt-set hash of every plan - what makes two records' prompts the same prompts.

    python -m lsa.evaluation.accuracy --hashes

A run recomputes the hash of each plan it sends and stops on one that differs from `hashes.json`. A pin
goes stale when the corpus is regenerated or a hub dataset's rendering changes - delete that key and fill
it again, which touches only the keys that are missing.
"""

from collections.abc import Callable
from typing import Any

from lsa import io
from lsa.evaluation.accuracy import tasks
from lsa.evaluation.accuracy.data import DATA_ROOT, suites


def make_hashes(
    suite_names: list[str] | tuple[str, ...] | None = None,
    data_root: str = DATA_ROOT,
    max_context: int | None = None,
    log: Callable[[str], Any] = print,
    pins: str | None = None,
) -> dict[str, str]:
    """Compute the prompt-set hash of every unpinned plan whose data exists - the values the runner checks.

    Args:
        suite_names: The suites to cover, every one when None.
        data_root: The fixed prompt files.
        max_context: Which LongBench v2 file to read, or None for the context its card prepares.
        log: Where progress lines go.
        pins: Which set of pins to fill (`suites.pins_path`), None for the default corpus's.

    Returns:
        Plan key to prompt-set hash, the pins already in the file included.
    """
    path = suites.pins_path(pins)
    out = io.read_json(path, {})
    for suite_name in suite_names or suites.SUITES:
        for plan in suites.SUITES[suite_name]():
            if plan.key in out:
                continue
            definition = tasks.resolve(plan.task, data_root, max_context)
            try:
                task = definition.build()
            # Broad on purpose - any failure here is data that is not generated yet
            except Exception as exc:
                log(f"skip {plan.key}: {type(exc).__name__}")
                continue
            out[plan.key] = tasks.prompt_set_hash(tasks.build_items(task, plan.ids()))
            log(f"{plan.key}: {out[plan.key]}")
    io.write_json(path, dict(sorted(out.items())), indent=1)
    return out
