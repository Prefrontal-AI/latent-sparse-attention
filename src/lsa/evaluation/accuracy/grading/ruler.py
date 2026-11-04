"""RULER's official scoring (scripts/eval/synthetic/constants.py) over lm-eval's answer post-processing."""

from collections.abc import Callable

# The QA tasks carry several gold answers and any one counts; every other task scores the fraction found
ANY_REFERENCE = ("qa_squad", "qa_hotpot")


def string_match_all(pred, refs):
    """Official RULER: the fraction of references found in the answer."""
    return sum(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs) / len(refs)


def string_match_part(pred, refs):
    """Official RULER: any reference found counts. lm-eval's port takes the max over predictions instead of
    over references, which under-scores the QA tasks when several gold answers exist - so not used."""
    return max(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs)


def results_for(task: str, any_reference: tuple[str, ...] = ANY_REFERENCE) -> Callable[[dict, list], dict]:
    """Return the process_results a RULER task is graded by - its string match over the first reply.

    Args:
        task: The RULER task.
        any_reference: The tasks a reply scores 1 on when any one reference is found, the rest scoring the
            fraction found. Empty grades every task by the fraction, as the paper's native RULER column did.

    Returns:
        lm-eval's `process_results` for the task.
    """
    # lm-eval's module reaches transformers and torch, so only a task that is graded loads it
    from lm_eval.tasks.ruler import common_utils as ruler_common

    match = string_match_part if task in any_reference else string_match_all

    def process_results(doc, results):
        return {"score": match(ruler_common.postprocess_pred(results)[0], doc["outputs"])}

    return process_results
