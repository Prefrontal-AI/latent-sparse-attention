"""Grading of math answers - two graders, both recorded, so a disagreement is visible.

math_verify: parses the last boxed (or last expression) of the reply and the gold answer and checks
symbolic equivalence - 0.5 equals 1/2. This is the primary metric.
boxed_match: lm-eval's own AIME/MATH grader - last \\boxed{} of the reply, string normalization, exact
string equality. Kept as the secondary metric.
"""

# lm_eval.tasks.aime is a namespace package, so mypy cannot see `utils` until it is imported
# => the submodule resolves at runtime, and `unused-ignore` covers a sync without the evaluation extra
from lm_eval.tasks.aime import utils as aime_utils  # type: ignore[attr-defined, unused-ignore]
from math_verify import parse, verify


def boxed_answer(text):
    """The content of the last \\boxed{...} in the text, or None."""
    boxed = aime_utils.last_boxed_only_string(text)
    if boxed is None:
        return None
    try:
        return aime_utils.remove_boxed(boxed)
    except (AssertionError, IndexError):
        return None


def boxed_match(reply, gold):
    answer = boxed_answer(reply)
    return 1 if answer is not None and aime_utils.is_equiv(answer, str(gold)) else 0


def math_verify_match(reply, gold):
    """gold is the final answer as a string - wrapped in \\boxed so the parser treats it as an answer.

    The reply's answer is its last \\boxed{} - math_verify alone would merge several boxes into a set.
    Without a box the whole reply goes to the parser, which then looks for a final-answer phrase.
    """
    boxed = boxed_answer(reply)
    candidate = "\\boxed{" + boxed + "}" if boxed is not None else reply
    try:
        parsed_gold = parse("\\boxed{" + str(gold) + "}")
        parsed_reply = parse(candidate)
        return 1 if parsed_gold and parsed_reply and verify(parsed_gold, parsed_reply) else 0
    # Broad on purpose - a parser failure is a wrong answer, never a crash
    except Exception:
        return 0


def grade(reply, gold):
    return {"math_verify": math_verify_match(reply, gold), "boxed_match": boxed_match(reply, gold)}
