"""The letter a multiple-choice answer names.

lm-eval's flexible filter takes the last "(X)" in the answer, which reads "(H)" out of a chemical formula
that follows the stated answer. We first look for the last final-answer phrase - "the answer is (B)",
"Answer: **(B)**", "correct choice is (B)", "**(B) 10 eV**", "\\boxed{\\text{(B) ...}}" - and only
without one fall back to the last "(X)". The letters are a task's own - A-D for GPQA, A-J for MMLU-Pro - and
the result is "(X)" like the target.
"""

import functools
import re

from lm_eval.api.filter import Filter

GPQA = "ABCD"
MMLU_PRO = "ABCDEFGHIJ"

# ornaments that may sit between a phrase and its letter: bold, math, braces, colon
_GAP = r"[\s:：*\\\[\]\(\{$]*?(?:text\{|boxed\{)?[\s*]*"


@functools.cache
def _patterns(choices: str) -> tuple[tuple[re.Pattern[str], ...], re.Pattern[str]]:
    """Return the final-answer patterns for these letters, and the last-"(X)" fallback."""
    one = f"[{choices}]"
    phrase = re.compile(
        r"(?:answer|choice|option)\s*(?:is|:|：|would be|should be)?" + _GAP + rf"\(?({one})\)",
        re.IGNORECASE,
    )
    bold = re.compile(rf"\*\*\(({one})\)")
    text = re.compile(rf"\\text\{{\s*\(({one})\)")
    # a boxed bare letter: \boxed{C}, \boxed{\text{C}}, \boxed{(C)}
    boxed = re.compile(rf"\\boxed\{{\s*(?:\\text\{{)?\s*\(?({one})\)?\s*\}}?\s*\}}")
    return (phrase, bold, text, boxed), re.compile(rf"\(({one})\)")


def letter(answer, choices: str = GPQA):
    """ "(X)" or None, X one of choices."""
    if not answer:
        return None
    phrases, fallback = _patterns(choices)
    last = None
    for pattern in phrases:
        for match in pattern.finditer(answer):
            if last is None or match.start() > last[0]:
                last = (match.start(), match.group(1))
    if last is not None:
        return f"({last[1].upper()})"
    found = fallback.findall(answer)
    return f"({found[-1].upper()})" if found else None


class LetterFilter(Filter):
    """lm-eval filter: one letter per response, "" when none is found."""

    def __init__(self, choices: str = GPQA, **kwargs) -> None:
        super().__init__(**kwargs)
        self.choices = choices

    def apply(self, resps, docs):
        return [[letter(r, self.choices) or "" for r in resp] for resp in resps]
