"""GraphWalks answers, read back and scored the way OpenAI's grader on the dataset card does.

The answer is the last line, `Final Answer: [a, b]`, and the score is the F1 of its set against the true one.
A reply that states no such line scores 0, and an empty answer to an empty true set scores 1 - which is where
lm-eval's port differs, scoring it 0 on the 113 of 750 short prompts and 31 of 400 long ones whose answer is
empty. The one liberty taken is trailing whitespace, which the card's grader would read as an empty last line.

`flexible_f1` is a secondary reading - the last `Final Answer: [...]` on any line, quotes stripped off the ids.
"""

import re
from typing import Any

FINAL_ANSWER = "Final Answer:"
# Where a prompt's operation starts - the last one, after the examples. The viewer finds the question by it
OPERATION = "Operation:\n"
STATED_LIST = re.compile(r"Final Answer:\s*\[(.*)\]")


def split_nodes(listed: str, quotes: str = "") -> list[str]:
    """Split the inside of a stated list into node ids, dropping empty entries and the given quote marks."""
    return [item.strip().strip(quotes) for item in listed.split(",") if item.strip().strip(quotes)]


def answer_nodes(response: str) -> list[str] | None:
    """Return the node ids on the reply's last line, as the card's grader reads them, or None when it states none."""
    line = response.rstrip().split("\n")[-1]
    if FINAL_ANSWER not in line:
        return None
    listed = re.search(r"\[.*\]", line)
    if listed is None:
        return None
    return split_nodes(listed.group(0).strip("[]"))


def flexible_answer_nodes(response: str) -> list[str] | None:
    """Return the node ids of the last line anywhere that states a list, or None when none does."""
    for line in reversed(response.rstrip().split("\n")):
        stated = STATED_LIST.search(line)
        if stated:
            return split_nodes(stated.group(1), quotes="'\"`")
    return None


def f1(predicted: list[str] | None, gold: list[str]) -> float:
    """Return the set F1 of a parsed answer against the true nodes - 0 when nothing parsed, 1 when both are empty."""
    if predicted is None:
        return 0.0
    predicted_set, gold_set = set(predicted), set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    overlap = len(predicted_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score one reply against its doc's `answer_nodes`."""
    gold = doc["answer_nodes"]
    return {
        "f1": f1(answer_nodes(results[0]), gold),
        "flexible_f1": f1(flexible_answer_nodes(results[0]), gold),
    }
