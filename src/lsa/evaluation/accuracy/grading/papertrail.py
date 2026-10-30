"""PaperTrail answers, read back and scored the way the benchmark's own grader (papertrail/grade.py) does.

The reply ends with `EVIDENCE: Paper NNNN, ...` and `ANSWER: ...`, and only the last ANSWER line is scored, one
number in [0, 1] per question by its answer type:

| type | correct when |
|---|---|
| number | the line's first number is within `tol` of the gold, relative - 1.2e5, 1.2 x 10^5, 1.2×10^{5}, 3,180 and 0.612% all read |
| text | the normalised line contains any accepted string |
| ids | set F1 between the papers the line names and the gold set - arXiv ids, or `Paper 0412` mapped through the context's paper order |
| abstain | the line says NOT IN CORPUS or an equivalent |

A question that is not an abstain scores 0 on an abstaining answer. The behaviour is the benchmark's verbatim,
and the reference replies it graded regrade to the same 540 scores.
"""

import json
import re
from typing import Any

ANSWER_LINE = re.compile(r"ANSWER\s*[:：]\s*(.+)")
ABSTAIN = re.compile(
    r"not\s+in\s+(the\s+)?(corpus|collection)|no\s+such|not\s+(mentioned|reported|present|found)|"
    r"does\s+not\s+(appear|exist|contain)|cannot\s+be\s+(found|determined)",
    re.I,
)
NUMBER = re.compile(
    r"(?<![A-Za-z_\d.])(?:10\^\{?[-−]?\d+\}?|[-−]?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:[x×]|\\times)\s*10\^?\{?\s*[-−]?\d+\}?|[eE][-+]?\d+)?)"
)
ARXIV_ID = re.compile(r"\b(\d{4}\.\d{5})\b")
PAPER_NUMBER = re.compile(r"\bPaper\s+0*(\d{1,5})\b", re.I)
BARE_PAPER_NUMBER = re.compile(r"(?<![\d.])0*(\d{1,5})(?![\d.])")
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
# The relative tolerance of a number whose question states none
DEFAULT_TOLERANCE = 0.02


def answer_line(reply: str) -> str:
    """Return the last ANSWER line's answer, or the reply's last non-empty line when it states none."""
    matches = list(ANSWER_LINE.finditer(reply or ""))
    if matches:
        return matches[-1].group(1).strip()
    lines = [line for line in (reply or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_number(text: str) -> float | None:
    """Return the first number written in a line, scientific forms included, or None."""
    text = text.replace(" ", "").replace("\\,", "").replace("{,}", ",").replace("$", "")
    # 7×10²⁹
    text = re.sub(
        r"10([⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda match: "10^" + match.group(1).translate(SUPERSCRIPTS), text
    )
    for match in NUMBER.finditer(text):
        token = match.group(0).replace("−", "-").replace(",", "")
        power = re.fullmatch(r"10\^\{?(-?\d+)\}?", token)
        if power:
            return 10.0 ** int(power.group(1))
        mantissa = re.match(r"(-?[\d.]+)\s*(?:[x×]|\\times)\s*10\^?\{?\s*(-?\d+)\}?", token)
        try:
            return float(mantissa.group(1)) * 10 ** int(mantissa.group(2)) if mantissa else float(token)
        except ValueError:
            continue
    return None


def normalise(text: str) -> str:
    """Lowercase a line and fold everything but letters, digits and `.+[]` into single spaces."""
    return re.sub(r"[^a-z0-9.+\[\]]+", " ", text.lower()).strip()


def named_papers(line: str, paper_ids: dict[int, str]) -> set[str]:
    """Return the papers an answer line names - arXiv ids, and paper numbers mapped through the context's order.

    Once a paper is named by number, bare numbers count too - "Paper 0012, 0040, Paper 7".
    """
    named = set(ARXIV_ID.findall(line))
    numbers = PAPER_NUMBER.findall(line)
    if numbers:
        numbers += BARE_PAPER_NUMBER.findall(PAPER_NUMBER.sub(" ", line))
    return named | {paper_ids[int(number)] for number in numbers if int(number) in paper_ids}


def score(doc: dict[str, Any], reply: str, paper_ids: dict[int, str] | None = None) -> dict[str, Any]:
    """Score one reply against its question's answer key.

    Args:
        doc: The question - `answer_type`, `answer`, and `tol`, `unit` and `answer_ids` where its type reads them.
        reply: The model's reply.
        paper_ids: Paper number in the context to paper id, for answers that name papers by number.

    Returns:
        `score` in [0, 1] and `extracted`, the answer line read, with what each type measured beside them.

    Raises:
        ValueError: The question's answer type is none of the four.
    """
    line = answer_line(reply)
    abstained = bool(ABSTAIN.search(line))
    answer_type = doc["answer_type"]
    out: dict[str, Any] = {"extracted": line[:300], "score": 0.0}
    if answer_type == "abstain":
        out["score"] = float(abstained)
        return out
    if abstained:
        return out
    if answer_type == "number":
        value = parse_number(line)
        gold, tolerance = float(doc["answer"]), float(doc.get("tol", DEFAULT_TOLERANCE))
        out["value"] = value
        # 0.612% may be written 0.00612
        candidates = [value] if value is None or doc.get("unit") != "%" else [value, 100 * value]
        out["score"] = float(
            any(
                candidate is not None and abs(candidate - gold) <= tolerance * abs(gold) + 1e-12
                for candidate in candidates
            )
        )
    elif answer_type == "text":
        accepted = doc["answer"] if isinstance(doc["answer"], list) else [doc["answer"]]
        out["score"] = float(any(normalise(spelling) in normalise(line) for spelling in accepted))
    elif answer_type == "ids":
        named = named_papers(line, paper_ids) if paper_ids else set(ARXIV_ID.findall(line))
        gold_ids = set(doc["answer_ids"])
        hits = len(named & gold_ids)
        precision = hits / len(named) if named else 0.0
        recall = hits / len(gold_ids) if gold_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update(pred_ids=sorted(named), precision=precision, recall=recall, score=f1)
    else:
        raise ValueError(f"unknown answer_type {answer_type}")
    return out


def extract_answer(reply: str) -> str:
    """Return what the grader reads of a reply - its last ANSWER line."""
    return answer_line(reply)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score one reply of a task's doc, whose `papers` are the context's paper ids in order.

    A doc carries its answer as json in `answer_json`, since lm-eval builds the docs into an Arrow table.
    """
    if "answer_json" in doc:
        doc = {**doc, "answer": json.loads(doc["answer_json"])}
    reply = results[0] if results else ""
    paper_ids = {position + 1: paper_id for position, paper_id in enumerate(doc.get("papers") or [])}
    return {"score": score(doc, reply, paper_ids)["score"]}
