"""LongBench v2 answers, read back the way the official evaluation (THUDM/LongBench, pred.py) reads them."""

import re


def extract_answer(response):
    """pred.py extract_answer, verbatim in behaviour: the first 'The correct answer is (X)' or '... is X'."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def process_results(doc, results):
    return {"acc": 1.0 if extract_answer(results[0]) == doc["answer"] else 0.0}
