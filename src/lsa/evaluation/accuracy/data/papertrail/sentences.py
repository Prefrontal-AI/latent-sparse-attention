"""The fixed sentence split a patch is anchored by - "after sentence k of this paragraph, insert ...".

The split has to be the one the snippet was written against, so it is a rule with no model or library in it:

- maths (`$...$`) and citation placeholders (`⟦...⟧`) are never split;
- a sentence ends at `.`, `?` or `!`, optionally followed by a closing quote or bracket, then whitespace and an
  upper-case letter, a digit, `(`, `[`, `$` or `⟦`;
- a common abbreviation (et al., e.g., i.e., Fig., Eq., Sect., Ref., cf., vs., approx., No., resp., or a single
  capital initial such as `J.`) never ends one.
"""

import re

PROTECTED = re.compile(r"\$[^$]*\$|⟦[^⟧]*⟧")
ABBREVIATION = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs|\bapprox|\bresp|\bFigs?|\bEqs?|\bSects?|\bSecs?|\bRefs?|\bTabs?|\bApps?"
    r"|\bNo|\bNos|\bVol|\bpp|\bDr|\bProf|\bMr|\bMs|\bSt|\bJr|\bal|\bca|\bviz|\bwrt|\b[A-Z])\.$"
)
SENTENCE_END = re.compile(r"[.?!][\"'”’)\]]*\s+(?=[A-Z0-9(\[$⟦])")


def split(text: str) -> list[str]:
    """Split a block's text into sentences, which joined by single spaces give back the text up to whitespace."""
    masked = list(text)
    for match in PROTECTED.finditer(text):
        for position in range(match.start(), match.end()):
            if masked[position] in ".?! ":
                masked[position] = "_"
    masked_text = "".join(masked)
    sentences, start = [], 0
    for match in SENTENCE_END.finditer(masked_text):
        cut = match.end()
        head = masked_text[start : match.start() + 1].rstrip()
        if ABBREVIATION.search(head):
            continue
        sentences.append(text[start:cut].strip())
        start = cut
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def insert_after(text: str, sentence: int, inserted: str) -> str:
    """Insert one or more sentences after sentence `sentence` of a block, 0 for before the first.

    Raises:
        ValueError: The block has fewer sentences than the anchor names.
    """
    sentences = split(text)
    if not 0 <= sentence <= len(sentences):
        raise ValueError(f"sentence index {sentence} out of range (block has {len(sentences)} sentences)")
    return " ".join(sentences[:sentence] + [inserted.strip()] + sentences[sentence:])
