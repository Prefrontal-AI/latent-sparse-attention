"""The benchmark's synthetic text inserted into a real paper - insertions only, never a change to the authors' text.

A snippet is one of two kinds:

    patch       {node, kind: "patch", host: <arXiv id>,
                 ops: [{op: "insert_sentences", block, after: k, text} | {op: "insert_paragraph", after_block, text}],
                 new_bib: [{key: "new.<node>", cites_node, text, tag}]}
    fake_paper  {node, kind: "fake_paper", date, title, authors, abstract, sections, bib, bib_style}

Numeric-style text cites a new reference by the placeholder `⟦new.<node>⟧`. The reference goes where a real author
would have put it - alphabetically in a sorted bibliography, in order of first citation otherwise - and the list is
renumbered, so no number in the text goes stale.
"""

import copy
import re
import unicodedata
from typing import Any

from lsa.evaluation.accuracy.data.papertrail.convert import CITATION
from lsa.evaluation.accuracy.data.papertrail.sentences import insert_after

# The invisible separator that brackets inserted text until it is rendered - then removed, its offsets kept
MARK = "⁣"

SURNAME_PARTICLES: tuple[str, ...] = (
    "van",
    "von",
    "der",
    "den",
    "de",
    "da",
    "di",
    "du",
    "le",
    "la",
    "dos",
    "del",
)
COLLECTIVE_NAMES: tuple[str, ...] = ("team", "collaboration", "collaborators", "consortium", "group")


def surname(entry: str) -> str:
    """Return a reference's first-author surname, lowercase, for alphabetical order.

    'J. D. Turner, ...' is 'turner', 'A. Afzal (2026). On ...' 'afzal', 'Gemma Team' 'gemma' and
    'K. van der Berg' 'berg'.
    """
    head = re.split(r"\s\(\d{4}|,|;| and |(?<![A-Z])\. (?=[A-Z][a-z])", entry, maxsplit=1)[0]
    raw_tokens = [token for token in re.split(r"\s+", head.strip()) if token]
    had_initials = any(re.fullmatch(r"(?:[A-Z]\.-?)+", token) for token in raw_tokens)
    tokens = [token for token in raw_tokens if not re.fullmatch(r"(?:[A-Z]\.-?)+", token)]
    # Vancouver style "Smith J"
    if len(tokens) > 1 and re.fullmatch(r"[A-Z]{1,3}", tokens[-1]):
        tokens.pop()
        had_initials = True
    # "T. Wolf et al (2020)"
    while len(tokens) > 1 and tokens[-1].lower().strip(".") in ("et", "al"):
        tokens.pop()
    if tokens and tokens[-1].lower() in COLLECTIVE_NAMES and len(tokens) > 1:
        return tokens[0].strip(".").lower()
    # The first surname token sorts a compound surname - "L. Aghababaie Beni" is "aghababaie"
    while len(tokens) > 1 and tokens[0] in SURNAME_PARTICLES:
        tokens.pop(0)
    # Full names without initials put the surname last - unless it is "Surname Surname, I.", a compound surname first
    rest = entry.lstrip()[len(head.strip()) :]
    surname_first = bool(re.match(r",\s*(?:[A-Z]\.|[A-Z][a-z]+,?\s+[A-Z]\.)", rest))
    if (
        not had_initials
        and not surname_first
        and len(tokens) > 1
        and all(token[:1].isupper() for token in tokens)
    ):
        return tokens[-1].strip(".").lower()
    return (tokens[0] if tokens else head).strip(".").lower()


def is_sorted(bib: list[dict[str, Any]]) -> bool:
    """Return whether a bibliography is alphabetical by first author, 85% of its neighbours in order."""
    names = [surname(entry["text"]) for entry in bib]
    in_order = sum(first <= second for first, second in zip(names, names[1:], strict=False))
    return in_order >= 0.85 * max(1, len(names) - 1)


def numbers_contiguous(bib: list[dict[str, Any]]) -> bool:
    """Return whether a numeric bibliography is numbered 1, 2, 3 ... in order."""
    return [entry["tag"] for entry in bib] == [str(position + 1) for position in range(len(bib))]


def sort_key(text: str) -> str:
    """Return a reference's surname folded to ascii, the key alphabetical order compares."""
    return unicodedata.normalize("NFKD", surname(text)).encode("ascii", "ignore").decode().lower()


def alphabetical_position(bib: list[dict[str, Any]], text: str) -> int:
    """Return the place for a new entry whose neighbours sort before and after it.

    Of the places that qualify, the one closest to the count of entries sorting before it - which alone misplaces an
    entry in a bibliography with a few out of order - and that count where none qualifies.
    """
    keys = [sort_key(entry["text"]) for entry in bib]
    new_key = sort_key(text)
    estimate = sum(key <= new_key for key in keys)
    qualifying = [
        position
        for position in range(len(keys) + 1)
        if (position == 0 or keys[position - 1] <= new_key)
        and (position == len(keys) or new_key <= keys[position])
    ]
    return min(qualifying, key=lambda position: abs(position - estimate)) if qualifying else estimate


def insert_reference(paper: dict[str, Any], reference: dict[str, Any], anchor_block: str) -> None:
    """Add a new reference to a paper's bibliography where its author would have, renumbering a numeric one.

    Args:
        paper: The structured paper, changed in place.
        reference: The new entry - its key, its text, and its tag in an author-year bibliography.
        anchor_block: The block that first cites it, which places it in a list ordered by first citation.

    Raises:
        ValueError: A numeric bibliography is not numbered 1, 2, 3 ... in order.
    """
    entry = {
        "key": reference["key"],
        "tag": reference.get("tag", ""),
        "text": reference["text"],
        "synthetic": reference.get("cites_node", True),
    }
    bib = paper["bib"]
    if paper["bib_style"] != "numeric":
        bib.insert(alphabetical_position(bib, reference["text"]), entry)
        return
    if not numbers_contiguous(bib):
        raise ValueError(f"{paper['id']}: numeric bibliography not contiguous")
    if is_sorted(bib):
        position = alphabetical_position(bib, reference["text"])
    else:
        # Order of first citation - after the highest number cited up to the anchor
        position = highest_cited_before(paper, anchor_block)
    bib.insert(position, entry)
    for number, renumbered in enumerate(bib):
        renumbered["tag"] = str(number + 1)


def highest_cited_before(paper: dict[str, Any], anchor_block: str) -> int:
    """Return the highest reference number a paper cites up to and including one block."""
    highest = 0
    for section in paper["sections"]:
        for block in section["blocks"]:
            for match in CITATION.finditer(block["text"]):
                tag = next((entry["tag"] for entry in paper["bib"] if entry["key"] == match.group(1)), None)
                if tag and tag.isdigit():
                    highest = max(highest, int(tag))
            if block["id"] == anchor_block:
                return highest
    return highest


def apply_patch(paper: dict[str, Any], snippet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Insert one patch's text into a copy of its host, each insertion bracketed by `MARK`.

    `insert_sentences` goes after sentence k of a paragraph, 0 for before the first, and `insert_paragraph` adds a
    paragraph after a block. Several insertions into one paragraph name its original sentences, so the highest is
    applied first.

    Args:
        paper: The structured host paper, left as it is.
        snippet: The patch.

    Returns:
        The patched paper.
        The ids of the blocks that carry its text.

    Raises:
        ValueError: An operation is empty, names a block the paper does not have or a sentence its block does not
            have, inserts sentences into a block that is not a paragraph, or is neither kind of insertion.
    """
    patched = copy.deepcopy(paper)
    node, host = snippet["node"], snippet.get("host")

    def locate() -> dict[str, tuple[int, int]]:
        """Return each block's section index and block index, by block id."""
        return {
            block["id"]: (section_index, block_index)
            for section_index, section in enumerate(patched["sections"])
            for block_index, block in enumerate(section["blocks"])
        }

    blocks = locate()
    operations = list(enumerate(snippet["ops"]))
    operations.sort(
        key=lambda numbered: (
            numbered[1]["op"] != "insert_sentences",
            numbered[1].get("block", ""),
            -numbered[1].get("after", 0),
        )
    )
    touched: list[str] = []
    for number, operation in operations:
        text = operation["text"].strip()
        if not text:
            raise ValueError(f"{node}: empty patch operation")
        if operation["op"] == "insert_sentences":
            if operation["block"] not in blocks:
                raise ValueError(f"{node}: anchor block {operation['block']} not found in {host}")
            section_index, block_index = blocks[operation["block"]]
            block = patched["sections"][section_index]["blocks"][block_index]
            if block["kind"] != "p":
                raise ValueError(
                    f"{node}: insert_sentences into the non-paragraph block {operation['block']}"
                )
            try:
                block["text"] = insert_after(block["text"], int(operation["after"]), MARK + text + MARK)
            except ValueError as error:
                raise ValueError(f"{node}: {error} in {host} block {operation['block']}") from error
            block["synthetic"] = node
            touched.append(operation["block"])
        elif operation["op"] == "insert_paragraph":
            anchor = operation["after_block"]
            if anchor not in blocks:
                raise ValueError(f"{node}: anchor block {anchor} not found in {host}")
            section_index, block_index = blocks[anchor]
            new_id = f"{anchor}+{node}.{number}"
            patched["sections"][section_index]["blocks"].insert(
                block_index + 1, {"id": new_id, "kind": "p", "text": MARK + text + MARK, "synthetic": node}
            )
            touched.append(new_id)
            blocks = locate()
        else:
            raise ValueError(
                f"{node}: only insert_sentences and insert_paragraph are allowed - got {operation['op']}"
            )
    for reference in snippet.get("new_bib", []):
        insert_reference(patched, reference, touched[0])
    return patched, touched
