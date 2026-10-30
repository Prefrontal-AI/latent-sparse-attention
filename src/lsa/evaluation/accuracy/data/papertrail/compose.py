"""The text a context is made of - a paper's body, a patched host, a fake paper, and the context around them.

No tokenizer and no network: given structured papers and the benchmark's snippets, this is the exact text the
model reads, which the rebuild checks byte for byte against the benchmark's checksums.
"""

import bisect
from collections.abc import Callable
from typing import Any

from lsa.evaluation.accuracy.data.papertrail.convert import header, render
from lsa.evaluation.accuracy.data.papertrail.patch import MARK, apply_patch, insert_reference

OMITTED = "[... rest of paper omitted ...]"
END = "\n=== END PAPERS ===\n"
# A node's inserted text is tagged by one private-use character after its opening mark
FIRST_TAG = 0xE000
LAST_TAG = ""


def setting_prompt(setting: str, paper_count: int) -> str:
    """Return what a context opens with - the sample's setting, then how to read the papers below it."""
    return (
        f"{setting.strip()}\n\nBelow are {paper_count} arXiv preprints, one after another. Each "
        "paper starts with a header line of the form [Paper NNNN | submission date]. Read all of them carefully. "
        "Afterwards you will be asked a question about this literature. Answering may require finding a single "
        "detail, combining findings from several papers, following how a result was later revised or renamed, "
        "or recognising that the collection does not contain the answer.\n\n=== BEGIN PAPERS ===\n\n"
    )


def fake_paper(prefix: str, snippet: dict[str, Any]) -> dict[str, Any]:
    """Return a fake-paper snippet as a structured paper, rendered by the renderer real papers go through.

    Its sections sit one level below their written level, as a real paper's `##` sections do.
    """
    sections = [
        {
            "title": section["title"],
            "level": section.get("level", 1) + 1,
            "blocks": [
                {
                    "id": f"S{section_index}.b{block_index}",
                    "kind": block.get("kind", "p"),
                    "text": block["text"],
                }
                for block_index, block in enumerate(section["blocks"])
            ],
        }
        for section_index, section in enumerate(snippet["sections"])
    ]
    return {
        "id": f"{prefix}-{snippet['node']}",
        "date": snippet["date"],
        "title": snippet["title"],
        "authors": list(snippet["authors"]) + list(snippet.get("affiliations", [])),
        "abstract": snippet["abstract"],
        "sections": sections,
        "bib": snippet["bib"],
        "bib_style": snippet.get("bib_style", "authoryear"),
        "index_cites": False,
    }


def body(paper: dict[str, Any]) -> tuple[str, dict[str, tuple[int, int]], list[tuple[int, int, str]]]:
    """Render a paper without its header line, its inserted text unmarked.

    Inserted text arrives as `MARK`, a tag, the text and `MARK`, the tag one private-use character per node. The
    marks are dropped here and their places kept.

    Returns:
        The text.
        Every block's span in it, by block id.
        Every inserted span, as its start, its end and its node's tag.

    Raises:
        ValueError: A mark opens no insertion or one insertion is left open.
    """
    text, spans = render(paper)
    cut = text.index("\n") + 1
    marked = text[cut:]
    removed: list[int] = []
    inserted: list[tuple[int, int, str]] = []
    clean: list[str] = []
    opened_at: int | None = None
    tag = ""
    position = 0
    while position < len(marked):
        character = marked[position]
        if character == MARK:
            if (
                opened_at is None
                and position + 1 < len(marked)
                and chr(FIRST_TAG) <= marked[position + 1] <= LAST_TAG
            ):
                opened_at, tag = len(clean), marked[position + 1]
                removed += [position, position + 1]
                position += 2
                continue
            if opened_at is not None:
                inserted.append((opened_at, len(clean), tag))
                opened_at = None
                removed.append(position)
                position += 1
                continue
            raise ValueError(f"{paper['id']}: stray synthetic mark")
        clean.append(character)
        position += 1
    if opened_at is not None:
        raise ValueError(f"{paper['id']}: unbalanced synthetic marks")

    def unmarked(offset: int) -> int:
        shifted = offset - cut
        return shifted - bisect.bisect_left(removed, shifted)

    blocks = {
        block_id: (unmarked(start), unmarked(end)) for block_id, (start, end) in spans.items() if start >= cut
    }
    return "".join(clean), blocks, inserted


def patch_host(
    paper: dict[str, Any], snippets: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply every node patched into one host, tagging each node's text so its span can be told apart.

    Returns:
        The patched paper.
        Tag to node.
    """
    tags: dict[str, str] = {}
    patched = paper
    for index, snippet in enumerate(sorted(snippets, key=lambda snippet: snippet["node"])):
        tag = chr(FIRST_TAG + index)
        tags[tag] = snippet["node"]
        tagged = {
            **snippet,
            "ops": [{**operation, "text": tag + operation["text"].strip()} for operation in snippet["ops"]],
            "new_bib": [],
        }
        patched, touched = apply_patch(patched, tagged)
        for reference in snippet.get("new_bib", []):
            insert_reference(patched, reference, touched[0])
    return patched, tags


def patched_host(
    original: dict[str, Any], snippets: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Patch one host with all its nodes.

    Returns:
        Its body text.
        Every span of our text in it - `node`, `start`, `end`, and `role`, `reference` for an inserted reference.
    """
    patched, tags = patch_host(original, snippets)
    text, blocks, inserted = body(patched)
    spans: list[dict[str, Any]] = [
        {"node": tags[tag], "start": start, "end": end, "role": None} for start, end, tag in inserted
    ]
    for snippet in snippets:
        for reference in snippet.get("new_bib", []):
            start, end = blocks[f"bib:{reference['key']}"]
            spans.append({"node": snippet["node"], "start": start, "end": end, "role": "reference"})
    return text, spans


def excerpt(text: str, spans: list[dict[str, Any]]) -> str:
    """Return a host as the synthetic-only control shows it - title and authors, an omission, then our text alone."""
    lines = text.split("\n", 2)
    kept = "\n".join(lines[:2]) + "\n\n" + OMITTED + "\n"
    for span in sorted(spans, key=lambda span: span["start"]):
        kept += "\n" + text[span["start"] : span["end"]].strip() + "\n"
    return kept


def context(
    setting: str, chosen: list[dict[str, Any]], text_of: Callable[[dict[str, Any]], str]
) -> tuple[str, list[dict[str, Any]]]:
    """Assemble a context - the setting, then each paper under its header, in order.

    Args:
        setting: The sample's setting.
        chosen: The papers in context order, each with its `id`, `date` and `kind`.
        text_of: A paper's body text.

    Returns:
        The context.
        Each paper's number, id, kind and character span.
    """
    parts = [setting_prompt(setting, len(chosen))]
    position = len(parts[0])
    papers = []
    for number, paper in enumerate(chosen, 1):
        text = text_of(paper)
        paper_header = header(number, paper) + "\n"
        start = position
        parts += [paper_header, text, "\n"]
        position += len(paper_header) + len(text) + 1
        papers.append(
            {
                "paper_no": number,
                "id": paper["id"],
                "kind": paper["kind"],
                "start": start,
                "body_start": start + len(paper_header),
                "end": position,
            }
        )
    parts.append(END)
    return "".join(parts), papers
