"""arXiv's HTML, as LaTeXML renders it, turned into a structured paper and the paper into the text the model reads.

A structured paper is what a patch works on:

    {id, date, title, authors, abstract, sections: [{title, level, blocks: [{id, kind, text}]}],
     bib: [{key, tag, text}], bib_style: "numeric" | "authoryear", index_cites}

Block text is plain text with LaTeX maths - `$..$` inline, `$$..$$` for a display block - and citation
placeholders. A numeric citation is stored as `⟦bib.bib12⟧` and rendered from its entry's current tag, so a reference
inserted into the list and the list renumbered leaves every number in the text right. An author-year citation keeps
its printed text. The fake papers are written in this form directly and rendered by the same `render`.

The conversion is the benchmark's own, rule for rule - a paper's text is checked against its checksum, so a rule
changed here is a paper that no longer matches.
"""

import re
from typing import Any

from lxml import html as lxml_html

CITATION = re.compile(r"⟦#?([^⟧\s|]+)(?:\|\d+)?⟧")
INDEX_CITATION = re.compile(r"⟦#([^⟧\s|]+)\|(\d+)⟧")
SPACES = re.compile(r"[ \t\r\f\v]+")

SKIPPED_CLASSES: tuple[str, ...] = (
    "ltx_page_footer",
    "ltx_page_header",
    "ltx_authors",
    "ltx_dates",
    "ltx_pagination",
    "ltx_rdf",
    "ltx_ERROR",
    "ltx_note_mark",
    "ltx_tag_note",
    "ltx_page_logo",
    "ar5iv-footer",
)
HEADINGS: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6")


def classes(element: Any) -> str:
    """Return an element's class attribute, empty when it has none."""
    return element.get("class") or ""


def normalise(text: str) -> str:
    """Fold runs of spaces into one, drop zero-width spaces and the spaces around line breaks."""
    text = SPACES.sub(" ", text.replace(" ", " ").replace("​", ""))
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def maths(element: Any) -> str:
    """Return a maths element as its LaTeX source, in `$$` for a display block and `$` inline."""
    tex = SPACES.sub(" ", element.get("alttext") or "").strip()
    return f"$${tex}$$" if element.get("display") == "block" else f"${tex}$"


def inline_text(element: Any, numeric: bool) -> str:
    """Flatten an element to text - maths to its source, citations to placeholders, footnotes inline.

    Args:
        element: The element.
        numeric: Whether the paper's bibliography is numbered.

    Returns:
        The text, normalised.
    """
    out: list[str] = []

    def walk(node: Any) -> None:
        """Append a node's text to `out`, with its children and their tails."""
        tag = node.tag if isinstance(node.tag, str) else ""
        node_classes = classes(node)
        if tag == "math":
            out.append(maths(node))
        elif any(name in node_classes for name in SKIPPED_CLASSES):
            pass
        elif tag in ("script", "style", "button", "svg", "img"):
            pass
        elif "ltx_note" in node_classes and "ltx_role_footnote" in node_classes:
            content = node.find_class("ltx_note_content")
            if content:
                footnote = inline_text(content[0], numeric)
                if footnote:
                    out.append(f" (footnote: {footnote})")
        elif tag == "a" and (node.get("href") or "").startswith("#bib."):
            key = node.get("href")[1:]
            printed = (node.text_content() or "").strip()
            # "12" or "Ziegler et al. [31]"
            number = re.fullmatch(r"(.*?)(\d+)(\]?)", printed, re.S)
            if numeric and number:
                out.append(f"{number.group(1)}⟦{key}⟧{number.group(3)}")
            elif not numeric and re.fullmatch(r"\d+", printed):
                # Author-year bibliography with a numeric in-text index
                out.append(f"⟦#{key}|{printed}⟧")
            else:
                out.append(printed)
        elif tag == "br":
            out.append("\n")
        else:
            if node.text:
                out.append(node.text)
            for child in node:
                walk(child)
                if child.tail:
                    out.append(child.tail)

    if element.text:
        out.append(element.text)
    for child in element:
        walk(child)
        if child.tail:
            out.append(child.tail)
    return normalise("".join(out))


def plain(text: str) -> str:
    """Drop index placeholders for their printed number - in titles and reference entries, which are not renumbered."""
    return INDEX_CITATION.sub(lambda match: match.group(2), text)


def table_text(table: Any, numeric: bool) -> str:
    """Return a table as one line a row, its cells joined by ` | `, leaving out empty rows."""
    rows = []
    for row in table.iter("tr"):
        cells = [
            inline_text(cell, numeric)
            for cell in row
            if isinstance(cell.tag, str) and cell.tag in ("td", "th")
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def bibliography(document: Any) -> tuple[list[dict[str, Any]], str]:
    """Read a paper's references and decide whether it numbers them.

    Returns:
        The entries in order, each with its key, its tag and its text.
        `numeric` or `authoryear`.
    """
    entries: list[dict[str, Any]] = []
    for position, item in enumerate(document.xpath('//li[contains(@class,"ltx_bibitem")]')):
        tag_element = item.find_class("ltx_tag_bibitem")
        tag = normalise(tag_element[0].text_content()) if tag_element else ""
        parts = []
        for block in item.find_class("ltx_bibblock"):
            text = plain(inline_text(block, False))
            text = re.split(r"External Links:|Cited by:", text)[0].strip()
            if text:
                parts.append(text if text[-1] in ".,;:?!" else text + ".")
        entries.append({"key": item.get("id") or f"bib.anon{position}", "tag": tag, "parts": parts})
    numeric = sum(bool(re.fullmatch(r"\[?\d+\]?", entry["tag"])) for entry in entries) > 0.6 * max(
        1, len(entries)
    )
    for entry in entries:
        if numeric:
            entry["tag"] = re.sub(r"[\[\]]", "", entry["tag"])
        else:
            # LaTeXML puts an author-year entry's year in its tag alone => put it back after the authors
            year_match = re.search(r"\((\d{4}[a-z]?)\)|(\d{4}[a-z]?)\s*$", entry["tag"])
            year = (year_match.group(1) or year_match.group(2)) if year_match else ""
            if year and entry["parts"] and not any(year in part for part in entry["parts"][:2]):
                entry["parts"][0] = entry["parts"][0].rstrip(".") + f" ({year})."
        entry["text"] = re.sub(r"\s+([.,;)])", r"\1", " ".join(entry.pop("parts")))
    return entries, ("numeric" if numeric else "authoryear")


def blocks_of(container: Any, numeric: bool) -> list[dict[str, Any]]:
    """Collect a section's content blocks in document order, leaving its nested sections to the caller."""
    blocks = []

    def visit(element: Any) -> None:
        """Append the content blocks under an element in document order, skipping headings and sections."""
        if not isinstance(element.tag, str):
            return
        element_classes = classes(element)
        if (
            element.tag == "section"
            or any(name in element_classes for name in ("ltx_bibliography", "ltx_appendix"))
            and element is not container
        ):
            return
        if any(name in element_classes for name in SKIPPED_CLASSES) or element.tag in HEADINGS:
            return
        if (
            element.tag == "p"
            or "ltx_para" in element_classes
            and not element.xpath(".//p|.//table|.//figure|.//ul|.//ol")
        ):
            text = re.sub(r"\s*\n\s*", " ", inline_text(element, numeric))
            if text:
                block_id = element.get("id") or (element.getparent().get("id") or "")
                blocks.append({"id": block_id, "kind": "p", "text": text})
            return
        if element.tag == "table" and (
            "ltx_equation" in element_classes or "ltx_equationgroup" in element_classes
        ):
            formulas = element.xpath(".//math")
            if formulas:
                tex = " \\\\ ".join((formula.get("alttext") or "").strip() for formula in formulas)
                number = element.xpath('.//span[contains(@class,"ltx_tag_equation")]')
                label = (" " + normalise(number[0].text_content())) if number else ""
                blocks.append(
                    {
                        "id": element.get("id") or "",
                        "kind": "eq",
                        "text": f"$${SPACES.sub(' ', tex)}$${label}",
                    }
                )
            return
        if element.tag == "figure":
            caption_element = element.find(".//figcaption")
            is_table = "ltx_table" in element_classes
            body = ""
            if is_table:
                tables = element.xpath('.//table[contains(@class,"ltx_tabular")]')
                body = "\n".join(table_text(table, numeric) for table in tables)
            caption = inline_text(caption_element, numeric) if caption_element is not None else ""
            if caption or body:
                blocks.append(
                    {
                        "id": element.get("id") or "",
                        "kind": "table" if is_table else "fig",
                        "text": (caption + ("\n" + body if body else "")).strip(),
                    }
                )
            return
        if element.tag in ("ul", "ol"):
            items = []
            for item in element.xpath("./li"):
                text = inline_text(item, numeric)
                if text:
                    items.append("- " + text)
            if items:
                blocks.append({"id": element.get("id") or "", "kind": "list", "text": "\n".join(items)})
            return
        for child in element:
            visit(child)

    for child in container:
        visit(child)
    return blocks


def parse_html(raw: str, arxiv_id: str, date: str) -> dict[str, Any]:
    """Turn one paper's arXiv HTML into a structured paper.

    Args:
        raw: The page.
        arxiv_id: The paper's id.
        date: Its v1 submission date.

    Returns:
        The structured paper.

    Raises:
        ValueError: The page holds no article.
    """
    document = lxml_html.fromstring(raw)
    articles = document.xpath("//article")
    if not articles:
        raise ValueError("no <article>")
    article = articles[0]
    bib, style = bibliography(document)
    numeric = style == "numeric"
    title_elements = article.xpath('.//h1[contains(@class,"ltx_title_document")]')
    title = plain(normalise(inline_text(title_elements[0], False))) if title_elements else ""
    authors = [
        normalise(name.text_content()) for name in article.xpath('.//span[contains(@class,"ltx_personname")]')
    ]
    authors = [author.split("\n")[0].strip() for author in authors if author]
    abstract_elements = article.xpath('.//div[contains(@class,"ltx_abstract")]')
    abstract = (
        "\n\n".join(
            re.sub(r"\s*\n\s*", " ", inline_text(paragraph, numeric))
            for paragraph in abstract_elements[0].xpath(".//p")
        )
        if abstract_elements
        else ""
    )

    sections: list[dict[str, Any]] = []

    def add_section(element: Any, level: int) -> None:
        """Append a section and, one level deeper, its subsections, each with its content blocks."""
        heading = element.xpath("./h2|./h3|./h4|./h5|./h6")
        section_title = plain(normalise(inline_text(heading[0], False))) if heading else ""
        sections.append({"title": section_title, "level": level, "blocks": blocks_of(element, numeric)})
        for subsection in element.xpath("./section"):
            if "ltx_bibliography" in classes(subsection):
                continue
            add_section(subsection, level + 1)

    # Content before the first section - some papers open with an untitled introduction
    lead = []
    for child in article:
        if isinstance(child.tag, str) and child.tag == "div" and "ltx_para" in classes(child):
            lead.extend(
                blocks_of(child, numeric)
                or [{"id": child.get("id") or "", "kind": "p", "text": inline_text(child, numeric)}]
            )
    if lead:
        sections.append({"title": "", "level": 2, "blocks": [block for block in lead if block["text"]]})
    for section in article.xpath("./section|./div/section"):
        if "ltx_bibliography" in classes(section):
            continue
        add_section(section, 2)
    position_of = {entry["key"]: index + 1 for index, entry in enumerate(bib)}
    indexed = [
        (key, int(number))
        for section in sections
        for block in section["blocks"]
        for key, number in INDEX_CITATION.findall(block["text"])
    ]
    index_cites = bool(indexed) and sum(
        position_of.get(key) == number for key, number in indexed
    ) >= 0.9 * len(indexed)
    seen: set[str] = set()
    count = 0
    for section in sections:
        for block in section["blocks"]:
            count += 1
            if not block["id"] or block["id"] in seen:
                block["id"] = f"{block['id'] or 'blk'}~{count}"
            seen.add(block["id"])
    return {
        "id": arxiv_id,
        "date": date,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "sections": sections,
        "bib": bib,
        "bib_style": style,
        "index_cites": index_cites,
    }


def render_block(text: str, tags: dict[str, str], index: dict[str, int] | None = None) -> str:
    """Resolve a block's citation placeholders.

    `⟦key⟧` becomes its entry's tag, and `⟦#key|n⟧` its entry's place in the bibliography - natbib's numbering of
    an alphabetical list - or the printed n when the paper does not number by place.
    """
    if index is not None:
        text = INDEX_CITATION.sub(
            lambda match: str(index[match.group(1)]) if match.group(1) in index else match.group(2), text
        )
    else:
        text = INDEX_CITATION.sub(lambda match: match.group(2), text)
    return CITATION.sub(lambda match: tags.get(match.group(1), "?"), text)


def header(paper_number: int | None, paper: dict[str, Any]) -> str:
    """Return the header the model sees above a paper - its number in the context and its v1 date, no arXiv id."""
    number = f"Paper {paper_number:04d} | " if paper_number is not None else ""
    return f"[{number}{paper['date']}]"


def render(paper: dict[str, Any], paper_number: int | None = None) -> tuple[str, dict[str, tuple[int, int]]]:
    """Render one paper's canonical text.

    Args:
        paper: The structured paper.
        paper_number: Its number in a context, or None for a header without one.

    Returns:
        The text.
        Every block's character span in it, by block id - `abstract`, a section block's id, `bib:<key>`.
    """
    tags = {entry["key"]: entry["tag"] for entry in paper["bib"]}
    index = (
        {entry["key"]: position + 1 for position, entry in enumerate(paper["bib"])}
        if paper.get("index_cites")
        else None
    )
    out: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    position = 0

    def emit(text: str, block_id: str | None = None) -> None:
        """Append text to the paper and record its character span under `block_id`."""
        nonlocal position
        if block_id is not None:
            spans[block_id] = (position, position + len(text))
        out.append(text)
        position += len(text)

    emit(header(paper_number, paper) + "\n")
    emit(f"# {paper['title']}\n")
    if paper["authors"]:
        emit("Authors: " + ", ".join(paper["authors"]) + "\n")
    if paper["abstract"]:
        emit("\n## Abstract\n")
        emit(render_block(paper["abstract"], tags, index) + "\n", "abstract")
    for section in paper["sections"]:
        if section["title"]:
            emit("\n" + "#" * min(section["level"], 4) + " " + section["title"] + "\n")
        for block in section["blocks"]:
            emit("\n")
            emit(render_block(block["text"], tags, index), block.get("id") or None)
            emit("\n")
    if paper["bib"]:
        emit("\n## References\n")
        for entry in paper["bib"]:
            lead = f"[{entry['tag']}] " if paper["bib_style"] == "numeric" else ""
            emit(lead + entry["text"] + "\n", "bib:" + entry["key"])
    return "".join(out), spans
