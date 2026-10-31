"""PaperTrail's contexts rebuilt from its kit - the paper ids, the benchmark's own text, and a checksum of each.

The kit ships no sentence of any real paper. Each one is fetched from arXiv at its pinned version, converted by the
benchmark's own rules, and checked against the checksum of the text the benchmark was built from. The hosts are
patched with the inserted text, the fake papers added, and every context assembled and checked in its turn, so a
rebuild is the benchmark byte for byte or says which paper is not.

    <kit>/papers.jsonl                  every real paper - id, version, v1 date, title, sha256 of its text, of its patched text
    <kit>/<sample>/sample.json          the setting, the hosts and their nodes, the fake papers' dates, each context's papers and sha256
    <kit>/<sample>/snippets/<node>.json the benchmark's text - two fake papers a trail, and insertion-only patches
    <kit>/<sample>/questions.jsonl      the 30 scored questions, and nic.jsonl the 6 not-in-corpus ones

The usual cause of a mismatch is arXiv rendering a paper's HTML again.
"""

import dataclasses
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lsa import io
from lsa.evaluation.accuracy.data.papertrail.arxiv import fetch_page, page_path
from lsa.evaluation.accuracy.data.papertrail.compose import body, context, excerpt, fake_paper, patched_host
from lsa.evaluation.accuracy.data.papertrail.convert import parse_html

HOSTS_UNPATCHED = "C2_hosts"
SYNTHETIC_ONLY = "C3_synthetic"


@dataclasses.dataclass(frozen=True)
class Kit:
    """The rebuild kit as read - its real papers by id, and each sample's layout."""

    root: Path
    papers: dict[str, dict[str, Any]]
    samples: dict[str, dict[str, Any]]

    def questions(self, sample: str) -> list[dict[str, Any]]:
        """Return a sample's scored questions, then its not-in-corpus ones."""
        directory = self.root / sample
        return [*io.read_jsonl(directory / "questions.jsonl"), *io.read_jsonl(directory / "nic.jsonl")]

    def snippets(self, sample: str) -> dict[str, dict[str, Any]]:
        """Return a sample's snippets by node."""
        return {
            path.stem: io.read_json(path) for path in sorted((self.root / sample / "snippets").glob("*.json"))
        }


@dataclasses.dataclass(frozen=True)
class Rebuilt:
    """One condition's context as rebuilt, its papers in order, and whether it is the benchmark's byte for byte."""

    context: str
    papers: list[dict[str, Any]]
    verified: bool


def read_kit(root: Path, samples: list[str]) -> Kit:
    """Read the kit's papers and the layout of the samples asked for."""
    papers = {paper["id"]: paper for paper in io.read_jsonl(root / "papers.jsonl")}
    return Kit(root, papers, {sample: io.read_json(root / sample / "sample.json") for sample in samples})


def sha256(text: str) -> str:
    """Return the sha256 of a text's utf-8 bytes."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def needed_papers(kit: Kit, conditions: list[str]) -> list[str]:
    """Return every real paper the asked-for conditions of the kit's samples hold, sorted."""
    return sorted(
        {
            paper_id
            for layout in kit.samples.values()
            for condition, entry in layout["contexts"].items()
            if condition in conditions
            for paper_id in entry["papers"]
            if paper_id in kit.papers
        }
    )


def fetch_pages(
    kit: Kit, paper_ids: list[str], cache: Path, delay: float, log: Callable[[str], Any] = print
) -> int:
    """Fetch every page the cache does not hold yet, so a job on a node without internet finds them all.

    Returns:
        How many pages were fetched.
    """
    missing = [
        paper_id
        for paper_id in paper_ids
        if not page_path(cache, paper_id, kit.papers[paper_id]["version"]).exists()
    ]
    for count, paper_id in enumerate(missing, 1):
        fetch_page(paper_id, kit.papers[paper_id]["version"], cache, delay)
        if count % 50 == 0 or count == len(missing):
            log(f"[fetch] {count}/{len(missing)} pages fetched from arXiv")
    return len(missing)


def convert_papers(
    kit: Kit, paper_ids: list[str], cache: Path, delay: float, log: Callable[[str], Any] = print
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Fetch and convert every paper, checking each one's text against the kit.

    A paper whose page has no title takes the kit's.

    Returns:
        The structured papers by id.
        The papers whose text differs from the kit's.
    """
    converted: dict[str, dict[str, Any]] = {}
    mismatched: list[str] = []
    start = time.perf_counter()
    for count, paper_id in enumerate(paper_ids, 1):
        pinned = kit.papers[paper_id]
        paper = parse_html(fetch_page(paper_id, pinned["version"], cache, delay), paper_id, pinned["date"])
        paper["title"] = paper["title"] or pinned["title"]
        converted[paper_id] = paper
        if sha256(body(paper)[0]) != pinned["sha256"]:
            mismatched.append(paper_id)
        if count % 50 == 0 or count == len(paper_ids):
            seconds = time.perf_counter() - start
            log(
                f"[papers] {count}/{len(paper_ids)} converted ({seconds:.0f} s), {len(mismatched)} mismatched"
            )
    return converted, mismatched


def rebuild_sample(
    kit: Kit, sample: str, converted: dict[str, dict[str, Any]], conditions: list[str]
) -> tuple[dict[str, Rebuilt], list[str]]:
    """Patch a sample's hosts, add its fake papers and assemble each asked-for condition.

    Args:
        kit: The kit.
        sample: The sample.
        converted: Every real paper the conditions hold, structured.
        conditions: The conditions to assemble - the controls with a context, and the lengths.

    Returns:
        Each condition's rebuilt context.
        The hosts whose patched text differs from the kit's.
    """
    layout = kit.samples[sample]
    snippets = kit.snippets(sample)
    texts: dict[str, str] = {}
    originals: dict[str, str] = {}
    spans: dict[str, list[dict[str, Any]]] = {}
    mismatched: list[str] = []
    for host, nodes in layout["hosts"].items():
        if host not in converted:
            continue
        texts[host], spans[host] = patched_host(converted[host], [snippets[node] for node in nodes])
        originals[host] = body(converted[host])[0]
        if sha256(texts[host]) != kit.papers[host].get("patched_sha256"):
            mismatched.append(f"{host} (after patching)")
    for snippet in snippets.values():
        if snippet["kind"] == "fake_paper":
            paper = fake_paper(layout["prefix"], snippet)
            texts[paper["id"]] = body(paper)[0]
    for paper_id, paper in converted.items():
        texts.setdefault(paper_id, body(paper)[0])

    rebuilt: dict[str, Rebuilt] = {}
    for condition, entry in layout["contexts"].items():
        if condition not in conditions:
            continue
        chosen = [paper_entry(kit, layout, paper_id) for paper_id in entry["papers"]]

        def text_of(paper: dict[str, Any], condition: str = condition) -> str:
            """Return a paper's text as the condition shows it."""
            if condition == HOSTS_UNPATCHED:
                return originals[paper["id"]]
            if condition == SYNTHETIC_ONLY and paper["kind"] == "host":
                return excerpt(texts[paper["id"]], spans[paper["id"]])
            return texts[paper["id"]]

        text, papers = context(layout["setting"], chosen, text_of)
        rebuilt[condition] = Rebuilt(text, papers, sha256(text) == entry["sha256"])
    return rebuilt, mismatched


def paper_entry(kit: Kit, layout: dict[str, Any], paper_id: str) -> dict[str, Any]:
    """Return a paper as a context lists it - its id, its v1 date and whether it is fake, a host or background."""
    if paper_id not in kit.papers:
        return {"id": paper_id, "date": layout["dates"][paper_id], "kind": "fake"}
    kind = "host" if paper_id in layout["hosts"] else "background"
    return {"id": paper_id, "date": kit.papers[paper_id]["date"], "kind": kind}
