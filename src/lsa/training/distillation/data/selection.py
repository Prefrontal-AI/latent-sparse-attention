"""Choose a target extraction's documents balanced over lengths and domains.

A selection is a file naming every document an extraction reads, the side it lands on and the length of its one
window. Every length bucket holds the same number of tokens unless the selection gives a bucket a budget of its own,
and inside a bucket every domain holds the same share wherever it has documents that long - a domain that runs
short gives the rest back to the others.

The longest bucket chooses first, because a long document is the scarce thing, and a document taken by one
bucket is out of every other, so no text is read at two lengths. Documents are taken in the order of a hash
of their id rather than of their place in a manifest, so growing a selection never moves a document it already
chose, and a smaller tier is rebalanced out of the full one so that extracting the full selection only extends
the tier's cache.

Training windows come from the side of the held-out hash language-model training reads, and scoring windows
from the held-out side alone - the target extractor refuses an entry that says otherwise.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import assets
from lsa.data.corpus import document_extents
from lsa.training.constants import Corpus
from lsa.training.data.corpus import document_share
from lsa.training.data.targets import extracted_from
from lsa.training.distillation.constants import Selection, Split


@dataclass
class SelectionConfig:
    """One selection - the datasets it draws on, its buckets and how its domains are made of sources.

    `tokens_per_length` gives a bucket a budget of its own in place of `tokens_per_bucket`, for a long end
    that should hold more documents than an equal share of tokens allows.

    `tokens_per_tier` optionally reduces training out of the full selection, in tokens per bucket. `min_length` keeps a
    source out of the buckets shorter than it, for text that is only worth its length. `from_start` reads a
    source's window from the front of the document rather than from a hashed slot, for a document whose
    beginning is the task it is about. `group_by` gives a source's separator, the part of a document id
    before which is its group - a bucket takes one document per group, and a group with any held-out member
    trains on none of them, so a scored task is never trained on through a sibling trajectory.

    `exclude_from_scoring` names cache collections, as `<root>/<collection>`, whose documents an earlier map
    trained on, so the new scoring set can score that map fairly.
    """

    datasets: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    tokens_per_bucket: int = 0
    tokens_per_length: dict[int, int] = field(default_factory=dict)
    tokens_per_tier: int = 0
    domains: dict[str, list[str]] = field(default_factory=dict)
    min_length: dict[str, int] = field(default_factory=dict)
    from_start: list[str] = field(default_factory=list)
    group_by: dict[str, str] = field(default_factory=dict)
    scoring_per_domain: int = 4
    exclude_from_scoring: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject a selection that would choose nothing or say one thing twice.

        Raises:
            ValueError: A field that has to be given is empty, or a length or budget is not positive.
            ValueError: A length's own budget is for a length the selection does not have, or is not positive.
            ValueError: A tier is not smaller than the full selection.
            ValueError: A source belongs to two domains.
        """
        if not (self.datasets and self.lengths and self.domains):
            raise ValueError("data needs datasets, lengths and domains")
        if min(self.lengths) <= 0 or self.tokens_per_bucket <= 0 or self.scoring_per_domain <= 0:
            raise ValueError("selection: lengths, tokens_per_bucket and scoring_per_domain must be positive")
        for length, tokens in self.tokens_per_length.items():
            if length not in self.lengths or tokens <= 0:
                raise ValueError(
                    f"selection: tokens_per_length needs a positive budget for one of {self.lengths}"
                    f" - got {length}: {tokens}"
                )
        if self.tokens_per_tier < 0 or self.tokens_per_tier >= self.tokens_per_bucket:
            raise ValueError("selection.tokens_per_tier must be zero or smaller than tokens_per_bucket")
        sources = [source for members in self.domains.values() for source in members]
        if len(set(sources)) != len(sources):
            raise ValueError(f"selection: a source belongs to two domains - {sorted(sources)}")

    @property
    def domain_of(self) -> dict[str, str]:
        """Each source's domain"""
        return {source: domain for domain, members in self.domains.items() for source in members}

    def budget(self, length: int) -> int:
        """Return the tokens one bucket of the full selection holds.

        Args:
            length: The bucket.

        Returns:
            The bucket's own budget if it has one, else `tokens_per_bucket`.
        """
        return self.tokens_per_length.get(length, self.tokens_per_bucket)


@dataclass(frozen=True)
class Candidate:
    """One document a selection may choose, and what it knows about it before choosing."""

    dataset: str
    source: str
    document: str
    tokens: int
    group: str
    held_out: bool


def hash_rank(key: str) -> float:
    """Place a key in the order a selection takes documents in.

    Args:
        key: A document id, or a document id and a slot.

    Returns:
        A fixed point in [0, 1), salted apart from both split hashes.
    """
    return document_share(key, Selection.SALT)


def candidates(
    selection: SelectionConfig, datasets: dict[str, assets.TokenizedCard] | None = None
) -> list[Candidate]:
    """Return every document of every dataset the selection draws on, with its usable length and its side.

    A document's usable length is the rows it was tokenized into, so a window never reads the remainder
    tokenizing dropped.

    Args:
        selection: The selection.
        datasets: Path-resolved cards overriding registered datasets by name.

    Returns:
        One candidate per document of a source some domain names.
    """
    excluded = frozenset().union(*(extracted_from(where) for where in selection.exclude_from_scoring))
    domain_of = selection.domain_of
    found = []
    for name in selection.datasets:
        asset = datasets[name] if datasets and name in datasets else assets.tokenized(name)
        for source in (name for name in asset.sources if name in domain_of):
            for document, (_, rows) in document_extents(Path(asset.path), source).items():
                held_out = document_share(document) < Corpus.HELD_OUT_FRACTION
                if held_out and (source, document) in excluded:
                    continue
                group = (
                    document.split(selection.group_by[source], 1)[0]
                    if source in selection.group_by
                    else document
                )
                found.append(
                    Candidate(name, source, document, int(rows) * asset.sample_length, group, held_out)
                )

    # A group with a held-out member trains on none of them
    scored_groups = {
        (one.source, one.group) for one in found if one.held_out and one.source in selection.group_by
    }
    return [one for one in found if one.held_out or (one.source, one.group) not in scored_groups]


def water_fill(total: int, supply: dict[str, int]) -> dict[str, int]:
    """Split `total` as evenly as possible over keys, none past its supply.

    Args:
        total: Units to hand out.
        supply: What each key can take.

    Returns:
        What each key takes, summing to `total` or to the whole supply when that is smaller.
    """
    taken = dict.fromkeys(supply, 0)
    left = total
    open_keys = [key for key in supply if supply[key] > 0]
    while left > 0 and open_keys:
        share = max(1, left // len(open_keys))
        for key in list(open_keys):
            step = min(share, supply[key] - taken[key], left)
            taken[key] += step
            left -= step
            if taken[key] == supply[key]:
                open_keys.remove(key)
            if left == 0:
                break
    return taken


def split_domain(windows: int, supply: dict[str, int], size: dict[str, int]) -> dict[str, int]:
    """Split a domain's windows over its sources in proportion to their size, topping up from the rest.

    Args:
        windows: The domain's windows in one bucket.
        supply: What each of its sources can give at that length.
        size: Each source's tokens, which is what proportion means.

    Returns:
        Windows per source.
    """
    size_total = sum(size[source] for source in supply) or 1
    proportional = {source: min(supply[source], windows * size[source] // size_total) for source in supply}
    top_up = water_fill(
        windows - sum(proportional.values()), {s: supply[s] - proportional[s] for s in supply}
    )
    return {source: proportional[source] + top_up[source] for source in supply}


def domain_supply(supply: dict[str, int], domain_of: dict[str, str]) -> dict[str, int]:
    """Sum what each source can give into what each domain can.

    Args:
        supply: Windows each source can give in one bucket.
        domain_of: Each source's domain.

    Returns:
        Windows per domain, domains in the order their first source appears.
    """
    per_domain: dict[str, int] = {}
    for source, count in supply.items():
        per_domain[domain_of[source]] = per_domain.get(domain_of[source], 0) + count
    return per_domain


def windows_per_source(
    domain_windows: dict[str, int], supply: dict[str, int], domain_of: dict[str, str], size: dict[str, int]
) -> dict[str, int]:
    """Split each domain's windows over its sources, in proportion to their size.

    Args:
        domain_windows: Windows each domain takes in one bucket, as `water_fill` handed them out.
        supply: Windows each source can give in that bucket.
        domain_of: Each source's domain.
        size: Each source's tokens, which is what proportion means.

    Returns:
        Windows per source, domain by domain in the order `domain_windows` holds them.
    """
    taken: dict[str, int] = {}
    for domain, windows in domain_windows.items():
        members = {source: supply[source] for source in supply if domain_of[source] == domain}
        taken.update(split_domain(windows, members, size))
    return taken


def allocate(
    selection: SelectionConfig, pool: list[Candidate], split: str, domain_cap: int | None
) -> list[dict[str, Any]]:
    """Choose one window per document, longest bucket first, balanced over domains then sources.

    Args:
        selection: The selection.
        pool: The documents this side may take.
        split: `Split.TRAIN` or `Split.EVAL`, written onto every entry.
        domain_cap: Windows a domain may take per bucket, or None for the budget alone to decide.

    Returns:
        One entry per chosen document.
    """
    domain_of = selection.domain_of
    size: dict[str, int] = {}
    for candidate in pool:
        size[candidate.source] = size.get(candidate.source, 0) + candidate.tokens
    free = sorted(pool, key=lambda candidate: hash_rank(candidate.document))

    chosen: list[dict[str, Any]] = []
    for length in sorted(selection.lengths, reverse=True):
        eligible = _eligible(selection, free, length)
        supply = {source: len(documents) for source, documents in eligible.items()}
        per_domain = domain_supply(supply, domain_of)
        if domain_cap is not None:
            per_domain = {domain: min(count, domain_cap) for domain, count in per_domain.items()}
        budget = selection.budget(length) // length if domain_cap is None else sum(per_domain.values())
        per_source = windows_per_source(water_fill(budget, per_domain), supply, domain_of, size)

        taken: set[tuple[str, str]] = set()
        for source, count in per_source.items():
            for candidate in eligible[source][:count]:
                chosen.append(_entry(selection, candidate, length, split))
                taken.add((candidate.source, candidate.document))
        free = [candidate for candidate in free if (candidate.source, candidate.document) not in taken]
    return chosen


def _eligible(selection: SelectionConfig, free: list[Candidate], length: int) -> dict[str, list[Candidate]]:
    """Return each source's documents that can give one window at this length, in hash order.

    Args:
        selection: The selection.
        free: Documents no longer bucket has taken, in hash order.
        length: The bucket.

    Returns:
        Candidates per source, one per group.
    """
    eligible: dict[str, list[Candidate]] = {}
    groups_seen: set[tuple[str, str]] = set()
    for candidate in free:
        if candidate.tokens < length or length < selection.min_length.get(candidate.source, 0):
            continue
        if (candidate.source, candidate.group) in groups_seen:
            continue
        groups_seen.add((candidate.source, candidate.group))
        eligible.setdefault(candidate.source, []).append(candidate)
    return eligible


def _entry(selection: SelectionConfig, candidate: Candidate, length: int, split: str) -> dict[str, Any]:
    """Write one chosen window, placed at a hashed slot unless its source reads from the front.

    Args:
        selection: The selection.
        candidate: The document.
        length: The bucket.
        split: The side.

    Returns:
        The entry as the selection file holds it.
    """
    slots = range(candidate.tokens // length)
    window = 0
    if candidate.source not in selection.from_start:
        window = min(slots, key=lambda slot: hash_rank(f"{candidate.document}:{slot}"))
    return {
        "dataset": candidate.dataset,
        "source": candidate.source,
        "domain": selection.domain_of[candidate.source],
        "document": candidate.document,
        "split": split,
        "length": length,
        "window": window,
    }


def tier(
    selection: SelectionConfig, entries: list[dict[str, Any]], tokens_per_bucket: int
) -> list[dict[str, Any]]:
    """Rebalance a smaller selection's training side out of a full one's, keeping its scoring side whole.

    Args:
        selection: The selection.
        entries: The full selection's entries.
        tokens_per_bucket: The tier's budget.

    Returns:
        The tier's entries - a subset of `entries`.
    """
    domain_of = selection.domain_of
    training = [entry for entry in entries if entry["split"] == Split.TRAIN]
    size: dict[str, int] = {}
    for entry in training:
        size[entry["source"]] = size.get(entry["source"], 0) + entry["length"]

    kept = [entry for entry in entries if entry["split"] == Split.EVAL]
    for length in selection.lengths:
        bucket = sorted(
            (entry for entry in training if entry["length"] == length),
            key=lambda entry: hash_rank(entry["document"]),
        )
        supply: dict[str, int] = {}
        for entry in bucket:
            supply[entry["source"]] = supply.get(entry["source"], 0) + 1
        domain_windows = water_fill(tokens_per_bucket // length, domain_supply(supply, domain_of))
        for source, count in windows_per_source(domain_windows, supply, domain_of, size).items():
            kept.extend([entry for entry in bucket if entry["source"] == source][:count])
    return kept


def build_full(
    selection: SelectionConfig, datasets: dict[str, assets.TokenizedCard] | None = None
) -> list[dict[str, Any]]:
    """Choose the full training budget and its held-out scoring documents."""
    pool = candidates(selection, datasets)
    training = allocate(selection, [one for one in pool if not one.held_out], Split.TRAIN, None)
    scoring = allocate(
        selection, [one for one in pool if one.held_out], Split.EVAL, selection.scoring_per_domain
    )
    return training + scoring


def build(selection: SelectionConfig) -> list[dict[str, Any]]:
    """Choose a deterministic selection, optionally reducing its training budget."""
    full = build_full(selection)
    return tier(selection, full, selection.tokens_per_tier) if selection.tokens_per_tier else full


def describe(selection: SelectionConfig, entries: Iterable[dict[str, Any]]) -> str:
    """Render windows per length and domain, and each bucket's tokens, as a markdown table.

    Args:
        selection: The selection.
        entries: One side of one selection.

    Returns:
        The table.
    """
    domains = list(selection.domains)
    counts: dict[tuple[int, str], int] = {}
    for entry in entries:
        counts[(entry["length"], entry["domain"])] = counts.get((entry["length"], entry["domain"]), 0) + 1
    lines = ["| length | " + " | ".join(domains) + " | tokens |", "|---" * (len(domains) + 2) + "|"]
    for length in selection.lengths:
        row = [counts.get((length, domain), 0) for domain in domains]
        lines.append(f"| {length} | " + " | ".join(map(str, row)) + f" | {sum(row) * length:,} |")
    return "\n".join(lines)
