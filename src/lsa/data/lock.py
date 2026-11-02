"""A lock file - what a collected source holds, as the sha256 of each of its documents.

A source's text cannot be redistributed, and the order its hub dataset streams in is not on any card. A lock file
is what stands in for both: it records every document a collection kept, in order, with its tokens and the sha256
of its text, so a reader who assembles the source from the hub learns whether they hold the same documents.

    lsa data prepare gutenberg --write-lock gutenberg.lock.parquet   # a maintainer, from the shards that exist
    lsa data prepare gutenberg --verify gutenberg.lock.parquet       # a reader, after collecting

Verifying names what differs rather than saying that something does. A document the lock holds and the shards
do not is missing, one the shards hold and the lock does not is extra, one whose text hashes differently is
changed, and two sources with the same documents in another order are a different corpus, because a tokenized
row lands where the documents before it put it.
"""

import dataclasses
from collections.abc import Iterator
from pathlib import Path

from lsa import assets
from lsa.data.collect import digest
from lsa.data.constants import Collected

# The columns of a lock file, in the order they are written
DOC_ID = "doc_id"
SHA256 = "sha256"
N_TOKENS = "n_tokens"
# What a lock file says about itself in its schema metadata
SOURCE_KEY = b"source"
REVISION_KEY = b"revision"
TOKENIZER_KEY = b"count_tokenizer"
# Documents named in a report before it says how many more there are
SHOWN = 10


@dataclasses.dataclass
class Verification:
    """What differs between a lock file and the shards of a source."""

    source: str
    locked: int
    collected: int
    missing: list[str]
    extra: list[str]
    changed: list[str]
    # The documents the two hold in common, which sit in a different order
    reordered: bool

    @property
    def ok(self) -> bool:
        """Whether the shards hold exactly the documents the lock does, in the same order."""
        return not (self.missing or self.extra or self.changed or self.reordered)

    def report(self) -> str:
        """Return what differs, a few of each kind named."""
        lines = [f"{self.source}: the lock holds {self.locked} documents and the shards {self.collected}"]
        for label, ids in (("missing", self.missing), ("extra", self.extra), ("changed", self.changed)):
            if ids:
                more = f" and {len(ids) - SHOWN} more" if len(ids) > SHOWN else ""
                lines.append(f"  {len(ids)} {label}: {', '.join(ids[:SHOWN])}{more}")
        if self.reordered:
            lines.append("  the documents they share are in a different order")
        if self.ok:
            lines.append("  every document is the one the lock holds, in the same order")
        return "\n".join(lines)


def shard_documents(card: assets.SourceCard) -> Iterator[tuple[str, str, int]]:
    """Return each document of a source's shards - its id, the sha256 of its text and its tokens - in shard order.

    Args:
        card: The source.

    Yields:
        The id, the digest and the tokens of each document.

    Raises:
        FileNotFoundError: The source has no shards.
    """
    import pyarrow.parquet as pq

    shards = sorted(Path(card.path).glob(Collected.SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no {Collected.SHARD_GLOB} under {card.path}")
    for shard in shards:
        table = pq.read_table(shard, columns=[Collected.DOC_ID, Collected.N_TOKENS, Collected.TEXT])
        for doc_id, tokens, text in zip(
            table.column(Collected.DOC_ID).to_pylist(),
            table.column(Collected.N_TOKENS).to_pylist(),
            table.column(Collected.TEXT).to_pylist(),
            strict=True,
        ):
            yield str(doc_id), digest(text), int(tokens)


def write(card: assets.SourceCard, out: str) -> Path:
    """Write the lock file of a source's collected shards.

    Args:
        card: The source.
        out: Where to write the lock file.

    Returns:
        The lock file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(shard_documents(card))
    table = pa.table(
        {
            DOC_ID: pa.array([row[0] for row in rows], pa.string()),
            SHA256: pa.array([row[1] for row in rows], pa.string()),
            N_TOKENS: pa.array([row[2] for row in rows], pa.int64()),
        }
    ).replace_schema_metadata(
        {
            SOURCE_KEY: card.name.encode(),
            REVISION_KEY: card.revision.encode(),
            TOKENIZER_KEY: f"{card.count_tokenizer}@{card.count_tokenizer_revision}".encode(),
        }
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"{card.name}: {len(rows)} documents locked into {path}", flush=True)
    return path


def compare(source: str, locked: list[tuple[str, str]], collected: list[tuple[str, str]]) -> Verification:
    """Compare two lists of documents, each an id and a digest in order, and say what differs.

    Args:
        source: The source's name, for the report.
        locked: What the lock file holds.
        collected: What the shards hold.

    Returns:
        What is missing, extra and changed, and whether the documents in common keep their order.
    """
    locked_digests = dict(locked)
    collected_digests = dict(collected)
    missing = [doc_id for doc_id, _ in locked if doc_id not in collected_digests]
    extra = [doc_id for doc_id, _ in collected if doc_id not in locked_digests]
    changed = [
        doc_id for doc_id, sha in locked if doc_id in collected_digests and collected_digests[doc_id] != sha
    ]
    shared = set(locked_digests) & set(collected_digests)
    reordered = [d for d, _ in locked if d in shared] != [d for d, _ in collected if d in shared]
    return Verification(source, len(locked), len(collected), missing, extra, changed, reordered)


def verify(card: assets.SourceCard, lock_path: str) -> Verification:
    """Compare a source's collected shards with a lock file, print what differs and refuse if anything does.

    Args:
        card: The source.
        lock_path: A lock file `write` made.

    Returns:
        What was compared, when the shards are exactly what the lock holds.

    Raises:
        ValueError: The shards differ from the lock, or the lock is another source's.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(lock_path)
    named = (table.schema.metadata or {}).get(SOURCE_KEY, b"").decode()
    if named and named != card.name:
        raise ValueError(f"{lock_path} is the lock of {named}, and this is {card.name}")
    locked = list(zip(table.column(DOC_ID).to_pylist(), table.column(SHA256).to_pylist(), strict=True))
    collected = [(doc_id, sha) for doc_id, sha, _ in shard_documents(card)]
    result = compare(card.name, locked, collected)
    print(result.report(), flush=True)
    if not result.ok:
        raise ValueError(f"{card.name} does not match its lock")
    return result
