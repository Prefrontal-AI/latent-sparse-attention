"""Collect a hub source into the shards a corpus is tokenized from.

    lsa data prepare gutenberg                      # one source, as a CPU job that reads the hub
    lsa data prepare gutenberg --limit-docs 30      # the first 30 documents, to see that it runs
    lsa data prepare gutenberg --verify <lock>      # compare the collected shards with a lock file
    lsa data prepare gutenberg --write-lock <out>   # write the lock file of the collected shards

This is the pipeline that first collected LongDocs, ported so a reader can assemble the same documents from the
hub rather than receive them. Each rule below is the first collector's, and a change to one is a different corpus.

**What a document is** depends on the card's `collector`. A `text` source is one row of its hub dataset. A
`repositories` source is the files of one repository, joined in the order of their paths, which The Stack stores
scattered rather than together - so each language is read in the stream's order until its files hold the card's
`buffer_chars` characters, and a repository's files beyond that slice are not in its document. A `traces` source
is one agent trajectory rendered as text.

**What is kept** is a document of at least the card's `min_tokens`, counted by the card's tokenizer at its pinned
revision with no special tokens. Documents too short in characters to ever reach it are dropped before they are
tokenized. Collection stops at the first batch that takes the kept tokens to `max_tokens`, so the last batch
can pass it.

**What is not reproducible from the card alone** is the order a hub dataset streams in and its head on the day it
was collected. A card pins the revision where one is known, and `code` is the one that is not - its collector
streamed `main` without recording it. A lock file is how that is told apart from a mistake: it holds the sha256
of every document that was collected, and [verifying](lock.py) it names each one a collection does not reproduce.

The shards are the format [the tokenizer](tokenize.py) reads: `shard_NNNNN.parquet` of `doc_id`, `source`,
`license`, `n_tokens` and `text`, and a `_DONE` marker holding what was counted, written last.
"""

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from lsa import assets, env
from lsa.assets.constants import Collector
from lsa.data.constants import Collected

Document = tuple[str, str]
Row = Mapping[str, Any]


class Collection:
    """The numbers and the licences the first collector fixed, which a card does not declare because they are
    how a source is read rather than what it is.
    """

    # Documents encoded in one call of the tokenizer, and written to one shard
    BATCH: int = 256
    SHARD_DOCS: int = 2000
    # A document of `min_tokens` has at least this many characters per token
    # => 2.5 is below the densest text, code at about 2.7, so no document that could be kept is dropped unread
    MIN_CHARS_PER_TOKEN: float = 2.5
    # The licences a file or a trajectory may carry. It is not "mit/apache/bsd" alone, which is what every
    # document of `code` was labelled - the list also holds ISC, MPL-2.0, CC0-1.0, CC-BY-4.0 and others
    PERMISSIVE: frozenset[str] = frozenset(
        {
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "BSD-3-Clause-Clear",
            "ISC",
            "Unlicense",
            "0BSD",
            "Zlib",
            "CC0-1.0",
            "CC-BY-4.0",
            "MPL-2.0",
        }
    )
    SPLIT: str = "train"
    # Where a source's collected shards are first written, beside their place until the collection is whole
    STAGING_SUFFIX: str = ".partial"


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Return `items` in lists of `size`, the last one shorter."""
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def text_documents(rows: Iterable[Row], name: str, id_field: str, text_field: str) -> Iterator[Document]:
    """Return one document for each row of a dataset of ready-to-use documents.

    Args:
        rows: The dataset's rows, in the order it streams.
        name: The source's name, which a row without an id is numbered under.
        id_field: The row field that holds the document's id.
        text_field: The row field that holds its text.

    Yields:
        The document's id and text.
    """
    for index, row in enumerate(rows):
        yield str(row.get(id_field) or f"{name}-{index}"), row[text_field]


def repository_documents(
    open_rows: Callable[[str], Iterable[Row]], languages: Sequence[str], buffer_chars: int
) -> Iterator[Document]:
    """Return one document for each repository in a slice of each language, files joined in path order.

    A language is opened only once the one before it is used up, so a collection that stops early never
    reads the languages after the one it stopped in.

    Args:
        open_rows: Opens one language's files as the dataset streams them.
        languages: The languages, in the order they are read.
        buffer_chars: Characters of a language's files read before its repositories are grouped.

    Yields:
        The document's id, `<language>/<repository>`, and its text.
    """
    for language in languages:
        repositories: dict[str, list[tuple[str, str]]] = {}
        buffered = 0
        for row in open_rows(language):
            licences = row["max_stars_repo_licenses"]
            # A file that lists no licence passes, and only its first listed licence is looked at
            if licences and licences[0] not in Collection.PERMISSIVE:
                continue
            repositories.setdefault(row["max_stars_repo_name"], []).append(
                (row["max_stars_repo_path"], row["content"])
            )
            buffered += len(row["content"])
            if buffered >= buffer_chars:
                break
        for repository, files in repositories.items():
            files.sort()
            yield f"{language}/{repository}", "\n\n".join(f"# {path}\n{content}" for path, content in files)


def render_trajectory(messages: Iterable[Mapping[str, Any]]) -> str:
    """Return one agent trajectory as text - each message's role, its reasoning, its content and its tool calls."""
    parts = []
    for message in messages:
        parts.append(f"<{message.get('role', '?')}>")
        for field_name in ("reasoning_content", "content"):
            if message.get(field_name):
                parts.append(message[field_name])
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"]))
    return "\n".join(parts)


def trace_documents(open_rows: Callable[[str], Iterable[Row]], configs: Sequence[str]) -> Iterator[Document]:
    """Return one document for each trajectory whose repository's licence is permissive.

    Args:
        open_rows: Opens one config of the dataset as it streams.
        configs: The configs, in the order they are read.

    Yields:
        The document's id, `<config>:<trajectory>`, and its text.
    """
    for config in configs:
        for row in open_rows(config):
            if row.get("license") not in Collection.PERMISSIVE:
                continue
            identifier = row.get("trajectory_id") or row.get("instance_id")
            yield f"{config}:{identifier}", render_trajectory(row["trajectory"])


def long_documents(
    documents: Iterable[Document],
    count_tokens: Callable[[list[str]], list[int]],
    min_tokens: int,
    max_tokens: int = 0,
    limit_docs: int = 0,
) -> Iterator[tuple[str, str, int]]:
    """Return the documents of at least `min_tokens`, stopping at the first batch that reaches `max_tokens`.

    Args:
        documents: Each document's id and text, in the order they were collected.
        count_tokens: A list of texts to the number of tokens of each.
        min_tokens: The shortest a kept document is.
        max_tokens: Stop after the batch that takes the kept tokens to this many, 0 for no cap.
        limit_docs: Read this many documents and stop, 0 for all of them - to see that a source runs.

    Yields:
        The document's id, its text and its tokens.
    """
    floor = int(min_tokens * Collection.MIN_CHARS_PER_TOKEN)

    def long_enough() -> Iterator[Document]:
        for seen, (doc_id, text) in enumerate(documents, start=1):
            if limit_docs and seen > limit_docs:
                return
            if text and len(text) >= floor:
                yield doc_id, text

    kept = 0
    for chunk in batched(long_enough(), Collection.BATCH):
        for (doc_id, text), tokens in zip(chunk, count_tokens([text for _, text in chunk]), strict=True):
            if tokens < min_tokens:
                continue
            kept += tokens
            yield doc_id, text, tokens
        if max_tokens and kept >= max_tokens:
            return


def write_shards(
    documents: Iterable[tuple[str, str, int]], out: Path, source: str, label: str
) -> dict[str, int]:
    """Write documents into numbered parquet shards under `out`, and the marker that says it is whole.

    The shards are written beside `out` and moved into place with the marker last, so a killed job leaves
    no directory that looks collected.

    Args:
        documents: Each document's id, text and tokens.
        out: Where the source's folder is to be.
        source: The source's name, which every row carries.
        label: The one licence label every row carries.

    Returns:
        The documents, the tokens and the shards written.

    Raises:
        FileExistsError: `out` is already there - a collection never replaces one, so move it aside.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists():
        raise FileExistsError(f"{out} is already there - a collection never replaces one, so move it aside")
    staging = out.with_name(out.name + Collection.STAGING_SUFFIX)
    if staging.exists():
        raise FileExistsError(f"{staging} is a collection that did not finish - remove it to start again")
    staging.mkdir(parents=True)
    schema = pa.schema(
        [
            (Collected.DOC_ID, pa.string()),
            (Collected.SOURCE, pa.string()),
            (Collected.LICENSE, pa.string()),
            (Collected.N_TOKENS, pa.int64()),
            (Collected.TEXT, pa.string()),
        ]
    )
    shards = 0
    count = 0
    tokens = 0
    for chunk in batched(documents, Collection.SHARD_DOCS):
        rows = [
            {
                Collected.DOC_ID: doc_id,
                Collected.SOURCE: source,
                Collected.LICENSE: label,
                Collected.N_TOKENS: n,
                Collected.TEXT: text,
            }
            for doc_id, text, n in chunk
        ]
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            staging / f"shard_{shards:05d}.parquet",
            compression="zstd",
        )
        shards += 1
        count += len(rows)
        tokens += sum(n for _, _, n in chunk)
    summary = {"n_docs": count, "kept_tokens": tokens, "n_shards": shards}
    (staging / Collected.DONE_MARKER).write_text(json.dumps(summary))
    staging.rename(out)
    return summary


def token_counter(card: assets.SourceCard) -> Callable[[list[str]], list[int]]:
    """Return the function that counts a card's documents, with the tokenizer and revision it pins.

    Args:
        card: A collected source.

    Returns:
        A list of texts to the number of tokens of each, with no special tokens.
    """
    from transformers import AutoTokenizer

    # The collection tokenizes batches, and a batch encodes on every core the job has
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(card.count_tokenizer, revision=card.count_tokenizer_revision)

    def count(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

    return count


def hub_rows(card: assets.SourceCard) -> Callable[[str | None], Iterable[Row]]:
    """Return a function that opens one part of a card's hub dataset as a stream, at the card's revision.

    Args:
        card: A collected source.

    Returns:
        A function from the part - a language or a config, None for a dataset with one - to its rows.
    """
    from datasets import load_dataset

    def open_rows(part: str | None) -> Iterable[Row]:
        arguments: dict[str, Any] = {
            "split": card.split or Collection.SPLIT,
            "streaming": True,
            "revision": card.revision or None,
        }
        if card.collector == Collector.REPOSITORIES:
            arguments["data_dir"] = f"data/{part}"
            return load_dataset(card.dataset, **arguments)
        return load_dataset(card.dataset, part, **arguments)

    return open_rows


def source_documents(
    card: assets.SourceCard, open_rows: Callable[[str | None], Iterable[Row]]
) -> Iterator[Document]:
    """Return a collected source's documents as its collector builds them, before any is dropped.

    Args:
        card: A collected source.
        open_rows: Opens a part of its hub dataset.

    Returns:
        The documents, lazily - a part is opened when the one before it is used up.

    Raises:
        ValueError: The card names a collector this module does not have.
    """
    if card.collector == Collector.TEXT:
        return text_documents(open_rows(None), card.name, card.id_field, card.text_field)
    if card.collector == Collector.REPOSITORIES:
        return repository_documents(lambda language: open_rows(language), card.subsets, card.buffer_chars)
    if card.collector == Collector.TRACES:
        return trace_documents(lambda config: open_rows(config), card.subsets)
    raise ValueError(f"{card.name}: this module has no collector {card.collector!r}")


def collect(card: assets.SourceCard, limit_docs: int = 0) -> Path:
    """Collect a source from the hub into the folder its card names, and return the folder.

    Args:
        card: A source with a `collector`.
        limit_docs: Read this many documents and stop, 0 for all of them.

    Returns:
        The folder of shards.
    """
    if not card.revision:
        print(f"{card.name}: the card pins no revision, so the hub's head is read - see its card", flush=True)
    summary = write_shards(
        long_documents(
            source_documents(card, hub_rows(card)),
            token_counter(card),
            card.min_tokens,
            card.max_tokens,
            limit_docs,
        ),
        Path(card.path),
        card.name,
        card.licence,
    )
    print(
        f"{card.name}: {summary['n_docs']} documents, {summary['kept_tokens'] / 1e9:.2f}B tokens, "
        f"{summary['n_shards']} shards into {card.path}",
        flush=True,
    )
    return Path(card.path)


def digest(text: str) -> str:
    """Return the sha256 of a document's text as utf-8, which is what a lock file records of it."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


# What a job asks for - a code language's slice is held in memory before it is grouped, and the tokenizer takes
# every core it is given
PARAMETERS: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 32, "mem_gb": 128, "timeout_min": 24 * 60}

# The hub is read, so the job is not offline as every other preparing job is
ENVIRONMENT: tuple[str, ...] = env.ONLINE_HF


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what a collected source is prepared with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument(
        "--limit-docs", type=int, default=0, help="read this many documents and stop, to see that it runs"
    )
    parser.add_argument(
        "--verify", default=None, metavar="LOCK", help="compare the collected shards with this lock file"
    )
    parser.add_argument(
        "--write-lock", default=None, metavar="OUT", help="write the lock file of the collected shards here"
    )


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs for a collected source - collecting it, or checking what was collected.

    Args:
        card: A source with a `collector`.
        args: The parsed command line.

    Returns:
        The function and what it is called with.

    Raises:
        ValueError: The card is not a collected source, or both a verification and a lock file were asked for.
    """
    if not isinstance(card, assets.SourceCard) or not card.collector:
        raise ValueError(f"{card.name} is not a source this module collects")
    if args.verify and args.write_lock:
        raise ValueError("--verify and --write-lock each read the collected shards - ask for one")
    from lsa.data import lock

    if args.verify:
        return lock.verify, (card, str(Path(args.verify).resolve()))
    if args.write_lock:
        return lock.write, (card, str(Path(args.write_lock).resolve()))
    return collect, (card, args.limit_docs)
