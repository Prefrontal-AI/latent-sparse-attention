"""Prepare one dataset card - whatever its kind needs before a training run can read it.

    lsa data prepare glm53_flash                    # every source the card names
    lsa data prepare glm53_flash --sources edgebench
    lsa data prepare edgebench --sessions <run shard dirs...>

`prepare(card)` is the function a job runs, and what it does follows from the card's kind.
`lsa data prepare` submits it, with the options `add_options` declares.

**A tokenized card** is some sources' text in one tokenizer's ids, so preparing it tokenizes the source cards
it names, with the model card it names, into rows of its sample length, and measures what that produced into
the dataset's manifest, `corpus.json` beside the rows.
A row is token ids and nothing else - no vocabulary, no name of what produced it - so a run that reads a
corpus tokenized for another model does not read a worse corpus, it reads ids that mean different words.
Where the vocabularies differ in size it is worse than that: Qwen3.5's is 248,320 against GLM-5.3-Flash's
154,880, so a third of a Qwen corpus's ids do not exist in that model at all. That is why one corpus is
tokenized per tokenizer, and why its manifest records which.

What is written is the format [the loader](corpus.py) reads, and the rule it is built on is the one the
corpus exists for: **a row is a contiguous span of one document**, never two concatenated. A document gives
`floor(tokens / sample_length)` rows and its remainder is dropped, which is what keeps every sample at every
sequence length real text read in order.

Determinism is by construction rather than by seed. Sources are taken in the order the card lists them,
shards in sorted order, documents in the order their shard holds them, and the row a document lands on
follows from the documents before it - so two runs of this produce the same bytes, and the held-out split,
which is a hash of the document id, picks the same documents whatever the tokenizer was.

**A source card collected into its own origin** is the agent trajectories, which [trajectories](trajectories.py) renders
from the harness's run directories into the directory the card names as where it came from.

**Any other source card** - one streamed from the hub, or read from elsewhere on the cluster - was collected
by a pipeline that predates this repository and has nothing to prepare here, and neither has a benchmark,
whose prompts a harness renders, or a workload, which its builder writes.
"""

import argparse
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from lsa import assets, io
from lsa.data import manifests
from lsa.data.constants import Collected, CorpusFiles
from lsa.data.corpus import document_extents
from lsa.data.tokenizer import load_tokenizer, tokenizer_checkpoint
from lsa.data.trajectories import collect_trajectories


class Tokenization:
    """How a corpus is tokenized, which no card declares because the bytes do not depend on it.

    The row length is the card's - a run never sets it, because a row read at the wrong length is not a worse
    sample, it is a different document.
    """

    # Documents encoded in one call to the tokenizer. Large enough that the Rust tokenizer's own threading
    # has work, small enough that one batch of long documents is not the peak memory of the job
    DOCUMENTS_PER_CALL: int = 32


@dataclass
class TokenizedSource:
    """What tokenizing one source produced."""

    source: str
    # Rows written, each `sample_length` tokens
    rows: int
    # Documents that gave at least one whole row
    documents: int
    # Documents dropped for being shorter than one row
    too_short: int


def prepare(
    card: assets.DatasetCard, sources: list[str] | None = None, sessions: list[str] | None = None
) -> Path:
    """Do what one card's kind needs before a run can read it, and return where it was written.

    Args:
        card: The dataset card to prepare.
        sources: The sources of a tokenized card to tokenize, or None for every one it lists - a long
            corpus is split across jobs this way.
        sessions: The harness run directories the trajectories are collected from, which their card does
            not record.

    Returns:
        The directory the card describes, now written.

    Raises:
        ValueError: The card is a source this module does not collect, a benchmark or a workload - there
            is nothing to prepare here for any of them.
        ValueError: The trajectories were given no run directories to collect from.
    """
    require_preparable(card, sessions)
    if isinstance(card, assets.TokenizedCard):
        return tokenize(card, sources)
    # Past the refusal only the source collected here is left
    return collect_trajectories(cast(assets.SourceCard, card), [Path(session) for session in sessions or []])


def require_preparable(card: assets.DatasetCard, sessions: list[str] | None = None) -> None:
    """Refuse a card this module has nothing to prepare for, or the trajectories with nowhere to collect from.

    Args:
        card: The dataset card to prepare.
        sessions: The harness run directories the trajectories are collected from.

    Raises:
        ValueError: The card is a source this module does not collect, a benchmark or a workload - there
            is nothing to prepare here for any of them.
        ValueError: The trajectories were given no run directories to collect from.
    """
    if isinstance(card, assets.TokenizedCard):
        return
    if isinstance(card, assets.SourceCard) and is_collected_here(card):
        if not sessions:
            raise ValueError(
                f"{card.name} is collected from harness run directories - name them with --sessions"
            )
        return
    if isinstance(card, assets.SourceCard):
        raise ValueError(
            f"{card.name} was collected by a pipeline that predates this repository - there is nothing to "
            "prepare here, and a corpus is tokenized from it by preparing that corpus's card"
        )
    raise ValueError(
        f"{card.name} is a {type(card).__name__} - the harness that reads it prepares it, and there is "
        "nothing to prepare here"
    )


def is_collected_here(card: assets.SourceCard) -> bool:
    """Say whether this module renders a source - one collected into the directory it names as its origin.

    The renderer writes one folder per source under the directory it is pointed at, so a source whose folder
    sits directly under its `collected_from` is one it wrote. A source read from elsewhere on the cluster
    names the directory it was read from, which is not its own.
    """
    return bool(card.collected_from) and Path(card.path).parent == Path(card.collected_from)


def tokenize(card: assets.TokenizedCard, sources: list[str] | None = None) -> Path:
    """Tokenize a card's sources into the rows it describes, and measure them into its manifest.

    Args:
        card: A tokenized card, naming its sources, its tokenizer and its sample length.
        sources: The sources to tokenize, or None for every one the card lists - a long corpus is split
            across jobs this way.

    Returns:
        The corpus directory.

    Raises:
        ValueError: A source asked for is not one the card lists.
    """
    wanted = sources or card.sources
    unknown = sorted(set(wanted) - set(card.sources))
    if unknown:
        raise ValueError(f"{card.name} has no source {unknown} - it has {card.sources}")
    snapshot = tokenizer_checkpoint(card)
    # The job's environment turns the tokenizer's threads off for processes that fork, and this one never does
    # => a batch encodes on every core the job has, some 3.5 times faster on 16 than on one
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = load_tokenizer(snapshot, require_fast=True)
    out = Path(card.path)

    print(f"tokenizing {len(wanted)} sources of {card.name} into {out}", flush=True)
    for name in wanted:
        result = tokenize_source(assets.source(name), out, tokenizer, card.sample_length)
        print(
            f"  {result.source}: {result.rows} rows from {result.documents} documents, "
            f"{result.too_short} too short",
            flush=True,
        )
    manifest = measure(card, snapshot, len(tokenizer))
    manifests.write_tokenized(out, manifest)
    shutil.copy(Path(snapshot) / "tokenizer.json", out / "tokenizer.json")
    print(f"{sum(manifest.rows.values())} rows - {manifest.tokens / 1e9:.2f}B tokens, measured into {out}")
    return out


def tokenize_source(
    source: assets.SourceCard, out: Path, tokenizer: Any, sample_length: int
) -> TokenizedSource:
    """Tokenize one source's documents into a bin and a manifest.

    Args:
        source: The source, whose name names the files written.
        out: Directory to write `<source>.bin` and `<source>.manifest.parquet` into.
        tokenizer: A HuggingFace fast tokenizer.
        sample_length: Tokens per row.

    Returns:
        What was written.

    Raises:
        FileNotFoundError: The source has no shards.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = sorted(Path(source.path).glob(Collected.SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no {Collected.SHARD_GLOB} under {source.path}")

    out.mkdir(parents=True, exist_ok=True)
    scratch = out / f"{source.name}{CorpusFiles.BIN_SUFFIX}.partial"
    doc_ids: list[str] = []
    row_starts: list[int] = []
    row_counts: list[int] = []
    rows = 0
    too_short = 0

    with scratch.open("wb") as sink:
        for shard in shards:
            table = pq.read_table(shard, columns=[Collected.DOC_ID, Collected.TEXT])
            ids = table.column(Collected.DOC_ID).to_pylist()
            texts = table.column(Collected.TEXT).to_pylist()
            for start in range(0, len(texts), Tokenization.DOCUMENTS_PER_CALL):
                batch = texts[start : start + Tokenization.DOCUMENTS_PER_CALL]
                encoded = tokenizer(batch, add_special_tokens=False).input_ids
                for offset, tokens in enumerate(encoded):
                    whole = len(tokens) // sample_length
                    if not whole:
                        too_short += 1
                        continue
                    kept = np.asarray(tokens[: whole * sample_length], dtype=np.uint32)
                    sink.write(kept.tobytes())
                    doc_ids.append(str(ids[start + offset]))
                    row_starts.append(rows)
                    row_counts.append(whole)
                    rows += whole

    # The manifest lands first and the bin is renamed into place last, because the loader takes a source as
    # present when its bin exists - so the other order leaves a killed job looking like a tokenized source
    # whose manifest is missing, and a run reading it fails on a file rather than on a half-written corpus
    manifest = pa.table(
        {
            CorpusFiles.DOC_ID: pa.array(doc_ids, pa.string()),
            CorpusFiles.ROW_START: pa.array(row_starts, pa.int64()),
            CorpusFiles.N_SAMPLES: pa.array(row_counts, pa.int64()),
        }
    )
    pq.write_table(manifest, out / f"{source.name}{CorpusFiles.MANIFEST_SUFFIX}")
    scratch.replace(out / f"{source.name}{CorpusFiles.BIN_SUFFIX}")
    return TokenizedSource(source=source.name, rows=rows, documents=len(doc_ids), too_short=too_short)


def measure(card: assets.TokenizedCard, snapshot: str, vocab_size: int) -> manifests.TokenizedManifest:
    """Measure what is tokenized under a card, every source on disk rather than the ones just tokenized.

    Counted off each source's own parquet manifest, so tokenizing a dataset a few sources at a time - which is
    how a long job is split across nodes - leaves one record of the whole dataset rather than a record of
    whichever piece finished last.

    Args:
        card: The tokenized dataset.
        snapshot: The snapshot the tokenizer came from.
        vocab_size: How many ids the tokenizer has.

    Returns:
        The dataset's manifest, with the card's declared fields.
    """
    root = Path(card.path)
    sources = [name for name in card.sources if card.bin_path(name).exists()]
    extents = {name: document_extents(root, name) for name in sources}
    return manifests.TokenizedManifest(
        declared=manifests.declared(card),
        tokenizer=snapshot,
        tokenizer_revision=manifests.snapshot_revision(snapshot),
        vocab_size=vocab_size,
        sample_length=card.sample_length,
        sources=sources,
        rows={name: sum(count for _, count in extents[name].values()) for name in sources},
        documents={name: len(extents[name]) for name in sources},
        git_sha=io.git_sha(Path(__file__).parent) or "",
        date=manifests.today(),
    )


# What a job asks for - the trajectories render a session per worker, and tokenizing is one pass over the text
PARAMETERS: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 32, "mem_gb": 128, "timeout_min": 24 * 60}


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what a source or a tokenized card is prepared with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("--sources", nargs="*", default=None, help="sources to tokenize, or all of them")
    parser.add_argument(
        "--sessions", nargs="*", default=None, help="harness run directories, for the trajectories"
    )


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs for one card, and what it is called with.

    Args:
        card: The card to prepare.
        args: The parsed command line.

    Returns:
        `prepare`, and the card with the sources and the absolute run directories the command line named.

    Raises:
        ValueError: There is nothing to prepare for the card, so nothing is submitted.
    """
    sessions = [str(Path(session).resolve()) for session in args.sessions] if args.sessions else None
    require_preparable(card, sessions)
    return prepare, (card, args.sources, sessions)
