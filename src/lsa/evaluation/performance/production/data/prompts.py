"""Build the prompt files a production run sends - a workload card's, one file per length.

    lsa data prepare production                     # the card's seven files, about four minutes, as a CPU job
    lsa data prepare production_qwen38              # Qwen 3.8's, sized in its own tokens
    lsa data prepare production --lengths 131072 --out /tmp/prompts

The card names the tokenized dataset and the source the text is decoded from, the lengths and the entries
per file. Each file is ShareGPT-shaped, which is what sglang's random dataset reads: the client tokenizes an
entry, cuts it to exactly the length and sends the ids. Entry i starts at the same row at every length, so the
first L tokens of stream i are a prefix of every longer prompt of it and the ladder is nested. The manifest
beside the files keeps each entry's rows, the tokens it encodes to and each file's sha256.
"""

import argparse
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lsa import assets, io
from lsa.data import manifests
from lsa.data.corpus import SourceRows
from lsa.data.tokenizer import load_tokenizer, tokenizer_checkpoint, tokenizer_repo

# What the building job asks for - the corpus is memory-mapped and one tokenizer decodes it, no GPU
PARAMETERS: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 60}


def corpus(card: assets.WorkloadCard) -> assets.TokenizedCard:
    """Return the tokenized dataset a workload's text is decoded from."""
    return assets.tokenized(card.tokenized)


def corpus_path(card: assets.WorkloadCard) -> str:
    """Return the tokenized rows of a workload's source."""
    return str(corpus(card).bin_path(card.source))


def row_tokens(card: assets.WorkloadCard) -> int:
    """Return the tokens one tokenized row holds, as the workload's tokenized card says."""
    return corpus(card).sample_length


def rows_for(card: assets.WorkloadCard, length: int) -> int:
    """Return the rows one entry of this length decodes - one over, since the text round trip may merge tokens."""
    return math.ceil(length / row_tokens(card)) + 1


def entry_rows(card: assets.WorkloadCard, entry: int, length: int) -> tuple[int, int]:
    """Return the corpus rows [start, end) of one entry.

    A nested entry owns the span its longest length needs, so it is a prefix of itself at every length. An entry of
    a card that counts its entries per length owns only its own length's span.
    """
    start = entry * rows_for(card, max(card.lengths) if card.nested else length)
    return start, start + rows_for(card, length)


def build(card: assets.WorkloadCard, length: int, rows: Any, tokenizer: Any, root: str) -> dict[str, Any]:
    """Write one length's file.

    Args:
        card: The workload card.
        length: The prompt length in tokens.
        rows: The token corpus as [row count, row tokens] uint32.
        tokenizer: The tokenized dataset's tokenizer.
        root: Where the file goes.

    Returns:
        Its manifest entry - file name, sha256, and each entry's rows, tokens and characters.

    Raises:
        ValueError: An entry encodes to fewer tokens than the length, so the client could not cut it there.
    """
    dataset, entries = [], []
    for entry in range(card.entries_at(length)):
        start, end = entry_rows(card, entry, length)
        text = tokenizer.decode(rows[start:end].reshape(-1).tolist())
        # The same call sglang's random dataset makes
        tokens = len(tokenizer.encode(text))
        if tokens < length:
            raise ValueError(f"entry {entry} of {length} encodes to {tokens} tokens")
        dataset.append({"conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": ""}]})
        entries.append({"rows": [start, end], "tokens": tokens, "chars": len(text)})
    path = card.file_path(length, root)
    # Compact and without a trailing newline, as the files these reproduce byte for byte were written
    with open(path + ".tmp", "w") as target:
        json.dump(dataset, target, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return {"file": os.path.basename(path), "sha256": io.file_sha256(path), "entries": entries}


def built_files(root: str) -> dict[str, Any]:
    """Return the files a manifest under a root already records, or none."""
    try:
        return manifests.read_workload(root).files
    except FileNotFoundError:
        return {}


def manifest(card: assets.WorkloadCard, files: dict[str, Any]) -> manifests.WorkloadManifest:
    """Return a workload's manifest - the corpus the files came from, one entry per length, and the card."""
    return manifests.WorkloadManifest(
        declared=manifests.declared(card),
        corpus=corpus_path(card),
        row=row_tokens(card),
        tokenizer=tokenizer_repo(corpus(card)),
        entries=card.entries,
        files=files,
        git_sha=io.git_sha(Path(__file__).parent) or "",
        date=manifests.today(),
    )


def build_files(name: str, lengths: list[int], out: str) -> str:
    """Build one file per length under a directory, and the manifest that records them.

    Args:
        name: The workload card, by name.
        lengths: The prompt lengths, in tokens.
        out: Where the files go.

    Returns:
        The directory.
    """
    card = assets.workload(name)
    tokenizer = load_tokenizer(tokenizer_checkpoint(corpus(card)), trust_remote_code=True)
    rows = SourceRows(corpus(card), sources=[card.source]).source_rows(card.source)
    os.makedirs(out, exist_ok=True)
    files = built_files(out)
    for length in lengths:
        files[str(length)] = build(card, length, rows, tokenizer, out)
        print(f"{length}: {[entry['tokens'] for entry in files[str(length)]['entries']]} tokens", flush=True)
    manifests.write_workload(out, manifest(card, files))
    return out


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what the workload's files are built with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("--lengths", type=int, nargs="+", help="the lengths to build - the card's by default")
    parser.add_argument("--out", help="where the files go - the card's path by default")


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs to build the files the command line names, and what it is called with.

    Args:
        card: The workload card.
        args: The parsed command line.

    Returns:
        `build_files`, and the card's name, the lengths and the absolute directory.
    """
    workload = assets.workload(card.name)
    return build_files, (
        workload.name,
        args.lengths or workload.lengths,
        os.path.abspath(args.out or workload.path),
    )
