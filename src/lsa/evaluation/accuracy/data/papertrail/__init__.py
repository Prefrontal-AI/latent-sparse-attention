"""PaperTrail - its samples rebuilt from the benchmark's kit into one file of questions and one context each.

  lsa data prepare papertrail                          # every sample at every length and control, as a CPU job
  lsa data prepare papertrail --max-length 1000000     # the lengths a one-node server holds

The kit is `data-assembly/papertrail/kit` in this repository - the paper ids, the benchmark's own text and the checksums,
and no sentence of any real paper ([rebuild](rebuild.py)). The card pins its digest, so an edited kit is refused.
The preparer fetches each paper from arXiv at its pinned version into a cache under the data
root, converts it ([convert](convert.py)), patches the hosts ([patch](patch.py)), assembles every context
([compose](compose.py)) and refuses any paper or context whose checksum differs from the benchmark's.

Each sample is read at its six lengths and under its four controls - `C1` the question alone, `C2_hosts` the
unpatched host papers, `C3_synthetic` our text alone and `C4_background` the background papers alone. A
condition is two files, so a 10M context is stored once rather than once per question:

    <root>/papertrail/<sample>_<condition>.jsonl          the 36 questions, their answer keys and the paper order
    <root>/papertrail/<sample>_<condition>.context.txt    the context every question of the condition reads
    <root>/papertrail/html/<id>v<version>.html.gz         the arXiv pages, fetched once

A prompt is the context with the question's own closing part after it, which asks for an `EVIDENCE:` and an
`ANSWER:` line - or under `C1` the question alone, worded for a reader without the papers. Every question of a
condition shares the context as its prefix, so a server with the radix cache prefills it once.
The manifest entry of a condition records its longest prompt in the card's tokenizer, chat template included,
so a run skips the lengths its server cannot hold.
"""

import argparse
import datetime
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lsa
from lsa import assets, io
from lsa.data import manifests
from lsa.data.tokenizer import load_tokenizer, tokenizer_checkpoint, tokenizer_repo
from lsa.evaluation.accuracy.data import DATA_ROOT, papertrail_card, papertrail_path
from lsa.evaluation.accuracy.data.papertrail.rebuild import (
    Rebuilt,
    convert_papers,
    fetch_pages,
    needed_papers,
    read_kit,
    rebuild_sample,
)
from lsa.evaluation.accuracy.data.papertrail_names import NOT_IN_CORPUS, QUESTION_ONLY, conditions

# What the preparing job asks for - 1,246 papers fetched 3 s apart, then one tokenizer over each context, no GPU
PARAMETERS: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 8, "mem_gb": 96, "timeout_min": 6 * 60}

# Seconds between two requests to arXiv
FETCH_DELAY = 3.0

# The benchmark's definition, kept in this repository beside `configs/` - found from the package the same way
KIT = Path(lsa.__file__).resolve().parents[2] / "data-assembly" / "papertrail" / "kit"

# The answer key a row carries, as the benchmark's questions name it
ANSWER_KEYS: tuple[str, ...] = (
    "trail",
    "tier",
    "question",
    "answer_type",
    "answer",
    "answer_ids",
    "answer_papers",
    "check_papers",
    "tol",
    "unit",
)


def context_path(path: str | os.PathLike[str]) -> Path:
    """Return where the context of a condition's question file sits, beside it."""
    return Path(path).with_suffix(".context.txt")


def question_row(question: dict[str, Any], sample: str, condition: str, papers: list[str]) -> dict[str, Any]:
    """Turn one of the benchmark's questions into the row a condition's file stores.

    `closing` is what follows the context in the prompt, or under `C1` the whole prompt.
    """
    closing = question["prompt_no_context"] if condition == QUESTION_ONLY else question["prompt_suffix"]
    return {
        "id": f"{sample}/{condition}/{question['id']}",
        "sample": sample,
        "condition": condition,
        "qid": question["id"],
        **{key: question.get(key) for key in ANSWER_KEYS},
        "closing": closing,
        "papers": papers,
    }


def read_rows(path: str | os.PathLike[str], not_in_corpus: bool) -> list[dict[str, Any]]:
    """Read a condition's scored or not-in-corpus questions with their prompts composed.

    Args:
        path: The condition's question file.
        not_in_corpus: The not-in-corpus questions rather than the scored ones.

    Returns:
        The rows, each with its `prompt` - the condition's context and the question's closing part.
    """
    context_file = context_path(path)
    context = context_file.read_text() if context_file.exists() else ""
    rows = [row for row in io.read_jsonl(path) if (row["tier"] == NOT_IN_CORPUS) == not_in_corpus]
    return [{**row, "prompt": context + row["closing"]} for row in rows]


# The field a task's doc carries its answer in, as json - a number for one question and a list of spellings for
# another is no Arrow column, and lm-eval builds its docs into one
ANSWER_JSON = "answer_json"


def as_docs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows as a task's docs, each answer moved into `ANSWER_JSON`."""
    return [
        {
            **{key: value for key, value in row.items() if key != "answer"},
            ANSWER_JSON: json.dumps(row["answer"]),
        }
        for row in rows
    ]


def answer_of(doc: dict[str, Any]) -> Any:
    """Return a question's answer, from a task's doc or a prepared row."""
    return json.loads(doc[ANSWER_JSON]) if ANSWER_JSON in doc else doc.get("answer")


def longest_prompt_tokens(context: str, closings: list[str], tokenizer: Any) -> int:
    """Count the longest prompt of a condition in tokens, under the tokenizer's chat template.

    The closings differ by a few hundred characters at most, so the longest one in characters is counted.
    """
    longest = max(closings, key=len)
    messages = [{"role": "user", "content": context + longest}]
    # Rendered then encoded - a tokenized template comes back as a dict of fields under transformers 5
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def html_cache(root: str) -> Path:
    """Return where the arXiv pages are kept under a data root."""
    return Path(papertrail_path(papertrail_card().tasks[0], QUESTION_ONLY, root)).parent / "html"


def write_condition(
    sample: str,
    condition: str,
    questions: list[dict[str, Any]],
    rebuilt: Rebuilt | None,
    root: str,
    revision: str,
    tokenizer: Any,
    tokenizer_name: str,
) -> None:
    """Write one condition's question file and context, and its manifest entry.

    Args:
        sample: The sample.
        condition: A control or a length.
        questions: The sample's questions, scored then not-in-corpus.
        rebuilt: The condition's rebuilt context, None for the question alone.
        root: The data root.
        revision: The kit's revision, which the entry records.
        tokenizer: The loaded tokenizer the prompt sizes are counted in.
        tokenizer_name: Its hub id, which the entry records.
    """
    start = time.perf_counter()
    path = papertrail_path(sample, condition, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    papers = [paper["id"] for paper in rebuilt.papers] if rebuilt else []
    context = rebuilt.context if rebuilt else ""
    context_entry: dict[str, Any] = {}
    if rebuilt:
        context_file = context_path(path)
        scratch = context_file.with_name(f".{context_file.name}.partial")
        scratch.write_text(context)
        scratch.replace(context_file)
        context_entry = {
            "context_sha256": io.file_sha256(context_file),
            "context_chars": len(context),
            "papers": len(papers),
            "verified": rebuilt.verified,
        }
    rows = [question_row(question, sample, condition, papers) for question in questions]
    io.write_jsonl(path, rows)
    entry = {
        "rows": len(rows),
        "sha256": io.file_sha256(path),
        "tokenizer": tokenizer_name,
        "generator": revision,
        "scored": sum(row["tier"] != NOT_IN_CORPUS for row in rows),
        "not_in_corpus": sum(row["tier"] == NOT_IN_CORPUS for row in rows),
        **context_entry,
        "prompt_max": longest_prompt_tokens(context, [row["closing"] for row in rows], tokenizer),
        "date": datetime.date.today().isoformat(),
        "seconds": round(time.perf_counter() - start, 1),
    }
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(papertrail_card())})
    manifests.write_benchmark_file(root, os.path.relpath(path, root), record)
    print(
        f"{path}: {len(rows)} rows, longest prompt {entry['prompt_max']} tokens, {entry['seconds']} s",
        flush=True,
    )


def kit_digest(directory: Path) -> str:
    """Return the sha256 of a kit's files - `sha256sum` of each, sorted by path, then the sha256 of that listing.

    The same digest as `find . -type f | sed 's|^\\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum` run
    inside the kit, so it can be checked without this code.

    Args:
        directory: The kit.

    Returns:
        The hex digest.
    """
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    listing = "".join(f"{io.file_sha256(path)}  {path.relative_to(directory).as_posix()}\n" for path in paths)
    return hashlib.sha256(listing.encode("utf8")).hexdigest()


def check_kit(directory: Path) -> str:
    """Return a kit's digest, refusing one other than the card's.

    Raises:
        ValueError: The kit is not the one the card pins, or is not a directory.
    """
    if not directory.is_dir():
        raise ValueError(
            f"{directory} is not a directory - the kit is data-assembly/papertrail/kit in this repository"
        )
    digest = kit_digest(directory)
    if digest != papertrail_card().revision:
        raise ValueError(
            f"{directory} has digest {digest} and the papertrail card pins {papertrail_card().revision} - "
            "a kit that was edited is not the benchmark, so restore it or update the card with the new digest"
        )
    return digest


def prepare_papertrail(
    root: str = DATA_ROOT,
    kit: str | None = None,
    cache: str | None = None,
    delay: float = FETCH_DELAY,
    max_length: int | None = None,
    tokenizer_name: str | None = None,
    keep_mismatched: bool = False,
) -> None:
    """Rebuild every sample at every condition up to a length, check it, and write each one's files and manifest entry.

    Args:
        root: The data root.
        kit: A copy of the kit the card pins, or None for `data-assembly/papertrail/kit` in this repository.
        cache: Where the arXiv pages are kept, or None for the data root's.
        delay: Seconds between two requests to arXiv.
        max_length: The longest length to prepare, or None for every length of the card.
        tokenizer_name: The tokenizer to count in, or None for the one the card names.
        keep_mismatched: Write the contexts that differ from the benchmark's, marked unverified, rather than refuse.

    Raises:
        ValueError: The kit has no context for a condition asked for, or a paper or a context differs from the
            benchmark's and `keep_mismatched` is off.
    """
    card = papertrail_card()
    kit_dir = Path(kit) if kit else KIT
    revision = check_kit(kit_dir)
    wanted = conditions(max_length)
    with_context = [condition for condition in wanted if condition != QUESTION_ONLY]
    kit_data = read_kit(kit_dir, card.tasks)
    converted, mismatched = convert_papers(
        kit_data, needed_papers(kit_data, with_context), Path(cache) if cache else html_cache(root), delay
    )
    rebuilt: dict[str, dict[str, Rebuilt]] = {}
    for sample in card.tasks:
        rebuilt[sample], hosts_mismatched = rebuild_sample(kit_data, sample, converted, with_context)
        mismatched += hosts_mismatched
    missing = [
        f"{sample}/{condition}"
        for sample in card.tasks
        for condition in with_context
        if condition not in rebuilt[sample]
    ]
    if missing:
        raise ValueError(f"the kit at {kit_dir} has no context for {missing}")
    unverified = [
        f"{sample}/{condition}"
        for sample, by_condition in rebuilt.items()
        for condition, one in by_condition.items()
        if not one.verified
    ]
    if mismatched or unverified:
        print(
            f"papers converted differently: {mismatched}\ncontexts not byte-identical: {unverified}",
            flush=True,
        )
        if not keep_mismatched:
            raise ValueError(
                f"{len(mismatched)} papers and {len(unverified)} contexts differ from the benchmark's - "
                "arXiv has most likely rendered them again; --keep-mismatched writes them marked unverified"
            )
    tokenizer = load_tokenizer(tokenizer_name or tokenizer_checkpoint(card), trust_remote_code=True)
    for sample in card.tasks:
        questions = kit_data.questions(sample)
        for condition in wanted:
            write_condition(
                sample,
                condition,
                questions,
                rebuilt[sample].get(condition),
                root,
                revision,
                tokenizer,
                tokenizer_name or tokenizer_repo(card),
            )


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what the PaperTrail files are prepared with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument(
        "--kit",
        default=None,
        help="a copy of the kit the card pins - data-assembly/papertrail/kit in this repository when unset",
    )
    parser.add_argument(
        "--cache", default=None, help="where the arXiv pages are kept - the data root's when unset"
    )
    parser.add_argument(
        "--delay", type=float, default=FETCH_DELAY, help="seconds between two requests to arXiv"
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="the longest length - every one when unset"
    )
    parser.add_argument(
        "--tokenizer", default=None, help="the tokenizer to count in - GLM-5.3-Flash's when unset"
    )
    parser.add_argument(
        "--keep-mismatched",
        action="store_true",
        help="write contexts that differ from the benchmark's, marked unverified, rather than refuse",
    )


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs to prepare the files the command line names, and what it is called with.

    The kit is checked against the card and the arXiv pages the cache does not hold are fetched here, before
    submission, since a compute node has no internet.

    Args:
        card: The PaperTrail card, which the preparer reads for itself.
        args: The parsed command line.

    Returns:
        `prepare_papertrail`, and the absolute root, the kit, the cache, the delay, the longest length, the
        tokenizer's name and whether mismatched contexts are kept.
    """
    cache = os.path.abspath(args.cache) if args.cache else None
    if not args.dry_run:
        kit_dir = Path(args.kit) if args.kit else KIT
        check_kit(kit_dir)
        kit = read_kit(kit_dir, papertrail_card().tasks)
        wanted = [condition for condition in conditions(args.max_length) if condition != QUESTION_ONLY]
        fetch_pages(
            kit,
            needed_papers(kit, wanted),
            Path(cache) if cache else html_cache(os.path.abspath(args.root)),
            args.delay,
        )
    arguments = (args.kit, cache, args.delay, args.max_length, args.tokenizer, args.keep_mismatched)
    return prepare_papertrail, (os.path.abspath(args.root), *arguments)
