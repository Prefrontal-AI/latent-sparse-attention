"""LongBench v2 prompts - the official template and middle truncation, prepared once per context limit.

lm-eval's longbench2 tasks score by log-likelihood, which a chat endpoint cannot provide. The official
evaluation (THUDM/LongBench, pred.py) is generative: the model is asked to answer in the form
"The correct answer is (X)", and the context is truncated in the middle to the model's length. The prompt
depends on the tokenizer and the limit, so it is built here once and stored; `grading.longbench` reads the
answer back.

  lsa data prepare longbench2                          # at the card's context, as a CPU job
  lsa data prepare longbench2 --max-context 131072

The template's markers are named here, because a records reader finds a prompt's question and its context's end
by them.
"""

import argparse
import datetime
import os
import time
from collections.abc import Callable
from typing import Any

from lsa import assets, io
from lsa.data import manifests
from lsa.data.tokenizer import load_tokenizer, tokenizer_checkpoint, tokenizer_repo
from lsa.evaluation.accuracy.data import default_data_root, longbench_card, longbench_path

# What the preparing job asks for - one tokenizer over 503 documents, no GPU
PARAMETERS: dict[str, Any] = {"gpus_per_node": 0, "cpus_per_task": 8, "mem_gb": 64, "timeout_min": 4 * 60}

# Where the document ends, where the question starts and where it ends
CONTEXT_END = "</text>"
QUESTION = "What is the correct answer to this question:"
CHOICES = "\nChoices:"

# THUDM/LongBench prompts/0shot.txt, verbatim
TEMPLATE = (
    f"Please read the following text and answer the question below.\n\n<text>\n$DOC$\n{CONTEXT_END}\n\n"
    f"{QUESTION} $Q${CHOICES}\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def tokenizer(name: str | None = None) -> Any:
    """Load the tokenizer a prompt's length is counted in, the card's own when none is named."""
    return load_tokenizer(name or tokenizer_checkpoint(longbench_card()), trust_remote_code=True)


def truncate_middle(context: str, tokenizer: Any, max_tokens: int) -> tuple[str, int, bool]:
    """Keep the first and the last half of the tokens - pred.py: ids[:max//2] + ids[-max//2:].

    Returns:
        The context as kept.
        How many tokens it had.
        Whether it was cut.
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return context, len(ids), False
    kept = ids[: max_tokens // 2] + ids[-(max_tokens // 2) :]
    return tokenizer.decode(kept, skip_special_tokens=True), len(ids), True


def prompt_for(doc: dict[str, Any], context: str) -> str:
    """Fill the official template with one row's question, choices and context."""
    return (
        TEMPLATE.replace("$DOC$", context.strip())
        .replace("$Q$", doc["question"].strip())
        .replace("$C_A$", doc["choice_A"].strip())
        .replace("$C_B$", doc["choice_B"].strip())
        .replace("$C_C$", doc["choice_C"].strip())
        .replace("$C_D$", doc["choice_D"].strip())
    )


def prepare(doc: dict[str, Any], tokenizer: Any, max_context_tokens: int) -> dict[str, Any]:
    """Turn one dataset row into the row the file stores - prompt, answer, and what is known of the context."""
    template_tokens = len(tokenizer.encode(prompt_for(doc, ""), add_special_tokens=False))
    budget = max_context_tokens - template_tokens
    context, context_tokens, truncated = truncate_middle(doc["context"], tokenizer, budget)
    return {
        "id": doc["_id"],
        "domain": doc["domain"],
        "sub_domain": doc["sub_domain"],
        "difficulty": doc["difficulty"],
        "length": doc["length"],
        "answer": doc["answer"],
        "context_tokens": context_tokens,
        "truncated": truncated,
        "prompt": prompt_for(doc, context),
    }


def prepare_longbench(
    max_context: int, root: str | None = None, tok: Any = None, tokenizer_name: str | None = None
) -> None:
    """Prepare every LongBench v2 row at one context limit and write its file and manifest entry.

    Args:
        max_context: The limit a prompt is truncated to, in tokens.
        root: The data root, or None for the default data root.
        tok: A loaded tokenizer, or None to load the one named.
        tokenizer_name: The tokenizer to count in, or None for the one the card names.
    """
    import datasets

    root = root or default_data_root()
    card = longbench_card()
    tok = tok or tokenizer(tokenizer_name)
    tokenizer_name = tokenizer_name or tokenizer_repo(card)
    start = time.perf_counter()
    rows = [
        prepare(doc, tok, max_context)
        for doc in datasets.load_dataset(card.dataset, split=card.split, revision=card.revision)
    ]
    path = longbench_path(max_context, root)
    io.write_jsonl(path, rows)
    entry = {
        "rows": len(rows),
        "sha256": io.file_sha256(path),
        "tokenizer": tokenizer_name,
        "max_context": max_context,
        "truncated": sum(row["truncated"] for row in rows),
        "date": datetime.date.today().isoformat(),
        "seconds": round(time.perf_counter() - start, 1),
    }
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(card)})
    manifests.write_benchmark_file(root, os.path.relpath(path, root), record)
    print(f"{path}: {len(rows)} rows, {entry['truncated']} truncated, {entry['seconds']} s", flush=True)


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add what the LongBench v2 file is prepared with to `lsa data prepare`'s parser.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("--root", default=default_data_root())
    parser.add_argument(
        "--tokenizer", default=None, help="the tokenizer to count in - GLM-5.3-Flash's when unset"
    )
    parser.add_argument("--max-context", type=int, default=longbench_card().context)


def job(card: assets.DatasetCard, args: argparse.Namespace) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Return what the job runs to prepare the file the command line names, and what it is called with.

    Args:
        card: The LongBench v2 card, which the preparer reads for itself.
        args: The parsed command line.

    Returns:
        `prepare_longbench`, and the context limit, the absolute root, no loaded tokenizer and its name.
    """
    return prepare_longbench, (args.max_context, os.path.abspath(args.root), None, args.tokenizer)
