"""A prepared RULER cell at test size, and a tokenizer that reports where each of its tokens sits in the text."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from lsa import assets
from lsa.data import manifests

LENGTH = 4096


def prepared_card(root, sources):
    return assets.BenchmarkCard(
        name="ruler_test",
        licence="Apache-2.0",
        generator="NVIDIA/RULER",
        revision="abc",
        path=str(root),
        file="{task}_{length}.jsonl",
        tokenizer="glm53_flash",
        tasks=list(sources),
        prompt=["input", "gen_prefix"],
        join="\n",
    )


def write_rows(card, source, rows, length=LENGTH):
    """Write a prepared file of (haystack, question, outputs) rows and the manifest entry that vouches for it"""
    path = Path(card.file_path(task=source, length=length))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"index": i, "input": haystack, "gen_prefix": question, "outputs": outputs, "length": length}
        )
        for i, (haystack, question, outputs) in enumerate(rows)
    ]
    path.write_text("\n".join(lines) + "\n")
    entry = {"rows": len(rows), "sha256": "", "tokenizer": "t", "date": "2026-10-01"}
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(card)})
    manifests.write_benchmark_file(card.data_root(), f"{path.parent.name}/{path.name}", record)


class SpanTokenizer:
    """One token a word, led by the space before it, with the character span the real tokenizers report"""

    bos_token_id = 99

    def __call__(self, text, return_tensors=None, add_special_tokens=True, return_offsets_mapping=False):
        found = list(re.finditer(r"\s?\S+", text))
        encoded = SimpleNamespace(input_ids=torch.arange(1, len(found) + 1)[None])
        if return_offsets_mapping:
            encoded.offset_mapping = torch.tensor([[[m.start(), m.end()] for m in found]])
        return encoded

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    ):
        (turn,) = messages
        closed = "" if enable_thinking else " think-closed"
        return f"user {turn['content']} assistant{closed}"

    def encode(self, text):
        return [99, 7, 8]
