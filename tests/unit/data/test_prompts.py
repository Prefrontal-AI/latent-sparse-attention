"""The kernels prompt shipped in data/prompts is the exact file its pin names."""

import hashlib
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[3] / "data" / "prompts"
PROMPT = "gutenberg_300k.txt"


def test_prompt_matches_its_pin():
    digest, name = (PROMPTS / f"{PROMPT}.sha256").read_text().split()
    assert name == PROMPT
    assert hashlib.sha256((PROMPTS / PROMPT).read_bytes()).hexdigest() == digest
