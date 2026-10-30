"""The manifest generation writes beside the prompt files, one entry per file."""

import json
import os
import tempfile

from lsa.data import manifests
from lsa.evaluation.accuracy import data


def entry(rows):
    """What generation records about a file of that many rows"""
    return {"rows": rows, "sha256": "f", "tokenizer": "t", "date": "2026-09-19"}


def test_an_entry_file_wins_over_the_single_manifest_and_every_entry_is_read():
    root = tempfile.mkdtemp()
    with open(os.path.join(root, "manifest.json"), "w") as target:
        json.dump(
            {"ruler/vt_4096.jsonl": entry(100), "longbench2/longbench2_1000000.jsonl": entry(503)},
            target,
        )
    declared = manifests.declared(data.ruler_card())
    for key in ("ruler/vt_4096.jsonl", "ruler/cwe_4096.jsonl"):
        manifests.write_benchmark_file(
            root, key, manifests.BenchmarkFile.from_entry({**entry(20), "declared": declared})
        )
    manifest = data.load_manifest(root)
    assert manifest["ruler/vt_4096.jsonl"] == entry(20)
    assert set(manifest) == {
        "ruler/vt_4096.jsonl",
        "ruler/cwe_4096.jsonl",
        "longbench2/longbench2_1000000.jsonl",
    }
    assert os.path.exists(os.path.join(root, "manifest", "ruler_cwe_4096.jsonl.json"))


def test_an_empty_root_has_an_empty_manifest():
    assert data.load_manifest(tempfile.mkdtemp()) == {}
