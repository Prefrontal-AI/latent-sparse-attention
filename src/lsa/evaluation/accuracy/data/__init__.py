"""What the accuracy suite reads - the fixed prompt files, where they live, what generation recorded, and what a suite asks for.

RULER comes from NVIDIA's own generators ([ruler](ruler.py)), LongBench v2 is prepared once per context
limit ([longbench](longbench.py)) and PaperTrail is copied from its repository ([papertrail](papertrail.py)). All land under one root, 12G of files too large for git, so it is a cache
root like HF_HOME that generation fills:

    <root>/ruler/<task>_<length>.jsonl
    <root>/longbench2/longbench2_<context>.jsonl
    <root>/papertrail/<sample>_<condition>.jsonl and .context.txt
    <root>/manifest/<file with / as _>.json    one per file - sha256, rows, and the card it was prepared under

What each benchmark is - the hub dataset and its revision, or the generator and its grid of tasks, lengths and
rows - is its [card](../../../assets/datasets/benchmarks), and what generating a file measured is its manifest
entry. This file is the paths and the manifest on the data root. Nothing here imports lm-eval, so the files
can be generated and checked without it. A prompt's length is counted in the tokenizer its card names, read
through [lsa.data.tokenizer](../../../data/tokenizer.py) as the production prompt set's is.

| module | what it does |
|---|---|
| [ruler](ruler.py) | NVIDIA's RULER generators, driven over the card's grid as a CPU job by `lsa data prepare ruler` |
| [longbench](longbench.py) | LongBench v2 under the official template, truncated in the middle to a limit - `lsa data prepare longbench2` |
| [papertrail](papertrail.py) | PaperTrail's samples at every length and control, one context each - `lsa data prepare papertrail` |
| [suites](suites.py) | `full`, `ruler` and `smoke` - which tasks a run asks for and how many prompts of each |
| [hashes](hashes.py) | the prompt-set hash of every plan, pinned in `hashes.json` |
"""

import os
from functools import cache
from typing import Any

from lsa import assets, paths
from lsa.data import manifests
from lsa.data.benchmarks import BenchmarkFiles
from lsa.evaluation.constants import Suite

# The shared root rather than a per-user path, since a directory another user writes to was rewritten
# under a run once already
DATA_ROOT = os.environ.get("LSA_EVAL_DATA") or str(paths.benchmarks())

# Each GraphWalks task and the hub file it reads - prompts to 128k tokens, and the long file's 256k and 1M
# tiers, a task each because a 1M server holds the first and not the second. Read from the hub as it is, so none
# lives under the data root
GRAPHWALKS_FILES = {
    "graphwalks_128k": "graphwalks_128k_and_shorter.parquet",
    "graphwalks_256k": "graphwalks_256k_to_1mil.parquet",
    "graphwalks_1m": "graphwalks_256k_to_1mil.parquet",
}

# The longest prompt of each GraphWalks task's fixed subset in GLM-5.3-Flash's tokenizer, chat template included,
# counted on a CPU job on 2026-10-03. The 1M tier is 1.14M tokens, past a 1,048,576 server, so a run skips it
# there the way it skips a RULER length, and serves it where the context holds it
GRAPHWALKS_PROMPT_MAX = {"graphwalks_128k": 142_921, "graphwalks_256k": 285_775, "graphwalks_1m": 1_142_712}

# The longest length one node serves - past it one prompt is 20 minutes of prefill and fills the batch alone
ONE_NODE_LENGTH = 1_000_000


@cache
def ruler_card() -> assets.BenchmarkCard:
    """Return RULER's card - its tasks, its lengths and the rows a file of each holds."""
    return assets.benchmark(Suite.RULER)


@cache
def longbench_card() -> assets.BenchmarkCard:
    """Return LongBench v2's card - the hub dataset and the one context it is prepared at."""
    return assets.benchmark(Suite.LONGBENCH)


@cache
def papertrail_card() -> assets.BenchmarkCard:
    """Return PaperTrail's card - its samples, its lengths and the repository they are copied from."""
    return assets.benchmark(Suite.PAPERTRAIL)


def graphwalks_rows(task: str) -> Any:
    """Read one GraphWalks task's file at the revision its card pins, from the hub cache.

    The parquet is fetched by name and read as a local file. Through `load_dataset` the cache of a hub dataset
    of files rather than configs is keyed by data files it can only resolve online, so no compute node finds it.

    Args:
        task: `graphwalks_128k`, `graphwalks_256k` or `graphwalks_1m`.

    Returns:
        The file's rows, as a `datasets.Dataset`.
    """
    import datasets
    from huggingface_hub import hf_hub_download

    card = assets.benchmark(Suite.GRAPHWALKS)
    path = hf_hub_download(card.dataset, GRAPHWALKS_FILES[task], repo_type="dataset", revision=card.revision)
    return datasets.load_dataset("parquet", data_files=path, split="train")


def ruler_path(ruler_task: str, length: int, data_root: str = DATA_ROOT) -> str:
    """Return where one RULER task's fixed samples at one length live."""
    return str(BenchmarkFiles(ruler_card(), data_root).path(task=ruler_task, length=length))


def longbench_path(max_context: int, data_root: str = DATA_ROOT) -> str:
    """Return where the LongBench v2 prompts prepared for one context limit live."""
    return str(BenchmarkFiles(longbench_card(), data_root).path(context=max_context))


def papertrail_path(sample: str, condition: str, data_root: str = DATA_ROOT) -> str:
    """Return where one PaperTrail sample's questions under one condition live - a length or a control."""
    return str(BenchmarkFiles(papertrail_card(), data_root).path(task=sample, condition=condition))


def load_manifest(root: str) -> dict[str, dict]:
    """Return what generation recorded about every fixed file under a root, as a record carries it.

    Args:
        root: The data root.

    Returns:
        Sample file, relative to the root, to its entry without the card it was prepared under.
    """
    return {key: record.recorded() for key, record in manifests.read_benchmark_files(root).items()}
