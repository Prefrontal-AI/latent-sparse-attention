"""The records a benchmark leaves - one directory per checkpoint and preset, with every sample scored.

    <records>/<checkpoint label>/<preset>/
        config.json          the checkpoint, its step, the attention, the options and what produced the numbers
        results.json         lm-eval's aggregate per task
        samples/<task>.jsonl one row per example, keyed by its id so two arms pair by example

A sample row keeps what a comparison reads - the metrics, the target and the reply - and leaves out the prompt, which
for a RULER cell is 16,000 tokens and is rebuilt from the same fixed file. The doc and prompt hashes are kept, so a
pair of rows that disagree on them is not a pair.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lsa import io

CONFIG = "config.json"
RESULTS = "results.json"
SAMPLES = "samples"
SCHEMA = "native-benchmark/1"


def run_folder(records: str | Path, label: str, preset: str) -> Path:
    """Return the directory one checkpoint's run of one preset writes into.

    Args:
        records: The records root.
        label: What names the checkpoint among its siblings.
        preset: The preset or task list's name.

    Returns:
        The directory.
    """
    return Path(records) / label / preset


def slim_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Reduce one lm-eval logged sample to what a comparison reads.

    Args:
        sample: lm-eval's logged sample, with its doc and prompt.

    Returns:
        The sample's id, hashes, target, replies and metric values.
    """
    return {
        "doc_id": sample["doc_id"],
        "doc_hash": sample["doc_hash"],
        "prompt_hash": sample["prompt_hash"],
        "target": sample["target"],
        "resps": sample["filtered_resps"],
        "metrics": {name: sample[name] for name in sample["metrics"]},
    }


def sample_rows(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Slim a task's samples and put them in id order.

    Args:
        samples: The task's logged samples, which several ranks gathered in rank order.

    Returns:
        One row per example, by id.
    """
    return sorted((slim_sample(sample) for sample in samples), key=lambda row: row["doc_id"])


def write_records(folder: str | Path, results: dict[str, Any], config: dict[str, Any]) -> None:
    """Write one run's config, aggregate results and every task's samples.

    Args:
        folder: The run's directory.
        results: What lm-eval returned, logged samples included.
        config: What produced the numbers.
    """
    folder = Path(folder)
    io.write_json(folder / CONFIG, {"schema": SCHEMA, **config})
    io.write_json(
        folder / RESULTS,
        {
            "results": results["results"],
            "versions": results.get("versions"),
            "n-samples": results.get("n-samples"),
        },
    )
    for task, samples in results["samples"].items():
        io.write_jsonl(folder / SAMPLES / f"{task}.jsonl", sample_rows(samples))


def read_samples(folder: str | Path, task: str) -> dict[int, dict[str, Any]]:
    """Read one task's sample rows back, by example id.

    Args:
        folder: The run's directory.
        task: The task.

    Returns:
        The rows, keyed by id.
    """
    return {row["doc_id"]: row for row in io.read_jsonl(Path(folder) / SAMPLES / f"{task}.jsonl")}


def read_config(folder: str | Path) -> dict[str, Any]:
    """Read the config a run recorded.

    Args:
        folder: The run's directory.

    Returns:
        The config.
    """
    config: dict[str, Any] = io.read_json(Path(folder) / CONFIG)
    return config
