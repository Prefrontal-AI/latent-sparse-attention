"""One run's layers trained in several processes, and the shards joined back into the run.

    lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml --shards 11

`lsa distill train --shards` submits the shards as one array and `join` as a job that waits on all of them.

The layers share nothing - each has its own map, its own documents and its own optimiser state, and a step
only ever touches one of them. So a run's layers can train apart, each process holding only its own share of
the cache, which is what lets a run fit a card whole rather than stream most of it from host memory, and
what lets a cache larger than one worker's host memory train at all.

A shard trains on the run's own step numbering and draws every step, its own or not, off the run's one host
generator. So it learns for its layers exactly the maps the whole run does, and reports at the same steps.
Joining the shards is then bookkeeping: the maps are merged, and a report is the mean over every scored
(layer, document) pair, which is each shard's mean weighed by its layer count, because every layer scores
the same documents.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.io import read_json, write_json
from lsa.training.distillation.checkpoint import run_directory
from lsa.training.distillation.constants import Results
from lsa.training.distillation.metrics import mean_reports
from lsa.training.distillation.tracking import Tracker

if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig


def split(layers: list[int], count: int) -> list[list[int]]:
    """Deal a run's layers into shards, round robin, so each shard spans the depth.

    Args:
        layers: The run's layers, in order.
        count: Shards wanted, more than there are layers giving one layer a shard.

    Returns:
        The shards, each a list of layers in order.

    Raises:
        ValueError: `count` is less than one.
    """
    if count < 1:
        raise ValueError(f"a run splits into at least one shard - got {count}")
    return [layers[start::count] for start in range(min(count, len(layers)))]


def join(experiment: "DistillConfig", layers: list[int], force: bool = False) -> Path:
    """Merge a run's finished shards into the run's own maps, snapshots, reports and meta.

    Args:
        experiment: The experiment.
        layers: Every layer the run trains, which the shards must cover exactly once between them.
        force: Join a run whose maps are already written.

    Returns:
        The run's directory.

    Raises:
        ValueError: A shard is unfinished, or the shards do not cover the run's layers exactly once.
    """
    out = run_directory(experiment)
    if (out / Results.MAPS).exists() and not force:
        print(f"{experiment.run.name}: joined already - nothing to do", flush=True)
        return out

    shards = sorted(path for path in (out / Results.SHARDS).iterdir() if path.is_dir())
    unfinished = [path.name for path in shards if not (path / Results.MAPS).exists()]
    if unfinished:
        raise ValueError(f"{experiment.run.name}: shards {unfinished} have not finished")
    metas = [read_json(path / Results.META) for path in shards]
    covered = sorted(layer for meta in metas for layer in meta["layers"])
    if covered != sorted(layers):
        raise ValueError(
            f"{experiment.run.name}: the shards train layers {covered} and the run trains {sorted(layers)}"
        )

    weights = [len(meta["layers"]) for meta in metas]
    tracker = Tracker(experiment, out, label=Results.JOINED)
    for report_path in sorted(shards[0].glob(Results.REPORT.format(step="*")), key=_step_of):
        joined = joined_report([read_json(path / report_path.name) for path in shards], weights)
        write_json(out / report_path.name, joined)
        tracker.report(joined["step"], joined)
    tracker.close()

    for maps_path in shards[0].glob(Results.SNAPSHOT.format(step="*")):
        torch.save(joined_maps([path / maps_path.name for path in shards]), out / maps_path.name)
    write_json(
        out / Results.META, {**metas[0], "layers": sorted(layers), "shards": [path.name for path in shards]}
    )
    torch.save(joined_maps([path / Results.MAPS for path in shards]), out / Results.MAPS)
    print(f"{experiment.run.name}: joined {len(shards)} shards into {out}", flush=True)
    return out


def joined_report(reports: list[dict[str, Any]], weights: list[int]) -> dict[str, Any]:
    """Merge one step's shard reports into the run's.

    Args:
        reports: One report per shard, all at the same step.
        weights: Each shard's layer count.

    Returns:
        The report the whole run would have written at that step.
    """
    tau = {layer: value for report in reports for layer, value in report["tau"].items()}
    measured = [
        {key: value for key, value in report.items() if key not in ("step", "tau")} for report in reports
    ]
    return {"step": reports[0]["step"], "tau": dict(sorted(tau.items(), key=lambda item: int(item[0])))} | (
        mean_reports(measured, weights)
    )


def joined_maps(paths: list[Path]) -> dict[str, Any]:
    """Merge the shards' checkpoints of one step into the run's.

    Args:
        paths: One checkpoint per shard, all of the same step.

    Returns:
        The checkpoint, with every shard's maps and the first shard's record of what produced them.
    """
    blobs = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    maps = {layer: state for blob in blobs for layer, state in blob["maps"].items()}
    return {**blobs[0], "maps": dict(sorted(maps.items()))}


def _step_of(path: Path) -> int:
    """Return the step a report file names."""
    return int(path.stem.removeprefix(Results.REPORT.split("{")[0]))
