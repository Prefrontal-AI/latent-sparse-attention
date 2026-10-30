"""Score maps that are already trained, at a read they were not trained for, one slurm job per run.

    lsa distill rescore configs/training/distillation/glm53_flash/long_16k_1m_heavy_x.yaml model.space.init=maps.pt

The loss never sees a bucket, so `top_k` changes what a report measures without changing what a map is.
Sweeping it is a re-report rather than a retrain, and this is the entry point that does it: one checkpoint,
one grid of reads, no gradient and no training document loaded.

`model.space.init` is what makes a run a rescore, and it is required - there is nothing to score without it.
The cache's visibility rule is the one thing that is not free, because `window_size` and `sink_size` define
the target the teacher was written under; the loader rejects a run that disagrees with the cache, which is
what keeps a swept grid comparable.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lsa import slurm
from lsa.io import write_json
from lsa.training import distilled_maps, require_extra
from lsa.training.distillation.checkpoint import load_maps, meta, provenance, resolve_layers
from lsa.training.distillation.constants import Results
from lsa.training.distillation.data.native import KeyAndQueryReader
from lsa.training.distillation.export import KEY, QUERY, is_projection_file
from lsa.training.distillation.model import IndexMap
from lsa.training.distillation.reporting import load_scoring, report
from lsa.training.distillation.run import PARAMETERS, submit
from lsa.training.distillation.trainer import input_reader

if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig


def rescore_maps(experiment: "DistillConfig", force: bool = False, device: str = "cuda:0") -> Path:
    """Score one run's checkpoint on its evaluation sets and write the report.

    The signature matches `train` so a submitter maps over either without a wrapper.

    Args:
        experiment: The experiment, whose `model.space.init` names the checkpoint to score.
        force: Rescore a run whose report is already written.
        device: Torch device the cache and the maps land on.

    Returns:
        The directory the run wrote into.

    Raises:
        ValueError: The run names no checkpoint to score.
    """
    require_extra("omegaconf")
    if not experiment.model.space.init:
        raise ValueError(f"{experiment.run.name}: rescoring needs model.space.init to name the maps to score")

    out = experiment.out
    if (out / Results.RESCORE).exists() and not force:
        print(f"{experiment.run.name}: complete - nothing to run", flush=True)
        return out

    recorded = provenance(experiment)
    layers = resolve_layers(experiment)
    scoring = load_scoring(experiment, layers)
    print(
        f"{experiment.run.name}: scored on {', '.join(scoring)}, layers {layers}, "
        f"top_k {experiment.model.indexer.top_k}",
        flush=True,
    )

    reader = input_reader(experiment, layers, device)
    maps = load_checkpoint(experiment, experiment.model.space.init, layers, device, reader)
    write_json(out / Results.META, meta(experiment, device, maps, recorded))
    write_json(out / Results.RESCORE, report(experiment, maps, scoring, Results.NO_STEP, reader))
    print(f"{experiment.run.name}: wrote {out}", flush=True)
    return out


def load_checkpoint(
    experiment: "DistillConfig",
    init: str,
    layers: list[int],
    device: str,
    reader: KeyAndQueryReader | None = None,
) -> dict[int, IndexMap]:
    """Build one map per layer straight from the checkpoint.

    A trained map replaces whatever it starts as, so nothing is initialised from `x` here and no document is
    read - the checkpoint's own projections say the dims the maps have to have.

    Args:
        experiment: The experiment, for the model the checkpoint has to belong to.
        init: The checkpoint to score, or a projection file - the form `lsa distill export` writes.
        layers: The layers to score.
        device: Where the maps land.
        reader: Reads the key and query, whose bases a map on them acts in, or None for `x`.

    Returns:
        One `IndexMap` per layer.
    """
    path = Path(init)
    blob = torch.load(path, map_location=device, weights_only=False)
    if is_projection_file(blob):
        return projection_maps(experiment, path, blob, layers, device)
    artifact = distilled_maps.decode(blob, path)
    held = artifact.checked_for(experiment.model.name, layers)
    maps = {
        layer: IndexMap(
            projection.model_dim,
            projection.index_dim,
            bases=reader.bases(layer) if reader is not None else None,
        ).to(device)
        for layer, projection in held.items()
    }
    load_maps(blob, maps)
    print(f"{experiment.run.name}: scoring {init}", flush=True)
    return maps


def projection_maps(
    experiment: "DistillConfig", path: Path, blob: dict[str, torch.Tensor], layers: list[int], device: str
) -> dict[int, IndexMap]:
    """Build one map per layer from a projection file, the form a server reads.

    The bases a key and query map was trained in are already folded into a projection file, so each map acts on
    what it reads as it is. A file carries no model, so the pairing with the cache is the caller's.

    Args:
        experiment: The experiment, for the line that names what is scored.
        path: The projection file.
        blob: Its tensors, `<layer>.W_Q_idx` and `<layer>.W_K_idx`.
        layers: The layers to score.
        device: Where the maps land.

    Returns:
        One `IndexMap` per layer.

    Raises:
        ValueError: The file holds no projection for a layer that was asked for.
    """
    missing = [
        layer
        for layer in layers
        if QUERY.format(layer=layer) not in blob or KEY.format(layer=layer) not in blob
    ]
    if missing:
        raise ValueError(f"{path} holds no projection for layers {missing}")
    maps = {}
    for layer in layers:
        query, key = blob[QUERY.format(layer=layer)], blob[KEY.format(layer=layer)]
        index_map = IndexMap(int(key.shape[0]), int(key.shape[1])).to(device)
        with torch.no_grad():
            index_map.query_projection.copy_(query.float())
            index_map.key_projection.copy_(key.float())
        maps[layer] = index_map
    print(f"{experiment.run.name}: scoring projection file {path}", flush=True)
    return maps


def main() -> None:
    """Parse the command line and submit a rescore of every run the file describes."""
    parser = argparse.ArgumentParser(
        description="Score trained maps at a different read, one slurm job per run"
    )
    parser.add_argument("config", help="path to the training yaml")
    slurm.add_flags(parser)
    parser.add_argument("--force", action="store_true", help="rescore a run whose report is already written")
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")

    from lsa.training.distillation.config import load

    experiments = load(args.config, args.overrides)
    for experiment in experiments:
        print(
            f"{experiment.run.name}: top_k {experiment.model.indexer.top_k}, maps {experiment.model.space.init}",
            flush=True,
        )
    if args.dry_run:
        return
    submit(experiments, {**PARAMETERS, **slurm.overrides(args)}, args.folder, args.force, stage=rescore_maps)
