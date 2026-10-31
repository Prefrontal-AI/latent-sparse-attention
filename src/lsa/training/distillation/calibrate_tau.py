"""Record the loss temperature of an exported map that has none, one job on a GPU.

    lsa distill calibrate-tau configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml \
        ${LSA_ROOT}/models/maps/glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt --partition main

A map trained before its temperature was exported cannot say what it was trained at, and a read that ranks by a
LogSumExp needs it. This runs the search a run calibrates its untrained map with, `losses.calibrate`'s - the grid value of
least cross-entropy to the teacher - on the trained map, a layer at a time, over the held-out documents of the
cache the config reads, and writes the result into the map's sidecar as `"tau": {"<layer>": <tau>}`.

The config is a training run's: its `data.evaluate` sets are the documents, `model.indexer` the visibility rule and input the
cache was written under, and `run.objective` the target the temperature is chosen for. The map is a projection file, so
it is read as a server reads it, and an `IndexMap` holds its projections with no basis.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa import env, io, slurm
from lsa.sglang.index import weights
from lsa.training import require_extra
from lsa.training.distillation.checkpoint import resolve_layers
from lsa.training.distillation.constants import Calibration
from lsa.training.distillation.data.pool import stream
from lsa.training.distillation.losses import calibration_grid, calibration_losses
from lsa.training.distillation.model import IndexMap
from lsa.training.distillation.served_read import PARAMETERS, document_path, held_out

if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig

# Documents of each evaluation set the search reads, so a set of 1M documents is not read whole
DOCUMENTS_PER_SET: int = 4


def map_of(path: Path, layer: int, device: str) -> IndexMap:
    """Hold one layer of an exported map as an `IndexMap`, as the server reads it.

    Args:
        path: The projection file.
        layer: The layer.
        device: Where the map lands.

    Returns:
        The map, whose temperature is the default until `set_tau`.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    input_dim, index_dim = next(iter(blob.values())).shape
    query, key = weights.read(path, input_dim, index_dim, [layer])[layer]
    index_map = IndexMap(input_dim, index_dim).to(device)
    with torch.no_grad():
        index_map.query_projection.copy_(query)
        index_map.key_projection.copy_(key)
    return index_map


def calibrate_tau(
    experiment: "DistillConfig", maps: Path, per_set: int = DOCUMENTS_PER_SET, device: str = "cuda:0"
) -> dict[str, float]:
    """Calibrate every layer of an exported map on the held-out documents of a cache, and write the sidecar.

    Args:
        experiment: The experiment, whose evaluation sets are the documents.
        maps: The projection file, whose `.json` sidecar is written.
        per_set: Documents of each evaluation set to read at most.
        device: Where the documents and the map land.

    Returns:
        The temperature by layer, as the sidecar records them.
    """
    require_extra("omegaconf")
    layers = resolve_layers(experiment)
    for document_set in experiment.data.evaluate:
        document_set.limit = min(document_set.limit or per_set, per_set)
    documents = held_out(experiment)
    objective = experiment.run.objective
    tau: dict[str, float] = {}
    with torch.inference_mode():
        for layer in layers:
            index_map = map_of(maps, layer, device)
            grid = calibration_grid(index_map)
            cross_entropy = torch.zeros_like(grid)
            paths = [document_path(entry, layer) for entry in documents]
            for waiting in stream(paths, experiment.model.indexer):
                cross_entropy += calibration_losses(
                    index_map,
                    waiting,
                    grid,
                    Calibration.ROWS,
                    objective.target_depth,
                    objective.target,
                    objective.tail,
                )
            tau[str(layer)] = float(grid[int(cross_entropy.argmin())])
            print(f"layer {layer}: tau {tau[str(layer)]:.4f} over {len(documents)} documents", flush=True)
    write_tau(maps, tau, experiment, len(documents))
    return tau


def write_tau(maps: Path, tau: dict[str, float], experiment: "DistillConfig", documents: int) -> None:
    """Record the temperatures in a map's sidecar, keeping what else it says.

    Args:
        maps: The projection file.
        tau: The temperature by layer.
        experiment: The experiment that calibrated them, which the sidecar names.
        documents: Documents read.
    """
    path = maps.with_suffix(".json")
    sidecar: dict[str, Any] = io.read_json(path, {}) if path.is_file() else {}
    sidecar["tau"] = tau
    sidecar["tau_calibrated"] = {
        "run": experiment.run.name,
        "target": experiment.run.objective.target,
        "target_depth": experiment.run.objective.target_depth,
        "tail": experiment.run.objective.tail,
        "documents": documents,
    }
    io.write_json(path, sidecar)


def main() -> None:
    """Submit the calibration of one exported map's temperature."""
    parser = argparse.ArgumentParser(
        description="Calibrate and record the loss temperature of an exported map"
    )
    parser.add_argument("config", help="path to the training yaml whose evaluation documents calibrate it")
    parser.add_argument("maps", type=Path, help="the projection file whose sidecar records the temperature")
    parser.add_argument(
        "--documents-per-set",
        type=int,
        default=DOCUMENTS_PER_SET,
        help="documents of each evaluation set to read",
    )
    slurm.add_flags(parser)
    parser.add_argument("overrides", nargs="*", help="dotted key=value assignments, applied last")
    args = parser.parse_args()

    require_extra("omegaconf", "hydra")

    from lsa.training.distillation.config import load

    (experiment,) = load(args.config, args.overrides)
    experiment.data.root = str(Path(experiment.data.root).resolve())
    for document_set in [*experiment.data.train, *experiment.data.evaluate]:
        if document_set.root:
            document_set.root = str(Path(document_set.root).resolve())
    slurm.submit(
        args,
        "lsa-calibrate-tau",
        env.DISTILL,
        PARAMETERS,
        calibrate_tau,
        experiment,
        args.maps.resolve(),
        args.documents_per_set,
    )
