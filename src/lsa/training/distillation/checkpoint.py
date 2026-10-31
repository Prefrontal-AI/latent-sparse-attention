"""Read, write and describe distillation checkpoints.

A run's maps are snapshotted at every report, and beside them the one resume state everything else needed to go on
from that step: the optimiser, the RNG, each pool's plan and how far its steps are, and the wandb run. A job told it is
about to be stopped - slurm's SIGUSR2, 90 s ahead of a preemption or the time limit - writes the same state at the step
it has reached and is requeued. A run started again into a directory that holds one continues from it rather than
from step 0, and `--force` starts it over.
"""

import os
import signal
import threading
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

import torch

from lsa.config_loader import run_provenance
from lsa.core import RotaryLayout
from lsa.sglang.settings.design import QSAConfig
from lsa.training.data.targets import read_manifest
from lsa.training.distillation.constants import Results
from lsa.training.distillation.data.cache import (
    cached_layers,
    recorded_rotary,
    set_root,
)
from lsa.training.distillation.model import Indexer, IndexMap, QSAIndexer

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig
    from lsa.training.distillation.data.online import OnlineTeacher


def save_maps(path: Path, maps: Mapping[int, Indexer], recorded: dict[str, Any], step: int) -> None:
    """Save the maps as they stand, with the step that produced them.

    A snapshot is written at the report boundary rather than on a cadence of its own, so that the map a
    later stage loads always has the routing report that says how it selects sitting beside it.

    Args:
        path: File to write.
        maps: One map or QSA indexer per layer.
        recorded: What produced the maps, as `provenance` built it.
        step: Steps trained when this was taken.
    """
    torch.save(checkpoint(maps, {**recorded, "step": step}), path)


def provenance(experiment: "DistillConfig", teacher: "OnlineTeacher | None" = None) -> dict[str, Any]:
    """Return what produced a run's maps, for the stage that loads them.

    The maps carry no trace of the model whose hidden states trained them, and their shapes will not say:
    two models of one dim share the shape of every projection. The extraction manifest of the collection
    trained on is where the model is named, and it has to be the model the run says it trains - or, for a run
    taught online, the extraction it reads, which the teacher has already held to the run's model.

    Args:
        experiment: The experiment.
        teacher: The online teacher, or None for a run that reads a cache.

    Returns:
        The run itself, and what its targets say they were extracted from.

    Raises:
        ValueError: The targets were extracted from another model than `model.name`.
    """
    if teacher is not None:
        return {"run": asdict(experiment), "teacher": teacher.provenance()}
    collection = experiment.data.train[0].source
    manifest = read_manifest(set_root(experiment.data.root, experiment.data.train[0]), collection)
    extracted_from = manifest["run"]["model"]["name"]
    for document_set in [*experiment.data.train, *experiment.data.evaluate]:
        current = read_manifest(set_root(experiment.data.root, document_set), document_set.source)
        model = current["run"]["model"]["name"]
        if model != experiment.model.name:
            raise ValueError(
                f"{experiment.run.name}: the targets under {document_set.source} were extracted from {model!r} and "
                f"model.name is {experiment.model.name!r} - a map is only valid on the weights whose states trained it"
            )
    return {
        "run": asdict(experiment),
        "teacher": {"collection": collection, "model": extracted_from, "layers": manifest["layers"]},
    }


def run_directory(experiment: "DistillConfig", shard: list[int] | None = None) -> Path:
    """Return where a run writes, or where one of its shards does.

    Args:
        experiment: The experiment.
        shard: The layers of one shard, or None for the run itself.

    Returns:
        The directory.
    """
    out = experiment.out
    return out if shard is None else out / Results.SHARDS / shard_label(shard)


def shard_label(shard: list[int]) -> str:
    """Return what one shard is called, which is the directory it writes into under its run's."""
    return Results.SHARD.format(layers="-".join(str(layer) for layer in shard))


def resolve_layers(experiment: "DistillConfig") -> list[int]:
    """Return the layers a run trains, from the config or from what the targets hold.

    Args:
        experiment: The experiment.

    Returns:
        Layer indices, in order.

    Raises:
        ValueError: The targets hold no layer, or not one the config names.
    """
    if experiment.data.teacher is not None:
        from lsa.training.distillation.data.online import extraction_of, teacher_layers

        held = teacher_layers(extraction_of(experiment.data.teacher))
    else:
        held = cached_layers(experiment.data.root, experiment.data.train[0])
        if not held:
            raise ValueError(
                f"{experiment.data.root} holds no layer of {experiment.data.train[0]} - name a train set the cache holds"
            )
    if experiment.data.layers is None:
        return held
    missing = sorted(set(experiment.data.layers) - set(held))
    if missing:
        raise ValueError(f"the targets hold no layers {missing} - they hold {held}")
    return list(experiment.data.layers)


def cached_rotary(experiment: "DistillConfig") -> RotaryLayout:
    """Return where the rotation sits in the heads of the model a run's cache was extracted from.

    Args:
        experiment: The experiment.

    Returns:
        The layout the extraction measured and recorded in its manifest - none for a cache that predates the record.
    """
    manifest = read_manifest(
        set_root(experiment.data.root, experiment.data.train[0]), experiment.data.train[0].source
    )
    return recorded_rotary(manifest)


def meta(
    experiment: "DistillConfig", device: str, maps: Mapping[int, Indexer], recorded: dict[str, Any]
) -> dict[str, Any]:
    """Record what produced the numbers next to them.

    Args:
        experiment: The experiment.
        device: Torch device the weights landed on.
        maps: The maps, for their dims.
        recorded: What produced the maps, as `provenance` built it.

    Returns:
        The metadata block, ready to write.
    """
    return {
        **run_provenance(experiment),
        "teacher": recorded["teacher"],
        "layers": sorted(maps),
        "model_dim": next(iter(maps.values())).model_dim,
        "index_dim": next(iter(maps.values())).index_dim,
        "device": device,
        "torch": torch.__version__,
    }


def checkpoint(maps: Mapping[int, Indexer], extra: dict[str, Any]) -> dict[str, Any]:
    """Build what a run saves - the maps by layer, and what produced them.

    Args:
        maps: One map or QSA indexer per layer.
        extra: Anything else worth recording next to the weights.

    Returns:
        The checkpoint, ready for `torch.save`.
    """
    return {"maps": {layer: item.state_dict() for layer, item in maps.items()}, **extra}


def load_maps(blob: dict[str, Any], maps: Mapping[int, Indexer]) -> None:
    """Warm start maps from a saved checkpoint, in place.

    Args:
        blob: A checkpoint as `checkpoint` built it.
        maps: The maps to load into, keyed by layer.

    Raises:
        ValueError: The checkpoint holds no map for a layer that was asked for.
    """
    saved = blob["maps"]
    missing = [layer for layer in maps if layer not in saved]
    if missing:
        raise ValueError(f"the checkpoint holds no map for layers {missing} - it has {sorted(saved)}")
    for layer, item in maps.items():
        item.load_state_dict(saved[layer])


class Preempted(Exception):
    """A run stopped at a step boundary, its resume state written, with no job scheduler to requeue it."""


class Preemption:
    """SIGUSR2 held until the trainer reaches a step it can stop at, then passed on to whoever was listening first.

    submitit listens for it in a job and requeues the job, so the handler this replaces is called once the resume
    state is written. Outside a job nothing was listening and the run raises `Preempted` instead.
    """

    def __init__(self) -> None:
        """Start listening - only the main thread can, so elsewhere the run is never told."""
        self.requested = threading.Event()
        self.previous: Any = None
        self.listening = threading.current_thread() is threading.main_thread()
        if self.listening:
            self.previous = signal.signal(signal.SIGUSR2, self.catch)

    def catch(self, _signal: int, _frame: FrameType | None) -> None:
        """Note the signal and return, so the step under way finishes."""
        print("preemption: SIGUSR2 caught - stopping at the next step boundary", flush=True)
        self.requested.set()

    def hand_over(self) -> None:
        """Stop listening and pass the signal on - submitit requeues the job and exits.

        Raises:
            Preempted: Nothing else was listening, so nothing requeues the run.
        """
        self.close()
        if callable(self.previous):
            self.previous(signal.SIGUSR2, None)
        raise Preempted("stopped at a step boundary with the resume state written")

    def close(self) -> None:
        """Give the signal back to whoever had it."""
        if self.listening:
            signal.signal(signal.SIGUSR2, self.previous)
            self.listening = False


def save_resume(path: Path, state: dict[str, Any]) -> None:
    """Write a run's resume state whole or not at all, over the one before it.

    Args:
        path: The run's resume file.
        state: What `read_resume` hands back.
    """
    partial = path.with_name(f"{path.name}.partial")
    torch.save(state, partial)
    os.replace(partial, path)


def read_resume(path: Path, experiment: "DistillConfig", shard: list[int] | None) -> dict[str, Any] | None:
    """Return the state a run continues from, or None for a run that starts at step 0.

    Args:
        path: The run's resume file.
        experiment: The experiment, which must be the one that wrote it.
        shard: The layers this process trains, which must be the ones it wrote.

    Returns:
        The state as `save_resume` wrote it, or None when there is none.

    Raises:
        ValueError: The state was written by another configuration or another shard.
    """
    if not path.exists():
        return None
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if state["run"] != asdict(experiment) or state["shard"] != shard:
        raise ValueError(
            f"{experiment.run.name}: {path} was written by another configuration than this one - a resumed run must "
            "train the steps it planned, so start it over with --force or give it its own run.output"
        )
    return state


def resumed_maps(
    state: dict[str, Any], device: str, qsa_config: QSAConfig | None = None
) -> dict[int, Indexer]:
    """Rebuild the maps a resume state holds, bases and temperature included.

    Args:
        state: The state as `read_resume` returned it.
        device: Where the maps land.
        qsa_config: The QSA indexer the run trains, or None for LSA maps.

    Returns:
        One map or QSA indexer per layer, as it stood at the state's step.
    """
    maps: dict[int, Indexer] = {}
    for layer, saved in state["maps"].items():
        item: Indexer
        if qsa_config is not None:
            model_dim = saved["index_qk_proj"].shape[1]
            item = QSAIndexer(
                model_dim, qsa_config, saved["inverse_frequencies"], float(saved["attention_scaling"])
            )
        else:
            model_dim, index_dim = saved["key_projection"].shape
            bases = (saved["key_basis"], saved["query_basis"]) if "key_basis" in saved else None
            item = IndexMap(model_dim, index_dim, bases=bases)
        item.load_state_dict(saved)
        maps[layer] = item.to(device)
    return maps
