"""Write a distilled index space in the form the server reads it, as a CPU job.

    lsa distill export results/training/glm53_flash/long_16k_1m_heavy_x/maps.pt projections.pt

A run saves its maps keyed by layer, one `IndexMap` state dict each, because that is what warm-starting
another run needs. The plugin asks a different question - one layer at a time, by name - and so reads a flat
file of `<layer>.W_Q_idx` and `<layer>.W_K_idx`. This is the whole of the difference between the two, and
writing it out is what lets a trained space be served rather than only scored offline.

Nothing is transformed on the way. `W_K_idx` is `key_projection` and `W_Q_idx` is `query_projection`, both
`[model_dim, index_dim]`, because the server normalises after projecting `x` exactly as the trainer does - so a
map that scored one way offline selects the same keys online.

A run on the key and query, `model.indexer.input: key_and_query`, trained its projections in the latent bases the
freeze distilled in, and is written with them folded in, `[kv_lora_rank, index_dim]` - what the server reads off the
plain key latent and summed query. Its sidecar records the input the run trained on.

A map distilled before the switch to `x` reads the key and the heads' queries, `[kv_lora_rank, index_dim]`, and
was exported then as a projection file of the same layout. Such a file is re-exported as it is - its tensors
untouched, read back through the server's own check - with a sidecar that names its input and the file and sha it
came from, so it is served under `model.indexer.input: key_and_query` - the design's input and the default:

    lsa distill export --input key_and_query glm53_flash__long-16k-1M.pt glm53_flash__long-16k-1M-latent.pt

The projection file is flat tensors and cannot hold the loss temperature a run trained each layer at, which an offline
read that ranks by a LogSumExp needs, so the sidecar does: `"tau": {"<layer>": <tau>}`, from the run's `log_tau` at the
checkpoint. A projection file re-exported keeps the one its own sidecar had, and a map that has none gets one from
`lsa distill calibrate-tau`. `map_tau` is the reader.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from lsa import env, io, slurm
from lsa.sglang.index import weights
from lsa.sglang.settings.design import ATTENTION_INPUT, INDEXER_INPUTS
from lsa.training import distilled_maps
from lsa.training.distillation.constants import Results

# What the plugin's loader looks a layer up by - see [weights](../../sglang/index/weights.py)
QUERY: str = "{layer}.W_Q_idx"
KEY: str = "{layer}.W_K_idx"


def projections(blob: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten a run's checkpoint into the plugin's projection file.

    Args:
        blob: A checkpoint as `checkpoint.checkpoint` built it.

    Returns:
        One `[model_dim, index_dim]` tensor per layer and direction, keyed the way the plugin reads them.

    Raises:
        ValueError: The checkpoint holds no maps, or a layer's two projections disagree on shape.
    """
    artifact = distilled_maps.decode(blob, Path("<checkpoint>"))
    return {
        name.format(layer=layer): tensor
        for layer, projection in artifact.projections.items()
        for name, tensor in ((QUERY, projection.query_projection), (KEY, projection.key_projection))
    }


def is_projection_file(blob: Any) -> bool:
    """Return whether a loaded file is already a projection file - only `<layer>.W_Q_idx` and `<layer>.W_K_idx`.

    Args:
        blob: What `torch.load` returned.
    """
    if not isinstance(blob, dict) or not blob:
        return False
    for name in blob:
        layer, _, kind = str(name).partition(".")
        if not layer.isdigit() or kind not in (weights.QUERY, weights.KEY):
            return False
    return True


def write(maps: Path, out: Path) -> dict[str, torch.Tensor]:
    """Read a run's maps, or a projection file, and write the projection file beside whatever asked for it.

    Args:
        maps: The run's `maps.pt`, a directory holding one, or a projection file to re-export as it is.
        out: Where to write the projection file.

    Returns:
        What was written.

    Raises:
        FileNotFoundError: No checkpoint at that path.
        ValueError: A projection file's layers disagree on their shapes, or `out` is the file itself.
    """
    path = maps / Results.MAPS if maps.is_dir() else maps
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    if out.resolve() == path.resolve():
        raise ValueError(f"{out} is the file being exported - write the export beside it")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if is_projection_file(blob):
        input_dim, index_dim = next(iter(blob.values())).shape
        layers = {int(str(name).partition(".")[0]) for name in blob}
        held = weights.read(path, int(input_dim), int(index_dim), layers)
        flat = {
            name.format(layer=layer): tensor
            for layer, (query, key) in held.items()
            for name, tensor in ((QUERY, query), (KEY, key))
        }
    else:
        flat = projections(blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, out)
    return flat


def trained_tau(blob: dict[str, Any]) -> dict[str, float]:
    """Return the temperature each layer of a run's checkpoint was trained at.

    Args:
        blob: A checkpoint as `checkpoint.checkpoint` built it.

    Returns:
        The temperature by layer, as the layer's name - the sidecar's keys are strings.
    """
    return {
        str(layer): float(math.exp(float(state["log_tau"])))
        for layer, state in sorted(blob["maps"].items(), key=lambda item: int(item[0]))
    }


def map_tau(path: str | Path, layers: Sequence[int]) -> dict[int, float]:
    """Read the temperatures an exported map's sidecar records, for the layers asked.

    Args:
        path: The projection file, whose `.json` sidecar sits beside it.
        layers: The layers wanted.

    Returns:
        The temperature of each layer.

    Raises:
        ValueError: The sidecar is missing, records no temperature, or lacks a layer - and says how to calibrate one.
    """
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    recorded = io.read_json(sidecar_path, {}).get("tau") if sidecar_path.is_file() else None
    asked = f"lsa distill calibrate-tau <training config> {path}"
    if not recorded:
        raise ValueError(f"{path} records no loss temperature in its sidecar - calibrate one with `{asked}`")
    missing = sorted(set(layers) - {int(layer) for layer in recorded})
    if missing:
        raise ValueError(
            f"{sidecar_path} records no temperature for layers {missing} - calibrate with `{asked}`"
        )
    return {layer: float(recorded[str(layer)]) for layer in layers}


def sidecar(maps: Path, out: Path, indexer_input: str = ATTENTION_INPUT) -> dict[str, Any]:
    """Say where an exported map came from, for the `<map>.json` written beside it.

    A projection file is a flat tensor dict and cannot say which run trained it, so this does - the run
    directory, the checkpoint in it, and the config, commit and date the run's own metadata recorded. A projection
    file re-exported as it is names the file and its sha, and carries the provenance its own sidecar had.

    Args:
        maps: The run's maps checkpoint the file was exported from, or the projection file re-exported.
        out: The projection file.
        indexer_input: The `model.indexer.input` the maps read.

    Returns:
        The sidecar, ready to write.
    """
    if maps.with_suffix(".json").is_file() and is_projection_file(
        torch.load(maps, map_location="cpu", weights_only=False)
    ):
        source = io.read_json(maps.with_suffix(".json"), {})
        return {
            "sha256": io.file_sha256(out),
            "input": indexer_input,
            "source": str(maps),
            "source_sha256": io.file_sha256(maps),
            "run": source.get("run", ""),
            "checkpoint": source.get("checkpoint", ""),
            "config": source.get("config", {}),
            "git": source.get("git", {}),
            "date": source.get("date", ""),
            "note": f"{maps.name} re-exported as it is, its tensors untouched",
            **({"tau": source["tau"]} if source.get("tau") else {}),
        }
    run = maps.parent
    meta_path = run / Results.META
    meta = io.read_json(meta_path, {})
    return {
        "sha256": io.file_sha256(out),
        "input": indexer_input,
        "run": str(run),
        "checkpoint": maps.name,
        "config": meta.get("config", {}),
        "git": meta.get("git", {}),
        "date": str(meta.get("date", ""))[:10],
        "note": "",
        "tau": trained_tau(torch.load(maps, map_location="cpu", weights_only=False)),
    }


def trained_input(path: Path) -> str | None:
    """Return the `model.indexer.input` a run's checkpoint was trained on, or None for a projection file.

    Args:
        path: A run's maps checkpoint, or a projection file.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if is_projection_file(blob):
        return None
    indexer = blob.get("run", {}).get("model", {}).get("indexer", {})
    return str(indexer.get("input", ATTENTION_INPUT))


def export(maps: Path, out: Path, indexer_input: str | None = None) -> Path:
    """Write one run's maps for the server with the sidecar naming the run, and print what the file holds.

    Args:
        maps: A run's maps checkpoint, the run directory holding it, or a projection file to re-export.
        out: Where to write the projection file.
        indexer_input: The `model.indexer.input` the maps read, which the sidecar records - a run's own when None,
            and `x` for a projection file.

    Returns:
        The projection file.

    Raises:
        ValueError: `indexer_input` is not one of `INDEXER_INPUTS`, or not the input the run trained on.
    """
    if indexer_input is not None and indexer_input not in INDEXER_INPUTS:
        raise ValueError(f"input must be one of {INDEXER_INPUTS} - got {indexer_input!r}")
    checkpoint = (maps / Results.MAPS if maps.is_dir() else maps).resolve()
    trained = trained_input(checkpoint) if checkpoint.exists() else None
    if trained is not None and indexer_input not in (None, trained):
        raise ValueError(
            f"{checkpoint} was trained on input {trained!r} - it cannot be exported as {indexer_input!r}"
        )
    flat = write(maps, out)
    recorded = trained or indexer_input or ATTENTION_INPUT
    io.write_json(out.with_suffix(".json"), sidecar(checkpoint, out.resolve(), recorded))
    layers = sorted({int(name.split(".")[0]) for name in flat})
    shape = next(iter(flat.values())).shape
    print(f"{out}: layers {layers}, each map {tuple(shape)}", flush=True)
    return out


def main() -> None:
    """Submit the export of one run's maps for the server."""
    parser = argparse.ArgumentParser(
        description="Write a run's maps as the file the server reads, as a CPU job"
    )
    parser.add_argument("maps", type=Path, help="a run's maps.pt, or the directory holding it")
    parser.add_argument("out", type=Path, help="where to write the projection file")
    parser.add_argument(
        "--input",
        choices=INDEXER_INPUTS,
        default=None,
        help="the model.indexer.input the maps read, recorded in the sidecar - a run's own by default",
    )
    slurm.add_flags(parser)
    args = parser.parse_args()
    slurm.submit(
        args,
        "lsa-export",
        env.OFFLINE_HF,
        slurm.CPU,
        export,
        args.maps.resolve(),
        args.out.resolve(),
        args.input,
    )
