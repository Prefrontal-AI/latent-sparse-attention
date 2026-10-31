"""The distilled maps, as every reader after the run that wrote them reads them.

One distillation run writes one pair of projections per indexed layer.
[inject](distillation/inject.py) writes them into a checkpoint, and the trainer and the exporter read them back.
The check here is the one whose absence is silent:
a map trained on another model has the right shape at every step and simply selects the wrong keys.

The model's dim is not among them. A projection that reads a different `model_dim` fails as a shape error in
the first forward, which is loud.
`x` is position-free, so a map has no rotary layout to be trained under.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class LayerProjections:
    """One layer's two projections, each `[model_dim, index_dim]` fp32 - the weights an index key and an index query are read off `x` with."""

    query_projection: torch.Tensor
    key_projection: torch.Tensor

    def __post_init__(self) -> None:
        """Reject projections that cannot be a pair.

        Raises:
            ValueError: A projection is not 2-D, or the two disagree on shape.
        """
        if self.query_projection.ndim != 2 or self.key_projection.ndim != 2:
            raise ValueError(
                f"both projections must be [model_dim, index_dim] - got {tuple(self.query_projection.shape)} "
                f"and {tuple(self.key_projection.shape)}"
            )
        if self.query_projection.shape != self.key_projection.shape:
            raise ValueError(
                f"the projections must share their shape - {tuple(self.query_projection.shape)} "
                f"!= {tuple(self.key_projection.shape)}"
            )

    @property
    def model_dim(self) -> int:
        """Dims of the `x` the projections read"""
        return int(self.key_projection.shape[0])

    @property
    def index_dim(self) -> int:
        """Dims of the index keys and index queries they write"""
        return int(self.key_projection.shape[1])


@dataclass(frozen=True)
class DistilledMaps:
    """One distillation artifact, loaded.

    `projections` is keyed by the layer index of the model the maps were trained on, which is the index a later reader
    indexes at as well - the same layer numbering off the same checkpoint.
    """

    path: Path
    # Steps the maps were trained for when this was saved
    step: int
    # The model whose hidden states trained them, as the distillation config spelled it
    model: str
    projections: dict[int, LayerProjections]

    @property
    def layers(self) -> list[int]:
        """Layers the artifact holds a map for, in order."""
        return sorted(self.projections)

    @property
    def index_dim(self) -> int:
        """Dims of the index keys and index queries the maps write."""
        return self.projections[self.layers[0]].index_dim

    def checked_for(self, model: str, layers: list[int]) -> dict[int, LayerProjections]:
        """Return the maps for the layers a run indexes, or refuse the pair.

        Args:
            model: The model the run trains, as its own config names it.
            layers: Layers the run replaces with LSA.

        Returns:
            One `LayerProjections` per layer in `layers`.

        Raises:
            ValueError: The artifact was distilled for another model.
            ValueError: The artifact holds no map for a layer the run indexes.
        """
        if self.model != model:
            raise ValueError(
                f"{self.path} was distilled for {self.model!r} and this run trains {model!r} - "
                "a map is only valid for the model whose states trained it"
            )
        missing = sorted(set(layers) - set(self.projections))
        if missing:
            raise ValueError(
                f"{self.path} holds no map for layers {missing} - it holds {self.layers}, "
                f"and this run indexes {sorted(layers)}"
            )
        return {layer: self.projections[layer] for layer in layers}


def load(path: str | Path) -> DistilledMaps:
    """Read one distillation artifact.

    Args:
        path: The `maps.pt` or `maps_step{step}.pt` a distillation run wrote.

    Returns:
        The artifact, with one `LayerProjections` per layer it holds.

    Raises:
        FileNotFoundError: There is no artifact there.
        ValueError: The file is not one of these artifacts, or holds no maps.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"no distilled maps at {artifact}")
    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    return decode(blob, artifact)


def decode(blob: Any, artifact: Path) -> DistilledMaps:
    """Decode a checkpoint into its learned projections and provenance."""
    if not isinstance(blob, dict) or "maps" not in blob:
        raise ValueError(f"{artifact} is not a distilled map artifact - it holds no 'maps'")
    saved: dict[Any, Any] = blob["maps"]
    if not saved:
        raise ValueError(f"{artifact} holds no maps")

    projections = {int(layer): _projection(artifact, int(layer), state) for layer, state in saved.items()}
    return DistilledMaps(
        path=artifact, step=int(blob["step"]), model=str(blob["teacher"]["model"]), projections=projections
    )


def _projection(path: Path, layer: int, state: dict[str, torch.Tensor]) -> LayerProjections:
    """Build one layer's projections from what distillation saved for it.

    Args:
        path: The artifact, for the message a broken entry raises.
        layer: The layer, likewise.
        state: That layer's `IndexMap` state dict.

    Returns:
        The two projections, as one pair, as a server reads them.

    Raises:
        ValueError: The entry holds neither projection.
    """
    if "query_projection" not in state or "key_projection" not in state:
        raise ValueError(
            f"{path} layer {layer} holds {sorted(state)} - a map needs 'query_projection' and 'key_projection'"
        )
    query_projection = state["query_projection"].detach().float()
    key_projection = state["key_projection"].detach().float()
    # A map on the key and query was trained in fixed bases, and is read with them folded in
    if "key_basis" in state:
        query_projection = state["query_basis"].float() @ query_projection
        key_projection = state["key_basis"].float() @ key_projection
    return LayerProjections(query_projection=query_projection, key_projection=key_projection)


def describe(artifact: DistilledMaps) -> str:
    """Return the line a run prints for the map it trains.

    Args:
        artifact: The loaded artifact.

    Returns:
        The summary, one line.
    """
    return (
        f"{artifact.path} - {artifact.model} at step {artifact.step}, "
        f"layers {artifact.layers}, index_dim {artifact.index_dim}"
    )
