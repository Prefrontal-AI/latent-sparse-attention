"""An extraction a span of the stack at a time - one card to a span, and the residual streams between them.

A checkpoint no card holds is spread over several by a device map, and its forward is then a serial pipeline:
one card computes while the rest wait. Cutting the stack into spans that each fit one card does the same work
on one card per span, and each span runs as a job of its own.

A span runs a document the way the whole stack prefills it - a chunk at a time into a cache - and nothing a
layer carries from one chunk to the next belongs to any other layer. So a span can take every chunk of a
document before the next span starts, and each layer still sees exactly what one pass over the whole stack
hands it.

What crosses a cut is the residual streams, written to disk a chunk at a time by the span before it and read
back a chunk at a time by the span after it. A cut's streams are published whole or not at all: the bytes go
to a scratch file that is renamed into place, and the small record that says what they are is written last.
"""

import math
import os
from itertools import pairwise
from pathlib import Path

import torch

from lsa.io import read_json, write_json
from lsa.training.distillation.constants import Stage


def stage_spans(cuts: list[int], layers: list[int]) -> list[tuple[int, int]]:
    """Cut the stack before each of `cuts`, ending the last span right after the last captured layer.

    Nothing past the last captured layer is read by an extraction, so no span loads it.

    Args:
        cuts: Layer indices each span after the first starts at, increasing.
        layers: The layers the extraction captures.

    Returns:
        The spans `[start, stop)`, in order.

    Raises:
        ValueError: The cuts do not increase inside the stack, or a span captures no layer.
    """
    edges = [0, *cuts, max(layers) + 1]
    if any(stop <= start for start, stop in pairwise(edges)):
        raise ValueError(
            f"stages must increase from 1 up to the last captured layer, {max(layers)} - got {cuts}"
        )
    spans = list(pairwise(edges))
    for start, stop in spans:
        if not any(start <= index < stop for index in layers):
            raise ValueError(f"span [{start}, {stop}) captures none of the layers {layers} - move a cut")
    return spans


def boundary_path(root: str | Path, cut: int, collection: str, split: str, document: str) -> Path:
    """Return where the residual streams layer `cut` reads for one document live.

    Args:
        root: The cache root.
        cut: The layer the streams are the input of.
        collection: The dataset's directory.
        split: `train` or `eval`.
        document: Document name inside the split.

    Returns:
        The streams file, which may not exist.
    """
    folder = Path(root) / Stage.BOUNDARIES / f"{Stage.CUT_TAG}{cut}" / collection / split
    return folder / f"{document}{Stage.STREAMS_SUFFIX}"


def _meta_path(path: Path) -> Path:
    """Return the metadata file beside a streams file."""
    return path.with_suffix(Stage.META_SUFFIX)


def boundary_ready(path: Path) -> bool:
    """Return whether a cut's streams are whole on disk.

    Args:
        path: The streams file.

    Returns:
        True once the span before the cut has published them.
    """
    return _meta_path(path).exists()


def remove_boundary(path: Path) -> None:
    """Delete a cut's streams once the span after it has read them, the record first so nothing sees them ready.

    Args:
        path: The streams file.
    """
    _meta_path(path).unlink(missing_ok=True)
    path.unlink(missing_ok=True)


class BoundaryWriter:
    """Write a cut's residual streams for one document a chunk at a time, and publish them whole."""

    def __init__(self, path: Path) -> None:
        """Open a scratch file beside where the streams will live.

        Args:
            path: The streams file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.scratch = path.with_suffix(f".{os.getpid()}.tmp")
        self.scratch.write_bytes(b"")
        self.rows = 0
        self.row_shape: list[int] = []
        self.dtype = ""

    def append(self, streams: torch.Tensor) -> None:
        """Write one chunk's streams after the ones before it.

        Args:
            streams: The chunk's streams [1, S, ...], on any device.
        """
        host = streams[0].contiguous().cpu()
        self.row_shape = list(host.shape[1:])
        self.dtype = str(host.dtype).removeprefix("torch.")
        with open(self.scratch, "ab") as handle:
            handle.write(host.view(torch.uint8).numpy().data)
        self.rows += host.shape[0]

    def publish(self) -> None:
        """Move the streams into place, then write the record that says they are whole."""
        os.replace(self.scratch, self.path)
        write_json(
            _meta_path(self.path), {"length": self.rows, "row_shape": self.row_shape, "dtype": self.dtype}
        )

    def discard(self) -> None:
        """Drop the scratch file of a document that failed part way."""
        self.scratch.unlink(missing_ok=True)


class Boundary:
    """A cut's published residual streams for one document, read a chunk at a time."""

    def __init__(self, path: Path) -> None:
        """Map the streams without reading them.

        Args:
            path: The streams file, which must be published.
        """
        # The training extra's, like everything an extraction runs - the configs import this module bare
        import numpy as np

        record = read_json(_meta_path(path))
        self.length: int = record["length"]
        self.row_shape: list[int] = record["row_shape"]
        self.dtype: torch.dtype = getattr(torch, record["dtype"])
        row_bytes = math.prod(self.row_shape) * torch.empty((), dtype=self.dtype).element_size()
        self.rows = np.memmap(path, dtype=np.uint8, mode="r", shape=(self.length, row_bytes))

    def read(self, start: int, stop: int) -> torch.Tensor:
        """Read the streams of positions `[start, stop)`.

        Args:
            start: First position.
            stop: One past the last.

        Returns:
            The streams [1, stop - start, ...] on the host.
        """
        chunk = torch.from_numpy(self.rows[start:stop].copy())
        return chunk.view(self.dtype).view(1, stop - start, *self.row_shape)
