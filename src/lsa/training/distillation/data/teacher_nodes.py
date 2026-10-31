"""The online teacher across nodes - every node's copies read a share of the drawn documents, and one trainer takes them.

    lsa distill train <config> --gpus 8 --nodes 2 --cpus 64 --mem-gb 1200 --partition <partition>

The job is one task a node. The first is the run as on one node - the trainer on its first card and a copy of the
teacher on each of the others. Every other task loads a copy on each of its cards and serves (`serve`): it reads
documents the trainer asked for and leaves their targets on the shared disk, where the trainer's pools load them.

The split is by document and decided as the copies go: the trainer publishes every document it asks for, in the order
it asks, and a copy on any node takes the first one nobody has claimed (`Staging.claim`, an exclusive create). So a
node of slow 1M documents does not hold up one of short ones, and the pools meet the order they asked for, whichever
node read a document. Every copy runs the same code on the same card type, so a document's targets are the same bits
whichever node read it.

The transport is files under the run directory rather than a process group. A document's layers are written as the
blobs an extraction builds and loaded back with the same `Document.from_blob` a copy on the trainer's node hands over -
no serialisation of its own, no rendezvous or port, and nothing to tear down when a node goes: a requeued job starts a
directory of its own. The rate asked of the disk is small - eight copies reading 1M documents some 1,000 s apiece,
27 GB each, is about 0.2 GB/s - and `staging_gb` bounds what waits there, past which the serving copies stop claiming.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lsa.training.distillation.constants import Online

# The trainer's requests, one json line each, in the order it made them
QUEUE: str = "queue.jsonl"
CLAIMS: str = "claims"
READY: str = "ready"
# Written by the trainer once it has every document it will take, which stops the serving nodes
FINISHED: str = "finished"


@dataclass(frozen=True)
class Request:
    """One document the trainer asked for, as every node numbers it."""

    index: int
    collection: str
    name: str
    layers: list[int]


@dataclass
class Staging:
    """One launch's staging directory, as one node sees it."""

    root: Path
    # This task's node - 0 is the trainer's
    node: int
    nodes: int
    # Bytes the documents waiting for the trainer may occupy, which the serving nodes stop claiming past
    budget: float
    _requests_seen: int = field(default=0, init=False)

    @classmethod
    def of_job(cls, out: Path, budget_gb: float) -> "Staging | None":
        """Return this task's view of the job's staging, or None for a job of one node.

        Args:
            out: The run directory.
            budget_gb: `data.teacher.staging_gb`.

        Returns:
            The staging, named for this launch of the job - the job and how many times slurm restarted it.
        """
        nodes = int(os.environ.get("SLURM_NTASKS", "1"))
        if nodes < 2:
            return None
        launch = f"{os.environ['SLURM_JOB_ID']}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        return cls(out / Online.STAGING / launch, int(os.environ["SLURM_PROCID"]), nodes, budget_gb * 1e9)

    @property
    def serving(self) -> bool:
        """Whether this node only reads, for the trainer on another"""
        return self.node != 0

    def open(self) -> None:
        """Start this launch's directory empty, and drop any launch before it - the trainer's node only."""
        if self.root.parent.exists():
            for launch in self.root.parent.iterdir():
                if launch != self.root:
                    shutil.rmtree(launch, ignore_errors=True)
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / CLAIMS).mkdir(parents=True)
        (self.root / READY).mkdir()
        (self.root / QUEUE).touch()

    def publish(self, requests: list[Request]) -> None:
        """Append the trainer's requests, in order, for every node to read.

        Args:
            requests: The documents asked for, numbered in the order they were asked.
        """
        with open(self.root / QUEUE, "a") as queue:
            for request in requests:
                queue.write(json.dumps(request.__dict__) + "\n")

    def requests(self) -> list[Request]:
        """Return the requests published since the last call, whole lines only."""
        path = self.root / QUEUE
        if not path.exists():
            return []
        lines = path.read_text().split("\n")
        # The last piece is empty or a line still being written
        complete = lines[:-1]
        new = complete[self._requests_seen :]
        self._requests_seen = len(complete)
        return [Request(**json.loads(line)) for line in new]

    def claim(self, index: int) -> bool:
        """Claim one request for this node's copy to read, and return whether nobody had.

        Args:
            index: The request.
        """
        try:
            descriptor = os.open(self.root / CLAIMS / str(index), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(descriptor, str(self.node).encode())
        os.close(descriptor)
        return True

    def put(self, index: int, blobs: dict[int, dict[str, Any]]) -> None:
        """Leave one read document's layers for the trainer, the marker that they are whole written last.

        Args:
            index: The request.
            blobs: Each layer's blob, as the trainer's own copies build it.
        """
        directory = self.root / READY / str(index)
        directory.mkdir()
        for layer, blob in blobs.items():
            partial = directory / f"layer{layer}.pt.partial"
            torch.save(blob, partial)
            os.replace(partial, directory / f"layer{layer}.pt")
        (self.root / READY / f"{index}.done").touch()

    def ready(self, index: int) -> bool:
        """Return whether another node has left a request's layers whole."""
        return (self.root / READY / f"{index}.done").exists()

    def claimed(self, index: int) -> bool:
        """Return whether a copy on any node has claimed a request."""
        return (self.root / CLAIMS / str(index)).exists()

    def take(self, index: int, layer: int) -> dict[str, Any]:
        """Load one layer of a document another node read, and free its file on the disk.

        The document's directory stays until `release`: each layer's pool takes its own layer on a thread of its own,
        and a thread listing the directory while another removes it fails on NFS with a stale handle.

        Args:
            index: The request.
            layer: The layer.

        Returns:
            The layer's blob, on the host.
        """
        path = self.root / READY / str(index) / f"layer{layer}.pt"
        blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        path.unlink()
        return blob

    def release(self, index: int) -> None:
        """Remove a document another node read, once the trainer has taken every layer of it.

        The trainer's teacher calls this once, under its lock, from the take that took the last layer - it alone knows
        which layers are left, so nothing lists the shared directory to find out.

        Args:
            index: The request.
        """
        (self.root / READY / f"{index}.done").unlink(missing_ok=True)
        shutil.rmtree(self.root / READY / str(index), ignore_errors=True)

    def footprint(self) -> int:
        """Return the bytes the documents waiting for the trainer occupy."""
        total = 0
        for directory in (self.root / READY).iterdir():
            # The trainer removes a document's directory once it has every layer, from another node
            try:
                total += sum(entry.stat().st_size for entry in os.scandir(directory))
            except (FileNotFoundError, NotADirectoryError):
                continue
        return total

    def has_room(self) -> bool:
        """Return whether a serving copy may claim another document - under the budget, or nothing waiting."""
        used = self.footprint()
        return used == 0 or used < self.budget

    def finish(self) -> None:
        """Tell the serving nodes the trainer takes nothing more."""
        if self.root.exists():
            (self.root / FINISHED).touch()

    def finished(self) -> bool:
        """Return whether the trainer has said it takes nothing more."""
        return (self.root / FINISHED).exists()
