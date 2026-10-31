"""The rolling pool - how a run trains on a layer's share of the cache that no node holds.

A layer's share of the `x` cache is some 3 TiB and a node has 1.4 TB of host memory, so the documents stream
from disk through a pool of bounded size and a run draws its steps from the documents that are in it.

The draw is the one a run always made - a step takes a document in proportion to its tokens, or equally under
`draw: document` - but taken for the whole run before it starts.
A document drawn six times over a run is then read once and trained on six times, in steps that other
documents' steps are mixed between, so a million-token document costs one 8.7 GB read rather than six.
The multiset of documents a run trains on is exactly the draw's, and only the order moves.

    draw       the document of every step, off the run's one generator           [steps]
    items      the distinct documents drawn, each with the steps that train on it
    sequence   which item each step trains on, interleaving a window of items     [steps]

The sequence is worked out ahead of time from the seed, the window and the card's budget alone, so a run
reproduces whatever the disk did.
A resumed run takes the items and the sequence its snapshot saved, less the steps already trained (`remaining`), so
it trains the steps an uninterrupted run would have - a document a step had already trained on is read again.
The window is the number of documents whose steps are mixed at once, and the card holds exactly those.
Three stages then follow it, each in order of first use:

    disk  --  readers  -->  host pool  --  copier  -->  card  --  steps

The readers fill the host pool up to its budget, the copier moves a document to the card once there is room
for it and drops the host copy, and a document leaves the card after its last step.
A step waits only when the document it trains on has not arrived, and the pool counts how long that was.
Under the key and query the copier also reads the key latents and summed queries off `x` on the card and lets `x`
go, so a document holds an eighth of what it is budgeted at.
"""

import itertools
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.constants import Cache, Draw, Pool
from lsa.training.distillation.data.cache import Document, layer_path, listing, native_path, set_root

if TYPE_CHECKING:
    from lsa.training.distillation.config import DistillConfig

# A document in memory is its file plus the int64 and fp32 copies of the teacher's ids and masses
# => about 1% over the file at 1M tokens and 5% at 16k, so a tenth is margin for both
RESIDENT_FACTOR: float = 1.1


@dataclass(frozen=True)
class Entry:
    """One document of the training sets, before it is drawn."""

    # Root, collection and split the document's files sit under
    where: tuple[Path, str, str]
    name: str
    tokens: int

    def path(self, layer: int) -> Path:
        """Return where this document's file for one layer is."""
        root, collection, split = self.where
        return layer_path(root, collection, split, self.name, layer)


@dataclass(frozen=True)
class Item:
    """One distinct document a layer's steps train on, and what it costs to hold."""

    layer: int
    entry: Entry
    # Steps of the layer that train on it
    uses: int
    # What it occupies in memory, estimated from its file
    size: int

    @property
    def path(self) -> Path:
        """Where the file is"""
        return self.entry.path(self.layer)


class Interrupted(Exception):
    """A step stopped waiting for its document because the run is being stopped - the pool is as it was before it."""


@dataclass(frozen=True)
class Drawn:
    """One step's draw - the layer it trains, the document and the query rows it reads."""

    step: int
    layer: int
    document: Document
    # [batch] int64 training query rows, on the host
    rows: torch.Tensor


def catalog(experiment: "DistillConfig") -> list[Entry]:
    """Return every document of the training sets, in a stable order.

    Args:
        experiment: The experiment.

    Returns:
        One entry per document, set by set and by name inside a set.
    """
    entries = []
    for document_set in experiment.data.train:
        root = set_root(experiment.data.root, document_set)
        for entry in listing(experiment.data.root, document_set):
            where = (root, document_set.source, document_set.split)
            entries.append(Entry(where, entry["name"], int(entry["length"])))
    return entries


def drawn_documents(entries: Sequence[Entry], steps: int, seed: int, draw: str) -> list[int]:
    """Draw the document of every step of a run.

    Every step of the run is drawn, whichever layer it trains, so a shard draws for its layers exactly what
    the whole run does.
    The draw reads only token counts, which every layer's copy of a document shares.

    Args:
        entries: The training documents.
        steps: Steps of the run.
        seed: The run's seed.
        draw: `Draw.TOKEN` for a document in proportion to its tokens, `Draw.DOCUMENT` for each equally.

    Returns:
        The index into `entries` of each step's document, step 1 first.
    """
    if steps == 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    if draw == Draw.TOKEN:
        weights = torch.tensor([entry.tokens for entry in entries], dtype=torch.float32)
        return [
            int(index) for index in torch.multinomial(weights, steps, replacement=True, generator=generator)
        ]
    return [int(index) for index in torch.randint(len(entries), (steps,), generator=generator)]


def layer_steps(layers: Sequence[int], steps: int) -> dict[int, list[int]]:
    """Deal a run's steps to its layers, which they rotate over.

    Args:
        layers: Every layer of the run, which steps rotate over.
        steps: Steps of the run, numbered from 1.

    Returns:
        The steps each layer trains, in order.
    """
    dealt: dict[int, list[int]] = {layer: [] for layer in layers}
    for step in range(1, steps + 1):
        dealt[layers[step % len(layers)]].append(step)
    return dealt


def items_of(
    entries: Sequence[Entry],
    drawn: Sequence[int],
    layer: int,
    steps: Sequence[int],
    seed: int,
    order: Sequence[int] | None = None,
    size_of: Callable[[Entry, int], int] | None = None,
) -> list[Item]:
    """Return the distinct documents a layer's steps train on, in the order they arrive.

    The order is a shuffle of the documents drawn, so the pool meets lengths and domains mixed rather than
    in the order of the catalog - the layer's own, or one every layer shares when the documents come from a
    teacher that reads each once for all of them.

    Args:
        entries: The training documents.
        drawn: The document of every step of the run, as `drawn_documents` returns it.
        layer: The layer.
        steps: The steps that train it.
        seed: The run's seed, which the arrival shuffle is taken from.
        order: Every document's place in an arrival order the layers share, as `shared_order` returns it, or None
            for the layer's own.
        size_of: What a document occupies at a layer, or None for its file's size.

    Returns:
        One item per distinct document, in arrival order.

    Raises:
        FileNotFoundError: A drawn document has no file for the layer.
    """
    counts: dict[int, int] = {}
    for step in steps:
        counts[drawn[step - 1]] = counts.get(drawn[step - 1], 0) + 1
    if order is None:
        generator = torch.Generator().manual_seed(seed + 1 + layer)
        arrival = [sorted(counts)[int(index)] for index in torch.randperm(len(counts), generator=generator)]
    else:
        arrival = [index for index in order if index in counts]
    items, absent = [], []
    for index in arrival:
        entry = entries[index]
        if size_of is not None:
            items.append(Item(layer, entry, counts[index], size_of(entry, layer)))
            continue
        try:
            size = os.stat(entry.path(layer)).st_size
        except FileNotFoundError:
            absent.append(str(entry.path(layer)))
            continue
        items.append(Item(layer, entry, counts[index], int(size * RESIDENT_FACTOR)))
    if absent:
        raise FileNotFoundError(
            f"layer {layer}: {len(absent)} of {len(arrival)} drawn documents have no file, the first being {absent[0]}"
        )
    return items


def shared_order(drawn: Sequence[int], seed: int) -> list[int]:
    """Return one arrival order of every document the run draws, for layers that share a teacher.

    Each layer meets its own documents in this order, so a teacher that reads a document once for every layer
    reads them in the order every pool asks for them.

    Args:
        drawn: The document of every step of the run, as `drawn_documents` returns it.
        seed: The run's seed.

    Returns:
        Indices into the catalog, each drawn document once.
    """
    distinct = sorted(set(drawn))
    generator = torch.Generator().manual_seed(seed + 1)
    return [distinct[int(index)] for index in torch.randperm(len(distinct), generator=generator)]


def require_native(entries: Sequence[Entry], layers: Sequence[int]) -> None:
    """Refuse a run whose objective reads the checkpoint's own selection when any document lacks it.

    A missing selection would otherwise surface hours in, when the pool first reaches that document.

    Args:
        entries: The training documents.
        layers: The layers the run trains.

    Raises:
        FileNotFoundError: Some (document, layer) has no native selection beside its file.
    """
    absent = [
        str(native_path(entry.path(layer)))
        for entry in entries
        for layer in layers
        if not native_path(entry.path(layer)).exists()
    ]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} of {len(entries) * len(layers)} native selections are missing, the first being "
            f"{absent[0]} - compute them with `lsa distill native-picks`"
        )


def interleave(items: Sequence[Item], device_bytes: float, window: int, seed: int) -> list[int]:
    """Work out which item each step of a layer trains on, mixing the steps of a window of items.

    Items become active in arrival order, up to `window` of them at once and while the active ones fit the
    card's budget, and each step trains on an active item chosen in proportion to the steps it has left.
    An item leaves when its steps are done.
    The first item is always admitted, so a document larger than the budget is refused by the card rather than
    by this plan.

    Args:
        items: The layer's items in arrival order.
        device_bytes: What the active items may occupy on the card.
        window: The most items active at once.
        seed: The seed of the choice.

    Returns:
        The index into `items` of each of the layer's steps, in order.
    """
    generator = torch.Generator().manual_seed(seed)
    left = [item.uses for item in items]
    active: list[int] = []
    sequence: list[int] = []
    arriving = 0
    while len(sequence) < sum(item.uses for item in items):
        while (
            arriving < len(items)
            and len(active) < window
            and (
                not active
                or sum(items[index].size for index in active) + items[arriving].size <= device_bytes
            )
        ):
            active.append(arriving)
            arriving += 1
        weights = torch.tensor([left[index] for index in active], dtype=torch.float32)
        chosen = active[int(torch.multinomial(weights, 1, generator=generator))]
        sequence.append(chosen)
        left[chosen] -= 1
        if not left[chosen]:
            active.remove(chosen)
    return sequence


def remaining(items: Sequence[Item], sequence: Sequence[int], cursor: int) -> tuple[list[Item], list[int]]:
    """Return what is left of a layer's plan once its first steps are trained - for a run resumed partway.

    An item no step left trains on is dropped, and one some steps trained on already keeps only the steps left, so the
    pool reads it again and frees it after its last.

    Args:
        items: The layer's items in arrival order.
        sequence: The item of every step of the layer, as `interleave` returned it.
        cursor: Steps of the layer already trained.

    Returns:
        items: The items some step left trains on, in arrival order, each with the steps left on it.
        sequence: The item of every step left, as indices into those items.
    """
    left = sequence[cursor:]
    uses: dict[int, int] = {}
    for index in left:
        uses[index] = uses.get(index, 0) + 1
    kept = [index for index in range(len(items)) if index in uses]
    renumbered = {index: position for position, index in enumerate(kept)}
    return (
        [replace(items[index], uses=uses[index]) for index in kept],
        [renumbered[index] for index in left],
    )


def stream(
    paths: Sequence[Path], indexer: IndexerConfig, ahead: int = Pool.EVAL_AHEAD, native: bool = False
) -> Iterator[Document]:
    """Read documents in order, a few ahead of the one being handled, each on the host.

    Args:
        paths: The files, in the order to yield them.
        indexer: The design's knobs the cache must agree with.
        ahead: Documents in flight at once, including the one yielded.
        native: Read each document's native selection too.

    Yields:
        One `Document` per path.
    """
    with ThreadPoolExecutor(max(1, ahead)) as readers:
        pending: deque[Future[Document]] = deque()
        remaining = iter(paths)
        for path in itertools.islice(remaining, ahead):
            pending.append(readers.submit(Document, path, indexer, Cache.HOST, native))
        while pending:
            document = pending.popleft().result()
            following = next(remaining, None)
            if following is not None:
                pending.append(readers.submit(Document, following, indexer, Cache.HOST, native))
            yield document


class Rolling:
    """One layer's pool - the readers, the copier and the steps' view of what has arrived.

    Built from a layer's items and the sequence of steps over them, it reads each item once, in order of first
    use, and hands the step its document.
    """

    def __init__(
        self,
        layer: int,
        items: Sequence[Item],
        sequence: Sequence[int],
        indexer: IndexerConfig,
        device: str | torch.device,
        host_bytes: float,
        device_bytes: float,
        batch: int,
        seed: int,
        workers: int = Cache.READ_WORKERS,
        native: bool = False,
        reader: Callable[[Document], Document] | None = None,
        load: Callable[[Item], Document] | None = None,
    ) -> None:
        """Start reading the layer's documents.

        Args:
            layer: The layer.
            items: The layer's items in arrival order.
            sequence: The item of every step, as `interleave` returns it.
            indexer: The design's knobs the cache must agree with.
            device: Where a step's document is staged.
            host_bytes: What documents read and not yet staged may occupy in host memory.
            device_bytes: What the staged documents may occupy on the device, which is the budget the plan
                was worked out under.
            batch: Query rows a step trains on.
            seed: The seed the rows are drawn from.
            workers: Documents read at once.
            native: Read each document's native selection too.
            reader: Reads the run's input off a document's `x` once it is staged - the key and query, which let
                `x` go - or None to train on `x`.
            load: Hands over one item's document on the host, or None to read its file - the online teacher's
                `take`, which waits for the teacher to have read it.
        """
        self.items = list(items)
        self.reader = reader
        self.load = load
        self.sequence = list(sequence)
        self.layer = layer
        self.indexer = indexer
        self.native = native
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.host_bytes = host_bytes
        self.device_bytes = device_bytes
        self.batch = batch
        self.generator = torch.Generator().manual_seed(seed + 1000 + self.layer)

        self._cond = threading.Condition()
        self._stop = False
        self._error: BaseException | None = None
        self._futures: dict[int, Future[Document]] = {}
        self._ready: dict[int, Document] = {}
        self._left = [item.uses for item in self.items]
        self._seen: set[int] = set()
        self._cursor = 0
        self.host_used = 0
        self.device_used = 0

        self.started = time.monotonic()
        self.documents_read = 0
        self.bytes_read = 0
        self.read_seconds = 0.0
        self.waited = 0.0

        # On a card the copy runs on a stream of its own, beside whatever the card is computing
        self._copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._readers = ThreadPoolExecutor(workers)
        self._threads = [
            threading.Thread(target=self._dispatch, name=f"pool-read-{self.layer}", daemon=True),
            threading.Thread(target=self._stage, name=f"pool-stage-{self.layer}", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def steps(self) -> int:
        """Steps this layer trains"""
        return len(self.sequence)

    def next(self, step: int, interrupt: threading.Event | None = None) -> Drawn:
        """Return the draw of the layer's next step, waiting for its document if it has not arrived.

        Args:
            step: The run's step number, which the draw is labelled with.
            interrupt: Set when the run is being stopped, which ends a wait without taking the step.

        Returns:
            The step's document, staged on the device, and its query rows.

        Raises:
            Interrupted: `interrupt` was set while the step waited - the pool has not moved.
            RuntimeError: A reader or the copier failed.
        """
        index = self.sequence[self._cursor]
        item = self.items[index]
        began = time.monotonic()
        said = began
        with self._cond:
            while index not in self._ready and self._error is None:
                if interrupt is not None and interrupt.is_set():
                    raise Interrupted(f"layer {self.layer} step {step} stopped waiting for {item.entry.name}")
                self._cond.wait(timeout=Pool.LOG_SECONDS if interrupt is None else Pool.INTERRUPT_SECONDS)
                if index not in self._ready and time.monotonic() - said >= Pool.LOG_SECONDS:
                    said = time.monotonic()
                    print(
                        f"pool: layer {self.layer} step {step} has waited {time.monotonic() - began:.0f} s for "
                        f"{item.entry.name} - {self.describe()}",
                        flush=True,
                    )
            if self._error is not None:
                raise RuntimeError(
                    f"the pool for layer {self.layer} failed reading {item.entry.name}"
                ) from self._error
            self._cursor += 1
            document = self._ready[index]
            self._left[index] -= 1
            if not self._left[index]:
                del self._ready[index]
                self.device_used -= item.size
                if self.device.type != "cuda":
                    self.host_used -= item.size
                self._cond.notify_all()
            self.waited += time.monotonic() - began
        if index not in self._seen:
            self._seen.add(index)
            if self.device.type == "cuda":
                # Memory the copy stream allocated must not be reused until the compute stream is done with it
                compute = torch.cuda.current_stream(self.device)
                for tensor in document.tensors():
                    tensor.record_stream(compute)
        # Every training row unless the closed-block rule hides the ones with none closed behind them
        trainable = document.train_rows().cpu()
        rows = trainable[torch.randperm(len(trainable), generator=self.generator)[: self.batch]]
        return Drawn(step, self.layer, document, rows)

    def state(self) -> dict[str, Any]:
        """Return what a resumed run rebuilds this pool from - its plan, how far the steps are, and its row generator.

        Returns:
            The items and the sequence as planned, the steps trained, the device budget the plan was worked out under
            and the generator's state.
        """
        with self._cond:
            return {
                "items": list(self.items),
                "sequence": list(self.sequence),
                "cursor": self._cursor,
                "device_bytes": self.device_bytes,
                "generator": self.generator.get_state(),
            }

    def describe(self) -> str:
        """Return what the pool has read so far and how long the steps waited, for the line a run prints."""
        elapsed = max(time.monotonic() - self.started, 1e-9)
        per_reader = self.bytes_read / self.read_seconds / 1e9 if self.read_seconds else 0.0
        return (
            f"{self.documents_read}/{len(self.items)} documents read, {self.bytes_read / 1e9:.0f} GB at "
            f"{self.bytes_read / elapsed / 1e9:.2f} GB/s overall and {per_reader:.2f} GB/s a reader, "
            f"host {self.host_used / 1e9:.0f}/{self.host_bytes / 1e9:.0f} GB, steps waited {self.waited:.1f} s"
        )

    def close(self) -> None:
        """Stop reading and let go of what is held."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._readers.shutdown(wait=False, cancel_futures=True)

    def _read(self, index: int) -> Document:
        """Read one item onto the host - off the disk, or from the teacher - and count what it cost."""
        began = time.monotonic()
        item = self.items[index]
        if self.load is not None:
            document = self.load(item)
            size = document.resident_bytes
        else:
            document = Document(item.path, self.indexer, Cache.HOST, self.native)
            size = os.stat(item.path).st_size
        took = time.monotonic() - began
        with self._cond:
            self.documents_read += 1
            self.bytes_read += size
            self.read_seconds += took
        return document

    def _dispatch(self) -> None:
        """Submit the items to the readers in arrival order, as host memory allows."""
        try:
            for index, item in enumerate(self.items):
                with self._cond:
                    while not (self._stop or self._host_has_room(item)):
                        self._cond.wait()
                    if self._stop:
                        return
                    self.host_used += item.size
                future = self._readers.submit(self._read, index)
                with self._cond:
                    self._futures[index] = future
                    self._cond.notify_all()
        except BaseException as failure:
            self._fail(failure)

    def _stage(self) -> None:
        """Move the read documents onto the device in arrival order, as the device's budget allows."""
        try:
            for index, item in enumerate(self.items):
                with self._cond:
                    while not (self._stop or index in self._futures):
                        self._cond.wait()
                    if self._stop:
                        return
                    future = self._futures.pop(index)
                document = future.result()
                with self._cond:
                    while not (self._stop or self._device_has_room(item)):
                        self._cond.wait()
                    if self._stop:
                        return
                    self.device_used += item.size
                if self._copy_stream is not None:
                    with torch.cuda.stream(self._copy_stream):
                        staged = document.to(self.device)
                        if self.reader is not None:
                            staged = self.reader(staged)
                    self._copy_stream.synchronize()
                else:
                    staged = document if self.reader is None else self.reader(document)
                del document, future
                with self._cond:
                    self._ready[index] = staged
                    if self._copy_stream is not None:
                        self.host_used -= item.size
                    self._cond.notify_all()
        except BaseException as failure:
            self._fail(failure)

    def _host_has_room(self, item: Item) -> bool:
        """Return whether the host pool can take an item, which it always can when it is empty."""
        return self.host_used == 0 or self.host_used + item.size <= self.host_bytes

    def _device_has_room(self, item: Item) -> bool:
        """Return whether the device can take an item, which it always can when it is empty."""
        return self.device_used == 0 or self.device_used + item.size <= self.device_bytes

    def _fail(self, failure: BaseException) -> None:
        """Record why the pool stopped, and wake whatever waits on it."""
        with self._cond:
            self._error = failure
            self._cond.notify_all()


def host_budget(configured_gb: float | None, layers: int) -> float:
    """Return the host memory one layer's pool may fill.

    A job sized with `--mem-gb` says how much memory it has in `SLURM_MEM_PER_NODE`, in MB, and the pool takes
    a share of it, less what a report reads ahead.
    A config that names `pool_gb` overrides that, and a run outside slurm with neither gets a small default.

    Args:
        configured_gb: `data.pool_gb`, or None to read it off the job.
        layers: Layers the process trains, which share the pool.

    Returns:
        Bytes for one layer's pool.
    """
    if configured_gb is not None:
        total = configured_gb * 1e9
    elif "SLURM_MEM_PER_NODE" in os.environ:
        total = max(
            float(os.environ["SLURM_MEM_PER_NODE"]) * 2**20 * Pool.HOST_SHARE - Pool.REPORT_BYTES, 0.0
        )
    else:
        total = float(Pool.DEFAULT_BYTES)
    return total / layers


def device_budget(device: str | torch.device, configured_gb: float | None, layers: int) -> float:
    """Return the card memory one layer's active documents may occupy.

    Args:
        device: The training device.
        configured_gb: `data.resident_gb`, or None to work it out from what the device reports free.
        layers: Layers the process trains, which share the card.

    Returns:
        Bytes for one layer, unbounded off a card.
    """
    resolved = torch.device(device)
    if configured_gb is not None:
        return configured_gb * 1e9 / layers
    if resolved.type != "cuda":
        return float("inf")
    free, _ = torch.cuda.mem_get_info(resolved)
    return max(free - Pool.HEADROOM_BYTES, 0) / layers
