"""Streams - one rank's endless, resumable supply of sequences, from one corpus or a weighted mixture of them.

Three layers, each knowing the one below only through what it yields.
An `ExampleStream` is one reader of one corpus: it deals the source's units out across every reader of every rank,
reads its own in an order a seed fixes, and hands out windows.
A `Mixture` draws each window from one of several corpora's readers, by weight.
A `BatchStream` takes a rank's readers in turn, a whole batch from each, which is how a data loader's workers
interleave and why `readers` is a setting of the run.

**Sharding.** With `world_size` ranks of `readers` readers each, reader `r` of rank `k` is global reader
`k * readers + r`. Every cycle the units are permuted by a generator seeded with the seed plus the cycle, and a
global reader takes every `world_size * readers`th unit from its own index, so no unit is read by two readers
in a cycle and every unit is read by one. A source with fewer units than readers is refused, since sharing
a unit would feed two readers the same tokens.

**Resuming.** A stream is a pure function of its seed and its layout, and its state is where it has got to - the
cycle, the unit, the record, the windows already taken from that record and the ids left over from the record
before - so a resumed run reads exactly what the interrupted one would have. A state saved under another seed
or layout is refused, because its position would mean something else.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from lsa import assets
from lsa.training.constants import Corpus, CorpusSplit
from lsa.training.data.corpus import batch_of
from lsa.training.data.sequencing import LanguageModelTask, TextEncoder
from lsa.training.data.sources import Source, open_source

# Ids of a carry are saved as bytes, which a checkpoint stores whole where it would need a tensor's shape to match
_CARRY_DTYPE = np.int32


class ExampleStream:
    """One reader of one corpus, as an endless stream of windows."""

    def __init__(self, source: Source, task: LanguageModelTask, reader: int, readers: int, seed: int) -> None:
        """Take this reader's share of the source.

        Args:
            source: Where the records are read from.
            task: How records become windows.
            reader: This reader's index among all `readers`.
            readers: Readers over every rank.
            seed: Seed of the unit order, which the cycle is added to.

        Raises:
            ValueError: The source holds fewer units than there are readers, or `reader` is outside them.
        """
        if not 0 <= reader < readers:
            raise ValueError(f"reader {reader} is outside {readers} readers")
        if source.units < readers:
            raise ValueError(
                f"{type(source).__name__} holds {source.units} units and cannot feed {readers} readers "
                "without duplicating data - lower the ranks or readers, or use a source with more units"
            )
        self.source = source
        self.task = task
        self.reader = reader
        self.readers = readers
        self.seed = seed
        self._reset()
        self._iterator = self._generate()

    def _reset(self) -> None:
        """Start at the first unit of the first cycle."""
        self.cycle = 0
        self.position = 0
        self.record = 0
        self.emitted = 0
        self.carry = np.empty(0, dtype=np.int64)

    def __iter__(self) -> "ExampleStream":
        """Return the stream, which is its own iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next window, ids [n] with `n` at most `seq_len + 1`."""
        return next(self._iterator)

    def _units(self) -> np.ndarray:
        """Return this cycle's units for this reader."""
        order = np.random.default_rng(self.seed + self.cycle).permutation(self.source.units)
        return order[self.reader :: self.readers]

    def _generate(self) -> Iterator[np.ndarray]:
        """Yield windows forever, keeping the position the state saves ahead of each one.

        Raises:
            ValueError: A whole pass over the units yielded no window.
        """
        while True:
            units = self._units()
            whole_pass = self.position == 0
            produced = 0
            while self.position < len(units):
                for record in self.source.read(int(units[self.position]), self.record):
                    windows, carry = self.task.cut(self.carry, self.task.documents(record))
                    while self.emitted < len(windows):
                        self.emitted += 1
                        produced += 1
                        yield windows[self.emitted - 1]
                    self.record += 1
                    self.emitted = 0
                    self.carry = carry
                # A unit's tail shorter than a window is dropped
                self.position += 1
                self.record = 0
                self.emitted = 0
                self.carry = np.empty(0, dtype=np.int64)
            if whole_pass and produced == 0:
                raise ValueError(
                    f"a whole pass over {type(self.source).__name__} yielded no sequence of {self.task.seq_len} tokens"
                )
            self.cycle += 1
            self.position = 0

    def state_dict(self) -> dict[str, Any]:
        """Return where the stream has got to, and what its order depends on."""
        return {
            "seed": self.seed,
            "reader": self.reader,
            "readers": self.readers,
            "cycle": self.cycle,
            "position": self.position,
            "record": self.record,
            "emitted": self.emitted,
            "carry": self.carry.astype(_CARRY_DTYPE).tobytes(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Seek to a saved position.

        Args:
            state: What `state_dict` returned.

        Raises:
            ValueError: The state was saved under another seed or layout.
        """
        for key, mine in (("seed", self.seed), ("reader", self.reader), ("readers", self.readers)):
            if state[key] != mine:
                raise ValueError(f"the stream was saved with {key} {state[key]} - this one has {mine}")
        self.cycle = int(state["cycle"])
        self.position = int(state["position"])
        self.record = int(state["record"])
        self.emitted = int(state["emitted"])
        self.carry = np.frombuffer(state["carry"], dtype=_CARRY_DTYPE).astype(np.int64)
        self._iterator = self._generate()


class Mixture:
    """Several corpora's readers, each window drawn from one of them by weight."""

    def __init__(self, children: list[ExampleStream], weights: list[float], seed: int, reader: int) -> None:
        """Draw from `children` in proportion to `weights`.

        Args:
            children: One reader of each corpus.
            weights: How often each is drawn, relative to the others.
            seed: Seed of the draws.
            reader: The global reader, so that readers do not draw one sequence of choices.

        Raises:
            ValueError: The counts differ or a weight is not positive.
        """
        if len(children) != len(weights) or not children:
            raise ValueError(
                f"a mixture needs one weight per child - got {len(weights)} weights for {len(children)}"
            )
        if min(weights) <= 0:
            raise ValueError(f"mixture weights must be positive - got {weights}")
        self.children = children
        self.probabilities = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.seed = seed
        self.reader = reader
        self.generator = np.random.default_rng([seed, reader])

    def __iter__(self) -> "Mixture":
        """Return the mixture, which is its own iterator."""
        return self

    def __next__(self) -> np.ndarray:
        """Return the next window of the child this draw picks."""
        if len(self.children) == 1:
            return next(self.children[0])
        return next(self.children[int(self.generator.choice(len(self.children), p=self.probabilities))])

    def state_dict(self) -> dict[str, Any]:
        """Return the draws' state and every child's."""
        return {
            "seed": self.seed,
            "reader": self.reader,
            "draws": self.generator.bit_generator.state,
            "children": [child.state_dict() for child in self.children],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Seek to a saved position.

        Args:
            state: What `state_dict` returned.

        Raises:
            ValueError: The state was saved under another seed or reader, or for another number of children.
        """
        if (state["seed"], state["reader"]) != (self.seed, self.reader) or len(state["children"]) != len(
            self.children
        ):
            raise ValueError("the mixture was saved under another seed, reader or set of children")
        self.generator.bit_generator.state = state["draws"]
        for child, saved in zip(self.children, state["children"], strict=True):
            child.load_state_dict(saved)


class BatchStream:
    """One rank's batches, a whole batch from each of its readers in turn."""

    def __init__(self, readers: list[Mixture], seq_len: int, sequences_per_batch: int) -> None:
        """Interleave this rank's readers.

        Args:
            readers: This rank's readers, in order.
            seq_len: Tokens per sequence.
            sequences_per_batch: Windows in a batch.
        """
        self.readers = readers
        self.seq_len = seq_len
        self.sequences_per_batch = sequences_per_batch
        self.batches = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield this rank's batches forever."""
        while True:
            reader = self.readers[self.batches % len(self.readers)]
            windows = [next(reader) for _ in range(self.sequences_per_batch)]
            self.batches += 1
            yield batch_of(windows, self.seq_len)

    def state_dict(self) -> dict[str, Any]:
        """Return the batches taken and every reader's position."""
        return {"batches": self.batches, "readers": [reader.state_dict() for reader in self.readers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the stream from `state_dict`.

        Args:
            state: What `state_dict` returned.

        Raises:
            ValueError: The state was saved for another number of readers.
        """
        if len(state["readers"]) != len(self.readers):
            raise ValueError(
                f"the stream was saved with {len(state['readers'])} readers - this one has {len(self.readers)}"
            )
        self.batches = int(state["batches"])
        for reader, saved in zip(self.readers, state["readers"], strict=True):
            reader.load_state_dict(saved)


@dataclass(frozen=True)
class Layout:
    """What makes one rank's stream: the ranks and readers it shares the corpora with, the sequence and the seed."""

    rank: int
    world_size: int
    readers: int
    seq_len: int
    tokens_per_batch: int
    seed: int = 0


def open_stream(
    corpora: list[tuple[assets.SourceCard | assets.TokenizedCard | assets.ShardCard, float]],
    layout: Layout,
    *,
    tokenizer_path: str,
    packing: str | None = None,
    split: str = CorpusSplit.TRAIN,
    sources: list[str] | None = None,
    held_out_fraction: float = Corpus.HELD_OUT_FRACTION,
) -> BatchStream:
    """Open one rank's batches over weighted corpora.

    Args:
        corpora: Each corpus's card and its weight in the mixture.
        layout: This rank's place among the others and the shape of a batch.
        tokenizer_path: The snapshot that tokenizes any corpus that holds text.
        packing: One of `Packing.ALL`, or None for each corpus's own first choice.
        split: Which side of the split every corpus is read from.
        sources: Sources of a tokenized corpus to read.
        held_out_fraction: Share of documents held out.

    Returns:
        The batches, endless.

    Raises:
        ValueError: The rank is outside the world, a batch is not whole sequences, or a corpus cannot feed the readers.
    """
    if not 0 <= layout.rank < layout.world_size:
        raise ValueError(f"rank {layout.rank} is outside a world of {layout.world_size}")
    if layout.tokens_per_batch % layout.seq_len:
        raise ValueError(
            f"seq_len must divide tokens_per_batch - {layout.seq_len} vs {layout.tokens_per_batch}"
        )
    built = []
    for card, weight in corpora:
        source = open_source(
            card, seq_len=layout.seq_len, split=split, sources=sources, held_out_fraction=held_out_fraction
        )
        encoder = TextEncoder(tokenizer_path) if isinstance(card, assets.SourceCard) else None
        built.append((source, LanguageModelTask(source, layout.seq_len, packing, encoder), weight))
    total = layout.world_size * layout.readers
    mixtures = []
    for reader in range(layout.readers):
        index = layout.rank * layout.readers + reader
        children = [ExampleStream(source, task, index, total, layout.seed) for source, task, _ in built]
        mixtures.append(Mixture(children, [weight for _, _, weight in built], layout.seed, index))
    return BatchStream(mixtures, layout.seq_len, layout.tokens_per_batch // layout.seq_len)
