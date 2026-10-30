"""A retrieval probe - a fact spliced into a held-out sequence at a controlled distance before a query for it.

A mean held-out loss can hide an attention that cannot reach far, because most tokens are predictable from the
last few thousand before them. The probe asks one question instead: a sequence's own tokens are overwritten with
a needle `The secret code for <name> is <digits>.`, a few distractors with other names and codes, and at the end
the query `The secret code for <name> is`, and the model is scored on the digits. Nothing is generated - it is a
forward pass on a modified held-out sequence, so every arm runs it the same way.

The control is the same sequence with the needle replaced by filler of the same length, so it differs from the
probe at the needle's tokens alone. What the model gets right there is what its priors give, and the gain of the
probe over it is what the needle taught.
Everything here is `torch` and the standard library, and the tokenizer comes in as a function, so `make check` reaches it.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch

from lsa.training.native.constants import Retrieval

Encoder = Callable[[str], list[int]]

# Placement tries per distractor before the sequence is called too full
PLACEMENT_TRIES: int = 1000


@dataclass(frozen=True)
class Fact:
    """One name and the code the facts about it state."""

    name: str
    code: str


@dataclass(frozen=True)
class Probe:
    """One held-out sequence with its needle in, and the same sequence with filler in the needle's place."""

    # Input ids [seq_len] int64 of the probe and of the control
    tokens: torch.Tensor
    control: torch.Tensor
    # Position of the first digit, so the input at `answer_start - 1` predicts it
    answer_start: int
    # The digits' token ids [digits] int64
    answer: torch.Tensor
    needle_start: int
    needle_stop: int
    query_start: int
    distractor_starts: tuple[int, ...]

    @property
    def distance(self) -> int:
        """Return the tokens between the needle's end and the query's start."""
        return self.query_start - self.needle_stop

    @property
    def answer_inputs(self) -> torch.Tensor:
        """Return the positions [digits] whose outputs predict the digits."""
        return torch.arange(self.answer_start - 1, self.answer_start - 1 + self.answer.numel())


def draw_facts(seed: int, index: int, count: int) -> list[Fact]:
    """Draw a needle's fact and its distractors' from the seed, no two with one name.

    Args:
        seed: The evaluation seed.
        index: Which probe sequence, so every sequence has facts of its own.
        count: Facts to draw, the needle's first.

    Returns:
        The facts, the needle's first.
    """
    rng = random.Random(f"facts:{seed}:{index}")
    pairs = rng.sample(
        [(first, last) for first in Retrieval.FIRST_NAMES for last in Retrieval.LAST_NAMES], count
    )
    return [
        Fact(name=f"{first} {last}", code=f"{rng.randrange(10**Retrieval.DIGITS):0{Retrieval.DIGITS}d}")
        for first, last in pairs
    ]


def fits(distance: int, seq_len: int, encode: Encoder) -> bool:
    """Say whether a distance leaves room for the needle and the query in a sequence.

    Args:
        distance: Tokens between the needle's end and the query's start.
        seq_len: Tokens per sequence.
        encode: The tokenizer's encoding of a string.

    Returns:
        Whether the needle starts at or after the sequence's first token, whichever name it states.
    """
    longest = 0
    for first in Retrieval.FIRST_NAMES:
        for last in Retrieval.LAST_NAMES:
            needle, query, answer = _tokens_of(Fact(f"{first} {last}", "0" * Retrieval.DIGITS), encode)
            longest = max(longest, len(needle) + len(query) + len(answer))
    return distance + longest <= seq_len


class ProbeBuilder:
    """Splices facts into held-out sequences, deterministically from a seed."""

    def __init__(self, encode: Encoder, seed: int, distractors: int, seq_len: int) -> None:
        """Keep what every splice shares.

        Args:
            encode: The tokenizer's encoding of a string, without special tokens.
            seed: The evaluation seed, which the names, codes and positions follow.
            distractors: Facts about other names spliced in beside the needle.
            seq_len: Tokens per sequence, which a splice leaves unchanged.
        """
        self.encode = encode
        self.seed = seed
        self.distractors = distractors
        self.seq_len = seq_len
        self.filler = encode(Retrieval.FILLER)

    def build(self, tokens: torch.Tensor, index: int, distance: int) -> Probe:
        """Splice the facts into a sequence and make its control.

        Args:
            tokens: Input ids [seq_len] int64 of the held-out sequence.
            index: Which probe sequence, which fixes its facts.
            distance: Tokens between the needle's end and the query's start.

        Returns:
            The probe and its control.

        Raises:
            ValueError: The sequence is not `seq_len` long, the needle does not fit before the query at
                this distance, or the distractors do not fit beside it.
        """
        if tokens.numel() != self.seq_len:
            raise ValueError(f"a probe needs {self.seq_len} tokens - got {tokens.numel()}")
        facts = draw_facts(self.seed, index, 1 + self.distractors)
        needle, query, answer = _tokens_of(facts[0], self.encode)
        query_start = self.seq_len - len(query) - len(answer)
        needle_stop = query_start - distance
        needle_start = needle_stop - len(needle)
        if needle_start < 0:
            raise ValueError(f"distance {distance} leaves no room for the needle in {self.seq_len} tokens")

        control_needle = (self.filler * (len(needle) // len(self.filler) + 1))[: len(needle)]
        rng = random.Random(f"positions:{self.seed}:{index}:{distance}")
        taken = [(needle_start, needle_stop), (query_start, self.seq_len)]
        distractor_starts: list[int] = []
        spliced = tokens.clone()
        for fact in facts[1:]:
            distractor, _, _ = _tokens_of(fact, self.encode)
            start = _place(rng, len(distractor), query_start, taken)
            taken.append((start, start + len(distractor)))
            distractor_starts.append(start)
            spliced[start : start + len(distractor)] = torch.tensor(distractor)

        control = spliced.clone()
        control[needle_start:needle_stop] = torch.tensor(control_needle)
        spliced[needle_start:needle_stop] = torch.tensor(needle)
        for sequence in (spliced, control):
            sequence[query_start : query_start + len(query)] = torch.tensor(query)
            sequence[query_start + len(query) :] = torch.tensor(answer)
        return Probe(
            tokens=spliced,
            control=control,
            answer_start=query_start + len(query),
            answer=torch.tensor(answer),
            needle_start=needle_start,
            needle_stop=needle_stop,
            query_start=query_start,
            distractor_starts=tuple(distractor_starts),
        )


def _tokens_of(fact: Fact, encode: Encoder) -> tuple[list[int], list[int], list[int]]:
    """Tokenize a fact as the needle, as the query that asks for it, and as the answer.

    Args:
        fact: The name and code.
        encode: The tokenizer's encoding of a string.

    Returns:
        A tuple of:
            The statement `The secret code for <name> is <digits>.` token ids.
            The query up to and including the space before the digits.
            The digits.

    Raises:
        ValueError: The statement does not tokenize as the query followed by the digits, which would make the
            query a different prefix than the one the needle carries.
    """
    head = Retrieval.HEAD.format(name=fact.name)
    statement = encode(f"{head} {fact.code}.")
    query = encode(f"{head} ")
    answer = encode(fact.code)
    if statement[: len(query) + len(answer)] != query + answer:
        raise ValueError(f"{fact} tokenizes differently in a statement than as a query and its digits")
    return statement, query, answer


def _place(rng: random.Random, length: int, stop: int, taken: list[tuple[int, int]]) -> int:
    """Draw a start for a span of tokens that overlaps none already placed.

    Args:
        rng: The draw's source.
        length: Tokens in the span.
        stop: The span ends at or before this position.
        taken: Spans `(start, stop)` already placed.

    Returns:
        The span's first position.

    Raises:
        ValueError: No place was found.
    """
    for _ in range(PLACEMENT_TRIES):
        start = rng.randrange(0, stop - length + 1)
        if all(start + length <= other_start or other_stop <= start for other_start, other_stop in taken):
            return start
    raise ValueError(f"no room for {length} more tokens before position {stop}")
