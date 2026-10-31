"""Run a native checkpoint through lm-eval in process, whatever attention it was trained under.

Mirrors the `ReferenceLM` wrapper of the reference training code this training code descends from, so a score here
means what it means there and for a HuggingFace model: a context and its continuation are encoded as lm-eval's HFLM
encodes them, left-truncated to the last `max_length + 1` tokens, and rolling windows score every token once.
The only prefix token is the end-of-text token, which is what separates documents in the training corpus.

Three things differ from the source, and all three come from the model.

**No padding mask.** The native attentions read a sequence as the model's own blocks, so a batch is right-padded and
the pad sits after every scored token - causal attention and the recurrent layers never read it back. A sparse arm
needs a length that its block divides, so every batch is padded up to the next power of two from 512.
That is also why generation batches prompts of any length: a row's next token is written into its own pad slot.

**No decode cache.** Generation recomputes the whole prefix at every step, for every arm. That is exact, and its
cost is one forward over the batch per generated token, which is why `max_gen_toks` can be capped.

**Hidden states, not logits.** A 16,384-token row of 248k-way logits is 16 GB in fp32, so the model returns hidden
states and the output projection runs on the rows that are scored, a chunk at a time.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lm_eval import utils as lm_eval_utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.utils import handle_stop_sequences

logger = logging.getLogger(__name__)

# The shortest padded length, which is the sparse arms' block
MIN_PADDED = 512
# Rows of hidden states projected to the vocabulary at once
PROJECTION_CHUNK = 2048
# Tokens one forward holds, a batch being as many rows as fit
BATCH_TOKENS = 16384
# What lm-eval's HFLM generates when a task does not say
DEFAULT_GEN_TOKS = 256


@dataclass
class Backbone:
    """The two halves of a language model, split so the output projection can run on a few rows.

    Attributes:
        hidden: Tokens [rows, length] int64 to final-norm hidden states [rows, length, model_dim].
        project: Hidden states [n, model_dim] to logits [n, vocab].
        device: Where the tokens go.
    """

    hidden: Callable[[torch.Tensor], torch.Tensor]
    project: Callable[[torch.Tensor], torch.Tensor]
    device: torch.device


@dataclass
class Tokenizer:
    """The three things lm-eval needs of a tokenizer.

    Attributes:
        encode: Text to token ids, with no special tokens added.
        decode: Token ids to text.
        end_id: The end-of-text id, which opens a context that has none and stops a generation.
    """

    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    end_id: int


def padded_length(length: int) -> int:
    """Return the length a batch is padded to - the next power of two, from `MIN_PADDED` up.

    Args:
        length: The longest row of the batch.

    Returns:
        A length every sparse arm's block divides.
    """
    padded = MIN_PADDED
    while padded < length:
        padded *= 2
    return padded


def encode_pair(
    encode: Callable[[str], list[int]], context: str, continuation: str
) -> tuple[list[int], list[int]]:
    """Encode a context and continuation as HFLM does - trailing context whitespace joins the continuation.

    The continuation's ids are those of context plus continuation past the context's own, so a word split across
    the boundary tokenizes as it would in running text.

    Args:
        encode: Text to token ids.
        context: The text conditioned on.
        continuation: The text scored.

    Returns:
        The context ids and the continuation ids.
    """
    spaces = len(context) - len(context.rstrip())
    if spaces > 0:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    whole = encode(context + continuation)
    context_ids = encode(context)
    return context_ids, whole[len(context_ids) :]


def encode_request(
    encode: Callable[[str], list[int]], end_id: int, context: str, continuation: str
) -> tuple[list[int], list[int]]:
    """Encode one loglikelihood request, conditioning an empty context on the end-of-text token.

    Args:
        encode: Text to token ids.
        end_id: The end-of-text id.
        context: The text conditioned on, possibly empty.
        continuation: The text scored.

    Returns:
        The context ids, never empty, and the continuation ids.
    """
    if context.strip():
        context_ids, continuation_ids = encode_pair(encode, context, continuation)
        if context_ids:
            return context_ids, continuation_ids
    continuation_ids = encode(context + continuation)
    # The text may open with the token itself
    if continuation_ids and continuation_ids[0] == end_id:
        return continuation_ids[:1], continuation_ids[1:]
    return [end_id], continuation_ids


def score_window(context_ids: list[int], continuation_ids: list[int], max_length: int) -> list[int]:
    """Return the tokens of a pair the model reads, left-truncated to its last `max_length + 1`.

    Args:
        context_ids: The context ids.
        continuation_ids: The continuation ids.
        max_length: The longest context fed to the model.

    Returns:
        The window, whose last token is the last one scored and is not an input.

    Raises:
        ValueError: The continuation alone is longer than the model can read.
    """
    if not 0 < len(continuation_ids) <= max_length:
        raise ValueError(
            f"a continuation of {len(continuation_ids)} tokens cannot be scored within max_length {max_length}"
        )
    return (context_ids + continuation_ids)[-(max_length + 1) :]


def pack_rows(sequences: list[list[int]], length: int, rows: int, pad_id: int) -> torch.Tensor:
    """Right-pad sequences into a batch of a fixed shape.

    Args:
        sequences: The token ids of each real row, each no longer than `length`.
        length: The padded length.
        rows: The batch's rows, at least as many as there are sequences. The rest are pad.
        pad_id: What fills the padding.

    Returns:
        The tokens [rows, length] int64.
    """
    tokens = torch.full((rows, length), pad_id, dtype=torch.long)
    for i, ids in enumerate(sequences):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens


def batches(lengths: list[int], batch_tokens: int) -> list[tuple[list[int], int, int]]:
    """Group requests into batches of one padded length, longest first.

    Args:
        lengths: The tokens each request needs.
        batch_tokens: The tokens one forward holds.

    Returns:
        One tuple per batch: the request indices, the padded length and the rows the batch is shaped to.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    grouped: dict[int, list[int]] = {}
    for i in order:
        grouped.setdefault(padded_length(lengths[i]), []).append(i)
    out: list[tuple[list[int], int, int]] = []
    for length, members in grouped.items():
        rows = max(1, batch_tokens // length)
        out.extend((members[start : start + rows], length, rows) for start in range(0, len(members), rows))
    return out


def score_continuation(
    project: Callable[[torch.Tensor], torch.Tensor], hidden: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a continuation a chunk of rows at a time.

    Args:
        project: Hidden states [n, model_dim] to logits [n, vocab].
        hidden: The hidden states [n, model_dim] that predict the continuation's tokens.
        targets: The continuation's token ids [n] int64.

    Returns:
        A tuple of:
            The summed log-probability of the continuation, a scalar fp32.
            Whether greedy decoding reproduces every token, a scalar bool.
    """
    log_likelihood = hidden.new_zeros((), dtype=torch.float32)
    greedy = torch.ones((), dtype=torch.bool, device=hidden.device)
    for rows, chunk_targets in zip(
        hidden.split(PROJECTION_CHUNK), targets.split(PROJECTION_CHUNK), strict=True
    ):
        logits = project(rows).float()
        log_probs = F.log_softmax(logits, dim=-1)
        log_likelihood = log_likelihood + log_probs.gather(-1, chunk_targets[:, None]).sum()
        greedy = greedy & (logits.argmax(-1) == chunk_targets).all()
    return log_likelihood, greedy


def stopped(tokenizer: Tokenizer, generated: list[int], until: tuple[str, ...]) -> bool:
    """Say whether a generation has produced the end token or a stop string.

    Only a tail is decoded, long enough to hold any stop string, so a step costs the same however long the
    generation already is.

    Args:
        tokenizer: The tokenizer.
        generated: The tokens generated so far.
        until: The stop strings.

    Returns:
        Whether to stop.
    """
    if generated and generated[-1] == tokenizer.end_id:
        return True
    tail = 4 * max(map(len, until), default=0) + 4
    decoded = tokenizer.decode(generated[-tail:])
    return any(stop in decoded for stop in until)


def finish(tokenizer: Tokenizer, generated: list[int], until: tuple[str, ...]) -> str:
    """Return the text before the end token and before the first stop string.

    Args:
        tokenizer: The tokenizer.
        generated: The tokens generated.
        until: The stop strings.

    Returns:
        The reply.
    """
    if tokenizer.end_id in generated:
        generated = generated[: generated.index(tokenizer.end_id)]
    text = tokenizer.decode(generated)
    for stop in until:
        text = text.split(stop)[0]
    return text


class NativeLM(LM):
    """lm-eval's view of a native checkpoint.

    Data-parallel over requests: launched with one process per GPU, each holding a full replica, lm-eval's evaluator
    hands every rank its share of a task's requests and gathers the results through the collectives below.
    """

    def __init__(
        self,
        backbone: Backbone,
        tokenizer: Tokenizer,
        max_length: int,
        batch_tokens: int = BATCH_TOKENS,
        max_gen_toks: int | None = None,
        process_group: Any = None,
    ) -> None:
        """Wrap a backbone for lm-eval.

        Args:
            backbone: The model, split at its output projection.
            tokenizer: The tokenizer the model was trained with.
            max_length: The longest context fed to the model.
            batch_tokens: The tokens one forward holds.
            max_gen_toks: A cap on the tokens any task generates, or None for the task's own.
            process_group: A CPU (gloo) group over the ranks for the collectives, or None to run on one process.
        """
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._batch_tokens = batch_tokens
        self._gen_cap = max_gen_toks
        self._group = process_group
        if process_group is not None:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        # Seconds spent in each request type on this rank
        self.seconds: dict[str, float] = {}
        self._born = time.perf_counter()

    @property
    def eot_token_id(self) -> int:
        """Return the end-of-text id."""
        return self.tokenizer.end_id

    @property
    def max_length(self) -> int:
        """Return the longest context fed to the model."""
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Return the tokens a task generates when it does not say."""
        return DEFAULT_GEN_TOKS if self._gen_cap is None else self._gen_cap

    @property
    def device(self) -> torch.device:
        """Return where the tokens go."""
        return self.backbone.device

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor from every rank, through the CPU group.

        Args:
            tensor: This rank's tensor, one dim.

        Returns:
            Every rank's, stacked.
        """
        if self._group is None:
            return tensor
        parts = [torch.zeros_like(tensor.cpu()) for _ in range(self._world_size)]
        dist.all_gather(parts, tensor.cpu(), group=self._group)
        return torch.stack(parts).to(tensor.device)

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        """Gather a python object from every rank to one.

        Args:
            obj: This rank's object.
            dst: The rank that receives them.

        Returns:
            Every rank's object on `dst`, None elsewhere.
        """
        if self._group is None:
            return [obj]
        gathered: list[Any] | None = [None] * self._world_size if self._rank == dst else None
        dist.gather_object(obj, gathered, dst=dst, group=self._group)
        return gathered

    def barrier(self) -> None:
        """Wait for every rank."""
        if self._group is not None:
            dist.barrier(group=self._group)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Score each request's continuation given its context.

        Args:
            requests: The requests, each carrying (context, continuation).

        Returns:
            One (log-probability, is-greedy) per request.
        """
        pairs = [
            encode_request(self.tokenizer.encode, self.eot_token_id, *request.args) for request in requests
        ]
        return self._timed("loglikelihood", self._score_pairs, pairs)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Score each request's whole text, through windows that give every token one prediction.

        Args:
            requests: The requests, each carrying the text.

        Returns:
            The log-probability of each text.
        """
        owners: list[int] = []
        pairs: list[tuple[list[int], list[int]]] = []
        for i, request in enumerate(requests):
            windows = lm_eval_utils.get_rolling_token_windows(
                token_list=self.tokenizer.encode(request.args[0]),
                prefix_token=self.eot_token_id,
                max_seq_len=self._max_length,
                context_len=1,
            )
            for pair in map(lm_eval_utils.make_disjoint_window, windows):
                owners.append(i)
                pairs.append(pair)
        scored = self._timed("loglikelihood_rolling", self._score_pairs, pairs)
        totals = [0.0] * len(requests)
        for owner, (log_likelihood, _) in zip(owners, scored, strict=True):
            totals[owner] += log_likelihood
        return totals

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate a greedy reply to each request, by recomputing the whole prefix at every token.

        Args:
            requests: The requests, each carrying (context, generation kwargs).

        Returns:
            One reply per request.
        """
        return self._timed("generate_until", self._generate, requests)

    def _report_progress(self, name: str, done: int, total: int) -> None:
        """Print how far this rank is through a request type, on rank 0 and about every tenth of the way."""
        if self._rank == 0 and (done == total or done % max(1, total // 10) == 0):
            print(f"{name}: batch {done}/{total}, {self.seconds_now():.0f} s", flush=True)

    def seconds_now(self) -> float:
        """Return the seconds since this model was built."""
        return time.perf_counter() - self._born

    def _timed(self, name: str, work: Callable[..., Any], *arguments: Any) -> Any:
        """Run one request type and add its seconds to the total."""
        start = time.perf_counter()
        result = work(*arguments)
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start
        return result

    @torch.no_grad()
    def _score_pairs(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """Score every pair, a batch of one padded length at a time.

        Args:
            pairs: The (context ids, continuation ids) of each request.

        Returns:
            One (log-probability, is-greedy) per pair, in order.
        """
        windows = [score_window(context, continuation, self._max_length) for context, continuation in pairs]
        results: list[tuple[float, bool]] = [(0.0, False)] * len(pairs)
        plan = batches([len(window) - 1 for window in windows], self._batch_tokens)
        for done, (members, length, rows) in enumerate(plan, 1):
            self._report_progress("score", done, len(plan))
            tokens = pack_rows([windows[i][:-1] for i in members], length, rows, self.eot_token_id)
            hidden = self.backbone.hidden(tokens.to(self.device))
            log_likelihoods, greedy = [], []
            for row, i in enumerate(members):
                count = len(pairs[i][1])
                end = len(windows[i]) - 1
                targets = torch.tensor(pairs[i][1], dtype=torch.long, device=self.device)
                log_likelihood, is_greedy = score_continuation(
                    self.backbone.project, hidden[row, end - count : end], targets
                )
                log_likelihoods.append(log_likelihood)
                greedy.append(is_greedy)
            for i, log_likelihood, is_greedy in zip(
                members, torch.stack(log_likelihoods).tolist(), torch.stack(greedy).tolist(), strict=True
            ):
                results[i] = (log_likelihood, is_greedy)
        return results

    @torch.no_grad()
    def _generate(self, requests: list[Instance]) -> list[str]:
        """Generate for every request, requests of one stop list and budget together.

        Args:
            requests: The requests.

        Returns:
            One reply per request, in order.
        """
        end_text = self.tokenizer.decode([self.eot_token_id])
        prompts: list[list[int]] = []
        settings: list[tuple[tuple[str, ...], int]] = []
        for request in requests:
            context, kwargs = request.args
            stop = kwargs.get("until")
            until = tuple(handle_stop_sequences(list(stop) if isinstance(stop, list) else stop, end_text))
            budget = kwargs.get("max_gen_toks", self.max_gen_toks)
            if self._gen_cap is not None:
                budget = min(budget, self._gen_cap)
            # Room for every new token inside max_length, as HFLM leaves
            prompts.append(
                self.tokenizer.encode(context)[-max(1, self._max_length - budget) :] or [self.eot_token_id]
            )
            settings.append((until, budget))

        replies = [""] * len(requests)
        grouped: dict[tuple[tuple[str, ...], int], list[int]] = {}
        for i, setting in enumerate(settings):
            grouped.setdefault(setting, []).append(i)
        for (until, budget), members in grouped.items():
            lengths = [len(prompts[i]) + budget for i in members]
            plan = batches(lengths, self._batch_tokens)
            for done, (batch, _, rows) in enumerate(plan, 1):
                self._report_progress("generate", done, len(plan))
                chosen = [members[j] for j in batch]
                generated = self._generate_batch([prompts[i] for i in chosen], rows, budget, until)
                for i, ids in zip(chosen, generated, strict=True):
                    replies[i] = finish(self.tokenizer, ids, until)
        return replies

    def _generate_batch(
        self, prompts: list[list[int]], rows: int, budget: int, until: tuple[str, ...]
    ) -> list[list[int]]:
        """Greedily extend a batch of prompts, each step a forward over everything so far.

        A row that has stopped keeps being generated, which holds the batch's shape, and its reply is cut afterwards.
        A row's next token is written into its own pad slot, so prompts of any length share a batch.

        Args:
            prompts: The prompt ids of each real row.
            rows: The batch's rows, at least as many as there are prompts.
            budget: The tokens to generate at most.
            until: The stop strings.

        Returns:
            The generated ids of each real row.
        """
        lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
        # Wide enough that every step's slice is a padded length
        buffer = pack_rows(prompts, padded_length(int(lengths.max()) + budget), rows, self.eot_token_id).to(
            self.device
        )
        current = torch.zeros(rows, dtype=torch.long)
        current[: len(prompts)] = lengths
        row_index = torch.arange(rows, device=self.device)
        generated: list[list[int]] = [[] for _ in prompts]
        done = [False] * len(prompts)
        for _ in range(budget):
            width = padded_length(int(current.max()))
            hidden = self.backbone.hidden(buffer[:, :width])
            last = hidden[row_index, (current - 1).clamp(min=0).to(self.device)]
            next_ids = self.backbone.project(last).argmax(-1)
            buffer[row_index, current.to(self.device)] = next_ids
            current += 1
            for i, token in enumerate(next_ids[: len(prompts)].tolist()):
                if not done[i]:
                    generated[i].append(token)
                    done[i] = stopped(self.tokenizer, generated[i], until)
            if all(done):
                break
        return generated
