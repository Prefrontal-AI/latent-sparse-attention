"""What a held-out pass measures - next-token loss per position bucket and per source - and what it logs of the attention.

Cross entropy is the only loss a native run trains on, and the trainer owns it. What lives here is how a
held-out pass reads it: a mean loss is a poor comparison between attentions on its own, because most of a long
sequence is predictable from its last few thousand tokens, which every arm reads in full. Where the arms differ is
the far end, so the loss is scored by position in buckets whose first edges are the open block and the fine
read's key budget - the buckets below the budget are a check that should agree across arms, and the ones above
it are where an attention that reads less may lose.

Scores are kept per sequence, so two runs on one seed pair sequence by sequence, and summed within a bucket,
so any subset's loss is a loss per token.
The retrieval probe is scored here too, by the loss and the exact match of its answer tokens alone.
Everything here is `torch` alone, so `make check` reaches it.
"""

from dataclasses import dataclass

import torch

from lsa.training.constants import Loss


@dataclass(frozen=True)
class SequenceScore:
    """One held-out sequence's loss, summed within each position bucket."""

    source: str
    document: str
    row: int
    slice_index: int
    # Summed token loss and token count, one entry per bucket
    loss_sums: list[float]
    token_counts: list[int]


@dataclass(frozen=True)
class RetrievalScore:
    """One probe sequence's answer, scored with the needle in and with filler in its place."""

    source: str
    document: str
    row: int
    slice_index: int
    distance: int
    answer_tokens: int
    probe_loss_sum: float
    control_loss_sum: float
    # Every answer token was the argmax, teacher-forced
    probe_exact: bool
    control_exact: bool


def bucket_of_position(positions: torch.Tensor, edges: tuple[int, ...]) -> torch.Tensor:
    """Assign each position to the bucket its edges put it in.

    Args:
        positions: Positions [n] int64 within their sequence.
        edges: Increasing bucket starts after the first bucket, which starts at 0.

    Returns:
        Bucket indices [n] int64 in `[0, len(edges)]`.
    """
    return torch.bucketize(positions, torch.tensor(edges, dtype=positions.dtype), right=True)


def score_batch(
    token_losses: torch.Tensor,
    labels: torch.Tensor,
    seq_len: int,
    edges: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum a batch's token losses per sequence and per position bucket.

    Args:
        token_losses: Per-token loss [n] fp32, over sequences laid end to end.
        labels: Labels [n] int64, `Loss.IGNORE_INDEX` where no token is predicted.
        seq_len: Tokens per sequence, which divides `n`.
        edges: Bucket edges, as `bucket_of_position` takes them.

    Returns:
        A tuple of:
            Summed loss [sequences, buckets] fp64.
            Counted tokens [sequences, buckets] int64.
    """
    bucket_count = len(edges) + 1
    sequence_count = labels.numel() // seq_len
    positions = torch.arange(labels.numel()) % seq_len
    sequence_of_token = torch.arange(labels.numel()) // seq_len
    # [n] flat (sequence, bucket) cell of every token
    cell = sequence_of_token * bucket_count + bucket_of_position(positions, edges)
    counted = labels.cpu() != Loss.IGNORE_INDEX

    loss_sums = torch.zeros(sequence_count * bucket_count, dtype=torch.float64)
    token_counts = torch.zeros(sequence_count * bucket_count, dtype=torch.long)
    loss_sums.index_add_(0, cell[counted], token_losses.detach().cpu().double()[counted])
    token_counts.index_add_(0, cell[counted], torch.ones(int(counted.sum()), dtype=torch.long))
    return loss_sums.view(sequence_count, bucket_count), token_counts.view(sequence_count, bucket_count)


def bucket_names(edges: tuple[int, ...], seq_len: int) -> list[str]:
    """Name each position bucket by the span it covers.

    Args:
        edges: Bucket edges.
        seq_len: Tokens per sequence, which closes the last bucket.

    Returns:
        One `"<start>_<stop>"` name per bucket.
    """
    starts = (0, *edges)
    stops = (*edges, seq_len)
    return [f"{start}_{stop}" for start, stop in zip(starts, stops, strict=True)]


def summarise(scores: list[SequenceScore], edges: tuple[int, ...], seq_len: int) -> dict[str, float]:
    """Reduce a pass's sequence scores to the scalars a run logs.

    Every mean is over tokens, so a subset's loss is the loss a token in it costs.

    Args:
        scores: Every sequence the pass read.
        edges: Bucket edges.
        seq_len: Tokens per sequence.

    Returns:
        The mean loss overall, per position bucket and per source.
    """
    loss_sums = torch.tensor([score.loss_sums for score in scores], dtype=torch.float64)
    token_counts = torch.tensor([score.token_counts for score in scores], dtype=torch.float64)

    def mean(rows: list[int], buckets: slice = slice(None)) -> float:
        taken = token_counts[rows, buckets].sum()
        return float(loss_sums[rows, buckets].sum() / taken) if taken else float("nan")

    everything = list(range(len(scores)))
    metrics = {"loss": mean(everything)}
    for bucket, name in enumerate(bucket_names(edges, seq_len)):
        metrics[f"position/{name}"] = mean(everything, slice(bucket, bucket + 1))
    for source in sorted({score.source for score in scores}):
        metrics[f"source/{source}"] = mean([i for i, score in enumerate(scores) if score.source == source])
    return metrics


def score_answers(
    token_losses: torch.Tensor, predictions: torch.Tensor, answers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each sequence's answer tokens by their summed loss and whether the model got every one.

    Args:
        token_losses: Loss [sequences, answer_tokens] fp32 at each answer token.
        predictions: Argmax [sequences, answer_tokens] int64 at the same positions.
        answers: The true tokens [sequences, answer_tokens] int64.

    Returns:
        A tuple of:
            Summed loss [sequences] fp64.
            Whether every token was predicted [sequences] bool.
    """
    return token_losses.double().sum(dim=1), (predictions == answers).all(dim=1)


def summarise_retrieval(scores: list[RetrievalScore]) -> dict[str, float]:
    """Reduce a pass's probe scores to the scalars a run logs, per distance.

    The loss is a mean over answer tokens, the exact match a share of sequences, and the gain what the control
    costs above the probe, so a positive gain is the needle helping.

    Args:
        scores: Every probe sequence the pass read, at every distance.

    Returns:
        `retrieval/d<distance>/` then `loss`, `control_loss`, `exact`, `control_exact` and `gain`.
    """
    metrics: dict[str, float] = {}
    for distance in sorted({score.distance for score in scores}):
        rows = [score for score in scores if score.distance == distance]
        tokens = sum(score.answer_tokens for score in rows)
        loss = sum(score.probe_loss_sum for score in rows) / tokens
        control_loss = sum(score.control_loss_sum for score in rows) / tokens
        prefix = f"retrieval/d{distance}"
        metrics[f"{prefix}/loss"] = loss
        metrics[f"{prefix}/control_loss"] = control_loss
        metrics[f"{prefix}/exact"] = sum(score.probe_exact for score in rows) / len(rows)
        metrics[f"{prefix}/control_exact"] = sum(score.control_exact for score in rows) / len(rows)
        metrics[f"{prefix}/gain"] = control_loss - loss
    return metrics
