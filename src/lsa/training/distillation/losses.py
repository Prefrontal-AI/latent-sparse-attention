"""The objective - a per-key divergence to the teacher's own attention, and the terms a step may add to it.

The per-key divergence reads no bucket.

    z[i, j] = q_i(idx) . k_j(idx) / tau   the index score, as logits over the keys query `i` may read
    L       = KL( p_i || softmax_j z )     forward, mode-covering, with `p_i` the teacher's renormalised mass

Forward KL is mode-covering, which is what a selector wants: paying for missing mass is the whole point, and
paying for spending score on a key the teacher ignores is not.
That argument is weaker once the read is bucketed, because score spent on a key the teacher ignores costs a
whole placement, so the symmetric form is here to be measured rather than argued about.
Reverse KL is not, because the target is zero almost everywhere over a document and the reverse direction is
infinite there; the Jensen-Shannon mixture is the finite way to ask the same question.

A QSA indexer is read over blocks instead, as QSA's own distillation reads it - the KL from the teacher's attention
max-pooled to blocks and renormalised, to the softmax of the block scores over the blocks a query has closed.

    I[i, b] = sum_h relu( q_i^h . k_b ) / sqrt(head_dim)   the block score, as logits over the closed blocks
    L       = KL( max-pooled p_i || softmax_b I / tau )

`depth` truncates the target to a prefix of the teacher's own ranking.
A selector is scored on its top 16 and its top 64, and the tail of a 256-key target is mass it never has to
rank correctly - so whether paying for that tail helps or hurts is a question the loss can be asked directly.
The truncated target drops the tail and renormalises what the kept keys hold.
The rest bucket keeps it as one more entry, which the student answers with its own mass on every visible key
outside the kept set, so a map is paid for the share the kept keys hold and not for ranking inside the tail.

    L = - sum_{j in S} p(j) log softmax_j z - (1 - rho) log (1 - sum_{j in S} softmax_j z)

with `p(j)` a kept key's share of the teacher's whole attention and `rho` the sum of those shares.
"""

from typing import TYPE_CHECKING

import torch

from lsa import constants as const
from lsa.baselines import qsa
from lsa.training.distillation.constants import Calibration, Divergence, Tail, Target, Term
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.model import IndexMap, QSAIndexer

# The config module reaches omegaconf, which the objective does not need to import
if TYPE_CHECKING:
    from lsa.training.distillation.config import ObjectiveConfig


def key_scores(index_map: IndexMap, document: Document, rows: torch.Tensor) -> torch.Tensor:
    """Score every key of a document against the given query rows, in the index space.

    Args:
        index_map: The layer's learned maps.
        document: The cached document.
        rows: Query rows [R] int64.

    Returns:
        Index scores [R, T] fp32.
    """
    # [T, index_dim] and [R, index_dim] fp32 unit rows
    index_keys = index_map.index_keys(document.key_inputs())
    index_queries = index_map.index_queries(document.query_inputs(rows))
    return index_queries @ index_keys.t()


def qsa_block_scores(indexer: QSAIndexer, document: Document, rows: torch.Tensor) -> torch.Tensor:
    """Score every closed block of a document against the given query rows, as QSA's indexer scores them.

    Args:
        indexer: The layer's QSA indexer.
        document: The document, read for a QSA indexer.
        rows: Query rows [R] int64.

    Returns:
        Block scores [R, T // block_size] fp32, causality not applied.
    """
    rows = rows.to(document.device)
    index_keys = indexer.index_keys(document.key_inputs())
    index_queries = indexer.index_queries(document.query_inputs(rows), document.query_positions[rows])
    return qsa.block_scores(index_queries, index_keys)


def block_divergence(
    scores: torch.Tensor, document: Document, rows: torch.Tensor, depth: int | None = None
) -> torch.Tensor:
    """Return the forward KL from the teacher's block-pooled attention to block scores already read at the temperature.

    Constant in the teacher's own entropy, as `key_divergence` is.

    Args:
        scores: Block logits [R, B] fp32, already divided by the temperature.
        document: The document, read for a QSA indexer.
        rows: Query rows [R] int64.
        depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.

    Returns:
        The mean loss over the rows, a scalar.
    """
    selectable = document.visible_blocks(rows)
    log_predicted = torch.log_softmax(scores.masked_fill(~selectable, const.Math.NINF), dim=-1)
    target_mass = document.block_target(rows, depth)
    return -(target_mass * log_predicted.masked_fill(~selectable, 0.0)).sum(-1).mean()


def key_divergence(
    scores: torch.Tensor,
    document: Document,
    rows: torch.Tensor,
    depth: int | None = None,
    divergence: str = Divergence.FORWARD,
    target: str = Target.DENSE,
    tail: str = Tail.TRUNCATE,
) -> torch.Tensor:
    """Return the divergence from the teacher's attention to index scores already read at the temperature.

    Under the forward KL this is constant in the teacher's own entropy, so it is a cross-entropy in
    everything but name and the number is not comparable across documents - nor across divergences, nor
    across depths, because each renormalises a different target.

    Args:
        scores: Index logits [R, T] fp32, already divided by the temperature.
        document: The cached document.
        rows: Query rows [R] int64.
        depth: Keep only this many of the teacher's strongest keys, or None to keep every cached one.
        divergence: One of `Divergence.BOTH`.
        target: What the index score is measured against, one of `Target.ALL`.
        tail: What the depth does with the mass beyond it, one of `Tail.BOTH`.

    Returns:
        The mean loss over the rows, a scalar.

    Raises:
        ValueError: `divergence` is not one the objective knows.
        ValueError: `tail` is not one the objective knows, or is the rest bucket under the symmetric divergence or without a depth.
    """
    if divergence not in Divergence.BOTH:
        raise ValueError(f"divergence must be one of {Divergence.BOTH} - got {divergence!r}")
    if tail not in Tail.BOTH:
        raise ValueError(f"tail must be one of {Tail.BOTH} - got {tail!r}")

    readable = document.visible_keys(rows)
    log_predicted = torch.log_softmax(scores.masked_fill(~readable, const.Math.NINF), dim=-1)
    if tail == Tail.REST_BUCKET:
        if divergence != Divergence.FORWARD or depth is None:
            raise ValueError(
                f"the rest bucket is the forward cross-entropy beyond a depth - got {divergence!r} at depth {depth}"
            )
        share, rest = rest_bucket_target(document, rows, depth, target)
        return rest_bucket_cross_entropy(log_predicted, readable, share, rest)
    target_mass = document.key_target(rows, depth, target)

    if divergence == Divergence.FORWARD:
        return -(target_mass * log_predicted.masked_fill(~readable, 0.0)).sum(-1).mean()

    # The mixture is what makes the symmetric form finite where the target is zero, which over a document
    # is almost everywhere
    predicted = log_predicted.exp()
    log_mixture = (0.5 * (target_mass + predicted)).clamp_min(Divergence.EPS).log()
    # log_softmax is -inf off the readable set and `predicted` is exactly zero there
    # => the product is a nan rather than the zero it stands for, so the log is floored first
    finite_log_predicted = log_predicted.masked_fill(~readable, 0.0)
    forward = (target_mass * (target_mass.clamp_min(Divergence.EPS).log() - log_mixture)).sum(-1)
    reverse = (predicted * (finite_log_predicted - log_mixture)).sum(-1)
    return (0.5 * (forward + reverse)).mean()


def rest_bucket_target(
    document: Document, rows: torch.Tensor, depth: int, target: str = Target.DENSE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the rest-bucket target - each kept key's share of the teacher's whole attention, and the share left over.

    The shares sum to the mass the kept keys hold, which rounding in the cache can lift a hair above one,
    so a row whose shares exceed one is renormalised to sum to it and has nothing left over.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        depth: Keep only this many of the teacher's strongest keys.
        target: One of `Target.RANKED`.

    Returns:
        share: Share of the whole attention on each kept key [R, T] fp32, zero elsewhere.
        rest: The share the kept keys do not hold [R] fp32 in [0, 1].
    """
    share = document.key_share(rows, depth, target)
    held = share.sum(-1, keepdim=True)
    return share / held.clamp_min(1.0), (1.0 - held.squeeze(-1)).clamp_min(0.0)


def rest_bucket_cross_entropy(
    log_predicted: torch.Tensor, readable: torch.Tensor, share: torch.Tensor, rest: torch.Tensor
) -> torch.Tensor:
    """Return the cross-entropy to a kept set's shares and one rest entry, from the student's log-softmax over the readable keys.

    The student's rest entry is its mass on the readable keys outside the kept set, taken as a log-sum-exp
    of their log-probabilities so a near-one kept mass costs no precision.
    A row with no key outside the kept set has no rest to answer with and pays nothing for it.

    Args:
        log_predicted: Log-softmax over the readable keys [R, T] fp32, -inf elsewhere.
        readable: Readable-key mask [R, T] bool.
        share: Share on each kept key [R, T] fp32, as `rest_bucket_target` returns.
        rest: The share left over [R] fp32.

    Returns:
        The mean loss over the rows, a scalar.
    """
    outside = readable & (share <= 0)
    has_outside = outside.any(-1)
    # A row with no outside key would be a log-sum-exp of -inf, whose gradient is a nan
    # => it reads a stand-in zero that its weight then removes
    outside_log = log_predicted.masked_fill(~outside, const.Math.NINF)
    outside_log = torch.where(has_outside[:, None], outside_log, torch.zeros_like(outside_log))
    log_rest = torch.logsumexp(outside_log, dim=-1)
    kept = -(share * log_predicted.masked_fill(~readable, 0.0)).sum(-1)
    return (kept - rest * has_outside * log_rest).mean()


def step_terms(
    index_map: IndexMap | QSAIndexer,
    document: Document,
    rows: torch.Tensor,
    objective: "ObjectiveConfig",
) -> dict[str, torch.Tensor]:
    """Return every term one step pays for, each unweighted.

    Args:
        index_map: The layer's learned maps, or its QSA indexer.
        document: The cached document.
        rows: Query rows [R] int64.
        objective: How deep the per-key target is, and which divergence reads it.

    Returns:
        The per-key divergence under `Term.KEY` - per block for a QSA indexer.
    """
    tau = index_map.log_tau.exp()
    if isinstance(index_map, QSAIndexer):
        scores = qsa_block_scores(index_map, document, rows)
        return {Term.KEY: block_divergence(scores / tau, document, rows, objective.target_depth)}
    scores = key_scores(index_map, document, rows)
    return {
        Term.KEY: key_divergence(
            scores / tau,
            document,
            rows,
            objective.target_depth,
            objective.divergence,
            objective.target,
            objective.tail,
        )
    }


def weighted(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the loss a step descends, which is the per-key divergence alone.

    Args:
        terms: The step's terms, as `step_terms` returns them.

    Returns:
        The loss the step descends, a scalar.
    """
    return terms[Term.KEY]


@torch.no_grad()
def target_entropy(
    document: Document,
    rows: torch.Tensor,
    depth: int | None = None,
    target: str = Target.DENSE,
    tail: str = Tail.TRUNCATE,
) -> torch.Tensor:
    """Return the entropy of the rows' target, which the forward divergence is a cross-entropy above.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        depth: The depth the target is truncated to, or None.
        target: Which target, one of `Target.ALL`.
        tail: What the depth does with the mass beyond it, one of `Tail.BOTH`.

    Returns:
        The mean entropy in nats, a scalar.
    """
    if tail == Tail.REST_BUCKET and depth is not None:
        share, rest = rest_bucket_target(document, rows, depth, target)
        entries = torch.cat([share, rest[:, None]], dim=-1)
    else:
        entries = target_of(document, rows, depth, target)
    return -(entries * entries.clamp_min(Calibration.ENTROPY_EPS).log()).sum(-1).mean()


def target_of(
    document: Document, rows: torch.Tensor, depth: int | None = None, target: str = Target.DENSE
) -> torch.Tensor:
    """Return the rows' target - over keys, or over blocks for a document read for a QSA indexer.

    Args:
        document: The document.
        rows: Query rows [R] int64.
        depth: The depth the target is truncated to, or None.
        target: Which target, one of `Target.ALL` - a QSA document has the dense one alone.

    Returns:
        The teacher's distribution [R, T] or [R, T // block_size] fp32.
    """
    if document.block_size is not None:
        return document.block_target(rows, depth)
    return document.key_target(rows, depth, target)


@torch.no_grad()
def calibrate(
    index_map: IndexMap,
    documents: list[Document],
    rows: int = Calibration.ROWS,
    depth: int | None = None,
    target: str = Target.DENSE,
    tail: str = Tail.TRUNCATE,
) -> float:
    """Set the temperature to the grid value the untrained map is least wrong at, and return it.

    Cosine scores over a whole document need a temperature near 0.05.
    At 1.0 the softmax is flat, the gradient carries almost no signal, and a run at a projection learning
    rate that would otherwise be fine destroys the map instead of improving it.
    A learning rate on `tau` cannot recover that inside a run, so it is fixed before step one.

    The criterion is the cross-entropy to the run's own target whatever the run's divergence is, because what is
    being chosen is the sharpness the scores are read at and a truncated target is sharper than a full one.
    Under the rest bucket that target is the kept shares and the rest entry, the loss the run descends.

    A document that waits on the host is staged onto the maps' own device for as long as it is read, which
    is what lets a run calibrate over documents that do not all fit the card at once.

    Args:
        index_map: The layer's learned maps, mutated.
        documents: The training documents of that layer, wherever they wait.
        rows: Query rows to read from each document.
        depth: The target depth the run trains at, so the temperature is chosen for that target.
        target: The target the run trains on, for the same reason.
        tail: What the run's depth does with the mass beyond it, for the same reason.

    Returns:
        The chosen temperature.
    """
    grid = calibration_grid(index_map)
    cross_entropy = torch.zeros_like(grid)
    for waiting in documents:
        cross_entropy += calibration_losses(index_map, waiting, grid, rows, depth, target, tail)
    return choose_tau(index_map, grid, cross_entropy)


def calibration_grid(index_map: IndexMap | QSAIndexer) -> torch.Tensor:
    """Return the temperatures `calibrate` tries, on the map's device - a QSA indexer's scores are not cosines."""
    low, high, count = Calibration.QSA_GRID if isinstance(index_map, QSAIndexer) else Calibration.GRID
    return torch.logspace(low, high, count, device=index_map.log_tau.device)


# Calibration only reads the scores
# => no graph, which at 1M tokens would keep a gigabyte per grid value per intermediate
@torch.no_grad()
def calibration_losses(
    index_map: IndexMap | QSAIndexer,
    waiting: Document,
    grid: torch.Tensor,
    rows: int = Calibration.ROWS,
    depth: int | None = None,
    target: str = Target.DENSE,
    tail: str = Tail.TRUNCATE,
) -> torch.Tensor:
    """Return one document's cross-entropy at every temperature of the grid, so a caller can hold one document at a time.

    Args:
        index_map: The layer's learned maps.
        waiting: The document, wherever it waits.
        grid: The temperatures, as `calibration_grid` returns them.
        rows: Query rows to read from the document.
        depth: The target depth the run trains at.
        target: The target the run trains on.
        tail: What the run's depth does with the mass beyond it.

    Returns:
        The cross-entropy at each temperature [len(grid)].
    """
    cross_entropy = torch.zeros_like(grid)
    document = waiting.to(index_map.log_tau.device)
    query_rows = document.train_rows()[:rows]
    if isinstance(index_map, QSAIndexer):
        readable = document.visible_blocks(query_rows)
        scores = qsa_block_scores(index_map, document, query_rows)
    else:
        readable = document.visible_keys(query_rows)
        scores = key_scores(index_map, document, query_rows)
    rest_target = None
    target_mass = None
    if tail == Tail.REST_BUCKET and depth is not None:
        rest_target = rest_bucket_target(document, query_rows, depth, target)
    else:
        target_mass = target_of(document, query_rows, depth, target)
    for index, tau in enumerate(grid):
        log_predicted = torch.log_softmax((scores / tau).masked_fill(~readable, const.Math.NINF), dim=-1)
        if rest_target is not None:
            cross_entropy[index] += rest_bucket_cross_entropy(log_predicted, readable, *rest_target)
        elif target_mass is not None:
            cross_entropy[index] += -(target_mass * log_predicted.masked_fill(~readable, 0.0)).sum(-1).mean()
    return cross_entropy


def choose_tau(index_map: IndexMap | QSAIndexer, grid: torch.Tensor, cross_entropy: torch.Tensor) -> float:
    """Set the temperature to the grid value with the least summed cross-entropy, and return it."""
    chosen = float(grid[int(cross_entropy.argmin())])
    index_map.set_tau(chosen)
    return chosen


def teacher_entropy(
    documents: list[Document],
    rows: int = Calibration.ROWS,
    device: str | torch.device | None = None,
    target: str = Target.DENSE,
    depth: int | None = None,
    tail: str = Tail.TRUNCATE,
) -> float:
    """Return the teacher's own entropy over the readable keys, which is the floor the loss walks down to.

    Args:
        documents: The documents to read, wherever they wait.
        rows: Query rows to read from each document.
        device: Where to read them, or None to read each where it already is.
        target: The target whose entropy is read.
        depth: The depth the target is truncated to, or None.
        tail: What the depth does with the mass beyond it.

    Returns:
        The mean entropy in nats.
    """
    total = 0.0
    for waiting in documents:
        document = waiting if device is None else waiting.to(device)
        total += float(target_entropy(document, document.train_rows()[:rows], depth, target, tail))
    return total / len(documents)
