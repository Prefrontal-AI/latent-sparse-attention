"""What a read of one document captures of the teacher's own attention.

A read is a list of key ids a query row, `-1` padded, in any order, the window and the sinks among them.
The exact read is the top `top_k` keys by index score among those a row may read, which is the ceiling any
bucketed read of the same scores and budget sits under; the served read is the one a server runs over its
buckets ([reporting](reporting.py)).

The teacher is the cache's strongest visible long-range keys of the row, in rank order, with their head-summed
mass. An entry of mass 0 is padding: a row that sees fewer keys than the cache keeps fills the rest of its list
with ids it may not read, the sinks and keys of the window among them, which every served read holds.
So only the entries of positive mass are teacher keys, and only they are counted.

| metric | what it counts |
|---|---|
| `mass` | share of the teacher's attention the read captures, per head |
| `mass_top` | share of the teacher's cached long-range mass the read captures |
| `recall16`, `recall64` | share of the teacher's top 16 and top 64 keys the read holds |
| `recall` | share of every cached teacher key the read holds |
| `peak` | share of heads whose single strongest visible key the read holds |
| `recall` at a depth | share of the teacher's top-K keys the space's own exact top-K holds, for a K the stored list is deep enough for |
| `stored_held` at a depth | share of the whole stored list the space's exact top-K holds - the only recall-like cell once K passes the list |
| `mass_lower`, `mass_upper`, `mass_width`, `outside_share` at a depth | bounds on the share of the teacher's attention the space's exact top-K captures, past the sinks and the window |
| `teacher_ids_read` | cached teacher keys the read holds |
| `ids_read` | key ids the read holds, the window and the sinks included |
| `far_share` | share of the read's long-range keys at least `Report.FAR_BEHIND` behind the row |
| `far_mass` | share of the teacher's cached mass that far back the read captures |
| `native_overlap` | share of GLM's own long-range picks the read holds, where the document carries them |
| `native_far_overlap` | the same over GLM's picks at least `Report.FAR_BEHIND` behind the row |
| `target_held` | share of a probe's long-range target keys the read holds - a needle's value tokens, or a cwe word's earlier occurrences |
| `target_any` | share of probes with a long-range target whose read holds at least one of them |
| `line_held` | share of the long-range keys of the needle's sentence the read holds, on a needle task |
| `agreement` | share of a served design's long-range keys a reference method also holds, for a method that names a design to agree with |

`far_mass` asks what the internal snapshot's targets asked - its cache saw only keys in coarse blocks of 8192 that closed
before the query's own, so a read that spends its budget on keys a few thousand back scores well on `mass` and
poorly here.

The depth metrics are the exact read of K keys in the space against the stored list, one read a depth.
The teacher softmaxes every head over its readable set - past the sinks and at least a window behind the row -
so a row's head-summed mass over that set is the head count, whatever the list holds.
The list holds the strongest keys only, so the mass the space's picks outside it hold is not stored:
`mass_lower` counts none of it, and `mass_upper` adds the most it could hold - the mass the list leaves over,
capped by every pick outside the list weighing as much as the list's weakest key.
`mass_width` is the gap between them and `outside_share` the share of the read's picks that fall off the list.

Each is taken a row and averaged over the rows. A row whose teacher has no key at a depth has no recall there
and is left out of that mean.

A QSA indexer's read is its own selection - the best closed blocks as keys, and the keys of the query's open block -
scored against QSA's teacher: the strongest keys of the closed blocks under the full attention, not renormalised.
So its `mass` is the share of all the attention the selected blocks capture, the open block's left out, where an LSA
map's is the share of the attention past its sinks and window - the two are not the same number, and the recalls
are each against the teacher's ranking of what that design may select.
"""

from collections.abc import Sequence
from typing import Any

import torch

from lsa import constants as const
from lsa.baselines import qsa
from lsa.sglang.settings.design import QSAConfig
from lsa.training.distillation.constants import Report
from lsa.training.distillation.data.cache import Document


def window_and_sinks(positions: torch.Tensor, sink_size: int, window_size: int) -> torch.Tensor:
    """Return the keys every read holds besides its long-range ones - the window, then the sinks the window does not hold.

    Args:
        positions: The rows' positions [R] int.
        sink_size: First keys always read.
        window_size: Recent keys always read, the query's own included.

    Returns:
        Key ids [R, window_size + sink_size] int32, -1 padded.
    """
    device = positions.device
    part = positions.long()
    window_start = (part + 1 - window_size).clamp_min(0)
    window = window_start[:, None] + torch.arange(window_size, device=device)
    window = torch.where(window <= part[:, None], window, -1)
    sink_ids = torch.arange(sink_size, device=device)
    sinks = torch.where(sink_ids < window_start[:, None], sink_ids, -1)
    return torch.cat([window, sinks], dim=1).to(torch.int32)


def exact_read(
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    positions: torch.Tensor,
    top_k: int,
    sink_size: int,
    window_size: int,
    chunk_size: int = const.Chunk.READ,
) -> torch.Tensor:
    """Read the exact top-k by index score among the visible keys, then the window and the sinks.

    Visible is past the sinks and at least a window behind the query, the rule the teacher is written under.
    The sinks are read where they are not already in the window, as the served read does.

    Args:
        index_queries: The rows' index queries [R, index_dim] fp32.
        index_keys: Every position's index key [T, index_dim].
        positions: The rows' positions [R] int.
        top_k: Long-range keys read at most.
        sink_size: First keys always read.
        window_size: Recent keys always read.
        chunk_size: Query rows scored per pass, which bounds the [rows, T] fp32 scores.

    Returns:
        Key ids [R, top_k + window_size + sink_size] int32 - the long-range keys best first, the window, the
        sinks, each -1 padded.
    """
    key_count = index_keys.shape[0]
    device = index_keys.device
    keys = index_keys.float()
    key_ids = torch.arange(key_count, device=device)
    reads = [torch.empty((0, top_k + window_size + sink_size), dtype=torch.int32, device=device)]
    for start in range(0, len(positions), chunk_size):
        part = positions[start : start + chunk_size].long().to(device)
        scores = index_queries[start : start + chunk_size].float().to(device) @ keys.t()
        visible = (key_ids >= sink_size) & (part[:, None] - key_ids >= window_size)
        best = scores.masked_fill(~visible, const.Math.NINF).topk(min(top_k, key_count), dim=1)
        long_range = torch.where(visible.gather(1, best.indices), best.indices, -1)
        long_range = torch.nn.functional.pad(long_range, (0, top_k - long_range.shape[1]), value=-1)
        kept = window_and_sinks(part, sink_size, window_size)
        reads.append(torch.cat([long_range.to(torch.int32), kept], dim=1))
    return torch.cat(reads)


def held(selected_key_ids: torch.Tensor, wanted_ids: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """Return which of each row's wanted keys the read holds.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        wanted_ids: Keys to look for [R, N] int.
        wanted: Which of them count [R, N] bool.

    Returns:
        Whether each wanted key was read [R, N] bool, False where it does not count.
    """
    read = selected_key_ids.long().sort(dim=1).values
    looked_for = wanted_ids.long().contiguous()
    slots = torch.searchsorted(read, looked_for).clamp_max(read.shape[1] - 1)
    return (read.gather(1, slots) == looked_for) & wanted


def row_metrics(
    selected_key_ids: torch.Tensor, top_key_ids: torch.Tensor, top_key_mass: torch.Tensor, head_count: int
) -> dict[str, torch.Tensor]:
    """Measure what a read captured of the teacher's strongest keys, one number a row.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        top_key_ids: The teacher's strongest keys [R, N] int, in rank order.
        top_key_mass: Their head-summed mass [R, N], 0 on padding.
        head_count: Heads the mass is summed over.

    Returns:
        Each metric but `peak` [R] fp32, by name - NaN for a recall where the row's teacher has no key.
    """
    mass = top_key_mass.float()
    teacher = mass > 0
    hit = held(selected_key_ids, top_key_ids, teacher)
    captured = (hit * mass).sum(1)
    metrics = {"mass": captured / head_count, "mass_top": captured / mass.sum(1)}
    for depth in Report.RECALL_DEPTHS:
        metrics[f"recall{depth}"] = hit[:, :depth].sum(1).float() / teacher[:, :depth].sum(1).float()
    metrics["recall"] = hit.sum(1).float() / teacher.sum(1).float()
    metrics["teacher_ids_read"] = hit.sum(1).float()
    metrics["ids_read"] = (selected_key_ids >= 0).sum(1).float()
    return metrics


def read_metrics(document: Document, rows: torch.Tensor, selected_key_ids: torch.Tensor) -> dict[str, float]:
    """Score one read of some rows of a document, each metric averaged over the rows.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        selected_key_ids: The read [R, W] int, -1 padded, on the document's device.

    Returns:
        One number per metric.
    """
    per_row = read_row_metrics(document, rows, selected_key_ids)
    return {name: float(values.nanmean()) for name, values in per_row.items()}


def read_row_metrics(
    document: Document, rows: torch.Tensor, selected_key_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Score one read of some rows of a document, one number a row.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        selected_key_ids: The read [R, W] int, -1 padded, on the document's device.

    Returns:
        Each metric [R] fp32, by name - NaN for a recall where the row's teacher has no key.
    """
    rows = rows.to(document.device)
    per_row = row_metrics(
        selected_key_ids, document.top_key_ids[rows], document.top_key_mass[rows], document.head_count
    )
    peak_ids = document.peak_key_id[rows]
    visible = document.selectable(peak_ids, rows)
    per_row["peak"] = held(selected_key_ids, peak_ids, visible).float().mean(1)
    return per_row


def served_row_metrics(
    document: Document, rows: torch.Tensor, selected_key_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Score one served read of some rows of a document, one number a row - `read_row_metrics`, its reach and its overlap with GLM's picks.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        selected_key_ids: The read [R, W] int, -1 padded, on the document's device.

    Returns:
        Each metric [R] fp32, by name - `native_overlap` and `native_far_overlap` only where the document carries GLM's picks.
    """
    per_row = read_row_metrics(document, rows, selected_key_ids)
    rows = rows.to(document.device)
    positions = document.query_positions[rows]
    per_row.update(
        reach_metrics(
            selected_key_ids,
            positions,
            document.top_key_ids[rows],
            document.top_key_mass[rows],
            document.sink_size,
            document.window_size,
        )
    )
    if document.target_key_ids is not None:
        per_row.update(
            probe_metrics(
                selected_key_ids,
                positions,
                document.target_key_ids[rows].long(),
                None if document.line_key_ids is None else document.line_key_ids[rows].long(),
                document.sink_size,
                document.window_size,
            )
        )
    if document.native_key_ids is not None:
        per_row.update(
            native_overlap(
                selected_key_ids,
                document.native_key_ids[rows],
                positions,
                document.sink_size,
                document.window_size,
            )
        )
    return per_row


def reach_metrics(
    selected_key_ids: torch.Tensor,
    positions: torch.Tensor,
    top_key_ids: torch.Tensor,
    top_key_mass: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Measure how far back a read reaches, and what it captures of the teacher that far back.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        positions: The rows' positions [R] int64.
        top_key_ids: The teacher's strongest keys [R, N] int, in rank order.
        top_key_mass: Their head-summed mass [R, N], 0 on padding.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        `far_share` and `far_mass` [R] fp32 - NaN where a row reads no long-range key or has no far teacher mass.
    """
    selected = selected_key_ids.long()
    behind = positions[:, None] - selected
    long_range = (selected >= sink_size) & (behind >= window_size)
    far_read = long_range & (behind >= Report.FAR_BEHIND)
    mass = top_key_mass.float()
    far_teacher = (mass > 0) & (positions[:, None] - top_key_ids.long() >= Report.FAR_BEHIND)
    far_teacher_mass = (mass * far_teacher).sum(1)
    captured = (held(selected_key_ids, top_key_ids, far_teacher) * mass).sum(1)
    return {
        "far_share": far_read.sum(1).float() / long_range.sum(1).float(),
        "far_mass": torch.where(far_teacher_mass > 0, captured / far_teacher_mass, torch.nan),
    }


def native_overlap(
    selected_key_ids: torch.Tensor,
    native_key_ids: torch.Tensor,
    positions: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Measure how much of GLM's own long-range selection a read holds.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        native_key_ids: GLM's own picks [R, K] int, -1 padded.
        positions: The rows' positions [R] int64.
        sink_size: First keys always read, which are not long-range.
        window_size: Recent keys always read, which are not long-range.

    Returns:
        `native_overlap` and `native_far_overlap` [R] fp32 - NaN where GLM picked no such key.
    """
    picked = native_key_ids.long()
    behind = positions[:, None] - picked
    long_range = (picked >= sink_size) & (behind >= window_size)
    far = long_range & (behind >= Report.FAR_BEHIND)
    hit = held(selected_key_ids, picked, long_range)
    return {
        "native_overlap": hit.sum(1).float() / long_range.sum(1).float(),
        "native_far_overlap": (hit & far).sum(1).float() / far.sum(1).float(),
    }


def long_range_keys(
    key_ids: torch.Tensor, positions: torch.Tensor, sink_size: int, window_size: int
) -> torch.Tensor:
    """Return which of some keys lie past a row's sinks and window, the keys a selector chooses.

    Args:
        key_ids: Key ids [R, N] int, -1 where there is none.
        positions: The rows' positions [R] int64.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        Whether each is a long-range key [R, N] bool.
    """
    ids = key_ids.long()
    return (ids >= sink_size) & (positions[:, None] - ids >= window_size)


def held_long_range(
    selected_key_ids: torch.Tensor,
    wanted_ids: torch.Tensor,
    positions: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure how much of some wanted keys a read holds, counting only those past the sinks and the window.

    Every read holds the window and the sinks, so a wanted key there is held by all of them and says nothing about
    the selector - it is left out of the count, and a row with none left has nothing to hold.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        wanted_ids: The keys the row was meant to find [R, N] int, -1 padded.
        positions: The rows' positions [R] int64.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        The share of the long-range wanted keys the read holds [R] fp32,
        and whether it holds any of them [R] fp32 - NaN for both on a row with no long-range wanted key.
    """
    counted = long_range_keys(wanted_ids, positions, sink_size, window_size)
    hit = held(selected_key_ids, wanted_ids, counted)
    wanted = counted.sum(1).float()
    share = torch.where(wanted > 0, hit.sum(1).float() / wanted, torch.nan)
    anything = torch.where(wanted > 0, hit.any(1).float(), torch.nan)
    return share, anything


def probe_metrics(
    selected_key_ids: torch.Tensor,
    positions: torch.Tensor,
    target_key_ids: torch.Tensor,
    line_key_ids: torch.Tensor | None,
    sink_size: int,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Measure what a read holds of the keys a probe was meant to find.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        positions: The rows' positions [R] int64.
        target_key_ids: The keys each probe is meant to find [R, N] int, -1 padded.
        line_key_ids: The keys of the needle's sentence [R, L] int, -1 padded, or None off a needle task.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        `target_held` and `target_any`, and `line_held` on a needle task, each [R] fp32 - NaN where a row has no
        long-range key to find.
    """
    share, anything = held_long_range(selected_key_ids, target_key_ids, positions, sink_size, window_size)
    metrics = {"target_held": share, "target_any": anything}
    if line_key_ids is not None:
        metrics["line_held"] = held_long_range(
            selected_key_ids, line_key_ids, positions, sink_size, window_size
        )[0]
    return metrics


def agreement(
    selected_key_ids: torch.Tensor,
    served_key_ids: torch.Tensor,
    positions: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> torch.Tensor:
    """Measure how much of a served design's long-range read another read holds.

    Args:
        selected_key_ids: The read [R, W] int, -1 padded, in any order.
        served_key_ids: The served design's read [R, K] int, -1 padded.
        positions: The rows' positions [R] int64.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        The share of the served read's long-range keys the read holds [R] fp32 - NaN where it reads none.
    """
    return native_overlap(selected_key_ids, served_key_ids, positions, sink_size, window_size)[
        "native_overlap"
    ]


def probe_rows(
    positions: torch.Tensor,
    query_kind: torch.Tensor,
    target_key_ids: torch.Tensor,
    sink_size: int,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Describe each probe row - its kind, how many long-range keys it looks for and how far back the nearest is.

    Args:
        positions: The rows' positions [R] int64.
        query_kind: Each row's kind, an index into the cache's `query_kinds` [R] int.
        target_key_ids: The keys each probe is meant to find [R, N] int, -1 padded.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        `query_kind`, `target_count` and `target_distance`, each [R] int64 - the nearest long-range target's distance
        behind the row, -1 where it has none.
    """
    counted = long_range_keys(target_key_ids, positions, sink_size, window_size)
    farthest = torch.iinfo(torch.int64).max
    behind = (positions[:, None] - target_key_ids.long()).masked_fill(~counted, farthest)
    return {
        "query_kind": query_kind.long(),
        "target_count": counted.sum(1),
        "target_distance": torch.where(counted.any(1), behind.min(1).values, -1),
    }


def depth_metrics(
    document: Document, rows: torch.Tensor, long_range: torch.Tensor, depth: int
) -> dict[str, float]:
    """Score the exact read of `depth` long-range keys against the teacher's stored list, a row at a time.

    Args:
        document: The cached document.
        rows: Query rows [R] int64.
        long_range: The space's long-range keys best first [R, W] int, -1 padded, at least `depth` wide.
        depth: Keys read.

    Returns:
        `recall` where the stored list is at least `depth` deep, `stored_held`, `mass_lower`, `mass_upper`,
        `mass_width` (their gap) and `outside_share` (the share of the read's picks off the stored list, which drives
        the gap), each averaged over the rows.
    """
    rows = rows.to(document.device)
    read = long_range[:, :depth]
    ids = document.top_key_ids[rows]
    mass = document.top_key_mass[rows].float()
    head_count = document.head_count
    teacher = mass > 0
    hit = held(read, ids, teacher)
    stored = teacher.sum(1).float()
    held_count = hit.sum(1).float()
    captured = (hit * mass).sum(1)
    # A pick outside the list weighs at most the list's weakest key, a list that is not full holding every readable key
    outside = ((read >= 0).sum(1) - hit.sum(1)).float()
    unseen = torch.minimum((head_count - mass.sum(1)).clamp_min(0), outside * mass.min(1).values)
    metrics = {
        "stored_held": held_count / stored,
        "mass_lower": captured / head_count,
        "mass_upper": (captured + unseen) / head_count,
        "mass_width": unseen / head_count,
        "outside_share": outside / (read >= 0).sum(1).float(),
    }
    if depth <= ids.shape[1]:
        metrics["recall"] = hit[:, :depth].sum(1).float() / teacher[:, :depth].sum(1).float()
    return {name: float(values.nanmean()) for name, values in metrics.items()}


def key_read(
    document: Document,
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    top_k: int,
    rows: torch.Tensor | None = None,
    chunk_size: int = const.Chunk.READ,
    depths: Sequence[int] = (),
) -> dict[str, Any]:
    """Score the exact top-`top_k` read of one document, and the exact read at each of some other depths.

    Args:
        document: The cached document.
        index_queries: Index queries of the rows to score [R, index_dim], in the row order of `rows`.
        index_keys: Index keys [T, index_dim] in that same space.
        top_k: Long-range keys read per query row.
        rows: Query rows to score [R] int64, or None for the held-out tail.
        chunk_size: Query rows scored per pass.
        depths: Keys to read, one exact read each, scored by `depth_metrics` under `Report.DEPTHS`.

    Returns:
        One number per metric, and the depth metrics by depth when `depths` is given.
    """
    rows = document.eval_rows() if rows is None else rows
    positions = document.query_positions[rows.to(document.device)]
    read = exact_read(
        index_queries, index_keys, positions, top_k, document.sink_size, document.window_size, chunk_size
    )
    out: dict[str, Any] = dict(read_metrics(document, rows, read))
    if depths:
        # The long-range keys come best first, so a shallower read is the front of the deepest
        most = max(depths)
        deepest = exact_read(
            index_queries, index_keys, positions, most, document.sink_size, document.window_size, chunk_size
        )[:, :most]
        out[Report.DEPTHS] = {str(depth): depth_metrics(document, rows, deepest, depth) for depth in depths}
    return out


def qsa_read(
    document: Document,
    index_queries: torch.Tensor,
    index_keys: torch.Tensor,
    config: QSAConfig,
    rows: torch.Tensor,
    chunk_size: int = Report.QSA_ROWS,
) -> dict[str, float]:
    """Score a QSA indexer's own read of one document - its best closed blocks as keys, and the open block's keys.

    Args:
        document: The document, read for a QSA indexer.
        index_queries: Index queries of the rows to score [R, H, head_dim] fp32, in the row order of `rows`.
        index_keys: Every closed block's index key [B, head_dim] fp32.
        config: The indexer's shape, whose `top_blocks` is the read's budget.
        rows: Query rows to score [R] int64.
        chunk_size: Query rows scored per pass, which bounds the [rows, B, H] fp32 dots.

    Returns:
        One number per metric.
    """
    positions = document.query_positions[rows.to(document.device)]
    reads = [
        qsa.select(
            qsa.block_scores(index_queries[start : start + chunk_size], index_keys),
            positions[start : start + chunk_size],
            config.block_size,
            config.top_blocks,
        )
        for start in range(0, len(rows), chunk_size)
    ]
    return read_metrics(document, rows, torch.cat(reads))


def mean_reports(reports: list[Any], weights: list[int] | None = None) -> Any:
    """Average a metric tree over documents, keeping its shape.

    Args:
        reports: One tree per document, all the same shape. A leaf is a number.
        weights: How many documents each tree stands for, or None when each is one.

    Returns:
        The same tree with each leaf averaged.
    """
    first = reports[0]
    if isinstance(first, dict):
        return {key: mean_reports([report[key] for report in reports], weights) for key in first}
    if weights is None:
        return sum(reports) / len(reports)
    return sum(weight * report for weight, report in zip(weights, reports, strict=True)) / sum(weights)
