"""Decide whether a selection is the true top-k, allowing for ties and for an unordered result.

The check is on values rather than on indices: every selected candidate must score at or above the
exact cutoff, and the number strictly above it must match. That admits any tie-break the kernel makes
and still fails a selection that missed a better candidate.

A short row - one that sees fewer than k candidates - has every visible candidate as its top-k, so it must select
all of them, once each, and pad the rest with -1, which no row that sees k candidates may do.
"""

from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from lsa.core.kernels.triton.keys import unpack_ids

# Differing rows of a rerun that are weighed one by one for the widest side
SELECTION_ROWS_WEIGHED = 512


class NoSelection:
    """The dense and the linear layer select nothing - every selection check is vacuous"""

    rows = None
    tree = False
    expected_tail = None
    allow_invalid_topk = False
    failure_path = None
    failures = ()

    def __init__(self) -> None:
        self.reference: dict[str, Any] = {}

    @contextmanager
    def capture(self, target):
        """Capture nothing."""
        yield

    def compare_scores(self, actual) -> None:
        """Compare nothing."""

    def prepare(self) -> None:
        """Prepare nothing."""

    def check(self, selected):
        """Return None - no verdict."""
        return None


class SelectionCheck:
    """The model's own selection held to the exact top-k of its index scores.

    `capture` keeps the scores the indexer ranks, `prepare` derives the exact cutoff from them and `check` holds
    a selection to it, allowing for ties and for any order.
    """

    # Whether the read went through a hierarchy, which has no one score row to hold to the exact top-k
    tree = False

    def __init__(self, module, slots, capacity, positions=None, score_rows=None, decode=False) -> None:
        """Map the request's KV slots back to positions.

        Args:
            module: The layer under test.
            slots: The request's KV slots in position order.
            capacity: Slots the pool holds.
            positions: Each query row's position, for the causal check.
            score_rows: The sampled query rows whose scores are kept, or None for all.
            decode: Whether the step is a decode step.
        """
        self.indexer = module.indexer
        self.decode = decode
        self.reverse = np.full(capacity + 64, -1, dtype=np.int64)
        self.reverse[slots.cpu().numpy()] = np.arange(len(slots))
        self.k = self.indexer.index_topk // self.indexer.index_kpool
        self.reference: dict[str, Any] = {}
        self.positions = positions.cpu().numpy() if positions is not None else None
        self.expected_tail = None
        self.rows = score_rows
        self.failures: list[dict[str, Any]] = []
        self.allow_invalid_topk = False
        self.failure_path = None

    @contextmanager
    def capture(self, target):
        """Keep the scores, visible counts and total counts the indexer ranks, for the sampled rows.

        Args:
            target: Where they go, which must be empty.

        Raises:
            RuntimeError: `target` is not empty.
        """
        if target:
            raise RuntimeError("Pilot expects one top-k capture per chunk")
        original = self.indexer._topk_from_kpool_logits
        # The indexer may split a chunk's query rows into consecutive slices, one call each, when its logits
        # would not fit free memory
        parts = []
        first_row = 0

        def call(*args, **kwargs):
            """Keep the sampled rows of one slice's scores, then rank them."""
            nonlocal first_row
            scores = kwargs["logits"] if "logits" in kwargs else args[0]
            lengths = kwargs["pool_lens"] if "pool_lens" in kwargs else args[1]
            row_count = scores.shape[0]
            if self.rows is None:
                picked = slice(None)
            else:
                picked = [row - first_row for row in self.rows if first_row <= row < first_row + row_count]
            parts.append((scores[picked].clone(), lengths[picked].clone(), lengths.clone()))
            first_row += row_count
            return original(*args, **kwargs)

        with patch.object(self.indexer, "_topk_from_kpool_logits", call):
            yield
        if parts:
            target["scores"] = torch.cat([scores for scores, _, _ in parts])
            target["lengths"] = torch.cat([lengths for _, lengths, _ in parts])
            target["all_lengths"] = torch.cat([all_lengths for _, _, all_lengths in parts])

    def compare_scores(self, actual):
        """Hold a rerun's captured scores bitwise to the reference's.

        Args:
            actual: The rerun's capture.

        Raises:
            AssertionError: The counts or visible scores differ.
        """
        for name in ("lengths", "all_lengths"):
            if not torch.equal(actual[name], self.reference[name]):
                raise AssertionError(f"Top-k {name} differ from the reference")
        expected = self.reference["scores"]
        observed = actual["scores"]
        if self.decode:
            width = int(self.reference["lengths"].max())
            valid = torch.arange(width, device=expected.device)[None, :] < self.reference["lengths"][:, None]
            expected, observed = expected[:, :width][valid], observed[:, :width][valid]
        if not torch.equal(observed, expected):
            raise AssertionError(
                f"Top-k visible scores differ: shapes {observed.shape}/{expected.shape}, "
                f"max_abs={(observed - expected).abs().max().item()}"
            )

    def prepare(self):
        """Derive the exact top-k cutoff of each sampled row from the reference scores, then drop them.

        Raises:
            RuntimeError: A row sees fewer than k candidates.
            AssertionError: A visible score is not finite.
        """
        scores = self.reference["scores"]
        lengths = self.reference["lengths"]
        if not (lengths >= self.k).all():
            raise RuntimeError("Pilot expects at least k visible candidates")
        invisible = torch.arange(scores.shape[1], device=scores.device)[None, :] >= lengths[:, None]
        if not (torch.isfinite(scores) | invisible).all():
            raise AssertionError("Non-finite visible index score")
        masked = scores.masked_fill(invisible, -torch.inf)
        self.cutoff = torch.topk(masked, self.k, dim=1).values[:, -1].cpu().numpy()
        self.above_cutoff = (
            (masked > torch.as_tensor(self.cutoff, device=scores.device)[:, None]).sum(1).cpu().numpy()
        )
        self.scores = scores.cpu().numpy()
        self.lengths = lengths.cpu().numpy()
        self.all_lengths = self.reference.get("all_lengths", lengths).cpu().numpy()
        self.reference.clear()

    def check(self, selected) -> bool | None:
        """Hold a selection to causality, to pool alignment and to the exact top-k.

        Args:
            selected: The selected KV slots [S, width] int32, -1 padded.

        Returns:
            Whether every pick is at or above the cutoff - False only when invalid picks are allowed.

        Raises:
            AssertionError: A slot is invalid, outside the request or non-causal, or the picks are not the top-k.
        """
        slots = selected.cpu().numpy()
        if ((slots < -1) | (slots >= len(self.reverse))).any():
            raise AssertionError("Invalid physical slot")
        valid = slots >= 0
        mapped = self.reverse[slots]
        if (valid & (mapped < 0)).any():
            raise AssertionError("Selected slot is outside the request")
        if self.positions is not None and (valid & (mapped > self.positions[:, None])).any():
            raise AssertionError("Non-causal selection")
        if self.expected_tail is not None and not np.array_equal(
            slots[:, self.indexer.index_topk :], self.expected_tail
        ):
            raise AssertionError("Native tail selection changed")
        raw = slots[:, : self.indexer.index_topk]
        positions = self.reverse[raw]
        if (positions < 0).any():
            raise AssertionError("Selected slot is outside the request")
        pools = positions.reshape(len(slots), self.k, self.indexer.index_kpool)
        if (pools[:, :, 0] % self.indexer.index_kpool).any():
            raise AssertionError("Pooled selection is not group-aligned")
        if not np.all(pools == pools[:, :, :1] + np.arange(self.indexer.index_kpool)):
            raise AssertionError("Pooled selection did not expand to complete key groups")
        logical = pools[:, :, 0] // self.indexer.index_kpool
        if ((logical < 0) | (logical >= self.all_lengths[:, None])).any():
            raise AssertionError("Selected an invisible candidate")
        return self.verify_picks(logical)

    def verify_picks(self, logical) -> bool:
        """Hold the picked candidates [rows, k] to the exact top-k, a pick of -1 being a row's padding.

        Only a row that sees fewer than k candidates may pad its picks.
        """
        picked = logical >= 0
        short = self.all_lengths[:, None] < self.k
        if ((~picked & ~short) | (logical >= self.all_lengths[:, None])).any():
            raise AssertionError("Selected an invisible candidate")
        # Each padding entry is its own value, so padding is never a duplicate
        distinct = np.where(picked, logical, -1 - np.arange(logical.shape[1])[None, :])
        if (np.diff(np.sort(distinct, axis=1), axis=1) == 0).any():
            raise AssertionError("Duplicate top-k candidate")
        if self.rows is not None:
            logical, picked = logical[self.rows], picked[self.rows]
        values = np.where(
            picked, self.scores[np.arange(len(logical))[:, None], np.maximum(logical, 0)], -np.inf
        )
        below = picked & (values < self.cutoff[:, None])
        if below.any():
            bad = np.flatnonzero(below.any(1))
            first = int(bad[0])
            failure = self.failure(logical, values, below, bad, first)
            if not self.failures and self.failure_path is not None:
                np.savez_compressed(
                    self.failure_path,
                    scores=self.scores[bad[:8]],
                    lengths=self.lengths[bad[:8]],
                    selected=logical[bad[:8]],
                    cutoff=self.cutoff[bad[:8]],
                    query_rows=np.asarray(self.rows if self.rows is not None else np.arange(len(logical)))[
                        bad[:8]
                    ],
                )
            self.failures.append(failure)
            if self.allow_invalid_topk:
                return False
            raise AssertionError(f"Selected below the true top-k cutoff: {failure}")
        if not np.array_equal((values > self.cutoff[:, None]).sum(1), self.above_cutoff):
            raise AssertionError("Omitted a score above the true top-k cutoff")
        return True

    def failure(self, logical, values, below, bad, first):
        """Describe the first row that selected below the cutoff, with where the native kernel's threshold bin sits."""
        row = self.scores[first, : self.lengths[first]]
        bits = row.astype(np.float16).view(np.uint16)
        bins = np.where(bits & 0x8000, ~bits, bits | 0x8000) >> 8
        cutoff_bits = np.array(self.cutoff[first], dtype=np.float16).view(np.uint16)
        cutoff_bin = int((~cutoff_bits if cutoff_bits & 0x8000 else cutoff_bits | 0x8000) >> 8) & 255
        return {
            "kind": "below_exact_topk_cutoff",
            "checked_rows": len(logical),
            "failing_rows": len(bad),
            "first_query": self.rows[first] if self.rows is not None else first,
            "cutoff": float(self.cutoff[first]),
            "lowest_selected": float(values[first].min()),
            "selected_below_cutoff": int(below[first].sum()),
            "native_threshold_bin_candidates": int((bins == cutoff_bin).sum()),
        }


class LSASelectionCheck(SelectionCheck):
    """The design's selection held to the exact top-p of its bucket scores.

    A flat read ranks one score row a query - a dense product and the exact top-k below `SCAN_SELECT_BUCKETS`
    visible buckets, a fused listing and its pick above it, with or without the pick when an expansion by
    threshold reads the listing's best buckets itself. The check holds the buckets that were picked to the exact top-p of
    those scores: none below the cutoff, as many above it as the true top-p has.

    A read through a hierarchy ranks a beam a level at a time, with no one score row to hold it to. It is
    recognised by the beam search running, and has no verdict on its top-p - its selection is held to the
    prepared one by `selection_rows`, and its output to the reference.
    """

    def __init__(self, module, slots, capacity, positions=None, score_rows=None, decode=False) -> None:
        """Map the request's KV slots back to positions and find the selector's pickers.

        Args:
            module: The layer under test.
            slots: The request's KV slots in position order.
            capacity: Slots the pool holds.
            positions: Each query row's position, for the causal check.
            score_rows: The sampled query rows whose scores are kept, or None for all.
            decode: Whether the step is a decode step.
        """
        self.indexer = module.indexer
        self.selector = self.indexer.index.selector
        self.decode = decode
        self.top_k: Any = self.selector.decode_top_k if decode else self.selector.prefill_top_k
        # A decode step never takes the fused select, and under CuTeDSL lists its buckets without a top-k
        self.scan_select: Any = None if decode else self.selector.scan_select
        self.decode_select: Any = self.selector.decode_select if decode else None
        self.tree = False
        # Each read of the chunk - whether it was fused, how many rows it held and what it listed
        self.parts: list[dict[str, Any]] = []
        self.beaming = 0
        self.reverse = np.full(capacity + 64, -1, dtype=np.int64)
        self.reverse[slots.cpu().numpy()] = np.arange(len(slots))
        self.k = self.selector.config.top_p
        self.reference: dict[str, Any] = {}
        self.positions = positions.cpu().numpy() if positions is not None else None
        self.expected_tail = None
        self.rows = score_rows
        self.failures: list[dict[str, Any]] = []
        self.allow_invalid_topk = False
        self.failure_path = None

    @contextmanager
    def capture(self, target):
        """Keep the scores, counts and listing of each read of the chunk, for the sampled rows.

        Args:
            target: Where the padded scores and counts go, which must be empty.

        Raises:
            RuntimeError: `target` is not empty.
        """
        from lsa.core.kernels.cutedsl import top_k as cutedsl_top_k
        from lsa.core.kernels.triton.paged import scan_paged
        from lsa.core.kernels.triton.scan_select import dense_scores
        from lsa.sglang.index.hierarchy import Hierarchy

        if target:
            raise RuntimeError("Pilot expects one top-k capture per chunk")
        top_k, scan_select, decode_select = self.top_k, self.scan_select, self.decode_select
        top_k_call, beam_search = type(top_k).listed, Hierarchy.beam_search
        # A chunk wider than a page of rows reads in consecutive segments, one listing each
        # => each part keeps its own rows of the sampled ones, offset by the rows read before it
        parts: list[dict[str, Any]] = []
        first_row = 0

        def rows_of(count):
            """Return the sampled rows inside the segment being read, relative to its start."""
            if self.rows is None:
                return slice(None)
            return [row - first_row for row in self.rows if first_row <= row < first_row + count]

        def keep(scores_of, lengths, fused, listing) -> None:
            """Store the sampled rows of one read and advance past its rows."""
            nonlocal first_row
            picked = rows_of(len(lengths))
            parts.append(
                {
                    "scores": scores_of(picked).clone(),
                    "lengths": lengths[picked].clone(),
                    "all_lengths": lengths.clone(),
                    "fused": fused,
                    "listing": listing,
                }
            )
            first_row += len(lengths)

        def ranked(picker, scores, lengths):
            """Run a ranking and keep the read's own pick, not a beam level's."""
            # Only the read's own pick is held to the scores - a beam's levels are other top-ks
            listing = top_k_call(picker, scores, lengths)
            if picker is top_k and not self.beaming:
                # The read releases its keys and counts from the top-k once it has expanded them
                keep(lambda picked: scores[picked], lengths, False, listing)
            return listing

        list_rows = cutedsl_top_k.list_rows

        # A short prefill under CuTeDSL lists its scores without the top-k
        # => its listing is put back into the top-k for the pick
        def short_listed(scores, lengths, top_k_wanted, ordered=False):
            """List a short prefill's rows and keep their scores."""
            listing = list_rows(scores, lengths, top_k_wanted, ordered=ordered)
            if not self.decode and not self.beaming:
                keep(lambda picked: scores[picked], lengths, False, listing)
            return listing

        def beamed(hierarchy, *args, **kwargs):
            """Mark the read as a hierarchy while its beam search runs."""
            self.tree = True
            self.beaming += 1
            try:
                return beam_search(hierarchy, *args, **kwargs)
            finally:
                self.beaming -= 1

        with ExitStack() as stack:
            stack.enter_context(patch.object(type(top_k), "listed", ranked))
            stack.enter_context(patch.object(Hierarchy, "beam_search", beamed))
            stack.enter_context(patch.object(cutedsl_top_k, "list_rows", short_listed))
            # A flat listing may fill its read without a pick
            # => its sampled rows are scored once a listing
            if scan_select is not None:
                listed = type(scan_select).listed

                # A shaped read lists by each bucket's estimated best key => its scores are the shaped ones
                def listed_call(select, queries, centroids, lengths, shapes=None):
                    """List a flat read and keep its sampled rows' dense scores."""
                    listing = listed(select, queries, centroids, lengths, shapes)
                    if select is scan_select and not self.beaming:
                        # A fused read writes no scores
                        # => only its sampled rows are scored, in full
                        keep(
                            lambda picked: dense_scores(
                                queries[picked], centroids, lengths[picked], shapes=shapes
                            ),
                            lengths,
                            True,
                            listing,
                        )
                    return listing

                stack.enter_context(patch.object(type(scan_select), "listed", listed_call))
            # A CuTeDSL decode step lists while it scans
            # => its scores are the paged scan's
            if decode_select is not None:
                decode_listed = type(decode_select).listed

                def decode_listed_call(select, queries, centroids, pages, counts, page_rows, max_rows):
                    """List a decode step and keep the paged scan's scores."""
                    scores = None
                    if select is decode_select:
                        scores = scan_paged(queries, centroids, pages, counts, page_rows, max_rows)
                    listing = decode_listed(select, queries, centroids, pages, counts, page_rows, max_rows)
                    if scores is not None:
                        keep(lambda picked: scores[picked], counts, True, listing)
                    return listing

                stack.enter_context(patch.object(type(decode_select), "listed", decode_listed_call))
            yield
        self.parts = parts
        if parts:
            width = max(part["scores"].shape[1] for part in parts)
            target["scores"] = torch.cat(
                [F.pad(part["scores"], (0, width - part["scores"].shape[1])) for part in parts]
            )
            target["lengths"] = torch.cat([part["lengths"] for part in parts])
            target["all_lengths"] = torch.cat([part["all_lengths"] for part in parts])

    def compare_scores(self, actual) -> None:
        """Hold a rerun's captured scores to the reference's, unless the read went through a hierarchy."""
        if not self.tree:
            super().compare_scores(actual)

    def prepare(self):
        """Derive the exact top-p cutoff of each sampled row from the reference scores, then drop them.

        Raises:
            AssertionError: A visible score is not finite.
        """
        if self.tree:
            return
        scores = self.reference["scores"]
        lengths = self.reference["lengths"]
        invisible = torch.arange(scores.shape[1], device=scores.device)[None, :] >= lengths[:, None]
        if not (torch.isfinite(scores) | invisible).all():
            raise AssertionError("Non-finite visible index score")
        masked = scores.masked_fill(invisible, -torch.inf)
        # A row that sees fewer buckets than the top-p has every one of them as its top-p
        self.cutoff = torch.topk(masked, min(self.k, scores.shape[1]), dim=1).values[:, -1].cpu().numpy()
        self.above_cutoff = (
            (masked > torch.as_tensor(self.cutoff, device=scores.device)[:, None]).sum(1).cpu().numpy()
        )
        self.scores = scores.cpu().numpy()
        self.lengths = lengths.cpu().numpy()
        self.all_lengths = self.reference.get("all_lengths", lengths).cpu().numpy()
        self.reference.clear()

    def check(self, selected) -> bool | None:
        """Hold a selection to causality and to the exact top-p of the bucket scores.

        Args:
            selected: The selected KV slots [S, width] int32, -1 padded.

        Returns:
            Whether every pick is at or above the cutoff, or None for a hierarchy, which has no verdict.

        Raises:
            AssertionError: A slot is invalid, outside the request or non-causal, or the picks are not the top-p.
        """
        slots = selected.cpu().numpy()
        if ((slots < -1) | (slots >= len(self.reverse))).any():
            raise AssertionError("Invalid physical slot")
        valid = slots >= 0
        mapped = self.reverse[slots]
        if (valid & (mapped < 0)).any():
            raise AssertionError("Selected slot is outside the request")
        if self.positions is not None and (valid & (mapped > self.positions[:, None])).any():
            raise AssertionError("Non-causal selection")
        if self.tree:
            return None
        logical = torch.cat([self.best_buckets(part) for part in self.parts])[: len(slots)].cpu().numpy()
        return self.verify_picks(logical)

    def best_buckets(self, part: dict[str, Any]) -> torch.Tensor:
        """Pick one read's best buckets from what it listed.

        Args:
            part: The read as `capture` kept it.

        Returns:
            Buckets [rows, top_p] int64, -1 past a row's visible count.
        """
        # Both pickers return a view of one buffer they reuse
        # => each read's picks are copied out before the next is picked
        if part["fused"]:
            # An expansion by threshold reads the walk of the listing's best picks without picking them
            # => pick them here, from the listing it read
            picker = self.decode_select if self.decode else self.scan_select
            return picker.pick(part["listing"]).clone().long()
        # A captured read released its listing from the top-k
        # => put it back before asking for the best
        self.top_k.keys, self.top_k.counts = part["listing"]
        return unpack_ids(self.top_k.best()).clone().long()

    def failure(self, logical, values, below, bad, first):
        """Describe the first row that selected below the cutoff - bucket scores have no threshold bin."""
        failure = super().failure(logical, values, below, bad, first)
        # The native kernel's threshold bin has no meaning for fp32 bucket scores
        failure["native_threshold_bin_candidates"] = None
        return failure


def weigh(rerun: dict[str, Any]) -> dict[str, Any]:
    """Say whether a rerun is exact - no row of it differs from the prepared selection.

    A selection is a set of KV slots a row. The prepared one is what the first pass over the chunk chose, and the
    rerun is the chunk run again from its snapshot. Every pick is the exact top-p of its scores with a tie to the
    lower bucket (invariant 30), so a rerun chooses the same buckets on every row and the bar is zero rows.
    The record weighs the facts a rerun wrote, so a ladder's record can be rebuilt without measuring again.

    Args:
        rerun: The facts `selection_rows` wrote - the rows, how many differ, the most ids a side a differing row
            lacks and the most keys a bucket holds.

    Returns:
        The facts, with whether the rerun is exact.
    """
    return {**rerun, "exact": rerun["differing"] == 0}


def selection_rows(observed: torch.Tensor, expected: torch.Tensor, bucket_cap: int) -> dict[str, Any]:
    """Count the rows of a selection that differ from the prepared one, and weigh them.

    Args:
        observed: The rerun's selection [S, width] int32 KV slots, -1 padded.
        expected: The prepared selection [S, width] int32.
        bucket_cap: Most keys a bucket holds.

    Returns:
        The rows, how many differ, the most ids a side a differing row lacks, the first few differing rows and
        the bucket cap, weighed by `weigh`.
    """
    differing = torch.nonzero((observed.sort(1).values != expected.sort(1).values).any(1)).flatten().tolist()
    widest = 0
    # Rows past a few hundred are not weighed for their widest side
    for row in differing[:SELECTION_ROWS_WEIGHED]:
        mine, theirs = observed[row][observed[row] >= 0], expected[row][expected[row] >= 0]
        extra = int((~torch.isin(mine, theirs)).sum())
        missing = int((~torch.isin(theirs, mine)).sum())
        widest = max(widest, extra, missing)
    return weigh(
        {
            "rows": len(expected),
            "differing": len(differing),
            "widest_side": widest,
            "first_differing": differing[:16],
            "bucket_cap": bucket_cap,
        }
    )
