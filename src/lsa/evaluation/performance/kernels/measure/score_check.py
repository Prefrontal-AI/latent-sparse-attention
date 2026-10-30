"""An exact reference for the index scores, to check the kernel's against."""

from contextlib import contextmanager
from unittest.mock import patch

import torch


def indices(size, count, device):
    """Return up to `count` evenly spaced, distinct positions in `range(size)`."""
    return torch.linspace(0, size - 1, min(size, count), device=device).long().unique()


def verify(actual, expected, bound, valid, result, strict=True):
    """Hold the valid cells of a score matrix to a reference within a per-cell bound.

    Args:
        actual: The kernel's scores.
        expected: The fp64 reference scores.
        bound: The error allowed per cell.
        valid: Cells inside each row's visible range.
        result: Where the cell count, errors and verdict go.
        strict: Raise when the bound is exceeded.

    Raises:
        AssertionError: `strict` and a cell exceeds its bound.
    """
    error = (actual.double() - expected).abs()[valid]
    bound = bound[valid].clamp_min(1e-12)
    ratio = (error / bound).max().item()
    result.update(
        checked_cells=int(valid.sum()),
        max_abs=error.max().item(),
        max_error_over_bound=ratio,
        passed=ratio <= 1,
    )
    if strict and not result["passed"]:
        raise AssertionError(f"Independent score oracle failed: {result}")


@contextmanager
def score_oracle(module, result):
    """Hold the native indexer's FP8 score kernel to an fp64 reference on sampled cells.

    Also checks that scoring a few rows alone gives the same scores as scoring them in the chunk.

    Args:
        module: The layer under test.
        result: Where the verdicts go.

    Raises:
        AssertionError: A row scored alone disagrees with the chunk.
    """
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool

    original = dsa_indexer_kpool.deep_gemm.fp8_mqa_logits

    def native(q, keys, weights, starts, lengths, **kwargs):
        """Run the kernel, then check it on sampled rows and columns."""
        actual = original(q, keys, weights, starts, lengths, **kwargs)
        k, scales = keys
        rows = indices(q.shape[0], 33, q.device)
        cols = indices(k.shape[0], 257, k.device)
        qr = q.view(torch.uint8)[rows].view(q.dtype).double()
        kc = k.view(torch.uint8)[cols].view(k.dtype).double()
        dots = torch.einsum("rhd,cd->rhc", qr, kc)
        terms = dots.relu() * weights[rows].double()[:, :, None]
        scale = scales[cols].double().reshape(-1)
        expected = terms.sum(1) * scale[None, :]
        bound = terms.abs().sum(1) * scale.abs()[None, :] * 2e-5
        valid = (cols[None, :] >= starts[rows, None]) & (cols[None, :] < lengths[rows, None])
        result["reference"] = "FP64 sum_h(relu(dot(FP8 q, FP8 k)) * weight) * key_scale"
        verify(
            actual[rows[:, None], cols],
            expected,
            bound,
            valid,
            result,
            strict=False,
        )
        result["fp64_within_nominal_fp32_bound"] = result.pop("passed")
        result["fp64_comparison"] = (
            "diagnostic: FP8 tensor-core accumulation is not a full FP32 arithmetic oracle"
        )
        # The indexer may score a chunk's rows in slices as short as one row
        small_rows = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], device=q.device) % q.shape[0]
        small_rows = small_rows.unique()
        reduced = original(
            q.view(torch.uint8)[small_rows].view(q.dtype).contiguous(),
            keys,
            weights[small_rows].contiguous(),
            starts[small_rows].contiguous(),
            lengths[small_rows].contiguous(),
            **kwargs,
        )
        result["reduced_query_rows"] = small_rows.cpu().tolist()
        result["reduced_query_scan_exact"] = torch.equal(actual[small_rows], reduced)
        result["passed"] = result["reduced_query_scan_exact"]
        if not result["passed"]:
            raise AssertionError(f"Reduced-query scan disagrees: {result}")
        return actual

    with patch.object(dsa_indexer_kpool.deep_gemm, "fp8_mqa_logits", native):
        yield


# A bf16 product accumulated in fp32 over a 64-wide dot errs by under 64 * 2^-24 of the sum of its terms' magnitudes
LSA_ACCUMULATION = 8e-6


@contextmanager
def lsa_score_oracle(indexer, result, tree=False):
    """Hold the sampled bucket scores of a flat prefill read to an fp64 reference.

    The reference is the dot of the same bf16 index query and bf16 bucket centroid, in fp64, with the fp32
    accumulation error allowed. A dense read's scores are the product the top-k reads, and a fused read's are
    the per-query scores of its listing. A read through a hierarchy ranks a beam a level at a time with no
    gathered row behind each level's top-k, so it has no scores to hold - its selection and output are checked.
    A shaped read scores each bucket by its estimated best key, which is no dot to hold - nor is it checked here.

    Args:
        indexer: The layer's `LSAIndexer`.
        result: Where the verdict goes.
        tree: Whether the read goes through a hierarchy.
    """
    selector = indexer.index.selector
    cache = selector.cache
    if tree:
        result.update(
            passed=None, skipped="a tree read's levels have no gathered rows to score independently"
        )
        yield
        return
    if cache.bucket_shapes is not None:
        result.update(
            passed=None,
            skipped="a shaped read scores each bucket's estimated best key, not the centroid's dot",
        )
        yield
        return
    from lsa.core.kernels.triton.scan_select import dense_scores

    top_k, scan_select = selector.prefill_top_k, selector.scan_select
    read_flat, gather, top_k_call, listed = (
        selector.read_flat,
        cache.gather,
        type(top_k).listed,
        type(scan_select).listed,
    )
    saved = {}
    # The centroids of each ranking, oldest first
    pending = []
    result.setdefault("passed", None)

    def capture_query(row, index_queries, *args, **kwargs):
        """Sample the index queries of a flat read, then run it."""
        rows = indices(index_queries.shape[0], 33, index_queries.device)
        saved.update(rows=rows, q=index_queries[rows].to(torch.bfloat16).double())
        return read_flat(row, index_queries, *args, **kwargs)

    def capture_keys(*args, **kwargs):
        """Sample the gathered centroids of a ranking, then return them."""
        centroids = gather(*args, **kwargs)
        cols = indices(centroids.shape[0], 257, centroids.device)
        pending.append((cols, centroids[cols].double()))
        return centroids

    def check_rows(row_scores, row_lengths) -> None:
        """Verify the sampled rows' scores [R, N] against the oldest pending ranking, with visible counts [R]."""
        cols, keys = pending.pop(0)
        expected = saved["q"] @ keys.t()
        bound = saved["q"].abs() @ keys.abs().t() * LSA_ACCUMULATION + 1e-7
        result["reference"] = "FP64 dot of BF16 operands, FP32 accumulation error bound"
        verify(row_scores[:, cols], expected, bound, cols[None, :] < row_lengths[:, None], result)

    def check_scores(picker, scores, lengths):
        """Verify a dense ranking's sampled rows, then run it."""
        if picker is top_k:
            rows = saved["rows"]
            check_rows(scores[rows], lengths[rows])
        return top_k_call(picker, scores, lengths)

    # A long flat prefill lists without its score matrix, and may fill its read without a pick
    # => its sampled rows are scored as it scores them, once a listing
    def check_listing(select, queries, centroids, lengths, shapes=None):
        """Run a fused listing, then verify its sampled rows against dense scores."""
        listing = listed(select, queries, centroids, lengths, shapes)
        if select is scan_select:
            rows = saved["rows"]
            check_rows(dense_scores(queries[rows], centroids, lengths[rows]), lengths[rows])
        return listing

    with (
        patch.object(selector, "read_flat", capture_query),
        patch.object(cache, "gather", capture_keys),
        patch.object(type(top_k), "listed", check_scores),
        patch.object(type(scan_select), "listed", check_listing),
    ):
        yield
