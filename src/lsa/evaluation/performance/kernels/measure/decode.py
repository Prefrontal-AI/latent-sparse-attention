"""The decode counterpart of the layer measurement - one token against a filled cache."""

import copy
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch


def pool_period(indexer):
    """Return how many keys the model's own indexer takes to close a pool - DSA's `index_kpool`, QSA's block of `compress_ratio`."""
    period = getattr(indexer, "index_kpool", None)
    return period if period is not None else indexer.compress_ratio


def maintenance_periods(variant):
    """Return how often each maintenance case recurs, in tokens, for the layer's indexer.

    The model's own indexer closes a pool of keys. The design's closes a cluster block every `window_size`
    keys, takes a page every `PAGE_SIZE`, and grows its hierarchy when `GROWTH_BUCKETS` buckets, or a node
    cluster block of them if that is more, are uncovered at the end of a page. PISA's closes a leaf block every
    `block_size` keys and maintains nothing else.

    Args:
        variant: The layer under test.

    Returns:
        One period per case the indexer maintains - empty for a layer that maintains nothing.
    """
    if variant.indexer is None:
        return {}
    if variant.pisa:
        return {"pool_close": variant.indexer.config.block_size}
    if not variant.lsa:
        return {"pool_close": pool_period(variant.indexer)}
    from lsa.sglang.index.hierarchy import GROWTH_BUCKETS
    from lsa.sglang.settings.design import PAGE_SIZE

    config = variant.design.indexer
    periods = {"cluster_block_close": config.window_size, "page_allocation": PAGE_SIZE}
    if config.top_level >= 1:
        buckets = max(config.at("node_cluster_block_size", 1), GROWTH_BUCKETS)
        periods["hierarchy_growth"] = -(-buckets * config.bucket_size // PAGE_SIZE) * PAGE_SIZE
    return periods


def decode_case(args, variant, snapshot, end, row):
    """Name what a decode token at `end` did, against the state saved before it.

    Args:
        args: The harness arguments, with the context the probes are offsets from.
        variant: The layer under test.
        snapshot: The state saved before the token.
        end: The token's sequence length.
        row: The request table row.

    Returns:
        The probe's offset, the prefix it reads, a flag for every case in `Kernels.CASES` and the periods the
        cases recur at.
    """
    from lsa.evaluation.constants import Kernels

    flags = dict.fromkeys(Kernels.CASES, False)
    if snapshot.state is not None:
        flags.update(snapshot.state.changed(row))
    elif variant.indexer is not None:
        flags["pool_close"] = end % pool_period(variant.indexer) == 0
    return {
        "offset": end - args.context,
        "prefix_tokens": end - 1,
        "query_tokens": 1,
        **flags,
        "periods": maintenance_periods(variant),
    }


def run_decode(args, runner, variant, req, tokens, started, source_token_count, source_hashes):
    """Prefill to just before the context, then decode the prompt's tokens and measure the layer at each target.

    The decode is teacher-forced: each step feeds the next prompt token.

    Args:
        args: The command line, whose `decode_offsets` say where to measure.
        runner: The model runner.
        variant: The layer under test.
        req: The request.
        tokens: The prompt's tokens.
        started: When the run began, from `time.perf_counter`.
        source_token_count: Tokens the prompt file had before any repeat.
        source_hashes: The source fingerprint taken at the start.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from lsa.evaluation.performance.kernels.measure.layer import measure, prepare_batch
    from lsa.evaluation.performance.kernels.measure.snapshot import clone_output, save_checkpoint
    from lsa.evaluation.performance.kernels.measure.workload import memory_admission

    module = variant.module
    if variant.lsa or variant.pisa:
        from lsa.sglang.index import registry

        if registry.INDEXERS != [module.indexer]:
            raise ValueError("Decode pilot requires exactly one sparse layer in the bootstrap")
    admission = memory_admission(
        args.chunk,
        args.context + 8,
        "qsa" if variant.qsa else args.indexer,
        args.validation_mib,
        args.memory_margin_gib,
        variant.design.indexer if variant.lsa else None,
    )
    admission["scope"] = "bootstrap prefill, conservative for decode"
    prefix = args.context - 513
    ends = list(range(args.chunk, prefix, args.chunk)) + [prefix]
    for end in ends:
        batch, schedule = prepare_batch(req, runner, end, return_schedule=True)
        runner.forward(batch)
        req.prefix_indices = runner.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :end].long()
        if end % 131072 == 0 or end == prefix:
            print(
                json.dumps(
                    {
                        "prepared_context": end,
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    checkpoints = variant.checkpoints()
    targets = {args.context + offset for offset in args.decode_offsets}
    captured = {}

    def capture_input(_module, positional, kwargs) -> None:
        """Keep the layer's arguments, with its input cloned."""
        captured["kwargs"] = variant.arguments(positional, kwargs)
        captured["kwargs"]["hidden_states"] = clone_output(captured["kwargs"]["hidden_states"])

    def capture_output(_module, positional, kwargs, output) -> None:
        """Keep a clone of the layer's output as the reference."""
        captured["output"] = clone_output(output)

    def capture_selection(_module, positional, kwargs, output) -> None:
        """Keep a clone of the selection the layer made as the reference."""
        captured["selection"] = clone_output(output)

    for end in range(prefix + 1, max(targets) + 1):
        token = tokens[end - 1]
        req.output_ids.append(token)
        schedule.input_ids = torch.tensor([token], dtype=torch.int64, device=runner.device)
        schedule.prepare_for_decode()
        batch = ForwardBatch.init_new(schedule, runner, return_hidden_states_before_norm=False)
        if not batch.forward_mode.is_decode() or batch.seq_lens_cpu.tolist() != [end]:
            raise RuntimeError(f"Expected one decode step at {end}")
        handles = []
        if end in targets:
            snapshot = variant.snapshot(batch)
            selection = variant.selection(
                runner.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :end],
                batch.positions,
                decode=True,
            )
            handles = [
                module.register_forward_pre_hook(capture_input, with_kwargs=True),
                module.register_forward_hook(capture_output, with_kwargs=True),
                variant.selection_hook(capture_selection, with_kwargs=True),
            ]
            handles += [
                part.register_forward_hook(save_checkpoint(name, captured))
                for name, part in checkpoints.items()
            ]
        with (
            selection.capture(selection.reference) if end in targets and not args.time_only else nullcontext()
        ):
            runner.forward(batch)
        for handle in handles:
            handle.remove()
        if end not in targets:
            continue
        measurement = copy.copy(args)
        measurement.context = end
        output = Path(args.output)
        measurement.output = str(output.with_name(f"{output.stem}-{end}.json"))
        measurement.decode_case = decode_case(args, variant, snapshot, end, req.kv.req_pool_idx)
        measure(
            measurement,
            variant,
            batch,
            captured,
            snapshot,
            selection,
            checkpoints,
            tokens[:end],
            time.perf_counter() - started,
            source_token_count,
            admission,
            source_hashes,
            True,
            args.context,
        )
        captured.clear()
        del snapshot, selection


@contextmanager
def decode_score_oracle(module, result):
    """Hold the native indexer's paged FP8 decode scores to an fp64 reference on sampled columns.

    Args:
        module: The layer under test.
        result: Where the verdict goes.
    """
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool

    from lsa.evaluation.performance.kernels.measure.score_check import indices, verify

    original = dsa_indexer_kpool.deep_gemm.fp8_paged_mqa_logits

    def native(q, cache, weights, lengths, pages, schedule, max_length, **kwargs):
        """Run the kernel, then check sampled columns against keys read from the paged bytes."""
        actual = original(q, cache, weights, lengths, pages, schedule, max_length, **kwargs)
        if q.shape[0] != 1 or q.shape[1] != 1:
            raise RuntimeError(f"Decode expects one query row, got {tuple(q.shape[:2])}")
        count = int(lengths[0, 0])
        cols = indices(count, 257, q.device)
        packed = cache.view(torch.uint8).reshape(cache.shape[0], -1)
        physical = pages[0, cols // 64].long()
        offsets = cols % 64
        keys = (
            packed[
                physical[:, None],
                offsets[:, None] * 128 + torch.arange(128, device=q.device),
            ]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        scale = (
            packed[
                physical[:, None],
                8192 + offsets[:, None] * 4 + torch.arange(4, device=q.device),
            ]
            .contiguous()
            .view(torch.float32)
            .flatten()
            .double()
        )
        dots = q[0, 0].double() @ keys.double().t()
        terms = dots.relu() * weights[0].double()[:, None]
        expected = terms.sum(0) * scale
        bound = terms.abs().sum(0) * scale.abs() * 2**-7 + 1e-5
        result["reference"] = (
            "FP64 paged FP8 score, independent byte addressing; conservative 2^-7 weighted-sum error bound"
        )
        result["checked_columns"] = cols.cpu().tolist()
        verify(
            actual[0, cols],
            expected,
            bound,
            torch.ones_like(expected, dtype=torch.bool),
            result,
        )
        return actual

    with patch.object(dsa_indexer_kpool.deep_gemm, "fp8_paged_mqa_logits", native):
        yield


@contextmanager
def lsa_decode_score_oracle(indexer, result, tree=False):
    """Hold a flat decode step's bucket scores to an fp64 reference, on a sample of its buckets.

    A step through a hierarchy scores a beam a level at a time and has no one row of bucket scores to hold.

    Args:
        indexer: The layer's `LSAIndexer`.
        result: Where the verdict goes.
        tree: Whether the step goes through a hierarchy.
    """
    from lsa.evaluation.performance.kernels.measure.score_check import LSA_ACCUMULATION, indices, verify
    from lsa.sglang.index import selector as selector_module

    result.setdefault("passed", None)
    if tree:
        result.update(skipped="a tree step's levels have no gathered rows to score independently")
        yield
        return
    if indexer.index.selector.cache.bucket_shapes is not None:
        result.update(skipped="a shaped step scores each bucket's estimated best key, not the centroid's dot")
        yield
        return
    original = selector_module.scan_paged

    def checked(index_queries, bucket_centroids, pages, counts, page_size, max_rows):
        """Run the paged scan, then check sampled buckets against a bf16 dot in fp64."""
        actual = original(index_queries, bucket_centroids, pages, counts, page_size, max_rows)
        if index_queries.shape[0] != 1:
            raise RuntimeError(f"Decode expects one query row, got {index_queries.shape[0]}")
        cols = indices(int(counts[0]), 257, index_queries.device)
        physical = pages[0, cols // page_size].long() * page_size + cols % page_size
        products = bucket_centroids[physical].double() * index_queries[0].double()
        expected = products.sum(1)
        bound = products.abs().sum(1) * LSA_ACCUMULATION + 1e-7
        result["reference"] = "FP64 BF16 paged dot, independent row addressing; FP32 accumulation error bound"
        result["checked_columns"] = cols.cpu().tolist()
        verify(actual[0, cols], expected, bound, torch.ones_like(expected, dtype=torch.bool), result)
        return actual

    with patch.object(selector_module, "scan_paged", checked):
        yield
