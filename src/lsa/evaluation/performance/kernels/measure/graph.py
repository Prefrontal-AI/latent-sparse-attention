"""Capture and compare CUDA graphs of the measured layer, against the same layer run eagerly."""

import time
import traceback
from contextlib import contextmanager

import torch


def bind_capture_metadata(variant, backend, batch) -> None:
    """Bind the backend's graph metadata to the batch before a decode graph is captured.

    A graph keeps the address of every buffer its kernels read at capture.
    Some backends keep the metadata of an eager step in tensors it allocates fresh on each call,
    and switch to their graph buffers only on a capture-mode init, so a capture taken after an eager step
    reads the eager tensors and a later eager step's allocation overwrites them.
    The server captures after this init for every attention layer.
    A linear layer's backend takes no such init here.

    Args:
        variant: The measured layer's variant, whose `linear` says whether it is a linear attention layer.
        backend: The attention backend, whose `init_cuda_graph_state` has run.
        batch: The decode batch the graph is captured over.
    """
    if variant.linear:
        return
    backend.init_forward_metadata_out_graph(batch, in_capture=True)


def check_replay(variant, execute, snapshot, compare, selection, replayed) -> None:
    """Hold a graph replay's output bitwise to the eager step over the same state and selection.

    A replay that reads a freed buffer can match for a while, since the block keeps what it held until another
    allocation reuses it.
    So every timed replay is held to the eager step, after its timing has been taken.

    Args:
        variant: The measured layer's variant, which pins the eager step to the replay's selection.
        execute: Runs the layer eagerly, metadata included.
        snapshot: The state the replay started from, restored before the eager step and after it.
        compare: Raises when two outputs differ in any bit.
        selection: The slots the replay selected.
        replayed: The replay's output.
    """
    snapshot.restore()
    with variant.pinned_selection(selection.clone()):
        compare(execute(), replayed)
    snapshot.restore()


def compare_graphs(
    args,
    variant,
    backend,
    batch,
    attention,
    execute,
    snapshot,
    reference_state,
    observed,
    selection_check,
    compare,
    zero,
):
    """Capture a prefill chunk of the layer as a breakable CUDA graph and time it against eager.

    Replays are held to the eager step over the same state and selection.
    A failure is recorded in the result rather than raised, and the state is restored either way.

    Args:
        args: The command line.
        variant: The layer under test.
        backend: The attention backend.
        batch: The forward batch the layer ran on.
        attention: Calls the layer.
        execute: Runs the layer eagerly, metadata included.
        snapshot: Put back before each run.
        reference_state: The state a run must leave unchanged.
        observed: Holds the selection of the latest run.
        selection_check: The check that a selection is the true top-k.
        compare: Raises when two outputs differ in any bit.
        zero: The zero allocator.

    Returns:
        The comparison record - `status` is `validated` or `unsupported_or_invalid`.
    """
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        enable_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        set_tc_piecewise_forward_context,
    )

    module = variant.module

    @contextmanager
    def context():
        """Enter the contexts a breakable graph captures and replays under."""
        with (
            enable_breakable_cuda_graph(),
            set_tc_piecewise_forward_context(
                batch,
                [None] * module.layer_id + [module.attn_mqa],
                None,
                [],
                [],
                dsa_indexers=[None] * module.layer_id + [module.indexer],
                num_tokens=args.chunk,
                raw_num_tokens=args.chunk,
            ),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            yield

    result = {
        "backend": "breakable",
        "bucket_tokens": args.chunk,
        "metadata_in_graph": False,
        "samples": [],
    }
    phase = "warmup"
    graph = BreakableCUDAGraph()
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), context():
            for _ in range(args.warmup):
                snapshot.restore()
                zero._pointer = 0
                backend.init_forward_metadata(batch)
                attention()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        snapshot.restore()
        zero._pointer = 0
        with forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata(batch)
        torch.cuda.synchronize()
        phase = "capture"
        started = time.perf_counter()
        with (
            context(),
            BreakableCUDAGraphCapture(
                graph,
                pool=torch.cuda.graph_pool_handle(),
                stream=stream,
            ),
        ):
            output = attention()
            selected = observed["selection"]
        torch.cuda.synchronize()
        result.update(
            capture_s=time.perf_counter() - started,
            segments=len(graph._segments),
            eager_breaks=len(graph._break_fns),
        )

        def replay():
            """Replay the graph and return its output."""
            with context():
                graph.replay()
            observed["selection"] = selected
            return output

        phase = "replay_validation"
        for _ in range(2):
            snapshot.restore()
            with forward_context(ForwardContext(attn_backend=backend)):
                backend.init_forward_metadata(batch)
            replay()
            torch.cuda.synchronize()
            if args.time_only:
                continue
            selection_check.check(selected)
            reference_state.check()
            expected = output[0].clone() if isinstance(output, tuple) else output.clone()
            snapshot.restore()
            with variant.pinned_selection(selected):
                eager = execute()
            compare(eager[0] if isinstance(eager, tuple) else eager, expected)
        if args.time_only:
            result["validation"] = "time only - replayed without the selection and output checks"
        else:
            result["validation"] = "exact output with the same selection; exact cache and host state"
        phase = "timing"
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for event in events:
            event.record()
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            modes = (False, True) if repeat % 2 == 0 else (True, False)
            for use_graph in modes:
                snapshot.restore()
                zero._pointer = 0
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with forward_context(ForwardContext(attn_backend=backend)):
                    events[0].record()
                    backend.init_forward_metadata(batch)
                    events[1].record()
                    actual = replay() if use_graph else attention()
                    events[2].record()
                events[2].synchronize()
                wall_ms = (time.perf_counter() - started) * 1000
                result["samples"].append(
                    {
                        "repeat": repeat,
                        "mode": "graph" if use_graph else "eager",
                        "module_ms": events[1].elapsed_time(events[2]),
                        "step_gpu_ms": events[0].elapsed_time(events[2]),
                        "step_wall_ms": wall_ms,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "topk_cutoff_valid": None
                        if args.time_only
                        else selection_check.check(observed["selection"]),
                    }
                )
                if not torch.isfinite(actual[0] if isinstance(actual, tuple) else actual).all():
                    raise RuntimeError("Non-finite attention output")
                reference_state.check()
        result["status"] = "validated"
    except Exception as error:
        result.update(
            status="unsupported_or_invalid",
            phase=phase,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        reference_state.restore()
    return result


def capture_free_indexer(variant, backend, batch, attention, snapshot, zero, stream, pinned):
    """Capture the decode step again with the indexer doing no work, handing the attention a fixed list.

    Runs inside the full graph's capture mode, on its stream, so the two graphs differ only by the indexer.

    Args:
        variant: The measured layer's variant, whose indexer is stood in for.
        backend: The attention backend, its graph state bound to the batch.
        batch: The decode batch both graphs are captured over.
        attention: Runs the layer without its metadata.
        snapshot: The state each step starts from.
        zero: The layer's bump allocator, rewound before each step.
        stream: The capture stream.
        pinned: The selection the stand-in hands over, filled before every replay.

    Returns:
        The captured graph and the output buffer it writes.
    """
    with variant.free_indexer(pinned) as served, torch.cuda.stream(stream):
        snapshot.restore()
        zero._pointer = 0
        backend.init_forward_metadata_out_graph(batch)
        backend.init_forward_metadata_in_graph(batch)
        attention()
        torch.cuda.synchronize()
        snapshot.restore()
        zero._pointer = 0
        backend.init_forward_metadata_out_graph(batch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
            backend.init_forward_metadata_in_graph(batch)
            output = attention()
        torch.cuda.synchronize()
    if not served:
        raise RuntimeError("The free indexer's selection did not reach the attention")
    return graph, output


def differing_slots(actual, expected) -> int:
    """Count the slots one selection holds and another does not, row by row, in whatever order they are listed.

    The top-k's order is not stable from one run to the next, and an approximate indexer's set is not either,
    even eagerly over the same state.

    Args:
        actual: One selection [S, width] int32 KV slots, -1 padded.
        expected: The selection it is read against.

    Returns:
        The slots of `actual` missing from `expected`, summed over the rows.
    """
    missing = 0
    for mine, theirs in zip(actual.tolist(), expected.tolist(), strict=True):
        missing += len({slot for slot in mine if slot >= 0} - {slot for slot in theirs if slot >= 0})
    return missing


def capture_indexer_call(variant):
    """Return a hook that keeps the indexer's last call, and the dict it keeps it in.

    The call's tensors are copied, since the layer's own are freed and reused once the step is done.

    Args:
        variant: The measured layer's variant, whose indexer is called.

    Returns:
        The hook's handle, and a dict whose `call` is the positional and keyword arguments of the last call.
    """
    kept = {}

    def keep(_module, positional, kwargs) -> None:
        """Keep a copy of the indexer's arguments."""

        def copy(value):
            """Return a tensor cloned, anything else as it is."""
            return value.clone() if isinstance(value, torch.Tensor) else value

        kept["call"] = (
            [copy(value) for value in positional],
            {key: copy(value) for key, value in kwargs.items()},
        )

    return variant.indexer_call_hook(keep), kept


def capture_indexer_only(variant, backend, batch, snapshot, stream, call):
    """Capture the indexer's call alone, on the inputs the step handed it, its metadata initialised outside.

    Runs inside the full graph's capture mode, on its stream, so the indexer takes the path it takes in the step.

    Args:
        variant: The measured layer's variant, whose indexer is captured.
        backend: The attention backend, its graph state bound to the batch.
        batch: The decode batch the graphs are captured over.
        snapshot: The state each step starts from.
        stream: The capture stream.
        call: The positional and keyword arguments of the step's indexer call.

    Returns:
        The captured graph and the selection buffer it writes.
    """
    positional, kwargs = call
    with torch.cuda.stream(stream):
        snapshot.restore()
        backend.init_forward_metadata_out_graph(batch)
        backend.init_forward_metadata_in_graph(batch)
        variant.call_indexer(*positional, **kwargs)
        torch.cuda.synchronize()
        snapshot.restore()
        backend.init_forward_metadata_out_graph(batch)
        backend.init_forward_metadata_in_graph(batch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
            output = variant.call_indexer(*positional, **kwargs)
        torch.cuda.synchronize()
    return graph, output


def compare_decode_graphs(
    args,
    variant,
    backend,
    batch,
    attention,
    execute,
    snapshot,
    reference_state,
    observed,
    selection_check,
    compare,
    zero,
    cache_flush=None,
):
    """Capture one decode step of the layer as a full CUDA graph and time it against eager.

    The arguments are those of `compare_graphs`, plus the buffer to zero before each run.
    Attention metadata is built outside the graph and maintenance runs inside the step time.
    Every timed replay is held bitwise to the eager step.

    Args:
        args: The command line.
        variant: The layer under test.
        backend: The attention backend.
        batch: The decode batch.
        attention: Calls the layer.
        execute: Runs the layer eagerly, metadata included.
        snapshot: Put back before each run.
        reference_state: The state a run must leave unchanged.
        observed: Holds the selection of the latest run.
        selection_check: The check that a selection is the true top-k.
        compare: Raises when two outputs differ in any bit.
        zero: The zero allocator.
        cache_flush: Buffer zeroed before each run to evict the cache, or None for a warm cache.

    Returns:
        The comparison record - `status` is `validated` or `unsupported_or_invalid`.
    """
    from sglang.srt.distributed import graph_capture
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    from lsa.evaluation.performance.kernels.measure.snapshot import clone_output

    dispatch = variant.dispatch
    result = {
        "backend": "full CUDA graph, one attention layer",
        "production_capture_mode": True,
        "metadata_outside_graph": True,
        "maintenance_outside_graph_and_in_step_time": True,
        "samples": [],
    }
    # A layer whose selection is its indexer's output can also be replayed with the indexer doing no work
    # => the indexer's cost under the graph is the difference, overlap with the rest of the layer included
    # Its indexer is also captured alone, on the step's inputs
    # => the free graph and the indexer alone add up to the step less whatever the indexer overlapped
    free = variant.frees_indexer
    result["free_indexer"] = free
    pinned = free_graph = free_output = indexer_graph = indexer_output = None
    # The list the attention reads, which Qwen's LSA hands over inside its backend and no module outputs
    # => observed["selection"] keeps what the harness checks, and this keeps what the free replay pins
    listing = {}

    def save_listing(_indexer, _arguments, output) -> None:
        """Keep the list the attention was handed."""
        listing["selection"] = output

    handle = variant.listing_hook(save_listing)
    phase = "warmup"
    graph = torch.cuda.CUDAGraph()
    try:
        backend.init_cuda_graph_state(max_bs=1, max_num_tokens=1)
        bind_capture_metadata(variant, backend, batch)
        with (
            model_capture_mode(),
            graph_capture() as capture,
            forward_context(ForwardContext(attn_backend=dispatch)),
        ):
            stream = capture.stream
            if free:
                call_hook, indexer_call = capture_indexer_call(variant)
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    snapshot.restore()
                    zero._pointer = 0
                    backend.init_forward_metadata_out_graph(batch)
                    backend.init_forward_metadata_in_graph(batch)
                    attention()
            torch.cuda.synchronize()
            # The warm-up's selection is the step's, since every warm-up step starts from the same state
            # => the stand-in indexer hands over a real list from its first call
            if free:
                call_hook.remove()
                pinned = listing["selection"].clone()
            snapshot.restore()
            zero._pointer = 0
            backend.init_forward_metadata_out_graph(batch)
            torch.cuda.synchronize()
            phase = "capture"
            started = time.perf_counter()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
                backend.init_forward_metadata_in_graph(batch)
                output = attention()
                selected = observed["selection"]
                graph_listing = listing.get("selection")
            torch.cuda.synchronize()
            result["capture_s"] = time.perf_counter() - started
            if free:
                phase = "free_indexer_capture"
                free_graph, free_output = capture_free_indexer(
                    variant, backend, batch, attention, snapshot, zero, stream, pinned
                )
                phase = "indexer_only_capture"
                indexer_graph, indexer_output = capture_indexer_only(
                    variant, backend, batch, snapshot, stream, indexer_call["call"]
                )

        def replay():
            """Replay the graph and return its output."""
            graph.replay()
            observed["selection"] = selected
            return output

        def replay_free():
            """Replay the free-indexer graph and return its output."""
            free_graph.replay()
            observed["selection"] = pinned if variant.hooked else selected
            return free_output

        def indexer_metadata() -> None:
            """Initialise the graph metadata the indexer reads, outside its graph."""
            # The indexer reads the graph metadata the step initialises inside its graph
            # => initialised here, outside the indexer's graph and its timing, in the mode the step captured in
            backend.init_forward_metadata_out_graph(batch)
            with model_capture_mode():
                backend.init_forward_metadata_in_graph(batch)

        def replay_indexer():
            """Replay the indexer's graph and return its selection."""
            indexer_graph.replay()
            observed["selection"] = indexer_output if variant.hooked else selected
            return indexer_output

        phase = "replay_validation"
        for _ in range(2):
            snapshot.restore()
            with forward_context(ForwardContext(attn_backend=dispatch)):
                backend.init_forward_metadata_out_graph(batch)
                replay()
            torch.cuda.synchronize()
            expected = clone_output(output)
            # The list the replay handed the attention, where the layer shows one
            replayed = (graph_listing if graph_listing is not None else selected).clone()
            chosen = selected.clone()
            if not args.time_only:
                selection_check.check(selected)
                reference_state.check()
            # One eager step from the same state, unpinned
            # => how far the replay's selection strays from what eager selects
            snapshot.restore()
            listing.pop("selection", None)
            execute()
            torch.cuda.synchronize()
            eager = listing.get("selection", observed["selection"])
            if replayed.numel() and eager.numel():
                differing = differing_slots(
                    replayed.reshape(-1, replayed.shape[-1]), eager.reshape(-1, eager.shape[-1])
                )
                result["graph_selection_differing_slots"] = max(
                    differing, result.get("graph_selection_differing_slots", 0)
                )
            observed["selection"] = selected
            if args.time_only:
                continue
            snapshot.restore()
            with variant.pinned_selection(chosen):
                compare(execute(), expected)
            if free:
                # The free graph attends over the full graph's selection, so its output is the full graph's
                pinned.copy_(graph_listing)
                snapshot.restore()
                with forward_context(ForwardContext(attn_backend=dispatch)):
                    backend.init_forward_metadata_out_graph(batch)
                    compare(replay_free(), expected)
                torch.cuda.synchronize()
                # The indexer alone, on the step's inputs, selects what the step selected
                snapshot.restore()
                with forward_context(ForwardContext(attn_backend=dispatch)):
                    indexer_metadata()
                    result["indexer_only_differing_slots"] = differing_slots(replay_indexer(), graph_listing)
                torch.cuda.synchronize()
        if args.time_only:
            result["validation"] = (
                "time only - no selection check; every timed replay bitwise equal to the eager step"
            )
        else:
            result["validation"] = (
                "bitwise output with the same selected slots, for the validation replays and every timed replay;"
                " exact cache and host state"
            )
        phase = "timing"
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for event in events:
            event.record()
        torch.cuda.synchronize()

        def trial(repeat, mode):
            """Time one eager, graph or free-indexer step from restored state, profiled when `repeat` is -1."""
            use_graph = mode != "eager"
            snapshot.restore()
            if cache_flush is not None:
                cache_flush.zero_()
            zero._pointer = 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_bytes = torch.cuda.memory_allocated()
            before = torch.cuda.memory_stats()
            started = time.perf_counter()
            with forward_context(ForwardContext(attn_backend=dispatch)):
                events[0].record()
                if repeat == -1:
                    torch.cuda.nvtx.range_push("attention metadata")
                if mode == "indexer_only":
                    indexer_metadata()
                elif use_graph:
                    backend.init_forward_metadata_out_graph(batch)
                else:
                    backend.init_forward_metadata(batch)
                if repeat == -1:
                    torch.cuda.nvtx.range_pop()
                events[1].record()
                if repeat == -1:
                    torch.cuda.nvtx.range_push("attention module")
                actual = {
                    "eager": attention,
                    "graph": replay,
                    "free_indexer": replay_free,
                    "indexer_only": replay_indexer,
                }[mode]()
                if repeat == -1:
                    torch.cuda.nvtx.range_pop()
                events[2].record()
            events[2].synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            after = torch.cuda.memory_stats()
            sample = {
                "repeat": repeat,
                "mode": mode,
                "module_ms": events[1].elapsed_time(events[2]),
                "metadata_ms": events[0].elapsed_time(events[1]),
                "step_gpu_ms": events[0].elapsed_time(events[2]),
                "step_wall_ms": elapsed,
                "allocated_before_bytes": baseline_bytes,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocator_retries": after["num_alloc_retries"] - before["num_alloc_retries"],
                "allocator_ooms": after["num_ooms"] - before["num_ooms"],
                "topk_cutoff_valid": None if args.time_only else selection_check.check(observed["selection"]),
            }
            if not torch.isfinite(actual[0] if isinstance(actual, tuple) else actual).all():
                raise RuntimeError("Non-finite attention output")
            # A free indexer writes no index key and the indexer alone no attention cache
            # => their replays leave the state short of the reference, and only their outputs are held
            if mode in ("eager", "graph"):
                reference_state.check()
                sample["cache_and_index_state_exact"] = True
            # The indexer alone does the step's indexing whatever it lists
            # => how far its list strays from the step's is recorded rather than held
            if mode == "indexer_only":
                sample["selection_differing_slots"] = differing_slots(actual, pinned)
            elif use_graph:
                check_replay(variant, execute, snapshot, compare, observed["selection"], actual)
            return sample

        modes = ["eager", "graph"] + (["free_indexer", "indexer_only"] if free else [])
        for repeat in range(args.repeats):
            # Each mode takes each place in the order equally often, so none always runs on a warmer card
            shift = repeat % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                result["samples"].append(trial(repeat, mode))
        if args.profile_graph:
            torch.cuda.cudart().cudaProfilerStart()
            result["profile_sample"] = trial(-1, "graph")
            torch.cuda.cudart().cudaProfilerStop()
        # A time-only run holds every timed replay to the eager step too, so a pass here is a verdict on the graph
        result["reference_order_output_exact"] = True
        result["status"] = "validated"
    except (RuntimeError, ValueError, AssertionError) as error:
        result.update(
            status="unsupported_or_invalid",
            phase=phase,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        handle.remove()
        reference_state.restore()
    return result
