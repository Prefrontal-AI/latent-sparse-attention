"""Time and validate one real attention layer in eager mode - sparse with the model's indexer or the design's,
full attention without one, or a linear attention layer. GLM-5.3-Flash (MLA, KDA) or Qwen3.5 (GQA, Gated
DeltaNet), each cut to its first layers with real weights.

    python -m lsa.evaluation.performance.kernels.measure.layer --model-path <cut> --indexer lsa \
        --indexer-config <settings.json> ...  (or --indexer pisa)

Run inside the serving container from the checkout root, never imported. Loads a checkpoint, builds one
batch, and measures a single attention layer against a reference run of the same layer - so a kernel change
is judged on the module it changed rather than on a whole server.

The arithmetic it leans on is `recorder`, `selection`, `snapshot` and `workload` beside it, which is the half
a CPU can test. This file is the half that needs a GPU and a checkpoint.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from array import array
from contextlib import nullcontext
from pathlib import Path

import torch

import lsa
from lsa import io
from lsa.evaluation.constants import Family
from lsa.evaluation.performance.kernels.family import checkpoint_family
from lsa.evaluation.performance.kernels.measure.selection import selection_rows
from lsa.evaluation.performance.kernels.measure.snapshot import (
    clone_output,
    compare,
    output_error,
    save_checkpoint,
    tensor_digest,
)
from lsa.evaluation.performance.telemetry import Sampler, sample_path
from lsa.sglang.constants import Plugin, Server
from lsa.sglang.launch import merge_override


def git_output(root, *arguments):
    """Return what git prints for this checkout, or None where it cannot be asked.

    A worktree's `.git` is a file pointing into the main repository, and a container that mounts the
    worktree alone cannot follow it - so provenance is best effort and a measurement is not lost over it.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def source_hashes():
    """Fingerprint this package, the cut beside it and the serving files it times, keyed by their path in the repository.

    The files are read where they were imported from, which inside the container is the checkout the
    harness installed - so the record names the code that ran, wherever it was installed to.
    """
    package = Path(lsa.__file__).resolve().parent
    measure = Path(__file__).resolve().parent
    paths = sorted(measure.glob("*.py")) + sorted((measure.parent / "data").glob("*.py"))
    paths += [
        package / "sglang" / name
        for name in (
            "index/probe.py",
            "models/glm5_next/indexer.py",
            "models/glm5_next/pisa_indexer.py",
            "index/pisa.py",
            "index/layer_index.py",
            "index/selector.py",
            "index/bucket_cache.py",
            "index/hierarchy.py",
            "index/clustering.py",
            "models/qwen3_5/backend.py",
            "models/qwen3_5/index_space.py",
        )
    ]
    return {f"src/lsa/{p.relative_to(package)}": hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


# What upstream sglang picks per GPU generation when no backend is named - `get_default_attn_backend` for the dense
# layers and the DSA default of `arg_groups/overrides.py` for GLM's bf16 sparse attention. A baseline method runs in
# upstream's layout; a deviation is named with `--prefill-attention-backend` / `--decode-attention-backend`
# (`run.args` in a config) and the record says so.
UPSTREAM_MLA = {"hopper": ("fa3", "fa3"), "blackwell": ("flashinfer", "flashinfer")}
UPSTREAM_MHA = {"hopper": ("fa3", "fa3"), "blackwell": ("trtllm_mha", "trtllm_mha")}
# Where the model cannot run upstream's pick, the cell it takes instead - one per exception, each cited
# Hopper, GLM dense decode: stock fa3's latent decode does not run on GLM's rope-free latent, so
# it decodes on FlashMLA's dense kernel; prefill stays on fa3.
# Blackwell, GLM dense: upstream's FlashInfer MLA fails on the rope-free latent in decode at either page size (`cannot
# reshape tensor of 0 elements into shape [-1, 64, 0]`; its prefill probe measured nothing), and the other decodes fail
# the same way (fa4, trtllm_mla, cutedsl_mla) or are SM90's (flashmla). The fastest validated alternatives are
# trtllm_mla for prefill (29.5 ms against triton's 1965 ms at a 128k chunk of 8,192, output exact) and triton for
# decode (0.855 ms graph-replayed at 128k, output and state exact, graph validated)
MODEL_REQUIRED_MLA = {"hopper": (None, "flashmla"), "blackwell": ("trtllm_mla", "triton")}
UPSTREAM_DSA = {"hopper": ("flashmla_sparse", "fa3"), "blackwell": ("flashmla_sparse", "trtllm")}


def device_generation():
    """Return `blackwell` on sm_100 and up and `hopper` otherwise - the two generations sglang picks backends for."""
    blackwell = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
    return "blackwell" if blackwell else "hopper"


def overrides(args):
    """Return the prefill and decode attention backends the run names, each None when it names none."""
    return getattr(args, "prefill_attention_backend", None), getattr(args, "decode_attention_backend", None)


def dense_backends(args, upstream):
    """Return the engine arguments of a dense layer, on upstream's backends unless the run names its own."""
    prefill, decode = overrides(args)
    prefill, decode = prefill or upstream[0], decode or upstream[1]
    if prefill == decode:
        return {"attention_backend": prefill}
    return {"prefill_attention_backend": prefill, "decode_attention_backend": decode}


def engine_backends(args):
    """Return the attention backend arguments as served, per family and indexer.

    The baseline methods run upstream sglang's per-GPU choice: dense MLA (GLM read densely) on FA3 on Hopper (decode on FlashMLA, which the model requires) and FlashInfer's
    MLA on Blackwell (GLM's latent runs neither there, see MODEL_REQUIRED_MLA), dense MHA (Qwen) on FA3 and trtllm_mha, and GLM's own sparse attention on FlashMLA's sparse prefill
    with FA3 (Hopper) or trtllm (Blackwell) decode. `--prefill-attention-backend` and `--decode-attention-backend`
    replace either for those methods. QSA is served with no backend named - the image's QSA backend builds its own for the attention layers
    and sets their page of 64. The design's methods (LSA, PISA) are ours: GLM reads its selection through the sparse kernels
    the run names, and Qwen's list is read on triton (Blackwell) or FA3 at a page of one slot, which the design's backend
    needs - the linear layer is measured in the served model with its own indexer.

    Raises:
        ValueError: The run names a backend for a method that has none to choose.
    """
    generation = device_generation()
    qwen = checkpoint_family(args.model_path) == Family.QWEN3_5
    if args.indexer == "full":
        if qwen:
            return dense_backends(args, UPSTREAM_MHA[generation])
        required = MODEL_REQUIRED_MLA.get(generation, (None, None))
        upstream = UPSTREAM_MLA[generation]
        return dense_backends(args, tuple(r or u for r, u in zip(required, upstream, strict=True)))
    if any(overrides(args)) and (qwen or args.indexer != "native"):
        raise ValueError(
            f"the {args.indexer} method is not a baseline and has no upstream backend to replace"
        )
    if qwen:
        if args.indexer == "native":
            return {}
        # An LSA or PISA list is a set of single slots, so the design's backend refuses any page but one
        backend = "triton" if generation == "blackwell" else "fa3"
        return {"attention_backend": backend, **({"page_size": 1} if args.indexer in ("lsa", "pisa") else {})}
    if args.indexer == "native":
        prefill, decode = (
            named or default for named, default in zip(overrides(args), UPSTREAM_DSA[generation], strict=True)
        )
    else:
        prefill, decode = args.dsa_prefill_backend, args.dsa_decode_backend
    return {"attention_backend": "dsa", "dsa_prefill_backend": prefill, "dsa_decode_backend": decode}


def backend_source(args) -> str:
    """Say where the backends come from - `override`, `upstream`, `design` for LSA and PISA, `fork` for QSA."""
    if any(overrides(args)):
        return "override"
    if args.indexer in ("lsa", "pisa"):
        return "design"
    if args.indexer == "native" and checkpoint_family(args.model_path) == Family.QWEN3_5:
        return "fork"
    return "upstream"


def backend_names(args, variant):
    """Return the backend each phase ran on, by name - the engine's own choice where it named one."""
    if variant.kind != "full":
        return variant.backends()
    chosen = engine_backends(args)
    shared = chosen.get("attention_backend")
    return {
        "prefill": chosen.get("prefill_attention_backend", shared),
        "decode": chosen.get("decode_attention_backend", shared),
    }


def backend_classes(backend):
    """Return the classes behind a layer's backend - a hybrid one holds a backend per phase."""
    prefill, decode = (
        getattr(backend, "prefill_backend", backend),
        getattr(backend, "decode_backend", backend),
    )
    return {"prefill": type(prefill).__name__, "decode": type(decode).__name__}


def model_override(args):
    """The checkpoint's config overrides the layer is built under, as JSON - with the indexer's settings block in it.

    GLM read densely is built with a null `index_topk`, which sglang reads as a model without DSA, so its attention
    layer builds no indexer and its KV pool no index keys.
    """
    override = json.loads(args.model_override or "{}")
    # The plugin reads its settings from the model config, so the harness hands it the block as a server's command line does
    if args.indexer_config:
        override = merge_override(override, {Plugin.BLOCK: io.read_json(args.indexer_config)})
    if args.indexer == "full" and checkpoint_family(args.model_path) == Family.GLM5_NEXT:
        override.setdefault(Server.TEXT_CONFIG, {})["index_topk"] = None
    return json.dumps(override)


def prepare_batch(req, runner, end, return_schedule=False):
    """Build the extend batch that prefills a request up to `end`.

    Args:
        req: The request, its prefix already in the cache.
        runner: The model runner.
        end: Context the batch ends at.
        return_schedule: Also return the schedule batch, for decode steps to continue from.

    Returns:
        The forward batch, or the forward batch and the schedule batch.
    """
    from sglang.benchmark.one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    req.full_untruncated_fill_ids = req.origin_input_ids[:end]
    req.set_extend_range(len(req.prefix_indices), end)
    tree = TreeCacheNamespace(
        page_size=runner.page_size,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    if batch.input_ids is None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device)
        batch.prefill_input_ids_cpu = None
    forward = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    return (forward, batch) if return_schedule else forward


@torch.inference_mode()
def run(args):
    """Load the cut model, prefill the prompt chunk by chunk and measure the layer at each recorded context.

    Args:
        args: The parsed command line.
    """
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import configure_logger

    from lsa.evaluation.performance.kernels.measure.variant import Variant
    from lsa.evaluation.performance.kernels.measure.workload import memory_admission, score_rows

    started = time.perf_counter()
    initial_source_hashes = source_hashes()
    torch.manual_seed(0)
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # No server entry point runs here
    # => load the plugin that arms the design's indexer (LSA or PISA) the way serving does
    if args.indexer in ("lsa", "pisa"):
        from sglang.srt.plugins import load_plugins

        load_plugins()
    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=True,
        tp_size=args.tp_size,
        **engine_backends(args),
        # trtllm_mla takes bf16 under that spelling alone
        kv_cache_dtype="bf16" if "trtllm_mla" in engine_backends(args).values() else "bfloat16",
        context_length=args.context + args.chunk,
        max_total_tokens=args.context + args.chunk,
        chunked_prefill_size=args.chunk,
        max_running_requests=1,
        mem_fraction_static=args.mem_fraction,
        disable_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        disable_radix_cache=True,
        # FlashInfer's autotune runs on sm_100 and up and forwards a dummy batch with no request ids, which LSA's
        # maintenance refuses - off on every method, as every B300 server runs
        disable_flashinfer_autotune=torch.cuda.get_device_capability()[0] >= 10,
        log_level="warning",
        json_model_override_args=model_override(args),
    )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    # One process on one device - load_model reads its rank and device from what is published
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    configure_logger(server_args)
    wrapper, tokenizer = load_model(server_args, PortArgs.init_new(server_args), 0, 0)
    runner = wrapper.torch_runner
    layers = runner.model.model.layers
    variant = Variant(runner, args)
    module = variant.module
    text = Path(args.prompt_file).read_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    source_token_count = len(tokens)
    required_tokens = args.context + 8 if args.phase == "decode" else args.context
    if len(tokens) < required_tokens:
        if not args.repeat_prompt:
            raise ValueError(
                f"Prompt contains {len(tokens)} tokens; need {args.context}; --repeat-prompt permits a repeated-token workload"
            )
        tokens = tokens * ((required_tokens + len(tokens) - 1) // len(tokens))
    tokens = tokens[:required_tokens]
    req = Req(
        rid="attention-layer-pilot",
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1024),
    )
    req.logprob_start_len = -1
    if args.phase == "decode":
        from lsa.evaluation.performance.kernels.measure.decode import run_decode

        run_decode(
            args,
            runner,
            variant,
            req,
            tokens,
            started,
            source_token_count,
            initial_source_hashes,
        )
        torch.distributed.destroy_process_group()
        return
    captured = {}
    config = variant.design.indexer if variant.lsa else None
    checked_rows = score_rows(
        args.chunk, args.context, args.indexer, args.validation_mib, config.bucket_size if config else None
    )
    admission = memory_admission(
        args.chunk,
        args.context,
        "qsa" if variant.qsa else args.indexer,
        args.validation_mib,
        args.memory_margin_gib,
        config,
    )
    print(
        json.dumps(
            {
                "memory_admission": admission,
                "validation_rows": len(checked_rows) if checked_rows is not None else args.chunk,
            }
        ),
        flush=True,
    )

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

    checkpoints = variant.checkpoints()

    first_measure = max(2 * args.chunk, args.context - (args.window_chunks - 1) * args.chunk)
    for end in range(args.chunk, args.context + 1, args.chunk):
        batch = prepare_batch(req, runner, end)
        handles = []
        if end >= first_measure:
            snapshot = variant.snapshot(batch)
            selection_check = variant.selection(
                runner.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :end],
                batch.positions,
                checked_rows,
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
        checking = end >= first_measure and not args.time_only
        with selection_check.capture(selection_check.reference) if checking else nullcontext():
            runner.forward(batch)
        for handle in handles:
            handle.remove()
        req.prefix_indices = runner.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :end].to(torch.int64)
        if end % 131072 == 0 or end == args.context:
            print(
                json.dumps(
                    {
                        "prepared_context": end,
                        "elapsed_s": time.perf_counter() - started,
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                ),
                flush=True,
            )
        if end >= first_measure:
            torch.cuda.synchronize()
            measurement_args = copy.copy(args)
            measurement_args.context = end
            if args.window_chunks > 1:
                output = Path(args.output)
                measurement_args.output = str(output.with_name(f"{output.stem}-{end}.json"))
            if args.window_chunks == 1:
                runner.model = None
                del layers, wrapper, runner
                gc.collect()
                torch.cuda.empty_cache()
            measure(
                measurement_args,
                variant,
                batch,
                captured,
                snapshot,
                selection_check,
                checkpoints,
                tokens[:end],
                time.perf_counter() - started,
                source_token_count,
                admission,
                initial_source_hashes,
                args.window_chunks > 1,
                args.context,
            )
            captured.clear()
            del snapshot, selection_check
    torch.distributed.destroy_process_group()


@torch.inference_mode()
def measure(
    args,
    variant,
    batch,
    captured,
    snapshot,
    selection_check,
    checkpoints,
    tokens,
    preparation_s,
    source_token_count,
    admission,
    initial_source_hashes,
    bootstrap_resident,
    window_end_context,
):
    """Validate the captured layer against its reference, time it and write the result.

    Args:
        args: The command line, with `context` and `output` set for this measurement.
        variant: The layer under test.
        batch: The forward batch the layer ran on.
        captured: The reference `kwargs`, `output`, `selection` and checkpoints from the prefill.
        snapshot: What one forward pass mutates, to put back between runs.
        selection_check: The check that a selection is the true top-k.
        checkpoints: The layer's parts, keyed by name.
        tokens: The prompt up to this context.
        preparation_s: Seconds spent before this measurement.
        source_token_count: Tokens the prompt file had before any repeat.
        admission: The memory admission verdict.
        initial_source_hashes: The source fingerprint taken at the start.
        bootstrap_resident: Whether the bootstrap weights are still resident.
        window_end_context: The last context of the recorded window.

    Returns:
        The result record, also written to `args.output`.

    Raises:
        RuntimeError: An output is not finite, a component is missing or the source changed mid-run.
    """
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.utils import BumpAllocator

    from lsa.evaluation.performance.kernels.measure.instrument import instrument
    from lsa.evaluation.performance.kernels.measure.recorder import Recorder

    module, backend, dispatch = variant.module, variant.backend, variant.dispatch
    # The harness starts this in the checkout - the installed copy carries no commit of its own
    repo_root = Path.cwd()
    checked_rows = selection_check.rows
    selection_check.allow_invalid_topk = args.diagnose_invalid_topk
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.save_topk_failures:
        selection_check.failure_path = str(Path(args.output).with_suffix(".invalid-topk.npz"))
    kwargs = captured["kwargs"]
    zero = BumpAllocator(16, torch.float32, "cuda")
    kwargs["zero_allocator"] = zero
    kwargs["forward_batch"] = batch
    expected = captured["output"]
    expected_selection = captured.get("selection", variant.empty_selection)
    selection_check.expected_tail = (
        expected_selection[:, module.indexer.index_topk :].cpu().numpy()
        if args.indexer == "native" and not variant.qsa
        else None
    )
    input_hash = tensor_digest(kwargs["hidden_states"])
    reference_state = variant.snapshot(batch)
    observed = {"selection": variant.empty_selection}
    cache_flush = (
        torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        if args.phase == "decode" and args.decode_cache_policy == "flush"
        else None
    )

    def save_selection(_module, positional, output) -> None:
        """Keep the selection the layer made on this run."""
        observed["selection"] = output

    def attention():
        """Call the layer on the captured arguments."""
        return variant.call(kwargs, batch)

    def execute():
        """Reset the zero allocator, build the attention metadata and call the layer."""
        zero._pointer = 0
        with forward_context(ForwardContext(attn_backend=dispatch)):
            backend.init_forward_metadata(batch)
            return attention()

    # The graph replay check pins the eager step to the list a pass chose, which a timing-only run still needs
    # => hooked in every run
    handle = variant.selection_hook(save_selection)
    expected_sorted = expected_selection.sort(1).values
    # What the pass changed in the index, read before the snapshot puts it back
    changed = snapshot.state.changed(int(batch.req_pool_indices[0])) if snapshot.state is not None else {}
    bucket_cap = variant.design.indexer.bucket_cap if variant.lsa else 0
    # A read that does not select the exact top-k of full logits - GLM's own indexer read through the native
    # space, say - has nothing for these checks to hold it to, so a timing-only run skips them
    if args.time_only:
        validation = {"time_only": True}
        exact_topk = None
    else:
        snapshot.restore()
        debug_handles = [
            part.register_forward_hook(save_checkpoint(name, observed)) for name, part in checkpoints.items()
        ]
        actual_scores = {}
        with selection_check.capture(actual_scores):
            actual = execute()
        selection_check.compare_scores(actual_scores)
        actual_scores.clear()
        independent_scores = {}
        snapshot.restore()
        with variant.oracle(independent_scores, args.phase == "decode", selection_check.tree):
            actual = execute()
        selection_check.prepare()
        exact_topk = selection_check.check(observed["selection"])
        diagnostics = {}
        for name in checkpoints:
            try:
                compare(observed[name], captured[name])
                diagnostics[name] = "exact"
            except AssertionError as error:
                diagnostics[name] = str(error)
        diagnostics["ordered_selection_exact"] = torch.equal(observed["selection"], expected_selection)
        diagnostics["selection_sets_exact"] = torch.equal(
            observed["selection"].sort(1).values, expected_sorted
        )
        print(json.dumps({"reference_diagnostics": diagnostics}), flush=True)
        for debug_handle in debug_handles:
            debug_handle.remove()
        observed = {"selection": observed["selection"]}
        rerun = selection_rows(observed["selection"], expected_selection, bucket_cap)
        validation = {
            "selection_multiset_exact": rerun["differing"] == 0,
            "selection_rows": rerun,
            "score_matrix_exact": checked_rows is None and args.phase != "decode",
            # A read through a hierarchy has no gathered scores to hold, so its scores have no verdict
            "checked_score_rows_exact": None if variant.indexer is None or selection_check.tree else True,
            "score_rows": checked_rows if checked_rows is not None else list(range(len(batch.positions))),
            "score_validation_exhaustive": checked_rows is None,
            "topk_cutoff_valid": exact_topk,
            "tree_read": selection_check.tree,
            "natural_order_output_error": output_error(actual, expected),
            "checkpoints": diagnostics,
            "independent_score_oracle": independent_scores,
        }
        reference_state.check()
        if args.phase == "decode":
            validation["visible_score_matrix_exact"] = True
        validation["cache_and_index_state_exact"] = True
        if not torch.isfinite(actual[0] if isinstance(actual, tuple) else actual).all():
            raise RuntimeError("Non-finite attention output")

        snapshot.restore()
        with variant.pinned_selection(expected_selection):
            validation["reference_order_output"] = compare(execute(), expected)
        selection_check.check(expected_selection)
        validation["selection_causal"] = True
        print(
            json.dumps({"validation": {k: v for k, v in validation.items() if k != "score_rows"}}),
            flush=True,
        )

    captured.clear()
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(args.warmup):
        snapshot.restore()
        execute()
    torch.cuda.synchronize()

    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    for event in events:
        event.record()
    torch.cuda.synchronize()
    recorder = Recorder()
    with instrument(variant, recorder):
        for _ in range(args.warmup):
            snapshot.restore()
            recorder.reset()
            execute()
        snapshot.restore()
        recorder.reset()
        if args.time_only:
            execute()
        else:
            with variant.pinned_selection(expected_selection):
                validation["instrumented_reference_order_output"] = compare(execute(), expected)
    torch.cuda.synchronize()
    rows = []

    def trial(repeat, detailed):
        """Time one run of the layer from restored state, with component ranges when `detailed`."""
        snapshot.restore()
        if cache_flush is not None:
            cache_flush.zero_()
        zero._pointer = 0
        recorder.reset()
        recorder.enabled = args.phase == "decode"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        allocator_before = torch.cuda.memory_stats()
        host_start = time.perf_counter()
        with forward_context(ForwardContext(attn_backend=dispatch)):
            events[0].record()
            if detailed and args.phase == "decode":
                torch.cuda.nvtx.range_push("attention metadata")
            backend.init_forward_metadata(batch)
            if detailed and args.phase == "decode":
                torch.cuda.nvtx.range_pop()
            events[1].record()
            metadata_ranges = len(recorder.ranges)
            recorder.enabled = True
            if detailed:
                torch.cuda.nvtx.range_push("attention module")
            try:
                actual = attention()
            finally:
                if detailed:
                    torch.cuda.nvtx.range_pop()
            events[2].record()
        events[2].synchronize()
        host_ms = (time.perf_counter() - host_start) * 1000
        allocator_after = torch.cuda.memory_stats()
        row = {
            "allocator_retries": allocator_after["num_alloc_retries"] - allocator_before["num_alloc_retries"],
            "allocator_ooms": allocator_after["num_ooms"] - allocator_before["num_ooms"],
            "repeat": repeat,
            "mode": "components" if detailed else "plain",
            "module_ms": events[1].elapsed_time(events[2]),
            "step_gpu_ms": events[0].elapsed_time(events[2]),
            "step_wall_ms": host_ms,
            "allocated_before_bytes": baseline_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if detailed:
            row["components"] = recorder.read(row["module_ms"], first=metadata_ranges)
            if args.phase == "decode":
                row["metadata_components"] = recorder.read(
                    events[0].elapsed_time(events[1]), last=metadata_ranges
                )
            row["backend_calls"] = dict(recorder.backends)
            for required in variant.required:
                if row["components"]["exclusive_ms"][required] <= 0:
                    raise RuntimeError(f"Missing component: {required}")
        if not args.time_only:
            row["topk_cutoff_valid"] = selection_check.check(observed["selection"])
            row["selection_rows_differing"] = int(
                (observed["selection"].sort(1).values != expected_sorted).any(1).sum()
            )
            row["selection_multiset_exact"] = row["selection_rows_differing"] == 0
        if not torch.isfinite(actual[0] if isinstance(actual, tuple) else actual).all():
            raise RuntimeError("Non-finite attention output")
        row["output_error_vs_reference"] = output_error(actual, expected)
        if args.phase == "decode":
            row["metadata_ms"] = events[0].elapsed_time(events[1])
            reference_state.check()
            row["cache_and_index_state_exact"] = True
        return row

    # Telemetry covers the timed trials and the graph comparison, not the warmup before them
    host = platform.node()
    card = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    sampler = Sampler(sample_path(Path(args.output).with_suffix(""), host), uuids=[card])
    with sampler:
        for repeat in range(args.repeats):
            modes = (False, True) if repeat % 2 == 0 else (True, False)
            for detailed in modes:
                context = instrument(variant, recorder) if detailed else nullcontext()
                with context:
                    rows.append(trial(repeat, detailed))
        profile_sample = None
        if args.profile and not args.profile_graph:
            torch.cuda.cudart().cudaProfilerStart()
            with instrument(variant, recorder):
                profile_sample = trial(-1, True)
            torch.cuda.cudart().cudaProfilerStop()
        graph_result = None
        if args.graph_compare:
            if args.phase == "decode":
                from lsa.evaluation.performance.kernels.measure.graph import (
                    compare_decode_graphs as compare_graphs,
                )
            else:
                from lsa.evaluation.performance.kernels.measure.graph import compare_graphs

            graph_result = compare_graphs(
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
                **({"cache_flush": cache_flush} if args.phase == "decode" else {}),
            )
    reference_state.check()
    validation["final_cache_and_index_state_exact"] = True
    validation["topk_failures"] = list(selection_check.failures)
    validation["topk_cutoff_valid"] = None if exact_topk is None else not selection_check.failures
    if handle is not None:
        handle.remove()
    if source_hashes() != initial_source_hashes:
        raise RuntimeError("Benchmark source changed during execution")
    result = {
        "schema": "attention-layer-perf/1",
        "args": vars(args),
        # Checkout commit - the installed copy carries none
        "lsa_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "lsa_diff": git_output(repo_root, "diff"),
        "source_sha256": initial_source_hashes,
        "environment": {
            k: v for k, v in os.environ.items() if k.startswith(("SGLANG_", "DG_", "CUDA_", "PYTORCH_"))
        },
        "gpu_uuid": card,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id={card}"], text=True
        ).strip(),
        "hostname": host,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "prompt_tokens_sha256": hashlib.sha256(array("q", tokens).tobytes()).hexdigest(),
        "attention_input_sha256": input_hash,
        "preparation_s": preparation_s,
        "memory_admission": admission,
        "prompt_source_tokens": source_token_count,
        "prompt_repeated": source_token_count < args.context,
        "validation": validation,
        "indexer_config": io.read_json(args.indexer_config) if args.indexer_config else None,
        "model_family": variant.family,
        "requested_prefill_backend": backend_names(args, variant)["prefill"],
        "backend_classes": backend_classes(variant.backend),
        "backend_source": backend_source(args),
        "max_attention_list_width": int(expected_selection.shape[-1]),
        "memory": variant.memory(),
        "snapshot_bytes": snapshot.nbytes,
        "verification_snapshot_bytes": reference_state.nbytes,
        "bootstrap_weights_resident": bootstrap_resident,
        "window_end_context": window_end_context,
        "hierarchy_growth": bool(changed.get("hierarchy_growth")),
        "samples": rows,
        "profile_sample": profile_sample,
        "graph_comparison": graph_result,
        "limitations": [
            "TP1",
            "eager",
            "one chunk in a recorded window",
            "bootstrap uses a truncated real model",
            "not full-model latency",
        ],
    }
    if args.phase == "decode":
        result.update(
            phase="decode",
            forward_mode=batch.forward_mode.name,
            query_tokens=len(batch.positions),
            requested_decode_backend=backend_names(args, variant)["decode"],
            decode_case=args.decode_case,
            decode_offsets=args.decode_offsets,
            cache_condition={
                "policy": args.decode_cache_policy,
                "eviction_buffer_bytes": 0 if cache_flush is None else cache_flush.nbytes,
                "eviction_outside_timing": True,
            },
            limitations=[
                "TP1, one active request, one decode token per call",
                "eager; optional graph comparison is recorded separately",
                "boundary probes are not a representative token average",
                "teacher-forced prompt continuation through a real five-layer truncation",
                "step includes this layer's backend metadata",
                "scheduler, sampling and full-model TPOT are excluded",
            ],
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output, "samples": len(rows)}), flush=True)
    return result


def main() -> None:
    """Parse and check the command line, then run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument(
        "--indexer",
        choices=("native", "lsa", "pisa", "full", "kda", "gdn"),
        default="native",
        help="native, lsa or pisa: the sparse layer with the model's own indexer, the design's or PISA's; full: "
        "the same layer with no indexer; kda (GLM) or gdn (Qwen): a linear attention layer",
    )
    parser.add_argument("--indexer-config", help="the settings file, for --indexer lsa or pisa")
    parser.add_argument(
        "--model-override",
        help="a JSON config override merged into the checkpoint's, sglang's --json-model-override-args - the "
        "model card's rotation",
    )
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument(
        "--prefill-attention-backend",
        help="replace upstream sglang's per-GPU prefill backend on a baseline method (full, GLM native) - recorded as an override",
    )
    parser.add_argument(
        "--decode-attention-backend",
        help="replace upstream sglang's per-GPU decode backend on a baseline method (full, GLM native) - recorded as an override",
    )
    parser.add_argument(
        "--dsa-prefill-backend",
        default="tilelang",
        help="GLM's sparse attention kernel over a prefill's selection, for the LSA and PISA methods",
    )
    parser.add_argument(
        "--dsa-decode-backend",
        default="tilelang",
        help="GLM's sparse attention kernel over a decode's selection, for the LSA and PISA methods",
    )
    parser.add_argument(
        "--decode-offsets",
        type=int,
        nargs="+",
        default=list(range(-512, -504)) + list(range(-7, 9)),
    )
    parser.add_argument("--decode-cache-policy", choices=("flush", "warm"), default="flush")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--layer", type=int, default=3)
    # One device - GLM-5.3-Flash needs 136 GiB of an H200 for its weights alone, so it is cut first
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-graph", action="store_true")
    parser.add_argument("--graph-compare", action="store_true")
    parser.add_argument("--repeat-prompt", action="store_true")
    parser.add_argument("--diagnose-invalid-topk", action="store_true")
    parser.add_argument(
        "--time-only", action="store_true", help="time the layer without checking its selection"
    )
    parser.add_argument("--save-topk-failures", action="store_true")
    parser.add_argument("--window-chunks", type=int, default=1)
    parser.add_argument("--validation-mib", type=int, default=1024)
    parser.add_argument("--memory-margin-gib", type=int, default=6)
    args = parser.parse_args()
    if args.validation_mib <= 0 or args.memory_margin_gib < 0:
        parser.error("Validation budget must be positive and margin nonnegative")
    if args.context <= args.chunk or args.context % args.chunk:
        parser.error("Context must be a multiple of chunk size and exceed one chunk")
    if args.indexer not in ("kda", "gdn") and args.layer != 3:
        parser.error("This pilot currently validates layer=3 only")
    if args.tp_size != 1:
        # Only rank 0 is started
        # => a larger tp-size waits ten minutes for peers
        parser.error(
            "This harness is one process and loads rank 0 only - a tp-size above 1 hangs in the "
            "rendezvous rather than sharding. Give it a checkpoint that fits one device."
        )
    if args.graph_compare and args.window_chunks != 1:
        parser.error("Graph comparison requires a single chunk")
    if args.window_chunks < 1 or args.window_chunks > 32:
        parser.error("Window must contain 1 to 32 chunks")
    if args.repeats < 2 or args.warmup < 1:
        parser.error("Need at least two repetitions and one warmup")
    if (args.indexer in ("lsa", "pisa")) is not bool(args.indexer_config):
        parser.error("--indexer lsa and pisa take --indexer-config, and no other variant does")
    if args.indexer == "pisa":
        # Its lists have no reference here - the served run's strict check holds them, and this times
        args.time_only = True
    if (
        args.indexer in ("lsa", "native")
        and not args.time_only
        and checkpoint_family(args.model_path) == Family.QWEN3_5
    ):
        parser.error(
            "Qwen's sparse layers - the design's and QSA - have no selection check in this harness, "
            "so they are timed without one - pass --time-only"
        )
    if args.phase == "decode":
        if args.profile and len(args.decode_offsets) != 1:
            parser.error("Decode profiling requires exactly one measured offset")
        if args.window_chunks != 1:
            parser.error("Decode uses token probes, not prefill chunk windows")
        if len(set(args.decode_offsets)) != len(args.decode_offsets) or any(
            offset < -512 or offset > 8 for offset in args.decode_offsets
        ):
            parser.error("Decode offsets must be unique and between -512 and 8")
    if args.profile_graph and not (args.profile and args.graph_compare and args.phase == "decode"):
        parser.error("Graph profiling requires decode, --profile and --graph-compare")
    run(args)


if __name__ == "__main__":
    main()
