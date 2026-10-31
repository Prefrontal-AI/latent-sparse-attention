"""Time everything a GLM-5.3-Flash layer runs besides its attention, on the modules sglang serves it with.

    python -m lsa.evaluation.performance.kernels.measure.modules --model-path <cut> --phase decode --output <file>

Run inside the serving container, never imported. Loads the five-layer cut [layer](layer.py) loads, with the same server
arguments, and times the real modules out of it - so the kernels are the ones sglang dispatches for this
checkpoint on this card, and the weights are the checkpoint's own:

    dense       layer 0's feed-forward - one of the three dense layers
    moe         layer 3's mixture of experts - routing, the fused shared expert and the routed experts
    boundary    one mHC boundary - the pre-map with its fused input norm, then the post-map
    embedding   the token lookup and the expansion into the residual streams
    final_norm  the contraction of the residual streams and the final norm
    head        the output projection of one token

A decoder layer has two boundaries, one around its attention and one around its feed-forward. The
attention itself is the harness's business, and nothing here touches it.

Prefill times one chunk eagerly with CUDA events, the way the harness times a prefill chunk, because this
model's prefill is never captured. Decode times one token as a replay of one captured graph, the way it is
served, beside the eager time.

There is no profiler kernel sum beside them. In this process the profiler returned no GPU activity at all for
the embedding, and three microseconds for a boundary whose replay takes fifteen - a session it loses is
indistinguishable from a module that launched nothing, so its number cannot be trusted either way.

The head is one token in both phases - a prefill computes the logits of its last position only.
"""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch

SCHEMA = "lsa-modules/1"
MODULES = ("dense", "moe", "boundary", "embedding", "final_norm", "head")


def load(args):
    """Load the cut as the harness does - the same server arguments, one device, no KV pool to speak of."""
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import configure_logger

    server_args = ServerArgs(
        model_path=args.model_path,
        trust_remote_code=True,
        tp_size=1,
        attention_backend="dsa",
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
        kv_cache_dtype="bfloat16",
        context_length=2 * args.chunk,
        max_total_tokens=2 * args.chunk,
        chunked_prefill_size=args.chunk,
        max_running_requests=1,
        mem_fraction_static=args.mem_fraction,
        disable_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        disable_radix_cache=True,
        log_level="warning",
    )
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    # One process on one device - load_model reads its rank and device from what is published
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    configure_logger(server_args)
    wrapper, _ = load_model(server_args, PortArgs.init_new(server_args), 0, 0)
    return wrapper.torch_runner.model


def build(model, args, tokens):
    """Return one callable per module, each running what the served forward runs for it.

    Args:
        model: The loaded cut.
        args: The command line.
        tokens: How many positions one call carries - a chunk at prefill, one at decode.

    Returns:
        Module name to a zero-argument callable, and a description of what each one calls.
    """
    from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
    from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
    from sglang.srt.runtime_context import get_forward

    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    text = model.model
    hc_mult = config.hc_mult
    hidden_size = config.hidden_size
    dense_layer = text.layers[args.dense_layer]
    moe_layer = text.layers[args.moe_layer]
    if moe_layer.mlp.__class__.__name__.endswith("MLP") or not dense_layer.mlp.__class__.__name__.endswith(
        "MLP"
    ):
        raise ValueError(
            f"layer {args.dense_layer} must be dense and layer {args.moe_layer} a mixture of experts - got "
            f"{dense_layer.mlp.__class__.__name__} and {moe_layer.mlp.__class__.__name__}"
        )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    # Activations at the scale the residual streams carry after the embedding
    hidden = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator)
    streams = torch.randn(
        tokens, hc_mult * hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    token_ids = torch.randint(0, config.vocab_size, (tokens,), device="cuda", generator=generator)
    last = hidden[-1:].contiguous()

    def dense():
        """Run layer 0's feed-forward."""
        return dense_layer.mlp(hidden)

    # The decoder layer calls the experts inside these two scopes, and an out-of-place runner writes into
    # the buffer the second one names
    moe_buffer = (
        nullcontext
        if moe_layer.mlp.experts.moe_runner_config.inplace
        else (lambda: moe_output_buffer_ctx(streams))
    )

    def moe():
        """Run the mixture of experts under the scopes the decoder layer sets."""
        with get_forward().scoped(fuse_mlp_allreduce=False, mlp_reduce_scatter=False):
            with moe_buffer():
                return moe_layer.mlp(hidden, None, None)

    communicator = moe_layer.layer_communicator.mhc

    def boundary():
        """Run one mHC boundary - the pre-map with its fused input norm, then the post-map."""
        layer_input, residual = communicator.attn_split(streams, out_norm=moe_layer.input_layernorm)
        combined = communicator.mlp_combine(layer_input, residual)
        communicator.reset_aux()
        return combined

    def embedding():
        """Look up the tokens and expand them into the residual streams."""
        return hc_expand(text.embed_tokens(token_ids), hc_mult)

    def final_norm():
        """Contract the residual streams and apply the final norm."""
        return text.norm(hc_contract(streams, hc_mult))

    def head():
        """Project the last position to logits."""
        return torch.matmul(last.to(model.lm_head.weight.dtype), model.lm_head.weight.T)

    calls = {
        "dense": f"layer {args.dense_layer} mlp - {dense_layer.mlp.__class__.__name__}",
        "moe": f"layer {args.moe_layer} mlp - {moe_layer.mlp.__class__.__name__}, "
        f"{moe_layer.mlp.num_fused_shared_experts} fused shared expert(s), "
        f"runner inplace {moe_layer.mlp.experts.moe_runner_config.inplace}",
        "boundary": f"layer {args.moe_layer} mHC attn_split with the fused input norm, then mlp_combine",
        "embedding": "embed_tokens, then hc_expand",
        "final_norm": "hc_contract, then the final norm",
        "head": "lm_head over the last position",
    }
    functions = {
        "dense": dense,
        "moe": moe,
        "boundary": boundary,
        "embedding": embedding,
        "final_norm": final_norm,
        "head": head,
    }
    return functions, calls


def check_finite(output):
    """Raise when a module's output holds a non-finite value.

    Args:
        output: A tensor or a tuple of them.

    Raises:
        RuntimeError: A tensor is not finite.
    """
    parts = output if isinstance(output, tuple) else (output,)
    for part in parts:
        if isinstance(part, torch.Tensor) and not torch.isfinite(part.float()).all():
            raise RuntimeError("a module returned non-finite values")


def event_times(function, warmup, repeats):
    """Return milliseconds per eager call, from CUDA events on either side of it."""
    for _ in range(warmup):
        check_finite(function())
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def graph_times(function, warmup, repeats):
    """Return milliseconds per replay of one captured call - what a served decode step pays."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = function()
    graph.replay()
    torch.cuda.synchronize()
    check_finite(captured)
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return times


def gpu_identity():
    """Return the card's name and uuid as nvidia-smi reports them - the uuid the harness records."""
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader", "-i", "0"], text=True
    ).strip()
    name, uuid = (part.strip() for part in line.split(","))
    return name, uuid


def text_config_summary(model):
    """Return the config values that size the timed modules."""
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    keys = (
        "hidden_size",
        "hc_mult",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "vocab_size",
    )
    return {key: getattr(config, key, None) for key in keys}


@torch.inference_mode()
def run(args):
    """Load the cut, time each named module and write the record.

    Args:
        args: The parsed command line.

    Raises:
        RuntimeError: This file changed while it was measuring.
    """
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    model = load(args)
    tokens = args.chunk if args.phase == "prefill" else 1
    functions, calls = build(model, args, tokens)
    modules = {}
    for name in args.modules:
        function = functions[name]
        entry = {
            "tokens": 1 if name == "head" else tokens,
            "calls": calls[name],
            "event_ms": event_times(function, args.warmup, args.repeats),
            "graph_ms": graph_times(function, args.warmup, args.repeats) if args.phase == "decode" else None,
        }
        modules[name] = entry
        graph = f"  graph {statistics.median(entry['graph_ms']):.4f}" if entry["graph_ms"] else ""
        print(
            f"{name:>10}  event {statistics.median(entry['event_ms']):.4f} ms{graph}",
            flush=True,
        )
    gpu, uuid = gpu_identity()
    body = {
        "schema": SCHEMA,
        "phase": args.phase,
        "chunk": args.chunk,
        "model": {"path": args.model_path, "text_config": text_config_summary(model)},
        "layers": {"dense": args.dense_layer, "moe": args.moe_layer},
        "warmup": args.warmup,
        "repeats": args.repeats,
        "modules": modules,
        "gpu": gpu,
        "gpu_uuid": uuid,
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "seconds": round(time.perf_counter() - started, 1),
        "source": {"src/lsa/evaluation/performance/kernels/measure/modules.py": source},
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12] != source:
        raise RuntimeError("the script changed while it was measuring")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=1))
    print(f"wrote {output}", flush=True)
    torch.distributed.destroy_process_group()


def main() -> None:
    """Parse and check the command line, then time the modules."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--modules", nargs="+", choices=MODULES, default=list(MODULES))
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--dense-layer", type=int, default=0)
    parser.add_argument("--moe-layer", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1:
        parser.error("Need at least two repetitions and one warmup")
    run(args)


if __name__ == "__main__":
    main()
