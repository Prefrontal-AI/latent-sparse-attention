"""The hooks that enable the design's indexer in a stock sglang, registered from outside it.

sglang discovers this through the `sglang.srt.plugins` entry point, calls `register` once per process before
the engine starts, and applies what it registered. Nothing in sglang is modified, so the server runs a stock
image and this package is a `pip install` next to it.

The indexer's settings have no command-line flag of their own, because the parser is the one thing a plugin cannot reach.
They are one block of the model config instead - `--json-model-override-args '{"lsa": {...}}'`, or the same key in a
checkpoint's config.json - which [settings](settings/__init__.py) parses and every reader below takes its config to find.

## When a process is enabled

The plugin loads in every process, before any model config exists, and sglang applies a hook for good, so the choice of hooks cannot
wait for a block and cannot be made without one. Two schemes were open: register every hook always and have each gate on the block, or
register one hook on the model config and enable the rest from it.
The second is the one here, for two reasons.
A gate on `_forward_fa3`, the backends' forwards and the memory estimates would cost a native server a wrapper on every call, and the
indexer is a class replacement, which cannot gate - the checkpoint's own indexer must stay the class a native server builds.
And every process that runs a model builds a model config before the hooked code runs: the scheduler builds it in its constructor,
so each tensor-parallel and pipeline rank has it before a layer, a pool or a backend exists, and the entrypoint's own resolution
builds one in the process that tokenizes.
So `register` carries `configure` on `ModelConfig.__init__`, and `configure` enables the process, once, from the
block it finds. A server with no block registers nothing else, and each of its configs costs one attribute read.
`apply_hooks` is idempotent and rebinds a name other modules imported, so enabling the indexer after sglang's modules loaded patches what a
registration at load would have.

The model calls its indexer with `x`, the attention's input, at the call site its own DSA indexer reads it at, so
the design's indexer replaces the class and no hook has to move that call. What the hooks add is the host-side
maintenance before each forward, the exact list lengths for the two attention backends, the request ids on the
batch views, the two answers that free GLM's own index keys, and the indexer's own memory counted into the KV
pool's sizing, which [memory](memory.py) explains.

A design whose `input` is the key and query - the design's input and the default, GLM's key latent and the heads' absorbed
queries - needs two hooks more, because neither exists where the model calls its indexer with `x`. `gate` stops an
LSA layer's indexer from running there and remembers that it wanted to, and `prepare` runs it at the end of
`forward_absorb_prepare`, where the absorbed query and the normed key latent are in the tuple the method returns,
and puts the selection where attention reads it - as the internal snapshot did. Its index keys are projected from the key
latents the KV pool holds, so the dense path that prefills a short prompt needs no call either. Under prefill context
parallel the key latents that method returns are the chunk's whole, which the fork's DSA attention gathered.

Context parallel needs no hook of its own. The indexer reads the layouts through [context_parallel](context_parallel.py),
and a list is one per position however the heads are laid out, so the two backends' hooks serve it as they are - the
prefill reads through FA3 or FlashMLA as under tensor parallel, and TileLang, which decode context parallel decodes
through, masks the padding itself. A server that runs context parallel - which sglang's own parallel state says, where
the scheduler has published it - has the symbols those layouts are read through resolved before its model loads.

Qwen3.8-27B is a dense model with no indexer to replace, so its hooks sit on the attention backends its sixteen
full-attention layers attend through - FlashAttention's, and triton's, which decodes under decode context parallel,
behind the hybrid wrapper - and [qwen3_5](models/qwen3_5/backend.py) says how: a backend's constructor enables it with an
indexer for every quadratic layer when the model is grouped-query, its two metadata calls maintain the index, and its
`forward_extend` and `forward_decode` select from the key and query input and read the selection. On any other model
they hand every call on unchanged, so both families' hooks are registered on every server and each acts on its own
model alone.

A flat decode captures one CUDA graph for each range of context and replays the narrowest that covers the batch's
longest request, so the index kernels are sized to the range instead of the server's context length.
`enable_context_graphs` hands sglang's decode runner the ranges as the attention variants it already keys a graph by,
and [context_buckets](context_buckets.py) says how a replay picks one and how a capture reads which it is sized to.
A descending decode captures one graph and takes no ranges: its shapes are fixed by the design and the levels the server's
longest context grows, never by the request in the batch.
It is captured as it is, and a design with shaped node buckets builds no tree for a graph to read and is served eagerly.

PISA is a second selector in the same plugin, named by the block's `selector`, and reads no projection file. On GLM it
replaces the same indexer class and takes the same backend and memory hooks and the same two hooks of the key and query input.
On a grouped-query model it takes LSA's backend hooks with its own setup and read, a list a key-value head. It serves tensor
parallel only, so it registers none of the context-parallel layouts.

An early budget shortens the lists below its position, which FA3 and FlashMLA read through the exact lengths and the
TileLang read does not - it walks every block of the width it is handed. A design with one hooks `_forward_tilelang`
as well, so a batch whose lists are all early reaches the kernel narrower, as `trim_early` decides.

The list width lives on the `hf_text_config` the model reads, which for a nested checkpoint is its `text_config`,
so `configure` derives it from the block and sets it there, puts the block there too, and checks the projection file
against the model's layers while it is at it.

`apply_hooks` and the loader around `register` log a failure rather than raise it, and a target that did not
attach leaves a server running the checkpoint's own indexer under an LSA label. So every target is resolved
before anything is registered, and the banner is printed only once they all did.

What another process reads - the two banners, the plugin's name and the block's key - is [constants](constants.py). The dotted
paths below are not, and stay next to the loop that resolves them: enabling indexes `TARGETS` positionally to
pair each target with its hook, and a reader who has to open another file to learn what `TARGETS[3]` is has lost
the registration.
"""

import importlib
import inspect
import logging
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch

from lsa.sglang import memory, settings
from lsa.sglang.constants import Banner, Plugin
from lsa.sglang.settings import PluginSettings
from lsa.sglang.settings.debug import DumpSettings
from lsa.sglang.settings.design import KEY_AND_QUERY, DesignSettings, PisaConfig

logger = logging.getLogger(__name__)

DSA_BACKEND: str = "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend"

# The checkpoint's own indexer classes - GLM's pooled one when its config pools index keys
NATIVE_INDEXERS: tuple[str, ...] = (
    "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer",
    "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.IndexerKPool",
)

# Qwen's own sparse-attention indexer, which an image without QSA does not carry
# => resolved on its own and skipped where absent
QSA_INDEXER: str = "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.forward_cuda"

# Where a server's settings arrive, and so the one hook every server carries
MODEL_CONFIG: str = "sglang.srt.configs.model_config.ModelConfig.__init__"

# Every dotted path a design patches, resolved before one of them is registered
TARGETS: tuple[str, ...] = (
    MODEL_CONFIG,
    "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer",
    f"{DSA_BACKEND}.init_forward_metadata",
    f"{DSA_BACKEND}.init_forward_metadata_out_graph",
    f"{DSA_BACKEND}._forward_fa3",
    f"{DSA_BACKEND}._forward_flashmla_sparse",
    "sglang.srt.model_executor.forward_batch_info.build_inner_fb_view",
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.build_replay_fb_view",
    "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._profile_available_bytes",
    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._resolve_memory_pool_config",
)

# The sparse read an image prefills on under `--dsa-prefill-backend tilelang`, which a design with an early budget wraps
# => resolved only then, since no other design hooks it
TILELANG: str = f"{DSA_BACKEND}._forward_tilelang"

# What a graph a context range needs - the decode runner's capture loop, which keys a graph by an attention variant,
# and the getter `context_buckets` reads the variant being captured through
# => resolved on their own and skipped where absent, since an image without them serves one graph as it always did
CONTEXT_GRAPH_TARGETS: tuple[str, ...] = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._capture_one_stream",
    "sglang.srt.model_executor.runner_utils.capture_mode.get_capture_attention_variant",
)

FORWARD_MLA: str = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla"

FLASH_ATTENTION: str = "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend"

# The backend a decode context-parallel Qwen server decodes on, the one the image shards a grouped-query pool for
TRITON: str = "sglang.srt.layers.attention.triton_backend.TritonAttnBackend"

# The backends a grouped-query model's quadratic layers are served through
GROUPED_QUERY_BACKENDS: tuple[str, ...] = (FLASH_ATTENTION, TRITON)

# What each of them is hooked at - the constructor, which enables it, its eager and captured metadata calls, which
# maintain the index, and the two forwards, which select and read
GROUPED_QUERY_METHODS: tuple[str, ...] = (
    "__init__",
    "init_forward_metadata",
    "init_forward_metadata_out_graph",
    "forward_extend",
    "forward_decode",
)
GROUPED_QUERY_TARGETS: tuple[str, ...] = tuple(
    f"{backend}.{method}" for backend in GROUPED_QUERY_BACKENDS for method in GROUPED_QUERY_METHODS
)


# What the key and query input hooks - where the model asks whether a layer runs its indexer, and the method whose
# returned tuple carries the absorbed query and the key latent
KEY_AND_QUERY_TARGETS: tuple[str, ...] = (
    f"{FORWARD_MLA}.DeepseekMLAForwardMixin.should_run_indexer",
    f"{FORWARD_MLA}.DeepseekMLAForwardMixin.forward_absorb_prepare",
)

# Where `gate` leaves the model's answer on an LSA layer, for `prepare` to read
WANTED_INDEXER: str = "_lsa_lsa_wants_indexer"

# Positions in `forward_absorb_prepare`'s returned tuple
ABSORBED_QUERY, KEY_LATENT, FORWARD_BATCH, POSITIONS, SELECTION, FUSION_PLAN = 2, 3, 4, 6, 7, 9

# Whether a layer reuses the top-k before it and so holds no index keys, asked by both pools' sizing
# => answering yes under LSA frees GLM's fp8 index-key buffer, 132 B a token a sparse layer
# Patched per module, not through the hook registry, which would also rewrite the model's own skip logic and the pipeline proxy's top-k size
ELIDED: tuple[str, ...] = (
    "sglang.srt.mem_cache.kv_cache_configurator.dsa_layer_skips_topk",
    "sglang.srt.model_executor.pool_configurator.dsa_layer_skips_topk",
)

# What PISA reads besides the symbols every server resolves - the gather that gives each rank every head's query
REQUIRED_PISA: tuple[str, ...] = ("sglang.srt.layers.dp_attention.attn_tp_all_gather_into_tensor",)

# Where the prefix cache repoints a request's positions at another request's KV slots for the same tokens
REPOINT_TARGET: str = "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.write"

# Symbols this package imports at call time instead of hooking
# => resolved up front, so a rename fails at startup rather than on the first forward
REQUIRED: tuple[str, ...] = (
    "sglang.srt.environ.envs",
    # Read by `refuse_metadata_glue` - unresolvable would leave the guard silently answering False
    "sglang.srt.environ.envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH",
    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
    # `query_heads: all` sums every rank's heads through it
    "sglang.srt.layers.dp_attention.attn_tp_all_reduce",
    "sglang.srt.configs.model_config.dsa_layer_skips_topk",
    "sglang.srt.layers.attention.dsa.utils.is_graph_dsa_split_op_surface",
    "sglang.srt.model_executor.forward_context.get_attn_backend",
    "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
    "sglang.srt.runtime_context.get_schedule",
    # Whether the server caches prefixes, which an index then shares
    "sglang.srt.runtime_context.get_memory",
    "sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute.compute_cu_seqlens",
)


# What the context-parallel layouts are read through - only the fork's image has them
# => resolved when the server runs context parallel, so a stock image fails at startup and not on a forward
REQUIRED_CP: tuple[str, ...] = (
    "sglang.srt.layers.cp.utils.is_cp_active",
    "sglang.srt.layers.cp.base.get_cp_strategy",
    "sglang.srt.runtime_context.get_parallel",
)

# What the key and query input reads besides under context parallel - whether the fork gathered a chunk's key latents
REQUIRED_CP_KEY_AND_QUERY: tuple[str, ...] = ("sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp",)


def resolve(target: str) -> object:
    """Return the object a dotted target names.

    Args:
        target: A `module.attr` or `module.Class.method` path.

    Returns:
        The attribute the path names.

    Raises:
        AttributeError: The module resolved but the attribute did not.
        ModuleNotFoundError: No prefix of the path imports.
    """
    module_path, _, attribute = target.rpartition(".")
    holder: Any
    try:
        holder = importlib.import_module(module_path)
    except ModuleNotFoundError:
        outer, _, class_name = module_path.rpartition(".")
        holder = getattr(importlib.import_module(outer), class_name)
    return getattr(holder, attribute)


def configure(result: None, self: Any, *args: Any, **kwargs: Any) -> None:
    """Enable this process from the settings its model config carries, and size the config for the design's list.

    A config with no block is left exactly as it was, which is every config of a native server.
    A block is read from wherever the override landed and put on the text config too, since the indexer
    is built from that one and a nested checkpoint's differs from the top-level config.
    Lists are `top_k + window_size + sink_size` wide, so there are no pooled index keys and nothing to compress.
    The fields live on `hf_text_config`, which for a checkpoint with a nested `text_config` is not the top-level
    config the block is read from, so they are set there.
    The projection file is read here too, before a model loads, so a file that is not this model's fails the
    launch rather than the first layer.
    A grouped-query model's file is checked the same way, against its quadratic layers, and it has no list width
    to set - its backend's hooks read the lists.

    Args:
        result: What `ModelConfig.__init__` returned, which is nothing.
        self: The config being built.
        *args: Its arguments, unused.
        **kwargs: The same, unused.

    Returns:
        None - the config is filled in place.

    Raises:
        ValueError: The block is refused.
        RuntimeError: This process was enabled with other settings, or a target no longer resolves.
    """
    text = getattr(self, "hf_text_config", None)
    if text is None or getattr(self, "is_draft_model", False):
        return None
    served = settings.of(self.hf_config)
    if served == PluginSettings():
        return None
    if settings.block_of(text) is None:
        setattr(text, Plugin.BLOCK, settings.block_of(self.hf_config))
    enable(served)
    if served.pisa is not None:
        return size_pisa_lists(text, served.pisa)
    if served.design is None:
        return None
    from lsa.sglang.index import weights
    from lsa.sglang.models.qwen3_5 import index_space

    design = served.design
    require_context_parallel(design)
    if index_space.is_grouped_query(text):
        weights.read(
            design.serving.projections,
            weights.input_dim(text, design.indexer.input),
            design.indexer.index_dim,
            weights.indexed_layers(text),
        )
        return None
    if not hasattr(text, "index_topk"):
        return None
    from sglang.srt.configs.model_config import dsa_layer_skips_topk

    layers = weights.indexer_layers(text, dsa_layer_skips_topk)
    weights.read(
        design.serving.projections,
        weights.input_dim(text, design.indexer.input),
        design.indexer.index_dim,
        layers,
    )
    text.index_topk = design.width
    text.index_kpool = 1
    text.index_kpool_compress = False
    return None


def size_pisa_lists(text: Any, config: PisaConfig) -> None:
    """Size the sparse attention backend for PISA's list, which has no projections to check.

    Args:
        text: The model's text config, which the model reads `index_topk` from.
        config: The pyramid, whose list width the backend is sized for.

    Returns:
        None - the config is filled in place.
    """
    if not hasattr(text, "index_topk"):
        return None
    text.index_topk = config.width
    text.index_kpool = 1
    text.index_kpool_compress = False
    return None


def unresolved(targets: tuple[str, ...]) -> list[str]:
    """Return the targets that no longer resolve, each with why.

    Args:
        targets: Dotted paths.

    Returns:
        One `target (reason)` for every path that does not name something.
    """
    missing = []
    for target in targets:
        try:
            resolve(target)
        except (AttributeError, ModuleNotFoundError, ImportError) as failure:
            missing.append(f"{target} ({failure})")
    return missing


def require_context_parallel(design: DesignSettings) -> None:
    """Refuse an image that cannot run the context-parallel layouts this server was launched with.

    Whether the server runs context parallel is sglang's own state, read where this process has it.
    A process whose parallel state is not published yet - the one that builds the config to resolve its arguments - has
    nothing to check, and each scheduler process checks before its model loads.

    Args:
        design: The design, whose input says which layouts' symbols it reads.

    Raises:
        RuntimeError: A symbol the layouts are read through is missing from this sglang.
    """
    from lsa.sglang import context_parallel

    if not context_parallel.is_served():
        return None
    wanted = REQUIRED_CP + (REQUIRED_CP_KEY_AND_QUERY if design.indexer.input == KEY_AND_QUERY else ())
    missing = unresolved(wanted)
    if missing:
        raise RuntimeError(
            "lsa cannot serve context parallel on this sglang - these symbols no longer resolve: "
            + "; ".join(missing)
        )
    return None


def skips_under_lsa(original: Any, config: Any, layer_id: int) -> bool:
    """Answer that a layer holds no index keys of its own, which is true of every layer under LSA and PISA.

    Args:
        original: sglang's `dsa_layer_skips_topk`.
        config: The model's config.
        layer_id: The layer.

    Returns:
        True.
    """
    return True


def elide_index_keys() -> None:
    """Make both pools' sizing leave out GLM's own index-key buffer.

    Raises:
        RuntimeError: A module no longer holds the function it was to answer in place of.
    """
    for target in ELIDED:
        module_path, _, name = target.rpartition(".")
        module = importlib.import_module(module_path)
        original = getattr(module, name)

        def answer(config: Any, layer_id: int, original: Any = original) -> bool:
            return skips_under_lsa(original, config, layer_id)

        setattr(module, name, answer)
        if getattr(module, name) is not answer:
            raise RuntimeError(f"{target} could not be replaced - the pool would reserve GLM's index keys")


def is_lsa(module: object) -> bool:
    """Return whether an attention layer indexes with the design's selector - LSA, or PISA where that is served.

    Args:
        module: An MLA attention layer, which carries an indexer only on the layers that own one.

    Returns:
        True when this layer's indexer is the design's.
    """
    indexer = getattr(module, "indexer", None)
    return bool(getattr(indexer, "is_lsa", False) or getattr(indexer, "is_pisa", False))


def gate(original: Any, self: Any, *args: Any, **kwargs: Any) -> bool:
    """Stop an LSA layer running its indexer where the model calls it with `x`, and remember that it wanted to.

    Under the key and query input the absorbed query does not exist yet at that call, so the call is made again
    from `prepare`, where it does.

    Args:
        original: `should_run_indexer`.
        self: The attention layer.
        *args: Its `prev_topk_indices`, if the caller passed one.
        **kwargs: The same, by name.

    Returns:
        The original answer on a layer without the design's indexer, and False on one with it.
    """
    answer = bool(original(self, *args, **kwargs))
    if not is_lsa(self):
        return answer
    setattr(self, WANTED_INDEXER, answer)
    return False


def prepare(result: tuple[Any, ...], self: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...] | None:
    """Select from the key and the heads' queries once both exist, and put the selection where attention reads it.

    Args:
        result: What `forward_absorb_prepare` returned.
        self: The attention layer.
        *args: Its arguments, unused.
        **kwargs: The same, unused.

    Returns:
        The tuple with the selection in it, or None on a layer this does not touch.

    Raises:
        RuntimeError: A fused bmm path ran, which leaves no eager absorbed query to sum.
    """
    if not is_lsa(self) or not getattr(self, WANTED_INDEXER, False):
        return None
    query, key, fusion_plan = result[ABSORBED_QUERY], result[KEY_LATENT], result[FUSION_PLAN]
    if fusion_plan is not None or query is None:
        raise RuntimeError(
            "the key and query input needs the eager absorbed query - disable the fused bmm graph paths"
        )
    # The attention input rides along for the debugging dump alone - the key and query input never reads it
    hidden_states = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
    # PISA scores with the attention's own softmax scale, which only the attention layer holds
    scaled = {"logit_scale": self.scaling} if getattr(self.indexer, "is_pisa", False) else {}
    selection = self.indexer(
        x=hidden_states,
        q_lora=None,
        positions=result[POSITIONS],
        forward_batch=result[FORWARD_BATCH],
        layer_id=self.layer_id,
        key=key.squeeze(1),
        query=query,
        **scaled,
    )
    return (*result[:SELECTION], selection, *result[SELECTION + 1 :])


def maintain(*args: Any, **kwargs: Any) -> None:
    """Bring every index up to the tokens already in the KV pool, before metadata is built.

    Args:
        *args: The backend and its forward batch.
        **kwargs: The same, by name.

    Returns:
        None, which leaves the arguments as they were.
    """
    from lsa.sglang.index import registry

    registry.maintain(kwargs.get("forward_batch") or args[1])
    return None


def maintain_out_graph(*args: Any, **kwargs: Any) -> None:
    """Do the same outside graph capture, where the batch is the live one.

    Args:
        *args: The backend, its forward batch, and possibly the capture flag.
        **kwargs: The same, by name.

    Returns:
        None, which leaves the arguments as they were.
    """
    from lsa.sglang.index import registry

    in_capture = kwargs.get("in_capture", args[2] if len(args) > 2 else False)
    if not in_capture:
        registry.maintain(kwargs.get("forward_batch") or args[1])
    return None


def fa3(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    """Give FA3 the exact list lengths instead of the checkpoint's fixed width.

    An LSA list puts its valid slots first and pads after them, and FA3 reads every entry it is given, so the
    count has to be exact.

    Args:
        original: `_forward_fa3`.
        self: The attention backend.
        *args: Its arguments.
        **kwargs: The same, by name.

    Returns:
        What FA3 returned.
    """
    from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import compute_cu_seqlens

    from lsa.sglang.index import read

    bound = bind(original, self, *args, **kwargs)
    page_table = bound.arguments["page_table"]
    lengths = read.list_lengths(page_table)
    bound.arguments["page_table"] = page_table.clamp(min=0)
    bound.arguments["cache_seqlens"] = lengths
    bound.arguments["cu_seqlens_k"] = compute_cu_seqlens(lengths)
    return original(*bound.args, **bound.kwargs)


def flashmla(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    """Do the same for the sparse FlashMLA path.

    Args:
        original: `_forward_flashmla_sparse`.
        self: The attention backend.
        *args: Its arguments.
        **kwargs: The same, by name.

    Returns:
        What FlashMLA returned.
    """
    from lsa.sglang.index import read

    bound = bind(original, self, *args, **kwargs)
    page_table = bound.arguments["page_table_1"]
    bound.arguments["page_table_1"] = page_table.clamp(min=0)
    bound.arguments["topk_length"] = read.list_lengths(page_table)
    return original(*bound.args, **bound.kwargs)


def tilelang(original: Any, self: Any, *args: Any, early_width: int, **kwargs: Any) -> Any:
    """Cut the lists to the early width when the batch is early, so the TileLang read walks fewer blocks.

    The kernel takes no list lengths and walks every 64-column block of the width it is handed, so an early budget's
    shorter lists save nothing unless they are narrower as well.
    Exactly two widths reach the kernel, the design's and the early one, so at most two kernels are compiled.

    Args:
        original: `_forward_tilelang`.
        self: The attention backend.
        *args: Its arguments.
        early_width: What `DesignSettings.early_width` gave, bound at registration.
        **kwargs: The same, by name.

    Returns:
        What the TileLang read returned.
    """
    from lsa.sglang.index import read

    bound = bind(original, self, *args, **kwargs)
    page_table = bound.arguments["page_table_1"]
    # A graph capture cannot read the host, and a captured decode keeps the width it was captured at
    if not (page_table.is_cuda and torch.cuda.is_current_stream_capturing()):
        bound.arguments["page_table_1"] = read.trim_early(page_table, early_width)
    return original(*bound.args, **bound.kwargs)


def enable_context_graphs(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    """Capture a decode graph a context range, by handing the runner the ranges as its attention variants.

    Called once per capture of the decode runner, before it loops over batch sizes, so every size is captured
    once for each range and a replay picks the range of the batch's longest request. A runner that already has
    variants is left as it is, and so is a descending decode: the width of its candidates follows the design's
    start width and beam, and the levels of the server's longest context, so a range of context narrows nothing of it.

    Args:
        original: `DecodeCudaGraphRunner._capture_one_stream`.
        self: The decode runner.
        *args: Its arguments.
        **kwargs: The same, by name.

    Returns:
        What the capture returned.
    """
    from lsa.sglang import context_buckets

    design = settings.design_of(self.model_runner.model_config.hf_config)
    if (
        self.attention_graph_variants is None
        and self.capture_forward_mode.is_decode()
        and design.serving.decode == "flat"
    ):
        context_length = self.model_runner.model_config.context_len
        buckets = context_buckets.ContextBuckets.under(design.serving.decode_context_limits, context_length)
        if buckets is not None:
            logger.info("lsa: decode graphs for contexts up to %s and %d", buckets.limits, context_length)
            self.attention_graph_variants = buckets
    return original(self, *args, **kwargs)


def bind(original: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the call's arguments by name, so a hook never counts positions.

    Args:
        original: The function being wrapped.
        *args: The call's positional arguments.
        **kwargs: Its keyword arguments.

    Returns:
        The bound arguments, with defaults filled in.
    """
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def carry_request_ids(result: SimpleNamespace, forward_batch: Any, *args: Any, **kwargs: Any) -> None:
    """Put the request ids on a batch view, which is how decode finds its index rows without a device sync.

    Args:
        result: The view that was built.
        forward_batch: The batch it was built from.
        *args: The builder's other arguments, unused.
        **kwargs: The same, unused.

    Returns:
        None - the view is filled in place.
    """
    result.rids = getattr(forward_batch, "rids", None)
    return None


def refuse_metadata_glue() -> None:
    """Refuse a server that would capture the per-replay metadata prep into a graph of its own.

    `MetadataGlueGraph` records `init_forward_metadata_out_graph` once per replay key and replays it, and
    that is where both families maintain their index - host work that has to run every step. A frozen
    index scores like a healthy one while it answers from a partition that stopped growing.

    Raises:
        RuntimeError: `SGLANG_ENABLE_METADATA_GLUE_GRAPH` is set.
    """
    from sglang.srt.environ import envs

    if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
        raise RuntimeError(
            "lsa maintains its index from init_forward_metadata_out_graph, and the metadata glue "
            "graph captures that call and replays the recording - which would leave every index frozen at "
            "its capture step and score as though nothing were wrong. Unset "
            "SGLANG_ENABLE_METADATA_GLUE_GRAPH to serve LSA"
        )


def register_context_graphs() -> bool:
    """Register the context-range graphs' hook, where the decode runner has the variants it hangs them on.

    Returns:
        Whether the hook was registered.
    """
    try:
        for target in CONTEXT_GRAPH_TARGETS:
            resolve(target)
    except (AttributeError, ModuleNotFoundError, ImportError) as absent:
        logger.info("lsa: one decode graph, no context ranges - %s", absent)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(CONTEXT_GRAPH_TARGETS[0], enable_context_graphs, HookType.AROUND)
    return True


def dump_native(result: Any, self: Any, *args: Any, dump: DumpSettings, **kwargs: Any) -> None:
    """Dump the `x` a native server's own indexer reads at a prefill, as the design's indexer dumps its own.

    Args:
        result: What the indexer returned.
        self: The checkpoint's indexer.
        *args: Its arguments.
        dump: The settings block's `debug.dump`, bound at registration.
        **kwargs: The same, by name, which is how the model calls it.

    Returns:
        None, which leaves the result as it was.
    """
    from lsa.sglang.index.dump import Dump

    forward_batch = kwargs["forward_batch"]
    if not forward_batch.forward_mode.is_extend():
        return None
    layer = kwargs["layer_id"]
    x = kwargs["x"]
    if not hasattr(self, "lsa_dump"):
        self.lsa_dump = Dump.of(dump, layer, x.shape[1], 1, x.device)
    if self.lsa_dump is not None:
        self.lsa_dump.prefill(x, kwargs["positions"], None, forward_batch, None)
    return None


def dump_qsa(result: Any, self: Any, *args: Any, dump: DumpSettings, **kwargs: Any) -> None:
    """Dump what Qwen's own sparse-attention indexer reads and selects at a prefill, to hold against the torch reference.

    The selection is already key positions, the open block's among them, so it is written as it is.

    Args:
        result: The selection [S, top_k + block_size - 1] int32, key positions, -1 padded.
        self: The QSA indexer.
        *args: `hidden_states`, `positions`, `forward_batch` and the indexer's metadata, in that order.
        dump: The settings block's `debug.dump`, bound at registration.
        **kwargs: The same, by name.

    Returns:
        None, which leaves the selection as it was.
    """
    from lsa.sglang.index.dump import Dump

    named = dict(zip(("hidden_states", "positions", "forward_batch"), args, strict=False)) | kwargs
    forward_batch = named["forward_batch"]
    if not forward_batch.forward_mode.is_extend():
        return None
    x = named["hidden_states"]
    if not hasattr(self, "lsa_dump"):
        self.lsa_dump = Dump.of(dump, self.layer_id, x.shape[1], result.shape[1], x.device)
    if self.lsa_dump is not None:
        positions = forward_batch.positions
        positions = positions[0] if positions.ndim == 2 else positions
        rows = result.shape[0]
        self.lsa_dump.prefill(x[:rows], positions[:rows], result, forward_batch, None)
    return None


def register_native_dump(dump: DumpSettings) -> None:
    """Register the native indexer's dump - GLM's own, and Qwen's where the image has it.

    Args:
        dump: The settings block's `debug.dump`, which names the directory.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for indexer in NATIVE_INDEXERS:
        HookRegistry.register(f"{indexer}.forward_cuda", partial(dump_native, dump=dump), HookType.AFTER)
    try:
        resolve(QSA_INDEXER)
    except (AttributeError, ModuleNotFoundError, ImportError) as absent:
        logger.info("lsa: no QSA indexer to dump - %s (%s)", QSA_INDEXER, absent)
        return None
    HookRegistry.register(QSA_INDEXER, partial(dump_qsa, dump=dump), HookType.AFTER)
    return None


def enable_pisa(config: PisaConfig) -> None:
    """Enable PISA on this process - GLM's through the same hooks as LSA's key and query input, without the projections, the
    context-parallel layouts and the decode graphs, and a grouped-query model's through LSA's backend hooks with PISA's
    selection and read.

    Every target is resolved before any hook is registered, as it is whenever a process is enabled.

    Args:
        config: The pyramid the settings block names.

    Raises:
        RuntimeError: A target no longer resolves.
        RuntimeError: The metadata glue graph is on, which would freeze index maintenance silently.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    missing = unresolved(
        TARGETS
        + ELIDED
        + REQUIRED
        + REQUIRED_PISA
        + (REPOINT_TARGET,)
        + KEY_AND_QUERY_TARGETS
        + GROUPED_QUERY_TARGETS
    )
    if missing:
        raise RuntimeError(
            "lsa cannot attach PISA to this sglang - these targets no longer resolve: " + "; ".join(missing)
        )

    refuse_metadata_glue()

    from lsa.sglang.index import pisa
    from lsa.sglang.models.glm5_next.pisa_indexer import PisaIndexer
    from lsa.sglang.models.qwen3_5 import backend
    from lsa.sglang.models.qwen3_5 import pisa as grouped_query_pisa

    hooks: tuple[tuple[str, Any, Any], ...] = (
        (TARGETS[1], PisaIndexer, HookType.REPLACE),
        (TARGETS[2], maintain, HookType.BEFORE),
        (TARGETS[3], maintain_out_graph, HookType.BEFORE),
        (TARGETS[4], fa3, HookType.AROUND),
        (TARGETS[5], flashmla, HookType.AROUND),
        (TARGETS[6], carry_request_ids, HookType.AFTER),
        (TARGETS[7], carry_request_ids, HookType.AFTER),
        (TARGETS[8], memory.count_cell, HookType.AFTER),
        (TARGETS[9], memory.reserve_fixed, HookType.AFTER),
        (TARGETS[10], memory.report, HookType.AFTER),
        (KEY_AND_QUERY_TARGETS[0], gate, HookType.AROUND),
        (KEY_AND_QUERY_TARGETS[1], prepare, HookType.AFTER),
        (REPOINT_TARGET, pisa.reopen_repointed, HookType.BEFORE),
    )
    # A grouped-query model's quadratic layers - enabled only where the model is one, so GLM's backends are left alone
    grouped_query_hooks: dict[str, tuple[Any, Any]] = {
        "__init__": (grouped_query_pisa.enable, HookType.AFTER),
        "init_forward_metadata": (backend.maintain, HookType.BEFORE),
        "init_forward_metadata_out_graph": (backend.maintain_out_graph, HookType.BEFORE),
        "forward_extend": (grouped_query_pisa.select_and_read, HookType.AROUND),
        "forward_decode": (grouped_query_pisa.select_and_read, HookType.AROUND),
    }
    hooks += tuple(
        (f"{attention}.{method}", *grouped_query_hooks[method])
        for attention in GROUPED_QUERY_BACKENDS
        for method in GROUPED_QUERY_METHODS
    )
    for target, hook, kind in hooks:
        HookRegistry.register(target, hook, kind)
    elide_index_keys()
    HookRegistry.apply_hooks()

    label = f"pisa-block_size{config.block_size}-keep_count{config.keep_count}-branching{config.branching}-beam{config.beam}"
    print(f"{Banner.READY} {label} width {config.width} from the model config's {Plugin.BLOCK!r}", flush=True)
    logger.info("lsa: %d hooks registered for PISA, index keys elided", len(hooks))


def enable_lsa(design: DesignSettings) -> None:
    """Enable the design's indexer on this process.

    Every target is resolved before any hook is registered, so a rename upstream fails here with the name in the
    message rather than silently leaving a server that runs the checkpoint's indexer under an LSA label.

    Args:
        design: The design the settings block names.

    Raises:
        RuntimeError: A target no longer resolves, or the design names no projection file.
        RuntimeError: The metadata glue graph is on, which would freeze index maintenance silently.
    """
    if not design.serving.projections:
        raise RuntimeError(
            f"the model config's {Plugin.BLOCK!r} names a design with no model.serving.projections - the indexer "
            "reads x through a trained map and there is no index without one"
        )

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # Resolved before this package's modules are imported
    # => importing the indexer first turns a renamed symbol into an ImportError, and nothing is registered
    key_and_query = design.indexer.input == KEY_AND_QUERY
    wanted_inputs = KEY_AND_QUERY_TARGETS if key_and_query else ()
    early_width = design.early_width
    wanted_tilelang = (TILELANG,) if early_width is not None else ()
    missing = unresolved(
        TARGETS + ELIDED + REQUIRED + GROUPED_QUERY_TARGETS + wanted_inputs + wanted_tilelang
    )
    if missing:
        raise RuntimeError(
            "lsa cannot attach to this sglang - these targets no longer resolve: " + "; ".join(missing)
        )

    refuse_metadata_glue()

    from lsa.sglang.models.glm5_next.indexer import LSAIndexer
    from lsa.sglang.models.qwen3_5 import backend

    hooks: tuple[tuple[str, Any, Any], ...] = (
        (TARGETS[1], LSAIndexer, HookType.REPLACE),
        (TARGETS[2], maintain, HookType.BEFORE),
        (TARGETS[3], maintain_out_graph, HookType.BEFORE),
        (TARGETS[4], fa3, HookType.AROUND),
        (TARGETS[5], flashmla, HookType.AROUND),
        (TARGETS[6], carry_request_ids, HookType.AFTER),
        (TARGETS[7], carry_request_ids, HookType.AFTER),
        (TARGETS[8], memory.count_cell, HookType.AFTER),
        (TARGETS[9], memory.reserve_fixed, HookType.AFTER),
        (TARGETS[10], memory.report, HookType.AFTER),
    )
    grouped_query_hooks: dict[str, tuple[Any, Any]] = {
        "__init__": (backend.enable, HookType.AFTER),
        "init_forward_metadata": (backend.maintain, HookType.BEFORE),
        "init_forward_metadata_out_graph": (backend.maintain_out_graph, HookType.BEFORE),
        "forward_extend": (backend.select_and_read, HookType.AROUND),
        "forward_decode": (backend.select_and_read, HookType.AROUND),
    }
    hooks += tuple(
        (f"{attention}.{method}", *grouped_query_hooks[method])
        for attention in GROUPED_QUERY_BACKENDS
        for method in GROUPED_QUERY_METHODS
    )
    if key_and_query:
        hooks += (
            (KEY_AND_QUERY_TARGETS[0], gate, HookType.AROUND),
            (KEY_AND_QUERY_TARGETS[1], prepare, HookType.AFTER),
        )
    if early_width is not None:
        hooks += ((TILELANG, partial(tilelang, early_width=early_width), HookType.AROUND),)
    for target, hook, kind in hooks:
        HookRegistry.register(target, hook, kind)
    elide_index_keys()

    # Registered where the decode runner has the variants, skipped where it does not
    registered = len(hooks) + int(register_context_graphs())
    HookRegistry.apply_hooks()

    label = f"lsa-top_level{design.indexer.top_level}-decode_{design.serving.decode}-{design.indexer.input}"
    # The input, the heads it sums and the map, so a server's log says which of them it serves
    heads = f" query_heads {design.indexer.query_heads}" if design.indexer.query_heads is not None else ""
    print(
        f"{Banner.READY} {label}{heads} width {design.width} map {design.serving.projections} "
        f"from the model config's {Plugin.BLOCK!r}",
        flush=True,
    )
    logger.info("lsa: %d hooks registered, index keys elided", registered)


def enable_native_dump(dump: DumpSettings) -> None:
    """Enable the dump of what the checkpoint's own indexer reads, for a server whose block names no design.

    Args:
        dump: The settings block's `debug.dump`.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry

    register_native_dump(dump)
    HookRegistry.apply_hooks()
    logger.info("lsa: dumping the native indexer's x")


# What this process was enabled with, so that a second config in it is held to the same settings
ENABLED: list[PluginSettings] = []


def enable(served: PluginSettings) -> None:
    """Register the hooks the settings ask for, once per process.

    Args:
        served: What the model config's block asks of this server, which is not the checkpoint's own with no checks.

    Raises:
        RuntimeError: This process was enabled with other settings, a target no longer resolves, or the metadata glue graph is on.
    """
    if ENABLED:
        if ENABLED[0] != served:
            raise RuntimeError(
                f"this process was enabled from one {Plugin.BLOCK!r} block, which names {ENABLED[0].selector}, and a "
                f"second model config carries another, which names {served.selector} - a server runs one block"
            )
        return None
    if served.pisa is not None:
        enable_pisa(served.pisa)
    elif served.design is not None:
        enable_lsa(served.design)
    elif served.debug.dumps:
        enable_native_dump(served.debug.dump)
    ENABLED.append(served)
    return None


def register() -> None:
    """Carry the plugin's one standing hook, on the model config.

    Called once per process by sglang's plugin loader, before any config exists.
    Whether to enable is not known yet, and sglang applies a hook for good, so nothing a design needs is
    registered here: `configure` runs after every model config is built, finds the block, and enables the process
    then - which every process that runs a model does before it builds one.
    A server with no block pays one attribute read per model config it builds.

    Raises:
        RuntimeError: The model config's constructor no longer resolves.
    """
    missing = unresolved((MODEL_CONFIG,))
    if missing:
        raise RuntimeError("lsa cannot read its settings from this sglang - " + "; ".join(missing))

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(MODEL_CONFIG, configure, HookType.AFTER)
    logger.info("lsa: settings are read from the model config's %r block", Plugin.BLOCK)
