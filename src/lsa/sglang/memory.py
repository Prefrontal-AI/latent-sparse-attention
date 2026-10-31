"""What the indexer holds on a GPU beside sglang's KV pool, counted into the pool's own sizing.

sglang sizes its KV pool from `--mem-fraction-static`: what is free after the weights, less a slack for activations,
divided by the bytes a KV slot costs across the layers - the cell. Every structure the indexer allocates is built
after that, on first use, so a pool sized without them leaves the activations' slack to hold them as well, and a long
prefill runs out of memory where native does not. On GLM-5.3-Flash at 0.85 that was 11.6 GiB a GPU, at 7.43M slots.

Most of it grows with the pool, so it is counted the way sglang counts its own: as bytes a slot, added to the cell.
The pool comes out smaller by exactly the share the indexer takes, and the slack is left to the activations as it
is under native. What does not grow with the pool is taken off the budget before it is divided: every layer's
projections and the pages a request's last partial page costs - and once, since the layers read in turn, the
selection buffers they share and what one layer's prefill selection allocates and frees at its peak, its listings
among it, which a top-k hands back after each read.

That peak is the one part not counted from the settings. A read's score rows, listings and expansion are set by the
counts it reads over - flat's dense scores below its fused gate, a tree's widest level, the level its beam starts at
- and the kernels that size them are tuned often, so it is measured on the GPU before the pool exists: one layer's
index grown to the context on random keys, read a chunk at a time at every power of two. It grows with the context
for a tree - fixed scores every level-1 node - and with the listing a read makes, it is what ran the fixed and
tree servers out of memory on a 512k prefill at 0.80.

What a slot costs a sparse layer:

| structure | bytes a slot | |
|---|---|---|
| index keys | `index_dim` bf16, none under the key and query input unless tensor parallel splits the key heads | `KeyBuffer`, at the KV slot - under decode context parallel, at the slots this rank owns alone |
| buckets | centroid, members, size and shape of a bucket, over `bucket_size` keys, and a page's position over `PAGE_SIZE` | `BucketCache` |
| hierarchy | every level's nodes and their shapes over the keys beneath them, twice for the doubling rows | `Hierarchy`, a bound |
| graph tree | a descending decode's mirror of every level's node buckets, one pooled page table a level | `GraphTree`, allocated whole |

The hierarchy is the one structure that is not sized up front - its rows grow by doubling, one hierarchy a
request. Level `l` holds a node over `bucket_size * node_bucket_size ** l` keys, with a centroid and its children.
Below the top a node first waits open with the fp32 key sum its pairing reads, and is then closed without it.
Counting every node as both open and closed, at twice the rows it fills, bounds what the doubling rows hold at any
moment - as long as a level's first growth fills the 64 rows a `Rows` starts with, which a node cluster block of
64 node buckets or more does, and every design's does.
A finished request keeps its hierarchy only as long as it keeps its bucket pages, so the pages bound it.

The key and query input keeps no index keys of its own: a key's index key is projected from the key the KV pool
already holds, when it is read - GLM's latent, or a grouped-query model's key heads past their rotated prefix - so the
slot costs the indexer its buckets and hierarchy alone, and the projection's reads are part of the selection's peak.
Tensor parallel splits a grouped-query model's key heads, so a rank's pool cannot make an index key: each is made when
written and kept at its KV slot on every rank, `index_dim` bf16 a slot - 128 B beside the 1 KiB of K and V a Qwen
layer's slot costs a rank under tensor parallel 8.

The graph tree is the one the hierarchy's mirror: a descending decode, which a CUDA graph captures, reads the node buckets
from pools at addresses that never move. Its pools are sized from the cache's bucket rows, so they grow with the pool
and count as bytes a slot, and a page of each level for each request of the table, the page tables and the counts
count as fixed. Unlike the hierarchy's rows it is allocated whole when the layer's index is built, not as requests grow.

Under decode context parallel a rank's pool holds its share of the slots, and the cell sglang divides by is the
cost of one of those rows. A rank writes index keys at its own slots, but keeps the bucket cache for every slot
of the context and every hierarchy over it - `size` slots a row - so the bucket, hierarchy and graph tree bytes count
`size` times, for a tree as for flat.

PISA keeps no bucket cache and no index keys. Its share has no slot cost - the cell is sglang's own - and holds the means of
the leaf blocks, a row of them for each of the request table's rows in every layer, and the peak of one layer's search: the
leaf blocks' gathered keys and their scores, and the queries gathered across the attention group.

[plugin](plugin.py) registers the three hooks at the bottom: the cell, the budget, and the line that says what
both came to.
A tree's count is a bound, so the bucket cache also says what a finished request actually held a key in a layer -
centroids apart from the rest of its index - each time a request is the longest the server has seen.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lsa.constants import Attention
from lsa.sglang.settings.design import KEY_AND_QUERY, PAGE_SIZE, DesignSettings, IndexerConfig, Shape

# The index is imported where it is built, so the job host reading these counts imports no kernel
if TYPE_CHECKING:
    from lsa.sglang.index.layer_index import KeySlot

# Request table size when the server sets no `--max-running-requests`, an upper bound on the engine's pick from 2,048 up
DEFAULT_REQUESTS: int = 4096

# Rows a doubling `Rows` holds at most against the rows it fills
ROWS_GROWTH: int = 2

# Context a request table row holds past the server's context length - sglang's own allowance is a handful
CONTEXT_ALLOWANCE: int = 64

# The request the selection probe grows - one row of its own request table
PROBE_REQUEST: str = "lsa-lsa-memory-probe"

# Bytes of the element types the structures are built from
BF16: int = 2
FP8: int = 1
FP32: int = 4
INT32: int = 4
INT64: int = 8


def index_key_bytes(config: IndexerConfig, split_heads: bool = False) -> int:
    """Return what one KV slot's index key costs a layer - one bf16 row of `index_dim`, or none under the key and query.

    A key and query rank whose pool holds some of the key heads keeps the row too.

    Args:
        config: The design's indexer.
        split_heads: Whether tensor parallel splits the key heads, so a rank's pool cannot make an index key.
    """
    return 0 if config.input == KEY_AND_QUERY and not split_heads else config.index_dim * BF16


def shape_bytes(config: IndexerConfig, shape: Shape | None) -> int:
    """Return what one bucket's or node's shape costs - a sign bit a dimension a direction, its variances and a floor.

    Args:
        config: The design's indexer.
        shape: The shape, or None for an unshaped bucket.
    """
    if shape is None:
        return 0
    return shape.rank * (config.index_dim // 8 + FP32) + FP32


def bucket_bytes(config: IndexerConfig) -> float:
    """Return what one key costs a layer's bucket cache.

    A bucket holds a bf16 centroid, `bucket_cap` int32 members, an int32 size and, under `bucket_shape`, its shape,
    one bucket for each `bucket_size` keys, and a page of buckets an int32 first position.

    Args:
        config: The design's indexer.
    """
    bucket = (
        config.index_dim * BF16 + config.bucket_cap * INT32 + INT32 + shape_bytes(config, config.bucket_shape)
    )
    return bucket / config.bucket_size + INT32 / PAGE_SIZE


def hierarchy_bytes(config: IndexerConfig, node_centroid_precision: str) -> float:
    """Return a bound on what one key costs a layer's hierarchies, from the design's settings.

    Args:
        config: The design's indexer.
        node_centroid_precision: `bf16` or `fp8`, what a node's centroid is stored as.

    Returns:
        Bytes a key, 0 for a flat design.
    """
    centroid = config.index_dim * (FP8 if node_centroid_precision == "fp8" else BF16)
    total, keys = 0.0, config.bucket_size
    for level in range(1, config.top_level + 1):
        size = config.at("node_bucket_size", level)
        keys *= size
        node = centroid + size * INT32 + shape_bytes(config, config.node_bucket_shape)
        # Below the top a node waits open with its key sum, then is closed without it - every node in both, at worst.
        # The top level is never paired, so its nodes stay open and keep no key sum
        held = node + config.index_dim * FP32 + node if level < config.top_level else node
        total += held / keys
    return ROWS_GROWTH * total


def graph_tree_bytes(settings: DesignSettings, requests: int, context: int) -> tuple[float, int]:
    """Return what a layer's graph tree costs a key and what it costs however many keys the pool holds.

    Args:
        settings: The design.
        requests: Rows of the request table.
        context: Longest context a request holds.

    Returns:
        Bytes a key, and bytes a layer - both 0 for a design that builds no graph tree.
    """
    # Imported where it is built, as the index is
    from lsa.sglang.index.graph_tree import GraphTree

    if not GraphTree.builds(settings):
        return 0.0, 0
    per_key = GraphTree.bytes_per_bucket(settings, context) / settings.indexer.bucket_size
    return per_key, GraphTree.fixed_bytes(settings, requests, context)


def fixed_bytes(
    settings: DesignSettings,
    requests: int,
    context: int,
    input_dim: int,
    page_size: int,
    decode_cp: int = 1,
    split_heads: bool = False,
) -> int:
    """Return what a layer's index holds that does not grow with the KV pool.

    Args:
        settings: The design.
        requests: Rows of the request table.
        context: Longest context a request holds.
        input_dim: Width of the indexer's input - `x`, or the key latent under the key and query.
        page_size: The KV pool's page, which the index-key buffer holds past the pool.
        decode_cp: The decode context-parallel group size, 1 without it.
        split_heads: Whether tensor parallel splits the key heads, under the key and query.

    Returns:
        Bytes a layer.
    """
    config = settings.indexer
    tree_per_key, tree_fixed = graph_tree_bytes(settings, requests, context)
    per_key = (
        bucket_bytes(config)
        + hierarchy_bytes(config, settings.serving.node_centroid_precision)
        + tree_per_key
    )
    # A request's last page is partial, and the cache rounds the pool up to a whole page
    # The spare contexts are whole contexts of pages beyond the pool's - the cache sizes them the same way
    pages = (requests + 1 + settings.serving.spare_contexts * -(-context // PAGE_SIZE)) * PAGE_SIZE * per_key
    page_table = requests * (-(-context // PAGE_SIZE) + 1) * INT32
    keys = (page_size + int(decode_cp > 1)) * index_key_bytes(config, split_heads)
    # The key and query's maps are held in bf16, the attention input's in fp32
    projections = input_dim * 2 * config.index_dim * (BF16 if config.input == KEY_AND_QUERY else FP32)
    # A decode step's listing is exact, k keys a row, in buffers sized once
    decode_listing = requests * (config.top_p * INT64 + INT32)
    return math.ceil(pages + page_table + keys + projections + decode_listing + tree_fixed)


def shared_bytes(settings: DesignSettings, prefill_chunk: int) -> int:
    """Return what the layers' prefill selection holds between them, in buffers sized once.

    Layers prefill one after another, so one fused select's output and one beam select for each width serve every
    layer. The top-ks keep no listing between reads, so what a read lists is the selection's peak.

    Args:
        settings: The design.
        prefill_chunk: Most query rows a prefill call is handed.

    Returns:
        Bytes, once for the rank.
    """
    config = settings.indexer
    scan_select = prefill_chunk * config.top_p * INT32
    beam_widths = {config.at("beam_width", level) for level in range(1, config.top_level + 1)}
    beam_selects = sum(prefill_chunk * width * INT32 for width in beam_widths)
    return scan_select + beam_selects


def probe_transient(
    settings: DesignSettings,
    context: int,
    prefill_chunk: int,
    input_dim: int,
    heads: int = 1,
    key_slot: "KeySlot | None" = None,
) -> int:
    """Return the most one layer's prefill selection allocates above what it held before, measured on this GPU.

    What a read allocates and frees - its score rows, listings and expansion - is set by the counts it reads over,
    which move with the context and step where a design changes how it reads: flat's dense scores below its fused
    gate, a tree's widest level, the level a beam starts at. Every such step falls on a power of two, so one request
    of random unit keys grows to the context by maintenance, and is read a chunk at a time over the last growth of
    its hierarchy before each power of two and before the context itself - the buckets no node covers yet rise and
    fall over a growth, and the read scores them. Layers read one after another, so a server holds one layer's peak
    at a time.

    Under the key and query input the index keys are projected from a KV pool's keys when they are read, so the
    probe holds a pool buffer of its own for the one layer, shaped as the model's slot is, and hands the read a
    chunk's keys and its heads' queries - all allocated before the peak is measured, as the attention's own are.

    Args:
        settings: The design.
        context: Longest context a request holds.
        prefill_chunk: Most query rows a prefill call is handed.
        input_dim: Width of the indexer's input - `x`, or the key latent under the key and query.
        heads: Heads a rank's queries hold, under the key and query - one for a grouped-query model, whose groups
            are summed before the read.
        key_slot: Where the key input sits in a KV pool slot under the key and query - GLM's latent alone when None.

    Returns:
        Bytes, with what the probe leaves allocated behind it.
    """
    import torch
    import torch.nn.functional as F

    from lsa.sglang.index.hierarchy import GROWTH_BUCKETS
    from lsa.sglang.index.layer_index import KeyAndQuery, KeySlot, LayerIndex
    from lsa.sglang.index.selector import clear_scratches

    device = torch.device("cuda", torch.cuda.current_device())
    held = torch.cuda.memory_allocated(device)
    index_dim = settings.indexer.index_dim
    chunk = min(prefill_chunk, context)
    # The keys of one growth, in whole chunks, as far as the context goes
    growth = GROWTH_BUCKETS * settings.indexer.bucket_size
    window = min(-(-growth // chunk) * chunk, context // chunk * chunk)
    powers = [
        1 << power
        for power in range(window.bit_length(), context.bit_length())
        if 1 << power <= context - window
    ]
    generator = torch.Generator(device=device).manual_seed(0)
    table = torch.arange(context, device=device, dtype=torch.int32)[None]
    maps = torch.randn(input_dim, index_dim), torch.randn(input_dim, index_dim)
    key_and_query = settings.indexer.input == KEY_AND_QUERY
    slot = key_slot or KeySlot.latent(input_dim)
    # The layer's KV pool under the key and query, random keys in it, counted by sglang's pool sizing
    pool = (
        [
            torch.randn(
                context, slot.heads, slot.head_dim, device=device, generator=generator, dtype=torch.bfloat16
            )
        ]
        if key_and_query
        else []
    )
    index = LayerIndex(
        settings,
        maps,
        context,
        context,
        lambda: table,
        context,
        chunk,
        device,
        key_buffer=(lambda: pool[0]) if pool else None,
        key_slot=slot if key_and_query else None,
    )
    if not key_and_query:
        index.index_keys[:context] = F.normalize(
            torch.randn(context, index_dim, device=device, generator=generator), dim=-1
        ).to(torch.bfloat16)
    peak = 0
    for end in [*powers, context]:
        index.maintain([PROBE_REQUEST], [0], [end - window])
        for start in range(end - window, end, chunk):
            inputs: torch.Tensor | KeyAndQuery
            if key_and_query:
                inputs = KeyAndQuery(
                    torch.randn(chunk, input_dim, device=device, generator=generator, dtype=torch.bfloat16),
                    torch.randn(
                        chunk, heads, input_dim, device=device, generator=generator, dtype=torch.bfloat16
                    ),
                )
            else:
                inputs = torch.randn(chunk, input_dim, device=device, generator=generator).to(torch.bfloat16)
            positions = torch.arange(start, start + chunk, device=device, dtype=torch.int32)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            index.extend(inputs, positions, positions.long(), [0], [chunk], [start])
            torch.cuda.synchronize(device)
            peak = max(peak, torch.cuda.max_memory_allocated(device) - before)
            del inputs
    del index, positions
    pool.clear()
    clear_scratches()
    torch.cuda.empty_cache()
    # Library workspaces from a first read stay
    # => added after the budget measured before them
    return peak + torch.cuda.memory_allocated(device) - held


@dataclass(frozen=True)
class Share:
    """What the indexer holds on one rank, as sglang's pool sizing counts it."""

    # Sparse layers this rank indexes
    layers: int
    # Bytes a KV slot of this rank costs a layer
    index_keys: float
    buckets: float
    hierarchy: float
    # Bytes this rank holds that the pool's size does not change, every layer together
    fixed: int
    # Bytes the layers share, once
    shared: int = 0
    # Bytes one layer's prefill selection allocates at its peak, which the layers take in turn
    transient: int = 0
    # Bytes a KV slot costs a layer's graph tree, 0 where the decode does not descend
    graph_tree: float = 0.0

    @property
    def slot(self) -> float:
        """Bytes a KV slot costs a layer."""
        return self.index_keys + self.buckets + self.hierarchy + self.graph_tree

    @property
    def cell(self) -> int:
        """Bytes a KV slot costs every layer together - what sglang's cell grows by."""
        return math.ceil(self.layers * self.slot)

    @property
    def reserved(self) -> int:
        """Bytes taken off the budget before it is divided into slots."""
        return self.fixed + self.shared + self.transient

    def pool_bytes(self, slots: int) -> int:
        """Return what the indexer holds over a pool of `slots` at its peak, the fixed part included."""
        return math.ceil(self.cell * slots) + self.reserved


def share(
    settings: DesignSettings,
    layers: int,
    requests: int,
    context: int,
    prefill_chunk: int,
    input_dim: int,
    page_size: int,
    decode_cp: int = 1,
    transient: int = 0,
    split_heads: bool = False,
) -> Share:
    """Return what the indexer holds on a rank, a slot at a time and fixed.

    Args:
        settings: The design.
        layers: Sparse layers on this rank.
        requests: Rows of the request table.
        context: Longest context a request holds.
        prefill_chunk: Most query rows a prefill call is handed.
        input_dim: Width of the indexer's input - `x`, or the key latent under the key and query.
        page_size: The KV pool's page.
        decode_cp: The decode context-parallel group size, 1 without it - a rank's bucket cache covers this many
            slots for each of its own.
        transient: One layer's prefill selection at its peak, as `probe_transient` measures it.
        split_heads: Whether tensor parallel splits the key heads, under the key and query - a rank then holds an
            index key a slot, as it does on the attention input.

    Returns:
        The share.
    """
    config = settings.indexer
    precision = settings.serving.node_centroid_precision
    tree_per_key, _ = graph_tree_bytes(settings, requests, context)
    return Share(
        layers=layers,
        index_keys=index_key_bytes(config, split_heads),
        buckets=decode_cp * bucket_bytes(config),
        hierarchy=decode_cp * hierarchy_bytes(config, precision),
        fixed=layers * fixed_bytes(settings, requests, context, input_dim, page_size, decode_cp, split_heads),
        shared=shared_bytes(settings, prefill_chunk) if layers else 0,
        transient=transient if layers else 0,
        graph_tree=decode_cp * tree_per_key,
    )


# What the hooks counted, read back by the line that reports it and by the indexer that checks it ran
COUNTED: dict[str, Any] = {}


# Gathered keys, their fp32 scores and the masked copy of those, each at most a tile of the search's bytes
PISA_TILE_COPIES: int = 3


def pisa_share(kvc: Any) -> Share:
    """Return PISA's share on the rank a KV cache configurator sizes the pool for.

    Args:
        kvc: sglang's `KVCacheConfigurator`.
    """
    from sglang.srt.runtime_context import get_parallel, get_schedule

    from lsa.sglang import settings
    from lsa.sglang.index import pisa, weights
    from lsa.sglang.models.qwen3_5 import index_space

    config = settings.pisa_of(kvc.model_config.hf_config)
    text = kvc.model_config.hf_text_config
    info = kvc.layer_info
    layers = [layer for layer in weights.indexed_layers(text) if info.start_layer <= layer < info.end_layer]
    schedule = get_schedule()
    requests = (schedule.max_running_requests or DEFAULT_REQUESTS) + 1
    leaves = math.ceil((kvc.model_config.context_len + CONTEXT_ALLOWANCE) / config.block_size)
    if index_space.is_grouped_query(text):
        # A leaf mean a key head this rank holds, and the queries are this rank's own, never gathered
        geometry = index_space.geometry_of(text)
        key_dim = index_space.kv_heads_held(geometry, get_parallel().attn_tp_size) * geometry.head_dim
        gathered_queries = 0
    else:
        key_dim = int(text.kv_lora_rank)
        gathered_queries = settings.prefill_chunk() * int(text.num_attention_heads) * key_dim * BF16
    # The leaf means and the pyramid above them, about as many summaries again
    blocks = sum(pisa.level_widths(leaves, config.branching))
    means = len(layers) * requests * blocks * key_dim * FP32
    transient = PISA_TILE_COPIES * pisa.TILE_BYTES + 2 * gathered_queries
    return Share(len(layers), 0, 0, 0, fixed=means, transient=transient if layers else 0)


def share_of(kvc: Any) -> Share:
    """Return the indexer's share on the rank a KV cache configurator sizes the pool for.

    The selection's peak is measured once, the first time the share is asked for, which is before the KV pool
    exists.

    Args:
        kvc: sglang's `KVCacheConfigurator`.
    """
    from sglang.srt.runtime_context import get_schedule

    from lsa.sglang import context_parallel, settings
    from lsa.sglang.index import weights
    from lsa.sglang.models.qwen3_5 import index_space

    if settings.of(kvc.model_config.hf_config).selector == Attention.PISA:
        return pisa_share(kvc)

    text = kvc.model_config.hf_text_config
    info = kvc.layer_info
    # The engine's own answer for GLM, which the plugin leaves in place where the model reads it
    layers = [layer for layer in weights.indexed_layers(text) if info.start_layer <= layer < info.end_layer]
    design = settings.design_of(kvc.model_config.hf_config)
    context = kvc.model_config.context_len
    chunk = settings.prefill_chunk()
    input_dim = weights.input_dim(text, design.indexer.input)
    if "transient" not in COUNTED:
        COUNTED["transient"] = (
            probe_transient(
                design,
                context,
                chunk,
                input_dim,
                weights.query_rows(text, context_parallel.attention_tp_size()),
                weights.key_slot(text),
            )
            if layers
            else 0
        )
    schedule = get_schedule()
    running = schedule.max_running_requests
    decode_size = context_parallel.decode_cp()[0]
    return share(
        design,
        len(layers),
        (running or DEFAULT_REQUESTS) + 1,
        context + CONTEXT_ALLOWANCE,
        chunk,
        input_dim,
        int(schedule.page_size or 1),
        decode_size,
        COUNTED["transient"],
        split_heads=index_space.is_grouped_query(text)
        and index_space.splits_key_heads(context_parallel.attention_tp_size(), decode_size),
    )


def count_cell(result: int, self: Any, kvc: Any, *args: Any, **kwargs: Any) -> int:
    """Add the indexer's bytes a slot to the cell sglang divides the KV budget by.

    Args:
        result: The cell sglang computed, its own KV across the layers.
        self: The pool configurator.
        kvc: The KV cache configurator it was built for.
        *args: The layer count, unused.
        **kwargs: The same, unused.

    Returns:
        The cell with the indexer's share in it.
    """
    if getattr(kvc, "is_draft_worker", False):
        return result
    counted = share_of(kvc)
    COUNTED.update(share=counted, native_cell=int(result))
    return int(result) + counted.cell


def reserve_fixed(result: int, self: Any, *args: Any, **kwargs: Any) -> int:
    """Take what the indexer holds regardless of the pool off the budget the pool is sized from.

    Args:
        result: The bytes sglang found for the pool.
        self: The KV cache configurator.
        *args: The memory before the model loaded, unused.
        **kwargs: The same, unused.

    Returns:
        The budget less the indexer's fixed bytes and its selection's peak.
    """
    counted = share_of(self)
    COUNTED.update(budget=int(result))
    return int(result) - counted.reserved


def report(result: Any, self: Any, *args: Any, **kwargs: Any) -> None:
    """Print what the indexer's share came to and the pool it left, once the pool is sized.

    Args:
        result: The memory pool config sglang resolved.
        self: The KV cache configurator.
        *args: Its arguments, unused.
        **kwargs: The same, unused.

    Returns:
        None, which leaves the config as it was.
    """
    counted: Share | None = COUNTED.get("share")
    if counted is None or "budget" not in COUNTED:
        return None
    slots = int(result.max_total_num_tokens)
    native_cell = COUNTED["native_cell"]
    page = max(1, int(getattr(self, "page_size", 1) or 1))
    # What the same budget would have held with nothing of the indexer's counted - before any token cap
    uncounted = COUNTED["budget"] // native_cell // page * page if native_cell else 0
    COUNTED.update(slots=slots, uncounted=uncounted)
    print(
        f"lsa-lsa: memory - {counted.slot:.1f} B a slot a sparse layer (index keys {counted.index_keys:.1f}, "
        f"buckets {counted.buckets:.1f}, hierarchy {counted.hierarchy:.1f}, graph tree {counted.graph_tree:.1f}) x {counted.layers} layers = "
        f"{counted.cell} B a slot beside sglang's {native_cell}, {counted.fixed / 2**30:.2f} GiB fixed, {counted.shared / 2**30:.2f} GiB shared and "
        f"{counted.transient / 2**30:.2f} GiB of one layer's selection at its peak; "
        f"KV pool {slots} slots, {uncounted} with none of it counted; "
        f"the indexer holds {counted.pool_bytes(slots) / 2**30:.2f} GiB at most",
        flush=True,
    )
    return None


def check_built(layer: int, built: int, slots: int) -> None:
    """Say what the first layer's index took when it was built, against what the pool's sizing counted for it.

    The beam's selects are built on first use, and its hierarchies grow with
    the requests, so the count runs over what is built here. A pool sized without the count says so, since its
    server can run out of memory on a long prefill where native does not.

    Args:
        layer: The layer just built.
        built: Bytes its build allocated on the device.
        slots: Slots of the KV pool on this rank.
    """
    if "built" in COUNTED:
        return
    COUNTED["built"] = built
    counted: Share | None = COUNTED.get("share")
    if counted is None:
        print(
            f"lsa-lsa: memory - layer {layer}'s index took {built / 2**20:.1f} MiB that sglang's KV pool sizing "
            "never counted - this pool configurator is not hooked, so lower --mem-fraction-static by what the "
            "sparse layers hold",
            flush=True,
        )
        return
    # The first layer built also built what the layers share
    accounted = (counted.cell * slots + counted.fixed) / max(counted.layers, 1) + counted.shared
    print(
        f"lsa-lsa: memory - layer {layer}'s index took {built / 2**20:.1f} MiB when built, "
        f"{accounted / 2**20:.1f} MiB counted a layer over {slots} slots",
        flush=True,
    )


def report_held(keys: int, held: dict[str, int]) -> None:
    """Say what a finished request's index held a key in one layer, when it is the longest request yet.

    The pool's count is a bound for a tree - its rows grow by doubling and every node counts as open and closed -
    so this line is what a request of `keys` keys actually held. Every layer releases the request, and only the
    first to report a new longest one prints, so a server says it once a new longest request.

    Args:
        keys: Keys the request appended.
        held: Bytes each structure of its index held, as `BucketCache.held` returns them.
    """
    if keys <= COUNTED.get("longest", 0):
        return
    COUNTED["longest"] = keys
    centroids = sum(count for name, count in held.items() if name.endswith("centroids"))
    rest = sum(held.values()) - centroids
    parts = ", ".join(f"{name} {count / keys:.2f}" for name, count in held.items())
    print(
        f"lsa-lsa: memory - a request of {keys} keys held {centroids / keys:.2f} B a key a layer of centroids and "
        f"{rest / keys:.2f} of the rest of its index ({parts})",
        flush=True,
    )
