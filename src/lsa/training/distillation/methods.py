"""The offline methods of the selection study - a grouping and a score read through `lsa.core`, and PISA's block pyramid.

Each method is read over the same rows of the same document as a served design, and scored as one is.
A reference method builds the buckets and the node bucket levels of a served design with the torch reference, at that
design's sizes, with the research options of the designs.
A PISA method searches a pyramid of positional blocks, over the model's own keys and the heads' queries - GLM's key
latent, or each of a model's key-value heads with the query heads that read it, one read a key-value head - or over a
design's index keys and index queries as one head.

Every method's read is its long-range keys, then the window, then the sinks - the layout a served read has - so a
metric that leaves the window and the sinks out of what it counts leaves them out for every method alike.
A method's cost is the vectors it scores a query against, and the dots that are, a PISA query head dotting each.

The torch reference runs on the host, so the methods read index keys and queries moved there.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from lsa.baselines import pisa
from lsa.core import Buckets, Hierarchy, build_buckets, build_hierarchy, read_hierarchy_counted
from lsa.sglang.index.hierarchy import GROWTH_BUCKETS
from lsa.sglang.index.layer_index import KeyAndQuery
from lsa.sglang.settings.design import PAGE_SIZE, DesignSettings, IndexerConfig
from lsa.training.distillation.constants import Served
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.export import map_tau
from lsa.training.distillation.metrics import window_and_sinks

# The config module reaches omegaconf and hydra, so importing it here would fail before `require_extra` runs
# => the guard names the extra and the import happens once past it
if TYPE_CHECKING:
    from lsa.training.distillation.config import MethodConfig


class MethodRead(NamedTuple):
    """What a method selected for some rows, and what it cost."""

    # The read [R, W] int32, or [R, kv_head_count, W] one a key-value head - long-range keys, the window, the sinks,
    # -1 padded
    selected_key_ids: torch.Tensor
    # Vectors each row scored [R] int64
    scored_vectors: torch.Tensor
    # Dot products each row took [R] int64 - the vectors, each once a query head
    scored_dots: torch.Tensor


@dataclass(frozen=True)
class Grouping:
    """What decides one set of buckets and levels - the sizes of a design and the algorithms that group over them."""

    pool_size: int
    bucket_size: int
    bucket_cap: int
    cluster_block_size: int
    cluster_algorithm: str


class Structures:
    """The buckets and levels of one document's index keys, built once for every method that reads the same grouping."""

    def __init__(self, index_keys: torch.Tensor) -> None:
        """Hold the keys the structures are built over.

        Args:
            index_keys: Every position's index key [T, index_dim] bf16 or fp32, unit rows, on the host.
        """
        self.index_keys = index_keys
        self._buckets: dict[Grouping, Buckets] = {}
        self._hierarchies: dict[tuple[Grouping, int, int, tuple[int, ...], str], Hierarchy] = {}

    def buckets(self, indexer: IndexerConfig, cluster_algorithm: str) -> tuple[Grouping, Buckets]:
        """Return the buckets of a design's sizes, built the first time they are asked for.

        Args:
            indexer: The design's knobs.
            cluster_algorithm: `ward` or `position`.

        Returns:
            What grouped them, and the buckets.
        """
        grouping = Grouping(
            indexer.pool_size or 1,
            indexer.bucket_size,
            indexer.bucket_cap,
            indexer.cluster_block_size,
            cluster_algorithm,
        )
        if grouping not in self._buckets:
            self._buckets[grouping] = build_buckets(
                self.index_keys,
                grouping.pool_size,
                grouping.bucket_size,
                grouping.bucket_cap,
                grouping.cluster_block_size,
                cluster_algorithm,
            )
        return grouping, self._buckets[grouping]

    def hierarchy(
        self, indexer: IndexerConfig, cluster_algorithm: str, node_cluster_algorithm: str
    ) -> Hierarchy:
        """Return the node bucket levels of a design's sizes, built the first time they are asked for.

        Args:
            indexer: The design's knobs, whose `top_level` is at least 1.
            cluster_algorithm: What grouped the buckets, `ward` or `position`.
            node_cluster_algorithm: What pairs them, `ward` or `position`.

        Returns:
            The levels.
        """
        grouping, buckets = self.buckets(indexer, cluster_algorithm)
        levels = range(1, indexer.top_level + 1)
        block_sizes = tuple(indexer.at("node_cluster_block_size", level) for level in levels)
        node_bucket_size = indexer.at("node_bucket_size", 1)
        key = (grouping, indexer.top_level, node_bucket_size, block_sizes, node_cluster_algorithm)
        if key not in self._hierarchies:
            self._hierarchies[key] = build_hierarchy(
                self.index_keys,
                buckets,
                indexer.top_level,
                node_bucket_size,
                block_sizes,
                node_cluster_algorithm,
            )
        return self._hierarchies[key]


def read_reference(
    method: "MethodConfig",
    indexer: IndexerConfig,
    structures: Structures,
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    tau: float,
) -> MethodRead:
    """Read some rows with a bucket cache and levels built at a design's sizes, as the torch reference reads them.

    Args:
        method: The method, whose algorithms and scores are the research options.
        indexer: The served design whose sizes it reads at, with a hierarchy - `top_level` of at least 1.
        structures: The buckets and levels of the document's index keys.
        index_queries: The rows' index queries [R, index_dim] fp32, unit rows, on the host.
        positions: The rows' positions [R] int64, on the host.
        tau: The temperature an `lse` score is read at.

    Returns:
        The read and its cost.

    Raises:
        ValueError: The design has no hierarchy to read.
    """
    if indexer.top_level < 1:
        raise ValueError(
            f"method {method.name!r} reads a hierarchy and design {method.design!r} has top_level 0"
        )
    _, buckets = structures.buckets(indexer, method.cluster_algorithm)
    hierarchy = structures.hierarchy(indexer, method.cluster_algorithm, method.node_cluster_algorithm)
    selected, _, scored = read_hierarchy_counted(
        index_queries,
        positions,
        buckets,
        hierarchy,
        [indexer.at("beam_width", level) for level in range(1, indexer.top_level + 1)],
        indexer.start_nodes,
        indexer.top_p,
        indexer.top_k,
        indexer.sink_size,
        indexer.window_size,
        PAGE_SIZE,
        GROWTH_BUCKETS,
        method.node_bucket_score,
        method.bucket_score,
        tau,
        structures.index_keys,
    )
    return MethodRead(selected, scored, scored)


def pisa_vectors(scored_counts: torch.Tensor, block_size: int, branching: int) -> torch.Tensor:
    """Count the vectors PISA scored a query against - each leaf block scored its keys, each block above its children.

    Args:
        scored_counts: Blocks scored per query, KV head and level [R, kv_head_count, levels] int32, the leaf level first.
        block_size: Keys in a leaf block.
        branching: Children of a block above the leaves.

    Returns:
        Vectors scored a query, over its KV heads [R] int64.
    """
    per_level = torch.full((scored_counts.shape[-1],), branching, dtype=torch.int64)
    per_level[0] = block_size
    return (scored_counts.long() * per_level).sum((1, 2))


def read_pisa(
    method: "MethodConfig",
    keys: torch.Tensor,
    queries: torch.Tensor,
    positions: torch.Tensor,
    logit_scale: float,
    sink_size: int,
    window_size: int,
) -> MethodRead:
    """Read some rows with PISA's block selection, one read a key-value head, then the window and the sinks.

    Args:
        method: The method, whose block size, keep count, branching and beam width shape the search.
        keys: Keys [T, kv_head_count, head_dim] bf16 or fp32, on the host.
        queries: The rows' queries [R, query_head_count, head_dim] bf16 or fp32, on the host, the heads of a key-value
            head consecutive.
        positions: The rows' positions [R] int64, on the host.
        logit_scale: What the dots are multiplied by.
        sink_size: First keys always read.
        window_size: Recent keys always read.

    Returns:
        The read - [R, W] for one key-value head, [R, kv_head_count, W] for more - and its cost over every head, a
        query head dotting every vector its key-value head scored.
    """
    selection = pisa.select(
        keys,
        queries,
        positions,
        method.block_size,
        method.keep_count,
        method.branching,
        method.beam_width,
        logit_scale,
        method.forced_blocks,
    )
    blocks = selection.selected_key_ids
    always = window_and_sinks(positions, sink_size, window_size)[:, None].expand(-1, blocks.shape[1], -1)
    selected = torch.cat([blocks, always], dim=2)
    vectors = pisa_vectors(selection.scored_counts, method.block_size, method.branching)
    dots = vectors * queries.shape[1] // keys.shape[1]
    return MethodRead(selected[:, 0] if keys.shape[1] == 1 else selected, vectors, dots)


def attention_inputs(
    method: "MethodConfig",
    document: Document,
    rows: torch.Tensor,
    key_and_query: Callable[[], KeyAndQuery],
    scaling: float | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the keys, the rows' queries and the logit scale a PISA method reads in the model's own space, on the host.

    Args:
        method: The method, whose space is `attention`.
        document: The cached document - its own keys and queries where it was extracted under `queries.attention`.
        rows: The rows scored [R] int64.
        key_and_query: Makes GLM's key latents [T, key_dim] and the heads' queries [R, heads, key_dim].
        scaling: GLM's softmax scale, or None where the model has none to give.

    Returns:
        The keys [T, kv_head_count, head_dim], the queries [R, query_head_count, head_dim] and what the dots are
        multiplied by - GLM's latent as one key-value head.

    Raises:
        ValueError: The method has no logit scale to read at.
    """
    if document.attention_keys is None or document.attention_queries is None:
        scale = scaling if method.logit_scale is None else method.logit_scale
        if scale is None:
            raise ValueError(
                f"method {method.name!r} reads GLM's attention and has no logit scale - set logit_scale"
            )
        inputs = key_and_query()
        return inputs.key.cpu()[:, None], inputs.query.cpu(), scale
    scale = document.attention_scale if method.logit_scale is None else method.logit_scale
    if scale is None:
        raise ValueError(
            f"method {method.name!r} reads the model's attention and the cache records no softmax scale"
        )
    queries = document.attention_queries[rows.to(document.attention_queries.device)]
    return document.attention_keys.cpu(), queries.cpu(), scale


@dataclass(frozen=True)
class MethodSet:
    """The methods a record reads, and the served designs whose sizes and spaces they read at."""

    methods: list["MethodConfig"]
    # Every served design by name, on a host without a GPU too, since a method reads a design's sizes and not its kernels
    designs: dict[str, DesignSettings]
    # The map each design reads, whose sidecar records the temperature a method reads at
    maps: dict[str, str]


def design_of(method: "MethodConfig") -> str:
    """Return the design a method reads at, which the config checked it names."""
    if method.design is None:
        raise ValueError(f"method {method.name!r} reads a design and names none")
    return method.design


def one_list(method: "MethodConfig", index_queries: torch.Tensor) -> torch.Tensor:
    """Return a design's index queries as one list a row, the form a method reads.

    Args:
        method: The method, for the message.
        index_queries: The rows' index queries [R, index_dim] or [R, ranks, index_dim] fp32.

    Returns:
        The index queries [R, index_dim] fp32.

    Raises:
        ValueError: The design reads one list a tensor-parallel rank.
    """
    if index_queries.ndim == 2:
        return index_queries
    if index_queries.shape[1] != 1:
        raise ValueError(
            f"method {method.name!r} reads one list a row and design {method.design!r} reads one a rank"
        )
    return index_queries[:, 0]


class MethodVariant(NamedTuple):
    """One method at one temperature - the method, its name, and the temperature it reads at."""

    name: str
    method: "MethodConfig"
    # A temperature the config fixed, or None to read at the map's, times `scale`
    tau: float | None
    scale: float


def uses_tau(method: "MethodConfig") -> bool:
    """Return whether a method reads at a temperature - a LogSumExp score, or PISA in the index space."""
    if method.kind == Served.METHOD_PISA:
        return method.space == Served.SPACE_INDEX and method.logit_scale is None
    return "lse" in (method.node_bucket_score, method.bucket_score)


def expand_methods(methods: Sequence["MethodConfig"]) -> list[MethodVariant]:
    """Expand each method into the temperatures it is read at - `tau_scale` times the map's, `taus` as they are.

    A swept method is named `<name>@tau<scale>` for a scale and `<name>@tau=<tau>` for a temperature.

    Args:
        methods: The methods of the config.

    Returns:
        One variant a temperature, and one of the method as it is when it sweeps none.

    Raises:
        ValueError: A method sweeps a temperature it does not read at.
    """
    variants = []
    for method in methods:
        if (method.tau_scale or method.taus) and not uses_tau(method):
            raise ValueError(f"method {method.name!r} sweeps a temperature and reads at none")
        variants += [
            MethodVariant(f"{method.name}@tau{scale:g}", method, method.tau, scale)
            for scale in method.tau_scale
        ]
        variants += [MethodVariant(f"{method.name}@tau={tau:g}", method, tau, 1.0) for tau in method.taus]
        if not (method.tau_scale or method.taus):
            variants.append(MethodVariant(method.name, method, method.tau, 1.0))
    return variants


def read_methods(
    method_set: MethodSet,
    indexed: dict[str, tuple[torch.Tensor, torch.Tensor]],
    document: Document,
    rows: torch.Tensor,
    key_and_query: Callable[[], KeyAndQuery],
    scaling: float | None,
) -> dict[str, MethodRead]:
    """Read every method over the same rows of one document, each on the host, a method at every temperature it sweeps.

    Args:
        method_set: The methods and the designs they read at.
        indexed: Each design's index queries [R, index_dim] or [R, 1, index_dim] fp32 and index keys [T, index_dim]
            bf16, on the document's device.
        document: The cached document, on the document's device.
        rows: The rows scored [R] int64.
        key_and_query: Makes the key latents [T, key_dim] and the heads' queries [R, heads, key_dim] a PISA method in
            the attention space reads.
        scaling: GLM's softmax scale, which a PISA method in the attention space multiplies its dots by unless it
            names a logit scale, or None where the model has none to give.

    Returns:
        Each method's read and cost, by name - `expand_methods`'s.

    Raises:
        ValueError: A method reads a design's index space as one list and the design reads one a rank, a method in the
            attention space has no logit scale to read at, or a method reads at a temperature its map's sidecar lacks.
    """
    positions = document.query_positions[rows].cpu()
    structures: dict[str, Structures] = {}
    reads = {}
    for variant in expand_methods(method_set.methods):
        method = variant.method
        if method.kind == Served.METHOD_PISA and method.space == Served.SPACE_ATTENTION:
            keys, queries, scale = attention_inputs(method, document, rows, key_and_query, scaling)
            reads[variant.name] = read_pisa(
                method, keys, queries, positions, scale, document.sink_size, document.window_size
            )
            continue
        design = design_of(method)
        index_queries, index_keys = indexed[design]
        index_queries = one_list(method, index_queries)
        tau = 1.0
        if uses_tau(method):
            tau = (
                variant.tau
                or variant.scale * map_tau(method_set.maps[design], [document.layer])[document.layer]
            )
        if method.kind == Served.METHOD_REFERENCE:
            if design not in structures:
                structures[design] = Structures(index_keys.cpu())
            reads[variant.name] = read_reference(
                method,
                method_set.designs[design].indexer,
                structures[design],
                index_queries.float().cpu(),
                positions,
                tau,
            )
        else:
            scale = 1.0 / tau if method.logit_scale is None else method.logit_scale
            reads[variant.name] = read_pisa(
                method,
                index_keys.cpu()[:, None],
                index_queries.cpu()[:, None],
                positions,
                scale,
                document.sink_size,
                document.window_size,
            )
    return reads
