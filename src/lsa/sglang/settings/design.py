"""A design's settings - the one place a design is defined.

A design is two yaml blocks, and `DesignSettings` holds both:

| block | class | what it says |
|---|---|---|
| `model.indexer` | `IndexerConfig` | what the design is - its knobs |
| `model.serving` | `ServingChoices` | how it is computed - node centroid precision, decode, the projection file |

A knob that varies by level takes either one scalar, meaning every level from 1 to `top_level`, or a list with
one value a level, first entry level 1.
Level 0 holds keys, so no per-level knob has a level 0 entry.
Both forms are accepted from yaml and read back through `IndexerConfig.at`.

Every refusal says what arrived and what was wanted.
This module runs on the job host as well as in the server, so it imports no sglang.
The block of the model config a server is armed with is the two blocks flattened into one json object - their keys never
collide, so `DesignSettings.body` writes it and `parse` reads it back, refusing any key neither block has.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

from lsa.budget import EarlyBudget
from lsa.constants import Attention

# The bucket cache's paging unit - where a prefill segment is cut and a tree grows
# => a constant of the cache, not a knob of any design
PAGE_SIZE: int = 8192

# The most the engine's fused top-k selects, which bounds the buckets a query picks and a beam keeps
TOPK_LIMIT: int = 2048

# What a node centroid is stored as - the bucket centroids are bf16 only
NODE_CENTROID_PRECISIONS: tuple[str, ...] = ("bf16", "fp8")

# How a decode step reads - descend for fixed and tree only
DECODES: tuple[str, ...] = ("flat", "descend")

# The language of a kernel that exists in both - cutedsl where a CuTeDSL kernel takes the inputs, triton for the rest
KERNELS: tuple[str, ...] = ("triton", "cutedsl")

# What a shape's directions are stored as - one sign bit a dimension; the kernels read no bf16 basis
SHAPE_BASES: tuple[str, ...] = ("sign",)

# Longest contexts a decode CUDA graph is captured for, ascending - a batch replays the graph of the first limit its
# longest request fits under, and the full-context graph beyond the last
# => the index kernels of a graph are sized to its limit, not to the server's context length
# Each is 18 pages times a power of two, so a request at a power of two plus its answer sits at 89% of its range
# and none sits under 50% - a row far under its width lists badly
DECODE_CONTEXT_LIMITS: tuple[int, ...] = tuple(18 * PAGE_SIZE * 2**i for i in range(7))

# The most directions a shape keeps
MAX_SHAPE_RANK: int = 8

# What the indexer projects: attention's key K^(attn) and its heads' queries Q^(attn), the design's input, or the
# attention input X, a research option - on RULER cwe 512k the key and query matched GLM's own indexer and X lost
# about 0.15
KEY_AND_QUERY: str = "key_and_query"
ATTENTION_INPUT: str = "attention_input"
INDEXER_INPUTS: tuple[str, ...] = (KEY_AND_QUERY, ATTENTION_INPUT)

# Which heads' queries an index query sums under the key and query: every head of the layer, so every rank selects
# the one list - the design's, the freeze's `query: all` - or the heads a tensor-parallel rank holds, each rank
# selecting for its own
QUERY_HEADS_RANK: str = "rank"
QUERY_HEADS_ALL: str = "all"
QUERY_HEADS: tuple[str, ...] = (QUERY_HEADS_ALL, QUERY_HEADS_RANK)

# The key of the settings block that names a selector other than LSA - a block without it is LSA's
SELECTOR: str = "selector"

# A knob that varies by level: one value for every level, or one a level from level 1
PerLevel = int | tuple[int, ...] | list[int] | None


def refuse_unknown(owner: str, body: Mapping[str, Any], names: set[str]) -> None:
    """Refuse a key a block does not have.

    Args:
        owner: The block the keys were read from, for the message.
        body: The block's contents.
        names: The keys the block has.

    Raises:
        ValueError: A key is not one of `names`.
    """
    unknown = sorted(set(body) - names)
    if unknown:
        raise ValueError(f"{owner} has no knobs {unknown} - it has {sorted(names)}")


def require_whole(name: str, value: object, least: int) -> None:
    """Refuse a value that is not an integer of at least `least`.

    Args:
        name: The knob, for the message.
        value: What arrived.
        least: The smallest value the knob takes.

    Raises:
        ValueError: `value` is not an int, or is below `least`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer - got {value!r}")
    if value < least:
        raise ValueError(f"{name} must be at least {least} - got {value}")


@dataclass(frozen=True)
class Shape:
    """A bucket's spread about its centroid - a sketch of rank $r$ that the score adds as $z$ deviations.

    A shaped bucket is scored by its estimated best key, `q . c + z sqrt(q^T Sigma q)`, with `Sigma` its keys'
    covariance about the centroid kept as `rank` directions and their variances, the rest of the trace spread
    evenly over the directions left out.
    """

    # Directions kept, $r$ - 0 keeps the trace alone, an even spread
    rank: int = 2
    # Standard deviations the score adds, $z$
    deviations: float = 2.0

    def __post_init__(self) -> None:
        """Reject a rank the kernels hold no lanes for, or deviations that are not a positive number.

        Raises:
            ValueError: `rank` is not an integer from 0 to `MAX_SHAPE_RANK`.
            ValueError: `deviations` is not a positive number.
        """
        require_whole("rank", self.rank, 0)
        if self.rank > MAX_SHAPE_RANK:
            raise ValueError(f"a shape keeps at most {MAX_SHAPE_RANK} directions - got rank {self.rank}")
        if isinstance(self.deviations, bool) or not isinstance(self.deviations, (int, float)):
            raise ValueError(f"deviations must be a number - got {self.deviations!r}")
        if not self.deviations > 0:
            raise ValueError(f"deviations must be positive - got {self.deviations}")
        object.__setattr__(self, "deviations", float(self.deviations))

    @classmethod
    def read(cls, owner: str, value: "Shape | Mapping[str, Any] | None") -> "Shape | None":
        """Build a shape knob from what a block holds - a mapping of `rank` and `deviations`, or none.

        Args:
            owner: The knob, for the message.
            value: What arrived.

        Returns:
            The shape, or None for none.

        Raises:
            ValueError: The value is neither a shape, a mapping nor None, or names a key a shape does not have.
        """
        if value is None or isinstance(value, Shape):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"{owner} is a mapping of rank and deviations, or null - got {value!r}")
        refuse_unknown(owner, value, {item.name for item in fields(cls)})
        return cls(**value)


@dataclass(frozen=True)
class IndexerConfig:
    """What a design is - the keys of `model.indexer`, each in the design's name.

    Frozen, because the bucket cache is built against these numbers and a change mid-sequence would leave
    buckets describing a partition that no longer exists.
    The cluster block size is `window_size`, so it has no field.
    """

    # Recent keys always read, and the keys a cluster block holds
    window_size: int = 512
    # First keys always read
    sink_size: int = 4
    # Long-range keys read besides the sink and the window
    top_k: int = 4096
    # Buckets a query picks for expansion
    top_p: int = 1024
    # Dims of index keys and index queries
    index_dim: int = 64
    # Consecutive keys clustered as one - None for no pooling
    pool_size: int | None = 4
    # Average keys in a bucket
    bucket_size: int = 8
    # Most keys in a bucket
    bucket_cap: int = 16
    # 0 flat, 1 fixed, grows with the context in tree
    top_level: int = 0
    # Level - 1 buckets a node bucket holds - set iff top_level >= 1
    node_bucket_size: PerLevel = None
    # Level - 1 buckets node buckets are formed over - set iff top_level >= 1
    node_cluster_block_size: PerLevel = None
    # Node buckets a query keeps at a level - set iff top_level >= 1
    beam_width: PerLevel = None
    # Where the beam begins, the lowest level of at most this many node buckets - None starts at the top level
    start_nodes: int | None = None
    # A bucket's spread, scored by its estimated best key - None scores the centroid
    bucket_shape: Shape | None = None
    # A node bucket's, formed over the keys beneath it, one at every level - None scores the centroid
    node_bucket_shape: Shape | None = None
    # What index keys and index queries are projected from - the key K^(attn) and the heads' queries Q^(attn)
    # summed, the design's input, or the attention input X, a research option
    input: str = KEY_AND_QUERY
    # Under the key and query, the heads an index query sums - every head of the layer, or this rank's. None takes
    # `all` under the key and query and stays None under the attention input, which has no heads
    query_heads: str | None = None
    # The budget of queries before a position, smaller than `top_k` and `top_p` - None reads them all at those. Flat only
    early_budget: EarlyBudget | None = None

    PER_LEVEL: ClassVar[tuple[str, ...]] = ("node_bucket_size", "node_cluster_block_size", "beam_width")

    def __post_init__(self) -> None:
        """Reject a design the clustering and the kernels cannot run.

        Lists from yaml become tuples, so the config stays hashable.

        Raises:
            ValueError: A size is not a whole number of the least it takes.
            ValueError: `bucket_cap` is under twice `bucket_size`, which lets Ward's merge stop short.
            ValueError: `bucket_size` does not divide `window_size`, so a cluster block has no whole bucket count.
            ValueError: `pool_size` does not divide `bucket_size`, `bucket_cap` or `window_size`.
            ValueError: `top_p` is past the engine's top-k limit, or `top_p` buckets of `bucket_cap` keys cannot fill `top_k`.
            ValueError: The node knobs are not all set exactly when `top_level` is 1 or more.
            ValueError: A per-level list is not `top_level` long, or holds a value under 1.
            ValueError: `start_nodes` is set with a `top_level` of 1 or less.
            ValueError: A shape is refused, keeps more directions than `index_dim`, or `node_bucket_shape` is set
                with `top_level` 0.
            ValueError: `input` is not one of `INDEXER_INPUTS`.
            ValueError: `query_heads` is set with an input that has no heads, or is not one of `QUERY_HEADS`.
            ValueError: `early_budget` is refused, is set with a `top_level` of 1 or more, or is larger than the design's
                own `top_k` or `top_p`, or its `top_p` buckets of `bucket_cap` keys cannot fill its `top_k`.
        """
        if self.input not in INDEXER_INPUTS:
            raise ValueError(f"input must be one of {INDEXER_INPUTS} - got {self.input!r}")
        if self.input != KEY_AND_QUERY:
            if self.query_heads is not None:
                raise ValueError(
                    f"query_heads {self.query_heads!r} sums the heads' queries, which only input {KEY_AND_QUERY!r} "
                    f"reads - input {self.input!r} has no heads"
                )
        elif self.query_heads is None:
            object.__setattr__(self, "query_heads", QUERY_HEADS_ALL)
        elif self.query_heads not in QUERY_HEADS:
            raise ValueError(f"query_heads must be one of {QUERY_HEADS} - got {self.query_heads!r}")
        for name, least in (
            ("window_size", 1),
            ("sink_size", 0),
            ("top_k", 1),
            ("top_p", 1),
            ("index_dim", 1),
            ("bucket_size", 1),
            ("bucket_cap", 1),
            ("top_level", 0),
        ):
            require_whole(name, getattr(self, name), least)
        if self.bucket_cap < 2 * self.bucket_size:
            raise ValueError(
                f"bucket_cap must be at least twice bucket_size, or Ward's merge can stop short of its "
                f"bucket count - got bucket_cap {self.bucket_cap} against bucket_size {self.bucket_size}"
            )
        self.check_pool()
        if self.window_size % self.bucket_size:
            raise ValueError(
                f"bucket_size must divide window_size, the cluster block - "
                f"{self.bucket_size} does not divide {self.window_size}"
            )
        if self.top_p > TOPK_LIMIT:
            raise ValueError(f"top_p is at most the engine's top-k limit {TOPK_LIMIT} - got {self.top_p}")
        if self.top_p * self.bucket_cap < self.top_k:
            raise ValueError(
                f"top_p buckets of at most bucket_cap keys must be able to fill top_k - "
                f"top_p {self.top_p} times bucket_cap {self.bucket_cap} is {self.top_p * self.bucket_cap}, "
                f"under top_k {self.top_k}"
            )
        self.check_levels()
        if self.start_nodes is not None:
            if self.top_level <= 1:
                raise ValueError(
                    f"start_nodes needs a hierarchy of more than one level - "
                    f"got start_nodes {self.start_nodes} with top_level {self.top_level}"
                )
            require_whole("start_nodes", self.start_nodes, 1)
        self.check_shapes()
        self.check_early_budget()

    def check_early_budget(self) -> None:
        """Build the early budget from its mapping and refuse one the flat read cannot serve.

        Raises:
            ValueError: `early_budget` is refused, or is set with a `top_level` of 1 or more, which reads a beam.
            ValueError: Its `top_k` or `top_p` is over the design's own.
            ValueError: Its `top_p` buckets of `bucket_cap` keys cannot fill its `top_k`.
        """
        early = EarlyBudget.read(self.early_budget)
        object.__setattr__(self, "early_budget", early)
        if early is None:
            return
        if self.top_level >= 1:
            raise ValueError(
                f"early_budget is for the flat design, which a hierarchy does not read - "
                f"got early_budget with top_level {self.top_level}"
            )
        for name in ("top_k", "top_p"):
            if getattr(early, name) > getattr(self, name):
                raise ValueError(
                    f"early_budget {name} is at most the design's own {name} - "
                    f"got {getattr(early, name)} against {getattr(self, name)}"
                )
        if early.top_p * self.bucket_cap < early.top_k:
            raise ValueError(
                f"early_budget top_p buckets of at most bucket_cap keys must be able to fill its top_k - "
                f"top_p {early.top_p} times bucket_cap {self.bucket_cap} is {early.top_p * self.bucket_cap}, "
                f"under top_k {early.top_k}"
            )

    def check_shapes(self) -> None:
        """Build the shape knobs from their mappings and refuse a shape the design cannot hold.

        Raises:
            ValueError: A shape is refused, a shape keeps more directions than `index_dim` or `index_dim` is not a
                multiple of 32, which the packed sign basis reads a word at a time.
            ValueError: `bucket_shape` keeps more directions than `bucket_cap`, the most keys a bucket holds.
            ValueError: `node_bucket_shape` is set with `top_level` 0, where there are no node buckets.
        """
        for name in ("bucket_shape", "node_bucket_shape"):
            shape = Shape.read(name, getattr(self, name))
            object.__setattr__(self, name, shape)
            if shape is None:
                continue
            if shape.rank > self.index_dim:
                raise ValueError(
                    f"{name} keeps at most index_dim {self.index_dim} directions - got rank {shape.rank}"
                )
            if self.index_dim % 32:
                raise ValueError(
                    f"{name} stores a sign bit a dimension, read 32 at a time - index_dim {self.index_dim} is "
                    f"not a multiple of 32"
                )
        if self.bucket_shape is not None and self.bucket_shape.rank > self.bucket_cap:
            raise ValueError(
                f"bucket_shape keeps at most bucket_cap {self.bucket_cap} directions, the keys a bucket holds - "
                f"got rank {self.bucket_shape.rank}"
            )
        if self.node_bucket_shape is not None and self.top_level == 0:
            raise ValueError(
                "node_bucket_shape shapes node buckets and top_level is 0 - set top_level or drop it"
            )

    def check_pool(self) -> None:
        """Refuse a pool size the buckets and the cluster block cannot be built from.

        Raises:
            ValueError: `pool_size` is under 1, or does not divide `bucket_size`, `bucket_cap` or `window_size`.
        """
        if self.pool_size is None:
            return
        require_whole("pool_size", self.pool_size, 1)
        for name in ("window_size", "bucket_size", "bucket_cap"):
            if getattr(self, name) % self.pool_size:
                raise ValueError(
                    f"pool_size must divide {name}, since a pool is never split - "
                    f"{self.pool_size} does not divide {getattr(self, name)}"
                )

    def check_levels(self) -> None:
        """Refuse node knobs that disagree with `top_level`, and normalize per-level lists to tuples.

        Raises:
            ValueError: The node knobs are set without a hierarchy, or a hierarchy lacks one.
            ValueError: A per-level list is not `top_level` long, or holds a value under 1.
        """
        values = {name: getattr(self, name) for name in self.PER_LEVEL}
        if self.top_level == 0:
            given = sorted(name for name, value in values.items() if value is not None)
            if given:
                raise ValueError(f"{given} are node knobs and top_level is 0 - set top_level or drop them")
            return
        missing = sorted(name for name, value in values.items() if value is None)
        if missing:
            raise ValueError(f"top_level {self.top_level} needs the node knobs {missing}, which are unset")
        for name, value in values.items():
            widest = max(value) if isinstance(value, (list, tuple)) and value else value
            if name == "beam_width" and isinstance(widest, int) and widest > TOPK_LIMIT:
                raise ValueError(
                    f"beam_width is at most the engine's top-k limit {TOPK_LIMIT} - got {widest}"
                )
            if isinstance(value, (list, tuple)):
                if len(value) != self.top_level:
                    raise ValueError(
                        f"{name} lists one value a level, from level 1 - "
                        f"got {len(value)} for top_level {self.top_level}"
                    )
                for entry in value:
                    require_whole(name, entry, 1)
                object.__setattr__(self, name, tuple(value))
            else:
                require_whole(name, value, 1)

    def at(self, name: str, level: int) -> int:
        """Return a per-level knob's value at a level.

        Args:
            name: One of `node_bucket_size`, `node_cluster_block_size` and `beam_width`.
            level: A level from 1 to `top_level`.

        Returns:
            The knob's value at that level.

        Raises:
            ValueError: `name` is not a per-level knob, or `level` is outside 1 to `top_level`.
        """
        if name not in self.PER_LEVEL:
            raise ValueError(f"{name} does not vary by level - the per-level knobs are {self.PER_LEVEL}")
        if not 1 <= level <= self.top_level:
            raise ValueError(f"{name} has levels 1 to {self.top_level} - asked for level {level}")
        value = getattr(self, name)
        return int(value[level - 1] if isinstance(value, tuple) else value)

    @property
    def cluster_block_size(self) -> int:
        """Keys a cluster block holds, which is the window size."""
        return self.window_size

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "IndexerConfig":
        """Build from the `model.indexer` block.

        Args:
            body: The block's contents.

        Returns:
            The config, checked.

        Raises:
            ValueError: A key is not a knob, or a value is refused.
        """
        refuse_unknown("model.indexer", body, {item.name for item in fields(cls)})
        return cls(**body)


@dataclass(frozen=True)
class QSAConfig:
    """Qwen's own sparse attention - the baseline LSA is measured against on Qwen, in the names its server reads.

    A lightweight MQA indexer on the attention input `x`: `query_head_count` query heads and one key head of
    `head_dim`, the keys mean-pooled into blocks of `block_size` before the rotation, which turns the first
    `rotary_dim` dims of each at the model's own frequencies - a query at its position, a block at its first.
    A block's score is `sum_h relu(q_h . k_b) / sqrt(head_dim)` over the blocks closed by the query, and the read is
    the `top_k / block_size` best blocks expanded to keys plus the keys of the query's own open block.
    sglang calls these `indexer_n_heads`, `indexer_head_dim`, `indexer_compress_ratio` and `indexer_budget`.

    `input` is what the two projections read: the attention input `x`, as Qwen3.8-Next's indexer does, or the attention's
    own key and query, as an LSA map on Qwen does - the key heads' position-free dims side by side, and each group's
    query heads summed onto them. The second asks whether QSA's method or its input is what an LSA comparison measures.
    """

    query_head_count: int = 4
    head_dim: int = 128
    # The dims each head rotates, the model's own rotary dim
    rotary_dim: int = 64
    # Consecutive keys pooled into one block
    block_size: int = 4
    # Keys the selected blocks hold, besides the open block's
    top_k: int = 2048
    # What the projections read - `x` or the attention's key and query
    input: str = ATTENTION_INPUT

    def __post_init__(self) -> None:
        """Reject an indexer the selection cannot run.

        Raises:
            ValueError: A size is not positive, the rotation is odd or wider than a head, a block is a single key,
                `block_size` does not divide `top_k`, or `input` is not one of `INDEXER_INPUTS`.
        """
        if self.input not in INDEXER_INPUTS:
            raise ValueError(f"input must be one of {INDEXER_INPUTS} - got {self.input!r}")
        for name in ("query_head_count", "head_dim", "rotary_dim", "top_k"):
            require_whole(name, getattr(self, name), 1)
        require_whole("block_size", self.block_size, 2)
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim must be even and at most head_dim {self.head_dim} - got {self.rotary_dim}"
            )
        if self.top_k % self.block_size:
            raise ValueError(f"block_size {self.block_size} must divide top_k {self.top_k}")

    @property
    def top_blocks(self) -> int:
        """Blocks a query selects"""
        return self.top_k // self.block_size


@dataclass(frozen=True)
class PisaConfig:
    """The original PISA design, served - leaf blocks of `block_size` keys under a mean-pooled pyramid searched coarse to fine.

    The names are those of `lsa.baselines.pisa.select`, whose paper symbols they are: `block_size` is C,
    `keep_count` is K and `branching` is g. A query keeps `keep_count` leaf blocks, `beam_width` blocks at every
    level above them, and reads their keys, so a list is `keep_count * block_size` wide.
    The first, previous and current leaf blocks and the blocks above them are kept unscored and count inside the keep
    count, which is what `forced_blocks` says.
    The defaults are the paper's, with a keep count that makes 4,608 keys, matched to LSA's 4,612-key list.
    """

    block_size: int = 64
    keep_count: int = 72
    branching: int = 2
    # Blocks kept at each level above the leaves - None keeps `keep_count`
    beam_width: int | None = None
    forced_blocks: bool = True

    def __post_init__(self) -> None:
        """Reject a pyramid the search is not defined for.

        Raises:
            ValueError: `block_size` is not a positive integer, `branching` is under 2, or `keep_count` or
                `beam_width` is under what the forced blocks need - 3 with them, 1 without.
            ValueError: `forced_blocks` is not true or false.
        """
        if not isinstance(self.forced_blocks, bool):
            raise ValueError(f"forced_blocks must be true or false - got {self.forced_blocks!r}")
        require_whole("block_size", self.block_size, 1)
        require_whole("branching", self.branching, 2)
        smallest = 3 if self.forced_blocks else 1
        require_whole("keep_count", self.keep_count, smallest)
        if self.beam_width is not None:
            require_whole("beam_width", self.beam_width, smallest)

    @property
    def beam(self) -> int:
        """Blocks kept at each level above the leaves."""
        return self.keep_count if self.beam_width is None else self.beam_width

    @property
    def width(self) -> int:
        """List width the attention kernels read, `keep_count * block_size`."""
        return self.keep_count * self.block_size

    def body(self) -> dict[str, Any]:
        """Return the settings block's json body - every knob, under the key that names the selector.

        Returns:
            The selector, then every field.
        """
        return {SELECTOR: Attention.PISA, **asdict(self)}

    @classmethod
    def parse(cls, body: Mapping[str, Any]) -> "PisaConfig":
        """Build from a settings block's body.

        Args:
            body: The block's contents, as `body` writes it.

        Returns:
            The config, checked.

        Raises:
            ValueError: The file names another selector, a key is not a knob, or a value is refused.
        """
        if body.get(SELECTOR) != Attention.PISA:
            raise ValueError(
                f"the settings name selector {body.get(SELECTOR, Attention.LSA)!r}, not {Attention.PISA!r}"
            )
        knobs = {name: value for name, value in body.items() if name != SELECTOR}
        refuse_unknown("the pisa settings", knobs, {item.name for item in fields(cls)})
        return cls(**knobs)

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "PisaConfig":
        """Build from the `model.pisa` block.

        Args:
            body: The block's contents.

        Returns:
            The config, checked.

        Raises:
            ValueError: A key is not a knob, or a value is refused.
        """
        refuse_unknown("model.pisa", body, {item.name for item in fields(cls)})
        return cls(**body)


@dataclass(frozen=True)
class ServingChoices:
    """How a design is computed rather than what it is - the keys of `model.serving`."""

    # Bucket centroids are bf16 always - this is the node centroids'
    node_centroid_precision: str = "bf16"
    # Flat reads every bucket by one scan, tree descends the hierarchy at decode too
    decode: str = "flat"
    # The file with "<layer>.W_Q_idx" and "<layer>.W_K_idx"
    projections: str = ""
    # What a shape's directions are stored as
    shape_basis: str = "sign"
    # The kernels a read runs where both languages have one - the same buckets either way
    kernels: str = "triton"
    # Contexts a flat decode captures a CUDA graph for besides the server's own, ascending - empty captures one
    decode_context_limits: tuple[int, ...] = DECODE_CONTEXT_LIMITS
    # Take a cached prefix's index where the server caches prefixes, rather than build it again - opt-in
    prefix_reuse: bool = False
    # Full contexts the bucket cache holds beyond the KV pool's pages - for requests that share a cached prefix in the pool
    # but not in the index, which the pool's size does not bound
    spare_contexts: int = 0

    def __post_init__(self) -> None:
        """Reject a choice the backend has no kernel for.

        Raises:
            ValueError: `node_centroid_precision`, `decode`, `shape_basis` or `kernels` is not one of its options,
                `projections` is not text, or `decode_context_limits` is not ascending integers of at least a page.
        """
        limits = tuple(self.decode_context_limits)
        object.__setattr__(self, "decode_context_limits", limits)
        for limit in limits:
            require_whole("decode_context_limits", limit, PAGE_SIZE)
        if list(limits) != sorted(set(limits)):
            raise ValueError(f"decode_context_limits must be ascending and distinct - got {list(limits)}")
        if self.shape_basis not in SHAPE_BASES:
            raise ValueError(
                f"shape_basis must be one of {SHAPE_BASES} - got {self.shape_basis!r}; the shape kernels read a "
                f"packed sign basis only"
            )
        if self.node_centroid_precision not in NODE_CENTROID_PRECISIONS:
            raise ValueError(
                f"node_centroid_precision must be one of {NODE_CENTROID_PRECISIONS} - "
                f"got {self.node_centroid_precision!r}"
            )
        if self.decode not in DECODES:
            raise ValueError(f"decode must be one of {DECODES} - got {self.decode!r}")
        if self.kernels not in KERNELS:
            raise ValueError(f"kernels must be one of {KERNELS} - got {self.kernels!r}")
        if not isinstance(self.projections, str):
            raise ValueError(f"projections must be a path - got {self.projections!r}")
        if not isinstance(self.prefix_reuse, bool):
            raise ValueError(f"prefix_reuse must be true or false - got {self.prefix_reuse!r}")
        require_whole("spare_contexts", self.spare_contexts, 0)

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "ServingChoices":
        """Build from the `model.serving` block.

        Args:
            body: The block's contents.

        Returns:
            The choices, checked.

        Raises:
            ValueError: A key is not a choice, or a value is refused.
        """
        refuse_unknown("model.serving", body, {item.name for item in fields(cls)})
        return cls(**body)


@dataclass(frozen=True)
class DesignSettings:
    """A design - what it is and how it is computed - built only from a combination the kernels can serve."""

    indexer: IndexerConfig
    serving: ServingChoices

    def __post_init__(self) -> None:
        """Reject a serving choice the hierarchy cannot honour.

        Raises:
            ValueError: `decode` is `tree` with `top_level` 0.
            ValueError: `node_centroid_precision` is `fp8` with `top_level` 0.
        """
        if self.serving.decode == "descend" and self.indexer.top_level < 1:
            raise ValueError(
                f"decode descend needs a hierarchy - got decode descend with top_level {self.indexer.top_level}"
            )
        if self.serving.node_centroid_precision == "fp8" and self.indexer.top_level < 1:
            raise ValueError(
                f"fp8 node centroids need node buckets - "
                f"got node_centroid_precision fp8 with top_level {self.indexer.top_level}"
            )

    @property
    def width(self) -> int:
        """List width the attention kernels read, `top_k + window_size + sink_size`."""
        return self.indexer.top_k + self.indexer.window_size + self.indexer.sink_size

    @property
    def early_width(self) -> int | None:
        """Columns an early query's list needs, `width` at the early budget's `top_k` rounded up to 64, or None for none.

        64 is the block the TileLang read walks a list in, so a list cut to this width is a whole number of blocks.
        """
        early = self.indexer.early_budget
        if early is None:
            return None
        needed = early.top_k + self.indexer.window_size + self.indexer.sink_size
        return -(-needed // 64) * 64

    def body(self) -> dict[str, Any]:
        """Return the settings block's json body - both blocks flattened under their config keys.

        Returns:
            Every `IndexerConfig` field, then every `ServingChoices` field.
        """
        return {**asdict(self.indexer), **asdict(self.serving)}

    @classmethod
    def parse(cls, body: Mapping[str, Any]) -> "DesignSettings":
        """Build from a settings block's flat body.

        Args:
            body: The block's contents, as `body` writes it.

        Returns:
            The settings, checked.

        Raises:
            ValueError: The body names a selector, a key is neither a knob nor a serving choice, or a value is refused.
        """
        if SELECTOR in body:
            raise ValueError(f"the settings name selector {body[SELECTOR]!r}, and LSA's block names none")
        indexer_names = {item.name for item in fields(IndexerConfig)}
        serving_names = {item.name for item in fields(ServingChoices)}
        unknown = sorted(set(body) - indexer_names - serving_names)
        if unknown:
            raise ValueError(
                f"the settings name {unknown}, which are neither knobs of model.indexer nor choices of "
                f"model.serving - they are {sorted(indexer_names)} and {sorted(serving_names)}"
            )
        return cls.from_blocks(
            {name: value for name, value in body.items() if name in indexer_names},
            {name: value for name, value in body.items() if name in serving_names},
        )

    @classmethod
    def from_blocks(cls, indexer: Mapping[str, Any], serving: Mapping[str, Any]) -> "DesignSettings":
        """Build from the two yaml blocks.

        Args:
            indexer: The `model.indexer` block.
            serving: The `model.serving` block.

        Returns:
            The settings, checked.

        Raises:
            ValueError: A key is unknown or a value is refused.
        """
        return cls(IndexerConfig.from_mapping(indexer), ServingChoices.from_mapping(serving))
