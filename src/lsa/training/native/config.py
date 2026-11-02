"""What a native run is, as a dataclass tree one yaml file fills - the attention, the model, the corpus and the schedule.

The same loader every other stage uses, so a run is described the same way here as anywhere: defaults, then the
file, then a sweep, then command-line overrides.

Four blocks. `attention` is the thing under study, and its fields are the design's own knobs. `model` is the architecture that carries it, `data` is
the corpus and the sequence length it is cut into, and `run` is how - the name, where it writes, the schedule,
the runtime and the held-out pass. The run is data-parallel only: a 0.8B model fits on one GPU, and the
sequence is the unit the attention partitions, so nothing here shards one.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lsa import assets, config_loader
from lsa.training.constants import Corpus, CorpusSplit, Packing
from lsa.training.native.constants import (
    Backend,
    DecayShape,
    Evaluation,
    Flavor,
    Kind,
    Partition,
    Qwen35,
    Rank,
    Retrieval,
    Runtime,
    Schedule,
    WeightDecayScope,
)
from lsa.training.native.schedule import decay_start_step, decay_step_count


@dataclass
class AttentionConfig:
    """Which attention the full-attention layers run, and the design's knobs when it is multi-resolution.

    `dense` is plain causal attention and `window` a sliding window, the two arms a multi-resolution run is
    compared against. `multires` reads every closed key exactly once - itself, when its bucket is selected, or
    through its bucket's summary otherwise. `lsa` reads what the flat LSA design serves and nothing else: its sinks,
    window and open block, and the keys of the buckets one selection shared by every head picks, in the index space of
    the model's own position-free q,k. Its `coarse` is false, its partition Ward, and `variance` and `rank` are unused.

    The window is the whole of the recent read: a bucket with any key inside it is read fine in full, so every
    key is read once whatever the window. The block is the span Ward clusters and the span a query's own open
    block covers.
    """

    kind: str = Kind.MULTIRES
    # The dense arm's kernel, torchtitan's name for it. The other arms run their own flex read
    backend: str = Backend.FLEX
    # Unselected buckets enter the softmax through their summaries. False reads the selection alone
    coarse: bool = True
    # The within-bucket variance term of a summary's logit, and of the estimated mass a selection ranks by
    variance: bool = True
    partition: str = Partition.WARD
    rank: str = Rank.MASS
    window_size: int = 512
    sink_size: int = 4
    # Keys of selected buckets a query reads at fine resolution, at most
    top_k: int = 1024
    # Buckets a query selects, at most. None takes the design's rule, twice the buckets top_k keys fill
    top_p: int | None = None
    pool_size: int = 4
    # Average keys in a bucket
    bucket_size: int = 8
    bucket_cap: int = 16
    # Keys in a block, which Ward clusters whole and which closes as one
    block_size: int = 512
    # Seed of the random partition's shuffle
    seed: int = 0
    leaf_block_size: int = 64
    branching: int = 2
    level_block_count: int = 24
    local_block_count: int = 2
    compress_block_size: int = 32
    compress_stride: int = 16
    compress_hidden_dim: int | None = None
    select_block_size: int = 64
    select_block_count: int = 16
    index_input: str = "none"
    index_training: str = "none"
    index_dim: int = 64
    index_rotary_dim: int = 0
    index_norm: str = "l2"
    query_head_reduction: str = "sum"
    index_query_heads: str = "sum"
    index_detached: bool = False
    index_loss_weight: float = 0.1
    summaries: bool = False
    fast_read: bool = False
    ste_window: bool = True
    index_head_count: int = 8
    index_bias: bool = False
    index_bias_scale_init: float = 0.0
    index_relu: bool = False
    sparse_start_step: int = 0
    kl_on_selection: bool = False
    kl_query_count: int = 128

    def __post_init__(self) -> None:
        """Reject a choice or a size the attention cannot honour.

        Raises:
            ValueError: A name is not one of its options, a size is not positive, a sink or a top-k is negative,
                the sizes do not nest, or the cap is under twice the bucket size.
        """
        if self.kind == Kind.DSA and self.index_input == "none":
            self.index_input = "attention_input"
        index_inputs: tuple[str, ...] = ("none", "key_and_query", "attention_input")
        if self.kind == Kind.DSA:
            index_inputs += ("sliced_key_and_query",)
        for field_name, value, options in (
            ("kind", self.kind, Kind.ALL),
            ("backend", self.backend, Backend.ALL),
            ("partition", self.partition, Partition.ALL),
            ("rank", self.rank, Rank.ALL),
            ("index_input", self.index_input, index_inputs),
            ("index_training", self.index_training, ("none", "kl", "ste")),
            ("index_norm", self.index_norm, ("l2", "rms")),
            ("query_head_reduction", self.query_head_reduction, ("sum", "max")),
            ("index_query_heads", self.index_query_heads, ("sum", "concat")),
        ):
            if value not in options:
                raise ValueError(f"attention.{field_name} must be one of {options} - got {value!r}")
        config_loader.require_positive(
            self, "attention", "window_size", "pool_size", "bucket_size", "bucket_cap", "block_size"
        )
        if self.kind == Kind.PISA:
            config_loader.require_positive(self, "attention", "leaf_block_size")
            if (
                self.branching < 2
                or self.local_block_count < 2
                or self.level_block_count < self.local_block_count + 1
            ):
                raise ValueError(
                    f"attention PISA needs branching >= 2, local_block_count >= 2 and level_block_count >= "
                    f"local_block_count + 1 - got {self.branching}, {self.local_block_count}, {self.level_block_count}"
                )
        if self.kind == Kind.NSA:
            config_loader.require_positive(
                self, "attention", "compress_block_size", "compress_stride", "select_block_size"
            )
            if self.compress_hidden_dim is not None:
                config_loader.require_positive(self, "attention", "compress_hidden_dim")
            if (
                self.compress_block_size % self.compress_stride
                or self.select_block_size % self.compress_stride
            ):
                raise ValueError(
                    f"attention.compress_stride {self.compress_stride} must divide compress_block_size "
                    f"{self.compress_block_size} and select_block_size {self.select_block_size}"
                )
            if self.select_block_count < 3:
                raise ValueError(
                    f"attention.select_block_count must be at least 3 - got {self.select_block_count}"
                )
        if self.sink_size < 0 or self.top_k < 0:
            raise ValueError("attention.sink_size and attention.top_k must not be negative")
        if self.top_p is not None and self.top_p < 0:
            raise ValueError(f"attention.top_p must not be negative - got {self.top_p}")
        config_loader.require_positive(self, "attention", "index_dim", "kl_query_count")
        if self.index_loss_weight < 0:
            raise ValueError(
                f"attention.index_loss_weight must not be negative - got {self.index_loss_weight}"
            )
        if self.index_rotary_dim < 0 or self.index_rotary_dim > self.index_dim or self.index_rotary_dim % 2:
            raise ValueError(
                f"attention.index_rotary_dim must be even and in [0, index_dim] - got {self.index_rotary_dim}"
            )
        if self.kind == Kind.LSA:
            if (self.index_input == "none") != (self.index_training == "none"):
                raise ValueError(
                    f"attention.index_input none requires index_training none; learned inputs require kl or ste - got {self.index_input!r}, {self.index_training!r}"
                )
            if self.query_head_reduction == "max" and (
                self.index_input != "none" or self.index_query_heads != "sum"
            ):
                raise ValueError(
                    "attention.query_head_reduction max requires index_input none and index_query_heads sum"
                )
            if self.summaries and self.block_size != self.window_size:
                raise ValueError("attention.summaries requires block_size equal to window_size")
        if self.kind == Kind.DSA:
            config_loader.require_positive(
                self, "attention", "index_head_count", "index_dim", "kl_query_count"
            )
            if self.index_input not in ("attention_input", "key_and_query", "sliced_key_and_query"):
                raise ValueError(
                    f"attention.index_input must be attention_input, key_and_query or sliced_key_and_query - got {self.index_input!r}"
                )
            if self.index_bias and (self.index_relu or self.index_input != "attention_input"):
                raise ValueError("attention.index_bias requires attention_input without index_relu")
            if self.sparse_start_step < 0 or self.index_loss_weight < 0:
                raise ValueError(
                    f"attention.sparse_start_step and index_loss_weight must be nonnegative - got {self.sparse_start_step}, {self.index_loss_weight}"
                )
        # The knobs as one string, which a custom op takes where it cannot take the config
        self.encoded = json.dumps(asdict(self))
        if self.kind not in (Kind.MULTIRES, Kind.LSA):
            return
        if self.kind == Kind.LSA and (self.coarse or self.partition != Partition.WARD):
            raise ValueError(
                "attention.kind lsa reads no summaries and clusters with Ward - set coarse false"
            )
        if self.block_size % self.bucket_size:
            raise ValueError(
                f"attention.bucket_size {self.bucket_size} must divide block_size {self.block_size}"
            )
        if self.partition != Partition.WARD:
            return
        if self.bucket_size % self.pool_size:
            raise ValueError(
                f"attention.pool_size {self.pool_size} must divide bucket_size {self.bucket_size}"
            )
        if self.bucket_cap < 2 * self.bucket_size:
            raise ValueError(
                f"attention.bucket_cap must be at least twice bucket_size {self.bucket_size} - "
                f"Ward needs the room - got {self.bucket_cap}"
            )

    @classmethod
    def decode(cls, encoded: str) -> "AttentionConfig":
        """Return the config an `encoded` string was made from."""
        return cls(**json.loads(encoded))

    @property
    def buckets_per_block(self) -> int:
        """Return the buckets a closed block is cut into."""
        return self.block_size // self.bucket_size

    @property
    def selection_width(self) -> int:
        """Return the buckets a query selects at most, the design's rule when the run did not say.

        The design reads twice the buckets its top-k keys fill, 1,024 buckets of about 8 keys at 4,096 keys
        and 512 at 2,048, so a list of the best buckets is long enough that the key budget decides where it ends.
        """
        return 2 * self.top_k // self.bucket_size if self.top_p is None else self.top_p

    @property
    def key_budget(self) -> int:
        """Return the keys a query reads at fine resolution beyond the open block, sinks and window included."""
        if self.kind == Kind.NSA:
            return self.select_block_count * self.select_block_size + self.window_size
        if self.kind == Kind.PISA:
            return self.level_block_count * self.leaf_block_size
        return self.sink_size + self.window_size + self.top_k


@dataclass
class ModelConfig:
    """Which model to build, and whose tokenizer to encode with.

    The flavor is torchtitan's own name for a Qwen3.5 size, or one of the two models at the GPT-2 vocabulary, and the
    model starts from random initialisation - a native run asks what an attention learns, not what it does to a
    checkpoint. The head shape is the flavor's own until a run names one: eight query heads over one key-value head
    of 128 dims with no rotation is the multi-query shape the sparse arms were compared on. The tokenizer is a model
    [card](../../assets/__init__.py)'s snapshot, because every Qwen3.5 size shares one vocabulary.
    """

    flavor: str = Flavor.QWEN35_0_8B
    # The full-attention layers' head shape, which the flavor sets when None
    attention_bias: bool = False
    query_head_count: int | None = None
    kv_head_count: int | None = None
    head_dim: int | None = None
    # Leading head dims the model rotates, 0 for none. The lsa arm's index space is the dims it leaves
    rotary_dim: int | None = None
    # The model card whose snapshot holds the tokenizer. None takes the tokenizer the corpus was tokenized with, which
    # a model trained from random initialisation has no other claim to
    tokenizer: str | None = None
    # Drop the vision tower, which a text corpus never reaches
    text_only: bool = True

    def __post_init__(self) -> None:
        """Reject a head shape no attention could have.

        Raises:
            ValueError: A count or the head dim is not positive, the rotation is negative, or the query heads are
                not a whole number of groups over the key-value heads.
        """
        for field_name in ("query_head_count", "kv_head_count", "head_dim"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"model.{field_name} must be positive - got {value}")
        if self.rotary_dim is not None and self.rotary_dim < 0:
            raise ValueError(f"model.rotary_dim must not be negative - got {self.rotary_dim}")
        if self.query_head_count and self.kv_head_count and self.query_head_count % self.kv_head_count:
            raise ValueError(
                f"model.kv_head_count {self.kv_head_count} must divide query_head_count {self.query_head_count}"
            )


@dataclass
class MixtureChild:
    """One corpus of a mixture, and how often a sequence is drawn from it relative to the others."""

    name: str
    weight: float = 1.0


@dataclass
class DataConfig:
    """Which corpus a run reads, how it is cut into sequences, and the sequence length.

    The corpus is named by its card - a source of text, a tokenized dataset or token shards - rather than pathed,
    because a path says where the bytes are and nothing about which tokenizer wrote them. `path` is the escape
    hatch for a tokenized corpus with no card. A `mixture` replaces `name` as what the run trains on, each child
    drawn by weight, and `name` is then only the corpus the held-out pass reads. `packing` is how documents are laid
    into sequences, and is left to the corpus's own default when it is None. A text corpus is tokenized by the model's
    tokenizer as it is read.
    """

    name: str = assets.Default.DATASET
    path: str | None = None
    mixture: list[MixtureChild] = field(default_factory=list)
    packing: str | None = None
    # Readers a rank interleaves a batch at a time, as a data loader's workers do
    readers: int = 1
    sources: list[str] | None = None
    held_out_fraction: float = Corpus.HELD_OUT_FRACTION
    split: str = CorpusSplit.TRAIN
    seed: int = 0
    # Tokens per sequence, which is the context the model trains at and has to divide a tokenized corpus's own row
    seq_len: int = Qwen35.SEQ_LEN

    def __post_init__(self) -> None:
        """Reject a split, share, packing or mixture the loader cannot honour.

        Raises:
            ValueError: `split` is neither half, `held_out_fraction` is not in `[0, 1)`, `path` is not absolute,
                `packing` is not one of its options, or a mixture weight is not positive.
        """
        if self.split not in CorpusSplit.ALL:
            raise ValueError(f"split must be one of {CorpusSplit.ALL} - got {self.split!r}")
        if not 0.0 <= self.held_out_fraction < 1.0:
            raise ValueError(f"held_out_fraction must be in [0, 1) - got {self.held_out_fraction}")
        if self.path is not None and not self.path.startswith("/"):
            raise ValueError(f"path must be an absolute path - got {self.path!r}")
        if self.packing is not None and self.packing not in Packing.ALL:
            raise ValueError(f"packing must be one of {Packing.ALL} - got {self.packing!r}")
        if any(child.weight <= 0 for child in self.mixture):
            raise ValueError(
                f"mixture weights must be positive - got {[child.weight for child in self.mixture]}"
            )
        config_loader.require_positive(self, "data", "seq_len", "readers")

    @property
    def asset(self) -> assets.SourceCard | assets.TokenizedCard | assets.ShardCard:
        """Return the corpus `name` names, by card or off its own manifest."""
        return assets.corpus(self.name, self.path)

    @property
    def corpora(self) -> list[tuple[str, float]]:
        """Return the corpora the run trains on and their weights - the mixture, or `name` alone."""
        if self.mixture:
            return [(child.name, child.weight) for child in self.mixture]
        return [(self.name, 1.0)]

    @property
    def cards(self) -> list[tuple[assets.SourceCard | assets.TokenizedCard | assets.ShardCard, float]]:
        """Return the cards of the corpora the run trains on and their weights."""
        single = len(self.corpora) == 1
        return [(assets.corpus(name, self.path if single else None), weight) for name, weight in self.corpora]


@dataclass
class OptimConfig:
    """AdamW, a learning-rate schedule, and the tokens of a step.

    The warmup is linear from zero, the stable stage holds `lr`, and the last `decay_ratio` of the run falls to
    `min_lr_factor` of it by `decay_shape`. A branch off a decay start has its own schedule, so its warmup is the re-warm.
    The cosine schedule instead decays over every step after warmup.
    `tokens_per_rank / data.seq_len` is how many sequences a rank sees at once, so the two have to divide.
    """

    steps: int = 100
    lr: float = 1e-3
    warmup_steps: int = 20
    schedule: str = Schedule.WSD
    decay_ratio: float = 0.8
    decay_shape: str = DecayShape.COSINE
    min_lr_factor: float = 0.1
    weight_decay: float = 0.1
    weight_decay_scope: str = WeightDecayScope.ALL_PARAMETERS
    grad_clip: float = 1.0
    tokens_per_rank: int = Qwen35.SEQ_LEN
    # Tokens of all the ranks in one optimiser step, which gradient accumulation makes up when a pass of every rank
    # reads fewer. None is one pass
    tokens_per_step: int | None = None

    @property
    def decay_step_count(self) -> int:
        """Return the steps of the run that decay."""
        return decay_step_count(self.steps, self.decay_ratio)

    @property
    def decay_start_step(self) -> int:
        """Return the step whose checkpoint is the decay start, the last step of the stable stage."""
        return decay_start_step(self.steps, self.decay_ratio)

    def __post_init__(self) -> None:
        """Reject a schedule that cannot run.

        Raises:
            ValueError: `schedule` or `decay_shape` is not one of its options, `decay_ratio` is not in `[0, 1]`, or the warmup is
                longer than the run, or `min_lr_factor` is outside `[0, 1]`.
        """
        config_loader.require_positive(self, "run.optim", "steps", "tokens_per_rank")
        if self.weight_decay_scope not in WeightDecayScope.ALL:
            raise ValueError(
                f"run.optim.weight_decay_scope must be one of {WeightDecayScope.ALL} - got {self.weight_decay_scope!r}"
            )
        if self.schedule not in Schedule.ALL:
            raise ValueError(f"run.optim.schedule must be one of {Schedule.ALL} - got {self.schedule!r}")
        if self.decay_shape not in DecayShape.ALL:
            raise ValueError(
                f"run.optim.decay_shape must be one of {DecayShape.ALL} - got {self.decay_shape!r}"
            )
        if not 0.0 <= self.decay_ratio <= 1.0:
            raise ValueError(f"run.optim.decay_ratio must be in [0, 1] - got {self.decay_ratio}")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError(
                f"run.optim.warmup_steps must be in [0, steps] - got {self.warmup_steps} of {self.steps} steps"
            )
        if not 0.0 <= self.min_lr_factor <= 1.0:
            raise ValueError(f"run.optim.min_lr_factor must be in [0, 1] - got {self.min_lr_factor}")
        if self.tokens_per_step is not None:
            config_loader.require_positive(self, "run.optim", "tokens_per_step")


@dataclass
class RuntimeConfig:
    """What the step is recomputed as, and where it reports."""

    activation_checkpoint: str = Runtime.NONE
    autocast_bf16: bool = False
    compile: str = Runtime.OFF
    # What seeds model initialisation, so that two runs are the same random model rather than two of them
    seed: int = 0
    log_every: int = 1
    wandb: bool = False

    def __post_init__(self) -> None:
        """Reject a recomputation or compile mode that does not exist.

        Raises:
            ValueError: A mode is not one of its options.
        """
        if self.activation_checkpoint not in Runtime.CHECKPOINTING:
            raise ValueError(
                f"activation_checkpoint must be one of {Runtime.CHECKPOINTING} - "
                f"got {self.activation_checkpoint!r}"
            )
        if self.compile not in Runtime.COMPILE:
            raise ValueError(f"compile must be one of {Runtime.COMPILE} - got {self.compile!r}")


@dataclass
class EvaluationConfig:
    """A fixed held-out set a run scores itself on as it trains, one sequence a document.

    Every run with the same seed reads the same sequences, so two runs pair sequence by sequence.
    """

    # Steps between passes, or 0 for none. torchtitan also runs one after step 1
    every: int = Evaluation.EVERY
    sequences: int = Evaluation.SEQUENCES
    seed: int = 0
    # Position bucket starts after the first. One at or past the sequence's end is dropped
    edges: list[int] = field(default_factory=lambda: list(Evaluation.EDGES))
    # Queries per sequence the attention diagnostics are measured on
    probes: int = Evaluation.PROBES
    # Tokens between a retrieval probe's needle and its query. One a sequence has no room for is dropped, and none
    # at all switches the probe off
    retrieval_distances: list[int] = field(default_factory=lambda: list(Retrieval.DISTANCES))
    # Held-out sequences a distance is probed on, rounded up to whole batches over the ranks
    retrieval_sequences: int = Retrieval.SEQUENCES
    retrieval_distractors: int = Retrieval.DISTRACTORS
    # A torchtitan step folder (`.../checkpoint/step-N`) `lsa native evaluate` scores when the command line names
    # none, or None for a config that trains. A preset of a released checkpoint sets it, so the config alone
    # says which weights an evaluation is of
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Reject a set or a cadence this evaluation cannot honour.

        Raises:
            ValueError: `every` is negative, `sequences`, `probes` or `retrieval_sequences` is not positive,
                `retrieval_distractors` is negative, the edges or retrieval distances are not positive
                and increasing, or `checkpoint` is not an absolute path.
        """
        if self.checkpoint is not None and not self.checkpoint.startswith("/"):
            raise ValueError(f"run.evaluation.checkpoint must be an absolute path - got {self.checkpoint!r}")
        if self.every < 0:
            raise ValueError(f"every must not be negative - got {self.every}")
        config_loader.require_positive(self, "run.evaluation", "sequences", "probes", "retrieval_sequences")
        if not self.edges or self.edges[0] <= 0 or self.edges != sorted(set(self.edges)):
            raise ValueError(f"edges must be positive and increasing - got {self.edges}")
        distances = self.retrieval_distances
        if distances and (distances[0] <= 0 or distances != sorted(set(distances))):
            raise ValueError(f"retrieval_distances must be positive and increasing - got {distances}")
        if self.retrieval_distractors < 0:
            raise ValueError(f"retrieval_distractors must not be negative - got {self.retrieval_distractors}")

    @property
    def enabled(self) -> bool:
        """Say whether the run evaluates at all."""
        return self.every > 0

    def check_ranks(self, rank_multiple: int, sequences_per_batch: int) -> None:
        """Refuse a held-out set that does not split into whole batches over the ranks.

        Args:
            rank_multiple: Ranks the set splits over, or a divisor of them when the run does not yet know how
                many it gets.
            sequences_per_batch: Sequences a rank reads at once.

        Raises:
            ValueError: `sequences` is not a whole number of batches on every rank.
        """
        per_round = rank_multiple * sequences_per_batch
        if self.enabled and self.sequences % per_round:
            raise ValueError(
                f"run.evaluation.sequences {self.sequences} must be a multiple of {per_round} - "
                f"{sequences_per_batch} sequences a batch over {rank_multiple} ranks"
            )


@dataclass
class CheckpointConfig:
    """What a run saves besides the checkpoints of its interval."""

    # Save at the last step of the stable stage and keep it, so a decay of any length can branch from this run. A run
    # with no decay has no such step
    decay_start: bool = False


@dataclass(kw_only=True)
class RunConfig:
    """How the model is trained - the name, where the run writes, the schedule and the held-out pass.

    `output` has no default: a config that does not say where is refused at load rather than writing
    somewhere nobody chose.
    """

    output: str
    name: str = "smoke"
    # Data-parallel degrees, -1 filling whatever the replicate degree leaves
    dp_shard: int = -1
    dp_replicate: int = 1
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Steps between checkpoints, or 0 to write none. torchtitan also writes one at the last step
    checkpoint_every: int = 0
    # Stage a checkpoint to pinned host memory and write it in the background, so a save pauses training for the
    # copy rather than the write. The last save is waited for before the run exits
    checkpoint_async: bool = True
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    # A torchtitan checkpoint step folder (`.../checkpoint/step-N`) whose weights seed step zero, or None for
    # random init. Model only: the optimiser, the schedule and the stream start fresh, which is what lets a
    # long-context phase branch off a shared trunk with its own schedule
    init_from: str | None = None

    def __post_init__(self) -> None:
        """Reject a mesh or a cadence that is not one.

        Raises:
            ValueError: A degree is zero or below `-1`, or `checkpoint_every` is negative.
            ValueError: `init_from` is not an absolute path.
            ValueError: `checkpoint.decay_start` is on without a decay or a stable stage after warmup.
        """
        if self.dp_replicate < 1 or self.dp_shard == 0 or self.dp_shard < -1:
            raise ValueError(
                f"dp_shard must be -1 or positive and dp_replicate positive - got {self.dp_shard}, {self.dp_replicate}"
            )
        if self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must not be negative - got {self.checkpoint_every}")
        if self.init_from is not None and not self.init_from.startswith("/"):
            raise ValueError(f"init_from must be an absolute path - got {self.init_from!r}")
        if self.checkpoint.decay_start:
            if self.optim.schedule != Schedule.WSD:
                raise ValueError("run.checkpoint.decay_start needs run.optim.schedule=wsd")
            if self.optim.decay_step_count == 0:
                raise ValueError(
                    "run.checkpoint.decay_start needs a decay - set run.optim.decay_ratio above 0"
                )
            if self.optim.decay_start_step <= self.optim.warmup_steps:
                raise ValueError(
                    "run.checkpoint.decay_start needs a stable stage after warmup - "
                    f"decay starts at {self.optim.decay_start_step}, warmup ends at {self.optim.warmup_steps}"
                )


@dataclass(kw_only=True)
class NativeConfig:
    """One native run - which attention, on what model, reading what, and how."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig

    def __post_init__(self) -> None:
        """Reject a run whose sequence does not fit the corpus, the batch or the attention's blocks.

        Raises:
            ValueError: A tokenized row does not divide into whole sequences, the held-out pass is on and the
                corpus is not tokenized, a rank's tokens are not whole sequences, the sequence is not a whole
                number of blocks, or the held-out set is not whole batches.
        """
        for card, _ in self.data.cards:
            if (
                isinstance(card, assets.TokenizedCard)
                and card.sample_length % self.data.seq_len
                and self.data.seq_len % card.sample_length
            ):
                raise ValueError(
                    f"seq_len must divide the tokenized row or span whole rows - {self.data.seq_len} vs "
                    f"{card.sample_length}"
                )
        if self.run.evaluation.enabled and not isinstance(self.data.asset, assets.TokenizedCard):
            raise ValueError(
                f"the held-out pass reads a tokenized corpus's documents and {self.data.name} is not one - "
                "set run.evaluation.every to 0, or name a tokenized corpus"
            )
        if self.run.optim.tokens_per_rank % self.data.seq_len:
            raise ValueError(
                f"seq_len must divide tokens_per_rank - {self.data.seq_len} does not divide "
                f"{self.run.optim.tokens_per_rank}"
            )
        if self.attention.kind in (Kind.MULTIRES, Kind.LSA) and self.data.seq_len % self.attention.block_size:
            raise ValueError(
                f"attention.block_size {self.attention.block_size} must divide seq_len {self.data.seq_len}"
            )
        ranks = self.run.dp_shard * self.run.dp_replicate if self.run.dp_shard > 0 else 1
        step = self.run.optim.tokens_per_step
        if step is not None and step % (self.run.optim.tokens_per_rank * ranks):
            raise ValueError(
                f"run.optim.tokens_per_step {step} must be a whole number of passes of every rank - "
                f"{self.run.optim.tokens_per_rank} tokens on each of {ranks} ranks"
            )
        self.check_ranks(ranks)

    @property
    def tokenizer(self) -> str:
        """Return the model card whose tokenizer the run uses, the model block's or else the corpus's.

        Raises:
            ValueError: The model block names none and the corpus is text, which carries no tokenizer.
        """
        if self.model.tokenizer is not None:
            return self.model.tokenizer
        for card, _ in self.data.cards:
            if not isinstance(card, assets.SourceCard):
                return card.tokenizer
        raise ValueError(f"{self.data.name} is text and has no tokenizer of its own - set model.tokenizer")

    @property
    def tokenizer_path(self) -> str:
        """Return the directory the run's tokenizer is read from."""
        return assets.model(self.tokenizer).checkpoint

    @property
    def sequences_per_batch(self) -> int:
        """Return the sequences a rank reads at once."""
        return self.run.optim.tokens_per_rank // self.data.seq_len

    def check_ranks(self, rank_multiple: int) -> None:
        """Refuse a held-out set that does not split into whole batches over the ranks.

        The evaluator refuses the same set when the job starts, which is after the nodes were allocated.

        Args:
            rank_multiple: Ranks the set splits over, or a divisor of them when the run does not yet know how
                many it gets.
        """
        self.run.evaluation.check_ranks(rank_multiple, self.sequences_per_batch)

    @property
    def out(self) -> Path:
        """Return the directory this run writes into."""
        return Path(self.run.output) / self.run.name

    def scored_checkpoint(self, given: str | None = None) -> str:
        """Return the step folder an evaluation of this run scores.

        Args:
            given: The folder the command line named, which wins, or None.

        Returns:
            `given`, else the `run.evaluation.checkpoint` the config names.

        Raises:
            ValueError: Neither names one.
        """
        checkpoint = given or self.run.evaluation.checkpoint
        if not checkpoint:
            raise ValueError(
                f"{self.run.name} names no checkpoint to score - pass --checkpoint <step folder>, "
                "or use a config that sets run.evaluation.checkpoint"
            )
        return checkpoint


def load(path: str | Path, overrides: list[str] | None = None) -> list[NativeConfig]:
    """Read a native run file and return the runs it describes.

    Args:
        path: Path to the yaml.
        overrides: Dotted `key=value` assignments applied after everything else.

    Returns:
        One `NativeConfig` per run, in sweep order.
    """
    return config_loader.load(NativeConfig, path, overrides, name_path="run.name")
