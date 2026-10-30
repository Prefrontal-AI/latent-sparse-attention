"""The values distilling a learned index space has to agree on, across its modules and with the cache on disk.

A group is imported by name, as in `from lsa.training.distillation.constants import Calibration`, because [the package's own constants](../../constants.py) own `const`.
A tunable a run might change is a field of [the run schema](config.py); what lives here is part of a contract or a default with no reason to differ.
"""


class Split:
    """Which side of the boundary a cached document sits, as a `<collection>/<split>/` directory.

    The side is a hash of the document id, so every window of a document lands on one side and more documents never move a cached one.
    """

    TRAIN: str = "train"
    EVAL: str = "eval"
    BOTH: tuple[str, str] = (TRAIN, EVAL)

    # Share of the held-out documents that score the map rather than train it
    EVAL_SHARE: float = 0.25

    # The digest this reads is salted apart from the held-out one, so the two decisions are independent
    # => a document being held out of pretraining says nothing about which side of this it falls
    SALT: str = "distill-split:"


class Selection:
    """What keeps a document selection reproducible."""

    # Fixed salt preserves the document ordering of existing datasets
    SALT: str = "mix:"


class Cache:
    """What a cached (document, layer) is called on disk, and what it holds."""

    # `<document>_layer<index>.pt`, under a collection directory
    LAYER_TAG: str = "_layer"
    SUFFIX: str = ".pt"
    GLOB: str = f"*{LAYER_TAG}*{SUFFIX}"

    # Where an extraction writes and a run reads, with the two agreeing by default
    ROOT: str = "caches/teacher"

    # The checkpoint's own sparse selection at each sampled position
    NATIVE: str = "native_key_ids"
    # The attention spent on each key of that selection, summed over heads
    NATIVE_MASS: str = "native_key_mass"

    # The sidecar `<document>_layer<index>_native.pt` that holds the selection when the extraction did not, from [native](data/native.py)
    NATIVE_TAG: str = "_native"

    # The one entry of a cached blob that is not a tensor
    META: str = "meta"

    # The tensors one cached (document, layer) holds
    FIELDS: tuple[str, ...] = (
        "x",
        "query_positions",
        "top_key_ids",
        "top_key_mass",
        "peak_key_id",
    )

    # The top-k cache format's blob fields in place of `x` - index keys and index queries, unit rows in the latent bases -
    # and the meta names of its coarse block, window and sinks
    INDEX_FIELDS: tuple[str, ...] = ("index_keys", "index_queries")
    INDEX_BLOCK: str = "block"
    INDEX_WINDOW: str = "window"
    INDEX_SINK: str = "sink"

    # Online teacher blob fields in place of `x` under the key and query: keys at every position, summed queries at the sampled rows
    KEY_AND_QUERY_FIELDS: tuple[str, ...] = ("key", "summed_query")
    # Under `run.input: key_and_query` also each head's folded query [Q, heads, key_dim] bf16, and the meta name of the input
    HEAD_QUERIES: str = "head_queries"
    INPUT: str = "input"
    # Under `queries.attention` also the rotated keys [T, kv_head_count, head_dim] bf16, queries [Q, query_head_count, head_dim] bf16 and the softmax scale
    ATTENTION_KEYS: str = "attention_keys"
    ATTENTION_QUERIES: str = "attention_queries"
    ATTENTION_SCALE: str = "attention_scale"

    # Online blob fields for a QSA indexer: each closed block's mean `x` and `x` at the sampled rows, a quarter of `x` at blocks of four
    QSA_FIELDS: tuple[str, ...] = ("block_x", "query_x")
    # On the key and query: each closed block's mean key input and the summed queries, a sixth of `x` on Qwen3.8-27B
    QSA_KEY_AND_QUERY_FIELDS: tuple[str, ...] = ("block_key", "summed_query")
    # The meta name of the pooling block, whose closing makes a key selectable
    BLOCK_SIZE: str = "block_size"

    # The layer input, residual streams mixed and normed
    X: str = "x"
    X_DTYPE: str = "bfloat16"

    # Meta name of the training query count
    TRAIN_QUERIES: str = "train_queries"

    # Under `queries.targets`, each query's kind as an index into `QUERY_KINDS` and the keys it should find, `NO_KEY` padded
    QUERY_KIND: str = "query_kind"
    TARGET_KEY_IDS: str = "target_key_ids"
    LINE_KEY_IDS: str = "line_key_ids"
    QUERY_KINDS: tuple[str, ...] = ("question", "answer", "control")
    QUERY_KIND_NAMES: str = "query_kinds"
    NO_KEY: int = -1

    # Meta names of the visibility rule, which a run's indexer has to agree with
    SINK_SIZE: str = "sink_size"
    WINDOW_SIZE: str = "window_size"
    CLUSTER_BLOCK_SIZE: str = "cluster_block_size"

    # Meta name of the document's token count
    LENGTH: str = "length"

    # Floor on the mass a teacher row is renormalised by, so a row with nothing readable stays zero
    MASS_EPS: float = 1e-9

    # Where a document waits before the training device has room for it
    HOST: str = "cpu"

    # Reading one document at a time gets a tenth of the disk's rate over thousands of files
    READ_WORKERS: int = 16


class Probe:
    """What `queries.targets` can ask a prepared prompt for."""

    # Query positions and targets read off a RULER row - see [teacher_plan](data/teacher_plan.py)
    RULER: str = "ruler"
    ALL: tuple[str, ...] = (RULER,)

    # The task whose answer is counted over a numbered word list, where every other task hides a needle
    LIST_TASK: str = "cwe"


class Pool:
    """The rolling pool of a layer's documents waiting in host memory, and the card they are staged to."""

    # Share of the job's host memory the pool may fill
    # => the rest is the interpreter, the CUDA context, the evaluation read-ahead and the page cache
    HOST_SHARE: float = 0.8

    # Host memory kept back for a report's read-ahead, in bytes
    # => `EVAL_AHEAD` documents of 8.7 GB at 1M tokens
    REPORT_BYTES: int = 40 * 10**9

    # Evaluation documents read ahead of the one being scored
    EVAL_AHEAD: int = 4

    # Pool size when neither the config nor the job gives host memory, in bytes
    DEFAULT_BYTES: int = 64 * 10**9

    # The largest staged document, 1M tokens, in bytes
    DOCUMENT_BYTES: int = 8_700_000_000

    # A step's card memory besides documents at 1M rows, in bytes
    # => the visible-key gather and 256-row divergence tiles, a few GB rounded up to 8 GiB for the allocator
    # `dense_full`'s chunked logits and its [256, 1M] target add under 3 GiB to that, inside the same bound
    STEP_BYTES: int = 8 * 2**30

    # Device memory kept back when sizing the cache the card holds
    # => `EVAL_AHEAD` read-ahead documents sit on the card, 4 x 8.7 GB + 8 GiB = 40.4 GiB
    HEADROOM_BYTES: int = EVAL_AHEAD * DOCUMENT_BYTES + STEP_BYTES

    # Seconds between the pool's progress lines
    LOG_SECONDS: float = 60.0

    # Seconds a waiting step goes between checks for a stop
    # => a preempted run is told 90 s ahead, and its snapshot must be written inside them
    INTERRUPT_SECONDS: float = 1.0

    # Documents whose steps are mixed at once, held on the card
    WINDOW: int = 8

    # Documents read whole to start a layer's PCA directions and temperature
    INIT_DOCUMENTS: int = 8


class Teacher:
    """Extraction, where the logits do not fit."""

    # fp32 cells of one block's [H, R, T] logits, which are never materialised whole - 128 query rows of 64
    # heads over a 64k document, and fewer rows as the document grows
    # => narrower than the deployed read's chunk, because every head's logits exist at once here
    SCORE_CELLS: int = 128 * 64 * 65536

    # A claim is a file a process writes before reading a document, so any number of processes share one cache
    # => survivors pick up what a killed job held, which a handed-out slice would orphan
    CLAIM_SUFFIX: str = ".claim"

    # How long a claim is honoured before it is taken over
    # => past one document so a live process is never overtaken
    CLAIM_STALE: float = 900.0

    # Seconds a process waits when everything left is claimed by a live one
    CLAIM_POLL: float = 30.0

    # One worker's job asks - a card holds one span, the host the layer inputs a chunked prefill keeps
    JOB_NAME: str = "lsa-targets"
    GPUS: int = 1
    CPUS: int = 16
    MEM_GB: int = 256
    HOURS: int = 24


class Mass:
    """What a dense attention-mass dump is called on disk, from [teacher_mass](data/teacher_mass.py)."""

    # `<root>/<collection>/<split>/<window>/layer<index>.safetensors`
    LAYER_PREFIX: str = "layer"
    SUFFIX: str = ".safetensors"

    # The format's name and version, in every file's metadata - a reader refuses a file it does not know
    FORMAT: str = "dense_mass"
    VERSION: int = 1

    # Bits of a shared entry's flag
    SINK_BIT: int = 1
    WINDOW_BIT: int = 2

    # fp32 cells of one block's logits - a block holds the logits, the all-causal softmax, its mask-sums and the
    # cache-style softmax at once
    # => half the plain extraction's block
    SCORE_CELLS: int = Teacher.SCORE_CELLS // 2


class Online:
    """The teacher run inside a distillation job in place of a cache, from [online](data/online.py)."""

    # Host memory for documents read but not yet asked for by a pool, in GB
    # => some ten 1M documents of every layer under the key and query
    AHEAD_GB: float = 200.0

    # Shared disk for documents another node's copies read, in GB
    # => a 1M document is some 27 GB over sixteen layers, a few per copy of a node of eight
    STAGING_GB: float = 400.0
    # Under the run directory, one directory per launch so a requeue starts clean
    STAGING: str = "teacher_staging"
    # Seconds between looks at the shared disk
    POLL_SECONDS: float = 1.0


class Stage:
    """An extraction run one span of the stack at a time, so a span fits one card."""

    # Under the cache root, the residual streams a cut hands the next span, as `layer<index>`
    BOUNDARIES: str = "boundaries"
    CUT_TAG: str = "layer"

    # The streams as raw bytes, and their description
    # => written last, so its presence says the streams are whole
    STREAMS_SUFFIX: str = ".streams"
    META_SUFFIX: str = ".json"

    # A span's claim tag, since each span claims apart
    CLAIM_TAG: str = ".from"


class Space:
    """The learned maps before any training."""

    # Rows of `x` the PCA init is taken from, spread evenly over the sampled documents
    INIT_ROWS: int = 8192

    # Positions projected at once
    # => a million-token document never holds its fp32 copy whole
    PROJECT_CHUNK: int = 65536

    # Uncalibrated temperature
    DEFAULT_TAU: float = 1.0

    # The QSA indexer's norm epsilon, Qwen3.8's `rms_norm_eps`
    QSA_NORM_EPS: float = 1e-6


class Draw:
    """What a training step's document is drawn in proportion to."""

    # Every document equally
    DOCUMENT: str = "document"
    # Every token equally, so a length bucket gets the steps its share of the tokens says
    # => a 1M document is drawn 64 times as often as a 16k one
    TOKEN: str = "token"

    BOTH: tuple[str, ...] = (DOCUMENT, TOKEN)


class Divergence:
    """How the loss measures the gap between the teacher's attention and the index score"""

    # Mode-covering
    FORWARD: str = "forward_kl"
    # Symmetric and bounded, defined against a target that is zero almost everywhere where reverse KL is not
    JENSEN_SHANNON: str = "jensen_shannon"

    BOTH: tuple[str, ...] = (FORWARD, JENSEN_SHANNON)

    # Floor under a probability before its log, for the symmetric mixture
    EPS: float = 1e-12


class Tail:
    """What the dense ranked target does with the mass beyond its depth"""

    # Drop it and renormalise what the kept keys hold
    TRUNCATE: str = "truncate"
    # Keep it as one entry the student answers with its own mass on the keys outside the kept set
    REST_BUCKET: str = "rest_bucket"

    BOTH: tuple[str, ...] = (TRUNCATE, REST_BUCKET)


class Rows:
    """Which of a document's query rows a scoring set reads"""

    # The held-out tail, which sees the whole document and is never trained on
    TAIL: str = "tail"
    # Training rows spread over the document, how far a map fits what it trained on
    TRAIN: str = "train"
    BOTH: tuple[str, ...] = (TAIL, TRAIN)

    # Training rows scored per document, evenly spaced
    TRAIN_SCORED: int = 128


class Term:
    """What each term a step pays for is called, in the log and the weights."""

    # The per-key divergence every run trains
    KEY: str = "key"


class Target:
    """What the loss measures the index score against."""

    # The teacher's dense attention over its strongest readable keys
    DENSE: str = "dense"
    # The same keys where the checkpoint's own selector also picked them
    DENSE_ON_NATIVE: str = "dense_on_native"
    # The attention the served model spends over everything its selector picked
    NATIVE_ATTENTION: str = "native_attention"
    # The selector's picks themselves, each weighted equally
    NATIVE_SELECTION: str = "native_selection"
    # The teacher's dense attention over every readable key, computed per step from the key latent and each head's query
    # => never truncated, and cached nowhere
    DENSE_FULL: str = "dense_full"

    ALL: tuple[str, ...] = (DENSE, DENSE_ON_NATIVE, NATIVE_ATTENTION, NATIVE_SELECTION, DENSE_FULL)
    # The ones a step computes from the key latent and the heads' queries, where the others read the cached ranking
    STEP_TIME: tuple[str, ...] = (DENSE_FULL,)

    # fp32 cells of one key chunk's [R * H, keys] logits, which are never held whole - 128 query rows of 64 heads, 16,384 keys
    # => 0.5 GiB a chunk, a few times that at the peak, at 1M keys as at 16k
    CHUNK_CELLS: int = 128 * 64 * 16384
    # The ones that need the selector's picks, from a `native_key_ids` sidecar
    PICKED: tuple[str, ...] = (DENSE_ON_NATIVE, NATIVE_SELECTION)
    # The ones `target_depth` truncates, read off the dense ranking
    RANKED: tuple[str, ...] = (DENSE, DENSE_ON_NATIVE)


class Native:
    """The offline pass that computes the checkpoint's own selection from the cached `x`."""

    JOB_NAME: str = "lsa-native-picks"
    GPUS: int = 1
    CPUS: int = 8
    # One document's mmap'd `x`, 8.7 GB at 1M tokens, and its page cache
    MEM_GB: int = 64
    HOURS: int = 12

    # A claim is honoured past the slowest document, 1M tokens
    CLAIM_STALE: float = 1800.0
    CLAIM_SUFFIX: str = ".claim"

    # Seconds between looks at a cache still being written
    POLL: float = 60.0

    # The (document, layer) files, whose names end in the layer index
    LAYER_PATTERN: str = r"^(?P<document>.+)_layer(?P<layer>\d+)\.pt$"

    # Query rows folded at once, 0.5 GB of fp32 at GLM's 64 heads of 512
    QUERY_ROWS: int = 4096

    # The family whose key and query input is a latent the heads share, GLM-5.3-Flash's MLA
    LATENT_FAMILY: str = "glm5_next"


class Calibration:
    """The temperature search that runs before step one."""

    # Log-spaced temperatures from sharp to flat, as `torch.logspace` takes them
    GRID: tuple[float, float, int] = (-2.5, 0.0, 26)
    # A QSA indexer's, whose scores are rectified dots of summed normed heads, order one at a random start
    QSA_GRID: tuple[float, float, int] = (-2.0, 1.0, 31)

    # Query rows read per document
    ROWS: int = 256

    # Floor under a teacher probability before its log is taken
    ENTROPY_EPS: float = 1e-12


class Report:
    """What a report measures, and how often a run logs."""

    # Prefixes of the teacher's top keys a report counts
    RECALL_DEPTHS: tuple[int, ...] = (16, 64)
    # Where a report keeps the exact reads at the depths a run asks for, and the per-layer means beside the mean
    DEPTHS: str = "depths"
    BY_LAYER: str = "by_layer"
    # Keys at least this far behind a row are far, the frozen coarse block
    FAR_BEHIND: int = 8192

    # Query rows a QSA read scores at once, 0.27 GB of dots over a 1M document
    QSA_ROWS: int = 64

    # Query rows a served read reads at most per document
    SERVED_ROWS: int = 256
    # What a served design takes from the run, the cache fixing the visibility rule and the maps the width
    SERVED_FIXED: tuple[str, ...] = ("window_size", "sink_size", "index_dim")
    # What a served design reads its maps with unless it names its own
    SERVED_INHERITED: tuple[str, ...] = ("input", "query_heads")
    # Tensor-parallel ranks the heads split over under the key and query, one node at TP8
    SERVED_RANKS: int = 8
    # A served read's place in a report
    SERVED: str = "served"

    # Steps between printed loss lines
    LOG_EVERY: int = 100

    # Steps between wandb scalars, which the sweeps are read from
    WANDB_LOG_EVERY: int = 10


class Results:
    """What a run writes into its own directory."""

    # The trained maps, the last snapshot under a stepless name
    # => their presence marks a run complete
    MAPS: str = "maps.pt"
    META: str = "meta.json"
    # The per-key divergence less the target's entropy
    KL: str = "kl"
    REPORT: str = "report_step{step}.json"
    # The maps at one report, which a later run starts from
    SNAPSHOT: str = "maps_step{step}.pt"
    # Everything a stopped run continues from, one file overwritten whole
    RESUME: str = "resume.pt"

    # A rescore's report and its step, a scored checkpoint not being a step
    RESCORE: str = "report.json"
    NO_STEP: int = -1

    # A served-read scoring's record per (document, layer), documents scored and tables
    RECORDS: str = "records"
    DOCUMENTS: str = "documents.json"
    TABLES: str = "report.md"
    SUMMARY: str = "summary.json"

    # A sharded run's shard directories and its joined wandb run
    SHARDS: str = "shards"
    SHARD: str = "layers-{layers}"
    JOINED: str = "joined"


class Job:
    """What a slurm array asks for per run by default."""

    NAME: str = "lsa-train"

    # One GPU, and its device inside the job
    GPUS: int = 1
    DEVICE: str = "cuda:0"
    CPUS: int = 8

    # The cache that does not fit the card waits in host memory
    # => the 27B's sixteen layers are 179 GB
    MEM_GB: int = 256
    HOURS: int = 4

    # The shard-joining job reads json and small checkpoints
    JOIN_MEM_GB: int = 32
    JOIN_HOURS: float = 0.5


class Served:
    """What a served-read scoring job asks for by default, and what it reads."""

    JOB_NAME: str = "lsa-served-read"
    GPUS: int = 1
    CPUS: int = 8
    # A report's read-ahead, 8.7 GB a document at 1M tokens, and the page cache
    MEM_GB: int = 128
    HOURS: int = 2

    # The exact read at the run's budget, and at another design's
    EXACT: str = "exact"
    EXACT_AT: str = "exact_k{top_k}"
    # A design's exact read in its own space at its budget
    EXACT_OF: str = "exact_{design}"
    # The union of the ranks' lists of a key and query read
    UNION: str = "{read}_union"
    # What a record keeps of each row
    PER_ROW: tuple[str, ...] = (
        "mass",
        "recall",
        "recall16",
        "ids_read",
        "far_share",
        "far_mass",
        "native_overlap",
    )
    # Key latents projected at a time
    KEY_PIECE: int = 65536
    # GLM's own picks, from the sidecars beside the cache
    NATIVE: str = "native"

    # A design's mass this far under the native read's is flagged
    BELOW_NATIVE: float = 0.02

    # An arm reads a bucket cache and its levels, or PISA's block pyramid
    ARM_REFERENCE: str = "reference"
    ARM_PISA: str = "pisa"
    ARM_KINDS: tuple[str, ...] = (ARM_REFERENCE, ARM_PISA)
    # The space PISA reads, the model's keys and queries or a design's index keys and index queries
    SPACE_ATTENTION: str = "attention"
    SPACE_INDEX: str = "index"
    ARM_SPACES: tuple[str, ...] = (SPACE_ATTENTION, SPACE_INDEX)

    # What a probe record keeps of each row, to group them afterwards
    PROBE_ROW_METRICS: tuple[str, ...] = (
        "mass",
        "recall",
        "native_overlap",
        "target_held",
        "target_any",
        "line_held",
        "agreement",
        "ids_read",
        "scored_vectors",
        "scored_dots",
    )
    # Table splits of a cwe prompt's rows, a bucket running from its edge to the next, the last open
    TARGET_COUNT_EDGES: tuple[int, ...] = (1, 2, 4, 8)
    DISTANCE_EDGES: tuple[int, ...] = (0, 4096, 16384, 65536, 262144)
    # The tables' cost and error columns
    COST_METRICS: tuple[str, ...] = ("scored_vectors", "scored_dots")


class Inject:
    """What an injected checkpoint is made of - the source plus one shard, a rewritten weight index and a config block."""

    # The projections' own shard, leaving the source shards untouched
    SHARD: str = "model-lsa-index.safetensors"

    # The flat per-layer file the sglang plugin reads, beside the checkpoint
    PROJECTIONS: str = "projections.pt"

    # The `config.json` block describing the injected tensors
    BLOCK: str = "lsa_index"
