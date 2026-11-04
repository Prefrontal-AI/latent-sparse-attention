"""The values the evaluation harness agrees on with the checkpoints it drives and every reader of its records.

A group is imported by name, as in `from lsa.evaluation.constants import Suite`, because [the package's own constants](../constants.py) own `const`.
A tunable a run might change is a field of [the experiment schema](config.py); what lives here is part of a contract or a default with no reason to differ.
Server launch arguments and the settings file's design knobs are `lsa.sglang`'s.
"""

from typing import ClassVar

from lsa import constants as const


class Adapter:
    """How the adapter that drives a checkpoint is found."""

    # The adapter a model_type with no entry of its own gets
    GENERIC: str = "hf"

    # The entry point group an installed package advertises its adapters through
    ENTRY_POINT_GROUP: str = "lsa.adapters"


class Layer:
    """What a stack calls the layers LSA replaces, and what one of them calls its parts"""

    # What a hybrid stack calls its quadratic layers when the config does not say otherwise
    DEFAULT_TYPES: tuple[str, ...] = ("full_attention",)

    # Per-head QK normalisation as (query, key) name pairs - Qwen and Llama say `q_norm`, phi, stablelm, persimmon and lfm2 `q_layernorm`
    # => one name alone would silently skip the other model's norm
    QK_NORMS: tuple[tuple[str, str], ...] = (("q_norm", "k_norm"), ("q_layernorm", "k_layernorm"))

    # A multimodal checkpoint's vision tower, left out of a span load
    VISION_TOWERS: tuple[str, ...] = ("visual", "vision_tower")

    # Positions a teacher projects at once
    # => a 1M document's queries and gates at Qwen3.8-27B's 24 heads are 24 GiB of bf16 whole
    TEACHER_SLICE: int = 65536

    # Attention for chunked prefill, causal from the bottom right
    # => new queries see every cached key without a dense mask
    LOWER_RIGHT_CAUSAL: str = "lsa_lower_right_causal"

    # FlashAttention-3 reading grouped key heads natively, at a pinned hub commit
    # => a moved kernel would move the targets' bits
    FLASH_KERNEL: str = "kernels-community/flash-attn3"
    FLASH_KERNEL_REVISION: str = "3c1451f803c146c54222305a8c350f7f46bb5135"


class Glm(const.Glm):
    """GLM-5.3-Flash as the harness reads it, over the names every package shares."""

    LAYER_TYPES: tuple[str, ...] = (const.Glm.SPARSE_LAYER,)

    # Query rows the DSA indexer scores per pass
    # => scores [B, rows, index_n_heads, pools] fp32 are 128 GiB over a 64k document, and blocks are exact since rows are independent
    INDEXER_QUERY_CHUNK: int = 4096
    # fp32 cells of one block's indexer scores, 4,096 queries by 32 heads by 16,384 pools
    # => a block narrows past a 64k document rather than growing
    INDEXER_SCORE_CELLS: int = 4096 * 32 * 16384

    # The attention a long extraction rebinds to read through its selection
    ATTENTION_CLASS: str = "Glm5NextTextAttention"
    # fp32 cells of one block's logits through the selection, heads by selected keys by some 1,000 queries
    ATTENTION_SCORE_CELLS: int = 2**27
    # Tokens projected to the latent at once when a teacher reads a whole document
    LATENT_SLICE: int = 65536

    # The module the block-at-a-time selection is bound onto, matched by name
    # => importing it would pull in the whole modelling file
    INDEXER_CLASS: str = "Glm5NextTextIndexer"


class Family:
    """The two architectures a kernel measurement drives, named as their model cards name the family."""

    GLM5_NEXT: str = "glm5_next"
    QWEN3_5: str = "qwen3_5"

    # Each family's quadratic layer types
    ATTENTION_LAYERS: ClassVar[dict[str, tuple[str, ...]]] = {
        GLM5_NEXT: Glm.LAYER_TYPES,
        QWEN3_5: Layer.DEFAULT_TYPES,
    }


class Attention(const.Attention):
    """Which keys a query reads, and the paths a kernel measurement times beside them."""

    # The linear attention layer beside the quadratic ones in a hybrid stack, with a constant state
    KDA: str = "kda"
    # Qwen3.5's gated delta net, in KDA's place
    GDN: str = "gdn"

    # The paths a config can name
    # => `full` (no selection, the floor beside `native`), `kda` and `gdn` are timings only, none a selection to score
    VARIANTS: tuple[str, ...] = (*const.Attention.KINDS, const.Attention.FULL, KDA, GDN)


class Suite:
    """What gets asked of a model once it is up."""

    # Wall-clock ceiling per suite, about twice the measured cost, which names the suites so a typo fails at `--dry-run`
    # => RULER past 1M is 61 hours on two nodes, one request at a time, and PaperTrail past 1M adds about 30 hours
    # A native run raises it with `run.hours`
    HOURS: ClassVar[dict[str, int]] = {"ruler": 120, "full": 240, "papertrail": 240, "smoke": 1}
    NAMES: tuple[str, ...] = tuple(HOURS)

    # The datasets with a card of their own the others are named by
    MMLU_PRO: str = "mmlu_pro"
    GPQA_DIAMOND: str = "gpqa_diamond"
    MATH500: str = "math500"
    AIME25: str = "aime25"
    LONGBENCH: str = "longbench2"
    # A RULER task is `ruler_<task>_<length>`, all one dataset
    RULER: str = "ruler"
    RULER_PREFIX: str = "ruler_"
    # A GraphWalks task is `graphwalks_<file>`, one per hub file, all one dataset
    GRAPHWALKS: str = "graphwalks"
    GRAPHWALKS_PREFIX: str = "graphwalks_"
    # A PaperTrail task is `papertrail_<sample>_<condition>`, `_nic` for not-in-corpus questions
    PAPERTRAIL: str = "papertrail"
    PAPERTRAIL_PREFIX: str = "papertrail_"

    # Phases a run takes in order, every task of a phase in flight at once
    # => a long prefill holds up decode, so reasoning runs first, long-prompt reasoning (LongBench v2, GraphWalks) next, PaperTrail apart and RULER last
    PHASES: tuple[tuple[str, ...], ...] = (
        (MMLU_PRO, GPQA_DIAMOND, MATH500, AIME25),
        (LONGBENCH, GRAPHWALKS),
        (PAPERTRAIL,),
        (RULER,),
    )

    # The datasets a `generation` block is keyed by
    DATASETS: tuple[str, ...] = tuple(dataset for phase in PHASES for dataset in phase)

    @classmethod
    def dataset_of(cls, task: str) -> str:
        """Name the dataset a task is drawn from - RULER's, GraphWalks' or PaperTrail's for one of theirs, else its own."""
        if task.startswith(cls.RULER_PREFIX):
            return cls.RULER
        if task.startswith(cls.GRAPHWALKS_PREFIX):
            return cls.GRAPHWALKS
        if task.startswith(cls.PAPERTRAIL_PREFIX):
            return cls.PAPERTRAIL
        return task


class Production:
    """What a served production run sends, and what its record agrees on with every reader."""

    # A production record's schema
    SCHEMA: str = "lsa-production/1"

    # sglang's serving benchmark, sending token ids to `/generate` streamed
    # => no tokenizer sits in the timed path
    CLIENT: str = "sglang.benchmark.serving"

    # The default workload card
    WORKLOAD: str = "production"

    # Profiler traces under the run directory, a folder a length
    PROFILES: str = "profiles"

    # The `/server_info` settings a number depends on
    SERVER_KEYS: tuple[str, ...] = (
        "model_path",
        "attention_backend",
        "dsa_prefill_backend",
        "dsa_decode_backend",
        "kv_cache_dtype",
        "tp_size",
        "pp_size",
        "nnodes",
        "context_length",
        "max_running_requests",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mem_fraction_static",
        "disable_radix_cache",
        "reasoning_parser",
        "speculative_algorithm",
        "quantization",
        "max_total_num_tokens",
        "version",
    )

    # Wall-clock ceiling, 13 minutes a configuration to 1M on one node plus a 15-minute start
    # => two nodes to 10M take 2.2 hours native and raise `run.hours`
    HOURS: int = 4


class Mesh:
    """The label every reader keys a run by, naming every axis with its size even at 1, as `tp1-cp8-dcp8-pp1-ep8`.

    | axis | what it splits | server setting |
    |---|---|---|
    | `tp` | attention's heads - tensor parallel inside attention | `tp_size / attn_cp_size / attention dp` |
    | `cp` | a prefill chunk's rows - prefill context parallel | `attn_cp_size` |
    | `dcp` | the KV pool's slots - decode context parallel | `dcp_size` |
    | `pp` | the layers - pipeline stages | `pp_size` |
    | `ep` | the experts - expert parallel | `ep_size` |

    `tp` is attention's own width, so `tp1-cp8-...` and `tp8-cp1-...` run on the same eight GPUs.
    `dcp` shares attention's ranks and never multiplies the GPU count.
    """

    # Each axis's label letters and mesh field, in label order
    AXES: tuple[tuple[str, str], ...] = (
        ("tp", "attn_tp_size"),
        ("cp", "attn_cp_size"),
        ("dcp", "dcp_size"),
        ("pp", "pp_size"),
        ("ep", "ep_size"),
    )

    # The server settings a mesh is read from, the context-parallel ones missing on a stock image
    SETTINGS: tuple[str, ...] = (
        "tp_size",
        "attn_cp_size",
        "dcp_size",
        "enable_prefill_cp",
        "cp_strategy",
        "pp_size",
        "ep_size",
        "dp_size",
        "enable_dp_attention",
        "nnodes",
        "moe_runner_backend",
    )

    # A record's mesh key
    FIELD: str = "mesh"


class Records:
    """Where a run's record lands and every reader looks.

    One root rather than one per experiment, so a records reader can put runs side by side.
    Each kind of run has its own directory inside it - `Evaluation.DIRS`.
    """

    # Runs without a shared root land under the working directory
    LOCAL_ROOT: tuple[str, ...] = ("results",)

    # A run directory's record, the launcher's metadata written last, and the full `/server_info`
    RECORD: str = "record.json"
    META: str = "meta.json"
    SERVER_INFO: str = "server_info.json"

    # An accuracy record's schema
    SCHEMA: str = "lsa-accuracy/2"

    # The cap key of rows that recorded none
    UNKNOWN_CAP: str = "unknown"

    # A production record's settings plus request templating, sampling and think budget
    SERVER_KEYS: tuple[str, ...] = (
        *Production.SERVER_KEYS,
        "served_model_name",
        "enable_strict_thinking",
        "grammar_backend",
        "default_chat_template_kwargs",
        "sampling_defaults",
    )

    # Settings the parts of a joined run may differ in, changing speed and fit but not an answer
    # => every other `SERVER_KEYS` entry but the model path must agree, and the joined record keeps each part's values
    SERVING_KNOBS: tuple[str, ...] = (
        "context_length",
        "max_running_requests",
        "max_total_num_tokens",
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mem_fraction_static",
    )

    # Where a joined run keeps each part's remaining files
    PARTS: str = "parts"

    # Percentile bootstrap of a record's `ci95` and a paired interval, resampling docs from a fixed seed
    BOOTSTRAP_DRAWS: int = 10_000
    BOOTSTRAP_SEED: int = 0
    BOOTSTRAP_LEVEL: float = 0.95

    # A run id's UTC start stamp, as `2026-09-23__15-37-53`, to the second and without colons
    RUN_STAMP: str = "%Y-%m-%d__%H-%M-%S"

    # A task entry says so when its score is better the lower it is, as a loss is. Absent, higher is better
    LOWER_IS_BETTER: str = "lower_is_better"


class Evaluation:
    """Which runner an experiment is for."""

    # Start a server and score it with the accuracy suite
    ACCURACY: str = "accuracy"

    # Time one attention layer through the attention-layer harness
    KERNELS: str = "kernels"

    # Start a server and time its requests - first token and every token after it
    PRODUCTION: str = "production"
    KINDS: tuple[str, ...] = (ACCURACY, KERNELS, PRODUCTION)

    # Task names a dry run lists before it counts the rest
    DRY_RUN_TASKS: int = 40

    # Each kind's run directories under the results root, nested as the package is
    DIRS: ClassVar[dict[str, tuple[str, ...]]] = {
        ACCURACY: ("accuracy",),
        KERNELS: ("performance", "kernels"),
        PRODUCTION: ("performance", "production"),
    }


class Launches:
    """What every launch leaves beside its record - who started it, in which job, and how it ended.

    Each launch appends a line as it starts and one as it ends, so a launch with no end line is one slurm stopped.
    """

    # The run directory's launch log, one JSON object per line, appended to and never rewritten
    FILE: str = "launches.jsonl"

    # End line states, written only for endings the launcher saw
    # => a launch slurm cancelled or timed out is named by slurm's accounting
    COMPLETED: str = "completed"
    FAILED: str = "failed"

    # Seconds a live job's state is trusted before slurm is asked again
    # => a finished job's state never changes and is asked once
    LIVE_TTL: float = 60.0

    # A single job's name, and the prefix of a pooled run's jobs before its run id
    # => a job's slurm comment carries its run directory, which leads from the queue to the run
    SINGLE_JOB: str = "lsa-eval"
    POOL_JOB: str = "lsa-pool-"


class Pool:
    """How several workers share one run - their queue under the run directory, and how long a claim holds.

    Everything a pooled run keeps beside its record is under one directory, so publishing and joining leave it out by one name.
    """

    # The queue directory, holding the rest
    DIR: str = "pool"
    # Per task, a claim file per (doc, repeat) and epoch
    CLAIMS: str = "claims"
    # Per worker, its sample rows as `<worker>/<task>.jsonl`
    SHARDS: str = "shards"
    # Per worker, its host, server, heartbeat and uptime
    WORKERS: str = "workers"
    # The config, overrides and records root for submitting another worker
    SUBMISSION: str = "submission.json"
    # Per task, the docs the data held, which a plan's `n` cannot say for a whole dataset
    UNITS: str = "units"
    # The record before the first worker, which the workers' seconds are added to
    PRIOR: str = "prior.json"
    # Held during a record rebuild, one worker grading at a time
    LOCK: str = "record.lock"

    # Seconds a claim holds without a heartbeat
    # => the heartbeat is a `utime` a minute from a thread that never waits on a request, so a twenty-minute unit keeps its claim
    LEASE_SECONDS: float = 900.0
    HEARTBEAT_SECONDS: float = 60.0
    # Failed requests of one unit across every worker before the pool stops sending it
    MAX_ATTEMPTS: int = 3
    # A rebuild holding the lock this long died
    LOCK_STALE_SECONDS: float = 1800.0
    # Seconds between a worker's record rebuilds
    REBUILD_SECONDS: float = 120.0
    # Seconds the last worker waits for other nodes' rows to arrive before its last rebuild
    SETTLE_SECONDS: float = 5.0


class Dispatch:
    """What the dispatcher defaults to, which a flag overrides."""

    # Seconds between two looks at the reservation
    INTERVAL: float = 300.0
    # Workers one run may have at once
    MAX_WORKERS: int = 8
    # A unit's seconds for a run nothing has measured, an ultralong sample being fifteen to twenty minutes
    UNIT_SECONDS: float = 1200.0
    # A worker whose heartbeat is older than this is gone, whatever its job says
    DEAD_AFTER: float = 3 * Pool.HEARTBEAT_SECONDS
    # Launches in a row that failed before the dispatcher stops adding workers to a run
    MAX_FAILURES: int = 3


class Status:
    """What a run came to, worked out from its launches and its record and never stored.

    A launch that ended well is complete when its record holds everything it set out to measure and partial when it does not.
    """

    IN_PROGRESS: str = "in progress"
    COMPLETE: str = "complete"
    PARTIAL: str = "partial"
    FAILED: str = "failed"
    CANCELLED: str = "cancelled"
    TIMED_OUT: str = "timed out"
    # A launch with no end line that slurm cannot place
    UNKNOWN: str = "unknown"
    ALL: tuple[str, ...] = (IN_PROGRESS, COMPLETE, PARTIAL, FAILED, CANCELLED, TIMED_OUT, UNKNOWN)

    # Slurm accounting states of a launch with no end line
    # => COMPLETED without one ended off the launcher's path, and its record says complete or partial
    SLURM: ClassVar[dict[str, str]] = {
        "PENDING": IN_PROGRESS,
        "RUNNING": IN_PROGRESS,
        "REQUEUED": IN_PROGRESS,
        "RESIZING": IN_PROGRESS,
        "SUSPENDED": IN_PROGRESS,
        "COMPLETED": COMPLETE,
        "CANCELLED": CANCELLED,
        "TIMEOUT": TIMED_OUT,
        "DEADLINE": TIMED_OUT,
        "FAILED": FAILED,
        "OUT_OF_MEMORY": FAILED,
        "NODE_FAIL": FAILED,
        "BOOT_FAIL": FAILED,
        "PREEMPTED": FAILED,
    }


class Kernels:
    """What the attention-layer harness writes, and what its readers agree on.

    The harness is run rather than imported, so only this class binds its output to its readers.
    """

    # A kernel record's schema
    SCHEMA: str = "lsa-kernels/1"

    # One harness process's per-chunk or per-token output
    RAW_SCHEMA: str = "attention-layer-perf/1"

    # The module timer's and a whole-model composition's schemas
    MODULES_SCHEMA: str = "lsa-modules/1"
    COMPOSITION_SCHEMA: str = "lsa-composition/1"

    # The two halves of serving, measured apart since a prefill chunk is thousands of query rows and a decode token one
    PHASES: tuple[str, ...] = ("prefill", "decode")

    # A decode token's case, the heaviest maintenance it triggers, heaviest first
    # => the design grows its hierarchy on a full page, clusters a closed block, takes a bucket-cache page; the model's indexer closes a pool of four keys
    CASES: tuple[str, ...] = ("hierarchy_growth", "cluster_block_close", "page_allocation", "pool_close")

    # Trials per mode and phase
    # => a decode token is tenths of a millisecond against tens for a prefill chunk, so it needs more
    REPEATS: ClassVar[dict[str, int]] = {"prefill": 5, "decode": 20}

    # Wall-clock ceiling per phase, dominated by the 10M prefill point, which loads a truncated checkpoint and times 32 chunks
    HOURS: ClassVar[dict[str, int]] = {"prefill": 12, "decode": 8}
