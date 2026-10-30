"""The values a native run has to agree on - the attention's options, torchtitan's labels and what a job asks slurm for.

A group is imported by name, as `from lsa.training.native.constants import Kind`.
A tunable a run might change is a field of [the run schema](config.py); what lives here is the set of names a field chooses between or a default with no reason to differ.
"""


class Kind:
    """Which attention a run's full-attention layers use."""

    DENSE: str = "dense"
    WINDOW: str = "window"
    MULTIRES: str = "multires"
    # The flat LSA read lsa serves: one selection shared by every head, in the q,k input's index space
    LSA: str = "lsa"
    PISA: str = "pisa"
    NSA: str = "nsa"
    DSA: str = "dsa"
    ALL: tuple[str, ...] = (DENSE, WINDOW, MULTIRES, LSA, NSA, PISA, DSA)


class Backend:
    """Which of torchtitan's kernels a dense run's full-attention layers use."""

    FLEX: str = "flex"
    FLEX_FLASH: str = "flex_flash"
    VARLEN: str = "varlen"
    ALL: tuple[str, ...] = (FLEX, FLEX_FLASH, VARLEN)


class Flavor:
    """Which model a run trains - torchtitan's Qwen3.5 sizes, and two models at the GPT-2 vocabulary."""

    QWEN35_0_8B: str = "0.8B"
    # Qwen3.5-0.8B's text decoder, three Gated DeltaNet layers to one full-attention layer, tied embeddings
    HYBRID_GPT2: str = "0.8B-gpt2"
    # Nineteen blocks of plain attention and a SwiGLU
    DENSE_GPT2: str = "340M-gpt2"
    GPT2: tuple[str, ...] = (HYBRID_GPT2, DENSE_GPT2)

    # GPT-2's 50,257 ids, rows padded to a multiple of 64
    GPT2_VOCAB_SIZE: int = 50304
    # Init and rotation at the GPT-2 vocabulary
    GPT2_INIT_STD: float = 0.02
    GPT2_ROPE_THETA: float = 10_000.0


class Partition:
    """How a closed block's keys are cut into buckets."""

    WARD: str = "ward"
    # Runs of consecutive keys, bucket_size each
    CHUNKS: str = "chunks"
    # A seeded shuffle of the block, then runs of bucket_size
    RANDOM: str = "random"
    ALL: tuple[str, ...] = (WARD, CHUNKS, RANDOM)


class Rank:
    """How a query orders the buckets it may read at fine resolution."""

    # The estimated softmax mass of the bucket, normalised per head
    MASS: str = "mass"
    # The estimated mass times the variance the coarse term gets wrong
    ERROR: str = "error"
    # The cosine of the query with the bucket's unit centroid, which is the flat design's own score
    COSINE: str = "cosine"
    ALL: tuple[str, ...] = (MASS, ERROR, COSINE)


class Schedule:
    """Which learning-rate trajectory follows the warmup."""

    WSD: str = "wsd"
    COSINE: str = "cosine"
    ALL: tuple[str, ...] = (WSD, COSINE)


class DecayShape:
    """How the learning rate falls over the decay stage."""

    LINEAR: str = "linear"
    # One minus the square root of the decay's progress
    SQRT: str = "sqrt"
    COSINE: str = "cosine"
    ALL: tuple[str, ...] = (LINEAR, SQRT, COSINE)


class Qwen35:
    """What the Qwen3.5 text model is, as torchtitan builds it."""

    # Tokens per sequence when a run says nothing
    SEQ_LEN: int = 16384


class Runtime:
    """How much of the forward is recomputed, and when a run writes a checkpoint."""

    NONE: str = "none"
    SELECTIVE: str = "selective"
    FULL: str = "full"
    CHECKPOINTING: tuple[str, ...] = (NONE, SELECTIVE, FULL)

    # The interval that means never, for a checkpointer switched off by `enable` alone
    NEVER: int = 1_000_000

    OFF: str = "off"
    LOSS: str = "loss"
    MODEL: str = "model"
    COMPILE: tuple[str, ...] = (OFF, LOSS, MODEL)


class Checkpoint:
    """torchtitan's names for how a checkpoint is written."""

    SYNC: str = "disabled"
    ASYNC: str = "async_with_pinned_mem"


class Results:
    """What a run writes into its own directory, beside what torchtitan writes."""

    CONFIG: str = "config.json"


class Job:
    """What a slurm job asks for per run by default."""

    NAME: str = "lsa-native"
    GPUS: int = 8
    NODES: int = 1
    CPUS: int = 8
    MEM_GB: int = 600
    HOURS: int = 4


class Evaluation:
    """What a held-out pass reads and how it is scored by default."""

    EVERY: int = 0
    SEQUENCES: int = 64

    # Position bucket starts after the first, 512 closing the open block and 1540 being sinks plus window plus top-k
    # => below 1540 every arm reads every key and the runs must agree
    EDGES: tuple[int, ...] = (512, 1540, 4096, 8192)

    # Sequence chunks the output projection runs over, so the logits are never whole
    CHUNKS: int = 8

    # Queries per sequence the coarse mass share is measured on, evenly spaced
    PROBES: int = 128

    FOLDER: str = "evaluation"
    RECORDS: str = "step-{step:06d}.jsonl"
    RETRIEVAL_RECORDS: str = "step-{step:06d}.retrieval.jsonl"
    SUMMARY: str = "step-{step:06d}.json"


class Retrieval:
    """The facts a retrieval probe splices into a held-out sequence, and its default distances."""

    # Tokens between the needle's end and the query. 256 sits inside the 512 window and 15,000 nearly spans a 16k sequence
    DISTANCES: tuple[int, ...] = (256, 1024, 2048, 4096, 8192, 15000)
    SEQUENCES: int = 32
    DISTRACTORS: int = 4
    # Digits of a code, one token each in Qwen3.5's vocabulary
    DIGITS: int = 6

    HEAD: str = "The secret code for {name} is"
    # Text that takes the needle's place in the control
    FILLER: str = "The archive was catalogued and filed without further remark. "

    FIRST_NAMES: tuple[str, ...] = (
        "Maria", "Tobias", "Ingrid", "Kofi", "Lucia", "Hamid", "Elena", "Jonas", "Priya", "Marcus", "Sofia", "Dmitri",
        "Amara", "Felix", "Noor", "Pablo", "Greta", "Tariq", "Yuki", "Oscar", "Leila", "Viktor", "Hana", "Rafael",
    )  # fmt: skip
    LAST_NAMES: tuple[str, ...] = (
        "Lopez", "Brandt", "Okafor", "Novak", "Haddad", "Larsen", "Moreau", "Tanaka", "Ferreira", "Kowalski", "Rossi", "Singh",
        "Duarte", "Weber", "Mbeki", "Sorensen", "Petrov", "Alvarez", "Nakamura", "Fischer", "Costa", "Baranov", "Quinn", "Volkov",
    )  # fmt: skip


class WeightDecayScope:
    """Which parameters AdamW decays."""

    ALL_PARAMETERS: str = "all"
    MATRICES: str = "matrices"
    ALL: tuple[str, ...] = (ALL_PARAMETERS, MATRICES)
