"""The values the plugin and its consumers have to agree on, across a process boundary.

A group is imported by name, as in `from lsa.sglang.constants import Banner`, because [the package's own constants](../constants.py) own `const`.
They live here rather than in [plugin](plugin.py) because [launch](launch.py) and the evaluation's submitter read them from other processes.
A number only one module reads is named in that module.
"""


class Banner:
    """What a server prints about itself, the only thing a reader outside the process sees."""

    # Printed once every target resolved and every hook is registered
    READY: str = "lsa-lsa: armed"

    # Printed once an indexer is actually built, which is the thing `armed` does not prove
    # => `lsa.sglang.launch` refuses a server whose log never carries both
    BUILT: str = "lsa-lsa: indexing"

    # Printed once a decode context-parallel group checks its index after every forward
    CHECKING: str = "lsa-lsa: checking the decode group's index"

    # Printed once an index is built with the server's prefix cache on, and once a request first takes a prefix's index
    SHARING: str = "lsa-lsa: sharing cached prefixes' index"
    TAKEN: str = "lsa-lsa: a request took a cached prefix's index"


class Plugin:
    """What a stock sglang is told about this plugin - the key it is configured through."""

    # The key of the model config's block that holds every setting of a server, and of its debugging sub-block
    # => `--json-model-override-args` or a checkpoint's own config.json carries it, and no block is the checkpoint's own indexer
    BLOCK: str = "lsa"
    DEBUG: str = "debug"


class Server:
    """Where a server runs, what its job reserves, and how long to wait on it."""

    # Merges a card's rotation into the nested text config of a multimodal checkpoint
    MODEL_OVERRIDE_FLAG: str = "--json-model-override-args"
    TEXT_CONFIG: str = "text_config"

    # Where the pipeline stages meet, on the first node
    DIST_INIT_PORT: int = 29548

    # Host threads a served job reserves per GPU
    # An overlapping `srun` step sees only the GPUs affine to its job's cores, so a fraction of the node hides GPUs
    # => 16 is 128 threads over 8 GPUs, and 8 threads showed four of eight H200s
    CPUS_PER_GPU: int = 16

    # Host memory a served accuracy job reserves per GPU, floored at one GPU's worth
    # => 5x the 47 GB scorer peak at 4 GPUs on 5M, room to tokenize 10M
    MEM_GB_PER_GPU: int = 64

    # Host memory the kernel timer reserves on its one GPU
    # => twice the 100 GB peak at 10M
    MEM_GB: int = 200

    # Between health polls, which is a model load rather than a request
    POLL_SECONDS: float = 10.0

    # A health poll that has not answered by then is a poll, not a hang
    POLL_TIMEOUT_SECONDS: float = 10.0

    # Between the terminate and the kill
    SHUTDOWN_SECONDS: float = 60.0
