"""The values that have to agree across modules and across backends.

Grouped so a call site reads as its own sentence:

    from lsa import constants as const

    scores.masked_fill(hidden, const.Math.NINF)

A tunable a caller might reasonably change is an `IndexerConfig` field or a keyword argument.
What lives here is either part of the contract, like the padding value, or a default with no reason to differ between one call site and the next.

Plain classes rather than `enum.Enum`, because these go straight into tensor calls where `.value` would be noise.
"""


class Sentinel:
    """Values that mean there is nothing here."""

    # No such entry - the only padding value an index list ever carries
    PAD: int = -1


class Math:
    """Numerical guards, and the score a hidden entry carries."""

    # A masked-out score - the test for one is `> NINF` rather than equality
    NINF: float = float("-inf")

    # Zero-norm guard - a row with no direction stays zero and scores 0 against everything
    NORM_EPS: float = 1e-12


class Rotary:
    """Where the rotation sits in a head vector."""

    NONE: str = "none"
    PREFIX: str = "prefix"
    SUFFIX: str = "suffix"
    ENDS: tuple[str, ...] = (NONE, PREFIX, SUFFIX)

    # A config's alternative to naming an end - the layout is measured from the model
    # => never reaches RotaryLayout, which takes ENDS alone
    AUTO: str = "auto"
    DECLARED: tuple[str, ...] = (AUTO, NONE, PREFIX, SUFFIX)


class Chunk:
    """Rows per pass, so an intermediate is never materialised whole."""

    # Queries per block in the sparse read, so a [Tq, Tk] matrix never exists
    QUERY: int = 512

    # Query rows scored per pass in an exact read, which bounds the [rows, keys] scores
    READ: int = 256


class Attention:
    """Which keys a query reads."""

    # `native` is the model's own attention, the ceiling `lsa` is read against - shipped sparse attention stays under it
    # `pisa` is the original PISA design, a second selector served beside `lsa` for GLM
    NATIVE: str = "native"
    LSA: str = "lsa"
    PISA: str = "pisa"
    KINDS: tuple[str, ...] = (NATIVE, LSA, PISA)

    # What `native` is on a checkpoint - every key, or a learned selection like GLM-5.3-Flash's DSA and Qwen3.8-27B's QSA
    FULL: str = "full"
    BLOCK_SPARSE: str = "block_sparse"
    SHIPPED: tuple[str, ...] = (FULL, BLOCK_SPARSE)
    # The paper's methods a record files a run under, `pisa` being its Block Sparse (Tree)
    # => never the checkpoint's own kind, GLM's DSA as `block_sparse`
    RECORDED: tuple[str, ...] = (FULL, BLOCK_SPARSE, LSA, PISA)

    # The arms of a natively trained model, which no server ran. `window` is a sliding window with sinks, the name the
    # native trainer gives it, and `nsa` is the compressed, selected and windowed design
    NSA: str = "nsa"
    WINDOW: str = "window"
    TRAINED: tuple[str, ...] = (FULL, LSA, PISA, NSA, WINDOW)


class Glm:
    """GLM-5.3-Flash, in the names its config declares."""

    # The layers that attend over per-token keys and values, which LSA replaces
    SPARSE_LAYER: str = "deepseek_sparse_attention"

    # The `indexer_types` value of a layer that runs its own selection
    OWN_INDEXER: str = "full"
