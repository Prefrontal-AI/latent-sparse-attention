"""The tokenized corpus format every training stage reads, and what else more than one stage reads.

[distillation](distillation/__init__.py) and the language-model run split the corpus by the same number here, or a run would train on the documents the index space was scored on.
What a corpus is, meaning its location, sources, tokenizer and row length, is a [card](../assets/__init__.py).
What distillation alone needs is its own [constants](distillation/constants.py).
"""


class Corpus:
    """What every tokenized corpus on disk looks like, and how every stage splits it."""

    # Share of documents held out of training, by document-id hash
    HELD_OUT_FRACTION: float = 0.02

    # Leading hex digits of the document-id digest the split reads, and the space they span
    # => every stage reads them the same way or the halves overlap
    HASH_DIGITS: int = 8
    HASH_SPACE: int = 0xFFFFFFFF


class Targets:
    """What a targets extraction records beside the tensors."""

    # The index a run's selection is resolved against, binding each cached name to its document id
    MANIFEST: str = "manifest.json"

    # The manifest's list of extracted documents, and a mix file's chosen ones
    DOCUMENTS: str = "documents"


class Loss:
    """What the loss is handed, and what it skips."""

    # Label value the loss skips, which is torch's own cross-entropy default and torchtitan's IGNORE_INDEX
    IGNORE_INDEX: int = -100


class Snapshot:
    """The files that describe a published HuggingFace snapshot."""

    CONFIG: str = "config.json"
    INDEX: str = "model.safetensors.index.json"


class CorpusSplit:
    """Which half of the tokenized corpus a run reads."""

    # `train` is what the targets never saw, `held_out` is what they were extracted from
    TRAIN: str = "train"
    HELD_OUT: str = "held_out"
    ALL: tuple[str, ...] = (TRAIN, HELD_OUT)


class Packing:
    """How a language-model sequence is cut from documents."""

    # Documents joined end to end, each ended by the end-of-document id, and cut into consecutive windows
    CONCATENATE: str = "concatenate"
    # Windows of one document each, never two documents in one, and a document's remainder past its last whole window dropped
    SPAN: str = "span"
    ALL: tuple[str, ...] = (CONCATENATE, SPAN)


class Field:
    """What a source reads off disk."""

    # Ids another pipeline or this one already wrote
    TOKENS: str = "tokens"
    # Text a tokenizer has still to turn into ids
    TEXT: str = "text"


class Stream:
    """How a stream reads its source, which no card declares."""

    # Tokens a token shard is read in at a time, and documents a text chunk holds
    SLAB_TOKENS: int = 1 << 22
    DOCUMENTS_PER_CHUNK: int = 256
