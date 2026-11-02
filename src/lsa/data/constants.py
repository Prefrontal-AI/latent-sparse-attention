"""Dataset file formats the preparing and the reading modules share."""


class CorpusFiles:
    """Name the two files holding one tokenized source, and the columns of the manifest."""

    BIN_SUFFIX: str = ".bin"
    MANIFEST_SUFFIX: str = ".manifest.parquet"

    # Per document: its id, the first row of the bin it was tokenized into, and how many rows it took
    DOC_ID: str = "doc_id"
    ROW_START: str = "row_start"
    N_SAMPLES: str = "n_samples"


class ShardFiles:
    """Name the files of a directory of token shards, which another pipeline wrote and this one reads in place."""

    # What the writer measured, and the flat token files - training shards in name order, and one validation file
    METADATA: str = "metadata.json"
    TRAIN_GLOB: str = "train_*.bin"
    VALIDATION: str = "val.bin"


class Collected:
    """What the documents a corpus is tokenized from look like on disk.

    They belong to no tokenizer, which is the property that matters: one set of documents produces both
    `qwen3_5` and `glm53_flash` under the same document-id split.
    """

    # One folder of shards per source, and the marker that says a source finished collecting
    SHARD_GLOB: str = "shard_*.parquet"
    DONE_MARKER: str = "_DONE"

    # The two columns a shard carries that tokenizing reads
    DOC_ID: str = "doc_id"
    TEXT: str = "text"

    # The three it carries beside them, which the collector writes - the source's name, its one licence label
    # (spelled as the first collector spelled it) and the document's tokens under the collector's tokenizer
    SOURCE: str = "source"
    LICENSE: str = "license"
    N_TOKENS: str = "n_tokens"
