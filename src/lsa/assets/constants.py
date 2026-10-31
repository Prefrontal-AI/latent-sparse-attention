"""The names a card is written in, and the sentinels it uses for what it does not pin.

Grouped so a call site reads as its own sentence:

    from lsa import assets

    corpus = assets.tokenized(assets.Default.DATASET)

A group is imported by name rather than behind one `const`, because [the package's own constants](../constants.py) own that name.
Where each kind of card lives on disk is [loading](loading.py)'s detail.
"""

from lsa import constants as const


class Default:
    """The cards a run falls back to when it names nothing."""

    # The tokenized dataset a training run reads when it names none
    DATASET: str = "qwen3_5"

    # The checkpoint a run builds when it names none
    MODEL: str = "glm53_flash"


class Kind:
    """What a dataset card describes, as its `kind:` says and as the directory it sits in.

    A benchmark is not a tokenized dataset even when rendered for one tokenizer: its rows are prompts and answers.
    """

    SOURCE: str = "source"
    TOKENIZED: str = "tokenized"
    SHARDS: str = "shards"
    BENCHMARK: str = "benchmark"
    WORKLOAD: str = "workload"
    ALL: tuple[str, ...] = (SOURCE, TOKENIZED, SHARDS, BENCHMARK, WORKLOAD)


class Licence:
    """What a source card's `licence` says when one SPDX id does not cover every document."""

    # Each document's own licence is a column of the manifest beside the source's shards
    PER_DOCUMENT: str = "per-document"
    # Made on the cluster and never published
    INTERNAL: str = "internal"

    # The per-document manifest, and the columns a reader takes from it
    MANIFEST: str = "manifest.parquet"
    DOCUMENT_COLUMN: str = "doc_id"
    LICENCE_COLUMN: str = "licence"


class Preparer:
    """Which generator writes a benchmark's files, as its card's `preparer` says.

    RULER is rendered by NVIDIA's generators over a grid, LongBench v2 is read from the hub and truncated to a context, and PaperTrail is copied from its repository's samples.
    """

    RULER: str = "ruler"
    LONGBENCH: str = "longbench"
    PAPERTRAIL: str = "papertrail"
    ALL: tuple[str, ...] = (RULER, LONGBENCH, PAPERTRAIL)


class Unpinned:
    """What a model card writes for a field it leaves to the checkpoint to answer.

    Two fields say `auto` and mean different things - the adapter is looked up, the rotary layout is measured.
    """

    # The harness resolves the adapter from the checkpoint's own `model_type`
    ADAPTER: str = "auto"

    # The harness measures where the rotation sits from the model
    ROTARY_END: str = const.Rotary.AUTO
