"""What a dataset and a checkpoint are, by name, so a run names one instead of repeating a path.

A path says where the bytes are and nothing else - not which tokenizer wrote a dataset, not which adapter
drives a checkpoint, not how wide either one's vocabulary is.
That is how a GLM run came to read Qwen's corpus, and how a 27B ladder run came to build its shape against a
4B snapshot: in both cases the two things differ in every way that matters and in nothing a config could see.

A card is the description, tracked here beside the code.

    from lsa import assets

    corpus = assets.tokenized("glm53_flash")
    model = assets.model("glm53_flash")

The cards live here rather than under `training/` or `evaluation/` because both read them - a training run
reads a tokenized dataset and the snapshot for its shape and its tokenizer, and the accuracy harness reads
the benchmarks and the same checkpoint through an adapter.

Every dataset a run reads has one [card](cards.py) of one of five kinds - a source, a tokenized dataset, token
shards, a benchmark or a workload - in `datasets/<kind>/`. A card holds only what is declared before anything is
prepared. What preparing measured lives in the dataset's own [manifest](../data/manifests.py) on disk beside the
data, with the card's declared fields copied in, and `card.manifest()` refuses one that has drifted.

A model card carries what the checkpoint does not say and what a run should not have to repeat.
It does not carry what the snapshot already states: whether the embedding is tied and whether the weights
are quantized are read off `config.json` at load, because a flag that has to be set by hand for every
checkpoint will be wrong for one of them.

Modelled on fairseq2's asset cards, and deliberately not on the rest of that machinery - there is no store
layering, no environment suffix and no dependency injection.
There is one directory per kind, a name is one card, and a tokenized dataset with none is named by `path`.

What a run falls back to when it names nothing, and what a card writes for a field it leaves to the
checkpoint, are [constants](constants.py).
"""

from lsa.assets.cards import (
    BenchmarkCard,
    BenchmarkLength,
    DatasetCard,
    ModelCard,
    ShardCard,
    SourceCard,
    TokenizedCard,
    WorkloadCard,
)
from lsa.assets.constants import Default, Kind, Licence, Preparer, Unpinned
from lsa.assets.loading import (
    benchmark,
    corpus,
    dataset,
    dataset_names,
    model,
    model_names,
    resolve,
    shards,
    source,
    tokenized,
    unregistered,
    workload,
)

__all__ = [
    "BenchmarkCard",
    "BenchmarkLength",
    "DatasetCard",
    "Default",
    "Kind",
    "Licence",
    "ModelCard",
    "Preparer",
    "ShardCard",
    "SourceCard",
    "TokenizedCard",
    "Unpinned",
    "WorkloadCard",
    "benchmark",
    "corpus",
    "dataset",
    "dataset_names",
    "model",
    "model_names",
    "resolve",
    "shards",
    "source",
    "tokenized",
    "unregistered",
    "workload",
]
