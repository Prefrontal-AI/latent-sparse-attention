"""What a distillation run is, written as yaml - what is trained, on what, and how.

Three blocks and nothing else. `model` is what is trained - the model the targets were extracted from, the
indexer the maps are read under and the learned space itself. `data` is what it reads - which targets, and
which of their documents train and score the maps. `run` is how - the name, where it writes, the objective,
the optimiser, the reports and the second copy of the numbers.
The yaml, the sweep and the overrides are [config_loader](../../config_loader.py); this file is the schema alone.

The indexer block is an `IndexerConfig`, the design's own, because the metrics are taken at a real read - `top_k` keys
a query - and reusing the design's settings is what keeps the two from drifting.
It carries `index_dim`, the dims the projections write.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsa import assets, config_loader, env
from lsa.core.hierarchy import BUCKET_SCORES
from lsa.core.reference import CLUSTER_ALGORITHMS
from lsa.sglang.settings.design import (
    KEY_AND_QUERY,
    QUERY_HEADS_ALL,
    QUERY_HEADS_RANK,
    DesignSettings,
    IndexerConfig,
    QSAConfig,
)
from lsa.training.distillation.constants import (
    Cache,
    Divergence,
    Draw,
    Online,
    Pool,
    Report,
    Rows,
    Served,
    Split,
    Tail,
    Target,
)


@dataclass
class DocumentSet:
    """One named group of documents the targets were extracted from.

    `source` is the dataset card the documents came from, and a directory under the targets root. `split` is
    `train` or `eval` and is a directory too, so a set that reads one side cannot reach the other however it
    is written.

    `sources` keeps only the domains it names, left out it keeps them all. That is what lets one dataset be
    reported per domain - one set per source - rather than as a single average over nine of them.
    `lengths` does the same for a mix's length buckets.

    `rows` is what a scoring set reads of each document - the held-out tail, or some of the training rows,
    which on a set from the training side is how well a map fits the rows it trained on. `limit` keeps the
    first that many documents, which is what keeps a set of training documents cheap to score.

    `root` reads the set from targets of its own instead of the run's. Held-out prompts extracted apart from
    what a run trains on - RULER at each length, one extraction a length - are scored beside it that way.
    """

    name: str = ""
    source: str = ""
    split: str = Split.TRAIN
    sources: list[str] | None = None
    lengths: list[int] | None = None
    rows: str = Rows.TAIL
    limit: int | None = None
    root: str | None = None

    def __post_init__(self) -> None:
        """Reject a set that names a side that does not exist.

        Raises:
            ValueError: `split` is not one of `Split.BOTH`.
            ValueError: `sources` is present but empty, or holds something that is not a name.
            ValueError: `lengths` is present but empty, or holds a length that is not positive.
            ValueError: `rows` is not one of `Rows.BOTH`, or `limit` is present and not positive.
        """
        if self.rows not in Rows.BOTH:
            raise ValueError(f"{self.name}.rows must be one of {Rows.BOTH} - got {self.rows!r}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"{self.name}.limit must be positive, or left out - got {self.limit}")
        if self.lengths is not None and (not self.lengths or min(self.lengths) <= 0):
            raise ValueError(
                f"{self.name}.lengths must be positive lengths, or left out - got {self.lengths!r}"
            )
        if self.split not in Split.BOTH:
            raise ValueError(
                f"{self.name or 'a document set'}.split must be one of {Split.BOTH} - got {self.split!r}"
            )
        if self.sources is not None:
            if not self.sources:
                raise ValueError(f"{self.name}.sources must name at least one, or be left out entirely")
            if not all(isinstance(source, str) and source for source in self.sources):
                raise ValueError(f"{self.name}.sources must be source names - got {self.sources!r}")


@dataclass
class TeacherSource:
    """The frozen teacher run inside the job in place of a cache - the online source of the targets.

    `extraction` is the yaml `lsa distill targets` would extract from, and `overrides` the dotted
    assignments it would be given. The run reads the documents, query positions and targets that extraction would
    have written, computed as the run goes and never written, so `data.root` is not read and a set's `source` names
    the extraction's collection.

    One copy of the model goes on every card the trainer leaves - seven on a node of eight - each reading one document
    at a time for every layer at once. `ahead_gb` is the host memory the documents read and not yet asked for may wait
    in. A job of several nodes puts copies on every card of the others too, and `staging_gb` is the room on the shared
    disk the documents they read may wait in for the trainer - see [teacher_nodes](data/teacher_nodes.py).
    """

    extraction: str = ""
    overrides: list[str] = field(default_factory=list)
    ahead_gb: float = Online.AHEAD_GB
    staging_gb: float = Online.STAGING_GB

    def __post_init__(self) -> None:
        """Reject a teacher that names no extraction, or a negative bound.

        Raises:
            ValueError: `extraction` is empty or `ahead_gb` or `staging_gb` is negative.
        """
        if not self.extraction:
            raise ValueError("data.teacher.extraction must name the extraction yaml the teacher reads")
        if self.ahead_gb < 0:
            raise ValueError(f"data.teacher.ahead_gb must not be negative - got {self.ahead_gb}")
        if self.staging_gb < 0:
            raise ValueError(f"data.teacher.staging_gb must not be negative - got {self.staging_gb}")


@dataclass
class DataConfig:
    """Where the targets are and which of them a run reads.

    A layer's share of the targets is some 3 TiB, so a run streams it - see [pool](data/pool.py).
    `pool_gb` is the host memory the documents waiting for the card may fill, and it is left out almost always:
    a job sized with `--mem-gb` says how much memory it has and the pool takes a share of it.
    `resident_gb` is the card memory the documents being trained on may occupy, which a run works out from what
    the card reports free when it is left out.
    `window` is how many documents' steps are mixed at once, and so how many the card holds - the steps of
    one document are spread over its window rather than taken in a row.

    `teacher` reads the targets off the frozen model as the run goes instead of off `root` - see `TeacherSource`.
    """

    root: str = Cache.ROOT
    teacher: TeacherSource | None = None
    layers: list[int] | None = None
    pool_gb: float | None = None
    resident_gb: float | None = None
    window: int = Pool.WINDOW
    train: list[DocumentSet] = field(default_factory=lambda: [DocumentSet(split=Split.TRAIN)])
    evaluate: list[DocumentSet] = field(default_factory=lambda: [DocumentSet(split=Split.EVAL)])

    def __post_init__(self) -> None:
        """Reject a selection that would train or score on nothing.

        Raises:
            ValueError: There are no training sets or no evaluation sets.
            ValueError: `layers` is present but empty.
            ValueError: `pool_gb` or `resident_gb` is present and negative, or `window` is not positive.
            ValueError: Two sets share a name, which would collapse their rows in a report.
        """
        if not self.train:
            raise ValueError("data.train must name at least one document set")
        if not self.evaluate:
            raise ValueError("data.evaluate must name at least one document set")
        if self.layers is not None and not self.layers:
            raise ValueError("data.layers must name at least one layer, or be left out entirely")
        for name in ("pool_gb", "resident_gb"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"data.{name} must not be negative - got {value}")
        config_loader.require_positive(self, "data", "window")
        names = [document_set.name for document_set in self.evaluate]
        if len(set(names)) != len(names):
            raise ValueError(f"two evaluation sets share a name - {sorted(names)}")


@dataclass
class SpaceConfig:
    """The learned space - how sharp the loss reads it, and the maps it starts from.

    Its dim is `model.indexer.index_dim`, the design's knob.

    `tau` left out is calibrated per layer on the untrained map before step one, which is what it is for:
    cosine scores over a whole document need a temperature near 0.05, and at 1.0 the softmax is flat enough
    that training destroys the projections rather than improving them.
    `init` names the maps a run starts from, and left out each layer starts at the leading PCA directions of its own `x`.
    """

    tau: float | None = None
    init: str | None = None

    def __post_init__(self) -> None:
        """Reject a space that cannot be built.

        Raises:
            ValueError: `tau` is present and not positive.
        """
        if self.tau is not None and self.tau <= 0:
            raise ValueError(f"space.tau must be positive - got {self.tau}")


@dataclass
class ObjectiveConfig:
    """What the loss measures the index score against.

    The defaults are the settled recipe - every extracted key of the teacher's target, under the forward KL -
    so a file that says nothing here trains what it trained before.

    `target` picks what the index score is measured against - the dense attention, or one of the targets
    read off the checkpoint's own sparse selection, which need a cache extracted with `queries.native`.
    `target_depth` truncates the dense ranking, so it applies to the targets read off it alone.
    `tail` says what the truncation does with the mass beyond the depth - drop it, or keep it as one `rest_bucket`
    entry against which the student pays for its mass outside the kept keys.

    `dense_full` is the dense attention over every readable key, computed each step from the key latent and the heads'
    queries in a cache written under the key and query.

    `closed_block` is the closed-block rule: a key is read, and carries target mass, only once the cluster block of that
    many keys it sits in has closed before the query's own, and a query row with no such block behind it is not trained
    on. An index-vector cache carries the rule itself, and the option lays it over any other cache's.
    It reads the cache's top-k ranking, so it applies to the targets read off that ranking alone.
    """

    target_depth: int | None = None
    divergence: str = Divergence.FORWARD
    target: str = Target.DENSE
    tail: str = Tail.TRUNCATE
    closed_block: int | None = None

    def __post_init__(self) -> None:
        """Reject an objective that names something the loss cannot compute.

        Raises:
            ValueError: `divergence` is not one of `Divergence.BOTH`.
            ValueError: `target_depth` is present and not positive.
            ValueError: `target` is not one of `Target.ALL`.
            ValueError: `target_depth` is present on a target that is not read off the dense ranking.
            ValueError: `tail` is not one of `Tail.BOTH`.
            ValueError: `tail` is the rest bucket on a target not read off the dense ranking, under another divergence
                than the forward KL, or without a `target_depth`.
            ValueError: `closed_block` is present and not positive, or on a target not read off the dense ranking.
        """
        if self.divergence not in Divergence.BOTH:
            raise ValueError(
                f"objective.divergence must be one of {Divergence.BOTH} - got {self.divergence!r}"
            )
        if self.target_depth is not None and self.target_depth <= 0:
            raise ValueError(
                f"objective.target_depth must be positive, or left out - got {self.target_depth}"
            )
        if self.target not in Target.ALL:
            raise ValueError(f"objective.target must be one of {Target.ALL} - got {self.target!r}")
        if self.closed_block is not None and self.closed_block <= 0:
            raise ValueError(
                f"objective.closed_block must be positive, or left out - got {self.closed_block}"
            )
        if self.closed_block is not None and self.target not in Target.RANKED:
            raise ValueError(
                f"objective.closed_block hides keys of the cached top-k ranking, which {self.target!r} is not read "
                f"off - wanted one of {Target.RANKED}"
            )
        if self.target_depth is not None and self.target not in Target.RANKED:
            raise ValueError(
                f"objective.target_depth truncates the dense ranking, which {self.target!r} is not read off"
            )
        if self.tail not in Tail.BOTH:
            raise ValueError(f"objective.tail must be one of {Tail.BOTH} - got {self.tail!r}")
        if self.tail != Tail.REST_BUCKET:
            return
        if self.target not in Target.RANKED:
            raise ValueError(
                f"objective.tail {self.tail!r} keeps the mass beyond the depth of the dense ranking, "
                f"which {self.target!r} is not read off - wanted one of {Target.RANKED}"
            )
        if self.divergence != Divergence.FORWARD:
            raise ValueError(
                f"objective.tail {self.tail!r} is the forward cross-entropy - got {self.divergence!r}"
            )
        if self.target_depth is None:
            raise ValueError(
                f"objective.tail {self.tail!r} needs an objective.target_depth to put the rest beyond"
            )


@dataclass
class OptimConfig:
    """How long the map trains and how fast it moves.

    The projections and the temperature move at different rates because they are different objects - an
    entry of a rotation is around `d^-1/2` and `log tau` is order one.

    `draw` is what a step's document is drawn in proportion to. Targets of one length are the same either
    way, and a mix balanced in tokens needs `token`, or its short buckets take most of the steps.
    """

    steps: int = 6000
    batch: int = 128
    lr: float = 1e-4
    tau_lr: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 0
    draw: str = Draw.DOCUMENT

    def __post_init__(self) -> None:
        """Reject an optimiser that would not train.

        Raises:
            ValueError: `steps` is negative - zero is the untrained report and is allowed.
            ValueError: `batch` is not positive.
            ValueError: A learning rate or the clip is not positive.
            ValueError: `draw` is not one of `Draw.BOTH`.
        """
        if self.draw not in Draw.BOTH:
            raise ValueError(f"optim.draw must be one of {Draw.BOTH} - got {self.draw!r}")
        if self.steps < 0:
            raise ValueError(f"optim.steps must not be negative - got {self.steps}")
        config_loader.require_positive(self, "optim", "batch", "lr", "tau_lr", "grad_clip")


@dataclass
class ServedDesign:
    """One design whose served read a report scores - its `model.indexer` and `model.serving` blocks.

    The blocks are deltas on the design's defaults, as the design's accuracy config writes them, not on the
    run's indexer: a run trains at the teacher's budget and a server reads at the design's.
    The visibility rule and the index width are the run's, because the cache and the maps fix them, and so is the input -
    `indexer.input` and `query_heads` - unless the design names its own.

    `lsa distill served-read` alone scores a design that reads its own map, `serving.projections`, or the key and the
    heads' queries, `indexer.input: key_and_query` - the latter from the cached `x` through the layer's own
    weights, each of `ranks` tensor-parallel ranks summing its own heads and selecting for them, as a server does.
    """

    name: str = "flat"
    indexer: dict[str, Any] = field(default_factory=dict)
    serving: dict[str, Any] = field(default_factory=dict)
    # Tensor-parallel ranks the heads split over under the key and query input, each selecting for its own
    ranks: int = Report.SERVED_RANKS

    def __post_init__(self) -> None:
        """Reject a design whose heads split over no rank.

        Raises:
            ValueError: `ranks` is not positive.
        """
        config_loader.require_positive(self, f"report.served design {self.name!r}", "ranks")

    def settings(self, run_indexer: IndexerConfig) -> DesignSettings:
        """Build the design as served over the run's cache and maps.

        Args:
            run_indexer: The run's `model.indexer`.

        Returns:
            The design's settings, its window, sinks and index width the run's, and its input the run's unless it names
            its own.

        Raises:
            ValueError: The design names a window, sinks or index width other than the run's, or a knob the
                design refuses.
        """
        fixed = {name: getattr(run_indexer, name) for name in Report.SERVED_FIXED}
        clashing = {name: value for name, value in self.indexer.items() if fixed.get(name, value) != value}
        if clashing:
            raise ValueError(
                f"run.report.served design {self.name!r} sets {clashing} - those are the run's, {fixed}, "
                "because the cache and the maps fix them"
            )
        inherited = {name: getattr(run_indexer, name) for name in Report.SERVED_INHERITED}
        indexer = {**inherited, **self.indexer, **fixed}
        # Under the key and query, ranks that each read their own list each sum their own heads
        if self.ranks > 1 and indexer.get("input") == KEY_AND_QUERY and indexer.get("query_heads") is None:
            indexer["query_heads"] = QUERY_HEADS_RANK
        return DesignSettings.from_blocks(indexer, self.serving)


@dataclass
class ArmConfig:
    """One offline arm `lsa distill served-read` scores beside the served designs - a reference read or PISA's selection.

    A `reference` arm builds the buckets and the levels of a served design `design` with `lsa.core`, at that design's
    sizes and over its index keys, with the research options below. A `pisa` arm selects blocks the way
    [PISA](../../baselines/pisa.py) does, over the model's own keys and the heads' queries when its `space` is
    `attention` - GLM's key latent, or each key-value head's rotated keys from a cache extracted under
    `queries.attention`, one selection a key-value head as the paper makes it - or over the index keys and index queries
    of `design` as one head when it is `index`. The model's own space reads at the layer's softmax scale unless
    `logit_scale` names another.
    Every arm's read is its long-range keys, then the window and the sinks, so the arms are scored as the designs are.

    `tau` is the temperature a LogSumExp score and PISA's index-space logits are read at, 1 / `tau` the logit scale.
    Left out it is the map's own, from its sidecar (`lsa distill calibrate-tau` records one for a map that has none).
    `tau_scale` reads the arm once at each multiple of that, and `taus` once at each temperature, as arms named
    `<name>@tau<scale>` and `<name>@tau=<tau>`.
    An arm that names `agrees_with` also scores how much of that design's long-range read it holds.
    """

    name: str
    kind: str = Served.ARM_REFERENCE
    # The served design whose sizes and space a reference arm reads, and whose space a PISA arm reads in the index space
    design: str | None = None
    agrees_with: str | None = None
    tau: float | None = None
    tau_scale: list[float] = field(default_factory=list)
    taus: list[float] = field(default_factory=list)
    # A reference arm's research options
    cluster_algorithm: str = "ward"
    node_cluster_algorithm: str = "ward"
    node_bucket_score: str = "centroid"
    bucket_score: str = "centroid"
    # A PISA arm's shape - the paper's C, K and g, the blocks kept above the leaves, the logit scale, and the forced blocks
    space: str = Served.SPACE_ATTENTION
    block_size: int = 64
    keep_count: int = 64
    branching: int = 2
    beam_width: int | None = None
    logit_scale: float | None = None
    forced_blocks: bool = True

    def __post_init__(self) -> None:
        """Reject an arm that names nothing it can read.

        Raises:
            ValueError: The kind, a space, an algorithm or a score is unknown, a reference arm or a PISA arm in the
                index space names no design, or a size or the temperature is not positive.
        """
        section = f"report.served arm {self.name!r}"
        if self.kind not in Served.ARM_KINDS:
            raise ValueError(f"{section} kind must be one of {Served.ARM_KINDS} - got {self.kind!r}")
        if self.space not in Served.ARM_SPACES:
            raise ValueError(f"{section} space must be one of {Served.ARM_SPACES} - got {self.space!r}")
        for name in ("cluster_algorithm", "node_cluster_algorithm"):
            if getattr(self, name) not in CLUSTER_ALGORITHMS:
                raise ValueError(
                    f"{section} {name} must be one of {CLUSTER_ALGORITHMS} - got {getattr(self, name)!r}"
                )
        for name in ("node_bucket_score", "bucket_score"):
            if getattr(self, name) not in BUCKET_SCORES:
                raise ValueError(
                    f"{section} {name} must be one of {BUCKET_SCORES} - got {getattr(self, name)!r}"
                )
        if self.design is None and (self.kind == Served.ARM_REFERENCE or self.space == Served.SPACE_INDEX):
            raise ValueError(f"{section} reads a design's sizes or space and names none - set design")
        config_loader.require_positive(self, section, "block_size", "keep_count", "branching")
        for name in ("tau", "logit_scale", "beam_width"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"{section} {name} must be positive - got {getattr(self, name)}")
        if any(value <= 0 for value in [*self.tau_scale, *self.taus]):
            raise ValueError(
                f"{section} tau_scale and taus must be positive - got {self.tau_scale}, {self.taus}"
            )
        if self.tau is not None and (self.tau_scale or self.taus):
            raise ValueError(f"{section} sets tau and sweeps one - set tau, or tau_scale or taus")


@dataclass
class ServedConfig:
    """The served read a report scores beside the exact one, on a GPU only.

    Each design's bucket cache is built over a whole held-out document and read at `rows` of its scored rows,
    spread over them - about a second a 1M document and layer on an H200, so a design costs a report about as
    much as reading the documents does. `designs` empty scores none.
    """

    designs: list[ServedDesign] = field(default_factory=lambda: [ServedDesign()])
    rows: int = Report.SERVED_ROWS
    # Keep each scored row's position and metrics in a `lsa distill served-read` record, to split the rows afterwards
    per_row: bool = False
    # Offline arms read beside the designs, by `lsa distill served-read` - see `ArmConfig`
    arms: list[ArmConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject a served read that reads nothing, two designs or arms of one name, or an arm that names a design that is not there.

        Raises:
            ValueError: `rows` is not positive, two designs or arms share a name, or an arm names an unknown design.
        """
        config_loader.require_positive(self, "report.served", "rows")
        names = [design.name for design in self.designs]
        if len(set(names)) != len(names):
            raise ValueError(f"two served designs share a name - {sorted(names)}")
        arm_names = [arm.name for arm in self.arms]
        if len(set(arm_names)) != len(arm_names) or set(arm_names) & set(names):
            raise ValueError(f"two served reads share a name - {sorted(arm_names)} beside {sorted(names)}")
        for arm in self.arms:
            unknown = {name for name in (arm.design, arm.agrees_with) if name is not None} - set(names)
            if unknown:
                raise ValueError(
                    f"report.served arm {arm.name!r} names {sorted(unknown)}, which are not served designs"
                )

    def settings(self, run_indexer: IndexerConfig) -> dict[str, DesignSettings]:
        """Build every design as served over the run's cache and maps, by name."""
        return {design.name: design.settings(run_indexer) for design in self.designs}


@dataclass
class ReportConfig:
    """How often a report is taken and a line printed, and the served reads it scores.

    `depths` are extra exact reads, one per K, that a report scores beside the run's own `top_k`: the space's top-K
    against the teacher's stored list, with the mean of each layer kept apart. None is the usual report.
    """

    every: int = 2000
    depths: list[int] = field(default_factory=list)
    # Steps between the printed training lines, which is not what wandb is sent - `wandb.log_every` is
    log_every: int = Report.LOG_EVERY
    served: ServedConfig = field(default_factory=ServedConfig)

    def __post_init__(self) -> None:
        """Reject a report that measures nothing.

        Raises:
            ValueError: `every` or `log_every` is not positive, or a depth is not.
        """
        config_loader.require_positive(self, "report", "every", "log_every")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError(f"report.depths must be positive - got {self.depths}")


@dataclass
class WandbConfig:
    """Where the scalars go, for the runs that send them anywhere.

    A run writes its reports to json whatever this says. This is the second copy, and it exists for the
    sweeps - a dim ladder is several runs whose point is the comparison between them.

    `prefix` names the family rather than the run: wandb picks its own two-word name, and this goes in front
    of it, so a ladder reads `ladder-fluent-sunset` and `ladder-brisk-meadow` and sorts together.

    `log_every` is this module's own cadence and is tied to neither of the other two. `report.every` is how
    often an evaluation is measured at all, `report.log_every` is how often a line is printed, and a chart
    wants points far tighter than a terminal wants lines. An evaluation goes up whenever one is measured,
    because there is no second copy of a number that was never taken.
    """

    enabled: bool = False
    project: str = env.WANDB_PROJECT
    # The team or user the run belongs to, or None for whatever the environment already logged in as
    entity: str | None = None
    prefix: str = ""
    tags: list[str] = field(default_factory=list)
    # Steps between the training scalars - the loss, the temperature and the gradient norm
    log_every: int = Report.WANDB_LOG_EVERY

    def __post_init__(self) -> None:
        """Reject a cadence that would log nothing.

        Raises:
            ValueError: `log_every` is not positive.
        """
        config_loader.require_positive(self, "wandb", "log_every")


@dataclass
class ModelConfig:
    """What is trained - the maps of one model, read under one design's indexer, in one learned space.

    `name` is the model card the targets were extracted from, and a run refuses targets that say they came
    from another model: a map is only valid on the weights whose states trained it, and Qwen3.5-4B and 9B
    share their layers and their index dim, so nothing about the tensors would catch the pairing.

    `indexer.input` is what the projections read, one setting for the run and the server: the design's own,
    `key_and_query` - GLM's key latent and every head's query summed, `query_heads: all`, which the layer's own weights
    read off the cached `x` once a document is on the card - or `x` when a run says so.

    `qsa` trains Qwen's own sparse-attention indexer in place of an LSA map, the baseline the maps are measured
    against, on the same documents, rows and steps. The indexer block still says which keys the extraction samples
    queries past, and the QSA indexer reads `x` whatever its `input` says.
    """

    name: str = assets.Default.MODEL
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    space: SpaceConfig = field(default_factory=SpaceConfig)
    qsa: QSAConfig | None = None


@dataclass(kw_only=True)
class RunConfig:
    """How the maps are trained - the name, where the run writes, and everything that is not the model or the data.

    `output` has no default. A run writes its maps and its reports there, and a config that does not say
    where is refused at load rather than writing somewhere nobody chose. A model's runs set it once in that
    model's `base.yaml`, as `${root:training/...}` for the shared root or a relative local path.
    """

    output: str
    name: str = "unnamed"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(kw_only=True)
class DistillConfig:
    """One distillation run - what is trained, on what, and how."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig

    def __post_init__(self) -> None:
        """Reject a served design the run's cache and maps cannot be read under.

        Raises:
            ValueError: A served design is not one the run's indexer can serve.
            ValueError: The indexer reads the key and query summed over a rank's heads - a run trains one index
                query a row, every head summed, `query_heads: all`.
            ValueError: The objective lays a closed block over a run that does not read the key and query.
            ValueError: A target the step computes from the key latent and the heads' queries is asked of a run that
                does not read the key and query.
            ValueError: An online run asks for the checkpoint's own picks, or a set reads a cache of its own.
            ValueError: A QSA indexer is trained off a cache, on another target or divergence than the dense forward
                KL, under a closed block, or with a served read of an LSA design.
        """
        if self.model.qsa is not None:
            self._check_qsa()
        indexer = self.model.indexer
        if indexer.input == KEY_AND_QUERY and indexer.query_heads != QUERY_HEADS_ALL:
            raise ValueError(
                f"model.indexer.query_heads {indexer.query_heads!r} sums a tensor-parallel rank's heads, and the teacher's "
                f"mass is every head's - a run on {KEY_AND_QUERY!r} trains query_heads: {QUERY_HEADS_ALL}"
            )
        if self.run.objective.closed_block is not None and indexer.input != KEY_AND_QUERY:
            raise ValueError(f"objective.closed_block is read on the key and query input, {KEY_AND_QUERY!r}")
        if self.run.objective.target in Target.STEP_TIME and indexer.input != KEY_AND_QUERY:
            raise ValueError(
                f"objective.target {self.run.objective.target!r} computes the teacher's logits from the key latent "
                f"and the heads' queries, and the run reads model.indexer.input {indexer.input!r} - "
                f"it needs {KEY_AND_QUERY!r}"
            )
        if self.data.teacher is not None:
            if self.run.objective.target in Target.PICKED:
                raise ValueError(
                    f"objective.target {self.run.objective.target!r} reads a checkpoint's own picks, which the "
                    "online teacher does not keep"
                )
            rooted = [one.name for one in [*self.data.train, *self.data.evaluate] if one.root]
            if rooted:
                raise ValueError(f"sets {rooted} name a root of their own, and an online run reads no cache")
        self.run.report.served.settings(indexer)

    def _check_qsa(self) -> None:
        """Refuse what a QSA indexer is not trained under.

        Raises:
            ValueError: The run reads a cache, reads another target, tail or divergence than the dense truncated forward KL,
                lays a closed block over the targets, or scores an LSA design's served read.
        """
        objective = self.run.objective
        if self.data.teacher is None:
            raise ValueError("model.qsa trains on the online teacher's blocks - name data.teacher")
        if (
            objective.target != Target.DENSE
            or objective.divergence != Divergence.FORWARD
            or objective.tail != Tail.TRUNCATE
        ):
            raise ValueError(
                f"model.qsa trains QSA's own objective, the forward KL to the dense target - got {objective.target!r} "
                f"under {objective.divergence!r} with tail {objective.tail!r}"
            )
        if objective.closed_block is not None:
            raise ValueError(
                "model.qsa selects closed blocks of its own - objective.closed_block is an LSA rule"
            )
        if self.run.report.served.designs:
            raise ValueError("model.qsa reads its own selection - set run.report.served.designs to []")

    @property
    def out(self) -> Path:
        """Return the directory this run writes into."""
        return Path(self.run.output) / self.run.name


def load(path: str | Path, overrides: list[str] | None = None) -> list[DistillConfig]:
    """Read a distillation file and return the runs it describes.

    Args:
        path: Path to the training yaml.
        overrides: Dotted `key=value` assignments applied after everything else.

    Returns:
        One `DistillConfig` per run, in sweep order.
    """
    return config_loader.load(DistillConfig, path, overrides, name_path="run.name")
