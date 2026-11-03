"""What an evaluation is, written as yaml - one schema per kind, each of three blocks.

An evaluation is one model read one way, asked one thing, run one way, and the file spelling those out is the
artifact - a run reproduces because its config does, not because someone wrote the command down.

Every kind has the same three blocks and differs in what they hold:

| block | what it says |
|---|---|
| `model` | the checkpoint, the attention path, the indexer and the knobs it is served with - and, for accuracy, how it is sampled |
| `data` | what is asked of it - the suite for accuracy, the contexts timed for kernels, the prompt ladder for production |
| `run` | what the run is called, where it writes and how its server is launched |

`AccuracyConfig`, `KernelsConfig` and `ProductionConfig` are the three schemas, and `SCHEMAS` names them by
kind. A config's tree under `configs/evaluation/` is its kind, which is what `load` reads it by.

The yaml, the sweep and the overrides are [config_loader](../config_loader.py); this file is the schema alone.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from lsa import assets, config_loader
from lsa import constants as const
from lsa.evaluation import gpu
from lsa.evaluation.constants import Attention, Evaluation, Kernels, Production, Suite
from lsa.sglang.launch import SglangConfig, with_rotation
from lsa.sglang.settings.design import DesignSettings, IndexerConfig, PisaConfig, ServingChoices


@dataclass
class ModelConfig:
    """Which checkpoint to load, and anything about it the harness should not work out for itself.

    Naming the checkpoint is enough for a supported architecture.
    `adapter` picks the code that drives it, and `auto` resolves that from the checkpoint's own `model_type`.
    The rotary layout is measured from the model at attach, so the two rotary fields exist to pin it against
    the measurement rather than to supply it - a disagreement raises instead of scoring.

    `layer_types` is what a hybrid model needs when it does not call its quadratic layers `full_attention`.
    There is no standard name - Qwen3.5 says `full_attention`, Qwen3.8-Flash-Next says `qwen_sparse_attention`
    and GLM-5.3-Flash says `deepseek_sparse_attention` - so a stack that matches nothing fails with the names
    it does declare.

    `rope_parameters` replaces the checkpoint's own rotation - YaRN past its native context - in every load the
    harness makes, and left out it is the card's, so a model distilled and served under one card sees one rotation.

    `name` is a [asset](../assets/__init__.py) when it holds no `/`, and a checkpoint id or a path when it
    does. An asset fills in whatever this block leaves unsaid, so two harnesses reading one checkpoint agree
    about its adapter and its rotary layout without either one repeating them. Empty names no model, which an
    experiment's block starts as so a config that never says which model it serves is refused rather than
    served with a default one.
    """

    name: str = assets.Default.MODEL
    dtype: str = "bfloat16"
    adapter: str = assets.Unpinned.ADAPTER
    rotary_end: str = const.Rotary.AUTO
    rotary_dim: int | None = None
    layer_types: list[str] | None = None
    rope_parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Take what the asset supplies, then reject a rotary declaration the index space has no meaning for.

        A field this block states wins over the asset's, because a run pinning a layout is asking for the
        disagreement to be caught rather than for the asset to be trusted.

        Raises:
            ValueError: `rotary_end` is neither `auto` nor one of `const.Rotary.ENDS`.
            ValueError: `rotary_dim` is not a nonnegative integer.
            ValueError: `rotary_dim` is given without an end for it to sit at.
            ValueError: `layer_types` is empty or holds something that is not a name.
        """
        asset = self.asset
        if asset is not None:
            self.adapter = self.adapter if self.adapter != assets.Unpinned.ADAPTER else asset.adapter
            self.layer_types = self.layer_types or list(asset.layer_types) or None
            if self.rope_parameters is None and asset.rope_parameters:
                self.rope_parameters = dict(asset.rope_parameters)
            pinned_here = self.rotary_end != const.Rotary.AUTO or self.rotary_dim is not None
            if not pinned_here and asset.rotary_end != const.Rotary.AUTO:
                self.rotary_end = asset.rotary_end
                self.rotary_dim = asset.rotary_dim

        if self.rotary_end not in const.Rotary.DECLARED:
            raise ValueError(f"rotary_end must be one of {const.Rotary.DECLARED} - got {self.rotary_end!r}")
        if self.rotary_dim is not None:
            if not isinstance(self.rotary_dim, int) or self.rotary_dim < 0:
                raise ValueError(f"rotary_dim must be a nonnegative integer - got {self.rotary_dim!r}")
            if self.rotary_end == const.Rotary.AUTO:
                raise ValueError(
                    "rotary_dim needs an end to sit at - name rotary_end too, or drop both and let "
                    "the layout be measured from the model"
                )
        if self.layer_types is not None:
            if not self.layer_types:
                raise ValueError("layer_types must name at least one layer type, or be left out entirely")
            if not all(isinstance(kind, str) for kind in self.layer_types):
                raise ValueError(f"layer_types must be names - got {self.layer_types!r}")

    @property
    def asset(self) -> assets.ModelCard | None:
        """Return the asset this names, or None when it names a checkpoint id, a path or nothing at all."""
        return None if not self.name or "/" in self.name else assets.model(self.name)

    @property
    def checkpoint(self) -> str:
        """Return what to hand a loader - the asset's snapshot, or the id this block already holds."""
        asset = self.asset
        return asset.checkpoint if asset is not None else self.name

    @property
    def repo(self) -> str:
        """Return the hub id this checkpoint is published under, or empty for a bare local path."""
        asset = self.asset
        if asset is not None:
            return asset.repo
        return "" if self.name.startswith("/") else self.name


@dataclass
class SamplingConfig:
    """What every request of one dataset sends, as written - the suite adds nothing to it.

    A field left None is not sent. `chat_template_kwargs` is where a template's switch for the think block
    goes. `think: false` is the reply starting past a closed think block, for a template with no switch, and
    `thinking_budget` is the server capping the block under strict thinking.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    think: bool = True

    def missing(self) -> list[str]:
        """Return the fields a request cannot be sent without."""
        needed = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens or None}
        return [name for name, value in needed.items() if value is None]

    def request(self) -> dict[str, Any]:
        """Return the settings as the suite's `Sampling` takes them."""
        settings = asdict(self)
        settings["chat_template_kwargs"] = settings["chat_template_kwargs"] or None
        return settings


@dataclass
class GenerationConfig:
    """How the model is sampled, one entry per dataset of the suite - `Suite.DATASETS`.

    It belongs to the model rather than the suite, because what "no reasoning" and "the recommended sampling"
    are depends on the checkpoint - so it sits in the yaml that names the model, beside it. Every RULER task
    is one dataset and takes the `ruler` entry, both GraphWalks tasks the `graphwalks` one and every PaperTrail
    task the `papertrail` one. An entry a run
    never draws from can stay empty.
    """

    mmlu_pro: SamplingConfig = field(default_factory=SamplingConfig)
    gpqa_diamond: SamplingConfig = field(default_factory=SamplingConfig)
    math500: SamplingConfig = field(default_factory=SamplingConfig)
    aime25: SamplingConfig = field(default_factory=SamplingConfig)
    longbench2: SamplingConfig = field(default_factory=SamplingConfig)
    graphwalks: SamplingConfig = field(default_factory=SamplingConfig)
    papertrail: SamplingConfig = field(default_factory=SamplingConfig)
    ruler: SamplingConfig = field(default_factory=SamplingConfig)

    def missing(self, tasks: list[str]) -> list[str]:
        """Return every field a run over these tasks still needs, as `dataset.field`.

        Args:
            tasks: The task names the run asks for, or empty for the whole suite.

        Returns:
            The unset fields of every dataset those tasks are drawn from.
        """
        used = {Suite.dataset_of(task) for task in tasks}
        return [
            f"{dataset}.{name}"
            for dataset, block in self.datasets().items()
            if not tasks or dataset in used
            for name in block.missing()
        ]

    def datasets(self) -> dict[str, SamplingConfig]:
        """Return the entries keyed by dataset name."""
        return {dataset: getattr(self, dataset) for dataset in Suite.DATASETS}

    def request(self) -> dict[str, object]:
        """Return the entries a request can be built from, as the suite's runner reads them."""
        return {dataset: block.request() for dataset, block in self.datasets().items() if not block.missing()}


@dataclass
class EvaluatedModelConfig(ModelConfig):
    """The checkpoint, the attention path it is read through, and the indexer and serving knobs LSA runs with.

    `attention` and `indexer` are one concept rather than two. `lsa` is what makes `indexer` mean anything, and
    under `native` it is dead config - so they live together, in the block that names the model they read.
    `pisa` is a second selector, served in LSA's place, and `pisa` holds its knobs the way `indexer` holds LSA's.

    `native` is the model's own attention, whatever it is. A checkpoint that ships sparse attention keeps it
    here, which is what makes the baseline the same model read a different way rather than a second model.
    `full`, `kda` and `gdn` are paths a kernel run times and no server runs, which the served schemas refuse.
    """

    attention: str = Attention.LSA
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    serving: ServingChoices = field(default_factory=ServingChoices)
    pisa: PisaConfig = field(default_factory=PisaConfig)

    def __post_init__(self) -> None:
        """Take what the asset supplies, then reject an attention path or an indexer the backend cannot run.

        The plugin cannot express every LSA config, and a `--dry-run` is where that should surface rather
        than twenty minutes into a server start.

        Raises:
            ValueError: `attention` is not one of `Attention.VARIANTS`.
            ValueError: The indexer and the serving choices are not a design the kernels can serve - see `DesignSettings`.
            ValueError: `pisa` is not a pyramid the search is defined for - see `PisaConfig`.
        """
        super().__post_init__()
        if self.attention not in Attention.VARIANTS:
            raise ValueError(f"model.attention must be one of {Attention.VARIANTS} - got {self.attention!r}")
        if self.attention == Attention.LSA:
            DesignSettings(self.indexer, self.serving)

    @property
    def recorded_attention(self) -> str:
        """Return the method a record files this run under - `lsa` or `pisa`, or what the checkpoint ships, empty when
        no card says."""
        if self.attention in (Attention.LSA, Attention.PISA):
            return self.attention
        asset = self.asset
        return asset.attention if asset is not None else ""

    @property
    def settings(self) -> dict[str, Any] | None:
        """Return the settings block's body the plugin is armed with, or None for the checkpoint's own indexer.

        Raises:
            ValueError: An LSA experiment names no `model.serving.projections`, and there is no index without a map.
        """
        if self.attention == Attention.PISA:
            return self.pisa.body()
        if self.attention != Attention.LSA:
            return None
        if not self.serving.projections:
            raise ValueError(
                "model.serving.projections must name a trained map - there is no index without one"
            )
        return DesignSettings(self.indexer, self.serving).body()


@dataclass
class AccuracyModelConfig(EvaluatedModelConfig):
    """A scored model - the evaluated model and how every request to it is sampled.

    Sampling belongs to the model rather than to the suite, because what "no reasoning" and "the recommended
    sampling" are depends on the checkpoint - so it sits in the block that names the model, beside it.
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)


def serve_argv(serve: Mapping[str, Any]) -> list[str]:
    """Turn a run's `serve` mapping into the server's command line.

    Each key is a server flag without its leading dashes. A scalar value gives `--flag value`, `true` the bare
    `--flag`, a list `--flag` followed by each item, and `null` or `false` no flag at all, which is how a
    config removes a flag a config before it set. The flags keep the order of the mapping.

    Args:
        serve: Flag name to value, as a config's `run.serve`.

    Returns:
        The server's arguments, in order.

    Raises:
        ValueError: A key starts with a dash, or a value is neither a scalar nor a non-empty list of scalars.
    """
    argv: list[str] = []
    for name, value in serve.items():
        if name.startswith("-"):
            raise ValueError(f"run.serve keys are flag names without the leading dashes - got {name!r}")
        if value is None or value is False:
            continue
        flag = f"--{name}"
        if value is True:
            argv.append(flag)
            continue
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        if not items or any(
            not isinstance(item, (str, int, float)) or isinstance(item, bool) for item in items
        ):
            raise ValueError(
                f"run.serve.{name} must be a scalar, true, false, null or a non-empty list of scalars - got {value!r}"
            )
        argv.extend([flag, *(str(item) for item in items)])
    return argv


@dataclass
class RunConfig:
    """What a run is called, where it writes, and how its server is launched.

    `output` is the records root, and the run id derived from the config picks the directory under it, so a
    launch and the record it produced cannot be filed apart. It is one root for the whole repo rather than one
    per experiment, because a records reader compares configurations by reading a root. It has no default - each
    tree's base names it, as `results` under the working directory or `${root:evaluation}` - and `run` and
    `submit` refuse a config that leaves it empty, the way they refuse one that names no model.

    `name` is what the console and a sweep point are called, the label its record is grouped by and where its
    run id starts. `id` names the run directory outright, and empty derives `<name>__<stamp>` closing on the
    UTC second the run started, so every launch is a run of its own and finishing an interrupted one means
    pinning its id. Two configs that share a name are told apart by the stamp alone, so each wants a name of
    its own.

    The launch fields are `SglangConfig`'s and default to its own, except `image`, the container, which every
    config names - a run's numbers belong to one build, so no default picks it for them. `tp` and `pp` are
    the mesh and the nodes it spans, `serve` the server's flags, `environment` what is exported to the
    server over the launcher's own variables. The checkpoint is not among them: it is
    `model.name`, said once.

    `serve` is the server's flags by name, so a config lays one flag over the configs before it, and sets a
    flag to `null` to drop what they gave. `serve_argv` says how a value becomes an argument.

    `hours` is the wall clock a submitted job may have, and 0 takes the ceiling its kind gives it. It is a
    ceiling rather than a reservation - the server is stopped when the run ends - so it is raised by a run
    slower than its kind usually is rather than tuned to what a run costs.

    `gpu` is the card the run is submitted for, as a token (`h200`) or a device name, and empty takes whatever
    the node has. The launch reads the device and refuses a different one, then writes the device name here
    and ends the run's name in its token. A pooled run's directory is named at submission, before any
    node is reached, so it must say which card its workers land on.

    `note` and `tag` are what the record is labelled with, the tag closing the run id, and `args` is passed
    verbatim to whatever the run starts - the suite, the harness or the client.
    """

    name: str = "unnamed"
    output: str = ""
    id: str = ""
    image: str = ""
    tp: int = SglangConfig.tp
    pp: int = SglangConfig.pp
    port: int = SglangConfig.port
    runner: str = SglangConfig.runner
    serve: dict[str, Any] = field(default_factory=dict)
    startup_minutes: int = SglangConfig.startup_minutes
    environment: dict[str, str] = field(default_factory=dict)
    hours: int = 0
    gpu: str = ""
    note: str = ""
    tag: str = ""
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject a wall clock no job could be given, and a `serve` value no flag takes.

        Raises:
            ValueError: `hours` is negative.
            ValueError: `serve` holds a value that is not a flag's.
        """
        serve_argv(self.serve)
        if self.hours < 0:
            raise ValueError(f"run.hours must be 0 for the kind's ceiling or positive - got {self.hours}")

    def launch_name(self) -> str:
        """Return the name the run launches under, the config's name with its card's token last.

        Returns:
            `<name>_<token>`, as `h200` or `b300`.

        Raises:
            ValueError: `gpu` is empty - a launch reads the device first, and a pooled run names it at submission.
        """
        if not self.gpu:
            raise ValueError(
                f"{self.name}: run.gpu is empty - the card is read from the device at launch, "
                "and a pooled run names it at submission, as run.gpu=h200"
            )
        return gpu.named(self.name, gpu.resolve(self.gpu))

    def server(self) -> SglangConfig:
        """Return the launch block the serving backend takes.

        Returns:
            The launch block, validated by `SglangConfig` itself.

        An empty image is not refused here, since loading a config on a site with no images is fine.
        A launch or a submission refuses it.
        """
        return SglangConfig(
            image=self.image,
            tp=self.tp,
            pp=self.pp,
            port=self.port,
            runner=self.runner,
            serve_args=serve_argv(self.serve),
            startup_minutes=self.startup_minutes,
            environment=dict(self.environment),
        )


@dataclass
class AccuracyRunConfig(RunConfig):
    """A scored run - the shared run block, and how the suite is driven.

    `python` is the interpreter that holds the suite, its tasks and its graders, and defaults to this one,
    because the suite is http and graders - no torch and no GPU - so the repo's own `evaluation` extra
    covers it. `concurrency_scale` scales the caps of the long-prompt tasks.
    `request_timeout` is the seconds one request may take - a reply that decodes past it is an error row, so a
    slow server needs it above the longest answer it will write.

    `pool` makes the job one worker of the run's pool - it claims only the requests no other worker holds and
    leaves when none is left, so several jobs given the same `id` drain one run. `lsa evaluate accuracy
    --workers` sets it; a run launched without it is the single job it always was.
    """

    python: str = ""
    concurrency_scale: float = 1.0
    request_timeout: float = 14400.0
    pool: bool = False

    def __post_init__(self) -> None:
        """Reject a run the suite runner would refuse.

        Raises:
            ValueError: `hours` is negative.
            ValueError: `concurrency_scale` or `request_timeout` is not positive.
        """
        super().__post_init__()
        config_loader.require_positive(self, "run", "concurrency_scale")
        config_loader.require_positive(self, "run", "request_timeout")


@dataclass
class AccuracyDataConfig:
    """What a scored run asks - one of the suite's named suites, or only some of it.

    `suite` is which of `Suite.NAMES` to run. `tasks` takes only those tasks of it and `lengths` only those
    RULER lengths, and either left empty takes all.
    `tasks` also takes `ruler`, a name for every RULER task, so `data.tasks=[ruler]` with `data.lengths` is the
    RULER cells of those lengths without a list of names - a config that pins other tasks is replaced by it.

    `root` is where the fixed RULER and LongBench files live. Empty leaves the suite its own default, and
    naming it here is what keeps a run off a directory somebody else writes to - the shared one was
    rewritten under a job once already, with fewer samples than the grid asks for.

    A corpus sized in another model's tokenizer names its own LongBench v2 file and pins. `longbench_context` is the
    limit that root's LongBench v2 file was prepared to, 0 for the card's own. `pins` is the set of prompt-set hashes
    its files are held to, `hashes_<pins>.json` beside the default corpus's `hashes.json` - empty for that one.
    """

    suite: str = "full"
    tasks: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    root: str = ""
    longbench_context: int = 0
    pins: str = ""

    def __post_init__(self) -> None:
        """Reject a suite the harness runner would refuse.

        Raises:
            ValueError: `suite` is not one the harness defines.
            ValueError: `tasks` or `lengths` names something `suite` does not offer, or together they match nothing.
        """
        if self.suite not in Suite.NAMES:
            raise ValueError(f"data.suite must be one of {Suite.NAMES} - got {self.suite!r}")
        from lsa.evaluation.accuracy.data import suites

        self.tasks = suites.expand_groups(self.tasks)
        try:
            suites.select(self.suite, self.tasks or None, self.lengths or None)
        except KeyError as error:
            raise ValueError(
                f"data.tasks and data.lengths do not fit data.suite {self.suite!r} - {error.args[0]}"
            ) from error


@dataclass
class KernelsDataConfig:
    """What one attention layer is timed over, inside the serving path rather than scored through it.

    The measurement itself is [the attention-layer harness](performance/kernels/measure/layer.py), which is run
    as a subprocess in the same container the server uses. It loads a checkpoint, prefills to a context,
    restores the same cache and index state before every trial and times the layer with CUDA events - so
    this block says which points to measure and the harness says what a point costs.

    `checkpoint` is the truncated cut of the model with real weights, built by
    `lsa.evaluation.performance.kernels.data.cut`. It is named here rather than taken from `model` because
    it is a different artifact: five layers, not the whole stack, and a full checkpoint would not fit beside a
    10M KV pool on one card. `prompt` is the text the harness prefills from.

    `phase` picks the half of serving to measure, and the two ask for different points. Prefill averages the
    last `window_chunks` chunks up to each context, which is where a coarse publication falls. Decode probes
    the tokens at `offsets` around each context, because maintenance is periodic and an average over
    arbitrary tokens hides it - what a token costs depends on which maintenance it triggers.

    `repeats` is trials per mode, and 0 takes the phase's default - a prefill chunk is tens of milliseconds
    and a decode token is tenths, so decode needs more of them to say anything.

    `prefill_backend` and `decode_backend` are the sparse attention kernels GLM's DSA backend reads a selection
    with - the attention after the selection, which neither the design nor the native indexer changes. The served
    meshes run `flashmla_sparse` and `fa3`. A dense or Qwen layer has its own backends and ignores them.

    `approximate_selection` declares a method whose selection differs from the exact top-k by design, as
    GLM's own fp8 indexer does. The record carries it, and a records reader then reads a differing selection beside
    exact output, state and scores as the method's nature rather than a failed check. A method that promises
    exact selection leaves it false.

    `contexts` is the ladder. Each is one harness process, and they are measured in the order given so an
    interrupted run keeps the short points it already has - once its id is pinned as `run.id`, since the
    derived one closes on the UTC second the run started.
    """

    phase: str = "prefill"
    checkpoint: str = ""
    prompt: str = ""
    contexts: list[int] = field(default_factory=lambda: [131072, 262144, 524288, 1048576, 2097152])
    layer: int = 3
    chunk: int = 8192
    window_chunks: int = 32
    offsets: list[int] = field(default_factory=list)
    warmup: int = 3
    repeats: int = 0
    prefill_backend: str = "tilelang"
    decode_backend: str = "tilelang"
    approximate_selection: bool = False

    def __post_init__(self) -> None:
        """Reject a measurement the harness would refuse, before a job is submitted for it.

        Raises:
            ValueError: `phase` is not one the harness measures.
            ValueError: `contexts` is empty, or holds a context that is not a positive multiple of `chunk`.
            ValueError: `warmup` or `repeats` is negative.
        """
        if self.phase not in Kernels.PHASES:
            raise ValueError(f"data.phase must be one of {Kernels.PHASES} - got {self.phase!r}")
        if not self.contexts:
            raise ValueError("data.contexts must name at least one context to measure")
        for context in self.contexts:
            if context <= self.chunk or context % self.chunk:
                raise ValueError(
                    f"every data.context must exceed chunk {self.chunk} and be a multiple of it - got {context}"
                )
        for name in ("warmup", "repeats"):
            if getattr(self, name) < 0:
                raise ValueError(f"data.{name} must not be negative - got {getattr(self, name)}")

    @property
    def trials(self) -> int:
        """Return the trials per mode, taking the phase's default when none was asked for."""
        return self.repeats or Kernels.REPEATS[self.phase]


@dataclass
class ProductionDataConfig:
    """The prompt ladder the whole served model is timed over, `concurrency` requests at a time.

    The client is sglang's own serving benchmark, run in the server's image against the server this run
    starts. Per length it sends `warmup` requests, flushes the cache, then `requests` more with at most
    `concurrency` in flight - prompt ids of exactly the length, `output_tokens` greedy tokens, streamed - and
    every request is kept. One in flight is the latency the paper's table reports; more is the throughput a
    node serves, and the server's running requests and decode graph batch must be raised to match.

    `workload` is the [card](../assets/datasets/workloads) of the prompt files - one per length, each with at
    least `requests` entries. Entry i at every length is the first tokens of the same text, so the ladder is
    nested.

    `lengths` is climbed in order and stops at the first failure, because a length that did not fit says
    nothing good about a longer one. A length whose samples already exist is kept - once `run.id` is pinned,
    since the derived one closes on the UTC second the run started - which is also how one run spans two
    deployments: the short lengths on one node, the long ones on two, each point naming its server.

    `profile_steps` above 0 asks the server's torch profiler for that many decode steps of each length, written under
    the run directory beside the samples - the timed requests are the ones profiled, so the run's own numbers carry
    the profiler's cost for those steps. The server's profiler records the first prefill chunks of a request too, which
    at a long length are the early, cheap ones.

    `profile_prefill_chunks` above 0 adds one more request per length, after the timed ones, whose prefill is profiled
    for that many chunks starting `profile_prefill_at` of the way through the prompt - a chunk late in the prompt is
    where a long context costs most. The run starts the profiler itself when the server's log shows that share of the
    chunks done, so the chunk it lands on is the one the server's log says it is, give or take one.
    """

    lengths: list[int] = field(default_factory=lambda: [131072, 262144, 524288, 1048576])
    workload: str = Production.WORKLOAD
    requests: int = 3
    concurrency: int = 1
    output_tokens: int = 256
    warmup: int = 1
    profile_steps: int = 0
    profile_prefill_chunks: int = 0
    profile_prefill_at: float = 0.9

    def __post_init__(self) -> None:
        """Reject a measurement the client would refuse or that could not say what a token costs.

        Raises:
            ValueError: `lengths` is empty or holds a length that is not positive.
            ValueError: `requests` is not positive, or `output_tokens` leaves no gap between two tokens.
            ValueError: `concurrency` is not positive or exceeds `requests`, so the cap it names never binds.
            ValueError: `requests` exceeds the workload's entries at one of `lengths` - the client sends one entry a
                request and stops when they run out.
            ValueError: `warmup`, `profile_steps` or `profile_prefill_chunks` is negative, or `profile_prefill_at` is not a share.
        """
        if not self.lengths or min(self.lengths) <= 0:
            raise ValueError(f"data.lengths must name at least one positive length - got {self.lengths}")
        config_loader.require_positive(self, "data", "requests")
        config_loader.require_positive(self, "data", "concurrency")
        if self.concurrency > self.requests:
            raise ValueError(
                f"data.concurrency must not exceed data.requests - got {self.concurrency} in flight "
                f"of {self.requests} requests"
            )
        card = assets.workload(self.workload)
        fewest = min(self.lengths, key=card.entries_at)
        if self.requests > card.entries_at(fewest):
            raise ValueError(
                f"data.requests must not exceed the {card.entries_at(fewest)} entries of workload {self.workload} "
                f"at {fewest} - got {self.requests}"
            )
        if self.output_tokens < 2:
            raise ValueError(
                f"data.output_tokens must be at least 2, since a token's cost is the gap after the first "
                f"- got {self.output_tokens}"
            )
        if self.warmup < 0:
            raise ValueError(f"data.warmup must not be negative - got {self.warmup}")
        if self.profile_steps < 0:
            raise ValueError(f"data.profile_steps must not be negative - got {self.profile_steps}")
        if self.profile_prefill_chunks < 0:
            raise ValueError(
                f"data.profile_prefill_chunks must not be negative - got {self.profile_prefill_chunks}"
            )
        if not 0 <= self.profile_prefill_at < 1:
            raise ValueError(f"data.profile_prefill_at must be in [0, 1) - got {self.profile_prefill_at}")


class EvaluationConfig(ABC):
    """What the three schemas share - the run's name, its server, and what its kind derives from them.

    Not a dataclass and not a block: each kind declares its own three blocks, and this is the behaviour
    reading `model` and `run` the same way whichever kind holds them.
    """

    kind: ClassVar[str]
    model: EvaluatedModelConfig
    run: RunConfig

    @property
    def name(self) -> str:
        """Return the run's name, which lives in the run block.

        `config_loader.load` names a sweep point by suffixing this, and every run it loads carries a name - so
        the block keeps the field and the schema keeps the convention.
        """
        return self.run.name

    @name.setter
    def name(self, value: str) -> None:
        """Rename the run, as a sweep does each of its points."""
        self.run.name = value

    def server(self) -> SglangConfig:
        """Return the launch block the serving backend takes, from the run's launch and the model's knobs.

        The model card's rotation reaches the server here, so a run never restates it in `serve`.
        """
        server = self.run.server()
        return replace(server, serve_args=with_rotation(server.serve_args, self.model.rope_parameters))

    @property
    @abstractmethod
    def hours(self) -> int:
        """Return the wall clock one job of this run may have - the run's own, or its kind's ceiling."""

    def refuse_unserved(self) -> None:
        """Reject an attention path only a kernel run can time, for a kind that starts a server.

        Raises:
            ValueError: `model.attention` is not one of `Attention.KINDS`.
        """
        if self.model.attention not in Attention.KINDS:
            raise ValueError(
                f"model.attention {self.model.attention!r} is a kernel measurement rather than a way to "
                f"serve a model - a served run takes one of {Attention.KINDS}"
            )


@dataclass
class AccuracyConfig(EvaluationConfig):
    """A served model scored by the accuracy suite.

    `model` starts naming nothing and `run.output` starts empty, and `run.require_model` and
    `run.require_output` refuse to run or submit a config that leaves either so.
    """

    kind: ClassVar[str] = Evaluation.ACCURACY
    model: AccuracyModelConfig = field(default_factory=lambda: AccuracyModelConfig(name=""))
    data: AccuracyDataConfig = field(default_factory=AccuracyDataConfig)
    run: AccuracyRunConfig = field(default_factory=AccuracyRunConfig)

    def __post_init__(self) -> None:
        """Reject a scored run whose path no server runs or whose launch the server would refuse.

        Raises:
            ValueError: The attention path is one only a kernel run can time.
            ValueError: The launch block is refused - see `SglangConfig`.
        """
        self.refuse_unserved()
        self.server()

    @property
    def hours(self) -> int:
        """Return the run's own wall clock, or the ceiling `Suite.HOURS` gives its suite."""
        return self.run.hours or Suite.HOURS[self.data.suite]


@dataclass
class KernelsConfig(EvaluationConfig):
    """One attention layer of a model, timed by the harness inside the serving container.

    Nothing is served: the harness loads the truncated checkpoint itself, in the image the server uses, so a
    kernel run and a served run measure the same build of the same kernels.
    """

    kind: ClassVar[str] = Evaluation.KERNELS
    model: EvaluatedModelConfig = field(default_factory=lambda: EvaluatedModelConfig(name=""))
    data: KernelsDataConfig = field(default_factory=KernelsDataConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        """Reject a launch the container runner would refuse.

        Raises:
            ValueError: The launch block is refused - see `SglangConfig`.
        """
        self.server()

    @property
    def hours(self) -> int:
        """Return the run's own wall clock, or the ceiling `Kernels.HOURS` gives its phase."""
        return self.run.hours or Kernels.HOURS[self.data.phase]


@dataclass
class ProductionConfig(EvaluationConfig):
    """A served model timed one request at a time - time to first token and time per output token."""

    kind: ClassVar[str] = Evaluation.PRODUCTION
    model: EvaluatedModelConfig = field(default_factory=lambda: EvaluatedModelConfig(name=""))
    data: ProductionDataConfig = field(default_factory=ProductionDataConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        """Reject a timed run whose path no server runs or whose launch the server would refuse.

        Raises:
            ValueError: The attention path is one only a kernel run can time.
            ValueError: The launch block is refused - see `SglangConfig`.
        """
        self.refuse_unserved()
        self.server()

    @property
    def hours(self) -> int:
        """Return the run's own wall clock, or the ceiling `Production.HOURS` gives every production run."""
        return self.run.hours or Production.HOURS


# Any one of the three, for the code that runs whichever kind it is handed
Experiment = AccuracyConfig | KernelsConfig | ProductionConfig

# The schema each kind is read through
SCHEMAS: dict[str, type[Experiment]] = {
    Evaluation.ACCURACY: AccuracyConfig,
    Evaluation.KERNELS: KernelsConfig,
    Evaluation.PRODUCTION: ProductionConfig,
}


def kind_of(path: str | Path) -> str:
    """Return the kind a config file is, from the tree it sits in - `configs/evaluation/<kind>/<model>/`.

    Args:
        path: Path to the yaml.

    Returns:
        The nearest enclosing directory named one of `Evaluation.KINDS`.

    Raises:
        ValueError: No enclosing directory names a kind.
    """
    for directory in Path(path).resolve().parents:
        if directory.name in Evaluation.KINDS:
            return directory.name
    raise ValueError(f"{path} sits in no {Evaluation.KINDS} tree, so its kind has to be named")


def load(path: str | Path, overrides: list[str] | None = None, kind: str = "") -> list[Experiment]:
    """Read an evaluation file through its kind's schema and return the runs it describes.

    Args:
        path: Path to the evaluation yaml.
        overrides: Dotted `key=value` assignments applied after everything else.
        kind: One of `Evaluation.KINDS`, or empty to take it from the tree the file sits in.

    Returns:
        One config of the kind's schema per run, in sweep order.

    Raises:
        ValueError: `kind` is not one of `Evaluation.KINDS`.
    """
    kind = kind or kind_of(path)
    if kind not in SCHEMAS:
        raise ValueError(f"kind must be one of {Evaluation.KINDS} - got {kind!r}")
    return [*config_loader.load(SCHEMAS[kind], path, overrides, name_path="run.name")]
