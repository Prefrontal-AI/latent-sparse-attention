"""Model adapters - what it takes to run LSA on a checkpoint.

An adapter is two objects with two jobs.
[ModelAdapter](model.py) knows about checkpoints: how to load one, which of its layers run full attention,
and where their rotation sits. [SparseAttention](layer.py) knows about one attention module: its projections, its
rotation and its read. Everything between - the cache bookkeeping, the index, the selection - is model-free
and is written once.

| module | what it does |
|---|---|
| [layer](layer.py) | the generic per-layer module, and the four hooks a model may override |
| [model](model.py) | loading, layer discovery and the rotary measurement |
| [rotary](rotary.py) | where the rotation sits, measured from the model rather than declared |
| [qwen3_5](qwen3_5.py) | a packed output gate and a vision tower |
| [glm5_next](glm5_next.py) | MLA projections and a NoPE index space |

## Running a checkpoint that is already supported

Name it, and leave the rest to the measurement.

    model:
      name: Qwen/Qwen3.5-4B-Base

`auto` resolves the adapter from the checkpoint's `model_type` and falls back to the generic HF one, which
covers any standard decoder - `q_proj`, `k_proj`, `v_proj`, an output projection called `o_proj` or `dense`,
QK-norm under either name transformers gives it, and `layer_types` if the model is hybrid.

The one thing a model has to have is a **position-free sub-term**, which means partial rotary or none at all.
A model that rotates every head dim has nothing for LSA to cluster, and measuring the layout says so by name
before anything reads through it.
NoPE is the best case rather than a missing one: the whole head vector is position-free, and an adapter says
so by returning None from `rotary_embedding`.

## Running your own model

Subclass what differs and register it under the `model_type` its checkpoints declare.

    class MyAttention(SparseAttention):
        def project(self, hidden_states): ...

    class MyAdapter(ModelAdapter):
        attention_class = MyAttention

    register("my_model_type", MyAdapter)

A package installed alongside `lsa` skips even that - it advertises its adapters through the
`lsa.adapters` entry point group, whose names are the model types, and this repo is untouched.

    [project.entry-points."lsa.adapters"]
    my_model_type = "my_package.adapter:MyAdapter"

`project` is the usual one - a fused QKV, a packed gate, an MLA latent, a head layout that is not
`[B, T, H * dh]`.
`combine` is next, for an output gate or a projection neither name finds.
`read` is the one that means real work, and it is rarer than it looks: MLA is a `project` override, because
transformers expands the latent into per-head keys and values and caches those. What needs `read` is a model
whose attention consumes something the cache never expands - a pooled entry standing for several tokens,
which owns the mapping from a selected id back to what it covers.
"""

from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from lsa.assets import Unpinned
from lsa.evaluation.adapter.glm5_next import Glm5NextAdapter, Glm5NextSparseAttention
from lsa.evaluation.adapter.layer import Projection, SparseAttention, heads, rotary_function
from lsa.evaluation.adapter.model import ModelAdapter, torch_dtype
from lsa.evaluation.adapter.qwen3_5 import Qwen35Adapter, Qwen35SparseAttention
from lsa.evaluation.adapter.rotary import detect_rotary, resolve_rotary
from lsa.evaluation.config import ModelConfig
from lsa.evaluation.constants import Adapter

# model_type to adapter, filled by `register` at the bottom of this module and by installed packages
ADAPTERS: dict[str, ModelAdapter] = {}

# model_type to why its installed adapter would not load - raised when that model is asked for, not before
BROKEN: dict[str, Exception] = {}

__all__ = [
    "ADAPTERS",
    "BROKEN",
    "Glm5NextAdapter",
    "Glm5NextSparseAttention",
    "ModelAdapter",
    "Projection",
    "Qwen35Adapter",
    "Qwen35SparseAttention",
    "SparseAttention",
    "adapter_for",
    "detect_rotary",
    "heads",
    "model_type_of",
    "register",
    "register_installed",
    "resolve_rotary",
    "rotary_function",
    "torch_dtype",
]


def register(
    model_type: str, adapter: ModelAdapter | type[ModelAdapter], replace: bool = False
) -> ModelAdapter:
    """Register the adapter that drives a `model_type`.

    One line at the bottom of the module that defines it, and `auto` resolves to it from then on.
    A package installed alongside `lsa` advertises its adapters through the `lsa.adapters` entry point
    group instead, and needs no line here at all.

    Args:
        model_type: The `model_type` the checkpoints it drives declare.
        adapter: The adapter, or its class - a class is instantiated with no arguments.
        replace: Take over a `model_type` another adapter already drives.

    Returns:
        The registered adapter.

    Raises:
        ValueError: Another adapter already drives this `model_type` and `replace` is False.
    """
    registered = adapter() if isinstance(adapter, type) else adapter
    existing = ADAPTERS.get(model_type)
    if existing is not None and not replace and type(existing) is not type(registered):
        raise ValueError(
            f"{model_type!r} is already driven by {type(existing).__name__} - "
            f"pass replace=True for {type(registered).__name__} to take it over"
        )
    ADAPTERS[model_type] = registered
    return registered


def register_installed(entries: Iterable[EntryPoint] | None = None) -> list[str]:
    """Register the adapters that installed packages advertise.

    An entry point's name is the `model_type` and its value is the adapter, so a package that installs
    alongside `lsa` needs no import and no edit here:

        [project.entry-points."lsa.adapters"]
        my_model_type = "my_package.adapter:MyAdapter"

    An entry point that will not load is recorded in `BROKEN` rather than raised here. This runs at import,
    and one broken package would otherwise take the whole harness down - including the models it has nothing
    to do with. Asking for the model type it claimed raises, with the original failure as the cause.

    Args:
        entries: The entry points to register. None reads the installed ones.

    Returns:
        The model types that were registered, in the order they were read.
    """
    if entries is None:
        entries = entry_points(group=Adapter.ENTRY_POINT_GROUP)

    registered = []
    for entry in entries:
        try:
            advertised: Any = entry.load()
        except Exception as error:
            BROKEN[entry.name] = error
            continue
        register(entry.name, advertised)
        registered.append(entry.name)
    return registered


def model_type_of(name: str) -> str:
    """Read a checkpoint's architecture name without loading its weights.

    Args:
        name: Checkpoint name or path.

    Returns:
        The `model_type` its config declares, or the empty string if it declares none.
    """
    from transformers import AutoConfig

    return str(getattr(AutoConfig.from_pretrained(name), "model_type", ""))


def adapter_for(model_config: ModelConfig) -> ModelAdapter:
    """Return the adapter that runs this checkpoint.

    Args:
        model_config: The model model_config, which may name an adapter or leave it to the checkpoint.

    Returns:
        The adapter.

    Raises:
        ValueError: The model_config names an adapter that is not registered.
        ImportError: An installed package claims this model type but its adapter would not load.
    """
    if model_config.adapter != Unpinned.ADAPTER:
        _check_not_broken(model_config.adapter)
        if model_config.adapter not in ADAPTERS:
            raise ValueError(
                f"no adapter named {model_config.adapter!r} - registered: {', '.join(sorted(ADAPTERS))}"
            )
        return ADAPTERS[model_config.adapter]

    model_type = model_type_of(model_config.checkpoint)
    _check_not_broken(model_type)
    return ADAPTERS.get(model_type, ADAPTERS[Adapter.GENERIC])


def _check_not_broken(model_type: str) -> None:
    """Raise the failure an installed adapter for this model type was parked with.

    Args:
        model_type: The model type about to be resolved.

    Raises:
        ImportError: An installed package claims this model type but its adapter would not load.
    """
    error = BROKEN.get(model_type)
    if error is None:
        return
    raise ImportError(
        f"an installed {Adapter.ENTRY_POINT_GROUP} entry point claims {model_type!r} but would not load"
    ) from error


register(Adapter.GENERIC, ModelAdapter())
register("qwen3_5", Qwen35Adapter())
register("glm5_next", Glm5NextAdapter())
register_installed()
