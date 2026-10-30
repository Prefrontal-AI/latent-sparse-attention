"""One yaml file, one dataclass tree, and the grid it can describe.

A run reproduces because its config does, so the file is the artifact and the dataclasses are its schema.
The same loader serves the accuracy harness and the trainer - a caller passes the root dataclass it wants
and gets back one instance per experiment.

The dataclasses a caller passes are `Config`, because a person sets them in a file.
What the code derives from a model rather than reads from a file is never a `Config` - `RotaryLayout` is the
shape a checkpoint's rotary section has, not a knob.

`defaults` turns one file into a delta on another, and it is Hydra's list rather than this module's.
A file names the configs it builds on and Hydra composes them, so a file that differs from another in a name
and a step count says that and no more.
A chain, a cycle and a missing parent are Hydra's to report; there is no second copy of those rules here.

`sweep` turns one file into a grid.
Every key is a dotted path into the config and every value a list, and the experiments are the product of
them, with the swept values appended to the name so two never write to one directory.

Command-line overrides use the same dotted paths and are applied last, so pinning a swept key on the command
line beats the sweep - and narrows it to that one value, so `run.optim.seed=1` on a three-seed file is the
one run named `_seed1` rather than three runs of seed 1 named for seeds 0, 1 and 2.
Precedence is the dataclass defaults, then the composed files, then the sweep, then the overrides.

`site` is where this machine's values live - its `root` and its `images` - in `configs/site/<site>.yaml`, read by [the site module](site.py).
It is composed into every file as the `site` node, so a config names an image as `${site.images.h200}` and a path under the root as `${root:models/maps/map.pt}`, and no path of the cluster is spelled in a run.
`site=local` on the command line picks another site for that run.
The node is for interpolation only - it is taken out before the schema is built, so no schema declares it.

Every config `load` returns remembers where it came from - the file, its point of the sweep and the overrides
- so the metadata a run writes beside its outputs can name the file that reproduces it, rather than only the
values the file resolved to. It is not a field: a file cannot set it, and `asdict` does not see it.
"""

import os
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from itertools import product
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, get_args, get_origin

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from lsa import io, paths, site

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _under_root(relative: str = "", *, _root_: Any) -> str:
    """Resolve `${root:...}` against the site composed into the config being resolved, or this process's site."""
    return paths.under_root(relative, _root_.get("site"))


# A path under the shared root, spelled `${root:models/maps/<file>.pt}`, so no config names the root itself
OmegaConf.register_new_resolver("root", _under_root, replace=True)


def _under_datasets(relative: str, *, _root_: Any) -> str:
    """Resolve `${datasets:...}` against the site composed into the config being resolved, or this process's site."""
    return paths.under_datasets(relative, _root_.get("site"))


# A path under the site's shared datasets directory, spelled `${datasets:fineweb-edu-100BT}`
OmegaConf.register_new_resolver("datasets", _under_datasets, replace=True)

# The attribute a loaded config keeps its `Source` under
SOURCE = "_lsa_source"


@dataclass(frozen=True)
class Source:
    """The file a run was read from, its point of the file's sweep, and the overrides applied on top.

    `path` is relative to the working directory when the file sits under it - a checkout's
    `configs/...` - and absolute otherwise.
    """

    path: str
    sweep: list[str]
    overrides: list[str]


def source_of(config: Any) -> dict[str, Any]:
    """Return where a loaded config came from, ready for a run's metadata.

    Args:
        config: A config `load` returned.

    Returns:
        The `Source` as a dict, or empty for a config built some other way.
    """
    found = getattr(config, SOURCE, None)
    return asdict(found) if isinstance(found, Source) else {}


def run_provenance(experiment: Any) -> dict[str, Any]:
    """Return what every run's metadata names first - the resolved config, the file it came from, the commit and the time.

    Args:
        experiment: A config `load` returned.

    Returns:
        `run`, `config`, `git` and `date`, for a run's own block to extend.
    """
    return {
        "run": asdict(experiment),
        "config": source_of(experiment),
        "git": io.git_state(Path(__file__).parent),
        "date": io.utc_now("auto"),
    }


def require_positive(config: Any, section: str, *names: str) -> None:
    """Reject a config whose named fields are not all above zero.

    Args:
        config: The dataclass being validated.
        section: Where the config sits in the yaml, which the message leads with.
        *names: The fields to check, in the order they are reported.

    Raises:
        ValueError: A field is zero or negative.
    """
    for name in names:
        value = getattr(config, name)
        if value <= 0:
            raise ValueError(f"{section}.{name} must be positive - got {value}")


def _named(path: Path) -> str:
    """Return a config file's path as a run's metadata spells it."""
    resolved = path.resolve()
    here = Path.cwd().resolve()
    return os.path.relpath(resolved, here) if resolved.is_relative_to(here) else str(resolved)


def load[Config: DataclassInstance](
    schema: type[Config], path: str | Path, overrides: list[str] | None = None, *, name_path: str = "name"
) -> list[Config]:
    """Read a config file and return the runs it describes.

    A file without a `sweep` block describes one.
    omegaconf reads the yaml and merges the layers; the dataclasses are what validate the result, so a bad
    value fails with the schema's own message rather than a second copy of its rules.

    Args:
        schema: The root dataclass.
        path: Path to the yaml.
        overrides: Dotted `key=value` assignments applied after everything else.
        name_path: Field the sweep suffixes, such as `run.name`.

    Returns:
        One `schema` per experiment, in sweep order.

    Raises:
        FileNotFoundError: There is no file at `path`.
        ValueError: The composition failed, a sweep key names no value, or the merged config fails its own
            validation.
    """
    given = list(overrides or [])
    site_name, overrides = _split_site(given)
    composed_site = OmegaConf.create({"site": site.load(site_name)})
    file_config = _read_file(Path(path))

    sweep = _take_sweep(file_config)
    # Defaults come from the dataclasses themselves, so there is one statement of what a field defaults to
    defaults = OmegaConf.create(declared_defaults(schema))
    override_config = OmegaConf.from_dotlist(overrides or [])
    # A swept key the command line pins is swept over that value alone, so the run is named for what it ran
    pinned = {assignment.split("=", 1)[0].strip() for assignment in overrides or []}
    sweep = {
        key: [OmegaConf.select(override_config, key)] if key in pinned else values
        for key, values in sweep.items()
    }

    configs = []
    for assignment in _sweep_assignments(sweep):
        merged = OmegaConf.merge(
            defaults, composed_site, file_config, OmegaConf.from_dotlist(assignment), override_config
        )
        values = _resolved(merged, path, site_name)
        if not isinstance(values, dict):
            raise ValueError(f"{path} must hold a mapping, not a list")
        if assignment:
            name = OmegaConf.select(merged, name_path)
            if not isinstance(name, str):
                raise ValueError(f"{path}: sweep name {name_path!r} must be a string")
            OmegaConf.update(merged, name_path, name + _name_suffix(assignment))
            values = _resolved(merged, path, site_name)
        built = construct(schema, values)
        object.__setattr__(built, SOURCE, Source(_named(Path(path)), list(assignment), given))
        configs.append(built)
    return configs


def _split_site(overrides: list[str] | None) -> tuple[str, list[str]]:
    """Take the `site=<name>` override out of a list of overrides.

    Args:
        overrides: Dotted `key=value` assignments.

    Returns:
        The site name, the one this process reads when none is given, and the remaining overrides.
    """
    name = site.current()
    rest = []
    for assignment in overrides or []:
        key, _, value = assignment.partition("=")
        if key.strip() == "site":
            name = value.strip()
        else:
            rest.append(assignment)
    return name, rest


def _resolved(merged: DictConfig | ListConfig, path: str | Path, site_name: str) -> Any:
    """Resolve every interpolation of a merged config and take the site node out.

    Args:
        merged: The layers merged, with the site node on it.
        path: The config file, for the message.
        site_name: The site composed, for the message.

    Returns:
        The values as a plain mapping, without `site`.

    Raises:
        ValueError: An interpolation names a key the site does not hold.
    """
    try:
        values = OmegaConf.to_container(merged, resolve=True)
    except InterpolationResolutionError as failure:
        raise ValueError(
            f"{path}: {failure} - set it in {site.SITES / (site_name + '.yaml')}, or pick another site with `site=<name>`"
        ) from failure
    if isinstance(values, dict):
        values.pop("site", None)
    return values


def declared_defaults(schema: Any) -> dict[str, Any]:
    """Return what a schema declares its fields default to, without building one.

    Read off the declarations rather than off a constructed instance, because a `__post_init__` that fills
    a field in - the way a model block takes its family from the asset it names - would otherwise bake that
    one asset's answers into the defaults layer, where they would survive a file naming a different model.
    A default is what the field says, and what an asset supplies is resolved once the file has been merged
    over it.

    Args:
        schema: A dataclass.

    Returns:
        Field name to declared default, recursing into nested dataclasses and leaving out a field that
        declares none.
    """
    values: dict[str, Any] = {}
    for declared_field in fields(schema):
        annotation = declared_field.type
        if isinstance(annotation, type) and is_dataclass(annotation):
            nested = declared_defaults(annotation)
            if declared_field.default_factory is not MISSING:
                nested.update(_factory_arguments(annotation, declared_field.default_factory))
            values[declared_field.name] = nested
        elif declared_field.default is not MISSING:
            values[declared_field.name] = declared_field.default
        elif declared_field.default_factory is not MISSING:
            values[declared_field.name] = declared_field.default_factory()
    return values


def _factory_arguments(schema: Any, factory: Any) -> dict[str, Any]:
    """Return the fields a default factory sets differently from the schema's own defaults.

    A block whose factory passes arguments - the indexer, which trains at a wider block than it serves at -
    means those arguments, and they have to survive into the defaults layer. A field the factory leaves
    alone reads the same either way and is left out, which is what keeps a value `__post_init__` derived
    from becoming a default.

    Args:
        schema: The nested dataclass.
        factory: Its field's default factory.

    Returns:
        Field name to value, for the fields the factory changed.
    """
    plain = asdict(schema())
    built = asdict(factory())
    return {name: value for name, value in built.items() if plain[name] != value}


def _read_file(path: Path) -> DictConfig:
    """Read one config file, composed with everything its `defaults` list names.

    Composition is Hydra's rather than this module's. A file says what it builds on in a `defaults` list and
    Hydra resolves it, which means a chain, a cycle and a missing parent are all its errors to report and
    there is no second implementation of the rules here to disagree with it.

    `_self_` decides where a file's own keys land in that list. Hydra puts them last when the list does not
    say, which is what a delta on a base config wants.

    Args:
        path: Path to the yaml.

    Returns:
        The composed config, with Hydra's own bookkeeping removed.

    Raises:
        FileNotFoundError: There is no file at `path`.
        ValueError: The composition failed, or the file holds a list rather than a mapping.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no config file at {path}")

    # Hydra keeps one global state, and a caller may load several files in a process
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
            composed = compose(config_name=path.stem)
    except Exception as failure:
        raise ValueError(f"{path}: {failure}") from failure
    finally:
        GlobalHydra.instance().clear()

    if not isinstance(composed, DictConfig):
        raise ValueError(f"{path} must hold a mapping, not a list")
    # Hydra composes in struct mode, and what comes back still has a `sweep` block to take out of it
    OmegaConf.set_struct(composed, False)
    return composed


def construct[Config: DataclassInstance](schema: type[Config], values: Any) -> Config:
    """Build one dataclass from a mapping, rejecting a field it does not have or one it needs and lacks.

    Recurses into nested dataclasses and into lists of them, so the whole tree is checked against the types
    it declares rather than against a schema written out a second time.

    A field that declares no default is one a file has to set - a run's output directory is one - and a file
    that leaves it out fails here with the field's name rather than as a constructor's missing argument.

    Args:
        schema: The dataclass to build.
        values: Field name to value, as the merged config holds them.

    Returns:
        An instance of `schema`, with its own `__post_init__` already run.

    Raises:
        ValueError: A key names no field of `schema`, or a value is not the type the field declares.
        ValueError: A field with no default is not set.
    """
    if not isinstance(values, dict):
        raise ValueError(f"{schema.__name__} needs a mapping - got {values!r}")

    declared = {declared_field.name: declared_field.type for declared_field in fields(schema)}
    unknown = sorted(set(values) - set(declared))
    if unknown:
        raise ValueError(f"{schema.__name__} has no field {unknown} - it has {sorted(declared)}")
    required = [
        declared_field.name
        for declared_field in fields(schema)
        if declared_field.default is MISSING and declared_field.default_factory is MISSING
    ]
    unset = [name for name in required if name not in values]
    if unset:
        raise ValueError(f"{schema.__name__} needs {unset} - it has no default, so the file has to set it")

    built = {}
    for name, value in values.items():
        built[name] = _field(name, value, declared[name])
    return schema(**built)


def _field(name: str, value: Any, annotation: Any) -> Any:
    """Build one field's value against the type it declares.

    Args:
        name: Field name, for the message.
        value: The value as the merged config holds it.
        annotation: The type the field declares.

    Returns:
        The value, with any nested dataclass built.
    """
    if isinstance(annotation, type) and is_dataclass(annotation):
        return construct(annotation, value)

    nested = _dataclass_element(annotation)
    if nested is not None and isinstance(value, list):
        return [construct(nested, element) for element in value]

    return _checked(name, value, annotation)


def _dataclass_element(annotation: Any) -> Any:
    """Return the dataclass a `list[...]` annotation holds, or None when it holds something else.

    `list[Thing] | None` is a union, so each member is tried in turn.

    Args:
        annotation: The type the field declares.

    Returns:
        The element dataclass, or None.
    """
    if isinstance(annotation, UnionType):
        for member in get_args(annotation):
            element = _dataclass_element(member)
            if element is not None:
                return element
        return None
    if get_origin(annotation) is not list:
        return None
    arguments = get_args(annotation)
    element = arguments[0] if arguments else None
    return element if isinstance(element, type) and is_dataclass(element) else None


def _checked(name: str, value: Any, annotation: Any) -> Any:
    """Validate scalars and list elements against their declared types.

    Args:
        name: Field name, for the message.
        value: The value as the merged config holds it.
        annotation: The type the field declares.

    Returns:
        The validated value, with nested dataclasses built.

    Raises:
        ValueError: A value or list element does not match its annotation.
    """
    if isinstance(annotation, UnionType):
        failures = []
        for member in get_args(annotation):
            try:
                return _field(name, value, member)
            except ValueError as failure:
                failures.append(failure)
        raise failures[0]
    if annotation is type(None) and value is not None:
        raise ValueError(f"{name} must be None - got {value!r}")
    if get_origin(annotation) is list:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list - got {value!r}")
        arguments = get_args(annotation)
        if not arguments:
            return value
        return [_field(f"{name}[{index}]", element, arguments[0]) for index, element in enumerate(value)]
    # `isinstance(True, int)` is true, so a yaml `true` would pass every numeric check
    # => a number never takes a bool
    if annotation in (int, float) and isinstance(value, bool):
        raise ValueError(f"{name} must be {annotation.__name__} - got the bool {value!r}")
    if annotation is float and isinstance(value, int):
        return value
    if annotation in (bool, int, float, str) and not isinstance(value, annotation):
        raise ValueError(f"{name} must be {annotation.__name__} - got {value!r}")
    return value


def _take_sweep(file_config: DictConfig) -> dict[str, list[Any]]:
    """Remove the sweep block from a loaded config and return it.

    Args:
        file_config: The config as loaded from the file, mutated in place.

    Returns:
        Dotted key to the list of values it takes, empty when the file has no sweep.

    Raises:
        ValueError: A sweep key holds something other than a nonempty list.
    """
    raw = file_config.pop("sweep", None)
    if raw is None:
        return {}
    container = OmegaConf.to_container(raw)
    if not isinstance(container, dict):
        raise ValueError("sweep must be a mapping of dotted key to a list of values")
    sweep: dict[str, list[Any]] = {}
    for key, values in container.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key!r} must be a nonempty list - got {values!r}")
        sweep[str(key)] = values
    return sweep


def _sweep_assignments(sweep: dict[str, list[Any]]) -> list[list[str]]:
    """Turn a sweep into one dotlist assignment per experiment.

    Args:
        sweep: Dotted key to the list of values it takes.

    Returns:
        One dotlist per experiment, in key order, and a single empty dotlist when there is no sweep.
    """
    if not sweep:
        return [[]]
    keys = list(sweep)
    return [
        [f"{key}={value}" for key, value in zip(keys, values, strict=True)]
        for values in product(*sweep.values())
    ]


def _name_suffix(assignment: list[str]) -> str:
    """Build the name suffix that keeps two experiments of one sweep apart.

    Args:
        assignment: Dotlist assignments for one experiment, as `indexer.bucket_size=32`.

    Returns:
        The suffix, as `_bucket_size32`, using the last segment of each dotted key.
    """
    parts = []
    for item in assignment:
        key, value = item.split("=", 1)
        parts.append(f"{key.rsplit('.', 1)[-1]}{value}")
    return "_" + "_".join(parts)
