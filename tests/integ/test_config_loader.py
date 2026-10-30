"""The loader - what a file may say, what it inherits, and what counts as a default.

The `defaults` list and the dataclass defaults are the two rules that decide what a run actually ran at,
and neither is visible in the file that names it. Both get a test.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from lsa import config_loader, paths, site


@dataclass
class Inner:
    width: int = 8
    label: str = ""

    def __post_init__(self):
        # Stands in for a block that takes what it did not say from the asset it names
        self.label = self.label or f"inner{self.width}"


@dataclass
class Outer:
    name: str = "run"
    plain: Inner = field(default_factory=Inner)
    widened: Inner = field(default_factory=lambda: Inner(width=64))


@dataclass
class Numbers:
    count: int = 1
    rate: float = 0.5
    enabled: bool = False


@dataclass
class Collections:
    args: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    rates: list[list[float]] = field(default_factory=list)
    optional: list[str] | None = None
    records: list[Inner] | None = None


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_a_derived_value_is_not_a_default():
    """Otherwise one block's derivation survives a file that names something else entirely"""
    assert config_loader.declared_defaults(Outer)["plain"]["label"] == ""


def test_a_factory_argument_is_a_default():
    """The indexer trains at a wider block than it serves at, and the factory is where that is said"""
    assert config_loader.declared_defaults(Outer)["widened"]["width"] == 64


def test_a_file_derives_its_own_value_rather_than_inheriting_one(tmp_path):
    built = config_loader.load(Outer, write(tmp_path, "run.yaml", "plain:\n  width: 32\n"))[0]
    assert built.plain.label == "inner32"


def test_a_defaults_list_merges_the_parent_under_the_child(tmp_path):
    write(tmp_path, "base.yaml", "name: base\nplain:\n  width: 32\nwidened:\n  width: 128\n")
    child = write(
        tmp_path, "child.yaml", "defaults:\n  - base\n  - _self_\nname: child\nwidened:\n  width: 256\n"
    )
    built = config_loader.load(Outer, child)[0]
    assert built.name == "child"
    assert built.plain.width == 32
    assert built.widened.width == 256


def test_a_defaults_list_follows_a_chain(tmp_path):
    write(tmp_path, "a.yaml", "name: a\nplain:\n  width: 16\n")
    write(tmp_path, "b.yaml", "defaults:\n  - a\n  - _self_\nwidened:\n  width: 32\n")
    child = write(tmp_path, "c.yaml", "defaults:\n  - b\n  - _self_\nname: c\n")
    built = config_loader.load(Outer, child)[0]
    assert (built.name, built.plain.width, built.widened.width) == ("c", 16, 32)


def test_a_cycle_is_refused_rather_than_followed(tmp_path):
    """Hydra's error rather than this module's, which is the point of composing through it"""
    write(tmp_path, "a.yaml", "defaults:\n  - b\n  - _self_\nname: a\n")
    write(tmp_path, "b.yaml", "defaults:\n  - a\n  - _self_\nname: b\n")
    with pytest.raises(ValueError):
        config_loader.load(Outer, tmp_path / "a.yaml")


def test_a_missing_parent_names_itself(tmp_path):
    child = write(tmp_path, "child.yaml", "defaults:\n  - nothing\n  - _self_\n")
    with pytest.raises(ValueError, match="nothing"):
        config_loader.load(Outer, child)


def test_an_override_beats_the_file_and_its_parents(tmp_path):
    write(tmp_path, "base.yaml", "plain:\n  width: 32\n")
    child = write(tmp_path, "child.yaml", "defaults:\n  - base\n  - _self_\n")
    built = config_loader.load(Outer, child, ["plain.width=99"])[0]
    assert built.plain.width == 99


def test_a_sweep_takes_the_product_and_names_each_run(tmp_path):
    path = write(
        tmp_path,
        "grid.yaml",
        "name: grid\nsweep:\n  plain.width: [16, 32]\n  widened.width: [64, 128]\n",
    )
    runs = config_loader.load(Outer, path)
    assert [(run.plain.width, run.widened.width) for run in runs] == [
        (16, 64),
        (16, 128),
        (32, 64),
        (32, 128),
    ]
    assert [run.name for run in runs] == [
        "grid_width16_width64",
        "grid_width16_width128",
        "grid_width32_width64",
        "grid_width32_width128",
    ]


def test_a_sweep_can_name_a_nested_run(tmp_path):
    path = write(tmp_path, "grid.yaml", "plain: {label: nested}\nsweep:\n  widened.width: [64, 128]\n")
    runs = config_loader.load(Outer, path, name_path="plain.label")
    assert [run.plain.label for run in runs] == ["nested_width64", "nested_width128"]


def test_an_override_narrows_the_sweep_to_the_one_run_it_names(tmp_path):
    """Two runs of width 96 named for 16 and 32 would write a run under the name of one it is not"""
    path = write(tmp_path, "grid.yaml", "sweep:\n  plain.width: [16, 32]\n  widened.width: [64, 128]\n")
    runs = config_loader.load(Outer, path, ["plain.width=96"])
    assert [(run.plain.width, run.widened.width) for run in runs] == [(96, 64), (96, 128)]
    assert [run.plain.label for run in runs] == ["inner96", "inner96"]
    assert [run.name for run in runs] == ["run_width96_width64", "run_width96_width128"]


@pytest.mark.parametrize("values", ["16", "[]"])
def test_a_sweep_needs_nonempty_lists(tmp_path, values):
    path = write(tmp_path, "grid.yaml", f"sweep:\n  plain.width: {values}\n")
    with pytest.raises(ValueError, match="nonempty list"):
        config_loader.load(Outer, path)


@pytest.mark.parametrize(
    ("schema", "text"),
    [
        (Outer, "unknown: 1\n"),
        (Outer, "plain: {unknown: 1}\n"),
        (Collections, "records: [{unknown: 1}]\n"),
    ],
    ids=["root", "nested-block", "list-of-blocks"],
)
def test_unknown_fields_are_refused_throughout_the_schema(tmp_path, schema, text):
    with pytest.raises(ValueError, match="has no field"):
        config_loader.load(schema, write(tmp_path, "run.yaml", text))


def test_a_scalar_of_the_wrong_type_is_refused(tmp_path):
    with pytest.raises(ValueError, match="width must be int"):
        config_loader.load(Inner, write(tmp_path, "run.yaml", "width: wide\n"))


def test_a_missing_file_is_reported_before_composition(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config file"):
        config_loader.load(Outer, tmp_path / "absent.yaml")


def test_a_loaded_config_remembers_its_file_its_sweep_point_and_its_overrides(tmp_path):
    """A run's metadata names the file that reproduces it, which the resolved values alone cannot"""
    path = write(tmp_path, "run.yaml", "plain:\n  width: 32\nsweep:\n  plain.label: [a, b]\n")
    first, second = config_loader.load(Outer, path, ["plain.width=99"])
    assert config_loader.source_of(first) == {
        "path": str(path.resolve()),
        "sweep": ["plain.label=a"],
        "overrides": ["plain.width=99"],
    }
    assert config_loader.source_of(second)["sweep"] == ["plain.label=b"]
    assert "_lsa_source" not in str(asdict(first))


def test_a_config_built_without_the_loader_names_no_source():
    assert config_loader.source_of(Outer()) == {}


def test_a_bool_is_not_a_number(tmp_path):
    """`isinstance(True, int)` is true, so an unguarded numeric check reads a yaml `true` as a 1"""
    for line in ("count: true", "rate: true"):
        with pytest.raises(ValueError, match="got the bool"):
            config_loader.load(Numbers, write(tmp_path, "run.yaml", line + "\n"))


def test_a_bool_field_still_takes_a_bool(tmp_path):
    built = config_loader.load(Numbers, write(tmp_path, "flag.yaml", "enabled: true\n"))[0]
    assert built.enabled is True


def test_a_path_under_the_root_follows_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_VARIABLE, "/elsewhere")
    built = config_loader.load(Inner, write(tmp_path, "run.yaml", "label: ${root:models/maps/map.pt}\n"))[0]
    assert built.label == "/elsewhere/models/maps/map.pt"


def test_a_path_under_the_root_follows_the_site_the_run_composes(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    path = write(tmp_path, "run.yaml", "label: ${root:models/maps/map.pt}\n")
    assert config_loader.load(Inner, path, ["site=local"])[0].label == str(
        tmp_path / "runs/models/maps/map.pt"
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("args: [--max-tokens, 128]", r"args\[1\] must be str"),
        ("counts: [1, true]", r"counts\[1\].*got the bool"),
        ("rates: [[1, false]]", r"rates\[0\]\[1\].*got the bool"),
        ("optional: [128]", r"optional\[0\] must be str"),
        ("args: --max-tokens", "args must be a list"),
    ],
)
def test_list_elements_are_validated_when_loading(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        config_loader.load(Collections, write(tmp_path, "run.yaml", line + "\n"))


def test_list_overrides_require_quoted_numeric_strings(tmp_path):
    path = write(tmp_path, "run.yaml", "{}\n")
    with pytest.raises(ValueError, match=r"args\[1\] must be str"):
        config_loader.load(Collections, path, ["args=[--max-tokens,128]"])
    built = config_loader.load(Collections, path, ['args=[--max-tokens,"128"]'])[0]
    assert built.args == ["--max-tokens", "128"]


def test_optional_lists_and_nested_numeric_lists_keep_valid_values(tmp_path):
    built = config_loader.load(
        Collections,
        write(tmp_path, "run.yaml", "rates: [[1, 0.5]]\noptional: null\nrecords: [{width: 16}]\n"),
    )[0]
    assert built.rates == [[1, 0.5]]
    assert built.optional is None
    assert built.records == [Inner(width=16)]


def test_evaluation_cli_arguments_reject_numeric_values_before_launch(tmp_path):
    from lsa.evaluation import config as evaluation

    path = write(tmp_path, "eval.yaml", "model: {name: qwen35_4b}\nrun: {image: /images/sglang.img}\n")
    with pytest.raises(ValueError, match=r"args\[1\] must be str"):
        evaluation.load(path, ["run.args=[--max-tokens,128]"], kind="accuracy")
    built = evaluation.load(path, ['run.args=[--max-tokens,"128"]'], kind="accuracy")[0]
    assert built.run.args == ["--max-tokens", "128"]


def test_a_field_that_is_not_positive_is_refused_by_its_section_and_name():
    @dataclass
    class Section:
        width: int = 1
        steps: int = 0

    config_loader.require_positive(Section(), "space", "width")
    with pytest.raises(ValueError, match="space.steps must be positive - got 0"):
        config_loader.require_positive(Section(), "space", "width", "steps")


SITE_RUN = "model: {name: qwen35_4b}\nrun: {image: '${site.images.h200}'}\n"


def test_the_default_site_names_the_image_and_the_site_node_is_not_a_field(tmp_path, monkeypatch):
    from lsa.evaluation import config as evaluation

    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "mine.yaml").write_text("root: /mine\nimages: {h200: registry/sglang:2}\n")
    monkeypatch.setattr(site, "SITES", sites)
    monkeypatch.setattr(site, "DEFAULT_SITE", "mine")
    path = write(tmp_path, "eval.yaml", SITE_RUN)
    built = evaluation.load(path, kind="accuracy")[0]
    assert built.run.image == "registry/sglang:2"


def test_a_site_override_picks_another_site_and_is_not_an_assignment(tmp_path, monkeypatch):
    from lsa.evaluation import config as evaluation

    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "mine.yaml").write_text("images: {h200: registry/sglang:1}\n")
    monkeypatch.setattr(site, "SITES", sites)
    path = write(tmp_path, "eval.yaml", SITE_RUN)
    built = evaluation.load(path, ["site=mine"], kind="accuracy")[0]
    assert built.run.image == "registry/sglang:1"
    assert built._lsa_source.overrides == ["site=mine"]


def test_an_unknown_site_is_refused_with_the_sites_there_are(tmp_path):
    from lsa.evaluation import config as evaluation

    path = write(tmp_path, "eval.yaml", SITE_RUN)
    with pytest.raises(ValueError, match=r"no site 'nowhere'.*local"):
        evaluation.load(path, ["site=nowhere"], kind="accuracy")


def test_an_image_the_site_does_not_name_is_refused_at_the_boundary_saying_what_to_set(tmp_path):
    from lsa.evaluation import config as evaluation

    path = write(tmp_path, "eval.yaml", "model: {name: qwen35_4b}\nrun: {image: '${site.images.missing}'}\n")
    with pytest.raises(ValueError, match=r"(?s)site.images.missing.*configs.*site"):
        evaluation.load(path, kind="accuracy")


def test_a_run_with_an_empty_image_loads_and_is_refused_at_the_launch(tmp_path):
    from lsa.evaluation import config as evaluation
    from lsa.sglang.launch import SglangConfig, container_command, require_image

    path = write(tmp_path, "eval.yaml", SITE_RUN)
    built = evaluation.load(path, ["site=local"], kind="accuracy")[0]
    assert built.server().image == ""
    with pytest.raises(site.NotNamed, match=r"run.image is not named.*configs/site"):
        require_image(built.server().image)
    with pytest.raises(site.NotNamed, match=r"run.image is not named.*configs/site"):
        container_command(SglangConfig(image=""), "true")


def test_the_default_site_is_chosen_in_one_place_and_the_local_site_holds_no_cluster_path():
    import re

    root = site.SITES.parent
    # Only the loader's constant names the default, and no config sets or names a site
    for path in sorted(root.rglob("*.yaml")):
        if path.parent == site.SITES:
            continue
        text = path.read_text()
        assert not re.search(r"^site\s*:", text, re.MULTILINE), path
        assert "site=" not in text, path
    source = Path(site.__file__).read_text()
    assert len(re.findall(r'^DEFAULT_SITE\s*=\s*"', source, re.MULTILINE)) == 1
    for path in sorted(Path(site.__file__).parent.rglob("*.py")):
        assert path == Path(site.__file__) or not re.search(
            r"^DEFAULT_SITE\b", path.read_text(), re.MULTILINE
        ), path

    local = site.load("local")
    values = [value for value in _leaves(local) if isinstance(value, str)]
    assert not [value for value in values if value.startswith(("/", "~"))], values
    # Its root is relative, and a site without one would leave the code no root to read
    assert local["root"] == "./runs"


def _leaves(node):
    if isinstance(node, (dict,)) or hasattr(node, "values"):
        for value in node.values():
            yield from _leaves(value)
    else:
        yield node


def test_a_submission_refuses_an_empty_image_before_queueing(tmp_path):
    from lsa.evaluation import config as evaluation
    from lsa.evaluation.run import require_image

    path = write(tmp_path, "eval.yaml", SITE_RUN)
    built = evaluation.load(path, ["site=local"], kind="accuracy")[0]
    with pytest.raises(site.NotNamed, match=r"run.image is not named"):
        require_image(built)
