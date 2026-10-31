"""The settings block of the model config - what a server reads, found wherever the override landed and refused by name."""

import json
from types import SimpleNamespace

import pytest

from lsa.sglang import settings
from lsa.sglang.settings import PluginSettings
from lsa.sglang.settings.debug import CHECK_STEPS, DUMP_LAYERS, DebugSettings, DumpSettings
from lsa.sglang.settings.design import DesignSettings, PisaConfig

DESIGN = {
    "window_size": 512,
    "sink_size": 4,
    "top_k": 2048,
    "top_p": 512,
    "index_dim": 8,
    "projections": "/maps/x.pt",
}


def test_no_block_is_the_checkpoints_own_indexer():
    served = settings.of(SimpleNamespace())
    assert served.selector == "native"
    assert served.design is None and served.pisa is None
    # Nothing is checked or dumped either
    assert served == PluginSettings()


def test_a_design_block_reads_back_the_design_it_was_written_from():
    made = DesignSettings.parse(DESIGN)
    served = settings.of(SimpleNamespace(lsa=json.loads(json.dumps(made.body()))))
    assert served.selector == "lsa" and served.design == made
    assert settings.design_of(SimpleNamespace(lsa=made.body())) == made


def test_a_block_naming_pisa_is_pisas_and_holds_no_lsa_design():
    config = SimpleNamespace(lsa=PisaConfig(keep_count=40).body())
    served = settings.of(config)
    assert served.selector == "pisa" and served.pisa == PisaConfig(keep_count=40)
    assert settings.pisa_of(config).width == 40 * 64
    with pytest.raises(ValueError, match="holds no LSA design - this server runs pisa"):
        settings.design_of(config)


def test_a_server_of_the_checkpoints_own_indexer_has_neither_design_to_read():
    with pytest.raises(ValueError, match="does not name PISA - this server runs native"):
        settings.pisa_of(SimpleNamespace())
    with pytest.raises(ValueError, match="holds no LSA design - this server runs native"):
        settings.design_of(SimpleNamespace())


def test_a_block_is_found_on_the_top_level_config_or_on_the_nested_text_config():
    top = SimpleNamespace(lsa=DESIGN, text_config=SimpleNamespace())
    nested = SimpleNamespace(text_config=SimpleNamespace(lsa=DESIGN))
    assert settings.of(top).design == settings.of(nested).design == DesignSettings.parse(DESIGN)
    # A config that is a dict, as a published config.json is, reads the same
    assert settings.of({"text_config": {"lsa": DESIGN}}).design == DesignSettings.parse(DESIGN)


def test_a_block_with_only_debug_is_the_checkpoints_own_indexer_with_its_checks():
    served = settings.of(SimpleNamespace(lsa={"debug": {"dump": {"directory": "/tmp/dump"}}}))
    assert served.selector == "native"
    assert served.debug.dumps and served.debug.dump.directory == "/tmp/dump"


def test_debug_is_off_with_the_variables_defaults_until_the_block_turns_it_on():
    debug = settings.of(SimpleNamespace(lsa=DESIGN)).debug
    assert debug == DebugSettings()
    assert not debug.check_index and not debug.dumps
    assert debug.check_pisa.rows == 0 and debug.check_pisa.from_keys == 0
    assert debug.check_pisa.steps == CHECK_STEPS == 6
    assert debug.dump == DumpSettings()
    assert debug.dump.layers == DUMP_LAYERS == (3, 23, 43)
    assert debug.dump.stride == 4096 and debug.dump.lists and not debug.dump.all_ranks


def test_a_debug_sub_block_reads_nested_and_does_not_touch_the_design():
    block = {
        **DESIGN,
        "debug": {
            "check_index": True,
            "check_pisa": {"rows": 8, "from_keys": 40, "steps": 2},
            "dump": {"directory": "/d", "layers": [3, 7], "stride": 1, "lists": False, "all_ranks": True},
        },
    }
    served = settings.of(SimpleNamespace(lsa=json.loads(json.dumps(block))))
    assert served.design == DesignSettings.parse(DESIGN)
    debug = served.debug
    assert debug.check_index and (debug.check_pisa.rows, debug.check_pisa.from_keys) == (8, 40)
    assert debug.check_pisa.steps == 2
    assert debug.dump == DumpSettings("/d", (3, 7), 1, False, True)


@pytest.mark.parametrize(
    ("debug", "message"),
    [
        ({"dumps": {}}, r"debug has no knobs \['dumps'\] - it has \['check_index', 'check_pisa', 'dump'\]"),
        ({"check_pisa": {"row": 2}}, r"debug.check_pisa has no knobs \['row'\]"),
        ({"dump": {"dir": "/d"}}, r"debug.dump has no knobs \['dir'\]"),
        ({"check_index": 1}, "check_index must be true or false - got 1"),
        ({"check_pisa": {"rows": -1}}, "check_pisa.rows must be at least 0 - got -1"),
        ({"check_pisa": {"steps": 0}}, "check_pisa.steps must be at least 1 - got 0"),
        ({"check_pisa": 4}, "debug.check_pisa must be an object - got 4"),
        ({"dump": {"directory": 5}}, "dump.directory must be a path or null - got 5"),
        ({"dump": {"stride": 0}}, "dump.stride must be at least 1 - got 0"),
        ({"dump": {"layers": ["a"]}}, "dump.layers must be an integer - got 'a'"),
        ({"dump": {"lists": "no"}}, "dump.lists must be true or false - got 'no'"),
        ("on", "debug must be an object"),
    ],
)
def test_a_debug_sub_block_with_an_unknown_key_or_a_bad_value_is_refused_with_what_arrived(debug, message):
    with pytest.raises(ValueError, match=message):
        settings.of(SimpleNamespace(lsa={**DESIGN, "debug": debug}))


def test_a_block_with_a_knob_the_design_lacks_is_refused_by_name():
    with pytest.raises(ValueError, match="bucket_shape"):
        settings.of(SimpleNamespace(lsa={**DESIGN, "bucket_shape": 5}))
    with pytest.raises(ValueError, match="recent"):
        settings.of(SimpleNamespace(lsa={**PisaConfig().body(), "recent": 1}))


def test_a_block_that_is_not_an_object_is_refused_with_what_arrived():
    with pytest.raises(ValueError, match=r"'lsa' must be an object of settings - got 'on'"):
        settings.of(SimpleNamespace(lsa="on"))


def test_a_selector_that_is_neither_pisa_nor_absent_is_refused():
    with pytest.raises(ValueError, match="selector"):
        settings.of(SimpleNamespace(lsa={**DESIGN, "selector": "nsa"}))
