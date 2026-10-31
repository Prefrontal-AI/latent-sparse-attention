"""The roofline page embeds its data and ships with the package."""

import json
import re
from pathlib import Path

import pytest

from lsa.evaluation.performance.roofline import page


def read_embedded(html: str, script: str = "roofline-data") -> object:
    match = re.search(rf'<script id="{script}" type="application/json">(.*?)</script>', html, re.S)
    assert match
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_render_embeds_the_json(tmp_path: Path) -> None:
    roofline = {"machine": {"name": "H200"}, "kernels": [{"kernel": "a"}], "steps": []}
    source = tmp_path / "roofline.json"
    source.write_text(json.dumps(roofline))
    output = page.render(source, tmp_path / "out" / "roofline.html")
    html = output.read_text()
    assert read_embedded(html) == roofline
    assert page.PLACEHOLDER not in html
    assert "<title>CuTeDSL Kernel Roofline</title>" in html


def test_render_escapes_a_closing_script_tag(tmp_path: Path) -> None:
    source = tmp_path / "roofline.json"
    source.write_text(json.dumps({"kernels": [{"kernel": "</script><b>"}]}))
    html = page.render(source, tmp_path / "roofline.html").read_text()
    assert html.count("</script>") == 3
    assert read_embedded(html) == {"kernels": [{"kernel": "</script><b>"}]}


def test_render_turns_non_finite_numbers_into_null(tmp_path: Path) -> None:
    source = tmp_path / "roofline.json"
    source.write_text('{"kernels": [{"flops": NaN, "bytes": Infinity}]}')
    html = page.render(source, tmp_path / "roofline.html").read_text()
    assert read_embedded(html) == {"kernels": [{"flops": None, "bytes": None}]}


def test_render_rejects_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "roofline.json"
    source.write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        page.render(source, tmp_path / "roofline.html")


def test_template_uses_no_dash_characters() -> None:
    template = (Path(page.__file__).parent / page.TEMPLATE).read_text(encoding="utf-8")
    assert "\u2013" not in template
    assert "\u2014" not in template


def test_template_opens_with_the_overview_before_the_flowcharts() -> None:
    template = (Path(page.__file__).parent / page.TEMPLATE).read_text(encoding="utf-8")
    assert template.index('<section id="overview">') < template.index('<section id="pipeline">')
    assert 'id="flow-prefill"' in template
    assert 'id="flow-decode"' in template


def test_template_sets_triton_beside_cutedsl_after_the_time_bars() -> None:
    template = (Path(page.__file__).parent / page.TEMPLATE).read_text(encoding="utf-8")
    assert (
        template.index('id="bars"')
        < template.index('<section id="versus">')
        < template.index("<h3>This profile</h3>")
    )
    assert template.index("<h3>This profile</h3>") < template.index('<section id="pipeline">')
    assert 'id="f-triton"' in template
    assert 'id="f-arm-table"' in template


def test_render_embeds_both_arms(tmp_path: Path) -> None:
    roofline = {
        "kernels": [{"kernels": "cutedsl", "kind": "cutedsl"}, {"kernels": "triton", "kind": "triton"}],
        "steps": [
            {"step": "expansion by threshold", "kernels": "cutedsl", "launches": [0]},
            {"step": "expansion by threshold", "kernels": "triton", "launches": [1]},
        ],
    }
    source = tmp_path / "roofline.json"
    source.write_text(json.dumps(roofline))
    html = page.render(source, tmp_path / "roofline.html").read_text()
    assert read_embedded(html) == roofline


def test_render_embeds_another_gpus_profile_to_compare(tmp_path: Path) -> None:
    this = {"machine": {"label": "B300"}, "kernels": [], "steps": [{"step": "fill", "median_ms": 1.0}]}
    other = {"machine": {"label": "H200"}, "kernels": [], "steps": [{"step": "fill", "median_ms": 2.0}]}
    (tmp_path / "b300.json").write_text(json.dumps(this))
    (tmp_path / "h200.json").write_text(json.dumps(other))
    html = page.render(tmp_path / "b300.json", tmp_path / "page.html", tmp_path / "h200.json").read_text()
    assert read_embedded(html) == this
    assert read_embedded(html, "compare-data") == other
    assert page.COMPARE_PLACEHOLDER not in html
    assert '<section id="compare" hidden>' in html


def test_render_without_a_comparison_embeds_null(tmp_path: Path) -> None:
    source = tmp_path / "roofline.json"
    source.write_text(json.dumps({"kernels": []}))
    html = page.render(source, tmp_path / "roofline.html").read_text()
    assert read_embedded(html, "compare-data") is None


def test_funnel_has_design_phase_decode_read_and_context_controls() -> None:
    template = (Path(page.__file__).parent / page.TEMPLATE).read_text(encoding="utf-8")
    for name in ("funnel-design", "funnel-phase", "funnel-read"):
        assert f'name="{name}"' in template
    for value in ("flat", "fixed", "tree", "prefill", "decode", "descend"):
        assert f'value="{value}"' in template
    assert 'id="funnel-context"' in template
    assert 'id="funnel-meaning"' in template
    assert "Read list" not in template
    assert 'name: "Selection"' in template


def test_funnel_stages_cover_every_configuration() -> None:
    template = (Path(page.__file__).parent / page.TEMPLATE).read_text(encoding="utf-8")
    block = template[template.index("var FUNNEL_PATHS = {") :]
    block = block[: block.index("};")]
    keys = set(re.findall(r'"([a-z]+-[a-z]+(?:-[a-z]+)?)":', block))
    expected = {
        f"{design}-{phase}" + (f"-{read}" if phase == "decode" and design != "flat" else "")
        for design in ("flat", "fixed", "tree")
        for phase in ("prefill", "decode")
        for read in ("flat", "descend")
    }
    assert keys == expected
