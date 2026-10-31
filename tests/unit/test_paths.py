from pathlib import Path

import pytest

from lsa import paths, site


def test_every_location_moves_with_the_root(monkeypatch):
    monkeypatch.setenv(paths.ROOT_VARIABLE, "/elsewhere")
    locations = (
        paths.data(),
        paths.benchmarks(),
        paths.targets(),
        paths.goldens(),
        paths.training(),
        paths.evaluations(),
        paths.models(),
        paths.maps(),
        paths.logs(),
    )
    assert all(location.is_relative_to("/elsewhere") for location in locations)


def test_under_root_is_the_root_when_given_nothing(monkeypatch):
    monkeypatch.setenv(paths.ROOT_VARIABLE, "/elsewhere")
    assert paths.under_root() == "/elsewhere"
    assert paths.under_root("logs/viewer") == "/elsewhere/logs/viewer"


def make_sites(tmp_path, monkeypatch, **sites):
    directory = tmp_path / "sites"
    directory.mkdir()
    for name, text in sites.items():
        (directory / f"{name}.yaml").write_text(text)
    monkeypatch.setattr(site, "SITES", directory)
    monkeypatch.setattr(site, "DEFAULT_SITE", next(iter(sites)))
    monkeypatch.setattr(site, "_selected", None)
    monkeypatch.delenv(paths.ROOT_VARIABLE, raising=False)


def test_the_site_gives_the_root_when_lsa_root_is_not_set(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /shared/lsa\n")
    assert paths.root() == Path("/shared/lsa")
    assert paths.logs() == Path("/shared/lsa/logs")


def test_lsa_root_wins_over_the_site(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /shared/lsa\n")
    monkeypatch.setenv(paths.ROOT_VARIABLE, "/elsewhere")
    assert paths.root() == Path("/elsewhere")
    assert paths.declared_root() == Path("/shared/lsa")


def test_an_empty_lsa_root_is_not_an_override(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /shared/lsa\n")
    monkeypatch.setenv(paths.ROOT_VARIABLE, "")
    assert paths.root() == Path("/shared/lsa")


def test_a_relative_root_is_taken_against_the_working_directory(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: ./runs\n")
    monkeypatch.chdir(tmp_path)
    assert paths.root() == tmp_path / "runs"


def test_a_selected_site_moves_the_root(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /shared/lsa\n", other="root: /other\n")
    site.select("other")
    assert paths.root() == Path("/other")


def test_a_site_without_a_root_has_no_third_fallback(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="images: {}\n")
    with pytest.raises(site.NotNamed, match=r"sets no root.*LSA_ROOT"):
        paths.root()


def test_a_card_path_under_the_sites_datasets(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /r\ndatasets: /shared/sets\n")
    assert paths.under_datasets("fineweb") == "/shared/sets/fineweb"


def test_a_site_without_datasets_says_where_to_set_them(tmp_path, monkeypatch):
    make_sites(tmp_path, monkeypatch, here="root: /r\n")
    with pytest.raises(site.NotNamed, match=r"sets no datasets.*datasets:"):
        paths.under_datasets("fineweb")
