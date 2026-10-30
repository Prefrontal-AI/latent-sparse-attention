"""A site written for one test, so a test never depends on which sites the checkout holds."""

from pathlib import Path

import pytest

from lsa import site


def use_site(monkeypatch: pytest.MonkeyPatch, directory: Path, text: str, name: str = "test") -> None:
    """Make `text` the site this process reads, and the only site there is."""
    (directory / f"{name}.yaml").write_text(text)
    monkeypatch.setattr(site, "SITES", directory)
    monkeypatch.setattr(site, "_selected", name)
