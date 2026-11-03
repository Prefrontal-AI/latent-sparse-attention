"""Pinned local prerequisites are verified before preparation or publication, without network access."""

import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsa import cli
from lsa.evaluation.accuracy.data import ruler

PATCHED = "scripts/data/synthetic/common_words_extraction.py"
RESOURCES = (
    "tokenizers/punkt/english.pickle",
    "tokenizers/punkt_tab/english/collocations.tab",
    "tokenizers/punkt_tab/english/sent_starters.txt",
    "tokenizers/punkt_tab/english/abbrev_types.txt",
    "tokenizers/punkt_tab/english/ortho_context.tab",
)


def resource_files(directory):
    for resource in RESOURCES:
        path = Path(directory) / resource
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture resource\n")


@pytest.fixture
def prerequisites(tmp_path, monkeypatch):
    """A real git pin and patch, measured haystacks, dist-info pins and local NLTK resources."""
    checkout = tmp_path / "existing"
    checkout.mkdir()

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments], capture_output=True, text=True, check=True
        ).stdout

    git("init", "--quiet")
    git("config", "user.name", "RULER fixture")
    git("config", "user.email", "fixture@example.invalid")
    generator = checkout / PATCHED
    generator.parent.mkdir(parents=True)
    generator.write_text("print('original')\n")
    (checkout / "untouched.py").write_text("print('untouched')\n")
    haystacks = generator.parent / "json"
    haystacks.mkdir()
    hashes = {}
    for name in ruler.ASSETS:
        content = (name + " fixture\n").encode()
        (haystacks / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    git("add", ".")
    git("commit", "--quiet", "-m", "Pinned fixture")
    revision = git("rev-parse", "HEAD").strip()
    generator.write_text("print('patched')\n")
    patch = tmp_path / "generator.patch"
    patch.write_text(git("diff", "--", PATCHED))
    for dependency in ruler.DEPENDENCIES:
        name, version = dependency.split("==")
        info = checkout / ".deps" / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    resources = tmp_path / "nltk"
    resource_files(resources)
    monkeypatch.setattr(ruler, "ruler_card", lambda: SimpleNamespace(revision=revision))
    monkeypatch.setattr(ruler, "PATCH", str(patch))
    monkeypatch.setattr(ruler, "ASSETS", hashes)
    monkeypatch.setattr(ruler, "RULER_NVIDIA", str(checkout))
    monkeypatch.setattr(ruler, "NLTK_DATA", str(resources))
    return checkout, resources


def downloader(monkeypatch, complete=True, fail=False, before_publish=None):
    """Replace only network installation; git verification and no-clobber publication remain real."""
    run = subprocess.run
    calls = []

    def local(command, **kwargs):
        if "nltk.downloader" in command:
            calls.append(command)
            destination = command[command.index("-d") + 1]
            if fail:
                raise subprocess.CalledProcessError(1, command)
            if complete:
                resource_files(destination)
            else:
                Path(destination).mkdir()
            return subprocess.CompletedProcess(command, 0)
        if command[0] == "mv" and before_publish is not None:
            before_publish(Path(command[-1]))
        return run(command, **kwargs)

    monkeypatch.setattr(ruler.subprocess, "run", local)
    return calls


def test_a_pinned_patched_installation_is_reused_without_installing(prerequisites, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("verified prerequisites must not download anything")

    monkeypatch.setattr(ruler, "install", refuse)
    monkeypatch.setattr(ruler, "fetch", refuse)
    ruler.ensure()


@pytest.mark.parametrize("changed", ["revision", "patch", "tracked", "haystack", "dependency"])
def test_changed_prerequisites_are_refused_and_never_replaced(prerequisites, monkeypatch, changed):
    checkout, _ = prerequisites
    if changed == "revision":
        monkeypatch.setattr(ruler, "ruler_card", lambda: SimpleNamespace(revision="0" * 40))
    elif changed == "patch":
        (checkout / PATCHED).write_text("print('wrong patch')\n")
    elif changed == "tracked":
        (checkout / "untouched.py").write_text("print('unexpected edit')\n")
    elif changed == "haystack":
        (checkout / "scripts/data/synthetic/json/squad.json").write_text("changed")
    else:
        next((checkout / ".deps").glob("*/METADATA")).write_text("Name: tenacity\nVersion: 0\n")
    before = {
        str(path.relative_to(checkout)): path.read_bytes() for path in checkout.rglob("*") if path.is_file()
    }

    def refuse(*args, **kwargs):
        raise AssertionError("an existing installation must never be repaired in place")

    monkeypatch.setattr(ruler, "install", refuse)
    with pytest.raises(ValueError, match="fresh private paths"):
        ruler.ensure()
    assert before == {
        str(path.relative_to(checkout)): path.read_bytes() for path in checkout.rglob("*") if path.is_file()
    }


def test_an_existing_generator_installs_absent_sentence_tokenizers(prerequisites, monkeypatch):
    _, resources = prerequisites
    shutil.rmtree(resources)
    calls = downloader(monkeypatch)
    ruler.ensure()
    assert len(calls) == 1
    ruler.verify_nltk()


@pytest.mark.parametrize("fail", [False, True], ids=["incomplete-success", "download-failure"])
def test_failed_sentence_tokenizers_are_never_published(prerequisites, monkeypatch, fail):
    checkout, resources = prerequisites
    shutil.rmtree(resources)
    generator_before = (checkout / PATCHED).read_bytes()
    downloader(monkeypatch, complete=False, fail=fail)
    with pytest.raises(ValueError, match="fresh private paths"):
        ruler.ensure()
    assert not resources.exists()
    assert (checkout / PATCHED).read_bytes() == generator_before
    assert not list(resources.parent.glob(".lsa-nltk-*"))


def test_a_failed_generator_install_is_cleaned_without_publication(prerequisites, tmp_path, monkeypatch):
    destination = tmp_path / "fresh-generator"
    monkeypatch.setattr(ruler, "RULER_NVIDIA", str(destination))

    def incomplete(path):
        Path(path).mkdir()
        (Path(path) / "partial").write_text("incomplete")
        raise ValueError("download failed")

    monkeypatch.setattr(ruler, "install", incomplete)
    with pytest.raises(ValueError, match="download failed"):
        ruler.ensure()
    assert not destination.exists()
    assert not list(tmp_path.glob(".lsa-ruler-*"))


@pytest.mark.parametrize("valid_winner", [True, False], ids=["verified-winner", "changed-winner"])
def test_concurrent_generator_publication_preserves_and_checks_the_winner(
    prerequisites, tmp_path, monkeypatch, valid_winner
):
    checkout, _ = prerequisites
    destination = tmp_path / "concurrent-generator"
    monkeypatch.setattr(ruler, "RULER_NVIDIA", str(destination))
    monkeypatch.setattr(ruler, "install", lambda path: shutil.copytree(checkout, path))

    def publish_winner(path):
        assert path == destination
        shutil.copytree(checkout, path)
        (path / "winner").write_text("concurrent installation")
        if not valid_winner:
            (path / PATCHED).write_text("print('wrong concurrent patch')\n")

    downloader(monkeypatch, before_publish=publish_winner)
    if valid_winner:
        ruler.ensure()
    else:
        with pytest.raises(ValueError, match="installed generator does not match"):
            ruler.ensure()
    assert (destination / "winner").read_text() == "concurrent installation"
    if not valid_winner:
        assert (destination / PATCHED).read_text() == "print('wrong concurrent patch')\n"
    assert not list(tmp_path.glob(".lsa-ruler-*"))


@pytest.mark.parametrize("valid_winner", [True, False], ids=["verified-winner", "incomplete-winner"])
def test_concurrent_sentence_tokenizers_are_preserved_and_verified(prerequisites, monkeypatch, valid_winner):
    _, resources = prerequisites
    shutil.rmtree(resources)

    def publish_winner(path):
        assert path == resources
        resource_files(path)
        (path / "winner").write_text("concurrent resources")
        if not valid_winner:
            (path / RESOURCES[0]).unlink()

    downloader(monkeypatch, before_publish=publish_winner)
    if valid_winner:
        ruler.ensure()
    else:
        with pytest.raises(ValueError, match="prerequisites could not be verified"):
            ruler.ensure()
    assert (resources / "winner").read_text() == "concurrent resources"
    assert (resources / RESOURCES[0]).exists() == valid_winner
    assert not list(resources.parent.glob(".lsa-nltk-*"))


def test_ruler_prerequisite_failure_prevents_submission(monkeypatch):
    def refuse():
        raise ValueError("prerequisite verification failed")

    def unexpected_submit(*args):
        raise AssertionError("prerequisite failure must stop submission")

    monkeypatch.setattr(ruler, "ensure", refuse)
    monkeypatch.setattr("lsa.slurm.submit", unexpected_submit)
    with pytest.raises(SystemExit) as finished:
        cli.main(["data", "prepare", "ruler", "--tasks", "vt", "--lengths", "4096"])
    assert finished.value.code == 2


RELEASE = "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/ruler-sources-v1/"


def test_the_essays_and_squad_come_from_the_release_assets_first():
    assert ruler.ASSET_URLS["PaulGrahamEssays.json"][0] == RELEASE + "PaulGrahamEssays.json"
    assert ruler.ASSET_URLS["squad.json"][0] == RELEASE + "squad.json"
    assert ruler.ASSET_URLS["squad.json"][1].startswith("https://rajpurkar.github.io/")


def test_squad_tries_the_release_then_rajpurkar(tmp_path, monkeypatch):
    asked = []

    def curl(command, **kwargs):
        asked.append(command[-1])
        return SimpleNamespace(returncode=22 if len(asked) == 1 else 0)

    monkeypatch.setattr(ruler.subprocess, "run", curl)
    ruler.fetch(str(tmp_path / "squad.json"), ruler.ASSET_URLS["squad.json"])
    assert asked == list(ruler.ASSET_URLS["squad.json"])


def test_install_fetches_the_essays_and_never_scrapes_them(tmp_path, monkeypatch):
    commands = []
    fetched = {}
    monkeypatch.setattr(ruler, "ruler_card", lambda: SimpleNamespace(revision="pinned"))
    monkeypatch.setattr(ruler.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(ruler, "fetch", lambda path, urls: fetched.update({Path(path).name: urls}))
    ruler.install(str(tmp_path / "checkout"))
    assert fetched["PaulGrahamEssays.json"] == (RELEASE + "PaulGrahamEssays.json",)
    assert not any("download_paulgraham_essay.py" in part for command in commands for part in command)


def prepare_submission(monkeypatch, *flags):
    fetched = []
    submitted = []
    monkeypatch.setattr(ruler, "ensure", lambda: fetched.append("fetched"))
    monkeypatch.setattr("lsa.slurm.submit", lambda *call: submitted.append(call))
    cli.main(["data", "prepare", "ruler", "--tasks", "vt", "--lengths", "4096", *flags])
    return fetched, submitted


def test_the_prerequisites_are_fetched_where_the_command_is_typed_by_default(monkeypatch):
    fetched, submitted = prepare_submission(monkeypatch)
    assert fetched == ["fetched"]
    assert submitted[0][-1] is False


def test_fetch_in_job_leaves_the_submitting_host_alone(monkeypatch):
    fetched, submitted = prepare_submission(monkeypatch, "--fetch-in-job")
    assert fetched == []
    assert submitted[0][-1] is True


def test_the_job_fetches_first_when_asked(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(ruler, "ensure", lambda: order.append("ensure"))
    monkeypatch.setattr(ruler, "load_manifest", lambda root: order.append("manifest") or {})
    ruler.prepare_ruler([], [], 1, str(tmp_path), "tok", fetch_first=True)
    assert order[0] == "ensure"


def test_the_default_job_asks_for_the_full_grid_s_resources():
    assert ruler.PARAMETERS["cpus_per_task"] == 64
    assert ruler.PARAMETERS["mem_gb"] == 512
    assert ruler.PARAMETERS["timeout_min"] == 12 * 60
