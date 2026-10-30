"""The cards - that each one describes its dataset or checkpoint completely, and that none has drifted from it.

The drift test is the one worth having. A card is a claim about bytes on a shared volume, and nothing stops the
two from parting company - a repack with a different tokenizer, a source added, a path moved. A card that lies
is worse than no card, because a run trusts it by name, which is why what preparing measured lives in the
manifest beside the bytes and reading it through the card refuses a manifest prepared under another.
"""

import dataclasses
import json

import pytest

from lsa import assets
from lsa.constants import Attention
from lsa.data import manifests

TOKENIZED = assets.TokenizedCard(
    name="tiny", path="/srv/tiny", sources=["arxiv", "code"], tokenizer="glm53_flash", sample_length=16
)


def measured(card, **changes):
    """A manifest for `card` as tokenizing would write it, with any field changed"""
    fields = {
        "declared": manifests.declared(card),
        "tokenizer": "/snapshot",
        "tokenizer_revision": "abc",
        "vocab_size": 8,
        "sample_length": card.sample_length,
        "sources": card.sources,
        "rows": dict.fromkeys(card.sources, 2),
        "documents": dict.fromkeys(card.sources, 1),
    }
    return manifests.TokenizedManifest(**{**fields, **changes})


def test_every_registered_card_loads():
    """A card that does not validate is a card no run can name"""
    for name in assets.dataset_names():
        assert assets.dataset(name).name == name
    for name in assets.model_names():
        assert assets.model(name).name == name


def test_the_defaults_are_registered():
    assert assets.Default.DATASET in assets.dataset_names(assets.Kind.TOKENIZED)
    assert assets.Default.MODEL in assets.model_names()


def test_an_unknown_name_says_what_there_is():
    with pytest.raises(ValueError, match="no dataset card named 'tokens_llama'"):
        assets.dataset("tokens_llama")
    with pytest.raises(ValueError, match="no tokenized card named 'ruler'"):
        assets.tokenized("ruler")
    with pytest.raises(ValueError, match="no model card named 'llama4'"):
        assets.model("llama4")


def test_a_card_path_is_absolute_so_a_compute_node_finds_it():
    with pytest.raises(ValueError, match="path must be absolute"):
        assets.SourceCard(name="x", path="relative/x", dataset="org/x", licence="MIT")
    with pytest.raises(ValueError, match="path must be absolute"):
        dataclasses.replace(TOKENIZED, path="relative/x")
    with pytest.raises(ValueError, match="path must be absolute"):
        assets.ModelCard(name="x", path="relative/x")


def test_a_source_comes_from_a_hub_dataset_or_a_path():
    with pytest.raises(ValueError, match="exactly one"):
        assets.SourceCard(name="x", path="/srv/x", licence="MIT")
    with pytest.raises(ValueError, match="exactly one"):
        assets.SourceCard(name="x", path="/srv/x", dataset="org/x", collected_from="/srv/y", licence="MIT")


def test_a_source_says_what_its_documents_may_be_used_for():
    with pytest.raises(ValueError, match="needs a licence"):
        assets.SourceCard(name="x", path="/srv/x", dataset="org/x", licence="")


def test_one_licence_for_every_document_has_no_column_to_read():
    with pytest.raises(ValueError, match="no per-document licence"):
        assets.source("games").document_licences()


def test_a_tokenized_dataset_says_which_tokenizer_writes_it():
    """The whole point - ids alone do not say which vocabulary they are in"""
    with pytest.raises(ValueError, match="names its sources and its tokenizer"):
        dataclasses.replace(TOKENIZED, tokenizer="")


@pytest.mark.parametrize("name", assets.dataset_names(assets.Kind.TOKENIZED))
def test_a_tokenized_dataset_names_registered_sources_and_tokenizer(name):
    card = assets.tokenized(name)
    assert card.tokenizer in assets.model_names()
    assert set(card.sources) <= set(assets.dataset_names(assets.Kind.SOURCE))


def test_a_benchmark_is_a_hub_dataset_or_a_generator():
    with pytest.raises(ValueError, match="exactly one"):
        assets.BenchmarkCard(name="x", licence="MIT", revision="abc")
    with pytest.raises(ValueError, match="pins the revision"):
        assets.BenchmarkCard(name="x", licence="MIT", dataset="org/x")
    with pytest.raises(ValueError, match="names its file and its tokenizer"):
        assets.BenchmarkCard(name="x", licence="MIT", dataset="org/x", revision="abc", path="/srv/x")


@pytest.mark.parametrize("name", assets.dataset_names(assets.Kind.BENCHMARK))
def test_a_prepared_benchmark_is_counted_in_a_registered_tokenizer(name):
    card = assets.benchmark(name)
    assert not card.path or card.tokenizer in assets.model_names()


@pytest.mark.parametrize("name", assets.dataset_names(assets.Kind.WORKLOAD))
def test_a_workload_is_built_from_a_registered_tokenized_dataset_and_source(name):
    card = assets.workload(name)
    assert card.source in assets.tokenized(card.tokenized).sources


def test_a_card_sits_in_the_directory_of_its_kind(tmp_path, monkeypatch):
    from lsa.assets import loading

    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "x.yaml").write_text("name: x\nkind: tokenized\npath: /srv/x\n")
    monkeypatch.setattr(loading, "_DATASETS", tmp_path)
    with pytest.raises(ValueError, match="sits with the source cards and says kind 'tokenized'"):
        assets.source("x")


def test_the_declared_fields_leave_out_where_a_dataset_is_and_how_it_reads():
    declared = manifests.declared(TOKENIZED)
    assert "path" not in declared and "about" not in declared
    assert declared["sources"] == ["arxiv", "code"]


def test_which_generator_prepares_a_benchmark_is_not_part_of_what_its_files_are():
    """A manifest written before the card named its generator still matches the card"""
    assert assets.benchmark("ruler").preparer == assets.Preparer.RULER
    assert "preparer" not in manifests.declared(assets.benchmark("ruler"))


def test_a_manifest_is_read_through_its_card(tmp_path):
    card = dataclasses.replace(TOKENIZED, path=str(tmp_path))
    manifests.write_tokenized(tmp_path, measured(card))
    assert card.manifest().vocab_size == 8
    assert card.manifest().tokens == 4 * 16


def test_a_manifest_tokenized_under_another_card_is_refused(tmp_path):
    """The card changed after the bytes were made, so the bytes are not what it says"""
    card = dataclasses.replace(TOKENIZED, path=str(tmp_path))
    manifests.write_tokenized(tmp_path, measured(card))
    with pytest.raises(ValueError, match=r"differs in \['sample_length'\]"):
        dataclasses.replace(card, sample_length=32).manifest()


def test_a_manifest_that_predates_cards_is_refused(tmp_path):
    (tmp_path / "corpus.json").write_text(json.dumps({"tokenizer": "/t", "vocab_size": 8}))
    with pytest.raises(ValueError, match="predates dataset cards"):
        dataclasses.replace(TOKENIZED, path=str(tmp_path)).manifest()


def test_an_unregistered_dataset_is_the_card_its_manifest_was_tokenized_under(tmp_path):
    manifests.write_tokenized(tmp_path, measured(TOKENIZED))
    card = assets.resolve(assets.Default.DATASET, str(tmp_path))
    assert card.path == str(tmp_path)
    assert card.sources == TOKENIZED.sources
    assert assets.resolve(assets.Default.DATASET).name == assets.Default.DATASET


def test_an_unregistered_dataset_needs_the_manifest_tokenizing_writes(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus.json"):
        assets.unregistered(tmp_path)


def test_a_benchmark_file_keeps_what_its_generator_recorded_apart_from_its_card(tmp_path):
    card = assets.benchmark("ruler")
    entry = {"rows": 20, "sha256": "f", "seed": 42, "tokenizer": "t", "date": "2026-09-19"}
    record = manifests.BenchmarkFile.from_entry({**entry, "declared": manifests.declared(card)})
    manifests.write_benchmark_file(tmp_path, "ruler/cwe_4096.jsonl", record)

    (read,) = manifests.read_benchmark_files(tmp_path).values()
    assert read.recorded() == entry
    assert read.declared == manifests.declared(card)
    assert list(card.manifest(str(tmp_path)).files) == ["ruler/cwe_4096.jsonl"]


def test_a_benchmark_file_prepared_under_another_grid_is_refused(tmp_path):
    card = assets.benchmark("ruler")
    other = dataclasses.replace(card, seed=7)
    entry = {
        "rows": 20,
        "sha256": "f",
        "tokenizer": "t",
        "date": "2026-09-19",
        "declared": manifests.declared(other),
    }
    manifests.write_benchmark_file(
        tmp_path, "ruler/cwe_4096.jsonl", manifests.BenchmarkFile.from_entry(entry)
    )
    with pytest.raises(ValueError, match=r"differs in \['seed'\]"):
        card.manifest(str(tmp_path))


def test_a_benchmark_with_nothing_prepared_has_no_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="records no file of ruler/"):
        assets.benchmark("ruler").manifest(str(tmp_path))


def test_a_workload_manifest_is_read_through_its_card(tmp_path):
    card = dataclasses.replace(assets.workload("production"), path=str(tmp_path))
    written = manifests.WorkloadManifest(
        declared=manifests.declared(card), corpus="/c.bin", row=16, tokenizer="t", entries=3, files={}
    )
    manifests.write_workload(tmp_path, written)
    assert card.manifest() == written
    with pytest.raises(ValueError, match=r"differs in \['entries'\]"):
        dataclasses.replace(card, entries=2).manifest()


def test_a_model_asset_says_where_the_checkpoint_is():
    for unplaced in ({}, {"repo": "org/x"}, {"revision": "abc"}):
        with pytest.raises(ValueError, match="needs a path, or a repo and the revision"):
            assets.ModelCard(name="x", **unplaced)


def test_a_published_snapshot_is_read_from_this_users_hub_cache(monkeypatch):
    """No egress, so the pinned commit is read where the hub cache keeps it, and a checkpoint of ours by path"""
    monkeypatch.setenv("HF_HOME", "/caches/hf")
    published = assets.ModelCard(name="x", repo="org/x", revision="abc")
    assert published.checkpoint == "/caches/hf/hub/models--org--x/snapshots/abc"
    assert assets.ModelCard(name="x", path="/srv/x", repo="org/x").checkpoint == "/srv/x"


def test_a_pinned_rotary_dim_needs_an_end_to_sit_at():
    with pytest.raises(ValueError, match="rotary_dim needs an end"):
        assets.ModelCard(name="x", path="/srv/x", rotary_dim=64)


@pytest.mark.parametrize("name", assets.model_names())
def test_a_model_asset_says_what_attention_its_checkpoint_ships(name):
    assert assets.model(name).attention in Attention.SHIPPED


def test_an_attention_no_checkpoint_ships_is_refused():
    with pytest.raises(ValueError, match="attention must say what the checkpoint ships"):
        assets.ModelCard(name="x", path="/srv/x", attention="lsa")


def text_config(published):
    """Return the block a model's own shape sits under, which a multimodal checkpoint nests."""
    return published.get("text_config") or published


def rotated_dims(text):
    """Return the head dims a snapshot rotates, or None when it does not say."""
    rope = text.get("rope_parameters") or {}
    factor = rope.get("partial_rotary_factor", text.get("partial_rotary_factor"))
    head_dim = text.get("head_dim")
    if factor is None or head_dim is None:
        return None
    return int(head_dim * factor)


@pytest.mark.parametrize("name", assets.dataset_names(assets.Kind.SHARDS))
def test_token_shards_name_a_registered_tokenizer_and_sit_where_they_are(name):
    card = assets.shards(name)
    assert card.tokenizer in assets.model_names()
    assert card.path.startswith("/")


def test_nothing_prepares_token_shards():
    from lsa.data.prepare import preparer_of

    with pytest.raises(ValueError, match="read where it is"):
        preparer_of(assets.shards("fineweb_edu_100bt"))


def test_a_training_run_reads_text_ids_or_shards_and_nothing_else():
    assert assets.corpus("fineweb_edu_100bt").name == "fineweb_edu_100bt"
    assert assets.corpus("gpt2").name == "gpt2"
    with pytest.raises(ValueError, match="a training run reads"):
        assets.corpus("ruler")


def test_the_two_gpt2_corpora_share_a_vocabulary_with_the_tokenizer_that_wrote_them():
    assert assets.shards("fineweb_edu_100bt").tokenizer == assets.tokenized("gpt2").tokenizer == "gpt2"
