"""Inline document allocation and extraction agree on their saved plan and cache identities."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lsa import assets
from lsa.core.index_space import RotaryLayout
from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.constants import Split
from lsa.training.distillation.data import selection, teacher, teacher_config, teacher_plan
from tests.helpers.training.data.fake_corpus import write_corpus


@pytest.fixture
def inline_extraction(tmp_path, monkeypatch):
    card = write_corpus(
        tmp_path / "corpus",
        sources=("arxiv",),
        documents=400,
        rows_per_document=4,
        sample_length=64,
    )
    monkeypatch.setattr(assets, "tokenized", lambda name: card)
    allocation = selection.SelectionConfig(
        datasets=[card.name],
        lengths=[64, 128],
        tokens_per_bucket=512,
        tokens_per_tier=256,
        domains={"academic": ["arxiv"]},
        scoring_per_domain=1,
    )
    return teacher_config.TeacherConfig(
        model=teacher_config.TeacherModelConfig(name="qwen35_4b"),
        data=teacher_config.DataConfig(selection=allocation),
        run=teacher_config.RunConfig(
            name="unit", collection="balanced", output=str(tmp_path / "targets"), layers=[0]
        ),
    )


def tokens_by_length(entries, split):
    counts = {}
    for entry in entries:
        if entry["split"] == split:
            counts[entry["length"]] = counts.get(entry["length"], 0) + entry["length"]
    return counts


def test_inline_allocation_persists_the_full_pool_and_final_extraction_tier(inline_extraction):
    run = inline_extraction
    resolved = teacher_plan.resolve_selection(run)
    path = Path(run.run.output) / "balanced" / "selection.json"

    assert json.loads(path.read_text()) == resolved
    assert resolved["config"] == dataclasses.asdict(run.data.selection)
    assert tokens_by_length(resolved["full_documents"], Split.TRAIN) == {64: 512, 128: 512}
    assert tokens_by_length(resolved["documents"], Split.TRAIN) == {64: 256, 128: 256}
    assert [entry for entry in resolved["documents"] if entry["split"] == Split.EVAL] == [
        entry for entry in resolved["full_documents"] if entry["split"] == Split.EVAL
    ]
    assert all(entry in resolved["full_documents"] for entry in resolved["documents"])


def test_the_saved_selection_and_extraction_plan_name_the_same_windows(inline_extraction):
    run = inline_extraction
    resolved = teacher_plan.resolve_selection(run)
    planned = teacher_plan.plan(run, tokenizer=None)
    assert len(planned) == 1
    corpus, selected = planned[0]

    assert corpus.collection == "balanced"
    assert len(selected) == len(resolved["documents"])
    assert {
        (entry.source, entry.document, entry.split, entry.length, entry.window) for entry in selected
    } == {
        (entry["source"], entry["document"], entry["split"], entry["length"], entry["window"])
        for entry in resolved["documents"]
    }
    assert len({entry.name for entry in selected}) == len(selected)
    assert all(corpus.tokens(entry, entry.length).shape == (1, entry.length) for entry in selected)

    again = teacher_plan.plan(run, tokenizer=None)
    assert [entry.name for _, entries in again for entry in entries] == [entry.name for entry in selected]


def test_resuming_keeps_the_same_saved_selection(inline_extraction):
    run = inline_extraction
    first = teacher_plan.resolve_selection(run)
    assert teacher_plan.resolve_selection(run) == first


@pytest.mark.parametrize("changed", ["allocation", "queries"])
def test_resuming_with_another_selection_or_target_sampling_is_refused(inline_extraction, changed):
    run = inline_extraction
    first = teacher_plan.resolve_selection(run)
    if changed == "allocation":
        allocation = dataclasses.replace(run.data.selection, tokens_per_bucket=768)
        data = dataclasses.replace(run.data, selection=allocation)
    else:
        data = dataclasses.replace(run.data, queries=dataclasses.replace(run.data.queries, top=128))
    other = dataclasses.replace(run, data=data)

    with pytest.raises(ValueError, match="selection|extraction"):
        teacher_plan.resolve_selection(other)
    assert teacher_plan.resolve_selection(run) == first


def test_the_collection_manifest_records_the_selection_and_each_dataset(inline_extraction):
    run = inline_extraction
    planned = teacher_plan.plan(run, tokenizer=None)
    root = Path(run.run.output)
    teacher_plan.write_collection_manifests(root, run, planned, [0], RotaryLayout("prefix", 64))
    manifest = json.loads((root / "balanced" / "manifest.json").read_text())

    assert manifest["run"]["data"]["selection"] == dataclasses.asdict(run.data.selection)
    assert manifest["selection"]["config"] == dataclasses.asdict(run.data.selection)
    assert {entry["dataset"] for entry in manifest["documents"]} == {"corpus"}
    assert {entry["domain"] for entry in manifest["documents"]} == {"academic"}
    assert len(manifest["documents"]) == sum(len(entries) for _, entries in planned)


def test_forced_extraction_refuses_changed_selection_before_loading_weights(inline_extraction, monkeypatch):
    from lsa.evaluation import adapter

    run = inline_extraction
    teacher_plan.resolve_selection(run)
    allocation = dataclasses.replace(run.data.selection, tokens_per_bucket=768)
    other = dataclasses.replace(run, data=dataclasses.replace(run.data, selection=allocation))

    def refuse(*args, **kwargs):
        raise AssertionError("a mismatched selection reached model loading")

    monkeypatch.setattr(adapter, "adapter_for", refuse)
    with pytest.raises(ValueError, match="settings differ"):
        teacher.extract(other, force=True, device="cpu")


def test_several_datasets_share_one_complete_collection_manifest(inline_extraction, monkeypatch, tmp_path):
    run = inline_extraction
    first = assets.tokenized("corpus")
    second = write_corpus(
        tmp_path / "other",
        sources=("code",),
        documents=400,
        rows_per_document=4,
        sample_length=64,
    )
    cards = {first.name: first, second.name: second}
    monkeypatch.setattr(assets, "tokenized", cards.__getitem__)
    allocation = dataclasses.replace(
        run.data.selection,
        datasets=list(cards),
        domains={"academic": ["arxiv"], "code": ["code"]},
    )
    run = dataclasses.replace(run, data=dataclasses.replace(run.data, selection=allocation))
    planned = teacher_plan.plan(run, tokenizer=None)
    root = Path(run.run.output)
    teacher_plan.write_collection_manifests(root, run, planned, [0], RotaryLayout("prefix", 64))
    manifest = json.loads((root / "balanced" / "manifest.json").read_text())

    assert len(planned) == 2
    assert {corpus.collection for corpus, _ in planned} == {"balanced"}
    assert {entry["dataset"] for entry in manifest["documents"]} == set(cards)
    assert len(manifest["documents"]) == sum(len(entries) for _, entries in planned)
    assert len({entry["name"] for entry in manifest["documents"]}) == len(manifest["documents"])
    assert all(entry["name"].startswith(entry["dataset"] + "_") for entry in manifest["documents"])


def test_counts_based_extraction_keeps_its_dataset_collection_and_side_counts(inline_extraction):
    run = inline_extraction
    tokenized = teacher_config.TokenizedConfig(
        name="corpus",
        path=str(Path(run.run.output).parent / "corpus"),
        sources=["arxiv"],
        train=5,
        evaluate=2,
        held_out_fraction=1.0,
    )
    run = dataclasses.replace(run, data=teacher_config.DataConfig(tokenized=tokenized))
    [(corpus, documents)] = teacher_plan.plan(run, tokenizer=None)

    assert corpus.collection == "corpus"
    assert sum(entry.split == Split.TRAIN for entry in documents) == 5
    assert sum(entry.split == Split.EVAL for entry in documents) == 2
    assert not (Path(run.run.output) / "balanced" / "selection.json").exists()


@pytest.mark.parametrize("force", [False, True])
def test_completed_documents_are_skipped_or_refreshed_once_and_extraction_exits(
    inline_extraction, monkeypatch, force
):
    from lsa.evaluation import adapter
    from lsa.training.distillation.data.cache import layer_path

    run = inline_extraction
    run = dataclasses.replace(
        run,
        model=dataclasses.replace(
            run.model,
            indexer=IndexerConfig(
                window_size=16, sink_size=4, bucket_size=4, bucket_cap=8, pool_size=None, top_k=8, top_p=1
            ),
        ),
        data=dataclasses.replace(
            run.data,
            queries=teacher_config.QueryConfig(train=2, tail=1, top=1),
            tokenized=teacher_config.TokenizedConfig(lengths=[64], splits=[Split.TRAIN]),
        ),
    )
    planned = teacher_plan.plan(run, tokenizer=None)
    documents = teacher_plan.extracted_now(run, teacher_plan.flatten_plan(planned))
    root = Path(run.run.output)
    files = {
        entry.selected.name: layer_path(
            root, entry.documents.collection, entry.selected.split, entry.selected.name, 0
        )
        for entry in documents
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": "old"}, path)
    embedding = torch.nn.Embedding(2, 1)
    attention = torch.nn.Linear(1, 1)
    stack = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    model_adapter = SimpleNamespace(
        load=lambda *args, **kwargs: (model, None),
        text_model=lambda model: stack,
        rotary_layout=lambda *args: RotaryLayout("prefix", 64),
        attention_class=lambda attention, *args: SimpleNamespace(attention=attention),
    )
    seen = []

    def capture(stack, layers, ids, prefill):
        return {0: (ids, None)}

    def layer_blob(
        reader, hidden, position_embeddings, positions, run, name, picked, reading=None, probes=None
    ):
        assert name not in seen, "forced extraction refreshed a completed document more than once"
        seen.append(name)
        return {"version": "new"}

    monkeypatch.setattr(adapter, "adapter_for", lambda model: model_adapter)
    monkeypatch.setattr(teacher, "capture_layer_inputs", capture)
    monkeypatch.setattr(teacher, "layer_blob", layer_blob)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    assert teacher.extract(run, force=force, device="cpu") == root
    assert set(seen) == (set(files) if force else set())
    assert {torch.load(path, weights_only=True)["version"] for path in files.values()} == {
        "new" if force else "old"
    }
