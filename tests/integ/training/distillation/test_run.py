"""The trainer end to end, on targets at test size and on a CPU."""

import json
import os
import signal

import pytest
import torch

from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation import trainer
from lsa.training.distillation.checkpoint import Preempted
from lsa.training.distillation.config import ServedDesign, load
from lsa.training.distillation.constants import Term
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.data.pool import Entry, Rolling, catalog, drawn_documents
from lsa.training.distillation.losses import step_terms
from lsa.training.distillation.rescore_maps import load_checkpoint
from lsa.training.distillation.trainer import input_reader, train
from tests.helpers.training.distillation.data import fake_attention, fake_cache

RUN = """
model:
  name: glm53_flash
  indexer:
    input: attention_input
    window_size: 32
    sink_size: 4
    top_k: 64
    index_dim: 8
data:
  root: {root}
  layers: [0]
  train:
    - {{name: natural, source: tokens, split: train}}
  evaluate:
    - {{name: natural, source: tokens, split: eval}}
run:
  name: unit
  output: {results}
  optim:
    steps: 4
    batch: 4
  report:
    every: 2
"""


def a_run(tmp_path, **overrides):
    cache = tmp_path / "cache"
    fake_cache.collection(cache, documents=("source0", "source1"), split="train")
    fake_cache.collection(cache, documents=("source2",), split="eval", seed=5)
    path = tmp_path / "run.yaml"
    path.write_text(RUN.format(root=cache, results=tmp_path / "results"))
    (experiment,) = load(path, [f"{key}={value}" for key, value in overrides.items()])
    return experiment


def test_a_run_writes_a_report_per_step_and_the_maps(tmp_path):
    experiment = a_run(tmp_path)
    out = train(experiment, device="cpu")
    assert (out / "maps.pt").exists()
    assert (out / "meta.json").exists()
    assert sorted(path.name for path in out.glob("report_step*.json")) == [
        "report_step0.json",
        "report_step2.json",
        "report_step4.json",
    ]


def test_a_served_design_reads_under_the_runs_window_sinks_and_width(tmp_path):
    experiment = a_run(tmp_path)
    flat = experiment.run.report.served.settings(experiment.model.indexer)["flat"]
    assert (flat.indexer.window_size, flat.indexer.sink_size) == (fake_cache.WINDOW, fake_cache.SINK)
    assert flat.indexer.index_dim == fake_cache.INDEX_DIM
    assert flat.indexer.top_k == IndexerConfig().top_k
    wider = ServedDesign(name="flat", indexer={"window_size": 2 * fake_cache.WINDOW})
    with pytest.raises(ValueError, match="those are the run's"):
        wider.settings(experiment.model.indexer)


def test_a_run_drawn_by_token_trains_to_the_end(tmp_path):
    out = train(a_run(tmp_path, **{"run.optim.draw": "token"}), device="cpu")
    assert (out / "maps.pt").exists()


def test_an_unknown_draw_is_refused(tmp_path):
    with pytest.raises(ValueError, match="optim.draw must be one of"):
        a_run(tmp_path, **{"run.optim.draw": "sentence"})


def test_the_untrained_report_is_a_pca_truncation(tmp_path):
    out = train(a_run(tmp_path, **{"run.optim.steps": 0}), device="cpu")
    report = json.loads((out / "report_step0.json").read_text())
    assert not (out / "report_step2.json").exists()
    # A 8-dim truncation of a 24-dim space loses something, and the report shows it rather than hiding it
    assert report["natural"]["recall16"] <= 1.0


def test_a_layer_the_targets_do_not_hold_says_so(tmp_path):
    with pytest.raises(ValueError, match="hold no layers"):
        train(a_run(tmp_path, **{"data.layers": "[5]"}), device="cpu")


def test_the_temperature_is_calibrated_per_layer_and_recorded(tmp_path):
    out = train(a_run(tmp_path), device="cpu")
    report = json.loads((out / "report_step0.json").read_text())
    assert set(report["tau"]) == {"0"}
    assert report["tau"]["0"] > 0


def test_a_pinned_temperature_is_used_as_given(tmp_path):
    out = train(a_run(tmp_path, **{"model.space.tau": 0.25}), device="cpu")
    assert json.loads((out / "report_step0.json").read_text())["tau"]["0"] == pytest.approx(0.25)


def test_the_meta_records_what_produced_the_numbers(tmp_path):
    out = train(a_run(tmp_path), device="cpu")
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["model_dim"], meta["index_dim"]) == (fake_cache.MODEL_DIM, 8)
    assert meta["layers"] == [0]
    assert meta["run"]["model"]["indexer"]["index_dim"] == 8
    assert meta["run"]["run"]["output"] == str(tmp_path / "results")


def test_a_warm_start_continues_from_the_checkpoint_rather_than_recalibrating(tmp_path):
    first = train(a_run(tmp_path / "a", **{"model.space.tau": 0.25}), device="cpu")
    trained = json.loads((first / "report_step4.json").read_text())["tau"]["0"]
    second = train(a_run(tmp_path / "b", **{"model.space.init": str(first / "maps.pt")}), device="cpu")
    assert json.loads((second / "report_step0.json").read_text())["tau"]["0"] == pytest.approx(trained)


def test_a_warm_start_from_a_checkpoint_of_another_index_dim_says_so(tmp_path):
    first = train(a_run(tmp_path / "a"), device="cpu")
    second = a_run(
        tmp_path / "b", **{"model.indexer.index_dim": 4, "model.space.init": str(first / "maps.pt")}
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        train(second, device="cpu")


def test_a_warm_start_requires_the_same_teacher(tmp_path):
    first = train(a_run(tmp_path / "a", **{"run.optim.steps": 0}), device="cpu")
    checkpoint = first / "maps.pt"
    artifact = torch.load(checkpoint, weights_only=False)
    artifact["teacher"]["model"] = "qwen35_4b"
    torch.save(artifact, checkpoint)
    second = a_run(tmp_path / "b", **{"model.space.init": str(checkpoint)})

    with pytest.raises(ValueError, match="was distilled for"):
        train(second, device="cpu")


def test_a_snapshot_lands_beside_every_report(tmp_path):
    """The intervals a later run can start from, each with the report that says how it selects"""
    out = train(a_run(tmp_path), device="cpu")
    assert sorted(path.name for path in out.glob("maps_step*.pt")) == [
        "maps_step0.pt",
        "maps_step2.pt",
        "maps_step4.pt",
    ]


def test_a_snapshot_records_the_step_that_produced_it(tmp_path):
    out = train(a_run(tmp_path), device="cpu")
    for step in (0, 2, 4):
        assert torch.load(out / f"maps_step{step}.pt", weights_only=False)["step"] == step
    assert torch.load(out / "maps.pt", weights_only=False)["step"] == 4


def test_the_finished_maps_are_the_last_snapshot(tmp_path):
    """`maps.pt` cannot carry a step in its name, because its presence is what marks a run complete"""
    out = train(a_run(tmp_path), device="cpu")
    final = torch.load(out / "maps.pt", weights_only=False)["maps"]
    last = torch.load(out / "maps_step4.pt", weights_only=False)["maps"]
    assert final.keys() == last.keys()
    for layer, saved in final.items():
        for name, weight in saved.items():
            assert torch.equal(weight, last[layer][name])


def test_a_snapshot_warm_starts_a_run_the_way_the_finished_maps_do(tmp_path):
    """A snapshot is a map like any other, so a mid-run step is something a later stage can read"""
    first = train(a_run(tmp_path / "a"), device="cpu")
    second = a_run(tmp_path / "b", **{"model.space.init": str(first / "maps_step2.pt")})
    assert (train(second, device="cpu") / "maps.pt").exists()


def test_every_saved_map_names_the_model_it_was_distilled_for(tmp_path):
    """Nothing about a map's shape says which model's states trained it, so the artifact has to"""
    out = train(a_run(tmp_path), device="cpu")
    for name in ("maps.pt", "maps_step0.pt", "maps_step2.pt", "maps_step4.pt"):
        teacher = torch.load(out / name, weights_only=False)["teacher"]
        assert teacher["model"] == fake_cache.MODEL
        assert teacher["layers"] == [0]
        assert teacher["collection"] == "tokens"


def test_the_meta_carries_the_teacher_too(tmp_path):
    out = train(a_run(tmp_path), device="cpu")
    assert json.loads((out / "meta.json").read_text())["teacher"]["model"] == fake_cache.MODEL


@pytest.mark.parametrize("side", ["train", "evaluate"])
def test_every_target_collection_must_match_the_teacher(tmp_path, side):
    experiment = a_run(tmp_path)
    split = "train" if side == "train" else "eval"
    fake_cache.collection(
        experiment.data.root,
        name="other",
        documents=("other0",),
        split=split,
        model="qwen35_4b",
    )
    collection = type(experiment.data.train[0])(name="other", source="other", split=split)
    getattr(experiment.data, side).append(collection)

    with pytest.raises(ValueError, match="extracted from"):
        train(experiment, device="cpu")


def test_targets_from_another_model_are_refused(tmp_path):
    """Two models of one dim share the shape of every projection, so only the name the targets carry catches this"""
    with pytest.raises(ValueError, match="were extracted from 'glm53_flash'"):
        train(a_run(tmp_path, **{"model.name": "qwen35_4b"}), device="cpu")


def test_a_run_reproduces_whatever_the_threads_did(tmp_path):
    """The pool reads and stages in background threads, and the sequence of steps is worked out before any of them runs"""
    first = torch.load(train(a_run(tmp_path / "a"), device="cpu") / "maps.pt", weights_only=False)["maps"]
    second = torch.load(train(a_run(tmp_path / "b"), device="cpu") / "maps.pt", weights_only=False)["maps"]

    for layer, state in first.items():
        for name, value in state.items():
            assert torch.equal(second[layer][name], value), f"layer {layer} {name}"


def test_a_pool_that_holds_one_document_still_trains_the_run(tmp_path):
    """A pool smaller than a window of documents narrows the window to what it holds rather than stalling"""
    small = a_run(tmp_path, **{"data.pool_gb": 0.000001})
    assert (train(small, device="cpu") / "maps.pt").exists()


def test_a_set_can_score_the_training_rows_of_training_documents(tmp_path):
    sets = (
        "[{name: natural, source: tokens, split: eval},"
        " {name: seen, source: tokens, split: train, rows: train, limit: 1}]"
    )
    out = train(a_run(tmp_path, **{"data.evaluate": sets}), device="cpu")
    report = json.loads((out / "report_step0.json").read_text())
    assert {"natural", "seen"} <= set(report)
    assert 0.0 <= report["seen"]["recall16"] <= 1.0


def test_a_set_that_names_rows_it_cannot_read_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rows"):
        a_run(tmp_path, **{"data.evaluate": "[{name: natural, source: tokens, split: eval, rows: head}]"})


def test_a_run_of_no_steps_initialises_calibrates_and_exports_the_maps(tmp_path):
    out = train(a_run(tmp_path, **{"run.optim.steps": 0}), device="cpu")
    assert (out / "maps.pt").exists()
    assert (out / "meta.json").exists()
    assert json.loads((out / "report_step0.json").read_text())["tau"]["0"] > 0


def test_a_draw_of_no_steps_is_empty():
    entries = [Entry(("root", "tokens", "train"), "source0", 10)]
    assert drawn_documents(entries, 0, 0, "token") == []
    assert drawn_documents(entries, 0, 0, "document") == []


def test_init_holds_one_document_at_a_time(tmp_path, monkeypatch):
    live, peak, loads = set(), [0], [0]

    class Counted(Document):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            loads[0] += 1
            live.add(id(self))
            peak[0] = max(peak[0], len(live))

        def __del__(self):
            live.discard(id(self))

    monkeypatch.setattr(trainer, "Document", Counted)
    experiment = a_run(tmp_path)
    maps = trainer.build_maps(experiment, catalog(experiment), [0], "cpu")
    # Each document is read once for its rows and once to calibrate, and never two together
    assert loads[0] == 4
    assert peak[0] == 1
    assert maps[0].tau > 0


KEY_AND_QUERY = {"model.indexer.input": "key_and_query", "model.indexer.query_heads": "all"}


def training_loss(experiment, maps_path, reader):
    """The key term over every training row of the training documents, under the maps at that path"""
    maps = load_checkpoint(experiment, str(maps_path), [0], "cpu", reader)
    losses = []
    for entry in catalog(experiment):
        document = reader(Document(entry.path(0), experiment.model.indexer))
        with torch.no_grad():
            terms = step_terms(maps[0], document, document.train_rows(), experiment.run.objective)
        losses.append(float(terms[Term.KEY]))
    return sum(losses) / len(losses)


def test_a_run_on_the_key_and_query_trains_its_loss_down_and_saves_maps_the_server_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trainer, "load_attention", fake_attention.load)
    steps = 300
    experiment = a_run(tmp_path, **KEY_AND_QUERY, **{"run.optim.steps": steps, "run.report.every": steps})
    out = train(experiment, device="cpu")
    reader = input_reader(experiment, [0], "cpu")

    assert training_loss(experiment, out / "maps.pt", reader) < training_loss(
        experiment, out / "maps_step0.pt", reader
    )
    assert json.loads((out / "meta.json").read_text())["model_dim"] == fake_attention.RANK
    report = json.loads((out / f"report_step{steps}.json").read_text())
    assert 0.0 < report["natural"]["recall16"] <= 1.0


# What a test that stops a run puts back before resuming it
ORIGINAL_NEXT = Rolling.next


def finished_maps(out):
    return torch.load(out / "maps.pt", weights_only=False)["maps"]


def assert_same_maps(first, second):
    assert sorted(first) == sorted(second)
    for layer, state in first.items():
        for name, value in state.items():
            assert torch.equal(second[layer][name], value), f"layer {layer} {name}"


def stop_at(monkeypatch, step, how="signal"):
    """Stop a run as its step is drawn - slurm's SIGUSR2, which lets the step finish, or a crash, which does not"""
    original = Rolling.next

    def drawn(pool, at, interrupt=None):
        if at == step and how == "signal":
            os.kill(os.getpid(), signal.SIGUSR2)
        if at == step and how == "crash":
            raise RuntimeError("the node went away")
        return original(pool, at, interrupt)

    monkeypatch.setattr(Rolling, "next", drawn)


@pytest.mark.parametrize("indexer_input", [{}, KEY_AND_QUERY])
def test_a_preempted_run_resumes_to_the_maps_an_uninterrupted_run_writes(
    tmp_path, monkeypatch, indexer_input
):
    monkeypatch.setattr(trainer, "load_attention", fake_attention.load)
    whole = train(a_run(tmp_path / "whole", **indexer_input), device="cpu")
    experiment = a_run(tmp_path / "stopped", **indexer_input)

    stop_at(monkeypatch, 3)
    with pytest.raises(Preempted):
        train(experiment, device="cpu")
    monkeypatch.setattr(Rolling, "next", ORIGINAL_NEXT)
    out = trainer.run_directory(experiment)
    assert torch.load(out / "resume.pt", weights_only=False)["step"] == 3
    assert not (out / "maps.pt").exists()

    train(experiment, device="cpu")

    assert_same_maps(finished_maps(whole), finished_maps(out))
    assert json.loads((out / "report_step4.json").read_text()) == json.loads(
        (whole / "report_step4.json").read_text()
    )


def test_a_run_that_died_resumes_from_its_last_snapshot_exactly(tmp_path, monkeypatch):
    whole = train(a_run(tmp_path / "whole"), device="cpu")
    experiment = a_run(tmp_path / "died")

    stop_at(monkeypatch, 3, how="crash")
    with pytest.raises(RuntimeError, match="went away"):
        train(experiment, device="cpu")
    monkeypatch.setattr(Rolling, "next", ORIGINAL_NEXT)
    out = trainer.run_directory(experiment)
    assert torch.load(out / "resume.pt", weights_only=False)["step"] == 2

    train(experiment, device="cpu")

    assert_same_maps(finished_maps(whole), finished_maps(out))


def test_a_run_resumes_only_under_the_configuration_that_stopped_and_force_starts_it_over(
    tmp_path, monkeypatch
):
    whole = train(a_run(tmp_path / "whole"), device="cpu")
    stop_at(monkeypatch, 3)
    with pytest.raises(Preempted):
        train(a_run(tmp_path / "stopped"), device="cpu")
    monkeypatch.setattr(Rolling, "next", ORIGINAL_NEXT)

    with pytest.raises(ValueError, match="another configuration"):
        train(a_run(tmp_path / "stopped", **{"run.optim.lr": 0.5}), device="cpu")
    out = train(a_run(tmp_path / "stopped"), force=True, device="cpu")

    assert_same_maps(finished_maps(whole), finished_maps(out))


def test_the_signal_goes_on_to_the_job_once_the_state_is_written(tmp_path, monkeypatch):
    """submitit's own handler requeues the job, and must see the state already on disk"""
    experiment = a_run(tmp_path)
    out = trainer.run_directory(experiment)
    seen = []
    previous = signal.signal(signal.SIGUSR2, lambda number, _frame: seen.append((out / "resume.pt").exists()))
    stop_at(monkeypatch, 3)
    try:
        with pytest.raises(Preempted):
            train(experiment, device="cpu")
    finally:
        signal.signal(signal.SIGUSR2, previous)

    assert seen == [True]


def test_a_requeued_job_resumes_rather_than_starting_over(tmp_path):
    experiment = a_run(tmp_path)

    submission = trainer.train.checkpoint(experiment, True, "cuda:0", [0])

    assert submission.function is train
    assert submission.args == (experiment, False, "cuda:0", [0])
