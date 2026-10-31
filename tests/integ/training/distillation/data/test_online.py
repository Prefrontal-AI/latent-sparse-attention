"""The online teacher - a document it reads is the document an extraction writes and a pool reads back.

A tiny random decoder with grouped-query attention and a partial rotation stands in for Qwen3.8-27B, on a CPU and in
bf16 as the model runs, so the input an extraction caches at bf16 is the input the teacher projects: the
extraction writes its cache, the teacher reads the same documents without one, and every field has to agree.
"""

import dataclasses
import errno
import json
import os
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, StableLmConfig, StableLmForCausalLM

from lsa.baselines import qsa
from lsa.sglang.settings.design import IndexerConfig, QSAConfig
from lsa.training.distillation import trainer
from lsa.training.distillation.checkpoint import Preempted
from lsa.training.distillation.config import DocumentSet, TeacherSource, load
from lsa.training.distillation.constants import Cache
from lsa.training.distillation.data import online
from lsa.training.distillation.data.cache import Document, cached_layers, layer_path
from lsa.training.distillation.data.native import KeyAndQueryReader
from lsa.training.distillation.data.online import OnlineTeacher, extraction_of
from lsa.training.distillation.data.pool import Entry, Rolling, shared_order
from lsa.training.distillation.data.teacher import extract
from lsa.training.distillation.data.teacher_config import (
    DataConfig,
    QueryConfig,
    RunConfig,
    TeacherConfig,
    TeacherModelConfig,
    TokenizedConfig,
)
from lsa.training.distillation.data.teacher_nodes import Request, Staging
from lsa.training.distillation.data.teacher_plan import extracted_now
from lsa.training.distillation.trainer import train
from tests.helpers.training.data.fake_corpus import write_corpus
from tests.helpers.training.distillation.data import fake_cache

VOCAB, LENGTH, WINDOW, SINK = 97, 256, 32, 4


def tiny_model(path):
    """Save a random decoder of two layers - four query heads on two key-value heads of 16 dims, 4 of them rotated."""
    config = StableLmConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        partial_rotary_factor=0.25,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    StableLmForCausalLM(config).save_pretrained(path)
    vocabulary = {f"t{index}": index for index in range(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer).save_pretrained(path)
    return path


def corpus(path):
    """A corpus every document of which is held out, its ids folded into the tiny vocabulary."""
    card = write_corpus(path, sources=("arxiv",), documents=8, rows_per_document=1, sample_length=LENGTH)
    rows = np.fromfile(path / "arxiv.bin", dtype=np.uint32)
    (rows % VOCAB).astype(np.uint32).tofile(path / "arxiv.bin")
    return card


def an_extraction(tmp_path, prefill=0):
    """Write the extraction a run's teacher names, and return its yaml."""
    extraction = TeacherConfig(
        model=TeacherModelConfig(
            name=str(tiny_model(tmp_path / "model")),
            dtype="bfloat16",
            indexer=IndexerConfig(window_size=WINDOW, sink_size=SINK),
        ),
        data=DataConfig(
            tokenized=TokenizedConfig(
                path=str(corpus(tmp_path / "corpus").path),
                sources=["arxiv"],
                train=3,
                evaluate=2,
                held_out_fraction=1.0,
            ),
            queries=QueryConfig(train=8, tail=4, top=16),
        ),
        run=RunConfig(output=str(tmp_path / "cache"), name="tiny", prefill=prefill),
    )
    path = tmp_path / "extraction.yaml"
    path.write_text(OmegaConf.to_yaml(asdict(extraction)))
    return extraction, TeacherSource(extraction=str(path))


# Reads an extraction's documents online and compares every field with the cache it wrote: the online teacher
# equals the cache it stands in for
def compare(source: TeacherSource, indexer_input: str, device: str = "cuda:0") -> dict[str, Any]:
    """Read an extraction's documents online and compare every field with the cache it wrote.

    The cache is the one an extraction wrote from the same config into its
    `run.output`, and its documents are the ones that extraction reads now - its lengths and sides. The online side is
    one copy of the teacher on `device`, reading each document as a run would.

    Args:
        source: The extraction and its overrides, as a run's `data.teacher` names them.
        indexer_input: The run's input - `x` alone, or the key and query read off it on both sides.
        device: Where the copy goes.

    Returns:
        The summary - per field, the documents that matched exactly and the largest difference, and the top keys' least
        and mean overlap - and a record per
        (document, layer).
    """

    extraction = extraction_of(source)
    indexer = IndexerConfig(
        sink_size=extraction.model.indexer.sink_size,
        window_size=extraction.model.indexer.window_size,
        input=indexer_input,
    )
    teacher = OnlineTeacher(source, indexer, extraction.model.name)
    documents = extracted_now(extraction, sorted(teacher.planned.values(), key=lambda one: one.index))
    sides = {(one.documents.collection, one.selected.split) for one in documents}
    held = set.intersection(
        *(
            set(cached_layers(teacher.root, DocumentSet(source=collection, split=split)))
            for collection, split in sides
        )
    )
    layers = [layer for layer in teacher.layers if layer in held]
    teacher.start([device], layers)
    replica = teacher.replicas[0]

    records = []
    for planned in documents:
        online = teacher.read_document(replica, planned, layers)
        for layer in layers:
            where = layer_path(
                teacher.root,
                planned.documents.collection,
                planned.selected.split,
                planned.selected.name,
                layer,
            )
            cached = Document(where, indexer, Cache.HOST)
            if replica.reading is not None:
                reader = KeyAndQueryReader({layer: replica.readers[layer].attention}, reading=replica.reading)
                cached = reader(cached.to(replica.inputs)).to(Cache.HOST)
            record: dict[str, Any] = {
                "document": planned.selected.name,
                "layer": layer,
                "length": cached.key_count,
            }
            record.update(compared(online[layer], cached))
            records.append(record)
    teacher.close()

    fields = [name for name in records[0] if name not in ("document", "layer", "length")] if records else []
    summary: dict[str, Any] = {"documents": len(documents), "layers": layers, "records": len(records)}
    for name in fields:
        values = [record[name] for record in records]
        if isinstance(values[0], bool):
            summary[name] = sum(bool(value) for value in values)
        elif name.endswith("_min"):
            summary[name] = min(values)
        elif name.endswith("_mean"):
            summary[name] = sum(values) / len(values)
        else:
            summary[name] = max(values)
    return summary


def compared(online: Document, cached: Document) -> dict[str, Any]:
    """Compare one (document, layer) read online with the same one read off the cache.

    Args:
        online: The teacher's.
        cached: The cache's, under the same input.

    Returns:
        Per field, whether it is identical - and for the masses and the inputs the largest difference, and for the
        teacher's keys the share of each row's set the two agree on.
    """
    result: dict[str, Any] = {}
    for name in ("query_positions", "top_key_ids", "top_key_mass", "peak_key_id", "x", "key", "summed_query"):
        mine, theirs = getattr(online, name), getattr(cached, name)
        if mine is None and theirs is None:
            continue
        if mine is None or theirs is None or mine.shape != theirs.shape:
            result[f"{name}_identical"] = False
            continue
        result[f"{name}_identical"] = bool(torch.equal(mine, theirs))
        if mine.is_floating_point():
            result[f"{name}_max_difference"] = float((mine.float() - theirs.float()).abs().max())
    if online.top_key_ids.shape == cached.top_key_ids.shape:
        shared = [
            len(set(mine.tolist()) & set(theirs.tolist())) / max(len(theirs), 1)
            for mine, theirs in zip(online.top_key_ids, cached.top_key_ids, strict=True)
        ]
        result["top_key_overlap_min"] = min(shared)
        result["top_key_overlap_mean"] = sum(shared) / max(len(shared), 1)
    return result


@pytest.mark.parametrize("prefill", [0, 96])
@pytest.mark.parametrize("indexer_input", ["attention_input", "key_and_query"])
def test_the_teacher_reads_every_field_the_extraction_wrote(tmp_path, prefill, indexer_input):
    extraction, source = an_extraction(tmp_path, prefill)
    extract(extraction, device="cpu")

    summary = compare(source, indexer_input, device="cpu")

    assert summary["records"] == summary["documents"] * len(summary["layers"]) > 0
    identical = [name for name in summary if name.endswith("_identical")]
    assert identical
    assert all(summary[name] == summary["records"] for name in identical)
    assert summary["top_key_overlap_min"] == 1.0


def test_under_the_key_and_query_a_document_holds_every_key_head_side_by_side_and_no_x(tmp_path):
    _, source = an_extraction(tmp_path)
    indexer = IndexerConfig(window_size=WINDOW, sink_size=SINK, input="key_and_query")
    teacher = OnlineTeacher(source, indexer, source_model(source))
    teacher.start(["cpu"], teacher.layers)
    (planned, *_) = teacher.planned.values()

    read = teacher.read_document(teacher.replicas[0], planned, teacher.layers)
    teacher.close()

    document = read[teacher.layers[0]]
    # Two key heads of 16 dims, 4 of them rotated and dropped
    assert document.x is None
    assert document.key.shape == (LENGTH, 2 * 12)
    assert document.summed_query.shape == (len(document.query_positions), 2 * 12)


@pytest.mark.parametrize("prefill", [0, 96])
def test_a_qsa_indexer_on_the_key_and_query_holds_the_lsa_inputs_pooled_and_qsas_targets(tmp_path, prefill):
    """The same document read for an LSA map on the key and query, for a QSA indexer on `x`, and for one on the key and
    query: the last holds the first's keys pooled into blocks and its summed queries, and the second's targets"""
    _, source = an_extraction(tmp_path, prefill)
    model = source_model(source)
    lsa = IndexerConfig(window_size=WINDOW, sink_size=SINK, input="key_and_query")
    on_x = QSAConfig(query_head_count=2, head_dim=8, rotary_dim=4, block_size=4, top_k=64)
    on_key_and_query = dataclasses.replace(on_x, input="key_and_query")
    read = {}
    for name, qsa_indexer in (("lsa", None), ("x", on_x), ("key_and_query", on_key_and_query)):
        teacher = OnlineTeacher(source, lsa, model, qsa_indexer=qsa_indexer)
        teacher.start(["cpu"], teacher.layers)
        (planned, *_) = teacher.planned.values()
        read[name] = teacher.read_document(teacher.replicas[0], planned, teacher.layers)
        teacher.close()

    for layer, document in read["key_and_query"].items():
        lsa_document, x_document = read["lsa"][layer], read["x"][layer]
        assert document.x is None and document.block_x is None
        # Two key heads of 16 dims, 4 of them rotated and dropped, pooled four keys a block
        assert document.block_key.shape == (LENGTH // 4, 2 * 12)
        assert document.model_dim == 2 * 12
        torch.testing.assert_close(
            document.block_key.float(), qsa.pool_blocks(lsa_document.key, 4), atol=2e-2, rtol=2e-2
        )
        assert torch.equal(document.summed_query, lsa_document.summed_query)
        assert torch.equal(
            document.query_inputs(document.train_rows()), lsa_document.summed_query[document.train_rows()]
        )
        for name in ("query_positions", "top_key_ids", "top_key_mass", "peak_key_id"):
            assert torch.equal(getattr(document, name), getattr(x_document, name)), name
    # A key and query block is a sixth of an `x` block here, as on Qwen3.8-27B
    x_bytes = next(iter(read["x"].values())).block_x.element_size() * read["x"][layer].block_x.shape[1]
    assert document.block_key.element_size() * document.block_key.shape[1] < x_bytes


def test_the_layers_meet_their_documents_in_one_order_the_teacher_reads_them_in():
    order = shared_order([3, 1, 3, 0, 2, 1], seed=0)
    assert sorted(order) == [0, 1, 2, 3]
    assert shared_order([3, 1, 3, 0, 2, 1], seed=0) == order


RUN = """
model:
  name: {model}
  indexer:
    window_size: 32
    sink_size: 4
    top_k: 64
    index_dim: 8
    input: key_and_query
    query_heads: all
data:
  teacher:
    extraction: {extraction}
    ahead_gb: 0
  train:
    - {{name: natural, source: corpus, split: train}}
  evaluate:
    - {{name: natural, source: corpus, split: eval}}
run:
  name: online
  output: {results}
  optim:
    steps: 6
    batch: 4
    draw: token
  report:
    every: 3
"""


def test_a_run_taught_online_trains_and_reports_without_a_cache(tmp_path):
    _, source = an_extraction(tmp_path)
    path = tmp_path / "run.yaml"
    path.write_text(
        RUN.format(model=source_model(source), extraction=source.extraction, results=tmp_path / "results")
    )
    (experiment,) = load(path)

    out = train(experiment, device="cpu")

    assert (out / "maps.pt").exists()
    assert not (tmp_path / "cache").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["teacher"]["online"]["extraction"] == source.extraction
    report = json.loads((out / "report_step6.json").read_text())
    assert 0.0 <= report["natural"]["recall16"] <= 1.0


def test_a_run_taught_online_resumes_to_the_maps_an_uninterrupted_run_writes(tmp_path, monkeypatch):
    """The documents a stopped step had already trained on are read again, and read the same"""
    _, source = an_extraction(tmp_path)
    runs = {}
    for name in ("whole", "stopped"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            RUN.format(model=source_model(source), extraction=source.extraction, results=tmp_path / name)
        )
        (runs[name],) = load(path)
    whole = train(runs["whole"], device="cpu")
    original = Rolling.next

    def drawn(pool, step, interrupt=None):
        if step == 4:
            os.kill(os.getpid(), signal.SIGUSR2)
        return original(pool, step, interrupt)

    monkeypatch.setattr(Rolling, "next", drawn)
    with pytest.raises(Preempted):
        train(runs["stopped"], device="cpu")
    monkeypatch.setattr(Rolling, "next", original)
    out = train(runs["stopped"], device="cpu")

    first = torch.load(whole / "maps.pt", weights_only=False)["maps"]
    second = torch.load(out / "maps.pt", weights_only=False)["maps"]
    for layer, state in first.items():
        for name, value in state.items():
            assert torch.equal(second[layer][name], value), f"layer {layer} {name}"
    assert json.loads((out / "report_step6.json").read_text()) == json.loads(
        (whole / "report_step6.json").read_text()
    )


def online_runs(tmp_path, *names):
    """The online run, once per name, each into its own directory, over one extraction"""
    _, source = an_extraction(tmp_path)
    runs = {}
    for name in names:
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            RUN.format(model=source_model(source), extraction=source.extraction, results=tmp_path / name)
        )
        (runs[name],) = load(path)
    return runs


def test_a_run_whose_documents_another_node_reads_trains_the_maps_one_node_does(tmp_path, monkeypatch):
    """The trainer's node keeps no copy here, so every document goes through the shared disk"""
    runs = online_runs(tmp_path, "one_node", "two_nodes")
    one_node = train(runs["one_node"], device="cpu")
    monkeypatch.setattr(trainer, "teacher_devices", lambda device: ["cpu"] if device is None else [])
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    out = trainer.run_directory(runs["two_nodes"])
    serving = Staging(out / "teacher_staging" / "7-0", 1, 2, 1e12)
    server = threading.Thread(target=trainer.serve, args=(runs["two_nodes"], serving), daemon=True)
    server.start()

    two_nodes = train(runs["two_nodes"], device="cpu")
    server.join(timeout=60)

    assert not server.is_alive()
    first = torch.load(one_node / "maps.pt", weights_only=False)["maps"]
    second = torch.load(two_nodes / "maps.pt", weights_only=False)["maps"]
    for layer, state in first.items():
        for name, value in state.items():
            assert torch.equal(second[layer][name], value), f"layer {layer} {name}"
    # Every document was taken off the disk and let go
    assert not list((serving.root / "ready").iterdir())
    assert len(list((serving.root / "claims").iterdir())) > 0


def test_a_serving_node_returns_once_the_trainer_finishes_with_every_copy_ended(tmp_path, monkeypatch):
    """A copy still reading when the trainer finishes leaves its document, and no copy outlives the serving task"""
    (run,) = online_runs(tmp_path, "two_nodes").values()
    monkeypatch.setattr(trainer, "teacher_devices", lambda device: ["cpu"])
    root = trainer.run_directory(run) / "teacher_staging" / "7-0"
    trainers = Staging(root, 0, 2, 1e12)
    trainers.open()
    collection, name = next(iter(OnlineTeacher(run.data.teacher, run.model.indexer, run.model.name).planned))
    serving = []

    class Serving(OnlineTeacher):
        def start(self, devices, layers):
            serving.append(self)
            super().start(devices, layers)

    reading = threading.Event()
    original = online.read_through

    def read_until_closed(*args, **kwargs):
        reading.set()
        while not serving[0]._stop:
            time.sleep(0.01)
        original(*args, **kwargs)

    monkeypatch.setattr(trainer, "OnlineTeacher", Serving)
    monkeypatch.setattr(online, "read_through", read_until_closed)
    server = threading.Thread(target=trainer.serve, args=(run, Staging(root, 1, 2, 1e12)), daemon=True)
    server.start()
    trainers.publish([Request(0, collection, name, trainer.resolve_layers(run))])
    assert reading.wait(timeout=60)

    trainers.finish()
    server.join(timeout=60)

    assert not server.is_alive()
    assert not [thread.name for thread in threading.enumerate() if thread.name.startswith("teacher-")]
    assert not list((root / "ready").iterdir())


def source_model(source):
    """Return the model the extraction a source names reads."""
    return OmegaConf.load(source.extraction).model.name


def a_trainers_teacher(tmp_path):
    """The trainer node's teacher over a launch's staging, no copy loaded, and the serving node's view of the staging."""
    (run,) = online_runs(tmp_path, "two_nodes").values()
    root = tmp_path / "teacher_staging" / "7-0"
    trainers, server = Staging(root, 0, 2, 1e12), Staging(root, 1, 2, 1e12)
    trainers.open()
    teacher = OnlineTeacher(run.data.teacher, run.model.indexer, run.model.name, trainers)
    return teacher, server


def a_staged_blob(layer):
    """One layer of a document as a serving node leaves it."""
    blob = fake_cache.document()
    blob["meta"] = {**blob["meta"], "layer": layer}
    return blob


def test_the_trainer_frees_a_staged_document_once_without_listing_the_shared_disk(tmp_path, monkeypatch):
    """Each layer's pool takes its layer on a thread of its own, and listing a directory another thread removes is a
    stale handle on NFS - so the take that empties the document frees it, and none lists the directory"""
    teacher, server = a_trainers_teacher(tmp_path)
    planned = next(iter(teacher.planned.values()))
    key = (planned.documents.collection, planned.selected.name)
    layers = list(range(8))
    teacher.submit([key], {key: layers})
    assert server.claim(0)
    server.put(0, {layer: a_staged_blob(layer) for layer in layers})

    ready = server.root / "ready"
    listing = Path.iterdir

    def stale(path):
        if ready in path.parents:
            raise OSError(errno.ESTALE, "Stale file handle", str(path))
        return listing(path)

    monkeypatch.setattr(Path, "iterdir", stale)
    entry = Entry((teacher.root, key[0], planned.selected.split), key[1], LENGTH)
    taken, failures = [], []

    def take(layer):
        try:
            taken.append(teacher.take(entry, layer))
        except BaseException as failure:
            failures.append(failure)

    threads = [threading.Thread(target=take, args=(layer,)) for layer in layers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not failures
    assert sorted(document.layer for document in taken) == layers
    assert not server.ready(0)
    assert not (ready / "0").exists()
    teacher.close()


def test_a_copy_reads_past_its_bound_only_for_a_document_no_node_has_claimed(tmp_path):
    """A pool waiting on a document another node claimed gets it from the shared disk, so the copies here hold"""
    teacher, server = a_trainers_teacher(tmp_path)
    first, second = list(teacher.planned)[:2]
    teacher.submit([first, second], {first: [0], second: [0]})
    teacher._ahead = 0
    teacher._waiting[(*first, 0)] = 1

    with teacher._cond:
        assert teacher._may_read()
    assert server.claim(0)
    with teacher._cond:
        assert not teacher._may_read()

    single = OnlineTeacher(teacher.source, teacher.indexer, teacher.extraction.model.name)
    single.submit([first, second], {first: [0], second: [0]})
    single._ahead = 0
    single._waiting[(*first, 0)] = 1
    with single._cond:
        assert single._may_read()
    teacher.close()
    single.close()


def test_past_its_bound_a_copy_reads_the_document_a_pool_waits_on_not_the_queues_head(tmp_path):
    """With the shared staging full no node claims, so the waited document is unclaimed and deeper in the queue - a
    copy past the bound reads it, and not the head, which only fills the host"""
    teacher, _ = a_trainers_teacher(tmp_path)
    first, second, third = list(teacher.planned)[:3]
    teacher.submit([first, second, third], {first: [0], second: [0], third: [0]})
    teacher._waiting[(*third, 0)] = 1

    teacher._ahead = 1e12
    with teacher._cond:
        index, planned, _ = teacher._next_request()
    assert (planned.documents.collection, planned.selected.name) == first

    teacher._ahead = 0
    with teacher._cond:
        assert teacher._may_read()
        index, planned, _ = teacher._next_request()
    assert (planned.documents.collection, planned.selected.name) == third
    assert index == 2
    assert [(one.documents.collection, one.selected.name) for _, one, _ in teacher._queue] == [second]
    with teacher._cond:
        assert not teacher._may_read()
    teacher.close()
