"""Scoring an exported map's served read - the documents, the shards, the records, the tables, and the report's served read."""

import json

import pytest
import torch

from lsa.io import read_json, write_json
from lsa.sglang.index.layer_index import KeyAndQuery, KeyAndQueryProjection
from lsa.training.distillation import served_read
from lsa.training.distillation.config import ServedDesign, load
from lsa.training.distillation.data.cache import Document, layer_path, native_path, write
from lsa.training.distillation.data.native import key_and_query
from lsa.training.distillation.export import write as export
from lsa.training.distillation.reporting import reads_own_space
from lsa.training.distillation.trainer import train
from tests.helpers.training.distillation.data import fake_attention, fake_cache

RUN = """
model:
  name: glm53_flash
  space:
    init: {init}
  indexer:
    input: attention_input
    window_size: {window}
    sink_size: 4
    top_k: 64
    index_dim: {index_dim}
data:
  root: {root}
  layers: [0]
  train:
    - {{name: natural, source: tokens, split: train}}
  evaluate:
    - {{name: natural, source: tokens, split: eval}}
    - {{name: first, source: tokens, split: eval, limit: 1}}
run:
  name: unit
  output: {results}
  optim:
    steps: {steps}
    batch: 4
  report:
    every: 2
    served:
      rows: 3
      designs:
        - {{name: flat}}
        - {{name: flat_k32, indexer: {{top_k: 32, top_p: 8}}}}
"""
EVAL = ("source2", "source3", "source4")
# Ward's kernel clusters at least 16 pools a block and index vectors at least 16 wide
SERVED_WINDOW, SERVED_INDEX_DIM = 64, 16


@pytest.fixture
def servable(monkeypatch):
    """A fake cache whose window a served read can cluster"""
    monkeypatch.setattr(fake_cache, "WINDOW", SERVED_WINDOW)
    return {"window": SERVED_WINDOW, "index_dim": SERVED_INDEX_DIM}


def a_run(
    tmp_path, init="null", steps=2, window=fake_cache.WINDOW, index_dim=fake_cache.INDEX_DIM, **overrides
):
    cache = tmp_path / "cache"
    if not cache.exists():
        fake_cache.collection(cache, documents=("source0", "source1"), split="train")
        fake_cache.collection(cache, documents=EVAL, split="eval", seed=5)
    path = tmp_path / "run.yaml"
    path.write_text(
        RUN.format(
            root=cache,
            results=tmp_path / "results",
            init=init,
            steps=steps,
            window=window,
            index_dim=index_dim,
        )
    )
    (experiment,) = load(path, [f"{key}={value}" for key, value in overrides.items()])
    return experiment


def with_native_picks(tmp_path):
    """GLM's picks beside each held-out layer file - the teacher's first keys, the rest empty"""
    for name in EVAL:
        path = layer_path(tmp_path / "cache", "tokens", "eval", name, 0)
        top_key_ids = torch.load(path, weights_only=False)["top_key_ids"]
        native_key_ids = torch.cat([top_key_ids[:, :8], torch.full_like(top_key_ids[:, :8], -1)], dim=1)
        write(native_path(path), {"native_key_ids": native_key_ids.int(), "meta": {}})


def record(document, layer, length, flat_mass, native_mass):
    metrics = dict.fromkeys(served_read.METRICS, 0.0)
    return {
        "document": document,
        "layer": layer,
        "length": length,
        "means": {"flat": {**metrics, "mass": flat_mass}, "native": {**metrics, "mass": native_mass}},
        "seconds": {"flat": 1.0},
    }


def test_the_documents_are_every_held_out_one_of_the_sets_each_once(tmp_path):
    documents = served_read.held_out(a_run(tmp_path))
    assert [entry["name"] for entry in documents] == list(EVAL)
    assert served_read.document_path(documents[0], 0).exists()


def test_a_dry_run_names_the_documents_it_would_score_and_submits_nothing(tmp_path, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run built an executor")

    a_run(tmp_path, init=tmp_path / "exported.pt")
    write_json(tmp_path / "documents.json", ["source3", {"name": "source4"}])
    monkeypatch.setattr("lsa.slurm.executor", refuse)
    command = [
        "lsa distill served-read",
        str(tmp_path / "run.yaml"),
        "--documents",
        str(tmp_path / "documents.json"),
    ]
    monkeypatch.setattr("sys.argv", [*command, "--shards", "2", "--dry-run"])
    served_read.main()
    assert "unit: 2 documents, designs ['flat', 'flat_k32']" in capsys.readouterr().out
    write_json(tmp_path / "documents.json", ["source9"])
    with pytest.raises(ValueError, match="source9"):
        served_read.main()


def test_shards_are_dealt_about_equal_lengths_longest_first():
    documents = [{"name": f"x{i}", "length": length} for i, length in enumerate([8, 1, 1, 4, 4, 2])]
    shards = served_read.deal(documents, 2)
    assert sorted(sum(item["length"] for item in shard) for shard in shards) == [10, 10]
    assert sorted(item["name"] for shard in shards for item in shard) == sorted(
        item["name"] for item in documents
    )


def test_a_length_averages_its_documents_and_all_weighs_each_length_alike(tmp_path):
    write_json(served_read.record_path(tmp_path, "a", 3), record("a", 3, 16384, 0.2, 0.3))
    write_json(served_read.record_path(tmp_path, "b", 3), record("b", 3, 16384, 0.4, 0.3))
    write_json(served_read.record_path(tmp_path, "c", 3), record("c", 3, 1 << 20, 0.9, 0.5))
    _, summary = served_read.tables(tmp_path)
    flat = summary["mass_by_length"]["flat"]
    assert flat["16k"] == pytest.approx(0.3)
    assert flat["1M"] == 0.9
    assert flat["all"] == pytest.approx(0.6)


def test_kept_rows_split_into_the_tail_and_the_rest_a_record_at_a_time(tmp_path):
    kept = record("a", 3, 16384, 0.55, 0.5)
    kept["per_row"] = {
        "row_ids": [0, 1, 2, 3],
        "tail_from": 2,
        "positions": [10, 20, 30, 31],
        "reads": {"flat": {"mass": [0.2, 0.4, 0.9, 0.7]}},
    }
    write_json(served_read.record_path(tmp_path, "a", 3), kept)

    _, summary = served_read.tables(tmp_path)

    assert summary["mass_by_region"]["flat"] == pytest.approx({"tail": 0.8, "rest": 0.3, "all": 0.55})


def test_a_design_under_native_by_more_than_the_margin_is_flagged(tmp_path):
    write_json(served_read.record_path(tmp_path, "a", 7), record("a", 7, 32768, 0.25, 0.3))
    write_json(served_read.record_path(tmp_path, "b", 11), record("b", 11, 32768, 0.29, 0.3))
    served_read.write_tables(tmp_path)
    flagged = read_json(tmp_path / "summary.json")["below_native"]
    assert len(flagged) == 1 and "layer 7" in flagged[0]


@pytest.mark.gpu
def test_a_scored_shard_writes_a_record_a_document_and_the_tables_once_all_are_written(tmp_path, servable):
    maps = train(a_run(tmp_path, **servable, **{"run.name": "trained"}), device="cpu") / "maps.pt"
    exported = tmp_path / "exported.pt"
    export(maps, exported)
    with_native_picks(tmp_path)
    experiment = a_run(tmp_path, init=exported, steps=0, **servable)
    documents = served_read.held_out(experiment)
    out = served_read.score(experiment, 0, 2, documents)
    assert not (out / "summary.json").exists()
    served_read.score(experiment, 1, 2, documents)
    written = json.loads(served_read.record_path(out, "source2", 0).read_text())
    assert list(written["means"]) == ["flat", "flat_k32", "exact", "exact_k4096", "exact_k32", "native"]
    assert written["rows"] == 3
    assert written["means"]["exact"]["recall"] >= written["means"]["exact_k32"]["recall"]
    assert set(read_json(out / "summary.json")["mass_by_length"]) == set(written["means"])


@pytest.mark.gpu
def test_a_report_on_a_gpu_scores_each_served_design_beside_the_exact_read(tmp_path, servable):
    out = train(a_run(tmp_path, **servable), device="cuda:0")
    report = json.loads((out / "report_step2.json").read_text())
    assert set(report["natural"]["served"]) == {"flat", "flat_k32"}
    for read in report["natural"]["served"].values():
        assert 0.0 < read["mass_top"] <= 1.0
        assert 0.0 < read["recall"] <= 1.0


def test_a_design_on_its_own_map_or_input_reads_in_its_own_space_with_an_exact_read_of_its_own(tmp_path):
    experiment = a_run(tmp_path, init=tmp_path / "exported.pt")
    latent = {"input": "key_and_query"}
    experiment.run.report.served.designs = [
        ServedDesign(name="flat"),
        ServedDesign(name="flat_k32", indexer={"top_k": 32, "top_p": 8}),
        ServedDesign(name="latent", indexer=latent, serving={"projections": "latent.pt"}),
        ServedDesign(
            name="latent_p8", indexer={**latent, "top_p": 512}, serving={"projections": "latent.pt"}
        ),
        ServedDesign(name="latent_one", indexer=latent, serving={"projections": "latent.pt"}, ranks=1),
    ]
    designs = experiment.run.report.served.settings(experiment.model.indexer)
    spaces = served_read.design_spaces(experiment, designs)
    run = served_read.Space(str(tmp_path / "exported.pt"))
    assert spaces["flat"] == spaces["flat_k32"] == run
    assert spaces["latent"] == spaces["latent_p8"] == served_read.Space("latent.pt", "key_and_query", 8)
    assert spaces["latent_one"] == served_read.Space("latent.pt", "key_and_query", 1)
    budgets = served_read.exact_budgets(experiment, designs, spaces)
    assert budgets == {
        "exact": (run, 64),
        "exact_k4096": (run, 4096),
        "exact_k32": (run, 32),
        "exact_latent": (spaces["latent"], 4096),
        "exact_latent_one": (spaces["latent_one"], 4096),
    }
    assert reads_own_space(designs["latent"], experiment.model.indexer)
    assert not reads_own_space(designs["flat_k32"], experiment.model.indexer)


def test_a_run_on_the_key_and_query_serves_its_designs_on_it_one_list_of_every_head(tmp_path):
    exported = tmp_path / "exported.pt"
    experiment = a_run(
        tmp_path,
        init=exported,
        **{"model.indexer.input": "key_and_query", "model.indexer.query_heads": "all"},
    )
    designs = experiment.run.report.served.settings(experiment.model.indexer)
    assert (designs["flat"].indexer.input, designs["flat"].indexer.query_heads) == ("key_and_query", "all")
    spaces = served_read.design_spaces(experiment, designs)
    run = served_read.Space(str(exported), "key_and_query", 1)
    assert spaces["flat"] == spaces["flat_k32"] == run
    assert served_read.exact_budgets(experiment, designs, spaces)["exact"] == (run, 64)
    assert not reads_own_space(designs["flat"], experiment.model.indexer)


def test_a_design_whose_heads_split_over_no_rank_is_refused():
    with pytest.raises(ValueError, match="ranks must be positive"):
        ServedDesign(name="latent", ranks=0)


@torch.no_grad()
def test_the_key_is_the_normed_latent_and_each_heads_query_is_folded_through_its_own_key_map():
    attention = fake_attention.attention(dtype=torch.float32)
    x = torch.randn(10, 24)
    positions = torch.tensor([2, 7])
    inputs = key_and_query(attention, x, positions)
    assert torch.equal(inputs.key, attention.kv_a_layernorm(attention.kv_a_proj_with_mqa(x)).bfloat16())
    q = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(x[positions]))).view(2, 4, 8)
    for head in range(4):
        # Head h's block of kv_b_proj is its 8 key rows, then its 4 value rows
        key_map = attention.kv_b_proj.weight[head * 12 : head * 12 + 8]
        assert torch.allclose(inputs.query[:, head].float(), q[:, head] @ key_map, atol=1e-2, rtol=1e-2)


def test_each_rank_sums_its_own_run_of_heads():
    projection = KeyAndQueryProjection(torch.randn(16, 8), torch.randn(16, 8), "cpu")
    inputs = KeyAndQuery(torch.randn(5, 16).bfloat16(), torch.randn(3, 4, 16).bfloat16())
    index_queries, index_keys = served_read.project_key_and_query(projection, inputs, ranks=2)
    assert index_queries.shape == (3, 2, 8)
    assert torch.equal(index_queries[:, 1], projection.index_queries(inputs.query[:, 2:]))
    assert torch.equal(index_keys, projection.index_keys(inputs.key))


def test_the_union_of_the_ranks_lists_holds_each_key_they_read_once():
    read = torch.tensor([[[5, 3, -1], [3, 9, 5]], [[-1, -1, -1], [2, -1, 2]]])
    merged = served_read.union(read)
    assert [sorted(row[row >= 0].tolist()) for row in merged] == [[3, 5, 9], [2]]


@torch.no_grad()
def test_a_file_holding_the_key_and_query_serves_the_heads_it_would_have_read_off_x(tmp_path):
    fake_cache.collection(tmp_path, documents=("source0",))
    path = layer_path(tmp_path, "tokens", "train", "source0", 0)
    attention = fake_attention.attention()
    indexer = fake_cache.config(input="key_and_query")
    rows = torch.tensor([0, 3, 9])
    off_x = served_read.key_and_query_of(Document(path, indexer), rows, attention, "maps.pt")
    fake_cache.keep_key_and_query(path, attention)
    kept = served_read.key_and_query_of(Document(path, indexer), rows, None, "maps.pt")
    assert torch.equal(kept.key, off_x.key) and torch.equal(kept.query, off_x.query)
