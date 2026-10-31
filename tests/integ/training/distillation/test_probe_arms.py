"""The offline arms of a selection study, read over a probe cache at test size on a CPU and written as records and tables.

The served kernels need a GPU, so the designs the arms read at are not served here - an arm reads a design's sizes - and
the one test that holds a reference arm to the served read is marked `gpu`.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from lsa.io import read_json
from lsa.training.distillation import arms, served_read
from lsa.training.distillation.config import ArmConfig, DocumentSet, ServedDesign, load
from lsa.training.distillation.data.cache import Document, layer_path, write
from tests.helpers.training.distillation.data import fake_attention, fake_probes

RUN = """
model:
  name: glm53_flash
  space:
    init: {init}
  indexer:
    window_size: {window}
    sink_size: {sink}
    top_k: 64
    index_dim: {index_dim}
    input: key_and_query
    query_heads: all
data:
  root: {root}
  layers: [0]
  train:
    - {{name: probes, source: ruler, split: eval}}
  evaluate:
    - {{name: probes, source: ruler, split: eval}}
run:
  name: probes
  output: {results}
  report:
    served:
      rows: 100
      designs:
        - name: tree
          indexer: {{top_k: 64, top_p: 16, top_level: 2, node_bucket_size: 4, node_cluster_block_size: 16, beam_width: 32}}
      arms:
        - {{name: A0, design: tree}}
        - {{name: A1, design: tree, node_cluster_algorithm: position}}
        - {{name: A2, design: tree, cluster_algorithm: position, node_cluster_algorithm: position}}
        - {{name: A3, design: tree, node_bucket_score: lse, tau: 0.05}}
        - {{name: A4, design: tree, node_bucket_score: lse, tau_scale: [0.5, 2]}}
        - {{name: A5, design: tree, cluster_algorithm: position, node_cluster_algorithm: position, node_bucket_score: lse, bucket_score: lse}}
        - {{name: B1, kind: pisa, space: attention, block_size: 16, keep_count: 8}}
        - {{name: B2, kind: pisa, space: attention, block_size: 16, keep_count: 8, beam_width: 16}}
        - {{name: B3, kind: pisa, space: index, design: tree, block_size: 16, keep_count: 8}}
"""
LAYER = 0


@pytest.fixture
def small(monkeypatch):
    """Levels that grow inside a 4,096 key document - pages of 512 keys and a level taken at once"""
    monkeypatch.setattr(arms, "PAGE_SIZE", 512)
    monkeypatch.setattr(arms, "GROWTH_BUCKETS", 0)


def identity_map(path):
    eye = torch.eye(fake_probes.KEY_DIM)
    torch.save({f"{LAYER}.W_Q_idx": eye, f"{LAYER}.W_K_idx": eye}, path)
    path.with_suffix(".json").write_text(json.dumps({"tau": {str(LAYER): 0.05}}))
    return path


def a_run(tmp_path, model="glm53_flash", make=fake_probes.blob):
    cache = fake_probes.collection(tmp_path / "cache", make=make)
    path = tmp_path / "run.yaml"
    path.write_text(
        RUN.replace("glm53_flash", model).format(
            init=identity_map(tmp_path / "identity.pt"),
            root=cache,
            results=tmp_path / "results",
            window=fake_probes.WINDOW,
            sink=fake_probes.SINK,
            index_dim=fake_probes.KEY_DIM,
        )
    )
    (experiment,) = load(path, [])
    return experiment


def test_the_arms_are_named_in_the_config_and_read_at_a_served_designs_sizes(tmp_path):
    experiment = a_run(tmp_path)
    served = experiment.run.report.served
    assert [arm.name for arm in served.arms] == ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3"]
    assert served.arms[0].kind == "reference" and served.arms[6].kind == "pisa"
    tree = served.settings(experiment.model.indexer)["tree"].indexer
    assert (tree.window_size, tree.sink_size, tree.top_level, tree.at("beam_width", 2)) == (64, 4, 2, 32)


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        (dict(kind="pisa", space="index"), "names none"),
        (dict(kind="reference"), "names none"),
        (dict(design="tree", cluster_algorithm="hash"), "cluster_algorithm"),
        (dict(design="tree", bucket_score="max"), "bucket_score"),
        (dict(design="tree", tau=0.0), "tau"),
        (dict(kind="tree"), "kind"),
    ],
)
def test_an_arm_that_names_nothing_it_can_read_is_refused(arm, message):
    with pytest.raises(ValueError, match=message):
        ArmConfig(name="x", **arm)


def test_an_arm_that_names_a_design_the_run_does_not_serve_is_refused():
    from lsa.training.distillation.config import ServedConfig

    with pytest.raises(ValueError, match="not served designs"):
        ServedConfig(designs=[ServedDesign(name="flat")], arms=[ArmConfig(name="A0", design="tree")])
    with pytest.raises(ValueError, match="share a name"):
        ServedConfig(designs=[ServedDesign(name="flat")], arms=[ArmConfig(name="flat", design="flat")])


@pytest.fixture
def scored(tmp_path, small):
    """Every arm's record of one needle prompt, and the document it was read from"""
    experiment = a_run(tmp_path)
    served = experiment.run.report.served
    designs = served.settings(experiment.model.indexer)
    spaces = served_read.design_spaces(experiment, designs)
    maps = served_read.load_map(experiment.model.space.init, [LAYER], "cpu", "key_and_query")
    path = layer_path(tmp_path / "cache", "ruler", "eval", "needle0", LAYER)
    document = Document(path, experiment.model.indexer, native=True)
    record = served_read.score_layer(
        document,
        {space: maps[LAYER] for space in set(spaces.values())},
        {},
        spaces,
        served_read.exact_budgets(experiment, {}, spaces),
        served.rows,
        fake_attention.attention(),
        False,
        arms.ArmSet(served.arms, designs, {name: space.projections for name, space in spaces.items()}),
        "needle",
    )
    return record, document


def test_every_arm_is_read_over_the_same_rows_and_scored_beside_the_exact_read(scored):
    record, _ = scored
    assert list(record["means"]) == [
        "exact",
        "native",
        "A0",
        "A1",
        "A2",
        "A3",
        "A4@tau0.5",
        "A4@tau2",
        "A5",
        "B1",
        "B2",
        "B3",
    ]
    assert record["rows"] == 5 and record["task"] == "needle"
    for name in ("A0", "A1", "A2", "A3", "A4@tau0.5", "A4@tau2", "A5", "B1", "B2", "B3"):
        means = record["means"][name]
        assert 0.0 <= means["mass"] <= 1.0
        assert means["scored_vectors"] > 0
        # Every arm reads the window and the sinks, so no read is shorter than they are
        assert means["ids_read"] >= fake_probes.WINDOW


def test_the_exact_read_and_the_flat_start_of_the_tree_hold_the_needle_every_probe_looks_for(scored):
    record, _ = scored
    assert record["means"]["exact"]["target_held"] == 1.0
    assert record["means"]["exact"]["line_held"] == pytest.approx(3 / 7)
    assert record["means"]["A0"]["target_any"] == 1.0
    assert record["means"]["native"]["target_held"] == 1.0


def test_a_probe_record_keeps_each_rows_kind_targets_and_distance_and_every_reads_metrics_a_row(scored):
    record, document = scored
    probe = record["probe"]
    assert probe["query_kinds"] == ["question", "answer", "control"]
    assert probe["rows"]["query_kind"] == fake_probes.KINDS
    assert probe["rows"]["target_count"] == [3, 3, 3, 3, 0]
    assert probe["rows"]["target_distance"] == [3000 - 1002, 4090 - 1002, 4091 - 1002, 4092 - 1002, -1]
    held = probe["reads"]["A0"]["target_held"]
    assert len(held) == 5 and math.isnan(held[4])
    assert set(probe["reads"]) == set(record["means"])
    assert document.target_key_ids is not None


def test_a_pisa_query_head_dots_every_vector_its_kv_head_scored(scored):
    record, _ = scored
    reads = record["probe"]["reads"]
    for vectors, dots in zip(reads["B1"]["scored_vectors"], reads["B1"]["scored_dots"], strict=True):
        assert dots == vectors * fake_probes.HEADS
    # In the index space the heads are summed first - one dot a vector
    assert reads["B3"]["scored_dots"] == reads["B3"]["scored_vectors"]
    assert reads["A0"]["scored_dots"] == reads["A0"]["scored_vectors"]


def test_an_lse_score_reads_each_vector_beneath_what_it_ranks_so_it_costs_more(scored):
    record, _ = scored
    assert record["means"]["A3"]["scored_vectors"] > record["means"]["A0"]["scored_vectors"]
    assert record["means"]["A5"]["scored_vectors"] > record["means"]["A3"]["scored_vectors"]


def test_the_structures_of_an_arm_are_built_once_per_grouping_at_the_sizes_of_its_design(tmp_path):
    experiment = a_run(tmp_path)
    indexer = experiment.run.report.served.settings(experiment.model.indexer)["tree"].indexer
    document = Document(
        layer_path(tmp_path / "cache", "ruler", "eval", "needle0", LAYER), experiment.model.indexer
    )
    structures = arms.Structures(document.key)
    ward = structures.buckets(indexer, "ward")[1]
    positional = structures.buckets(indexer, "position")[1]
    assert structures.buckets(indexer, "ward")[1] is ward
    assert torch.equal(positional.bucket_members[0, :8].long(), torch.arange(8))
    assert not torch.equal(ward.bucket_members, positional.bucket_members)
    levels = structures.hierarchy(indexer, "ward", "position")
    assert [len(children) for children in levels.children] == [128, 32]
    assert structures.hierarchy(indexer, "ward", "position") is levels


def test_a_logit_scale_is_the_attentions_unless_the_arm_names_its_own(tmp_path, small):
    experiment = a_run(tmp_path)
    document = Document(
        layer_path(tmp_path / "cache", "ruler", "eval", "needle0", LAYER),
        experiment.model.indexer,
        native=True,
    )
    rows = torch.arange(len(document.query_positions))
    arm_set = arms.ArmSet(
        [ArmConfig(name="B1", kind="pisa", block_size=16, keep_count=8)],
        experiment.run.report.served.settings(experiment.model.indexer),
        {},
    )
    inputs = served_read.key_and_query_of(document, rows, None, "maps.pt")
    with pytest.raises(ValueError, match="no logit scale"):
        arms.read_arms(arm_set, {}, document, rows, lambda: inputs, None)
    read = arms.read_arms(arm_set, {}, document, rows, lambda: inputs, 0.25)["B1"]
    assert read.selected_key_ids.shape[0] == len(rows)


def test_a_swept_arm_is_read_once_at_each_multiple_of_the_maps_temperature_and_each_named_one(tmp_path):
    experiment = a_run(tmp_path)
    arm = ArmConfig(name="A3", design="tree", node_bucket_score="lse", tau_scale=[0.5, 1, 2], taus=[0.3])
    names = [(v.name, v.tau, v.scale) for v in arms.expand_arms([arm, ArmConfig(name="A0", design="tree")])]
    assert names == [
        ("A3@tau0.5", None, 0.5),
        ("A3@tau1", None, 1),
        ("A3@tau2", None, 2),
        ("A3@tau=0.3", 0.3, 1.0),
        ("A0", None, 1.0),
    ]
    with pytest.raises(ValueError, match="reads at none"):
        arms.expand_arms([ArmConfig(name="A0", design="tree", tau_scale=[2])])
    with pytest.raises(ValueError, match="sets tau and sweeps one"):
        ArmConfig(name="A3", design="tree", node_bucket_score="lse", tau=0.1, tau_scale=[2])
    assert experiment.run.report.served.arms[4].tau_scale == [0.5, 2]


def test_an_arm_reads_at_the_temperature_of_its_maps_sidecar_and_a_map_without_one_says_how_to_calibrate(
    tmp_path,
):
    experiment = a_run(tmp_path)
    document = Document(
        layer_path(tmp_path / "cache", "ruler", "eval", "needle0", LAYER),
        experiment.model.indexer,
        native=True,
    )
    rows = torch.arange(len(document.query_positions))
    designs = experiment.run.report.served.settings(experiment.model.indexer)
    spaces = served_read.design_spaces(experiment, designs)
    maps = served_read.load_map(experiment.model.space.init, [LAYER], "cpu", "key_and_query")
    indexed = {
        "tree": served_read.project_key_and_query(
            maps[LAYER], served_read.key_and_query_of(document, rows, None, "m"), 1
        )
    }
    arm = ArmConfig(name="A3", design="tree", node_bucket_score="lse")
    held = {"tree": spaces["tree"].projections}
    inputs = lambda: None  # noqa: E731
    read = arms.read_arms(arms.ArmSet([arm], designs, held), indexed, document, rows, inputs, None)
    assert list(read) == ["A3"]
    Path(spaces["tree"].projections).with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="lsa distill calibrate-tau"):
        arms.read_arms(arms.ArmSet([arm], designs, held), indexed, document, rows, inputs, None)


def test_a_shard_writes_the_records_and_the_tables_with_the_error_over_prompts(tmp_path, small, monkeypatch):
    monkeypatch.setattr(served_read, "load_attention", fake_attention.load)
    experiment = a_run(tmp_path)
    documents = served_read.held_out(experiment)
    out = served_read.score(experiment, 0, 1, documents, device="cpu")
    assert [entry["source"] for entry in documents] == ["cwe", "cwe", "needle", "needle"]
    written = read_json(served_read.record_path(out, "needle1", LAYER))
    assert written["task"] == "needle" and written["key_count"] == fake_probes.LENGTH
    # A record's length is its document's cell, not the keys the prompt holds
    assert written["length"] == next(entry["length"] for entry in documents if entry["name"] == "needle1")
    report = (out / "report.md").read_text()
    assert "## target_held by task and length - question queries" in report
    assert "| A0 |" in report and "| B1 |" in report
    assert "Prompts a column: cwe 4k: 2, needle 4k: 2." in report
    assert "## target_held by target count - cwe 4k, answer queries" in report
    summary = read_json(out / "summary.json")
    cell = summary["probe"]["question"]["target_held"]["exact"]["needle 4k"]
    assert cell["prompts"] == 2 and cell["mean"] == 1.0
    assert "scored_vectors" in summary["probe"]["question"]


def test_a_shard_over_a_grouped_query_cache_reads_its_own_keys_and_loads_no_attention(
    tmp_path, small, monkeypatch
):
    monkeypatch.setattr(served_read, "load_attention", refuse_attention)
    experiment = a_run(tmp_path, model="qwen38_27b", make=fake_probes.grouped_blob)
    documents = served_read.held_out(experiment)
    out = served_read.score(experiment, 0, 1, documents, device="cpu")
    written = read_json(served_read.record_path(out, "needle1", LAYER))
    assert {"B1", "B1_union"} <= set(written["means"])
    assert written["means"]["B1"]["target_held"] == 1.0


def test_a_shard_whose_train_set_the_cache_does_not_hold_is_refused_before_it_scores(tmp_path):
    experiment = a_run(tmp_path)
    experiment.data.train = [DocumentSet()]
    documents = served_read.held_out(experiment)
    with pytest.raises(ValueError, match="holds no layer"):
        served_read.score(experiment, 0, 1, documents, device="cpu")
    assert not (experiment.out / "report.md").exists()


def refuse_attention(*args, **kwargs):
    raise AssertionError("a cache that keeps the key and query loads no attention")


@pytest.mark.gpu
def test_the_reference_arm_of_the_design_reads_what_the_served_design_reads(tmp_path):
    experiment = a_run(tmp_path)
    length = 1 << 16
    rows = [40000, 50000, 60000, 65000, length - 1]
    path = layer_path(tmp_path / "cache", "ruler", "eval", "long0", LAYER)
    write(path, fake_probes.blob(length=length, rows=rows, name="long0"))
    served = experiment.run.report.served
    tree = {"top_k": 64, "top_p": 16, "top_level": 2, "node_bucket_size": 4, "node_cluster_block_size": 256}
    served.designs = [ServedDesign(name="tree", indexer={**tree, "beam_width": 32})]
    served.arms = [ArmConfig(name="A0", design="tree", agrees_with="tree")]
    designs = served.settings(experiment.model.indexer)
    spaces = served_read.design_spaces(experiment, designs)
    maps = served_read.load_map(experiment.model.space.init, [LAYER], "cuda:0", "key_and_query")
    document = Document(path, experiment.model.indexer, "cuda:0", native=True)
    record = served_read.score_layer(
        document,
        {space: maps[LAYER] for space in set(spaces.values())},
        designs,
        spaces,
        {},
        served.rows,
        fake_attention.attention(),
        True,
        arms.ArmSet(served.arms, designs, {name: space.projections for name, space in spaces.items()}),
        "needle",
    )
    agreed = record["means"]["A0"]["agreement"]
    print(f"A0 holds {agreed:.4f} of the served tree's long-range keys")
    assert agreed >= 0.9
    assert record["means"]["A0"]["target_held"] == record["means"]["tree"]["target_held"]


@pytest.fixture
def grouped(tmp_path):
    """A grouped-query probe document under `queries.attention`, read without a native selection"""
    path = layer_path(tmp_path, "ruler", "eval", "needle0", LAYER)
    path.parent.mkdir(parents=True)
    write(path, fake_probes.grouped_blob())
    return Document(path, fake_probes.indexer())


def test_pisa_on_a_grouped_query_cache_reads_one_list_a_kv_head_and_each_holds_the_needle(grouped):
    rows = torch.arange(len(grouped.query_positions))
    arm = ArmConfig(name="B1", kind="pisa", block_size=16, keep_count=8)
    read = arms.read_arms(arms.ArmSet([arm], {}, {}), {}, grouped, rows, pisa_reads_no_latent, None)["B1"]
    assert grouped.native_key_ids is None
    assert read.selected_key_ids.shape[:2] == (len(rows), fake_probes.KV_HEADS)
    for kv_head in range(fake_probes.KV_HEADS):
        assert torch.isin(torch.tensor(fake_probes.NEEDLE), read.selected_key_ids[:, kv_head]).all()
    # Each query head dots every vector its own key-value head scored
    group_size = fake_probes.GROUPED_HEADS // fake_probes.KV_HEADS
    assert torch.equal(read.scored_dots, read.scored_vectors * group_size)


def test_a_read_a_kv_head_is_scored_each_list_averaged_and_their_union(grouped):
    rows = torch.arange(len(grouped.query_positions))
    arm = ArmConfig(name="B1", kind="pisa", block_size=16, keep_count=8)
    read = arms.read_arms(arms.ArmSet([arm], {}, {}), {}, grouped, rows, pisa_reads_no_latent, None)["B1"]
    metrics = served_read.ranked_row_metrics(grouped, rows, read.selected_key_ids, "B1")
    assert set(metrics) == {"B1", "B1_union"}
    assert torch.allclose(metrics["B1"]["target_held"][fake_probes.KINDS.index(1)], torch.tensor(1.0))


def test_the_attention_space_reads_at_the_recorded_scale_unless_the_arm_names_its_own(grouped):
    rows = torch.arange(2)
    arm = ArmConfig(name="B1", kind="pisa")
    keys, queries, scale = arms.attention_inputs(arm, grouped, rows, pisa_reads_no_latent, None)
    assert keys.shape == (fake_probes.LENGTH, fake_probes.KV_HEADS, fake_probes.HEAD_DIM)
    assert queries.shape == (2, fake_probes.GROUPED_HEADS, fake_probes.HEAD_DIM)
    assert scale == fake_probes.ATTENTION_SCALE
    named = ArmConfig(name="B1", kind="pisa", logit_scale=0.5)
    assert arms.attention_inputs(named, grouped, rows, pisa_reads_no_latent, None)[2] == 0.5


def pisa_reads_no_latent():
    raise AssertionError("a grouped-query cache reads its own keys, not GLM's latent")
