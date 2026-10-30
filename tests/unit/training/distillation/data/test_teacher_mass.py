"""The dense attention-mass dump against a naive reference - what it keeps, how it splits and how it is written."""

from types import SimpleNamespace

import pytest
import torch

from lsa.evaluation.adapter.layer import TeacherView
from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.constants import Mass
from lsa.training.distillation.data import teacher_mass
from lsa.training.distillation.data.teacher_attention import teacher_attention
from lsa.training.distillation.data.teacher_config import (
    DataConfig,
    MassConfig,
    QueryConfig,
    RunConfig,
    TeacherConfig,
    TokenizedConfig,
)
from lsa.training.distillation.data.teacher_mass import MassRequest, dense_mass, mass_fields
from lsa.training.distillation.data.teacher_plan import nested_entries

SINK, WINDOW = 4, 32
HEADS, KEY_DIM, SCALE = 4, 8, 0.35
VISIBILITY = IndexerConfig(window_size=WINDOW, sink_size=SINK)


def a_layer(length, rows, seed=0):
    """Folded queries, a shared latent and the last `rows` positions of a document."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(HEADS, rows, KEY_DIM, generator=generator)
    k = torch.randn(1, length, KEY_DIM, generator=generator)
    return q, k, torch.arange(length - rows, length)


def naive(q, k, positions):
    """Every head's softmax over its causal keys [H, Q, T] and the logits it came from."""
    logits = torch.einsum("hqd,td->hqt", q, k[0]) * SCALE
    causal = torch.arange(k.shape[1])[None] <= positions[:, None]
    logits = logits.masked_fill(~causal[None], float("-inf"))
    return torch.softmax(logits, -1), logits


def test_every_head_softmaxes_over_all_causal_keys_and_the_whole_set_sums_to_one():
    q, k, positions = a_layer(300, 5)
    length = k.shape[1]
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(length, (length,)))
    # Keeping every key, the heads' summed and divided mass is one in every row, and so is each head's own
    assert rows[teacher_mass.SHARED_KEY_MASS].sum(-1) == pytest.approx(1.0, abs=1e-5)
    assert rows[teacher_mass.HEAD_TOP_MASS][:, 0] == pytest.approx(1.0, abs=1e-5)
    # A key past the row is never taken
    assert (rows[teacher_mass.SHARED_KEY_IDS] <= positions[:, None]).all() or (
        rows[teacher_mass.SHARED_KEY_MASS].masked_select(
            rows[teacher_mass.SHARED_KEY_IDS] > positions[:, None]
        )
        == 0
    ).all()


def test_the_masses_are_the_naive_softmax_summed_over_heads_and_divided_by_the_head_count():
    q, k, positions = a_layer(400, 6)
    attention, logits = naive(q, k, positions)
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(64, (16, 64)))
    expected = (attention.sum(0) / HEADS).topk(64, dim=-1)
    assert torch.equal(rows[teacher_mass.SHARED_KEY_IDS], expected.indices.int())
    assert torch.allclose(rows[teacher_mass.SHARED_KEY_MASS], expected.values, atol=1e-6)
    assert torch.allclose(rows[teacher_mass.HEAD_LSE], torch.logsumexp(logits, -1).t(), atol=1e-5)
    own = attention.topk(64, dim=-1).values
    assert torch.allclose(rows[teacher_mass.HEAD_TOP_MASS][:, 0], own[..., :16].sum(-1).t(), atol=1e-6)
    assert torch.allclose(rows[teacher_mass.HEAD_TOP_MASS][:, 1], own.sum(-1).t(), atol=1e-6)


def test_sink_window_and_the_rest_make_one_per_head_and_in_the_summed_set():
    q, k, positions = a_layer(300, 5)
    length = k.shape[1]
    attention, _ = naive(q, k, positions)
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(length, (length,)))
    keys = torch.arange(length)
    sink = keys < SINK
    window = (positions[:, None] - keys[None] < WINDOW) & ~sink[None] & (keys[None] <= positions[:, None])
    rest = attention.masked_fill(sink[None, None] | window[None], 0.0).sum(-1)
    head_sink, head_window = rows[teacher_mass.HEAD_SINK_MASS], rows[teacher_mass.HEAD_WINDOW_MASS]
    assert torch.allclose(head_sink + head_window + rest.t(), torch.ones(5, HEADS), atol=1e-5)
    assert torch.allclose(rows[teacher_mass.SINK_MASS], head_sink.mean(-1), atol=1e-6)
    assert torch.allclose(rows[teacher_mass.WINDOW_MASS], head_window.mean(-1), atol=1e-6)
    # The flagged entries of the shared set carry exactly the sink and window mass, and the unflagged the rest
    flags, mass = rows[teacher_mass.SHARED_KEY_FLAGS], rows[teacher_mass.SHARED_KEY_MASS]
    in_sink = (flags & Mass.SINK_BIT) > 0
    in_window = (flags & Mass.WINDOW_BIT) > 0
    assert not (in_sink & in_window).any()
    assert torch.allclose((mass * in_sink).sum(-1), rows[teacher_mass.SINK_MASS], atol=1e-6)
    assert torch.allclose((mass * in_window).sum(-1), rows[teacher_mass.WINDOW_MASS], atol=1e-6)
    unflagged = (mass * (flags == 0)).sum(-1)
    assert torch.allclose(
        rows[teacher_mass.SINK_MASS] + rows[teacher_mass.WINDOW_MASS] + unflagged, torch.ones(5), atol=1e-5
    )


def test_a_row_inside_the_sinks_and_the_window_still_splits_into_sink_window_and_rest():
    q, k, _ = a_layer(300, 5)
    positions = torch.arange(10, 15)
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(10, (10,)))
    # Nothing is double counted where the window overlaps the sinks - the window's share starts past them
    total = rows[teacher_mass.HEAD_SINK_MASS] + rows[teacher_mass.HEAD_WINDOW_MASS]
    assert torch.allclose(total, torch.ones(5, HEADS), atol=1e-5)


def test_the_sets_come_strongest_first_with_distinct_keys_and_the_head_masses_grow_with_their_count():
    q, k, positions = a_layer(600, 6)
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(128, (8, 32, 128)))
    for name in (teacher_mass.SHARED_KEY_MASS, teacher_mass.CACHE_KEY_MASS):
        assert (rows[name][:, 1:] <= rows[name][:, :-1]).all()
    for name in (teacher_mass.SHARED_KEY_IDS, teacher_mass.CACHE_KEY_IDS):
        assert all(len(set(row.tolist())) == 128 for row in rows[name])
    own = rows[teacher_mass.HEAD_TOP_MASS]
    assert (own[:, 1:] >= own[:, :-1]).all()
    assert (own <= 1 + 1e-5).all()


def test_the_cache_style_set_is_the_existing_extraction_s_at_two_thousand_and_forty_eight_keys():
    q, k, positions = a_layer(5000, 4)
    rows = dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(4096, (2048, 4096)))
    ids, mass, _ = teacher_attention(q, k, SCALE, positions, VISIBILITY, 2048)
    # The same keys with the same mass, in the same order but for keys the fp32 sum ties, which a top-k of another
    # size may break the other way
    first = rows[teacher_mass.CACHE_KEY_IDS][:, :2048]
    assert torch.equal(first.sort(-1).values, ids.sort(-1).values)
    assert torch.equal(rows[teacher_mass.CACHE_KEY_MASS][:, :2048].half(), mass)
    assert (first == ids).float().mean() > 0.99
    # And the sinks and the window are masked from it, as they are there
    taken = rows[teacher_mass.CACHE_KEY_IDS]
    assert (taken >= SINK).all()
    assert (positions[:, None] - taken >= WINDOW).all()


def test_the_dump_is_the_same_however_few_rows_a_block_holds(monkeypatch):
    q, k, positions = a_layer(500, 7)
    request = MassRequest(64, (16, 64))
    whole = dense_mass(q, k, SCALE, positions, VISIBILITY, request)
    monkeypatch.setattr(Mass, "SCORE_CELLS", HEADS * 500 * 2)
    narrow = dense_mass(q, k, SCALE, positions, VISIBILITY, request)
    for name, tensor in whole.items():
        assert torch.equal(tensor, narrow[name]), name


def test_a_document_shorter_than_what_a_set_keeps_is_refused():
    q, k, positions = a_layer(100, 2)
    with pytest.raises(ValueError, match="fewer keys than the 128"):
        dense_mass(q, k, SCALE, positions, VISIBILITY, MassRequest(128, (64,)))


def test_a_key_with_several_heads_is_refused():
    q, _, positions = a_layer(200, 2)
    with pytest.raises(ValueError, match="one latent shared by every head"):
        dense_mass(q, torch.randn(2, 200, KEY_DIM), SCALE, positions, VISIBILITY, MassRequest(16, (16,)))


def test_the_fields_have_the_shapes_and_dtypes_the_format_states():
    length, rows_count, top = 300, 5, 64
    q, k, positions = a_layer(length, rows_count)
    fields = mass_fields(q, k, SCALE, positions, VISIBILITY, MassRequest(top, (16, 64)))
    expected = {
        teacher_mass.QUERY_POSITIONS: ((rows_count,), torch.int32),
        teacher_mass.HEAD_QUERIES: ((rows_count, HEADS, KEY_DIM), torch.bfloat16),
        teacher_mass.KEY: ((length, KEY_DIM), torch.bfloat16),
        teacher_mass.SHARED_KEY_IDS: ((rows_count, top), torch.int32),
        teacher_mass.SHARED_KEY_MASS: ((rows_count, top), torch.float32),
        teacher_mass.SHARED_KEY_FLAGS: ((rows_count, top), torch.uint8),
        teacher_mass.SINK_MASS: ((rows_count,), torch.float32),
        teacher_mass.WINDOW_MASS: ((rows_count,), torch.float32),
        teacher_mass.HEAD_SINK_MASS: ((rows_count, HEADS), torch.float32),
        teacher_mass.HEAD_WINDOW_MASS: ((rows_count, HEADS), torch.float32),
        teacher_mass.HEAD_TOP_MASS: ((rows_count, 2, HEADS), torch.float32),
        teacher_mass.HEAD_LSE: ((rows_count, HEADS), torch.float32),
        teacher_mass.CACHE_KEY_IDS: ((rows_count, top), torch.int32),
        teacher_mass.CACHE_KEY_MASS: ((rows_count, top), torch.float32),
    }
    assert set(fields) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert tuple(fields[name].shape) == shape, name
        assert fields[name].dtype == dtype, name
        assert fields[name].is_contiguous() and fields[name].device.type == "cpu", name
    assert torch.equal(fields[teacher_mass.QUERY_POSITIONS], positions.int())


def test_the_stored_key_and_queries_reproduce_the_logits_the_lse_was_taken_over():
    q, k, positions = a_layer(300, 5)
    fields = mass_fields(q, k, SCALE, positions, VISIBILITY, MassRequest(64, (16, 64)))
    queries = fields[teacher_mass.HEAD_QUERIES].float().transpose(0, 1)
    logits = torch.einsum("hqd,td->hqt", queries, fields[teacher_mass.KEY].float()) * SCALE
    causal = torch.arange(300)[None] <= positions[:, None]
    lse = torch.logsumexp(logits.masked_fill(~causal[None], float("-inf")), -1).t()
    # Only the bf16 rounding of the stored operands separates them
    assert torch.allclose(lse, fields[teacher_mass.HEAD_LSE], atol=0.1)


def a_reader(q, k):
    """A layer wrapper whose view is a given folded query and latent."""
    attention = SimpleNamespace(layer_idx=7, scaling=SCALE, parameters=lambda: iter([torch.zeros(1)]))
    view = TeacherView(torch.zeros(1), torch.zeros(1), q, k)
    return SimpleNamespace(attention=attention, teacher_view=lambda hidden, positions, embeddings: view)


def test_a_dumped_layer_reads_back_whole_or_by_field_with_its_metadata(tmp_path):
    q, k, positions = a_layer(300, 5)
    request = MassRequest(64, (16, 64))
    path = teacher_mass.mass_path(tmp_path, "glm53_flash", "eval", "doc_l300_w0", 7)
    assert path == tmp_path / "glm53_flash" / "eval" / "doc_l300_w0" / "layer7.safetensors"
    teacher_mass.dump_layer(
        a_reader(q, k),
        torch.zeros(1, 300, 2),
        None,
        positions,
        path,
        "glm53_flash",
        "doc_l300_w0",
        VISIBILITY,
        request,
    )
    fields, metadata = teacher_mass.read_mass(path)
    assert torch.equal(fields[teacher_mass.KEY], k[0].bfloat16())
    assert metadata["format"] == Mass.FORMAT and metadata["version"] == str(Mass.VERSION)
    assert metadata["length"] == "300" and metadata["layer"] == "7"
    assert metadata["head_tops"] == "[16, 64]" and float(metadata["scale"]) == SCALE
    assert metadata["sink_size"] == str(SINK) and metadata["window_size"] == str(WINDOW)
    assert metadata["head_count"] == str(HEADS) and metadata["queries"] == "5"
    rows, _ = teacher_mass.read_mass(path, [teacher_mass.SINK_MASS])
    assert list(rows) == [teacher_mass.SINK_MASS]
    assert not list(path.parent.glob("*.tmp"))


def test_a_file_of_another_format_or_a_missing_field_is_refused(tmp_path):
    from safetensors.torch import save_file

    other = tmp_path / "other.safetensors"
    save_file({"a": torch.zeros(1)}, str(other), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="wanted 'dense_mass'"):
        teacher_mass.read_mass(other)
    q, k, positions = a_layer(300, 5)
    path = tmp_path / "layer7.safetensors"
    teacher_mass.dump_layer(
        a_reader(q, k),
        torch.zeros(1, 300, 2),
        None,
        positions,
        path,
        "m",
        "d",
        VISIBILITY,
        MassRequest(8, (8,)),
    )
    with pytest.raises(ValueError, match="holds no"):
        teacher_mass.read_mass(path, ["nothing"])


def entry(split, length, window=0, document="a"):
    return {
        "dataset": "d",
        "source": "s",
        "document": document,
        "split": split,
        "length": length,
        "window": window,
    }


def test_a_window_nests_into_every_shorter_length_from_its_first_token():
    nested = nested_entries([entry("eval", 1024, window=1), entry("train", 1024)], ["eval"], [256, 512, 1024])
    assert [(e["split"], e["length"], e["window"]) for e in nested] == [
        ("eval", 1024, 1),
        ("eval", 512, 2),
        ("eval", 256, 4),
        ("train", 1024, 0),
    ]
    # The shorter window starts where the longer does
    assert all(e["window"] * e["length"] == 1024 for e in nested[:3])


def test_a_length_that_does_not_divide_the_window_cannot_nest():
    with pytest.raises(ValueError, match="does not divide"):
        nested_entries([entry("eval", 1000)], ["eval"], [300])


def a_run(**data):
    return TeacherConfig(data=DataConfig(**data), run=RunConfig(output="out"))


def test_a_dump_may_sample_no_training_row_and_an_extraction_of_layer_files_may_not():
    a_run(queries=QueryConfig(train=0, tail=256), mass=MassConfig())
    with pytest.raises(ValueError, match="only for a dense mass dump"):
        a_run(queries=QueryConfig(train=0, tail=256))
    with pytest.raises(ValueError, match="must not be negative"):
        QueryConfig(train=-1)


def test_the_head_counts_of_a_dump_run_up_to_what_it_keeps():
    MassConfig(top=4096, head_tops=[2048, 4096])
    for bad in ([], [4096, 2048], [0, 8], [2048, 8192]):
        with pytest.raises(ValueError, match="head_tops"):
            MassConfig(top=4096, head_tops=bad)


def test_nesting_needs_the_manifest_the_lengths_and_the_sides():
    TokenizedConfig(documents="mix.json", lengths=[256], splits=["eval"], nest=True)
    with pytest.raises(ValueError, match="tokenized.nest needs"):
        TokenizedConfig(lengths=[256], splits=["eval"], nest=True)
