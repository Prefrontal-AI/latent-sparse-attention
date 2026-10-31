"""The MLA adapter, against the real transformers module rather than a fake of it.

GLM-5.3-Flash is 306GB and does not fit on one GPU, but its attention is ordinary python: the same
`Glm5NextTextAttention` the checkpoint runs is built here at a size that fits in a test, with random
weights. That is what says `project` unpacks the MLA latent the way the model does - the module's own
`expand_kv` is what produces the keys and values, so a mistake shows up as a number, not a shape.

Skipped without the evaluation extra, which is what CI installs.
"""

import inspect

import pytest
import torch

from lsa.core import RotaryLayout
from lsa.evaluation.adapter.glm5_next import Glm5NextAdapter
from lsa.evaluation.config import ModelConfig
from lsa.sglang.settings.design import IndexerConfig

TOKENS = 48

# A budget past every key any query can see, so the selection covers the context and the read is exact
COVERING = IndexerConfig(
    window_size=8, sink_size=2, top_k=TOKENS * 2, bucket_size=4, bucket_cap=8, pool_size=None
)

# The teacher's visibility rule at the same window and sinks
VISIBLE = IndexerConfig(
    window_size=8, sink_size=2, top_k=TOKENS * 2, top_p=TOKENS, pool_size=4, bucket_size=4, bucket_cap=8
)

SPEC = ModelConfig(name="zai-org/GLM-5.3-Flash", dtype="float32")


def build(indexer_types=None, layers=2, **overrides):
    """A GLM-5.3-Flash-shaped text stack small enough to run on CPU.

    `index_topk` is wider than the context so the model's own DSA selects everything - the baseline has to
    be dense attention, or a budget that covers the context has nothing to reproduce. `overrides` narrows it
    for a test whose point is a sparse selection.
    """
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    torch.manual_seed(0)
    values = dict(
        vocab_size=64,
        pad_token_id=0,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        layer_types=["deepseek_sparse_attention"] * layers,
        indexer_types=indexer_types or ["full"] * layers,
        mlp_layer_types=["dense"] * layers,
        index_head_dim=16,
        index_n_heads=2,
        index_topk=TOKENS * 2,
        linear_head_dim=16,
        linear_num_heads=4,
        attn_implementation="eager",
    )
    config = Glm5NextTextConfig(**{**values, **overrides})
    return Glm5NextTextModel(config).eval()


def sparse(layers=2):
    """A stack whose own DSA selects 8 of 48 tokens in pools of 4, so reading the selection is a real read"""
    from lsa.evaluation.adapter.glm5_next import chunk_indexer_queries

    model = build(layers=layers, index_topk=8, index_kpool=4)
    chunk_indexer_queries(model)
    return model


def test_reading_through_the_selection_is_the_models_own_sparse_attention():
    """The dense mask and the gather by id read the same keys, so the stack's output is the same"""
    from lsa.evaluation.adapter.glm5_next import bind_selected_attention

    model = sparse()
    ids = token_ids()
    with torch.no_grad():
        masked = model(input_ids=ids).last_hidden_state

    assert bind_selected_attention(model) == 2
    with torch.no_grad():
        gathered = model(input_ids=ids).last_hidden_state

    torch.testing.assert_close(gathered, masked, atol=1e-5, rtol=1e-4)


def test_prefilling_in_chunks_hands_every_layer_what_one_pass_hands_it():
    """The latent cache and the indexer's own cache carry a chunk's keys into the next one exactly"""
    from lsa.evaluation.adapter.glm5_next import bind_selected_attention
    from lsa.training.distillation.data.teacher import capture_layer_inputs

    model = sparse()
    ids = token_ids()
    one_pass = capture_layer_inputs(model, [0, 1], ids)

    bind_selected_attention(model)
    chunked = capture_layer_inputs(model, [0, 1], ids, prefill=16)

    for layer in (0, 1):
        assert chunked[layer][0].device.type == "cpu"
        torch.testing.assert_close(chunked[layer][0], one_pass[layer][0], atol=1e-5, rtol=1e-4)


HYBRID = ["linear_attention", "deepseek_sparse_attention"] * 2

# What the linear-attention layers call, bound at import to a CUDA package when one is installed
KERNELS = (
    "chunk_kimi_delta_attention",
    "recurrent_kimi_delta_attention",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def reference(kernel):
    """The PyTorch function under a kernel binding - a kernels-hub module around it, a package's function in it"""
    if isinstance(kernel, torch.nn.Module):
        cells = type(kernel).forward.__closure__ or ()
        kernel = next(cell.cell_contents for cell in cells if callable(cell.cell_contents))
    return inspect.unwrap(kernel)


@pytest.fixture
def reference_kernels(monkeypatch):
    """The reference PyTorch kernels, so a hybrid stack runs on a host with the training extra and no GPU"""
    from transformers.models.glm5_next import modeling_glm5_next

    for name in KERNELS:
        monkeypatch.setattr(modeling_glm5_next, name, reference(getattr(modeling_glm5_next, name)))


def hybrid():
    """Linear-attention layers between sparse ones, as the checkpoint has, selecting 8 of 48 in pools of 4"""
    from lsa.evaluation.adapter.glm5_next import chunk_indexer_queries

    model = build(layers=4, layer_types=HYBRID, indexer_types=["full"] * 4, index_topk=8, index_kpool=4)
    chunk_indexer_queries(model)
    return model


def run_in_spans(model, spans, ids, prefill, tmp_path):
    """Run each span over the whole document before the next, through the streams each cut writes"""
    from lsa.training.distillation.data.stage import Boundary, BoundaryWriter
    from lsa.training.distillation.data.teacher import capture_span_inputs

    adapter = Glm5NextAdapter()
    captured = {}
    for start, stop in spans:
        layers = [index for index in (1, 3) if start <= index < stop]
        source = Boundary(tmp_path / f"{start}.streams") if start > 0 else None
        sink = BoundaryWriter(tmp_path / f"{stop}.streams") if stop < len(model.layers) else None
        captured.update(
            capture_span_inputs(adapter, model, (start, stop), layers, ids, prefill, source, sink)
        )
        if sink is not None:
            sink.publish()
    return captured


@pytest.mark.parametrize("prefill", [0, 16])
def test_running_the_stack_a_span_at_a_time_hands_every_layer_what_one_pass_hands_it(
    prefill, tmp_path, reference_kernels
):
    """Every chunk of a document through one span before the next changes nothing a layer is handed"""
    from lsa.evaluation.adapter.glm5_next import bind_selected_attention
    from lsa.training.distillation.data.teacher import capture_layer_inputs

    model = hybrid()
    ids = token_ids()
    bind_selected_attention(model)
    one_pass = capture_layer_inputs(model, [1, 3], ids)

    in_spans = run_in_spans(model, [(0, 2), (2, 4)], ids, prefill, tmp_path)

    for layer in (1, 3):
        torch.testing.assert_close(in_spans[layer][0].cpu(), one_pass[layer][0], atol=1e-5, rtol=1e-4)


def test_a_span_starting_the_stack_reads_the_ids_the_way_the_stack_does(reference_kernels):
    """The embedding copied into every stream is what the stack's own forward hands its first layer"""
    model = hybrid()
    ids = token_ids()
    positions = torch.arange(TOKENS)[None]
    adapter = Glm5NextAdapter()
    with torch.no_grad():
        streams = adapter.forward_span(model, (0, 4), adapter.stack_input(model, ids), positions, None)
        whole = model(input_ids=ids).last_hidden_state
    torch.testing.assert_close(model.norm(model.hc_head(streams)), whole)


def test_a_teacher_reading_the_latent_sees_what_the_expanded_heads_see():
    """Folding each query through its key map scores it against the latent, which is the model's own logit"""
    from lsa.evaluation.adapter.glm5_next import Glm5NextSparseAttention
    from lsa.evaluation.adapter.layer import SparseAttention
    from lsa.training.distillation.data.teacher import teacher_attention

    model = build()
    reader = Glm5NextSparseAttention(model.layers[0].self_attn, COVERING, RotaryLayout())
    torch.manual_seed(2)
    hidden = torch.randn(1, TOKENS, 64)
    positions = torch.tensor([20, 31, 40, 47])

    expanded = SparseAttention.teacher_view(reader, hidden, positions, None)
    latent = reader.teacher_view(hidden, positions, None)

    torch.testing.assert_close(latent.index_queries, expanded.index_queries)
    torch.testing.assert_close(latent.index_keys, expanded.index_keys)
    scale = reader.attention.scaling
    by_heads = teacher_attention(expanded.q, expanded.k, scale, positions, VISIBLE, 8)
    by_latent = teacher_attention(latent.q, latent.k, scale, positions, VISIBLE, 8)
    assert torch.equal(by_latent[0], by_heads[0])
    torch.testing.assert_close(by_latent[1], by_heads[1], atol=1e-3, rtol=1e-3)
    assert torch.equal(by_latent[2], by_heads[2])


def token_ids(query_count=TOKENS):
    torch.manual_seed(1)
    return torch.randint(0, 64, (1, query_count))


def swap(model):
    """Replace every DSA layer's attention with the sparse read, as a teacher builds its readers."""
    adapter = Glm5NextAdapter()
    text_model = adapter.text_model(model)
    layers = adapter.full_attention_layers(text_model.config, SPEC)
    rotary = adapter.rotary_layout(model, SPEC, layers)
    for index in layers:
        text_model.layers[index].self_attn = adapter.attention_class(
            text_model.layers[index].self_attn, COVERING, rotary
        )
    return layers, rotary


def test_the_dsa_layers_are_swapped_and_measured_as_nope():
    assert swap(build()) == ([0, 1], RotaryLayout())


def test_a_stack_that_shares_one_selection_is_refused():
    """A swapped layer makes no top-k, so a layer that would reuse one cannot be left in place"""
    model = build(indexer_types=["full", "shared"])
    with pytest.raises(ValueError, match="reuse another layer's top-k"):
        swap(model)


def test_the_adapter_loads_through_an_auto_class_that_knows_this_config():
    """GLM-5.3-Flash is vision-language, so its head is not in the causal-LM mapping the default names"""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

    from lsa.evaluation.adapter.model import ModelAdapter

    assert Glm5NextConfig in Glm5NextAdapter().model_class()._model_mapping
    assert Glm5NextConfig not in ModelAdapter().model_class()._model_mapping


def test_scoring_the_indexer_in_blocks_selects_what_scoring_it_whole_selects():
    """Nothing downstream of the score tensor couples one query to another, so blocking it changes nothing"""
    from lsa.evaluation.adapter.glm5_next import chunk_indexer_queries

    model = build()
    attention = model.layers[0].self_attn
    indexer = attention.indexer
    # Narrow, so the top-k ranks rather than selecting the whole context
    indexer.index_topk = 8

    hidden = model.embed_tokens(token_ids())
    q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)

    with torch.no_grad():
        whole = indexer(hidden, q_resid, mask, None)

    assert chunk_indexer_queries(model, chunk=16) == 2
    with torch.no_grad():
        blocked = indexer(hidden, q_resid, mask, None)

    assert blocked.shape == whole.shape
    assert torch.equal(blocked, whole)


def test_selecting_for_some_positions_returns_those_rows_of_selecting_for_all():
    """The offline pass scores only the cached query rows, against the keys of the whole document"""
    from lsa.evaluation.adapter.glm5_next import chunk_indexer_queries

    model = build()
    attention = model.layers[0].self_attn
    attention.indexer.index_topk = 8
    chunk_indexer_queries(model, chunk=5)

    hidden = model.embed_tokens(token_ids())
    positions = torch.tensor([9, 10, 23, 30, 47])
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)

    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
        whole = attention.indexer(hidden, q_resid, mask, None)
        rows = attention.indexer(hidden, q_resid[:, positions], mask, None, positions=positions)

    assert torch.equal(rows, whole[:, positions])
