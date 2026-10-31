from types import SimpleNamespace

import pytest
import torch
from torchtitan.distributed import fsdp as titan_fsdp
from torchtitan.models.common.attention import FlexInnerAttention

from lsa.training.native import config as native
from lsa.training.native.layers import MultiresInnerAttention
from lsa.training.native.models import OffsetGatedDeltaKernel, model_spec, plain_parameter_fsdp


def experiment(kind, flavor="0.8B", model=None, **attention):
    return native.NativeConfig(
        attention=native.AttentionConfig(kind=kind, **attention),
        model=native.ModelConfig(flavor=flavor, **(model or {})),
        data=native.DataConfig(name="qwen3_5", seq_len=4096),
        run=native.RunConfig(output="/tmp/native-test"),
    )


def attentions(spec):
    return [layer.attention for layer in spec.model.layers if layer.attention is not None]


def test_the_dense_arm_leaves_torchtitans_own_attention_in_place():
    built = model_spec(experiment("dense"))
    assert attentions(built)
    assert all(
        type(attention.inner_attention) is FlexInnerAttention.Config for attention in attentions(built)
    )


@pytest.mark.parametrize(
    ("kind", "knobs"),
    [("window", {}), ("multires", {}), ("lsa", {"coarse": False})],
    ids=["window", "multires", "lsa"],
)
def test_every_full_attention_layer_gets_the_arm_and_the_recurrent_layers_are_untouched(kind, knobs):
    built = model_spec(experiment(kind, **knobs))
    layers = built.model.layers
    full = attentions(built)
    # Qwen3.5 is a recurrent layer three times in four
    assert len(full) == len(layers) // 4
    assert all(isinstance(attention.inner_attention, MultiresInnerAttention.Config) for attention in full)
    assert all(attention.inner_attention.attention.kind == kind for attention in full)
    assert all(attention.inner_attention.seq_len == 4096 for attention in full)
    assert sum(layer.delta_net is not None for layer in layers) == len(layers) - len(full)


@pytest.mark.parametrize("kind", ["dense", "window"])
def test_every_recurrent_layer_gets_the_kernel_that_always_has_offsets(kind):
    recurrent = [layer.delta_net for layer in model_spec(experiment(kind)).model.layers if layer.delta_net]
    assert recurrent
    assert all(type(net.inner_gated_delta_net.kernel) is OffsetGatedDeltaKernel.Config for net in recurrent)


def test_the_text_model_has_no_vision_tower():
    assert model_spec(experiment("dense")).model.vision_encoder is None


def test_the_arms_differ_in_the_attention_alone():
    dense, multires = model_spec(experiment("dense")), model_spec(experiment("multires"))
    assert dense.model.dim == multires.model.dim
    assert len(dense.model.layers) == len(multires.model.layers)
    assert dense.model.vocab_size == multires.model.vocab_size


def parameter_count(**model):
    built = model_spec(
        native.NativeConfig(
            attention=native.AttentionConfig(kind="window"),
            model=native.ModelConfig(**model),
            data=native.DataConfig(name="qwen3_5", seq_len=4096),
            run=native.RunConfig(output="/tmp/native-test"),
        )
    )
    with torch.device("meta"):
        return sum(parameter.numel() for parameter in built.model.build().parameters())


# The reference training code's own models instantiated on the meta device at the 50,257-token GPT-2 vocabulary, which it pads to
# 50,304 rows: lm_800m_qwen35.yml, lm_800m_qwen35_mqa8_nope_wsd_4k_lr5e-4.yml and lm_340m.yml
HYBRID_PARAMETERS = 549_624_640
HYBRID_MQA_PARAMETERS = 526_030_144
DENSE_PARAMETERS = 342_449_152


def test_the_hybrid_has_the_reference_training_codes_parameter_count():
    assert parameter_count(flavor="0.8B-gpt2") == HYBRID_PARAMETERS


def test_the_multi_query_shape_has_the_reference_training_codes_parameter_count():
    shape = {"query_head_count": 8, "kv_head_count": 1, "head_dim": 128, "rotary_dim": 0}
    assert parameter_count(flavor="0.8B-gpt2", **shape) == HYBRID_MQA_PARAMETERS


def test_the_340m_model_has_the_reference_training_codes_parameter_count():
    assert parameter_count(flavor="340M-gpt2") == DENSE_PARAMETERS


def test_the_hybrid_ties_its_embeddings_and_the_340m_model_does_not():
    assert model_spec(experiment("dense", flavor="0.8B-gpt2")).model.enable_weight_tying
    assert not model_spec(experiment("dense", flavor="340M-gpt2")).model.enable_weight_tying


def test_the_340m_model_is_nothing_but_plain_attention_and_swiglu():
    layers = model_spec(experiment("window", flavor="340M-gpt2")).model.layers
    assert len(layers) == 19
    assert all(layer.delta_net is None and layer.attention is not None for layer in layers)
    assert all(not layer.attention.output_gate and not layer.attention.qk_norm for layer in layers)
    assert {layer.attention.rotary_dim for layer in layers} == {128}
    assert {layer.feed_forward.w1.out_features for layer in layers} == {2736}


def test_a_head_shape_is_the_flavours_until_a_run_names_one():
    layer = next(layer for layer in model_spec(experiment("window")).model.layers if layer.attention)
    assert (layer.attention.n_heads, layer.attention.n_kv_heads, layer.attention.head_dim) == (8, 2, 256)
    shaped = model_spec(experiment("window", model={"kv_head_count": 1, "rotary_dim": 0})).model
    layer = next(layer for layer in shaped.layers if layer.attention)
    assert (layer.attention.n_heads, layer.attention.n_kv_heads, layer.attention.head_dim) == (8, 1, 256)
    assert layer.attention.rotary_dim == 0


def test_the_lsa_arm_reads_the_rotation_the_model_has():
    shaped = model_spec(experiment("lsa", model={"rotary_dim": 0}, coarse=False)).model
    inner = [layer.attention.inner_attention for layer in shaped.layers if layer.attention]
    assert {attention.rotary_dim for attention in inner} == {0}


def test_a_rotation_wider_than_the_head_is_refused():
    with pytest.raises(ValueError, match="rotation within the head"):
        model_spec(experiment("window", model={"head_dim": 32}))


@pytest.mark.parametrize("flavor", ["0.8B-gpt2", "340M-gpt2"])
def test_every_weight_of_a_gpt2_flavour_has_an_initialiser(flavor):
    spec = model_spec(experiment("window", flavor=flavor))
    with torch.device("meta"):
        model = spec.model.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=None)
    embedding = model.tok_embeddings.weight
    assert torch.isfinite(embedding).all()
    assert abs(embedding.std().item() - 0.02) < 1e-3


@pytest.mark.parametrize(
    ("axes", "passed"),
    [(("dp_shard",), None), (("dp_replicate", "dp_shard"), None), (("dp_shard", "tp"), "dims")],
    ids=["fsdp", "hsdp", "tensor-parallel"],
)
def test_data_parallel_meshes_shard_plain_parameters_and_other_meshes_keep_the_spmd_dims(
    axes, passed, monkeypatch
):
    calls = []
    monkeypatch.setattr(titan_fsdp, "fully_shard", lambda module, **kwargs: calls.append(kwargs))
    with plain_parameter_fsdp():
        titan_fsdp.fully_shard(
            "block", mesh=SimpleNamespace(mesh_dim_names=axes), dp_mesh_dims="dims", reshard=True
        )
    assert calls[0]["dp_mesh_dims"] == passed
    assert calls[0]["reshard"] is True
