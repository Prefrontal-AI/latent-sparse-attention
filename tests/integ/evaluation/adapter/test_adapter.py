"""The registry - which code drives which checkpoint.

Resolving `auto` reads the checkpoint's config and needs transformers, so what is gated here is everything
either side of that: a named adapter, an unknown name, and the class each one wraps with.
"""

from types import SimpleNamespace

import pytest

from lsa.evaluation import adapter
from lsa.evaluation.config import ModelConfig
from lsa.evaluation.constants import Adapter


def test_a_named_adapter_is_the_one_that_runs():
    assert isinstance(adapter.adapter_for(ModelConfig(adapter="qwen3_5")), adapter.Qwen35Adapter)


def test_the_generic_adapter_is_named_too():
    assert type(adapter.adapter_for(ModelConfig(adapter="hf"))) is adapter.ModelAdapter


def test_an_unregistered_name_says_what_is_registered():
    with pytest.raises(ValueError, match="no adapter named 'llama_42'") as raised:
        adapter.adapter_for(ModelConfig(adapter="llama_42"))
    assert ", ".join(sorted(adapter.ADAPTERS)) in str(raised.value)


def test_a_dtype_name_that_is_not_a_dtype_is_rejected():
    with pytest.raises(ValueError, match="must name a torch dtype"):
        adapter.torch_dtype("float9")


def stack(layer_types=None, layer_count=4):
    """A decoder config shaped like the two attributes layer discovery reads."""
    return SimpleNamespace(layer_types=layer_types, num_hidden_layers=layer_count)


def test_a_stack_that_declares_no_types_is_swapped_whole():
    found = adapter.ModelAdapter().full_attention_layers(stack(layer_count=6), ModelConfig())
    assert found == [0, 1, 2, 3, 4, 5]


def test_a_hybrid_stack_is_swapped_where_it_says_full_attention():
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    found = adapter.ModelAdapter().full_attention_layers(stack(layer_types * 2), ModelConfig())
    assert found == [3, 7]


def test_a_stack_with_its_own_vocabulary_fails_with_the_names_it_uses():
    layer_types = ["linear_attention"] * 3 + ["qwen_sparse_attention"]
    with pytest.raises(ValueError, match="it declares linear_attention, qwen_sparse_attention"):
        adapter.ModelAdapter().full_attention_layers(stack(layer_types), ModelConfig())


def test_naming_that_vocabulary_in_the_config_is_all_it_takes():
    layer_types = ["linear_attention"] * 3 + ["qwen_sparse_attention"]
    model_config = ModelConfig(layer_types=["qwen_sparse_attention"])
    assert adapter.ModelAdapter().full_attention_layers(stack(layer_types * 2), model_config) == [3, 7]


def test_a_multimodal_checkpoint_is_read_through_to_its_decoder():
    text_model = SimpleNamespace(config=SimpleNamespace(layer_types=None), layers=[0, 1])
    model = SimpleNamespace(model=SimpleNamespace(language_model=text_model, visual=object()))
    assert adapter.ModelAdapter().text_model(model) is text_model


def test_a_stack_with_no_rotary_tables_says_so_rather_than_guessing():
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(), layers=[]))
    with pytest.raises(ValueError, match="carries no rotary_emb"):
        adapter.ModelAdapter().rotary_embedding(model)


@pytest.fixture
def registry():
    """The registry is module state - snapshot it so a test that registers does not leak."""
    saved = dict(adapter.ADAPTERS)
    saved_broken = dict(adapter.BROKEN)
    yield adapter.ADAPTERS
    adapter.ADAPTERS.clear()
    adapter.ADAPTERS.update(saved)
    adapter.BROKEN.clear()
    adapter.BROKEN.update(saved_broken)


class Mine(adapter.ModelAdapter):
    pass


class Other(adapter.ModelAdapter):
    pass


class FakeEntryPoint:
    """The two attributes `register_installed` reads off an entry point."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def test_registering_an_adapter_is_one_line(registry):
    adapter.register("my_model_type", Mine())
    assert isinstance(adapter.adapter_for(ModelConfig(adapter="my_model_type")), Mine)


def test_a_class_is_registered_by_instantiating_it(registry):
    assert isinstance(adapter.register("my_model_type", Mine), Mine)


def test_registering_the_same_adapter_twice_is_not_a_clash(registry):
    adapter.register("my_model_type", Mine)
    adapter.register("my_model_type", Mine)
    assert isinstance(registry["my_model_type"], Mine)


def test_taking_over_a_model_type_someone_else_drives_is_refused(registry):
    adapter.register("my_model_type", Mine)
    with pytest.raises(ValueError, match="already driven by Mine"):
        adapter.register("my_model_type", Other)


def test_a_takeover_can_be_asked_for(registry):
    adapter.register("my_model_type", Mine)
    adapter.register("my_model_type", Other, replace=True)
    assert isinstance(registry["my_model_type"], Other)


def test_an_installed_package_registers_without_an_edit_here(registry):
    assert adapter.register_installed([FakeEntryPoint("my_model_type", Mine)]) == ["my_model_type"]
    assert isinstance(adapter.adapter_for(ModelConfig(adapter="my_model_type")), Mine)


def test_an_entry_point_that_will_not_load_costs_only_the_model_it_claimed(registry):
    """Discovery runs at import, so one broken package must not take the harness down with it"""
    entry = FakeEntryPoint("broken_model", ImportError("no module named my_package"))
    assert adapter.register_installed([entry]) == []
    assert "broken_model" in adapter.BROKEN

    # Every other model still resolves
    assert isinstance(adapter.adapter_for(ModelConfig(adapter=Adapter.GENERIC)), adapter.ModelAdapter)


def test_asking_for_a_broken_adapter_raises_with_the_original_failure(registry):
    adapter.register_installed([FakeEntryPoint("broken_model", ImportError("no module named my_package"))])

    with pytest.raises(ImportError, match="claims 'broken_model' but would not load") as raised:
        adapter.adapter_for(ModelConfig(adapter="broken_model"))
    assert "no module named my_package" in str(raised.value.__cause__)


def test_a_broken_adapter_is_not_quietly_replaced_by_the_generic_one(registry, monkeypatch):
    """Falling back would run the wrong code rather than say the package the user installed is broken"""
    adapter.register_installed([FakeEntryPoint("broken_model", ImportError("boom"))])
    monkeypatch.setattr(adapter, "model_type_of", lambda name: "broken_model")

    with pytest.raises(ImportError, match="claims 'broken_model'"):
        adapter.adapter_for(ModelConfig(name="me/my-model"))
