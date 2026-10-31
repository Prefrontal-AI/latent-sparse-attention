"""Loading one span of a decoder stack - the rest left out before a weight is read.

A tiny Llama saved to disk stands in for a checkpoint no card holds, loaded through the same
`from_pretrained` the real one is.
"""

import pytest
import torch

from lsa.evaluation.adapter import ModelAdapter
from lsa.evaluation.config import ModelConfig


@pytest.fixture
def saved(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    return model, ModelConfig(name=str(tmp_path), dtype="float32")


def load_span(model_config, span):
    from transformers import AutoModelForCausalLM

    model_class = ModelAdapter().span_class(AutoModelForCausalLM, model_config, span)
    return model_class.from_pretrained(model_config.checkpoint, dtype=torch.float32)


def test_a_span_holds_its_own_layers_and_nothing_else(saved):
    whole, model_config = saved
    span = load_span(model_config, (1, 3))

    names = {name for name, _ in span.named_parameters()}
    assert names == {
        name for name, _ in whole.named_parameters() if ".layers.1." in name or ".layers.2." in name
    }
    for name, parameter in span.named_parameters():
        assert torch.equal(parameter, whole.get_parameter(name))


def test_the_span_that_starts_the_stack_keeps_the_embedding(saved):
    whole, model_config = saved
    span = load_span(model_config, (0, 2))
    assert torch.equal(span.model.embed_tokens.weight, whole.model.embed_tokens.weight)
    assert not any("lm_head" in name or "model.norm" in name for name, _ in span.named_parameters())


@pytest.mark.parametrize("span", [(2, 2), (3, 5)])
def test_an_empty_span_or_one_past_the_stack_is_refused(saved, span):
    _, model_config = saved
    with pytest.raises(ValueError, match="a span must be non-empty"):
        load_span(model_config, span)
