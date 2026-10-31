# Writing an adapter

A new model is one file in `src/lsa/evaluation/adapter/`, imported and registered by its `__init__.py`.
Everything between the projections and the output - the cache bookkeeping, the index, the selection, the `query_start` arithmetic - is model-free and is already written.

An adapter is how teacher extraction reads a checkpoint - its layers, its projections, its rotation and its index space - and what the probe screens one against.
Serving does not go through it: a served model runs in sglang, and the plugin drives its families itself.

First check whether you need one at all: [models.md](models.md) is what the generic HF decoder already drives.

## The smallest one

Start from what your attention module does that the generic one does not.
The default `project` reads `q_proj`, `k_proj` and `v_proj`, applies `q_norm` and `k_norm` when the model carries them, and hands back head tensors.
A model with one fused projection overrides it and nothing else.

```python
# src/lsa/evaluation/adapter/my_model.py
from lsa.evaluation.adapter.layer import Projection, SparseAttention, heads
from lsa.evaluation.adapter.model import ModelAdapter


class MyAttention(SparseAttention):
    """A model whose q, k and v come out of one fused projection."""

    def project(self, hidden_states):
        attention = self.attention
        head_dim = attention.head_dim
        query_dim = attention.config.num_attention_heads * head_dim
        kv_dim = attention.config.num_key_value_heads * head_dim

        fused = attention.qkv_proj(hidden_states)
        q, k, v = fused.split([query_dim, kv_dim, kv_dim], dim=-1)
        return Projection(q=heads(q, head_dim), k=heads(k, head_dim), v=heads(v, head_dim))


class MyAdapter(ModelAdapter):
    attention_class = MyAttention
```

Import from `adapter.layer` and `adapter.model` rather than from `adapter` itself.
`adapter/__init__.py` imports your module before defining `register`, so reaching back into the package from inside the module is a circular import.

`Projection` carries `[B, H, T, dh]` head tensors in the model's own dtype, and `heads` is the split and transpose that gets there from `[B, T, H * dh]`.
`q` and `k` are **pre-RoPE** - that is the space the index vectors are taken in, and the rotation is applied after.
Apply the model's own QK-norm here if it has one, because overriding `project` skips the default that would have.

Register it.
The key is the checkpoint's `model_type`, which is what `auto` resolves against.

Add the import beside the existing adapter imports in `src/lsa/evaluation/adapter/__init__.py`:

```python
from lsa.evaluation.adapter.my_model import MyAdapter
```

Then add the registration beside the built-in registrations at the bottom of that file, after `register` is defined:

```python
register("my_model_type", MyAdapter)
```

Then screen the checkpoint and extract from it.

```bash
lsa probe me/my-model                   # reports against your adapter now
lsa distill targets <extraction>.yaml model.name=me/my-model
```

## One that lives in your own package

Nothing above requires a fork.
A package installed alongside `lsa` advertises its adapters through an entry point group, and the name is the `model_type`.

```toml
# your package's pyproject.toml
[project.entry-points."lsa.adapters"]
my_model_type = "my_package.adapter:MyAdapter"
```

```bash
pip install lsa my-package
lsa probe me/my-model
```

No import, no `register` call and no edit to this repo - `lsa.evaluation.adapter` reads the installed entry points when it loads.
A package that claims a `model_type` another adapter already drives is a conflict and raises, naming both, rather than shadowing it silently.

## The five hooks, in the order a model needs them

`project` is the usual one - a fused QKV, a packed gate, a head layout that is not `[B, T, H * dh]`.

`index_space` is the space the indexer clusters and selects in, and an extraction caches.
The default is the position-free sub-term of the logit, regrouped across GQA heads, which is right for any model that keeps its keys per head.
A model that stores one latent per token instead overrides it to read that latent - GLM-5.3-Flash's MLA is the one here.
The extraction stores `x`, the attention block's input; the trainer computes the key latent and the heads' queries from it with the layer's own weights, and the served design reads them through the map it trains.

`combine` is next, for an output gate or a projection that is not `o_proj`.
It receives the attention output and whatever `project` put in `Projection.gate`.

`rotate` only matters when the rotation does not come from the model's own `apply_rotary_pos_emb`, which is resolved out of the attention class's module and so needs no code for almost every model.
It is static, because the rotary layout is measured by calling it before any layer is built.

`read` is the one that means real work, and it is needed less often than it looks.
The default attends over raw keys and values per head, and an MLA model still qualifies: transformers expands the latent into per-head keys and values and caches those, so MLA is a `project` override and the read is untouched.
What actually needs `read` is a model whose attention consumes something the cache never expands - a pooled or compressed entry standing for several tokens.

Checkpoint-level differences go on `ModelAdapter` instead: `load` for a model class that is not `AutoModelForCausalLM` or a tower to drop, `text_model` for an unusual nesting, `full_attention_layers` for a layer stack that needs more than a name, `rotary_embedding` for tables that do not sit at `text_model.rotary_emb`.

Two worked examples are in the tree.
[qwen3_5.py](../../src/lsa/evaluation/adapter/qwen3_5.py) is the small one - a packed output gate and a vision tower, and nothing else.
[glm5_next.py](../../src/lsa/evaluation/adapter/glm5_next.py) is the one that shows the shape of a harder model: MLA projections, a NoPE index space, a per-family default for which layer types to swap, and an attention module whose decoder layer unpacks three values instead of two.

## Checking it

An adapter that is subtly wrong produces plausible numbers, so check it against the model it wraps rather than against a score.

**Give it a top-k that covers the context.**
Selection plus window plus sinks then reaches every causal position, the read is exact, and the wrapped layer must reproduce the one it wraps.

```python
config = IndexerConfig(window_size=32, sink_size=4, bucket_size=8, top_k=256)
native = text_model.layers[index].self_attn
layer = MyAttention(native, config, model_adapter.rotary_layout(model, model_config, [index]))

with torch.no_grad():
    sparse, _ = layer(hidden_states, (cos, sin), past_key_values=cache)
# the native module's own output on the same hidden states and tables is `dense`
print((dense - sparse).abs().max())    # ~0 - the budget covers the context, so the read is exact
```

Any gap here is the adapter, not the indexer - the read has its own exactness test in `tests/unit/core/test_attention.py`.

**Then decode.**
One token at a time from a half-length prefill must reproduce one whole prefill, which is what says the index grows across steps instead of being rebuilt, and that a fresh cache is a fresh sequence.
`tests/integ/evaluation/adapter/test_layer.py` does both on CPU against a fake model, and [tests/helpers/evaluation/adapter/fake_model.py](../../tests/helpers/evaluation/adapter/fake_model.py) is the attribute contract written down - copy it for a layout the fake does not cover.

**Better, check against the real module.**
A model too large to load is still ordinary python: build its config at a size that fits in a test, instantiate the transformers class with random weights, and run both checks against it.
[tests/integ/evaluation/adapter/test_glm5_next.py](../../tests/integ/evaluation/adapter/test_glm5_next.py) does exactly that for a 306GB checkpoint, and it is what says `project` unpacks the MLA latent the way the model does - the module's own `expand_kv` produces the keys and values, so a mistake is a number rather than a shape.
