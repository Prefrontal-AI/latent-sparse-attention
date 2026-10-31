"""Build a small DSA layer with the native model configuration path."""

from dataclasses import replace

from lsa.training.native.config import AttentionConfig, DataConfig, ModelConfig, NativeConfig, RunConfig
from lsa.training.native.models import model_spec


def attention(length=8, **knobs):
    config = AttentionConfig(
        **(
            dict(
                kind="dsa",
                index_head_count=2,
                index_dim=4,
                window_size=2,
                sink_size=1,
                top_k=2,
                kl_query_count=8,
            )
            | knobs
        )
    )
    spec = model_spec(
        NativeConfig(
            attention=config,
            model=ModelConfig(flavor="340M-gpt2"),
            data=DataConfig(seq_len=length),
            run=RunConfig(output="/tmp/dsa-tests"),
        )
    )
    template = spec.model.layers[0].attention
    tiny = replace(
        template,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        rotary_dim=0,
        wq=replace(template.wq, in_features=16, out_features=16),
        wk=replace(template.wk, in_features=16, out_features=16),
        wv=replace(template.wv, in_features=16, out_features=16),
        wo=replace(template.wo, in_features=16, out_features=16),
    )
    layer = tiny.build()
    layer.init_states(buffer_device=None)
    return layer
