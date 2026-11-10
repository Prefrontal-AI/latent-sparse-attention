# Models

The weights are assets of the repository's releases, not files in it.

| Release | What it holds | Read by |
|---|---|---|
| [`d-lsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1) | the distilled maps of D-LSA for Qwen3.8-27B and GLM-5.3-Flash | every `lsa_*` served config |
| [`qsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/qsa-weights-v1) | the QSA indexer distilled on frozen Qwen3.8-27B (`long_16k_1m_heavy_qsa`, step 48000), the Block Sparse (Flat) baseline | the `qsa` configs on Qwen3.8-27B, through `lsa distill inject` |
| [`n-lsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/n-lsa-weights-v1) | the checkpoints of the ten 128k decays of the native comparison, N-LSA and its baselines | `lsa native evaluate` |

```bash
gh release download d-lsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention --pattern '*.pt' --dir "$LSA_ROOT/models/maps"
gh release download qsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention --pattern '*.pt' --dir "$LSA_ROOT/models/maps"
lsa distill inject --model qwen38_27b --maps "$LSA_ROOT/models/maps/qwen38_27b__long-16k-1M-heavy-qsa.pt" --out "$LSA_ROOT/models/qwen38_27b_qsa"
```

A native checkpoint is split in parts below the 2 GiB limit of a release asset.
[Weights](../README.md#weights) in the main README says how to join and unpack one, and which config evaluates each.
