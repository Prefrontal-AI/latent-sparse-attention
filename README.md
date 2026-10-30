# Latent Sparse Attention

**Towards Subquadratic Ultra-Long Context Attention**

[Paper](#citation) (link to come) | [SGLang fork](https://github.com/Prefrontal-AI/sglang/tree/lsa-v0.1.0) | [Distilled maps](#released-artifacts) (to come)

![D-LSA and N-LSA overview](assets/overview.png)

*Top: one attention layer. Keys are pooled and clustered into buckets, and a query scores the bucket centroids, then attends only to the selected long-range keys plus the sink and window. Bottom: accuracy and speedup (P: prefill, D: decode, T: training) for D-LSA with flat search and N-LSA.*

## TL;DR

Latent Sparse Attention (LSA) clusters keys hierarchically in a learned latent space, and each query scores the clusters instead of every key.
Prefill, decode and training get faster at long context, while accuracy stays close to the original model's.

- **D-LSA** (Distilled) is applied to a pre-trained model by distilling its indexer.
  On GLM-5.3-Flash at 10M tokens it speeds up one attention layer by 22.7x in prefill and 4.6x in decode over the model's own sparse attention.
  On Qwen3.8-27B it is 72.4x and 42.5x faster than full attention.
- **N-LSA** (Native) is trained from scratch on the next-token loss alone.
  It matches the accuracy of full attention, and its training is 1.6x faster per layer at 128k tokens.

The attention kernel is unchanged.
LSA only decides which keys it reads.

## Contents

| Method | Paper name | Where |
|---|---|---|
| Distilled LSA, flat search | D-LSA (Flat) | `src/lsa/core`, served by `src/lsa/sglang` |
| Distilled LSA, tree search | D-LSA (Tree) | same |
| Training-free LSA, a PCA map | D-LSA (Training-Free) | same |
| Distillation of the index space | - | `src/lsa/training/distillation` |
| Native LSA and the native baselines (Full Attention, SWA, NSA, PISA) | N-LSA | `src/lsa/training/native` |
| Served baselines | Block Sparse Attention (Flat, Tree) | `src/lsa/baselines` |
| Accuracy, kernel and end-to-end evaluations | - | `src/lsa/evaluation` |

Everything runs through one command, `lsa`.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are needed.

```bash
git clone https://github.com/Prefrontal-AI/latent-sparse-attention.git
cd latent-sparse-attention
uv sync --locked --extra evaluation --extra training
uv run lsa --help
```

The `training` extra builds `causal-conv1d`, which needs the CUDA toolkit.
Without `nvcc`, add `--no-install-package causal-conv1d`.
For evaluation alone, use `uv sync --locked --extra evaluation`.

The results in the paper were measured on NVIDIA H200 GPUs.

## Quick start: serve a model with LSA

LSA is an SGLang plugin.
It runs in the image built from our [SGLang fork](https://github.com/Prefrontal-AI/sglang/tree/lsa-v0.1.0), which adds the context-parallel and long-context support the 10M-token runs need.
LSA itself is not in the fork.

```bash
# Build the serving image once - Docker and enroot, no GPU
git clone --branch lsa-v0.1.0 https://github.com/Prefrontal-AI/sglang.git
sbatch sglang/docker/enroot/build.sh --out-dir <image directory>
```

Inside the image, install this repository and pass an `lsa` block in the model config:

```bash
pip install --no-deps /path/to/latent-sparse-attention
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27b --tp 1 \
  --attention-backend fa3 --page-size 1 \
  --json-model-override-args '{"lsa": {"projections": "/path/to/qwen38_27b.pt", "query_heads": "all"}}'
```

The plugin registers through SGLang's `sglang.srt.plugins` entry point and arms itself from the `lsa` block.
Look for `armed` in the server log.
Without it, the server is running the checkpoint's own attention.
Every key of the block defaults to the paper's flat design, and `docs/reference/notation.md` names each one.
`projections` is a distilled map: train one (see [Distillation](#distillation)), or use the released maps once they are published.

For GLM-5.3-Flash, the full server settings are in `configs/evaluation/accuracy/glm53_flash/lsa_flat_early_budget_tp8.yaml`.

## Reproducing the paper

The evaluation harness starts the server, runs a suite against it and writes a record.
Each command below submits a Slurm job through a *site* (see [Running on a cluster](#running-on-a-cluster)).
Add `--dry-run` to print what would be submitted.

Configs are named `<method>_<mesh>[_<slice>].yaml` under one directory per model, `qwen38_27b` or `glm53_flash`:

| Paper method | Config stem |
|---|---|
| Full Attention | `full` (Qwen3.8-27B) |
| Block Sparse Attention (Flat) | `qsa` on Qwen3.8-27B, `native` on GLM-5.3-Flash - each model's own indexer |
| Block Sparse Attention (Tree) | `pisa_c8` |
| D-LSA (Flat) | `lsa_flat`, and `lsa_flat_early_budget` on GLM-5.3-Flash |
| D-LSA (Tree) | `lsa_tree` |
| D-LSA (Training-Free) | `lsa_flat_pca` |

### Accuracy (Section 4.2)

Prepare RULER once for the model's tokenizer, then score a config:

```bash
lsa --site mycluster data prepare ruler --tokenizer Qwen/Qwen3.8-27b \
  --root <root>/benchmarks-qwen38 --lengths 4096 32768 --cpus 16 --mem-gb 64 --hours 2
lsa --site mycluster evaluate accuracy configs/evaluation/accuracy/qwen38_27b/lsa_flat_tp1.yaml \
  data.suite=ruler "data.lengths=[4096,32768]" "data.tasks=[<the 26 tasks below>]" \
  run.output=<dir> --gpus 1 --hours 4
```

<details>
<summary>The 26 RULER tasks at 4k and 32k</summary>

```text
ruler_niah_single_1_4096,ruler_niah_single_2_4096,ruler_niah_single_3_4096,ruler_niah_multikey_1_4096,ruler_niah_multikey_2_4096,ruler_niah_multikey_3_4096,ruler_niah_multiquery_4096,ruler_niah_multivalue_4096,ruler_vt_4096,ruler_cwe_4096,ruler_fwe_4096,ruler_qa_squad_4096,ruler_qa_hotpot_4096,ruler_niah_single_1_32768,ruler_niah_single_2_32768,ruler_niah_single_3_32768,ruler_niah_multikey_1_32768,ruler_niah_multikey_2_32768,ruler_niah_multikey_3_32768,ruler_niah_multiquery_32768,ruler_niah_multivalue_32768,ruler_vt_32768,ruler_cwe_32768,ruler_fwe_32768,ruler_qa_squad_32768,ruler_qa_hotpot_32768
```

The Qwen configs pin the paper's task list, so `data.tasks` selects the cells.
For GLM-5.3-Flash, drop `--tokenizer` and `--root`, since its tokenizer is the default.

</details>

Expected results, RULER mean over 13 tasks of 100 rows each, from a clean checkout:

| Run | 4k | 32k | Paper 4k | Paper 32k | Wall time |
|---|---|---|---|---|---|
| Qwen3.8-27B, D-LSA (Flat), 1 H200 | 0.974 | 0.968 | 0.975 | 0.966 | 1 h 52 min |
| GLM-5.3-Flash, D-LSA (Flat) with early budget, 8 H200 | 0.973 | 0.968 | 0.973 | 0.968 | about 40 min |

`data.suite` also takes `papertrail`, `full` and `smoke` (four rows per task, for a quick check).
`full_tp1.yaml` scores the dense baseline, which needs no map.

### Performance (Section 4.3)

```bash
# One attention layer on one GPU, prefill and decode, 128k to 10M tokens
lsa --site mycluster evaluate kernels configs/evaluation/kernels/glm53_flash/lsa_flat.yaml
# Complete served requests end to end
lsa --site mycluster evaluate production configs/evaluation/production/glm53_flash/lsa_flat_cp8_10m.yaml
```

Swap the stem for each method, and `glm53_flash` for `qwen38_27b`.
The configs prefixed `b300_` measure on B300.

### Native training (Section 4.4)

The native arms pre-train a 24-layer hybrid model on 100B tokens of FineWeb-Edu at 4k, then decay at 128k on long documents, each arm from its own checkpoint.
Each sparse arm reads about 2,560 keys per query in the decay.

| Arm | Pre-train | 128k decay |
|---|---|---|
| Full Attention | `pretrain_fineweb.yaml` | `decay_fineweb128k.yaml` |
| SWA | `pretrain_fineweb_window.yaml` | `decay_fineweb128k_window.yaml` |
| NSA | `nsam8.yaml` | `decay_fineweb128k_nsa.yaml` |
| PISA | `pisam8.yaml` | `decay_fineweb128k_pisa.yaml` |
| N-LSA | `pretrain_fineweb_lsa.yaml` | `decay_fineweb128k_lsa.yaml` |

```bash
lsa --site mycluster native train configs/training/native/qwen35_0_8b_gpt2/pretrain_fineweb_lsa.yaml --gpus 8
lsa --site mycluster native train configs/training/native/qwen35_0_8b_gpt2/decay_fineweb128k_lsa.yaml --gpus 8
lsa --site mycluster native evaluate configs/training/native/qwen35_0_8b_gpt2/decay_fineweb128k_lsa.yaml \
  --checkpoint <checkpoint> --suite benchmark
```

The configs live in `configs/training/native/qwen35_0_8b_gpt2/`, and its README lists the other presets.
Add `--dry-run` to print each resolved run before submitting it.

### Distillation

D-LSA learns its index space from a frozen model, then exports the map the server reads:

```bash
lsa --site mycluster distill targets configs/data/targets/qwen38_27b/long_16k_1m_heavy.yaml
lsa --site mycluster distill train configs/training/distillation/qwen38_27b/long_16k_1m_heavy_key_and_query_d2048.yaml
lsa distill export <run's maps.pt> <out.pt>
```

`long_16k_1m_heavy_key_and_query_d2048_smoke.yaml` runs a single step.

## Released artifacts

| Artifact | Needed for | Status |
|---|---|---|
| SGLang fork, tag `lsa-v0.1.0` | every served run | [available](https://github.com/Prefrontal-AI/sglang/tree/lsa-v0.1.0) |
| Distilled maps for Qwen3.8-27B and GLM-5.3-Flash | every `lsa_*` served config | to come |
| PaperTrail | the `papertrail` suite | to come |
| Prepared RULER sets | RULER, when an upstream source no longer matches its checksum | to come |

## Running on a cluster

A site holds everything specific to one machine, and no run spells it.
It is `configs/site/<name>.yaml`, chosen with `lsa --site <name>`.
The default, `local`, is an example to copy.

<details>
<summary>A Slurm site</summary>

```yaml
# configs/site/mycluster.yaml
launcher: slurm
slurm:
  partition: <partition>
  account: <account>
root: /path/to/shared/storage/lsa
datasets: /path/to/shared/datasets
hf_home: /path/to/hub/cache
mounts: []
images:
  h200: /path/to/lsa-sglang-fork-<sha10>.sqsh
  b300: ""
```

Every `slurm` key is optional, and an unset key adds no sbatch flag.
Large artifacts - targets, maps, records and logs - live under `root`.
`LSA_ROOT` and `HF_HOME` override `root` and `hf_home`, and `LSA_EXTRA_MOUNTS` adds directories to the serving container.
Jobs read the Hugging Face cache offline, so download a model first, for example `hf download Qwen/Qwen3.8-27b`.
`images` names the image built above, or a docker tag your runner can pull, and `run.image=<file or tag>` overrides it for one run.

</details>

## Documentation

| Document | Covers |
|---|---|
| `docs/guide/evaluation.md` | Accuracy, kernel and end-to-end evaluations |
| `docs/guide/training.md` | Distillation and native training |
| `docs/guide/data.md` | Preparing benchmarks, targets and corpora |
| `docs/guide/models.md` | Running LSA on another checkpoint, and the `lsa` block |
| `docs/guide/adapters.md` | Writing an adapter for a new model |
| `docs/guide/development.md` | Checks, tests and conventions |
| `docs/reference/notation.md` | Every symbol and config key, mapped to its name in the code |

## Development

```bash
make sync     # uv sync --locked
make format   # apply every fix a tool can make
make check    # lint, types and the tests the installed environment can run
```

The `gpu` and `cluster` test markers need a GPU and are deselected in CI.

## Citation

```bibtex
@article{lsa2026,
  title  = {Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention},
  author = {To come},
  year   = {2026}
}
```

## License

Apache-2.0, see `LICENSE`.
Third-party code keeps its own license, see `THIRD_PARTY_NOTICES.md`.

## Acknowledgements

Serving builds on [SGLang](https://github.com/sgl-project/sglang).
The NSA training kernels are adapted from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
Long-context accuracy uses [RULER](https://github.com/NVIDIA/RULER) and [LongBench v2](https://github.com/THUDM/LongBench).
