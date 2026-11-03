# Configs

One file, one run - or one file and the file it is a delta on.

```text
configs/
  data/targets/<model>/          select documents and extract frozen-model supervision
  training/distillation/<model>/ train index projections on that supervision
  training/native/<model>/       train a model from scratch under each attention, see its README
  evaluation/accuracy/<model>/   serve and score a model
  evaluation/kernels/<model>/    time one attention layer
  evaluation/production/<model>/ time complete served requests
```

The command chooses the schema: `lsa distill targets`, `lsa distill train`, `lsa native train` or `lsa evaluate <kind>`.
The native presets and the reference they reproduce are in [training/native](training/native/README.md).
`lsa data prepare <card>` tokenizes corpora or generates benchmark prompts from dataset cards rather than a run config.
Target extraction is data preparation, so its configs live under `data/targets/`.
GLM extraction configs declare the document allocation under `data.selection`; there is no separate selection command or config.
GLM recipes name the selected dataset, such as `long_16k_1m`, and the setting that varies, such as `long_16k_1m_heavy_key_and_query_d2048.yaml`.
Each training tree's `base.yaml` holds what its runs share and is not itself a run.
An evaluation file that is not a run starts with an underscore, `_base.yaml` or `_cp8.yaml`, and says so in its first line.

Extraction, training and evaluation have `model`, `data` and `run` blocks.
The schemas reject unknown fields and missing required values when the config is loaded.

**A training run is three blocks.**
`model` is what is trained, `data` is what it reads and `run` is how, and nothing else sits at the top beside `defaults` and `sweep`.
`run.output` has no default, so a run that does not say where it writes is refused, and each model's `base.yaml` or `_base.yaml` says it once.

**A config is named for what it trains on and what varies.**
Everything held fixed is the recipe and is already stated by the directory, so `long_16k_1m_heavy_key_and_query_d2048.yaml` is the recipe on the key and query cache, and a `sweep` over `model.indexer.index_dim` generates one run per width.
The key is the paper's name for it, `index_dim`.

**One dashboard project, and the run name says what produced it.**
`wandb.project` is `lsa` for everything, set once in [env.py](../src/lsa/env.py) rather than per config, because a project split by model or by stage cannot put runs of different models or stages beside each other.
`wandb.prefix` is then `<model>-<stage>-<parameter>` - `glm53flash-distill-long_16k_1m_heavy_key_and_query_d2048` - and wandb's own two words go after it, so the runs of one sweep sort together and still tell each other apart.

**A shared artifact is named under the root, never spelled out.**
A path under the shared root is written `${root:data/targets/glm53_flash/long_16k_1m}`, and the loader resolves it against the site's `root`, which `LSA_ROOT` overrides.
Target caches can be large, so every `data.root` and every extraction's `run.output` names a path there, never one under the repository - `caches/` is gitignored but it is not the right filesystem for it.
A trainer's `run.output` is `${root:training/<trainer>/<model card>}` too, so a run's directory is `<run.name>` there and its `meta.json` names this file, its sweep point and its overrides; only evaluations write a local `results/` first, and `lsa evaluate publish` copies a finished run up.
The root is shaped like this repository - `data/` holds a directory per dataset card kind and the targets `configs/data/targets` extracts, `training/` one per trainer and model card as here, `evaluations/` the scored and timed runs as `lsa.evaluation` nests them, `models/` a directory per model card with the maps under `models/maps/`, and `logs/` the job output - so a config's own path says where its artifact is.
[paths.py](../src/lsa/paths.py) is the same list in code, and the site names the per-user caches and extra mounts.

The schema is the dataclass tree, not a document: [distillation](../src/lsa/training/distillation/config.py), [targets](../src/lsa/training/distillation/data/teacher_config.py) and [evaluation](../src/lsa/evaluation/config.py), one schema per evaluation kind.
A key no field matches fails at load with the fields there are, so the file cannot drift from the schema.

## Two rules

**An asset is named, not pathed.**
`model.name` and `data.name` are [assets](../src/lsa/assets), which carry where the thing is along with everything a path cannot say - which tokenizer wrote a corpus, which family builds a checkpoint, where its rotation sits.
A model card names a hub repository and a pinned revision, resolved in the running user's own HuggingFace cache.
`run.checkpoint.load: pretrained` is the model card's own snapshot; a path there is a run resuming itself.

**A file may be a delta on another.**
A `defaults:` list names what a file builds on and Hydra composes it, with `_self_` saying where the file's own keys land - last, which is what a delta wants.
An entry is a config name relative to the composing file or on its Hydra search path; it has no `.yaml` suffix.
Precedence is the dataclass defaults, then the composed files, then the sweep, then the command line.

A parent holds only what every child wants, because a child can overwrite a key but cannot remove one.
A file that is not a run says so in its first line - [`evaluation/accuracy/glm53_flash/_base.yaml`](evaluation/accuracy/glm53_flash/_base.yaml) is the shared serving setup and has no attention path of its own.

## Sites

A site holds the values that belong to one machine, so no run spells them: the shared `root` and the server images, by name, in `site/<site>.yaml`.
The loader composes the chosen site into every run as `site`, and a config names an image as `${site.images.h200}`, directly or through `_image_b300`, and a path under the root as `${root:models/maps/<map>.pt}`.
The site also holds the hub cache `hf_home`, the `mounts` the serving container adds and the `datasets` directory a card names as `${datasets:<folder>}`.
The hub cache is `HF_HOME` when it is set, then the site's `hf_home`, and otherwise Hugging Face's own default.
The serving container mounts that cache at its own path, so the snapshot a model card names opens inside it, and `mounts` holds only what else a machine keeps weights in.
A name the site does not hold fails at load, and a name it leaves empty fails when the run is built, saying `run.image` is not named.

```bash
lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/lsa_flat_tp8.yaml --dry-run               # the default site
lsa evaluate accuracy configs/evaluation/accuracy/glm53_flash/lsa_flat_tp8.yaml --dry-run site=local    # another site, for that run
lsa --site local records show <a record directory>                                                       # another site, for a command that takes no config
```

The root is `LSA_ROOT` when it is set and the site's `root` otherwise, with no third fallback - a site without a root and no `LSA_ROOT` fails, saying where to set one.
So a shell, a test or an sbatch script that exports `LSA_ROOT` moves every path, and one that does not reads the site.
A relative `root` is taken against the working directory.
Code outside a loaded run reads the same site through one function, the default site or the one `--site` selects.

The default site is chosen in one place, the site module.
The default is the site the site module names, and `--site` selects another.
[local](site/local.yaml) is an example to copy: a root next to where you run, no hub cache and no mounts, images as docker tags or squashfs files you built, and nothing of a cluster in it.

## Naming an evaluation file

A production or accuracy file is `<method>_<mesh>[_<slice>][_b300].yaml`, in the directory of its model.
A kernel file is `<method>.yaml` with no mesh, since a layer is timed on one GPU, and its B300 twin still opens with `b300_`.
The method is the config's own vocabulary - `full` or `native`, `qsa`, `pisa_c8`, `lsa_flat`, `lsa_flat_early_budget`, `lsa_tree`.
Qwen's dense model is `full` in a file name and a run name, and GLM's `native` is its own sparse indexer.
The mesh is always written - `tp1`, `tp8`, `cp8`, `dcp8`, `tp8pp2` - since a bare name meant TP8 on GLM and one card on Qwen.
A slice is written only when the file runs part of the suite on a server sized for it, named by the longest context it serves or the benchmark it alone serves - `1m`, `10m`, `longbench`.
The card comes last, as in a run name, and H200 is the default and is not written.
A file that reruns a paper cell on the server it was scored on names that server's partial under `defaults`: `_cp8_early` is the first battery's context-parallel server, on the image it ran, and `_cp8_10m_early` its request budget at 10.5M tokens, `_cp8_10m_early_chunk32768` with the 32768-token chunk the D-LSA (Flat) and Block Sparse (Flat) RULER cells at 5M and 10M ran at.
The PaperTrail cells past 1M ran on two other builds, `_cp8_papertrail5m` and `_cp8_papertrail10m`, and a slice name `ruler10m` or `papertrail5m` says which cells a file reruns.
A file whose name ends `_early` or `_first` reruns cells of the first battery on a server as it ran then, and `_first` also on the sampling budgets it used, `_first_generation` in the GLM directory.
The file names a partial for each setting a current server has moved - `_no_cache_report`, `_no_kv_splits`, `_tp1_early` - and the records of those cells ran without it.

The file name is the run's name less the model, and the launch appends the card, so a record id reads straight back to its file: `glm53_flash_lsa_flat_cp8_b300__2026-10-22__14-46-15` is `production/glm53_flash/lsa_flat_cp8_b300.yaml`.
A name that starts with an underscore is a partial, `_<mesh>[_<slice>][_b300]` or `_base` or `_image_b300`, and carries no method.

## Running one

```bash
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml --shards 11 --mem-gb 300
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml --shards 11 --mem-gb 300 run.optim.steps=10
```

Command-line overrides are dotted paths into the same tree and beat everything else, including a `sweep`.
