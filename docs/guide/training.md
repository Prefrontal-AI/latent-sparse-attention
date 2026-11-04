# Training

Learned LSA distils the indexer while the language model is frozen, and this is how to run that.

```bash
uv sync --extra training
eval "$(lsa env)"   # LSA_ROOT and the offline HuggingFace caches, as a job has them
```

One extra installs the whole training path, and B300 (Blackwell) nodes add `--extra blackwell` for the FlashAttention-4 kernels.
Every command below submits slurm jobs and runs nothing where it is typed - `--dry-run` prints the runs and submits nothing, and `--gpus`, `--cpus`, `--mem-gb`, `--hours`, `--partition` and `--reservation` change what a job asks for.
A `<partition>` or `<reservation>` in an example is a placeholder: your cluster's own values are the `slurm:` keys of its site file, `configs/site/<site>.yaml`, and the site's `launcher` decides whether a job is submitted through sbatch or runs on this machine.
[development.md](development.md#torch) is what torch resolves from and why.

Native attention needs none of this, and neither does anything that serves a map already trained; [evaluation.md](evaluation.md) is how to run those.

## Distilling the indexer

One linear map per layer, a key projection and a query projection, carries the key and query the indexer reads into the narrow space the indexer clusters and scores in.
That narrow space is the whole index: Ward clusters it, the centroids are means of it and the scan scores against it.

### On GLM-5.3-Flash

GLM's recipe trains on a balanced dataset of lengths from 16k to 1M.
Prepare the tokenized corpus, extract targets with its configured document selection, then train.

```bash
lsa data prepare glm53_flash
lsa distill targets configs/data/targets/glm53_flash/long_16k_1m_heavy_key_and_query.yaml \
    --reservation <reservation> --mem-gb 175 --workers 8
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml \
    --shards 11 --mem-gb 300 --hours 12 --partition <partition>
lsa distill export $LSA_ROOT/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048 \
    $LSA_ROOT/models/maps/glm53_flash__long-16k-1M-heavy-key-and-query-d2048.pt
```

The indexer reads GLM's key latent and every head's query summed, which is what the server serves.
The extraction keeps only those - `run.input: key_and_query`, a fifth of the attention input `x` on GLM - and the run trains a map of them, so `lsa distill export` writes the map the server reads.

The extraction's `data.selection` declares which documents and windows it reads, balanced by length and domain.
Extraction records the resolved selection as `selection.json` inside its target collection before loading weights.
Every length bucket holds the same tokens, and inside a bucket every domain holds the same share wherever it has documents that long.
`configs/data/targets/` holds extraction configs with their document allocation; `configs/training/distillation/` holds runs that train and score on those targets.
`run.collection` names the extracted collection, and the trainer names the extraction cache root as `data.root`.
Existing artifact directories retain their historical names even when a config has been renamed.

An extraction that names `run.stages` runs one span of the stack per job, each on one card, and `lsa distill targets` submits every span, or the one `--stage s` names.
`--workers` submits that many jobs per span, and several go faster, because each claims the documents it reads.
An extraction without stages loads the whole stack, and `--device auto` with `--gpus` spreads it over the node as a transformers device map, each layer's vectors built where that layer's weights already are.
`configs/data/targets/glm53_flash/published.yaml` is the older 64k-only extraction.

The model does not fit one card, so GLM distils from a cache rather than online: one extraction pass, then any number of runs.

The index space is the key and query the layers read: the key latent and the heads' queries summed on GLM, the position-free part of the logit on Qwen.
`configs/data/targets/glm53_flash/long_16k_1m_heavy_key_and_query.yaml` extracts it, and `lsa distill targets` on that config is how the cache is reproduced.

The attention input `x` is a research option: `model_dim` dims a token, with no heads to regroup and no rotary to undo.
Its run is `configs/training/distillation/glm53_flash/long_16k_1m_heavy_x.yaml`, which reads a cache of `x` that no shipped extraction config writes any more, and no shipped evaluation config serves it.

```bash
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_x.yaml --shards 11 --mem-gb 300
```

### On Qwen3.8-27B - the teacher online

A cache of `x` for Qwen3.8-27B over the heavy mix would be some 60 TiB, so its run reads none: `data.teacher` names the extraction config, and the frozen model reads the documents the run draws as it trains.

```bash
lsa data prepare qwen38_27b --sources <source>          # the heavy mix's corpus in Qwen's ids, a job a source
lsa distill train configs/training/distillation/qwen38_27b/long_16k_1m_heavy_key_and_query_d2048.yaml \
    --gpus 8 --cpus 64 --mem-gb 1200 --hours 24 --partition <partition>
```

The maps train on the first card and a copy of the model reads documents on each of the others, every layer of a document at once, into the same pool a cache feeds.
The documents, their query positions and their targets are the extraction's own, so the targets are the ones `lsa distill targets` would have written.
A run is one job rather than a shard a layer, because a shard would read every document again for its own layers.

```bash
lsa distill train configs/training/distillation/qwen38_27b/long_16k_1m_heavy_key_and_query_d2048.yaml \
    --gpus 8 --nodes 2 --cpus 64 --mem-gb 1200 --hours 12 --partition <partition>
```

`--nodes n` spreads the teacher rather than the layers: the trainer stays on the first node's first card, and every other card of every node holds a copy of the model.
A copy on any node takes the next document no other copy has claimed, so each node reads a disjoint share of the documents, and the copies on the other nodes leave what they read under the run directory's `teacher_staging/` for the trainer to load.
The targets are the same whichever node reads a document, and the pools meet the documents in the same order.
`data.teacher.staging_gb` bounds what waits on the shared disk; past it the other nodes stop claiming until the trainer catches up.
The held-out sets are read once when the run starts and kept in host memory, which is why the long ones name a `limit`.

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('kernels-community/flash-attn3', repo_type='kernel', revision='3c1451f803c146c54222305a8c350f7f46bb5135')"
```

Qwen's full-attention layers read a prefill chunk through FlashAttention-3 off the kernel hub, its 4 key heads read by their groups of query heads as they are, at about twice the rate of one key head expanded a group - a 1M document in some 560 s on one card rather than 1,080.
A job may have no network, so the kernel is fetched into the HF cache once, from a machine that has it, at the revision the code pins - the whole repository, 7.3 GB, because the hub loader offline resolves the whole snapshot before it picks the build for this torch.
An extraction and the online teacher read through the same kernel and agree bit for bit; a cache written before the kernel changed does not.

The model card carries the rotation the model is served under - YaRN to 10M as `rope_parameters` - and every load applies it, so the targets are the attention the served model pays.
On Qwen the key and query input is the position-free part of the logit: each key head's unrotated dims side by side, and each group's query heads summed onto the slice of the key head they read.

### The model is frozen

One pass per document extracts what the objective and the metrics need - the key and query at every position (the attention input `x` for the research option), and the attention the teacher actually paid to the keys a whole-bucket read may see - so a run loads in minutes rather than holding a model for the whole of training.
The extraction drives the model through its [adapter](adapters.md), so a checkpoint an adapter exists for is one this extracts from.

The objective is per-key KL to the teacher.
A loss aimed at the bucket score or the served read is a dead end.
The loss shapes the space and Ward shapes the partition.
What comes out is the indexer an injected checkpoint carries, not a trained language model.

### Configuring a run

A run is a yaml file with the same sweep and the same dotted overrides an experiment has.

```bash
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml run.optim.steps=0  # the untrained report
```

```yaml
sweep:
  run.optim.lr: [0.0003, 0.001]     # one run per rate
```

A run is three blocks and nothing else: `model` is what is trained - the model card, the indexer and the learned space - `data` is which targets it reads and which of their documents train and score it, and `run` is how - the name, the output, the objective, the optimiser, the reports and wandb.
`run.output` has no default, so a file that does not say where it writes is refused.
Each model's `base.yaml` sets it once, `${root:training/distillation/<model card>}` under the shared root, the same directory names `configs/training/distillation` has, along with `model.name`, and every run in that directory names `base` first in its `defaults`.
A run writes to `<run.output>/<run.name>`, so a config's run directory is its name - `long_16k_1m_heavy_key_and_query_d2048_seed1` for seed 1 of `long_16k_1m_heavy_key_and_query_d2048.yaml` - and never sits inside a checkout a worktree removal can take with it.
Its `meta.json` names the config file, its point of the sweep and the command-line overrides, and the full commit with whether the checkout had uncommitted edits.
A swept key pinned on the command line narrows the sweep to that value, so `run.optim.seed=1` is the one run named `_seed1`.

### Training a run in shards

```bash
lsa distill train configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml --shards 11
```

A run's layers share nothing, so they can train in separate jobs, each holding only its own share of the targets.
A shard keeps the run's step numbering and draws every step off the run's one generator, so it learns for its layers the maps the whole run would.
`--shards n` submits the shards as an array job and a join that waits on it, which merges the maps and takes each report as the layer-weighted mean.
A finished shard is skipped, so the same command after a failed join trains nothing and joins again.

### Stopping and resuming a run

```bash
lsa distill train <config>            # again, into the same run.output - continues from the last resume state
lsa distill train <config> --force    # starts it over
```

Every report writes `resume.pt` beside its `maps_step<N>.pt`: the maps, the optimiser, the RNG, each layer's pool plan and how far its steps are, and the wandb run.
slurm warns a job 90 s before a preemption or its time limit, and the run then stops at the next step boundary, writes the same state at the step it reached, and is requeued to continue from it.
A run that died without warning continues from its last report.

A resumed run trains exactly the steps an uninterrupted one would - the same documents in the same order, the same query rows, the same maps bit for bit.
Off the online teacher too: the order is the run's plan rather than the order the copies finished in, and a document some step had already trained on is read again and comes out the same.
That re-read is the cost of a resume - every document a pool held, up to a window of 8 a layer.
A state written under another configuration is refused rather than resumed, since the plan it holds is another run's.

### Scoring a map at another read

```bash
lsa distill rescore configs/training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048.yaml \
    'model.space.init=${root:training/distillation/glm53_flash/long_16k_1m_heavy_key_and_query_d2048/maps.pt}'
```

For the per-key objective, the read's settings - the bucket size, the top-k and the pool size - change what a report measures without changing what a map is.
Rescoring scores maps that are already trained, named by `model.space.init`, over a grid of reads, with no gradient and no training document loaded.
The targets' visibility rule is the one thing it cannot change, and the loader refuses a grid that disagrees with the targets.

### Scoring an exported map's served read

```bash
lsa distill served-read configs/training/distillation/glm53_flash/long_16k_1m_heavy_x_dense_on_native_served.yaml \
    --shards 8 --reservation <reservation>
```

The served read of every design the config lists, the exact read at each design's budget and GLM's own picks, on the same rows of every held-out document and layer, a GPU a shard.
`model.space.init` names the exported map, and the picks are the `_native.pt` sidecars `lsa distill native-picks` writes.
Each (document, layer) is a record, and the last shard writes the tables by read, layer and length beside them, with the cells where a design reads more than 0.02 of mass under native flagged; `--tables` rewrites them, and `--documents` narrows the documents to a list.

### Scoring the RULER probe set

```bash
lsa distill served-read configs/training/distillation/glm53_flash/ruler_probes_served_512k.yaml --shards 8 --partition <partition>
lsa distill served-read configs/training/distillation/glm53_flash/ruler_probes_served_512k.yaml --tables
```

The same command over a probe cache - `ruler_probes_128k`, `512k` or `1m`, written by `lsa distill targets` - scores what each read holds of the keys a probe was meant to find.
A needle task's rows are its last 32 positions, aimed at the needle's value tokens and at its sentence.
A cwe row's are its list's items, aimed at that word's earlier occurrences, and its last 32 positions, aimed at nothing.
A target in the window or the sinks is left out of every count, because every read holds them, so `target_held`, `target_any` and `line_held` are over the long-range keys alone.

`run.report.served.arms` names the offline arms read beside the served designs.
A `reference` arm builds the buckets and node bucket levels of a served design with `lsa.core` at that design's sizes, with the research options, and `agrees_with` also scores how much of a served design's long-range read it holds.
A `pisa` arm is [PISA](../../src/lsa/baselines/pisa.py) over GLM's own key latent and heads' queries at GLM's softmax scale, or over a design's index keys and index queries as one head at `1 / tau`.

```yaml
arms:
  - {name: A0, design: tree, agrees_with: tree_bf16}
  - {name: A2, design: tree, cluster_algorithm: position, node_cluster_algorithm: position}
  - {name: A3, design: tree, node_bucket_score: lse}
  - {name: B1, kind: pisa, space: attention, block_size: 64, keep_count: 64}
  - {name: B3, kind: pisa, space: index, design: tree, block_size: 64, keep_count: 64}
```

An arm that reads at a temperature - a LogSumExp score, or B3 - takes the map's own from its sidecar, which `lsa distill export` writes and `lsa distill calibrate-tau <config> <map>` records for a map that has none, and the read refuses to start without it.
`tau` fixes it on an arm, `tau_scale: [0.5, 1, 2]` reads the arm once at each multiple of the map's as arms named `A3@tau0.5`, `A3@tau1` and `A3@tau2`, and `taus: [0.05]` sweeps explicit ones as `A3@tau=0.05`.
The torch reference runs on the host, so a 1M prompt's buckets are built once for the arms that group alike and each arm is read a row at a time.
`report.md` then adds, beside the tables of every read, one table of a metric by task and length for each kind of query, and the cwe rows split by number of targets and by distance back.
A cell is the mean over prompts of each prompt's mean, its rows first and its layers next, with its standard error.

### Reading a report

A report scores two reads of the held-out documents against the teacher's cached keys.
The exact read is the top `top_k` keys by index score, the ceiling any bucketed read of the same space sits under.
The served read is what a server reads - each design of `run.report.served` built over the whole document and read by the serving selector, Ward's buckets, the pick, the expansion and the beam included - on a GPU only, at `run.report.served.rows` rows a document, and it sits under `served` in the report.
Flat is the default design; `designs: []` turns the served read off.
The teacher's padding entries, of mass 0, are not teacher keys and count for neither.

Attention-mass recall is the direct mechanistic measure of how much teacher probability the fixed-budget read retains.
Recall16, recall64 and peak capture expose failures that the average mass can hide, and downstream tasks remain the final check because an offline gain does not predict a served one.

### Serving a map

```bash
lsa distill inject --model glm53_flash --maps "$LSA_ROOT/training/distillation/glm53_flash/<run>/maps.pt" \
    --out "$LSA_ROOT/models/glm53_flash_index"
```

The checkpoint's files are copied rather than linked, because a link would point into the injecting user's own HuggingFace cache, and its `lsa_index` block names the maps it was read from by path and sha256.

A run's maps mean nothing without the checkpoint they belong to, and `lsa distill inject` writes the pair as one thing: the source checkpoint with the two index projections added per indexed layer, a `lsa_index` block in its `config.json`, and the flat projections file the serving plugin reads.
The weight shards are symlinked rather than copied, so a 27B checkpoint costs megabytes.
It refuses a map recorded against another model, a layer the checkpoint does not attend on, and two maps of one layer that disagree on shape.
`--maps` takes several files when a rung was distilled in halves.
Serving the result, and timing the layer, the rest of the model and the served model under it, are in [evaluation.md](evaluation.md#serving-a-distilled-space).

## The corpus

[data.md](data.md) is what the datasets hold, how to tokenize one and where the eval prompts come from.
`prepare` does what a dataset card needs, and the agent traces are fed from outside the collected sources.
Download the `edgebench` source from the `traces-v1` release as [data.md](data.md#sources) describes, then tokenize it:

```bash
lsa data prepare glm53_flash --sources edgebench
```

The release holds the agent sessions as documents the way a GLM teacher reads them.
One session is one document - its longest main-thread call with that call's response as the final assistant turn, through the teacher's own chat template, reasoning kept.
The command tokenizes the `edgebench` source into `glm53_flash` beside the collected ones.

## A card is named, not pathed

`data.name` and `model.name` are cards under [`src/lsa/assets`](../../src/lsa/assets), and a card carries where the thing is along with everything else about it.
A tokenized dataset card names its directory; a model card names a hub repository and a pinned revision, resolved in the running user's own HuggingFace cache.

```yaml
data:
  name: qwen3_5
  split: train
```

| tokenized card | what it is |
|---|---|
| [qwen3_5](../../src/lsa/assets/datasets/tokenized/qwen3_5.yaml) | the nine collected sources, tokenized with Qwen3.5's tokenizer |
| [glm53_flash](../../src/lsa/assets/datasets/tokenized/glm53_flash.yaml) | the same sources and the agent trajectories, tokenized with GLM-5.3-Flash's |

| model card | what it is |
|---|---|
| [qwen35_4b](../../src/lsa/assets/models/qwen35_4b.yaml) | what the 4B distillation runs load |
| [glm53_flash](../../src/lsa/assets/models/glm53_flash.yaml) | the target model, 45 layers and fp8 on disk |

A path says where bytes are and nothing else.
A tokenized card says which sources and which tokenizer, and how long a row is, and its manifest beside the bytes says how wide the vocabulary is and how many rows each source gave; a model card says which family builds the checkpoint, which flavour the published one is, and where its rotation sits.
Those are the facts that decide whether a run is reading what it thinks it is, and none of them is in a path.
`path` is still there for a thing that has no card yet, and a dataset read that way takes the card it was tokenized under off the manifest preparing it leaves beside the bytes.

A tokenized row is token ids and nothing else, so a run that names the wrong corpus is naming the wrong vocabulary.

## Native - a model trained under the multi-resolution attention

```bash
lsa native train configs/training/native/qwen35_0_8b/multires.yaml --gpus 8 --hours 4
lsa native train configs/training/native/qwen35_0_8b/dense.yaml --dry-run run.optim.steps=20
lsa native evaluate configs/training/native/qwen35_0_8b/multires.yaml --checkpoint ${LSA_ROOT}/training/native/qwen35_0_8b/b16k-multires/checkpoint/step-8000 attention.coarse=false run.name=evaluate-multires-coarse-false
```

The native indexer kernels are compiled the first time a job uses them, on the GPU node, through torch's inline extension loader.
The node needs `nvcc` on its `PATH`, at the CUDA version of the installed torch, and `ninja`, which the `training` extra installs into `.venv/bin`.
The `cuda_sparse` kernels add the interpreter's own `bin` directory to `PATH` when they find no `ninja`, and the FA3 indexer does not.
A script of your own that builds the FA3 indexer needs `export PATH=$PWD/.venv/bin:$PATH`, or `uv run`, which puts `.venv/bin` first.
The `cuda_sparse` build needs neither.
The first step of a run waits for the build, and later runs reuse torch's extension cache.

One file is one arm: `base.yaml` holds the model from random initialisation, the corpus and the schedule, and each arm names it and sets `attention`.
Dense, a sliding window of the same key budget, multi-resolution attention, the same selection with no summaries, the equal-chunk partition and the flat LSA read lsa serves are the six that ship.
`data.name` names the corpus by card - `qwen3_5` or `gpt2` for tokenized rows, `fineweb_edu_100bt` for token shards, a source card for text tokenized as it is read - and `data.mixture` replaces it with weighted cards:

```yaml
data:
  name: gpt2
  seq_len: 4096
  packing: concatenate
  readers: 4
  seed: 42
  mixture:
    - {name: fineweb_edu_100bt, weight: 0.9}
    - {name: gpt2, weight: 0.1}
```

`packing` is `concatenate` for pretraining and `span` for a long-document decay, and defaults to what the corpus allows.
The held-out pass reads a tokenized corpus, so a run on shards or text sets `run.evaluation.every: 0`.
`evaluate` scores a checkpoint's weights on the held-out set, the default `--suite held-out`, with no optimiser and no training data, under the attention the file and its overrides name, so a trained model can be read with the summaries off.
The held-out pass, in a run or an evaluation, writes a summary and the per-sequence scores under the run's `evaluation/` directory, with the loss in position buckets and the share of the softmax on coarse terms.
`evaluate --suite benchmark` scores a checkpoint on standard pretraining tasks through lm-eval, in process, under the attention the file names:

```bash
lsa native evaluate --suite benchmark configs/training/native/qwen35_0_8b/multires.yaml --checkpoint ${LSA_ROOT}/training/native/qwen35_0_8b/b16k-multires/checkpoint/step-8000 --tasks core --gpus 8
lsa native evaluate --suite benchmark configs/training/native/qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml --tasks ruler-4k --gpus 1
lsa native report ${LSA_ROOT}/training/native/benchmark --presets core,ruler-4k
```

`--tasks` is a preset - `core`, `ruler`, its 13 tasks at 4,096, 8,192 and 16,384, `ruler-4k`, the same 13 at 4,096, and `ruler-32k` past the training length - or a comma-separated list of tasks, with a RULER cell spelled `ruler_<task>_<length>`.
Every rank holds a full replica and lm-eval hands it a share of each task's requests.
A run leaves `<records>/<label>/<tasks>/` with a `config.json` naming the checkpoint, its step and the attention, lm-eval's `results.json`, and one row per example under `samples/`, so two arms can be compared example by example with `lsa native report`.
The RULER files are lm-eval's port of RULER, sized by the GPT-2 tokenizer of the `qwen35_0_8b_gpt2` family as the paper's evaluation did, and prepared once into a root of their own with `lsa native prepare-ruler --lengths 4096 8192 16384`.
They are for that family only: another tokenizer, such as the Qwen3.5 family's under `configs/training/native/qwen35_0_8b`, would see mis-sized prompts, so score RULER with a config of the GPT-2 family.
The [native preset README](../../configs/training/native/qwen35_0_8b_gpt2/README.md) says where the files land and why `lsa data prepare ruler` is not the command.
The datasets of the core tasks are fetched once, from a machine with internet access, into the caches that `lsa env` exports:

```bash
eval "$(lsa env)"
HF_HUB_OFFLINE=0 uv run --no-sync python -c 'from lm_eval.tasks import TaskManager, get_task_dict; get_task_dict(["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "lambada_openai", "wikitext", "sciq"], TaskManager())'
```
