# Reproduce the paper's training families

Run `lsa native train <config> --dry-run <overrides>` to inspect every resolved run before submitting it.
The paper trained with a different trainer, which is not released.
Schedules convert token budgets and warmups to steps with `ceil`, matching the paper's trainer even where its comments print a smaller number.
Seed 42 initializes the model and corpus order; decay branches use corpus seed 1042 with a fresh optimizer and schedule.
An identical seed does not establish identical initial weights across implementations.

The hybrid base and dense, window, unlearned LSA and learned LSA presets select the 18B long-document trunk.
NSA and PISA select the 100B FineWeb trunk through the shared pretrain phase.
The 340M family selects the 20B FineWeb recipe and its cosine schedule.
The universal attention configuration keeps sum reduction; the upstream MQA LSA preset explicitly selects max reduction.

| Family | Config under `qwen35_0_8b_gpt2/` | Resolved attention |
|---|---|---|
| MQA dense and window | `dense_mqa8.yaml`, `window_mqa8.yaml` | 8Q/1KV,128 dims,NoPE; window1536,sinks4 |
| MQA LSA | `lsa_mqa8.yaml` | 8Q/1KV,128 dims,NoPE,max query-head ranking; k512/1024/2048/4096 |
| Learned LSA from q/k | `lsa_indexer.yaml` | 8Q/2KV,256 dims,64 rotary dims; concatenated pre-RoPE heads projected to64 index dims |
| Learned LSA from hidden state | `lsa_input.yaml` | Same attention shape; hidden-state index input |
| One-head LSA | `lsa_onehead.yaml`, `lsa_onehead_indexer.yaml` | 1Q/1KV,128 dims,NoPE; optional summaries; learned variant64dim,L2,STE,NoPE |
| NSA | `nsam8.yaml`, `nsam8_selection.yaml` | compress32/stride16,select64,count16 or32/64,window512 |
| PISA | `pisam8.yaml` | leaf64,branch2,count24,local2 |
| Historical PISA | `pisam8_historical_budgets.yaml` | count40/72,local2 |
| Current PISA window budgets | `pisam8_window_budgets.yaml` | count25/41/73,local8 |

The learned LSA sweep includes additional combinations beyond the original runs.
Pin fields to reproduce a particular run; this resolves the upstream q/k-index KL recipe:

```bash
lsa native train configs/training/native/qwen35_0_8b_gpt2/lsa_indexer.yaml --dry-run attention.top_k=1024 attention.index_training=kl attention.summaries=false attention.index_norm=rms attention.index_rotary_dim=16
```

Use `lsa_input.yaml` for the original hidden-state input, `index_training=ste` for the straight-through runs, `index_rotary_dim=0` for NoPE, and `index_norm=l2` for the L2 ablations.
Unlearned 8Q/2KV LSA is expressible with the learned preset by pinning `attention.index_input=none attention.index_training=none` and the remaining swept fields.
`lsa_mqa8_lr_seeds.yaml` sweeps the long-document MQA LSA peak rates 5e-4/1e-3 and seeds 42/43 at k1024.
Pin `run.runtime.seed=43` on dense or window presets for the corresponding seed 43 runs; `data.seed` follows it.

# Compose another phase with an arm

Keep the arm preset and override the common recipe fields.
For a 100B FineWeb trunk from any long-document preset:

```bash
lsa native train configs/training/native/qwen35_0_8b_gpt2/lsa_mqa8.yaml --dry-run data.name=fineweb_edu_100bt run.optim.steps=95368 run.optim.warmup_steps=954 run.optim.decay_ratio=0.01
```

For an 18B long-document trunk from an NSA or PISA preset, override `data.name=gpt2 run.optim.steps=17167 run.optim.warmup_steps=859 run.optim.decay_ratio=0.2`.
Keep `run.optim.schedule=wsd`, global tokens 1048576, and the selected peak rate.
The long trunk decay-start checkpoint is step 13733; the FineWeb trunk checkpoint is step 94414.
The provided `pretrain_long.yaml` and `pretrain_fineweb.yaml` compose these same phases with the dense MQA arm.

For a 32k decay from the FineWeb trunk, use the same arm and branch from its own checkpoint:

```bash
lsa native train configs/training/native/qwen35_0_8b_gpt2/nsam8_selection.yaml --dry-run attention.select_block_count=32 data.name=gpt2 data.seq_len=32768 data.seed=1042 run.init_from=/absolute/path/to/nsam8/checkpoint/step-94414 run.optim.steps=17548 run.optim.warmup_steps=96 run.optim.decay_ratio=0.97 run.optim.tokens_per_rank=32768 run.optim.tokens_per_step=1048576 run.runtime.activation_checkpoint=full run.checkpoint.decay_start=false
```

A 64k branch changes `data.seq_len=65536 run.optim.tokens_per_rank=65536` and keeps that budget.
The 128k FineWeb-source recipe changes `data.seq_len=131072 run.optim.tokens_per_rank=131072 run.optim.tokens_per_step=2097152 run.optim.steps=5866 run.optim.warmup_steps=48`, preserving the source 12.3B budget.
A branch from the long-document 18B trunk uses checkpoint 13733 and `run.optim.steps=3434 run.optim.warmup_steps=96`, global 1048576 at all three lengths.
The dense MQA examples are `decays.yaml` for the long trunk, `decays_fineweb.yaml` for 32k/64k FineWeb branches, and `decay_fineweb128k.yaml` for 128k.
These examples never substitute another arm when loading its checkpoint; compose the matching arm or supply its overrides.

# FineWeb 128k decays for every arm

Each arm pretrains on the 100B FineWeb trunk and decays at 128k from its own decay-start checkpoint, step 94414.
The decay presets share `phase/decay128k_fineweb.yaml`: 5866 steps, warmup 48, global tokens 2097152, sequence 131072.
Pretrain the arm with the preset in the second column, then submit the decay with the preset in the third.

| Arm | Pretrain preset | 128k decay preset | Pretrain budget | Decay budget |
|---|---|---|---|---|
| Dense | `pretrain_fineweb.yaml` | `decay_fineweb128k.yaml` | all keys | all keys |
| Window | `pretrain_fineweb_window.yaml` | `decay_fineweb128k_window.yaml` | sink 4, window 1536 | sink 4, window 2560 |
| NSA | `nsam8.yaml` | `decay_fineweb128k_nsa.yaml` | 16 blocks of 64, window 512 | 32 blocks of 64, window 512 |
| PISA | `pisam8.yaml` | `decay_fineweb128k_pisa.yaml` | 24 blocks of 64, 2 local | 40 blocks of 64, 2 local |
| LSA | `pretrain_fineweb_lsa.yaml` | `decay_fineweb128k_lsa.yaml` | sink 4, window 512, k 1024 | sink 4, window 512, k 2048 |

The pretrain presets write `<output>/<run.name>/checkpoint/step-94414` and each decay preset reads it, so no override is needed when both run under the same output.
Resolve any of them with `lsa native train <config> --dry-run`.

# Released 128k checkpoints

The ten 128k decays of the paper's comparison are released as torchtitan step folders under `training/native/released/<checkpoint>/checkpoint/step-5866`.
Each `evaluate_released_<arm>.yaml` is the arm's 128k decay preset with its checkpoint in `run.evaluation.checkpoint`, so `evaluate` scores it with no `--checkpoint`:

| Arm | Checkpoint | Preset |
|---|---|---|
| Dense | `decay128k-dense` | `evaluate_released_dense.yaml` |
| Window 2,560 and 4,608 | `decay128k-swa-2560`, `decay128k-swa-4608` | `evaluate_released_swa_2560.yaml`, `evaluate_released_swa_4608.yaml` |
| NSA 32 and 64 blocks | `decay128k-nsa-2048`, `decay128k-nsa-4096` | `evaluate_released_nsa_2048.yaml`, `evaluate_released_nsa_4096.yaml` |
| PISA 40 and 72 blocks | `decay128k-pisa-2368`, `decay128k-pisa-4416` | `evaluate_released_pisa_2368.yaml`, `evaluate_released_pisa_4416.yaml` |
| LSA k 1024, 2048 and 4096 | `decay128k-lsa-1024`, `decay128k-lsa-2048`, `decay128k-lsa-4096` | `evaluate_released_lsa_k1024.yaml`, `evaluate_released_lsa_k2048.yaml`, `evaluate_released_lsa_k4096.yaml` |

```bash
lsa native evaluate configs/training/native/qwen35_0_8b_gpt2/evaluate_released_lsa_k2048.yaml --suite benchmark --tasks core
```

A `--checkpoint` on the command line still wins, and a config that names none and gets none is refused.
The records land under the preset's own run name, `released-decay128k-lsa-<k>` with the sequence length appended, so they never mix with a retrain's.
The held-out suite builds a block mask for a whole 128k sequence, which does not fit an H200, so score it at a shorter length with `data.seq_len=16384 run.optim.tokens_per_rank=16384`.

# DSA matrix

The files under `dense_340m_gpt2/` express the live/detached hidden-state indexer, projected or sliced q/k inputs, 64/32 dim indexers, bias mixing, ReLU, sparse-after-warmup, and KL-on-selection variants.
The base side-channel arm preserves its historical qkv/output biases; the other variants turn them off.
`dsa_lr.yaml` sweeps peak rates 1e-4/3e-4/1e-3 and seeds 42/43 on the detached projected 64 dim sparse arm.
The resolved global batch is 1048576, steps 19074, warmup 401, followed by cosine decay to one tenth of the peak.
A cosine run has no stable-stage checkpoint.
