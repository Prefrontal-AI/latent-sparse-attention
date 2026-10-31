# Running LSA on another model

A model qualifies for training if the trainer can read its attention block's input, `x`, which carries no position for any model.
The design serves GLM-5.3-Flash on its key latent and its heads' queries, through a map trained on that input.

The sections below on the rotary layout describe the extraction that read the position-free part of the key, which is the index space before the overhaul.
They are re-done with the extraction that stores `x`, and the adapter hook they describe changes with it.

Two things read a checkpoint, and they qualify it separately.
Serving is sglang with the plugin, which drives the family it has hooks for - GLM-5.3-Flash - and a served config names the checkpoint in `model.name` and nothing else about it.
Everything below is the other one: the adapter that teacher extraction and the probe read a checkpoint through, which is what a distillation needs before a model can be served with a learned map.

## Screen a checkpoint

The probe submits a CPU Slurm job that reads `config.json` without loading model weights.
Add `--dry-run` to inspect the submission.

```bash
lsa probe Qwen/Qwen3.5-4B-Base            # supported, and prints the config block
lsa probe zai-org/GLM-5.3-Flash           # supported - MLA, NoPE, its own adapter
lsa probe HuggingFaceTB/SmolLM3-3B-Base   # full RoPE - says so and stops
```

It reports against the adapter that would actually drive the checkpoint, so a hybrid model with an adapter of its own is read with that adapter's layer types rather than against `full_attention`.
A checkpoint the installed transformers cannot read reports the reason instead of attempting extraction.

## Name it

A shape that is already supported needs nothing else in the model block.

```yaml
model:
  name: Qwen/Qwen3.5-4B-Base
```

A name with no `/` is a model asset under [`src/lsa/assets/models`](../../src/lsa/assets/models), which names the hub `repo` and a pinned `revision` and fills in whatever the block leaves unsaid.
Its snapshot is read from your own hub cache, `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>`, so the cache has to hold that revision before a job starts on a cluster with no egress.
A name with a `/` is a hub id or a path, taken as it is.

The adapter comes from the checkpoint's own `model_type`, falling back to a generic HF decoder that reads `q_proj`, `k_proj` and `v_proj`, applies QK-norm where a model has it, projects out through `o_proj` or `dense`, and takes the rotation from the model's own modeling file.

Where transformers gives one thing two names, the generic path knows both, because knowing one is worse than knowing neither: a QK-norm looked up as `q_norm` alone is silently skipped on the six families that call it `q_layernorm`, and a skipped norm scores rather than raises.
The same goes for the rotation - some models slice the partial-rotary dims inside `apply_rotary_pos_emb` and some slice in their attention forward before calling it, so the generic `rotate` slices to the dims of the model's own `cos` table and passes the rest through, which is correct for both.

## The rotary layout is measured, not declared

Reading a prefix as a suffix indexes on the rotary half and nothing fails - it just scores noise.
So an extraction rotates a key at several positions before it caches anything, sees which dims move, and builds the index space from that.

Pinning `rotary_end` and `rotary_dim` in the file is still allowed, and then the measurement checks them instead of supplying them.
Either way what was resolved lands in each collection's `manifest.json` beside the targets, the vectors it describes, so a config that names only the checkpoint still reproduces.

A model that does not rotate at all needs no measurement, because there is nothing to measure.
An adapter says so by returning None from `rotary_embedding`, and the default raises rather than reading a missing `rotary_emb` as NoPE - guessing that wrong indexes on rotated dims and only ever shows up as a bad score.

## A hybrid

One more line, when it does not call its quadratic layers `full_attention`.

```yaml
model:
  name: <a hybrid checkpoint>
  layer_types: [<what it calls its quadratic attention>]
```

There is no standard name - Qwen3.5 says `full_attention`, Qwen3.8-Flash-Next says `qwen_sparse_attention`, GLM-5.3-Flash says `deepseek_sparse_attention`, DeepSeek-V4-Flash declares three kinds and none of them that.
A stack that matches nothing fails with the names it does declare, and the probe prints them too, so naming the right one is the fix.

That only helps when the layer really is attention over per-token keys and values.
A layer that pools or compresses its keys before attending is a different thing, and is the last section.

## What is out of reach

**DeepSeek-V4-Flash**, and none of it is the index space - that part is a clean `RotarySpec("suffix", 64)` of a 512-dim entry.

Its attention differs from every model above in five ways at once.
Keys and values are the same tensor.
Each head carries a learnable attention sink, which is an extra logit in the softmax that `topk_attention` does not model.
The rotation is interleaved within the trailing slice and applied by a function taking one tensor rather than a query and a key, so neither the layout probe nor the generic `rotate` can call it.
Its rotary tables arrive as a `{"main", "compress"}` dict rather than a pair.
And the conjugate rotation applied to the attention *output* has to be undone before the grouped output projection.

On top of that its compressor concatenates compressed entries onto the key axis after the sliding-window cache update, so the thing being attended over is not one sequence of per-token keys.
That is the case `read` exists for, and it is a design question rather than an adapter: nothing in `lsa.core` means "token", only "index slot `i` sits at position `i`", so an adapter would have to keep the window size, the sink size and the top-k in entries and expand a selected entry id back to the tokens it covers.
Deciding what a window of 512 *means* when an entry is eight tokens is the actual work, and it is not written.

An interleaved partial rotation is rejected rather than read as a prefix.
`position_free` drops a contiguous slice, so that model needs its own.

Teaching the adapter a model it does not already drive is [adapters.md](adapters.md).
