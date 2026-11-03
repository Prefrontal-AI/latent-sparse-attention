"""Native training - a model trained from random initialisation under a multi-resolution attention.

The model has had the attention since its first step: every closed key enters the softmax once, itself
or through its bucket's summary, so cross entropy is the only loss and its gradient reaches every bucket.
"Native" is the training regime. It has nothing to do with the native arm, a model's own sparse attention such as
GLM's indexer.

This package is independent of its sibling: it imports the repository's shared pieces - the core Ward, the
corpus, the config loader, slurm - and nothing from [distillation](../distillation/__init__.py).

| module | what it does |
|---|---|
| [config](config.py) | the run schema - the attention's knobs, the model, the corpus and the schedule |
| [constants](constants.py) | the names a knob chooses between, and what a job and a held-out pass default to |
| [buckets](buckets.py) | the partition of the closed keys, what each bucket is summarised by, and which a query reads |
| [layers](layers.py) | the attention - a reference that is the oracle, the flex training path, and torchtitan's inner attention |
| [models](models.py) | Qwen3.5 with its full-attention layers swapped for the arm a run names |
| [losses](losses.py) | what a held-out pass measures - loss per position bucket and per source |
| [lsa](lsa.py) | the flat LSA read as an arm - its index space, buckets and the keys a query reads |
| [evaluation](evaluation.py) | the held-out evaluation, and building a model to score a checkpoint with |
| [trainer](trainer.py) | torchtitan's trainer configured from a run, with a corpus loader |
| [benchmark](benchmark/__init__.py) | `lsa native evaluate --suite benchmark`, a checkpoint scored on standard tasks through lm-eval, in process |
| [ruler_data](ruler_data.py) | `lsa native prepare-ruler`, the RULER files the benchmark reads, rendered as the paper's evaluation did |
| [run](run.py) | `lsa native train` and `lsa native evaluate`, which submit one slurm job per run and one task per GPU |

`config`, `constants`, `buckets`, `lsa` and `losses` reach torch and `lsa` alone, so `make check` gates them with no
framework installed; `layers`, `models`, `evaluation`, `trainer` and `run` need torchtitan.
"""
