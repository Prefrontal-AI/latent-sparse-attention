"""Benchmark a native checkpoint on the standard pretraining tasks, in process, under the attention it trained with.

    lsa native evaluate --suite benchmark configs/training/native/qwen35_0_8b/multires.yaml --checkpoint <step dir> --tasks core

The checkpoint is loaded the way `lsa native evaluate` loads it for the held-out set and handed to lm-eval as a model, so a task sees
a language model and the arm is whatever attention the config names. Arms are compared example by example, which is
why the records keep every sample.

| module | what it does |
|---|---|
| [lm](lm.py) | lm-eval's model interface over a native checkpoint - loglikelihood, rolling loglikelihood and greedy generation |
| [tasks](tasks.py) | the presets, and RULER's fixed prompt files as lm-eval tasks |
| [records](records.py) | the directory a run leaves, one row per example |
| [report](report.py) | the arms side by side, with paired bootstrap differences |
| [run](run.py) | one checkpoint through one preset into one records directory |

The hub is read offline, as every stage reads it, so the datasets a task needs are in the cache before a job starts.
"""

from lsa import env

# Before any module of the package imports lm-eval, datasets or transformers - the hub reads its offline
# switches at import
env.prefer_offline_hub()
