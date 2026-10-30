# Guide

How to run the thing.

These documents hold no decisions and no evidence.
What is here is the operational surface: the commands, the config fields, and the extension points.

| doc | what it runs |
|---|---|
| [evaluation.md](evaluation.md) | a served experiment - scoring it, timing its kernels or its requests, the yaml, the overrides, the sweeps, slurm and the viewer |
| [models.md](models.md) | LSA on another checkpoint - screening one, configuring it, and what is out of reach |
| [adapters.md](adapters.md) | a model teacher extraction does not already read - the five hooks, and how to check one |
| [training.md](training.md) | learned LSA - distil the indexer, extract its targets, export and inject the map |
| [development.md](development.md) | the local loop - what the gates fix, adding a dependency, and the torch pin |
| [data.md](data.md) | tokenize a corpus, select distillation documents and prepare evaluation prompts |

Setting the repository up is the [README](../../README.md), because that is true before any of this.
