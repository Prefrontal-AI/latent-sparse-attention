"""The distillation stage - the learned index space: the targets, the trainer, and the yaml that says what a run is.

The model is frozen. What is learned is one linear map per indexer-carrying layer and nothing else, and what
comes out of it is what [inject](inject.py) writes into a checkpoint.

| module | what it does |
|---|---|
| [data](data/__init__.py) | the teacher's attention targets - extracted from a model, and read back by a run |
| [config](config.py) | the run schema, the yaml loader and the sweep |
| [constants](constants.py) | what the targets, a run and its reports have to agree on |
| [model](model.py) | the learned maps themselves |
| [losses](losses.py) | the per-key divergence, and the temperature it is read at |
| [metrics](metrics.py) | what a read captures of the teacher's own attention |
| [reporting](reporting.py) | a report - the exact read and the served read of the held-out documents |
| [run](run.py) | the trainer, and `lsa distill train`, which submits it one slurm job per run |
| [shard](shard.py) | one run trained a few layers per job, and joined |
| [rescore_maps](rescore_maps.py) | trained maps scored again at a read they were not trained for - `lsa distill rescore` |
| [served_read](served_read.py) | an exported map's served read per design, layer and length - `lsa distill served-read` |
| [tracking](tracking.py) | the second copy of the numbers, which the sweeps are read from |
| [export](export.py) | the maps as the flat per-layer file the serving plugin reads |
| [inject](inject.py) | a checkpoint that carries its own index space |

What a written map is, and what a targets extraction records about itself, are read outside this package too, so
they live above this package in [distilled_maps](../distilled_maps.py) and [targets](../data/targets.py).

"""
