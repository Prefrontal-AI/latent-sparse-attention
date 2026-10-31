"""LSA as an sglang plugin - the hooks that attach the kernels to a stock server, and the launch around it.

This is the serving backend. `lsa.core` says what LSA selects and is correct on a CPU; this is the same
selection at a scale the reference path cannot reach, and the two are read against each other.

The kernels themselves are not here.
They are in [core/kernels](../core/kernels/), because the training path wants the same ones.

[plugin](plugin.py) registers from outside sglang what a fork once edited into it, so nothing in sglang is
modified.

| module | what it does |
|---|---|
| [plugin](plugin.py) | the entry point sglang calls, and the hooks that stand in for the fork's edits |
| [constants](constants.py) | what a reader outside the process matches on - the banners above all - and where a server runs |
| [launch](launch.py) | the container, the server's command line, and the wait until it says LSA is armed |
| [settings](settings/__init__.py) | the block of the model config a server runs under - the design's `DesignSettings`, its indexer knobs and serving choices under their config keys, the `debug` checks - and which combinations the kernels can serve |
| [index](index/__init__.py) | the selection every model shares - the bucket cache, the clustering, the hierarchy, the selector, the paged read, the top-k, the live indexers |
| [models](models/__init__.py) | one package per model, named by its `model_type` - what LSA needs of that checkpoint alone |

Two halves, and they run in different places.
[launch](launch.py) runs on the job host, before any container starts, and imports no sglang - an evaluation
drives it, and `import lsa` never reaches it. [settings.design](settings/design.py) is read on both sides,
so it imports no sglang either - the job host writes the settings block and the server reads it back.
Everything else runs inside the server's container, against the sglang that container ships, and needs a GPU
- so the package is installed there rather than sglang resolved into this repository's lock.
"""
