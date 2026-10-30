"""What LSA needs of one checkpoint alone, one package per model, named by its `model_type`.

The selection both share is in [index](../index/__init__.py), and nothing there imports a model.
A piece of serving logic that names a model lives here.

| package | what it does |
|---|---|
| [glm5_next](glm5_next/__init__.py) | GLM-5.3-Flash - the design's indexer reading `x`, in place of the checkpoint's own |
| [qwen3_5](qwen3_5/__init__.py) | Qwen3.8-27B - a dense model's quadratic layers reading the design's selection, from the attention backend |
"""
