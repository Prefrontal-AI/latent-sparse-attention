"""Towards ultra-long context attention.

LSA clusters keys into fixed-capacity buckets, ranks bucket centroids instead of keys, and emits nothing but
a list of key indices - so the attention kernel underneath stays untouched.

`lsa` itself needs only torch.
`lsa.core` is the pure-pytorch implementation and the specification.
`lsa.evaluation` is the accuracy harness and `lsa.training` is both training stages, each behind an extra.
`lsa.sglang` is the serving backend, a plugin sglang loads through its own entry point.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
