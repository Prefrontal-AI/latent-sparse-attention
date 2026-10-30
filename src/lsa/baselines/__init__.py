"""Sparse selectors other than LSA, in torch - the methods LSA is measured against.

Each module is one method as its authors or its model's server compute it, kept apart from `lsa.core`, which
holds LSA alone.

| module | what it does |
|---|---|
| [qsa](qsa.py) | Qwen's own sparse attention - its indexer's scores over pooled blocks, and its read |
| [pisa](pisa.py) | PISA's block selection - a positional pyramid of pooled key blocks, searched coarse to fine |
"""
