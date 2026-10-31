"""GLM-5.3-Flash under the design - index keys and index queries read off `x`, in place of the checkpoint's own indexer.

| module | what it does |
|---|---|
| [indexer](indexer.py) | the checkpoint's indexer class replaced by one that hands `x` to a `LayerIndex`, and the layers that run one |

It imports sglang, so nothing but the plugin reaches it, and only after the plugin resolved every target.
"""
