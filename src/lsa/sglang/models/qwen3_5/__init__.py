"""Qwen3.5's architecture under the design - Qwen3.8-27B's quadratic layers reading an LSA selection.

| module | what it does |
|---|---|
| [index_space](index_space.py) | the geometry off the config, and the key and query input off the heads attention is handed |
| [backend](backend.py) | the FlashAttention backend's quadratic layers selecting and reading, and the index maintained before each forward |

`backend` imports sglang, so nothing but the plugin reaches it, and only after the plugin resolved every target.
"""
