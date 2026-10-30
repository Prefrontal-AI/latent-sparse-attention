"""The CuTeDSL kernels built on an instruction only Blackwell has - `tcgen05` MMA into tensor memory - so they run on Blackwell alone.

| module | what it does |
|---|---|
| [bucket_scan](bucket_scan.py) | the buckets each index query scores at or above its threshold, scored on `tcgen05` MMA |
| [tile](tile.py) | a tile of 64 by 64 tf32x3 merge costs on `tcgen05` MMA, read back as the Hopper rounds read theirs - [the Hopper tile](../hopper/tile.py) with its hooks overridden |
| [node_bucket_clustering](node_bucket_clustering.py) | the Hopper node-bucket clustering's rounds on Blackwell tiles |
| [ward](ward.py) | the Hopper ward clustering's rounds on Blackwell tiles |
"""
