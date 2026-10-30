"""The CuTeDSL kernels built on an instruction only Hopper has, so they run on Hopper alone.

The pairing's and the Ward merge's rounds use no such instruction - only their tiles of merge costs do.
They inherit those from [tile](tile.py), so
[blackwell](../blackwell/__init__.py) runs the same rounds with its own tiles.

| module | what it does |
|---|---|
| [bucket_scan](bucket_scan.py) | the buckets each index query scores at or above its threshold, scored on warpgroup MMA |
| [tile](tile.py) | a tile of 64 by 64 tf32x3 merge costs on warpgroup MMA, and the hooks another GPU's MMA replaces it by |
| [node_bucket_clustering](node_bucket_clustering.py) | a hierarchy growth's pairing of buckets by the Ward cost, every round in one launch |
| [ward](ward.py) | a cluster block's Ward merge of its pools into buckets, every round in its CTAs' shared memory - a cluster of them for a wide block |
"""
