"""The CuTeDSL kernels - where a kernel stages shared memory itself, which Triton does not expose.

The kernels here use no instruction only one GPU has; [hopper](hopper/__init__.py) and
[blackwell](blackwell/__init__.py) hold those that do, and [bucket_scan](bucket_scan.py), [node_bucket_clustering](node_bucket_clustering.py) and
[ward](ward.py) pick between the two by the GPU at hand. [dispatch](../dispatch.py) names the GPUs each one runs on.

| module | what it does |
|---|---|
| [thresholds](thresholds.py) | flat's sampled thresholds, bit for bit the Triton ones |
| [scoring](scoring.py) | the listed buckets, or every bucket a row sees, scored and packed into keys, bit for bit the Triton keys |
| [expansion](expansion.py) | flat's expansion by threshold and the expansion in rank order of picked buckets - a row's buckets read as their members |
| [decode](decode.py) | a flat decode step's buckets scored through their pages and its best listed, the scan and the select in one pass |
| [top_k](top_k.py) | a short prefill's listing - each row's sampled threshold and the columns at or above it in one streaming pass |
| [beam](beam.py) | a tree's levels - the carried children scored, the best kept, their children or the picked buckets' keys named |
| [bucket_scan](bucket_scan.py) | the bucket scan by the kernel built for the GPU |
| [node_bucket_clustering](node_bucket_clustering.py) | a growth's node-bucket clustering by the kernel built for the GPU |
| [ward](ward.py) | a cluster block's ward clustering by the kernel built for the GPU |
"""
