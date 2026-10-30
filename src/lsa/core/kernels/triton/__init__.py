"""The Triton kernels, and the host code that orders every kernel of a read.

| module | what it does |
|---|---|
| [block_scores](block_scores.py) | PISA's block scores of a level, reading leaf keys where they sit in the pool |
| [descend](descend.py) | one level of PISA's descent - mask, forced blocks, keep and children in one launch |
| [pyramid](pyramid.py) | PISA's summaries above the leaves, pooled again over the leaves that close |
| [keys](keys.py) | a score packed with its id in one int64, and the one ordered-float conversion |
| [ward](ward.py) | the Ward merge of a cluster block's pools, the pairing of buckets for the hierarchy, the member table |
| [scan_select](scan_select.py) | the top-p of a prefill chunk's bucket scores without the score matrix |
| [top_k](top_k.py) | the exact top-k of a score matrix, best first and the same on every run |
| [expansion](expansion.py) | the picked buckets read as keys, by walk or by threshold, with the window and the sink |
| [paged](paged.py) | the bucket scan over a paged cache, for decode |
| [beam](beam.py) | the children of the node buckets a beam keeps, and their scores |
| [shape](shape.py) | a bucket's spread about its centroid, and the estimated best key it scores by |
"""
