"""What serving shares across models - the per-layer index, what a query reads from it, and what the read becomes.

The bucket cache and the hierarchy over it, the clustering that fills them, the selector that reads them, the
layer index that projects `x` into them, the registry and the learned projections, with each model's own package under
[models](../models/__init__.py). Nothing here imports a family: a model hands the
selector its request table and the index keys of a position, and takes back lists of KV slots.

| module | what it does |
|---|---|
| [bucket_cache](bucket_cache.py) | `BucketCache` - the pages each request holds, their bucket centroids, members and sizes; allocating, releasing and reclaiming them, the row maps and the counts a decode step reads |
| [clustering](clustering.py) | Ward over a cluster block's pools, and the buckets it leaves - one call for every block a chunk closes |
| [hierarchy](hierarchy.py) | `Hierarchy` - the node bucket levels over a request's closed buckets, grown in full blocks, and the beam search down them |
| [selector](selector.py) | `Selector` - the read of every design: flat, fixed and tree prefill, flat and descending decode |
| [layer_index](layer_index.py) | `LayerIndex` - one layer's projections of `x`, its index-key buffer and its selector: what a forward batch hands in and gets lists back from |
| [consistency](consistency.py) | whether every rank of a decode context-parallel group holds the same buckets and tree - a checksum summed across the group, checked when a server asks |
| [read](read.py) | what a selection becomes for a paged read - the page table, and the exact count per row |
| [registry](registry.py) | every live indexer, so the attention backend can bring them all up to date, and the banners each says once |
| [weights](weights.py) | the learned projections of `x` a server reads, refused unless they are the model's |
| [probe](probe.py) | the timing ranges the cache and the selector open, a no-op unless a benchmark records them |
"""
