"""The core implementation - pure pytorch, correct on CPU, and the specification every backend answers to.

The torch reference of the design is every core design end to end - pools, Ward, buckets, node bucket levels, a
read flat or by a beam - and is the oracle the kernels and the server are tested against. Its public boundary is
`build_buckets`, `read`, `build_hierarchy` and `read_hierarchy`, which raise on a wrong dtype or a row that is not
unit length.
`read_hierarchy_counted` is `read_hierarchy` that also returns the vectors each query scored, the read's cost.

| module | what it does |
|---|---|
| [reference](reference.py) | the flat design: Ward over a cache's closed cluster blocks, then a read that scores, picks and expands |
| [hierarchy](hierarchy.py) | fixed and tree: node bucket levels Ward-paired over the buckets, then a beam down the levels a query's page saw |
| [shape](shape.py) | bucket shapes: a bucket's spread about its centroid, and the estimated best key a shaped bucket scores by |
| [ward](ward.py) | Ward over a cluster block's pools, mutual cheapest pairs under the cap, and the member table |
| [index_space](index_space.py) | the position-free sub-term of the logit, GQA-collapsed, and the learned map onto it |
| [attention](attention.py) | the sparse read, for checking the indexer against dense attention |
| [kernels](kernels/) | the device kernels every backend draws from, triton where a fused launch wins |
"""

from lsa.core.attention import topk_attention
from lsa.core.hierarchy import Hierarchy, build_hierarchy, read_hierarchy, read_hierarchy_counted
from lsa.core.index_space import (
    LatentMetric,
    LearnedProjection,
    RotaryLayout,
    position_free,
    project,
    regroup,
    regroup_latent,
    regroup_partial,
    rownorm,
)
from lsa.core.reference import Buckets, build_buckets, read, shape_buckets
from lsa.core.shape import Shapes

__all__ = [
    "Buckets",
    "Hierarchy",
    "LatentMetric",
    "LearnedProjection",
    "RotaryLayout",
    "Shapes",
    "build_buckets",
    "build_hierarchy",
    "position_free",
    "project",
    "regroup",
    "regroup_latent",
    "regroup_partial",
    "read",
    "read_hierarchy",
    "read_hierarchy_counted",
    "rownorm",
    "shape_buckets",
    "topk_attention",
]
