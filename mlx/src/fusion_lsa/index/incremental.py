import logging
from dataclasses import dataclass, field

import mlx.core as mx

from fusion_lsa.core.reference import Buckets, build_buckets

logger = logging.getLogger(__name__)


@dataclass
class IncrementalBucketIndex:
    # Page-boundary contract: buckets are built only over whole cluster blocks, one page of
    # keys at a time. A prefill that stops mid page accumulates keys outside this index and
    # feeds them here once the page closes - mid-page ward merges are refused, not attempted.
    pool_size: int
    bucket_size: int
    bucket_cap: int
    cluster_block_size: int
    _blocks: list[Buckets] = field(default_factory=list, init=False, repr=False)
    _key_count: int = field(default=0, init=False, repr=False)
    _cached_buckets: Buckets | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bucket_size > self.cluster_block_size or self.cluster_block_size % self.bucket_size:
            raise ValueError(
                f"bucket_size {self.bucket_size} must divide cluster_block_size {self.cluster_block_size}"
            )

    def append(self, keys: mx.array) -> None:
        # One closed page of keys [cluster_block_size, index_dim]; anything else is refused
        if keys.ndim != 2 or keys.shape[0] != self.cluster_block_size:
            raise ValueError(
                f"append takes exactly one page of {self.cluster_block_size} keys - got shape {keys.shape}"
            )
        block = build_buckets(keys, self.pool_size, self.bucket_size, self.bucket_cap, self.cluster_block_size)
        # build_buckets over one page numbers keys locally; shift them to positions in the whole sequence
        offset = len(self._blocks) * self.cluster_block_size
        members = block.bucket_members
        block = block._replace(
            bucket_members=mx.where(members >= 0, members + offset, members).astype(mx.int32)
        )
        self._blocks.append(block)
        self._key_count += self.cluster_block_size
        self._cached_buckets = None
        logger.debug(
            "incremental index: block %d appended, %d blocks, %d keys",
            len(self._blocks),
            len(self._blocks),
            self._key_count,
        )

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def key_count(self) -> int:
        return self._key_count

    def buckets(self) -> Buckets:
        # The concatenated buckets of every closed block; block i's rows are immutable once built.
        # Concat is cached: called every decode step, and blocks only grow on page close.
        if not self._blocks:
            raise ValueError("the index holds no closed blocks")
        if self._cached_buckets is None:
            self._cached_buckets = Buckets(
                mx.concatenate([block.bucket_centroids for block in self._blocks], axis=0),
                mx.concatenate([block.bucket_members for block in self._blocks], axis=0),
                mx.concatenate([block.bucket_sizes for block in self._blocks], axis=0),
                self._blocks[0].block_bucket_count,
                self.cluster_block_size,
            )
        return self._cached_buckets
