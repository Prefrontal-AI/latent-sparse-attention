"""Save everything one measured forward pass mutates, and put it back.

A timing pass runs the same module over the same batch many times, so the KV cache, the index buffers and the
design's bucket cache all have to look untouched at the start of each one. `restore` does that and `check` proves
it, which is what makes repeated passes comparable.

`compare` and the output helpers live here because they are what a restored pass is judged by.
"""

import copy
import hashlib
from typing import Any

import torch

from lsa.sglang.index.graph_tree import NODE_BUCKET_PAGE


def clone_output(value):
    """Return a detached copy of a tensor, or of every tensor in a tuple."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_output(item) for item in value)
    return value


def compare(actual, expected):
    """Hold an output bitwise to the expected one, tuples element by element.

    Returns:
        The shape, dtype and exactness of each tensor compared.

    Raises:
        AssertionError: Any value differs.
    """
    if isinstance(expected, tuple):
        return [compare(a, b) for a, b in zip(actual, expected, strict=True)]
    if expected is None:
        if actual is not None:
            raise AssertionError("Expected no tensor and got one")
        return None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"shape": list(actual.shape), "dtype": str(actual.dtype), "exact": True}


def tensor_digest(tensor):
    """Return the sha256 of a tensor's bytes."""
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def output_error(actual, expected):
    """Return the max absolute and relative l2 error of an output against the expected one, a tuple's first element."""
    if isinstance(expected, tuple):
        actual, expected = actual[0], expected[0]
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.float().norm()).item(),
    }


def save_checkpoint(name, destination):
    """Return a forward hook that stores a clone of the module's output as `destination[name]`."""

    def hook(_module, positional, output) -> None:
        destination[name] = clone_output(output)

    return hook


def linear_layer_id(module):
    """Return the layer index of a linear attention module."""
    # KDA names its layer `layer_idx`, Gated DeltaNet `layer_id`
    return module.layer_idx if hasattr(module, "layer_idx") else module.layer_id


def kv_buffers(pool, layer_id):
    """The layer's cache buffers - one latent for MLA, keys and values for GQA"""
    key = pool.get_key_buffer(layer_id)
    getter = getattr(pool, "get_value_buffer", None)
    value = getter(layer_id) if getter is not None else key
    return [key] if value.data_ptr() == key.data_ptr() else [key, value]


def index_buffer(pool, layer_id):
    """The native indexer's key buffer - QSA's compressed keys, DSA's pooled keys - or None for a dense model's pool"""
    qsa_keys = getattr(pool, "get_qsa_compressed_k_buffer", None)
    # A graph step that closes no block writes its group to compressed slot 0, the padding slot no request owns
    # => scratch, so only the slots after it are state
    if qsa_keys is not None:
        return qsa_keys(layer_id)[1:]
    getter = getattr(pool, "get_index_k_with_scale_buffer", None)
    # A dense model's pool still has the getter, and it asserts rather than returning nothing
    if getter is None or not getattr(pool, "use_dsa", True):
        return None
    try:
        return getter(layer_id)
    except (AttributeError, NotImplementedError):
        return None


class LinearSnapshot:
    """The linear layer's state for the request - its conv window and its recurrent state.

    Nothing here grows with the context, which is the whole point of the layer: one conv window and one
    recurrent state per request, restored by row rather than by cache location.
    """

    # The linear layer has no index of its own
    state = None

    def __init__(self, module, req_pool, batch) -> None:
        """Save the batch's conv window and recurrent state.

        Args:
            module: The linear attention layer.
            req_pool: The request pool holding its state.
            batch: The forward batch.
        """
        cache = req_pool.mamba2_layer_cache(linear_layer_id(module))
        self.row = req_pool.get_mamba_indices(batch.req_pool_indices).long()
        self.tensors = [(tensor, tensor[self.row].clone()) for tensor in (cache.conv[0], cache.temporal)]

    def restore(self) -> None:
        """Put the saved state back."""
        for tensor, saved in self.tensors:
            tensor[self.row] = saved

    def check(self) -> None:
        """Raise unless the state is exactly what was saved."""
        for tensor, saved in self.tensors:
            compare(tensor[self.row], saved)

    @property
    def nbytes(self):
        """Bytes the saved state takes."""
        return sum(saved.nbytes for _, saved in self.tensors)


def same_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Tell whether two tensors hold the same bytes - fp8 has no equality kernel on every build, and nan is not equal to itself."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def same_hierarchy(left: Any, right: Any) -> bool:
    """Tell whether two hierarchies hold the same node buckets, level by level, compared by their bytes."""
    if left.covered_buckets != right.covered_buckets:
        return False
    for first, second in (
        (left.open_node_buckets, right.open_node_buckets),
        (left.closed_node_buckets, right.closed_node_buckets),
    ):
        if len(first) != len(second):
            return False
        for rows, other in zip(first, second, strict=True):
            if len(rows) != len(other) or rows.parts.keys() != other.parts.keys():
                return False
            if not all(same_bytes(rows[name], other[name]) for name in rows.parts):
                return False
    return True


def same_field(name: str, left: Any, right: Any) -> bool:
    """Tell whether a host field of the bucket cache is what it was - the hierarchies compared by their bytes."""
    if name != "hierarchies" or left is None or right is None:
        return bool(left == right)
    return len(left) == len(right) and all(same_hierarchy(a, b) for a, b in zip(left, right, strict=True))


class GraphTreeState:
    """The graph tree's mirror of the node buckets - the tensors a captured descend reads and the host state behind them.

    A step that grows a hierarchy takes pages, writes node buckets into them and moves the counts. Restored in place,
    since a captured graph holds the addresses:

    - the counts, start levels, covered buckets and page tables, whole
    - the node centroids and children of every page a request holds, since a reset can free a page and a later sync
      write another request's node buckets into it
    - the free lists, each request's held pages and its copied counts, by value
    """

    TENSORS = ("made", "closed", "start", "covered")

    def __init__(self, tree: Any) -> None:
        """Save the mirror.

        Args:
            tree: The cache's `GraphTree`.
        """
        self.tree = tree
        self.tensors = [
            (t, t.clone()) for t in (*(getattr(tree, name) for name in self.TENSORS), *tree.pages)
        ]
        # The rows of each level's pools that the held pages cover
        self.pools = []
        for level in range(tree.level_count):
            held = sorted({page for row in tree.held_pages for page in row[level]})
            pages = torch.tensor(held, dtype=torch.int64, device=tree.device)
            offsets = torch.arange(NODE_BUCKET_PAGE, device=tree.device)
            rows = (pages[:, None] * NODE_BUCKET_PAGE + offsets).flatten()
            centroids, children = tree.node_centroids[level], tree.children[level]
            self.pools.append((rows, centroids, centroids[rows].clone(), children, children[rows].clone()))
        self.free = copy.deepcopy(tree.free)
        self.held_pages = copy.deepcopy(tree.held_pages)
        self.copied = copy.deepcopy(tree.copied)

    def restore(self) -> None:
        """Put the tensors and the host state back."""
        for tensor, saved in self.tensors:
            tensor.copy_(saved)
        for rows, centroids, saved_centroids, children, saved_children in self.pools:
            centroids[rows] = saved_centroids
            children[rows] = saved_children
        self.tree.free = copy.deepcopy(self.free)
        self.tree.held_pages = copy.deepcopy(self.held_pages)
        self.tree.copied = copy.deepcopy(self.copied)

    def check(self, hierarchies: list[Any] | None) -> None:
        """Raise unless the mirror is what was saved, and holds as many node buckets as the hierarchies do.

        Args:
            hierarchies: The cache's hierarchies, or None where it keeps none.

        Raises:
            AssertionError: A tensor or a host field differs, or a level's copied count is not its hierarchy's.
        """
        for tensor, saved in self.tensors:
            compare(tensor, saved)
        for rows, centroids, saved_centroids, children, saved_children in self.pools:
            compare(centroids[rows], saved_centroids)
            compare(children[rows], saved_children)
        tree = self.tree
        if tree.free != self.free or tree.held_pages != self.held_pages or tree.copied != self.copied:
            raise AssertionError("Graph tree host state was not restored")
        for row, hierarchy in enumerate(hierarchies or []):
            made = [
                len(c) + len(o)
                for c, o in zip(hierarchy.closed_node_buckets, hierarchy.open_node_buckets, strict=True)
            ]
            if tree.copied[row][: len(made)] != made or any(tree.copied[row][len(made) :]):
                raise AssertionError(
                    f"Graph tree row {row} holds {tree.copied[row]} node buckets, hierarchy {made}"
                )

    @property
    def nbytes(self) -> int:
        """Bytes the saved state takes."""
        pools = sum(c.nbytes + k.nbytes for _, _, c, _, k in self.pools)
        return pools + sum(saved.nbytes for _, saved in self.tensors)


class IndexState:
    """The design's index of one layer - its index keys and bucket cache - saved and put back.

    A pass over the module advances the bucket cache, which clusters closed blocks into buckets, takes pages
    and grows the hierarchy. All of it is restored, and a restore is exact:

    - the cache's tensors - centroids, members, member counts (`bucket_sizes`), block positions
      (`page_positions`), the request table of pages and the closed key count the CUDA graph reads - are copied
      back in place, since a captured graph holds their addresses
    - the host fields - the free pages, each request's pages, its run of consecutive pages, its end and closed
      key count, its owner and the hierarchy it grew - are copied back by value
    - the index keys the pass wrote are put back at their slots
    - the graph tree's mirror of the node buckets, when the cache keeps one, is put back by `GraphTreeState`
    """

    # The cache's tensors the reads and the CUDA graph address
    TENSORS = (
        "bucket_centroids",
        "bucket_members",
        "bucket_sizes",
        "page_positions",
        "request_pages",
        "closed_keys_device",
    )
    # The cache's host fields, copied by value. `run` is each request's leading pages that are consecutive,
    # which a lone request's centroids are read as a view of
    HOST = ("free", "pages", "run", "end", "closed_keys", "rid", "seen", "hierarchies")

    def __init__(self, index: Any, locations: torch.Tensor) -> None:
        """Save the index's state.

        Args:
            index: The layer's `LayerIndex`.
            locations: The KV slots the pass writes index keys at [S] int64.
        """
        self.index = index
        self.selector = index.selector
        self.cache = index.selector.cache
        self.locations = locations
        # The index keys the layer holds itself - none under the key and query, whose keys are the KV pool's and
        # are saved with it
        self.own_keys = index.index_keys.numel() > 0
        self.keys = index.index_keys[locations].clone() if self.own_keys else index.index_keys
        self.tensors = [
            (getattr(self.cache, name), getattr(self.cache, name).clone()) for name in self.TENSORS
        ]
        self.host = {name: copy.deepcopy(getattr(self.cache, name)) for name in self.HOST}
        self.step = self.selector.step
        tree = self.cache.graph_tree
        self.tree = GraphTreeState(tree) if tree is not None else None

    def restore(self) -> None:
        """Put every saved tensor and field back."""
        if self.own_keys:
            self.index.index_keys[self.locations] = self.keys
        for tensor, saved in self.tensors:
            tensor.copy_(saved)
        for name, saved in self.host.items():
            setattr(self.cache, name, copy.deepcopy(saved))
        self.selector.step = self.step
        if self.tree is not None:
            self.tree.restore()

    def check(self) -> None:
        """Raise unless the state is exactly what was saved.

        Raises:
            AssertionError: A tensor or a field differs.
        """
        if self.own_keys:
            compare(self.index.index_keys[self.locations], self.keys)
        for tensor, saved in self.tensors:
            compare(tensor, saved)
        for name, saved in self.host.items():
            if not same_field(name, getattr(self.cache, name), saved):
                raise AssertionError(f"Bucket cache field {name} was not restored")
        if self.selector.step != self.step:
            raise AssertionError("Selector step was not restored")
        if self.tree is not None:
            self.tree.check(self.cache.hierarchies)

    def changed(self, row: int) -> dict[str, bool]:
        """Say what a pass did to one request's cache, against the state saved.

        Args:
            row: The request table row.

        Returns:
            Whether the pass took a page, closed a cluster block, and grew the hierarchy.
        """
        cache, saved = self.cache, self.host
        hierarchies = cache.hierarchies
        return {
            "page_allocation": cache.pages[row] != saved["pages"][row],
            "cluster_block_close": cache.closed_keys[row] != saved["closed_keys"][row],
            "hierarchy_growth": hierarchies is not None
            and not same_hierarchy(hierarchies[row], saved["hierarchies"][row]),
        }

    @property
    def nbytes(self) -> int:
        """Bytes the saved state takes."""
        return (
            self.keys.nbytes
            + sum(saved.nbytes for _, saved in self.tensors)
            + (self.tree.nbytes if self.tree is not None else 0)
        )


def pisa_indexes(indexer: Any) -> list[Any]:
    """Return the `PisaIndex` a layer's indexer holds, GLM's one or Qwen's one a key head.

    Args:
        indexer: The layer's PISA indexer.

    Returns:
        The indexes, in key head order.
    """
    indexes = indexer.ready()
    return indexes if isinstance(indexes, list) else [indexes]


class PisaState:
    """PISA's index of one layer - the means of the closed leaf blocks of the batch's request rows and the pyramid above
    them - saved and put back.

    A pass closes the blocks its keys complete, which moves a row's closed count and writes its means and the
    summaries above. The keys themselves are the KV pool's, saved with it. Both are copied back in place, since a
    captured decode graph holds their address. Qwen holds one index a key head, and each is saved.
    """

    def __init__(self, indexes: list[Any], rows: list[int]) -> None:
        """Save the rows' means, their pyramid and what each row holds.

        Args:
            indexes: The layer's `PisaIndex`, one a key head.
            rows: The request table rows of the batch.
        """
        self.leaf_means = [index.leaf_means for index in indexes]
        self.rows = rows
        self.means = [leaf.means[rows].clone() for leaf in self.leaf_means]
        self.upper = [leaf.upper[rows].clone() for leaf in self.leaf_means]
        self.closed = [list(leaf.closed) for leaf in self.leaf_means]
        self.rids = [list(leaf.rids) for leaf in self.leaf_means]

    def restore(self) -> None:
        """Put the means, the pyramid and the host fields back."""
        for at, leaf in enumerate(self.leaf_means):
            leaf.means[self.rows] = self.means[at]
            leaf.upper[self.rows] = self.upper[at]
            leaf.closed = list(self.closed[at])
            leaf.rids = list(self.rids[at])

    def check(self) -> None:
        """Raise unless the state is exactly what was saved.

        Raises:
            AssertionError: A mean, a summary or a host field differs.
        """
        for at, leaf in enumerate(self.leaf_means):
            compare(leaf.means[self.rows], self.means[at])
            compare(leaf.upper[self.rows], self.upper[at])
            if leaf.closed != self.closed[at] or leaf.rids != self.rids[at]:
                raise AssertionError("Leaf means host fields were not restored")

    def changed(self, row: int) -> dict[str, bool]:
        """Say whether a pass closed a leaf block of one request, against the state saved.

        Args:
            row: The request table row.

        Returns:
            Whether the pass closed a block in any index, which is the one maintenance PISA has.
        """
        return {
            "pool_close": any(
                leaf.closed[row] != self.closed[at][row] for at, leaf in enumerate(self.leaf_means)
            )
        }

    @property
    def nbytes(self) -> int:
        """Bytes the saved state takes."""
        return sum(m.nbytes for m in self.means) + sum(u.nbytes for u in self.upper)


class Snapshot:
    """Everything one forward pass of a layer mutates - its KV rows, its index and its tails - saved and put back."""

    def __init__(self, module, pool, batch) -> None:
        """Save the layer's state for the batch.

        Args:
            module: The layer under test.
            pool: The KV pool.
            batch: The forward batch, whose cache locations the pass writes.
        """
        self.module = module
        self.pool = pool
        self.locations = batch.out_cache_loc.long()
        self.buffers = kv_buffers(pool, module.layer_id)
        self.rows = [buffer[self.locations].clone() for buffer in self.buffers]
        # The design's indexer writes its keys to a buffer of its own, and the pool holds none of GLM's
        design = getattr(module.indexer, "is_lsa", False)
        # PISA's keys are the KV pool's latents, so it holds only the means of its closed blocks
        pisa = getattr(module.indexer, "is_pisa", False)
        self.state: IndexState | PisaState | None = None
        if design:
            self.state = IndexState(module.indexer.ready(), self.locations)
        if pisa:
            self.state = PisaState(pisa_indexes(module.indexer), batch.req_pool_indices.tolist())
        self.index = None if design or pisa else index_buffer(pool, module.layer_id)
        self.index_copy = self.index.clone() if self.index is not None else None
        full_pool = getattr(pool, "full_kv_pool", pool)
        self.tails = []
        for name in ("_compress_tail_k", "_compress_tail_score"):
            for tensor in getattr(full_pool, name, None) or []:
                self.tails.append((tensor, tensor.clone()))
        # QSA's keys of each request's open block wait in a ring, with their rotary positions, until the block closes.
        # Its first `compress_ratio` rows are no request's - a prefill dumps every key before its open block there
        # => scratch whose last writer is a race, so only the requests' rows are state
        if hasattr(pool, "get_qsa_key_state_buffer"):
            owned = slice(pool.qsa_compress_ratio, None)
            for tensor in (pool.get_qsa_key_state_buffer(module.layer_id), pool.qsa_rope_position_buffer):
                self.tails.append((tensor[owned], tensor[owned].clone()))

    def restore(self) -> None:
        """Put every saved buffer and the index state back."""
        for buffer, rows in zip(self.buffers, self.rows, strict=True):
            buffer[self.locations] = rows
        if self.index is not None:
            self.index.copy_(self.index_copy)
        for tensor, saved in self.tails:
            tensor.copy_(saved)
        if self.state is not None:
            self.state.restore()

    def check(self) -> None:
        """Raise unless every buffer and the index state are exactly what was saved."""
        for buffer, rows in zip(self.buffers, self.rows, strict=True):
            compare(buffer[self.locations], rows)
        if self.index is not None:
            compare(self.index, self.index_copy)
        for tensor, saved in self.tails:
            compare(tensor, saved)
        if self.state is not None:
            self.state.check()

    @property
    def nbytes(self):
        """Bytes the saved state takes."""
        return (
            self.locations.nbytes
            + sum(rows.nbytes for rows in self.rows)
            + (self.index_copy.nbytes if self.index_copy is not None else 0)
            + sum(saved.nbytes for _, saved in self.tails)
            + (self.state.nbytes if self.state is not None else 0)
        )
