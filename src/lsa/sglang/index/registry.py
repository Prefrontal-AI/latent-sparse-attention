"""Every LSA indexer registers here - the attention backend runs their maintenance before each forward."""

from typing import Any

# The live indexers, one per sparse layer
INDEXERS: list[Any] = []

# Request pool row of each request id, so decode finds its index without a device sync
ROWS: dict[Any, int] = {}

# Request ids kept before the oldest half is dropped - four batches' worth over a floor
ROWS_KEPT_PER_BATCH_ROW: int = 4
ROWS_KEPT_FLOOR: int = 4096

# Banners already printed - each is said once per server
ANNOUNCED: set[str] = set()


def register(indexer: Any) -> None:
    """Add one layer's indexer to the set maintained before every forward.

    Args:
        indexer: The layer's indexer, which has a `maintain(forward_batch)`.
    """
    INDEXERS.append(indexer)


def announce_once(banner: str, layer_id: object) -> None:
    """Print a banner the first time any layer earns it, which is what readiness waits for.

    Args:
        banner: One of `Banner`'s lines.
        layer_id: The layer that earned it first.
    """
    if banner in ANNOUNCED:
        return
    ANNOUNCED.add(banner)
    print(f"{banner} layer {layer_id} and every sparse layer after", flush=True)


def rows_of(forward_batch: Any) -> list[int]:
    """Request pool rows of the batch - one device sync per prefill, none in decode

    Every request enters through a prefill, which records its row. Decode looks the rows up by
    request id instead of copying the tensor off the device once per layer and step.

    Args:
        forward_batch: The batch, with `rids`, `forward_mode` and `req_pool_indices`.

    Returns:
        The request pool row of each request, in batch order.
    """
    rids = forward_batch.rids
    if not forward_batch.forward_mode.is_extend() and all(rid in ROWS for rid in rids):
        return [ROWS[rid] for rid in rids]
    rows = forward_batch.req_pool_indices.tolist()
    ROWS.update(zip(rids, rows, strict=False))
    if len(ROWS) > ROWS_KEPT_PER_BATCH_ROW * len(rows) + ROWS_KEPT_FLOOR:
        # Finished requests never come back - keep the map small
        for rid in list(ROWS)[: len(ROWS) // 2]:
            del ROWS[rid]
    return rows


def ends_of(forward_batch: Any, request_count: int) -> list[int]:
    """Keys each real request has in the pool before this forward's own.

    A decode batch replayed in a CUDA graph is padded to the graph's size, so its length lists run past the requests.

    Args:
        forward_batch: The batch, with `seq_lens_cpu`, `forward_mode` and, for a prefill, `extend_seq_lens_cpu`.
        request_count: Real requests in the batch.

    Returns:
        The tokens before the forward of each real request, in batch order.
    """
    seq_lens = forward_batch.seq_lens_cpu.tolist()[:request_count]
    if forward_batch.forward_mode.is_extend():
        lengths = list(forward_batch.extend_seq_lens_cpu)[:request_count]
    else:
        lengths = [1] * request_count
    return [seq_len - length for seq_len, length in zip(seq_lens, lengths, strict=True)]


def maintain(forward_batch: Any) -> None:
    """Put the request pool rows on the batch and run every live indexer's maintenance.

    Args:
        forward_batch: The batch about to be forwarded.

    Raises:
        RuntimeError: The batch carries no request ids.
    """
    if not INDEXERS or forward_batch.forward_mode.is_idle():
        return
    if getattr(forward_batch, "rids", None) is None:
        raise RuntimeError("LSA maintenance needs request ids on the forward batch")
    forward_batch.lsa_rows = rows_of(forward_batch)
    for indexer in INDEXERS:
        indexer.maintain(forward_batch)
