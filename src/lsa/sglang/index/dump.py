"""A debugging dump of what a served layer reads - its `x`, positions and lists - to hold against an offline read.

Off unless the settings block's `debug.dump.directory` names a directory. Then tensor-parallel rank 0 writes, for the
layers `debug.dump.layers` lists, one torch file per prefill chunk and per decode step of a request:

    <dir>/layer<L>/<rid>/prefill_<first position>.pt   x, positions, lists at sampled rows of the chunk
    <dir>/layer<L>/<rid>/write_<first position>.pt     x at sampled rows of a chunk the dense path prefilled
    <dir>/layer<L>/<rid>/decode_<position>.pt          x, position, list of one decode step

A list is written as key positions, -1 padded, the request table read backwards, so it compares with
`read_sampled` directly - or as it is, for Qwen's own sparse-attention indexer, whose selection is key positions
already. A prefill keeps every row at a multiple of `debug.dump.stride` and the chunk's last
`TAIL` rows. A decode step runs inside the CUDA graph, so the step copies its rows into buffers the graph owns and the
host writes them out before the next forward, once that step is done.
"""

from pathlib import Path
from typing import Any

import torch

from lsa.sglang.settings.debug import DumpSettings

# Rows a prefill chunk keeps from its end - the prompt's last chunk holds the question
TAIL: int = 64
# Decode steps a request keeps, its first ones
DECODE_STEPS: int = 64
# Most rows a decode batch has, the largest CUDA graph sglang captures
MOST_DECODE_ROWS: int = 512


def first_rank() -> bool:
    """Return whether this process is tensor-parallel rank 0, the one that writes."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def slot_positions(lists: torch.Tensor, table_row: torch.Tensor, length: int) -> torch.Tensor:
    """Return lists of KV slots as the key positions they hold, -1 kept and an unknown slot -2.

    Args:
        lists: KV slots [S, width] int32, -1 padded.
        table_row: One request's positions to KV slots [context] int32.
        length: Positions the request holds.

    Returns:
        Key positions [S, width] int32.
    """
    slots = table_row[:length].long()
    inverse = torch.full((int(slots.max()) + 2,), -2, device=lists.device, dtype=torch.int32)
    inverse[slots] = torch.arange(length, device=lists.device, dtype=torch.int32)
    valid = lists >= 0
    found = inverse[lists.long().clamp(0, len(inverse) - 1)]
    found = torch.where(lists.long() < len(inverse), found, -2)
    return torch.where(valid, found, -1)


class Dump:
    """One layer's dump."""

    def __init__(
        self,
        root: Path,
        layer: int,
        stride: int,
        lists: bool,
        model_dim: int,
        width: int,
        device: torch.device | str,
    ) -> None:
        """Start a layer's dump, with the decode buffers allocated outside any graph capture.

        Args:
            root: The dump directory.
            layer: The layer.
            stride: A prefill row is kept at a multiple of this position.
            lists: Whether a prefill writes its lists.
            model_dim: Dims of `x`.
            width: Entries of a list.
            device: Where the decode buffers live.
        """
        self.root = root / f"layer{layer}"
        self.stride = stride
        self.write_lists = lists
        self.steps: dict[object, int] = {}
        self.pending: list[tuple[object, int, int]] | None = None
        self.x = torch.zeros(MOST_DECODE_ROWS, model_dim, device=device, dtype=torch.bfloat16)
        self.positions = torch.zeros(MOST_DECODE_ROWS, device=device, dtype=torch.int32)
        self.lists = torch.zeros(MOST_DECODE_ROWS, width, device=device, dtype=torch.int32)

    @classmethod
    def of(
        cls, dump: DumpSettings, layer: int, model_dim: int, width: int, device: torch.device | str
    ) -> "Dump | None":
        """Return the layer's dump, or None when the dump is off, not on this layer or not on this rank.

        Args:
            dump: The settings block's `debug.dump`.
            layer: The layer.
            model_dim: Dims of `x`.
            width: Entries of a list.
            device: Where the decode buffers live.
        """
        if dump.directory is None or layer not in dump.layers or not (dump.all_ranks or first_rank()):
            return None
        root = Path(dump.directory)
        if dump.all_ranks and torch.distributed.is_initialized():
            root = root / f"rank{torch.distributed.get_rank()}"
        return cls(root, layer, dump.stride, dump.lists, model_dim, width, device)

    def save(self, rid: object, name: str, blob: dict[str, Any]) -> None:
        """Write one file under the request's directory."""
        where = self.root / str(rid)
        where.mkdir(parents=True, exist_ok=True)
        torch.save(blob, where / f"{name}.pt")

    def kept_rows(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the rows of a chunk to keep - at a multiple of the stride, or among its last rows."""
        rows = torch.arange(len(positions), device=positions.device)
        return ((positions % self.stride == 0) | (rows >= len(positions) - TAIL)).nonzero().squeeze(1)

    def prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        lists: torch.Tensor | None,
        forward_batch: Any,
        table: torch.Tensor | None,
    ) -> None:
        """Write the sampled rows of a prefill batch, request by request.

        Args:
            x: The batch's attention input [S, model_dim] bf16.
            positions: Each row's position [S].
            lists: KV slots [S, width] int32, or key positions when `table` is None, or None for a chunk the dense
                path prefilled.
            forward_batch: The batch.
            table: The request table, which KV slots need, or None for lists of key positions.
        """
        rows = forward_batch.req_pool_indices.tolist()
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        start = 0
        for rid, row, seq_len, count in zip(
            forward_batch.rids, rows, seq_lens, forward_batch.extend_seq_lens_cpu, strict=False
        ):
            segment = slice(start, start + count)
            start += count
            kept = self.kept_rows(positions[segment])
            blob = {
                "rid": rid,
                "row": row,
                "seq_len": seq_len,
                "positions": positions[segment][kept].cpu(),
                "x": x[segment][kept].cpu(),
            }
            if lists is not None and self.write_lists:
                kept_lists = lists[segment][kept]
                blob["lists"] = (
                    kept_lists if table is None else slot_positions(kept_lists, table[row], seq_len)
                ).cpu()
            first = int(positions[segment][0])
            self.save(rid, f"{'prefill' if lists is not None else 'write'}_{first}", blob)

    def decode(self, x: torch.Tensor, positions: torch.Tensor, lists: torch.Tensor) -> None:
        """Copy a decode step's rows into the buffers the host writes out - captured with the step.

        Args:
            x: The attention input [B, model_dim] bf16.
            positions: Each request's position [B] int32.
            lists: KV slots [B, width] int32.
        """
        count = x.shape[0]
        self.x[:count].copy_(x)
        self.positions[:count].copy_(positions)
        self.lists[:count].copy_(lists)

    def before_forward(self, forward_batch: Any, rows: list[int], table: torch.Tensor) -> None:
        """Write out the decode step before this forward, then note this one if it is a decode step.

        Args:
            forward_batch: The batch about to be forwarded.
            rows: Its request table rows.
            table: The request table.
        """
        if self.pending is not None:
            if self.x.is_cuda:
                torch.cuda.synchronize(self.x.device)
            for index, (rid, row, seq_len) in enumerate(self.pending):
                if rid is None:
                    continue
                position = int(self.positions[index])
                self.save(
                    rid,
                    f"decode_{position}",
                    {
                        "rid": rid,
                        "row": row,
                        "seq_len": seq_len,
                        "positions": self.positions[index : index + 1].cpu(),
                        "x": self.x[index : index + 1].cpu(),
                        "lists": slot_positions(self.lists[index : index + 1], table[row], seq_len).cpu(),
                    },
                )
        self.pending = None
        if not forward_batch.forward_mode.is_decode():
            return
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        pending: list[tuple[object, int, int]] = []
        for rid, row, seq_len in zip(forward_batch.rids, rows, seq_lens, strict=False):
            step = self.steps.get(rid, 0)
            self.steps[rid] = step + 1
            # A request past its steps keeps its place in the batch and is not written
            pending.append((rid if step < DECODE_STEPS else None, row, seq_len))
        self.pending = pending
