import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

QUANT_BITS = (2, 4, 8)


def hadamard_matrix(dim: int) -> mx.array:
    # Normalized Sylvester Hadamard, dim a power of two - the rotation that spreads outlier channels
    if dim < 1 or dim & (dim - 1):
        raise ValueError(f"hadamard needs a power-of-two dim - got {dim}")
    h = mx.ones((1, 1), dtype=mx.float32)
    while h.shape[0] < dim:
        h = mx.concatenate(
            [mx.concatenate([h, h], axis=1), mx.concatenate([h, -h], axis=1)], axis=0
        )
    return h / (dim ** 0.5)


def next_pow_two(dim: int) -> int:
    size = 1
    while size < dim:
        size *= 2
    return size


class HadamardRotation:
    # Orthogonal rotation by a normalized Hadamard over the next power-of-two size, zero-padding
    # the rows - the inverse applies the transpose and drops the padding
    def __init__(self, dim: int):
        self.dim = dim
        self.size = next_pow_two(dim)
        self.matrix = hadamard_matrix(self.size)

    def apply(self, rows: mx.array) -> mx.array:
        if rows.shape[-1] != self.dim:
            raise ValueError(f"rows have {rows.shape[-1]} dims, rotation built for {self.dim}")
        pad = mx.zeros((*rows.shape[:-1], self.size - self.dim), dtype=mx.float32)
        padded = mx.concatenate([rows.astype(mx.float32), pad], axis=-1)
        return padded @ self.matrix

    def invert(self, rows: mx.array) -> mx.array:
        unrotated = rows @ self.matrix.T
        return unrotated[..., : self.dim]


@dataclass
class TqCache:
    # Quantized rows: uint values in [0, 2^bits - 1] [N, d_kept], per-row scales and zero points [N, 1]
    values: mx.array
    scales: mx.array
    zeros: mx.array
    # Channels kept in fp16 [N, k] and their column indices [k]
    fp16_values: mx.array | None
    fp16_channels: mx.array | None
    bits: int


class TqQuantizer:
    # TurboQuant-style per-row asymmetric quantization with fp16 escape channels for the
    # outlier/sink columns; 2/4/8 bit, stored as uint8 pending packed layouts
    def __init__(self, bits: int = 4, fp16_channels: list[int] | None = None):
        if bits not in QUANT_BITS:
            raise ValueError(f"bits must be one of {QUANT_BITS} - got {bits}")
        self.bits = bits
        self.fp16_channels = sorted(set(fp16_channels or []))

    def quantize(self, rows: mx.array) -> TqCache:
        if rows.ndim != 2:
            raise ValueError(f"quantize takes [N, d] rows - got shape {rows.shape}")
        rows = rows.astype(mx.float32)
        dim = rows.shape[1]
        bad = [c for c in self.fp16_channels if c >= dim]
        if bad:
            raise ValueError(f"fp16 channels {bad} out of range for {dim} columns")
        kept = [c for c in range(dim) if c not in self.fp16_channels]
        fp16_values = None
        fp16_channels = None
        if self.fp16_channels:
            fp16_values = rows[:, self.fp16_channels]
            fp16_channels = mx.array(self.fp16_channels, dtype=mx.int32)
        block = rows[:, kept] if kept else rows[:, :0]
        levels = (1 << self.bits) - 1
        low = block.min(axis=1, keepdims=True)
        high = block.max(axis=1, keepdims=True)
        scales = (high - low) / levels
        scales = mx.maximum(scales, mx.array(1e-8, dtype=mx.float32))
        zeros = -low / scales
        values = mx.round(block / scales + zeros)
        values = mx.clip(values, 0.0, float(levels)).astype(mx.uint8)
        return TqCache(values, scales, zeros, fp16_values, fp16_channels, self.bits)

    def dequantize(self, cache: TqCache) -> mx.array:
        block = (cache.values.astype(mx.float32) - cache.zeros) * cache.scales
        if cache.fp16_values is None:
            return block
        kept_set = set(self.fp16_channels)
        dim = block.shape[1] + cache.fp16_values.shape[1]
        columns = []
        block_i = 0
        fp16_i = 0
        for channel in range(dim):
            if channel in kept_set:
                columns.append(cache.fp16_values[:, fp16_i : fp16_i + 1])
                fp16_i += 1
            else:
                columns.append(block[:, block_i : block_i + 1])
                block_i += 1
        return mx.concatenate(columns, axis=1)
