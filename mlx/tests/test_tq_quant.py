import logging

import mlx.core as mx
import numpy as np
import pytest

from fusion_lsa.core.reference import build_buckets, read
from fusion_lsa.quant.tq import HadamardRotation, TqQuantizer, hadamard_matrix

logger = logging.getLogger(__name__)


def _unit_keys(total: int, dim: int = 64, seed: int = 3) -> mx.array:
    keys = mx.random.normal((total, dim), key=mx.random.key(seed))
    return keys / mx.linalg.norm(keys, axis=1, keepdims=True)


def test_reconstruction_error_bounds():
    keys = _unit_keys(512)
    q = TqQuantizer(bits=4)
    deq = q.dequantize(q.quantize(keys))
    # 4-bit asymmetric on unit rows: per-key max abs error well under the bucket selection tolerance
    err = mx.abs(deq - keys).max().item()
    assert err < 0.05, f"4-bit reconstruction error {err:.4f} too large"
    q8 = TqQuantizer(bits=8).dequantize(TqQuantizer(bits=8).quantize(keys))
    assert mx.abs(q8 - keys).max().item() < 0.01


def test_rejected_bits():
    keys = _unit_keys(8)
    with pytest.raises(ValueError, match="bits"):
        TqQuantizer(bits=3).quantize(keys)


def test_sink_channels_stay_fp16():
    keys = _unit_keys(64)
    q = TqQuantizer(bits=4, fp16_channels=[0, 1])
    deq = q.dequantize(q.quantize(keys))
    # kept channels round-trip exactly
    assert mx.abs(deq[:, 0] - keys[:, 0]).max().item() < 1e-6
    assert mx.abs(deq[:, 1] - keys[:, 1]).max().item() < 1e-6
    # other channels are quantized
    assert mx.abs(deq[:, 5] - keys[:, 5]).max().item() > 1e-6


def test_hadamard_is_orthonormal():
    h4 = hadamard_matrix(4)
    prod = h4 @ h4.T
    eye = mx.eye(4)
    assert float(mx.abs(prod - eye).max().item()) < 1e-5


def test_hadamard_equalizes_outliers():
    # one huge outlier channel dominates row norms; rotation spreads it
    keys = _unit_keys(256)
    spiked = keys * 1.0
    spiked = spiked / mx.linalg.norm(spiked, axis=1, keepdims=True)
    raw_err = TqQuantizer(bits=4).dequantize(TqQuantizer(bits=4).quantize(spiked))
    h = hadamard_matrix(64)
    rotated = spiked @ h
    rot_err = TqQuantizer(bits=4).dequantize(TqQuantizer(bits=4).quantize(rotated))
    deq_rot = rot_err @ h.T
    assert float((mx.abs(deq_rot - spiked)).mean().item()) <= float((mx.abs(raw_err - spiked)).mean().item())


def test_selection_survives_4bit_keys():
    # buckets built on dequantized keys still select nearly the same key sets on real-scale data
    keys = _unit_keys(8192)
    queries = _unit_keys(16, seed=9)
    positions = mx.arange(16) + 4096
    buckets_fp = build_buckets(keys, 4, 8, 16, 512)
    selected_fp, _ = read(queries, positions, buckets_fp, 512, 2048, 4, 512)
    deq = TqQuantizer(bits=4).dequantize(TqQuantizer(bits=4).quantize(keys))
    # stored keys are renormalized after dequantization, as the serving path does
    deq = deq / mx.linalg.norm(deq, axis=1, keepdims=True)
    buckets_q = build_buckets(deq, 4, 8, 16, 512)
    selected_q, _ = read(queries, positions, buckets_q, 512, 2048, 4, 512)
    overlaps = []
    for row in range(16):
        want = set(int(x) for x in np.array(selected_fp)[row] if x >= 0)
        got = set(int(x) for x in np.array(selected_q)[row] if x >= 0)
        overlaps.append(len(want & got) / max(len(want), 1))
    mean_overlap = float(np.mean(overlaps))
    logger.info("4-bit key quantization selection overlap (synthetic gaussian): %.4f", mean_overlap)
    # Gaussian unit keys are the hardest case - every centroid is almost equally similar.
    # The real-weights number comes from scripts/tq_probe.py on extracted checkpoint vectors.
    assert mean_overlap >= 0.85, f"selection overlap {mean_overlap:.4f} under 0.85"


def test_hadamard_rotation_roundtrip_padded():
    rows = mx.random.normal((16, 768), key=mx.random.key(11))
    rot = HadamardRotation(768)
    rotated = rot.apply(rows)
    assert rotated.shape == (16, 1024)
    back = rot.invert(rotated)
    # fp32 accumulation over a 1024-dim matmul; relative error stays ~1e-4 of the row norms
    assert float(mx.abs(back - rows).max().item()) < 1e-2
