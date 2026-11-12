import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def to_mlx(t):
    return mx.array(np.asarray(t.detach().cpu().numpy()))


def to_numpy(a):
    return np.asarray(a)
