"""Which of the two architectures a kernel measurement drives a checkpoint is."""

from pathlib import Path

from lsa import io
from lsa.config import text_config
from lsa.evaluation.constants import Family


def model_family(model_type: str) -> str:
    """Return `Family.QWEN3_5` for a Qwen3.5 text `model_type` and `Family.GLM5_NEXT` for any other."""
    return Family.QWEN3_5 if model_type.startswith(Family.QWEN3_5) else Family.GLM5_NEXT


def checkpoint_family(model_path: str) -> str:
    """Return the family of a checkpoint from its own config, before an engine loads it."""
    config = io.read_json(Path(model_path) / "config.json")
    return model_family(text_config(config).get("model_type", ""))
