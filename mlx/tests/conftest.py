import logging
import os
import sys
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

UPSTREAM_DEFAULT = Path.home() / "model" / "latent-sparse-attention" / "src"


@pytest.fixture(scope="session")
def upstream_src() -> Path:
    root = Path(os.environ.get("LSA_UPSTREAM", UPSTREAM_DEFAULT))
    if not root.is_dir():
        pytest.skip(f"upstream oracle not available at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
