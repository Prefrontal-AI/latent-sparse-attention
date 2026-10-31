"""A stand-in for the subprocess module the telemetry sampler starts nvidia-smi through."""

import subprocess
from types import SimpleNamespace
from typing import Any


def fake_subprocess(popen: Any) -> SimpleNamespace:
    """Return a module-shaped object whose `Popen` is the given class, so no other caller of subprocess is touched."""
    return SimpleNamespace(Popen=popen, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired)
