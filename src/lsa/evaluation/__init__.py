"""Evaluation - what a model scores, what its attention costs, and what it takes to serve.

One experiment yaml says what a run is and the tree it sits in says its kind, and `run` sends it to slurm and
carries it out in the job - `lsa evaluate <kind> <config>`. Each kind of measurement is a package of its own and writes its own record into
the one records root the viewer reads.
The server a served run scores or times is the serving backend's - `lsa.sglang.launch` starts it, from the
settings `lsa.sglang.settings.design` holds.

Behind the `evaluation` extra, because `import lsa` reaches only torch and this reaches a model.
The config layer needs `omegaconf` and `hydra`; the accuracy suite needs `lm_eval` for two of its graders and
the probe needs `transformers`, and `require_extra` says so by name rather than letting a stack trace arrive
from inside a vendor package.

This file imports nothing of the package, so `import lsa.evaluation` runs under a bare `uv sync` and the guard is
what a caller meets rather than a `ModuleNotFoundError` from a vendor import.

| module | what it does |
|---|---|
| [config](config.py) | one schema per kind - `model`, `data` and `run` each - and the loader that reads a file through its kind's |
| [run](run.py) | submit one file's experiments to slurm, and the runner each kind's job calls |
| [accuracy](accuracy/__init__.py) | what a served model scores - its data, tasks, graders, the client, the record |
| [performance](performance/__init__.py) | what one attention layer costs (`kernels`) and what the whole served model takes to answer (`production`), each with its data |
| viewer (optional) | the page that puts all three kinds of record side by side |
| [adapter](adapter/__init__.py) | the model surgery a teacher extraction reads a checkpoint through, and the probe that says whether LSA can run on it |
"""

from importlib.util import find_spec

__all__ = ["require_extra"]


def require_extra(*packages: str) -> None:
    """Fail with the missing extra rather than an import error from inside a vendor package.

    Nothing of the package is imported by this file: a package that re-exported its own config would reach
    omegaconf before this could name it.

    Args:
        *packages: The packages this caller needs - the config layer needs `omegaconf` and `hydra`, the
            probe `transformers`.

    Raises:
        ImportError: A package the caller needs is not installed.
    """
    missing = [name for name in packages if find_spec(name) is None]
    if missing:
        raise ImportError(
            f"lsa.evaluation needs the evaluation extra - missing {', '.join(missing)} - run `uv sync --extra evaluation`"
        )
