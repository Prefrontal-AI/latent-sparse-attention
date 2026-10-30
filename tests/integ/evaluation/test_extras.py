"""The `evaluation` extra's guard - the friendly error has to be what a caller meets, not a vendor import error.

The subprocess hides omegaconf, hydra, transformers and lm_eval from the import machinery, which is what a
bare `uv sync` looks like. It has to be a subprocess: a package already imported in this interpreter cannot
be un-imported.

A package that re-exported its own config would die on omegaconf before `require_extra` could name anything,
so the first test here is that `import lsa.evaluation` reaches none of them.
"""

import subprocess
import sys

HIDE = """
import sys
from importlib.machinery import PathFinder

BLOCKED = {"omegaconf", "hydra", "transformers", "lm_eval"}


class HidingPathFinder(PathFinder):
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            return None
        return super().find_spec(name, path, target)


sys.meta_path = [HidingPathFinder if f is PathFinder else f for f in sys.meta_path]
"""


def run_without_the_extra(body):
    return subprocess.run([sys.executable, "-c", HIDE + body], capture_output=True, text=True)


def test_the_packages_are_really_hidden():
    finished = run_without_the_extra("import omegaconf\n")
    assert "ModuleNotFoundError" in finished.stderr
    assert "omegaconf" in finished.stderr


def test_the_package_imports_without_a_single_package_of_the_extra():
    finished = run_without_the_extra("import lsa.evaluation\nprint('imported')\n")
    assert finished.returncode == 0, finished.stderr
    assert "imported" in finished.stdout


def test_the_guard_names_the_extra_rather_than_the_vendor_import_that_failed():
    finished = run_without_the_extra(
        "from lsa.evaluation import require_extra\n"
        "try:\n"
        "    require_extra('omegaconf', 'hydra', 'transformers', 'lm_eval')\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    assert finished.returncode == 0, finished.stderr
    assert "uv sync --extra evaluation" in finished.stdout
    for name in ("omegaconf", "hydra", "transformers", "lm_eval"):
        assert name in finished.stdout


def test_the_entry_point_meets_the_guard_rather_than_a_missing_hydra():
    finished = run_without_the_extra(
        "import sys\nsys.argv = ['lsa', 'evaluate', 'accuracy', 'configs/evaluation/does-not-exist.yaml']\n"
        "from lsa.cli import main\nmain()\n"
    )
    assert finished.returncode != 0
    assert "uv sync --extra evaluation" in finished.stderr
