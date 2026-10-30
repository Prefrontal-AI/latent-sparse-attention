"""What every test run shares.

A CI job sets `LSA_NO_SKIPS` and deselects the `gpu` and `cluster` tests by name, so any skip left is a
package its job should have installed or a test in the wrong job - and the run fails rather than reporting it
as a skip nobody reads. A local run without the variable skips as before, and says in one line which files it
skipped and why, so a bare sync does not pass the extras' tests unseen.
"""

import os

import pytest
import torch
import torch._dynamo.config

# One process runs every test, and each shape a test feeds `compiled_flex` is one recompile of the same function
# => past dynamo's default of 8 it quietly runs flex eagerly, whose backward materialises the full score matrix
# and exhausts a B300 at 32k. A training run sees a few shapes, so only a whole suite reaches the limit
torch._dynamo.config.recompile_limit = 128


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not os.environ.get("LSA_NO_SKIPS"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return
    reporter.write_sep("=", f"{len(skipped)} skipped in a run that allows none", red=True)
    for report in skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reporter.write_line(f"{report.nodeid}: {reason}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if os.environ.get("LSA_NO_SKIPS"):
        return
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return
    files_by_reason: dict[str, set[str]] = {}
    for report in skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ").splitlines()[0][:70]
        files_by_reason.setdefault(reason, set()).add(report.nodeid.split("::")[0])
    files = set().union(*files_by_reason.values())
    why = "; ".join(
        f"{len(found)} {'file' if len(found) == 1 else 'files'}: {reason}"
        for reason, found in sorted(files_by_reason.items(), key=lambda item: -len(item[1]))
    )
    terminalreporter.write_line(
        f"skipped {len(skipped)} tests in {len(files)} files - {why} - `make check-ci` runs what CI runs"
    )


@pytest.fixture(autouse=True)
def an_h200_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # No test node has a GPU, and a launch reads its card from the device
    # => every test runs on an H200 unless it says otherwise
    monkeypatch.setattr("lsa.evaluation.gpu.detect", lambda: "NVIDIA H200")


def pytest_collection_modifyitems(items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="the numerical kernels need a GPU")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
