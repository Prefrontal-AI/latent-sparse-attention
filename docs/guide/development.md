# Development

```bash
make hooks     # install the pre-commit hooks - once per clone
make format    # apply every fix a tool can make
make check     # lint, types and test - what CI gates on
make sync      # the 3.12 env from uv.lock
```

What follows is what they do that is not obvious from the target name, and the environment they run in.

## The gates

`make format` runs `autotyping` before ruff, so an annotation mypy would ask for and a tool can infer - a `-> None` on a function that never returns - is filled in rather than reported.
Nothing else about a type error is autofixable: a missing stub package is `uv add --dev types-<name>`, not something a formatter should install behind the lock's back.

The hooks run the project's own tools through `uv`, so a commit is checked against the versions the lock pins rather than a separately pinned mirror that would drift from `make check`.

Two hooks take no file list.
mypy needs the whole package to resolve.
The repository checks - links and dashes - run on every commit whatever is staged, since a dash can land in any file and a staged-file filter leaves out deletions, and `make lint` runs the same `tools/check_repo.py`.

## Test scopes

```bash
make test-unit
make test-integ
make test-core
make test-evaluation
make test-training
```

`tests/unit/` checks isolated contracts and numerical operations; `tests/integ/` checks workflows spanning modules, models, services and on-disk corpora.
Both keep domain directories, with shared fakes in `tests/helpers/` and collection policy in `tests/conftest.py`.
The dependency targets run both scopes, and GPU and cluster requirements stay independent pytest markers.

## Reproducing CI

```bash
make sync                                    # a bare sync - what test-core and types run on
make check                                   # lint, types, and the tests this env can run
uv sync --locked --extra evaluation --extra training --no-install-package causal-conv1d
make check-ci                                # CI's targets in this one env: lint, types, test-core, test-evaluation, test-training
UV_PROJECT_ENVIRONMENT=$TMPDIR/ci-bare uv sync --locked && UV_PROJECT_ENVIRONMENT=$TMPDIR/ci-bare make types test-core   # the two bare-sync jobs, in a bare env
```

`make test` skips what the installed env cannot run - a GPU test, a test that needs a package - and still passes.
It ends with one line naming the files skipped and why.
On a bare sync a test module that imports an extra's package at the top fails at collection instead.
`make check-ci` runs CI's targets in one env with both extras installed, each test target failing on any skip.
It does not stand in for the two jobs that run on a bare sync, `types` and `test-core`: the second also proves the package imports without an extra, which an env holding the extras cannot.

CI's `types` job runs on a bare sync, where the extras' packages resolve to `Any` and hide their errors.
Every package the evaluation or training code imports from an extra needs an `ignore_missing_imports` entry in `pyproject.toml`, or CI fails where a local run with the extra passes.
The last command above runs both of them against a bare environment without tearing down the one the extras are installed in.

## Environment variables

| Variable | Read by | Meaning |
| --- | --- | --- |
| `LSA_ROOT` | every path under the shared root | the shared root, the site's `root` by default |
| `LSA_EVAL_DATA` | the benchmark data | where the fixed benchmark files live, `data/benchmarks` under the root by default - see [data](data.md) |
| `RULER_NVIDIA`, `NLTK_DATA` | the RULER setup | the pinned generator clone and the sentence tokenizer, both under `LSA_EVAL_DATA` by default - see [data](data.md) |
| `LSA_NO_SKIPS` | the test run | fail on any skipped test - the `test-*` targets set it |

`eval "$(lsa env)"` exports the first two and the cache variables a job needs; a submitted job gets them without it.

## Dependencies

`uv add` edits `pyproject.toml`, re-resolves `uv.lock`, and installs into `.venv` in one step - the lock is generated, never hand-edited.

```bash
uv add torch                     # runtime dep -> [project] dependencies
uv add --dev pytest-xdist        # tool the package does not import -> [dependency-groups] dev
uv remove torch                  # drop one, both files stay in step
uv lock --upgrade-package torch  # move a single dep, leave the rest where they are
make check                       # lint and test against the new env
```

`uv add` writes a `>=` bound at the version it just resolved - that is the version the dep was validated against, so leave the bound as it lands rather than widening it.
Add a dep in the same change as the code that imports it, never ahead of it.

Commit `pyproject.toml` and `uv.lock` together: CI runs `make sync` (`uv sync --locked`), which fails on a lock that has drifted from the pyproject.

For what `uv add` cannot express - an environment marker, a constraint, an extra - edit `pyproject.toml` by hand and follow it with `make lock && make sync`.

### Analysis

The analysis notebooks need jupyter, matplotlib, pandas and seaborn, which sit in the `analysis` dependency group - outside the default sync, so CI never installs them.

```bash
uv sync --group analysis         # the dev group stays, the notebook libraries are added
uv sync --group analysis --extra evaluation  # with an extra as well
```

### Torch

`torch` is one of those: `[tool.uv.sources]` resolves it from the cu129 index rather than PyPI, because the H200 cluster is on driver 570 and a `cu130` wheel will not load there.

The driver is the ceiling, not the index.
It takes any CUDA 12.x runtime, and cu129 is the last 12.x build of torch 2.13.0 and the first with kernels for Blackwell, so the same lock runs on the H200 and the B300 nodes.
A cu126 wheel holds no Blackwell kernel, and every torch operation on a B300 fails with no kernel image for the device.
The pin is exact, and so is the CuTeDSL compiler's, because the image installs this package without its deps.
Triton's order of adding a sum decides the kernels' bits, so a test on a torch the image does not run proves nothing about what serves.
Move the pin only with the image.

`torchtitan` is a git pin rather than a release, because it moves with torch and a floating version would change the trainer under a run.
An extra may name a direct reference like that only with `tool.hatch.metadata.allow-direct-references`.
