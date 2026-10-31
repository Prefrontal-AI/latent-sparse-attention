# Every command the repo needs. CI runs lint and types as jobs of their own, types a second time with the evaluation
# extra installed, because a bare sync resolves the extras' packages to Any and hides their errors, and the
# tests as test-core, test-evaluation and test-training - one job per dependency set, each failing on any skip.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help sync lock lint types format test test-unit test-integ test-core test-evaluation test-training check check-ci hooks build clean

help:  ## List the targets
	@awk -F':.*##' '/^[a-z][a-z-]*:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync:  ## Install the env from uv.lock - fails if the lock is stale
	$(UV) sync --locked

lock:  ## Re-resolve uv.lock after editing pyproject
	$(UV) lock

lint:  ## Check style without changing files
	$(UV) run ruff check
	$(UV) run ruff format --check
	$(UV) run python tools/check_repo.py

types:  ## Type-check the package and the tests
	$(UV) run mypy

# autotyping fills in the annotations mypy would ask for and can infer - a `-> None` on a function that
# never returns, a scalar return type. Nothing else about a type error is autofixable, and a missing stub
# package is `uv add --dev types-<name>` rather than a fix a tool should make behind the lock's back.
format:  ## Apply formatting and every fix a tool can make
	$(UV) run autotyping --safe src
	$(UV) run ruff format
	$(UV) run ruff check --fix

test-unit:  ## Run isolated contracts and numerical tests
	$(UV) run pytest tests/unit
test-integ:  ## Run workflow, model, service and corpus integration tests
	$(UV) run pytest tests/integ

test:  ## Run the tests the installed env can run, skipping the rest
	$(UV) run pytest
# What CI runs, one job per dependency set. Each deselects what no runner can give - a GPU, a Slurm cluster's
# data - and fails on any other skip, so a test that needs a package its job did not install fails instead
# of passing unseen. Both scopes keep evaluation and training domains in their own dependency jobs
EVERY_TEST := LSA_NO_SKIPS=1 $(UV) run pytest -m "not gpu and not cluster"
test-core:  ## CI's test-core - the package on a bare sync
	$(EVERY_TEST) tests/unit tests/integ --ignore=tests/unit/evaluation --ignore=tests/integ/evaluation --ignore=tests/unit/training --ignore=tests/integ/training
test-evaluation:  ## CI's test-evaluation - needs --extra evaluation
	$(EVERY_TEST) tests/unit/evaluation tests/integ/evaluation
test-training:  ## CI's test-training - needs --extra training
	$(EVERY_TEST) tests/unit/training tests/integ/training

check: lint types test  ## Every gate at once - what CI's jobs add up to

# check passes on what the installed env can run; this fails on any skip, as CI's jobs do. Needs both extras
# installed where CI's jobs install them one at a time, so it cannot stand in for the bare-sync jobs (types and
# test-core)
check-ci: lint types test-core test-evaluation test-training  ## The CI jobs' targets in one env - needs both extras

hooks:  ## Install the pre-commit hooks into .git/hooks
	$(UV) run pre-commit install

build:  ## Build the sdist and wheel into dist/
	$(UV) build

clean:  ## Remove build output and tool caches
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find src tests -name __pycache__ -type d -exec rm -rf {} +
