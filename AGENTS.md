# Agents

## Where things go

How to run each part lives in `docs/guide/`, and [the glossary](docs/reference/notation.md) binds every name.

An artifact is described by its card under `src/lsa/assets`, a run by its yaml, and a yaml names its artifacts by card rather than by path.
What a step measured lives in the artifact's manifest on disk, never on the card.

## Pull requests

The description is the whole record of a change: results, dead ends and corrections go there.
There is no logbook and no running state file.

Fill in [the template](.github/pull_request_template.md) - `gh pr create` does not apply it, so pass `--body-file`.
Say what you could not test: `make check` is CPU-only, so name any GPU path, multi-device load or unsubmitted job in `Testing`.
Delete a heading with nothing under it.

## CI

Run `make format`, then `make check`, before saying something works.
`make hooks` installs the commit hook once per clone - ruff, autotyping, mypy and the repository checks, not the tests.
Seed a new worktree's mypy cache from an existing one (`cp -r <other-worktree>/.mypy_cache .`) before the first commit.

`tests/unit/` holds isolated contracts, `tests/integ/` workflows crossing module, model, service or corpus boundaries, `tests/helpers/` the shared fakes.
The `gpu` and `cluster` markers run on the cluster only.
CI jobs `test-core`, `test-evaluation` and `test-training` are Makefile targets of the same names, and fail on any skip other than those two markers.
A test that needs the `evaluation` or `training` extra goes under that domain's directory, or its job fails.

## Style

### Prose

One sentence per line, never hard-wrapped to a column.
A long sentence breaks at a clause boundary after " - " or a comma, or becomes two sentences.
Tables, code blocks and link targets are never broken.
Short sentences, no fluff, no restating the heading.
Lead with the commands in anything operational.
Prefer a heading and a paragraph to a bulleted list unless the items are parallel and unordered.

Use only the plain hyphen `-` everywhere, including comments and commit messages.
Never an em or en dash.
Join short clauses with " - ".

### Code

Simple over clever, top to bottom, one job per function, delete before adding.
Order a function as setup, work, return; return early rather than nest.
No banners - a file that needs section banners is too big.

Comments are rare.
On their own line above the code, uppercase start, no final period, terse noun phrases.
State the fact, then the conclusion on the next line with `=>`.

```python
# Cache may be stale
# => refetch
```

Names say what the thing is: a verb for a function, a noun for a value, spelled out (`seats_left`, not `room`).
Short names are for tensor math (`q`, `k`, `v`, the model's own attention tensors), a loop counter `i` and one-line comprehensions.
One concept, one name across every module; a rename is not done until grep finds the same name everywhere.
[The glossary](docs/reference/notation.md) binds every symbol, config key and code name, its case and extents and head counts included - check it before inventing a name, and extend it rather than spelling a concept twice.

Every function in `src/` has full annotations (tests are exempt) and a Google-style docstring.
The first docstring line is one imperative sentence ending in a period.
`Args:`, `Returns:` and `Raises:` follow a blank line, and empty sections are left out.
Shapes and dtypes ride on the entry that carries them, as `index_queries: Index queries [R, index_dim] fp32.`
Never restate an annotation in prose.

### Checks

No `assert` in `src/`.
An invariant is a test.
Bad input from outside the package raises at the public boundary, says what arrived and what was wanted, and is covered by a test.
