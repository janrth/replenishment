# Contributing

Thanks for contributing to `replenishment`.

## Development Setup

The project supports Python `3.10`, `3.11`, `3.12`, and `3.13`.

Using `uv` is the simplest setup:

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
```

If you want to validate a different supported version, replace `3.13` with
`3.10`, `3.11`, or `3.12`.

## Running Checks

Run the test suite with:

```bash
uv run pytest
```

If you change notebook-facing behavior, also rerun the relevant example notebook
or plot-generation snippet and update the checked-in plot assets under
`docs/plots/` when needed.

## Pull Requests

To keep reviews fast and predictable:

- keep each PR focused on one logical change
- add or update tests for behavior changes
- update README examples, notebooks, and plot assets when public APIs or outputs change
- include screenshots or plots for visualization changes
- call out any follow-up work explicitly instead of leaving it implicit

## Good First Contributions

Useful contribution areas include:

- new replenishment policies or policy constraints
- additional synthetic-data generators and scenario notebooks
- API docs and notebook cleanup
- CI, packaging, and release automation improvements
- bug fixes with regression tests
