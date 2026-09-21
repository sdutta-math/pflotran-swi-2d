# Contributing

## Setup

From the repo root:

```bash
conda env create -f norfolkenv.yml
conda activate norfolk
pip install -e .
```

## Running tests

```bash
pytest src/pflotran_swi/tests
```

## Pull requests

- Keep PRs scoped to one change.
- Add or update tests under `src/pflotran_swi/tests/` for any behavior change.
- Run the test suite locally before opening a PR.
