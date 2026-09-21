# Contributing

## Setup

From the repo root:

```bash
conda env create -f src/norfolkenv.yml
conda activate norfolk
```

## Running tests

```bash
pytest src/model/tests
```

## Pull requests

- Keep PRs scoped to one change.
- Add or update tests under `src/model/tests/` for any behavior change.
- Run the test suite locally before opening a PR.
