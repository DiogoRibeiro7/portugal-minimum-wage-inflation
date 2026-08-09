# Contributing

Thanks for your interest in this project. It is a reproducible-research
repository, so contributions are judged on two axes: whether the code is sound,
and whether the empirical result it produces can be regenerated from scratch by
someone else.

## Ways to contribute

- **Data corrections.** A wrong minimum-wage schedule, a misread gazette, a
  mismatched NUTS II code, or a broken source URL is the highest-value bug in
  this repository. Open a *Data issue* and cite the primary source.
- **Methodological review.** Challenges to the identification strategy,
  exposure construction, or inference are welcome as issues. See
  `docs/research_design.md` first.
- **Code.** Bug fixes, new source adapters, tests, and performance work.
- **Documentation.** Clarifications to the data dictionary and research design.

## Development setup

Python 3.12 and Poetry are required.

```bash
poetry install
poetry run pre-commit install
```

`pre-commit install` is not optional — the same Ruff and mypy versions run in
CI, and installing the hooks is the cheapest way to avoid a red build.

## Before opening a pull request

```bash
make check   # ruff check + ruff format --check + mypy + pytest
```

All four must pass. Individually:

```bash
make lint        # poetry run ruff check src tests && ruff format --check src tests
make typecheck   # poetry run mypy src
make test        # poetry run pytest
```

mypy runs in `strict` mode over `src/`. New code needs full annotations; do not
add `# type: ignore` without a comment explaining why the type cannot be
expressed.

## Reproducibility rules

These are the constraints that make the repository citable. A pull request that
breaks one will not be merged.

- **Raw files are immutable.** Nothing under `data/raw/` is ever edited in
  place. A correction is a new retrieval with a new checksum.
- **Every source carries provenance**: URL, provider, retrieval date, and a
  SHA-256 hash, recorded in `config/sources.yaml`.
- **Transformations are deterministic and tested.** No wall-clock, no unseeded
  randomness, no network access inside processing or analysis code.
- **Specifications live in YAML**, not in notebooks and not hard-coded in
  function bodies. If your change introduces a tunable, it belongs in
  `config/analysis.yaml`.
- **Figures and tables are generated**, never hand-edited. Nothing under
  `report/figures/` or `report/tables/` is committed; both are build outputs.
- **No result is manually copied into the manuscript.** The LaTeX sources
  reference generated files.
- **Notebooks are for exploration only.** Anything that produces a published
  number moves into `src/pt_mw_inflation/` with a test.

## Tests

New data adapters need a contract test against a small fixture, not against the
live API — CI must pass with no network. New processing or analysis functions
need a unit test that pins the numerical result, so that a refactor that changes
an estimate is visible in the diff.

## Commit and pull-request style

- Write commit subjects in the imperative mood, under ~72 characters.
- Explain *why* in the body when the change is not self-evident.
- Keep a pull request to one logical change.
- If a change alters any published estimate, say so explicitly in the PR
  description and show the before/after numbers.

## Adding a new data source

1. Add the source to `config/sources.yaml` with provider, URL, licence, and the
   expected checksum.
2. Add an adapter under `src/pt_mw_inflation/data/`.
3. Document every resulting variable in `docs/data_dictionary.md`.
4. Add a fixture-based test.
5. Confirm the provider's terms permit redistribution of derived data, and note
   the licence in the PR.

## Licensing of contributions

By contributing you agree that your contributions to `src/`, `tests/`,
`notebooks/`, and `config/` are licensed under the MIT Licence, and that
contributions to `report/` and `docs/` are licensed under CC BY 4.0, matching
the repository's `LICENSE`.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
