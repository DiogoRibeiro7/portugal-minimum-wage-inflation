# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, one extra rule applies: **any change that
alters a published estimate is a breaking change** and is recorded under
`Changed` with the before/after values, regardless of how small the code diff
was.

## [Unreleased]

### Added

- Repository governance: `LICENSE` (MIT) and `LICENSE-CC-BY-4.0.txt`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`,
  issue and pull-request templates.
- Continuous integration running Ruff, mypy (strict), and pytest on Python 3.12
  and 3.13.
- Dependabot updates for Poetry dependencies and GitHub Actions.
- `.gitattributes` enforcing LF line endings and correct binary handling.

## [0.1.0] - 2026-08-09

### Added

- Project scaffold: Poetry packaging, Ruff, mypy strict, pytest, pre-commit.
- `src/pt_mw_inflation/data/`: HTTP retrieval with checksum recording, a source
  registry driven by `config/sources.yaml`, and a Eurostat HICP adapter.
- `src/pt_mw_inflation/processing/`: minimum-wage series construction and
  exposure-measure construction.
- `src/pt_mw_inflation/analysis/`: descriptive statistics and panel local
  projections.
- `ptmw` command-line interface covering data download, dataset builds, and
  analysis entry points.
- Pydantic schemas for data contracts.
- `docs/`: research design, data dictionary, and literature map.
- `report/`: LaTeX manuscript skeleton with sections and bibliography.
- Unit tests for minimum-wage and exposure processing.

[Unreleased]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/releases/tag/v0.1.0
