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

- Concept and version DOIs from the first Zenodo deposit, recorded in
  `CITATION.cff` and as a README badge.
- Statutory minimum-wage panel built from the official DGERT history: one row
  per legal act, scope and effective date, covering 1974 to 2026 with the
  citation of every act. Written to `data/processed/minimum_wage_policy.parquet`
  by `ptmw build minimum-wage`.
- Annual collapse of the statutory panel offering both the 1 January level and
  the day-weighted average, which diverge in the eight years with a mid-year act.
- Payload verification on download: a response whose media type contradicts the
  declared source kind is rejected instead of being written under a data file
  name.
- Snapshot-on-change retrieval: a changed upstream file is retained under a
  timestamped name rather than overwritten.
- Registry validation for empty registries, duplicate destinations, and
  withdrawn sources, which must state why they are unreachable.
- Cross-validation against Eurostat's independently compiled minimum-wage
  series, pinned in the test suite at five years.

### Changed

- Source registry records licence terms and a verification date per source.
- Provenance manifest uses forward-slash paths, so it is byte-identical across
  platforms for identical downloads.

### Fixed

- The registered GEP bulletin link is withdrawn upstream: the host now answers
  200 with an HTML landing page for every path. It is disabled with a recorded
  reason instead of silently storing that page as a spreadsheet.

## [0.2.0] - 2026-08-09

First archived release. No estimate changes: this release makes the repository
citable, licensed, and continuously verified, and does not alter any result.

### Added

- Repository governance: `LICENSE` (MIT) and `LICENSE-CC-BY-4.0.txt`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`,
  issue and pull-request templates.
- Continuous integration running Ruff, mypy (strict), and pytest on Python 3.12
  and 3.13.
- Dependabot updates for Poetry dependencies and GitHub Actions.
- `.gitattributes` enforcing LF line endings and correct binary handling.
- `.zenodo.json` supplying deposit metadata for Zenodo archiving, so each
  tagged release is citable by DOI. ORCID and affiliation added to
  `CITATION.cff` to match.
- CI validation of `CITATION.cff` against the CFF 1.2.0 schema and of
  `.zenodo.json`, including a cross-check that both agree on title and version.

### Changed

- Packaging migrated to the PEP 621 `[project]` table, clearing the Poetry 2
  deprecation warnings.
- `poetry.lock` is now committed, so a rerun resolves to the same dependency
  versions.
- Dependency floors raised past known advisories: `lxml` to 6.1 (XXE via the
  default `iterparse()` configuration), `pyarrow` to 23.0.1 (use-after-free
  reading IPC files), `pytest` to 9.0.3 (tmpdir handling).

### Fixed

- The scaffold did not satisfy its own lint and type gates. Corrected a
  `groupby` aggregation that returned a Series where a DataFrame was renamed,
  an under-typed row dictionary in the Eurostat adapter, a flake8-bugbear
  false positive on Typer's call-valued defaults, and missing import handling
  for the untyped `statsmodels`.

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

[Unreleased]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/releases/tag/v0.1.0
