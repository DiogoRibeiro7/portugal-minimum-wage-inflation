# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, one extra rule applies: **any change that
alters a published estimate is a breaking change** and is recorded under
`Changed` with the before/after values, regardless of how small the code diff
was.

## [Unreleased]

## [0.3.0] - 2026-08-14

Three designs are now estimated rather than argued about, and the paper's causal
layer reaches a settled conclusion: the binding constraint is that policy is
assigned to nine regions, not any missing structure.

**This release alters published estimates.** Every bootstrap $p$-value may move
by up to 0.016, for the reason recorded under `Changed`. No conclusion changes.

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
- Few-cluster inference: restricted wild cluster bootstrap with Rademacher
  weights, exhaustive over all `2**G` sign vectors when the cluster count is
  small, and cluster-level randomization inference.
- Panel local projections reporting both the conventional clustered p-value and
  the bootstrap p-value at every horizon, so the difference between them is
  visible rather than hidden.
- Event-study estimation with a normalised reference lead, and a pre-trend
  verdict naming every offending lead.
- Falsification checks: leave-one-region-out with a deviation summary in
  standard errors, placebo shock dating, and comparison across exposure
  definitions.
- Exposure bridge validation: bridge weights must sum to one per consumption
  category, shares must be proportions rather than percentages, and the
  baseline bite is frozen before each policy episode.
- Synthetic panel generator with a known dynamic response, used to verify that
  the estimators recover the truth and that each falsification check fires.
- Long-run annual macro dataset covering 1974 to 2025: statutory wage, consumer
  prices, real productivity, the productivity-plus-lagged-inflation benchmark
  and the policy residual, with the accounting identities checked on every
  build.
- AMECO adapter for real GDP per person employed from 1960, and a World Bank
  adapter for the consumer price index from 1960. Both reach back to the
  introduction of the minimum wage, where the Eurostat equivalents begin in
  1995 and 1996.
- Generated figures and LaTeX tables for the long-run layer, plus headline
  quantities emitted as LaTeX macros so the manuscript cannot quote a number
  the dataset does not produce.
- Manuscript sections for data and long-run evidence, written entirely against
  generated macros, and tests asserting that the prose cites no undefined
  quantity and hard-codes no result.
- `make paper`, which rebuilds the datasets, regenerates every output and
  compiles the manuscript.
- Withdrawn GEP publications restored from the Internet Archive and registered
  with their archival addresses, so the exposure sources are retrievable again.
- Reconciliation of the annual wage series against Eurostat, which fills years
  whose act is missing from the national history and records the compiler each
  observation came from.
- `construct_regional_bite`, the shift-share aggregation of a national
  industry-level bite to regions, for use once a genuine employment
  cross-tabulation is available.
- `assess_identifying_variation` and `require_regional_variation`, which measure
  the between-region share of exposure variance and refuse a measure that
  carries none. An exposure built from national inputs otherwise fails
  silently: it populates, passes every range check, runs, and identifies
  nothing.
- A documented feasibility assessment of the region-by-industry exposure
  measure, recording that no Portuguese source crosses a regional with an
  industry dimension, and what would unblock it.
- Retrieval of Portuguese legal acts from the Diário da República by ELI
  permalink, with parsing of the amounts and percentages they state. The
  gazette's web interface returns no content to a plain request; the permalink
  redirects to the published PDF, which is stable, official and citable.
- Regional statutory minimum wages for Madeira and the Azores, built from the
  acts themselves, together with the premium over the national wage.
- `config/legal_acts.yaml`, the register of acts retrieved as primary sources.
- Regional employment by NUTS II region and NACE activity from Eurostat's
  regional accounts, and the shift-share exposure it makes possible. Frozen on
  2015 composition, exposure ranges from 18.4 per cent in Norte to 20.5 in the
  Algarve, with nine distinct regional values.
- The national minimum-wage bite by economic activity, registered in
  `config/minimum_wage_bite.yaml` from the GEP monitoring report, keyed by
  activity name because the published table's letters and names are misaligned
  in its text layer.
- Regional consumer price panel from Statistics Portugal indicator `0014659`:
  price indices by NUTS II region and consumption purpose, monthly, built for
  2000 onwards by `ptmw data ine-cpi`. This is the last data dependency of the
  pass-through design, and no other source publishes it.
- The region-by-category design, built and estimated. `ptmw build
  structural-exposure` composes `B[r,c] = sum_s q[r,s] b[s] l[s] w[c,s]` and
  `ptmw analyse structural-design` estimates its interaction with the national
  statutory change, absorbing region-category, region-time and category-time
  effects together. It is the only design in the paper that rejects anything ---
  two of seven horizons at five per cent, none surviving Holm --- and it is
  disqualified by its own magnitudes. The coefficient is on an exposure index
  rather than a cost share and is not interpretable directly; scaled into
  points, three horizons imply differential price responses larger than complete
  pass-through of the entire minimum-wage cost bill could produce. What the
  exercise settles is that the binding constraint is the assignment of policy to
  nine regions rather than any missing structure: the structure was built, it
  absorbs the confounds the simpler designs are exposed to, and the answer did
  not improve.
- `bootstrap_with_interval`, which pays for a design's decomposition once and
  reads the estimate, its cluster-robust standard error, the bootstrap p-value
  and the inverted interval off it. The three-way design carries 2,695 columns
  against 12,077 rows, where the least-squares solve behind the standard error
  alone cost about ten minutes a horizon; the estimation went from uncomputable
  to twenty-two minutes. It also makes the estimate reported beside a p-value the
  estimate that p-value was computed from by construction rather than by
  coincidence. Every published figure of the other two designs is unchanged, and
  a test pins the equivalence.
- `build_absorbing_design`, generalising the design builder to any number of
  factors. Beyond two the design is rank deficient by construction, since
  region-time and category-time dummies both span the calendar-month main
  effects; the pseudo-inverse resolves it without touching any identified
  coefficient, and `joint_wald_test` gained a matching fallback reached only on
  the singular case, so the two-factor designs keep the solve they were computed
  with.
- The production-to-consumption bridge, which is the last term the
  region-by-category exposure needed. `ptmw build consumption-bridge` composes
  it from Eurostat's use table and the concordance recorded in
  `config/consumption_bridge.yaml`, and writes the labour shares and the
  consumption vector alongside it. The bridge is moderately concentrated, at an
  identifying spread of 2.31 percentage points against 2.22 for the moderate
  simulation and 0.23 for the diffuse one, so the design the decision log gated
  on this answer is worth estimating.
- `pt_mw_inflation.data.supply_use`, reading household final consumption by CPA
  product at basic prices and domestic uses. Both are arguments rather than
  defaults and both change the answer: at purchasers' prices a good's retail
  margin is credited to the industry that made it rather than the one that sold
  it, and including imports credits Portuguese employment with costs incurred
  abroad. It checks the accounting identity that domestic content, imported
  content and product taxes reconstruct the published total, since the excluded
  share is only meaningful if they do.
- Confidence intervals obtained by inverting the wild cluster bootstrap test,
  reported beside the $p$-value at every horizon of both estimated designs and
  led with in the manuscript. A null $p$-value says only that zero survives;
  the interval says what else does, which is what separates a design that found
  no effect from one that could not have found any. The exposure design's
  widest runs from -73.70 to 64.65, admitting full pass-through, sixty times
  full pass-through, and the same magnitudes negative.
- `invert_bootstrap_interval` widens its search when the interval runs past it,
  rather than reporting the ends of the search as the ends of the interval. The
  starting range is quoted in cluster-robust standard errors, which is exactly
  the statistic this module exists because it cannot be trusted: at impact on
  the regional design a $t$ of 8.2 carries a bootstrap $p$ of 0.23, so every
  candidate within six standard errors survived and the search never reached
  zero.

### Changed

- **Published estimates: every bootstrap $p$-value may move by up to 0.016.**
  Two draws of the enumerated sign space reproduce the observed statistic
  exactly, and whether they were counted was previously decided by which way
  the last bits of a floating-point comparison fell. They are now counted, and
  the sign symmetry that guarantees they come in pairs is imposed rather than
  recomputed. The regional design's bootstrap $p$-values move from
  0.234/0.029/0.303/0.441/0.494/0.383/0.352 to
  0.238/0.031/0.305/0.441/0.496/0.391/0.359 across horizons 0 to 24, and its
  smallest Holm-adjusted value from 0.205 to 0.219. The exposure design's move
  from 0.512/0.871/0.904/0.516/0.984/0.461/0.914 to
  0.512/0.871/0.906/0.516/0.984/0.469/0.922. No conclusion changes: one horizon
  still survives at five per cent before the Holm correction and none after it,
  and the exposure design still rejects nothing.
- Manuscript restructured around the long-run record. The paper now documents
  the statutory series built from primary law, the real erosion of the wage
  floor against productivity, and a negative identification result explaining
  why regional variation cannot identify pass-through in Portugal. No causal
  claim is made.
- The 2000 statutory values are read from Decreto-Lei n.º 573/99 rather than
  taken from Eurostat. With that act and its domestic-service value included,
  every stated increase in the published history reconciles with the act
  preceding it, and no year is sourced to a secondary compiler.
- `annual_minimum_wage` takes a geography. The panel holds both autonomous
  regions under the general regime, so filtering on scope alone returned a
  regional wage as the national one for every year a region legislated.
- Packaging migrated to the PEP 621 `[project]` table, clearing the Poetry 2
  deprecation warnings.
- `poetry.lock` is now committed, so a rerun resolves to the same dependency
  versions.
- Dependency floors raised past known advisories: `lxml` to 6.1, `pyarrow` to
  23.0.1, `pytest` to 9.0.3, `cryptography` to 50.0. The last is required by
  pypdf to read the AES-encrypted PDFs the gazette publishes, and the initial
  constraint capped it below the patched release.
- Source registry records licence terms and a verification date per source.
- Provenance manifest uses forward-slash paths, so it is byte-identical across
  platforms for identical downloads.

### Fixed

- `structural_exposure` records whether the bite it was given carried the share
  of each group's employment the bite was measured on, and the builder warns
  when it did not. The column is optional, and passing the bite as a bare column
  rather than the full frame dropped it silently, turning the coverage weighting
  off while looking identical to having asked for it. That is how
  `docs/consumption_bridge_feasibility.py` came to report 2.31 points of
  identifying spread against the pipeline's 1.92; the script now calls the
  production builder, so there is one number and one code path.
- `invert_bootstrap_interval` is fast enough to run in the pipeline, and is now
  wired into it. It rebuilt the decomposition of a design carrying one dummy per
  region-category and per month on every one of its candidate values, so a
  seven-horizon path took roughly fifty minutes per design and was excluded from
  `make paper`. The projection is now built once per horizon and reused, the
  restricted fit comes from it by Frisch-Waugh instead of a second
  least-squares solve of the reduced design, and the sign space is evaluated in
  the cluster coordinates it actually occupies rather than one draw at a time.
  A horizon's interval takes under two seconds; the bootstrap $p$-value alone
  went from about ten seconds to under a third of one.
- `write_regional_design_table` ignored its `command` argument and always named
  `ptmw analyse pass-through`, so `exposure_design.tex` told the reader to
  regenerate it with a command that does not produce it. This is the third
  instance of that defect, and the second in a writer that took the argument and
  dropped it; the existing regression test covered only the pre-trend writer.
- The manuscript bibliography. Nothing in the paper cited anything, and the
  build ran a single LaTeX pass with no bibtex step, so the reference list was
  empty and the build still exited zero. `make paper` now runs the full cycle
  and fails on unresolved citations, and the test suite checks citation keys
  and entry completeness independently of the build.
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

[Unreleased]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/releases/tag/v0.1.0
