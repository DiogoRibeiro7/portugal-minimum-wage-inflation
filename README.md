# Portugal Minimum Wage Inflation

[![CI](https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21864603.svg)](https://doi.org/10.5281/zenodo.21864603)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0.txt)

Reproducible research repository for studying the interaction between Portugal's statutory minimum wage, labour productivity, and consumer-price inflation from the introduction of the national minimum wage in 1974 to the present.

The repository is designed around two complementary empirical layers:

1. **Long-run macro accounting, 1974-present**: decompose minimum-wage growth into compensation for prior inflation, productivity growth, and a residual policy component.
2. **Price pass-through identification, post-2000**: exploit heterogeneous minimum-wage exposure across Portuguese regions and industries, together with regional minimum-wage differentiation in Madeira and the Azores where available, to estimate the effect of minimum-wage policy shocks on consumer prices.

The publication target is not a simple correlation between minimum-wage growth and inflation. The main contribution is intended to be a transparent exposure-based design linking statutory wage shocks to the parts of the price index most exposed to minimum-wage labour costs.

## Research questions

- Has the Portuguese minimum wage historically grown faster or slower than productivity plus inflation?
- Do minimum-wage increases above a productivity-and-inflation benchmark predict subsequent inflation acceleration?
- Are price responses larger in regions and sectors with a higher pre-existing minimum-wage bite?
- How much of an observed minimum-wage increase is absorbed by productivity, margins, employment, or consumer prices?
- Did pass-through change across macroeconomic regimes such as EU accession, euro adoption, the sovereign-debt crisis, the 2014 wage-policy restart, and the post-pandemic inflation episode?

## Core policy residual

For annual minimum wage `MW_t`, productivity `A_t`, and inflation `pi_t`, define the benchmark nominal wage growth as

```text
g_benchmark_t = (1 + g_productivity_t) * (1 + pi_{t-1}) - 1
```

and the policy residual as

```text
policy_residual_t = g_minimum_wage_t - g_benchmark_t
```

This residual is descriptive rather than automatically causal. The causal analysis uses predetermined exposure measures and panel variation.

## Identification strategy, and what it delivered

Two designs were pursued and both are reported as negative results. The README
describes what the repository does, not what was hoped for; the reasoning behind
each step is in `docs/research_design.md`.

**Regional policy variation.** Madeira legislates its own statutory wage, so its
change can differ from the mainland's. Dynamic responses are estimated with panel
local projections:

```text
Delta_h log(P[r,c,t+h]) = alpha[r,c] + lambda[t]
                          + beta_h * shock[r,t]
                          + error[r,c,t+h]
```

Ten region-months diverge over the window. Conventional clustered inference calls
five horizons significant; the wild cluster bootstrap leaves one, and Holm's
correction across the horizon family leaves none. Every horizon also carries a
confidence interval built by inverting the bootstrap test rather than from the
standard error, so the interval and the p-value cannot disagree.

**Shift-share exposure.** Regional industry composition from Eurostat's regional
accounts is combined with the national minimum-wage bite by activity:

```text
exposure[r] = sum_s employment_share[r, s] * minimum_wage_bite[s]
```

and interacted with the *national* statutory change. Exposure is predetermined:
the October 2015 survey round precedes every shock in the window. The measure
spans about two percentage points across nine regions, and not one horizon is
significant even by conventional clustered inference.

**Built and estimated.** A fuller structural exposure adds a labour-cost
share by industry and a production-to-consumption bridge:

```text
shock[r, c, t] = exposure[r, c] * national_minimum_wage_change[t]
exposure[r, c] = sum_s employment_share[r, s] * bite[s]
                 * labour_cost_share[s] * bridge[c, s]
```

An earlier version of this README argued that building these would not help,
because both new terms are national. That was wrong: with region-time and
category-time effects absorbed, what identifies the coefficient is the
non-additive part of the region-by-category matrix, which does not vanish.

Both terms now exist. `ptmw build consumption-bridge` composes the bridge from
Eurostat's use table, read at basic prices and domestic uses, and the
concordance recorded in `config/consumption_bridge.yaml`; `ptmw build
structural-exposure` and `ptmw analyse structural-design` build and estimate the
design. Its value is less a wider regressor than the fixed effects it can carry:
region-time effects absorb the tourism and island supply shocks the regional
comparison is otherwise exposed to, and category-time effects absorb the January
sales cycle that defeats the category design.

It is the only design here that rejects anything — two of seven horizons at five
per cent, none surviving Holm — and its magnitudes disqualify it. The
coefficient is on an exposure index rather than a cost share and is not
interpretable directly; scaled into points, the estimates imply differential
price responses larger than complete pass-through of the entire minimum-wage
cost bill could produce. Absorbing the confounds did not make the effect
identifiable, which is what establishes that the confounds were never the
constraint: policy is assigned to nine regions, and finer structure adds cells
rather than clusters. See `docs/decision_log.md`.

Diagnostics that are implemented: a joint pre-trend test over the leads,
wild-cluster and randomization inference for the small number of regions,
confidence intervals obtained by inverting the bootstrap test, Holm correction
across horizons, sixteen alternative constructions of the exposure measure,
minimum detectable effects, and a seasonal-confound diagnostic.

## Public data sources

The repository uses public or openly downloadable sources wherever possible:

- DGERT / Portuguese legislation: statutory national minimum-wage history.
- Regional official gazettes: Madeira and Azores regional minimum-wage schedules.
- GEP / MTSSS: minimum-wage coverage (`RMMG`) by economic activity and NUTS II region.
- INE: national and NUTS II CPI by consumption purpose; national accounts and regional controls.
- Eurostat: detailed monthly HICP/ECOICOP indices, HICP weights, constant-tax HICP, labour productivity and national-accounts controls; regional accounts employment by NUTS II region and activity (`nama_10r_3empers`); compensation of employees and value added by activity (`nama_10_a64`); and the use table at basic prices (`naio_10_cp1610`), read at domestic uses for household final consumption by product.
- AMECO / European Commission: long-run macroeconomic series where a consistent pre-Eurostat history is required.

Every downloaded file is retained under `data/raw/` with metadata and a checksum. Processed datasets are generated from raw inputs and never edited by hand.

## Repository layout

```text
.github/
  workflows/ci.yml       Lint, type check, tests, package build
  ISSUE_TEMPLATE/        Bug, data-correction, and feature templates
config/                  Source registry and analysis configuration
data/
  raw/                   Immutable downloaded source files
  interim/               Parsed but not analysis-ready data
  processed/             Analysis-ready datasets
  external/              Manually supplied public files when an API is unavailable
docs/
  research_design.md     Identification, estimands, assumptions, robustness
  decision_log.md        How positions were reached, including the wrong ones
  roadmap.md             What is left, and what is deliberately not
  data_dictionary.md     Variable definitions and provenance
  literature_map.md      Literature and novelty map
notebooks/               Exploration only; production results live in src/
report/
  main.tex               Reproducible article manuscript
  sections/              LaTeX manuscript sections
  figures/               Generated figures
  tables/                Generated LaTeX tables
src/pt_mw_inflation/
  data/                   Downloaders and source adapters
  processing/             Cleaning, harmonisation, exposure construction
  analysis/               Descriptive, panel, event-study, local-projection models
tests/                    Unit and data-contract tests
```

## Environment

Python 3.12 and Poetry are used for reproducibility.

```bash
poetry install
poetry run pre-commit install
```

Dependencies are pinned in `poetry.lock`, which is committed. Install from the
lock file rather than resolving fresh, so that a rerun years from now produces
the same numbers.

## Quality gates

```bash
make check       # everything below
make lint        # ruff check + ruff format --check
make typecheck   # mypy, strict, over src/
make test        # pytest
```

The same three gates run in CI on every push and pull request, against Python
3.12 and 3.13.

## Main commands

```bash
# Download every configured raw public source
poetry run ptmw data download-sources

# Regional consumer price panel from Statistics Portugal
poetry run ptmw data ine-cpi

# Regional and national employment by industry from Eurostat
poetry run ptmw data regional-employment

# Statutory minimum-wage panel, national and regional
poetry run ptmw build minimum-wage

# Long-run annual macro dataset
poetry run ptmw build macro

# Shift-share exposure, predetermined for shocks from 2016
poetry run ptmw build regional-exposure --bite-period 2015-10 --first-shock-year 2016

# Long-run figures, tables and headline macros
poetry run ptmw analyse macro

# Regional policy design, pre-trend test and seasonal diagnostic
poetry run ptmw analyse pass-through

# Shift-share exposure design
poetry run ptmw analyse exposure-design

# The exposure design under sixteen alternative constructions
poetry run ptmw analyse exposure-robustness
```

Everything above, plus the LaTeX build, runs as `make paper`. That target is
verified to work from an empty tree.

## Reproducibility rules

- Raw files are immutable.
- Every source has URL, provider, retrieval date, and SHA-256 hash.
- Transformations are deterministic and tested.
- Statistical specifications are configured in YAML rather than hidden in notebooks.
- Figures and tables are generated directly from analysis-ready datasets.
- The manuscript references generated table and figure files.
- No result is manually copied into the paper.

## Publication strategy

The claim rests on the long-run statutory and accounting layer, and on a
documented negative identification result. It does not rest on the exposure
analysis, which is reported precisely because it fails: the paper shows what the
available variation can and cannot support rather than presenting a weakly
identified estimate. See `docs/research_design.md`, `docs/decision_log.md` and
`docs/literature_map.md`.

## Contributing

Data corrections are the most valuable contribution here: a wrong minimum-wage
schedule or a misread gazette invalidates everything downstream. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and the reproducibility
rules a pull request must respect, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
for expected conduct. Security reports go through
[SECURITY.md](SECURITY.md), not public issues.

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md). Any change that
moves a published estimate is logged there with before/after values.

## Citation

If you use this repository, please cite it. GitHub renders `CITATION.cff` as a
"Cite this repository" button; the same metadata is machine-readable for
reference managers.

## Archiving and DOI

Each tagged release is archived on Zenodo, which mints a DOI so a specific state
of the analysis can be cited and retrieved even if this repository moves or
disappears. `.zenodo.json` supplies the deposit metadata — author, ORCID,
affiliation, licence, keywords — so the record does not depend on whatever
Zenodo infers from the repository.

Zenodo issues two kinds of DOI:

| DOI | Value | Cite it when |
| --- | ----- | ------------ |
| **Concept** | [`10.5281/zenodo.21864603`](https://doi.org/10.5281/zenodo.21864603) | The citation should track the current analysis. Always resolves to the latest version. |
| **Version** | [`10.5281/zenodo.21939240`](https://doi.org/10.5281/zenodo.21939240) (v0.3.0) | A result must be reproducible exactly. Pins one release. |

Which to cite is not a formatting preference here. v0.3.0 changed every bootstrap
p-value by up to 0.016, so a result quoted against v0.2.0
([`10.5281/zenodo.21864604`](https://doi.org/10.5281/zenodo.21864604)) does not
reproduce against v0.3.0. Anything reporting a specific number wants the version
DOI; anything referring to the analysis in general wants the concept DOI.

Archiving is already enabled: publishing a GitHub release is enough, and Zenodo
mints a new version DOI under the same concept DOI automatically. When cutting
a release, bump the version in `pyproject.toml`, `CITATION.cff` and
`.zenodo.json` together — `.zenodo.json` is what the deposit records, so a
stale value there mislabels the archive. The new version DOI exists only after
the release is published, so recording it in `CITATION.cff` and in the table
above is a step that necessarily follows the release rather than accompanying
it.

`CITATION.cff` and `.zenodo.json` are validated in CI, including a check that
they agree on title and version, so a release cannot be archived with
contradictory metadata.

## Licence

Dual-licensed:

- **Code** — `src/`, `tests/`, `notebooks/`, `config/`, `Makefile`, packaging
  and tooling: [MIT](LICENSE).
- **Written output** — `report/` and `docs/`, including the manuscript and
  generated figures and tables:
  [CC BY 4.0](LICENSE-CC-BY-4.0.txt).

Third-party data downloaded into `data/` is covered by neither licence and
remains subject to its original provider's terms. Per-source provenance and
licensing are recorded in `config/sources.yaml` and `docs/data_dictionary.md`.
