# Portugal Minimum Wage Inflation

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

## Planned identification strategy

The preferred price-pass-through design is based on an exposure shock of the form

```text
shock[r, c, t] = regional_minimum_wage_change[r, t]
                 * predetermined_minimum_wage_bite[r, s]
                 * labour_cost_share[s]
                 * production_to_consumption_weight[c, s]
```

where `r` is region, `s` is production industry, and `c` is a CPI/HICP consumption category.

Dynamic responses are estimated with panel local projections:

```text
Delta_h log(P[r,c,t+h]) = alpha[r,c] + lambda[t]
                          + beta_h * shock[r,c,t]
                          + Gamma X[r,c,t]
                          + error[r,c,t+h]
```

The design is accompanied by pre-trend diagnostics, alternative exposure definitions, wild-cluster or randomization-based inference for the small number of regions, and robustness to energy, VAT, tourism, imported inflation, and pandemic shocks.

## Public data sources

The repository uses public or openly downloadable sources wherever possible:

- DGERT / Portuguese legislation: statutory national minimum-wage history.
- Regional official gazettes: Madeira and Azores regional minimum-wage schedules.
- GEP / MTSSS: minimum-wage coverage (`RMMG`) by economic activity and NUTS II region.
- INE: national and NUTS II CPI by consumption purpose; national accounts and regional controls.
- Eurostat: detailed monthly HICP/ECOICOP indices, HICP weights, constant-tax HICP, labour productivity and national-accounts controls.
- AMECO / European Commission: long-run macroeconomic series where a consistent pre-Eurostat history is required.

Every downloaded file is retained under `data/raw/` with metadata and a checksum. Processed datasets are generated from raw inputs and never edited by hand.

## Repository layout

```text
config/                  Source registry and analysis configuration
data/
  raw/                   Immutable downloaded source files
  interim/               Parsed but not analysis-ready data
  processed/             Analysis-ready datasets
  external/              Manually supplied public files when an API is unavailable
docs/
  research_design.md     Identification, estimands, assumptions, robustness
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

## Main commands

```bash
# Download configured raw public sources
poetry run ptmw data download-sources

# Download detailed Portuguese HICP data from Eurostat
poetry run ptmw data eurostat-hicp --geo PT

# Build the long-run annual macro dataset
poetry run ptmw build macro

# Construct the minimum-wage policy residual
poetry run ptmw build policy-residual

# Run baseline descriptive analysis
poetry run ptmw analyse macro

# Run the panel/local-projection analysis after exposure data are available
poetry run ptmw analyse pass-through
```

## Reproducibility rules

- Raw files are immutable.
- Every source has URL, provider, retrieval date, and SHA-256 hash.
- Transformations are deterministic and tested.
- Statistical specifications are configured in YAML rather than hidden in notebooks.
- Figures and tables are generated directly from analysis-ready datasets.
- The manuscript references generated table and figure files.
- No result is manually copied into the paper.

## Publication strategy

The macro history is the context and validation layer. The publication claim should rest on the exposure-based price analysis, not on a raw national time-series correlation. See `docs/research_design.md` and `docs/literature_map.md`.
