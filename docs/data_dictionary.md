# Data dictionary

Every variable produced by the pipeline is defined here, with its provenance.
Nothing enters a table or figure without an entry.

## Conventions

- Rates are decimals, not percentage points, unless the name ends in `_pct`.
- Monetary values are euro unless the name ends in `_pte`.
- Escudo amounts convert at 200.482 PTE per euro, the irrevocable rate fixed by
  Council Regulation (EC) No 2866/98.
- Dates are ISO 8601. `effective_date` is the date a legal act took effect, not
  the date it was published.

## `data/raw/source_manifest.json`

One record per retrieval attempt, written by `ptmw data download-sources`.

| Field | Meaning |
| ----- | ------- |
| `source_name` | Registry key from `config/sources.yaml`. |
| `provider` | Institution publishing the file. |
| `url` | Canonical URL requested. |
| `destination` | Path the bytes were written to, relative to the repository. |
| `retrieved_at_utc` | ISO 8601 UTC timestamp of the retrieval. |
| `sha256` | Digest of the retrieved bytes. |
| `bytes` | Payload size. |
| `media_type` | Response media type, checked against the declared source kind. |
| `status` | `created`, `unchanged`, or `changed`. |
| `previous_sha256` | Digest of the superseded file, when `status` is `changed`. |
| `snapshot_path` | Dated copy of the superseded file, when `status` is `changed`. |

Raw files are immutable. When upstream content changes, the previous bytes are
retained under a timestamped name rather than overwritten, so a rerun can never
silently redefine the inputs behind a published estimate.

## `data/processed/minimum_wage_policy.parquet`

Statutory minimum-wage regimes, one row per geography, scope and effective
date. Built by `ptmw build minimum-wage` from the DGERT history.

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `geography` | str | Geography the act applies to. `PT` is mainland and national law. |
| `effective_date` | date | Date the act took effect. |
| `scope` | str | Legal regime: `general`, `agriculture`, or `domestic_service`. |
| `minimum_wage_monthly_eur` | float | Monthly statutory wage in euro. |
| `payments_per_year` | int | Statutory payments per year. 14 in Portugal. |
| `annualised_minimum_wage_eur` | float | `minimum_wage_monthly_eur × payments_per_year`. |
| `original_amount` | float | Amount as published, before currency conversion. |
| `original_currency` | str | `PTE` or `EUR`, as published. |
| `legal_source` | str | Citation of the act, e.g. `Decreto-Lei n.º 217/74 de 27 de maio`. |
| `national_or_regional` | str | `national` or `regional`. |
| `notes` | str | Empty unless the record needs a caveat. |

### Scope

Portugal did not have a single minimum wage until 2005. The published history
carries up to three legally distinct wages, and the number of columns on the
provider's page changes as regimes converge:

| Period | Regimes in force |
| ------ | ---------------- |
| 1974–1976 | General only. |
| 1977–1991 | General, agriculture, domestic service. |
| 1992–2004 | General, domestic service. Agriculture folded into the general regime. |
| 2005 onwards | A single unified wage. |

`general` corresponds to *Restantes Atividades* and is the series normally
quoted as the Portuguese minimum wage. **Downstream analysis uses `general`.**
The special regimes were always set at or below the general wage, which the
test suite asserts on every date where regimes coexist.

### Known gap in the upstream history

The DGERT page does not list the act that took effect in 2000, but the 2001
entries still state their increase relative to that missing regime. Affected
rows carry an explanatory `notes` value, and `find_unexplained_jumps` reports
them. Eurostat's independent series implies a 2000 general wage of about
318 EUR per month (63,800\$), which is consistent with the missing act, but no
value is imputed: an unsourced number will not be written into the panel.

### Cross-validation

Eurostat's `earn_mw_cur` is compiled independently from national law and is
used as a check rather than an input. Eurostat spreads Portugal's fourteen
statutory payments over twelve months, so its figure equals
`minimum_wage_monthly_eur × 14 / 12`. The test suite pins this correspondence
at five separate years, which validates both the parsed level and the
fourteen-payment convention.

## Annual series from `annual_minimum_wage`

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `year` | int | Calendar year. |
| `minimum_wage_january` | float | Level legally in force on 1 January. |
| `minimum_wage_mean` | float | Level averaged over the days of the year. |
| `statutory_acts` | int | Number of acts taking effect during the year. |
| `coverage_fraction` | float | Share of days in the year with a statutory wage in force. |
| `scope` | str | Regime the series was extracted for. |

The two conventions diverge only in years with a mid-year act: 1975, 1978,
1979, 1980, 1981, 1989 and 2014. Use `minimum_wage_mean` against annual
inflation and national-accounts aggregates; use `minimum_wage_january` when a
change should be dated to the year it was legislated for.

`coverage_fraction` is below 1 only in 1974, when the wage was introduced on
27 May. Exclude that year from full-year comparisons.

`statutory_acts` is 0 in years with no change, including 1976, 1982, 2000,
2012, 2013 and 2015. For 2012 and 2013 this reflects the wage freeze under the
adjustment programme; for 2000 it reflects the gap in the upstream page.

## `data/processed/macro_annual.parquet`

Built by `ptmw build macro` from the statutory panel, the World Bank consumer
price index and AMECO productivity. One row per year, 1974 onwards.

| Column | Meaning |
| ------ | ------- |
| `year` | Calendar year. |
| `minimum_wage` | Day-weighted statutory monthly wage, euro. |
| `cpi` | Consumer price index (World Bank, `FP.CPI.TOTL`). |
| `productivity` | Real GDP per person employed (AMECO `PRT.1.1.0.0.RVGDE`). |
| `inflation` | Growth of `cpi`. |
| `productivity_growth` | Growth of `productivity`. |
| `minimum_wage_growth` | Growth of the nominal minimum wage. |
| `lagged_inflation` | Inflation lagged by `benchmark_inflation_lag` years. |
| `benchmark_wage_growth` | `(1 + productivity_growth)(1 + lagged_inflation) − 1`. |
| `policy_residual` | `minimum_wage_growth − benchmark_wage_growth`. |
| `real_minimum_wage` | Nominal wage deflated to the first year's prices. |
| `minimum_wage_index` | Nominal wage, first year = 100. |
| `real_minimum_wage_index` | Real wage, first year = 100. |
| `productivity_index` | Productivity, first year = 100. |
| `minimum_wage_to_productivity_index` | Real wage per unit of productivity, first year = 100. |
| `log_minimum_wage_growth` | First difference of the log nominal wage. |
| `cumulative_policy_residual` | Running sum of the residual. |

The benchmark **compounds** rather than adds. At the inflation rates of the
early 1980s the additive approximation understates it by more than a
percentage point, which exceeds most of the residuals being measured.

`check_accounting_identities` verifies the residual, benchmark and real-wage
identities on every build, not only in tests.

The residual is descriptive. It is not a causal estimate, and the causal work
uses predetermined exposure and panel variation instead.

### Why these providers

| Series | Provider | Coverage | Why not the alternative |
| ------ | -------- | -------- | ----------------------- |
| Consumer prices | World Bank | 1960– | Eurostat HICP for Portugal begins in 1996. |
| Productivity | AMECO | 1960– | Eurostat national accounts begin in 1995 for Portugal. |

AMECO and Eurostat overlap for 31 years and agree to within 0.003 percentage
points of annual growth, so the longer series costs nothing in consistency.
AMECO publishes Commission projections past the last outturn; `last_actual_year`
in `config/analysis.yaml` excludes them. Raise it only when the year is an
outturn.

### Provenance of the wage series

`minimum_wage_source` on the annual series records which compiler each
observation came from. It is `DGERT statutory history` everywhere except 2000,
whose act is absent from the national page; that year is taken from Eurostat and
labelled accordingly. The correction fires only where the two compilers
materially disagree, so genuine freezes such as 2012 and 2013 are left
untouched.

## Exposure and price panels

Not yet populated. The exposure panel depends on the GEP coverage tables, which
the provider has withdrawn; see `config/sources.yaml` for the recorded
citation and the reason the source is disabled.

## Estimation output

Produced by `estimate_panel_local_projections`, written by
`ptmw analyse pass-through`.

| Column | Meaning |
| ------ | ------- |
| `horizon` | Horizon in months, from `config/analysis.yaml`. |
| `coefficient` | Cumulative log-price response to the exposure shock. |
| `standard_error` | Cluster-robust standard error, clustered on region. |
| `t_statistic` | Coefficient divided by the cluster-robust standard error. |
| `p_value_clustered` | Two-sided normal p-value. **Reported for comparison only.** |
| `p_value_bootstrap` | Restricted wild-cluster-bootstrap p-value. |
| `observations` | Rows entering the horizon regression. |
| `clusters` | Number of clusters, which is the binding constraint on inference. |

`p_value_clustered` is not a valid basis for a conclusion in this design.
Cluster-robust inference is asymptotic in the number of clusters, and with the
seven NUTS II regions the test over-rejects severely: in simulation, at a
nominal five per cent it rejects a true null about twenty per cent of the time,
while the bootstrap holds its size. Both are reported so the gap is visible;
**cite `p_value_bootstrap`.**

With seven clusters there are only `2**7 = 128` distinct Rademacher sign
vectors, so the bootstrap enumerates all of them and the p-value is exact
rather than simulated. The `exhaustive` flag on `ClusterInference` records
this.

## Event study

Produced by `estimate_event_study`.

| Column | Meaning |
| ------ | ------- |
| `event_time` | Periods relative to the shock. Negative values are leads. |
| `coefficient` | Effect on price growth at that event time. |
| `standard_error` | Cluster-robust standard error. Exactly 0 at the reference. |
| `t_statistic` | Ratio of the two. |

The reference lead is normalised to zero and retained in the output, so plots
cannot silently drop it. `assess_pre_trends` returns a verdict naming every
lead significant at `|t| > 1.96`.

Passing the pre-trend test does not establish parallel trends; it fails to
refute them, and an underpowered design passes trivially.
