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

## Macro annual dataset

Built by `ptmw build macro`. Requires `year`, `minimum_wage`, `inflation` and
`productivity_growth`, with rates as decimals.

| Column | Meaning |
| ------ | ------- |
| `minimum_wage_growth` | Year-on-year growth of the nominal minimum wage. |
| `lagged_inflation` | CPI inflation lagged by `benchmark_inflation_lag` years. |
| `benchmark_wage_growth` | `(1 + productivity_growth)(1 + lagged_inflation) − 1`. |
| `policy_residual` | `minimum_wage_growth − benchmark_wage_growth`. |
| `log_minimum_wage_growth` | First difference of the log nominal minimum wage. |

The residual is descriptive. It is not a causal estimate, and the causal work
uses predetermined exposure and panel variation instead.

## Exposure and price panels

Not yet populated. The exposure panel depends on the GEP coverage tables, which
the provider has withdrawn; see `config/sources.yaml` for the recorded
citation and the reason the source is disabled.
