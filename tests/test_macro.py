"""Tests for the long-run macro dataset and its inputs."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from pt_mw_inflation.data.ameco import parse_chapter, to_frame
from pt_mw_inflation.processing.macro import (
    MacroDatasetError,
    build_macro_annual,
    check_accounting_identities,
    summarise_by_regime,
)
from pt_mw_inflation.processing.minimum_wage import reconcile_annual_with_eurostat

AMECO_TEXT = (
    "CODE;COUNTRY;SUB-CHAPTER;TITLE;UNIT;1974;1975;1976;1977\n"
    "PRT.1.1.0.0.RVGDE;Portugal;06 Domestic product;GDP per person employed;1000 EURO-PTE;"
    "10.0;10.5;11.0;NA\n"
    "ESP.1.1.0.0.RVGDE;Spain;06 Domestic product;GDP per person employed;1000 EURO-PTE;"
    "12.0;12.5;13.0;13.5\n"
)


def _ameco_archive() -> bytes:
    """Build a minimal AMECO chapter archive in memory."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("AMECO6.TXT", AMECO_TEXT.encode("latin-1"))
    return buffer.getvalue()


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three coherent annual inputs with a known growth path."""
    years = list(range(1974, 1980))
    wages = pd.DataFrame(
        {"year": years, "minimum_wage_mean": [100.0 * 1.10**index for index in range(len(years))]}
    )
    prices = pd.DataFrame(
        {"year": years, "value": [50.0 * 1.05**index for index in range(len(years))]}
    )
    productivity = pd.DataFrame(
        {"year": years, "value": [80.0 * 1.02**index for index in range(len(years))]}
    )
    return wages, prices, productivity


def test_ameco_series_is_extracted_by_code() -> None:
    """The right country's series must be selected, not the first row."""
    series = parse_chapter(_ameco_archive(), "PRT.1.1.0.0.RVGDE")
    assert series.country == "Portugal"
    assert series.observations == {1974: 10.0, 1975: 10.5, 1976: 11.0}


def test_ameco_missing_values_are_dropped_not_zeroed() -> None:
    """An 'NA' cell must not become a zero observation."""
    series = parse_chapter(_ameco_archive(), "PRT.1.1.0.0.RVGDE")
    assert 1977 not in series.observations


def test_ameco_forecast_years_are_excluded() -> None:
    """Commission projections are not data and must not extend the series."""
    series = parse_chapter(_ameco_archive(), "PRT.1.1.0.0.RVGDE")
    frame = to_frame(series, last_actual_year=1975)
    assert frame["year"].max() == 1975
    assert len(frame) == 2


def test_ameco_unknown_code_is_rejected() -> None:
    """A renamed series must fail loudly rather than return nothing."""
    with pytest.raises(ValueError, match="not found"):
        parse_chapter(_ameco_archive(), "PRT.9.9.9.9.NOPE")


def test_macro_dataset_satisfies_its_identities() -> None:
    """The residual and benchmark identities must hold exactly."""
    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    check_accounting_identities(macro)


def test_benchmark_compounds_rather_than_adds() -> None:
    """Compounding matters at the inflation rates Portugal saw in the 1970s.

    With 30 per cent inflation and 5 per cent productivity growth the additive
    shortcut understates the benchmark by 1.5 points, which is larger than most
    of the residuals being measured.
    """
    years = [1980, 1981, 1982]
    wages = pd.DataFrame({"year": years, "minimum_wage_mean": [100.0, 130.0, 169.0]})
    prices = pd.DataFrame({"year": years, "value": [100.0, 130.0, 169.0]})
    productivity = pd.DataFrame({"year": years, "value": [100.0, 105.0, 110.25]})

    macro = build_macro_annual(wages, prices, productivity, start_year=1980).set_index("year")
    benchmark = float(macro.loc[1982, "benchmark_wage_growth"])

    assert benchmark == pytest.approx(1.05 * 1.30 - 1.0)
    assert benchmark > 0.05 + 0.30  # strictly above the additive approximation


def test_real_wage_is_deflated_to_the_base_year() -> None:
    """The real wage index must start at 100 and track nominal over prices."""
    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    assert macro["real_minimum_wage_index"].iloc[0] == pytest.approx(100.0)
    # Nominal grows 10 per cent, prices 5 per cent, so the real wage rises.
    assert macro["real_minimum_wage_index"].is_monotonic_increasing


def test_missing_columns_are_reported() -> None:
    """A misnamed input column must name the offending frame."""
    wages, prices, productivity = _inputs()
    with pytest.raises(MacroDatasetError, match="consumer_prices missing"):
        build_macro_annual(wages, prices.rename(columns={"value": "cpi"}), productivity)


def test_disjoint_inputs_are_rejected() -> None:
    """Inputs that barely overlap cannot form a growth rate."""
    wages, prices, productivity = _inputs()
    with pytest.raises(MacroDatasetError, match="at least two"):
        build_macro_annual(wages, prices, productivity.assign(year=productivity["year"] + 50))


def test_non_positive_prices_are_rejected() -> None:
    """A zero index would make the deflated series infinite."""
    wages, prices, productivity = _inputs()
    with pytest.raises(MacroDatasetError, match="strictly positive"):
        build_macro_annual(wages, prices.assign(value=0.0), productivity)


def test_regime_summary_counts_only_observed_years() -> None:
    """A regime extending beyond the data reports what was actually observed."""
    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    regimes = summarise_by_regime(
        macro,
        [
            {"name": "early", "start": 1974, "end": 1976},
            {"name": "late", "start": 1977, "end": 2100},
            {"name": "absent", "start": 1900, "end": 1950},
        ],
    )
    assert regimes["regime"].to_list() == ["early", "late"]
    assert int(regimes.loc[regimes["regime"] == "early", "years_observed"].iloc[0]) == 3


def test_malformed_regime_is_rejected() -> None:
    """A regime without bounds is a configuration error."""
    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    with pytest.raises(MacroDatasetError, match="malformed regime"):
        summarise_by_regime(macro, [{"name": "broken"}])


def test_identity_violation_is_detected() -> None:
    """Tampering with the residual must be caught, not carried into the paper."""
    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    macro.loc[3, "policy_residual"] = 0.42
    with pytest.raises(MacroDatasetError, match="residual identity"):
        check_accounting_identities(macro)


def test_reconciliation_fills_only_genuinely_missing_acts() -> None:
    """A year with no act is corrected only when an independent source disagrees.

    Portugal genuinely froze the wage in 2012 and 2013, so those years must be
    left alone; a year the national page omits must not be.
    """
    annual = pd.DataFrame(
        {
            "year": [2011, 2012, 2013, 2014],
            "minimum_wage_january": [485.0, 485.0, 485.0, 485.0],
            "minimum_wage_mean": [485.0, 485.0, 485.0, 485.0],
            "statutory_acts": [1, 0, 0, 0],
        }
    )
    eurostat = pd.DataFrame(
        {
            "year": [2011, 2012, 2013, 2014],
            # 2014 is materially higher: an act is missing from the national page.
            "implied_monthly_statutory_eur": [485.0, 485.0, 485.0, 560.0],
        }
    )
    result = reconcile_annual_with_eurostat(annual, eurostat).set_index("year")

    assert result.loc[2012, "minimum_wage_source"] == "DGERT statutory history"
    assert result.loc[2013, "minimum_wage_january"] == 485.0
    assert result.loc[2014, "minimum_wage_january"] == 560.0
    assert "Eurostat" in result.loc[2014, "minimum_wage_source"]


def test_reconciliation_never_overrides_a_year_with_an_act() -> None:
    """A year whose level comes from the legal text is authoritative."""
    annual = pd.DataFrame(
        {
            "year": [2020],
            "minimum_wage_january": [635.0],
            "minimum_wage_mean": [635.0],
            "statutory_acts": [1],
        }
    )
    eurostat = pd.DataFrame({"year": [2020], "implied_monthly_statutory_eur": [900.0]})
    result = reconcile_annual_with_eurostat(annual, eurostat)
    assert float(result["minimum_wage_january"].iloc[0]) == 635.0


def test_reconciliation_requires_the_reference_column() -> None:
    """A frame without the Eurostat level cannot be reconciled against."""
    annual = pd.DataFrame(
        {
            "year": [2000],
            "minimum_wage_january": [300.0],
            "minimum_wage_mean": [300.0],
            "statutory_acts": [0],
        }
    )
    with pytest.raises(ValueError, match="missing columns"):
        reconcile_annual_with_eurostat(annual, pd.DataFrame({"year": [2000]}))


def test_outputs_are_written(tmp_path: Path) -> None:
    """Every figure and table must be produced and be non-empty."""
    from pt_mw_inflation.analysis.outputs import generate_macro_outputs

    wages, prices, productivity = _inputs()
    macro = build_macro_annual(wages, prices, productivity, start_year=1974)
    regimes = summarise_by_regime(macro, [{"name": "all", "start": 1974, "end": 1979}])

    written = generate_macro_outputs(
        macro, regimes, figures_dir=tmp_path / "figures", tables_dir=tmp_path / "tables"
    )
    assert len(written) == 5
    for path in written:
        assert path.exists() and path.stat().st_size > 0

    macros = (tmp_path / "tables" / "headline_macros.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\MacroFirstYear}{1974}" in macros
    assert "Do not edit" in macros
