"""Tests for the statutory minimum-wage panel and its annual collapse."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pt_mw_inflation.data.dgert import parse_minimum_wage_history
from pt_mw_inflation.processing.minimum_wage import (
    PAYMENTS_PER_YEAR,
    annual_minimum_wage,
    build_statutory_panel,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dgert_minimum_wage_history.html"

#: Eurostat's bi-annual statutory minimum wage for Portugal, in euro, as
#: published (retrieved 2026-08-09). Eurostat spreads the fourteen statutory
#: payments over twelve months, so these equal the monthly level times 14/12.
#: Eurostat compiles this from its own reading of national law, which makes it
#: an independent check on the parsed legal history rather than a restatement
#: of it.
EUROSTAT_PUBLISHED = {2003: 416, 2019: 700, 2024: 957, 2025: 1015, 2026: 1073}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Build the panel once from the captured page."""
    return build_statutory_panel(parse_minimum_wage_history(FIXTURE.read_text(encoding="utf-8")))


def test_panel_has_the_required_columns(panel: pd.DataFrame) -> None:
    """The processed contract must carry provenance alongside the values."""
    required = {
        "geography",
        "effective_date",
        "minimum_wage_monthly_eur",
        "payments_per_year",
        "annualised_minimum_wage_eur",
        "legal_source",
        "national_or_regional",
        "notes",
    }
    assert required.issubset(panel.columns)
    assert (panel["minimum_wage_monthly_eur"] > 0).all()
    assert panel["national_or_regional"].eq("national").all()


def test_annualisation_identity_holds(panel: pd.DataFrame) -> None:
    """Annualised pay must equal the monthly level times the payment count."""
    expected = panel["minimum_wage_monthly_eur"] * panel["payments_per_year"]
    pd.testing.assert_series_equal(
        panel["annualised_minimum_wage_eur"], expected, check_names=False
    )
    assert panel["payments_per_year"].eq(PAYMENTS_PER_YEAR).all()


def test_incomplete_upstream_history_is_flagged_in_notes(panel: pd.DataFrame) -> None:
    """The regime whose stated increase cannot be reconciled carries a note."""
    flagged = panel.loc[panel["notes"].str.contains("missing", case=False, na=False)]
    assert set(pd.to_datetime(flagged["effective_date"]).dt.year) == {2001}
    unflagged = panel.loc[~panel.index.isin(flagged.index)]
    assert unflagged["notes"].eq("").all()


def test_annual_series_spans_the_whole_history(panel: pd.DataFrame) -> None:
    """The annual general series must run from 1974 with no interior gaps."""
    annual = annual_minimum_wage(panel, scope="general")
    years = annual["year"].to_list()
    assert years[0] == 1974
    assert years == list(range(years[0], years[-1] + 1))
    assert annual["minimum_wage_january"].is_monotonic_increasing


def test_first_year_is_marked_as_partial(panel: pd.DataFrame) -> None:
    """1974 was not a full year of statutory coverage and must say so."""
    annual = annual_minimum_wage(panel, scope="general").set_index("year")
    assert annual.loc[1974, "coverage_fraction"] < 0.7
    assert (annual.loc[1975:, "coverage_fraction"] == 1.0).all()


def test_mid_year_acts_separate_the_two_annual_conventions(panel: pd.DataFrame) -> None:
    """January and day-weighted levels must differ exactly in mid-year years.

    Portuguese acts took effect mid-year in 1975, 1978, 1979, 1980, 1981, 1989
    and 2014. Anywhere else the two conventions must agree, which is what makes
    the day-weighted average trustworthy.
    """
    annual = annual_minimum_wage(panel, scope="general")
    differing = annual.loc[
        (annual["minimum_wage_january"] - annual["minimum_wage_mean"]).abs() > 1e-9, "year"
    ]
    assert set(differing) == {1975, 1978, 1979, 1980, 1981, 1989, 2014}


def test_two_acts_in_1989_are_both_retained(panel: pd.DataFrame) -> None:
    """1989 had increases in January and July; neither may be dropped."""
    annual = annual_minimum_wage(panel, scope="general").set_index("year")
    assert annual.loc[1989, "statutory_acts"] == 2
    assert annual.loc[1989, "minimum_wage_mean"] > annual.loc[1989, "minimum_wage_january"]


def test_frozen_wage_years_carry_no_act(panel: pd.DataFrame) -> None:
    """2012 and 2013 saw no statutory increase and must forward-fill 2011."""
    annual = annual_minimum_wage(panel, scope="general").set_index("year")
    assert annual.loc[[2012, 2013], "statutory_acts"].eq(0).all()
    assert annual.loc[2012, "minimum_wage_january"] == annual.loc[2011, "minimum_wage_january"]


@pytest.mark.parametrize(("year", "eurostat_value"), sorted(EUROSTAT_PUBLISHED.items()))
def test_levels_agree_with_the_independent_eurostat_series(
    panel: pd.DataFrame, year: int, eurostat_value: int
) -> None:
    """The parsed legal history must reproduce Eurostat's published figures.

    Eurostat annualises over twelve months while Portuguese law states a
    monthly wage paid fourteen times, so the comparison has to undo that
    convention. Agreement to within a euro confirms both the parsed level and
    the fourteen-payment assumption used for annualisation.
    """
    annual = annual_minimum_wage(panel, scope="general").set_index("year")
    implied = annual.loc[year, "minimum_wage_january"] * PAYMENTS_PER_YEAR / 12
    assert abs(implied - eurostat_value) < 1.0


def test_unknown_scope_is_rejected(panel: pd.DataFrame) -> None:
    """Asking for a regime that does not exist is an error, not an empty frame."""
    with pytest.raises(ValueError, match="not present"):
        annual_minimum_wage(panel, scope="fishing")


def test_empty_input_is_rejected() -> None:
    """Building a panel from nothing is a programming error."""
    with pytest.raises(ValueError, match="no statutory changes"):
        build_statutory_panel([])


def test_effective_dates_are_unique_within_a_scope(panel: pd.DataFrame) -> None:
    """One regime cannot have two different wages on the same date."""
    duplicated = panel.duplicated(subset=["geography", "scope", "effective_date"])
    assert not duplicated.any()
    assert pd.to_datetime(panel["effective_date"]).min() == pd.Timestamp(date(1974, 5, 27))
