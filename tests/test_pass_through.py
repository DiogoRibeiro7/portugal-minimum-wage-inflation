"""Tests for the two pass-through estimation panels."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pt_mw_inflation.processing.pass_through import (
    PassThroughError,
    add_category_interactions,
    build_estimation_panel,
    build_regional_shock,
    count_identifying_events,
    diagnose_seasonal_confound,
    monthly_statutory_wage,
)

MONTHS = pd.date_range("2024-01-01", "2026-06-01", freq="MS")


def _wage_panel() -> pd.DataFrame:
    """National acts plus a region that legislates its own value."""
    return pd.DataFrame(
        [
            {
                "geography": "PT",
                "effective_date": pd.Timestamp("2024-01-01"),
                "minimum_wage_monthly_eur": 820.0,
            },
            {
                "geography": "PT",
                "effective_date": pd.Timestamp("2025-01-01"),
                "minimum_wage_monthly_eur": 870.0,
            },
            {
                "geography": "PT",
                "effective_date": pd.Timestamp("2026-01-01"),
                "minimum_wage_monthly_eur": 920.0,
            },
            {
                "geography": "PT30",
                "effective_date": pd.Timestamp("2025-01-01"),
                "minimum_wage_monthly_eur": 915.0,
            },
            {
                "geography": "PT30",
                "effective_date": pd.Timestamp("2026-01-01"),
                "minimum_wage_monthly_eur": 980.0,
            },
            # A proportional supplement: always 5 per cent above the national.
            {
                "geography": "PT20",
                "effective_date": pd.Timestamp("2024-01-01"),
                "minimum_wage_monthly_eur": 861.0,
            },
            {
                "geography": "PT20",
                "effective_date": pd.Timestamp("2025-01-01"),
                "minimum_wage_monthly_eur": 913.5,
            },
            {
                "geography": "PT20",
                "effective_date": pd.Timestamp("2026-01-01"),
                "minimum_wage_monthly_eur": 966.0,
            },
        ]
    )


def _prices() -> pd.DataFrame:
    """A small price panel over three regions and two categories."""
    rows = []
    for region in ("PT11", "PT20", "PT30", "PT"):
        for category in ("01", "11", "T"):
            for index, month in enumerate(MONTHS):
                rows.append(
                    {
                        "month": month,
                        "nuts_code": region,
                        "category_code": category,
                        "price_index": 100.0 + index * 0.2,
                    }
                )
    return pd.DataFrame(rows)


def test_wage_steps_forward_until_the_next_act() -> None:
    """A level holds until an act replaces it."""
    series = monthly_statutory_wage(_wage_panel(), MONTHS, geography="PT")
    assert series.loc[pd.Timestamp("2024-06-01")] == 820.0
    assert series.loc[pd.Timestamp("2025-06-01")] == 870.0


def test_region_falls_back_to_the_national_wage_before_it_legislates() -> None:
    """Madeira is governed by the national wage until its own act takes effect.

    Without the fallback the region would have no wage at all in those months
    and would drop out of the panel, which would silently change the sample
    rather than the shock.
    """
    national = monthly_statutory_wage(_wage_panel(), MONTHS, geography="PT")
    series = monthly_statutory_wage(_wage_panel(), MONTHS, geography="PT30", fallback=national)
    assert series.loc[pd.Timestamp("2024-06-01")] == 820.0  # national
    assert series.loc[pd.Timestamp("2025-06-01")] == 915.0  # its own act


def test_proportional_supplement_contributes_no_independent_variation() -> None:
    """A fixed percentage supplement has the same log change as the national wage.

    This is the finding that governs how much the regional design can carry: a
    region on a permanent supplement adds rows to the panel and not one degree
    of freedom, so counting its observations would overstate the evidence.
    """
    shock = build_regional_shock(_wage_panel(), MONTHS, ["PT11", "PT20", "PT30"])
    azores = shock.loc[shock["nuts_code"] == "PT20"].set_index("month")
    mainland = shock.loc[shock["nuts_code"] == "PT11"].set_index("month")

    difference = (azores["delta_log_minimum_wage"] - mainland["delta_log_minimum_wage"]).abs()
    assert float(difference.max()) < 1e-12


def test_identifying_events_counts_only_genuine_divergence() -> None:
    """Only the region legislating its own value diverges, and only when it acts."""
    shock = build_regional_shock(_wage_panel(), MONTHS, ["PT11", "PT20", "PT30"])
    variation = count_identifying_events(shock, national="PT11")

    assert variation.regions == ("PT30",)
    assert variation.months == ("2025-01", "2026-01")
    assert variation.region_months == 2


def test_panel_excludes_aggregates_and_the_all_items_index() -> None:
    """Portugal contains the regions, and the total contains the categories."""
    panel = build_estimation_panel(_prices(), _wage_panel(), start="2024-01")
    assert "PT" not in set(panel["nuts_code"])
    assert "T" not in set(panel["category_code"])
    assert set(panel["nuts_code"]) == {"PT11", "PT20", "PT30"}


def test_panel_carries_log_price_and_the_region_specific_shock() -> None:
    """Each row must have the outcome and the wage change that applies to it."""
    panel = build_estimation_panel(_prices(), _wage_panel(), start="2024-01")
    assert np.allclose(panel["log_price"], np.log(panel["price_index"]))

    january = panel.loc[panel["month"] == pd.Timestamp("2026-01-01")]
    madeira = january.loc[january["nuts_code"] == "PT30", "delta_log_minimum_wage"].iloc[0]
    mainland = january.loc[january["nuts_code"] == "PT11", "delta_log_minimum_wage"].iloc[0]
    # Madeira went 915 -> 980; the mainland 870 -> 920.
    assert madeira == pytest.approx(np.log(980 / 915))
    assert mainland == pytest.approx(np.log(920 / 870))
    assert madeira > mainland


def test_category_interactions_cover_every_category_exactly_once() -> None:
    """One interaction per category, and each row loads on only its own."""
    panel = build_estimation_panel(_prices(), _wage_panel(), start="2024-01")
    interacted, names = add_category_interactions(panel)

    assert len(names) == panel["category_code"].nunique()

    # The first month of each series has no predecessor, so its change is NaN
    # rather than zero and its interactions are NaN too. Estimation drops those
    # rows, so the loading check is made on the sample that is actually used.
    estimable = interacted.dropna(subset=["delta_log_minimum_wage"])
    loadings = estimable[names].ne(0).sum(axis=1)
    # A row loads on at most its own category, and on none at all in a month
    # when the wage did not change.
    assert loadings.max() <= 1
    assert len(estimable) < len(interacted)


def test_missing_price_columns_are_reported() -> None:
    """A renamed column must fail loudly rather than estimate on a subset."""
    with pytest.raises(PassThroughError, match="prices missing columns"):
        build_estimation_panel(_prices().drop(columns=["price_index"]), _wage_panel())


def test_empty_window_is_rejected() -> None:
    """A start date beyond the data is a configuration error."""
    with pytest.raises(PassThroughError, match="no price observations"):
        build_estimation_panel(_prices(), _wage_panel(), start="2100-01")


def test_the_national_wage_is_a_floor_under_a_stale_regional_act() -> None:
    """A region overtaken by the national wage takes the national wage.

    Madeira legislates intermittently and its acts state a euro figure, so a
    value can stand while the national wage rises past it: the 2017 act's 570
    was still the region's own last figure in 2018, by which time the national
    wage was 580. Carrying the regional level unconditionally would report a
    wage no employer could lawfully pay, and would record the region as
    diverging downwards in a year its premium had simply been extinguished.
    """
    months = pd.date_range("2017-01-01", "2018-12-01", freq="MS")
    panel = pd.DataFrame(
        [
            {
                "geography": "PT30",
                "effective_date": date(2017, 1, 1),
                "minimum_wage_monthly_eur": 570.0,
            }
        ]
    )
    national = pd.Series([557.0 if month.year == 2017 else 580.0 for month in months], index=months)

    stepped = monthly_statutory_wage(panel, months, geography="PT30", fallback=national)

    assert stepped.loc["2017-06-01"] == pytest.approx(570.0)
    # Overtaken: the standing regional figure is below the national floor.
    assert stepped.loc["2018-06-01"] == pytest.approx(580.0)
    assert (stepped >= national).all()


def test_a_regional_premium_above_the_floor_is_preserved() -> None:
    """The floor must not flatten a region that genuinely pays more."""
    months = pd.date_range("2019-01-01", "2019-06-01", freq="MS")
    panel = pd.DataFrame(
        [
            {
                "geography": "PT30",
                "effective_date": date(2019, 1, 1),
                "minimum_wage_monthly_eur": 615.0,
            }
        ]
    )
    national = pd.Series(600.0, index=months)

    stepped = monthly_statutory_wage(panel, months, geography="PT30", fallback=national)
    assert (stepped == 615.0).all()


def _seasonal_panel() -> tuple[pd.DataFrame, pd.Series]:
    """A shock that only ever moves in January, and a category that crashes then.

    This is the Portuguese case in miniature: the statutory wage steps every
    1 January, and clothing collapses in the same month because winter sales
    enter the index. Nothing about the price series is wrong.
    """
    months = pd.date_range("2015-01-01", "2019-12-01", freq="MS")
    rows = []
    for code, january_swing in (("03", -0.16), ("01", 0.01)):
        level = 100.0
        for month in months:
            level *= 1.0 + (january_swing if month.month == 1 else 0.02)
            rows.append({"category_code": code, "month": month, "price_index": level})
    prices = pd.DataFrame(rows)

    wage = pd.Series([600.0 + 20.0 * (month.year - 2015) for month in months], index=months)
    return prices, np.log(wage)


def test_a_january_only_shock_is_reported_as_confounded_with_january() -> None:
    """A policy that always moves in one month is collinear with that month."""
    prices, shock = _seasonal_panel()
    confound = diagnose_seasonal_confound(prices, shock)

    assert confound.modal_month == 1
    assert confound.modal_share == pytest.approx(1.0)


def test_the_confound_names_the_category_it_distorts_most() -> None:
    """The diagnosis has to point at the series, not merely report a number.

    Clothing swings sixteen times as far as food in January, which is why its
    coefficient is the one that looks broken.
    """
    prices, shock = _seasonal_panel()
    confound = diagnose_seasonal_confound(prices, shock)

    assert confound.worst_category == "03"
    assert confound.worst_category_swing < -10.0


def test_month_effects_leave_nothing_when_the_shock_is_purely_seasonal() -> None:
    """Correcting the artefact costs the variation that identified it.

    A shock occurring in the same month every year, at the same size, is
    perfectly explained by a month-of-year indicator, so nothing survives. That
    is the design failing, not the correction working.
    """
    months = pd.date_range("2015-01-01", "2019-12-01", freq="MS")
    prices = pd.DataFrame(
        [{"category_code": "01", "month": month, "price_index": 100.0} for month in months]
    )
    constant_step = pd.Series(
        [np.log(600.0 * 1.05 ** (month.year - 2015)) for month in months], index=months
    )
    confound = diagnose_seasonal_confound(prices, constant_step)
    assert confound.surviving_variance_share == pytest.approx(0.0, abs=1e-9)


def test_a_shock_spread_across_months_is_not_confounded() -> None:
    """The diagnostic must not condemn a policy that moves at varied dates."""
    months = pd.date_range("2015-01-01", "2019-12-01", freq="MS")
    prices = pd.DataFrame(
        [{"category_code": "01", "month": month, "price_index": 100.0} for month in months]
    )
    # Steps at irregular intervals, so they fall in different calendar months
    # and no single month can explain them.
    steps = {3, 9, 16, 26, 31, 40, 44, 53}
    levels, level = [], 600.0
    for index in range(len(months)):
        if index in steps:
            level *= 1.03
        levels.append(level)
    confound = diagnose_seasonal_confound(prices, pd.Series(np.log(levels), index=months))

    assert confound.modal_share < 0.5
    assert confound.surviving_variance_share > 0.3


def test_a_shock_that_never_moves_is_refused() -> None:
    """A constant wage carries no variation to diagnose."""
    months = pd.date_range("2015-01-01", "2015-12-01", freq="MS")
    prices = pd.DataFrame(
        [{"category_code": "01", "month": month, "price_index": 100.0} for month in months]
    )
    with pytest.raises(PassThroughError, match="never changes"):
        diagnose_seasonal_confound(prices, pd.Series(np.log(600.0), index=months))
