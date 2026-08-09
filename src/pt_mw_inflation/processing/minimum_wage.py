"""Minimum-wage transformations and policy-shock definitions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from pt_mw_inflation.data.dgert import Scope, StatutoryChange

_REQUIRED_COLUMNS = {"year", "minimum_wage", "inflation", "productivity_growth"}

#: Portugal pays the monthly statutory wage fourteen times a year: twelve
#: months plus holiday and Christmas subsidies. Eurostat's published series
#: equals the monthly value times 14/12, which is the independent confirmation
#: that this figure, and the parsed monthly level, are both correct.
PAYMENTS_PER_YEAR = 14

#: Largest accepted gap between the percentage change DGERT prints and the one
#: implied by consecutive parsed values, in percentage points. The provider
#: rounds to one decimal, so a tenth of a point is normal.
_PERCENT_TOLERANCE = Decimal("0.15")


def find_unexplained_jumps(changes: Sequence[StatutoryChange]) -> list[tuple[Scope, date]]:
    """Locate regimes whose stated increase disagrees with the parsed history.

    The published page states, for each act, the percentage increase over the
    wage it replaced. When that percentage cannot be reproduced from the
    previous parsed value, an intermediate act is missing from the page: the
    stated figure refers to a regime that was never listed.

    This is a data-completeness check on the provider, not on the parser, and it
    is the mechanism by which a silently truncated upstream history is caught.

    Args:
        changes: Statutory changes as parsed, in any order.

    Returns:
        Scope and effective date of each regime whose stated increase cannot be
        reconciled with its predecessor.
    """
    unexplained: list[tuple[Scope, date]] = []

    by_scope: dict[Scope, list[StatutoryChange]] = {}
    for change in changes:
        by_scope.setdefault(change.scope, []).append(change)

    for scope, series in by_scope.items():
        ordered = sorted(series, key=lambda change: change.effective_date)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.stated_percent_change is None:
                continue
            if previous.original_currency == current.original_currency:
                implied = (current.original_amount / previous.original_amount - 1) * 100
            else:
                implied = (current.amount_eur / previous.amount_eur - 1) * 100
            if abs(implied - current.stated_percent_change) > _PERCENT_TOLERANCE:
                unexplained.append((scope, current.effective_date))

    return sorted(unexplained, key=lambda item: (item[1], item[0]))


def build_statutory_panel(
    changes: Iterable[StatutoryChange],
    *,
    geography: str = "PT",
    national_or_regional: str = "national",
    payments_per_year: int = PAYMENTS_PER_YEAR,
) -> pd.DataFrame:
    """Assemble the statutory minimum-wage panel from parsed legal acts.

    Args:
        changes: Parsed statutory changes.
        geography: Geography code the acts apply to.
        national_or_regional: Whether the acts are national or regional law.
        payments_per_year: Statutory payments per year used for annualisation.

    Returns:
        One row per geography, scope and effective date, with the wage in euro,
        the amount as originally published, and the legal citation.

    Raises:
        ValueError: If no changes were supplied.
    """
    ordered = sorted(changes, key=lambda change: (change.scope, change.effective_date))
    if not ordered:
        raise ValueError("no statutory changes supplied")

    unexplained = set(find_unexplained_jumps(ordered))

    records = []
    for change in ordered:
        monthly = float(change.amount_eur)
        note = ""
        if (change.scope, change.effective_date) in unexplained:
            note = (
                "Stated increase does not reconcile with the previous listed act; "
                "an intermediate regime is missing from the provider's page."
            )
        records.append(
            {
                "geography": geography,
                "effective_date": change.effective_date,
                "scope": change.scope,
                "minimum_wage_monthly_eur": monthly,
                "payments_per_year": payments_per_year,
                "annualised_minimum_wage_eur": monthly * payments_per_year,
                "original_amount": float(change.original_amount),
                "original_currency": change.original_currency,
                "legal_source": change.legal_source,
                "national_or_regional": national_or_regional,
                "notes": note,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    return frame.sort_values(["scope", "effective_date"]).reset_index(drop=True)


def annual_minimum_wage(
    panel: pd.DataFrame,
    *,
    scope: str = "general",
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Collapse the event-based panel into an annual series.

    Two annual conventions are produced, because they answer different
    questions and disagree whenever an act takes effect mid-year:

    ``minimum_wage_january``
        The level legally in force on 1 January. This is the convention used
        when a policy change is dated to the year it was announced for.
    ``minimum_wage_mean``
        The level averaged over the days of the year. This is the convention
        that matches annual national-accounts aggregates, and it is the one to
        use against annual inflation.

    Portugal changed the wage mid-year in 1974, 1975, 1978, 1979, 1980, 1981,
    1989 and 2014, so the choice is not cosmetic.

    Args:
        panel: Output of :func:`build_statutory_panel`.
        scope: Which legal regime to extract.
        start_year: First year of the returned series. Defaults to the year of
            the first act.
        end_year: Last year of the returned series. Defaults to the last year
            for which a full calendar year of law is known.

    Returns:
        Annual series with both conventions and the number of acts in the year.

    Raises:
        ValueError: If the requested scope is absent from the panel.
    """
    selected = panel.loc[panel["scope"] == scope].copy()
    if selected.empty:
        raise ValueError(f"scope {scope!r} not present in the panel")

    selected = selected.sort_values("effective_date")
    dates = pd.to_datetime(selected["effective_date"])
    levels = selected["minimum_wage_monthly_eur"].to_numpy(dtype=float)

    first_year = start_year if start_year is not None else int(dates.iloc[0].year)
    last_year = end_year if end_year is not None else int(dates.iloc[-1].year)

    # Daily step function of the statutory level, evaluated across the whole span.
    span: pd.DatetimeIndex = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
    year_values = np.asarray(span.year, dtype=int)
    positions = np.searchsorted(dates.to_numpy(), span.to_numpy(), side="right") - 1
    daily = pd.Series(
        np.where(positions >= 0, levels[np.clip(positions, 0, None)], np.nan),
        index=span,
    )

    grouped = daily.groupby(year_values)
    annual = pd.DataFrame(
        {
            "year": np.unique(year_values),
            "minimum_wage_january": grouped.first().to_numpy(),
            "minimum_wage_mean": grouped.mean().to_numpy(),
            # 1974 is a partial year: the wage was introduced on 27 May, so no
            # statutory level existed for the first five months. Downstream code
            # must be able to exclude partial years rather than treat them as
            # full-year observations.
            "coverage_fraction": grouped.apply(lambda days: days.notna().mean()).to_numpy(),
        }
    )
    acts_per_year = dates.dt.year.value_counts()
    annual["statutory_acts"] = annual["year"].map(acts_per_year).fillna(0).astype(int)
    annual["scope"] = scope
    return annual.dropna(subset=["minimum_wage_january"]).reset_index(drop=True)


def build_policy_residual(frame: pd.DataFrame, inflation_lag: int = 1) -> pd.DataFrame:
    """Compute minimum-wage growth and the productivity-plus-inflation residual.

    Args:
        frame: Annual data with `year`, `minimum_wage`, `inflation`, and
            `productivity_growth`. Rates are decimals, not percentage points.
        inflation_lag: Number of years by which CPI inflation enters the wage benchmark.

    Returns:
        Copy of the input with exact growth rates, benchmark growth and policy residual.

    Raises:
        ValueError: If required columns are absent or the minimum wage is non-positive.
    """
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (frame["minimum_wage"] <= 0).any():
        raise ValueError("minimum_wage must be strictly positive")

    result = frame.sort_values("year").copy()
    result["minimum_wage_growth"] = result["minimum_wage"].pct_change()
    result["lagged_inflation"] = result["inflation"].shift(inflation_lag)
    result["benchmark_wage_growth"] = (1.0 + result["productivity_growth"]) * (
        1.0 + result["lagged_inflation"]
    ) - 1.0
    result["policy_residual"] = result["minimum_wage_growth"] - result["benchmark_wage_growth"]
    result["log_minimum_wage_growth"] = np.log(result["minimum_wage"]).diff()
    return result
