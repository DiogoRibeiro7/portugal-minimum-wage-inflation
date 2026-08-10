"""Estimation panels for the two pass-through designs the data supports.

Both designs use the same region-by-category-by-month price panel and differ in
where their identifying variation comes from.

**Regional policy variation.** The autonomous regions may legislate a minimum
wage above the mainland's, so the statutory change itself differs across
regions in the months they do. Calendar-time fixed effects are admissible here,
because the shock is not collinear with them.

The variation is narrow and its width should be measured, not assumed. The
Azores fix a permanent proportional supplement, so after the month it was
introduced their log change equals the national one exactly and they contribute
no independent timing at all. Madeira legislates a value, so it contributes one
month per act. :func:`count_identifying_events` reports how many region-months
actually diverge, and it is the number a reader should judge the design by.

**Category-differential response.** The mainland faces a single national
change, so the only variation left is across consumption categories. Rather
than impose an exposure measure — which would require a region-by-industry bite
that no Portuguese source publishes — the differential response is estimated
directly, one coefficient per category.

That design cannot carry calendar-time fixed effects: the shock is constant
across regions within a month, so time effects would absorb it entirely. What
it identifies is therefore the response of each category relative to the
others, and it is exposed to anything else moving national prices at the same
time. It is reported as a ranking to be compared against known coverage by
activity, not as a level of pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Regions whose prices are published but which are aggregates of the others.
_AGGREGATE_CODES = frozenset({"PT", "PT1"})

#: The all-items index, excluded when estimating by category.
TOTAL_CATEGORY = "T"


class PassThroughError(ValueError):
    """Raised when an estimation panel cannot be assembled as configured."""


@dataclass(frozen=True)
class VariationCount:
    """How much independent regional variation a shock series carries."""

    region_months: int
    regions: tuple[str, ...]
    months: tuple[str, ...]


def monthly_statutory_wage(
    panel: pd.DataFrame,
    months: pd.DatetimeIndex,
    *,
    geography: str,
    fallback: pd.Series | None = None,
    gap_years: frozenset[int] | None = None,
) -> pd.Series:
    """Step the statutory wage for one geography onto a monthly index.

    Args:
        panel: Statutory panel, general regime.
        months: Months to evaluate.
        geography: Geography to extract.
        fallback: Series used where the geography has no act of its own in
            force. This applies to interior gaps as well as to months before the
            first act, and the distinction matters: a region that legislated in
            a year whose act is not registered would otherwise be held at its
            last known level while the national wage rose, which is
            indistinguishable from a deliberate freeze and enters the estimation
            as a policy shock that never happened.
        gap_years: Years in which this geography is known to have legislated
            without the act being registered. Those months take the fallback
            rather than the carried-forward level.

    Returns:
        The wage in force in each month.

    Raises:
        PassThroughError: If neither the geography nor a fallback covers a month.
    """
    selected = panel.loc[panel["geography"] == geography].sort_values("effective_date")
    if selected.empty and fallback is None:
        raise PassThroughError(f"no acts for {geography} and no fallback supplied")

    if selected.empty:
        if fallback is None:
            raise PassThroughError(f"no acts for {geography} and no fallback supplied")
        stepped = pd.Series(fallback, index=months).reindex(months)
        if stepped.isna().any():
            raise PassThroughError(f"fallback does not cover every month for {geography}")
        return stepped

    dates = pd.to_datetime(selected["effective_date"]).to_numpy()
    levels = selected["minimum_wage_monthly_eur"].to_numpy(dtype=float)
    positions = np.searchsorted(dates, months.to_numpy(), side="right") - 1
    stepped = pd.Series(
        np.where(positions >= 0, levels[np.clip(positions, 0, None)], np.nan), index=months
    )

    if gap_years:
        # A registered act cannot speak for a year it does not cover. Rather
        # than carry the previous level across a known hole, defer to the
        # fallback so the hole cannot masquerade as a frozen wage.
        in_gap = pd.Series(months.year, index=months).isin(gap_years)
        stepped = stepped.mask(in_gap)

    if fallback is not None:
        stepped = stepped.fillna(pd.Series(fallback, index=months))
    if stepped.isna().any():
        raise PassThroughError(f"{geography} has no wage in force for some months")
    return stepped


def build_regional_shock(
    wage_panel: pd.DataFrame,
    months: pd.DatetimeIndex,
    regions: list[str],
    *,
    national: str = "PT",
    gap_years: dict[str, frozenset[int]] | None = None,
) -> pd.DataFrame:
    """Build the applicable statutory wage and its log change for each region.

    Args:
        wage_panel: Statutory panel filtered to the general regime.
        months: Months to cover.
        regions: NUTS codes to build the shock for.
        national: Geography supplying the mainland wage and the fallback.
        gap_years: Per-region years known to be missing from the register,
            which take the national wage rather than a carried-forward level.

    Returns:
        Columns `nuts_code`, `month`, `minimum_wage`, `delta_log_minimum_wage`.

    Raises:
        PassThroughError: If the national series cannot be built.
    """
    national_wage = monthly_statutory_wage(wage_panel, months, geography=national)

    blocks = []
    for region in regions:
        has_own = (wage_panel["geography"] == region).any()
        wage = (
            monthly_statutory_wage(
                wage_panel,
                months,
                geography=region,
                fallback=national_wage,
                gap_years=(gap_years or {}).get(region),
            )
            if has_own
            else national_wage
        )
        block = pd.DataFrame(
            {
                "nuts_code": region,
                "month": months,
                "minimum_wage": wage.to_numpy(dtype=float),
            }
        )
        # The first month has no predecessor, so its change is undefined rather
        # than zero. Filling it with zero would silently drop an act effective
        # in exactly that month, which is where a region's divergence lives.
        block["delta_log_minimum_wage"] = np.log(block["minimum_wage"]).diff()
        block.loc[block.index[0], "delta_log_minimum_wage"] = np.nan
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def count_identifying_events(shock: pd.DataFrame, *, national: str = "PT") -> VariationCount:
    """Count the region-months whose statutory change differs from the national one.

    This is the honest measure of what the regional design rests on. A region
    with a permanent proportional supplement has the same log change as the
    mainland in every month after it was introduced, so it inflates the sample
    without adding a single degree of freedom.

    Args:
        shock: Output of :func:`build_regional_shock`.
        national: Region code whose change is the reference.

    Returns:
        The number of diverging region-months, and which they are.

    Raises:
        PassThroughError: If the reference region is absent.
    """
    if shock.empty:
        raise PassThroughError("shock frame is empty; nothing to compare")

    reference = shock.loc[shock["nuts_code"] == national]
    if reference.empty:
        # The mainland regions all carry the national change, so any of them
        # serves as the reference when the national code itself is not present.
        reference = shock.loc[shock["nuts_code"] == shock["nuts_code"].iloc[0]]
        if reference.empty:
            raise PassThroughError("no reference region available")

    baseline = reference.set_index("month")["delta_log_minimum_wage"]
    merged = shock.copy()
    merged["baseline"] = merged["month"].map(baseline)
    diverging = merged.loc[(merged["delta_log_minimum_wage"] - merged["baseline"]).abs() > 1e-9]

    return VariationCount(
        region_months=int(len(diverging)),
        regions=tuple(sorted(diverging["nuts_code"].unique())),
        months=tuple(sorted(month.strftime("%Y-%m") for month in diverging["month"].unique())),
    )


def build_estimation_panel(
    prices: pd.DataFrame,
    wage_panel: pd.DataFrame,
    *,
    start: str = "2000-01",
    include_total: bool = False,
) -> pd.DataFrame:
    """Join regional prices to the applicable statutory wage.

    Args:
        prices: Regional price panel.
        wage_panel: Statutory panel, general regime.
        start: First month to retain.
        include_total: Keep the all-items index alongside the categories.

    Returns:
        One row per region, category and month, with `log_price`, the wage in
        force and its log change.

    Raises:
        PassThroughError: If required columns are missing or nothing survives
            the filters.
    """
    required = {"month", "nuts_code", "category_code", "price_index"}
    missing = required.difference(prices.columns)
    if missing:
        raise PassThroughError(f"prices missing columns: {sorted(missing)}")

    frame = prices.loc[~prices["nuts_code"].isin(_AGGREGATE_CODES)].copy()
    if not include_total:
        frame = frame.loc[frame["category_code"] != TOTAL_CATEGORY]
    frame = frame.loc[frame["month"] >= pd.Timestamp(start)]
    if frame.empty:
        raise PassThroughError(f"no price observations at or after {start}")

    months = pd.DatetimeIndex(sorted(frame["month"].unique()))
    regions = sorted(frame["nuts_code"].unique())
    shock = build_regional_shock(wage_panel, months, regions)

    merged = frame.merge(shock, on=["nuts_code", "month"], how="inner")
    if merged.empty:
        raise PassThroughError("prices and wages do not overlap")

    merged["log_price"] = np.log(merged["price_index"])
    merged["region_category"] = merged["nuts_code"] + "_" + merged["category_code"].astype(str)
    merged["region"] = merged["nuts_code"]
    return merged.sort_values(["region_category", "month"]).reset_index(drop=True)


def add_category_interactions(
    panel: pd.DataFrame,
    *,
    shock_column: str = "delta_log_minimum_wage",
) -> tuple[pd.DataFrame, list[str]]:
    """Interact the statutory change with each consumption category.

    Estimating one coefficient per category avoids imposing an exposure measure
    that no source supports. What it buys is a ranking; what it costs is that
    calendar-time effects cannot be included, since the shock does not vary
    across regions within a month once the mainland is the sample.

    Args:
        panel: Output of :func:`build_estimation_panel`.
        shock_column: Column holding the statutory change.

    Returns:
        The panel with one interaction column per category, and their names.

    Raises:
        PassThroughError: If the shock column is absent.
    """
    if shock_column not in panel.columns:
        raise PassThroughError(f"panel has no {shock_column!r} column")

    frame = panel.copy()
    names: list[str] = []
    for category in sorted(frame["category_code"].unique()):
        column = f"shock_x_{category}"
        frame[column] = frame[shock_column] * (frame["category_code"] == category).astype(float)
        names.append(column)
    return frame, names
