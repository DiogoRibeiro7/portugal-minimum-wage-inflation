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
        national = pd.Series(fallback, index=months)
        stepped = stepped.fillna(national)
        # The national wage is a floor everywhere in Portugal, and a regional
        # act sets a supplement above it rather than a wage instead of it. A
        # region that legislates intermittently can therefore be overtaken:
        # Madeira's 2017 act stood at 570 through 2018, by which time the
        # national wage was 580, so 580 was what actually bound. Carrying the
        # regional level unconditionally would report a wage no employer could
        # lawfully pay, and would record the region as diverging downwards in a
        # year its premium had simply been extinguished.
        stepped = stepped.where(stepped >= national, national)
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

        # Entering or leaving a declared gap moves the series between the
        # region's own level and the national one. That step is an artefact of
        # the register, not a wage change anybody legislated, so it is marked
        # unestimable rather than passed to the estimator as a shock.
        region_gaps = (gap_years or {}).get(region)
        if region_gaps:
            years = pd.Series(months.year, index=months)
            in_gap = years.isin(region_gaps).to_numpy()
            boundary = np.zeros(len(months), dtype=bool)
            boundary[1:] = in_gap[1:] != in_gap[:-1]
            block.loc[boundary, "delta_log_minimum_wage"] = np.nan
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
    gap_years: dict[str, frozenset[int]] | None = None,
) -> pd.DataFrame:
    """Join regional prices to the applicable statutory wage.

    Args:
        prices: Regional price panel.
        wage_panel: Statutory panel, general regime.
        start: First month to retain.
        include_total: Keep the all-items index alongside the categories.
        gap_years: Per-region years missing from the statutory register. Passing
            them is what keeps a hole in the register out of the estimation; a
            builder that accepts them but never forwards them leaves the
            artefact in the output while appearing to remove it.

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
    shock = build_regional_shock(wage_panel, months, regions, gap_years=gap_years)

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


@dataclass(frozen=True)
class SeasonalConfound:
    """How far a statutory shock is confounded with the calendar.

    Attributes:
        modal_month: Calendar month carrying most of the statutory change.
        modal_share: Share of total log change falling in that month.
        surviving_variance_share: Share of the shock's variance left once
            month-of-year effects are absorbed. This is the variation any
            seasonally-controlled estimate is actually identified from.
        worst_category: Category whose own seasonal is largest in the modal
            month, which is the category the confound distorts most.
        worst_category_swing: That category's mean log change in the modal
            month, in per cent.
    """

    modal_month: int
    modal_share: float
    surviving_variance_share: float
    worst_category: str
    worst_category_swing: float


def diagnose_seasonal_confound(
    prices: pd.DataFrame,
    shock: pd.Series,
    *,
    category_column: str = "category_code",
    month_column: str = "month",
    price_column: str = "price_index",
) -> SeasonalConfound:
    """Measure how far the statutory shock is collinear with the calendar.

    Portugal moves its minimum wage on 1 January in almost every year, so the
    shock is nearly a January indicator. Any consumption category with a large
    January seasonal is then attributed a response it did not have: clothing
    falls sharply every January because winter sales enter the index, and a
    specification without month-of-year effects reads that fall as the price
    response to the wage rise that shares its date.

    The diagnosis matters more than the correction. Absorbing month-of-year
    effects removes the artefact but also removes most of the shock, because
    most of the shock *is* the calendar. That is a property of the policy, not
    of the data, and no amount of cleaning changes it.

    Args:
        prices: Long price panel with category, month and index columns.
        shock: Log statutory wage indexed by month.
        category_column: Column naming the consumption category.
        month_column: Column holding the month.
        price_column: Column holding the price index.

    Returns:
        The confound diagnostics.

    Raises:
        PassThroughError: If the shock never changes, or the panel and the
            shock share no months.
    """
    frame = prices.copy()
    frame[month_column] = pd.to_datetime(frame[month_column])
    wide = frame.groupby([category_column, month_column])[price_column].mean().unstack(0)
    if wide.empty:
        raise PassThroughError("price panel carries no observations")

    changes = pd.Series(shock).sort_index().diff().dropna()
    changes.index = pd.DatetimeIndex(changes.index)
    changes = changes.loc[changes.index.isin(wide.index)]
    if changes.empty:
        raise PassThroughError("the shock and the price panel share no months")

    magnitude = changes.abs()
    by_month = magnitude.groupby(pd.DatetimeIndex(changes.index).month).sum()
    if float(by_month.sum()) == 0.0:
        raise PassThroughError("the statutory shock never changes")

    months = pd.DatetimeIndex(changes.index)
    modal_month = int(by_month.idxmax())
    modal_share = float(by_month.max() / by_month.sum())

    # What a month-of-year specification would have left to work with. Regressing
    # the shock on calendar-month indicators and keeping the residual is exactly
    # what including those indicators does to it.
    indicators = pd.get_dummies(months.month, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([indicators, np.ones(len(changes))])
    values = changes.to_numpy(dtype=float)
    fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    total = float(np.var(values))
    surviving = float(np.var(values - fitted)) / total if total > 0 else 0.0

    growth = pd.DataFrame(np.log(wide.to_numpy()), index=wide.index, columns=wide.columns).diff()
    price_months = pd.DatetimeIndex(growth.index)
    modal_swing = growth.loc[price_months.month == modal_month].mean()
    worst = str(modal_swing.abs().idxmax())

    return SeasonalConfound(
        modal_month=modal_month,
        modal_share=modal_share,
        surviving_variance_share=surviving,
        worst_category=worst,
        worst_category_swing=100.0 * float(modal_swing[worst]),
    )


def add_exposure_interaction(
    panel: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    shock_column: str = "delta_log_minimum_wage",
    exposure_column: str = "regional_bite_exposure",
    region_column: str = "region",
) -> pd.DataFrame:
    """Interact the statutory change with each region's predetermined exposure.

    This is the design the literature favours, and it has one decisive advantage
    over the category-differential alternative: because exposure varies across
    regions, the interaction survives calendar-time fixed effects. Those effects
    absorb everything moving national prices in a month --- energy, taxes, the
    pandemic --- which the category design cannot do and is confounded by.

    What it cannot do is manufacture variation. The interaction is the national
    shock scaled by a regional constant, so its cross-sectional spread is the
    spread of that constant, and no estimator recovers more information than the
    exposure measure carries.

    Args:
        panel: Output of :func:`build_estimation_panel`.
        exposure: Predetermined exposure by region.
        shock_column: Column holding the log statutory change.
        exposure_column: Column holding regional exposure.
        region_column: Column identifying the region in the panel.

    Returns:
        The panel with `exposure_shock`, and the exposure it was built from.

    Raises:
        PassThroughError: If a column is missing, or no region matches, which is
            what a NUTS coding mismatch between the two sources looks like.
    """
    for name, frame, columns in (
        ("panel", panel, {shock_column, region_column}),
        ("exposure", exposure, {exposure_column, "region"}),
    ):
        missing = columns.difference(frame.columns)
        if missing:
            raise PassThroughError(f"{name} missing columns: {sorted(missing)}")

    merged = panel.merge(
        exposure[["region", exposure_column]],
        left_on=region_column,
        right_on="region",
        how="inner",
        suffixes=("", "_exposure"),
    )
    if merged.empty:
        raise PassThroughError(
            "no region matches between the panel and the exposure measure; "
            f"panel has {sorted(panel[region_column].unique())[:4]}..., "
            f"exposure has {sorted(exposure['region'].unique())[:4]}..."
        )

    # Demeaned exposure, so the coefficient is the differential response of a
    # region one unit more exposed than average rather than a level whose
    # interpretation depends on where the exposure scale happens to sit.
    centred = merged[exposure_column] - merged[exposure_column].mean()
    merged["exposure_shock"] = centred * merged[shock_column]
    return merged
