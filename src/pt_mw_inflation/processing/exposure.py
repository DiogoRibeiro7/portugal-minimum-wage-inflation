"""Construction of predetermined minimum-wage cost exposure measures.

The structural exposure index follows the research design:

.. math::

    E_{rct} = \\sum_s b_{rs,0}\\,\\ell_{st}\\,\\omega_{cs}\\,\\Delta\\log MW_{rt}

where :math:`b_{rs,0}` is a *predetermined* minimum-wage bite for region ``r``
and industry ``s``, :math:`\\ell_{st}` the labour-cost share, :math:`\\omega_{cs}`
the production-to-consumption bridge weight, and :math:`\\Delta\\log MW_{rt}`
the applicable statutory change.

Two properties matter more than the arithmetic:

*Predetermined bite.* Coverage measured after a wage rise is mechanically
caused by it, so the baseline bite is frozen before each policy episode and the
freezing is enforced here rather than left to the caller.

*Documented denominators.* A weight is meaningless without the total it is a
share of. Every weight this module accepts is checked against its stated
denominator, and a bridge whose weights do not sum to one for a consumption
category is rejected instead of silently rescaling the exposure of that
category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BITE_COLUMNS = frozenset({"region", "industry", "reference_period", "minimum_wage_bite"})
LABOUR_SHARE_COLUMNS = frozenset({"industry", "labour_cost_share"})
BRIDGE_COLUMNS = frozenset({"category", "industry", "production_weight"})

#: Tolerance for a bridge column summing to one. Published supply-use tables are
#: rounded, so exact equality is not attainable, but a gap beyond this points to
#: a missing industry rather than to rounding.
WEIGHT_SUM_TOLERANCE = 1e-6


class ExposureError(ValueError):
    """Raised when exposure inputs violate a documented structural requirement."""


@dataclass(frozen=True)
class ExposureDefinition:
    """One exposure variant, so robustness is a first-class object.

    Attributes:
        name: Identifier carried into results tables.
        baseline_period: Period whose bite is used, held fixed thereafter.
        use_labour_share: Whether to weight by the industry labour-cost share.
        description: Why this variant exists and what it tests.
    """

    name: str
    baseline_period: int
    use_labour_share: bool = True
    description: str = ""


def _require_columns(name: str, frame: pd.DataFrame, required: frozenset[str]) -> None:
    """Raise if a frame is missing any required column."""
    missing = required.difference(frame.columns)
    if missing:
        raise ExposureError(f"{name} missing columns: {sorted(missing)}")


def validate_shares(
    frame: pd.DataFrame,
    column: str,
    *,
    name: str,
    lower: float = 0.0,
    upper: float = 1.0,
) -> None:
    """Check that a share column lies in its admissible range.

    Args:
        frame: Frame holding the share.
        column: Share column name.
        name: Human-readable frame name for the error message.
        lower: Smallest admissible value.
        upper: Largest admissible value.

    Raises:
        ExposureError: If any value is missing or outside the range, which
            normally means a percentage was supplied where a proportion was
            expected.
    """
    values = frame[column]
    if values.isna().any():
        raise ExposureError(f"{name}.{column} contains missing values")
    if (values < lower).any() or (values > upper).any():
        observed = (float(values.min()), float(values.max()))
        raise ExposureError(
            f"{name}.{column} must lie in [{lower}, {upper}]; observed range {observed}. "
            "A share expressed in percent rather than as a proportion is the usual cause."
        )


def validate_bridge(bridge: pd.DataFrame) -> pd.Series:
    """Check that bridge weights form a proper distribution over industries.

    Each consumption category's weights must sum to one across production
    industries, because the bridge apportions one unit of consumption spending.
    A category summing to less than one has industries missing from the table,
    and its exposure would be understated by exactly the missing mass.

    Args:
        bridge: Production-to-consumption bridge.

    Returns:
        The realised weight sum per category, for reporting.

    Raises:
        ExposureError: If any category's weights do not sum to one.
    """
    _require_columns("consumption_bridge", bridge, BRIDGE_COLUMNS)
    validate_shares(bridge, "production_weight", name="consumption_bridge")

    sums = bridge.groupby("category")["production_weight"].sum()
    offending = sums[(sums - 1.0).abs() > WEIGHT_SUM_TOLERANCE]
    if not offending.empty:
        detail = ", ".join(f"{category}={total:.6f}" for category, total in offending.items())
        raise ExposureError(
            f"bridge weights must sum to 1 per consumption category; offending: {detail}"
        )
    return sums


def freeze_baseline_bite(bite: pd.DataFrame, baseline_period: int) -> pd.DataFrame:
    """Select the predetermined bite and drop its time dimension.

    Args:
        bite: Bite observations by region, industry and reference period.
        baseline_period: Period to freeze on.

    Returns:
        Bite for the baseline period only, with `reference_period` retained for
        provenance.

    Raises:
        ExposureError: If the baseline period is absent, or if a region and
            industry appear more than once in it.
    """
    _require_columns("bite", bite, BITE_COLUMNS)
    validate_shares(bite, "minimum_wage_bite", name="bite")

    frozen = bite.loc[bite["reference_period"] == baseline_period].copy()
    if frozen.empty:
        available = sorted(bite["reference_period"].unique())
        raise ExposureError(f"baseline period {baseline_period} absent; available: {available}")

    duplicated = frozen.duplicated(subset=["region", "industry"])
    if duplicated.any():
        raise ExposureError(
            f"{int(duplicated.sum())} duplicate region-industry rows in baseline "
            f"period {baseline_period}"
        )
    return frozen


def construct_cost_exposure(
    bite: pd.DataFrame,
    labour_share: pd.DataFrame,
    consumption_bridge: pd.DataFrame,
    *,
    definition: ExposureDefinition | None = None,
) -> pd.DataFrame:
    """Map industry minimum-wage exposure into consumption-category cost exposure.

    Args:
        bite: Columns `region`, `industry`, `reference_period`, `minimum_wage_bite`.
        labour_share: Columns `industry`, `labour_cost_share`.
        consumption_bridge: Columns `category`, `industry`, `production_weight`.
        definition: Exposure variant. When omitted, the latest available period
            is used as the baseline and the labour share is applied.

    Returns:
        Structural exposure per region and consumption category, together with
        the variant name and the baseline period it was frozen at. This is the
        unit-cost term only; multiply by the statutory change with
        :func:`apply_policy_shock`.

    Raises:
        ExposureError: If any input violates its documented contract.
    """
    _require_columns("bite", bite, BITE_COLUMNS)
    _require_columns("labour_share", labour_share, LABOUR_SHARE_COLUMNS)
    validate_bridge(consumption_bridge)
    validate_shares(labour_share, "labour_cost_share", name="labour_share")

    if definition is None:
        definition = ExposureDefinition(
            name="baseline",
            baseline_period=int(bite["reference_period"].max()),
            description="Latest available bite, labour-share weighted.",
        )

    frozen = freeze_baseline_bite(bite, definition.baseline_period)

    merged = frozen.merge(labour_share, on="industry", validate="many_to_one")
    merged = merged.merge(consumption_bridge, on="industry", validate="many_to_many")
    if merged.empty:
        raise ExposureError("no industries survive the merge; check industry coding consistency")

    weight = merged["minimum_wage_bite"] * merged["production_weight"]
    if definition.use_labour_share:
        weight = weight * merged["labour_cost_share"]
    merged["component_exposure"] = weight

    aggregated = merged.groupby(["region", "category"], as_index=False)[
        ["component_exposure"]
    ].sum()
    aggregated = aggregated.rename(columns={"component_exposure": "cost_exposure"})
    aggregated["exposure_definition"] = definition.name
    aggregated["baseline_period"] = definition.baseline_period
    return aggregated


def apply_policy_shock(
    exposure: pd.DataFrame,
    policy_change: pd.DataFrame,
    *,
    shock_column: str = "delta_log_minimum_wage",
) -> pd.DataFrame:
    """Scale structural exposure by the applicable statutory change.

    Separating this from :func:`construct_cost_exposure` keeps the
    predetermined component and the time-varying policy component visible as
    distinct objects, which is what makes the exposure interpretable as a
    shift-share.

    Args:
        exposure: Output of :func:`construct_cost_exposure`.
        policy_change: Columns `region`, `period`, and the shock column.
        shock_column: Name of the statutory change column.

    Returns:
        Exposure by region, category and period.

    Raises:
        ExposureError: If required columns are absent.
    """
    _require_columns("policy_change", policy_change, frozenset({"region", "period", shock_column}))

    merged = exposure.merge(policy_change, on="region", validate="many_to_many")
    merged["exposure_shock"] = merged["cost_exposure"] * merged[shock_column]
    return merged


def build_exposure_variants(
    bite: pd.DataFrame,
    labour_share: pd.DataFrame,
    consumption_bridge: pd.DataFrame,
    definitions: list[ExposureDefinition],
) -> pd.DataFrame:
    """Construct several exposure definitions for robustness.

    The design requires that results survive alternative exposure definitions,
    so producing them is part of the pipeline rather than an afterthought.

    Args:
        bite: Bite observations.
        labour_share: Industry labour-cost shares.
        consumption_bridge: Production-to-consumption bridge.
        definitions: Variants to construct.

    Returns:
        Stacked exposures, one block per definition.

    Raises:
        ExposureError: If no definitions were supplied.
    """
    if not definitions:
        raise ExposureError("at least one exposure definition is required")

    return pd.concat(
        [
            construct_cost_exposure(bite, labour_share, consumption_bridge, definition=definition)
            for definition in definitions
        ],
        ignore_index=True,
    )


def exposure_correlation(variants: pd.DataFrame) -> pd.DataFrame:
    """Correlate exposure definitions with one another.

    Near-perfect correlation means the robustness checks are not independent
    evidence, and reporting them as if they were overstates the result.

    Args:
        variants: Output of :func:`build_exposure_variants`.

    Returns:
        Correlation matrix across definitions, aligned on region and category.
    """
    wide = variants.pivot_table(
        index=["region", "category"],
        columns="exposure_definition",
        values="cost_exposure",
    )
    return wide.corr(method="pearson").replace({np.nan: 1.0})


@dataclass(frozen=True)
class VariationDiagnostic:
    """How much of an exposure measure's variance is available to identify a regional effect."""

    between_region_share: float
    regions: int
    distinct_values_per_region: int
    identifying: bool
    detail: str


def construct_regional_bite(
    employment: pd.DataFrame,
    industry_bite: pd.DataFrame,
    *,
    employment_column: str = "employees",
) -> pd.DataFrame:
    r"""Aggregate an industry-level bite to regions using local industry mix.

    This is the shift-share construction: where the bite is measured only
    nationally by industry, a region's exposure is the employment-weighted
    average of national industry bites,

    .. math::

        B_r = \sum_s w_{rs}\, b_s,
        \qquad w_{rs} = \frac{L_{rs}}{\sum_{s'} L_{rs'}},

    so all regional variation comes from industry composition. The maintained
    assumption is that the bite within an industry does not vary across regions.

    That assumption is substantive, not technical. Accommodation and food has by
    far the highest bite of any Portuguese industry, and the regions differ
    sharply in how much of it they have and at what wages, so assuming a common
    within-industry bite attenuates real differences. Anything built this way is
    a robustness variant, never the baseline.

    The construction also requires a genuine joint distribution of employment
    over regions and industries. Deriving ``L_{rs}`` from separate regional and
    industry totals assumes the two are independent, which forces
    :math:`w_{rs} = w_s` and yields an exposure that is identical in every
    region. :func:`assess_identifying_variation` detects that case.

    Args:
        employment: Columns `region`, `industry` and the employment count. Must
            be a cross-tabulation, not marginals.
        industry_bite: Columns `industry` and `minimum_wage_bite`.
        employment_column: Name of the employment count column.

    Returns:
        Columns `region`, `minimum_wage_bite` and `employment`, one row per
        region.

    Raises:
        ExposureError: If required columns are missing, if a region-industry
            pair is duplicated, or if no industry survives the merge.
    """
    _require_columns("employment", employment, frozenset({"region", "industry", employment_column}))
    _require_columns("industry_bite", industry_bite, frozenset({"industry", "minimum_wage_bite"}))
    validate_shares(industry_bite, "minimum_wage_bite", name="industry_bite")

    if employment[employment_column].lt(0).any():
        raise ExposureError(f"employment.{employment_column} contains negative counts")

    duplicated = employment.duplicated(subset=["region", "industry"])
    if duplicated.any():
        raise ExposureError(
            f"{int(duplicated.sum())} duplicate region-industry rows in employment; "
            "the input must be a cross-tabulation"
        )

    merged = employment.merge(industry_bite, on="industry", validate="many_to_one")
    if merged.empty:
        raise ExposureError("no industries survive the merge; check industry coding consistency")

    totals = merged.groupby("region")[employment_column].transform("sum")
    if (totals <= 0).any():
        raise ExposureError("every region must have positive total employment")
    merged["weight"] = merged[employment_column] / totals
    merged["weighted_bite"] = merged["weight"] * merged["minimum_wage_bite"]

    aggregated = merged.groupby("region", as_index=False).agg(
        minimum_wage_bite=("weighted_bite", "sum"),
        employment=(employment_column, "sum"),
    )
    return aggregated.sort_values("region").reset_index(drop=True)


def assess_identifying_variation(
    frame: pd.DataFrame,
    *,
    value_column: str = "cost_exposure",
    region_column: str = "region",
    minimum_share: float = 0.01,
) -> VariationDiagnostic:
    """Measure whether an exposure measure varies across regions at all.

    A regional design is identified by differences between regions. When an
    exposure is built from national inputs those differences can be exactly
    zero while every other diagnostic looks healthy: the column is populated,
    its values are plausible, and the regression runs. The coefficient is then
    identified by nothing, or absorbed entirely by the fixed effects.

    Args:
        frame: Exposure measure by region.
        value_column: Column holding the exposure.
        region_column: Column identifying the region.
        minimum_share: Smallest between-region variance share treated as
            identifying.

    Returns:
        The between-region share of total variance and a verdict.

    Raises:
        ExposureError: If the columns are absent or fewer than two regions are
            present.
    """
    _require_columns("exposure", frame, frozenset({value_column, region_column}))

    regions = int(frame[region_column].nunique())
    if regions < 2:
        raise ExposureError(f"at least two regions are required, found {regions}")

    values = frame[value_column].astype(float)
    total_variance = float(values.var(ddof=0))
    region_means = frame.groupby(region_column)[value_column].transform("mean").astype(float)
    between_variance = float(region_means.var(ddof=0))

    share = 0.0 if total_variance <= 0 else between_variance / total_variance
    distinct = int(frame.groupby(region_column)[value_column].mean().round(12).nunique())
    identifying = distinct > 1 and share >= minimum_share

    if distinct <= 1:
        detail = (
            "exposure is identical in every region; the measure was built from "
            "national inputs and carries no regional variation to identify from"
        )
    elif not identifying:
        detail = f"only {share:.1%} of exposure variance lies between regions"
    else:
        detail = f"{share:.1%} of exposure variance lies between regions"

    return VariationDiagnostic(
        between_region_share=share,
        regions=regions,
        distinct_values_per_region=distinct,
        identifying=identifying,
        detail=detail,
    )


def require_regional_variation(
    frame: pd.DataFrame,
    *,
    value_column: str = "cost_exposure",
    region_column: str = "region",
    minimum_share: float = 0.01,
) -> VariationDiagnostic:
    """Refuse to proceed with an exposure that cannot identify a regional effect.

    Args:
        frame: Exposure measure by region.
        value_column: Column holding the exposure.
        region_column: Column identifying the region.
        minimum_share: Smallest between-region variance share accepted.

    Returns:
        The diagnostic, when it passes.

    Raises:
        ExposureError: If the exposure carries no usable regional variation.
    """
    diagnostic = assess_identifying_variation(
        frame,
        value_column=value_column,
        region_column=region_column,
        minimum_share=minimum_share,
    )
    if not diagnostic.identifying:
        raise ExposureError(
            f"exposure cannot identify a regional effect: {diagnostic.detail}. "
            "A region-by-industry measure requires a joint distribution of employment "
            "over regions and industries; separate regional and industry totals are "
            "not sufficient, because assuming they are independent makes every region "
            "identical."
        )
    return diagnostic


def activity_bite_from_registry(registry: dict[str, Any]) -> pd.DataFrame:
    """Aggregate the published bite to the activity groups employment uses.

    The survey reports the bite at a finer activity breakdown than the regional
    accounts publish employment at, so each group takes the mean of the sections
    it contains. Groups with no measured section -- agriculture and public
    administration are outside the survey -- are returned as missing rather than
    as zero, since an unmeasured bite is not an absent one.

    Args:
        registry: Parsed `config/minimum_wage_bite.yaml`.

    Returns:
        Columns `industry` and `minimum_wage_bite`, the bite as a proportion.

    Raises:
        ExposureError: If a group names a section the bite table does not carry.
    """
    bite = registry["bite_by_activity"]
    rows = []
    for group, sections in registry["nace_aggregates"].items():
        missing = [section for section in sections if section not in bite]
        if missing:
            raise ExposureError(f"{group} names unknown activities: {missing}")
        if not sections:
            rows.append({"industry": group, "minimum_wage_bite": float("nan")})
            continue
        rows.append(
            {
                "industry": group,
                "minimum_wage_bite": sum(bite[s] for s in sections) / len(sections) / 100.0,
            }
        )
    return pd.DataFrame(rows)


def shift_share_exposure(
    shares: pd.DataFrame,
    activity_bite: pd.DataFrame,
    *,
    share_column: str = "employment_share",
) -> pd.DataFrame:
    """Combine regional industry composition with the national industry bite.

    This is the measure a previous version of this project recorded as
    impossible to build. Regional composition comes from the regional accounts;
    the bite is national and is held constant within industry, which is the
    measure's maintained assumption rather than a property of the data.

    Args:
        shares: Regional employment shares by activity.
        activity_bite: National bite by activity.
        share_column: Column holding the employment share.

    Returns:
        One row per region, with the exposure and the share of regional
        employment for which a bite was measured. A region whose covered share
        is low has an exposure resting on a minority of its workers, and the
        column is returned so that can be judged rather than assumed away.

    Raises:
        ExposureError: If required columns are missing or nothing merges.
    """
    _require_columns("shares", shares, frozenset({"region", "activity", share_column}))
    _require_columns("activity_bite", activity_bite, frozenset({"industry", "minimum_wage_bite"}))

    merged = shares.merge(
        activity_bite, left_on="activity", right_on="industry", how="left", validate="many_to_one"
    )
    if merged.empty:
        raise ExposureError("no activities merged; check the activity coding")

    measured = merged.dropna(subset=["minimum_wage_bite"]).copy()
    measured["contribution"] = measured[share_column] * measured["minimum_wage_bite"]

    exposure = measured.groupby("region", as_index=False).agg(
        cost_exposure=("contribution", "sum"),
        covered_employment_share=(share_column, "sum"),
    )
    # Renormalise onto measured employment, so a region is not scored low merely
    # because more of its workforce sits outside the survey's scope.
    exposure["cost_exposure"] = exposure["cost_exposure"] / exposure["covered_employment_share"]
    exposure["exposure_definition"] = "shift_share_national_bite"
    return exposure.sort_values("region").reset_index(drop=True)
