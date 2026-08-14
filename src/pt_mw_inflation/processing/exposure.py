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

import re
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

    **This is not the path the pipeline takes**, and the difference is the first
    argument. It wants a bite that varies by *region* and industry, which is the
    measure no Portuguese source publishes and whose absence is the constraint
    the whole exposure design works around. Nothing in `ptmw` calls this;
    :func:`structural_exposure` builds the region-by-category measure that is
    actually estimated, substituting regional employment composition for the
    regional bite this function assumes.

    It is kept, and kept tested, because it states what the design would be if
    the bite were published by region: the shift-share is a workaround, and
    deleting the thing it works around would leave that invisible. A reader
    looking for the estimated measure wants :func:`structural_exposure`.

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


@dataclass(frozen=True)
class VariationStrength:
    """How much usable spread an exposure measure carries, not merely whether any."""

    coefficient_of_variation: float
    spread: float
    interquartile_spread: float
    regions: int
    detail: str


def measure_variation_strength(
    frame: pd.DataFrame,
    *,
    value_column: str = "regional_bite_exposure",
    region_column: str = "region",
) -> VariationStrength:
    """Quantify the spread of an exposure measure across regions.

    :func:`assess_identifying_variation` answers whether the measure varies at
    all, which is a precondition and not a recommendation. A measure can take a
    distinct value in every region and still be too flat to estimate anything
    from once fixed effects have absorbed the level. This reports the size of
    the variation so that judgement can be made on a number.

    Args:
        frame: Exposure by region.
        value_column: Column holding the exposure.
        region_column: Column identifying the region.

    Returns:
        The coefficient of variation, the high-low spread, and the
        interquartile spread, which is the more robust of the two when one
        region is unusual.

    Raises:
        ExposureError: If the columns are absent, fewer than two regions are
            present, or the mean is zero and the coefficient undefined.
    """
    _require_columns("exposure", frame, frozenset({value_column, region_column}))

    values = frame.groupby(region_column)[value_column].mean().astype(float)
    if values.size < 2:
        raise ExposureError(f"at least two regions are required, found {values.size}")

    mean = float(values.mean())
    if mean == 0.0:
        raise ExposureError("mean exposure is zero; the coefficient of variation is undefined")

    coefficient = float(values.std(ddof=0)) / abs(mean)
    spread = float(values.max() - values.min())
    interquartile = float(values.quantile(0.75) - values.quantile(0.25))

    return VariationStrength(
        coefficient_of_variation=coefficient,
        spread=spread,
        interquartile_spread=interquartile,
        regions=int(values.size),
        detail=(
            f"coefficient of variation {coefficient:.3f} across {values.size} regions; "
            f"high-low spread {spread:.4f}"
        ),
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


class PredeterminationError(ExposureError):
    """Raised when an exposure measure postdates the episode it is applied to."""


def check_predetermined(
    registry: dict[str, Any], composition_year: int, first_shock_year: int
) -> None:
    """Refuse an exposure measured after the policy episode it is applied to.

    Coverage measured after a wage rise is partly caused by it, which is why the
    design calls for a predetermined bite. Nothing in the data prevents a 2017
    bite being applied to a 2015 shock, so the check has to be explicit: the
    resulting estimate would have the outcome built into the regressor and
    would look entirely healthy.

    Args:
        registry: Parsed bite registry, carrying `source.reference_period`.
        composition_year: Year the regional employment shares are frozen at.
        first_shock_year: First year of the episode being estimated.

    Raises:
        PredeterminationError: If either input is dated at or after the first
            shock, or if the registry states no usable reference period.
    """
    period = str(registry.get("source", {}).get("reference_period", ""))
    match = re.match(r"(\d{4})", period)
    if match is None:
        raise PredeterminationError(f"registry states no usable reference_period: {period!r}")

    offending = [
        f"{name} measured in {year}"
        for name, year in (("bite", int(match.group(1))), ("composition", composition_year))
        if year >= first_shock_year
    ]
    if offending:
        raise PredeterminationError(
            f"exposure is not predetermined for shocks from {first_shock_year}: "
            + ", ".join(offending)
            + ". Use an earlier snapshot, or start the window after the exposure was measured."
        )


def activity_bite_from_registry(
    registry: dict[str, Any], national_employment: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate the published bite onto the groups regional employment uses.

    Each group's bite is the employment-weighted mean of the sections that have
    one. An unweighted mean would treat manufacturing and a near-zero-bite
    utility sector as equally large, and they are not.

    Coverage is measured rather than assumed. A group can contain sections the
    survey never looked at -- public administration sits in the same aggregate
    as education and health -- and assigning the surveyed sections' bite to the
    whole group would impute a minimum wage to workers nobody surveyed, while
    reporting the group as fully covered.

    Args:
        registry: Parsed `config/minimum_wage_bite.yaml`.
        national_employment: Columns `activity` and `employment_thousands` at
            NACE section level. Omitting it weights sections equally, which is
            correct only for a group holding a single section.

    Returns:
        Columns `industry`, `minimum_wage_bite` and `measured_employment_share`,
        the last being the fraction of the group's employment the bite was
        actually measured on.

    Raises:
        ExposureError: If a group names an unknown activity or one with no
            section mapping, or if employment is supplied but does not cover
            every section the registry weights.
    """
    bite = registry["bite_by_activity"]
    section_of = registry["section_of_activity"]

    weights: dict[str, float] = {}
    weighted_by_employment = national_employment is not None
    if national_employment is not None:
        _require_columns(
            "national_employment",
            national_employment,
            frozenset({"activity", "employment_thousands"}),
        )
        weights = dict(
            zip(
                national_employment["activity"].astype(str),
                national_employment["employment_thousands"].astype(float),
                strict=False,
            )
        )

        # A section absent from the response would otherwise fall back to a
        # weight of one thousand workers: not zero, not its true size, and small
        # enough to look like a rounding artefact in the aggregate. An
        # incomplete Eurostat response has to be a failure, because the numbers
        # it produces are wrong in a way nothing downstream can detect.
        required = {
            code
            for block in registry["nace_aggregates"].values()
            for code in [
                *block.get("sections", []),
                *(section_of[name] for name in block.get("measured", []) if name in section_of),
            ]
        }
        absent = sorted(required - weights.keys())
        if absent:
            raise ExposureError(
                f"national employment is missing sections {absent}; "
                "weighting them by default would silently distort every group they belong to"
            )

    rows = []
    for group, block in registry["nace_aggregates"].items():
        measured = list(block.get("measured", []))
        sections = list(block.get("sections", []))

        unknown = [name for name in measured if name not in bite]
        if unknown:
            raise ExposureError(f"{group} names unknown activities: {unknown}")
        unmapped = [name for name in measured if name not in section_of]
        if unmapped:
            raise ExposureError(f"{group} names activities with no section: {unmapped}")

        if not measured:
            rows.append(
                {
                    "industry": group,
                    "minimum_wage_bite": float("nan"),
                    "measured_employment_share": 0.0,
                }
            )
            continue

        # Without an employment frame every section counts once, which is right
        # only where a group holds one section. The validation above guarantees
        # that a supplied frame covers all of them, so the default is reached
        # only in the unweighted mode.
        default = 0.0 if weighted_by_employment else 1.0
        section_weights = [weights.get(section_of[name], default) for name in measured]
        measured_weight = sum(section_weights)
        group_weight = sum(weights.get(code, default) for code in sections) or measured_weight

        weighted = (
            sum(bite[name] * weight for name, weight in zip(measured, section_weights, strict=True))
            / measured_weight
            / 100.0
        )
        rows.append(
            {
                "industry": group,
                "minimum_wage_bite": weighted,
                "measured_employment_share": min(measured_weight / group_weight, 1.0),
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
        One row per region, with `regional_bite_exposure` and the share of
        regional employment the bite was measured on. The name is deliberate:
        this is an employment-weighted bite, not a cost share. It becomes cost
        exposure only once multiplied by labour's share of costs, which is not
        yet in the pipeline, and naming it that now would overstate what it
        measures.

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
    if "measured_employment_share" not in measured.columns:
        measured["measured_employment_share"] = 1.0

    # Weight by the employment the bite was measured on, not by the whole group.
    # A partly surveyed group contributes only its surveyed part; counting the
    # rest would impute a bite to workers nobody looked at and would report the
    # region as better covered than it is.
    measured["covered"] = measured[share_column] * measured["measured_employment_share"]
    measured["contribution"] = measured["covered"] * measured["minimum_wage_bite"]

    exposure = measured.groupby("region", as_index=False).agg(
        measured_exposure=("contribution", "sum"),
        covered_employment_share=("covered", "sum"),
    )

    # Three distinct quantities, kept apart because they answer different
    # questions and only one of them is what the design wants.
    #
    # `measured_exposure` is the minimum-wage bite as a share of *all* the
    # region's employees: the sectors nobody surveyed contribute nothing to the
    # numerator and still sit in the denominator.
    #
    # `regional_bite_exposure` divides that by coverage, giving the bite
    # conditional on working in a surveyed sector. Reporting only this
    # conditions the unsurveyed sectors away, and they are not distributed
    # evenly: agriculture is unsurveyed and is a far larger share of employment
    # in the Alentejo than in Lisbon, so dividing it out compresses precisely
    # the regional differences the measure exists to capture.
    #
    # Neither is the unconditional probability that an employee earns the
    # minimum wage, which is unobservable without an assumption about the
    # unsurveyed sectors. :func:`bound_unmeasured_exposure` supplies that
    # assumption explicitly instead of burying it in a normalisation.
    exposure["regional_bite_exposure"] = (
        exposure["measured_exposure"] / exposure["covered_employment_share"]
    )
    exposure["exposure_definition"] = "shift_share_national_bite"
    return exposure.sort_values("region").reset_index(drop=True)


def bound_unmeasured_exposure(
    exposure: pd.DataFrame, incidence: float, *, label: str | None = None
) -> pd.DataFrame:
    """Complete the exposure measure under a stated bite for unsurveyed sectors.

    The survey covers neither agriculture nor public administration, and their
    weight differs sharply across regions. Two conventions are usually applied
    silently: setting their bite to zero, which asserts that no agricultural
    worker earns the minimum wage, or renormalising them away, which assumes
    their bite equals the average of the sectors that were surveyed. Both are
    assumptions, and neither is stated.

    This makes the assumption a parameter. Sweeping it over a defensible range
    is what turns "the spread is narrow" from a property of one normalisation
    into a claim about the measure.

    Args:
        exposure: Output of :func:`shift_share_exposure`.
        incidence: Assumed minimum-wage bite in the unsurveyed sectors, as a
            proportion.
        label: Name recorded in `exposure_definition`.

    Returns:
        A frame whose `regional_bite_exposure` is
        ``measured_exposure + (1 - coverage) * incidence``, the unconditional
        bite implied by that assumption.

    Raises:
        ExposureError: If the columns are absent or the incidence is not a
            proportion.
    """
    _require_columns(
        "exposure", exposure, frozenset({"measured_exposure", "covered_employment_share"})
    )
    if not 0.0 <= incidence <= 1.0:
        raise ExposureError(f"incidence must be a proportion, got {incidence}")

    bounded = exposure.copy()
    uncovered = 1.0 - bounded["covered_employment_share"]
    bounded["regional_bite_exposure"] = bounded["measured_exposure"] + uncovered * incidence
    bounded["exposure_definition"] = label or f"unmeasured_bite_{incidence:.2f}"
    return bounded


def select_snapshot(registry: dict[str, Any], period: str | None = None) -> dict[str, Any]:
    """Return the registry with one measurement round selected.

    The published table reports several survey rounds. Which one is used is an
    identification decision, not a detail: only a round measured before the
    first shock is predetermined, and the most recent round --- the natural
    default --- is the one least likely to qualify.

    Args:
        registry: Parsed `config/minimum_wage_bite.yaml`.
        period: Round to select, as `YYYY-MM`. The registry's own default round
            is used when omitted.

    Returns:
        A registry whose `bite_by_activity` and `source.reference_period` are
        those of the selected round. Every other key is carried through, so the
        aggregation and coverage rules do not vary with the round.

    Raises:
        ExposureError: If the round is not recorded, or records no bite.
    """
    if period is None:
        return registry

    snapshots = registry.get("snapshots") or {}
    if period not in snapshots:
        available = sorted(snapshots) or ["none"]
        raise ExposureError(f"no snapshot for {period}; available: {available}")

    snapshot = snapshots[period]
    bite = snapshot.get("bite_by_activity")
    if not bite:
        raise ExposureError(f"snapshot {period} records no bite_by_activity")

    selected = dict(registry)
    selected["bite_by_activity"] = bite
    selected["source"] = {
        **dict(registry.get("source") or {}),
        "reference_period": period,
        "national_total": snapshot.get("national_total"),
    }
    return selected


@dataclass(frozen=True)
class StructuralCoverage:
    """What the region-by-category exposure was built on, and what it missed.

    Attributes:
        measured_share: Share of the consumption bridge's weight that reaches an
            activity carrying both a bite and a labour share. The rest
            contributes nothing, and it is not spread evenly across categories.
        unmeasured_activities: Activities absent from one term or the other,
            with which term is missing.
        identifying_spread: Range of the double-demeaned exposure matrix, in
            percentage points. This, and not the raw spread, is what survives
            region-time and category-time effects.
        coverage_weighted: Whether the bite carried the share of each group's
            employment it was measured on. False means every group was treated
            as fully surveyed, which overstates the exposure of the partly
            surveyed ones and is usually a caller that passed the bite column
            without the frame around it.
    """

    measured_share: float
    unmeasured_activities: tuple[tuple[str, str], ...]
    identifying_spread: float
    coverage_weighted: bool


def structural_exposure(
    shares: pd.DataFrame,
    activity_bite: pd.DataFrame,
    labour_share: pd.DataFrame,
    consumption_bridge: pd.DataFrame,
    *,
    share_column: str = "employment_share",
) -> tuple[pd.DataFrame, StructuralCoverage]:
    r"""Build the region-by-category exposure from all four of its terms.

    .. math::

        B_{rc} = \sum_s q_{rs}\, b_s\, \ell_s\, \omega_{cs}

    Three of the terms are national and only the composition varies by region.
    An earlier version of this project dismissed the two extra terms on the
    grounds that national factors cannot create regional variation, which
    confused "these factors do not vary across regions" with "their product does
    not vary across region-category cells". Once region-time and category-time
    effects are absorbed, what identifies the coefficient is the non-additive
    part of this matrix, and the national terms enter it.

    The quantity that matters is therefore reported alongside the measure. The
    raw spread of :math:`B` overstates what the design has, because the fixed
    effects remove its row and column means; `identifying_spread` is what is
    left after that removal.

    Args:
        shares: Regional employment shares by activity.
        activity_bite: National bite by activity, from
            :func:`activity_bite_from_registry`.
        labour_share: Columns `activity` and `labour_cost_share`.
        consumption_bridge: Columns `category`, `industry` and
            `production_weight`, from
            :func:`~pt_mw_inflation.processing.consumption_bridge.build_consumption_bridge`.
        share_column: Column holding the employment share.

    Returns:
        One row per region and category with `structural_exposure`, and a record
        of the activities the measure could not reach.

    Raises:
        ExposureError: If a required column is missing, if the bridge violates
            its contract, or if nothing merges, which is what an activity-coding
            mismatch between the four sources looks like.
    """
    _require_columns("shares", shares, frozenset({"region", "activity", share_column}))
    _require_columns("activity_bite", activity_bite, frozenset({"industry", "minimum_wage_bite"}))
    _require_columns("labour_share", labour_share, frozenset({"activity", "labour_cost_share"}))
    validate_bridge(consumption_bridge)

    bite = activity_bite.set_index("industry")["minimum_wage_bite"]

    # A group can be only partly surveyed, and `activity_bite_from_registry`
    # reports how much of it was. Weighting by that is what stops a bite
    # measured on part of a group being credited to all of it.
    #
    # The column is optional because a caller may supply a bite from elsewhere,
    # but its absence is recorded rather than absorbed. Passing the bite as a
    # bare column instead of the full frame drops it silently and turns the
    # weighting off, which is worth 0.4 of a point of identifying spread here
    # and looks identical to having asked for it.
    coverage_weighted = "measured_employment_share" in activity_bite.columns
    measured_cover = (
        activity_bite.set_index("industry")["measured_employment_share"]
        if coverage_weighted
        else pd.Series(1.0, index=activity_bite["industry"])
    )
    labour = labour_share.set_index("activity")["labour_cost_share"]

    # An activity missing either term contributes nothing rather than being
    # dropped or imputed. Both absences are real and neither is an error: the
    # survey excludes agriculture, so it has no bite, and real estate value
    # added is imputed rent, which employs nobody, so its labour share is
    # suppressed as an artefact. Treating them as zero exposure is the
    # conservative reading, and which activities it applies to is reported.
    activities = sorted(set(shares["activity"].astype(str)))
    unmeasured: list[tuple[str, str]] = []
    weight: dict[str, float] = {}
    for activity in activities:
        this_bite = float(bite.get(activity, float("nan")))
        this_labour = float(labour.get(activity, float("nan")))
        missing = []
        if not np.isfinite(this_bite):
            missing.append("bite")
        if not np.isfinite(this_labour):
            missing.append("labour share")
        if missing:
            unmeasured.append((activity, " and ".join(missing)))
            weight[activity] = 0.0
            continue
        weight[activity] = this_bite * this_labour * float(measured_cover.get(activity, 1.0))

    composition = shares.assign(
        _weight=shares["activity"].astype(str).map(weight).astype(float),
    )
    composition["_contribution"] = composition[share_column] * composition["_weight"]

    merged = composition.merge(
        consumption_bridge, left_on="activity", right_on="industry", validate="many_to_many"
    )
    if merged.empty:
        raise ExposureError(
            "no activities merged between composition and the bridge; "
            "check the activity coding on both sides"
        )

    merged["_cell"] = merged["_contribution"] * merged["production_weight"]
    exposure = merged.groupby(["region", "category"], as_index=False)[["_cell"]].sum()
    exposure = exposure.rename(columns={"_cell": "structural_exposure"})

    # The national minimum-wage cost share of each category, carried alongside
    # because it is what makes an estimate readable.
    #
    # The exposure above is *not* a cost share and must not be read as one. It
    # weights each industry by the region's employment share in it, which is
    # what introduces the regional dimension the design needs, and that weight
    # is a fraction summing to one across industries. So the exposure runs about
    # a third of the cost share, and a coefficient of one on it is not complete
    # pass-through. The cost share here is on the scale where it would be:
    # sum_s w[c,s] b[s] l[s], the share of category c's costs that is
    # minimum-wage labour, with no regional weighting at all.
    cost_share = (
        consumption_bridge.assign(
            _unit=consumption_bridge["industry"]
            .astype(str)
            .map(lambda activity: weight.get(activity, 0.0))
        )
        .assign(_x=lambda frame: frame["production_weight"] * frame["_unit"])
        .groupby("category")["_x"]
        .sum()
    )
    exposure["category_cost_share"] = exposure["category"].map(cost_share).astype(float)
    exposure["exposure_definition"] = "structural_region_category"

    # The share of the bridge's weight that reaches a measured activity. Below
    # one it means categories are supplied partly by industries this measure
    # cannot see, and the shortfall differs across categories.
    reachable = consumption_bridge.assign(
        _measured=consumption_bridge["industry"]
        .astype(str)
        .map(lambda activity: 1.0 if weight.get(activity, 0.0) > 0 else 0.0)
    )
    measured_share = float(
        (reachable["production_weight"] * reachable["_measured"]).sum()
        / reachable["production_weight"].sum()
    )

    matrix = exposure.pivot(index="region", columns="category", values="structural_exposure")
    values = matrix.to_numpy(dtype=float)
    residual = (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )

    coverage = StructuralCoverage(
        measured_share=measured_share,
        unmeasured_activities=tuple(unmeasured),
        identifying_spread=100.0 * float(residual.max() - residual.min()),
        coverage_weighted=coverage_weighted,
    )
    return exposure, coverage
