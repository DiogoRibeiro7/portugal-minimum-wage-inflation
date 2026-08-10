"""Tests for regional industry composition and the shift-share exposure.

These cover the construction an earlier version of this project recorded as
impossible, and the two ways it can be built wrongly: averaging a bite over
sections of wildly different size, and assigning a measured bite to workers the
survey never looked at. No test touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pt_mw_inflation.data.eurostat_regional import (
    NACE_PARTITION,
    RegionalEmploymentError,
    industry_shares,
    require_matched_inputs,
)
from pt_mw_inflation.processing.exposure import (
    ExposureError,
    PredeterminationError,
    activity_bite_from_registry,
    assess_identifying_variation,
    check_predetermined,
    measure_variation_strength,
    require_regional_variation,
    shift_share_exposure,
)

REGISTRY = {
    "source": {"reference_period": "2017-10"},
    "bite_by_activity": {"Tourism": 36.0, "Utilities": 0.6, "Factories": 24.0},
    "section_of_activity": {"Tourism": "I", "Utilities": "D", "Factories": "C"},
    "nace_aggregates": {
        # Only accommodation is surveyed inside trade-transport-accommodation,
        # so the group is partly observed and must say so.
        "G-I": {"sections": ["G", "H", "I"], "measured": ["Tourism"]},
        "B-E": {"sections": ["C", "D"], "measured": ["Utilities", "Factories"]},
        "A": {"sections": ["A"], "measured": []},
    },
}

#: Manufacturing employs thirty-five times as many people as utilities here,
#: which is the whole reason an unweighted mean of their bites is wrong.
NATIONAL_EMPLOYMENT = pd.DataFrame(
    [
        {"activity": "C", "employment_thousands": 700.0},
        {"activity": "D", "employment_thousands": 20.0},
        {"activity": "G", "employment_thousands": 400.0},
        {"activity": "H", "employment_thousands": 150.0},
        {"activity": "I", "employment_thousands": 250.0},
        {"activity": "A", "employment_thousands": 300.0},
    ]
)


def _employment() -> pd.DataFrame:
    """Two regions with sharply different industry mixes."""
    rows = []
    for region, mix in {
        "PT15": {"G-I": 60.0, "B-E": 20.0, "A": 20.0},
        "PT11": {"G-I": 20.0, "B-E": 60.0, "A": 20.0},
    }.items():
        for activity, value in mix.items():
            rows.append(
                {
                    "region": region,
                    "activity": activity,
                    "year": 2015,
                    "employment_thousands": value,
                }
            )
        rows.append(
            {"region": region, "activity": "TOTAL", "year": 2015, "employment_thousands": 100.0}
        )
    return pd.DataFrame(rows)


def _bite() -> pd.DataFrame:
    """The registry aggregated with real employment weights."""
    return activity_bite_from_registry(REGISTRY, NATIONAL_EMPLOYMENT)


def test_shares_sum_to_one_within_each_region() -> None:
    """The partition must reconstruct each region's total exactly."""
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    assert shares.groupby("region")["employment_share"].sum().round(9).eq(1.0).all()


def test_overlapping_aggregate_is_rejected() -> None:
    """Eurostat publishes aggregates alongside their parts.

    Summing both double counts most workers, and the resulting shares would be
    silently wrong rather than obviously so, which is why the reconstruction of
    the published total is checked instead of trusted.
    """
    doubled = pd.concat(
        [
            _employment(),
            pd.DataFrame(
                [{"region": "PT15", "activity": "C", "year": 2015, "employment_thousands": 15.0}]
            ),
        ]
    )
    with pytest.raises(RegionalEmploymentError, match="miss the published total"):
        industry_shares(doubled, year=2015, partition=("G-I", "B-E", "A", "C"))


def test_absent_baseline_year_is_reported() -> None:
    """Freezing on a year with no data is a configuration error."""
    with pytest.raises(RegionalEmploymentError, match="year 1999 absent"):
        industry_shares(_employment(), year=1999, partition=("G-I", "B-E", "A"))


def test_unmeasured_activity_is_missing_not_zero() -> None:
    """Agriculture is outside the survey; its bite is unknown, not nil.

    Recording it as zero would state that no agricultural worker earns the
    minimum wage, which the source does not say and which is false.
    """
    bite = _bite().set_index("industry")
    assert pd.isna(bite.loc["A", "minimum_wage_bite"])
    assert bite.loc["A", "measured_employment_share"] == 0.0


def test_group_bite_is_weighted_by_employment_not_averaged() -> None:
    """An unweighted mean treats a huge sector and a tiny one as equal.

    Manufacturing employs thirty-five times as many people as utilities and has
    a bite forty times higher, so the unweighted mean sits far below the
    weighted one and would understate industrial regions.
    """
    bite = _bite().set_index("industry")
    weighted = (0.006 * 20.0 + 0.24 * 700.0) / 720.0
    assert bite.loc["B-E", "minimum_wage_bite"] == pytest.approx(weighted)
    assert bite.loc["B-E", "minimum_wage_bite"] > (0.006 + 0.24) / 2


def test_partly_surveyed_group_reports_its_true_coverage() -> None:
    """A group's bite may not be spread over sections nobody surveyed.

    Only accommodation is surveyed within trade-transport-accommodation, so the
    group carries that bite over its own employment share alone. Reporting the
    group as fully covered would impute a minimum wage to retail and transport
    workers the survey never looked at.
    """
    bite = _bite().set_index("industry")
    assert bite.loc["G-I", "minimum_wage_bite"] == pytest.approx(0.36)
    # Accommodation is 250 of the group's 800 thousand workers.
    assert bite.loc["G-I", "measured_employment_share"] == pytest.approx(250.0 / 800.0)
    assert bite.loc["B-E", "measured_employment_share"] == pytest.approx(1.0)


def test_unknown_activity_in_a_group_is_rejected() -> None:
    """A group naming a section the bite table lacks must fail loudly."""
    broken = {
        **REGISTRY,
        "nace_aggregates": {"G-I": {"sections": ["I"], "measured": ["Nightclubs"]}},
    }
    with pytest.raises(ExposureError, match="unknown activities"):
        activity_bite_from_registry(broken, NATIONAL_EMPLOYMENT)


def test_activity_without_a_section_mapping_is_rejected() -> None:
    """An activity with no NACE section cannot be weighted."""
    broken = {**REGISTRY, "section_of_activity": {"Utilities": "D", "Factories": "C"}}
    with pytest.raises(ExposureError, match="no section"):
        activity_bite_from_registry(broken, NATIONAL_EMPLOYMENT)


def test_different_industry_mixes_give_different_exposure() -> None:
    """This is the construction previously recorded as impossible.

    Both regions face the same national bite, so every difference in exposure
    comes from composition. A tourism-heavy region must score above an
    industry-heavy one when accommodation carries the higher bite.
    """
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    exposure = shift_share_exposure(shares, _bite()).set_index("region")

    assert (
        exposure.loc["PT15", "regional_bite_exposure"]
        > exposure.loc["PT11", "regional_bite_exposure"]
    )
    diagnostic = assess_identifying_variation(
        exposure.reset_index(), value_column="regional_bite_exposure"
    )
    assert diagnostic.identifying
    require_regional_variation(exposure.reset_index(), value_column="regional_bite_exposure")


def test_exposure_reports_how_much_employment_it_covers() -> None:
    """A region scored on a minority of its workers must be visible as such.

    Agriculture is a fifth of each region and unsurveyed, and within
    trade-transport-accommodation only accommodation is surveyed, so the
    tourism-heavy region rests on less measured employment than the industrial
    one despite scoring higher.
    """
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    exposure = shift_share_exposure(shares, _bite()).set_index("region")

    assert exposure["covered_employment_share"].max() < 0.8
    assert (
        exposure.loc["PT15", "covered_employment_share"]
        < exposure.loc["PT11", "covered_employment_share"]
    )


def test_exposure_measured_after_the_shock_is_refused() -> None:
    """A 2017 bite cannot be predetermined for a 2015 policy episode.

    Coverage measured after a rise is partly caused by it, so the estimate would
    carry the outcome inside the regressor while looking entirely healthy.
    """
    with pytest.raises(PredeterminationError, match="not predetermined"):
        check_predetermined(REGISTRY, composition_year=2015, first_shock_year=2015)


def test_exposure_measured_before_the_shock_is_accepted() -> None:
    """The same measure is admissible for a later episode."""
    check_predetermined(REGISTRY, composition_year=2015, first_shock_year=2019)


def test_registry_without_a_reference_period_is_refused() -> None:
    """An undated bite cannot be shown to be predetermined at all."""
    with pytest.raises(PredeterminationError, match="no usable reference_period"):
        check_predetermined({"source": {}}, composition_year=2015, first_shock_year=2019)


def test_partition_is_a_partition() -> None:
    """The configured activity list must not contain nested aggregates."""
    assert {"B-E", "C"} & set(NACE_PARTITION) != {"B-E", "C"}


def test_variation_strength_separates_flat_from_dispersed() -> None:
    """Distinct values are not the same thing as usable variation.

    ``require_regional_variation`` accepts any measure whose regions differ,
    which is a precondition and not a recommendation. The Portuguese measure
    passes it and is still nearly flat, so the strength of the variation has to
    be reported as a number rather than inferred from the guard having passed.
    """
    flat = pd.DataFrame(
        {"region": ["A", "B", "C", "D"], "regional_bite_exposure": [0.200, 0.201, 0.202, 0.203]}
    )
    dispersed = pd.DataFrame(
        {"region": ["A", "B", "C", "D"], "regional_bite_exposure": [0.05, 0.20, 0.35, 0.50]}
    )

    # The guard cannot tell these apart; both have distinct values everywhere.
    for frame in (flat, dispersed):
        require_regional_variation(frame, value_column="regional_bite_exposure")

    weak = measure_variation_strength(flat)
    strong = measure_variation_strength(dispersed)

    assert weak.regions == 4
    assert weak.spread == pytest.approx(0.003)
    assert weak.coefficient_of_variation < 0.01
    assert strong.coefficient_of_variation > 20 * weak.coefficient_of_variation


def test_variation_strength_averages_within_region_before_measuring() -> None:
    """A region appearing on several rows must not count as several regions.

    An exposure frame carrying one row per region-category would otherwise
    report its category count as a region count and understate how few clusters
    the design actually has.
    """
    duplicated = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "regional_bite_exposure": [0.10, 0.30, 0.50, 0.70],
        }
    )
    strength = measure_variation_strength(duplicated)

    assert strength.regions == 2
    assert strength.spread == pytest.approx(0.40)


def test_variation_strength_needs_more_than_one_region() -> None:
    """Spread across a single region is not a quantity."""
    single = pd.DataFrame({"region": ["A"], "regional_bite_exposure": [0.21]})
    with pytest.raises(ExposureError, match="at least two regions"):
        measure_variation_strength(single)


def test_variation_strength_refuses_an_undefined_coefficient() -> None:
    """A zero mean makes the coefficient of variation a division by zero."""
    centred = pd.DataFrame({"region": ["A", "B"], "regional_bite_exposure": [-0.2, 0.2]})
    with pytest.raises(ExposureError, match="coefficient of variation is undefined"):
        measure_variation_strength(centred)


def test_incomplete_national_employment_is_refused_not_defaulted() -> None:
    """A section absent from the weights must fail, not fall back to a default.

    A missing section previously took a weight of one, which is not zero, not
    its true size, and small enough in thousands of workers to look like a
    rounding artefact. Every group containing it would carry a quietly wrong
    bite and a quietly wrong coverage figure, and nothing downstream could
    detect either.
    """
    truncated = NATIONAL_EMPLOYMENT.loc[NATIONAL_EMPLOYMENT["activity"] != "D"]
    with pytest.raises(ExposureError, match=r"missing sections \['D'\]"):
        activity_bite_from_registry(REGISTRY, truncated)


def test_omitting_employment_entirely_still_weights_sections_equally() -> None:
    """The unweighted mode remains available and is not caught by the guard.

    It is correct for a group holding a single section, and the tests for
    partial coverage depend on being able to reach it.
    """
    unweighted = activity_bite_from_registry(REGISTRY).set_index("industry")
    assert unweighted.loc["B-E", "minimum_wage_bite"] == pytest.approx((0.006 + 0.24) / 2)


def _stamped(population: str, year: int = 2015) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A regional and national frame carrying matching provenance stamps."""
    regional = _employment()
    regional["population"] = "SAL" if population == "employees" else "EMP"
    national = NATIONAL_EMPLOYMENT.copy()
    national["population"] = "SAL_DC" if population == "employees" else "EMP_DC"
    national["reference_year"] = year
    return regional, national


def test_matched_inputs_accept_either_population_consistently_used() -> None:
    """Both bases are admissible; mixing them is what is not."""
    for population, expected in (("employees", ("SAL", "SAL_DC")), ("all", ("EMP", "EMP_DC"))):
        regional, national = _stamped(population)
        assert require_matched_inputs(regional, national, baseline_year=2015) == expected


def test_mixed_populations_are_refused() -> None:
    """Employee weights against all-worker composition put coverage on the wrong base.

    The coverage figure is then a fraction of employees reported against a
    denominator of all workers, which is wrong by however much self-employment
    differs between the measured and unmeasured sections.
    """
    regional, _ = _stamped("all")
    _, national = _stamped("employees")
    with pytest.raises(RegionalEmploymentError, match="counts EMP but the weights count SAL_DC"):
        require_matched_inputs(regional, national, baseline_year=2015)


def test_weights_frozen_at_another_year_are_refused() -> None:
    """Half of a measure frozen at 2015 is not a measure frozen at 2015."""
    regional, national = _stamped("employees", year=2019)
    with pytest.raises(RegionalEmploymentError, match=r"weights are for \[2019\]"):
        require_matched_inputs(regional, national, baseline_year=2015)


@pytest.mark.parametrize("dropped", ["population", "reference_year"])
def test_an_unstamped_frame_is_refused_not_warned(dropped: str) -> None:
    """A missing stamp must fail, whatever the other frame carries.

    Conditioning the population check on the year stamp let a half-stamped pair
    through: a national file with its year but no population, beside a regional
    file with neither, satisfied the year check and skipped the denominator
    check entirely.
    """
    regional, national = _stamped("employees")
    with pytest.raises(RegionalEmploymentError, match="predate the provenance stamps"):
        require_matched_inputs(regional, national.drop(columns=[dropped]), baseline_year=2015)


def test_a_frame_mixing_populations_within_itself_is_refused() -> None:
    """One file carrying two populations cannot be assigned a single denominator."""
    regional, national = _stamped("employees")
    regional.loc[regional.index[0], "population"] = "EMP"
    with pytest.raises(RegionalEmploymentError, match="mix populations"):
        require_matched_inputs(regional, national, baseline_year=2015)
