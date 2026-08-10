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
)
from pt_mw_inflation.processing.exposure import (
    ExposureError,
    PredeterminationError,
    activity_bite_from_registry,
    assess_identifying_variation,
    check_predetermined,
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
