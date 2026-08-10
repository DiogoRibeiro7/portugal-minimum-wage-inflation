"""Tests for regional industry composition and the shift-share exposure.

These cover the construction that an earlier version of this project recorded
as impossible. No test touches the network.
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
    activity_bite_from_registry,
    assess_identifying_variation,
    require_regional_variation,
    shift_share_exposure,
)

REGISTRY = {
    "bite_by_activity": {"Tourism": 36.0, "Utilities": 0.6, "Factories": 24.0},
    "nace_aggregates": {
        "G-I": ["Tourism"],
        "B-E": ["Utilities", "Factories"],
        "A": [],
    },
}


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


def test_shares_sum_to_one_within_each_region() -> None:
    """The partition must reconstruct each region's total exactly."""
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    totals = shares.groupby("region")["employment_share"].sum()
    assert totals.round(9).eq(1.0).all()


def test_overlapping_aggregate_is_rejected() -> None:
    """Eurostat publishes aggregates alongside their parts.

    Summing both double counts most workers, and the resulting shares would be
    silently wrong rather than obviously so, which is why the reconstruction of
    the published total is checked instead of trusted.
    """
    employment = _employment()
    doubled = pd.concat(
        [
            employment,
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
    bite = activity_bite_from_registry(REGISTRY).set_index("industry")
    assert pd.isna(bite.loc["A", "minimum_wage_bite"])
    assert bite.loc["G-I", "minimum_wage_bite"] == pytest.approx(0.36)
    # The group takes the mean of the sections it contains.
    assert bite.loc["B-E", "minimum_wage_bite"] == pytest.approx((0.006 + 0.24) / 2)


def test_unknown_activity_in_a_group_is_rejected() -> None:
    """A group naming a section the bite table lacks must fail loudly."""
    broken = {**REGISTRY, "nace_aggregates": {"G-I": ["Nightclubs"]}}
    with pytest.raises(ExposureError, match="unknown activities"):
        activity_bite_from_registry(broken)


def test_different_industry_mixes_give_different_exposure() -> None:
    """This is the construction previously recorded as impossible.

    Both regions face the same national bite, so every difference in exposure
    comes from composition. A tourism-heavy region must score above an
    industry-heavy one when accommodation carries the higher bite.
    """
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    exposure = shift_share_exposure(shares, activity_bite_from_registry(REGISTRY)).set_index(
        "region"
    )

    assert exposure.loc["PT15", "cost_exposure"] > exposure.loc["PT11", "cost_exposure"]
    diagnostic = assess_identifying_variation(exposure.reset_index(), value_column="cost_exposure")
    assert diagnostic.identifying
    require_regional_variation(exposure.reset_index(), value_column="cost_exposure")


def test_exposure_reports_how_much_employment_it_covers() -> None:
    """A region scored on a minority of its workers must be visible as such.

    Agriculture is outside the survey, so a farming region's exposure rests on
    less of its workforce. Renormalising without reporting the covered share
    would hide that.
    """
    shares = industry_shares(_employment(), year=2015, partition=("G-I", "B-E", "A"))
    exposure = shift_share_exposure(shares, activity_bite_from_registry(REGISTRY))
    # Twenty per cent of each region is agriculture, which has no measured bite.
    assert exposure["covered_employment_share"].round(6).eq(0.8).all()


def test_partition_is_a_partition() -> None:
    """The configured activity list must not contain nested aggregates."""
    nested = {"B-E", "C"} & set(NACE_PARTITION)
    assert nested != {"B-E", "C"}, "partition contains an aggregate and its component"
