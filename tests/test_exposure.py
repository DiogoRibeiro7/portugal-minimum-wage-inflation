"""Tests for the structural exposure bridge."""

from __future__ import annotations

import pandas as pd
import pytest

from pt_mw_inflation.processing.exposure import (
    ExposureDefinition,
    ExposureError,
    apply_policy_shock,
    assess_identifying_variation,
    build_exposure_variants,
    construct_cost_exposure,
    construct_regional_bite,
    exposure_correlation,
    freeze_baseline_bite,
    require_regional_variation,
    validate_bridge,
)
from tests.synthetic import make_exposure_inputs


def test_exposure_matches_the_hand_computed_value() -> None:
    """The index must equal the sum of bite x labour share x bridge weight."""
    bite, labour_share, bridge = make_exposure_inputs()
    definition = ExposureDefinition(name="b2018", baseline_period=2018)
    result = construct_cost_exposure(bite, labour_share, bridge, definition=definition)

    # Norte, restaurants: 0.40*0.50*0.80 + 0.20*0.25*0.20 = 0.16 + 0.01
    value = result.loc[
        (result["region"] == "Norte") & (result["category"] == "restaurants"), "cost_exposure"
    ].iloc[0]
    assert value == pytest.approx(0.17)

    # Algarve, clothing: 0.55*0.50*0.10 + 0.15*0.25*0.90 = 0.0275 + 0.03375
    value = result.loc[
        (result["region"] == "Algarve") & (result["category"] == "clothing"), "cost_exposure"
    ].iloc[0]
    assert value == pytest.approx(0.06125)


def test_exposure_records_its_baseline() -> None:
    """Every exposure carries the period it was frozen at."""
    bite, labour_share, bridge = make_exposure_inputs()
    result = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("b2018", 2018)
    )
    assert (result["baseline_period"] == 2018).all()
    assert (result["exposure_definition"] == "b2018").all()


def test_bite_is_frozen_at_the_baseline_period() -> None:
    """Later bite values must not leak into a predetermined measure.

    Coverage measured after a wage rise is caused by it, so using it would build
    the outcome into the regressor.
    """
    bite, labour_share, bridge = make_exposure_inputs()
    early = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("b2018", 2018)
    )
    late = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("b2019", 2019)
    )
    merged = early.merge(late, on=["region", "category"], suffixes=("_2018", "_2019"))
    assert (merged["cost_exposure_2018"] != merged["cost_exposure_2019"]).all()


def test_absent_baseline_period_is_rejected() -> None:
    """Silently falling back to another year would break predetermination."""
    bite, labour_share, bridge = make_exposure_inputs()
    with pytest.raises(ExposureError, match="baseline period 1999 absent"):
        freeze_baseline_bite(bite, 1999)


def test_duplicate_baseline_rows_are_rejected() -> None:
    """A region-industry pair may not appear twice in the baseline."""
    bite, _, _ = make_exposure_inputs()
    duplicated = pd.concat([bite, bite.iloc[[0]]], ignore_index=True)
    with pytest.raises(ExposureError, match="duplicate region-industry"):
        freeze_baseline_bite(duplicated, 2018)


def test_bridge_weights_must_sum_to_one() -> None:
    """A category missing industries understates its exposure by that mass."""
    _, _, bridge = make_exposure_inputs()
    incomplete = bridge.loc[
        ~((bridge["category"] == "restaurants") & (bridge["industry"] == "manufacturing"))
    ]
    with pytest.raises(ExposureError, match="must sum to 1"):
        validate_bridge(incomplete)


def test_complete_bridge_passes_validation() -> None:
    """The supplied bridge is a proper distribution over industries."""
    _, _, bridge = make_exposure_inputs()
    sums = validate_bridge(bridge)
    assert set(sums.index) == {"restaurants", "clothing"}
    assert sums.round(9).eq(1.0).all()


def test_shares_given_as_percentages_are_rejected() -> None:
    """A bite of 45 rather than 0.45 would inflate exposure a hundredfold."""
    bite, labour_share, bridge = make_exposure_inputs()
    as_percent = bite.assign(minimum_wage_bite=bite["minimum_wage_bite"] * 100)
    with pytest.raises(ExposureError, match=r"must lie in \[0.0, 1.0\]"):
        construct_cost_exposure(as_percent, labour_share, bridge)


def test_missing_columns_are_named() -> None:
    """The error must say which column is missing."""
    bite, labour_share, bridge = make_exposure_inputs()
    with pytest.raises(ExposureError, match="labour_cost_share"):
        construct_cost_exposure(bite, labour_share.drop(columns=["labour_cost_share"]), bridge)


def test_labour_share_can_be_switched_off() -> None:
    """An unweighted variant is available for robustness."""
    bite, labour_share, bridge = make_exposure_inputs()
    weighted = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("w", 2018, use_labour_share=True)
    )
    unweighted = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("u", 2018, use_labour_share=False)
    )
    merged = weighted.merge(unweighted, on=["region", "category"], suffixes=("_w", "_u"))
    # Labour shares are below one, so weighting can only shrink the index.
    assert (merged["cost_exposure_w"] < merged["cost_exposure_u"]).all()


def test_policy_shock_scales_the_structural_index() -> None:
    """Exposure times the statutory change gives the shift-share shock."""
    bite, labour_share, bridge = make_exposure_inputs()
    exposure = construct_cost_exposure(
        bite, labour_share, bridge, definition=ExposureDefinition("b", 2018)
    )
    policy = pd.DataFrame(
        [
            {"region": "Norte", "period": 2019, "delta_log_minimum_wage": 0.05},
            {"region": "Algarve", "period": 2019, "delta_log_minimum_wage": 0.05},
        ]
    )
    shocked = apply_policy_shock(exposure, policy)
    expected = shocked["cost_exposure"] * 0.05
    pd.testing.assert_series_equal(shocked["exposure_shock"], expected, check_names=False)


def test_variants_are_stacked_and_correlated() -> None:
    """Robustness definitions are produced together and their overlap reported."""
    bite, labour_share, bridge = make_exposure_inputs()
    variants = build_exposure_variants(
        bite,
        labour_share,
        bridge,
        [
            ExposureDefinition("baseline_2018", 2018),
            ExposureDefinition("baseline_2019", 2019),
            ExposureDefinition("unweighted_2018", 2018, use_labour_share=False),
        ],
    )
    assert set(variants["exposure_definition"]) == {
        "baseline_2018",
        "baseline_2019",
        "unweighted_2018",
    }

    correlations = exposure_correlation(variants)
    assert correlations.shape == (3, 3)
    assert correlations.to_numpy().diagonal().round(9).tolist() == [1.0, 1.0, 1.0]


def test_at_least_one_variant_is_required() -> None:
    """An empty robustness set is a configuration error."""
    bite, labour_share, bridge = make_exposure_inputs()
    with pytest.raises(ExposureError, match="at least one exposure definition"):
        build_exposure_variants(bite, labour_share, bridge, [])


def _cross_tab() -> pd.DataFrame:
    """A genuine joint distribution of employment over regions and industries."""
    return pd.DataFrame(
        [
            # Algarve is concentrated in accommodation; Norte in manufacturing.
            {"region": "Algarve", "industry": "accommodation", "employees": 700},
            {"region": "Algarve", "industry": "manufacturing", "employees": 300},
            {"region": "Norte", "industry": "accommodation", "employees": 200},
            {"region": "Norte", "industry": "manufacturing", "employees": 800},
        ]
    )


def _industry_bite() -> pd.DataFrame:
    """National bite by industry, in the range the RMMG report documents."""
    return pd.DataFrame(
        [
            {"industry": "accommodation", "minimum_wage_bite": 0.35},
            {"industry": "manufacturing", "minimum_wage_bite": 0.24},
        ]
    )


def test_regional_bite_is_the_employment_weighted_industry_bite() -> None:
    """The shift-share aggregation must reproduce a hand computation."""
    result = construct_regional_bite(_cross_tab(), _industry_bite()).set_index("region")

    # Algarve: 0.7*0.35 + 0.3*0.24 = 0.245 + 0.072
    assert result.loc["Algarve", "minimum_wage_bite"] == pytest.approx(0.317)
    # Norte: 0.2*0.35 + 0.8*0.24 = 0.07 + 0.192
    assert result.loc["Norte", "minimum_wage_bite"] == pytest.approx(0.262)
    assert result.loc["Algarve", "employment"] == 1000


def test_genuine_cross_tab_yields_identifying_variation() -> None:
    """Different industry mixes must produce different regional exposure."""
    regional = construct_regional_bite(_cross_tab(), _industry_bite())
    diagnostic = assess_identifying_variation(regional, value_column="minimum_wage_bite")
    assert diagnostic.identifying
    assert diagnostic.distinct_values_per_region == 2


def test_marginals_cannot_identify_a_regional_effect() -> None:
    """Employment totals that are only marginals produce no regional variation.

    This is the case the recovered Portuguese sources are actually in: employment
    is published by industry for the whole country, and separately by district
    with no industry detail. Reconstructing the joint distribution from the two
    means assuming region and industry are independent, which makes every
    region's industry mix the national mix, and every region's exposure
    identical. The measure looks populated and plausible and identifies nothing.
    """
    national_mix = {"accommodation": 0.30, "manufacturing": 0.70}
    regional_totals = {"Algarve": 1000, "Norte": 4000, "Centro": 2500}

    # Independence: L_rs = (regional total) x (national industry share).
    from_marginals = pd.DataFrame(
        [
            {"region": region, "industry": industry, "employees": total * share}
            for region, total in regional_totals.items()
            for industry, share in national_mix.items()
        ]
    )

    regional = construct_regional_bite(from_marginals, _industry_bite())

    # Every region gets the same number, to floating-point exactness.
    assert regional["minimum_wage_bite"].round(12).nunique() == 1

    diagnostic = assess_identifying_variation(regional, value_column="minimum_wage_bite")
    assert not diagnostic.identifying
    assert "identical in every region" in diagnostic.detail

    with pytest.raises(ExposureError, match="cannot identify a regional effect"):
        require_regional_variation(regional, value_column="minimum_wage_bite")


def test_regional_bite_rejects_marginal_shaped_input() -> None:
    """A region repeated without industry detail is not a cross-tabulation."""
    duplicated = pd.DataFrame(
        [
            {"region": "Norte", "industry": "accommodation", "employees": 100},
            {"region": "Norte", "industry": "accommodation", "employees": 200},
        ]
    )
    with pytest.raises(ExposureError, match="cross-tabulation"):
        construct_regional_bite(duplicated, _industry_bite())


def test_regional_bite_rejects_negative_employment() -> None:
    """A negative count is a parsing failure, not a small region."""
    broken = _cross_tab()
    broken.loc[0, "employees"] = -5
    with pytest.raises(ExposureError, match="negative counts"):
        construct_regional_bite(broken, _industry_bite())


def test_variation_diagnostic_needs_two_regions() -> None:
    """A single region cannot exhibit between-region variation."""
    single = pd.DataFrame([{"region": "Norte", "cost_exposure": 0.3}])
    with pytest.raises(ExposureError, match="at least two regions"):
        assess_identifying_variation(single)


def test_require_regional_variation_passes_a_healthy_measure() -> None:
    """A measure with real spread is accepted and reports its share."""
    regional = construct_regional_bite(_cross_tab(), _industry_bite())
    diagnostic = require_regional_variation(regional, value_column="minimum_wage_bite")
    assert diagnostic.between_region_share == pytest.approx(1.0)
    assert diagnostic.regions == 2
