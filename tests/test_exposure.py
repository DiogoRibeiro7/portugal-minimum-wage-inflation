"""Tests for the structural exposure bridge."""

from __future__ import annotations

import pandas as pd
import pytest

from pt_mw_inflation.processing.exposure import (
    ExposureDefinition,
    ExposureError,
    apply_policy_shock,
    build_exposure_variants,
    construct_cost_exposure,
    exposure_correlation,
    freeze_baseline_bite,
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
