"""Tests for household consumption by product, the measured half of the bridge.

Two properties of this request decide what the bridge measures, and neither is
visible in the result once it is wrong. Taking the table at purchasers' prices
would credit the retail margin on a good to the industry that made it. Taking
total rather than domestic uses would credit Portuguese employment with costs
incurred abroad. These tests pin both, and the accounting identity that makes
the excluded share meaningful.
"""

from __future__ import annotations

import pytest

from pt_mw_inflation.data.supply_use import (
    DOMESTIC_USES,
    HOUSEHOLD_CONSUMPTION,
    USE_TABLE_DATASET,
    HouseholdConsumption,
    to_frame,
)


def _result(**overrides: object) -> HouseholdConsumption:
    base = {
        "consumption": {"CPA_I": 12_522.0, "CPA_G47": 9_665.0, "CPA_C10-12": 7_200.0},
        "reference_year": 2015,
        "product_total": 91_005.0,
        "purchasers_total": 121_926.0,
        "imported_share": 14_827.0 / 121_926.0,
        "tax_share": 16_094.0 / 121_926.0,
    }
    base.update(overrides)
    return HouseholdConsumption(**base)  # type: ignore[arg-type]


def test_the_request_asks_for_basic_prices_and_domestic_uses() -> None:
    """The two choices that decide what the bridge measures are pinned here.

    `naio_10_cp16` is the purchasers' price table and is the wrong one: it
    records the retail margin on a loaf of bread inside food products, so the
    whole shelf price is attributed to manufacturing rather than to the retail
    employment that earned the margin.
    """
    assert USE_TABLE_DATASET == "naio_10_cp1610"
    assert HOUSEHOLD_CONSUMPTION == "P3_S14"
    assert DOMESTIC_USES == "DOM"


def test_the_excluded_share_is_what_no_concordance_can_reach() -> None:
    """Imports and product taxes have no producing industry behind them.

    This bounds the whole exercise: a quarter of Portuguese household spending
    cannot be attributed to any Portuguese industry, however the concordance is
    written.
    """
    result = _result()

    assert result.excluded_share == pytest.approx(0.2536, abs=1e-3)
    assert result.imported_share + result.tax_share == pytest.approx(result.excluded_share)
    # Domestic content plus what is excluded is the published headline.
    reconstructed = result.product_total + result.excluded_share * result.purchasers_total
    assert reconstructed == pytest.approx(result.purchasers_total, rel=1e-6)


def test_the_frame_carries_shares_that_sum_to_one() -> None:
    """Weights are shares of domestic content, so the denominator travels with them."""
    frame = to_frame(_result(consumption={"CPA_I": 3.0, "CPA_G47": 1.0}, product_total=4.0))

    assert list(frame.columns) == [
        "product",
        "consumption_meur",
        "consumption_share",
        "reference_year",
    ]
    assert frame["consumption_share"].sum() == pytest.approx(1.0)
    assert frame["product"].is_monotonic_increasing


def test_the_frame_is_ordered_so_two_runs_agree() -> None:
    """An unordered frame makes identical runs write different files."""
    frame = to_frame(_result(consumption={"CPA_Z": 1.0, "CPA_A01": 2.0, "CPA_M71": 3.0}))
    assert frame["product"].tolist() == ["CPA_A01", "CPA_M71", "CPA_Z"]
