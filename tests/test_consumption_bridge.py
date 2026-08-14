"""Tests for the production-to-consumption bridge.

The bridge is the one term of the region-by-category exposure that is not
published, so it is built from a measured consumption vector and a recorded
concordance. These tests cover the join between the two: that the concordance
cannot silently disagree with the classification, that the trade margin is
allocated as documented, and that the result satisfies the contract the exposure
builder already enforces on a bridge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pt_mw_inflation.processing.consumption_bridge import (
    ConsumptionBridgeError,
    activity_group,
    build_consumption_bridge,
)
from pt_mw_inflation.processing.exposure import validate_bridge

CONFIG = Path(__file__).resolve().parents[1] / "config/consumption_bridge.yaml"


def _registry() -> dict:
    """Read the concordance the pipeline actually uses."""
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _consumption(**overrides: float) -> pd.DataFrame:
    """Build a consumption vector covering every product the concordance names."""
    registry = _registry()
    products = {
        product for block in registry["categories"].values() for product in block["products"]
    }
    products |= set(registry["margin_allocation"]["margin_products"])
    values = dict.fromkeys(sorted(products), 100.0)
    values.update(overrides)
    return pd.DataFrame({"product": list(values), "consumption_meur": [values[p] for p in values]})


def test_the_bridge_satisfies_the_contract_the_exposure_builder_enforces() -> None:
    """Each category's weights must be a distribution over industries.

    `validate_bridge` is what `construct_cost_exposure` checks before using a
    bridge, so building one that fails it would produce a table nothing could
    consume.
    """
    bridge, _ = build_consumption_bridge(_registry(), _consumption())

    sums = validate_bridge(bridge)
    assert len(sums) == 13
    assert set(bridge.columns) == {"category", "industry", "production_weight"}


def test_every_product_the_concordance_names_is_placed() -> None:
    """A product no category claims is consumption the bridge cannot attribute."""
    _, coverage = build_consumption_bridge(_registry(), _consumption())

    assert coverage.unmatched_products == ()
    assert coverage.matched_share == pytest.approx(1.0)


def test_a_product_missing_from_the_use_table_is_refused() -> None:
    """The concordance and the classification must not disagree silently.

    Eurostat revises the product breakdown between vintages. A concordance
    naming a product the table no longer carries would otherwise drop that
    consumption without a word, and the category would be understated by
    exactly the missing mass.
    """
    consumption = _consumption()
    consumption = consumption[consumption["product"] != "CPA_I"]

    with pytest.raises(ConsumptionBridgeError, match="does not carry"):
        build_consumption_bridge(_registry(), consumption)


def test_the_margin_rule_changes_where_trade_lands() -> None:
    """Spreading the margin over goods is what puts retail behind the basket.

    Under the documented rule a goods-heavy category carries trade weight it
    never buys directly. Restaurants buy no margin at all, so their weight is
    the same under every rule, which is what makes them the comparison.
    """
    registry, consumption = _registry(), _consumption()

    def weight(rule: str, category: str, industry: str) -> float:
        bridge, _ = build_consumption_bridge(registry, consumption, margin_rule=rule)
        row = bridge[(bridge["category"] == category) & (bridge["industry"] == industry)]
        return float(row["production_weight"].sum())

    # Clothing buys from manufacturing alone, so any trade weight it carries is
    # margin and nothing else. That it carries some under the documented rule
    # and none when the margin is left unallocated is the whole mechanism.
    assert weight("own", "03", "G-I") == pytest.approx(0.0)
    assert weight("goods", "03", "G-I") > 0.0
    assert weight("uniform", "03", "G-I") > 0.0

    # Restaurants are one product in one section, and no rule touches them.
    for rule in ("goods", "uniform", "own"):
        assert weight(rule, "11", "G-I") == pytest.approx(1.0)


def test_an_unknown_margin_rule_is_refused() -> None:
    """A misspelled rule must not silently fall through to the default."""
    with pytest.raises(ConsumptionBridgeError, match="margin_rule must be"):
        build_consumption_bridge(_registry(), _consumption(), margin_rule="proportional")


def test_a_category_with_no_consumption_is_refused() -> None:
    """A row of zeros cannot be normalised and would drop the category."""
    # Restaurants draw on one product; zeroing it empties the category.
    with pytest.raises(ConsumptionBridgeError, match="carry no consumption"):
        build_consumption_bridge(_registry(), _consumption(CPA_I=0.0))


def test_products_map_to_the_groups_the_regional_accounts_publish() -> None:
    """Trade, transport and hospitality share one group, and that is the constraint.

    The regional accounts publish ten activity groups, so no concordance can
    separate a retail margin from a restaurant meal. Recording it as a test
    rather than a comment because it is the reason the bridge cannot be more
    concentrated than it is.
    """
    assert activity_group("CPA_G47") == "G-I"
    assert activity_group("CPA_I") == "G-I"
    assert activity_group("CPA_H49") == "G-I"
    assert activity_group("CPA_C10-12") == "B-E"
    assert activity_group("CPA_L68A") == "L"


def test_an_unclassifiable_product_is_refused() -> None:
    """A code carrying no NACE section must raise rather than vanish."""
    with pytest.raises(ConsumptionBridgeError, match="maps to no NACE section"):
        activity_group("CPA_9ZZ")


def test_the_concordance_covers_every_consumption_category() -> None:
    """The price panel has thirteen divisions and the bridge must carry all of them.

    A missing division would not fail loudly: the design would simply estimate
    on fewer cells than the panel holds.
    """
    registry = _registry()
    assert sorted(registry["categories"]) == [f"{index:02d}" for index in range(1, 14)]
