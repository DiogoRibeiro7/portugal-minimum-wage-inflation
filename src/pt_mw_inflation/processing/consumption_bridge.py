"""Compose the production-to-consumption bridge from measurement and judgement.

The region-by-category exposure is

.. math::

    B_{rc} = \\sum_s q_{rs}\\, b_s\\, \\ell_s\\, \\omega_{cs},

and this module builds the last term. Three of the four are published: regional
industry composition from the regional accounts, the national bite from the GEP
monitoring reports, the labour-cost share from national accounts. The bridge is
not, because no source crosses a consumption purpose with a producing industry
for Portugal, and it is therefore built from a published measurement and a
recorded judgement rather than from data alone.

The measurement is household final consumption by CPA product, at basic prices
and domestic uses only, from :mod:`pt_mw_inflation.data.supply_use`. The
judgement is which products serve which consumption purpose, recorded in
``config/consumption_bridge.yaml`` with its reasoning, in the manner of
``config/minimum_wage_bite.yaml``.

One rule in that file decides more than the rest and is implemented here.
Wholesale and retail trade are nearly a fifth of domestic household consumption
at basic prices, and nobody buys retail trade services as such: the margin is
earned on the goods it distributes. Spreading it across categories in
proportion to their goods content is what puts retail employment behind the
shopping basket, and it works *against* the design, because it places a large
common block of one industry into every goods-carrying category, which is
exactly the additive structure the fixed effects absorb. The alternatives are
offered as named variants rather than hidden, since the choice is contestable
and its effect is measurable: the identifying spread runs from 2.00 to 2.88
percentage points across the three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

#: CPA product code prefix to the activity groups the regional accounts publish.
#: The regional accounts are the binding constraint on granularity: ten groups,
#: one of which carries trade, transport and hospitality together, so no
#: concordance can separate a retail margin from a restaurant meal.
NACE_SECTION_GROUPS: dict[str, str] = {
    "A": "A",
    "B": "B-E",
    "C": "B-E",
    "D": "B-E",
    "E": "B-E",
    "F": "F",
    "G": "G-I",
    "H": "G-I",
    "I": "G-I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M_N",
    "N": "M_N",
    "O": "O-Q",
    "P": "O-Q",
    "Q": "O-Q",
    "R": "R-U",
    "S": "R-U",
    "T": "R-U",
    "U": "R-U",
}

#: How the trade margin is allocated across consumption categories.
MARGIN_RULES = ("goods", "uniform", "own")


class ConsumptionBridgeError(RuntimeError):
    """Raised when the bridge cannot be composed from its inputs."""


@dataclass(frozen=True)
class BridgeCoverage:
    """What the composed bridge reached, and what it did not.

    Attributes:
        matched_share: Share of domestic household consumption the concordance
            assigns to some category. Anything below one is consumption the
            bridge cannot place.
        unmatched_products: Products carrying consumption that no category
            claims, largest first.
        margin_rule: How the trade margin was allocated.
    """

    matched_share: float
    unmatched_products: tuple[str, ...]
    margin_rule: str


def activity_group(product: str) -> str:
    """Map a CPA product code to its regional-accounts activity group.

    Args:
        product: CPA code such as ``CPA_C10-12``.

    Returns:
        The activity group, one of the ten the regional accounts publish.

    Raises:
        ConsumptionBridgeError: If the code carries no recognisable NACE
            section, which would otherwise drop the product silently.
    """
    section = product.removeprefix("CPA_")[:1]
    group = NACE_SECTION_GROUPS.get(section)
    if group is None:
        raise ConsumptionBridgeError(
            f"product {product!r} maps to no NACE section; the concordance names a "
            "product this classification does not contain"
        )
    return group


def build_consumption_bridge(
    registry: dict[str, Any],
    consumption: pd.DataFrame,
    *,
    margin_rule: str = "goods",
) -> tuple[pd.DataFrame, BridgeCoverage]:
    """Compose the category-by-industry bridge.

    Args:
        registry: Parsed ``config/consumption_bridge.yaml``.
        consumption: Columns `product` and `consumption_meur`, from
            :func:`pt_mw_inflation.data.supply_use.to_frame`.
        margin_rule: One of :data:`MARGIN_RULES`. ``goods`` distributes the
            trade margin in proportion to each category's goods content, which
            is the documented rule; ``uniform`` spreads it evenly across
            categories; ``own`` leaves it unallocated, as though trade were
            bought in its own right.

    Returns:
        The bridge, with columns `category`, `industry` and `production_weight`
        in the shape :func:`pt_mw_inflation.processing.exposure.validate_bridge`
        expects, and a record of what it covered.

    Raises:
        ConsumptionBridgeError: If the rule is unknown, a required column is
            missing, the registry names a product the table does not carry, or
            a category ends up with no consumption at all.
    """
    if margin_rule not in MARGIN_RULES:
        raise ConsumptionBridgeError(
            f"margin_rule must be one of {MARGIN_RULES}, got {margin_rule!r}"
        )

    missing = {"product", "consumption_meur"}.difference(consumption.columns)
    if missing:
        raise ConsumptionBridgeError(f"consumption missing columns: {sorted(missing)}")

    values = dict(
        zip(
            consumption["product"].astype(str),
            consumption["consumption_meur"].astype(float),
            strict=True,
        )
    )

    allocation = registry["margin_allocation"]
    margin_products = list(allocation["margin_products"])
    goods_prefixes = tuple(allocation["goods_prefixes"])

    unknown = [product for product in margin_products if product not in values]
    if unknown:
        raise ConsumptionBridgeError(f"margin products absent from the use table: {unknown}")

    margin_total = float(sum(values[product] for product in margin_products))
    direct = {
        product: value for product, value in values.items() if product not in set(margin_products)
    }

    categories: dict[str, dict[str, float]] = {}
    goods_value: dict[str, float] = {}
    claimed: set[str] = set(margin_products)

    for code, block in registry["categories"].items():
        weights: dict[str, float] = {}
        goods = 0.0
        for product, share in block["products"].items():
            if product not in values:
                raise ConsumptionBridgeError(
                    f"category {code} names product {product!r}, which the use table "
                    "does not carry; the concordance and the classification disagree"
                )
            claimed.add(product)
            value = float(direct.get(product, 0.0)) * float(share)
            if value <= 0.0:
                continue
            group = activity_group(product)
            weights[group] = weights.get(group, 0.0) + value
            if product.startswith(goods_prefixes):
                goods += value
        categories[str(code)] = weights
        goods_value[str(code)] = goods

    total_goods = sum(goods_value.values())
    for code, weights in categories.items():
        if margin_rule == "goods":
            share = goods_value[code] / total_goods if total_goods else 0.0
        elif margin_rule == "uniform":
            share = 1.0 / len(categories)
        else:
            share = 0.0
        if share:
            weights["G-I"] = weights.get("G-I", 0.0) + margin_total * share

    empty = sorted(code for code, weights in categories.items() if not weights)
    if empty:
        raise ConsumptionBridgeError(
            f"categories {empty} carry no consumption; a bridge row summing to zero "
            "cannot be normalised and would silently drop the category"
        )

    rows = [
        {
            "category": code,
            "industry": industry,
            "production_weight": value / sum(weights.values()),
        }
        for code, weights in sorted(categories.items())
        for industry, value in sorted(weights.items())
    ]
    bridge = pd.DataFrame(rows)

    total = float(sum(values.values()))
    matched = float(sum(values[product] for product in claimed))
    unmatched = sorted(
        (product for product in values if product not in claimed),
        key=lambda product: values[product],
        reverse=True,
    )
    coverage = BridgeCoverage(
        matched_share=matched / total if total else 0.0,
        unmatched_products=tuple(product for product in unmatched if values[product] > 0),
        margin_rule=margin_rule,
    )
    return bridge, coverage
