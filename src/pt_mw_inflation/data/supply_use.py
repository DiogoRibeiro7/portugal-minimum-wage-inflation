"""Household final consumption by product, from Eurostat's use table.

This is the measured half of the production-to-consumption bridge. The bridge
answers which industries supply each consumption category, and it has two
halves: what households buy from each industry, which is published, and how that
splits across consumption purposes, which is not. This module supplies the
first; `config/consumption_bridge.yaml` records the judgement behind the second.

Two properties of the request decide the answer, and both are arguments rather
than defaults.

**Basic prices, not purchasers' prices.** In the purchasers' price table the
retail margin on a loaf of bread sits inside "food products", so the whole shelf
price is attributed to manufacturing. At basic prices the margin appears where
it is earned, as consumption of retail trade services. The two conventions
describe Portuguese household consumption completely differently: food,
beverages and tobacco are 17.0 per cent of the basket under one and 7.9 under
the other, while wholesale and retail trade go from almost nothing to 18.6.
Retail is minimum-wage intensive and manufacturing much less so, so the choice
moves the measured wage exposure of the shopping basket between industries.

**Domestic uses, not total.** A rise in the Portuguese minimum wage does not
raise the cost of the imported content of a television. Excluding imports is
what keeps a goods-heavy category from being credited with exposure that is
incurred abroad, and it is not a small correction: textiles and apparel are 7.4
per cent of the basket at purchasers' prices including imports, and 2.0 per cent
of domestic content at basic prices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from pt_mw_inflation.data.eurostat_regional import EUROSTAT_API, _decode
from pt_mw_inflation.data.http import USER_AGENT

#: Use table at basic prices. The purchasers' price table is `naio_10_cp16` and
#: is deliberately not used; see the module docstring.
USE_TABLE_DATASET = "naio_10_cp1610"

#: Final consumption expenditure by households.
HOUSEHOLD_CONSUMPTION = "P3_S14"

#: Domestic uses, excluding the imported content of each product.
DOMESTIC_USES = "DOM"

CURRENT_PRICES = "MIO_EUR"

#: Product rows carry this prefix. The column also holds rows that are not
#: products --- taxes on products, imported content, adjustments for purchases
#: by residents abroad --- which belong to the total but have no producing
#: industry and so cannot enter a bridge.
NON_PRODUCT_PREFIX = "CPA_"

#: The all-products total. Note this sums the *product* rows only, not the
#: column: under `stk_flow=DOM` it equals domestic content at basic prices.
PRODUCT_TOTAL = "CPA_TOTAL"

#: Household final consumption expenditure at purchasers' prices, the published
#: headline. It decomposes exactly into domestic content at basic prices, the
#: imported content, and taxes less subsidies on products.
COLUMN_TOTAL = "P2_ADJ"

#: The two rows that make up the difference between the two totals.
IMPORTED_CONTENT = "IMP"
PRODUCT_TAXES = "D21X31"

#: The identity above should hold to rounding in a published table. A wider gap
#: means the three rows are not the decomposition assumed here, and the excluded
#: share would be describing something else.
_IDENTITY_TOLERANCE = 0.005


class SupplyUseError(RuntimeError):
    """Raised when the use-table response cannot be used."""


@dataclass(frozen=True)
class HouseholdConsumption:
    """Domestic household consumption by product, with what it excludes.

    Attributes:
        consumption: CPA product code mapped to million euro at basic prices.
        reference_year: Year the table was taken from.
        product_total: Domestic content at basic prices, summed over products.
            This is the denominator of every bridge weight.
        purchasers_total: Household final consumption at purchasers' prices.
        imported_share: Imported content as a share of that total.
        tax_share: Taxes less subsidies on products, as a share of that total.
        excluded_share: The two together. This bounds what any bridge built from
            this table can attribute to Portuguese employment: a quarter of what
            Portuguese households spend has no Portuguese producing industry
            behind it, and no concordance can change that.
    """

    consumption: dict[str, float]
    reference_year: int
    product_total: float
    purchasers_total: float
    imported_share: float
    tax_share: float

    @property
    def excluded_share(self) -> float:
        """Share of household spending with no producing industry behind it."""
        return self.imported_share + self.tax_share


def fetch_household_consumption(
    *,
    year: int,
    country: str = "PT",
    dataset: str = USE_TABLE_DATASET,
    timeout_seconds: int = 240,
) -> HouseholdConsumption:
    """Fetch domestic household final consumption by CPA product.

    Args:
        year: Reference year of the use table.
        country: Geography code.
        dataset: Eurostat dataset code.
        timeout_seconds: Request timeout. The use table is large and the
            dissemination API answers slowly for it.

    Returns:
        Consumption by product, with the non-product share retained so a caller
        can report what the bridge cannot reach.

    Raises:
        requests.HTTPError: If the request fails.
        SupplyUseError: If the response carries no product rows, or they sum to
            nothing, either of which would make every bridge weight undefined.
    """
    response = requests.get(
        f"{EUROSTAT_API}/{dataset}",
        params={
            "format": "JSON",
            "lang": "EN",
            "geo": country,
            "time": str(year),
            "unit": CURRENT_PRICES,
            "ind_use": HOUSEHOLD_CONSUMPTION,
            "stk_flow": DOMESTIC_USES,
        },
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    frame = _decode(json.loads(response.content))
    frame = frame[frame["value"].notna()]
    if frame.empty:
        raise SupplyUseError(f"{dataset} returned no observations for {country} in {year}")

    products = frame[
        frame["prd_ava"].str.startswith(NON_PRODUCT_PREFIX) & (frame["prd_ava"] != PRODUCT_TOTAL)
    ]
    if products.empty:
        raise SupplyUseError(f"{dataset} carried no CPA product rows for {country} in {year}")

    consumption = dict(
        zip(products["prd_ava"].astype(str), products["value"].astype(float), strict=True)
    )
    product_total = float(sum(consumption.values()))
    if product_total <= 0:
        raise SupplyUseError(f"product rows sum to {product_total}; the bridge would be undefined")

    def row(code: str) -> float:
        return float(frame.loc[frame["prd_ava"] == code, "value"].sum())

    purchasers_total = row(COLUMN_TOTAL)
    if purchasers_total <= 0:
        raise SupplyUseError(f"{dataset} carried no {COLUMN_TOTAL} total for {country} in {year}")

    imported = row(IMPORTED_CONTENT)
    taxes = row(PRODUCT_TAXES)

    # Household consumption at purchasers' prices is domestic content at basic
    # prices, plus the imported content, plus taxes less subsidies on products.
    # Checked rather than assumed: the excluded share below is only meaningful
    # if these three rows are that decomposition, and a silent mismatch would
    # understate how much of the basket the bridge cannot reach.
    residual = abs(product_total + imported + taxes - purchasers_total) / purchasers_total
    if residual > _IDENTITY_TOLERANCE:
        raise SupplyUseError(
            f"domestic content, imports and product taxes miss the published total by "
            f"{100 * residual:.1f}%; the household column is not decomposing as assumed"
        )

    return HouseholdConsumption(
        consumption=consumption,
        reference_year=year,
        product_total=product_total,
        purchasers_total=purchasers_total,
        imported_share=imported / purchasers_total,
        tax_share=taxes / purchasers_total,
    )


def to_frame(result: HouseholdConsumption) -> pd.DataFrame:
    """Tabulate consumption by product for joining onto a concordance.

    Args:
        result: Output of :func:`fetch_household_consumption`.

    Returns:
        Columns `product`, `consumption_meur`, `consumption_share` and
        `reference_year`, sorted by product code.
    """
    frame = pd.DataFrame(
        sorted(result.consumption.items()), columns=["product", "consumption_meur"]
    )
    frame["consumption_share"] = frame["consumption_meur"] / result.product_total
    frame["reference_year"] = result.reference_year
    return frame
