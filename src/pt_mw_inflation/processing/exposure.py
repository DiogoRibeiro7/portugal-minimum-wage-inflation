"""Construction of predetermined minimum-wage cost exposure measures."""

from __future__ import annotations

import pandas as pd


def construct_cost_exposure(
    bite: pd.DataFrame,
    labour_share: pd.DataFrame,
    consumption_bridge: pd.DataFrame,
) -> pd.DataFrame:
    """Map industry minimum-wage exposure into consumption-category cost exposure.

    Required columns:
      * bite: region, industry, reference_period, minimum_wage_bite
      * labour_share: industry, labour_cost_share
      * consumption_bridge: category, industry, production_weight

    The resulting exposure equals bite × labour share × production-to-consumption
    weight, summed across industries for each region/category/reference period.
    """
    bite_required = {"region", "industry", "reference_period", "minimum_wage_bite"}
    labour_required = {"industry", "labour_cost_share"}
    bridge_required = {"category", "industry", "production_weight"}

    for name, frame, required in (
        ("bite", bite, bite_required),
        ("labour_share", labour_share, labour_required),
        ("consumption_bridge", consumption_bridge, bridge_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    merged = bite.merge(labour_share, on="industry", validate="many_to_one")
    merged = merged.merge(consumption_bridge, on="industry", validate="many_to_many")
    merged["component_exposure"] = (
        merged["minimum_wage_bite"] * merged["labour_cost_share"] * merged["production_weight"]
    )

    # Select with a list, not a bare label, so the aggregation stays a DataFrame.
    aggregated = merged.groupby(["region", "category", "reference_period"], as_index=False)[
        ["component_exposure"]
    ].sum()

    return aggregated.rename(columns={"component_exposure": "cost_exposure"})
