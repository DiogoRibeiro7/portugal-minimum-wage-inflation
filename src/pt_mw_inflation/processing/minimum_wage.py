"""Minimum-wage transformations and policy-shock definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"year", "minimum_wage", "inflation", "productivity_growth"}


def build_policy_residual(frame: pd.DataFrame, inflation_lag: int = 1) -> pd.DataFrame:
    """Compute minimum-wage growth and the productivity-plus-inflation residual.

    Args:
        frame: Annual data with `year`, `minimum_wage`, `inflation`, and
            `productivity_growth`. Rates are decimals, not percentage points.
        inflation_lag: Number of years by which CPI inflation enters the wage benchmark.

    Returns:
        Copy of the input with exact growth rates, benchmark growth and policy residual.

    Raises:
        ValueError: If required columns are absent or the minimum wage is non-positive.
    """
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (frame["minimum_wage"] <= 0).any():
        raise ValueError("minimum_wage must be strictly positive")

    result = frame.sort_values("year").copy()
    result["minimum_wage_growth"] = result["minimum_wage"].pct_change()
    result["lagged_inflation"] = result["inflation"].shift(inflation_lag)
    result["benchmark_wage_growth"] = (
        (1.0 + result["productivity_growth"]) * (1.0 + result["lagged_inflation"]) - 1.0
    )
    result["policy_residual"] = (
        result["minimum_wage_growth"] - result["benchmark_wage_growth"]
    )
    result["log_minimum_wage_growth"] = np.log(result["minimum_wage"]).diff()
    return result
