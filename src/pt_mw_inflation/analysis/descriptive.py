"""Transparent descriptive models for the long-run macro layer."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


def inflation_acceleration_regression(frame: pd.DataFrame) -> RegressionResultsWrapper:
    """Regress inflation acceleration on the minimum-wage policy residual.

    This is a descriptive diagnostic, not the primary causal specification.
    Newey-West/HAC covariance is used to reduce sensitivity to serial correlation.
    """
    required = {"inflation", "policy_residual"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    sample = frame.copy()
    sample["inflation_acceleration"] = sample["inflation"].diff()
    sample = sample.dropna(subset=["inflation_acceleration", "policy_residual"])

    exog = sm.add_constant(sample[["policy_residual"]])
    model = sm.OLS(sample["inflation_acceleration"], exog)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": 2})
