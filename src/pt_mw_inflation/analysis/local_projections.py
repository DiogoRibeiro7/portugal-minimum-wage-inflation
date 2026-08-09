"""Panel local-projection helpers for dynamic price pass-through."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class LocalProjectionEstimate:
    """One horizon-specific local-projection coefficient."""

    horizon: int
    coefficient: float
    standard_error: float
    p_value: float
    observations: int


def estimate_panel_local_projections(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    horizons: list[int],
    entity: str = "region_category",
    time: str = "month",
) -> pd.DataFrame:
    """Estimate a transparent two-way fixed-effect local-projection sequence.

    This baseline implementation is intentionally explicit. Publication specifications
    should add small-cluster inference and robustness estimators before interpretation.
    """
    required = {outcome, shock, entity, time}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.sort_values([entity, time]).copy()
    estimates: list[LocalProjectionEstimate] = []

    for horizon in horizons:
        lead_name = f"__lead_{horizon}"
        ordered[lead_name] = ordered.groupby(entity, observed=True)[outcome].shift(-horizon)
        sample = ordered.dropna(subset=[lead_name, shock])
        if sample.empty:
            continue

        formula = f"{lead_name} ~ {shock} + C({entity}) + C({time})"
        result = smf.ols(formula, data=sample).fit()
        estimates.append(
            LocalProjectionEstimate(
                horizon=horizon,
                coefficient=float(result.params[shock]),
                standard_error=float(result.bse[shock]),
                p_value=float(result.pvalues[shock]),
                observations=int(result.nobs),
            )
        )

    return pd.DataFrame([estimate.__dict__ for estimate in estimates])
