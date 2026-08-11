"""Panel local projections and event-study diagnostics for price pass-through.

The estimating equation is the one in the research design:

    Delta_h log P[r,c,t+h] = alpha[r,c] + lambda[t] + beta_h E[r,c,t]
                             + Gamma X[r,c,t] + error[r,c,t+h]

with region-category and calendar-time fixed effects. The sequence of beta_h is
the dynamic pass-through function.

Every horizon is reported with both a conventional cluster-robust standard error
and a wild-cluster-bootstrap p-value. The two are kept side by side deliberately:
with seven regions the conventional p-value over-rejects severely, and showing
only the smaller number would misrepresent the strength of the evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from pt_mw_inflation.analysis.inference import (
    clustered_t_statistic,
    holm_adjusted,
    wild_cluster_bootstrap,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LocalProjectionEstimate:
    """One horizon-specific local-projection coefficient."""

    horizon: int
    coefficient: float
    standard_error: float
    t_statistic: float
    p_value_clustered: float
    p_value_bootstrap: float
    observations: int
    clusters: int
    #: Whether the bootstrap enumerated the whole sign space rather than
    #: sampling it. Recorded rather than inferred from the cluster count,
    #: because the manuscript states the $p$-value is exact and that claim
    #: has to come from the run that produced it.
    bootstrap_exhaustive: bool


def build_two_way_design(
    frame: pd.DataFrame,
    *,
    shock: str,
    entity: str,
    time: str,
    controls: list[str] | None = None,
) -> tuple[FloatArray, list[str]]:
    """Build a design matrix with entity and calendar-time fixed effects.

    Fixed effects enter as explicit dummies rather than by demeaning, so the
    matrix that is bootstrapped is exactly the matrix that was estimated. The
    intercept is omitted and one level of each factor is dropped, which keeps
    the design full rank.

    Args:
        frame: Estimation sample.
        shock: Regressor of interest; it occupies column 0 of the result.
        entity: Region-category identifier.
        time: Calendar period identifier.
        controls: Additional regressors.

    Returns:
        The design matrix and its column names.
    """
    controls = controls or []
    blocks: list[FloatArray] = [frame[[shock, *controls]].to_numpy(dtype=np.float64)]
    names: list[str] = [shock, *controls]

    for factor in (entity, time):
        dummies = pd.get_dummies(frame[factor], prefix=factor, drop_first=True, dtype=float)
        blocks.append(dummies.to_numpy(dtype=np.float64))
        names.extend(dummies.columns.astype(str).tolist())

    intercept = np.ones((len(frame), 1), dtype=np.float64)
    return np.hstack([*blocks, intercept]), [*names, "intercept"]


def estimate_panel_local_projections(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    horizons: list[int],
    entity: str = "region_category",
    time: str = "month",
    cluster: str = "region",
    controls: list[str] | None = None,
    bootstrap_draws: int = 999,
    cumulative: bool = True,
) -> pd.DataFrame:
    """Estimate the dynamic pass-through function.

    Args:
        frame: Panel with one row per entity and period.
        outcome: Log price level. Horizon responses are built from it.
        shock: Exposure shock.
        horizons: Horizons in periods, as configured in `config/analysis.yaml`.
        entity: Region-category identifier.
        time: Calendar period identifier.
        cluster: Level at which policy is assigned; inference clusters on it.
        controls: Additional regressors.
        bootstrap_draws: Bootstrap draws per horizon when the sign space is too
            large to enumerate.
        cumulative: Estimate the cumulative response from ``t-1`` to ``t+h``.
            When false, the response at ``t+h`` alone is estimated.

    Returns:
        One row per horizon with the coefficient, cluster-robust standard error,
        and both p-values.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {outcome, shock, entity, time, cluster}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.sort_values([entity, time]).copy()
    grouped = ordered.groupby(entity, observed=True)[outcome]
    baseline = grouped.shift(1) if cumulative else grouped.shift(0)

    estimates: list[LocalProjectionEstimate] = []

    for horizon in horizons:
        response = f"__response_{horizon}"
        ordered[response] = grouped.shift(-horizon) - baseline

        sample = ordered.dropna(subset=[response, shock, *(controls or [])])
        if sample.empty or sample[cluster].nunique() < 2:
            continue

        design, _ = build_two_way_design(
            sample, shock=shock, entity=entity, time=time, controls=controls
        )
        outcome_values = sample[response].to_numpy(dtype=np.float64)
        cluster_labels = pd.factorize(sample[cluster])[0]

        coefficient, standard_error, t_statistic = clustered_t_statistic(
            design, outcome_values, cluster_labels, 0
        )
        bootstrap = wild_cluster_bootstrap(
            design, outcome_values, cluster_labels, target=0, draws=bootstrap_draws
        )

        estimates.append(
            LocalProjectionEstimate(
                horizon=horizon,
                coefficient=coefficient,
                standard_error=standard_error,
                t_statistic=t_statistic,
                p_value_clustered=_normal_two_sided(t_statistic),
                p_value_bootstrap=bootstrap.p_value,
                observations=int(len(sample)),
                clusters=bootstrap.clusters,
                bootstrap_exhaustive=bootstrap.exhaustive,
            )
        )

    frame = pd.DataFrame([asdict(estimate) for estimate in estimates])
    if not frame.empty:
        # The horizons are estimated together and reported together, so they are
        # one family of tests. Adjusting here rather than at the reporting stage
        # keeps the correction attached to the set it was computed over: a later
        # caller that drops horizons would otherwise adjust for a family larger
        # than the one it shows.
        frame["p_value_bootstrap_holm"] = holm_adjusted(frame["p_value_bootstrap"].tolist())
    return frame


def _normal_two_sided(t_statistic: float) -> float:
    """Two-sided normal p-value, reported only for comparison."""
    if not np.isfinite(t_statistic):
        return float("nan")
    # Normal survival function without a SciPy dependency in the hot path.
    from math import erfc, sqrt

    return float(erfc(abs(t_statistic) / sqrt(2.0)))


def estimate_event_study(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    leads: int = 6,
    lags: int = 12,
    entity: str = "region_category",
    time: str = "month",
    cluster: str = "region",
    reference_lead: int = 1,
) -> pd.DataFrame:
    """Estimate leads and lags of the shock for pre-trend diagnostics.

    The design requires that leads of the statutory shock do not predict
    pre-treatment inflation. Estimating them jointly with the lags, against a
    normalised reference period, is what makes that testable.

    Args:
        frame: Panel with one row per entity and period.
        outcome: Log price level; the outcome is its first difference.
        shock: Exposure shock.
        leads: Number of pre-periods to estimate.
        lags: Number of post-periods to estimate.
        entity: Region-category identifier.
        time: Calendar period identifier.
        cluster: Clustering level.
        reference_lead: Lead normalised to zero and omitted from the design.

    Returns:
        One row per event time with the coefficient and cluster-robust standard
        error. The reference period is present with a coefficient of exactly
        zero, so plots do not silently drop it.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {outcome, shock, entity, time, cluster}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.sort_values([entity, time]).copy()
    ordered["__growth"] = ordered.groupby(entity, observed=True)[outcome].diff()

    event_times = [offset for offset in range(-leads, lags + 1) if offset != -reference_lead]
    shock_columns: list[str] = []
    for offset in event_times:
        column = f"__event_{offset}"
        ordered[column] = ordered.groupby(entity, observed=True)[shock].shift(offset)
        shock_columns.append(column)

    sample = ordered.dropna(subset=["__growth", *shock_columns])
    if sample.empty:
        return pd.DataFrame(columns=["event_time", "coefficient", "standard_error", "t_statistic"])

    design, _ = build_two_way_design(
        sample,
        shock=shock_columns[0],
        entity=entity,
        time=time,
        controls=shock_columns[1:],
    )
    outcome_values = sample["__growth"].to_numpy(dtype=np.float64)
    cluster_labels = pd.factorize(sample[cluster])[0]

    rows = []
    for index, offset in enumerate(event_times):
        coefficient, standard_error, t_statistic = clustered_t_statistic(
            design, outcome_values, cluster_labels, index
        )
        rows.append(
            {
                "event_time": offset,
                "coefficient": coefficient,
                "standard_error": standard_error,
                "t_statistic": t_statistic,
            }
        )

    rows.append(
        {
            "event_time": -reference_lead,
            "coefficient": 0.0,
            "standard_error": 0.0,
            "t_statistic": 0.0,
        }
    )
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
