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
    JointTest,
    bootstrap_with_interval,
    clustered_t_statistic,
    holm_adjusted,
    joint_wald_test,
)

FloatArray = npt.NDArray[np.float64]

#: Seed shared by the bootstrap test and the interval that inverts it. They must
#: draw the same sign vectors or the interval could exclude a value the reported
#: test does not reject, which is the defect that sank an earlier construction.
DEFAULT_INFERENCE_SEED = 20260809


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
    #: Ends of the interval obtained by inverting the bootstrap test. A null
    #: p-value says only that zero survives; these say what else does.
    interval_lower: float
    interval_upper: float
    #: Whether the search closed on both endpoints. False means the interval
    #: runs past the widest range searched, so the ends understate it.
    interval_bounded: bool


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
    return build_absorbing_design(frame, shock=shock, factors=[entity, time], controls=controls)


def build_absorbing_design(
    frame: pd.DataFrame,
    *,
    shock: str,
    factors: list[str],
    controls: list[str] | None = None,
) -> tuple[FloatArray, list[str]]:
    """Build a design absorbing any number of factors as explicit dummies.

    Two factors give the panel design; three give the region-by-category design,
    which absorbs region-category, region-time and category-time effects
    together. The third is what lets that design carry a region-time effect, and
    so absorb the tourism and island-supply shocks the regional comparison is
    otherwise exposed to.

    Beyond two factors the design is *rank deficient by construction* and that
    is expected rather than a fault. Region-time and category-time dummies both
    span the calendar-month main effects, so those effects are represented
    twice and dropping one level of each factor cannot remove the overlap. The
    inference machinery uses a pseudo-inverse throughout, which resolves the
    redundancy without changing any identified coefficient, so the shock's
    estimate and its cluster-robust variance are unaffected.

    Args:
        frame: Estimation sample.
        shock: Regressor of interest; it occupies column 0 of the result.
        factors: Categorical columns to absorb, in order.
        controls: Additional regressors.

    Returns:
        The design matrix and its column names.

    Raises:
        ValueError: If no factor is given, since the caller then wants a plain
            regression and should say so.
    """
    if not factors:
        raise ValueError("at least one factor is required to absorb")

    controls = controls or []
    blocks: list[FloatArray] = [frame[[shock, *controls]].to_numpy(dtype=np.float64)]
    names: list[str] = [shock, *controls]

    for factor in factors:
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
    intervals: bool = True,
    seed: int = DEFAULT_INFERENCE_SEED,
    absorb: list[str] | None = None,
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
        intervals: Also invert the bootstrap test to obtain an interval per
            horizon. Worth turning off only where the caller reduces the
            estimates to rejection counts and would discard them anyway.
        seed: Seed shared by the test and the interval that inverts it.
        absorb: Factors to absorb, replacing `entity` and `time`. The
            region-by-category design passes three, which is what lets it carry
            a region-time effect alongside a category-time one.

    Returns:
        One row per horizon with the coefficient, cluster-robust standard error,
        both p-values, and the inverted interval when it was asked for.

    Raises:
        ValueError: If required columns are missing.
    """
    factors = list(absorb) if absorb else [entity, time]
    required = {outcome, shock, entity, time, cluster, *factors}
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

        design, _ = build_absorbing_design(sample, shock=shock, factors=factors, controls=controls)
        outcome_values = sample[response].to_numpy(dtype=np.float64)
        cluster_labels = pd.factorize(sample[cluster])[0]

        # One decomposition answers all of it. The p-value alone would leave the
        # reader unable to tell a design that found nothing from one that could
        # not have found anything, which is the distinction this paper turns on,
        # so the interval is built from the same projection rather than from a
        # second pass over a matrix that can carry thousands of columns.
        bootstrap, interval = bootstrap_with_interval(
            design,
            outcome_values,
            cluster_labels,
            target=0,
            interval=intervals,
            draws=bootstrap_draws,
            seed=seed,
        )

        estimates.append(
            LocalProjectionEstimate(
                horizon=horizon,
                coefficient=bootstrap.coefficient,
                standard_error=bootstrap.standard_error,
                t_statistic=bootstrap.t_statistic,
                p_value_clustered=_normal_two_sided(bootstrap.t_statistic),
                p_value_bootstrap=bootstrap.p_value,
                observations=int(len(sample)),
                clusters=bootstrap.clusters,
                bootstrap_exhaustive=bootstrap.exhaustive,
                interval_lower=interval.lower if interval else float("nan"),
                interval_upper=interval.upper if interval else float("nan"),
                interval_bounded=bool(interval.bounded) if interval else False,
            )
        )

    frame = pd.DataFrame([asdict(estimate) for estimate in estimates])
    if not frame.empty:
        if not intervals:
            # Dropped rather than left as NaN: an absent column is a caller that
            # did not ask, which a reader can see; a NaN column reads as a
            # search that ran and failed.
            frame = frame.drop(columns=["interval_lower", "interval_upper", "interval_bounded"])
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


def assess_pre_trends(
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
    draws: int = 999,
    absorb: list[str] | None = None,
) -> JointTest:
    """Test whether the leads of the shock are jointly zero.

    The falsification battery this project set itself requires that leads of the
    statutory change do not predict pre-treatment inflation. Reading each lead
    off an event-study plot answers that badly: it multiplies the chance one lead
    looks significant, and it misses a trend spread thinly across several leads
    without any one standing out. This restricts them together.

    Args:
        frame: Panel with one row per entity and period.
        outcome: Log price level; the response is its first difference.
        shock: Exposure or statutory shock.
        leads: Pre-periods to include.
        lags: Post-periods to include, which stay unrestricted.
        entity: Region-category identifier.
        time: Calendar period identifier.
        cluster: Clustering level.
        reference_lead: Lead normalised to zero and omitted from the design.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        absorb: Factors to absorb, replacing `entity` and `time`. The test has
            to run under the same effects as the estimate it falsifies, or it
            answers a question about a design nobody reported.

    Returns:
        The joint test over the lead coefficients.

    Raises:
        ValueError: If required columns are missing, or no lead survives the
            sample, which would silently make this a test of nothing.
    """
    factors = list(absorb) if absorb else [entity, time]
    required = {outcome, shock, entity, time, cluster, *factors}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ordered = frame.sort_values([entity, time]).copy()
    grouped = ordered.groupby(entity, observed=True)[outcome]
    ordered["__growth"] = grouped.diff()

    event_times = [offset for offset in range(-leads, lags + 1) if offset != -reference_lead]
    shock_columns: list[str] = []
    for offset in event_times:
        column = f"__event_{offset}"
        ordered[column] = ordered.groupby(entity, observed=True)[shock].shift(-offset)
        shock_columns.append(column)

    sample = ordered.dropna(subset=["__growth", *shock_columns])
    if sample.empty:
        raise ValueError("no observations survive the event window")

    design, _ = build_absorbing_design(
        sample,
        shock=shock_columns[0],
        factors=factors,
        controls=shock_columns[1:],
    )
    outcome_values = sample["__growth"].to_numpy(dtype=np.float64)
    cluster_labels = pd.factorize(sample[cluster])[0]

    # Only the leads are restricted. The lags carry whatever effect exists and
    # must stay free, or the test would reject whenever the policy did anything.
    lead_positions = [index for index, offset in enumerate(event_times) if offset < 0]
    if not lead_positions:
        raise ValueError("no lead terms in the event window; nothing to falsify")

    return joint_wald_test(
        design, outcome_values, cluster_labels, targets=lead_positions, draws=draws
    )
