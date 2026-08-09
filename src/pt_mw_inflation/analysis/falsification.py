"""Falsification and robustness checks required before any causal reading.

The research design sets a minimum publication standard: a causal conclusion is
not written unless pre-trend diagnostics are acceptable, small-cluster inference
is implemented, exposure is predetermined, and the result survives alternative
exposure definitions and the exclusion of any single region.

These checks are built to fail loudly. A pass here is not evidence of an effect;
a failure is evidence against one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pt_mw_inflation.analysis.local_projections import (
    estimate_event_study,
    estimate_panel_local_projections,
)


@dataclass(frozen=True)
class PreTrendVerdict:
    """Outcome of the pre-trend diagnostic."""

    passed: bool
    max_absolute_t: float
    failing_leads: tuple[int, ...]
    detail: str


def assess_pre_trends(
    event_study: pd.DataFrame,
    *,
    critical_value: float = 1.96,
) -> PreTrendVerdict:
    """Check that leads of the shock do not predict pre-treatment price growth.

    Args:
        event_study: Output of
            :func:`pt_mw_inflation.analysis.local_projections.estimate_event_study`.
        critical_value: Absolute t above which a lead counts as failing.

    Returns:
        A verdict naming every offending lead. Note the asymmetry of the test:
        passing it does not establish parallel trends, it only fails to refute
        them, and an underpowered design passes trivially.

    Raises:
        ValueError: If the event study is empty.
    """
    if event_study.empty:
        raise ValueError("event study produced no estimates")

    leads = event_study.loc[event_study["event_time"] < 0]
    informative = leads.loc[leads["standard_error"] > 0]
    if informative.empty:
        return PreTrendVerdict(
            passed=False,
            max_absolute_t=float("nan"),
            failing_leads=(),
            detail="no estimable leads; the pre-trend test could not be run",
        )

    absolute_t = informative["t_statistic"].abs()
    failing = tuple(
        int(offset)
        for offset, value in zip(informative["event_time"], absolute_t, strict=True)
        if value > critical_value
    )
    return PreTrendVerdict(
        passed=not failing,
        max_absolute_t=float(absolute_t.max()),
        failing_leads=failing,
        detail=(
            "no lead is individually significant"
            if not failing
            else f"{len(failing)} lead(s) significant at |t|>{critical_value}"
        ),
    )


def leave_one_cluster_out(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    horizons: list[int],
    cluster: str = "region",
    **kwargs: object,
) -> pd.DataFrame:
    """Re-estimate the pass-through function dropping one cluster at a time.

    With seven regions a single one can drive an entire result, and the design
    explicitly requires that the finding not rest on one region or on
    hospitality alone.

    Args:
        frame: Estimation panel.
        outcome: Log price level.
        shock: Exposure shock.
        horizons: Horizons to estimate.
        cluster: Column identifying the cluster to drop.
        **kwargs: Passed through to the estimator.

    Returns:
        Estimates stacked with an `excluded` column, including the full sample
        labelled ``"(none)"`` for comparison.

    Raises:
        ValueError: If fewer than three clusters are present, since dropping one
            would leave too few for inference.
    """
    labels = frame[cluster].dropna().unique()
    if len(labels) < 3:
        raise ValueError(f"leave-one-out needs at least three clusters, found {len(labels)}")

    blocks = []
    for excluded in [None, *sorted(labels)]:
        sample = frame if excluded is None else frame.loc[frame[cluster] != excluded]
        estimates = estimate_panel_local_projections(
            sample,
            outcome=outcome,
            shock=shock,
            horizons=horizons,
            cluster=cluster,
            **kwargs,  # type: ignore[arg-type]
        )
        if estimates.empty:
            continue
        estimates["excluded"] = "(none)" if excluded is None else str(excluded)
        blocks.append(estimates)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def summarise_leave_one_out(results: pd.DataFrame, *, horizon: int) -> dict[str, float]:
    """Quantify how much one cluster moves a horizon's estimate.

    Args:
        results: Output of :func:`leave_one_cluster_out`.
        horizon: Horizon to summarise.

    Returns:
        The full-sample estimate, the range across exclusions, and the largest
        deviation expressed in full-sample standard errors. A deviation beyond
        about one standard error means the result is carried by one region.

    Raises:
        ValueError: If the horizon is absent.
    """
    at_horizon = results.loc[results["horizon"] == horizon]
    if at_horizon.empty:
        raise ValueError(f"horizon {horizon} not present in results")

    full = at_horizon.loc[at_horizon["excluded"] == "(none)"]
    others = at_horizon.loc[at_horizon["excluded"] != "(none)"]
    if full.empty or others.empty:
        raise ValueError("both full-sample and leave-one-out estimates are required")

    baseline = float(full["coefficient"].iloc[0])
    standard_error = float(full["standard_error"].iloc[0])
    deviations = (others["coefficient"] - baseline).abs()

    return {
        "full_sample": baseline,
        "minimum": float(others["coefficient"].min()),
        "maximum": float(others["coefficient"].max()),
        "max_deviation": float(deviations.max()),
        "max_deviation_in_standard_errors": (
            float(deviations.max() / standard_error) if standard_error > 0 else float("nan")
        ),
    }


def placebo_shock_dates(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    horizon: int,
    offsets: list[int],
    entity: str = "region_category",
    time: str = "month",
    cluster: str = "region",
    **kwargs: object,
) -> pd.DataFrame:
    """Re-estimate with the shock moved to dates on which nothing happened.

    A fake implementation date must not reproduce the event profile. Shifting
    the shock forward in time is the sharpest version of the test, because a
    shock dated *after* the outcome cannot cause it: a coefficient that survives
    that shift is measuring something other than the policy.

    Args:
        frame: Estimation panel.
        outcome: Log price level.
        shock: Exposure shock.
        horizon: Horizon to estimate at.
        offsets: Period offsets to apply. Negative values move the shock earlier.
        entity: Region-category identifier.
        time: Calendar period identifier.
        cluster: Clustering level.
        **kwargs: Passed through to the estimator.

    Returns:
        One row per offset, with offset zero being the true dating.
    """
    ordered = frame.sort_values([entity, time]).copy()
    blocks = []

    for offset in offsets:
        placebo = f"__placebo_{offset}"
        ordered[placebo] = ordered.groupby(entity, observed=True)[shock].shift(offset)
        sample = ordered.dropna(subset=[placebo])
        if sample.empty:
            continue
        estimates = estimate_panel_local_projections(
            sample,
            outcome=outcome,
            shock=placebo,
            horizons=[horizon],
            entity=entity,
            time=time,
            cluster=cluster,
            **kwargs,  # type: ignore[arg-type]
        )
        if estimates.empty:
            continue
        estimates["offset"] = offset
        estimates["is_true_dating"] = offset == 0
        blocks.append(estimates)

    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def compare_exposure_definitions(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shocks: list[str],
    horizon: int,
    **kwargs: object,
) -> pd.DataFrame:
    """Estimate the same specification under alternative exposure definitions.

    Args:
        frame: Estimation panel carrying one column per exposure definition.
        outcome: Log price level.
        shocks: Exposure columns to compare.
        horizon: Horizon to estimate at.
        **kwargs: Passed through to the estimator.

    Returns:
        One row per definition, with the sign-agreement flag that the minimum
        publication standard turns on.
    """
    blocks = []
    for shock in shocks:
        estimates = estimate_panel_local_projections(
            frame,
            outcome=outcome,
            shock=shock,
            horizons=[horizon],
            **kwargs,  # type: ignore[arg-type]
        )
        if estimates.empty:
            continue
        estimates["exposure_definition"] = shock
        blocks.append(estimates)

    if not blocks:
        return pd.DataFrame()

    combined = pd.concat(blocks, ignore_index=True)
    signs = np.sign(combined["coefficient"])
    combined["sign_agrees_with_first"] = signs == signs.iloc[0]
    return combined


def run_pre_trend_diagnostic(
    frame: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    **kwargs: object,
) -> tuple[pd.DataFrame, PreTrendVerdict]:
    """Estimate the event study and assess its leads in one call.

    Args:
        frame: Estimation panel.
        outcome: Log price level.
        shock: Exposure shock.
        **kwargs: Passed through to the event-study estimator.

    Returns:
        The event study and its pre-trend verdict.
    """
    event_study = estimate_event_study(frame, outcome=outcome, shock=shock, **kwargs)  # type: ignore[arg-type]
    return event_study, assess_pre_trends(event_study)
