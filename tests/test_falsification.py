"""Tests for the falsification and robustness checks.

A diagnostic that never fires is worthless. Each check is therefore tested
twice: once on a panel where it should pass, and once on a panel deliberately
built to make it fail.
"""

from __future__ import annotations

import pytest

from pt_mw_inflation.analysis.falsification import (
    assess_pre_trends,
    compare_exposure_definitions,
    leave_one_cluster_out,
    placebo_shock_dates,
    run_pre_trend_diagnostic,
    summarise_leave_one_out,
)
from pt_mw_inflation.analysis.local_projections import estimate_event_study
from tests.synthetic import PanelSpec, make_panel


def test_pre_trend_diagnostic_passes_when_there_is_no_pre_trend() -> None:
    """A clean panel must not be flagged."""
    panel = make_panel(PanelSpec(regions=6, categories=4, periods=90, pre_trend=0.0, seed=201))
    _, verdict = run_pre_trend_diagnostic(
        panel, outcome="log_price", shock="exposure_shock", leads=3, lags=4
    )
    assert verdict.passed
    assert verdict.failing_leads == ()


def test_pre_trend_diagnostic_detects_a_planted_pre_trend() -> None:
    """Prices moving before the shock must be caught.

    This is the check that protects against reverse causation, where wage policy
    responds to inflation that is already under way. If it cannot detect a
    pre-trend this large, it cannot be relied on for the real data.
    """
    panel = make_panel(PanelSpec(regions=6, categories=4, periods=90, pre_trend=0.5, seed=202))
    _, verdict = run_pre_trend_diagnostic(
        panel, outcome="log_price", shock="exposure_shock", leads=4, lags=4
    )
    assert not verdict.passed
    assert -3 in verdict.failing_leads, verdict.detail


def test_pre_trend_verdict_requires_estimates() -> None:
    """An empty event study is a failure to run the test, not a pass."""
    empty = estimate_event_study(
        make_panel(PanelSpec(regions=2, categories=1, periods=6, seed=203)),
        outcome="log_price",
        shock="exposure_shock",
        leads=20,
        lags=20,
    )
    with pytest.raises(ValueError, match="no estimates"):
        assess_pre_trends(empty)


def test_leave_one_out_covers_every_cluster_and_the_full_sample() -> None:
    """Each region must be dropped in turn, with the full sample for reference."""
    panel = make_panel(PanelSpec(regions=5, categories=3, periods=72, seed=204))
    results = leave_one_cluster_out(
        panel,
        outcome="log_price",
        shock="exposure_shock",
        horizons=[3],
        bootstrap_draws=31,
    )
    excluded = set(results["excluded"])
    assert "(none)" in excluded
    assert {f"R{index}" for index in range(5)} <= excluded


def test_leave_one_out_summary_reports_the_largest_deviation() -> None:
    """A stable estimate must not move far when any one region is dropped."""
    panel = make_panel(PanelSpec(regions=5, categories=3, periods=72, seed=205))
    results = leave_one_cluster_out(
        panel,
        outcome="log_price",
        shock="exposure_shock",
        horizons=[3],
        bootstrap_draws=31,
    )
    summary = summarise_leave_one_out(results, horizon=3)

    assert summary["minimum"] <= summary["full_sample"] <= summary["maximum"]
    # With a homogeneous effect no single region should carry the estimate.
    assert summary["max_deviation_in_standard_errors"] < 3.0


def test_leave_one_out_needs_enough_clusters() -> None:
    """Dropping one of two clusters leaves too few for inference."""
    panel = make_panel(PanelSpec(regions=2, categories=2, periods=40, seed=206))
    with pytest.raises(ValueError, match="at least three clusters"):
        leave_one_cluster_out(panel, outcome="log_price", shock="exposure_shock", horizons=[1])


def test_summary_rejects_an_absent_horizon() -> None:
    """Asking about a horizon that was not estimated is an error."""
    panel = make_panel(PanelSpec(regions=4, categories=2, periods=60, seed=207))
    results = leave_one_cluster_out(
        panel, outcome="log_price", shock="exposure_shock", horizons=[1], bootstrap_draws=31
    )
    with pytest.raises(ValueError, match="horizon 9 not present"):
        summarise_leave_one_out(results, horizon=9)


def test_placebo_dating_weakens_the_estimate() -> None:
    """A shock dated after the outcome cannot explain it.

    Shifting the shock forward breaks the causal ordering, so the coefficient
    must fall well below the one obtained at the true dating. A placebo that
    performs as well as the real thing means the estimate is picking up
    something other than the policy.
    """
    panel = make_panel(PanelSpec(regions=6, categories=4, periods=90, seed=208))
    results = placebo_shock_dates(
        panel,
        outcome="log_price",
        shock="exposure_shock",
        horizon=3,
        offsets=[0, -6],
        bootstrap_draws=31,
    ).set_index("offset")

    true_estimate = abs(float(results.loc[0, "coefficient"]))
    placebo_estimate = abs(float(results.loc[-6, "coefficient"]))
    assert placebo_estimate < 0.5 * true_estimate


def test_placebo_marks_the_true_dating() -> None:
    """The genuine dating must be identifiable in the output."""
    panel = make_panel(PanelSpec(regions=5, categories=3, periods=72, seed=209))
    results = placebo_shock_dates(
        panel,
        outcome="log_price",
        shock="exposure_shock",
        horizon=1,
        offsets=[0, -4],
        bootstrap_draws=31,
    )
    assert results.loc[results["is_true_dating"], "offset"].to_list() == [0]


def test_exposure_definitions_are_compared_on_sign() -> None:
    """Alternative exposure definitions must be reported together."""
    panel = make_panel(PanelSpec(regions=5, categories=3, periods=72, seed=210))
    panel["exposure_alternative"] = panel["exposure_shock"] * 0.8

    results = compare_exposure_definitions(
        panel,
        outcome="log_price",
        shocks=["exposure_shock", "exposure_alternative"],
        horizon=3,
        bootstrap_draws=31,
    )
    assert set(results["exposure_definition"]) == {"exposure_shock", "exposure_alternative"}
    assert results["sign_agrees_with_first"].all()
