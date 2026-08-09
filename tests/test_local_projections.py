"""Tests for panel local projections and the event study.

The estimator is checked against a panel whose true dynamic response is known
by construction, so these assert recovery of specific numbers rather than that
the code runs.
"""

from __future__ import annotations

import pytest

from pt_mw_inflation.analysis.local_projections import (
    build_two_way_design,
    estimate_event_study,
    estimate_panel_local_projections,
)
from tests.synthetic import PanelSpec, make_panel

HORIZONS = [0, 1, 3, 6]


def test_recovers_the_true_dynamic_response() -> None:
    """Estimated coefficients must reproduce the planted response profile.

    The tolerance is the estimator's own reported precision rather than a fixed
    number: a coefficient is correct if it lands within three standard errors of
    the truth. A fixed absolute bound would either be so loose it tests nothing
    or so tight it fails on sampling noise from one seed to the next.
    """
    spec = PanelSpec(response={0: 0.0, 1: 0.05, 3: 0.15, 6: 0.30}, seed=101)
    estimates = estimate_panel_local_projections(
        make_panel(spec),
        outcome="log_price",
        shock="exposure_shock",
        horizons=HORIZONS,
        bootstrap_draws=99,
    ).set_index("horizon")

    for horizon, truth in spec.response.items():
        coefficient = float(estimates.loc[horizon, "coefficient"])
        standard_error = float(estimates.loc[horizon, "standard_error"])
        deviation = abs(coefficient - truth)
        assert deviation < 3 * standard_error, (
            f"horizon {horizon}: {coefficient:.4f} is {deviation / standard_error:.1f} "
            f"standard errors from the true {truth}"
        )

    # The profile must also be increasing, which is the shape being estimated.
    assert estimates["coefficient"].is_monotonic_increasing


def test_response_is_reported_at_every_requested_horizon() -> None:
    """No horizon may be silently dropped from the pass-through function."""
    estimates = estimate_panel_local_projections(
        make_panel(PanelSpec(seed=102)),
        outcome="log_price",
        shock="exposure_shock",
        horizons=HORIZONS,
        bootstrap_draws=49,
    )
    assert estimates["horizon"].to_list() == HORIZONS
    assert (estimates["clusters"] == 7).all()
    assert (estimates["observations"] > 0).all()


def test_both_p_values_are_reported() -> None:
    """The clustered and bootstrap p-values must both be visible.

    Reporting only the clustered value would overstate significance, which is
    exactly what the research design forbids with few regions.
    """
    estimates = estimate_panel_local_projections(
        make_panel(PanelSpec(response={0: 0.0, 1: 0.0, 3: 0.0, 6: 0.0}, seed=103)),
        outcome="log_price",
        shock="exposure_shock",
        horizons=[1, 6],
        bootstrap_draws=99,
    )
    assert {"p_value_clustered", "p_value_bootstrap"}.issubset(estimates.columns)
    assert (estimates["p_value_bootstrap"] > 0).all()
    assert (estimates["p_value_bootstrap"] <= 1).all()


def test_no_effect_is_not_detected() -> None:
    """A panel with no pass-through must not produce significant estimates."""
    estimates = estimate_panel_local_projections(
        make_panel(PanelSpec(response={1: 0.0, 6: 0.0}, seed=104)),
        outcome="log_price",
        shock="exposure_shock",
        horizons=[1, 6],
        bootstrap_draws=127,
    )
    assert (estimates["p_value_bootstrap"] > 0.05).all()
    assert estimates["coefficient"].abs().max() < 0.02


def test_missing_columns_are_rejected() -> None:
    """A misnamed column must fail loudly rather than estimate on a subset."""
    panel = make_panel(PanelSpec(regions=3, categories=2, periods=40, seed=105))
    with pytest.raises(ValueError, match="Missing columns"):
        estimate_panel_local_projections(
            panel.drop(columns=["region"]),
            outcome="log_price",
            shock="exposure_shock",
            horizons=[1],
        )


def test_design_places_the_shock_first_and_includes_both_fixed_effects() -> None:
    """The bootstrap targets column zero, so the shock must occupy it."""
    panel = make_panel(PanelSpec(regions=3, categories=2, periods=24, seed=106))
    design, names = build_two_way_design(
        panel, shock="exposure_shock", entity="region_category", time="month"
    )
    assert names[0] == "exposure_shock"
    assert names[-1] == "intercept"
    assert any(name.startswith("region_category_") for name in names)
    assert any(name.startswith("month_") for name in names)
    assert design.shape[0] == len(panel)


def test_event_study_normalises_the_reference_period() -> None:
    """The omitted lead is reported with a coefficient of exactly zero."""
    event_study = estimate_event_study(
        make_panel(PanelSpec(regions=5, categories=3, periods=60, seed=107)),
        outcome="log_price",
        shock="exposure_shock",
        leads=3,
        lags=4,
        reference_lead=1,
    )
    reference = event_study.loc[event_study["event_time"] == -1]
    assert len(reference) == 1
    assert float(reference["coefficient"].iloc[0]) == 0.0
    assert event_study["event_time"].is_monotonic_increasing


def test_event_study_covers_the_requested_window() -> None:
    """Every lead and lag asked for must appear exactly once."""
    event_study = estimate_event_study(
        make_panel(PanelSpec(regions=5, categories=3, periods=60, seed=108)),
        outcome="log_price",
        shock="exposure_shock",
        leads=2,
        lags=3,
    )
    assert event_study["event_time"].to_list() == [-2, -1, 0, 1, 2, 3]
    assert not event_study["event_time"].duplicated().any()
