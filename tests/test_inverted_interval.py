"""Tests for the interval built by inverting the restricted bootstrap test.

The property that matters is agreement. An interval built by resampling around
the estimate declared horizons significant that the null-imposed test does not
reject, which is why that construction was discarded. Inversion cannot disagree
with the test, because it *is* the test, and these tests establish that rather
than assuming it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pt_mw_inflation.analysis.inference import (
    _ClusterProjector,
    _restricted_bootstrap,
    _sign_vectors,
    bootstrap_with_interval,
    invert_bootstrap_interval,
    wild_cluster_bootstrap,
)


def _clustered(seed: int, effect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A panel with cluster-correlated noise and a known effect."""
    rng = np.random.default_rng(seed)
    n_clusters, per_cluster = 8, 30
    clusters = np.repeat(np.arange(n_clusters), per_cluster)

    design = rng.normal(size=(n_clusters * per_cluster, 3))
    design[:, -1] = 1.0
    shocks = rng.normal(size=n_clusters)[clusters]
    outcome = effect * design[:, 0] + shocks + rng.normal(size=len(design))
    return design, outcome, clusters


def test_the_interval_agrees_with_the_test_it_inverts() -> None:
    """Zero lies inside the interval exactly when the test fails to reject.

    This is the property the discarded band lacked. Because inversion tests the
    same null with the same procedure, the interval and the reported p-value
    cannot tell different stories about whether zero is plausible.
    """
    for seed, effect in ((5, 0.0), (6, 1.4)):
        design, outcome, clusters = _clustered(seed, effect)
        p_value = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=199).p_value
        interval = invert_bootstrap_interval(
            design, outcome, clusters, target=0, grid_points=41, draws=199
        )

        covers_zero = interval.lower <= 0.0 <= interval.upper
        assert covers_zero == (p_value > 0.05), (
            f"seed {seed}: p={p_value:.3f} but interval "
            f"[{interval.lower:.3f}, {interval.upper:.3f}]"
        )


def test_the_interval_contains_the_point_estimate() -> None:
    """A candidate equal to the estimate leaves nothing for the test to reject."""
    design, outcome, clusters = _clustered(seed=11, effect=0.8)
    interval = invert_bootstrap_interval(design, outcome, clusters, target=0, draws=199)

    assert interval.lower <= interval.coefficient <= interval.upper


def test_a_real_effect_gives_an_interval_away_from_zero() -> None:
    """A large effect must produce an interval that excludes zero."""
    design, outcome, clusters = _clustered(seed=3, effect=3.0)
    interval = invert_bootstrap_interval(design, outcome, clusters, target=0, draws=199)

    assert interval.lower > 0.0
    assert interval.bounded


def test_a_wider_level_gives_a_wider_interval() -> None:
    """Coverage must be monotone, or the level is not doing anything."""
    design, outcome, clusters = _clustered(seed=9, effect=0.6)
    narrow = invert_bootstrap_interval(
        design, outcome, clusters, target=0, level=0.80, grid_points=61, draws=199
    )
    wide = invert_bootstrap_interval(
        design, outcome, clusters, target=0, level=0.99, grid_points=61, draws=199
    )

    assert wide.lower <= narrow.lower
    assert wide.upper >= narrow.upper


def test_an_interval_running_past_an_exhausted_search_says_so() -> None:
    """Reporting a search limit as an endpoint would understate the uncertainty.

    A very narrow search cannot contain a wide interval, and silently returning
    its own edges would read as precision the data does not support. With no
    expansions left the search is exhausted, and the flag has to show it.
    """
    design, outcome, clusters = _clustered(seed=4, effect=0.0)
    interval = invert_bootstrap_interval(
        design, outcome, clusters, target=0, span=0.05, grid_points=5, expansions=0, draws=99
    )
    assert not interval.bounded


def test_a_narrow_search_widens_until_the_interval_closes() -> None:
    """The starting range is a guess, and a wrong guess must not become the answer.

    The range is quoted in cluster-robust standard errors, which is the statistic
    this whole module exists because it cannot be trusted. Where it most
    understates the uncertainty the interval is widest and the range built from
    it is narrowest, so the search has to be able to correct itself.
    """
    design, outcome, clusters = _clustered(seed=4, effect=0.0)
    exhausted = invert_bootstrap_interval(
        design, outcome, clusters, target=0, span=0.05, grid_points=5, expansions=0, draws=99
    )
    widened = invert_bootstrap_interval(
        design, outcome, clusters, target=0, span=0.05, grid_points=5, draws=99
    )

    assert widened.bounded
    assert widened.lower < exhausted.lower
    assert widened.upper > exhausted.upper


def test_the_interval_at_zero_is_the_reported_test() -> None:
    """The agreement is by shared implementation, not by shared definition.

    A candidate of zero subtracts nothing, so inverting the test at that point
    must run exactly the computation the reported p-value comes from. Testing it
    on a design whose clustered standard error is badly understated is the case
    where a second implementation would have drifted.
    """
    design, outcome, clusters = _clustered(seed=6, effect=1.4)
    reported = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=199, seed=20260814)

    projector = _ClusterProjector(design, np.unique(clusters, return_inverse=True)[1], 0)
    signs, exhaustive = _sign_vectors(8, 199, np.random.default_rng(20260814))
    *_, p_value = _restricted_bootstrap(projector, outcome, signs, exhaustive=exhaustive)

    assert p_value == reported.p_value


def test_an_impossible_level_is_refused() -> None:
    """A level outside (0, 1) is a caller error, not a wide interval."""
    design, outcome, clusters = _clustered(seed=2, effect=0.0)
    with pytest.raises(ValueError, match="level must lie"):
        invert_bootstrap_interval(design, outcome, clusters, target=0, level=1.0, draws=99)


def test_too_few_grid_points_are_refused() -> None:
    """Two candidates cannot describe an interval."""
    design, outcome, clusters = _clustered(seed=2, effect=0.0)
    with pytest.raises(ValueError, match="three grid points"):
        invert_bootstrap_interval(design, outcome, clusters, target=0, grid_points=2, draws=99)


def test_sharing_the_projection_changes_no_number() -> None:
    """One decomposition must give what four separate ones gave.

    The estimator pays for the projection once and reads the estimate, its
    standard error, the p-value and the interval off it. On the
    region-by-category design that is the difference between minutes and an
    hour per horizon, but it is only worth having if it is the same arithmetic,
    so this compares it against the procedures run separately.
    """
    design, outcome, clusters = _clustered(seed=6, effect=1.4)

    combined, bounds = bootstrap_with_interval(
        design, outcome, clusters, target=0, draws=199, seed=4242
    )
    separate = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=199, seed=4242)
    apart = invert_bootstrap_interval(design, outcome, clusters, target=0, draws=199, seed=4242)

    assert combined == separate
    assert bounds == apart


def test_declining_the_interval_still_runs_the_test() -> None:
    """A caller that wants only the p-value must not pay for the search."""
    design, outcome, clusters = _clustered(seed=11, effect=0.8)

    test, bounds = bootstrap_with_interval(
        design, outcome, clusters, target=0, interval=False, draws=199
    )

    assert bounds is None
    assert 0.0 < test.p_value <= 1.0
