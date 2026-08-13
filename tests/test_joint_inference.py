"""Tests for the joint restriction test used as a pre-trend diagnostic.

The efficient path inside :func:`joint_wald_test` never forms the full
cluster-robust covariance, so the first thing these tests establish is that it
agrees with the covariance routine it replaces. An optimisation that silently
changed the statistic would be worse than the slow version it removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pt_mw_inflation.analysis.inference import cluster_robust_covariance, joint_wald_test


def _panel(seed: int, effect: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A clustered panel with an optional effect on the first two regressors."""
    rng = np.random.default_rng(seed)
    n_clusters, per_cluster = 8, 40
    clusters = np.repeat(np.arange(n_clusters), per_cluster)

    design = rng.normal(size=(n_clusters * per_cluster, 4))
    design[:, -1] = 1.0
    shocks = rng.normal(size=n_clusters)[clusters]
    outcome = effect * design[:, 0] + effect * design[:, 1] + shocks + rng.normal(size=len(design))
    return design, outcome, clusters


def test_the_restricted_block_matches_the_full_sandwich() -> None:
    """The fast path must reproduce the covariance block it avoids computing.

    Only the restricted block is needed, so the implementation accumulates
    outer products of projected cluster scores instead of building a covariance
    the size of the design squared. That is an optimisation, and it is only
    legitimate if the number it produces is the same one.
    """
    design, outcome, clusters = _panel(seed=11, effect=0.4)
    selected = [0, 1]

    beta = np.linalg.solve(design.T @ design, design.T @ outcome)
    residuals = outcome - design @ beta
    reference = cluster_robust_covariance(design, residuals, clusters)
    block = reference[np.ix_(selected, selected)]
    restricted = beta[selected]
    expected = float(restricted @ np.linalg.solve(block, restricted))

    observed = joint_wald_test(design, outcome, clusters, targets=selected, draws=2).statistic
    assert observed == pytest.approx(expected, rel=1e-9)


def test_the_test_holds_its_size_under_a_true_null() -> None:
    """Rejection under the null must happen about as often as it is supposed to.

    A single replication cannot establish this: a valid test at five per cent
    rejects one draw in twenty, so asserting a large p-value on one seed is a
    test that fails on schedule. The size is therefore measured over many
    replications, with a bound loose enough not to be flaky and tight enough to
    catch a procedure that rejects indiscriminately.
    """
    rejections = 0
    replications = 40
    for seed in range(replications):
        design, outcome, clusters = _panel(seed=1000 + seed, effect=0.0)
        result = joint_wald_test(design, outcome, clusters, targets=[0, 1], draws=199)
        rejections += int(result.p_value < 0.05)

    assert rejections / replications < 0.20


def test_the_result_describes_the_restriction_it_tested() -> None:
    """The report has to say what was restricted and on how many clusters."""
    design, outcome, clusters = _panel(seed=3, effect=0.0)
    result = joint_wald_test(design, outcome, clusters, targets=[0, 1], draws=199)

    assert result.restrictions == 2
    assert result.clusters == 8
    assert 0.0 <= result.p_value <= 1.0


def test_a_real_joint_effect_is_detected() -> None:
    """A shared effect across the restricted coefficients should reject."""
    design, outcome, clusters = _panel(seed=5, effect=1.5)
    result = joint_wald_test(design, outcome, clusters, targets=[0, 1], draws=299)

    assert result.p_value < 0.05
    assert result.statistic > 0.0


def test_the_test_catches_an_effect_spread_across_restrictions() -> None:
    """This is why the joint form exists rather than reading leads one by one.

    An effect divided between two coefficients can leave neither individually
    striking while the pair is jointly informative.
    """
    rng = np.random.default_rng(17)
    n_clusters, per_cluster = 10, 60
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    design = rng.normal(size=(n_clusters * per_cluster, 3))
    design[:, -1] = 1.0
    outcome = 0.35 * design[:, 0] + 0.35 * design[:, 1] + rng.normal(size=len(design))

    joint = joint_wald_test(design, outcome, clusters, targets=[0, 1], draws=299)
    assert joint.p_value < 0.05


def test_an_empty_restriction_is_refused() -> None:
    """Restricting nothing would report a test of nothing as a pass."""
    design, outcome, clusters = _panel(seed=2)
    with pytest.raises(ValueError, match="at least one restriction"):
        joint_wald_test(design, outcome, clusters, targets=[], draws=99)


def test_mismatched_shapes_are_refused() -> None:
    """Silently truncating to the shortest input would bias the clustering."""
    design, outcome, clusters = _panel(seed=2)
    with pytest.raises(ValueError, match="same number of rows"):
        joint_wald_test(design, outcome[:-5], clusters, targets=[0], draws=99)


def test_the_sign_space_is_enumerated_when_it_is_small() -> None:
    """With few clusters the p-value should carry no simulation error."""
    design, outcome, clusters = _panel(seed=7, effect=0.2)
    result = joint_wald_test(design, outcome, clusters, targets=[0, 1], draws=9999)

    assert result.exhaustive
    assert result.draws == 2**8
