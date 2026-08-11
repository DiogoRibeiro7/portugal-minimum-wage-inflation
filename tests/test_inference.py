"""Tests for few-cluster inference.

The point of these procedures is that they remain valid where conventional
clustered standard errors do not, so the tests measure that property directly
rather than checking that the functions return numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pt_mw_inflation.analysis.inference import (
    _ClusterProjector,
    clustered_t_statistic,
    holm_adjusted,
    randomization_inference,
    wild_cluster_bootstrap,
)


def _clustered_sample(
    n_clusters: int,
    per_cluster: int,
    effect: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a panel with cluster-level treatment and within-cluster correlation."""
    rng = np.random.default_rng(seed)
    treatment = rng.normal(size=n_clusters)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    regressor = np.repeat(treatment, per_cluster)
    # Within-cluster correlated errors: the reason clustering is needed at all.
    errors = np.repeat(rng.normal(size=n_clusters), per_cluster) + rng.normal(
        scale=0.5, size=n_clusters * per_cluster
    )
    outcome = effect * regressor + errors
    design = np.column_stack([regressor, np.ones(clusters.size)])
    return design, outcome, clusters


def test_bootstrap_has_correct_size_where_clustered_errors_do_not() -> None:
    """With seven clusters the bootstrap holds its nominal size; the t test does not.

    This is the empirical justification for the whole module. Cluster-robust
    inference is asymptotic in the number of clusters, and Portugal has seven
    NUTS II regions, so the conventional test rejects a true null far too often.
    """
    simulations = 120
    rejected_clustered = 0
    rejected_bootstrap = 0

    for simulation in range(simulations):
        design, outcome, clusters = _clustered_sample(7, 30, effect=0.0, seed=500 + simulation)
        _, _, t_statistic = clustered_t_statistic(design, outcome, clusters, 0)
        if abs(t_statistic) > 1.96:
            rejected_clustered += 1
        result = wild_cluster_bootstrap(design, outcome, clusters, target=0, seed=9000 + simulation)
        if result.p_value < 0.05:
            rejected_bootstrap += 1

    size_clustered = rejected_clustered / simulations
    size_bootstrap = rejected_bootstrap / simulations

    # The conventional test must visibly over-reject, and the bootstrap must not.
    assert size_clustered > 0.12, f"expected over-rejection, saw {size_clustered:.3f}"
    assert size_bootstrap < 0.11, f"bootstrap size {size_bootstrap:.3f} is too high"
    assert size_bootstrap < size_clustered


def test_bootstrap_enumerates_the_sign_space_for_few_clusters() -> None:
    """With seven clusters there are 128 sign vectors, so the test is exact."""
    design, outcome, clusters = _clustered_sample(7, 20, effect=0.0, seed=1)
    result = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=9999)
    assert result.exhaustive
    assert result.draws == 2**7


def test_bootstrap_samples_when_the_sign_space_is_large() -> None:
    """With many clusters enumeration is infeasible and draws are sampled."""
    design, outcome, clusters = _clustered_sample(30, 5, effect=0.0, seed=2)
    result = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=199)
    assert not result.exhaustive
    assert result.draws == 199


def test_bootstrap_detects_a_real_effect() -> None:
    """A large effect must still be rejected at conventional levels."""
    design, outcome, clusters = _clustered_sample(7, 60, effect=3.0, seed=3)
    result = wild_cluster_bootstrap(design, outcome, clusters, target=0, seed=4)
    assert result.p_value < 0.05
    assert result.coefficient == pytest.approx(3.0, abs=0.6)


def test_bootstrap_is_deterministic_for_a_given_seed() -> None:
    """Reproducibility: the same inputs and seed give the same p-value."""
    design, outcome, clusters = _clustered_sample(12, 20, effect=0.4, seed=5)
    first = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=99, seed=77)
    second = wild_cluster_bootstrap(design, outcome, clusters, target=0, draws=99, seed=77)
    assert first.p_value == second.p_value


def test_p_value_can_never_be_zero() -> None:
    """The observed sample counts as a draw, so the p-value is bounded below."""
    design, outcome, clusters = _clustered_sample(7, 40, effect=50.0, seed=6)
    result = wild_cluster_bootstrap(design, outcome, clusters, target=0)
    assert result.p_value >= 1.0 / (result.draws + 1)
    assert result.p_value > 0.0


def test_projector_matches_direct_estimation() -> None:
    """The fast path must be numerically identical to the direct computation."""
    design, outcome, clusters = _clustered_sample(6, 40, effect=1.0, seed=8)
    direct = clustered_t_statistic(design, outcome, clusters, 0)
    projected = _ClusterProjector(design, clusters, 0).fit(outcome)
    assert np.allclose(direct, projected, rtol=1e-10)


def test_single_cluster_is_rejected() -> None:
    """Inference on one cluster is not defined and must not be attempted."""
    design, outcome, clusters = _clustered_sample(1, 40, effect=0.0, seed=9)
    with pytest.raises(ValueError, match="at least two clusters"):
        wild_cluster_bootstrap(design, outcome, clusters, target=0)


def test_mismatched_shapes_are_rejected() -> None:
    """Silent broadcasting of misaligned inputs would corrupt the estimate."""
    design, outcome, clusters = _clustered_sample(4, 10, effect=0.0, seed=10)
    with pytest.raises(ValueError, match="same number of rows"):
        wild_cluster_bootstrap(design, outcome[:-1], clusters, target=0)


def test_randomization_inference_recovers_a_null() -> None:
    """Permuting cluster-level treatment gives a large p-value under the null."""
    design, outcome, clusters = _clustered_sample(8, 25, effect=0.0, seed=11)
    result = randomization_inference(design, outcome, clusters, target=0, draws=299)
    assert result.p_value > 0.10
    assert result.method == "randomization_inference"


def test_randomization_inference_rejects_a_strong_effect() -> None:
    """A large cluster-level effect is detected by permutation."""
    design, outcome, clusters = _clustered_sample(10, 40, effect=5.0, seed=12)
    result = randomization_inference(design, outcome, clusters, target=0, draws=299)
    assert result.p_value < 0.05


def test_randomization_requires_cluster_level_treatment() -> None:
    """Permuting a within-cluster-varying regressor would be meaningless."""
    rng = np.random.default_rng(13)
    clusters = np.repeat(np.arange(5), 20)
    design = np.column_stack([rng.normal(size=clusters.size), np.ones(clusters.size)])
    outcome = rng.normal(size=clusters.size)
    with pytest.raises(ValueError, match="constant within clusters"):
        randomization_inference(design, outcome, clusters, target=0, draws=10)


def test_holm_leaves_a_single_test_untouched() -> None:
    """A family of one has nothing to correct for."""
    assert holm_adjusted([0.027]) == pytest.approx([0.027])


def test_holm_scales_the_smallest_by_the_family_size() -> None:
    """The most significant member is compared against the whole family.

    This is the case the regional design turns on: one horizon of seven
    rejects at five per cent, and reporting that p-value alone would treat
    seven tests as though one had been run.
    """
    adjusted = holm_adjusted([0.238, 0.027, 0.301, 0.891, 0.500, 0.367, 0.352])
    assert adjusted[1] == pytest.approx(7 * 0.027)
    assert adjusted[1] > 0.05


def test_holm_is_monotone_in_the_original_ranking() -> None:
    """Once a hypothesis is not rejected, nothing ranked below it may be.

    Without enforced monotonicity the step-down rule can return an adjusted
    value smaller than one assigned to a more significant test, which reads as
    the weaker result being the stronger one.
    """
    adjusted = holm_adjusted([0.01, 0.02, 0.03])
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_holm_never_exceeds_one() -> None:
    """Scaling a large p-value by the family size must not leave probability."""
    assert max(holm_adjusted([0.4, 0.5, 0.9])) == pytest.approx(1.0)


def test_holm_is_at_least_as_large_as_the_raw_value() -> None:
    """A correction that could shrink a p-value would not be a correction."""
    raw = [0.001, 0.2, 0.049, 0.6]
    assert all(a >= r for a, r in zip(holm_adjusted(raw), raw, strict=True))


def test_holm_rejects_an_empty_family() -> None:
    """Adjusting nothing is a caller error, not an empty result."""
    with pytest.raises(ValueError, match="no p-values"):
        holm_adjusted([])
