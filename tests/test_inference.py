"""Tests for few-cluster inference.

The point of these procedures is that they remain valid where conventional
clustered standard errors do not, so the tests measure that property directly
rather than checking that the functions return numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pt_mw_inflation.analysis.inference import (
    _ClusterProjector,
    _sign_vectors,
    clustered_t_statistic,
    detectable_effects,
    holm_adjusted,
    randomization_inference,
    summarise_run,
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


def test_the_enumerated_bootstrap_distribution_is_exactly_symmetric() -> None:
    """Flipping every sign must flip the statistic and nothing else.

    Under the restricted bootstrap the resampled statistic is odd in the sign
    vector, so an enumerated sign space produces exact plus/minus pairs and half
    as many distinct magnitudes as draws. This is not a decorative property: the
    pair containing the all-positive vector is the one that reproduces the
    observed statistic, and if rounding separates a pair the p-value moves by
    whole draws. Computing the restricted fit's contribution rather than
    imposing the zero it is defined to be did exactly that.
    """
    design, outcome, clusters = _clustered_sample(7, 20, effect=0.5, seed=21)
    _, index = np.unique(clusters, return_inverse=True)
    projector = _ClusterProjector(design, index, 0)
    signs, exhaustive = _sign_vectors(7, 9999, np.random.default_rng(1))
    assert exhaustive

    statistics = projector.bootstrap_t_statistics(outcome, signs)

    # Enumeration pairs each vector with its negation at the mirrored index.
    mirrored = statistics[::-1]
    assert np.array_equal(statistics, -mirrored)
    assert np.unique(np.abs(statistics)).size == statistics.size // 2


def test_the_observed_sample_counts_as_its_own_draw() -> None:
    """The all-positive draw rebuilds the sample, so it cannot be rounded away.

    With every sign positive the resampled outcome is the original outcome, so
    that draw's statistic *is* the observed one and belongs in the tail. The two
    are computed by different routes and agree only to rounding, so a bare
    comparison decided it by the last bits.
    """
    design, outcome, clusters = _clustered_sample(7, 40, effect=50.0, seed=6)
    _, index = np.unique(clusters, return_inverse=True)
    projector = _ClusterProjector(design, index, 0)
    signs, _ = _sign_vectors(7, 9999, np.random.default_rng(1))

    _, _, observed = projector.fit(outcome)
    statistics = projector.bootstrap_t_statistics(outcome, signs)

    # A colossal effect puts every other draw far inside the observed statistic,
    # so the tail is exactly the pair that reproduces it.
    assert abs(statistics[-1]) == pytest.approx(abs(observed), rel=1e-9)
    assert int(np.sum(np.abs(statistics) >= abs(observed) * (1 - 1e-6))) == 2


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


def test_a_run_summary_reports_what_survived_each_correction() -> None:
    """A robustness table needs rejections before and after multiplicity."""
    estimates = pd.DataFrame(
        {
            "horizon": [0, 1, 3],
            "coefficient": [-2.36, 0.49, 0.15],
            "p_value_bootstrap": [0.55, 0.03, 0.93],
            "p_value_bootstrap_holm": [1.0, 0.09, 1.0],
        }
    )
    run = summarise_run("baseline", estimates)

    assert run.horizons == 3
    assert (run.min_coefficient, run.max_coefficient) == pytest.approx((-2.36, 0.49))
    # One horizon rejects raw; none survives the correction across the family.
    assert (run.rejections, run.rejections_holm) == (1, 0)


def test_summarising_nothing_is_an_error_not_an_empty_row() -> None:
    """A specification that estimated nothing must not read as one that found nothing."""
    with pytest.raises(ValueError, match="no estimates"):
        summarise_run("empty", pd.DataFrame())


def test_detectable_effects_scale_the_estimate_into_price_points() -> None:
    """A null is only informative if the design could have seen something.

    Reporting the coefficient beside the smallest response it could have
    rejected a null against is what separates "no pass-through" from "this
    cannot see pass-through".
    """
    estimates = pd.DataFrame({"horizon": [0], "coefficient": [0.5], "standard_error": [0.2]})
    effect = detectable_effects(estimates, contrast=0.02, statutory_rise=0.10)[0]

    # 0.5 x 0.02 x 0.10 = 0.001 in logs, a tenth of a percentage point.
    assert effect.effect_pp == pytest.approx(0.1)
    assert effect.lower_pp < effect.effect_pp < effect.upper_pp
    # The design cannot resolve anything smaller than this, however small the p.
    assert effect.minimum_detectable_pp == pytest.approx(100 * 0.02 * 0.10 * 2.802 * 0.2)


def test_a_zero_exposure_contrast_is_refused() -> None:
    """Scaling by a contrast of zero would report perfect precision."""
    estimates = pd.DataFrame({"horizon": [0], "coefficient": [0.5], "standard_error": [0.2]})
    with pytest.raises(ValueError, match="contrast must be positive"):
        detectable_effects(estimates, contrast=0.0)
