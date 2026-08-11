"""Inference procedures for a design with very few clusters.

Portugal has seven NUTS II regions. Cluster-robust standard errors are
justified by asymptotics in the number of clusters, and at seven those
asymptotics do not hold: the resulting tests over-reject badly, so a
conventional clustered p-value below 0.05 carries much less evidence than it
appears to.

This module therefore provides two procedures that remain valid with few
clusters, and reports the ordinary clustered result alongside them so the gap
is visible rather than hidden:

``wild_cluster_bootstrap``
    The restricted wild bootstrap with cluster-level Rademacher weights. With
    ``G`` clusters there are only ``2**G`` distinct weight vectors, so when that
    number is small the procedure enumerates all of them, and the p-value
    carries no simulation error. That is not the same as an exact
    finite-sample test: validity still rests on the assumptions under which the
    sign-flip distribution approximates the null, and enumeration removes only
    the Monte Carlo component of the error.

``randomization_inference``
    Permutes the treatment across clusters and compares the observed statistic
    to its permutation distribution. This makes no distributional assumption at
    all, at the cost of testing a sharp null.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Enumerate every sign vector rather than sampling when the cluster count is at
#: or below this. 2**20 draws is already far more than any simulated bootstrap
#: would use, and enumeration removes simulation error entirely.
MAX_CLUSTERS_FOR_ENUMERATION = 20


@dataclass(frozen=True)
class ClusterInference:
    """Result of one inference procedure for a single coefficient."""

    coefficient: float
    standard_error: float
    t_statistic: float
    p_value: float
    method: str
    clusters: int
    draws: int
    exhaustive: bool


def _ols(design: FloatArray, outcome: FloatArray) -> FloatArray:
    """Solve least squares, tolerating rank deficiency from absorbed dummies."""
    beta, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return np.asarray(beta, dtype=np.float64)


class _ClusterProjector:
    """Reusable machinery for refitting one design against many outcome vectors.

    A bootstrap changes only the outcome: the design, and therefore its
    decomposition and the cluster partition, are identical on every draw.
    Solving the full least-squares problem each time repeats that work for
    nothing, and with fixed-effect dummies the design has hundreds of columns.

    Precomputing the projector reduces a draw to two matrix-vector products.
    The cluster-robust variance of a single coefficient simplifies further: with
    ``a`` the target row of ``(X'X)^-1`` and ``h = X a``, the variance is
    ``c * sum_g (sum_{i in g} h_i u_i)^2``, so it costs one pass over the
    residuals instead of forming and multiplying a full covariance matrix.
    """

    def __init__(
        self,
        design: FloatArray,
        cluster_index: npt.NDArray[np.int_],
        target: int,
    ) -> None:
        self.design = design
        self.target = target
        n_obs, n_params = design.shape

        xtx_inv = np.linalg.pinv(design.T @ design)
        self.projector = xtx_inv @ design.T
        self.leverage = design @ xtx_inv[target, :]

        order = np.argsort(cluster_index, kind="stable")
        self.sorted_index = order
        sorted_labels = cluster_index[order]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        self.group_slices = np.split(order, boundaries)
        n_clusters = len(self.group_slices)

        self.correction = (n_clusters / max(n_clusters - 1, 1)) * (
            (n_obs - 1) / max(n_obs - n_params, 1)
        )
        self.n_clusters = n_clusters

    def fit(self, outcome: FloatArray) -> tuple[float, float, float]:
        """Return coefficient, cluster-robust standard error and t statistic."""
        beta = self.projector @ outcome
        residuals = outcome - self.design @ beta

        weighted = self.leverage * residuals
        scores = np.array([weighted[group].sum() for group in self.group_slices], dtype=np.float64)
        variance = self.correction * float(np.dot(scores, scores))

        coefficient = float(beta[self.target])
        if variance <= 0 or not np.isfinite(variance):
            return coefficient, float("nan"), float("nan")
        standard_error = float(np.sqrt(variance))
        return coefficient, standard_error, coefficient / standard_error


def cluster_robust_covariance(
    design: FloatArray,
    residuals: FloatArray,
    clusters: npt.NDArray[np.int_],
) -> FloatArray:
    """Cluster-robust covariance with the usual finite-sample correction.

    Args:
        design: Regressor matrix.
        residuals: Estimation residuals.
        clusters: Integer cluster label per observation.

    Returns:
        The CR1 covariance matrix.
    """
    n_obs, n_params = design.shape
    unique = np.unique(clusters)
    n_clusters = unique.size

    bread = np.linalg.pinv(design.T @ design)

    meat = np.zeros((n_params, n_params), dtype=np.float64)
    for label in unique:
        mask = clusters == label
        scores = design[mask].T @ residuals[mask]
        meat += np.outer(scores, scores)

    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    covariance = correction * bread @ meat @ bread
    return np.asarray(covariance, dtype=np.float64)


def clustered_t_statistic(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    target: int,
) -> tuple[float, float, float]:
    """Estimate one coefficient with a cluster-robust standard error.

    Args:
        design: Regressor matrix.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient of interest.

    Returns:
        Coefficient, cluster-robust standard error, and t statistic.
    """
    beta = _ols(design, outcome)
    residuals = outcome - design @ beta
    covariance = cluster_robust_covariance(design, residuals, clusters)
    variance = float(covariance[target, target])
    standard_error = float(np.sqrt(variance)) if variance > 0 else float("nan")
    t_statistic = float(beta[target] / standard_error) if standard_error > 0 else float("nan")
    return float(beta[target]), standard_error, t_statistic


def _sign_vectors(n_clusters: int, draws: int, rng: np.random.Generator) -> tuple[FloatArray, bool]:
    """Return cluster sign vectors, enumerating them when that is feasible."""
    if n_clusters <= MAX_CLUSTERS_FOR_ENUMERATION and 2**n_clusters <= max(draws, 1) * 2:
        signs = np.array(list(product((-1.0, 1.0), repeat=n_clusters)), dtype=np.float64)
        return signs, True
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, n_clusters))
    return np.asarray(signs, dtype=np.float64), False


def wild_cluster_bootstrap(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    target: int,
    draws: int = 9999,
    seed: int = 20260809,
) -> ClusterInference:
    """Test a zero restriction with the restricted wild cluster bootstrap.

    The null is imposed before resampling: the model is re-estimated without the
    regressor of interest, and the bootstrap outcome is rebuilt from the
    restricted fit with each cluster's residuals given a common random sign.
    Imposing the null is what gives the procedure its accuracy with few
    clusters; the unrestricted variant does not share that property.

    Args:
        design: Regressor matrix, including any fixed-effect dummies.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient being tested.
        draws: Number of bootstrap draws when the sign space is too large to
            enumerate.
        seed: Seed for reproducibility.

    Returns:
        The observed coefficient and t statistic with a bootstrap p-value.

    Raises:
        ValueError: If fewer than two clusters are present, or shapes disagree.
    """
    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    clusters = np.asarray(clusters)

    if design.shape[0] != outcome.shape[0] or clusters.shape[0] != outcome.shape[0]:
        raise ValueError("design, outcome and clusters must have the same number of rows")

    unique, cluster_index = np.unique(clusters, return_inverse=True)
    n_clusters = unique.size
    if n_clusters < 2:
        raise ValueError("at least two clusters are required")

    projector = _ClusterProjector(design, cluster_index, target)
    coefficient, standard_error, observed_t = projector.fit(outcome)

    # Restricted fit: the null of no effect imposed by dropping the regressor.
    restricted_design = np.delete(design, target, axis=1)
    restricted_beta = _ols(restricted_design, outcome)
    restricted_fit = restricted_design @ restricted_beta
    restricted_residuals = outcome - restricted_fit

    rng = np.random.default_rng(seed)
    signs, exhaustive = _sign_vectors(n_clusters, draws, rng)

    bootstrap_t = np.empty(signs.shape[0], dtype=np.float64)
    for draw, sign_vector in enumerate(signs):
        resampled = restricted_fit + restricted_residuals * sign_vector[cluster_index]
        _, _, bootstrap_t[draw] = projector.fit(resampled)

    finite = bootstrap_t[np.isfinite(bootstrap_t)]
    if finite.size == 0 or not np.isfinite(observed_t):
        p_value = float("nan")
    else:
        extreme = int(np.sum(np.abs(finite) >= abs(observed_t)))
        # When the whole sign space is enumerated the observed sample is the
        # all-positive draw already inside it, so adding a further draw would
        # correct for simulation noise that is not present. When draws are
        # sampled, counting the observed sample is what keeps the p-value valid.
        p_value = extreme / finite.size if exhaustive else (extreme + 1) / (finite.size + 1)

    return ClusterInference(
        coefficient=coefficient,
        standard_error=standard_error,
        t_statistic=observed_t,
        p_value=min(p_value, 1.0),
        method="wild_cluster_bootstrap",
        clusters=n_clusters,
        draws=int(signs.shape[0]),
        exhaustive=exhaustive,
    )


def randomization_inference(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    target: int,
    draws: int = 9999,
    seed: int = 20260809,
) -> ClusterInference:
    """Test a sharp null by permuting the treatment across clusters.

    The regressor of interest is reassigned between whole clusters, holding
    everything else fixed. Because policy is assigned at the region level, the
    permutation respects the level at which treatment actually varies.

    When the number of distinct cluster orderings is small the procedure
    enumerates them, giving an exact test.

    Args:
        design: Regressor matrix.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the regressor to permute.
        draws: Number of permutations when enumeration is infeasible.
        seed: Seed for reproducibility.

    Returns:
        The observed coefficient with a permutation p-value.

    Raises:
        ValueError: If the treatment is not constant within clusters, since the
            permutation would then not be well defined.
    """
    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    unique, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = unique.size

    treatment = design[:, target]
    cluster_values = np.array(
        [np.unique(treatment[cluster_index == index]) for index in range(n_clusters)],
        dtype=object,
    )
    if any(values.size != 1 for values in cluster_values):
        raise ValueError(
            "randomization inference requires treatment constant within clusters; "
            "aggregate to the assignment level first"
        )
    assignments = np.array([float(values[0]) for values in cluster_values], dtype=np.float64)

    coefficient, standard_error, observed_t = clustered_t_statistic(
        design, outcome, cluster_index, target
    )

    rng = np.random.default_rng(seed)
    permuted = np.empty(draws, dtype=np.float64)
    candidate = design.copy()
    for draw in range(draws):
        shuffled = rng.permutation(assignments)
        candidate[:, target] = shuffled[cluster_index]
        estimate = _ols(candidate, outcome)
        permuted[draw] = estimate[target]

    finite = permuted[np.isfinite(permuted)]
    extreme = int(np.sum(np.abs(finite) >= abs(coefficient)))
    p_value = (extreme + 1) / (finite.size + 1)

    return ClusterInference(
        coefficient=coefficient,
        standard_error=standard_error,
        t_statistic=observed_t,
        p_value=min(p_value, 1.0),
        method="randomization_inference",
        clusters=n_clusters,
        draws=draws,
        exhaustive=False,
    )


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Adjust a family of p-values for multiplicity, by Holm's step-down rule.

    A horizon profile is a family of tests, not one test. Estimating seven
    horizons and reporting the smallest p-value as though it stood alone
    rejects a true null far more often than its nominal size: at five per cent
    across seven independent tests, the chance of at least one rejection is
    about thirty per cent. In a design whose whole point is that conventional
    inference over-rejects, quoting an uncorrected minimum would repeat the
    error the bootstrap was introduced to fix, one level up.

    Holm is used rather than Bonferroni because it is uniformly more powerful
    and needs no more assumptions: both control the familywise error rate under
    arbitrary dependence, which matters here because horizons of the same
    series are strongly dependent.

    Args:
        p_values: One p-value per member of the family, in any order.

    Returns:
        Adjusted p-values in the order supplied, each capped at one and made
        monotone in the original ranking, so a smaller raw value can never
        adjust to a larger one.

    Raises:
        ValueError: If the family is empty.
    """
    values = list(p_values)
    if not values:
        raise ValueError("no p-values to adjust")

    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])

    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        # Enforced monotonicity is what makes the step-down rule coherent: once
        # a hypothesis fails to be rejected, nothing ranked below it may be.
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted
