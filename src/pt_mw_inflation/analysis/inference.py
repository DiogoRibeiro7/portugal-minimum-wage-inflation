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
import pandas as pd

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


@dataclass(frozen=True)
class DetectableEffect:
    """What a design could have found, alongside what it did find.

    Attributes:
        horizon: Horizon the estimate belongs to.
        coefficient: Point estimate.
        effect_pp: Implied differential price response, in percentage points,
            between the most and least exposed region for the stated statutory
            rise.
        lower_pp: Lower end of the 95 per cent interval, same units.
        upper_pp: Upper end of the 95 per cent interval, same units.
        minimum_detectable_pp: Smallest such response the design would reject a
            null against, at 80 per cent power and five per cent size.
    """

    horizon: int
    coefficient: float
    effect_pp: float
    lower_pp: float
    upper_pp: float
    minimum_detectable_pp: float


#: Two-sided 5 per cent critical value plus the 80 per cent power quantile. The
#: conventional multiplier for a minimum detectable effect.
_MDE_MULTIPLIER = 2.802


def detectable_effects(
    estimates: pd.DataFrame, *, contrast: float, statutory_rise: float = 0.10
) -> list[DetectableEffect]:
    """Express each estimate, and the design's resolution, in price points.

    Failing to reject a null is not evidence that the effect is zero. It is
    only informative if the design could have detected an effect worth caring
    about, and whether it could is a property of the standard error and the
    spread of the regressor rather than of the p-value. Reporting the point
    estimate beside the smallest effect the design could have found is what
    separates "no pass-through" from "this cannot see pass-through".

    Args:
        estimates: Horizon estimates carrying `coefficient` and
            `standard_error`.
        contrast: Exposure difference between the most and least exposed
            region, in the same units the coefficient was estimated on.
        statutory_rise: Log statutory increase the effect is quoted for.

    Returns:
        One record per horizon, in percentage points.

    Raises:
        ValueError: If columns are missing, or the contrast is not positive,
            which would make every scaled quantity zero and read as precision.
    """
    missing = {"horizon", "coefficient", "standard_error"}.difference(estimates.columns)
    if missing:
        raise ValueError(f"estimates missing columns: {sorted(missing)}")
    if contrast <= 0:
        raise ValueError(f"exposure contrast must be positive, got {contrast}")

    scale = 100.0 * contrast * statutory_rise
    horizons = estimates["horizon"].astype(int).tolist()
    coefficients = estimates["coefficient"].astype(float).tolist()
    errors = estimates["standard_error"].astype(float).tolist()

    return [
        DetectableEffect(
            horizon=horizon,
            coefficient=coefficient,
            effect_pp=scale * coefficient,
            lower_pp=scale * (coefficient - 1.96 * error),
            upper_pp=scale * (coefficient + 1.96 * error),
            minimum_detectable_pp=scale * _MDE_MULTIPLIER * error,
        )
        for horizon, coefficient, error in zip(horizons, coefficients, errors, strict=True)
    ]


@dataclass(frozen=True)
class RobustnessRun:
    """One specification of a design, and what it found.

    Attributes:
        label: What was varied.
        horizons: Number of horizons estimated.
        min_coefficient: Smallest coefficient across horizons.
        max_coefficient: Largest coefficient across horizons.
        rejections: Horizons significant at five per cent by the bootstrap.
        rejections_holm: Horizons surviving the Holm correction.
    """

    label: str
    horizons: int
    min_coefficient: float
    max_coefficient: float
    rejections: int
    rejections_holm: int


def summarise_run(label: str, estimates: pd.DataFrame) -> RobustnessRun:
    """Reduce one estimated specification to what a robustness table needs.

    A null that holds in the specification its author chose, and nowhere else,
    is not a finding. Reducing every variant to the same few numbers is what
    makes the comparison possible without reproducing seven horizons for each.

    Args:
        label: What this specification varied.
        estimates: Horizon estimates.

    Returns:
        The summary.

    Raises:
        ValueError: If there are no estimates to summarise.
    """
    if estimates.empty:
        raise ValueError(f"{label}: no estimates to summarise")

    holm = (
        int((estimates["p_value_bootstrap_holm"] < 0.05).sum())
        if "p_value_bootstrap_holm" in estimates
        else 0
    )
    return RobustnessRun(
        label=label,
        horizons=len(estimates),
        min_coefficient=float(estimates["coefficient"].min()),
        max_coefficient=float(estimates["coefficient"].max()),
        rejections=int((estimates["p_value_bootstrap"] < 0.05).sum()),
        rejections_holm=holm,
    )


@dataclass(frozen=True)
class JointTest:
    """A joint restriction that several coefficients are simultaneously zero.

    Attributes:
        statistic: Observed Wald statistic.
        p_value: Bootstrap p-value for the joint null.
        restrictions: How many coefficients were restricted.
        clusters: Clusters the inference is based on.
        draws: Sign vectors used.
        exhaustive: Whether the sign space was enumerated rather than sampled.
    """

    statistic: float
    p_value: float
    restrictions: int
    clusters: int
    draws: int
    exhaustive: bool


def joint_wald_test(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    targets: Sequence[int],
    draws: int = 9999,
    seed: int = 20260811,
) -> JointTest:
    """Test that several coefficients are zero together, with few clusters.

    A pre-trend diagnostic that inspects each lead separately answers the wrong
    question twice over. It multiplies the chance that one lead looks
    significant, and it cannot detect a trend that is spread thinly across
    several leads without any single one standing out. The honest test is
    whether the leads are jointly zero.

    The asymptotic chi-squared reference is not usable here for the same reason
    the t reference is not: it is justified in the number of clusters, and there
    are nine. The null is therefore imposed and the Wald statistic is referred to
    its own wild cluster bootstrap distribution.

    Args:
        design: Regressor matrix including fixed-effect dummies.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        targets: Column indices restricted to zero under the null.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        seed: Seed for reproducibility.

    Returns:
        The observed statistic with its bootstrap p-value.

    Raises:
        ValueError: If shapes disagree, fewer than two clusters are present, or
            no restriction is supplied.
    """
    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    clusters = np.asarray(clusters)
    selected = list(targets)

    if not selected:
        raise ValueError("at least one restriction is required")
    if design.shape[0] != outcome.shape[0] or clusters.shape[0] != outcome.shape[0]:
        raise ValueError("design, outcome and clusters must have the same number of rows")

    unique, cluster_index = np.unique(clusters, return_inverse=True)
    n_clusters = unique.size
    if n_clusters < 2:
        raise ValueError("at least two clusters are required")

    # The design is fixed across draws and carries one dummy per region-category
    # and per month, so it has thousands of columns. Two things make the naive
    # loop infeasible and both are avoidable.
    #
    # Only the outcome changes, so the projection onto the coefficients is built
    # once and each draw becomes a matrix-vector product.
    #
    # And the full sandwich would form a covariance the size of the design
    # squared, when only the restricted block is needed. Writing
    # $A^{-1}$ for the bread and $w_g = [A^{-1}]_S X_g^{\top} u_g$ for each
    # cluster's contribution, the block is $\sum_g w_g w_g^{\top}$: a sum of
    # outer products of length equal to the number of restrictions, never
    # touching the large matrix.
    n_obs, n_params = design.shape
    gram = design.T @ design
    try:
        projector = np.linalg.solve(gram, design.T)
    except np.linalg.LinAlgError as error:
        raise ValueError("design is rank deficient; cannot form the projection") from error

    projector_selected = projector[selected]
    cluster_masks = [cluster_index == label for label in range(n_clusters)]
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))

    def wald(values: FloatArray) -> float:
        beta = projector @ values
        residuals = values - design @ beta

        block = np.zeros((len(selected), len(selected)), dtype=np.float64)
        for mask in cluster_masks:
            scores = projector_selected[:, mask] @ residuals[mask]
            block += np.outer(scores, scores)
        block *= correction

        restricted = beta[selected]
        try:
            solved = np.linalg.solve(block, restricted)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(restricted @ solved)

    observed = wald(outcome)

    # The null imposed by dropping the restricted columns, exactly as the
    # single-coefficient bootstrap does. Resampling around an unrestricted fit
    # would not have the same accuracy with this few clusters.
    restricted_design = np.delete(design, selected, axis=1)
    restricted_beta = _ols(restricted_design, outcome)
    restricted_fit = restricted_design @ restricted_beta
    restricted_residuals = outcome - restricted_fit

    rng = np.random.default_rng(seed)
    signs, exhaustive = _sign_vectors(n_clusters, draws, rng)

    bootstrap = np.empty(signs.shape[0], dtype=np.float64)
    for draw, sign_vector in enumerate(signs):
        bootstrap[draw] = wald(restricted_fit + restricted_residuals * sign_vector[cluster_index])

    finite = bootstrap[np.isfinite(bootstrap)]
    if finite.size == 0 or not np.isfinite(observed):
        p_value = float("nan")
    else:
        extreme = int(np.sum(finite >= observed))
        p_value = extreme / finite.size if exhaustive else (extreme + 1) / (finite.size + 1)

    return JointTest(
        statistic=observed,
        p_value=min(p_value, 1.0),
        restrictions=len(selected),
        clusters=n_clusters,
        draws=int(signs.shape[0]),
        exhaustive=exhaustive,
    )


@dataclass(frozen=True)
class InvertedInterval:
    """A confidence interval built by inverting the restricted bootstrap test.

    Attributes:
        coefficient: The point estimate.
        lower: Smallest value in the grid not rejected.
        upper: Largest value in the grid not rejected.
        level: Coverage the interval was built at.
        bounded: Whether both endpoints fell strictly inside the search grid. A
            false value means the interval runs past where we looked, so the
            endpoints are the grid limits rather than the interval's.
        grid_points: How many candidate values were tested.
    """

    coefficient: float
    lower: float
    upper: float
    level: float
    bounded: bool
    grid_points: int


def invert_bootstrap_interval(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    target: int,
    level: float = 0.95,
    grid_points: int = 41,
    span: float = 6.0,
    draws: int = 999,
    seed: int = 20260814,
) -> InvertedInterval:
    """Invert the restricted bootstrap test to obtain an interval.

    The natural way to attach an interval to a bootstrap p-value is to resample
    around the estimate and take quantiles. With this few clusters that performs
    badly, and demonstrably so here: a band built that way declared horizons
    significant that the null-imposed test does not reject. The accuracy of the
    wild cluster bootstrap comes from imposing the null, and an interval that
    abandons the null gives up exactly what makes the procedure trustworthy.

    Inversion keeps it. To test :math:`\beta = \beta_0` the candidate effect is
    subtracted from the outcome and the resulting coefficient is tested against
    zero by the same restricted bootstrap the paper reports. The interval is the
    set of candidates not rejected. At :math:`\beta_0 = 0` this reproduces the
    reported p-value exactly, so the interval and the test cannot disagree.

    Args:
        design: Regressor matrix including fixed-effect dummies.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient of interest.
        level: Coverage level.
        grid_points: Candidates tested across the search range.
        span: Half-width of the search range, in cluster-robust standard errors.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        seed: Seed for reproducibility.

    This is correct and, on a design carrying one dummy per region-category and
    per month, too slow to run in the pipeline as written. Each candidate calls
    the bootstrap afresh, and each call rebuilds the projection of a design with
    thousands of columns, so a seven-horizon path costs some hundreds of
    pseudo-inverses. The fix is the one already applied to
    :func:`joint_wald_test`: build the projection once and reuse it across
    candidates, since only the outcome changes. Until that is done this is used
    on small designs and in tests, and is deliberately not wired into
    ``make paper``.

    Returns:
        The interval, with a flag for whether it is bounded by the grid.

    Raises:
        ValueError: If the level is not a probability, or the grid is too small
            to describe an interval.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if grid_points < 3:
        raise ValueError(f"at least three grid points are required, got {grid_points}")

    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)

    coefficient, standard_error, _ = clustered_t_statistic(design, outcome, clusters, target)
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("the coefficient has no usable standard error to search around")

    alpha = 1.0 - level
    candidates = np.linspace(
        coefficient - span * standard_error,
        coefficient + span * standard_error,
        grid_points,
    )
    regressor = design[:, target]

    accepted: list[float] = []
    for candidate in candidates:
        # Testing beta = candidate is testing zero once the candidate effect is
        # removed, which is what lets the reported procedure be reused unchanged.
        adjusted = outcome - candidate * regressor
        result = wild_cluster_bootstrap(
            design, adjusted, clusters, target=target, draws=draws, seed=seed
        )
        if np.isfinite(result.p_value) and result.p_value > alpha:
            accepted.append(float(candidate))

    if not accepted:
        # Every candidate rejected. Reporting a degenerate interval would read as
        # extreme precision, when it means the grid missed the region entirely.
        return InvertedInterval(
            coefficient=coefficient,
            lower=float("nan"),
            upper=float("nan"),
            level=level,
            bounded=False,
            grid_points=grid_points,
        )

    lower, upper = min(accepted), max(accepted)
    bounded = lower > candidates[0] and upper < candidates[-1]
    return InvertedInterval(
        coefficient=coefficient,
        lower=lower,
        upper=upper,
        level=level,
        bounded=bounded,
        grid_points=grid_points,
    )
