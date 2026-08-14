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

#: Relative slack when comparing a bootstrap statistic to the observed one.
#:
#: When the sign space is enumerated it contains the all-positive vector, which
#: rebuilds the original sample: its statistic *is* the observed statistic and
#: has to be counted, which is why the enumerated branch adds no further draw.
#: The two are computed by different routes, so they agree to rounding rather
#: than to the bit, and a bare ``>=`` then settled that draw by which way the
#: last bits fell. Because the draws come in plus/minus pairs, that was worth
#: two draws of the p-value, and on a design where whole clusters contribute no
#: variation each distinct statistic repeats and it was worth more still: eight
#: of 512 at one horizon of the regional design.
#:
#: The value is measured rather than guessed, because the two quantities it has
#: to separate are both observable. Across the fourteen horizons of the two
#: reported designs, the draw that reproduces the observed statistic sits
#: between 7e-12 and 5e-8 of it in relative terms --- the loosest cases being
#: the horizons whose statistic is nearest zero, where the cancellation is
#: worst --- while the closest genuinely distinct draw never comes nearer than
#: 2e-4. Anything inside that band settles the tie by arithmetic rather than by
#: luck, and this sits near the middle of it on a log scale.
#:
#: The band is worth checking again if the designs change: a tolerance below it
#: misses the tie it exists for, which is what a tighter 1e-9 did at the
#: twenty-four-month horizon of the exposure design.
_TIE_TOLERANCE = 1e-6


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

    def restricted_fit(self, outcome: FloatArray) -> FloatArray:
        """Fit the design without the target column, reusing the full projection.

        Imposing the null means fitting the reduced design, and that fit is
        needed afresh for every bootstrap and, when the test is inverted, for
        every candidate value. Decomposing the reduced matrix each time was the
        most expensive thing in this module by a wide margin: on the regional
        design a least-squares solve of it costs twelve seconds, against two
        milliseconds here.

        Frisch-Waugh removes the need for a second decomposition altogether.
        With ``b`` the full coefficient vector and ``x_perp`` the target
        regressor after the other columns are projected out, the fit on the
        reduced design differs from the full fit by exactly the part of the
        target regressor the other columns cannot explain::

            restricted_fit = X @ b - b[target] * x_perp

        and the projector already holds ``x_perp``: the target row of
        ``(X'X)^-1 X'`` is ``x_perp / (x_perp' x_perp)``, which is
        ``self.leverage``. Where the other columns explain the target regressor
        exactly there is nothing to remove and the two fits coincide, which is
        the degenerate case the guard below returns.
        """
        beta = self.projector @ outcome
        fitted = self.design @ beta
        scale = float(self.leverage @ self.leverage)
        if not np.isfinite(scale) or scale <= 0.0:
            return np.asarray(fitted, dtype=np.float64)
        residualised = self.leverage / scale
        return np.asarray(fitted - beta[self.target] * residualised, dtype=np.float64)

    def bootstrap_t_statistics(self, outcome: FloatArray, signs: FloatArray) -> FloatArray:
        """Statistics for every sign vector at once, without a loop over draws.

        The wild *cluster* bootstrap gives each cluster's whole residual block a
        single sign, so however many draws are taken, every resampled outcome is
        a combination of the same ``clusters + 1`` vectors: the restricted fit,
        and the residuals masked to one cluster at a time. Projecting those once
        and recombining is not an approximation to the per-draw loop; it is the
        same arithmetic with the shared work done once.

        That is what makes inverting the test affordable. On the regional design
        the loop projected a matrix of 309 columns once per draw, 512 times for
        every candidate value; this projects it ten times, whatever the number
        of draws.

        Args:
            outcome: Dependent variable the null is imposed on.
            signs: Cluster sign vectors, one row per draw.

        Returns:
            One t statistic per draw, ``nan`` where the variance is unusable.
        """
        fitted = self.restricted_fit(outcome)
        residuals = outcome - fitted

        # The sign space in the coordinates it actually occupies: column g
        # carries cluster g's residuals and zero elsewhere, so the resampled
        # outcome for sign vector s is `fitted + basis @ s`.
        basis = np.zeros((outcome.shape[0], self.n_clusters), dtype=np.float64)
        for column, group in enumerate(self.group_slices):
            basis[group, column] = residuals[group]

        basis_beta = self.projector @ basis
        basis_residual = basis - self.design @ basis_beta

        weighted_basis = self.leverage[:, None] * basis_residual
        loading = np.array(
            [weighted_basis[group].sum(axis=0) for group in self.group_slices], dtype=np.float64
        )

        # The restricted fit contributes nothing to either quantity, and that is
        # definitional rather than approximate: it is the part of the outcome
        # the *other* columns explain, so its coefficient on the target is zero
        # and it leaves no residual against the full design. Recomputing the two
        # numerically and carrying the result is not harmless. The rounding
        # residue breaks the sign symmetry the procedure has, and the draws come
        # in plus/minus pairs whose statistics differ only in sign, so a tie that
        # ought to be exact became a coin flip worth a whole draw of the p-value.
        sign_matrix = np.asarray(signs, dtype=np.float64).T
        scores = loading @ sign_matrix
        variance = self.correction * np.einsum("gd,gd->d", scores, scores)
        coefficients = basis_beta[self.target] @ sign_matrix

        statistics = np.full(coefficients.shape, np.nan, dtype=np.float64)
        usable = np.isfinite(variance) & (variance > 0.0)
        statistics[usable] = coefficients[usable] / np.sqrt(variance[usable])
        return statistics


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


def _prepare(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    target: int,
    draws: int,
    seed: int,
) -> tuple[_ClusterProjector, FloatArray, bool, int]:
    """Validate the inputs and build everything that depends on the design alone.

    Every procedure here needs the same three things and they cost the same to
    build regardless of which one asks: the projection, the cluster partition
    and the sign vectors. On the region-by-category design the projection is a
    pseudo-inverse of a matrix with 2,695 columns and takes about a minute, so
    building it once per horizon rather than once per procedure is the
    difference between a design that can be estimated and one that cannot.

    Args:
        design: Regressor matrix including fixed-effect dummies.
        outcome: Dependent variable, used only to check the shapes agree.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient of interest.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        seed: Seed for reproducibility.

    Returns:
        The projector, the sign vectors, whether they enumerate the space, and
        the number of clusters.

    Raises:
        ValueError: If shapes disagree or fewer than two clusters are present.
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
    signs, exhaustive = _sign_vectors(n_clusters, draws, np.random.default_rng(seed))
    return projector, signs, exhaustive, n_clusters


def _restricted_bootstrap(
    projector: _ClusterProjector,
    outcome: FloatArray,
    signs: FloatArray,
    *,
    exhaustive: bool,
) -> tuple[float, float, float, float]:
    """Run the restricted bootstrap for one outcome against precomputed signs.

    Both the reported test and the interval that inverts it come through here.
    Sharing the implementation rather than the definition is what makes the two
    agree at a candidate of zero: it is not that they compute the same thing, it
    is that they are the same call.

    Args:
        projector: Precomputed machinery for the design being tested.
        outcome: Dependent variable.
        signs: Cluster sign vectors, one row per draw.
        exhaustive: Whether ``signs`` enumerates the space rather than sampling.

    Returns:
        Coefficient, cluster-robust standard error, t statistic and p-value.
    """
    coefficient, standard_error, observed = projector.fit(outcome)
    statistics = projector.bootstrap_t_statistics(outcome, signs)

    finite = statistics[np.isfinite(statistics)]
    if finite.size == 0 or not np.isfinite(observed):
        return coefficient, standard_error, observed, float("nan")

    threshold = abs(observed) * (1.0 - _TIE_TOLERANCE)
    extreme = int(np.sum(np.abs(finite) >= threshold))
    # When the whole sign space is enumerated the observed sample is the
    # all-positive draw already inside it, so adding a further draw would
    # correct for simulation noise that is not present. When draws are
    # sampled, counting the observed sample is what keeps the p-value valid.
    p_value = extreme / finite.size if exhaustive else (extreme + 1) / (finite.size + 1)
    return coefficient, standard_error, observed, min(p_value, 1.0)


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
    projector, signs, exhaustive, n_clusters = _prepare(
        design, outcome, clusters, target=target, draws=draws, seed=seed
    )

    # The null of no effect is imposed inside the projector, by fitting the
    # design without the regressor of interest and resampling around that fit.
    coefficient, standard_error, observed_t, p_value = _restricted_bootstrap(
        projector, np.asarray(outcome, dtype=np.float64), signs, exhaustive=exhaustive
    )

    return ClusterInference(
        coefficient=coefficient,
        standard_error=standard_error,
        t_statistic=observed_t,
        p_value=p_value,
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
    except np.linalg.LinAlgError:
        # A design absorbing three factors is rank deficient by construction:
        # region-time and category-time dummies both span the calendar-month
        # main effects. The pseudo-inverse resolves the redundancy without
        # touching any identified coefficient, so the restricted block and its
        # Wald statistic are unchanged. Reached only on the singular case, so
        # the two-factor designs keep the solve they were computed with.
        projector = np.linalg.pinv(gram) @ design.T

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
    expansions: int = 6,
    draws: int = 999,
    seed: int = 20260814,
) -> InvertedInterval:
    r"""Invert the restricted bootstrap test to obtain an interval.

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
    reported p-value exactly, because it is not merely the same procedure but
    the same call: both go through :func:`_restricted_bootstrap`. The interval
    and the test therefore cannot disagree.

    Every quantity that depends on the design alone --- its projection, the
    cluster partition, and the sign vectors --- is built once and reused across
    candidates, since only the outcome changes from one to the next. Rebuilding
    them per candidate is what previously made this too slow to run in the
    pipeline, at some hundreds of decompositions for a seven-horizon path.

    Args:
        design: Regressor matrix including fixed-effect dummies.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient of interest.
        level: Coverage level.
        grid_points: Candidates tested across the search range.
        span: Half-width of the first search range, in cluster-robust standard
            errors.
        expansions: How many times the range may double when the interval runs
            past it. Zero searches ``span`` alone and reports what it finds.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        seed: Seed for reproducibility.

    Returns:
        The interval, with a flag for whether it is bounded by the grid. An
        unbounded interval means the search was exhausted before the interval
        closed, so the endpoints are the search limits and understate it.

    Raises:
        ValueError: If the level is not a probability, the grid is too small to
            describe an interval, shapes disagree, or fewer than two clusters
            are present.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if grid_points < 3:
        raise ValueError(f"at least three grid points are required, got {grid_points}")

    projector, signs, exhaustive, _ = _prepare(
        design, outcome, clusters, target=target, draws=draws, seed=seed
    )
    return _invert(
        projector,
        np.asarray(design, dtype=np.float64),
        np.asarray(outcome, dtype=np.float64),
        signs,
        exhaustive=exhaustive,
        target=target,
        level=level,
        grid_points=grid_points,
        span=span,
        expansions=expansions,
    )


def _invert(
    projector: _ClusterProjector,
    design: FloatArray,
    outcome: FloatArray,
    signs: FloatArray,
    *,
    exhaustive: bool,
    target: int,
    level: float,
    grid_points: int,
    span: float,
    expansions: int,
) -> InvertedInterval:
    """Search for the candidates the restricted bootstrap does not reject.

    Split out so the interval can be built from a projector that has already
    been paid for, which is what
    :func:`bootstrap_with_interval` does.

    Raises:
        ValueError: If the coefficient has no usable standard error to search
            around, which leaves no scale to build a search range from.
    """
    coefficient, standard_error, _ = projector.fit(outcome)
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("the coefficient has no usable standard error to search around")

    alpha = 1.0 - level
    regressor = design[:, target]

    def scan(half_width: float) -> tuple[list[float], FloatArray]:
        """Test every candidate across one search range."""
        grid = np.linspace(coefficient - half_width, coefficient + half_width, grid_points)
        kept = []
        for candidate in grid:
            # Testing beta = candidate is testing zero once the candidate effect
            # is removed, which is what lets the reported procedure be reused
            # unchanged.
            adjusted = outcome - candidate * regressor
            *_, p_value = _restricted_bootstrap(projector, adjusted, signs, exhaustive=exhaustive)
            if np.isfinite(p_value) and p_value > alpha:
                kept.append(float(candidate))
        return kept, grid

    # The search range is quoted in cluster-robust standard errors because they
    # are the only scale available in advance, but this module exists because
    # that scale cannot be trusted with few clusters, and it fails here in the
    # direction that matters. Where the clustered error most understates the
    # uncertainty, the bootstrap interval is widest and the range built from it
    # is narrowest: on the regional design at impact a t statistic of 8.2 comes
    # with a bootstrap p-value of 0.23, and every candidate within six standard
    # errors survives, so the endpoints returned would have been the edges of
    # the search rather than of the interval.
    #
    # Widening until the interval closes costs one scan each time and settles
    # the scale from the test itself instead of from a statistic it distrusts.
    half_width = span * standard_error
    accepted, candidates = scan(half_width)
    for _ in range(expansions):
        if accepted and min(accepted) > candidates[0] and max(accepted) < candidates[-1]:
            break
        half_width *= 2.0
        accepted, candidates = scan(half_width)

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


def bootstrap_with_interval(
    design: FloatArray,
    outcome: FloatArray,
    clusters: npt.NDArray[np.int_],
    *,
    target: int,
    interval: bool = True,
    level: float = 0.95,
    grid_points: int = 41,
    span: float = 6.0,
    expansions: int = 6,
    draws: int = 999,
    seed: int = 20260809,
) -> tuple[ClusterInference, InvertedInterval | None]:
    """Test one coefficient and bound it, paying for the projection once.

    The point estimate, its cluster-robust standard error, the bootstrap
    p-value and the inverted interval are four answers from one decomposition,
    and computing them separately pays for that decomposition four times. On the
    two-factor designs that was merely wasteful. On the region-by-category
    design, whose matrix carries one dummy per region-month and per
    category-month, it was prohibitive: the pseudo-inverse alone takes about a
    minute, and the least-squares solve the standard error used to come from
    took nearer ten.

    Sharing the projection also removes a subtler problem. The estimate reported
    beside a bootstrap p-value ought to be the estimate that p-value was
    computed from, and running two procedures over the same design left that as
    a coincidence rather than a guarantee.

    Args:
        design: Regressor matrix including fixed-effect dummies.
        outcome: Dependent variable.
        clusters: Integer cluster label per observation.
        target: Column index of the coefficient of interest.
        interval: Whether to invert the test as well as run it.
        level: Coverage level for the interval.
        grid_points: Candidates tested across each search range.
        span: Half-width of the first search range, in standard errors.
        expansions: How many times the range may double.
        draws: Bootstrap draws when the sign space is too large to enumerate.
        seed: Seed shared by the test and the interval, so the two draw the same
            sign vectors and cannot disagree.

    Returns:
        The test, and the interval when it was asked for.

    Raises:
        ValueError: If shapes disagree, fewer than two clusters are present, or
            the interval was asked for on a coefficient with no usable standard
            error to search around.
    """
    if interval:
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        if grid_points < 3:
            raise ValueError(f"at least three grid points are required, got {grid_points}")

    projector, signs, exhaustive, n_clusters = _prepare(
        design, outcome, clusters, target=target, draws=draws, seed=seed
    )
    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)

    coefficient, standard_error, observed_t, p_value = _restricted_bootstrap(
        projector, outcome, signs, exhaustive=exhaustive
    )
    test = ClusterInference(
        coefficient=coefficient,
        standard_error=standard_error,
        t_statistic=observed_t,
        p_value=p_value,
        method="wild_cluster_bootstrap",
        clusters=n_clusters,
        draws=int(signs.shape[0]),
        exhaustive=exhaustive,
    )
    if not interval:
        return test, None

    bounds = _invert(
        projector,
        design,
        outcome,
        signs,
        exhaustive=exhaustive,
        target=target,
        level=level,
        grid_points=grid_points,
        span=span,
        expansions=expansions,
    )
    return test, bounds
