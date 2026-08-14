"""Tests for the region-by-category exposure and the design that carries it.

The measure exists because of a reasoning error recorded in the decision log:
that national terms cannot create regional variation. They can, through their
product with regional composition, and what identifies the coefficient is the
non-additive part of the resulting matrix. These tests pin that property
directly rather than checking that the function returns numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pt_mw_inflation.analysis.local_projections import build_absorbing_design
from pt_mw_inflation.processing.exposure import ExposureError, structural_exposure
from pt_mw_inflation.processing.pass_through import (
    PassThroughError,
    add_structural_interaction,
)


def _shares() -> pd.DataFrame:
    """Two regions with sharply different composition across three activities."""
    return pd.DataFrame(
        {
            "region": ["PT11", "PT11", "PT11", "PT15", "PT15", "PT15"],
            "activity": ["G-I", "B-E", "O-Q", "G-I", "B-E", "O-Q"],
            "employment_share": [0.2, 0.5, 0.3, 0.6, 0.1, 0.3],
        }
    )


def _bite() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "industry": ["G-I", "B-E", "O-Q"],
            "minimum_wage_bite": [0.226, 0.253, 0.161],
            "measured_employment_share": [1.0, 1.0, 1.0],
        }
    )


def _labour() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activity": ["G-I", "B-E", "O-Q"],
            "labour_cost_share": [0.49, 0.453, 0.783],
        }
    )


def _bridge() -> pd.DataFrame:
    """Two categories drawing on different activities."""
    return pd.DataFrame(
        {
            "category": ["01", "01", "11", "11"],
            "industry": ["G-I", "B-E", "G-I", "O-Q"],
            "production_weight": [0.5, 0.5, 0.9, 0.1],
        }
    )


def test_national_terms_produce_region_by_category_variation() -> None:
    """The claim the decision log records as wrong, tested directly.

    Every term but composition is national, and the earlier reasoning was that
    national factors therefore cannot vary the shock across region-category
    cells. They can: the product does, and what survives region-time and
    category-time effects is the non-additive part of it.
    """
    exposure, coverage = structural_exposure(_shares(), _bite(), _labour(), _bridge())

    matrix = exposure.pivot(index="region", columns="category", values="structural_exposure")
    values = matrix.to_numpy(dtype=float)
    residual = (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )

    assert values.shape == (2, 2)
    # Not additively separable: something survives both sets of effects.
    assert np.abs(residual).max() > 0
    assert coverage.identifying_spread > 0


def test_an_additively_separable_bridge_leaves_nothing() -> None:
    """The converse, which is what makes the test above meaningful.

    Give every category the same industry mix and the matrix becomes a product
    of a row effect and a column effect. Region-time and category-time effects
    then absorb it entirely, and the design has nothing to estimate.
    """
    flat = pd.DataFrame(
        {
            "category": ["01", "01", "11", "11"],
            "industry": ["G-I", "B-E", "G-I", "B-E"],
            "production_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    _, coverage = structural_exposure(_shares(), _bite(), _labour(), flat)

    assert coverage.identifying_spread == pytest.approx(0.0, abs=1e-12)


def test_an_activity_missing_a_term_contributes_nothing_and_is_named() -> None:
    """Agriculture has no bite and real estate no labour share; neither is imputed.

    Both absences are real. The survey excludes agriculture, and real estate
    value added is imputed rent, which employs nobody. Treating them as zero is
    the conservative reading, and which activities it applied to has to be
    reported or the measure looks more complete than it is.
    """
    shares = pd.concat(
        [
            _shares(),
            pd.DataFrame(
                {
                    "region": ["PT11", "PT11", "PT15", "PT15"],
                    "activity": ["A", "L", "A", "L"],
                    "employment_share": [0.1, 0.05, 0.2, 0.05],
                }
            ),
        ]
    )
    # Agriculture is surveyed by nobody, so it has a labour share and no bite.
    # Real estate has a bite and no labour share, its value added being imputed
    # rent. The two absences have different causes and both must be named.
    labour = pd.concat([_labour(), pd.DataFrame({"activity": ["A"], "labour_cost_share": [0.287]})])
    bite = pd.concat(
        [
            _bite(),
            pd.DataFrame(
                {
                    "industry": ["L"],
                    "minimum_wage_bite": [0.199],
                    "measured_employment_share": [1.0],
                }
            ),
        ]
    )
    bridge = pd.concat(
        [
            _bridge(),
            pd.DataFrame(
                {
                    "category": ["01", "11"],
                    "industry": ["A", "L"],
                    "production_weight": [0.0, 0.0],
                }
            ),
        ]
    )
    _, coverage = structural_exposure(shares, bite, labour, bridge)

    assert ("A", "bite") in coverage.unmeasured_activities
    assert ("L", "labour share") in coverage.unmeasured_activities


def test_a_bridge_that_is_not_a_distribution_is_refused() -> None:
    """Weights summing to less than one understate a category by the missing mass."""
    broken = _bridge().copy()
    broken.loc[broken["category"] == "01", "production_weight"] = 0.25

    with pytest.raises(ExposureError, match="must sum to 1"):
        structural_exposure(_shares(), _bite(), _labour(), broken)


def _panel() -> pd.DataFrame:
    months = pd.to_datetime(["2016-01-01", "2016-02-01"])
    rows = []
    for region in ("PT11", "PT15"):
        for category in ("01", "11"):
            for month in months:
                rows.append(
                    {
                        "region": region,
                        "category_code": category,
                        "month": month,
                        "region_category": f"{region}_{category}",
                        "delta_log_national_minimum_wage": 0.05 if month == months[0] else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_the_interaction_carries_both_absorbed_dimensions() -> None:
    """The estimator needs the region-time and category-time keys built from data.

    Building them from a product of the two margins would invent cells the panel
    does not hold; building them from the rows keeps a region-month that exists
    for one category and not another exactly as it is.
    """
    exposure, _ = structural_exposure(_shares(), _bite(), _labour(), _bridge())
    merged = add_structural_interaction(_panel(), exposure)

    assert {"structural_shock", "region_month", "category_month"}.issubset(merged.columns)
    assert merged["region_month"].nunique() == 4
    assert merged["category_month"].nunique() == 4
    # A month with no statutory change carries no shock, whatever the exposure.
    quiet = merged[merged["delta_log_national_minimum_wage"] == 0.0]
    assert (quiet["structural_shock"] == 0.0).all()


def test_a_category_coding_mismatch_is_refused() -> None:
    """An empty merge is what a COICOP coding disagreement looks like."""
    exposure, _ = structural_exposure(_shares(), _bite(), _labour(), _bridge())
    exposure = exposure.assign(category="X" + exposure["category"])

    with pytest.raises(PassThroughError, match="no region-category cell matches"):
        add_structural_interaction(_panel(), exposure)


def test_three_factors_absorb_more_than_two() -> None:
    """The third factor is what buys the region-time effect.

    It also makes the design rank deficient by construction, since region-time
    and category-time dummies both span the month main effects. That is expected
    and the pseudo-inverse resolves it; a design that refused to build here
    would have blocked the whole approach.
    """
    exposure, _ = structural_exposure(_shares(), _bite(), _labour(), _bridge())
    frame = add_structural_interaction(_panel(), exposure)

    two, two_names = build_absorbing_design(
        frame, shock="structural_shock", factors=["region_category", "month"]
    )
    three, three_names = build_absorbing_design(
        frame,
        shock="structural_shock",
        factors=["region_category", "region_month", "category_month"],
    )

    assert three.shape[1] > two.shape[1]
    assert two_names[0] == three_names[0] == "structural_shock"
    assert np.linalg.matrix_rank(three) < three.shape[1]


def test_absorbing_nothing_is_refused() -> None:
    """A caller wanting a plain regression should ask for one explicitly."""
    exposure, _ = structural_exposure(_shares(), _bite(), _labour(), _bridge())
    frame = add_structural_interaction(_panel(), exposure)

    with pytest.raises(ValueError, match="at least one factor"):
        build_absorbing_design(frame, shock="structural_shock", factors=[])


def test_a_bite_without_its_coverage_column_is_reported_not_absorbed() -> None:
    """Dropping the coverage column silently overstates the exposure.

    `activity_bite_from_registry` returns the bite alongside the share of each
    group's employment it was measured on. Passing only the bite column --- a
    `set_index(...)[...]` away --- turns the weighting off and looks identical
    to having asked for it. On the real measure that is worth 0.4 of a point of
    identifying spread, so the absence is recorded rather than absorbed.
    """
    full = _bite()
    bare = full[["industry", "minimum_wage_bite"]]

    _, weighted = structural_exposure(_shares(), full, _labour(), _bridge())
    _, unweighted = structural_exposure(_shares(), bare, _labour(), _bridge())

    assert weighted.coverage_weighted
    assert not unweighted.coverage_weighted


def test_partial_coverage_lowers_the_identifying_spread() -> None:
    """A partly surveyed group must contribute only the part that was surveyed.

    Crediting a bite measured on half a group to all of it overstates exactly
    the activities the survey covers worst, which here are the ones carrying
    public administration and personal services.
    """
    full = _bite()
    half = full.assign(measured_employment_share=[1.0, 1.0, 0.5])

    _, complete = structural_exposure(_shares(), full, _labour(), _bridge())
    _, partial = structural_exposure(_shares(), half, _labour(), _bridge())

    assert partial.identifying_spread < complete.identifying_spread
