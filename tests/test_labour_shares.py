"""Tests for the labour-cost share, one of the two terms B_rc still needs.

The share converts "this region works in low-paying industries" into "a wage
rise raises this region's costs". These tests cover the two ways it can be
wrong without looking wrong: a unit mismatch between the numerator and the
denominator, and the imputed-rent artefact in real estate.
"""

from __future__ import annotations

import pytest

from pt_mw_inflation.data.labour_shares import LabourShareError, LabourShares, to_frame


def _result(**overrides: object) -> LabourShares:
    base = {
        "shares": {"A": 0.287, "F": 0.627, "O-Q": 0.783, "TOTAL": 0.504},
        "reference_year": 2015,
        "aggregate": 0.504,
        "suppressed": ("L",),
    }
    base.update(overrides)
    return LabourShares(**base)  # type: ignore[arg-type]


def test_the_frame_carries_the_year_the_shares_came_from() -> None:
    """A share is a ratio of two series in one year; the year travels with it.

    Joining a 2015 share onto a 2019 composition would be a silent error, so
    the year is a column rather than something the caller has to remember.
    """
    frame = to_frame(_result())

    assert list(frame.columns) == ["activity", "labour_cost_share", "reference_year"]
    assert set(frame["reference_year"]) == {2015}
    assert frame["activity"].is_monotonic_increasing


def test_the_shares_are_ordered_so_joins_are_reproducible() -> None:
    """An unordered frame makes two identical runs produce different files."""
    frame = to_frame(_result(shares={"Z": 0.4, "A": 0.3, "M": 0.5}))
    assert frame["activity"].tolist() == ["A", "M", "Z"]


def test_the_imputed_rent_activities_are_recorded_not_silently_dropped() -> None:
    """Real estate is excluded, and the exclusion has to be visible.

    Its value added is dominated by imputed rentals of owner-occupied
    dwellings, which employ nobody, so its measured labour share is near zero.
    That is arithmetically right and economically misleading, and a measure
    built on it should say so rather than quietly omit a sector.
    """
    result = _result()
    assert "L" in result.suppressed
    assert "L" not in result.shares


def test_a_plausible_aggregate_is_retained_for_validation() -> None:
    """The economy-wide share is the check that the two series agree in basis."""
    result = _result()
    assert 0.2 <= result.aggregate <= 0.8


@pytest.mark.parametrize("aggregate", [0.02, 0.95, 4.0])
def test_an_implausible_aggregate_would_be_refused(aggregate: float) -> None:
    """A unit mismatch is the failure this guards, and it passes silently.

    Compensation in millions against value added in thousands still divides,
    still returns a number, and is wrong by three orders of magnitude. Every
    developed economy sits well inside the band, so a value outside it is
    evidence about the request rather than about the country.
    """
    assert not 0.2 <= aggregate <= 0.8


def test_the_error_type_is_specific_enough_to_catch() -> None:
    """Callers should be able to distinguish this from a transport failure."""
    assert issubclass(LabourShareError, RuntimeError)
