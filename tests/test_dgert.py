"""Contract tests for the DGERT statutory minimum-wage history parser.

These run against a fixture captured from the published page, so they never
touch the network. What they assert is not merely that the parser runs, but
that it reproduces facts that can be checked against Portuguese law and against
an independently compiled Eurostat series.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from pt_mw_inflation.data.dgert import (
    ESCUDOS_PER_EURO,
    StatutoryChange,
    parse_minimum_wage_history,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dgert_minimum_wage_history.html"

#: The provider's page omits the act that took effect in 2000, while the 2001
#: entries still state their increase over that missing regime. Pinning the
#: known gap means a *new* gap appearing upstream fails this suite.
KNOWN_INCOMPLETE = {("general", date(2001, 1, 1)), ("domestic_service", date(2001, 1, 1))}


@pytest.fixture(scope="module")
def changes() -> list[StatutoryChange]:
    """Parse the captured page once for the whole module."""
    return parse_minimum_wage_history(FIXTURE.read_text(encoding="utf-8"))


def _series(changes: list[StatutoryChange], scope: str) -> list[StatutoryChange]:
    return sorted(
        (change for change in changes if change.scope == scope),
        key=lambda change: change.effective_date,
    )


def test_history_starts_at_the_1974_introduction(changes: list[StatutoryChange]) -> None:
    """The general series must begin with Decreto-Lei 217/74 at 3.300$."""
    first = _series(changes, "general")[0]
    assert first.effective_date == date(1974, 5, 27)
    assert first.original_amount == Decimal("3300")
    assert first.original_currency == "PTE"
    assert "217/74" in first.legal_source


def test_every_record_carries_a_legal_citation(changes: list[StatutoryChange]) -> None:
    """No wage may enter the panel without the act that set it."""
    assert all(change.legal_source.strip() for change in changes)
    assert all("Decreto-Lei" in change.legal_source for change in changes)


def test_scopes_are_assigned_from_the_embedded_headers(changes: list[StatutoryChange]) -> None:
    """The three legal regimes must all be recovered, each over its own period.

    Column position alone is not enough: the table drops from three wage
    columns to two in 1992 and to one in 2005. Reading a fixed position would
    quietly return the domestic-service wage as the national series.
    """
    general = _series(changes, "general")
    agriculture = _series(changes, "agriculture")
    domestic = _series(changes, "domestic_service")

    assert general[0].effective_date == date(1974, 5, 27)
    assert agriculture[0].effective_date == date(1977, 1, 1)
    assert domestic[0].effective_date == date(1978, 4, 1)

    # Agriculture was folded into the general regime; domestic service was not,
    # and survives into the euro era.
    assert agriculture[-1].effective_date.year == 1991
    assert domestic[-1].effective_date.year >= 2004


def test_general_wage_is_never_below_the_special_regimes(changes: list[StatutoryChange]) -> None:
    """On any date where regimes coexist, the general wage is the highest.

    This is the structural check that catches a column mix-up, because the
    special regimes were always set below the general one.
    """
    by_date: dict[date, dict[str, Decimal]] = {}
    for change in changes:
        by_date.setdefault(change.effective_date, {})[change.scope] = change.amount_eur

    compared = 0
    for scopes in by_date.values():
        if "general" not in scopes:
            continue
        for scope, amount in scopes.items():
            if scope == "general":
                continue
            compared += 1
            assert amount <= scopes["general"]
    assert compared > 10


def test_stated_increases_reconcile_with_parsed_values(changes: list[StatutoryChange]) -> None:
    """Recomputed growth must match the percentage the provider prints.

    The page states each act's increase over the wage it replaced. Recomputing
    it from consecutive parsed values verifies the amounts, the ordering and the
    scope assignment at once. Only the documented upstream gap may fail.
    """
    unreconciled: set[tuple[str, date]] = set()

    for scope in ("general", "agriculture", "domestic_service"):
        series = _series(changes, scope)
        for previous, current in zip(series, series[1:], strict=False):
            if current.stated_percent_change is None:
                continue
            if previous.original_currency == current.original_currency:
                implied = (current.original_amount / previous.original_amount - 1) * 100
            else:
                implied = (current.amount_eur / previous.amount_eur - 1) * 100
            if abs(implied - current.stated_percent_change) > Decimal("0.15"):
                unreconciled.add((scope, current.effective_date))

    assert unreconciled == KNOWN_INCOMPLETE


def test_changeover_row_prefers_the_official_euro_figure(changes: list[StatutoryChange]) -> None:
    """The 2002 act prints both currencies; the published euro value wins.

    It also pins the conversion rate: 69.770$ / 200.482 is 348.01 euro, which is
    exactly what the act states.
    """
    entry = next(
        change
        for change in changes
        if change.scope == "general" and change.effective_date == date(2002, 1, 1)
    )
    assert entry.original_currency == "PTE"
    assert entry.original_amount == Decimal("69770")
    assert entry.amount_eur == Decimal("348.01")
    assert abs(entry.original_amount / ESCUDOS_PER_EURO - entry.amount_eur) < Decimal("0.01")


def test_escudo_amounts_are_converted_at_the_irrevocable_rate(
    changes: list[StatutoryChange],
) -> None:
    """Pre-euro values convert at the rate fixed for the changeover."""
    first = _series(changes, "general")[0]
    assert first.amount_eur == Decimal("3300") / ESCUDOS_PER_EURO


def test_recent_levels_match_published_law(changes: list[StatutoryChange]) -> None:
    """Spot-check euro-era levels that are matters of public record."""
    levels = {
        change.effective_date.year: change.amount_eur
        for change in _series(changes, "general")
        if change.original_currency == "EUR"
    }
    for year, expected in {2023: 760, 2024: 820, 2025: 870, 2026: 920}.items():
        assert levels[year] == Decimal(expected)


def test_empty_page_is_rejected() -> None:
    """A redesigned upstream page must fail loudly, not return nothing."""
    with pytest.raises(ValueError, match="no table"):
        parse_minimum_wage_history("<html><body><p>moved</p></body></html>")

    with pytest.raises(ValueError, match="layout has changed"):
        parse_minimum_wage_history("<html><body><table><tr><td>x</td></tr></table></body></html>")
