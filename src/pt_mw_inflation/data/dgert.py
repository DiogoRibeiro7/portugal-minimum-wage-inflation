"""Parser for the official DGERT history of the guaranteed minimum monthly wage.

The published page is a single ragged HTML table covering 1974 to the present.
Its shape changes twice, because the number of legally distinct minimum wages
fell over time:

===================  =========================================================
Period               Value columns, in the order they appear
===================  =========================================================
1974 - 1991          Serviço Doméstico, Agricultura, Restantes Atividades
1992 - 2004          Serviço Doméstico, Restantes Atividades
2005 onwards         a single unified value
===================  =========================================================

Each shape change is announced by an embedded header row inside the table body,
which is what this parser keys on. Taking a fixed column position instead would
silently read the domestic-service wage as the national series for the whole
escudo era, and the resulting numbers would still look entirely plausible.

``Restantes Atividades`` is the general regime and the series normally quoted as
*the* Portuguese minimum wage; it is mapped to the ``general`` scope here.

Amounts before 2002 are in escudos and are converted at the irrevocable rate
fixed for the euro changeover. The 2002 row states both currencies, and the
official euro figure is preferred over the converted one when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from bs4 import BeautifulSoup, Tag

#: Irrevocable conversion rate fixed by Council Regulation (EC) No 2866/98.
ESCUDOS_PER_EURO = Decimal("200.482")

Scope = Literal["general", "agriculture", "domestic_service"]

#: Column headings used by the provider, normalised to lowercase without accents.
_SCOPE_BY_LABEL: dict[str, Scope] = {
    "servico domestico": "domestic_service",
    "agricultura": "agriculture",
    "restantes atividades": "general",
    "restantes actividades": "general",
}

_EURO_PATTERN = re.compile(r"€\s*([\d.]+,\d{2})")
_ESCUDO_PATTERN = re.compile(r"([\d.]+)\s*\$")
_PERCENT_PATTERN = re.compile(r"(\d+(?:,\d+)?)\s*%")
_DATE_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})")

_ACCENTS = str.maketrans("áàâãéêíóôõúç", "aaaaeeiooouc")


@dataclass(frozen=True)
class StatutoryChange:
    """One statutory minimum-wage value announced by one legal act."""

    legal_source: str
    effective_date: date
    scope: Scope
    amount_eur: Decimal
    original_amount: Decimal
    original_currency: Literal["PTE", "EUR"]
    stated_percent_change: Decimal | None


def _normalise(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace for label matching."""
    return " ".join(text.split()).lower().translate(_ACCENTS)


def _parse_portuguese_number(text: str) -> Decimal:
    """Convert a Portuguese-formatted number, with '.' thousands and ',' decimals."""
    return Decimal(text.replace(".", "").replace(",", "."))


def _cell_text(cell: Tag) -> str:
    """Return the collapsed visible text of a table cell."""
    return " ".join(cell.get_text(" ", strip=True).split())


def _parse_date(text: str) -> date | None:
    """Parse a dd/mm/yyyy effective date, returning None when absent."""
    match = _DATE_PATTERN.search(text)
    if match is None:
        return None
    day, month, year = (int(part) for part in match.groups())
    return date(year, month, day)


def _parse_amount(text: str) -> tuple[Decimal, Decimal, Literal["PTE", "EUR"]] | None:
    """Extract the wage from one value cell.

    Returns:
        A triple of amount in euro, amount in the currency as published, and the
        published currency code. None when the cell holds no wage, which happens
        for regimes that did not yet exist in that year.
    """
    euro_match = _EURO_PATTERN.search(text)
    escudo_match = _ESCUDO_PATTERN.search(text)

    if euro_match is not None:
        euro = _parse_portuguese_number(euro_match.group(1))
        if escudo_match is not None:
            # Changeover row: both currencies printed. Keep the official euro
            # figure, and retain the escudo amount as published.
            return euro, _parse_portuguese_number(escudo_match.group(1)), "PTE"
        return euro, euro, "EUR"

    if escudo_match is not None:
        escudos = _parse_portuguese_number(escudo_match.group(1))
        return escudos / ESCUDOS_PER_EURO, escudos, "PTE"

    return None


def _parse_percent(text: str) -> Decimal | None:
    """Extract the provider's stated percentage increase from a cell."""
    match = _PERCENT_PATTERN.search(text)
    return _parse_portuguese_number(match.group(1)) if match is not None else None


def _scope_layout(cells: list[Tag]) -> list[Scope] | None:
    """Return the scope order announced by an embedded header row, if it is one."""
    labels = [_normalise(_cell_text(cell)) for cell in cells]
    scopes = [_SCOPE_BY_LABEL[label] for label in labels if label in _SCOPE_BY_LABEL]
    return scopes if len(scopes) == len(labels) and scopes else None


def parse_minimum_wage_history(html: str) -> list[StatutoryChange]:
    """Parse the DGERT page into one record per legal act, scope and date.

    Args:
        html: The decoded HTML of the published history page.

    Returns:
        Statutory changes sorted by effective date and scope.

    Raises:
        ValueError: If the page contains no table, or no rows could be parsed,
            which is how a redesign of the upstream page surfaces.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not isinstance(table, Tag):
        raise ValueError("no table found on the DGERT page; the layout has changed")

    # Before the first embedded header the page is in its modern single-wage
    # shape, where the percentage sits in its own cell rather than beside the
    # amount.
    layout: list[Scope] = ["general"]
    percent_beside_amount = False

    changes: list[StatutoryChange] = []

    for row in table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = [cell for cell in row.find_all(["th", "td"]) if isinstance(cell, Tag)]
        if not cells:
            continue

        # Scope headers must be tested before any width guard: the two-column
        # header that opens the 1992-2004 block is itself only two cells wide,
        # and skipping it leaves the layout stuck on the modern single-wage
        # shape, which silently reads domestic service as the national series.
        announced = _scope_layout(cells)
        if announced is not None:
            layout = announced
            percent_beside_amount = True
            continue

        if len(cells) < 3 or any(cell.name == "th" for cell in cells):
            continue

        legal_source = _cell_text(cells[0])
        effective_date = _parse_date(_cell_text(cells[1]))
        if effective_date is None or not legal_source:
            continue

        value_cells = cells[2 : 2 + len(layout)]
        for scope, cell in zip(layout, value_cells, strict=False):
            text = _cell_text(cell)
            parsed = _parse_amount(text)
            if parsed is None:
                continue
            amount_eur, original_amount, currency = parsed
            percent_text = text if percent_beside_amount else _cell_text(cells[-1])
            changes.append(
                StatutoryChange(
                    legal_source=legal_source,
                    effective_date=effective_date,
                    scope=scope,
                    amount_eur=amount_eur,
                    original_amount=original_amount,
                    original_currency=currency,
                    stated_percent_change=_parse_percent(percent_text),
                )
            )

    if not changes:
        raise ValueError("no statutory changes parsed; the DGERT page layout has changed")

    return sorted(changes, key=lambda change: (change.effective_date, change.scope))
