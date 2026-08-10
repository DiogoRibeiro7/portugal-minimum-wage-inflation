"""Regional statutory minimum wages for Madeira and the Azores.

Mainland statutory changes are national, so a design that identifies price
pass-through from regional differences in the *policy* needs the autonomous
regions. Both may set a wage above the national one, and they do it by different
mechanisms:

Madeira
    legislates an explicit value, by a regional decree, most years. The premium
    over the mainland is not a fixed rule and has widened: 3.3 per cent in 2023,
    6.5 per cent in 2026.

Azores
    legislate a permanent proportional supplement instead of an annual figure.
    Article 3 of Decreto Legislativo Regional 8/2002/A adds 5 per cent to
    whatever the national wage is, so the regional series is derived from the
    national one rather than read off annual acts.

The distinction matters for identification. Madeira supplies genuine
independent variation, because its value moves for reasons of regional politics
that are not a function of the national figure. The Azores supplement is a fixed
transformation of the national wage, so on its own it adds a level difference
but no independent timing: in a specification with calendar-time fixed effects
the Azores contribute through the *interaction* of the supplement with national
changes, not through variation of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Literal

import pandas as pd

from pt_mw_inflation.data.dgert import ESCUDOS_PER_EURO, StatutoryChange
from pt_mw_inflation.data.dre import (
    LegalAct,
    extract_minimum_wage,
    extract_regional_supplement,
    fetch_act,
    parse_amounts,
)

Mechanism = Literal["explicit_value", "proportional_supplement"]


class RegionalScheduleError(ValueError):
    """Raised when a regional schedule cannot be built as configured."""


@dataclass(frozen=True)
class RegionalWage:
    """One regional statutory wage in force from a date."""

    geography: str
    effective_date: date
    minimum_wage_monthly_eur: Decimal
    legal_source: str
    source_url: str
    mechanism: Mechanism


def _act_from_config(entry: dict[str, Any]) -> LegalAct:
    """Retrieve one act described by a registry entry."""
    return fetch_act(
        entry["act_type"],
        int(entry["number"]),
        entry["act_date"],
        jurisdiction=entry.get("jurisdiction", "p"),
    )


def build_explicit_schedule(
    geography: str,
    acts: list[dict[str, Any]],
    *,
    jurisdiction: str = "m",
) -> list[RegionalWage]:
    """Retrieve each act and read the wage it sets.

    Args:
        geography: NUTS code for the region.
        acts: Registry entries with `act_type`, `number`, `act_date` and
            `effective_date`.
        jurisdiction: ELI jurisdiction token for the region.

    Returns:
        One wage per act, ordered by effective date.

    Raises:
        RegionalScheduleError: If an act states no wage, or states one in a
            currency the regional series cannot use.
    """
    schedule: list[RegionalWage] = []
    for entry in acts:
        act = _act_from_config({**entry, "jurisdiction": jurisdiction})
        try:
            amount, currency = extract_minimum_wage(act)
        except ValueError as error:
            raise RegionalScheduleError(f"{act.citation}: {error}") from error
        if currency != "EUR":
            raise RegionalScheduleError(
                f"{act.citation} states {currency}; regional acts postdate the changeover"
            )
        schedule.append(
            RegionalWage(
                geography=geography,
                effective_date=entry["effective_date"],
                minimum_wage_monthly_eur=amount,
                legal_source=act.citation,
                source_url=act.url,
                mechanism="explicit_value",
            )
        )
    return sorted(schedule, key=lambda wage: wage.effective_date)


def build_supplement_schedule(
    geography: str,
    supplement_act: dict[str, Any],
    national: pd.DataFrame,
    *,
    jurisdiction: str = "a",
    wage_column: str = "minimum_wage_monthly_eur",
) -> list[RegionalWage]:
    """Derive a regional schedule from a proportional supplement.

    Args:
        geography: NUTS code for the region.
        supplement_act: Registry entry for the act fixing the supplement.
        national: National statutory panel, one row per act.
        jurisdiction: ELI jurisdiction token for the region.
        wage_column: Column of `national` holding the monthly wage.

    Returns:
        One regional wage per national act from the supplement's effective date,
        each carrying both the national act and the supplement act as its source.

    Raises:
        RegionalScheduleError: If the act states no supplement, or the national
            panel has no acts after it takes effect.
    """
    act = _act_from_config({**supplement_act, "jurisdiction": jurisdiction})
    try:
        percentage = extract_regional_supplement(act)
    except ValueError as error:
        raise RegionalScheduleError(f"{act.citation}: {error}") from error

    factor = Decimal(1) + percentage / Decimal(100)
    start = supplement_act["effective_date"]

    applicable = national.loc[pd.to_datetime(national["effective_date"]).dt.date >= start]
    if applicable.empty:
        raise RegionalScheduleError(
            f"no national acts on or after {start}; the supplement cannot be applied"
        )

    schedule = []
    for _, row in applicable.iterrows():
        national_amount = Decimal(str(row[wage_column]))
        schedule.append(
            RegionalWage(
                geography=geography,
                effective_date=pd.Timestamp(row["effective_date"]).date(),
                minimum_wage_monthly_eur=national_amount * factor,
                legal_source=f"{row['legal_source']} + {act.citation} (+{percentage}%)",
                source_url=act.url,
                mechanism="proportional_supplement",
            )
        )
    return schedule


def build_regional_panel(
    registry: dict[str, Any],
    national: pd.DataFrame,
    *,
    payments_per_year: int = 14,
) -> pd.DataFrame:
    """Assemble the regional statutory panel from the configured acts.

    Args:
        registry: The `regional` block of `config/legal_acts.yaml`.
        national: National statutory panel, filtered to the general regime.
        payments_per_year: Statutory payments per year, for annualisation.

    Returns:
        A panel with the same contract as the national one, so the two stack.

    Raises:
        RegionalScheduleError: If a region declares an unknown mechanism.
    """
    wages: list[RegionalWage] = []

    for name, config in registry.items():
        mechanism = config["mechanism"]
        geography = config["geography"]
        # Read from the registry rather than inferred from the key: regional
        # decrees are numbered per region, so fetching the wrong jurisdiction
        # returns a real but unrelated act and succeeds silently.
        jurisdiction = config.get("jurisdiction")
        if not jurisdiction:
            raise RegionalScheduleError(f"{name}: registry entry declares no jurisdiction")

        if mechanism == "explicit_value":
            wages.extend(
                build_explicit_schedule(geography, config["acts"], jurisdiction=jurisdiction)
            )
        elif mechanism == "proportional_supplement":
            wages.extend(
                build_supplement_schedule(
                    geography, config["supplement_act"], national, jurisdiction=jurisdiction
                )
            )
        else:
            raise RegionalScheduleError(f"{name}: unknown mechanism {mechanism!r}")

    if not wages:
        return pd.DataFrame()

    records = []
    for wage in sorted(wages, key=lambda item: (item.geography, item.effective_date)):
        monthly = float(wage.minimum_wage_monthly_eur)
        records.append(
            {
                "geography": wage.geography,
                "effective_date": wage.effective_date,
                "scope": "general",
                "minimum_wage_monthly_eur": monthly,
                "payments_per_year": payments_per_year,
                "annualised_minimum_wage_eur": monthly * payments_per_year,
                "original_amount": monthly,
                "original_currency": "EUR",
                "legal_source": wage.legal_source,
                "national_or_regional": "regional",
                "notes": f"mechanism: {wage.mechanism}; source: {wage.source_url}",
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    return frame


def regional_premium(regional: pd.DataFrame, national_annual: pd.DataFrame) -> pd.DataFrame:
    """Compare each regional wage with the national wage in force that year.

    The premium is the object the identification strategy needs: it is the part
    of the regional wage that is not the national policy.

    Args:
        regional: Output of :func:`build_regional_panel`.
        national_annual: Annual national series with `year` and
            `minimum_wage_january`.

    Returns:
        One row per geography and year with both levels and the premium.

    Raises:
        RegionalScheduleError: If required columns are absent.
    """
    required = {"year", "minimum_wage_january"}
    missing = required.difference(national_annual.columns)
    if missing:
        raise RegionalScheduleError(f"national_annual missing columns: {sorted(missing)}")

    frame = regional.copy()
    frame["year"] = pd.to_datetime(frame["effective_date"]).dt.year

    # A region that does not legislate in a given year keeps the level it last
    # set, while the national wage usually rises. Those are exactly the years
    # the premium narrows, so joining only on act years would drop them and
    # overstate the premium's stability.
    span = national_annual.loc[national_annual["year"] >= int(frame["year"].min()), ["year"]]
    carried = []
    for geography, block in frame.groupby("geography"):
        filled = span.merge(block, on="year", how="left").sort_values("year")
        filled["geography"] = geography
        for column in ("minimum_wage_monthly_eur", "legal_source"):
            filled[column] = filled[column].ffill()
        carried.append(filled.dropna(subset=["minimum_wage_monthly_eur"]))

    frame = pd.concat(carried, ignore_index=True) if carried else frame
    merged = frame.merge(national_annual[["year", "minimum_wage_january"]], on="year", how="inner")
    merged["national_minimum_wage_eur"] = merged["minimum_wage_january"]
    merged["premium"] = (
        merged["minimum_wage_monthly_eur"] / merged["national_minimum_wage_eur"] - 1.0
    )
    return merged[
        [
            "geography",
            "year",
            "minimum_wage_monthly_eur",
            "national_minimum_wage_eur",
            "premium",
            "legal_source",
        ]
    ].sort_values(["geography", "year"])


def merge_supplements(
    parsed: list[StatutoryChange], supplements: list[StatutoryChange]
) -> list[StatutoryChange]:
    """Add retrieved acts to the parsed history, skipping any already present.

    The summary page may start listing an act that is registered here. Appending
    unconditionally would then put two rows on one date for one regime, which
    makes the reconciliation check report a spurious zero-per-cent increase.

    Args:
        parsed: Changes read from the summary history.
        supplements: Changes read from the gazette.

    Returns:
        The union, preferring the parsed history where both cover a regime-date.
    """
    present = {(change.scope, change.effective_date) for change in parsed}
    return parsed + [
        change for change in supplements if (change.scope, change.effective_date) not in present
    ]


def supplementary_statutory_changes(
    supplements: list[dict[str, Any]],
) -> list[StatutoryChange]:
    """Retrieve acts the summary history omits, as statutory changes.

    The parsed DGERT table is authoritative for what it lists, but it has holes:
    it does not carry the act effective in 2000, while its 2001 entries state
    their increase relative to it. Rather than impute the missing level, or take
    it from a secondary compiler, the act itself is retrieved and read.

    Args:
        supplements: Registry entries with `act_type`, `number`, `act_date`,
            `effective_date` and `scope`.

    Returns:
        Changes that merge into the parsed history, each carrying the citation
        of the act it was read from.

    Raises:
        RegionalScheduleError: If an act states no wage.
    """
    changes: list[StatutoryChange] = []
    for entry in supplements:
        act = _act_from_config(entry)
        index = int(entry.get("amount_index", 0))
        try:
            if index == 0:
                amount, currency = extract_minimum_wage(act)
            else:
                # An act may fix several regimes in one article. The index
                # selects which stated amount this entry refers to.
                anchor = act.text.lower().rfind("passam a ser de")
                if anchor < 0:
                    raise ValueError(
                        "no 'passam a ser de' clause found; amount_index cannot "
                        "select among the amounts this act states"
                    )
                stated = parse_amounts(act.text[anchor:])
                if len(stated) <= index:
                    raise ValueError(f"act states {len(stated)} amounts, index {index} requested")
                amount, currency = stated[index]
        except ValueError as error:
            raise RegionalScheduleError(f"{act.citation}: {error}") from error

        amount_eur = amount / ESCUDOS_PER_EURO if currency == "PTE" else amount
        changes.append(
            StatutoryChange(
                legal_source=act.citation,
                effective_date=entry["effective_date"],
                scope=entry.get("scope", "general"),
                amount_eur=amount_eur,
                original_amount=amount,
                original_currency=currency,
                stated_percent_change=None,
            )
        )
    return changes
