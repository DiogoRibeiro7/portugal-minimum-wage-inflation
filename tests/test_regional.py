"""Tests for legal-act retrieval and the regional statutory schedules.

The text fixtures below are excerpts captured from the published acts, so the
parsing is tested against the wording the gazette actually uses rather than
against an idealisation of it. No test touches the network.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pt_mw_inflation.data.dre import (
    LegalAct,
    eli_url,
    extract_minimum_wage,
    extract_regional_supplement,
    parse_amounts,
)
from pt_mw_inflation.processing.regional import (
    RegionalScheduleError,
    build_regional_panel,
    regional_premium,
)

# Decreto-Lei 573/99, article 1. The escudo amounts are printed with a thin
# space and no cents, and the article also cites the act it amends.
DL_573_99 = (
    "Artigo 1.o Os valores de remuneracao minima mensal a que se refere o n. o 1 do "
    "artigo 1. o eon . o 2 do artigo 3. o do Decreto-Lei n. o 69-A/87, de 9 de Fevereiro, "
    "passam a ser de 63 800$ e de 60 000$. Artigo 2.o E revogado o Decreto-Lei n. o 49/99, "
    "de 16 de Fevereiro. Artigo 3.o O presente diploma produz efeitos a partir de "
    "1 de Janeiro de 2000."
)

# Decreto Legislativo Regional 3/2024/M. The euro sign follows the amount and
# the value carries no cents.
DLR_3_2024_M = (
    "Artigo 2.o Valor da retribuicao minima mensal garantida O valor da retribuicao "
    "minima mensal garantida para vigorar na Regiao Autonoma da Madeira e de 850 €, "
    "nos termos do artigo 6.o do Decreto Legislativo Regional n.o 21/2009/M, de 4 de agosto."
)

# Decreto Legislativo Regional 1/2026/M. Here the sign precedes the amount and
# cents are present, and the act also quotes the mainland figure for comparison.
DLR_1_2026_M = (
    "Artigo 2.o Valor da retribuicao minima mensal garantida O valor da retribuicao "
    "minima mensal garantida para vigorar na Regiao Autonoma da Madeira e de € 980,00, "
    "o que representa um acrescimo face ao valor de € 920,00 em vigor no continente."
)

# Decreto Legislativo Regional 8/2002/A. The operative article is preceded by
# pension bands quoting other percentages and amounts.
DLR_8_2002_A = (
    "b) 90% para aqueles cuja pensao seja superior ao salario minimo e inferior ou igual "
    "a 75 000$ (E 374,10); c) 70% para aqueles cuja pensao seja superior a 75 000$. "
    "CAPITULO II Acrescimo regional ao salario minimo Artigo 3. o Montante O montante do "
    "salario minimo, estabelecido ao nivel nacional para os trabalhadores por contra de "
    "outrem, tem, na Regiao Autonoma dos Acores, o acrescimo de 5%."
)


def _act(text: str, **overrides: object) -> LegalAct:
    """Build an act around captured text."""
    payload: dict[str, object] = {
        "act_type": "dec-lei",
        "number": 573,
        "act_date": date(1999, 12, 30),
        "jurisdiction": "p",
        "url": "https://example.invalid",
        "text": text,
    }
    payload.update(overrides)
    return LegalAct(**payload)  # type: ignore[arg-type]


def test_permalink_matches_the_published_scheme() -> None:
    """The permalink must be built exactly as the gazette expects."""
    assert eli_url("dec-lei", 573, date(1999, 12, 30)) == (
        "https://data.dre.pt/eli/dec-lei/573/1999/12/30/p/dre/pt/pdf"
    )
    assert eli_url("declegreg", 3, date(2024, 2, 8), jurisdiction="m").endswith(
        "/declegreg/3/2024/02/08/m/dre/pt/pdf"
    )


def test_escudo_amounts_are_parsed_with_thin_spaces() -> None:
    """Escudo figures are printed with a space as the thousands separator."""
    amounts = parse_amounts(DL_573_99)
    assert (Decimal("63800"), "PTE") in amounts
    assert (Decimal("60000"), "PTE") in amounts


def test_euro_sign_is_parsed_on_either_side_of_the_amount() -> None:
    """Acts put the sign before or after the figure; both must parse."""
    assert (Decimal("850"), "EUR") in parse_amounts(DLR_3_2024_M)
    assert (Decimal("980.00"), "EUR") in parse_amounts(DLR_1_2026_M)


def test_wage_is_read_from_the_operative_article() -> None:
    """The value must come from the article that sets it, not the first number.

    Article 1 of the 1999 decree names two earlier acts before stating any
    amount, so an unanchored parse would return a cross-reference.
    """
    amount, currency = extract_minimum_wage(_act(DL_573_99))
    assert (amount, currency) == (Decimal("63800"), "PTE")


def test_madeira_value_is_read_despite_the_comparison_figure() -> None:
    """The 2026 act quotes the mainland wage too; the regional one must win."""
    amount, currency = extract_minimum_wage(
        _act(DLR_1_2026_M, act_type="declegreg", jurisdiction="m")
    )
    assert (amount, currency) == (Decimal("980.00"), "EUR")


def test_supplement_is_read_from_its_own_chapter() -> None:
    """Pension bands quote percentages too, so the supplement must be anchored.

    An unanchored search would return 90 per cent, from the pension rules that
    precede the article fixing the wage supplement.
    """
    act = _act(DLR_8_2002_A, act_type="declegreg", number=8, jurisdiction="a")
    assert extract_regional_supplement(act) == Decimal("5")


def test_act_without_an_amount_is_rejected() -> None:
    """An act that sets a regime rather than a value must fail loudly."""
    with pytest.raises(ValueError, match="no monetary amount"):
        extract_minimum_wage(_act("Artigo 1.o O presente diploma estabelece o regime."))


def test_act_without_a_supplement_is_rejected() -> None:
    """A missing supplement is an error, not a zero."""
    with pytest.raises(ValueError, match="no regional supplement"):
        extract_regional_supplement(_act("Artigo 1.o Objeto."))


def _national_panel() -> pd.DataFrame:
    """A minimal national panel covering the supplement period."""
    return pd.DataFrame(
        [
            {
                "effective_date": pd.Timestamp("2024-01-01"),
                "minimum_wage_monthly_eur": 820.0,
                "legal_source": "Decreto-Lei n.º 107/2023",
            },
            {
                "effective_date": pd.Timestamp("2025-01-01"),
                "minimum_wage_monthly_eur": 870.0,
                "legal_source": "Decreto-Lei n.º 112/2024",
            },
        ]
    )


def test_regional_panel_applies_each_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Madeira takes its stated value; the Azores derive theirs from the rule."""
    import pt_mw_inflation.processing.regional as regional_module

    def fake_fetch(entry: dict[str, object]) -> LegalAct:
        if entry.get("jurisdiction") == "a":
            return _act(DLR_8_2002_A, act_type="declegreg", number=8, jurisdiction="a")
        return _act(DLR_3_2024_M, act_type="declegreg", number=3, jurisdiction="m")

    monkeypatch.setattr(regional_module, "_act_from_config", fake_fetch)

    registry = {
        "madeira": {
            "geography": "PT30",
            "mechanism": "explicit_value",
            "acts": [
                {
                    "act_type": "declegreg",
                    "number": 3,
                    "act_date": date(2024, 2, 8),
                    "effective_date": date(2024, 1, 1),
                }
            ],
        },
        "azores": {
            "geography": "PT20",
            "mechanism": "proportional_supplement",
            "supplement_act": {
                "act_type": "declegreg",
                "number": 8,
                "act_date": date(2002, 4, 10),
                "effective_date": date(2002, 1, 1),
            },
        },
    }

    panel = build_regional_panel(registry, _national_panel()).set_index(
        ["geography", "effective_date"]
    )

    assert panel.loc[("PT30", pd.Timestamp("2024-01-01")), "minimum_wage_monthly_eur"] == 850.0
    # 820 x 1.05, derived rather than legislated.
    assert panel.loc[
        ("PT20", pd.Timestamp("2024-01-01")), "minimum_wage_monthly_eur"
    ] == pytest.approx(861.0)
    assert panel["national_or_regional"].eq("regional").all()


def test_unknown_mechanism_is_rejected() -> None:
    """A region must declare how its wage is set."""
    registry = {"nowhere": {"geography": "PTXX", "mechanism": "vibes"}}
    with pytest.raises(RegionalScheduleError, match="unknown mechanism"):
        build_regional_panel(registry, _national_panel())


def test_premium_separates_regional_policy_from_national() -> None:
    """The premium is the part of the regional wage that is not national policy.

    Madeira's premium varies year to year, which is the independent variation
    the design needs; a fixed supplement would produce a constant column.
    """
    regional = pd.DataFrame(
        [
            {
                "geography": "PT30",
                "effective_date": pd.Timestamp("2024-01-01"),
                "minimum_wage_monthly_eur": 850.0,
                "legal_source": "DLR 3/2024/M",
            },
            {
                "geography": "PT30",
                "effective_date": pd.Timestamp("2025-01-01"),
                "minimum_wage_monthly_eur": 915.0,
                "legal_source": "DLR 20/2024/M",
            },
        ]
    )
    national_annual = pd.DataFrame({"year": [2024, 2025], "minimum_wage_january": [820.0, 870.0]})
    premium = regional_premium(regional, national_annual).set_index("year")

    assert premium.loc[2024, "premium"] == pytest.approx(30 / 820)
    assert premium.loc[2025, "premium"] == pytest.approx(45 / 870)
    assert premium.loc[2025, "premium"] > premium.loc[2024, "premium"]


def test_premium_requires_the_national_series() -> None:
    """Comparing against a frame without the national wage is an error."""
    with pytest.raises(RegionalScheduleError, match="missing columns"):
        regional_premium(pd.DataFrame(), pd.DataFrame({"year": [2024]}))
