"""Tests for the Statistics Portugal indicator adapter.

The fixture below is an excerpt of a real response, including the mangled
accent the API returns for month names, so the parsing is tested against what
the service actually sends. No test touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pt_mw_inflation.data.ine import (
    AGGREGATE_GEOGRAPHIES,
    IneError,
    monthly_periods,
    parse_observations,
    parse_period_label,
    to_nuts_code,
)

RESPONSE = {
    "IndicadorCod": "0014659",
    "Dados": {
        "Janeiro de 2026": [
            {
                "geocod": "11",
                "geodsg": "Norte",
                "dim_3": "11",
                "dim_3_t": "Restaurantes e serviços de alojamento",
                "valor": "99.107",
            },
            {
                "geocod": "2",
                "geodsg": "Região Autónoma dos Açores",
                "dim_3": "11",
                "dim_3_t": "Restaurantes e serviços de alojamento",
                "valor": "98.731",
            },
            {
                "geocod": "3",
                "geodsg": "Região Autónoma da Madeira",
                "dim_3": "11",
                "dim_3_t": "Restaurantes e serviços de alojamento",
                "valor": "98.783",
            },
            {
                "geocod": "PT",
                "geodsg": "Portugal",
                "dim_3": "T",
                "dim_3_t": "Total",
                "valor": "99.512",
            },
            # A suppressed observation, which must not become a zero.
            {"geocod": "1C", "geodsg": "Alentejo", "dim_3": "11", "dim_3_t": "x", "valor": ""},
        ]
    },
}


def test_autonomous_regions_map_to_their_nuts_codes() -> None:
    """The regions are published as single digits, not as NUTS codes.

    They are the two geographies with their own statutory minimum wage, so
    mapping them wrongly would silently misalign prices and policy.
    """
    assert to_nuts_code("2") == "PT20"
    assert to_nuts_code("3") == "PT30"


def test_mainland_codes_take_the_country_prefix() -> None:
    """Every other code is a NUTS code missing its prefix."""
    assert to_nuts_code("11") == "PT11"
    assert to_nuts_code("1A") == "PT1A"
    assert to_nuts_code("PT") == "PT"


def test_aggregates_are_identified() -> None:
    """Portugal and Continente overlap their own components."""
    frame = pd.DataFrame(parse_observations(RESPONSE))
    frame["nuts_code"] = frame["ine_geocode"].map(to_nuts_code)
    aggregates = set(frame.loc[frame["nuts_code"].isin(AGGREGATE_GEOGRAPHIES), "nuts_code"])
    assert aggregates == {"PT"}


def test_period_labels_parse_including_mangled_accents() -> None:
    """The API returns prose labels, and the encoding mangles accented names."""
    assert parse_period_label("Janeiro de 2026") == pd.Timestamp("2026-01-01")
    assert parse_period_label("Dezembro de 1991") == pd.Timestamp("1991-12-01")
    # "Março" as returned when the accent does not survive the response encoding.
    assert parse_period_label("Mar�o de 2025") == pd.Timestamp("2025-03-01")


def test_unrecognised_label_is_rejected() -> None:
    """A changed label format must fail rather than yield a wrong month."""
    with pytest.raises(IneError, match="unrecognised period label"):
        parse_period_label("Q1 2026")


def test_observations_are_flattened_with_the_month_attached() -> None:
    """Each row must carry region, category and month."""
    records = parse_observations(RESPONSE)
    assert len(records) == 4  # the suppressed row is dropped
    assert all(record["month"] == pd.Timestamp("2026-01-01") for record in records)

    azores = next(r for r in records if r["ine_geocode"] == "2")
    assert azores["price_index"] == pytest.approx(98.731)
    assert azores["category_code"] == "11"


def test_suppressed_values_are_dropped_not_zeroed() -> None:
    """An empty value is a withheld observation, not a price index of zero."""
    records = parse_observations(RESPONSE)
    assert all(record["price_index"] > 0 for record in records)
    assert not any(record["ine_geocode"] == "1C" for record in records)


def test_monthly_period_codes_match_the_api_scheme() -> None:
    """Period codes are the monthly prefix followed by year and month."""
    codes = monthly_periods("1991-01", "1991-03")
    assert codes == ["S3A199101", "S3A199102", "S3A199103"]
    assert len(monthly_periods("2000-01", "2026-06")) == 318


def test_reversed_range_is_rejected() -> None:
    """An empty range is a caller error, not an empty panel."""
    with pytest.raises(IneError, match="no months between"):
        monthly_periods("2026-06", "2020-01")


def test_missing_data_block_is_reported() -> None:
    """A retired indicator answers 200 with an error body, not a failure code."""
    assert parse_observations({"IndicadorCod": "0007320"}) == []
