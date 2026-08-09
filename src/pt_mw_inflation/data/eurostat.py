"""Eurostat Statistics API adapter for JSON-stat 2.0 datasets."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import requests

EUROSTAT_STATISTICS_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _ordered_categories(dimension: dict[str, Any]) -> list[str]:
    """Return category codes in JSON-stat positional order."""
    index = dimension["category"]["index"]
    if isinstance(index, dict):
        return [code for code, _ in sorted(index.items(), key=lambda item: item[1])]
    if isinstance(index, list):
        return [str(code) for code in index]
    raise TypeError("Unsupported JSON-stat category index representation")


def jsonstat_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a dense/sparse JSON-stat 2.0 dataset into a tidy DataFrame."""
    dimensions: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    categories = [_ordered_categories(payload["dimension"][dim]) for dim in dimensions]

    expected_size = 1
    for size in sizes:
        expected_size *= int(size)

    raw_values = payload.get("value", {})
    if isinstance(raw_values, list):
        values: dict[int, float | None] = dict(enumerate(raw_values))
    elif isinstance(raw_values, dict):
        values = {int(i): value for i, value in raw_values.items()}
    else:
        raise TypeError("Unsupported JSON-stat value representation")

    rows: list[dict[str, object]] = []
    for flat_index, coordinates in enumerate(product(*categories)):
        if flat_index >= expected_size:
            break
        value = values.get(flat_index)
        if value is None:
            continue
        row: dict[str, object] = dict(zip(dimensions, coordinates, strict=True))
        row["value"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_dataset(
    dataset_code: str,
    filters: dict[str, str | list[str]],
    *,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """Fetch a filtered Eurostat dataset and return tidy observations."""
    params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN")]
    for dimension, value in filters.items():
        if isinstance(value, list):
            params.extend((dimension, item) for item in value)
        else:
            params.append((dimension, value))

    url = f"{EUROSTAT_STATISTICS_API}/{dataset_code}"
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    frame = jsonstat_to_frame(payload)
    frame.attrs["source_url"] = response.url
    frame.attrs["dataset_code"] = dataset_code
    return frame


def fetch_portugal_hicp(geo: str = "PT") -> pd.DataFrame:
    """Fetch detailed monthly HICP indices for Portugal.

    The query deliberately leaves ECOICOP unfiltered so the returned data can be
    used for category-level pass-through analysis. The index unit is selected
    from the current Eurostat HICP dataset; if Eurostat changes its base-unit code,
    the caller can use :func:`fetch_dataset` directly with an updated filter.
    """
    return fetch_dataset(
        "prc_hicp_midx",
        {
            "freq": "M",
            "geo": geo,
            "unit": "I15",
        },
    )


#: Eurostat publishes national minimum wages on a twelve-month basis. Portugal
#: pays the statutory monthly wage fourteen times a year, so Eurostat's figure
#: is the monthly level times 14/12. Dividing it back gives a series directly
#: comparable to the level in Portuguese law.
PORTUGUESE_PAYMENTS_PER_YEAR = 14
EUROSTAT_MONTHS_PER_YEAR = 12


def fetch_minimum_wage(geo: str = "PT", currency: str = "NAC") -> pd.DataFrame:
    """Fetch Eurostat's bi-annual statutory minimum wage series.

    This is an independent check on the national legal history: the two are
    compiled by different institutions from different documents, so agreement
    between them is meaningful evidence that the legal history was read
    correctly.

    Args:
        geo: Eurostat geography code.
        currency: Eurostat currency code; ``NAC`` is the national currency.

    Returns:
        Observations with the published value and, for Portugal, the implied
        monthly statutory level after removing the twelve-month convention.
    """
    frame = fetch_dataset("earn_mw_cur", {"geo": geo, "currency": currency})
    # Observations are bi-annual, labelled like "2024-S1"; the calendar year is
    # what the annual series joins on.
    frame["year"] = frame["time"].astype(str).str.slice(0, 4).astype(int)
    if geo == "PT":
        frame["implied_monthly_statutory_eur"] = frame["value"] * (
            EUROSTAT_MONTHS_PER_YEAR / PORTUGUESE_PAYMENTS_PER_YEAR
        )
    return frame


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    """Persist an API result as Parquet, including a sidecar source URL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    source_url = frame.attrs.get("source_url")
    if source_url:
        path.with_suffix(path.suffix + ".source.txt").write_text(str(source_url), encoding="utf-8")
