"""Adapter for the World Bank indicator API.

Used for the long consumer-price index. Eurostat's HICP for Portugal begins in
1996, which is two decades after the minimum wage was introduced, so the
historical layer needs a series that reaches back to the 1970s. The World Bank
compiles the national CPI from 1960 and exposes it through a stable, versioned
API.

The index is a level with an arbitrary base. Only its growth rate is used, so
the base is irrelevant, but it is retained in the dataset so the deflation of
nominal wages can be reproduced.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from pt_mw_inflation.data.http import USER_AGENT

WORLD_BANK_API = "https://api.worldbank.org/v2"

#: Consumer price index, 2010 = 100.
CPI_INDICATOR = "FP.CPI.TOTL"

PORTUGAL = "PRT"


class WorldBankError(RuntimeError):
    """Raised when the API responds in a shape the adapter does not expect."""


def fetch_indicator(
    indicator: str = CPI_INDICATOR,
    *,
    country: str = PORTUGAL,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """Fetch one indicator for one country as an annual series.

    Args:
        indicator: World Bank indicator code.
        country: ISO-3 country code.
        timeout_seconds: Request timeout.

    Returns:
        Columns `year`, `value` and `indicator`, sorted by year, with missing
        observations dropped.

    Raises:
        requests.HTTPError: If the request fails.
        WorldBankError: If the response is not the documented two-element
            envelope, or carries no observations.
    """
    response = requests.get(
        f"{WORLD_BANK_API}/country/{country}/indicator/{indicator}",
        params={"format": "json", "per_page": "500"},
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload: Any = response.json()

    # The API returns [metadata, observations]; an error returns a single dict.
    if not isinstance(payload, list) or len(payload) < 2:
        raise WorldBankError(f"unexpected response for {indicator}: {str(payload)[:200]}")

    observations = payload[1]
    if not observations:
        raise WorldBankError(f"no observations returned for {indicator} and {country}")

    records = [
        {"year": int(entry["date"]), "value": float(entry["value"]), "indicator": indicator}
        for entry in observations
        if entry.get("value") is not None
    ]
    if not records:
        raise WorldBankError(f"all observations for {indicator} are missing")

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)
