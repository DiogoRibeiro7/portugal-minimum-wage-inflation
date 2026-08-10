"""Adapter for Statistics Portugal's indicator API.

This supplies the regional price panel the identification strategy needs:
consumer price indices by NUTS II region and consumption purpose, monthly. No
other source publishes it. Eurostat's HICP for Portugal is national only, and
the World Bank series used for the long-run layer is a single national index.

Finding the indicator is the hard part, and worth recording. INE's own portal
search is unreliable from a script, and indicator codes are retired without
redirect when a series is rebased: the code the open-data catalogue still
advertises for the 2012-base index answers "o codigo do indicador nao existe".
The catalogue at dados.gov.pt does, however, list each live indicator together
with its API URL, which is how ``0014659`` was identified.

The API itself is intermittent -- connections are refused under load -- so every
request retries. Periods may be requested in batches, which keeps a full
history to a few dozen calls rather than one per month.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pt_mw_inflation.data.http import USER_AGENT, sha256_bytes

INE_API = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

#: Consumer price index, base 2025, by NUTS II (2024) and consumption purpose,
#: monthly from January 1991.
REGIONAL_CPI_INDICATOR = "0014659"

#: Monthly period codes are "S3A" followed by year and month.
MONTHLY_PREFIX = "S3A"

#: INE geographic codes are NUTS codes without the country prefix, except that
#: the autonomous regions are published as a single digit.
_NUTS_OVERRIDES = {"2": "PT20", "3": "PT30", "1": "PT1", "PT": "PT"}

#: Codes that are not NUTS II regions. They are kept but flagged: including
#: Portugal or Continente alongside their own components would count every
#: observation twice in a panel regression.
AGGREGATE_GEOGRAPHIES = frozenset({"PT", "PT1"})

_MONTH_NUMBERS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

_ACCENTS = str.maketrans("áàâãéêíóôõúç", "aaaaeeiooouc")


class IneError(RuntimeError):
    """Raised when the API responds in a shape the adapter does not expect."""


def to_nuts_code(ine_code: str) -> str:
    """Translate an INE geographic code into a NUTS code.

    Args:
        ine_code: Code as published, such as ``11`` or ``2``.

    Returns:
        The NUTS code, such as ``PT11`` or ``PT20``.
    """
    return _NUTS_OVERRIDES.get(ine_code, f"PT{ine_code}")


def monthly_periods(start: str, end: str) -> list[str]:
    """List the monthly period codes between two months, inclusive.

    Args:
        start: First month as ``YYYY-MM``.
        end: Last month as ``YYYY-MM``.

    Returns:
        Period codes in ascending order.

    Raises:
        IneError: If the range is empty or reversed.
    """
    months = pd.period_range(start=start, end=end, freq="M")
    if len(months) == 0:
        raise IneError(f"no months between {start} and {end}")
    return [f"{MONTHLY_PREFIX}{month.year}{month.month:02d}" for month in months]


def parse_period_label(label: str) -> pd.Timestamp:
    """Convert a Portuguese period label into a month timestamp.

    The API labels periods in prose, as "Janeiro de 2026", and the accented
    month name is frequently mangled by the response encoding, so accents are
    stripped before matching.

    Args:
        label: Period label as returned.

    Returns:
        The first day of that month.

    Raises:
        IneError: If the label is not a recognisable month.
    """
    text = " ".join(str(label).split()).lower().translate(_ACCENTS)
    name, _, year = text.partition(" de ")
    # A mangled accent leaves a replacement character in place of the letter.
    number = _MONTH_NUMBERS.get(name.strip()) or next(
        (
            value
            for month, value in _MONTH_NUMBERS.items()
            if len(month) == len(name.strip()) and month[:2] == name.strip()[:2]
        ),
        None,
    )
    if number is None or not year.strip().isdigit():
        raise IneError(f"unrecognised period label: {label!r}")
    return pd.Timestamp(year=int(year.strip()), month=number, day=1)


def _request(params: dict[str, str], *, attempts: int = 8, timeout_seconds: int = 90) -> Any:
    """Call the API, retrying the refused connections it returns under load."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                INE_API,
                params=params,
                timeout=timeout_seconds,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == attempts:
                raise IneError(f"INE request failed after {attempts} attempts: {error}") from error
            # Linear backoff. The API refuses connections under load rather
            # than returning 429, so the only signal is the failure itself.
            time.sleep(4.0 * attempt)
            continue

        if not isinstance(payload, list) or not payload:
            raise IneError(f"unexpected response shape: {str(payload)[:200]}")

        node = payload[0]
        if "Dados" not in node:
            # A retired or unknown indicator is reported in the body with a 200
            # status, so raise_for_status cannot catch it.
            raise IneError(f"no data returned for {params.get('varcd')}: {str(node)[:200]}")
        return node

    raise IneError(f"unreachable retry state: {last_error}")


def parse_observations(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one API response into observation records.

    Args:
        node: A decoded response node carrying a ``Dados`` block.

    Returns:
        One record per region, category and month, with missing values dropped.
    """
    records: list[dict[str, Any]] = []
    for period_label, rows in node.get("Dados", {}).items():
        month = parse_period_label(period_label)
        for row in rows:
            value = row.get("valor")
            if value in (None, ""):
                continue
            records.append(
                {
                    "month": month,
                    "ine_geocode": row["geocod"],
                    "region": row["geodsg"],
                    "category_code": row.get("dim_3"),
                    "category": row.get("dim_3_t"),
                    "price_index": float(value),
                }
            )
    return records


def fetch_regional_cpi(
    *,
    start: str = "1991-01",
    end: str | None = None,
    indicator: str = REGIONAL_CPI_INDICATOR,
    batch_months: int = 12,
    pause_seconds: float = 1.5,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch the regional consumer price panel.

    Args:
        start: First month as ``YYYY-MM``.
        end: Last month as ``YYYY-MM``. Defaults to the previous calendar month.
        indicator: INE indicator code.
        batch_months: Periods requested per call.
        pause_seconds: Delay between calls. The API is rate limited and starts
            refusing connections when a long history is pulled without pacing,
            so this is what makes a full run complete rather than an increase
            in retries.
        raw_dir: Directory to retain the raw responses in, with a checksum
            manifest. The processed panel is derived, and the indicator it comes
            from is retired without redirect when the series is rebased, so
            without the raw payloads the panel cannot be reproduced.

    Returns:
        Tidy observations with `month`, `nuts_code`, `region`, `category_code`,
        `category`, `price_index` and `is_aggregate`.

    Raises:
        IneError: If the API fails, returns no observations, or is asked for a
            non-positive batch size.
    """
    if batch_months < 1:
        raise IneError(f"batch_months must be positive, got {batch_months}")
    if end is None:
        previous = pd.Timestamp.today().to_period("M") - 1
        end = f"{previous.year}-{previous.month:02d}"

    periods = monthly_periods(start, end)
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    for index in range(0, len(periods), batch_months):
        if index and pause_seconds:
            time.sleep(pause_seconds)
        batch = periods[index : index + batch_months]
        params = {"op": "2", "varcd": indicator, "Dim1": ",".join(batch), "lang": "PT"}
        node = _request(params)
        records.extend(parse_observations(node))

        if raw_dir is not None:
            provenance.append(_retain(node, params, raw_dir, batch))

    if raw_dir is not None and provenance:
        manifest = raw_dir / "regional_cpi_manifest.json"
        manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    if not records:
        raise IneError(f"indicator {indicator} returned no observations for {start}..{end}")

    frame = pd.DataFrame.from_records(records)
    frame["nuts_code"] = frame["ine_geocode"].map(to_nuts_code)
    frame["is_aggregate"] = frame["nuts_code"].isin(AGGREGATE_GEOGRAPHIES)

    return frame.sort_values(["nuts_code", "category_code", "month"]).reset_index(drop=True)


def _retain(
    node: dict[str, Any], params: dict[str, str], raw_dir: Path, batch: list[str]
) -> dict[str, Any]:
    """Write one raw response to disk and return its provenance record.

    The panel is a derived artefact and the indicator behind it is retired
    without redirect when the series is rebased, so the responses are kept with
    their checksums. Without them the processed file could not be reproduced,
    only re-downloaded from a source that may no longer exist.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(node, ensure_ascii=False, sort_keys=True).encode("utf-8")
    name = f"regional_cpi_{batch[0]}_{batch[-1]}.json"
    (raw_dir / name).write_bytes(payload)

    return {
        "file": name,
        "indicator": params["varcd"],
        "periods": batch,
        "url": f"{INE_API}?" + "&".join(f"{k}={v}" for k, v in params.items()),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }
