"""Adapter for the European Commission's AMECO annual macroeconomic database.

AMECO is the source the research design names for long-run series, and it is
the only one that reaches back to the introduction of the Portuguese minimum
wage. Eurostat's national-accounts productivity series begins in 1995 for
Portugal, which would truncate the historical layer at the point where it
becomes least interesting.

The database is published as semicolon-delimited text inside per-chapter zip
archives. Each row is one series: an identifier, country, sub-chapter, title,
unit, and then one column per year from 1960.

AMECO carries Commission forecasts beyond the last observed year. They are not
data and are excluded by default; keeping them would silently extend every
series with projections and make an estimate look as though it were measured.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import pandas as pd
import requests

from pt_mw_inflation.data.http import USER_AGENT

AMECO_BASE = "https://ec.europa.eu/economy_finance/db_indicators/ameco/documents"

#: Real GDP per person employed, Portugal, at 2015 reference levels. This is
#: the productivity concept the policy benchmark is defined on.
PORTUGAL_REAL_PRODUCTIVITY = "PRT.1.1.0.0.RVGDE"

#: Chapter holding domestic-product series.
DOMESTIC_PRODUCT_CHAPTER = 6

#: AMECO is published in Latin-1, and country names carry accents.
ENCODING = "latin-1"


@dataclass(frozen=True)
class AmecoSeries:
    """One AMECO series with its documentation."""

    code: str
    country: str
    sub_chapter: str
    title: str
    unit: str
    observations: dict[int, float]


def chapter_url(chapter: int) -> str:
    """Return the download URL for one AMECO chapter."""
    return f"{AMECO_BASE}/ameco{chapter}.zip"


def parse_chapter(payload: bytes, code: str) -> AmecoSeries:
    """Extract one series from a downloaded AMECO chapter archive.

    Args:
        payload: Raw bytes of the chapter zip archive.
        code: AMECO series identifier, such as ``PRT.1.1.0.0.RVGDE``.

    Returns:
        The series with its metadata and year-indexed observations.

    Raises:
        ValueError: If the archive is empty or the code is absent, which is how
            a renamed series or a restructured release surfaces.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        if not members:
            raise ValueError("AMECO archive is empty")
        text = archive.read(members[0]).decode(ENCODING)

    lines = text.splitlines()
    header = lines[0].split(";")
    years = [int(value) for value in header[5:] if value.strip().isdigit()]

    for line in lines[1:]:
        fields = line.split(";")
        if fields and fields[0] == code:
            observations: dict[int, float] = {}
            for year, raw in zip(years, fields[5:], strict=False):
                cleaned = raw.strip().replace(",", "")
                if cleaned and cleaned.upper() != "NA":
                    observations[year] = float(cleaned)
            return AmecoSeries(
                code=code,
                country=fields[1],
                sub_chapter=fields[2],
                title=fields[3].strip(),
                unit=fields[4].strip(),
                observations=observations,
            )

    raise ValueError(f"series {code!r} not found in the AMECO chapter")


def fetch_series(
    code: str = PORTUGAL_REAL_PRODUCTIVITY,
    *,
    chapter: int = DOMESTIC_PRODUCT_CHAPTER,
    timeout_seconds: int = 180,
) -> AmecoSeries:
    """Download one AMECO chapter and extract a single series.

    Args:
        code: AMECO series identifier.
        chapter: Chapter number holding the series.
        timeout_seconds: Request timeout.

    Returns:
        The parsed series.

    Raises:
        requests.HTTPError: If the download fails.
        ValueError: If the series is absent from the archive.
    """
    response = requests.get(
        chapter_url(chapter), timeout=timeout_seconds, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    return parse_chapter(response.content, code)


def to_frame(series: AmecoSeries, *, last_actual_year: int | None = None) -> pd.DataFrame:
    """Convert a series to a frame, dropping forecast years.

    Args:
        series: Parsed AMECO series.
        last_actual_year: Final year to retain. Years beyond it are Commission
            projections and are dropped. When omitted, everything is kept, which
            is only appropriate when the caller has already truncated.

    Returns:
        Columns `year`, `value`, and the series `code` for provenance.
    """
    records = sorted(series.observations.items())
    if last_actual_year is not None:
        records = [(year, value) for year, value in records if year <= last_actual_year]

    frame = pd.DataFrame(records, columns=["year", "value"])
    frame["series_code"] = series.code
    return frame
