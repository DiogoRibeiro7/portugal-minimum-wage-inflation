"""Regional employment by industry, from Eurostat's regional accounts.

This supplies the regional industry composition that the shift-share exposure
measure needs. An earlier version of this project concluded that no source
crossed a regional dimension with an industry dimension, and disabled the
exposure design on that basis. That conclusion was drawn from the Portuguese
labour-ministry publications alone and was wrong: Eurostat's regional accounts
publish employment by NUTS region and NACE activity for Portugal from 2000.

What the measure assumes is unchanged, and remains substantive. Regional
composition is observed; the minimum-wage bite within an industry is not
observed regionally and is taken at its national value. Accommodation and food
service has by far the highest bite of any activity and differs across
Portuguese regions in both its size and its wage distribution, so holding the
within-industry bite constant attenuates exactly the variation the design
exploits. The construction belongs in robustness, clearly labelled, not in a
baseline.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests

from pt_mw_inflation.data.http import USER_AGENT

EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

#: Employment by NUTS region and NACE activity, thousands of persons.
REGIONAL_EMPLOYMENT_DATASET = "nama_10r_3empers"

#: Mainland and island NUTS II regions under the 2024 classification.
PORTUGUESE_REGIONS = ("PT11", "PT15", "PT19", "PT1A", "PT1B", "PT1C", "PT1D", "PT20", "PT30")

#: A partition of total employment. Eurostat publishes overlapping aggregates
#: alongside their components, so the set used has to be chosen deliberately:
#: summing everything published would count most workers several times.
NACE_PARTITION = ("A", "B-E", "F", "G-I", "J", "K", "L", "M_N", "O-Q", "R-U")

TOTAL_ACTIVITY = "TOTAL"


class RegionalEmploymentError(RuntimeError):
    """Raised when the regional employment response cannot be used."""


def _decode(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a JSON-stat response into tidy observations."""
    dimensions: list[str] = payload["id"]
    sizes: list[int] = payload["size"]

    categories: dict[str, list[str]] = {}
    for name in dimensions:
        index = payload["dimension"][name]["category"]["index"]
        if isinstance(index, dict):
            categories[name] = [code for code, _ in sorted(index.items(), key=lambda kv: kv[1])]
        else:
            categories[name] = [str(code) for code in index]

    strides: list[int] = [1] * len(sizes)
    for position in range(len(sizes) - 2, -1, -1):
        strides[position] = strides[position + 1] * sizes[position + 1]

    records: list[dict[str, Any]] = []
    for flat, value in payload.get("value", {}).items():
        if value is None:
            continue
        remainder = int(flat)
        row: dict[str, Any] = {}
        for position, name in enumerate(dimensions):
            row[name] = categories[name][remainder // strides[position] % sizes[position]]
        row["value"] = float(value)
        records.append(row)

    if not records:
        raise RegionalEmploymentError("response carried no observations")
    return pd.DataFrame.from_records(records)


#: Employees, in the regional accounts' vocabulary. The counterpart of
#: :data:`EMPLOYEES` below, which names the same population in the national
#: accounts under a different code.
REGIONAL_EMPLOYEES = "SAL"

#: All persons in employment, employees and self-employed together.
REGIONAL_TOTAL_EMPLOYMENT = "EMP"


def fetch_regional_employment(
    *,
    regions: tuple[str, ...] = PORTUGUESE_REGIONS,
    dataset: str = REGIONAL_EMPLOYMENT_DATASET,
    wstatus: str = REGIONAL_EMPLOYEES,
    timeout_seconds: int = 180,
) -> pd.DataFrame:
    """Fetch employment by region and activity.

    Counts employees by default. The exposure measure multiplies a region's
    industry composition by the share of employees in that industry paid the
    minimum wage, so composition and bite have to be shares of the same
    population. Counting the self-employed in the composition and not in the
    bite would make a region's exposure depend on how much self-employment its
    industries happen to carry, which is not what the measure is about, and
    would put the coverage figures on a different denominator from the weights.

    Args:
        regions: NUTS codes to request.
        dataset: Eurostat dataset code.
        wstatus: Population to count.
        timeout_seconds: Request timeout.

    Returns:
        Columns `region`, `activity`, `year`, `employment_thousands` and
        `population`, the last so a later stage can refuse to combine
        populations that do not match.

    Raises:
        requests.HTTPError: If the request fails.
        RegionalEmploymentError: If the response carries no observations.
    """
    params: list[tuple[str, str]] = [
        ("format", "JSON"),
        ("lang", "EN"),
        ("wstatus", wstatus),
        ("unit", "THS"),
    ]
    params.extend(("geo", region) for region in regions)

    response = requests.get(
        f"{EUROSTAT_API}/{dataset}",
        params=params,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    frame = _decode(json.loads(response.content))

    result = (
        frame.rename(
            columns={
                "geo": "region",
                "nace_r2": "activity",
                "time": "year",
                "value": "employment_thousands",
            }
        )[["region", "activity", "year", "employment_thousands"]]
        .astype({"year": int})
        .sort_values(["region", "activity", "year"])
    )
    result["population"] = wstatus
    return result.reset_index(drop=True)


#: Employees, in the national accounts' vocabulary. The counterpart of
#: :data:`REGIONAL_EMPLOYEES`, which names the same people under another code.
#: The bite these weights aggregate is a share of *employees* paid the minimum
#: wage, so weighting it by a population that also contains the self-employed
#: would give a sector's bite an influence proportional to a headcount the bite
#: was never measured over. Agriculture and construction carry far more
#: self-employment than finance, so the mismatch is not a wash across sectors.
EMPLOYEES = "SAL_DC"

#: All persons in employment, employees and self-employed together. Retained as
#: an alternative basis, admissible as long as it is used on both sides.
TOTAL_EMPLOYMENT = "EMP_DC"

#: The pairings that name the same people in the two datasets, each with the
#: plain-English name of who that is. The vocabularies differ, so a pairing is
#: checked against this rather than the two codes compared for equality; and it
#: is reported by name, because "both count SAL" beside a frame stamped SAL_DC
#: reads as a mismatch that has just been accepted.
MATCHED_POPULATIONS: dict[tuple[str, str], str] = {
    (REGIONAL_EMPLOYEES, EMPLOYEES): "employees",
    (REGIONAL_TOTAL_EMPLOYMENT, TOTAL_EMPLOYMENT): "all persons in employment",
}


def require_matched_inputs(
    regional: pd.DataFrame, national: pd.DataFrame, *, baseline_year: int
) -> str:
    """Check that composition and weights describe the same people in the same year.

    Composition is frozen at ``baseline_year``; the weights were frozen at
    whatever year was passed to the download. Nothing tied the two together, so
    changing one produced a measure labelled as frozen at a year only half of it
    was frozen at. The populations are the same trap one level down: composition
    counting the self-employed against a bite weighted on employees alone puts
    the coverage figures on a denominator the weights do not share.

    An unstamped frame is refused rather than warned about. Both files are one
    command away from being regenerated, so there is no case for proceeding on a
    pairing that cannot be verified.

    Args:
        regional: Regional employment, as downloaded.
        national: National employment by NACE section, as downloaded.
        baseline_year: Year the composition is frozen at.

    Returns:
        The plain-English name of the population both frames count.

    Raises:
        RegionalEmploymentError: If either frame lacks its provenance stamps,
            mixes populations, was frozen at another year, or counts a different
            population from the other.
    """
    unstamped = [
        name
        for name, frame in (("regional", regional), ("national", national))
        if "population" not in frame.columns
    ]
    if "reference_year" not in national.columns:
        unstamped.append("national")
    if unstamped:
        raise RegionalEmploymentError(
            f"employment frames {sorted(set(unstamped))} predate the provenance stamps; "
            "re-run 'ptmw data regional-employment' so the year and population can be checked"
        )

    weight_years = sorted({int(year) for year in national["reference_year"]})
    if weight_years != [baseline_year]:
        raise RegionalEmploymentError(
            f"national weights are for {weight_years} but composition is frozen at "
            f"{baseline_year}; re-run 'ptmw data regional-employment --year {baseline_year}'"
        )

    populations = {"regional": set(regional["population"]), "national": set(national["population"])}
    if any(len(values) != 1 for values in populations.values()):
        raise RegionalEmploymentError(f"employment frames mix populations: {populations}")

    pairing = (next(iter(populations["regional"])), next(iter(populations["national"])))
    named = MATCHED_POPULATIONS.get(pairing)
    if named is None:
        raise RegionalEmploymentError(
            f"composition counts {pairing[0]} but the weights count {pairing[1]}; "
            "re-run 'ptmw data regional-employment' so both count the same people"
        )
    return named


def industry_shares(
    employment: pd.DataFrame,
    *,
    year: int,
    partition: tuple[str, ...] = NACE_PARTITION,
    tolerance: float = 0.06,
) -> pd.DataFrame:
    """Compute each region's employment share by activity in one year.

    Args:
        employment: Output of :func:`fetch_regional_employment`.
        year: The predetermined year the shares are frozen at.
        partition: Activities forming a partition of total employment.
        tolerance: Largest accepted gap between the partition's sum and the
            published total, as a share.

    Returns:
        Columns `region`, `activity`, `employment_share` and the frozen year.

    Raises:
        RegionalEmploymentError: If the year is absent, or if the chosen
            activities do not reconstruct the published total, which is how an
            overlapping aggregate slipped into the partition would surface.
    """
    frozen = employment.loc[employment["year"] == year]
    if frozen.empty:
        available = sorted(employment["year"].unique())
        raise RegionalEmploymentError(
            f"year {year} absent; available {available[:3]}..{available[-3:]}"
        )

    selected = frozen.loc[frozen["activity"].isin(partition)]
    totals = frozen.loc[frozen["activity"] == TOTAL_ACTIVITY].set_index("region")[
        "employment_thousands"
    ]

    summed = selected.groupby("region")["employment_thousands"].sum()
    common = summed.index.intersection(totals.index)
    if common.empty:
        raise RegionalEmploymentError("no region has both a total and its components")

    gap = ((summed[common] - totals[common]).abs() / totals[common]).max()
    if float(gap) > tolerance:
        raise RegionalEmploymentError(
            f"the chosen activities miss the published total by {float(gap):.1%}; "
            "the partition either overlaps or omits an activity"
        )

    shares = selected.copy()
    shares["employment_share"] = shares["employment_thousands"] / shares["region"].map(summed)
    shares["baseline_year"] = year
    return shares[["region", "activity", "employment_share", "baseline_year"]].reset_index(
        drop=True
    )


#: National employment by detailed NACE activity. Supplies the weights needed to
#: aggregate a section-level bite onto the coarser groups the regional accounts
#: publish, and to see how much of a group has no measured bite at all.
NATIONAL_EMPLOYMENT_DATASET = "nama_10_a64_e"


def fetch_national_employment(
    *,
    year: int,
    country: str = "PT",
    dataset: str = NATIONAL_EMPLOYMENT_DATASET,
    na_item: str = EMPLOYEES,
    timeout_seconds: int = 180,
) -> pd.DataFrame:
    """Fetch national employment by NACE activity for one year.

    Args:
        year: Reference year.
        country: Geography code.
        dataset: Eurostat dataset code.
        na_item: Population to count. Defaults to employees, which is the
            population the minimum-wage bite is measured over.
        timeout_seconds: Request timeout.

    Returns:
        Columns `activity`, `employment_thousands`, `reference_year` and
        `population`. The last two travel with the data so a later stage can
        refuse to combine weights with a composition frozen at another year, or
        with a bite measured over a different population.

    Raises:
        requests.HTTPError: If the request fails.
        RegionalEmploymentError: If the response carries no observations.
    """
    response = requests.get(
        f"{EUROSTAT_API}/{dataset}",
        params={
            "format": "JSON",
            "lang": "EN",
            "geo": country,
            "na_item": na_item,
            "unit": "THS_PER",
            "time": str(year),
        },
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    frame = _decode(json.loads(response.content))

    result = frame.rename(columns={"nace_r2": "activity", "value": "employment_thousands"})[
        ["activity", "employment_thousands"]
    ].sort_values("activity")
    result["reference_year"] = year
    result["population"] = na_item
    return result.reset_index(drop=True)
