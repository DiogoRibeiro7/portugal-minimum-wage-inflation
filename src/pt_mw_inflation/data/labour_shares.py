"""Labour-cost shares by economic activity, from Eurostat national accounts.

This is one of the two terms the fuller structural exposure needs. Writing
:math:`\\ell_s` for the share of value added paid as compensation of employees,

.. math::

    B_{rc} = \\sum_s q_{rs}\\, b_s\\, \\ell_s\\, \\omega_{cs},

the labour-cost share is what converts "this region works in low-paying
industries" into "a wage rise raises this region's costs". An industry with a
large minimum-wage bite but little labour in its cost structure passes less of a
statutory rise into prices than the bite alone suggests.

The share is national and is not, on its own, a source of regional variation.
What makes it useful is the interaction: once region-time and category-time
effects are absorbed, identification comes from the non-additive part of the
region-by-category matrix, and the labour share enters that part. An earlier
version of this project dismissed these terms for exactly the reason that turns
out to be wrong; ``docs/decision_log.md`` records the error.

One artefact of the source deserves naming rather than silent handling. Value
added in real estate is dominated by imputed rentals of owner-occupied
dwellings, which involve no employment, so the measured labour share there is
close to zero. That is arithmetically correct and economically misleading, and
any exposure measure using it should say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from pt_mw_inflation.data.eurostat_regional import EUROSTAT_API, _decode
from pt_mw_inflation.data.http import USER_AGENT

#: National accounts by 64 activities, the source of both series.
NATIONAL_ACCOUNTS_DATASET = "nama_10_a64"

#: Compensation of employees, and value added at basic prices.
COMPENSATION = "D1"
VALUE_ADDED = "B1G"

#: Current prices, million euro. Both series must share a unit or their ratio is
#: not a share of anything.
CURRENT_PRICES = "CP_MEUR"

#: Activities whose measured share is an accounting artefact rather than a
#: description of their cost structure. Real estate value added is dominated by
#: imputed rentals of owner-occupied dwellings, which employ nobody.
IMPUTED_RENT_ACTIVITIES = frozenset({"L", "L68", "L68A"})


class LabourShareError(RuntimeError):
    """Raised when the labour-share response cannot be used."""


@dataclass(frozen=True)
class LabourShares:
    """Labour-cost shares with the checks that make them usable.

    Attributes:
        shares: Activity code mapped to compensation over value added.
        reference_year: Year both series were taken from.
        aggregate: The economy-wide share, for validation against a known value.
        suppressed: Activities excluded because their share is an artefact.
    """

    shares: dict[str, float]
    reference_year: int
    aggregate: float
    suppressed: tuple[str, ...]


def fetch_labour_shares(
    *,
    year: int,
    country: str = "PT",
    dataset: str = NATIONAL_ACCOUNTS_DATASET,
    timeout_seconds: int = 180,
    drop_imputed_rent: bool = True,
) -> LabourShares:
    """Fetch compensation of employees over value added, by activity.

    Args:
        year: Reference year. Both series are taken from the same year, because
            a ratio of two years is not a share.
        country: Geography code.
        dataset: Eurostat dataset code.
        timeout_seconds: Request timeout.
        drop_imputed_rent: Exclude the real-estate activities whose share is
            dominated by imputed rents.

    Returns:
        The shares, with the aggregate retained for validation.

    Raises:
        requests.HTTPError: If either request fails.
        LabourShareError: If the two series share no activity, or the aggregate
            is outside the range any national accounts should produce.
    """

    def series(na_item: str) -> dict[str, float]:
        response = requests.get(
            f"{EUROSTAT_API}/{dataset}",
            params={
                "format": "JSON",
                "lang": "EN",
                "geo": country,
                "na_item": na_item,
                "unit": CURRENT_PRICES,
                "time": str(year),
            },
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        frame = _decode(json.loads(response.content))
        return dict(zip(frame["nace_r2"].astype(str), frame["value"].astype(float), strict=True))

    compensation = series(COMPENSATION)
    value_added = series(VALUE_ADDED)

    shared = sorted(set(compensation) & set(value_added))
    if not shared:
        raise LabourShareError(f"{COMPENSATION} and {VALUE_ADDED} share no activity for {year}")

    shares: dict[str, float] = {}
    suppressed: list[str] = []
    for activity in shared:
        denominator = value_added[activity]
        if denominator <= 0:
            continue
        if drop_imputed_rent and activity in IMPUTED_RENT_ACTIVITIES:
            suppressed.append(activity)
            continue
        shares[activity] = compensation[activity] / denominator

    aggregate = shares.get("TOTAL")
    if aggregate is None:
        total_denominator = value_added.get("TOTAL")
        if not total_denominator:
            raise LabourShareError("no economy-wide total to validate the shares against")
        aggregate = compensation["TOTAL"] / total_denominator

    # A national labour share sits well inside this band in every developed
    # economy. Outside it, the two series are not on the same basis --- a unit
    # mismatch is the usual cause, and it would otherwise pass silently.
    if not 0.2 <= aggregate <= 0.8:
        raise LabourShareError(
            f"economy-wide labour share {aggregate:.3f} is implausible; "
            "the two series are probably not on the same basis"
        )

    return LabourShares(
        shares=shares,
        reference_year=year,
        aggregate=aggregate,
        suppressed=tuple(suppressed),
    )


def to_frame(result: LabourShares) -> pd.DataFrame:
    """Tabulate the shares for joining onto an activity-level measure.

    Args:
        result: Output of :func:`fetch_labour_shares`.

    Returns:
        Columns `activity`, `labour_cost_share` and `reference_year`.
    """
    frame = pd.DataFrame(sorted(result.shares.items()), columns=["activity", "labour_cost_share"])
    frame["reference_year"] = result.reference_year
    return frame
