"""Synthetic panel with a known pass-through function, for testing estimators.

An econometric routine that merely runs proves nothing. These generators build
panels whose true dynamic response is known by construction, so a test can
assert that the estimator recovers it, and that the falsification checks fire
when — and only when — they should.

Two shock regimes are available, and the difference between them is itself
informative:

``annual_shocks=False`` (default)
    Shocks are independent across periods. Successive shocks do not contaminate
    one another's horizon windows, so the local projections identify the true
    cumulative response and a test can assert recovery.

``annual_shocks=True``
    Shocks land every January in every region, as Portuguese statutory changes
    do. Because exposure is then serially correlated and nearly collinear with
    calendar time, the horizon-``h`` window overlaps the next shock and the
    time fixed effects absorb much of the variation. Estimates are attenuated
    even though the estimator is correct. This is a real feature of the
    Portuguese setting rather than an artefact, and it is why the design leans
    on cross-regional exposure differences instead of on timing alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelSpec:
    """Shape of a synthetic region-by-category panel."""

    regions: int = 7
    categories: int = 4
    periods: int = 120
    seed: int = 20260809
    #: True cumulative response at each horizon, applied to the exposure shock.
    response: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.05, 3: 0.15, 6: 0.30})
    noise: float = 0.002
    shock_scale: float = 0.05
    annual_shocks: bool = False
    #: Effect of a shock on price growth three periods *before* it happens.
    #: Non-zero values plant a pre-trend that the diagnostic must detect.
    pre_trend: float = 0.0


def _increments(response: dict[int, float]) -> dict[int, float]:
    """Convert a cumulative response profile into per-period increments."""
    increments: dict[int, float] = {}
    previous = 0.0
    for horizon in sorted(response):
        increments[horizon] = response[horizon] - previous
        previous = response[horizon]
    return increments


def make_panel(spec: PanelSpec | None = None) -> pd.DataFrame:
    """Generate a panel whose prices respond to exposure with a known profile.

    Prices accumulate growth, and a shock at ``t`` raises growth over the
    following periods so that the cumulative log-price change from ``t-1`` to
    ``t+h`` equals ``response[h] * shock`` — exactly the object the local
    projections estimate.

    Args:
        spec: Panel configuration.

    Returns:
        A panel with `region`, `category`, `region_category`, `month`,
        `exposure_shock` and `log_price`.
    """
    spec = spec or PanelSpec()
    rng = np.random.default_rng(spec.seed)
    increments = _increments(spec.response)

    rows = []
    for region_index in range(spec.regions):
        region = f"R{region_index}"
        for category_index in range(spec.categories):
            category = f"C{category_index}"

            if spec.annual_shocks:
                shock = np.zeros(spec.periods)
                intensity = float(rng.uniform(0.5, 1.5))
                for period in range(12, spec.periods, 12):
                    shock[period] = intensity * float(rng.uniform(0.03, 0.09))
            else:
                shock = rng.normal(size=spec.periods) * spec.shock_scale

            growth = rng.normal(scale=spec.noise, size=spec.periods)
            for period in range(spec.periods):
                if shock[period] == 0.0:
                    continue
                for horizon, increment in increments.items():
                    target = period + horizon
                    if target < spec.periods:
                        growth[target] += increment * shock[period]
                lead = period - 3
                if spec.pre_trend and 0 <= lead < spec.periods:
                    growth[lead] += spec.pre_trend * shock[period]

            log_price = np.cumsum(growth) + 4.0
            for period in range(spec.periods):
                rows.append(
                    {
                        "region": region,
                        "category": category,
                        "region_category": f"{region}_{category}",
                        "month": period,
                        "exposure_shock": float(shock[period]),
                        "log_price": float(log_price[period]),
                    }
                )

    return pd.DataFrame(rows)


def make_exposure_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a small, internally consistent set of exposure inputs.

    Bridge weights sum to one for every consumption category, so the inputs
    satisfy the structural contract the exposure module enforces.
    """
    bite = pd.DataFrame(
        [
            {
                "region": region,
                "industry": industry,
                "reference_period": period,
                "minimum_wage_bite": value,
            }
            for region, industry, period, value in [
                ("Norte", "accommodation", 2018, 0.40),
                ("Norte", "manufacturing", 2018, 0.20),
                ("Norte", "accommodation", 2019, 0.45),
                ("Norte", "manufacturing", 2019, 0.22),
                ("Algarve", "accommodation", 2018, 0.55),
                ("Algarve", "manufacturing", 2018, 0.15),
                ("Algarve", "accommodation", 2019, 0.60),
                ("Algarve", "manufacturing", 2019, 0.18),
            ]
        ]
    )
    labour_share = pd.DataFrame(
        [
            {"industry": "accommodation", "labour_cost_share": 0.50},
            {"industry": "manufacturing", "labour_cost_share": 0.25},
        ]
    )
    bridge = pd.DataFrame(
        [
            {"category": "restaurants", "industry": "accommodation", "production_weight": 0.80},
            {"category": "restaurants", "industry": "manufacturing", "production_weight": 0.20},
            {"category": "clothing", "industry": "accommodation", "production_weight": 0.10},
            {"category": "clothing", "industry": "manufacturing", "production_weight": 0.90},
        ]
    )
    return bite, labour_share, bridge
