"""Long-run annual macro dataset: wages, prices, productivity and the residual.

This is the historical accounting layer of the research design. It answers a
descriptive question — has the statutory minimum wage grown faster or slower
than productivity plus prior inflation — and motivates the causal design
without itself being causal evidence.

The policy residual is defined exactly as in the design. For year ``t``:

    benchmark_t = (1 + productivity_growth_t)(1 + inflation_{t-1}) - 1
    residual_t  = minimum_wage_growth_t - benchmark_t

The benchmark compounds rather than adding, so a decade of high inflation does
not accumulate an approximation error into the residual. In 1984 Portuguese
inflation was near thirty per cent, where the additive shortcut is wrong by
more than a percentage point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_INPUTS = frozenset({"year", "minimum_wage", "cpi", "productivity"})


class MacroDatasetError(ValueError):
    """Raised when the macro inputs cannot produce a coherent dataset."""


def build_macro_annual(
    minimum_wage: pd.DataFrame,
    consumer_prices: pd.DataFrame,
    productivity: pd.DataFrame,
    *,
    start_year: int = 1974,
    end_year: int | None = None,
    inflation_lag: int = 1,
    wage_column: str = "minimum_wage_mean",
) -> pd.DataFrame:
    """Assemble the annual macro dataset from its three inputs.

    Args:
        minimum_wage: Annual statutory wage, with `year` and `wage_column`.
        consumer_prices: Annual CPI, with `year` and `value`.
        productivity: Annual real productivity, with `year` and `value`.
        start_year: First year retained.
        end_year: Last year retained. Defaults to the last year present in all
            three inputs.
        inflation_lag: Years by which inflation enters the wage benchmark.
        wage_column: Which annual wage convention to use.

    Returns:
        One row per year with levels, growth rates, the benchmark and the
        residual, plus indices normalised to the first common year.

    Raises:
        MacroDatasetError: If an input lacks required columns, or the three
            series share fewer than two years, which cannot yield a growth rate.
    """
    for name, frame, needed in (
        ("minimum_wage", minimum_wage, {"year", wage_column}),
        ("consumer_prices", consumer_prices, {"year", "value"}),
        ("productivity", productivity, {"year", "value"}),
    ):
        missing = needed.difference(frame.columns)
        if missing:
            raise MacroDatasetError(f"{name} missing columns: {sorted(missing)}")

    wages = minimum_wage[["year", wage_column]].rename(columns={wage_column: "minimum_wage"})
    prices = consumer_prices[["year", "value"]].rename(columns={"value": "cpi"})
    output = productivity[["year", "value"]].rename(columns={"value": "productivity"})

    frame = wages.merge(prices, on="year", how="inner").merge(output, on="year", how="inner")
    frame = frame.loc[frame["year"] >= start_year]
    if end_year is not None:
        frame = frame.loc[frame["year"] <= end_year]
    frame = frame.sort_values("year").reset_index(drop=True)

    if len(frame) < 2:
        raise MacroDatasetError(
            f"only {len(frame)} year(s) common to all three inputs; "
            "at least two are needed to form a growth rate"
        )

    if (frame["cpi"] <= 0).any() or (frame["productivity"] <= 0).any():
        raise MacroDatasetError("cpi and productivity must be strictly positive")

    frame["inflation"] = frame["cpi"].pct_change()
    frame["productivity_growth"] = frame["productivity"].pct_change()
    frame["minimum_wage_growth"] = frame["minimum_wage"].pct_change()
    frame["lagged_inflation"] = frame["inflation"].shift(inflation_lag)

    frame["benchmark_wage_growth"] = (1.0 + frame["productivity_growth"]) * (
        1.0 + frame["lagged_inflation"]
    ) - 1.0
    frame["policy_residual"] = frame["minimum_wage_growth"] - frame["benchmark_wage_growth"]

    base_cpi = frame["cpi"].iloc[0]
    base_wage = frame["minimum_wage"].iloc[0]
    base_productivity = frame["productivity"].iloc[0]

    frame["real_minimum_wage"] = frame["minimum_wage"] / frame["cpi"] * base_cpi
    frame["minimum_wage_index"] = frame["minimum_wage"] / base_wage * 100.0
    frame["real_minimum_wage_index"] = frame["real_minimum_wage"] / base_wage * 100.0
    frame["productivity_index"] = frame["productivity"] / base_productivity * 100.0
    # Minimum wage measured against productivity: above 100 means the real wage
    # floor has risen faster than output per worker since the base year.
    frame["minimum_wage_to_productivity_index"] = (
        frame["real_minimum_wage_index"] / frame["productivity_index"] * 100.0
    )
    frame["log_minimum_wage_growth"] = np.log(frame["minimum_wage"]).diff()
    # Compounded, not summed. The benchmark itself compounds productivity with
    # prior inflation precisely because addition is wrong over long horizons at
    # high inflation; summing the resulting residuals would reintroduce the
    # error the benchmark avoids. The arithmetic sum is retained separately and
    # named for what it is.
    growth_ratio = (1.0 + frame["minimum_wage_growth"]) / (1.0 + frame["benchmark_wage_growth"])
    frame["cumulative_policy_gap"] = growth_ratio.fillna(1.0).cumprod() - 1.0
    frame["summed_annual_residual"] = frame["policy_residual"].fillna(0.0).cumsum()

    return frame


def summarise_by_regime(frame: pd.DataFrame, regimes: list[dict[str, object]]) -> pd.DataFrame:
    """Average the annual series within each configured policy regime.

    Args:
        frame: Output of :func:`build_macro_annual`.
        regimes: Regime definitions with `name`, `start` and `end`, as held in
            `config/analysis.yaml`.

    Returns:
        One row per regime with mean growth rates and the residual, plus the
        number of years actually observed. Regimes with no observed years are
        dropped rather than reported as empty.

    Raises:
        MacroDatasetError: If a regime definition is malformed.
    """
    rows = []
    for regime in regimes:
        try:
            name = str(regime["name"])
            start = int(regime["start"])  # type: ignore[call-overload]
            end = int(regime["end"])  # type: ignore[call-overload]
        except (KeyError, TypeError, ValueError) as error:
            raise MacroDatasetError(f"malformed regime definition {regime!r}") from error

        window = frame.loc[frame["year"].between(start, end)]
        if window.empty:
            continue

        rows.append(
            {
                "regime": name,
                "start": start,
                "end": end,
                "years_observed": int(window["year"].nunique()),
                "mean_inflation": float(window["inflation"].mean()),
                "mean_productivity_growth": float(window["productivity_growth"].mean()),
                "mean_minimum_wage_growth": float(window["minimum_wage_growth"].mean()),
                "mean_benchmark_growth": float(window["benchmark_wage_growth"].mean()),
                "mean_policy_residual": float(window["policy_residual"].mean()),
                "real_wage_change": float(
                    window["real_minimum_wage_index"].iloc[-1]
                    / window["real_minimum_wage_index"].iloc[0]
                    - 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def check_accounting_identities(frame: pd.DataFrame, *, tolerance: float = 1e-9) -> None:
    """Verify the identities the dataset is built on.

    A silent violation would mean the residual no longer measures what the
    manuscript says it measures, so this is checked on every build rather than
    only in the test suite.

    Args:
        frame: Output of :func:`build_macro_annual`.
        tolerance: Largest tolerated absolute deviation.

    Raises:
        MacroDatasetError: If any identity fails.
    """
    sample = frame.dropna(
        subset=["policy_residual", "benchmark_wage_growth", "minimum_wage_growth"]
    )
    if sample.empty:
        raise MacroDatasetError("no complete observations to verify")

    residual_gap = (
        sample["policy_residual"]
        - (sample["minimum_wage_growth"] - sample["benchmark_wage_growth"])
    ).abs()
    if float(residual_gap.max()) > tolerance:
        raise MacroDatasetError(f"residual identity violated by {float(residual_gap.max()):.3e}")

    benchmark = (1.0 + sample["productivity_growth"]) * (1.0 + sample["lagged_inflation"]) - 1.0
    benchmark_gap = (sample["benchmark_wage_growth"] - benchmark).abs()
    if float(benchmark_gap.max()) > tolerance:
        raise MacroDatasetError(f"benchmark identity violated by {float(benchmark_gap.max()):.3e}")

    real = sample["minimum_wage"] / sample["cpi"] * frame["cpi"].iloc[0]
    real_gap = (sample["real_minimum_wage"] - real).abs()
    if float(real_gap.max()) > tolerance * max(1.0, float(real.abs().max())):
        raise MacroDatasetError(f"real wage identity violated by {float(real_gap.max()):.3e}")
