"""Tests for the policy-residual accounting identity."""

import pandas as pd
import pytest

from pt_mw_inflation.processing.minimum_wage import build_policy_residual


def test_policy_residual_is_zero_when_wage_matches_productivity_and_lagged_inflation() -> None:
    frame = pd.DataFrame(
        {
            "year": [2020, 2021],
            "minimum_wage": [100.0, 107.1],
            "inflation": [0.05, 0.02],
            "productivity_growth": [0.00, 0.02],
        }
    )
    result = build_policy_residual(frame)
    assert result.loc[1, "benchmark_wage_growth"] == pytest.approx(0.071)
    assert result.loc[1, "policy_residual"] == pytest.approx(0.0)
