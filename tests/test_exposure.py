"""Tests for the structural exposure construction."""

import pandas as pd
import pytest

from pt_mw_inflation.processing.exposure import construct_cost_exposure


def test_cost_exposure_multiplies_bite_labour_share_and_bridge_weight() -> None:
    bite = pd.DataFrame(
        {"region": ["Norte"], "industry": ["I"], "reference_period": [2019], "minimum_wage_bite": [0.4]}
    )
    labour = pd.DataFrame({"industry": ["I"], "labour_cost_share": [0.5]})
    bridge = pd.DataFrame({"category": ["restaurants"], "industry": ["I"], "production_weight": [0.8]})

    result = construct_cost_exposure(bite, labour, bridge)
    assert result.loc[0, "cost_exposure"] == pytest.approx(0.16)
