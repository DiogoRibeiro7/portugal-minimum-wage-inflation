"""Tests for the generated quantities the manuscript quotes.

The reproducibility rule is that the paper contains no transcribed number. That
only holds if the generators are correct, so the two that were added last --- the
exposure summary and the description of how the bootstrap p-value was obtained
--- are checked here rather than trusted because the paper compiled.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pt_mw_inflation.analysis.outputs import write_exposure_macros, write_identification_macros
from pt_mw_inflation.processing.exposure import measure_variation_strength

_DEFINE = re.compile(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}")

REGISTRY: dict[str, object] = {"source": {"reference_period": "2017-10"}}

EXPOSURE = pd.DataFrame(
    {
        "region": ["PT11", "PT15", "PT1A"],
        "regional_bite_exposure": [0.218303, 0.216662, 0.199988],
        "covered_employment_share": [0.833041, 0.808750, 0.894908],
    }
)


def _macros(path: Path) -> dict[str, str]:
    return dict(_DEFINE.findall(path.read_text(encoding="utf-8")))


def test_exposure_macros_carry_the_values_the_prose_cites(tmp_path: Path) -> None:
    """Percentages, the spread and coverage bounds must come from the frame."""
    strength = measure_variation_strength(EXPOSURE)
    written = _macros(write_exposure_macros(EXPOSURE, strength, REGISTRY, tmp_path / "e.tex"))

    assert written["ExposureRegions"] == "3"
    assert written["ExposureMinPct"] == "20.00"
    assert written["ExposureMaxPct"] == "21.83"
    # Reported in percentage points, not in the frame's units.
    assert written["ExposureSpreadPP"] == "1.83"
    assert written["ExposureDistinctValues"] == "3"
    assert written["ExposureCoverageMinPct"] == "81"
    assert written["ExposureCoverageMaxPct"] == "89"


def test_the_bite_date_is_rendered_for_a_sentence(tmp_path: Path) -> None:
    """The configuration keys the bite as YYYY-MM, which reads badly in prose.

    Formatting it in the generator rather than in the manuscript is what stops
    the paper naming a date the pipeline did not read.
    """
    strength = measure_variation_strength(EXPOSURE)
    written = _macros(write_exposure_macros(EXPOSURE, strength, REGISTRY, tmp_path / "e.tex"))
    assert written["ExposureBitePeriod"] == "October 2017"


def test_an_unparseable_bite_date_is_refused(tmp_path: Path) -> None:
    """An undated bite must fail here, not render as an empty macro."""
    strength = measure_variation_strength(EXPOSURE)
    with pytest.raises(ValueError, match="reference period"):
        write_exposure_macros(EXPOSURE, strength, {"source": {}}, tmp_path / "e.tex")


def _estimates(exhaustive: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horizon": [0, 1],
            "coefficient": [0.34, 1.60],
            "standard_error": [0.05, 0.20],
            "t_statistic": [6.8, 8.0],
            "p_value_clustered": [0.0002, 0.0004],
            "p_value_bootstrap": [0.246, 0.098],
            "observations": [900, 900],
            "clusters": [9, 9],
            "bootstrap_exhaustive": [exhaustive, exhaustive],
            "p_value_bootstrap_holm": [0.246, 0.196],
        }
    )


@pytest.mark.parametrize(
    ("exhaustive", "expected"),
    [(True, "exact"), (False, "simulation error")],
)
def test_the_bootstrap_claim_follows_the_run(
    exhaustive: bool, expected: str, tmp_path: Path
) -> None:
    """The paper says its p-value is exact; that has to come from the run.

    Whether the sign space was enumerated depends on how many clusters survive
    the merges, which is a property of the data and not of the method. Asserting
    exactness in prose would let a change in the panel silently falsify the
    sentence, so the sentence is emitted here.
    """
    written = _macros(write_identification_macros(_estimates(exhaustive), 3, 1, tmp_path / "i.tex"))
    assert written["BootstrapClusters"] == "9"
    assert expected in written["BootstrapBasis"]
