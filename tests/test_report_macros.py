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

from pt_mw_inflation.analysis.outputs import (
    write_exposure_macros,
    write_headline_macros,
    write_identification_macros,
)
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
    [(True, "no simulation error"), (False, "carries simulation error")],
)
def test_the_bootstrap_claim_follows_the_run(
    exhaustive: bool, expected: str, tmp_path: Path
) -> None:
    """What the bootstrap achieved has to come from the run that achieved it.

    Whether the sign space was enumerated depends on how many clusters survive
    the merges, which is a property of the data and not of the method, so the
    sentence is emitted rather than written.

    The claim itself is bounded deliberately. Enumeration removes simulation
    error; it does not make the test exact, because the sign-flip distribution
    approximates the null rather than being it. An earlier version of this test
    asserted the word "exact" and so defended a claim the inference module's own
    docstring rejects.
    """
    written = _macros(write_identification_macros(_estimates(exhaustive), 3, 1, tmp_path / "i.tex"))
    assert written["BootstrapClusters"] == "9"
    assert expected in written["BootstrapBasis"]


def test_the_base_year_sensitivity_is_generated(tmp_path: Path) -> None:
    """The wage floor arrived in May, so its first year is not a full year.

    Pairing a rate in force for eight months with a full-calendar-year price
    and productivity observation is a convention, and the obvious check is to
    rebase on the first complete year. Reporting it turns "the base year is
    awkward" into a quantity a reader can weigh rather than a caveat.
    """
    macro = pd.DataFrame(
        {
            "year": [1974, 1975, 2025],
            "minimum_wage": [16.46, 18.36, 870.0],
            "policy_residual": [None, -0.12, 0.03],
            "cumulative_policy_gap": [0.0, -0.1, -0.56],
            "summed_annual_residual": [0.0, -0.1, -1.02],
            "real_minimum_wage_index": [100.0, 96.8, 128.3],
            "productivity_index": [100.0, 96.0, 257.0],
            "minimum_wage_to_productivity_index": [100.0, 50.0, 25.0],
        }
    )
    written = _macros(write_headline_macros(macro, tmp_path / "headline.tex"))

    assert written["MacroSecondYear"] == "1975"
    # Rebasing on a second year worth half the first doubles the end value.
    assert written["WageToProductivityEnd"] == "25.0"
    assert written["WageToProductivityEndAlt"] == "50.0"


def test_a_shared_writer_names_the_command_that_invoked_it(tmp_path: Path) -> None:
    """A banner naming the wrong command is worse than no banner at all.

    Two designs share the pre-trend writer, and two share the design-table
    writer. Both defaulted to hardcoding one command, so the file produced by
    the other told its reader to regenerate it with something that would not
    produce it --- and, being plausible, would appear to have been followed.

    This has been fixed twice: once in the design-table writer, and again in the
    pre-trend writer after the same shape was copied. The third time is what
    this test is for.
    """
    from pt_mw_inflation.analysis.inference import JointTest
    from pt_mw_inflation.analysis.outputs import write_pre_trend_macros

    result = JointTest(
        statistic=112.0,
        p_value=0.711,
        restrictions=5,
        clusters=9,
        draws=512,
        exhaustive=True,
    )

    default = write_pre_trend_macros(result, tmp_path / "a.tex")
    named = write_pre_trend_macros(
        result,
        tmp_path / "b.tex",
        prefix="Exposure",
        command="ptmw analyse exposure-design",
    )

    assert "ptmw analyse pass-through" in default.read_text(encoding="utf-8")
    banner = named.read_text(encoding="utf-8")
    assert "ptmw analyse exposure-design" in banner
    assert "ptmw analyse pass-through" not in banner


def test_a_prefixed_writer_does_not_collide_with_the_unprefixed_one(tmp_path: Path) -> None:
    """Two designs must emit distinct quantities, not overwrite each other.

    Without the prefix both files define the same macro names, so whichever
    ran last would silently supply the figures the manuscript attributes to
    the other design.
    """
    from pt_mw_inflation.analysis.inference import JointTest
    from pt_mw_inflation.analysis.outputs import write_pre_trend_macros

    result = JointTest(
        statistic=7332.0, p_value=0.109, restrictions=5, clusters=9, draws=512, exhaustive=True
    )
    plain = _macros(write_pre_trend_macros(result, tmp_path / "a.tex"))
    prefixed = _macros(write_pre_trend_macros(result, tmp_path / "b.tex", prefix="Exposure"))

    assert not set(plain) & set(prefixed)
    assert "PreTrendP" in plain
    assert "ExposurePreTrendP" in prefixed
