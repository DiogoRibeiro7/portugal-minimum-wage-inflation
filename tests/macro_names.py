"""Every macro name the pipeline can define, derived without touching the data.

The manuscript consistency checks ask a question about the *writers* --- which
quantities does the pipeline define? --- and an earlier version answered it by
reading whatever macro files happened to be lying in ``report/tables``. Those
files are build outputs and are not committed, so on a clean checkout there was
nothing to read and the checks skipped. They therefore never ran in continuous
integration, only on a machine that had already built the paper, which is the
one place they were least needed.

Running each writer over synthetic inputs answers the same question offline, in
milliseconds, and answers it better: a macro that survives in a stale local file
but is no longer emitted by any writer is now visible as undefined, which
reading the directory could never detect.

Adding a macro writer means adding it here. That is deliberate: a writer nothing
exercises is a writer whose output the manuscript cannot safely cite.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from pt_mw_inflation.analysis.inference import JointTest
from pt_mw_inflation.analysis.outputs import (
    write_exposure_design_macros,
    write_exposure_macros,
    write_headline_macros,
    write_identification_macros,
    write_pre_trend_macros,
    write_regional_premium_macros,
    write_seasonality_macros,
    write_structural_design_macros,
)
from pt_mw_inflation.processing.exposure import measure_variation_strength
from pt_mw_inflation.processing.pass_through import SeasonalConfound

_DEFINE = re.compile(r"\\providecommand\{\\([A-Za-z]+)\}")


def _macro_frame() -> pd.DataFrame:
    """Two years of the long-run annual dataset, enough to exercise the writer."""
    return pd.DataFrame(
        {
            "year": [1974, 2025],
            "minimum_wage": [16.46, 870.0],
            "policy_residual": [-0.12, 0.03],
            "cumulative_policy_gap": [0.0, -0.56],
            "summed_annual_residual": [0.0, -1.02],
            "real_minimum_wage_index": [100.0, 128.3],
            "productivity_index": [100.0, 257.0],
            "minimum_wage_to_productivity_index": [100.0, 49.9],
        }
    )


def _estimates() -> pd.DataFrame:
    """Two horizons carrying every column the identification writer reads.

    The inverted interval is included because the manuscript now leads with it.
    A writer that emits interval macros only when the estimator produced them
    would otherwise define nothing here, and the prose citing them would fail
    this check rather than the pipeline being fixed.
    """
    return pd.DataFrame(
        {
            "horizon": [0, 1],
            "coefficient": [0.12, 0.23],
            "standard_error": [0.02, 0.04],
            "t_statistic": [6.0, 5.8],
            "p_value_clustered": [0.0002, 0.0004],
            "p_value_bootstrap": [0.238, 0.027],
            "p_value_bootstrap_holm": [1.0, 0.191],
            "observations": [900, 900],
            "clusters": [9, 9],
            "bootstrap_exhaustive": [True, True],
            "interval_lower": [-0.11, 0.03],
            "interval_upper": [0.32, 0.40],
            "interval_bounded": [True, True],
        }
    )


def _exposure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["PT11", "PT1A"],
            "regional_bite_exposure": [0.2168, 0.1974],
            "covered_employment_share": [0.8935, 0.8872],
        }
    )


def _wage_panel() -> pd.DataFrame:
    """A mainland series and a region on a fixed proportional premium."""
    rows = []
    for year, national in ((2015, 505.0), (2016, 530.0), (2017, 557.0)):
        rows.append(
            {
                "geography": "PT",
                "effective_date": date(year, 1, 1),
                "minimum_wage_monthly_eur": national,
            }
        )
        premium = 1.02 if year < 2017 else 1.023
        rows.append(
            {
                "geography": "PT30",
                "effective_date": date(year, 1, 1),
                "minimum_wage_monthly_eur": round(national * premium, 2),
            }
        )
    return pd.DataFrame(rows)


def definable_macros(directory: Path) -> set[str]:
    """Run every macro writer over synthetic inputs and collect what it defines.

    Args:
        directory: Scratch directory the writers may write into.

    Returns:
        Every macro name the pipeline is capable of defining.
    """
    written = [
        write_headline_macros(_macro_frame(), directory / "headline.tex"),
        write_identification_macros(_estimates(), 10, 1, directory / "identification.tex"),
        write_exposure_macros(
            _exposure(),
            measure_variation_strength(_exposure()),
            {"source": {"reference_period": "2017-10"}},
            directory / "exposure.tex",
        ),
        write_regional_premium_macros(_wage_panel(), directory / "premium.tex"),
        write_exposure_design_macros(_estimates(), directory / "exposure_design.tex"),
        write_structural_design_macros(
            _estimates(),
            directory / "structural_design.tex",
            identifying_spread=1.92,
            cost_share_ceiling=0.1127,
        ),
        write_pre_trend_macros(
            JointTest(
                statistic=7332.0,
                p_value=0.109,
                restrictions=5,
                clusters=9,
                draws=512,
                exhaustive=True,
            ),
            directory / "pre_trend.tex",
        ),
        write_pre_trend_macros(
            JointTest(
                statistic=112.0,
                p_value=0.711,
                restrictions=5,
                clusters=9,
                draws=512,
                exhaustive=True,
            ),
            directory / "exposure_pre_trend.tex",
            prefix="Exposure",
        ),
        write_pre_trend_macros(
            JointTest(
                statistic=23.0,
                p_value=0.646,
                restrictions=5,
                clusters=9,
                draws=512,
                exhaustive=True,
            ),
            directory / "structural_pre_trend.tex",
            prefix="Structural",
        ),
        write_seasonality_macros(
            SeasonalConfound(
                modal_month=1,
                modal_share=0.94,
                surviving_variance_share=0.36,
                worst_category="03",
                worst_category_swing=-15.96,
            ),
            directory / "seasonality.tex",
        ),
    ]

    defined: set[str] = set()
    for path in written:
        defined |= set(_DEFINE.findall(path.read_text(encoding="utf-8")))
    return defined


def written_macro_files(directory: Path) -> list[Path]:
    """The files the writers produced, for checks on their contents."""
    definable_macros(directory)
    return sorted(directory.glob("*.tex"))
