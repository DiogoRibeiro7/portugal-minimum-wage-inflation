"""Deterministic generation of the manuscript's figures and tables.

Every figure and table the paper imports is produced here from the analysis
datasets. Nothing is drawn by hand and no number is transcribed, so a change in
the data propagates to the manuscript on the next build rather than leaving a
stale value in the text.

Figure conventions, chosen for print rather than for a screen:

* Both index figures share one vertical axis. Two measures are never given two
  scales; where they differ in level they are indexed to a common base year, so
  the visual comparison is the real one.
* The two-series palette is validated for colour-vision deficiency, and every
  series is also direct-labelled, so identity never rests on colour alone.
* The residual carries a sign, so it uses a diverging pair with a neutral zero
  line rather than a single hue.
* Grid and axes are recessive; marks are thin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Deterministic, headless rendering for reproducible builds.

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pt_mw_inflation.analysis.inference import JointTest, RobustnessRun  # noqa: E402
from pt_mw_inflation.processing.exposure import VariationStrength  # noqa: E402
from pt_mw_inflation.processing.pass_through import SeasonalConfound  # noqa: E402

#: Categorical slots 1 and 2, validated for CVD separation against the light
#: chart surface (worst adjacent pair ΔE 24.7 protan, 33.6 normal vision).
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"

#: Diverging poles for a signed quantity, with a neutral midpoint.
POSITIVE_RED = "#d03b3b"
NEGATIVE_BLUE = "#2a78d6"
NEUTRAL_GREY = "#8a8a85"

SURFACE = "#ffffff"
INK = "#1a1a19"
MUTED_INK = "#5c5c58"

FIGURE_SIZE = (7.0, 4.2)
DPI = 300


def _style_axes(axes: Axes) -> None:
    """Apply the recessive grid and axis treatment used by every figure."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, axis="y", color="#e4e4e0", linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c9c9c4")
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=MUTED_INK, labelsize=9, length=3, width=0.8)


def _save(figure: Figure, path: Path) -> Path:
    """Write a figure and close it, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_real_wage_and_productivity(macro: pd.DataFrame, path: Path) -> Path:
    """Plot the real minimum wage against productivity, both on one axis.

    The comparison the paper turns on is whether the wage floor kept pace with
    output per worker. Indexing both to the first year puts them on one scale,
    which is the only honest way to show them together.

    Args:
        macro: Output of :func:`pt_mw_inflation.processing.macro.build_macro_annual`.
        path: Destination file.

    Returns:
        The written path.
    """
    base_year = int(macro["year"].iloc[0])
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(axes)

    axes.plot(macro["year"], macro["real_minimum_wage_index"], color=SERIES_BLUE, linewidth=2.0)
    axes.plot(macro["year"], macro["productivity_index"], color=SERIES_ORANGE, linewidth=2.0)
    axes.axhline(100.0, color=NEUTRAL_GREY, linewidth=0.8, linestyle=(0, (4, 3)))

    last = macro.iloc[-1]
    axes.annotate(
        "Real minimum wage",
        xy=(last["year"], last["real_minimum_wage_index"]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=9,
    )
    axes.annotate(
        "Labour productivity",
        xy=(last["year"], last["productivity_index"]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=9,
    )

    axes.set_ylabel(f"Index, {base_year} = 100", color=MUTED_INK, fontsize=9)
    axes.set_xlabel("")
    axes.set_title(
        "The real minimum wage against output per worker",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    return _save(figure, path)


def plot_policy_residual(macro: pd.DataFrame, path: Path) -> Path:
    """Plot the annual policy residual as signed bars around zero.

    Args:
        macro: Macro dataset.
        path: Destination file.

    Returns:
        The written path.
    """
    sample = macro.dropna(subset=["policy_residual"])
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(axes)

    values = sample["policy_residual"] * 100.0
    colours = [POSITIVE_RED if value > 0 else NEGATIVE_BLUE for value in values]
    axes.bar(sample["year"], values, color=colours, width=0.75, linewidth=0)
    axes.axhline(0.0, color=NEUTRAL_GREY, linewidth=1.0)

    axes.set_ylabel("Percentage points per year", color=MUTED_INK, fontsize=9)
    axes.set_title(
        "Minimum-wage growth relative to productivity plus prior inflation",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    axes.annotate(
        "Above zero: the wage floor rose faster than the benchmark",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, -12),
        textcoords="offset points",
        color=MUTED_INK,
        fontsize=8,
    )
    return _save(figure, path)


def plot_wage_to_productivity(macro: pd.DataFrame, path: Path) -> Path:
    """Plot the minimum wage measured against productivity.

    A single series needs no legend: the title names it.

    Args:
        macro: Macro dataset.
        path: Destination file.

    Returns:
        The written path.
    """
    base_year = int(macro["year"].iloc[0])
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(axes)

    axes.plot(
        macro["year"],
        macro["minimum_wage_to_productivity_index"],
        color=SERIES_BLUE,
        linewidth=2.0,
    )
    axes.axhline(100.0, color=NEUTRAL_GREY, linewidth=0.8, linestyle=(0, (4, 3)))
    axes.set_ylabel(f"Index, {base_year} = 100", color=MUTED_INK, fontsize=9)
    axes.set_title(
        "Real minimum wage per unit of labour productivity",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )
    return _save(figure, path)


def _latex_escape(text: str) -> str:
    """Escape the LaTeX specials that appear in regime names."""
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_regime_table(regimes: pd.DataFrame, path: Path) -> Path:
    """Write the regime summary as a LaTeX table body.

    The manuscript inputs this file, so the numbers in the paper are the
    numbers in the dataset by construction.

    Args:
        regimes: Output of
            :func:`pt_mw_inflation.processing.macro.summarise_by_regime`.
        path: Destination `.tex` file.

    Returns:
        The written path.
    """
    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_regime_table.",
        "% Do not edit: regenerate with 'ptmw analyse macro'.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Regime & Years & $\pi$ & $g_A$ & $g_{MW}$ & Residual \\",
        r"\midrule",
    ]
    for _, row in regimes.iterrows():
        lines.append(
            f"{_latex_escape(str(row['regime']))} & "
            f"{int(row['start'])}--{int(row['end'])} & "
            f"{row['mean_inflation'] * 100:.1f} & "
            f"{row['mean_productivity_growth'] * 100:.1f} & "
            f"{row['mean_minimum_wage_growth'] * 100:.1f} & "
            f"{row['mean_policy_residual'] * 100:+.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_headline_macros(macro: pd.DataFrame, path: Path) -> Path:
    """Write headline figures as LaTeX macros for the manuscript body.

    Defining them as macros is what keeps prose honest: the text cannot quote a
    number that the dataset does not currently produce, because the number is
    not written in the text at all.

    Args:
        macro: Macro dataset.
        path: Destination `.tex` file.

    Returns:
        The written path.
    """
    first = macro.iloc[0]
    last = macro.iloc[-1]
    residual = macro["policy_residual"].dropna()
    gap = float(macro["cumulative_policy_gap"].iloc[-1])
    trough = macro["minimum_wage_to_productivity_index"].astype(float)
    trough_year = int(macro["year"].astype(int).to_numpy()[int(trough.to_numpy().argmin())])

    values = {
        "MacroFirstYear": f"{int(first['year'])}",
        "MacroLastYear": f"{int(last['year'])}",
        "RealWageIndexEnd": f"{last['real_minimum_wage_index']:.1f}",
        "ProductivityIndexEnd": f"{last['productivity_index']:.1f}",
        "WageToProductivityEnd": f"{last['minimum_wage_to_productivity_index']:.1f}",
        "WageToProductivityMin": f"{macro['minimum_wage_to_productivity_index'].min():.1f}",
        "WageToProductivityMinYear": f"{trough_year}",
        "MeanResidualPct": f"{residual.mean() * 100:+.2f}",
        # The compounded gap, not the sum of annual residuals. Summing them
        # over fifty years at Portuguese inflation rates gives a figure beyond
        # -100 per cent, which cannot be read as a cumulative shortfall.
        # Emitted as an unsigned magnitude with a separate direction word. A
        # signed macro placed next to a hard-coded "below" reads as a double
        # negative, and would invert the claim outright if the sign ever flipped.
        "CumulativeGapPct": f"{gap * 100:+.1f}",
        "CumulativeGapMagnitudePct": f"{abs(gap) * 100:.1f}",
        "CumulativeGapDirection": "below" if gap < 0 else "above",
        "SummedResidualPct": f"{residual.sum() * 100:+.1f}",
        "ObservedYears": f"{len(macro)}",
    }

    # The wage floor was introduced in May, so its first year pairs a rate in
    # force for eight months with a full-calendar-year price and productivity
    # observation. Rebasing on the first complete year is the obvious check, and
    # reporting it is what turns "the base year is awkward" into a quantity a
    # reader can weigh.
    if len(macro) > 1:
        ratio = trough / float(trough.iloc[0])
        second = float(trough.iloc[1]) / float(trough.iloc[0])
        values["WageToProductivityEndAlt"] = f"{100 * float(ratio.iloc[-1]) / second:.1f}"
        values["WageToProductivityMinAlt"] = f"{100 * float(ratio.min()) / second:.1f}"
        values["MacroSecondYear"] = f"{int(macro['year'].astype(int).to_numpy()[1])}"

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_headline_macros.",
        "% Do not edit: regenerate with 'ptmw analyse macro'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_macro_outputs(
    macro: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    figures_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Produce every macro figure and table in one call.

    Args:
        macro: Macro dataset.
        regimes: Regime summary.
        figures_dir: Destination for figures.
        tables_dir: Destination for tables.

    Returns:
        Paths written, in a stable order.
    """
    return [
        plot_real_wage_and_productivity(macro, figures_dir / "real_wage_productivity.pdf"),
        plot_policy_residual(macro, figures_dir / "policy_residual.pdf"),
        plot_wage_to_productivity(macro, figures_dir / "wage_to_productivity.pdf"),
        write_regime_table(regimes, tables_dir / "regime_summary.tex"),
        write_headline_macros(macro, tables_dir / "headline_macros.tex"),
    ]


def write_regional_design_table(
    estimates: pd.DataFrame, path: Path, *, command: str = "ptmw analyse pass-through"
) -> Path:
    """Write the regional pass-through estimates as a LaTeX table body.

    The manuscript reports these numbers to contrast conventional inference with
    the bootstrap, which is an argument about rigour. Transcribing them by hand
    would undercut that argument, and would let the paper keep reporting a
    figure the pipeline no longer produces.

    Args:
        estimates: Horizon estimates from the local-projection estimator.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If there are no estimates to tabulate.
    """
    if estimates.empty:
        raise ValueError("no estimates to tabulate")

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_regional_design_table.",
        "% Do not edit: regenerate with 'ptmw analyse pass-through'.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Horizon (months) & Coefficient & Clustered $p$ & Bootstrap $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for _, row in estimates.iterrows():
        lines.append(
            f"{int(row['horizon'])} & {row['coefficient']:.3f} & "
            f"{row['p_value_clustered']:.3f} & {row['p_value_bootstrap']:.3f} & "
            f"{row['p_value_bootstrap_holm']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_identification_macros(
    estimates: pd.DataFrame,
    identifying_events: int,
    identifying_regions: int,
    path: Path,
) -> Path:
    """Emit the identification quantities the prose cites, as LaTeX macros.

    Args:
        estimates: Horizon estimates.
        identifying_events: Region-months whose statutory change diverges.
        identifying_regions: How many regions contribute any divergence.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If there are no estimates.
    """
    if estimates.empty:
        raise ValueError("no estimates to summarise")

    ordered = estimates.reset_index(drop=True)
    peak_row = int(ordered["coefficient"].abs().to_numpy().argmax())
    peak_coefficient = float(ordered["coefficient"].to_numpy()[peak_row])
    peak_horizon = int(ordered["horizon"].to_numpy()[peak_row])

    values = {
        "IdentifyingEvents": f"{identifying_events}",
        "IdentifyingRegions": f"{identifying_regions}",
        "PeakCoefficient": f"{peak_coefficient:.2f}",
        "PeakHorizon": f"{peak_horizon}",
        "MinBootstrapP": f"{estimates['p_value_bootstrap'].min():.3f}",
        "MinBootstrapHolmP": f"{estimates['p_value_bootstrap_holm'].min():.3f}",
        # Seven horizons are a family, not seven separate papers. Reporting how
        # many survive before and after the correction is what stops the
        # smallest of several p-values being read as though it stood alone.
        "BootstrapSurvivors": f"{int((estimates['p_value_bootstrap'] < 0.05).sum())}",
        "BootstrapSurvivorsHolm": f"{int((estimates['p_value_bootstrap_holm'] < 0.05).sum())}",
        "ClusteredHighlySignificant": f"{int((estimates['p_value_clustered'] < 0.001).sum())}",
        "HorizonsEstimated": f"{len(estimates)}",
        "BootstrapClusters": f"{int(ordered['clusters'].max())}",
        # The manuscript says the bootstrap p-value is exact. Whether it is
        # depends on whether the sign space was enumerated, which depends on the
        # cluster count in the sample that survived the merges. Asserting it in
        # prose would let a change in the panel silently falsify the sentence.
        # "Exact" was wrong and this module was the only place asserting it.
        # inference.py says so in its own docstring: enumerating the sign space
        # removes the Monte Carlo component of the error and nothing else, since
        # the sign-flip distribution approximates the null rather than being it.
        # A test was pinning the wrong word, so the suite defended the error.
        "BootstrapBasis": (
            r"enumerates the complete Rademacher sign space, so its $p$-value "
            r"carries no simulation error"
            if bool(ordered["bootstrap_exhaustive"].all())
            else r"samples the sign space, so its $p$-value carries simulation error"
        ),
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_identification_macros.",
        "% Do not edit: regenerate with 'ptmw analyse pass-through'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_exposure_macros(
    exposure: pd.DataFrame,
    strength: VariationStrength,
    registry: dict[str, object],
    path: Path,
) -> Path:
    """Emit the shift-share exposure quantities the manuscript cites.

    The exposure section previously described a measure that could not be built.
    It now describes one that can, and every figure in that description --- the
    spread, the coefficient of variation, how much employment carries a measured
    bite, and the date the bite refers to --- comes from here rather than from
    the prose, so a change in the source moves the sentence with it.

    Args:
        exposure: Output of :func:`~pt_mw_inflation.processing.exposure.shift_share_exposure`.
        strength: Output of
            :func:`~pt_mw_inflation.processing.exposure.measure_variation_strength`.
        registry: The parsed bite configuration, read for its reference period.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If the exposure frame is empty.
    """
    if exposure.empty:
        raise ValueError("no exposure to summarise")

    values_series = exposure["regional_bite_exposure"]
    coverage = exposure["covered_employment_share"]
    source = registry.get("source", {})
    reference = str(source.get("reference_period", "")) if isinstance(source, dict) else ""
    # The configuration dates the bite as YYYY-MM, which is right for a key and
    # wrong in a sentence. Rendered here rather than restated in the prose, so
    # the paper cannot drift from the file the pipeline actually read.
    try:
        reference = pd.Period(reference, freq="M").strftime("%B %Y")
    except ValueError as error:  # pragma: no cover - configuration error
        raise ValueError(f"unparseable bite reference period {reference!r}") from error

    # The endpoints and the spread must be readable against each other. At one
    # decimal the endpoints imply a range that disagrees with a two-decimal
    # spread, so a reader checking the arithmetic finds a discrepancy that is
    # not in the data.
    values = {
        "ExposureRegions": f"{strength.regions}",
        "ExposureDistinctValues": f"{int(values_series.round(6).nunique())}",
        "ExposureMinPct": f"{100 * float(values_series.min()):.2f}",
        "ExposureMaxPct": f"{100 * float(values_series.max()):.2f}",
        "ExposureSpreadPP": f"{100 * strength.spread:.2f}",
        "ExposureCV": f"{strength.coefficient_of_variation:.3f}",
        "ExposureCoverageMinPct": f"{100 * float(coverage.min()):.0f}",
        "ExposureCoverageMaxPct": f"{100 * float(coverage.max()):.0f}",
        "ExposureBitePeriod": reference,
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_exposure_macros.",
        "% Do not edit: regenerate with 'ptmw build regional-exposure'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_regional_premium_macros(
    wage_panel: pd.DataFrame,
    path: Path,
    *,
    geography: str = "PT30",
    national: str = "PT",
    tolerance: float = 5e-4,
) -> Path:
    """Emit Madeira's premium over the mainland, and where it was a fixed rule.

    The manuscript claims that Madeira's legislated figures follow a two per
    cent rule until they stop. That claim is arithmetic on the statutory panel,
    so it is computed here rather than transcribed: the run of years over which
    the premium is constant is found, not asserted, and if a newly registered
    act broke the run the prose would move with it instead of becoming false.

    Args:
        wage_panel: Statutory panel, general regime, national and regional rows.
        path: Destination `.tex` file.
        geography: The region whose premium is described.
        national: Geography code carrying the mainland series.
        tolerance: Largest premium difference still counted as the same rule.

    Returns:
        The written path.

    Raises:
        ValueError: If either geography is absent, or no constant run exists.
    """
    frame = wage_panel.copy()
    frame["year"] = pd.to_datetime(frame["effective_date"]).dt.year

    def _january(code: str) -> pd.Series:
        rows = frame.loc[frame["geography"] == code].sort_values("effective_date")
        if rows.empty:
            raise ValueError(f"no statutory rows for {code}")
        return rows.groupby("year")["minimum_wage_monthly_eur"].last()

    regional, mainland = _january(geography), _january(national)
    years = sorted(set(regional.index) & set(mainland.index))
    if not years:
        raise ValueError(f"{geography} and {national} share no year")

    premium = {year: float(regional[year]) / float(mainland[year]) - 1.0 for year in years}

    # The leading run of years sharing one premium. Anchored at the first shared
    # year rather than searched for anywhere, because the claim is that the rule
    # held from the start of the register and was later abandoned.
    first = years[0]
    rule = premium[first]
    last = first
    for year in years[1:]:
        if abs(premium[year] - rule) > tolerance:
            break
        last = year
    if last == first:
        raise ValueError(f"{geography} shows no constant premium run from {first}")

    values = {
        "MadeiraRuleFactorPct": f"{100 * rule:.0f}",
        "MadeiraRuleFirstYear": f"{first}",
        "MadeiraRuleLastYear": f"{last}",
        "MadeiraPremiumEndPct": f"{100 * premium[years[-1]]:.1f}",
        "MadeiraPremiumEndYear": f"{years[-1]}",
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_regional_premium_macros.",
        "% Do not edit: regenerate with 'ptmw analyse pass-through'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


#: Month names, so the manuscript names a month rather than printing its number.
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

#: COICOP divisions, for naming the category the confound distorts most.
_CATEGORY_NAMES = {
    "01": "food and non-alcoholic beverages",
    "02": "alcoholic beverages and tobacco",
    "03": "clothing and footwear",
    "04": "housing, water and energy",
    "05": "furnishings and household equipment",
    "06": "health",
    "07": "transport",
    "08": "information and communication",
    "09": "recreation, sport and culture",
    "10": "education services",
    "11": "restaurants and accommodation",
    "12": "financial and insurance services",
    "13": "personal care and miscellaneous services",
}


def write_seasonality_macros(confound: SeasonalConfound, path: Path) -> Path:
    """Emit the seasonal-confound diagnosis the manuscript reports.

    An earlier version of the paper attributed the clothing coefficient to "a
    break or a rebasing in that series". It is not that. It is that Portugal
    moves its minimum wage on 1 January, so the shock is very nearly a January
    indicator, and clothing collapses every January on winter sales. Since that
    claim is arithmetic on the panel, it is computed rather than described.

    Args:
        confound: Output of
            :func:`~pt_mw_inflation.processing.pass_through.diagnose_seasonal_confound`.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If the modal month is not a calendar month.
    """
    if not 1 <= confound.modal_month <= 12:
        raise ValueError(f"{confound.modal_month} is not a calendar month")

    values = {
        "ShockModalMonth": _MONTH_NAMES[confound.modal_month - 1],
        "ShockModalSharePct": f"{100 * confound.modal_share:.0f}",
        "ShockSurvivingSharePct": f"{100 * confound.surviving_variance_share:.0f}",
        "SeasonalWorstCategory": _CATEGORY_NAMES.get(
            confound.worst_category, confound.worst_category
        ),
        "SeasonalWorstSwingPct": f"{abs(confound.worst_category_swing):.1f}",
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_seasonality_macros.",
        "% Do not edit: regenerate with 'ptmw analyse pass-through'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_exposure_design_macros(estimates: pd.DataFrame, path: Path) -> Path:
    """Emit what the shift-share design supports, which is nothing.

    Reported as generated quantities because a null result stated in prose is
    the easiest kind of claim to leave standing after the data moves under it.

    Args:
        estimates: Horizon estimates from the exposure design.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If there are no estimates.
    """
    if estimates.empty:
        raise ValueError("no estimates to summarise")

    widest = int(estimates["coefficient"].abs().to_numpy().argmax())
    values = {
        "ExposureHorizons": f"{len(estimates)}",
        "ExposureClusteredSignificant": f"{int((estimates['p_value_clustered'] < 0.05).sum())}",
        "ExposureBootstrapSignificant": f"{int((estimates['p_value_bootstrap'] < 0.05).sum())}",
        "ExposureMinClusteredP": f"{estimates['p_value_clustered'].min():.2f}",
        "ExposureWidestCoefficient": f"{estimates['coefficient'].to_numpy()[widest]:.2f}",
        "ExposureWidestHorizon": f"{int(estimates['horizon'].to_numpy()[widest])}",
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_exposure_design_macros.",
        "% Do not edit: regenerate with 'ptmw analyse exposure-design'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_robustness_table(runs: list[RobustnessRun], path: Path) -> Path:
    """Tabulate what each specification of a design yielded.

    Args:
        runs: Summaries from
            :func:`~pt_mw_inflation.analysis.inference.summarise_run`.
        path: Destination `.tex` file.

    Returns:
        The written path.

    Raises:
        ValueError: If no specification was estimated.
    """
    if not runs:
        raise ValueError("no specifications to tabulate")

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_robustness_table.",
        "% Do not edit: regenerate with 'ptmw analyse exposure-robustness'.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Specification & Coefficient range & Rejects & After Holm \\",
        r"\midrule",
    ]
    for run in runs:
        lines.append(
            f"{_latex_escape(run.label)} & "
            f"[{run.min_coefficient:.2f}, {run.max_coefficient:.2f}] & "
            f"{run.rejections} & {run.rejections_holm} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_pre_trend_macros(
    result: JointTest,
    path: Path,
    *,
    prefix: str = "",
    command: str = "ptmw analyse pass-through",
) -> Path:
    """Emit the joint pre-trend diagnostic the manuscript reports.

    The falsification battery requires that leads of the statutory change do not
    predict pre-treatment inflation. Emitting the joint test rather than a
    sentence keeps the paper from claiming the battery passed after the panel
    changed underneath it.

    Args:
        result: Output of
            :func:`~pt_mw_inflation.analysis.local_projections.assess_pre_trends`.
        path: Destination `.tex` file.
        prefix: Prepended to each macro name. The paper reports two designs and
            the battery applies to both, so each needs its own diagnostic rather
            than one standing in for the other.
        command: Named in the do-not-edit banner. Two commands now share this
            writer, and a banner naming the wrong one is worse than none because
            it will appear to have been followed.

    Returns:
        The written path.

    Raises:
        ValueError: If the test restricted nothing.
    """
    if result.restrictions < 1:
        raise ValueError("a pre-trend test must restrict at least one lead")

    values = {
        f"{prefix}PreTrendLeads": f"{result.restrictions}",
        f"{prefix}PreTrendWald": f"{result.statistic:.0f}",
        f"{prefix}PreTrendP": f"{result.p_value:.3f}",
        f"{prefix}PreTrendRejects": ("rejects" if result.p_value < 0.05 else "does not reject"),
    }

    lines = [
        "% Generated by pt_mw_inflation.analysis.outputs.write_pre_trend_macros.",
        f"% Do not edit: regenerate with '{command}'.",
    ]
    lines += [rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
