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
        "CumulativeResidualPct": f"{residual.sum() * 100:+.1f}",
        "ObservedYears": f"{len(macro)}",
    }

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
