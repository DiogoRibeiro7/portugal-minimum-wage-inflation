"""Command-line interface for data acquisition and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml

from pt_mw_inflation.analysis.falsification import run_pre_trend_diagnostic
from pt_mw_inflation.analysis.local_projections import estimate_panel_local_projections
from pt_mw_inflation.analysis.outputs import generate_macro_outputs
from pt_mw_inflation.data.ameco import fetch_series as fetch_ameco_series
from pt_mw_inflation.data.ameco import to_frame as ameco_to_frame
from pt_mw_inflation.data.dgert import parse_minimum_wage_history
from pt_mw_inflation.data.eurostat import fetch_minimum_wage as fetch_eurostat_minimum_wage
from pt_mw_inflation.data.eurostat import fetch_portugal_hicp, save_frame
from pt_mw_inflation.data.registry import download_registry
from pt_mw_inflation.data.worldbank import fetch_indicator
from pt_mw_inflation.processing.macro import (
    build_macro_annual,
    check_accounting_identities,
    summarise_by_regime,
)
from pt_mw_inflation.processing.minimum_wage import (
    annual_minimum_wage,
    build_statutory_panel,
    find_unexplained_jumps,
    reconcile_annual_with_eurostat,
)
from pt_mw_inflation.processing.regional import (
    build_regional_panel,
    merge_supplements,
    supplementary_statutory_changes,
)

app = typer.Typer(help="Portugal minimum-wage inflation research pipeline.")
data_app = typer.Typer(help="Download raw public data.")
build_app = typer.Typer(help="Build analysis-ready datasets.")
analyse_app = typer.Typer(help="Run empirical analysis.")
app.add_typer(data_app, name="data")
app.add_typer(build_app, name="build")
app.add_typer(analyse_app, name="analyse")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@data_app.command("download-sources")
def download_sources() -> None:
    """Download every source currently listed in config/sources.yaml."""
    root = _repo_root()
    records = download_registry(root / "config/sources.yaml", root)
    typer.echo(
        f"Downloaded {len(records)} sources; manifest written to data/raw/source_manifest.json"
    )


@data_app.command("eurostat-hicp")
def eurostat_hicp(
    geo: str = typer.Option("PT", help="Eurostat geography code."),
    output: Path = typer.Option(
        Path("data/raw/eurostat/portugal_hicp.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Download detailed monthly HICP observations from Eurostat."""
    root = _repo_root()
    frame = fetch_portugal_hicp(geo=geo)
    save_frame(frame, root / output)
    typer.echo(f"Saved {len(frame):,} observations to {output}")


@build_app.command("minimum-wage")
def build_minimum_wage(
    source: Path = typer.Option(
        Path("data/raw/dgert/minimum_wage_history.html"),
        help="Raw DGERT history previously retrieved by 'data download-sources'.",
    ),
    output: Path = typer.Option(
        Path("data/processed/minimum_wage_policy.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Build the statutory minimum-wage panel from the retrieved DGERT history."""
    root = _repo_root()
    raw = root / source
    if not raw.exists():
        raise typer.BadParameter(f"{source} not found; run 'ptmw data download-sources' first")

    registry = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    changes = parse_minimum_wage_history(raw.read_bytes().decode("utf-8"))
    changes = merge_supplements(
        changes, supplementary_statutory_changes(registry.get("national_supplements", []))
    )
    panel = build_statutory_panel(changes)

    regional = build_regional_panel(registry["regional"], panel[panel["scope"] == "general"])
    panel = pd.concat([panel, regional], ignore_index=True)

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)

    typer.echo(f"Wrote {len(panel):,} statutory regimes to {output}")
    for scope, count in panel["scope"].value_counts().sort_index().items():
        typer.echo(f"  {scope}: {count} acts")

    if regional.empty:
        typer.echo("  no regional acts configured")
    else:
        for geography, count in regional["geography"].value_counts().sort_index().items():
            typer.echo(f"  {geography}: {count} regional acts")

    unexplained = find_unexplained_jumps(changes)
    for scope, effective in unexplained:
        typer.echo(
            f"  incomplete upstream history: {scope} {effective} states an increase "
            "that does not reconcile with the previous listed act"
        )
    if not unexplained:
        typer.echo("  every stated increase reconciles with the preceding act")


def _load_settings(root: Path) -> dict[str, Any]:
    """Read the analysis configuration."""
    loaded = yaml.safe_load((root / "config/analysis.yaml").read_text(encoding="utf-8"))
    return dict(loaded or {})


@build_app.command("macro")
def build_macro(
    source: Path = typer.Option(
        Path("data/raw/dgert/minimum_wage_history.html"),
        help="Raw DGERT history previously retrieved by 'data download-sources'.",
    ),
    output: Path = typer.Option(
        Path("data/processed/macro_annual.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Build the long-run annual macro dataset from wages, prices and productivity.

    Prices come from the World Bank consumer price index and productivity from
    AMECO, because both reach back to the introduction of the minimum wage in
    1974, where Eurostat's national-accounts series would begin only in 1995.
    """
    root = _repo_root()
    settings = _load_settings(root)
    macro_settings = settings.get("macro", {})

    raw = root / source
    if not raw.exists():
        raise typer.BadParameter(f"{source} not found; run 'ptmw data download-sources' first")

    registry = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    changes = parse_minimum_wage_history(raw.read_bytes().decode("utf-8"))
    # Acts the summary page omits are read from the gazette, so the macro layer
    # rests on the same primary sources as the statutory panel instead of
    # falling back on a secondary compiler.
    changes = merge_supplements(
        changes, supplementary_statutory_changes(registry.get("national_supplements", []))
    )
    panel = build_statutory_panel(changes)

    wages = reconcile_annual_with_eurostat(
        annual_minimum_wage(panel, scope="general", geography="PT"),
        fetch_eurostat_minimum_wage(),
    )
    # Reconciliation is now a check rather than a source: with every act
    # registered it should find nothing to correct.
    corrected = wages.loc[wages["minimum_wage_source"] != "DGERT statutory history", "year"]
    for year in corrected:
        typer.echo(f"  {int(year)}: no act registered; level taken from Eurostat")
    if corrected.empty:
        typer.echo("  every year is sourced to an act; Eurostat agrees throughout")

    prices = fetch_indicator()
    productivity = ameco_to_frame(
        fetch_ameco_series(), last_actual_year=int(macro_settings["last_actual_year"])
    )

    macro = build_macro_annual(
        wages,
        prices,
        productivity,
        start_year=int(macro_settings.get("start_year", 1974)),
        end_year=int(macro_settings["last_actual_year"]),
        inflation_lag=int(macro_settings.get("benchmark_inflation_lag", 1)),
    )
    check_accounting_identities(macro)

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(destination, index=False)

    typer.echo(
        f"Wrote {len(macro)} years ({int(macro['year'].min())}-{int(macro['year'].max())}) "
        f"to {output}"
    )
    residual = macro["policy_residual"].dropna()
    typer.echo(f"  mean policy residual: {residual.mean() * 100:+.2f} pp per year")


@build_app.command("policy-residual")
def build_policy_residual_command() -> None:
    """Placeholder command wired for the next implementation milestone."""
    typer.echo("Use processing.minimum_wage.build_policy_residual on macro_annual.parquet.")


@analyse_app.command("macro")
def analyse_macro(
    source: Path = typer.Option(
        Path("data/processed/macro_annual.parquet"), help="Macro dataset to analyse."
    ),
) -> None:
    """Generate every long-run figure and table from the macro dataset."""
    root = _repo_root()
    dataset = root / source
    if not dataset.exists():
        raise typer.BadParameter(f"{source} not found; run 'ptmw build macro' first")

    macro = pd.read_parquet(dataset)
    check_accounting_identities(macro)

    settings = _load_settings(root)
    regimes = summarise_by_regime(macro, list(settings["macro"]["regimes"]))

    written = generate_macro_outputs(
        macro,
        regimes,
        figures_dir=root / "report/figures",
        tables_dir=root / "report/tables",
    )
    for path in written:
        typer.echo(f"  wrote {path.relative_to(root)}")
    typer.echo(f"Generated {len(written)} outputs from {len(macro)} years.")


@analyse_app.command("pass-through")
def analyse_pass_through(
    panel: Path = typer.Option(
        Path("data/processed/exposure_price_panel.parquet"),
        help="Region-category-month panel with log prices and the exposure shock.",
    ),
    output: Path = typer.Option(
        Path("report/tables/pass_through.csv"), help="Where to write the estimates."
    ),
) -> None:
    """Estimate the dynamic pass-through function with few-cluster inference.

    Runs the horizons configured in config/analysis.yaml, then the pre-trend
    diagnostic and the leave-one-region-out check. Nothing is interpreted here:
    the command reports the estimates and whether the falsification checks pass.
    """
    root = _repo_root()
    source = root / panel
    if not source.exists():
        raise typer.BadParameter(
            f"{panel} not found. The exposure panel depends on the regional coverage "
            "tables, whose source is recorded as unavailable in config/sources.yaml."
        )

    settings = yaml.safe_load((root / "config/analysis.yaml").read_text(encoding="utf-8"))
    horizons = list(settings["pass_through"]["horizons_months"])

    frame = pd.read_parquet(source)
    estimates = estimate_panel_local_projections(
        frame, outcome="log_price", shock="exposure_shock", horizons=horizons
    )

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(destination, index=False)
    typer.echo(f"Wrote {len(estimates)} horizon estimates to {output}")

    _, verdict = run_pre_trend_diagnostic(frame, outcome="log_price", shock="exposure_shock")
    outcome_label = "passed" if verdict.passed else "FAILED"
    typer.echo(f"Pre-trend diagnostic: {outcome_label} - {verdict.detail}")
    if not verdict.passed:
        typer.echo("  A causal reading is not supported while leads predict pre-treatment prices.")


if __name__ == "__main__":
    app()
