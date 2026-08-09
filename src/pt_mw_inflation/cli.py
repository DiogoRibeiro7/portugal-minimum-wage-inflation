"""Command-line interface for data acquisition and analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml

from pt_mw_inflation.analysis.falsification import run_pre_trend_diagnostic
from pt_mw_inflation.analysis.local_projections import estimate_panel_local_projections
from pt_mw_inflation.data.dgert import parse_minimum_wage_history
from pt_mw_inflation.data.eurostat import fetch_portugal_hicp, save_frame
from pt_mw_inflation.data.registry import download_registry
from pt_mw_inflation.processing.minimum_wage import (
    build_statutory_panel,
    find_unexplained_jumps,
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

    changes = parse_minimum_wage_history(raw.read_bytes().decode("utf-8"))
    panel = build_statutory_panel(changes)

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False)

    typer.echo(f"Wrote {len(panel):,} statutory regimes to {output}")
    for scope, count in panel["scope"].value_counts().sort_index().items():
        typer.echo(f"  {scope}: {count} acts")

    unexplained = find_unexplained_jumps(changes)
    for scope, effective in unexplained:
        typer.echo(
            f"  incomplete upstream history: {scope} {effective} states an increase "
            "that does not reconcile with the previous listed act"
        )


@build_app.command("macro")
def build_macro() -> None:
    """Explain the expected macro build inputs until all source adapters are populated."""
    typer.echo(
        "Macro build contract: year, minimum_wage, inflation, productivity_growth. "
        "Populate adapters for DGERT/INE/AMECO, then write data/processed/macro_annual.parquet."
    )


@build_app.command("policy-residual")
def build_policy_residual_command() -> None:
    """Placeholder command wired for the next implementation milestone."""
    typer.echo("Use processing.minimum_wage.build_policy_residual on macro_annual.parquet.")


@analyse_app.command("macro")
def analyse_macro() -> None:
    """Placeholder for the reproducible long-run tables and figures."""
    typer.echo("Macro analysis module is available; connect it after macro data build.")


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
