"""Command-line interface for data acquisition and analysis."""

from __future__ import annotations

from pathlib import Path

import typer

from pt_mw_inflation.data.eurostat import fetch_portugal_hicp, save_frame
from pt_mw_inflation.data.registry import download_registry

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
    typer.echo(f"Downloaded {len(records)} sources; manifest written to data/raw/source_manifest.json")


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
def analyse_pass_through() -> None:
    """Placeholder for exposure-based panel/local-projection estimation."""
    typer.echo("Pass-through analysis requires the regional/category exposure panel.")


if __name__ == "__main__":
    app()
