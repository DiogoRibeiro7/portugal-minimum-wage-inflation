"""Command-line interface for data acquisition and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import yaml

from pt_mw_inflation.analysis.inference import detectable_effects, summarise_run
from pt_mw_inflation.analysis.local_projections import (
    assess_pre_trends,
    estimate_panel_local_projections,
)
from pt_mw_inflation.analysis.outputs import (
    generate_macro_outputs,
    write_exposure_design_macros,
    write_exposure_macros,
    write_identification_macros,
    write_pre_trend_macros,
    write_regional_design_table,
    write_regional_premium_macros,
    write_robustness_table,
    write_seasonality_macros,
)
from pt_mw_inflation.data.ameco import fetch_series as fetch_ameco_series
from pt_mw_inflation.data.ameco import to_frame as ameco_to_frame
from pt_mw_inflation.data.dgert import parse_minimum_wage_history
from pt_mw_inflation.data.eurostat import fetch_minimum_wage as fetch_eurostat_minimum_wage
from pt_mw_inflation.data.eurostat import fetch_portugal_hicp, save_frame
from pt_mw_inflation.data.eurostat_regional import (
    RegionalEmploymentError,
    fetch_national_employment,
    fetch_regional_employment,
    industry_shares,
    require_matched_inputs,
)
from pt_mw_inflation.data.freeze import FreezeError, verify_manifest, write_manifest
from pt_mw_inflation.data.ine import fetch_regional_cpi
from pt_mw_inflation.data.registry import download_registry
from pt_mw_inflation.data.worldbank import fetch_indicator
from pt_mw_inflation.processing.exposure import (
    ExposureError,
    PredeterminationError,
    activity_bite_from_registry,
    bound_unmeasured_exposure,
    check_predetermined,
    measure_variation_strength,
    select_snapshot,
    shift_share_exposure,
)
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
from pt_mw_inflation.processing.pass_through import (
    PassThroughError,
    add_exposure_interaction,
    build_estimation_panel,
    build_regional_shock,
    count_identifying_events,
    diagnose_seasonal_confound,
    monthly_statutory_wage,
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


@data_app.command("ine-cpi")
def data_ine_cpi(
    start: str = typer.Option("1991-01", help="First month, as YYYY-MM."),
    end: str = typer.Option("", help="Last month, as YYYY-MM. Defaults to last complete month."),
    output: Path = typer.Option(
        Path("data/processed/regional_price_panel.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Download the regional consumer price panel from Statistics Portugal."""
    root = _repo_root()
    frame = fetch_regional_cpi(start=start, end=end or None, raw_dir=root / "data/raw/ine")

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)

    regions = frame.loc[~frame["is_aggregate"], "nuts_code"].nunique()
    typer.echo(f"Wrote {len(frame):,} observations to {output}")
    typer.echo(
        f"  {regions} NUTS II regions, {frame['category_code'].nunique()} consumption "
        f"categories, {frame['month'].min():%Y-%m} to {frame['month'].max():%Y-%m}"
    )


@data_app.command("regional-employment")
def data_regional_employment(
    output: Path = typer.Option(
        Path("data/processed/regional_employment.parquet"), help="Output Parquet path."
    ),
    national_output: Path = typer.Option(
        Path("data/processed/national_employment.parquet"),
        help="National employment by NACE section, used to weight the bite.",
    ),
    year: int = typer.Option(2015, help="Year the national weights are taken from."),
) -> None:
    """Download regional and national employment by industry from Eurostat."""
    root = _repo_root()
    regional = fetch_regional_employment()
    national = fetch_national_employment(year=year)

    for frame, path in ((regional, output), (national, national_output)):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)

    first, last = int(regional["year"].min()), int(regional["year"].max())
    typer.echo(f"Wrote {len(regional):,} regional observations to {output}")
    typer.echo(f"  {regional['region'].nunique()} regions, {first}-{last}")
    typer.echo(f"Wrote {len(national):,} national activity rows for {year} to {national_output}")
    typer.echo(
        f"  counted over {national['population'].iat[0]}, the population the bite is measured on"
    )


@build_app.command("regional-exposure")
def build_regional_exposure(
    regional: Path = typer.Option(
        Path("data/processed/regional_employment.parquet"),
        help="Regional employment from 'ptmw data regional-employment'.",
    ),
    national: Path = typer.Option(
        Path("data/processed/national_employment.parquet"),
        help="National employment by NACE section.",
    ),
    baseline_year: int = typer.Option(2015, help="Year the composition is frozen at."),
    first_shock_year: int = typer.Option(
        0, help="First year of the episode. When set, predetermination is enforced."
    ),
    bite_period: str = typer.Option(
        "", help="Survey round to take the bite from, as YYYY-MM. Defaults to the latest."
    ),
    output: Path = typer.Option(
        Path("data/processed/regional_exposure.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Build the shift-share regional exposure from composition and the national bite."""
    root = _repo_root()
    for path in (regional, national):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; run 'ptmw data regional-employment' first")

    registry = yaml.safe_load((root / "config/minimum_wage_bite.yaml").read_text(encoding="utf-8"))
    try:
        registry = select_snapshot(registry, bite_period or None)
    except ExposureError as error:
        raise typer.BadParameter(str(error)) from error

    if first_shock_year:
        # Refuses a bite or composition dated at or after the first shock, since
        # coverage measured after a rise is partly caused by it. Surfaced as a
        # parameter error rather than a traceback: it is a statement about the
        # window the caller asked for, not a fault in the code.
        try:
            check_predetermined(registry, baseline_year, first_shock_year)
        except PredeterminationError as error:
            raise typer.BadParameter(str(error)) from error

    national_employment = pd.read_parquet(root / national)
    regional_employment = pd.read_parquet(root / regional)

    # Composition is frozen at baseline_year here; the weights were frozen at
    # whatever year was passed to 'data regional-employment'. Nothing tied the
    # two together, so changing one option produced a measure labelled as frozen
    # at a year only half of it was frozen at. The download now stamps its year
    # and this refuses the mismatch.
    try:
        population = require_matched_inputs(
            regional_employment, national_employment, baseline_year=baseline_year
        )
    except RegionalEmploymentError as error:
        raise typer.BadParameter(str(error)) from error

    shares = industry_shares(regional_employment, year=baseline_year)
    bite = activity_bite_from_registry(registry, national_employment)
    exposure = shift_share_exposure(shares, bite)

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_parquet(destination, index=False)

    typer.echo(f"Wrote exposure for {len(exposure)} regions to {output}")
    typer.echo(f"  composition and weights both count {population}, frozen at {baseline_year}")
    typer.echo(f"  bite from the {registry['source']['reference_period']} survey round")
    covered = exposure["covered_employment_share"]
    typer.echo(f"  bite measured on {covered.min():.0%}-{covered.max():.0%} of regional employment")

    strength = measure_variation_strength(exposure)
    macros = write_exposure_macros(
        exposure, strength, registry, root / "report/tables/exposure_macros.tex"
    )
    typer.echo(f"  macros written to {macros.relative_to(root)}")
    typer.echo(f"  {strength.detail}")
    if strength.coefficient_of_variation < 0.05:
        typer.echo(
            "  This is flat. Distinct values establish identification in principle, "
            "not that it is precise enough to be informative."
        )


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
    prices: Path = typer.Option(
        Path("data/processed/regional_price_panel.parquet"),
        help="Regional price panel from 'ptmw data ine-cpi'.",
    ),
    wages: Path = typer.Option(
        Path("data/processed/minimum_wage_policy.parquet"),
        help="Statutory panel from 'ptmw build minimum-wage'.",
    ),
    start: str = typer.Option(
        "2010-01", help="First month; the window where the register is contiguous."
    ),
) -> None:
    """Estimate the regional pass-through design and write its outputs.

    Reports the estimates, the number of region-months that actually identify
    them, and both p-values. Nothing is interpreted here: the command produces
    the table the manuscript imports, so no result reaches the paper by hand.
    """
    root = _repo_root()
    for path in (prices, wages):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; build it first")

    settings = _load_settings(root)
    horizons = list(settings["pass_through"]["horizons_months"])

    registry = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    gaps = {
        block["geography"]: frozenset(block.get("gap_years", []) or [])
        for block in registry["regional"].values()
    }

    price_panel = pd.read_parquet(root / prices)
    wage_panel = pd.read_parquet(root / wages).query("scope == 'general'")

    panel = build_estimation_panel(price_panel, wage_panel, start=start, gap_years=gaps)
    months = pd.DatetimeIndex(sorted(panel["month"].unique()))
    shock = build_regional_shock(
        wage_panel, months, sorted(panel["nuts_code"].unique()), gap_years=gaps
    )
    write_regional_premium_macros(wage_panel, root / "report/tables/premium_macros.tex")

    # The category design carries no calendar-time effects, so a shock that
    # always lands in the same month is confounded with that month. Diagnosed
    # here rather than described in the prose, because it is arithmetic.
    national_wage = monthly_statutory_wage(wage_panel, months, geography="PT")
    confound = diagnose_seasonal_confound(price_panel, np.log(national_wage))
    write_seasonality_macros(confound, root / "report/tables/seasonality_macros.tex")
    typer.echo(
        f"  {confound.modal_share:.0%} of statutory change lands in month "
        f"{confound.modal_month}; {confound.surviving_variance_share:.0%} of it "
        "survives month-of-year effects"
    )
    # Leads of the statutory change must not predict pre-treatment inflation.
    # Tested jointly rather than lead by lead: reading them one at a time
    # multiplies the chance one looks significant and misses a trend spread
    # thinly across several.
    pre_trend = assess_pre_trends(panel, outcome="log_price", shock="delta_log_minimum_wage")
    write_pre_trend_macros(pre_trend, root / "report/tables/pre_trend_macros.tex")
    typer.echo(
        f"  joint pre-trend test on {pre_trend.restrictions} leads: p = {pre_trend.p_value:.3f}"
    )

    variation = count_identifying_events(shock, national="PT11")

    typer.echo(
        f"Identifying variation: {variation.region_months} region-months "
        f"across {len(variation.regions)} region(s)"
    )
    if variation.region_months < 10:
        typer.echo("  This is thin. Judge the estimates by this number, not by the row count.")

    estimates = estimate_panel_local_projections(
        panel,
        outcome="log_price",
        shock="delta_log_minimum_wage",
        horizons=horizons,
        cluster="region",
    )
    if estimates.empty:
        raise typer.BadParameter("no horizon could be estimated on this window")

    tables = root / "report/tables"
    write_regional_design_table(estimates, tables / "regional_design.tex")
    write_identification_macros(
        estimates,
        variation.region_months,
        len(variation.regions),
        tables / "identification_macros.tex",
    )
    typer.echo(f"Wrote {len(estimates)} horizon estimates to report/tables/")

    conventional = int((estimates["p_value_clustered"] < 0.05).sum())
    bootstrap = int((estimates["p_value_bootstrap"] < 0.05).sum())
    typer.echo(
        f"  significant at 5%: {conventional} horizon(s) by clustered inference, "
        f"{bootstrap} by the bootstrap"
    )
    if conventional > bootstrap:
        typer.echo(
            "  Cite the bootstrap. With this few clusters the clustered p-value "
            "rejects a true null far above its nominal size."
        )


@analyse_app.command("exposure-design")
def analyse_exposure_design(
    prices: Path = typer.Option(
        Path("data/processed/regional_price_panel.parquet"), help="Regional price panel."
    ),
    wages: Path = typer.Option(
        Path("data/processed/minimum_wage_policy.parquet"), help="Statutory panel."
    ),
    exposure: Path = typer.Option(
        Path("data/processed/regional_exposure.parquet"), help="Predetermined exposure."
    ),
    start: str = typer.Option("2016-01", help="First month; must follow the bite's survey round."),
) -> None:
    """Estimate the shift-share exposure design and write its table.

    Reported so the design can be judged on an estimate rather than on the
    spread of its regressor. Unlike the category design this one carries
    calendar-time fixed effects, because exposure varies across regions, so
    everything moving national prices in a month is absorbed.
    """
    root = _repo_root()
    for path in (prices, wages, exposure):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; build it first")

    settings = _load_settings(root)
    horizons = list(settings["pass_through"]["horizons_months"])

    registry = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    gaps = {
        block["geography"]: frozenset(block.get("gap_years", []) or [])
        for block in registry["regional"].values()
    }

    panel = build_estimation_panel(
        pd.read_parquet(root / prices),
        pd.read_parquet(root / wages).query("scope == 'general'"),
        start=start,
        gap_years=gaps,
    )
    try:
        panel = add_exposure_interaction(panel, pd.read_parquet(root / exposure))
    except PassThroughError as error:
        raise typer.BadParameter(str(error)) from error

    estimates = estimate_panel_local_projections(
        panel,
        outcome="log_price",
        shock="exposure_shock",
        horizons=horizons,
        cluster="region",
    )
    if estimates.empty:
        raise typer.BadParameter("no horizon could be estimated")

    destination = root / "report/tables/exposure_design.tex"
    write_regional_design_table(estimates, destination, command="ptmw analyse exposure-design")
    write_exposure_design_macros(estimates, root / "report/tables/exposure_design_macros.tex")

    # The falsification battery applies to this design too. Reporting it for one
    # design and not the other would let the untested one borrow the tested
    # one's credibility.
    exposure_pre_trend = assess_pre_trends(panel, outcome="log_price", shock="exposure_shock")
    write_pre_trend_macros(
        exposure_pre_trend,
        root / "report/tables/exposure_pre_trend_macros.tex",
        prefix="Exposure",
        command="ptmw analyse exposure-design",
    )
    typer.echo(
        f"  joint pre-trend test on {exposure_pre_trend.restrictions} leads: "
        f"p = {exposure_pre_trend.p_value:.3f}"
    )

    exposure_frame = pd.read_parquet(root / exposure)
    contrast = float(
        exposure_frame["regional_bite_exposure"].max()
        - exposure_frame["regional_bite_exposure"].min()
    )
    resolution = detectable_effects(estimates, contrast=contrast)
    widest = max(resolution, key=lambda item: item.upper_pp - item.lower_pp)
    smallest = min(resolution, key=lambda item: item.minimum_detectable_pp)

    typer.echo(f"Wrote {len(estimates)} horizon estimates to {destination.relative_to(root)}")
    typer.echo(f"  window from {start}, {panel['region'].nunique()} regions")
    typer.echo(
        f"  high-low exposure contrast {100 * contrast:.2f}pp; for a 10% statutory rise the "
        f"design resolves differential responses down to {smallest.minimum_detectable_pp:.2f}pp"
    )
    typer.echo(
        f"  widest interval: horizon {widest.horizon}, "
        f"[{widest.lower_pp:.2f}, {widest.upper_pp:.2f}]pp. Judge the null by this, not by p."
    )
    survivors = int((estimates["p_value_bootstrap_holm"] < 0.05).sum())
    typer.echo(
        f"  significant at 5%: {int((estimates['p_value_clustered'] < 0.05).sum())} by clustered "
        f"inference, {int((estimates['p_value_bootstrap'] < 0.05).sum())} by the bootstrap, "
        f"{survivors} after Holm"
    )


@analyse_app.command("exposure-robustness")
def analyse_exposure_robustness(
    prices: Path = typer.Option(
        Path("data/processed/regional_price_panel.parquet"), help="Regional price panel."
    ),
    wages: Path = typer.Option(
        Path("data/processed/minimum_wage_policy.parquet"), help="Statutory panel."
    ),
    regional: Path = typer.Option(
        Path("data/processed/regional_employment.parquet"), help="Regional employment."
    ),
    national: Path = typer.Option(
        Path("data/processed/national_employment.parquet"), help="National employment."
    ),
    start: str = typer.Option("2016-01", help="First month of the baseline window."),
) -> None:
    """Re-estimate the exposure design under every discretionary choice.

    A null that holds in the specification its author picked, and nowhere else,
    is not a finding. This varies the survey round the bite comes from, the year
    composition is frozen at, the assumed bite in the sectors the survey misses,
    and which region is dropped, and reports what each yields.
    """
    root = _repo_root()
    for path in (prices, wages, regional, national):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; build it first")

    settings = _load_settings(root)
    horizons = list(settings["pass_through"]["horizons_months"])

    acts = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    gaps = {
        block["geography"]: frozenset(block.get("gap_years", []) or [])
        for block in acts["regional"].values()
    }
    registry = yaml.safe_load((root / "config/minimum_wage_bite.yaml").read_text(encoding="utf-8"))
    regional_employment = pd.read_parquet(root / regional)
    national_employment = pd.read_parquet(root / national)
    price_panel = pd.read_parquet(root / prices)
    wage_panel = pd.read_parquet(root / wages).query("scope == 'general'")

    def build(period: str, baseline: int) -> pd.DataFrame:
        selected = select_snapshot(registry, period)
        shares = industry_shares(regional_employment, year=baseline)
        bite = activity_bite_from_registry(selected, national_employment)
        return shift_share_exposure(shares, bite)

    def estimate(exposure: pd.DataFrame, drop: str | None = None) -> pd.DataFrame:
        panel = build_estimation_panel(price_panel, wage_panel, start=start, gap_years=gaps)
        if drop is not None:
            panel = panel.loc[panel["region"] != drop]
        return estimate_panel_local_projections(
            add_exposure_interaction(panel, exposure),
            outcome="log_price",
            shock="exposure_shock",
            horizons=horizons,
            cluster="region",
        )

    runs = []
    baseline_exposure = build("2015-10", 2015)
    runs.append(
        summarise_run("baseline: 2015-10 bite, 2015 composition", estimate(baseline_exposure))
    )

    for period in sorted(registry.get("snapshots") or {}):
        if period != "2015-10":
            runs.append(summarise_run(f"bite from {period}", estimate(build(period, 2015))))

    for baseline in (2013, 2014):
        runs.append(
            summarise_run(f"composition frozen at {baseline}", estimate(build("2015-10", baseline)))
        )

    for incidence in (0.0, 0.10, 0.21):
        bounded = bound_unmeasured_exposure(baseline_exposure, incidence)
        runs.append(summarise_run(f"unsurveyed bite u={incidence:.2f}", estimate(bounded)))

    for region in sorted(baseline_exposure["region"]):
        runs.append(summarise_run(f"leave out {region}", estimate(baseline_exposure, drop=region)))

    write_robustness_table(runs, root / "report/tables/exposure_robustness.tex")

    typer.echo(f"Estimated {len(runs)} specifications")
    rejecting = [run for run in runs if run.rejections_holm]
    typer.echo(f"  specifications where any horizon survives Holm: {len(rejecting)} of {len(runs)}")
    widest = max(runs, key=lambda run: run.max_coefficient - run.min_coefficient)
    typer.echo(
        f"  widest coefficient range: {widest.label} "
        f"[{widest.min_coefficient:.2f}, {widest.max_coefficient:.2f}]"
    )


@data_app.command("freeze-inputs")
def data_freeze_inputs(
    manifest: Path = typer.Option(
        Path("config/publication_inputs.json"),
        help="Committed manifest recording the checksum of every raw input.",
    ),
) -> None:
    """Record the checksums of the raw inputs behind a publishable run."""
    root = _repo_root()
    try:
        frozen = write_manifest(root / "data/raw", root / manifest)
    except FreezeError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"Froze {frozen} inputs to {manifest}")
    typer.echo("  commit this file; the raw bytes themselves are not distributed")


@data_app.command("verify-inputs")
def data_verify_inputs(
    manifest: Path = typer.Option(
        Path("config/publication_inputs.json"),
        help="Committed manifest to check the working tree against.",
    ),
    strict: bool = typer.Option(
        False, help="Exit non-zero when an input has changed or gone missing."
    ),
) -> None:
    """Report raw inputs that have changed since the manifest was frozen.

    Statistical agencies revise their history. This does not prevent that; it
    makes it visible at the moment it happens, rather than after a figure in the
    manuscript has quietly moved.
    """
    root = _repo_root()
    try:
        report = verify_manifest(root / "data/raw", root / manifest)
    except FreezeError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"Verified {report.verified} inputs against {manifest}")
    for label, names in (
        ("changed upstream", report.changed),
        ("missing locally", report.missing),
        ("not yet frozen", report.added),
    ):
        if names:
            typer.echo(f"  {len(names)} {label}:")
            for name in names[:5]:
                typer.echo(f"    {name}")
            if len(names) > 5:
                typer.echo(f"    ... and {len(names) - 5} more")

    if report.changed:
        typer.echo(
            "  A changed input means the published numbers were computed from "
            "different bytes. Re-freeze deliberately, and rebuild the paper."
        )
    if strict and not report.clean:
        raise typer.Exit(code=1)


# Kept last on purpose. Every command must be registered before ``app()`` is
# reached, and this module registers some of them below where this block used
# to sit: ``python -m pt_mw_inflation.cli analyse --help`` was omitting
# ``exposure-design`` entirely. The installed console script was unaffected,
# because import completes before it calls ``app()``, so the two entrypoints
# disagreed about which commands existed.
if __name__ == "__main__":
    app()
