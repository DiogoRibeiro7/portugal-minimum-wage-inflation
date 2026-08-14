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
    write_structural_design_macros,
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
from pt_mw_inflation.data.labour_shares import fetch_labour_shares
from pt_mw_inflation.data.labour_shares import to_frame as labour_shares_to_frame
from pt_mw_inflation.data.registry import download_registry
from pt_mw_inflation.data.supply_use import fetch_household_consumption
from pt_mw_inflation.data.supply_use import to_frame as consumption_to_frame
from pt_mw_inflation.data.worldbank import fetch_indicator
from pt_mw_inflation.processing.consumption_bridge import (
    ConsumptionBridgeError,
    build_consumption_bridge,
)
from pt_mw_inflation.processing.exposure import (
    ExposureError,
    PredeterminationError,
    activity_bite_from_registry,
    bound_unmeasured_exposure,
    check_predetermined,
    measure_variation_strength,
    select_snapshot,
    shift_share_exposure,
    structural_exposure,
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
    add_structural_interaction,
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


@build_app.command("consumption-bridge")
def build_consumption_bridge_command(
    year: int = typer.Option(2015, help="Reference year of the use table."),
    margin_rule: str = typer.Option(
        "goods",
        help="Trade-margin allocation: 'goods' in proportion to goods content, "
        "'uniform' evenly across categories, 'own' unallocated.",
    ),
    consumption_output: Path = typer.Option(
        Path("data/processed/household_consumption.parquet"),
        help="Household consumption by CPA product, at basic prices, domestic uses.",
    ),
    labour_output: Path = typer.Option(
        Path("data/processed/labour_shares.parquet"),
        help="Labour-cost share by activity, the other national term.",
    ),
    output: Path = typer.Option(
        Path("data/processed/consumption_bridge.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Build the production-to-consumption bridge from the use table and the concordance.

    Three of the four terms the region-by-category exposure needs are published.
    This builds the fourth, which is not: no source crosses a consumption purpose
    with a producing industry for Portugal, so the bridge combines a measured
    consumption vector with the concordance recorded in
    `config/consumption_bridge.yaml`.

    The measurement is taken at basic prices and domestic uses only. Both are
    arguments rather than defaults, and the module docstring says why: at
    purchasers' prices the retail margin on a good is credited to the industry
    that made it, and including imports credits Portuguese employment with costs
    incurred abroad.
    """
    root = _repo_root()
    registry = yaml.safe_load((root / "config/consumption_bridge.yaml").read_text(encoding="utf-8"))

    consumption = fetch_household_consumption(year=year)
    consumption_frame = consumption_to_frame(consumption)
    labour = fetch_labour_shares(year=year)

    try:
        bridge, coverage = build_consumption_bridge(
            registry, consumption_frame, margin_rule=margin_rule
        )
    except ConsumptionBridgeError as error:
        raise typer.BadParameter(str(error)) from error

    for frame, path in (
        (consumption_frame, consumption_output),
        (labour_shares_to_frame(labour), labour_output),
        (bridge, output),
    ):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)

    typer.echo(
        f"Wrote {bridge['category'].nunique()} categories x "
        f"{bridge['industry'].nunique()} industries to {output}"
    )
    typer.echo(
        f"  consumption from the {year} use table at basic prices, domestic uses: "
        f"{consumption.product_total:,.0f} of {consumption.purchasers_total:,.0f} MEUR"
    )
    typer.echo(
        f"  {100 * consumption.excluded_share:.1f}% of household spending has no producing "
        f"industry behind it ({100 * consumption.imported_share:.1f}% imported content, "
        f"{100 * consumption.tax_share:.1f}% product taxes). No concordance can reach it."
    )
    typer.echo(f"  concordance places {100 * coverage.matched_share:.1f}% of domestic consumption")
    if coverage.unmatched_products:
        typer.echo(
            f"  {len(coverage.unmatched_products)} product(s) unplaced: "
            f"{', '.join(coverage.unmatched_products[:5])}"
        )
    typer.echo(f"  labour shares validated against an economy-wide {labour.aggregate:.3f}")
    if labour.suppressed:
        typer.echo(
            f"  suppressed as an accounting artefact: {', '.join(labour.suppressed)}. "
            "Real estate value added is imputed rent, which employs nobody."
        )


@build_app.command("structural-exposure")
def build_structural_exposure(
    regional: Path = typer.Option(
        Path("data/processed/regional_employment.parquet"),
        help="Regional employment from 'ptmw data regional-employment'.",
    ),
    national: Path = typer.Option(
        Path("data/processed/national_employment.parquet"),
        help="National employment by NACE section.",
    ),
    bridge: Path = typer.Option(
        Path("data/processed/consumption_bridge.parquet"),
        help="Production-to-consumption bridge from 'ptmw build consumption-bridge'.",
    ),
    labour: Path = typer.Option(
        Path("data/processed/labour_shares.parquet"), help="Labour-cost share by activity."
    ),
    baseline_year: int = typer.Option(2015, help="Year the composition is frozen at."),
    bite_period: str = typer.Option("2015-10", help="Survey round the bite comes from."),
    first_shock_year: int = typer.Option(
        0, help="First year of the episode. When set, predetermination is enforced."
    ),
    output: Path = typer.Option(
        Path("data/processed/structural_exposure.parquet"), help="Output Parquet path."
    ),
) -> None:
    """Build the region-by-category exposure from all four of its terms.

    Composition varies by region and the other three terms are national. That is
    not an objection to the measure: once region-time and category-time effects
    are absorbed, what identifies the coefficient is the non-additive part of the
    region-by-category matrix, and national factors enter it through their
    product with composition.

    The number to judge it by is the identifying spread reported below, which is
    the range left after those effects remove the matrix's row and column means.
    The raw spread is larger and does not describe what the design has.
    """
    root = _repo_root()
    for path in (regional, national, bridge, labour):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; build it first")

    registry = yaml.safe_load((root / "config/minimum_wage_bite.yaml").read_text(encoding="utf-8"))
    try:
        registry = select_snapshot(registry, bite_period or None)
    except ExposureError as error:
        raise typer.BadParameter(str(error)) from error

    if first_shock_year:
        try:
            check_predetermined(registry, baseline_year, first_shock_year)
        except PredeterminationError as error:
            raise typer.BadParameter(str(error)) from error

    shares = industry_shares(pd.read_parquet(root / regional), year=baseline_year)
    bite = activity_bite_from_registry(registry, pd.read_parquet(root / national))

    try:
        exposure, coverage = structural_exposure(
            shares,
            bite,
            pd.read_parquet(root / labour),
            pd.read_parquet(root / bridge),
        )
    except ExposureError as error:
        raise typer.BadParameter(str(error)) from error

    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_parquet(destination, index=False)

    typer.echo(
        f"Wrote {len(exposure)} region-category cells "
        f"({exposure['region'].nunique()} x {exposure['category'].nunique()}) to {output}"
    )
    typer.echo(
        f"  identifying spread {coverage.identifying_spread:.2f}pp after region-time and "
        "category-time effects. Judge the design by this, not by the raw range."
    )
    typer.echo(f"  bite from the {registry['source']['reference_period']} survey round")
    typer.echo(
        f"  {100 * coverage.measured_share:.1f}% of the bridge's weight reaches an activity "
        "carrying both a bite and a labour share"
    )
    for activity, missing in coverage.unmeasured_activities:
        typer.echo(f"    {activity}: no {missing}; contributes nothing rather than being imputed")
    if not coverage.coverage_weighted:
        typer.echo(
            "  WARNING: the bite carried no measured_employment_share, so every group was "
            "treated as fully surveyed. The spread above is overstated."
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


def _echo_intervals(estimates: pd.DataFrame) -> None:
    """Report the inverted interval, which is what a null p-value cannot say.

    Reported at the widest horizon rather than the narrowest. The narrowest
    flatters the design, and the question a reader has after a table of nulls is
    what the design failed to rule out, which is what the widest one answers.
    """
    if "interval_lower" not in estimates.columns:
        return

    ordered = estimates.reset_index(drop=True)
    widths = (ordered["interval_upper"] - ordered["interval_lower"]).to_numpy(dtype=float)
    widest = int(widths.argmax())
    horizon = int(ordered["horizon"].to_numpy()[widest])
    lower = float(ordered["interval_lower"].to_numpy()[widest])
    upper = float(ordered["interval_upper"].to_numpy()[widest])
    typer.echo(f"  95% inverted interval at horizon {horizon} (widest): [{lower:.3f}, {upper:.3f}]")
    if not bool(estimates["interval_bounded"].all()):
        unbounded = int((~estimates["interval_bounded"]).sum())
        typer.echo(
            f"  {unbounded} interval(s) ran past the widest range searched. "
            "Their endpoints are where the search stopped, not where the interval does."
        )


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
    _echo_intervals(estimates)
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
    _echo_intervals(estimates)


@analyse_app.command("structural-design")
def analyse_structural_design(
    prices: Path = typer.Option(
        Path("data/processed/regional_price_panel.parquet"), help="Regional price panel."
    ),
    wages: Path = typer.Option(
        Path("data/processed/minimum_wage_policy.parquet"), help="Statutory panel."
    ),
    exposure: Path = typer.Option(
        Path("data/processed/structural_exposure.parquet"),
        help="Region-by-category exposure from 'ptmw build structural-exposure'.",
    ),
    start: str = typer.Option("2016-01", help="First month; must follow the bite's survey round."),
) -> None:
    """Estimate the region-by-category design and write its table.

    This is the design the decision log gated on the consumption bridge being
    concentrated enough to build. Its advantage over the shift-share design is
    not a wider regressor but the fixed effects it can carry: because exposure
    varies across regions *and* categories, the interaction survives region-time
    effects, which absorb the tourism, transport and island-supply shocks that
    make the autonomous regions a poor control, and category-time effects, which
    absorb the January sales cycle that defeated the category design.

    It does not relax the constraint the rest of the paper documents. Policy is
    assigned by region, so inference still clusters on nine regions however many
    region-category cells the panel holds.
    """
    root = _repo_root()
    for path in (prices, wages, exposure):
        if not (root / path).exists():
            raise typer.BadParameter(f"{path} not found; build it first")

    settings = _load_settings(root)
    horizons = list(settings["pass_through"]["horizons_months"])

    acts = yaml.safe_load((root / "config/legal_acts.yaml").read_text(encoding="utf-8"))
    gaps = {
        block["geography"]: frozenset(block.get("gap_years", []) or [])
        for block in acts["regional"].values()
    }

    panel = build_estimation_panel(
        pd.read_parquet(root / prices),
        pd.read_parquet(root / wages).query("scope == 'general'"),
        start=start,
        gap_years=gaps,
    )
    exposure_frame = pd.read_parquet(root / exposure)
    try:
        panel = add_structural_interaction(panel, exposure_frame)
    except PassThroughError as error:
        raise typer.BadParameter(str(error)) from error

    # What the design has after its own fixed effects, recomputed here rather
    # than carried from the build step so the reported figure describes the
    # exposure file this run actually read.
    matrix = exposure_frame.pivot(
        index="region", columns="category", values="structural_exposure"
    ).to_numpy(dtype=float)
    residual = (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )
    identifying_spread = 100.0 * float(residual.max() - residual.min())

    # What complete pass-through could produce. The exposure is not a cost share
    # --- it weights industries by regional employment and runs about a third of
    # one --- so the coefficient cannot be read as a multiple of pass-through.
    # The largest category cost share can be, and it bounds the differential
    # response that passing every euro of minimum-wage cost into prices would
    # give. An estimate above it is impossible rather than merely large.
    cost_share_ceiling = float(exposure_frame["category_cost_share"].max())

    absorb = ["region_category", "region_month", "category_month"]
    typer.echo(
        f"Estimating on {len(panel):,} rows, {panel['region_category'].nunique()} cells, "
        f"{panel['month'].nunique()} months, absorbing {', '.join(absorb)}"
    )
    typer.echo("  The three-way design is large; each horizon takes about a minute.")

    estimates = estimate_panel_local_projections(
        panel,
        outcome="log_price",
        shock="structural_shock",
        horizons=horizons,
        cluster="region",
        absorb=absorb,
    )
    if estimates.empty:
        raise typer.BadParameter("no horizon could be estimated")

    destination = root / "report/tables/structural_design.tex"
    write_regional_design_table(estimates, destination, command="ptmw analyse structural-design")
    write_structural_design_macros(
        estimates,
        root / "report/tables/structural_design_macros.tex",
        identifying_spread=identifying_spread,
        cost_share_ceiling=cost_share_ceiling,
    )

    # The falsification battery applies here too, for the reason it applies to
    # the exposure design: reporting it for two designs and not the third would
    # let the untested one borrow the others' credibility.
    pre_trend = assess_pre_trends(
        panel, outcome="log_price", shock="structural_shock", absorb=absorb
    )
    write_pre_trend_macros(
        pre_trend,
        root / "report/tables/structural_pre_trend_macros.tex",
        prefix="Structural",
        command="ptmw analyse structural-design",
    )

    typer.echo(f"Wrote {len(estimates)} horizon estimates to {destination.relative_to(root)}")
    typer.echo(
        f"  joint pre-trend test on {pre_trend.restrictions} leads: p = {pre_trend.p_value:.3f}"
    )
    typer.echo(
        f"  significant at 5%: {int((estimates['p_value_clustered'] < 0.05).sum())} by clustered "
        f"inference, {int((estimates['p_value_bootstrap'] < 0.05).sum())} by the bootstrap, "
        f"{int((estimates['p_value_bootstrap_holm'] < 0.05).sum())} after Holm"
    )

    # The coefficient is not readable on its own, so it is reported scaled into
    # points against the ceiling complete pass-through could reach.
    implied = estimates["coefficient"].abs() * (identifying_spread / 100.0) * 0.10 * 100.0
    ceiling = 100.0 * cost_share_ceiling * 0.10
    typer.echo(
        f"  identifying spread {identifying_spread:.2f}pp; for a 10% statutory rise the "
        f"estimates imply differential responses up to {implied.max():.1f}pp"
    )
    typer.echo(
        f"  complete pass-through of the largest category cost share "
        f"({100 * cost_share_ceiling:.0f}%) could reach {ceiling:.1f}pp. "
        f"{int((implied > ceiling).sum())} horizon(s) exceed it, which is impossible, not large."
    )
    _echo_intervals(estimates)


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
            # Every run here is reduced to a coefficient range and two rejection
            # counts, so inverting the test per horizon would be discarded work.
            intervals=False,
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
