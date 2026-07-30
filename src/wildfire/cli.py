"""Command line: ingest -> build -> backtest.

    python -m wildfire ingest-fires  --years 2024 2025 2026
    python -m wildfire ingest-hotspots --years 2024 2025
    python -m wildfire diagnose
    python -m wildfire build
    python -m wildfire backtest --test-years 2026
    python -m wildfire scrape-ciffc
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from . import config
from .config import SPEC
from .features import build as feat_build
from .models import escalation
from .sources import cwfis_fires, cwfis_hotspots

app = typer.Typer(add_completion=False, help="Canadian wildfire escalation forecaster")
console = Console()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)


@app.command("ingest-fires")
def ingest_fires(
    years: list[int] = typer.Option(..., "--years", "-y"),
    force: bool = typer.Option(False, "--force"),
):
    """Pull the bitemporal reported-fires layer for the given fire years."""
    df = cwfis_fires.load(years, force=force)
    console.print(f"[green]{df.height:,}[/] revision rows for {df['national_fire_id'].n_unique():,} fires")
    _print_depth(cwfis_fires.bitemporal_depth(df))


@app.command("ingest-hotspots")
def ingest_hotspots(
    years: list[int] = typer.Option(..., "--years", "-y"),
    force: bool = typer.Option(False, "--force"),
):
    """Pull CWFIS satellite hotspot archives."""
    df = cwfis_hotspots.load_seasons(years, force=force)
    console.print(f"[green]{df.height:,}[/] hotspot detections")


@app.command("diagnose")
def diagnose():
    """Check whether the bitemporal history is deep enough to model on.

    A year whose rows_per_fire is close to 1.0 was backfilled as a snapshot:
    the 'what was known at T' premise does not hold there and it must not be
    used for training.
    """
    path = config.CURATED / "reported_fires.parquet"
    if not path.exists():
        raise typer.BadParameter("run ingest-fires first")
    _print_depth(cwfis_fires.bitemporal_depth(pl.read_parquet(path)))


def _print_depth(depth: pl.DataFrame):
    t = Table(title="Bitemporal depth by fire year", header_style="bold")
    for c in ("fire_year", "n_fires", "n_rows", "rows_per_fire", "first_record", "last_record"):
        t.add_column(c)
    for r in depth.iter_rows(named=True):
        rpf = r["rows_per_fire"]
        colour = "red" if rpf is not None and rpf < 1.5 else "green"
        t.add_row(
            str(r["fire_year"]), f"{r['n_fires']:,}", f"{r['n_rows']:,}",
            f"[{colour}]{rpf}[/]",
            str(r["first_record"])[:16], str(r["last_record"])[:16],
        )
    console.print(t)
    console.print("[dim]rows_per_fire < 1.5 (red) = snapshot backfill, not usable for as-of features[/]")


@app.command("build")
def build(
    with_weather: bool = typer.Option(False, "--weather", help="add ERA5 (slow: 1 request per fire-cell)"),
):
    """Assemble the point-in-time modelling table."""
    fires = pl.read_parquet(config.CURATED / "reported_fires.parquet")
    hs_path = config.CURATED / "hotspots.parquet"
    hotspots = pl.read_parquet(hs_path) if hs_path.exists() else pl.DataFrame()
    if hotspots.is_empty():
        console.print("[yellow]no hotspots ingested - satellite features will be absent[/]")

    df = feat_build.build(fires, hotspots, with_weather=with_weather)
    console.print(
        f"[green]{df.height:,}[/] fires; "
        f"[bold]{int(df['escalated'].sum()):,}[/] escalated "
        f"({100 * df['escalated'].mean():.2f}%) at >= {SPEC.size_threshold_ha} ha "
        f"by T0+{SPEC.horizon_hours}h"
    )


@app.command("backtest")
def backtest(test_years: list[int] = typer.Option(None, "--test-years")):
    """Train on earlier seasons, evaluate on held-out ones."""
    df = pl.read_parquet(config.CURATED / "modelling_table.parquet")
    res = escalation.train_and_backtest(df, test_years=test_years or None)

    t = Table(title="Escalation backtest", header_style="bold")
    t.add_column("metric"); t.add_column("value", justify="right")
    d = asdict(res)
    for k in ("train_years", "test_years", "n_train", "n_test",
              "positive_rate_train", "positive_rate_test",
              "pr_auc_model", "pr_auc_size_only", "pr_auc_prevalence",
              "lift_over_size_baseline", "roc_auc_model",
              "brier_model", "brier_uncalibrated", "brier_prevalence"):
        t.add_row(k, str(d[k]))
    console.print(t)

    ft = Table(title="Top features", header_style="bold")
    ft.add_column("feature"); ft.add_column("gain", justify="right")
    for name, gain in res.top_features:
        ft.add_row(name, str(gain))
    console.print(ft)

    rt = Table(title="Calibration (test)", header_style="bold")
    for c in ("bucket", "n", "mean_predicted", "observed_rate"):
        rt.add_column(c, justify="right")
    for r in res.reliability:
        rt.add_row(str(r["bucket"]), str(r["n"]), str(r["mean_predicted"]), str(r["observed_rate"]))
    console.print(rt)


@app.command("scrape-ciffc")
def scrape_ciffc():
    """Render and parse today's CIFFC national situation report."""
    from .sources import ciffc

    try:
        snap = ciffc.snapshot()
    except ciffc.PlaywrightMissing as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1)
    console.print(
        f"sitrep [bold]{snap['sitrep_date']}[/] "
        f"preparedness level [bold]{snap['preparedness_level']}[/], "
        f"{len(snap['new_fires_by_agency'])} agency rows"
    )


if __name__ == "__main__":
    app()
