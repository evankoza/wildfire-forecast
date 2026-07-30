"""Command line: ingest -> build -> backtest.

    python -m wildfire ingest-fires  --years 2024 2025 2026
    python -m wildfire ingest-hotspots --years 2024 2025
    python -m wildfire diagnose
    python -m wildfire build
    python -m wildfire backtest --test-years 2026
    python -m wildfire backtest --holdout-agency ALL     # region-blocked
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


def _fmt_ci(ci: dict, key: str) -> str:
    lo_hi = ci.get(key)
    if not lo_hi:
        return "-"
    return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def _interval_header(ci: dict) -> str:
    """Label the interval column with the percentiles actually used."""
    pct = ci.get("percentiles")
    return f"{pct[0]}-{pct[-1]} pctile" if pct else "interval"


@app.command("backtest")
def backtest(
    test_years: list[int] = typer.Option(None, "--test-years"),
    holdout_agency: list[str] = typer.Option(
        None, "--holdout-agency", "-a",
        help="hold out a REGION, not a season: repeat the flag per agency, "
             "or pass ALL to sweep every agency with enough positives",
    ),
    min_positives: int = typer.Option(
        10, "--min-positives", help="skip agency folds thinner than this"
    ),
    drop_geography: bool = typer.Option(
        False, "--drop-geography",
        help="refit without lat/lon/agency/region to see what actually transfers",
    ),
    n_boot: int = typer.Option(1000, "--bootstrap", help="0 to skip the interval"),
):
    """Train on earlier seasons (or other regions), evaluate on held-out ones."""
    df = pl.read_parquet(config.CURATED / "modelling_table.parquet")

    if holdout_agency:
        agencies = None if [a.upper() for a in holdout_agency] == ["ALL"] else \
            [a.upper() for a in holdout_agency]
        _spatial(df, agencies, test_years, min_positives, drop_geography, n_boot)
        return

    res = escalation.train_and_backtest(
        df, test_years=test_years or None, n_boot=n_boot
    )

    t = Table(title="Escalation backtest (season-blocked)", header_style="bold")
    t.add_column("metric"); t.add_column("value", justify="right")
    t.add_column(_interval_header(res.ci), justify="right")

    d = asdict(res)
    bootstrapped = {"pr_auc_model", "pr_auc_size_only"}  # CI keys match field names
    for k in ("train_years", "test_years", "n_train", "n_test",
              "positive_rate_train", "positive_rate_test",
              "pr_auc_model", "pr_auc_size_only", "pr_auc_prevalence",
              "lift_over_size_baseline", "roc_auc_model",
              "brier_model", "brier_uncalibrated", "brier_prevalence"):
        t.add_row(k, str(d[k]), _fmt_ci(res.ci, k) if k in bootstrapped else "")
    if res.ci.get("pr_auc_delta"):
        t.add_row("[bold]model - size baseline[/]",
                  str(round(res.pr_auc_model - res.pr_auc_size_only, 4)),
                  _fmt_ci(res.ci, "pr_auc_delta"))
        t.add_row("P(model > size baseline)", str(res.ci["p_model_beats_size"]), "")
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


def _spatial(df, agencies, test_years, min_positives, drop_geography, n_boot):
    """Leave-one-agency-out: does it work in country it has never seen?"""
    res = escalation.spatial_backtest(
        df,
        agencies=agencies,
        test_years=test_years or None,
        min_test_positives=min_positives,
        drop_geography=drop_geography,
        n_boot=n_boot,
    )

    title = f"Leave-one-agency-out - {res.mode}"
    if res.dropped_features:
        title += f" - without {', '.join(res.dropped_features)}"
    t = Table(title=title, header_style="bold")
    for c in ("held out", "n_test", "pos", "PR-AUC",
              _interval_header(res.pooled["ci"]), "size base", "lift", "P(beats)"):
        t.add_column(c, justify="right" if c != "held out" else "left")
    for f in res.folds:
        colour = "green" if f.pr_auc_model > f.pr_auc_size_only else "red"
        t.add_row(
            f.holdout_agency, f"{f.n_test:,}", str(f.n_positives_test),
            f"[{colour}]{f.pr_auc_model}[/]", _fmt_ci(f.ci, "pr_auc_model"),
            str(f.pr_auc_size_only), str(f.lift_over_size_baseline),
            str(f.ci.get("p_model_beats_size", "-")),
        )
    p = res.pooled
    t.add_section()
    t.add_row("[bold]POOLED[/]", f"{p['n_test']:,}", str(p["n_positives"]),
              f"[bold]{p['pr_auc_model']}[/]", _fmt_ci(p["ci"], "pr_auc_model"),
              str(p["pr_auc_size_only"]), str(p["lift_over_size_baseline"]),
              str(p["ci"].get("p_model_beats_size", "-")))
    console.print(t)

    console.print(
        f"train seasons {res.train_years} / test seasons {res.test_years} - "
        f"{res.macro['folds_beating_baseline']}/{res.macro['n_folds']} folds beat "
        f"the size-at-decision baseline, median lift {res.macro['median_lift']}x"
    )
    for s in res.skipped:
        console.print(f"[yellow]skipped {s['agency']}: {s['reason']}[/]")

    rt = Table(title="Calibration (pooled out-of-region)", header_style="bold")
    for c in ("bucket", "n", "mean_predicted", "observed_rate"):
        rt.add_column(c, justify="right")
    for r in p["reliability"]:
        rt.add_row(str(r["bucket"]), str(r["n"]), str(r["mean_predicted"]),
                   str(r["observed_rate"]))
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
