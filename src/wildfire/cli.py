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
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Refresh only these seasons, keeping the others already curated. "
        "Without it the curated table is replaced by exactly the years given.",
    ),
):
    """Pull CWFIS satellite hotspot archives."""
    df = cwfis_hotspots.load_seasons(years, force=force, merge=merge)
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
    """Format an interval with enough digits to be an interval.

    The escalation model's PR-AUCs are around 0.26 and three decimals are
    plenty; the ignition model's are around 0.019 against a 0.0009 base rate,
    and three decimals round two of the three numbers in a row to the same
    value. Scale the precision to the magnitude rather than picking one and
    making half the tables useless.
    """
    lo_hi = ci.get(key)
    if not lo_hi:
        return "-"
    digits = 3 if max(abs(v) for v in lo_hi) >= 0.1 else 5
    return f"[{lo_hi[0]:.{digits}f}, {lo_hi[1]:.{digits}f}]"


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


def _ablation_table(res: dict, title: str, digits: int = 4) -> Table:
    t = Table(title=title, header_style="bold")
    lo, hi = res["percentiles"][0], res["percentiles"][-1]
    for c in ("feature set", "PR-AUC", f"{lo}-{hi} pctile", "delta vs full",
              "paired interval", "P(full better)"):
        t.add_column(c, justify="right" if c != "feature set" else "left")
    for r in res["rows"]:
        d = r.get("delta_vs_full")
        t.add_row(
            r["feature_set"], str(r["pr_auc"]),
            f"[{r['interval'][0]}, {r['interval'][1]}]",
            "-" if d is None else f"{d:+.{digits}f}",
            "-" if d is None else f"[{r['delta_interval'][0]}, {r['delta_interval'][1]}]",
            "-" if d is None else str(r["p_full_is_better"]),
        )
    return t


@app.command("ablate")
def ablate(
    test_years: list[int] = typer.Option(None, "--test-years"),
    n_boot: int = typer.Option(1000, "--bootstrap"),
):
    """Which blocks of features the escalation model actually needs.

    Every variant is refit on the same split and the differences are
    bootstrapped paired, which is the only way this table is readable: with
    ~117 positives every marginal interval swallows every other point estimate.
    """
    df = pl.read_parquet(config.CURATED / "modelling_table.parquet")
    res = escalation.ablation(df, test_years=test_years or None, n_boot=n_boot)
    console.print(_ablation_table(
        res, f"Escalation feature ablation - test {res['test_years']}, "
             f"{res['n_positives_test']} positives"))
    console.print(f"[dim]size-at-decision baseline: {res['pr_auc_size_only']}[/]")


@app.command("compare-geography")
def compare_geography(
    test_years: list[int] = typer.Option(None, "--test-years"),
    min_positives: int = typer.Option(10, "--min-positives"),
    n_boot: int = typer.Option(1000, "--bootstrap"),
):
    """Do lat/lon/agency/region help or hurt in country the model has not seen?

    Runs the leave-one-agency-out backtest with and without them and pairs the
    two on the same pooled out-of-region fires.
    """
    df = pl.read_parquet(config.CURATED / "modelling_table.parquet")
    res = escalation.geography_paired(
        df, test_years=test_years or None, min_test_positives=min_positives,
        n_boot=n_boot,
    )
    t = Table(title="Pooled out-of-region PR-AUC, paired", header_style="bold")
    t.add_column("metric"); t.add_column("value", justify="right")
    for k in ("n_pooled", "n_positives", "pr_auc_with_geography",
              "pr_auc_without_geography", "paired_delta", "delta_interval",
              "p_geography_helps"):
        t.add_row(k, str(res[k]))
    console.print(t)
    console.print(f"[dim]folds: {', '.join(res['agencies'])}[/]")


@app.command("ingest-ciffc")
def ingest_ciffc(
    years: list[int] = typer.Option(None, "--years", "-y"),
    force: bool = typer.Option(False, "--force"),
):
    """Backfill the CIFFC preparedness series from the dated public API.

    One request per published sitrep date; reports exist only for the fire
    season, so a year costs ~140 requests and responses are cached verbatim.
    """
    from .sources import ciffc

    df = ciffc.load_history(years or None, force=force)
    console.print(
        f"[green]{df.height:,}[/] agency-day rows across "
        f"{df['agency_code'].n_unique()} agencies"
    )

    t = Table(title="CIFFC preparedness coverage", header_style="bold")
    for c in ("year", "n_days", "n_rows", "mean_national_pl", "max_national_pl",
              "mean_agency_pl"):
        t.add_column(c, justify="right")
    for r in ciffc.preparedness_coverage(df).iter_rows(named=True):
        t.add_row(*(str(r[c]) for c in ("year", "n_days", "n_rows",
                                        "mean_national_pl", "max_national_pl",
                                        "mean_agency_pl")))
    console.print(t)


@app.command("fit-final")
def fit_final():
    """Refit on every labelled season, for scoring live fires.

    Separate from `backtest` on purpose: the backtest model is handicapped so
    a season can be held out honestly, and nothing should be *scored* with it.
    """
    df = pl.read_parquet(config.CURATED / "modelling_table.parquet")
    payload = escalation.fit_final(df)
    console.print(
        f"final model: [green]{payload['n_train']:,}[/] fires, "
        f"[bold]{payload['n_positives']}[/] escalations, seasons {payload['train_years']}"
    )
    console.print("[dim]accuracy for this model is inferred from the backtest, "
                  "not measured on the rows it was fitted on[/]")


@app.command("predict")
def predict_cmd(
    window_days: int = typer.Option(14, "--window-days",
                                    help="how far back a first report can be"),
    top: int = typer.Option(15, "--top"),
    as_of: str = typer.Option(None, "--as-of", help="ISO timestamp; default = newest record"),
):
    """Rank fires burning now by probability of exceeding the size threshold."""
    from datetime import datetime as _dt

    from . import predict as predict_mod

    stamp = _dt.fromisoformat(as_of) if as_of else None
    df = predict_mod.score(as_of=stamp, window_days=window_days)
    if df.is_empty():
        console.print("[yellow]no fires in the scoring window[/]")
        return

    t = Table(title=f"Escalation risk - top {min(top, df.height)} of {df.height:,} fires",
              header_style="bold")
    for c, j in (("fire", "left"), ("age", "right"), ("ha @ decision", "right"),
                 ("status @ decision", "left"), ("hotspots", "right"),
                 ("risk", "right"), ("size now", "right")):
        t.add_column(c, justify=j, overflow="fold")

    for r in df.head(top).iter_rows(named=True):
        risk = r["risk"]
        colour = "red" if risk >= 0.25 else "yellow" if risk >= 0.10 else "white"
        size_now = r.get("size_now")
        # The national id is "<year>_<agency>_<agency's own id>"; the first two
        # parts are constant down the column, so show the part that identifies
        # the fire and put the agency in front of it.
        fid = r["national_fire_id"] or ""
        short = fid.split("_", 2)[-1] if fid.count("_") >= 2 else fid
        t.add_row(
            f"{r['agency_code'] or '--'}  {short}",
            f"{r['age_hours'] // 24}d",
            f"{(r['size_at_decision'] or 0):,.1f}",
            predict_mod.STATUS_LABEL.get(r["status_at_decision"], r["status_at_decision"] or "-"),
            str(int(r["hs_count"] or 0)),
            f"[{colour}]{risk:.3f}[/]",
            f"{size_now:,.0f}" if size_now is not None else "-",
        )
    console.print(t)
    console.print(f"[dim]risk = P(>= {SPEC.size_threshold_ha:.0f} ha by T0+{SPEC.horizon_hours}h), "
                  f"as known {SPEC.decision_hours}h after first report[/]")


@app.command("ingest-fwi")
def ingest_fwi(
    years: list[int] = typer.Option(..., "--years", "-y"),
    force: bool = typer.Option(False, "--force"),
    daily: bool = typer.Option(
        False, "--daily",
        help="pull the season in progress from fwi_obs/current/ one day at a "
             "time and merge it in, keeping the other seasons. The decadal "
             "archive lags by most of a year, so this is the only way to score "
             "today.",
    ),
):
    """Pull station weather and Fire Weather Index observations.

    The exogenous leg of the ignition model: the hotspot feed's FWI exists
    only where something was already burning, which would make it circular
    evidence for where a fire will start. Backfill is one large decadal file,
    streamed once and re-parsed thereafter.
    """
    from .sources import cwfis_fwi

    df = (cwfis_fwi.load_current(years, force=force) if daily
          else cwfis_fwi.load(years, force=force))
    console.print(
        f"[green]{df.height:,}[/] station-days across "
        f"{df['aes'].n_unique():,} stations"
    )
    t = Table(title="Station FWI coverage", header_style="bold")
    for c in ("year", "n_days", "first_day", "last_day", "n_stations", "n_rows",
              "fwi_reported", "mean_fwi"):
        t.add_column(c, justify="right")
    for r in cwfis_fwi.coverage(df).iter_rows(named=True):
        t.add_row(*(str(r[c]) for c in ("year", "n_days", "first_day", "last_day",
                                        "n_stations", "n_rows", "fwi_reported",
                                        "mean_fwi")))
    console.print(t)


@app.command("build-ignition")
def build_ignition(
    years: list[int] = typer.Option(None, "--years", "-y",
                                    help="seasons to pose the question over"),
    area_years: list[int] = typer.Option(
        None, "--area-years",
        help="seasons the study area is drawn from; must EXCLUDE any season "
             "you intend to evaluate on, or the domain is defined by the "
             "fires you are about to forecast",
    ),
    neg_rate: float = typer.Option(
        None, "--neg-rate", help="fraction of non-event cell-days kept"
    ),
):
    """Assemble the ignition panel: one row per 10 km cell per day."""
    from .features import ignition as ig_feat

    fires = pl.read_parquet(config.CURATED / "reported_fires.parquet")
    hs_path = config.CURATED / "hotspots.parquet"
    hotspots = pl.read_parquet(hs_path) if hs_path.exists() else pl.DataFrame()
    obs_path = config.CURATED / "station_fwi.parquet"
    if not obs_path.exists():
        raise typer.BadParameter("run ingest-fwi first")
    obs = pl.read_parquet(obs_path)

    seasons = years or sorted({d.year for d in obs["obs_date"].to_list()})
    area = area_years or seasons[:-1] or seasons
    if set(area) & set(seasons[-1:]) and len(seasons) > 1:
        console.print(
            "[yellow]the study area includes the last season you are modelling; "
            "that is a look-ahead unless you mean to score, not evaluate[/]"
        )

    df = ig_feat.build_panel(
        fires, hotspots, obs, years=seasons, area_years=area,
        neg_rate=neg_rate if neg_rate is not None else ig_feat.NEG_RATE,
    )
    console.print(
        f"[green]{df.height:,}[/] cell-days; "
        f"[bold]{int(df[ig_feat.TARGET].sum()):,}[/] with an ignition "
        f"({100 * df[ig_feat.TARGET].mean():.2f}% of the sample)"
    )


@app.command("backtest-ignition")
def backtest_ignition(
    test_years: list[int] = typer.Option(None, "--test-years"),
    holdout_agency: list[str] = typer.Option(
        None, "--holdout-agency", "-a",
        help="hold out a REGION: repeat per agency, or ALL to sweep",
    ),
    min_positives: int = typer.Option(50, "--min-positives"),
    drop_geography: bool = typer.Option(False, "--drop-geography"),
    drop_network: bool = typer.Option(
        False, "--drop-network",
        help="refit without wx_dist_km / wx_n_stations, which measure the "
             "observing network rather than the ground",
    ),
    drop_ciffc: bool = typer.Option(False, "--drop-ciffc"),
    n_boot: int = typer.Option(1000, "--bootstrap"),
):
    """Train on earlier seasons (or other regions), evaluate on held-out ones."""
    from .models import ignition as ig_model

    df = pl.read_parquet(config.CURATED / "ignition_table.parquet")

    if holdout_agency:
        agencies = None if [a.upper() for a in holdout_agency] == ["ALL"] else \
            [a.upper() for a in holdout_agency]
        _spatial_ignition(df, agencies, test_years, min_positives,
                          drop_geography, n_boot)
        return

    res = ig_model.train_and_backtest(
        df, test_years=test_years or None, n_boot=n_boot,
        drop_network=drop_network, drop_ciffc=drop_ciffc,
    )

    t = Table(title="Ignition backtest (season-blocked)", header_style="bold")
    t.add_column("metric"); t.add_column("value", justify="right")
    t.add_column(_interval_header(res.ci), justify="right")
    d = asdict(res)
    for k in ("train_years", "test_years", "n_train", "n_test",
              "n_positives_test", "neg_rate", "population_prevalence",
              "pr_auc_model", "pr_auc_fwi", "pr_auc_climatology",
              "lift_over_fwi", "lift_over_climatology",
              "roc_auc_model", "brier_model", "brier_prevalence"):
        t.add_row(k, str(d[k]), _fmt_ci(res.ci, k) if k == "pr_auc_model" else "")
    if res.ci.get("pr_auc_delta"):
        t.add_row(f"[bold]model - {res.ci['baseline']}[/]",
                  str(round(res.pr_auc_model - (
                      res.pr_auc_climatology if res.ci["baseline"] == "climatology"
                      else res.pr_auc_fwi), 5)),
                  _fmt_ci(res.ci, "pr_auc_delta"))
        t.add_row("P(model > baseline)", str(res.ci["p_model_beats_baseline"]), "")
    console.print(t)
    console.print(
        "[dim]PR-AUC is sample-weighted back onto the true cell-day prevalence, "
        "so it is comparable with the baselines and not with an unweighted "
        "figure.[/]"
    )
    if res.brier_model > res.brier_prevalence:
        console.print(
            "[yellow]Brier is worse than predicting the base rate: the ranking "
            "is informative but the probability scale is not. See the "
            "reliability table.[/]"
        )

    ft = Table(title="Top features", header_style="bold")
    ft.add_column("feature"); ft.add_column("gain", justify="right")
    for name, gain in res.top_features:
        ft.add_row(name, str(gain))
    console.print(ft)

    rt = Table(title="Calibration (test, population-weighted)", header_style="bold")
    for c in ("bucket", "n", "mean_predicted", "observed_rate"):
        rt.add_column(c, justify="right")
    for r in res.reliability:
        rt.add_row(str(r["bucket"]), str(r["n"]), str(r["mean_predicted"]),
                   str(r["observed_rate"]))
    console.print(rt)


def _spatial_ignition(df, agencies, test_years, min_positives, drop_geography, n_boot):
    from .models import ignition as ig_model

    res = ig_model.spatial_backtest(
        df, agencies=agencies, test_years=test_years or None,
        min_test_positives=min_positives, drop_geography=drop_geography,
        n_boot=n_boot,
    )
    title = f"Ignition leave-one-agency-out - {res.mode}"
    if res.dropped_features:
        title += f" - without {', '.join(res.dropped_features)}"
    t = Table(title=title, header_style="bold")
    for c in ("held out", "n_test", "pos", "PR-AUC",
              _interval_header(res.pooled["ci"]), "fwi", "clim", "P(beats)"):
        t.add_column(c, justify="right" if c != "held out" else "left")
    for f in res.folds:
        best = max(f.pr_auc_fwi, f.pr_auc_climatology)
        colour = "green" if f.pr_auc_model > best else "red"
        t.add_row(
            f.holdout_agency, f"{f.n_test:,}", str(f.n_positives_test),
            f"[{colour}]{f.pr_auc_model}[/]", _fmt_ci(f.ci, "pr_auc_model"),
            str(f.pr_auc_fwi), str(f.pr_auc_climatology),
            str(f.ci.get("p_model_beats_baseline", "-")),
        )
    p = res.pooled
    t.add_section()
    t.add_row("[bold]POOLED[/]", f"{p['n_test']:,}", str(p["n_positives"]),
              f"[bold]{p['pr_auc_model']}[/]", _fmt_ci(p["ci"], "pr_auc_model"),
              str(p["pr_auc_fwi"]), str(p["pr_auc_climatology"]),
              str(p["ci"].get("p_model_beats_baseline", "-")))
    console.print(t)
    console.print(
        f"train seasons {res.train_years} / test seasons {res.test_years} - "
        f"{res.macro['folds_beating_climatology']}/{res.macro['n_folds']} folds "
        f"beat their own climatology, median lift over fire weather "
        f"{res.macro['median_lift_over_fwi']}x"
    )
    for s in res.skipped:
        console.print(f"[yellow]skipped {s['agency']}: {s['reason']}[/]")


@app.command("ablate-ignition")
def ablate_ignition(
    test_years: list[int] = typer.Option(None, "--test-years"),
    n_boot: int = typer.Option(1000, "--bootstrap"),
):
    """Which blocks of features earn their place, paired on one resample."""
    from .models import ignition as ig_model

    df = pl.read_parquet(config.CURATED / "ignition_table.parquet")
    res = ig_model.ablation(df, test_years=test_years or None, n_boot=n_boot)
    console.print(_ablation_table(
        res, f"Ignition feature ablation - test {res['test_years']}, "
             f"{res['n_positives_test']:,} positives", digits=5))
    console.print(
        "[dim]deltas are bootstrapped paired on the same resampled test rows, "
        "so they are readable where the marginal intervals overlap[/]"
    )


@app.command("fit-final-ignition")
def fit_final_ignition():
    """Refit the ignition model on every season in the panel, for scoring."""
    from .models import ignition as ig_model

    df = pl.read_parquet(config.CURATED / "ignition_table.parquet")
    payload = ig_model.fit_final(df)
    console.print(
        f"final ignition model: [green]{payload['n_train']:,}[/] cell-days, "
        f"[bold]{payload['n_positives']:,}[/] ignitions, seasons "
        f"{payload['train_years']}"
    )
    console.print("[dim]accuracy for this model is inferred from the backtest, "
                  "not measured on the rows it was fitted on[/]")


@app.command("predict-ignition")
def predict_ignition_cmd(
    day: str = typer.Option(None, "--day", help="ISO date; default = the day "
                                                "after the last observation"),
    days: int = typer.Option(1, "--days", help="how many days back to score"),
    top: int = typer.Option(15, "--top"),
):
    """Rank grid cells by the probability of a fire being reported today."""
    from datetime import date as _date

    from . import predict as predict_mod

    target = _date.fromisoformat(day) if day else None
    df = predict_mod.score_ignition(day=target, days=days)
    if df.is_empty():
        console.print("[yellow]no cells scored[/]")
        return

    latest = df["day"].max()
    shown = df.filter(pl.col("day") == latest).head(top)
    t = Table(title=f"Ignition risk {latest} - top {shown.height} of "
                    f"{df.filter(pl.col('day') == latest).height:,} cells",
              header_style="bold")
    for c, j in (("cell", "left"), ("lat, lon", "left"), ("agency", "left"),
                 ("FWI", "right"), ("km to stn", "right"),
                 ("fires/yr", "right"), ("risk", "right")):
        t.add_column(c, justify=j)
    for r in shown.iter_rows(named=True):
        risk = r["risk"]
        colour = "red" if risk >= 0.05 else "yellow" if risk >= 0.01 else "white"
        t.add_row(
            f"{r['cell_x']},{r['cell_y']}",
            f"{r['lat']:.2f}, {r['lon']:.2f}",
            r["agency_code"] or "--",
            f"{r['wx_fwi']:.1f}" if r["wx_fwi"] is not None else "-",
            f"{r['wx_dist_km']:.0f}" if r["wx_dist_km"] is not None else "-",
            str(int(r["ig_n_365d"] or 0)),
            f"[{colour}]{risk:.4f}[/]",
        )
    console.print(t)
    console.print("[dim]risk = P(at least one new fire reported in this 10 km "
                  "cell on this day), on the population scale[/]")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Serve the scored artefacts over HTTP.

    Reads what the CLI has already written; nothing here ingests or refits.
    Needs the optional extra: pip install -e ".[serve]"
    """
    from . import api

    console.print(f"[green]http://{host}:{port}[/]  docs at /docs")
    api.serve(host=host, port=port, reload=reload)


@app.command("report")
def report(
    outdir: str = typer.Option(None, "--outdir", help="default: docs/img/"),
):
    """Render the backtest as charts (light and dark, for the README)."""
    from pathlib import Path

    from . import report as report_mod

    written = report_mod.generate(Path(outdir) if outdir else None)
    console.print(f"[green]{len(written)}[/] figures -> {written[0].parent}")
    for p in written:
        console.print(f"  {p.name}", style="dim")


@app.command("dashboard")
def dashboard_cmd(
    out: str = typer.Option(None, "--out", help="default: docs/dashboard.html"),
):
    """Build the self-contained HTML dashboard from the current artefacts."""
    from pathlib import Path

    from . import dashboard as dash

    path = dash.build(out=Path(out) if out else None)
    console.print(f"[green]{path}[/]  ({path.stat().st_size / 1024:.0f} KB, no external requests)")


@app.command("scrape-ciffc")
def scrape_ciffc():
    """Render and parse today's sitrep page (fallback for `ingest-ciffc`).

    The API path needs no browser and can backfill; this cannot. Kept for the
    day the API changes shape.
    """
    from .sources import ciffc

    try:
        snap = ciffc.snapshot()
    except ciffc.PlaywrightMissing as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1)
    console.print(
        f"sitrep [bold]{snap['sitrep_date']}[/] "
        f"preparedness level [bold]{snap['preparedness_level']}[/], "
        f"{len(snap['new_fires_by_agency'])} agency rows, "
        f"{len(snap['agency_preparedness'])} APL rows"
    )


if __name__ == "__main__":
    app()
