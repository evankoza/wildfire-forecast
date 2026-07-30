"""Assemble the modelling table.

One row per fire, features as known at T0 + decision_hours, label from
T0 + horizon_hours. The spatial-temporal hotspot join runs in DuckDB: a
bounding-box prefilter followed by an exact haversine test, which is what
keeps a few thousand fires x a few million detections tractable without a
spatial index.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import duckdb
import polars as pl

from .. import config
from ..config import SPEC
from . import asof

log = logging.getLogger(__name__)

# Fuel types collapsed to the groups that behave differently. The raw feed has
# ~30 codes including 50/50 mixes (M2_50), which fragment badly as categories.
FUEL_GROUPS = {
    "C": "conifer",
    "D": "deciduous",
    "M": "mixedwood",
    "S": "slash",
    "O": "grass",
}


def _fuel_group_expr(col: str = "fuel") -> pl.Expr:
    return (
        pl.col(col)
        .str.to_uppercase()
        .str.slice(0, 1)
        .replace_strict(FUEL_GROUPS, default="other")
        .alias("fuel_group")
    )


def hotspot_features(
    fires: pl.DataFrame,
    hotspots: pl.DataFrame,
    *,
    radius_km: float = SPEC.hotspot_radius_km,
    decision_hours: int = SPEC.decision_hours,
) -> pl.DataFrame:
    """Satellite-observed fire behaviour near each fire, up to decision time.

    Window starts 24 h before T0 because a fire is frequently detected from
    orbit before an agency reports it -- that lead time is real signal.
    """
    if hotspots.is_empty() or fires.is_empty():
        return pl.DataFrame({asof.FIRE_KEY: [], "hs_count": []})

    hs = hotspots.with_columns(_fuel_group_expr()).drop_nulls(["lat", "lon", "rep_date"])
    fr = fires.select(asof.FIRE_KEY, "t0", "lat", "lon").drop_nulls(["lat", "lon", "t0"])

    # The season archives and the rolling daily files do not carry identical
    # columns (the archives drop `estarea` and `bfc`), so build the projection
    # from what is actually present rather than assuming a fixed schema.
    available = set(hs.columns)
    measures = {
        "hfi": ("max", "mean"),
        "fwi": ("max", "mean"),
        "ros": ("max", "mean"),
        "sfc": ("mean",),
        "tfc": ("max", "mean"),
        "estarea": ("sum",),
        "bfc": ("mean",),
    }
    carried = [c for c in measures if c in available]

    select_cols = ", ".join(f"h.{c}" for c in carried)

    # Sums are accumulated in DECIMAL, not float64.
    #
    # DuckDB aggregates in parallel and reduces partial sums in whatever order
    # the threads finish. Float addition is not associative, so a mean drifts
    # in its last bits between runs of the same query on the same data. That
    # is physically meaningless -- FWI to 1e-12 is noise -- but a gradient
    # booster is not physical: a handful of values land the other side of a
    # split threshold, and the headline PR-AUC moves by ~0.008 with no change
    # to the data whatsoever.
    #
    # Decimal addition *is* associative, so the sum is identical regardless of
    # reduction order, and the single trailing division is then deterministic
    # too. Forcing `SET threads TO 1` also fixes it but costs several minutes
    # on a full rebuild; this keeps the parallelism.
    # MAX/MIN are order-independent already and left exact.
    DEC = "DECIMAL(18,6)"

    def _agg(col: str, how: str) -> str:
        if how == "mean":
            # SUM/COUNT rather than AVG: AVG promotes to double internally,
            # which is the thing being avoided. COUNT(col) skips nulls, so
            # this matches AVG's semantics exactly.
            expr = (f"CAST(SUM(CAST({col} AS {DEC})) AS DOUBLE) "
                    f"/ NULLIF(COUNT({col}), 0)")
        elif how == "sum":
            expr = f"CAST(SUM(CAST({col} AS {DEC})) AS DOUBLE)"
        else:
            expr = f"{how.upper()}({col})"
        return f"{expr} AS hs_{col}_{how}"

    aggs = [_agg(col, how) for col in carried for how in measures[col]]
    agg_sql = ",\n        ".join(aggs)

    con = duckdb.connect()
    con.register("hs", hs.to_arrow())
    con.register("fr", fr.to_arrow())

    sql = f"""
    WITH win AS (
        SELECT
            f.national_fire_id, f.lat, f.lon, f.t0,
            f.t0 - INTERVAL 24 HOUR                    AS win_start,
            f.t0 + INTERVAL {decision_hours} HOUR      AS win_end,
            {radius_km} / 111.32                       AS dlat,
            {radius_km} / (111.32 * GREATEST(COS(RADIANS(f.lat)), 0.01)) AS dlon
        FROM fr f
    ),
    matched AS (
        SELECT
            w.national_fire_id,
            {select_cols}, h.fuel_group, h.rep_date,
            w.t0,
            6371.0 * 2 * ASIN(SQRT(
                POWER(SIN(RADIANS(h.lat - w.lat) / 2), 2) +
                COS(RADIANS(w.lat)) * COS(RADIANS(h.lat)) *
                POWER(SIN(RADIANS(h.lon - w.lon) / 2), 2)
            )) AS dist_km
        FROM win w
        JOIN hs h
          ON h.rep_date >= w.win_start
         AND h.rep_date <= w.win_end
         AND h.lat BETWEEN w.lat - w.dlat AND w.lat + w.dlat
         AND h.lon BETWEEN w.lon - w.dlon AND w.lon + w.dlon
    ),
    in_radius AS (
        SELECT * FROM matched WHERE dist_km <= {radius_km}
    ),
    -- The dominant fuel type, with ties broken alphabetically rather than by
    -- whichever row the scan happened to reach first. DuckDB's MODE() is
    -- order-dependent on ties, which made this column silently non-reproducible.
    fuel AS (
        SELECT national_fire_id, fuel_group AS hs_fuel_group
        FROM (
            SELECT
                national_fire_id,
                fuel_group,
                ROW_NUMBER() OVER (
                    PARTITION BY national_fire_id
                    ORDER BY COUNT(*) DESC, fuel_group ASC
                ) AS rn
            FROM in_radius
            WHERE fuel_group IS NOT NULL
            GROUP BY national_fire_id, fuel_group
        )
        WHERE rn = 1
    )
    SELECT
        n.national_fire_id,
        COUNT(*)                                        AS hs_count,
        {agg_sql},
        MIN(dist_km)                                    AS hs_dist_min_km,
        ANY_VALUE(f.hs_fuel_group)                      AS hs_fuel_group,
        COUNT(DISTINCT CAST(rep_date AS DATE))          AS hs_active_days,
        DATE_DIFF('hour', MIN(rep_date), n.t0)          AS hs_detection_lead_hours
    FROM in_radius n
    LEFT JOIN fuel f USING (national_fire_id)
    GROUP BY n.national_fire_id, n.t0
    """
    out = con.execute(sql).arrow()
    con.close()
    return pl.from_arrow(out)


PREP_FEATURES = {
    "national_preparedness_level": "ciffc_national_pl",
    "agency_preparedness": "ciffc_agency_pl",
    "prep_hazard": "ciffc_prep_hazard",
    "prep_current_load": "ciffc_prep_current_load",
    "prep_expected_load": "ciffc_prep_expected_load",
    "prep_resource_levels": "ciffc_prep_resource_levels",
    "prep_resource_availability": "ciffc_prep_resource_availability",
    "occurrence_pred_lightning": "ciffc_occurrence_pred_lightning",
    "occurrence_pred_human": "ciffc_occurrence_pred_human",
}


def preparedness_features(
    fires: pl.DataFrame,
    sitreps: pl.DataFrame,
    *,
    decision_hours: int = SPEC.decision_hours,
) -> pl.DataFrame:
    """CIFFC preparedness as it stood at decision time.

    An as-of backward join on `published_at`: the most recent sitrep for that
    fire's agency that had actually gone out by T0 + decision_hours.

    Joining on `sitrep_date` instead would leak, and subtly. The report for
    the 15th is published late *on* the 15th, so a fire first reported that
    morning would pick up a judgement -- including an explicit forecast of
    tomorrow's fire load -- made hours after the decision instant. Publication
    time is carried through ingestion precisely so this join can be honest.

    `ciffc_sitrep_lag_hours` is kept as a feature in its own right: a stale
    sitrep is weaker evidence, and off-season fires have none at all.
    """
    if sitreps.is_empty() or fires.is_empty():
        return pl.DataFrame({asof.FIRE_KEY: []})

    left = (
        fires.select(
            asof.FIRE_KEY,
            "agency_code",
            (pl.col("t0") + pl.duration(hours=decision_hours)).alias("decision_at"),
        )
        .drop_nulls(["decision_at", "agency_code"])
        .sort("decision_at")
    )
    right = (
        sitreps.select(
            "agency_code",
            pl.col("published_at").cast(pl.Datetime("us")),
            *[pl.col(src).alias(dst) for src, dst in PREP_FEATURES.items()],
        )
        .drop_nulls(["published_at", "agency_code"])
        .sort("published_at")
    )

    joined = left.join_asof(
        right,
        left_on="decision_at",
        right_on="published_at",
        by="agency_code",
        strategy="backward",
    )

    return joined.select(
        asof.FIRE_KEY,
        *PREP_FEATURES.values(),
        (
            (pl.col("decision_at") - pl.col("published_at")).dt.total_minutes() / 60.0
        ).alias("ciffc_sitrep_lag_hours"),
    )


def weather_features(fires: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    """Observed ERA5 weather from T0 to decision time, per fire.

    Off by default in the CLI: this is one HTTP request per distinct
    (0.1 deg cell, date window), which is thousands of requests for a full
    season. The hotspot feed already carries FWI at the pixel, so the model
    is not blind without it -- this adds wind/gust/VPD detail that FWI
    compresses away.
    """
    from ..sources import openmeteo

    rows = []
    subset = fires if limit is None else fires.head(limit)
    for rec in subset.iter_rows(named=True):
        lat, lon, t0 = rec.get("lat"), rec.get("lon"), rec.get("t0")
        if lat is None or lon is None or t0 is None:
            continue
        start = t0.date()
        end = (t0 + timedelta(hours=SPEC.decision_hours)).date()
        try:
            wx = openmeteo.fetch_window(lat, lon, start, end)
            summary = openmeteo.summarise(wx)
        except Exception as exc:  # noqa: BLE001
            log.debug("weather failed for %s: %s", rec[asof.FIRE_KEY], exc)
            continue
        if summary:
            rows.append({asof.FIRE_KEY: rec[asof.FIRE_KEY], **summary})

    return pl.DataFrame(rows) if rows else pl.DataFrame({asof.FIRE_KEY: []})


def assemble_features(
    fires: pl.DataFrame,
    offsets: pl.DataFrame,
    hotspots: pl.DataFrame,
    *,
    spec=SPEC,
    with_weather: bool = False,
) -> pl.DataFrame:
    """Everything known about each fire at T0 + decision_hours. No label.

    Deliberately label-free so that `build` (training, where the horizon has
    already passed) and `predict` (serving, where it has not) construct
    features through *one* code path. Two paths would drift, and the model
    would be scored at serving time on a feature distribution it was never
    fitted on -- training/serving skew is the standard way a model that
    backtests well fails in production.
    """
    known = asof.state_asof(fires, offsets, spec.decision_hours)
    revs = asof.revisions_before(fires, offsets, spec.decision_hours)

    base = offsets.join(
        known.select(
            asof.FIRE_KEY,
            pl.col("fire_size").alias("size_at_decision"),
            pl.col("stage_of_control_status").alias("status_at_decision"),
            pl.col("response_type"),
            pl.col("national_fire_cause"),
            pl.col("percent_contained"),
        ),
        on=asof.FIRE_KEY,
        how="inner",
    ).join(revs, on=asof.FIRE_KEY, how="left")

    hs = hotspot_features(offsets, hotspots, radius_km=spec.hotspot_radius_km,
                          decision_hours=spec.decision_hours)
    if hs.height:
        base = base.join(hs, on=asof.FIRE_KEY, how="left")

    sitrep_path = config.CURATED / "ciffc_sitreps.parquet"
    if sitrep_path.exists():
        prep = preparedness_features(offsets, pl.read_parquet(sitrep_path),
                                     decision_hours=spec.decision_hours)
        if prep.height:
            base = base.join(prep, on=asof.FIRE_KEY, how="left")
            log.info("preparedness features joined for %s of %s fires",
                     base["ciffc_national_pl"].drop_nulls().len(), base.height)
    else:
        log.info("no CIFFC sitreps ingested - preparedness features will be absent")

    if with_weather:
        wx = weather_features(offsets)
        if wx.height:
            base = base.join(wx, on=asof.FIRE_KEY, how="left")

    return base.with_columns(
        pl.col("t0").dt.month().alias("month"),
        pl.col("t0").dt.ordinal_day().alias("doy"),
        pl.col("t0").dt.hour().alias("t0_hour"),
        pl.col("hs_count").fill_null(0),
        # Growth already visible by decision time.
        (pl.col("size_at_decision").fill_null(0) - pl.col("size_first").fill_null(0))
        .alias("size_growth_to_decision"),
        pl.col("size_at_decision").fill_null(0).log1p().alias("log_size_at_decision"),
    )


def build(
    fires: pl.DataFrame,
    hotspots: pl.DataFrame,
    *,
    spec=SPEC,
    data_cutoff: datetime | None = None,
    with_weather: bool = False,
) -> pl.DataFrame:
    """The modelling table: features as known at decision time, plus the label."""
    offsets = asof.first_seen(fires)

    if data_cutoff is None:
        data_cutoff = fires["record_start"].max()
    offsets = asof.observable(offsets, data_cutoff, spec.horizon_hours)
    log.info("fires with an observable label horizon: %s", offsets.height)

    base = assemble_features(fires, offsets, hotspots, spec=spec,
                             with_weather=with_weather)
    base = base.join(
        asof.label_at(fires, offsets, spec.horizon_hours),
        on=asof.FIRE_KEY,
        how="inner",
    )

    # ---- label -----------------------------------------------------------
    # A fire "escalates" if by the horizon it is at least `size_threshold_ha`.
    # Fires already over the threshold at decision time are excluded: the
    # question is whether a fire *becomes* large, and leaving them in lets the
    # model score a trivially-true carry-over as a win.
    base = base.with_columns(
        (pl.col("size_at_horizon") >= spec.size_threshold_ha)
        .cast(pl.Int8)
        .alias("escalated"),
        (pl.col("status_at_horizon") == "OC").cast(pl.Int8).alias("still_out_of_control"),
    )
    base = base.filter(
        pl.col("size_at_horizon").is_not_null()
        & (pl.col("size_at_decision").fill_null(0) < spec.size_threshold_ha)
    )

    dest = config.CURATED / "modelling_table.parquet"
    base.write_parquet(dest)
    log.info("modelling table: %s rows, %s escalations (%.2f%%) -> %s",
             base.height, int(base["escalated"].sum()),
             100 * base["escalated"].mean() if base.height else 0, dest)
    return base
