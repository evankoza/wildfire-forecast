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
    aggs = [
        f"{how.upper()}({col}) AS hs_{col}_{how}"
        for col in carried
        for how in measures[col]
    ]
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
    )
    SELECT
        national_fire_id,
        COUNT(*)                                        AS hs_count,
        {agg_sql},
        MIN(dist_km)                                    AS hs_dist_min_km,
        MODE(fuel_group)                                AS hs_fuel_group,
        COUNT(DISTINCT CAST(rep_date AS DATE))          AS hs_active_days,
        DATE_DIFF('hour', MIN(rep_date), t0)            AS hs_detection_lead_hours
    FROM matched
    WHERE dist_km <= {radius_km}
    GROUP BY national_fire_id, t0
    """
    out = con.execute(sql).arrow()
    con.close()
    return pl.from_arrow(out)


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


def build(
    fires: pl.DataFrame,
    hotspots: pl.DataFrame,
    *,
    spec=SPEC,
    data_cutoff: datetime | None = None,
    with_weather: bool = False,
) -> pl.DataFrame:
    """The modelling table."""
    offsets = asof.first_seen(fires)

    if data_cutoff is None:
        data_cutoff = fires["record_start"].max()
    offsets = asof.observable(offsets, data_cutoff, spec.horizon_hours)
    log.info("fires with an observable label horizon: %s", offsets.height)

    known = asof.state_asof(fires, offsets, spec.decision_hours)
    revs = asof.revisions_before(fires, offsets, spec.decision_hours)
    labels = asof.label_at(fires, offsets, spec.horizon_hours)

    base = (
        offsets.join(
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
        )
        .join(revs, on=asof.FIRE_KEY, how="left")
        .join(labels, on=asof.FIRE_KEY, how="inner")
    )

    hs = hotspot_features(offsets, hotspots, radius_km=spec.hotspot_radius_km,
                          decision_hours=spec.decision_hours)
    if hs.height:
        base = base.join(hs, on=asof.FIRE_KEY, how="left")

    if with_weather:
        wx = weather_features(offsets)
        if wx.height:
            base = base.join(wx, on=asof.FIRE_KEY, how="left")

    base = base.with_columns(
        pl.col("t0").dt.month().alias("month"),
        pl.col("t0").dt.ordinal_day().alias("doy"),
        pl.col("t0").dt.hour().alias("t0_hour"),
        pl.col("hs_count").fill_null(0),
        # Growth already visible by decision time.
        (pl.col("size_at_decision").fill_null(0) - pl.col("size_first").fill_null(0))
        .alias("size_growth_to_decision"),
        pl.col("size_at_decision").fill_null(0).log1p().alias("log_size_at_decision"),
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
