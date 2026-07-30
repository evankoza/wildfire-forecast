"""ERA5 reanalysis weather from Open-Meteo. No API key, ~10k requests/day.

Used to attach observed weather to a fire's location for the window between
ignition report and the decision time.

IMPORTANT -- leakage: ERA5 is *reanalysis*, i.e. the best retrospective
estimate. It is legitimate to use it for the window that has already elapsed
at decision time (T0 .. T_decision), because a forecaster standing at T_d
would have had observations for that period. It is NOT legitimate to use it
for T_d .. T_horizon: at T_d that period is the future, and only a forecast
existed. `fetch_window` therefore refuses to look past `until`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import polars as pl

from .. import config, http

log = logging.getLogger(__name__)

HOURLY = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "vapour_pressure_deficit",
)

# Open-Meteo rounds coordinates to its grid anyway, so snapping to ~0.1 deg
# lets many fires share one cached request.
GRID = 0.1


def snap(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat / GRID) * GRID, round(lon / GRID) * GRID)


def fetch_window(
    lat: float,
    lon: float,
    start: date,
    until: date,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Hourly ERA5 for one grid cell over [start, until] inclusive."""
    if until < start:
        raise ValueError(f"until ({until}) precedes start ({start})")

    slat, slon = snap(lat, lon)
    key = f"{slat:+.1f}_{slon:+.1f}_{start:%Y%m%d}_{until:%Y%m%d}"
    dest = config.RAW / "openmeteo" / f"{key}.json"

    if not dest.exists() or force:
        from urllib.parse import urlencode

        url = f"{config.OPENMETEO_ARCHIVE}?" + urlencode(
            {
                "latitude": f"{slat:.4f}",
                "longitude": f"{slon:.4f}",
                "start_date": start.isoformat(),
                "end_date": until.isoformat(),
                "hourly": ",".join(HOURLY),
                "timezone": "UTC",
            }
        )
        http.fetch_to_file(url, dest, conditional=False)

    import json

    payload = json.loads(dest.read_text())
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        return pl.DataFrame()

    df = pl.DataFrame(hourly).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", strict=False)
    )
    return df.with_columns(
        pl.lit(slat).alias("grid_lat"), pl.lit(slon).alias("grid_lon")
    )


def summarise(df: pl.DataFrame, prefix: str = "wx") -> dict[str, float | None]:
    """Collapse an hourly window into the handful of numbers fire behaviour cares about."""
    if df.is_empty():
        return {}

    def agg(col: str, how: str):
        if col not in df.columns:
            return None
        s = df[col].drop_nulls()
        if s.is_empty():
            return None
        return {"max": s.max, "min": s.min, "mean": s.mean, "sum": s.sum}[how]()

    out = {
        f"{prefix}_temp_max": agg("temperature_2m", "max"),
        f"{prefix}_temp_mean": agg("temperature_2m", "mean"),
        f"{prefix}_rh_min": agg("relative_humidity_2m", "min"),
        f"{prefix}_rh_mean": agg("relative_humidity_2m", "mean"),
        f"{prefix}_wind_max": agg("wind_speed_10m", "max"),
        f"{prefix}_wind_mean": agg("wind_speed_10m", "mean"),
        f"{prefix}_gust_max": agg("wind_gusts_10m", "max"),
        f"{prefix}_precip_sum": agg("precipitation", "sum"),
        f"{prefix}_vpd_max": agg("vapour_pressure_deficit", "max"),
    }
    # Dry-and-windy hours: the classic blow-up combination.
    if {"relative_humidity_2m", "wind_speed_10m"} <= set(df.columns):
        crossover = df.filter(
            (pl.col("relative_humidity_2m") < 30) & (pl.col("wind_speed_10m") > 15)
        ).height
        out[f"{prefix}_crossover_hours"] = float(crossover)
    return out
