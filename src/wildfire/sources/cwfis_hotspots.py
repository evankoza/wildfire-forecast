"""CWFIS satellite hotspots -- the feature source.

NRCan's hotspot feed is considerably richer than raw NASA FIRMS: each
detection already carries the Canadian Forest Fire Behaviour Prediction
outputs computed at that pixel.

  fwi    Fire Weather Index at the pixel
  fuel   FBP fuel type (C1-C7, D1, M1-M4, S1-S3, O1a/b, water, urban...)
  ros    rate of spread (m/min)
  sfc/tfc/bfc  surface / total / crown fuel consumption (kg/m^2)
  hfi    head fire intensity (kW/m)  <- the operational severity number
  estarea  estimated area represented by the detection

Daily files live at downloads/hotspots/YYYYMMDD.csv (rolling, ~1.8 MB/day);
whole seasons are zipped under downloads/hotspots/archive/YYYY_hotspots.zip.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta

import polars as pl

from .. import config, http

log = logging.getLogger(__name__)

SCHEMA = {
    "lat": pl.Float64,
    "lon": pl.Float64,
    "rep_date": pl.Utf8,
    "source": pl.Utf8,
    "sensor": pl.Utf8,
    "fwi": pl.Float64,
    "fuel": pl.Utf8,
    "ros": pl.Float64,
    "sfc": pl.Float64,
    "tfc": pl.Float64,
    "bfc": pl.Float64,
    "hfi": pl.Float64,
    "estarea": pl.Float64,
}

# Non-vegetation detections: industrial heat, flares, water glint. Keeping
# them would teach the model that "hotspot near fire" includes gas plants.
NON_FUEL = ("water", "urban", "unknown", "non-fuel", "vegetated non-fuel")


def fetch_day(day: date, *, force: bool = False) -> pl.DataFrame | None:
    """One rolling daily file. Returns None if the day is off the window."""
    stamp = day.strftime("%Y%m%d")
    dest = config.RAW / "cwfis_hotspots" / f"{stamp}.csv"
    url = f"{config.CWFIS_DOWNLOADS}/hotspots/{stamp}.csv"

    if not dest.exists() or force:
        try:
            http.fetch_to_file(url, dest, conditional=False)
        except Exception as exc:  # noqa: BLE001 - the rolling window has gaps
            log.warning("hotspots %s unavailable: %s", stamp, exc)
            return None

    return _parse(dest.read_bytes())


def fetch_season(year: int, *, force: bool = False) -> pl.DataFrame:
    """A whole season from the archive zip -- far cheaper than 200 day files."""
    dest = config.RAW / "cwfis_hotspots" / f"{year}_hotspots.zip"
    url = f"{config.CWFIS_DOWNLOADS}/hotspots/archive/{year}_hotspots.zip"

    if not dest.exists() or force:
        http.fetch_to_file(url, dest, conditional=False)

    frames = []
    with zipfile.ZipFile(dest) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"{dest.name} contains no CSV: {z.namelist()[:10]}")
        for name in members:
            with z.open(name) as fh:
                frames.append(_parse(fh.read()))
    df = pl.concat([f for f in frames if f is not None and f.height], how="vertical_relaxed")
    log.info("hotspots %s: %s detections", year, df.height)
    return df


def _parse(raw: bytes) -> pl.DataFrame | None:
    if not raw.strip():
        return None
    df = pl.read_csv(
        io.BytesIO(raw),
        infer_schema_length=5_000,
        ignore_errors=True,
    )
    # The daily files ship with a leading space in every header after the first.
    df = df.rename({c: c.strip() for c in df.columns})

    keep = [c for c in SCHEMA if c in df.columns]
    df = df.select(keep)

    casts = [pl.col(c).cast(SCHEMA[c], strict=False) for c in keep if c != "rep_date"]
    df = df.with_columns(casts)

    if "rep_date" in df.columns:
        df = df.with_columns(
            pl.col("rep_date")
            .cast(pl.Utf8)
            .str.replace("T", " ")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
            .alias("rep_date")
        )
    return df


def drop_non_fuel(df: pl.DataFrame) -> pl.DataFrame:
    if "fuel" not in df.columns:
        return df
    return df.filter(
        pl.col("fuel").is_null()
        | ~pl.col("fuel").str.to_lowercase().is_in(NON_FUEL)
    )


def load_seasons(years: list[int], *, force: bool = False) -> pl.DataFrame:
    frames = []
    for y in years:
        try:
            frames.append(fetch_season(y, force=force))
        except Exception as exc:  # noqa: BLE001
            log.warning("season %s archive unavailable (%s); trying daily files", y, exc)
            frames.append(_load_days_for_year(y))

    frames = [f for f in frames if f is not None and f.height]
    if not frames:
        raise RuntimeError(f"no hotspot detections loaded for {years}")

    # The rolling daily files carry two columns the season archives do not
    # (`estarea`, `bfc`), so a naive concat raises on mismatched schemas -- and
    # papering over that with a diagonal concat would be worse: the current
    # season would contribute feature columns that are null for every training
    # season, and the feature set would depend on which source a year happened
    # to come from. Intersect instead, so the schema is the same no matter how
    # a season was fetched.
    common = set(frames[0].columns).intersection(*(set(f.columns) for f in frames[1:]))
    ordered = [c for c in SCHEMA if c in common]
    dropped = sorted(set().union(*(set(f.columns) for f in frames)) - common)
    if dropped:
        log.info("hotspot columns absent from at least one season, dropped: %s", dropped)

    df = pl.concat([f.select(ordered) for f in frames], how="vertical_relaxed")
    df = drop_non_fuel(df)
    dest = config.CURATED / "hotspots.parquet"
    df.write_parquet(dest)
    log.info("wrote %s hotspot detections -> %s", df.height, dest)
    return df


def _load_days_for_year(year: int, *, season_only: bool = True) -> pl.DataFrame:
    """Fallback for the current season, which has no archive zip yet.

    One request per day, so restricted to the fire season by default: outside
    April-October the daily files are nearly empty and the whole point is to
    avoid ~200 round trips for detections that do not exist.
    """
    from ..config import FIRE_SEASON_MONTHS

    frames = []
    d = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today())
    while d <= end:
        if season_only and d.month not in FIRE_SEASON_MONTHS:
            d += timedelta(days=1)
            continue
        got = fetch_day(d)
        if got is not None and got.height:
            frames.append(got)
        d += timedelta(days=1)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
