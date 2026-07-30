"""NASA FIRMS -- the near-real-time / global leg. Requires a free MAP_KEY.

The Canadian historical backbone of this project comes from CWFIS, which
needs no credentials and ships fire-behaviour-enriched detections. FIRMS
matters for two things CWFIS cannot do:

  1. Latency. FIRMS publishes within ~3 hours of overpass; the CWFIS daily
     roll-up lands the next morning.
  2. Coverage outside Canada, if the model is ever pointed elsewhere.

Get a key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put it in
.env as FIRMS_MAP_KEY. Limit is 5000 transactions / 10 minutes; the area
endpoint accepts a day range of 1-5 per call, so backfill is a date loop.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import polars as pl

from .. import config, http

log = logging.getLogger(__name__)

# west, south, east, north -- Canada.
CANADA_BBOX = "-141,41,-52,84"

SOURCES = (
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
    "MODIS_NRT",
)

MAX_DAY_RANGE = 5


class MissingKeyError(RuntimeError):
    pass


def available() -> bool:
    return bool(config.FIRMS_MAP_KEY)


def fetch_area(
    start: date,
    days: int = MAX_DAY_RANGE,
    *,
    source: str = "VIIRS_NOAA20_NRT",
    bbox: str = CANADA_BBOX,
    force: bool = False,
) -> pl.DataFrame:
    if not available():
        raise MissingKeyError(
            "FIRMS_MAP_KEY is not set. This source is optional -- the Canadian "
            "pipeline runs entirely on CWFIS. Set the key in .env to enable it."
        )
    if not 1 <= days <= MAX_DAY_RANGE:
        raise ValueError(f"FIRMS day range must be 1..{MAX_DAY_RANGE}, got {days}")

    url = (
        f"{config.FIRMS_AREA}/{config.FIRMS_MAP_KEY}/{source}/{bbox}/"
        f"{days}/{start.isoformat()}"
    )
    dest = config.RAW / "firms" / f"{source}_{start:%Y%m%d}_{days}d.csv"

    if not dest.exists() or force:
        http.fetch_to_file(url, dest, conditional=False)

    raw = dest.read_bytes()
    # FIRMS answers errors as plain text with a 200, so sniff the body.
    head = raw[:200].lower()
    if b"invalid" in head or b"error" in head or not raw.strip():
        raise RuntimeError(f"FIRMS returned an error body: {raw[:200]!r}")

    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=5_000, ignore_errors=True)
    if "acq_date" in df.columns and "acq_time" in df.columns:
        df = df.with_columns(
            (
                pl.col("acq_date").cast(pl.Utf8)
                + " "
                + pl.col("acq_time").cast(pl.Utf8).str.zfill(4)
            )
            .str.to_datetime("%Y-%m-%d %H%M", strict=False)
            .alias("acq_datetime")
        )
    return df


def backfill(start: date, end: date, *, source: str = "VIIRS_NOAA20_NRT") -> pl.DataFrame:
    frames, cursor = [], start
    while cursor <= end:
        span = min(MAX_DAY_RANGE, (end - cursor).days + 1)
        try:
            frames.append(fetch_area(cursor, span, source=source))
        except Exception as exc:  # noqa: BLE001
            log.warning("FIRMS %s +%sd failed: %s", cursor, span, exc)
        cursor += timedelta(days=span)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
