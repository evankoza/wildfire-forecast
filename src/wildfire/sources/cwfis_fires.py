"""Reported fires from the CWFIF national GeoServer -- the label source.

This layer is *bitemporal*, which is the single most useful property in the
whole project. Each fire (`national_fire_id`) has many rows; each row carries
`record_start` / `record_end`, the window during which that row was the
system's current belief about the fire.

That means we can reconstruct exactly what was known about a fire at any past
instant, and therefore build features that are point-in-time correct by
construction rather than by careful bookkeeping on our side.

Column notes (from the layer's own CSV header):
  national_fire_id        stable key, e.g. 2026_AB_SWF-054-2026
  fire_size               hectares (running estimate; -1 / blank = unknown)
  stage_of_control_status OC=out of control, BH=being held,
                          UC=under control, EX=extinguished/out
  national_fire_cause     N=natural, H=human, U=undetermined
  response_type           FUL=full, MOD=modified, MON=monitored
  status_date             agency's timestamp for the status
  record_start/_end       transaction time -- when *we* could have known it
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import polars as pl

from .. import config, http

log = logging.getLogger(__name__)

LAYER = "public:cwfif_national_reportedfires"

# GeoServer caps this layer at 10k features per response regardless of what
# `count` asks for, so the page size must match the cap exactly -- otherwise
# a short page reads as "end of data" and silently truncates the pull.
PAGE = 10_000

# Everything the layer gives us, with the types we want downstream.
SCHEMA_OVERRIDES = {
    "id": pl.Int64,
    "fire_size": pl.Float64,
    "percent_contained": pl.Float64,
    "severity_nearest_dsr": pl.Float64,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "fire_year": pl.Int32,
    "status_year": pl.Int32,
}

_TIMESTAMP_COLS = ("situation_report_date", "status_date", "record_start", "record_end")


def _wfs_url(cql: str, start_index: int, count: int) -> str:
    from urllib.parse import urlencode

    params = {
        "service": "WFS",
        "version": "2.0.1",
        "request": "GetFeature",
        "outputFormat": "csv",
        "typeName": LAYER,
        "count": count,
        "startIndex": start_index,
        "sortBy": "id",  # stable order is required for correct paging
        "CQL_FILTER": cql,
    }
    return f"{config.CWFIF_WFS}?{urlencode(params)}"


def fetch_year(year: int, *, force: bool = False) -> pl.DataFrame:
    """Pull every bitemporal row for fires of a given `fire_year`.

    Pages through the WFS. Raw CSV pages are kept in the landing zone so a
    re-parse never costs another request.
    """
    out_dir = config.RAW / "cwfis_reportedfires" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pl.DataFrame] = []
    start = 0
    while True:
        page_file = out_dir / f"page_{start:07d}.csv"
        url = _wfs_url(f"fire_year={year}", start, PAGE)

        if page_file.exists() and not force:
            raw = page_file.read_bytes()
        else:
            http.fetch_to_file(url, page_file, conditional=False)
            raw = page_file.read_bytes()

        if raw.lstrip().startswith(b"<"):
            raise RuntimeError(
                f"WFS returned XML (an exception report) for {year} at offset "
                f"{start}. First 300 bytes: {raw[:300]!r}"
            )

        df = pl.read_csv(
            io.BytesIO(raw),
            schema_overrides=SCHEMA_OVERRIDES,
            infer_schema_length=10_000,
            try_parse_dates=False,
            ignore_errors=True,
        )
        if df.height == 0:
            page_file.unlink(missing_ok=True)  # don't cache an empty tail page
            break

        frames.append(df)
        log.info("cwfis reportedfires %s: +%s rows (offset %s)", year, df.height, start)
        if df.height < PAGE:
            break
        start += PAGE

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames, how="vertical_relaxed")
    return _normalise(df)


# A revision's transaction time has to sit inside a plausible window around
# the season it belongs to. A handful of rows do not: 2023 fires carrying a
# record_start in 2011, which is not a late correction but a bad timestamp.
#
# They matter out of all proportion to their number. T0 is `min(record_start)`
# per fire, so one row dated twelve years early moves that fire's whole
# decision instant, and every as-of feature is then computed from a window
# that ended before the fire existed.
#
# The window is deliberately loose: from the start of the *previous* calendar
# year (a fire discovered in late December can be filed against the next fire
# year, and some agencies backfill early) to the end of the year after the
# season. Anything outside that is a data fault, not a reporting quirk.
RECORD_START_LOOKBACK_YEARS = 1
RECORD_START_LOOKAHEAD_YEARS = 1


def quarantine_record_start(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split off revision rows whose `record_start` cannot belong to their season.

    Returns `(kept, quarantined)`. Nothing is deleted -- the rejected rows are
    written alongside the curated table so the count can be audited and the
    rule argued with, rather than becoming a filter nobody can see.
    """
    if "record_start" not in df.columns or "fire_year" not in df.columns:
        return df, df.head(0)

    lo = pl.date(pl.col("fire_year") - RECORD_START_LOOKBACK_YEARS, 1, 1)
    hi = pl.date(pl.col("fire_year") + RECORD_START_LOOKAHEAD_YEARS, 12, 31)
    plausible = (
        pl.col("record_start").is_null()
        | pl.col("fire_year").is_null()
        | pl.col("record_start").dt.date().is_between(lo, hi)
    )
    return df.filter(plausible), df.filter(~plausible)


def _normalise(df: pl.DataFrame) -> pl.DataFrame:
    """Parse timestamps, clean sentinels, drop the projected geometry blob."""
    exprs = []
    for c in _TIMESTAMP_COLS:
        if c in df.columns:
            exprs.append(
                pl.col(c)
                .str.replace("T", " ")
                .str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
                .alias(c)
            )
    df = df.with_columns(exprs)

    # -1 is this feed's "not reported" sentinel; keep it out of the maths.
    for c in ("fire_size", "percent_contained", "severity_nearest_dsr"):
        if c in df.columns:
            df = df.with_columns(
                pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
            )

    drop = [c for c in ("geometry", "FID") if c in df.columns]
    return df.drop(drop)


def load(years: list[int], *, force: bool = False) -> pl.DataFrame:
    frames = [fetch_year(y, force=force) for y in years]
    frames = [f for f in frames if f.height]
    if not frames:
        raise RuntimeError(f"no reported-fire rows returned for years {years}")
    df = pl.concat(frames, how="vertical_relaxed")

    df, rejected = quarantine_record_start(df)
    if rejected.height:
        qdest = config.CURATED / "reported_fires_quarantine.parquet"
        rejected.write_parquet(qdest)
        log.warning(
            "quarantined %s of %s revision rows on an implausible record_start "
            "(%s fires affected) -> %s",
            rejected.height, rejected.height + df.height,
            rejected["national_fire_id"].n_unique(), qdest.name,
        )

    dest = config.CURATED / "reported_fires.parquet"
    df.write_parquet(dest)
    log.info("wrote %s rows -> %s", df.height, dest)
    return df


def bitemporal_depth(df: pl.DataFrame) -> pl.DataFrame:
    """Diagnostic: how much genuine revision history does each year carry?

    Backfilled years collapse to ~1 row per fire, which silently destroys the
    "what did we know at T?" premise. Check this before trusting a training
    window -- a year with mean_rows_per_fire near 1.0 cannot support the
    escalation label at all.
    """
    return (
        df.group_by("fire_year")
        .agg(
            n_rows=pl.len(),
            n_fires=pl.col("national_fire_id").n_unique(),
            first_record=pl.col("record_start").min(),
            last_record=pl.col("record_start").max(),
        )
        .with_columns(
            (pl.col("n_rows") / pl.col("n_fires")).round(2).alias("rows_per_fire")
        )
        .sort("fire_year")
    )
