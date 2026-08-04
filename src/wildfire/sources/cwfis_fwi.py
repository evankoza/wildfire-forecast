"""Station weather and Fire Weather Index observations -- the ignition model's
one *exogenous* input.

The hotspot feed already carries FWI, but only at pixels that were on fire.
Using it to answer "where will a fire start tomorrow" would be circular: the
covariate exists precisely where the outcome already happened, so "no FWI" and
"no fire" would be almost the same column. This feed is the honest
alternative -- noon-local observations from the national weather station
network, taken whether or not anything is burning nearby.

Columns kept (from the file's own header):
  aes            station id
  rep_date       noon local standard time: the FWI System's observation hour
  lat/lon/elev   the station, carried in the file itself
  temp/rh/ws/precip    the four inputs the FWI System is computed from
  ffmc/dmc/dc          fuel moisture codes: fine, duff, deep drought
  isi/bui/fwi/dsr      spread, buildup, the index, the daily severity rating

Three awkward facts about the source, all handled here:

* **The backfill is one very large file per decade.** There is no per-day
  archive before the current season -- `fwi_obs/current/` holds this year
  only. So a decade is streamed to the landing zone once and every season
  after that is a re-parse, not another request.
* **There are two decadal variants and they are not interchangeable.** The
  plain `cwfis_fwi2020s.csv` stops on 2025-01-21 and carries no coordinates;
  `cwfis_fwi2020sv3.0_ll.csv` runs to 2025-08-31, carries lat/lon inline, and
  includes provincial stations the plain file is not licensed to publish. The
  `_ll` variant is the one used. `PLAIN_URL` is kept named rather than deleted
  because the difference is the sort of thing that is rediscovered
  expensively.
* **The archive ends mid-season.** Whatever is downloaded, the last day is a
  hard boundary on the ignition panel, so `load` records the range it actually
  wrote and `features.ignition` clips the panel to it rather than silently
  posing the question on days with no weather.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import polars as pl

from .. import config, http

log = logging.getLogger(__name__)

ARCHIVE_URL = f"{config.CWFIS_DOWNLOADS}/fwi_obs/cwfis_fwi{{decade}}sv3.0_ll.csv"
PLAIN_URL = f"{config.CWFIS_DOWNLOADS}/fwi_obs/cwfis_fwi{{decade}}s.csv"
DAILY_URL = f"{config.CWFIS_DOWNLOADS}/fwi_obs/current/cwfis_fwi_{{stamp}}.csv"
STATION_URL = f"{config.CWFIS_DOWNLOADS}/fwi_obs/cwfis_allstn{{year}}.csv"

MEASURES = ("temp", "rh", "ws", "precip", "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")
KEEP = ("aes", "rep_date", "lat", "lon", "elev", *MEASURES)

# Physically impossible values do appear -- an FWI of 10^4, a relative humidity
# of 999. They are sentinels, not measurements, and a gradient booster has no
# way to tell the difference.
LIMITS = {
    "temp": (-60.0, 50.0),
    "rh": (0.0, 100.0),
    "ws": (0.0, 200.0),
    "precip": (0.0, 500.0),
    "ffmc": (0.0, 101.0),
    "dmc": (0.0, 1000.0),
    "dc": (0.0, 2000.0),
    "isi": (0.0, 200.0),
    "bui": (0.0, 1000.0),
    "fwi": (0.0, 200.0),
    "dsr": (0.0, 1000.0),
}


def _decades(years: list[int]) -> list[int]:
    return sorted({(y // 10) * 10 for y in years})


def fetch_decade(decade: int, *, force: bool = False) -> pl.LazyFrame:
    """Stream one decadal archive to the landing zone and scan it lazily.

    ~670 MB for the 2020s. `http.fetch_to_file` buffers a whole response in
    memory, which is the right trade for a 400 KB sitrep and the wrong one
    here, so this streams to disk instead.
    """
    dest = config.RAW / "cwfis_fwi" / f"cwfis_fwi{decade}sv3.0_ll.csv"
    if not dest.exists() or force:
        http.stream_to_file(ARCHIVE_URL.format(decade=decade), dest)
    return pl.scan_csv(dest, infer_schema_length=10_000, ignore_errors=True)


def fetch_day(day: date, *, force: bool = False) -> pl.DataFrame | None:
    """One day from `fwi_obs/current/`, for the season in progress.

    Returns None when the day is not published; that directory is rebuilt
    periodically, so gaps are normal rather than an error.

    These files are not the archive in miniature. The header is upper-case and
    **repeated as the first data row**, `REPDATE` is a bare `YYYYMMDD` rather
    than a timestamp, every field is space-padded, and there are no
    coordinates -- so the station list supplies those. Parsing them with the
    archive's reader silently yields an empty frame.
    """
    stamp = day.strftime("%Y%m%d")
    dest = config.RAW / "cwfis_fwi" / "current" / f"cwfis_fwi_{stamp}.csv"
    if not dest.exists() or force:
        try:
            http.fetch_to_file(DAILY_URL.format(stamp=stamp), dest, conditional=False)
        except Exception as exc:  # noqa: BLE001 - the window has holes
            log.debug("station fwi %s unavailable: %s", stamp, exc)
            return None
    raw = dest.read_bytes()
    if not raw.strip():
        return None

    df = pl.read_csv(
        io.BytesIO(raw), infer_schema_length=0, ignore_errors=True,
        encoding="utf8-lossy",
    )
    df = df.rename({c: c.strip().lower() for c in df.columns})
    need = ["aes", "repdate", *MEASURES]
    if any(c not in df.columns for c in need):
        log.warning("daily station file %s has an unexpected header", stamp)
        return None

    out = (
        df.select(
            pl.col("aes").cast(pl.Utf8).str.strip_chars(),
            pl.col("repdate").cast(pl.Utf8).str.strip_chars(),
            *[pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
              for c in MEASURES],
        )
        # The repeated header line parses as a row whose date is the literal
        # string "REPDATE"; a strict date parse turns it into a null and this
        # drops it, rather than needing a rule about which row to skip.
        .with_columns(
            pl.col("repdate").str.to_date("%Y%m%d", strict=False).alias("obs_date")
        )
        .drop_nulls(["obs_date"])
        .drop("repdate")
    )
    clipped = [
        pl.when(pl.col(c).is_between(lo, hi)).then(pl.col(c)).otherwise(None).alias(c)
        for c, (lo, hi) in LIMITS.items()
    ]
    out = out.with_columns(clipped)
    if out.is_empty():
        return None

    return out.join(
        stations().select("aes", "lat", "lon", "elev", "prov"), on="aes", how="inner"
    )


def load_current(
    years: list[int], *, force: bool = False, season_only: bool = True
) -> pl.DataFrame:
    """Refresh the season in progress from `fwi_obs/current/`, keeping the rest.

    The decadal archive is republished occasionally and lags by the better part
    of a year, so it cannot serve today. This walks the daily files instead and
    merges them into the curated table by observation year -- the same
    replace-a-season-wholesale shape `ingest-hotspots --merge` uses, and for the
    same reason: the daily files are the authority for a season in progress, so
    a partial overlay would leave stale rows behind alongside fresh ones.

    One request per day, restricted to the fire season by default.
    """
    from ..config import FIRE_SEASON_MONTHS

    frames = []
    for year in sorted(years):
        d = date(year, 1, 1)
        end = min(date(year, 12, 31), date.today())
        while d <= end:
            if season_only and d.month not in FIRE_SEASON_MONTHS:
                d += timedelta(days=1)
                continue
            got = fetch_day(d, force=force)
            if got is not None and got.height:
                frames.append(got)
            d += timedelta(days=1)

    if not frames:
        raise RuntimeError(f"no daily station files available for {years}")
    fresh = pl.concat(frames, how="vertical_relaxed")

    dest = config.CURATED / "station_fwi.parquet"
    if dest.exists():
        existing = pl.read_parquet(dest)
        refreshed = set(fresh["obs_date"].dt.year().unique().to_list())
        kept = existing.filter(~pl.col("obs_date").dt.year().is_in(list(refreshed)))
        log.info("merging %s season(s) into %s: keeping %s rows, replacing %s",
                 len(refreshed), dest.name, kept.height, existing.height - kept.height)
        common = [c for c in kept.columns if c in fresh.columns]
        fresh = pl.concat([kept.select(common), fresh.select(common)],
                          how="vertical_relaxed")

    out = fresh.sort(["obs_date", "aes"])
    out.write_parquet(dest)
    log.info("wrote %s station-days (%s..%s) -> %s",
             f"{out.height:,}", out["obs_date"].min(), out["obs_date"].max(), dest)
    return out


def stations(*, force: bool = False) -> pl.DataFrame:
    """Station coordinates and province, from the most recent published list.

    The `_ll` archive already carries coordinates; this is fetched for `prov`,
    which is what maps a cell to a CIFFC reporting agency. Tries this year
    first and falls back a year at a time, because the current year's list
    appears partway through it.
    """
    today = date.today().year
    last_error: Exception | None = None
    for year in (today, today - 1, today - 2):
        dest = config.RAW / "cwfis_fwi" / f"cwfis_allstn{year}.csv"
        try:
            if not dest.exists() or force:
                http.fetch_to_file(STATION_URL.format(year=year), dest, conditional=False)
            df = pl.read_csv(
                dest, infer_schema_length=10_000, ignore_errors=True,
                encoding="utf8-lossy",
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        # Several columns arrive space-padded, so every cast strips first: a
        # bare `.cast(Float64)` on " 60.34" raises rather than coercing.
        def num(c):
            return pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)

        out = (
            df.select(
                pl.col("aes").cast(pl.Utf8).str.strip_chars(),
                num("lat").alias("lat"),
                num("lon").alias("lon"),
                num("elev").alias("elev"),
                pl.col("prov").cast(pl.Utf8).str.strip_chars().alias("prov"),
            )
            .drop_nulls(["aes", "lat", "lon"])
            # A few ids repeat across instruments at one site; they share
            # coordinates, so keeping the first is not a choice about which
            # reading wins -- observations join on `aes` either way.
            .unique(subset=["aes"], keep="first")
            .sort("aes")
        )
        log.info("station list %s: %s stations", year, out.height)
        return out

    raise RuntimeError(f"no station list could be fetched: {last_error}")


def _clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Parse the timestamp, strip padding, null out impossible values."""
    clipped = [
        pl.when(pl.col(c).is_between(lo, hi)).then(pl.col(c)).otherwise(None).alias(c)
        for c, (lo, hi) in LIMITS.items()
        if c in lf.collect_schema().names()
    ]
    return lf.with_columns(
        pl.col("aes").cast(pl.Utf8).str.strip_chars(),
        pl.col("rep_date")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace("T", " ")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("rep_date"),
        *[
            pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
            for c in MEASURES
            if c in lf.collect_schema().names()
        ],
    ).with_columns(clipped)


def load(
    years: list[int],
    *,
    force: bool = False,
    season_only: bool = True,
) -> pl.DataFrame:
    """Parse the decades covering `years`, attach province, write parquet.

    Filtered to the fire season by default: the ignition model only poses the
    question April-October, and carrying the winter triples a table nothing
    reads.
    """
    from ..config import FIRE_SEASON_MONTHS

    frames = []
    for decade in _decades(years):
        lf = fetch_decade(decade, force=force)
        have = lf.collect_schema().names()
        missing = [c for c in KEEP if c not in have]
        if missing:
            raise RuntimeError(
                f"the {decade}s archive is missing {missing}; it is probably the "
                f"plain variant rather than the _ll one"
            )
        lf = _clean(lf.select(KEEP))
        lf = lf.filter(pl.col("rep_date").dt.year().is_in(years))
        if season_only:
            lf = lf.filter(pl.col("rep_date").dt.month().is_in(FIRE_SEASON_MONTHS))
        frames.append(lf.collect())

    obs = pl.concat([f for f in frames if f.height], how="vertical_relaxed")
    if obs.is_empty():
        raise RuntimeError(f"no station observations for {years}")

    prov = stations(force=force).select("aes", "prov")
    obs = obs.drop_nulls(["rep_date", "lat", "lon"]).join(prov, on="aes", how="left")
    known_prov = obs["prov"].is_not_null().mean()

    # One row per station-day. Duplicates exist where a station reports under
    # two instrument records; neither is more authoritative, so take the mean
    # of what was reported rather than an arbitrary row -- with an explicit
    # sort, so the result cannot depend on scan order.
    obs = (
        obs.with_columns(pl.col("rep_date").dt.date().alias("obs_date"))
        .group_by(["aes", "obs_date"])
        .agg(
            *[pl.col(c).mean().alias(c) for c in MEASURES],
            pl.col("lat").first(),
            pl.col("lon").first(),
            pl.col("elev").first(),
            pl.col("prov").first(),
        )
        .sort(["obs_date", "aes"])
    )

    dest = config.CURATED / "station_fwi.parquet"
    obs.write_parquet(dest)
    log.info(
        "wrote %s station-days (%s stations, %s..%s, %.1f%% with a province) -> %s",
        f"{obs.height:,}", f"{obs['aes'].n_unique():,}",
        obs["obs_date"].min(), obs["obs_date"].max(), 100 * known_prov, dest,
    )
    return obs


def coverage(df: pl.DataFrame) -> pl.DataFrame:
    """Diagnostic: how much of the network actually reported, by season."""
    return (
        df.with_columns(pl.col("obs_date").dt.year().alias("year"))
        .group_by("year")
        .agg(
            n_days=pl.col("obs_date").n_unique(),
            first_day=pl.col("obs_date").min(),
            last_day=pl.col("obs_date").max(),
            n_stations=pl.col("aes").n_unique(),
            n_rows=pl.len(),
            fwi_reported=pl.col("fwi").is_not_null().mean().round(3),
            mean_fwi=pl.col("fwi").mean().round(2),
        )
        .sort("year")
    )
