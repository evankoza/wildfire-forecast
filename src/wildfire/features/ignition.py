"""Assemble the ignition panel: one row per grid cell per day.

The escalation model asks "this fire exists -- will it get big". This one asks
the question one step earlier: **standing at midnight, will a new fire be
reported anywhere in this 10 km cell today?**

Four things make that a different problem rather than the same problem on a
different key, and each of them is a decision in this file:

1. **The unit has to be invented.** A fire brings its own coordinates; a
   non-event does not. `grid.py` defines the cell; the day is a UTC calendar
   day and the decision instant is its 00:00.

2. **The domain has to be bounded.** Canada is about a million 10 km cells and
   almost all of them are tundra, ice or open water where no fire has ever
   been reported. Posing the question there would hand the model a million
   free negatives a day and make every metric meaningless. The study area is
   therefore the cells that showed *any* fire activity -- a report or a
   satellite detection -- during the **training** seasons only. Defining it
   from all seasons would be a look-ahead: the test season's own fires would
   be telling us where to look. What it costs is stated rather than hidden:
   `build_panel` reports the share of test-season ignitions that fall outside
   the study area and are therefore unreachable.

3. **Negatives have to be sampled.** About one cell-day in two thousand
   carries an ignition. Every negative is kept in the *evaluation* through a
   sample weight rather than by materialising 12 million rows a season -- see
   `models/ignition.py`, where the weight reconstructs the population PR curve
   and the prior correction puts the probabilities back on the true scale.

4. **The fire-weather covariate cannot come from the hotspot feed.** The
   hotspot rows carry FWI, but only where something was already burning: a
   cell with no detection has no FWI, and "no FWI" would be almost perfectly
   collinear with "no fire". That is circular. So fire weather is interpolated
   from the *station* network (`sources/cwfis_fwi.py`), which observes whether
   or not anything is alight.

Everything a row can see is stamped strictly before its decision instant.
Station observations are noon-local on the *previous* day, hotspot and
ignition history windows end at `day - 1`, and the CIFFC join is the same
as-of backward join on `published_at` that the escalation table uses.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import duckdb
import numpy as np
import polars as pl

from .. import config
from ..config import FIRE_SEASON_MONTHS
from . import asof, grid

log = logging.getLogger(__name__)

# One row per cell-day; the label is "at least one new fire report today".
TARGET = "ignited"

# Fraction of the non-event cell-days kept. Chosen so the panel lands in the
# low hundreds of thousands of rows with roughly twenty negatives per
# positive: enough that the trees are not fitting the rare class against a
# handful of quiet days, small enough to refit in seconds. The rate is written
# into the artefact and into the fitted model, because every probability the
# model emits has to be divided back out by it and a mismatch would not raise.
NEG_RATE = 0.03

# Inverse-distance interpolation of the station field. Squared weights, and a
# cap: a station 400 km away is not evidence about this cell's fine fuel
# moisture, and including it would quietly turn the feature into a national
# average in the parts of the north where the network is thinnest.
IDW_POWER = 2.0
IDW_MAX_KM = 300.0
IDW_K = 12

# Trailing windows, in days, ending the day before the decision instant.
HS_WINDOWS = (1, 3, 7)
IG_WINDOWS = (7, 30, 365)

# `hs_days_since` / `ig_days_since` are censored at these horizons rather than
# unbounded. Past a couple of months "when did this cell last burn" stops
# being a state of the fuel and starts being the cell's climatology, which
# `ig_n_365d` already carries -- and the wider the window, the more cell-days
# the range join has to touch. Censoring is visible in the data as a null,
# which is the truthful encoding of "not within the window we looked at".
HS_HISTORY_DAYS = 60
IG_HISTORY_DAYS = 365



def season_days(years: list[int], *, months=FIRE_SEASON_MONTHS) -> list[date]:
    """Every fire-season day in the given seasons."""
    out: list[date] = []
    for y in sorted(years):
        d = date(y, 1, 1)
        end = date(y, 12, 31)
        while d <= end:
            if d.month in months:
                out.append(d)
            d += timedelta(days=1)
    return out


def ignition_events(fires: pl.DataFrame) -> pl.DataFrame:
    """One row per (cell, day) on which at least one new fire entered the feed.

    T0 is `min(record_start)` -- transaction time, when the national system
    first held the fire -- which is exactly what a forecaster at midnight
    could later be scored against. The agency's own `status_date` would be a
    better estimate of when the fire actually started and a worse definition
    of the target: it is frequently backdated after the fact, so a model
    trained against it would be predicting information that did not exist.
    """
    offsets = asof.first_seen(fires)
    ev = grid.with_cells(
        offsets.select(asof.FIRE_KEY, "t0", "lat", "lon", "agency_code")
        .drop_nulls(["t0"])
    )
    return (
        ev.with_columns(pl.col("t0").dt.date().alias("day"))
        .group_by(["cell_x", "cell_y", "day"])
        .agg(
            n_ignitions=pl.len(),
            agency_code=pl.col("agency_code").drop_nulls().sort().first(),
        )
        .sort(["day", "cell_x", "cell_y"])
    )


def cell_agency(cells: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Attach a reporting agency to each cell, from two sources in order.

    1. **The fire feed itself**, for any cell that has ever had a fire reported
       in it. That is authoritative -- it is the agency that filed the report.
       Parks Canada is skipped: PC is federal parkland scattered inside
       provinces, so a PC fire says nothing about which agency covers the
       ground around it.
    2. **The provincial outline**, for everything else.

    A cell that neither source can place is outside Canada, and `study_area`
    drops it. That is the country filter: without it the domain inherits the
    hotspot feed's continental footprint.
    """
    fire_agency = (
        events.filter(pl.col("agency_code").is_not_null() & (pl.col("agency_code") != "PC"))
        .group_by(["cell_x", "cell_y", "agency_code"])
        .agg(pl.len().alias("n"))
        # Deterministic: most reports win, ties alphabetically, never scan order.
        .sort(["cell_x", "cell_y", "n", "agency_code"], descending=[False, False, True, False])
        .unique(subset=["cell_x", "cell_y"], keep="first", maintain_order=True)
        .select("cell_x", "cell_y", pl.col("agency_code").alias("_feed_agency"))
    )

    placed = grid.with_centres(cells.select("cell_x", "cell_y").unique().sort(["cell_x", "cell_y"]))
    prov = grid.province_of(placed["lat"].to_numpy(), placed["lon"].to_numpy())
    placed = placed.with_columns(
        pl.Series("_polygon_agency", [p if p is not None else None for p in prov], dtype=pl.Utf8)
    )
    return (
        placed.join(fire_agency, on=["cell_x", "cell_y"], how="left")
        .with_columns(
            pl.coalesce("_feed_agency", "_polygon_agency").alias("agency_code")
        )
        .select("cell_x", "cell_y", "lat", "lon", "agency_code")
    )


def study_area(
    events: pl.DataFrame,
    hotspots: pl.DataFrame,
    *,
    years: list[int],
) -> pl.DataFrame:
    """Cells inside Canada with any fire activity during `years`.

    `years` should be the *training* seasons. Widening it to include the test
    season is the natural mistake and it is a real look-ahead -- the domain
    would then be drawn around fires the model is about to be asked to
    forecast.

    The country filter is not a detail. The CWFIS hotspot feed covers North
    America, so an activity-only domain reaches to 25 N; two fifths of the
    resulting cell-days sat in the United States, where a Canadian fire report
    is impossible by construction. They are guaranteed negatives, they dilute
    every rate in the table, and because the nearest Canadian station to
    Florida is in Ontario they all arrived labelled `ON`.
    """
    frames = [
        events.filter(pl.col("day").dt.year().is_in(years)).select("cell_x", "cell_y")
    ]
    if not hotspots.is_empty():
        hs = grid.with_cells(
            hotspots.select("lat", "lon", "rep_date").drop_nulls(["rep_date"])
        )
        frames.append(
            hs.filter(pl.col("rep_date").dt.year().is_in(years)).select("cell_x", "cell_y")
        )
    active = pl.concat(frames, how="vertical").unique()

    placed = cell_agency(active, events)
    inside = placed.filter(pl.col("agency_code").is_not_null())
    log.info(
        "study area: %s active cells, %s inside Canada",
        f"{active.height:,}", f"{inside.height:,}",
    )
    return inside.select("cell_x", "cell_y", "agency_code").sort(["cell_x", "cell_y"])


def _sample_panel(
    area: pl.DataFrame,
    events: pl.DataFrame,
    days: list[date],
    *,
    neg_rate: float,
    seed: int,
) -> pl.DataFrame:
    """Every positive cell-day in the domain, plus a random sample of the rest.

    Negatives are drawn per day without replacement, so the sample is a clean
    Bernoulli-by-day design rather than "the first N rows of a cross product"
    -- which would correlate the sample with the cell ordering and, because
    cell ids are spatial, with geography.

    The cross product is never materialised. 57,000 cells x 214 days x three
    seasons is 36 million rows to build and throw away 99% of.
    """
    rng = np.random.default_rng(seed)
    cells_x = area["cell_x"].to_numpy()
    cells_y = area["cell_y"].to_numpy()
    n_cells = len(cells_x)
    per_day = int(round(neg_rate * n_cells))
    if per_day < 1:
        raise RuntimeError(f"neg_rate {neg_rate} samples zero cells from {n_cells}")

    day_arr = np.array(days, dtype="datetime64[D]")
    day_idx, xs, ys = [], [], []
    for i in range(len(days)):
        idx = rng.choice(n_cells, size=per_day, replace=False)
        xs.append(cells_x[idx])
        ys.append(cells_y[idx])
        day_idx.append(np.full(per_day, i, dtype=np.int64))

    sampled = pl.DataFrame(
        {
            "cell_x": pl.Series(np.concatenate(xs), dtype=pl.Int32),
            "cell_y": pl.Series(np.concatenate(ys), dtype=pl.Int32),
            "day": pl.Series(day_arr[np.concatenate(day_idx)], dtype=pl.Date),
        }
    ).unique(subset=["cell_x", "cell_y", "day"])

    day_set = pl.DataFrame({"day": pl.Series(days, dtype=pl.Date)})
    positives = (
        events.join(area, on=["cell_x", "cell_y"], how="semi")
        .join(day_set, on="day", how="semi")
        .select("cell_x", "cell_y", "day", "n_ignitions")
    )

    # A sampled cell-day that turns out to be an ignition is not a negative;
    # drop it from the sample and keep the positive, or the same row would
    # appear twice with two different labels.
    negatives = sampled.join(
        positives.select("cell_x", "cell_y", "day"),
        on=["cell_x", "cell_y", "day"],
        how="anti",
    ).with_columns(pl.lit(0, dtype=pl.UInt32).alias("n_ignitions"))

    panel = pl.concat(
        [positives.with_columns(pl.col("n_ignitions").cast(pl.UInt32)), negatives],
        how="vertical",
    ).with_columns(
        (pl.col("n_ignitions") > 0).cast(pl.Int8).alias(TARGET),
    )
    return panel.join(area, on=["cell_x", "cell_y"], how="left").sort(
        ["day", "cell_x", "cell_y"]
    )


def _register_ring(con, panel: pl.DataFrame) -> None:
    """Materialise the panel expanded over the eight surrounding cells.

    The obvious way to write a ring feature is `JOIN ... ON h.cell_x =
    p.cell_x + o.ox` against a cross-joined offset table. Do not: the join key
    is then an expression over two relations, which DuckDB cannot hash, so it
    falls back to a nested loop over the whole activity table. On this panel
    that turns a four-second query into one that had not finished in twenty
    minutes.

    Computing the eight neighbour keys into a real table first makes the join
    a plain equality on materialised columns, and the planner hashes it.
    """
    ring = pl.concat(
        [
            panel.select(
                "pid",
                (pl.col("cell_x") + ox).cast(pl.Int32).alias("cell_x"),
                (pl.col("cell_y") + oy).cast(pl.Int32).alias("cell_y"),
                "day",
            )
            for ox in (-1, 0, 1)
            for oy in (-1, 0, 1)
            if not (ox == 0 and oy == 0)
        ],
        how="vertical",
    )
    con.register("ring", ring.to_arrow())


def _hotspot_history(panel: pl.DataFrame, hotspots: pl.DataFrame) -> pl.DataFrame:
    """Satellite fire activity in the cell and its ring, before the decision.

    Two radii, deliberately disjoint: the cell itself, and the eight cells
    around it (~10-15 km out). A fire burning next door is the strongest
    single hint that this cell is about to be reported -- large fires spread
    across cell boundaries and are then reported as new fires -- and keeping
    the ring separate from the cell means the model can tell "already burning
    here" from "burning nearby", which are operationally different situations.
    """
    empty = panel.select("pid").with_columns(
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in _HS_COLUMNS]
    )
    if hotspots.is_empty():
        return empty

    hs = grid.with_cells(
        hotspots.select("lat", "lon", "rep_date", "fwi", "hfi", "ros")
        .drop_nulls(["rep_date"])
    )
    hs_day = (
        hs.with_columns(pl.col("rep_date").dt.date().alias("d"))
        .group_by(["cell_x", "cell_y", "d"])
        .agg(
            n=pl.len(),
            fwi_max=pl.col("fwi").max(),
            hfi_max=pl.col("hfi").max(),
            ros_max=pl.col("ros").max(),
        )
    )
    if hs_day.is_empty():
        return empty

    con = duckdb.connect()
    con.register("panel", panel.select("pid", "cell_x", "cell_y", "day").to_arrow())
    con.register("hs_day", hs_day.to_arrow())
    _register_ring(con, panel)

    windows = ",\n            ".join(
        f"""COALESCE(SUM(h.n) FILTER (WHERE h.d > p.day - {w} - 1), 0) AS hs_n_{w}d,
            MAX(h.fwi_max) FILTER (WHERE h.d > p.day - {w} - 1) AS hs_fwi_max_{w}d,
            MAX(h.hfi_max) FILTER (WHERE h.d > p.day - {w} - 1) AS hs_hfi_max_{w}d,
            MAX(h.ros_max) FILTER (WHERE h.d > p.day - {w} - 1) AS hs_ros_max_{w}d"""
        for w in HS_WINDOWS
    )
    ring_windows = ",\n            ".join(
        f"""COALESCE(SUM(h.n) FILTER (WHERE h.d > p.day - {w} - 1), 0) AS hs_ring_n_{w}d,
            MAX(h.hfi_max) FILTER (WHERE h.d > p.day - {w} - 1) AS hs_ring_hfi_max_{w}d"""
        for w in HS_WINDOWS
    )

    cell_sql = f"""
    SELECT p.pid,
        {windows},
        MIN(DATE_DIFF('day', h.d, p.day)) AS hs_days_since
    FROM panel p
    JOIN hs_day h
      ON h.cell_x = p.cell_x AND h.cell_y = p.cell_y
     AND h.d < p.day AND h.d >= p.day - {HS_HISTORY_DAYS}
    GROUP BY p.pid
    """
    ring_sql = f"""
    SELECT p.pid,
        {ring_windows}
    FROM ring p
    JOIN hs_day h
      ON h.cell_x = p.cell_x AND h.cell_y = p.cell_y
     AND h.d < p.day AND h.d >= p.day - {max(HS_WINDOWS)}
    GROUP BY p.pid
    """
    cell = con.execute(cell_sql).pl()
    ring = con.execute(ring_sql).pl()
    con.close()

    out = panel.select("pid").join(cell, on="pid", how="left").join(ring, on="pid", how="left")
    # A cell with no detection in the window has a count of zero, not a missing
    # count -- the absence is observed. `hs_days_since` really is missing, and
    # is left so: LightGBM routes it, and filling it with a sentinel would put
    # "never burned" on the same axis as "burned a long time ago".
    # DuckDB widens SUM over an unsigned integer to HUGEINT, which arrives as a
    # pandas `object` column of Decimals and which LightGBM refuses outright.
    counts = [c for c in out.columns if c.startswith(("hs_n_", "hs_ring_n_"))]
    return out.with_columns(
        [pl.col(c).cast(pl.Int64).fill_null(0) for c in counts]
    )


_HS_COLUMNS = [
    *[f"hs_{stat}_{w}d" for w in HS_WINDOWS for stat in ("n", "fwi_max", "hfi_max", "ros_max")],
    *[f"hs_ring_{stat}_{w}d" for w in HS_WINDOWS for stat in ("n", "hfi_max")],
    "hs_days_since",
]


def _ignition_history(panel: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """How often this cell and its ring have ignited before today.

    The rolling 365-day count is the model's climatology, and it is rolling
    rather than fixed on purpose: a per-cell rate estimated over the whole
    record would be computed partly from the test season. The cost is that
    rows in the first modelled season see a short history, which is real and
    shows up as a weaker feature there rather than as a hidden advantage
    later.
    """
    con = duckdb.connect()
    con.register("panel", panel.select("pid", "cell_x", "cell_y", "day").to_arrow())
    con.register("ev", events.select("cell_x", "cell_y", "day", "n_ignitions").to_arrow())
    _register_ring(con, panel)

    windows = ",\n        ".join(
        f"COALESCE(SUM(e.n_ignitions) FILTER (WHERE e.day > p.day - {w} - 1), 0) "
        f"AS ig_n_{w}d"
        for w in IG_WINDOWS
    )
    cell_sql = f"""
    SELECT p.pid,
        {windows},
        MIN(DATE_DIFF('day', e.day, p.day)) AS ig_days_since
    FROM panel p
    JOIN ev e
      ON e.cell_x = p.cell_x AND e.cell_y = p.cell_y
     AND e.day < p.day AND e.day >= p.day - {IG_HISTORY_DAYS}
    GROUP BY p.pid
    """
    ring_sql = f"""
    SELECT p.pid,
        COALESCE(SUM(e.n_ignitions) FILTER (WHERE e.day > p.day - 8), 0) AS ig_ring_n_7d,
        COALESCE(SUM(e.n_ignitions) FILTER (WHERE e.day > p.day - 366), 0) AS ig_ring_n_365d
    FROM ring p
    JOIN ev e
      ON e.cell_x = p.cell_x AND e.cell_y = p.cell_y
     AND e.day < p.day AND e.day >= p.day - {IG_HISTORY_DAYS}
    GROUP BY p.pid
    """
    cell = con.execute(cell_sql).pl()
    ring = con.execute(ring_sql).pl()
    con.close()

    out = panel.select("pid").join(cell, on="pid", how="left").join(ring, on="pid", how="left")
    counts = [c for c in out.columns if c.startswith(("ig_n_", "ig_ring_n_"))]
    return out.with_columns(
        [pl.col(c).cast(pl.Int64).fill_null(0) for c in counts]
    )


def _station_weather(
    panel: pl.DataFrame,
    neighbours: pl.DataFrame,
    obs: pl.DataFrame,
) -> pl.DataFrame:
    """Inverse-distance interpolation of the station field to each cell centre.

    The observation used is the previous day's, because the FWI System's daily
    reading is taken at noon local standard time: at the 00:00 decision
    instant today's has not been made yet. That is a 12-36 hour lag and it is
    the honest one -- an operational run at dawn would face exactly the same
    lag.

    `wx_n_stations` and `wx_dist_km` come out alongside the values so a thin
    interpolation is legible rather than silent. In the far north a cell can be
    250 km from its nearest reporting station, and an FWI carried that far is a
    regional average wearing a local label.
    """
    from ..sources.cwfis_fwi import MEASURES

    if obs.is_empty():
        return panel.select("pid").with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(f"wx_{m}") for m in MEASURES],
            pl.lit(0, dtype=pl.Int64).alias("wx_n_stations"),
            pl.lit(None, dtype=pl.Float64).alias("wx_dist_km"),
        )

    con = duckdb.connect()
    con.register("panel", panel.select("pid", "cell_x", "cell_y", "day").to_arrow())
    con.register("nb", neighbours.to_arrow())
    con.register("obs", obs.to_arrow())

    # Squared inverse distance, floored at 1 km so a station inside the cell
    # does not divide by zero and take the whole weight.
    weighted = ",\n        ".join(
        f"SUM(o.{m} * w.wt) FILTER (WHERE o.{m} IS NOT NULL) "
        f"/ NULLIF(SUM(w.wt) FILTER (WHERE o.{m} IS NOT NULL), 0) AS wx_{m}"
        for m in MEASURES
    )
    sql = f"""
    WITH w AS (
        SELECT
            p.pid,
            n.aes,
            n.station_dist_km,
            p.day - 1 AS obs_date,
            POWER(1.0 / GREATEST(n.station_dist_km, 1.0), {IDW_POWER}) AS wt
        FROM panel p
        JOIN nb n ON n.cell_x = p.cell_x AND n.cell_y = p.cell_y
        WHERE n.station_rank < {IDW_K} AND n.station_dist_km <= {IDW_MAX_KM}
    )
    SELECT
        w.pid,
        {weighted},
        COUNT(*) AS wx_n_stations,
        MIN(w.station_dist_km) AS wx_dist_km
    FROM w
    JOIN obs o ON o.aes = w.aes AND o.obs_date = w.obs_date
    GROUP BY w.pid
    """
    # `.pl()` rather than `pl.from_arrow(... .arrow())`: an empty DuckDB result
    # carries no record batch, and pyarrow refuses to build a table from none.
    # That happens whenever a window matches nothing, which is a normal day.
    got = con.execute(sql).pl()
    con.close()

    out = panel.select("pid").join(got, on="pid", how="left")
    return out.with_columns(pl.col("wx_n_stations").fill_null(0))


def _preparedness(panel: pl.DataFrame, sitreps: pl.DataFrame) -> pl.DataFrame:
    """CIFFC preparedness for the cell's agency, as published before midnight.

    This is the covariate the escalation model could not use. Preparedness is
    one number per agency per day, so on that task every fire burning in one
    agency shared a value that `agency_code` and `doy` already encoded, and it
    contributed nothing. Here the unit of prediction *is* an agency-day patch
    of ground, and the report carries each agency's own forecast of tomorrow's
    lightning- and human-caused ignition load -- which is, almost literally,
    this model's target written down by a human. Whether that beats the
    machine features is measured in `models/ignition.py`, not assumed.
    """
    from .build import PREP_FEATURES

    if sitreps.is_empty() or "agency_code" not in panel.columns:
        return panel.select("pid")

    left = (
        panel.select("pid", "agency_code", "day")
        .with_columns(pl.col("day").cast(pl.Datetime("us")).alias("decision_at"))
        .drop_nulls(["agency_code"])
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
    return panel.select("pid").join(
        joined.select("pid", *PREP_FEATURES.values()), on="pid", how="left"
    )


def assemble_features(
    panel: pl.DataFrame,
    events: pl.DataFrame,
    hotspots: pl.DataFrame,
    station_obs: pl.DataFrame,
) -> pl.DataFrame:
    """Everything a cell-day can see before its decision instant. No label.

    Deliberately label-free, and the *only* path to a feature row: `build_panel`
    calls it to make the training table and `predict.score_ignition` calls it to
    score tomorrow. Two paths would drift, and the model would end up scored at
    serving time on a feature distribution it was never fitted on. The
    escalation side of the project makes the same promise with the same
    function name, for the same reason.

    `panel` needs `cell_x`, `cell_y` and `day`; anything else it carries -- a
    label, a count -- is passed through untouched.
    """
    panel = panel.with_row_index("pid")
    if "lat" not in panel.columns or "lon" not in panel.columns:
        panel = grid.with_centres(panel)
    if "agency_code" not in panel.columns:
        panel = panel.join(
            cell_agency(panel, events).select("cell_x", "cell_y", "agency_code"),
            on=["cell_x", "cell_y"],
            how="left",
        )

    stations = (
        station_obs.select("aes", "lat", "lon")
        .unique(subset=["aes"], keep="first")
        .sort("aes")
    )
    # Every station, including the American ones near the border: a station
    # forty kilometres away is good evidence about a cell's fuel moisture
    # whichever country it stands in. The border only matters for the *agency*,
    # which comes from the fire feed and the provincial outlines instead.
    neighbours = grid.nearest_stations(panel, stations, k=IDW_K)

    obs = station_obs.select("aes", "obs_date", *_measure_names())
    parts = [
        _station_weather(panel, neighbours, obs),
        _hotspot_history(panel, hotspots),
        _ignition_history(panel, events),
    ]
    sitrep_path = config.CURATED / "ciffc_sitreps.parquet"
    if sitrep_path.exists():
        parts.append(_preparedness(panel, pl.read_parquet(sitrep_path)))
    else:
        log.info("no CIFFC sitreps ingested - preparedness features will be absent")

    out = panel
    for part in parts:
        out = out.join(part, on="pid", how="left")

    return out.with_columns(
        pl.col("day").dt.year().alias("fire_year"),
        pl.col("day").dt.month().alias("month"),
        pl.col("day").dt.ordinal_day().alias("doy"),
    )


def build_panel(
    fires: pl.DataFrame,
    hotspots: pl.DataFrame,
    station_obs: pl.DataFrame,
    *,
    years: list[int],
    area_years: list[int],
    neg_rate: float = NEG_RATE,
    seed: int = 17,
    write: bool = True,
) -> pl.DataFrame:
    """The ignition modelling table.

    `years` are the seasons to pose the question over; `area_years` are the
    seasons the study area is drawn from and must not include a season you
    intend to evaluate on.
    """
    events = ignition_events(fires)
    area = study_area(events, hotspots, years=area_years)
    days = season_days(years)

    # The weather archive ends where it ends -- the current decadal file stops
    # part-way through a season. Posing the question past that point would
    # produce rows whose entire exogenous block is null, which a tree happily
    # learns to read as "late season". Clip instead, loudly.
    if not station_obs.is_empty():
        first_obs = station_obs["obs_date"].min()
        last_obs = station_obs["obs_date"].max()
        kept = [d for d in days if first_obs < d <= last_obs + timedelta(days=1)]
        if len(kept) != len(days):
            log.warning(
                "clipping the panel to the station record: %s of %s days kept "
                "(%s..%s)", len(kept), len(days), kept[0], kept[-1],
            )
        days = kept
    if not days:
        raise RuntimeError("no days left after clipping to the station record")

    log.info(
        "study area %s cells from seasons %s; %s days in %s",
        f"{area.height:,}", area_years, len(days), years,
    )

    reachable = events.join(
        pl.DataFrame({"day": pl.Series(days, dtype=pl.Date)}), on="day", how="semi"
    )
    n_reachable = int(reachable["n_ignitions"].sum())
    inside = reachable.join(area, on=["cell_x", "cell_y"], how="semi")
    coverage = int(inside["n_ignitions"].sum()) / max(n_reachable, 1)
    log.info(
        "study-area coverage: %.1f%% of the %s ignitions on those days fall inside it",
        100 * coverage, f"{n_reachable:,}",
    )

    panel = _sample_panel(area, events, days, neg_rate=neg_rate, seed=seed)
    log.info(
        "panel %s rows: %s positive (%.3f%% of the sample)",
        f"{panel.height:,}", f"{int(panel[TARGET].sum()):,}",
        100 * panel[TARGET].mean(),
    )

    out = assemble_features(panel, events, hotspots, station_obs)

    if write:
        dest = config.CURATED / "ignition_table.parquet"
        out.write_parquet(dest)
        meta = config.CURATED / "ignition_table_meta.json"
        import json

        meta.write_text(
            json.dumps(
                {
                    "years": years,
                    "area_years": area_years,
                    "neg_rate": neg_rate,
                    "seed": seed,
                    "n_cells": area.height,
                    "n_days": len(days),
                    "n_rows": out.height,
                    "n_positive": int(out[TARGET].sum()),
                    "study_area_coverage": round(float(coverage), 4),
                    "cell_km": grid.CELL_KM,
                },
                indent=2,
            )
        )
        log.info("ignition table: %s rows -> %s", out.height, dest)
    return out


def _measure_names() -> list[str]:
    from ..sources.cwfis_fwi import MEASURES

    return list(MEASURES)
