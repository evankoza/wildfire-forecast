"""Bitemporal as-of reconstruction.

The one rule this project is built around:

    A feature for a fire at decision time T may only be computed from rows
    whose transaction-time window had already *opened* at T, i.e.
    record_start <= T.

Because the CWFIF feed is system-versioned, obeying that rule is a filter
rather than an act of discipline. `state_asof` is the only sanctioned way to
ask "what did we know then", and the feature builder calls nothing else.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

FIRE_KEY = "national_fire_id"


def first_seen(fires: pl.DataFrame) -> pl.DataFrame:
    """T0 per fire: the instant it first entered the national feed."""
    return (
        fires.group_by(FIRE_KEY)
        .agg(
            t0=pl.col("record_start").min(),
            lat=pl.col("latitude").drop_nulls().first(),
            lon=pl.col("longitude").drop_nulls().first(),
            agency_code=pl.col("agency_code").drop_nulls().first(),
            region_code=pl.col("region_code").drop_nulls().first(),
            fire_year=pl.col("fire_year").drop_nulls().first(),
            n_revisions=pl.len(),
        )
        .filter(pl.col("t0").is_not_null())
    )


def state_asof(fires: pl.DataFrame, offsets: pl.DataFrame, hours: int) -> pl.DataFrame:
    """The believed state of each fire at T0 + `hours`.

    `offsets` must carry `national_fire_id` and `t0`. Returns one row per
    fire: the latest revision whose record_start is at or before the cutoff.
    Fires with no revision yet visible at that cutoff drop out.
    """
    cutoff = offsets.select(
        FIRE_KEY,
        (pl.col("t0") + pl.duration(hours=hours)).alias("_cutoff"),
    )

    joined = fires.join(cutoff, on=FIRE_KEY, how="inner")
    visible = joined.filter(pl.col("record_start") <= pl.col("_cutoff"))

    # Latest belief wins; `id` breaks ties within the same second.
    latest = (
        visible.sort(["record_start", "id"], descending=[False, False])
        .group_by(FIRE_KEY)
        .last()
    )
    return latest.drop("_cutoff")


def revisions_before(fires: pl.DataFrame, offsets: pl.DataFrame, hours: int) -> pl.DataFrame:
    """How many times a fire had been re-reported by T0 + `hours`.

    Reporting cadence is itself informative: agencies update the fires they
    are worried about. This is an operational-attention proxy, and it is
    legitimately available at decision time.
    """
    cutoff = offsets.select(
        FIRE_KEY, (pl.col("t0") + pl.duration(hours=hours)).alias("_cutoff")
    )
    return (
        fires.join(cutoff, on=FIRE_KEY, how="inner")
        .filter(pl.col("record_start") <= pl.col("_cutoff"))
        .group_by(FIRE_KEY)
        .agg(
            n_revisions_by_decision=pl.len(),
            size_first=pl.col("fire_size").drop_nulls().first(),
            size_max_so_far=pl.col("fire_size").max(),
            n_distinct_status=pl.col("stage_of_control_status").n_unique(),
        )
    )


def label_at(fires: pl.DataFrame, offsets: pl.DataFrame, hours: int) -> pl.DataFrame:
    """Outcome state at the label horizon. Same mechanism, later cutoff."""
    lab = state_asof(fires, offsets, hours)
    return lab.select(
        FIRE_KEY,
        pl.col("fire_size").alias("size_at_horizon"),
        pl.col("stage_of_control_status").alias("status_at_horizon"),
    )


def observable(offsets: pl.DataFrame, data_cutoff: datetime, hours: int) -> pl.DataFrame:
    """Drop fires too recent to have a label yet.

    Without this the most recent fires all look like non-escalations simply
    because their later revisions have not happened, which biases the model
    toward optimism exactly where it is used.
    """
    return offsets.filter(
        pl.col("t0") + pl.duration(hours=hours) <= pl.lit(data_cutoff)
    )
