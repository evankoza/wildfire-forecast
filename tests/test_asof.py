"""The point-in-time guarantee is the project's main claim. Test it directly.

If `state_asof` ever returns a revision that had not yet been recorded at the
decision time, every metric downstream becomes fiction. These tests build a
tiny bitemporal fixture where the right answer is obvious by inspection.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from wildfire.features import asof

T0 = datetime(2025, 6, 1, 12, 0)


def _fires() -> pl.DataFrame:
    """One fire, three revisions: 0.5 ha -> 40 ha -> 900 ha."""
    return pl.DataFrame(
        {
            "national_fire_id": ["F1"] * 3,
            "id": [1, 2, 3],
            "fire_size": [0.5, 40.0, 900.0],
            "stage_of_control_status": ["UC", "OC", "OC"],
            "record_start": [T0, T0 + timedelta(hours=20), T0 + timedelta(hours=60)],
            "record_end": [
                T0 + timedelta(hours=20),
                T0 + timedelta(hours=60),
                datetime(2026, 12, 31),
            ],
            "latitude": [55.0] * 3,
            "longitude": [-115.0] * 3,
            "agency_code": ["AB"] * 3,
            "region_code": ["SWF"] * 3,
            "fire_year": [2025] * 3,
            "response_type": ["FUL"] * 3,
            "national_fire_cause": ["N"] * 3,
            "percent_contained": [None] * 3,
        }
    )


def test_first_seen_picks_earliest_record_start():
    off = asof.first_seen(_fires())
    assert off.height == 1
    assert off["t0"][0] == T0


def test_state_asof_cannot_see_the_future():
    """At T0+24h only the first two revisions exist, so size must be 40 ha."""
    fires = _fires()
    off = asof.first_seen(fires)
    state = asof.state_asof(fires, off, hours=24)

    assert state["fire_size"][0] == 40.0, "leaked a revision recorded at T0+60h"
    assert state["stage_of_control_status"][0] == "OC"


def test_state_asof_at_horizon_sees_the_later_revision():
    fires = _fires()
    off = asof.first_seen(fires)
    assert asof.state_asof(fires, off, hours=72)["fire_size"][0] == 900.0


def test_label_at_horizon_matches_state():
    fires = _fires()
    off = asof.first_seen(fires)
    lab = asof.label_at(fires, off, hours=72)
    assert lab["size_at_horizon"][0] == 900.0
    assert lab["status_at_horizon"][0] == "OC"


def test_revisions_before_counts_only_visible_rows():
    fires = _fires()
    off = asof.first_seen(fires)
    revs = asof.revisions_before(fires, off, hours=24)
    assert revs["n_revisions_by_decision"][0] == 2
    assert revs["size_first"][0] == 0.5


def test_observable_drops_fires_without_a_full_horizon():
    """A fire reported an hour ago cannot have a 72 h outcome yet."""
    fires = _fires()
    off = asof.first_seen(fires)

    # Cutoff sits inside the horizon -> the fire must be excluded.
    early = asof.observable(off, T0 + timedelta(hours=40), hours=72)
    assert early.height == 0

    late = asof.observable(off, T0 + timedelta(hours=100), hours=72)
    assert late.height == 1


def test_state_asof_excludes_fire_with_no_visible_revision():
    """A cutoff before the first record_start yields no state at all."""
    fires = _fires()
    off = asof.first_seen(fires).with_columns(
        (pl.col("t0") - pl.duration(hours=10)).alias("t0")
    )
    # Decision at t0+1h is still 9h before the real first record.
    assert asof.state_asof(fires, off, hours=1).height == 0
