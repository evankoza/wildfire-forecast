"""The hotspot aggregation must not depend on the order of its input.

This is a regression test for a bug that cost real time. DuckDB aggregates in
parallel and reduces partial sums in whatever order the threads finish. Float
addition is not associative, so `AVG` over float64 drifted in its last bits
between runs of the *same query on the same data* -- and `MODE` broke ties by
whichever row the scan reached first.

Physically that drift is nothing. To a gradient booster it is not nothing:
about 35 of 21,500 fires had a feature land the other side of a split
threshold, and the headline PR-AUC moved by ~0.008 with no change to the data
at all. It was found only because a refactor that could not possibly have
changed a number appeared to change one.

The fixes are in `hotspot_features`: sums accumulate in DECIMAL (associative,
so reduction order cannot matter), and the dominant fuel type is chosen by an
explicit `ROW_NUMBER() ... ORDER BY COUNT(*) DESC, fuel_group ASC` rather than
`MODE`. These tests fail if either is reverted.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from wildfire.features.build import hotspot_features

T0 = datetime(2025, 6, 1, 12, 0)


def _fires() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "national_fire_id": ["F1", "F2"],
            "t0": [T0, T0 + timedelta(hours=5)],
            "lat": [55.0, 52.0],
            "lon": [-115.0, -106.0],
        }
    )


def _hotspots(n: int = 400) -> pl.DataFrame:
    """Detections around both fires, with values chosen to make float
    summation order actually matter -- many values whose sum needs more
    precision than float64 keeps."""
    rows = []
    for i in range(n):
        near_f1 = i % 2 == 0
        rows.append(
            {
                "lat": (55.0 if near_f1 else 52.0) + (i % 7) * 0.001,
                "lon": (-115.0 if near_f1 else -106.0) + (i % 5) * 0.001,
                "rep_date": T0 + timedelta(minutes=i),
                "fwi": 10.0 + i * 0.1234567891,
                "hfi": 1000.0 + i * 7.7777777,
                "ros": 0.1 + i * 0.0333333333,
                "sfc": 1.0 + i * 0.0111111111,
                "tfc": 2.0 + i * 0.0222222222,
                # Deliberate exact tie between two fuel groups for F1.
                "fuel": "C2" if i % 4 in (0, 1) else "D1",
            }
        )
    return pl.DataFrame(rows)


def _shuffled(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    return df.sample(fraction=1.0, shuffle=True, seed=seed)


def test_aggregates_do_not_depend_on_input_order():
    fires, hs = _fires(), _hotspots()

    base = hotspot_features(fires, hs).sort("national_fire_id")
    for seed in (1, 2, 3):
        other = hotspot_features(fires, _shuffled(hs, seed)).sort("national_fire_id")
        assert base.columns == other.columns
        for col in base.columns:
            assert base[col].equals(other[col]), (
                f"{col} changed when only the input row order changed -- the "
                f"aggregation has become order-dependent again"
            )


def test_repeated_calls_are_identical():
    """Same input twice: parallel reduction order must not leak through."""
    fires, hs = _fires(), _hotspots()
    a = hotspot_features(fires, hs).sort("national_fire_id")
    b = hotspot_features(fires, hs).sort("national_fire_id")
    assert a.equals(b)


def test_fuel_group_tie_breaks_alphabetically():
    """An exact tie resolves to the alphabetically first group, every time.

    C2 and D1 appear equally often around F1 by construction, so `MODE` could
    legitimately return either -- and did, depending on scan order.
    """
    fires, hs = _fires(), _hotspots()
    seen = {
        hotspot_features(fires, _shuffled(hs, s))
        .sort("national_fire_id")["hs_fuel_group"][0]
        for s in range(6)
    }
    assert seen == {"conifer"}, f"tie-break is unstable, saw {seen}"
