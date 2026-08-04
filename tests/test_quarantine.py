"""Revision rows whose transaction time cannot belong to their season.

A handful of 2023 fires carry a `record_start` in 2011. Two rows in twenty-three
thousand fires, and they matter out of all proportion: T0 is
`min(record_start)` per fire, so one row dated twelve years early moves that
fire's entire decision instant, and every as-of feature is then computed from a
window that closed before the fire existed. The label comes from the same
mechanism, so the fire is scored against a horizon in the wrong decade too.

The rule is deliberately loose -- a whole year either side of the fire year --
because reporting genuinely does spill across a new year. What it catches is
not late reporting, it is a broken timestamp.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from wildfire.features import asof
from wildfire.sources.cwfis_fires import quarantine_record_start


def _rows(pairs):
    return pl.DataFrame(
        {
            "national_fire_id": [p[0] for p in pairs],
            "fire_year": [p[1] for p in pairs],
            "record_start": [p[2] for p in pairs],
            "fire_size": [1.0] * len(pairs),
            "id": list(range(len(pairs))),
        },
        schema_overrides={"fire_year": pl.Int32},
    )


def test_a_decade_early_record_is_quarantined():
    df = _rows(
        [
            ("2023_AB_1", 2023, datetime(2011, 5, 4, 10)),
            ("2023_AB_1", 2023, datetime(2023, 6, 1, 8)),
            ("2023_AB_1", 2023, datetime(2023, 6, 1, 20)),
        ]
    )
    kept, rejected = quarantine_record_start(df)
    assert kept.height == 2
    assert rejected.height == 1
    assert rejected["record_start"][0].year == 2011


def test_quarantine_moves_t0_back_to_the_real_first_report():
    """The reason this matters at all."""
    df = _rows(
        [
            ("2023_AB_1", 2023, datetime(2011, 5, 4, 10)),
            ("2023_AB_1", 2023, datetime(2023, 6, 1, 8)),
        ]
    ).with_columns(
        pl.lit(55.0).alias("latitude"),
        pl.lit(-114.0).alias("longitude"),
        pl.lit("AB").alias("agency_code"),
        pl.lit("HWF").alias("region_code"),
    )

    dirty = asof.first_seen(df)["t0"][0]
    clean = asof.first_seen(quarantine_record_start(df)[0])["t0"][0]
    assert dirty.year == 2011
    assert clean.year == 2023


def test_reporting_that_spills_across_the_new_year_is_kept():
    """A fire filed against the next fire year in late December is normal."""
    df = _rows(
        [
            ("2024_ON_9", 2024, datetime(2023, 12, 29, 22)),
            ("2024_ON_9", 2024, datetime(2024, 4, 2, 9)),
            ("2024_ON_9", 2024, datetime(2025, 3, 3, 9)),
        ]
    )
    kept, rejected = quarantine_record_start(df)
    assert rejected.height == 0
    assert kept.height == 3


def test_a_record_two_years_late_is_quarantined():
    df = _rows([("2023_SK_2", 2023, datetime(2026, 8, 1, 12))])
    kept, rejected = quarantine_record_start(df)
    assert kept.height == 0 and rejected.height == 1


def test_nulls_are_never_quarantined():
    """A missing timestamp is a different problem, handled elsewhere."""
    df = _rows(
        [
            ("2023_QC_3", 2023, None),
            ("2023_QC_3", None, datetime(1999, 1, 1)),
        ]
    )
    kept, rejected = quarantine_record_start(df)
    assert kept.height == 2 and rejected.height == 0


def test_the_split_is_lossless():
    df = _rows(
        [
            ("A", 2023, datetime(2011, 1, 1)),
            ("B", 2023, datetime(2023, 7, 1)),
            ("C", 2024, datetime(2026, 7, 1)),
            ("D", 2024, datetime(2024, 7, 1)),
        ]
    )
    kept, rejected = quarantine_record_start(df)
    assert kept.height + rejected.height == df.height
    assert set(kept["national_fire_id"]) | set(rejected["national_fire_id"]) == {
        "A", "B", "C", "D"
    }


def test_a_frame_without_the_columns_passes_through():
    df = pl.DataFrame({"national_fire_id": ["A"], "fire_size": [1.0]})
    kept, rejected = quarantine_record_start(df)
    assert kept.height == 1 and rejected.height == 0
