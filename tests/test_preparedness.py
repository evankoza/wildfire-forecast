"""The preparedness join is the easiest place in this project to leak.

A CIFFC sitrep is dated for a day but *published* late that same day, and it
contains an explicit forecast of tomorrow's fire load. Join it on `sitrep_date`
and a fire first reported that morning gets to read a judgement made hours
after its decision instant -- a leak that would flatter the model and never
raise anything. So the join is on `published_at`, and that is asserted here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from wildfire.features import build

T0 = datetime(2025, 6, 15, 6, 0)  # fire first reported 06:00 on the 15th


def _fires() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "national_fire_id": ["F1"],
            "t0": [T0],
            "agency_code": ["AB"],
        }
    )


def _sitreps() -> pl.DataFrame:
    """Three reports: two safely before the decision time, one just after."""
    return pl.DataFrame(
        {
            "agency_code": ["AB", "AB", "AB"],
            "sitrep_date": [
                datetime(2025, 6, 13).date(),
                datetime(2025, 6, 14).date(),
                datetime(2025, 6, 15).date(),
            ],
            # Reports go out at 20:00 on the day they are dated for.
            "published_at": [
                datetime(2025, 6, 13, 20, 0),
                datetime(2025, 6, 14, 20, 0),
                datetime(2025, 6, 15, 20, 0),
            ],
            "national_preparedness_level": [1.0, 2.0, 3.0],
            "agency_preparedness": [1.0, 2.0, 3.0],
            "prep_hazard": [1.0, 2.0, 3.0],
            "prep_current_load": [1.0, 2.0, 3.0],
            "prep_expected_load": [1.0, 2.0, 3.0],
            "prep_resource_levels": [1.0, 2.0, 3.0],
            "prep_resource_availability": [1.0, 2.0, 3.0],
            "occurrence_pred_lightning": [1.0, 2.0, 3.0],
            "occurrence_pred_human": [1.0, 2.0, 3.0],
        }
    )


def test_join_takes_the_latest_report_published_before_the_decision():
    """Decision at T0+24h = 06:00 on the 16th; the 15th's 20:00 report wins."""
    out = build.preparedness_features(_fires(), _sitreps(), decision_hours=24)
    assert out.height == 1
    assert out["ciffc_national_pl"][0] == 3.0
    # Published 20:00 on the 15th, decision 06:00 on the 16th -> 10 hours old.
    assert out["ciffc_sitrep_lag_hours"][0] == 10.0


def test_a_report_published_after_the_decision_is_invisible():
    """The failure mode this join exists to prevent."""
    future = _sitreps().vstack(
        pl.DataFrame(
            {
                "agency_code": ["AB"],
                "sitrep_date": [datetime(2025, 6, 16).date()],
                "published_at": [datetime(2025, 6, 16, 20, 0)],  # after decision
                "national_preparedness_level": [5.0],
                "agency_preparedness": [5.0],
                "prep_hazard": [5.0],
                "prep_current_load": [5.0],
                "prep_expected_load": [5.0],
                "prep_resource_levels": [5.0],
                "prep_resource_availability": [5.0],
                "occurrence_pred_lightning": [5.0],
                "occurrence_pred_human": [5.0],
            }
        )
    )
    out = build.preparedness_features(_fires(), future, decision_hours=24)
    assert out["ciffc_national_pl"][0] == 3.0, "leaked a sitrep published after the decision"


def test_same_day_publication_is_excluded_when_it_lands_after_the_decision():
    """A fire reported at 06:00 decides at 06:00 next day, not at 20:00 today."""
    early = _fires().with_columns(pl.lit(datetime(2025, 6, 15, 6, 0)).alias("t0"))
    # Decision at T0+6h = 12:00 on the 15th, before that day's 20:00 report.
    out = build.preparedness_features(early, _sitreps(), decision_hours=6)
    assert out["ciffc_national_pl"][0] == 2.0, "saw the 15th's report before it existed"


def test_agencies_do_not_borrow_each_others_reports():
    fires = pl.DataFrame(
        {"national_fire_id": ["F1"], "t0": [T0], "agency_code": ["BC"]}
    )
    out = build.preparedness_features(fires, _sitreps(), decision_hours=24)
    assert out.height == 1
    assert out["ciffc_national_pl"][0] is None


def test_no_sitreps_yields_no_rows_rather_than_wrong_ones():
    empty = pl.DataFrame()
    assert build.preparedness_features(_fires(), empty, decision_hours=24).height == 0
