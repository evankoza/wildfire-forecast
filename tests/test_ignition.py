"""The ignition panel: point-in-time discipline, and undoing the sampling.

The escalation table gets its point-in-time guarantee from the feed's
transaction time -- obeying it is a `WHERE` clause. The ignition panel does
not have that luxury: its unit is a cell and a day, so every trailing window
is arithmetic that a fencepost error would silently widen by one day. These
tests plant an event exactly on the decision instant and demand that it does
not appear.

The second half is about negative sampling, which is the only thing here that
makes a *correct* pipeline produce wrong numbers: probabilities inflated by
the sampling rate, and a PR-AUC inflated by the same factor because precision
is a function of prevalence.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wildfire.features import grid
from wildfire.features import ignition as ig
from wildfire.models import ignition as ig_model

# Somewhere unambiguously in Alberta, so the country filter keeps it.
LAT, LON = 55.0, -114.0
DAY = date(2024, 7, 15)


def _cell():
    cx, cy = grid.cell_of(LAT, LON)
    return int(cx), int(cy)


def _fires(rows: list[tuple[str, datetime, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "national_fire_id": [r[0] for r in rows],
            "record_start": [r[1] for r in rows],
            "latitude": [r[2] for r in rows],
            "longitude": [r[3] for r in rows],
            "agency_code": ["AB"] * len(rows),
            "region_code": ["HWF"] * len(rows),
            "fire_year": [r[1].year for r in rows],
        }
    )


def _panel(days: list[date]) -> pl.DataFrame:
    cx, cy = _cell()
    return pl.DataFrame(
        {
            "cell_x": pl.Series([cx] * len(days), dtype=pl.Int32),
            "cell_y": pl.Series([cy] * len(days), dtype=pl.Int32),
            "day": pl.Series(days, dtype=pl.Date),
        }
    )


def _stations(days: list[date]) -> pl.DataFrame:
    """One station 5 km from the cell, reporting every day."""
    rows = []
    for d in days:
        rows.append(
            {
                "aes": "S1", "obs_date": d, "lat": LAT + 0.05, "lon": LON,
                "temp": 20.0, "rh": 40.0, "ws": 10.0, "precip": 0.0,
                "ffmc": 90.0, "dmc": 30.0, "dc": 200.0, "isi": 8.0,
                "bui": 40.0, "fwi": 20.0, "dsr": 4.0, "prov": "AB",
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("obs_date").cast(pl.Date))


# --- point-in-time -------------------------------------------------------


def test_an_ignition_on_the_decision_day_is_not_a_feature():
    """The label day itself must be invisible to the history windows.

    A fire reported at 00:05 on the forecast day is the thing being predicted.
    If the trailing count includes it, `ig_n_7d` becomes the answer and the
    model looks superb.
    """
    fires = _fires(
        [
            ("F_today", datetime.combine(DAY, datetime.min.time()) + timedelta(minutes=5), LAT, LON),
            ("F_before", datetime.combine(DAY, datetime.min.time()) - timedelta(days=3), LAT, LON),
        ]
    )
    events = ig.ignition_events(fires)
    got = ig._ignition_history(_panel([DAY]).with_row_index("pid"), events)
    assert got["ig_n_7d"][0] == 1, "only the fire three days earlier may be counted"
    assert got["ig_days_since"][0] == 3


def test_a_hotspot_on_the_decision_day_is_not_a_feature():
    cx, cy = _cell()
    hs = pl.DataFrame(
        {
            "lat": [LAT, LAT],
            "lon": [LON, LON],
            "rep_date": [
                datetime.combine(DAY, datetime.min.time()) + timedelta(hours=2),
                datetime.combine(DAY, datetime.min.time()) - timedelta(hours=2),
            ],
            "fwi": [50.0, 10.0],
            "hfi": [9000.0, 100.0],
            "ros": [30.0, 1.0],
        }
    )
    got = ig._hotspot_history(_panel([DAY]).with_row_index("pid"), hs)
    assert got["hs_n_1d"][0] == 1
    assert got["hs_hfi_max_1d"][0] == pytest.approx(100.0), (
        "the detection two hours *after* the decision instant leaked in"
    )


def test_station_weather_uses_the_previous_days_observation():
    """Noon-local readings are not available at the preceding midnight."""
    days = [DAY - timedelta(days=1), DAY]
    obs = _stations(days).with_columns(
        pl.when(pl.col("obs_date") == DAY).then(99.0).otherwise(11.0).alias("fwi")
    )
    panel = grid.with_centres(_panel([DAY]).with_row_index("pid"))
    stations = obs.select("aes", "lat", "lon").unique(subset=["aes"])
    nb = grid.nearest_stations(panel, stations, k=1)
    got = ig._station_weather(panel, nb, obs.select("aes", "obs_date", *ig._measure_names()))
    assert got["wx_fwi"][0] == pytest.approx(11.0), (
        "the forecast day's own noon reading was used, which does not exist yet"
    )


def test_trailing_windows_are_closed_at_the_right_fencepost():
    """`ig_n_7d` counts days d-7 .. d-1 inclusive, and nothing else."""
    base = datetime.combine(DAY, datetime.min.time())
    fires = _fires(
        [(f"F{i}", base - timedelta(days=i), LAT, LON) for i in (1, 7, 8)]
    )
    events = ig.ignition_events(fires)
    got = ig._ignition_history(_panel([DAY]).with_row_index("pid"), events)
    assert got["ig_n_7d"][0] == 2, "d-8 must be outside the seven-day window"
    assert got["ig_n_30d"][0] == 3


def test_label_is_at_least_one_ignition_not_a_count():
    base = datetime.combine(DAY, datetime.min.time())
    fires = _fires([(f"F{i}", base + timedelta(hours=i), LAT, LON) for i in range(3)])
    events = ig.ignition_events(fires)
    assert events["n_ignitions"][0] == 3

    cx, cy = _cell()
    area = pl.DataFrame(
        {
            "cell_x": pl.Series([cx], dtype=pl.Int32),
            "cell_y": pl.Series([cy], dtype=pl.Int32),
            "agency_code": ["AB"],
        }
    )
    panel = ig._sample_panel(area, events, [DAY], neg_rate=1.0, seed=1)
    assert panel.height == 1
    assert panel[ig.TARGET][0] == 1


# --- the study area ------------------------------------------------------


def test_study_area_excludes_cells_outside_canada():
    """The hotspot feed is continental; the fire feed is not.

    Without the country filter, a detection over Montana or Florida becomes a
    cell-day that can never be positive, and the panel fills up with free
    negatives.
    """
    hs = pl.DataFrame(
        {
            "lat": [LAT, 46.9, 25.8],          # Alberta, Montana, Miami
            "lon": [LON, -110.0, -80.2],
            "rep_date": [datetime(2024, 7, 1)] * 3,
        }
    )
    events = ig.ignition_events(
        _fires([("F1", datetime(2024, 6, 1), LAT, LON)])
    )
    area = ig.study_area(events, hs, years=[2024])
    assert area.height == 1
    assert (int(area["cell_x"][0]), int(area["cell_y"][0])) == _cell()
    assert area["agency_code"][0] == "AB"


def test_cell_agency_prefers_the_feed_over_the_polygon():
    """The outlines are 23 KB of simplified Natural Earth; the feed is truth.

    Where an agency has actually filed a report in a cell, that is the answer,
    and the polygon only fills in cells no one has ever reported a fire in.
    """
    events = ig.ignition_events(
        _fires([("F1", datetime(2024, 6, 1), LAT, LON)])
    ).with_columns(pl.lit("SK").alias("agency_code"))  # deliberately not AB
    cx, cy = _cell()
    cells = pl.DataFrame(
        {"cell_x": pl.Series([cx], dtype=pl.Int32),
         "cell_y": pl.Series([cy], dtype=pl.Int32)}
    )
    got = ig.cell_agency(cells, events)
    assert got["agency_code"][0] == "SK"


def test_parks_canada_is_never_assigned_to_a_cell():
    """PC is scattered federal parkland, not a region.

    A PC fire says nothing about which agency covers the ground around it, so
    a PC-only cell falls through to the provincial outline.
    """
    events = ig.ignition_events(
        _fires([("F1", datetime(2024, 6, 1), LAT, LON)])
    ).with_columns(pl.lit("PC").alias("agency_code"))
    cx, cy = _cell()
    cells = pl.DataFrame(
        {"cell_x": pl.Series([cx], dtype=pl.Int32),
         "cell_y": pl.Series([cy], dtype=pl.Int32)}
    )
    assert ig.cell_agency(cells, events)["agency_code"][0] == "AB"


# --- negative sampling ---------------------------------------------------


def test_prior_correction_recovers_the_population_rate():
    """A calibrated probability on the sample is wrong by the sampling odds.

    Constructed from a known population: prevalence 1 in 1000, negatives kept
    at 2%. The sampled prevalence is then about 1 in 21, and correcting it
    must land back on 1 in 1000.
    """
    rate = 0.02
    p_pop = 0.001
    odds_pop = p_pop / (1 - p_pop)
    p_sample = (odds_pop / rate) / (1 + odds_pop / rate)
    back = ig_model.correct_prior(np.array([p_sample]), rate)[0]
    assert back == pytest.approx(p_pop, rel=1e-9)


def test_prior_correction_is_monotone_and_bounded():
    p = np.linspace(0.001, 0.999, 50)
    out = ig_model.correct_prior(p, 0.05)
    assert np.all(np.diff(out) > 0)
    assert out.min() > 0 and out.max() < 1
    assert np.all(out < p), "correcting must always lower a probability when r<1"


def test_sample_weights_reconstruct_the_population_prevalence():
    y = np.array([1, 1, 0, 0, 0, 0])
    rate = 0.01
    w = ig_model.sample_weights(y, rate)
    # Two positives against four sampled negatives standing in for 400.
    assert float(np.average(y, weights=w)) == pytest.approx(2 / 402)


def test_weighting_moves_the_floor_from_sampled_to_population_prevalence():
    """The reason every metric in the module takes weights.

    A useless scorer has a PR-AUC equal to the prevalence. Unweighted, the
    sampled panel's prevalence is ~5%, so noise scores 0.05 and a mediocre
    model reads as impressive. Weighted, the floor drops to the population's
    0.05%, which is the number every result on this task has to be read
    against. The ratio between the two is exactly the sampling rate.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    n = 20_000
    rate = 0.01
    y = (rng.random(n) < 0.05).astype(int)
    noise = rng.random(n)
    w = ig_model.sample_weights(y, rate)

    unweighted = average_precision_score(y, noise)
    weighted = average_precision_score(y, noise, sample_weight=w)
    assert unweighted == pytest.approx(y.mean(), abs=0.01)
    assert weighted == pytest.approx(float(np.average(y, weights=w)), abs=1e-3)
    assert weighted < unweighted / 40


def test_weighting_shrinks_a_real_models_pr_auc():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.05).astype(int)
    # Informative but far from separable, like the real thing.
    p = np.clip(rng.normal(0.3 * y, 0.35, n), 0, 1)
    w = ig_model.sample_weights(y, 0.01)
    assert average_precision_score(y, p, sample_weight=w) < average_precision_score(y, p)


def test_negative_sampling_never_mislabels_a_positive():
    """A sampled cell-day that turns out to be an ignition is not a negative.

    Without the anti-join the same (cell, day) can appear twice with two
    different labels, and nothing raises.
    """
    base = datetime(2024, 6, 1)
    rows = []
    for i in range(60):
        rows.append((f"F{i}", base + timedelta(days=i % 30), LAT + 0.2 * (i % 7), LON))
    events = ig.ignition_events(_fires(rows))
    area = events.select("cell_x", "cell_y").unique().with_columns(
        pl.lit("AB").alias("agency_code")
    )
    days = [date(2024, 6, 1) + timedelta(days=i) for i in range(30)]

    panel = ig._sample_panel(area, events, days, neg_rate=1.0, seed=5)
    dupes = panel.group_by(["cell_x", "cell_y", "day"]).agg(pl.len().alias("n"))
    assert dupes["n"].max() == 1

    truth = events.select("cell_x", "cell_y", "day").with_columns(pl.lit(1).alias("t"))
    check = panel.join(truth, on=["cell_x", "cell_y", "day"], how="left")
    assert (
        check.select((pl.col(ig.TARGET) == pl.col("t").fill_null(0)).all()).item()
    )


def test_sampling_is_reproducible_for_a_seed():
    base = datetime(2024, 6, 1)
    events = ig.ignition_events(
        _fires([(f"F{i}", base + timedelta(days=i % 10), LAT + 0.2 * i, LON) for i in range(20)])
    )
    area = events.select("cell_x", "cell_y").unique().with_columns(
        pl.lit("AB").alias("agency_code")
    )
    days = [date(2024, 6, 1) + timedelta(days=i) for i in range(10)]
    a = ig._sample_panel(area, events, days, neg_rate=0.5, seed=3)
    b = ig._sample_panel(area, events, days, neg_rate=0.5, seed=3)
    assert a.equals(b)


# --- the hand-rolled metric ----------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_weighted_average_precision_matches_sklearn(seed):
    """It is written out by hand, so it has to be pinned to the reference.

    Scores are rounded hard on purpose: ties are where a hand-rolled PR curve
    goes wrong, and the real model produces plenty of them because isotonic
    calibration maps whole ranges onto one value.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    n = int(rng.integers(200, 4000))
    y = (rng.random(n) < rng.uniform(0.02, 0.4)).astype(int)
    if y.sum() == 0:
        pytest.skip("no positives drawn")
    p = np.round(rng.random(n), int(rng.integers(1, 4)))
    w = ig_model.sample_weights(y, 0.03) if seed % 2 else np.ones(n)

    assert ig_model._ap(y, p, w) == pytest.approx(
        average_precision_score(y, p, sample_weight=w), abs=1e-12
    )


def test_bootstrap_by_weight_equals_bootstrap_by_index():
    """The optimisation the pooled table depends on.

    Resampling rows with replacement is a multinomial over the originals, and
    average precision reads only the score order plus each row's weight -- so
    drawing a row k times is exactly giving it k times its weight. That lets
    the sort happen once instead of once per draw. If this identity ever stops
    holding, every interval in the ignition tables is wrong.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(4)
    n = 1500
    y = (rng.random(n) < 0.08).astype(int)
    p = np.round(rng.random(n), 2)
    w = ig_model.sample_weights(y, 0.02)

    counts = rng.multinomial(n, np.full(n, 1 / n))
    idx = np.repeat(np.arange(n), counts)

    by_index = average_precision_score(y[idx], p[idx], sample_weight=w[idx])
    ys, ws, runs, order = ig_model._sorted_view(y, p, w)
    by_weight = ig_model._ap_presorted(ys, ws * counts[order].astype(float), runs)
    assert by_weight == pytest.approx(by_index, abs=1e-12)


def test_average_precision_of_a_constant_scorer_is_the_prevalence():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.full(10, 0.3)
    w = np.ones(10)
    assert ig_model._ap(y, p, w) == pytest.approx(0.2)


def test_average_precision_is_zero_without_positives():
    y = np.zeros(20)
    assert ig_model._ap(y, np.linspace(0, 1, 20), np.ones(20)) == 0.0


# --- the model contract --------------------------------------------------


def test_forbidden_columns_cannot_reach_the_feature_matrix():
    """The label and the row's identity are guarded explicitly."""
    assert ig.TARGET in ig_model.FORBIDDEN
    assert "n_ignitions" in ig_model.FORBIDDEN
    leaked = ig_model.FORBIDDEN & set(ig_model.NUMERIC + ig_model.CATEGORICAL)
    assert not leaked, leaked


def test_feature_frame_pins_category_levels_from_training():
    """Unpinned levels make the model read one agency as another.

    pandas numbers categories per frame, so an agency missing from one side
    shifts every code after it -- which is exactly the situation a
    region-blocked fold creates by construction.
    """
    train = pl.DataFrame({"agency_code": ["AB", "BC", "SK"], "lat": [1.0, 2.0, 3.0]})
    test = pl.DataFrame({"agency_code": ["SK", "YT"], "lat": [4.0, 5.0]})

    _, levels = ig_model._feature_frame(train)
    X_te, _ = ig_model._feature_frame(test, categories=levels)
    codes = X_te["agency_code"].cat.codes.tolist()
    assert list(X_te["agency_code"].cat.categories) == ["AB", "BC", "SK"]
    assert codes[0] == 2, "SK must keep the code it had in training"
    assert codes[1] == -1, "an unseen agency must arrive as missing, not as a level"


def test_drop_groups_name_real_columns():
    every = set()
    for group in ("ciffc", "network", "weather", "hotspots", "history", "geography"):
        cols = set(ig_model._drop_for([group]))
        assert cols, group
        every |= cols
    # `network` is deliberately a subset of `weather` -- both are wx_ columns.
    assert set(ig_model._drop_for(["network"])) < set(ig_model._drop_for(["weather"]))
    assert every <= set(ig_model.NUMERIC) | set(ig_model.CATEGORICAL)


def test_every_declared_feature_set_is_droppable():
    for name, groups in ig_model.FEATURE_SETS.items():
        assert isinstance(ig_model._drop_for(groups), list), name


def test_unknown_feature_group_raises():
    with pytest.raises(ValueError):
        ig_model._drop_for(["nonsense"])
