"""The region-blocked backtest is only worth reading if the block is real.

These tests pin the two things that would silently invalidate it: a held-out
agency leaking into the training frame, and pandas re-numbering category codes
between the train and test frames. The second is the nastier of the two --
nothing raises, the model just quietly reads `agency_code == BC` as `ON`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wildfire import config
from wildfire.models import escalation

AGENCIES = ["AB", "BC", "ON", "SK"]


def _table(n: int = 800, seed: int = 3) -> pl.DataFrame:
    """A synthetic modelling table with a learnable, region-independent signal."""
    rng = np.random.default_rng(seed)
    size = rng.gamma(1.5, 4.0, n)
    hfi = rng.gamma(2.0, 800.0, n)
    agency = rng.choice(AGENCIES, n)
    t0 = [datetime(2024, 5, 1) + timedelta(hours=int(h)) for h in rng.integers(0, 3000, n)]

    # Escalation depends on behaviour (size, intensity), not on which agency
    # reported it -- so a model that generalises spatially *should* work here.
    # The intercept is set for a base rate in the same ballpark as the real
    # table (~5%); a fixture that escalates a quarter of the time would make
    # every PR-AUC assertion below far too easy to pass. The slopes are steep
    # enough that the signal is actually recoverable from a few hundred
    # positives -- an oracle scores ~0.75 AP here, so a fold that lands near
    # the base rate is a real regression and not just a thin sample.
    logit = -11.0 + 0.35 * size + 0.0018 * hfi
    escalated = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)

    return pl.DataFrame(
        {
            "national_fire_id": [f"F{i}" for i in range(n)],
            "t0": t0,
            "fire_year": rng.choice([2023, 2024, 2025], n).astype(np.int32),
            "agency_code": agency,
            "region_code": [f"{a}-{i % 3}" for i, a in enumerate(agency)],
            "status_at_decision": rng.choice(["UC", "OC", "BH"], n),
            "size_at_decision": size,
            "log_size_at_decision": np.log1p(size),
            "hs_hfi_max": hfi,
            "hs_count": rng.integers(0, 40, n).astype(float),
            "lat": rng.uniform(49, 62, n),
            "lon": rng.uniform(-135, -60, n),
            "doy": rng.integers(120, 270, n).astype(np.int32),
            "escalated": escalated,
        }
    )


# --- category pinning -------------------------------------------------------


def test_test_frame_inherits_training_category_codes():
    """Same label must mean the same integer code on both sides of the split."""
    train = _table(300, seed=1)
    test = _table(200, seed=2).with_columns(
        pl.col("agency_code").str.replace("AB", "BC")  # AB absent from the test frame
    )

    X_tr, levels = escalation._feature_frame(train)
    X_te, _ = escalation._feature_frame(test, categories=levels)

    assert list(X_te["agency_code"].cat.categories) == list(X_tr["agency_code"].cat.categories)
    on_code_tr = list(X_tr["agency_code"].cat.categories).index("ON")
    on_rows = X_te["agency_code"] == "ON"
    assert (X_te["agency_code"].cat.codes[on_rows] == on_code_tr).all()


def test_unseen_category_becomes_missing_not_a_neighbour():
    """A held-out agency is unseen by construction; it must arrive as NaN."""
    train = _table(300, seed=1).filter(pl.col("agency_code") != "SK")
    test = _table(200, seed=2).filter(pl.col("agency_code") == "SK")

    _, levels = escalation._feature_frame(train)
    X_te, _ = escalation._feature_frame(test, categories=levels)

    assert "SK" not in list(levels["agency_code"])
    assert X_te["agency_code"].isna().all(), "unseen agency was mapped onto a trained level"


def test_drop_removes_geographic_features():
    X, _ = escalation._feature_frame(_table(100), drop=escalation.GEOGRAPHIC)
    assert not (set(escalation.GEOGRAPHIC) & set(X.columns))
    assert "hs_hfi_max" in X.columns


def test_feature_frame_rejects_a_leaked_column(monkeypatch):
    monkeypatch.setattr(escalation, "NUMERIC", escalation.NUMERIC + ["size_at_horizon"])
    df = _table(50).with_columns(pl.lit(1.0).alias("size_at_horizon"))
    with pytest.raises(RuntimeError, match="leakage"):
        escalation._feature_frame(df)


# --- the spatial block itself -----------------------------------------------


def test_holdout_agency_never_appears_in_training(monkeypatch, tmp_path):
    """The whole claim of the region-blocked backtest, asserted directly."""
    seen: list[tuple[str, set]] = []

    def spy(train, test, **kw):
        seen.append((set(train["agency_code"].unique()), set(test["agency_code"].unique())))
        y_te = test["escalated"].to_numpy().astype(int)
        return {
            "model": None,
            "p": np.linspace(0.01, 0.9, len(y_te)),
            "p_raw": np.linspace(0.01, 0.9, len(y_te)),
            "size_only": test["size_at_decision"].to_numpy(),
            "y_tr": train["escalated"].to_numpy().astype(int),
            "y_te": y_te,
            "columns": [],
            "importances": np.array([]),
            "calibrated": True,
        }

    monkeypatch.setattr(escalation, "_fit_and_predict", spy)
    monkeypatch.setattr(config, "MODELS", tmp_path)

    res = escalation.spatial_backtest(_table(900), min_test_positives=1, n_boot=20)

    assert res.folds, "no fold ran"
    for fold, (train_agencies, test_agencies) in zip(res.folds, seen):
        assert fold.holdout_agency not in train_agencies
        assert test_agencies == {fold.holdout_agency}


def test_thin_folds_are_skipped_rather_than_reported_as_noise(monkeypatch, tmp_path):
    """A fold with three positives yields a meaningless PR-AUC. Say so, loudly."""
    monkeypatch.setattr(config, "MODELS", tmp_path)
    df = _table(1200)
    thin = df.filter(pl.col("agency_code") == "SK").head(4)
    df = df.filter(pl.col("agency_code").is_in(["AB", "BC"])).vstack(thin)

    res = escalation.spatial_backtest(
        df, agencies=["AB", "BC", "SK", "YT"], min_test_positives=5, n_boot=20
    )

    reported = {f.holdout_agency for f in res.folds}
    assert "SK" not in reported and "YT" not in reported
    reasons = {s["agency"]: s["reason"] for s in res.skipped}
    assert "fewer than 5 positives" in reasons["SK"]
    assert reasons["YT"] == "no test fires"


def test_every_fold_needs_a_fold_to_compare_against(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS", tmp_path)
    with pytest.raises(RuntimeError, match="no agency cleared"):
        escalation.spatial_backtest(_table(600), min_test_positives=10_000_000, n_boot=10)


def test_spatial_backtest_runs_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS", tmp_path)
    res = escalation.spatial_backtest(_table(3000), min_test_positives=5, n_boot=25)

    assert res.mode.startswith("region blocked")
    assert res.pooled["n_test"] == sum(f.n_test for f in res.folds)
    # The filename records which blocking mode ran, so the strict and pooled
    # runs cannot overwrite each other.
    assert (tmp_path / "escalation_spatial_backtest_region.json").exists()
    # Signal is behavioural in this fixture, so an out-of-region model should
    # still clear the base rate by a wide margin.
    assert res.pooled["pr_auc_model"] > 3 * res.pooled["pr_auc_prevalence"]


# --- reproducibility --------------------------------------------------------


def test_row_order_does_not_change_the_score():
    """The bug this guards: adding an upstream join moved PR-AUC by ~0.03.

    Row order is not a modelling choice, but LightGBM's row subsampling and
    the calibration tie-break both read row positions. Before `_deterministic`
    the same table in a different order scored anywhere from 0.258 to 0.296 on
    the real data -- a spread wider than any feature effect being measured.
    """
    from sklearn.metrics import average_precision_score

    df = _table(1500, seed=11)
    train = df.filter(pl.col("fire_year") < 2025)
    test = df.filter(pl.col("fire_year") == 2025)

    scores = []
    for seed in (0, 1, 2):
        shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
        fit = escalation._fit_and_predict(shuffled, test)
        scores.append(round(float(average_precision_score(fit["y_te"], fit["p"])), 10))

    assert len(set(scores)) == 1, f"row order changed the score: {scores}"


def test_deterministic_imposes_a_total_order():
    df = _table(200, seed=5).sample(fraction=1.0, shuffle=True, seed=9)
    a = escalation._deterministic(df)
    b = escalation._deterministic(df.sample(fraction=1.0, shuffle=True, seed=4))
    assert a["national_fire_id"].to_list() == b["national_fire_id"].to_list()


# --- bootstrap --------------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(1500) < 0.05).astype(int)
    p = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.2, 1500), 0, 1)
    base = np.clip(p + rng.normal(0, 0.3, 1500), 0, 1)

    from sklearn.metrics import average_precision_score

    point = average_precision_score(y, p)
    ci = escalation._bootstrap(y, p, base, n_boot=200)

    lo, hi = ci["pr_auc_model"]
    assert lo <= point <= hi
    assert lo < hi
    assert ci["percentiles"] == [5, 95]
    assert 0.0 <= ci["p_model_beats_size"] <= 1.0


def test_bootstrap_delta_is_paired():
    """A predictor identical to the baseline must show zero difference."""
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.08).astype(int)
    p = np.clip(0.08 + 0.4 * y + rng.normal(0, 0.25, 800), 0, 1)

    ci = escalation._bootstrap(y, p, p.copy(), n_boot=100)
    assert ci["pr_auc_delta"] == [0.0, 0.0]
    assert ci["p_model_beats_size"] == 0.0
