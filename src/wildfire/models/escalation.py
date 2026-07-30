"""Escalation model: will this newly-reported fire be large in three days?

Evaluation choices worth defending:

* **Season-blocked split.** Test years are held out whole. A random split
  would put revisions of the same fire, and neighbouring fires burning in the
  same weather, on both sides of the divide and return a flattering number
  that means nothing operationally.
* **PR-AUC over ROC-AUC.** Escalations are a small minority; ROC-AUC is
  dominated by the easy negatives and stays high even for a useless model.
* **Two baselines, not zero.** Prevalence (predict the base rate) and
  size-at-decision alone. A gradient-booster that cannot beat "how big is it
  already" has learned nothing worth deploying, and that comparison is the
  first thing a reviewer should see.
* **Calibration is reported, not assumed.** These scores would be read as
  probabilities by anyone allocating crews, so Brier score and a reliability
  table are part of the result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .. import config
from ..config import SPEC

log = logging.getLogger(__name__)

TARGET = "escalated"

CATEGORICAL = [
    "agency_code",
    "region_code",
    "status_at_decision",
    "response_type",
    "national_fire_cause",
    "hs_fuel_group",
]

NUMERIC = [
    "size_at_decision",
    "log_size_at_decision",
    "size_growth_to_decision",
    "size_first",
    "size_max_so_far",
    "percent_contained",
    "n_revisions_by_decision",
    "n_distinct_status",
    "hs_count",
    "hs_hfi_max",
    "hs_hfi_mean",
    "hs_fwi_max",
    "hs_fwi_mean",
    "hs_ros_max",
    "hs_ros_mean",
    "hs_estarea_sum",
    "hs_sfc_mean",
    "hs_tfc_max",
    "hs_tfc_mean",
    "hs_bfc_mean",
    "hs_dist_min_km",
    "hs_active_days",
    "hs_detection_lead_hours",
    "lat",
    "lon",
    "month",
    "doy",
    "t0_hour",
]

# Columns that encode the answer. Guarding explicitly because a leaked label
# is the failure mode that looks like success.
FORBIDDEN = {
    "size_at_horizon",
    "status_at_horizon",
    "still_out_of_control",
    "escalated",
    "t0",
    "national_fire_id",
    "n_revisions",
    "fire_year",
}


@dataclass
class Result:
    n_train: int
    n_test: int
    train_years: list[int]
    test_years: list[int]
    positive_rate_train: float
    positive_rate_test: float
    pr_auc_model: float
    pr_auc_size_only: float
    pr_auc_prevalence: float
    roc_auc_model: float
    brier_model: float
    brier_uncalibrated: float
    brier_prevalence: float
    lift_over_size_baseline: float
    top_features: list[tuple[str, float]]
    reliability: list[dict]
    trained_at: str


def _feature_frame(df: pl.DataFrame):
    import pandas as pd

    cols = [c for c in NUMERIC + CATEGORICAL if c in df.columns]
    leaked = FORBIDDEN & set(cols)
    if leaked:
        raise RuntimeError(f"label leakage: {leaked} must not be features")

    pdf = df.select(cols).to_pandas()
    for c in CATEGORICAL:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("category")
    return pdf


def _reliability(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    """Predicted vs observed frequency, by score decile-ish bucket."""
    out = []
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append(
            {
                "bucket": i + 1,
                "n": int(m.sum()),
                "mean_predicted": round(float(p[m].mean()), 4),
                "observed_rate": round(float(y[m].mean()), 4),
            }
        )
    return out


def train_and_backtest(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    spec=SPEC,
) -> Result:
    import lightgbm as lgb

    if "fire_year" not in df.columns:
        raise RuntimeError("modelling table needs fire_year for the temporal split")

    years = sorted(y for y in df["fire_year"].unique().to_list() if y is not None)
    if len(years) < 2:
        raise RuntimeError(
            f"need at least two fire years to hold one out; got {years}. "
            "Ingest more seasons before backtesting."
        )
    if test_years is None:
        test_years = [years[-1]]

    # Train strictly on the past. Holding out 2025 while training on 2026
    # would still be a "held-out" split by year, and still be nonsense: the
    # model would have seen the future of the process it is forecasting.
    cutoff = min(test_years)
    train_years = [y for y in years if y < cutoff]
    if not train_years:
        raise RuntimeError(
            f"no seasons before {cutoff} to train on; available years {years}"
        )
    dropped = [y for y in years if y > max(test_years)]
    if dropped:
        log.info("excluding future seasons from the split entirely: %s", dropped)

    train = df.filter(pl.col("fire_year").is_in(train_years))
    test = df.filter(pl.col("fire_year").is_in(test_years))
    if train.is_empty() or test.is_empty():
        raise RuntimeError(f"empty split: train={train.height} test={test.height}")

    X_tr, X_te = _feature_frame(train), _feature_frame(test)
    y_tr = train[TARGET].to_numpy().astype(int)
    y_te = test[TARGET].to_numpy().astype(int)

    if y_tr.sum() == 0 or y_te.sum() == 0:
        raise RuntimeError(
            f"no positive examples in a split (train={y_tr.sum()}, test={y_te.sum()}); "
            f"threshold {spec.size_threshold_ha} ha may be too high for this data"
        )

    # Two-stage fit. `class_weight="balanced"` helps the trees split on a rare
    # class but destroys the probability scale -- it optimises ranking at the
    # cost of calibration, and these scores would be read as probabilities by
    # anyone deciding where to send a crew. So: fit unweighted on the earlier
    # part of the training window, then isotonic-calibrate on the later part.
    # The calibration slice is temporal, never random, for the same reason the
    # outer split is.
    cat_cols = [c for c in CATEGORICAL if c in X_tr.columns]
    order = np.argsort(train["t0"].to_numpy())
    split = int(len(order) * 0.75)
    fit_idx, cal_idx = order[:split], order[split:]

    if y_tr[cal_idx].sum() < 10:
        log.warning("calibration slice has %s positives; skipping calibration",
                    int(y_tr[cal_idx].sum()))
        fit_idx, cal_idx = order, None

    base = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
        random_state=17,
    )
    base.fit(X_tr.iloc[fit_idx], y_tr[fit_idx], categorical_feature=cat_cols)
    p_raw = base.predict_proba(X_te)[:, 1]

    calibrator = None
    if cal_idx is not None:
        from sklearn.isotonic import IsotonicRegression

        # Fit the isotonic map on the base model's scores over the held-back
        # later slice, then apply it to the test scores. Done explicitly rather
        # than via CalibratedClassifierCV because that estimator's `prefit`
        # mode has changed API twice, and the mechanism is three lines anyway.
        p_cal_in = base.predict_proba(X_tr.iloc[cal_idx])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p_cal_in, y_tr[cal_idx])
        p = calibrator.predict(p_raw)
    else:
        p = p_raw

    model = {"base": base, "calibrator": calibrator}

    # Baselines.
    prevalence = float(y_tr.mean())
    p_prev = np.full_like(p, prevalence, dtype=float)
    size_only = np.nan_to_num(test["size_at_decision"].fill_null(0).to_numpy(), nan=0.0)

    pr_model = float(average_precision_score(y_te, p))
    pr_size = float(average_precision_score(y_te, size_only))
    pr_prev = float(y_te.mean())  # AP of a constant predictor is the base rate

    imp = sorted(
        zip(X_tr.columns, base.feature_importances_.astype(float)),
        key=lambda t: t[1],
        reverse=True,
    )[:15]

    res = Result(
        n_train=train.height,
        n_test=test.height,
        train_years=train_years,
        test_years=test_years,
        positive_rate_train=round(prevalence, 4),
        positive_rate_test=round(float(y_te.mean()), 4),
        pr_auc_model=round(pr_model, 4),
        pr_auc_size_only=round(pr_size, 4),
        pr_auc_prevalence=round(pr_prev, 4),
        roc_auc_model=round(float(roc_auc_score(y_te, p)), 4),
        brier_model=round(float(brier_score_loss(y_te, p)), 4),
        brier_uncalibrated=round(float(brier_score_loss(y_te, p_raw)), 4),
        brier_prevalence=round(float(brier_score_loss(y_te, p_prev)), 4),
        lift_over_size_baseline=round(pr_model / pr_size, 3) if pr_size > 0 else float("nan"),
        top_features=[(k, round(v, 1)) for k, v in imp],
        reliability=_reliability(y_te, p),
        trained_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    out = config.MODELS / "escalation_backtest.json"
    out.write_text(json.dumps({"spec": asdict(spec), "result": asdict(res)}, indent=2))

    import joblib

    joblib.dump(model, config.MODELS / "escalation_lgbm.joblib")
    log.info("backtest written -> %s", out)
    return res
