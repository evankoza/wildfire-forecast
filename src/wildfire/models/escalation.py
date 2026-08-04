"""Escalation model: will this newly-reported fire be large in three days?

Evaluation choices worth defending:

* **Season-blocked split.** Test years are held out whole. A random split
  would put revisions of the same fire, and neighbouring fires burning in the
  same weather, on both sides of the divide and return a flattering number
  that means nothing operationally.
* **Region-blocked split, as a second axis.** Holding out a season answers
  "does it work next year"; it does not answer "did it just memorise
  Alberta". `spatial_backtest` holds out an entire reporting agency, so the
  test fires are in country the model has never seen. `lat`, `lon` and
  `region_code` carry heavy gain, so this is the split that decides whether
  the model learned fire behaviour or geography.
* **PR-AUC over ROC-AUC.** Escalations are a small minority; ROC-AUC is
  dominated by the easy negatives and stays high even for a useless model.
* **Two baselines, not zero.** Prevalence (predict the base rate) and
  size-at-decision alone. A gradient-booster that cannot beat "how big is it
  already" has learned nothing worth deploying, and that comparison is the
  first thing a reviewer should see.
* **Intervals, not point estimates.** ~117 positives carry a wide interval.
  Every PR-AUC here comes with a percentile bootstrap, and the model-vs-
  baseline comparison is bootstrapped *paired* on the same resample, because
  overlapping marginal intervals do not mean two scores are indistinguishable.
* **Calibration is reported, not assumed.** These scores would be read as
  probabilities by anyone allocating crews, so Brier score and a reliability
  table are part of the result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

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
    # `percent_contained` is deliberately absent. The feed reports it for 0.6%
    # of fires and with two distinct values, so LightGBM never split on it:
    # `ablation()` measured the difference at exactly 0.0000 PR-AUC with a
    # [0, 0] paired interval -- the two models' predictions were bit-identical.
    # It stays in the modelling table as a diagnostic and out of the feature
    # contract, on the same principle as `ciffc_sitrep_lag_hours`.
    # `severity_nearest_dsr` never reached the table at all; it is nulled at
    # ingest for the same reason and `assemble_features` does not carry it.
    "n_revisions_by_decision",
    "n_distinct_status",
    "hs_count",
    "hs_hfi_max",
    "hs_hfi_mean",
    "hs_fwi_max",
    "hs_fwi_mean",
    "hs_ros_max",
    "hs_ros_mean",
    # `estarea` and `bfc` are deliberately absent: the season archives do not
    # carry them, so they are dropped at ingest to keep the feature schema
    # identical across seasons however they were fetched.
    "hs_sfc_mean",
    "hs_tfc_max",
    "hs_tfc_mean",
    "hs_dist_min_km",
    "hs_active_days",
    "hs_detection_lead_hours",
    "lat",
    "lon",
    "month",
    "doy",
    "t0_hour",
    # CIFFC situation report, as published before the decision instant. These
    # are human judgements about how stretched suppression is -- the one class
    # of covariate with no machine-feed equivalent.
    "ciffc_national_pl",
    "ciffc_agency_pl",
    "ciffc_prep_hazard",
    "ciffc_prep_current_load",
    "ciffc_prep_expected_load",
    "ciffc_prep_resource_levels",
    "ciffc_prep_resource_availability",
    "ciffc_occurrence_pred_lightning",
    "ciffc_occurrence_pred_human",
    # `ciffc_sitrep_lag_hours` is built and kept in the modelling table, but
    # deliberately not a feature. It measures how stale the sitrep was, which
    # is a property of the reporting calendar rather than of the fire, and it
    # is the only CIFFC variant that scored *below* the no-CIFFC baseline
    # (-0.007 PR-AUC). Useful as a diagnostic, not as evidence.
]

# Features that encode *where* rather than *how a fire behaves*. Dropping them
# is the natural follow-up if the region-blocked backtest shows the model is
# leaning on geography: `spatial_backtest(..., drop_geography=True)` refits
# without them so the two numbers are directly comparable.
GEOGRAPHIC = ["lat", "lon", "agency_code", "region_code"]

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

N_BOOT = 1000
PCTILES = (5, 95)


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


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
    # Percentile bootstrap over the test fires; see `_bootstrap`.
    ci: dict = field(default_factory=dict)


@dataclass
class FoldResult:
    """One leave-one-agency-out fold."""

    holdout_agency: str
    n_train: int
    n_test: int
    n_positives_test: int
    positive_rate_train: float
    positive_rate_test: float
    pr_auc_model: float
    pr_auc_size_only: float
    pr_auc_prevalence: float
    lift_over_size_baseline: float
    roc_auc_model: float
    brier_model: float
    brier_prevalence: float
    calibrated: bool
    ci: dict


@dataclass
class SpatialResult:
    mode: str
    train_years: list[int]
    test_years: list[int]
    agencies: list[str]
    skipped: list[dict]
    folds: list[FoldResult]
    pooled: dict
    macro: dict
    dropped_features: list[str]
    trained_at: str


def _feature_frame(df: pl.DataFrame, *, categories: dict | None = None,
                   drop: list[str] | None = None):
    """Feature matrix, plus the category levels it was built with.

    `categories` pins a test frame to the *training* frame's category levels.
    Without it pandas assigns integer codes per frame, so a level missing from
    one side silently shifts every code after it -- and in a region-blocked
    fold the held-out agency is absent from training by construction. Levels
    unseen in training become NaN, which LightGBM handles as missing.
    """
    import pandas as pd

    excluded = set(drop or ())
    cols = [c for c in NUMERIC + CATEGORICAL if c in df.columns and c not in excluded]
    leaked = FORBIDDEN & set(cols)
    if leaked:
        raise RuntimeError(f"label leakage: {leaked} must not be features")

    pdf = df.select(cols).to_pandas()
    levels: dict[str, object] = {}
    for c in CATEGORICAL:
        if c not in pdf.columns:
            continue
        if categories is None:
            pdf[c] = pdf[c].astype("category")
        else:
            # Mask levels the training frame never saw before constructing the
            # categorical: they become missing, which is the truthful encoding
            # of "no evidence" and is what LightGBM already knows how to route.
            # Done explicitly because passing unknown values straight to
            # `pd.Categorical(..., categories=...)` is deprecated in pandas 3.
            known = pdf[c].isin(list(categories[c]))
            pdf[c] = pd.Categorical(pdf[c].where(known), categories=categories[c])
        levels[c] = pdf[c].cat.categories
    return pdf, levels


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


def _bootstrap(
    y: np.ndarray,
    p_model: np.ndarray,
    p_base: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 17,
) -> dict:
    """Percentile bootstrap of PR-AUC over the test fires.

    Resamples *fires* with replacement, which is the right unit: the question
    an interval answers here is "how much of this number is the particular
    handful of escalations we happened to draw this season".

    Model and baseline are scored on the *same* resample so their difference
    can be bootstrapped paired. That matters: marginal intervals on two
    correlated statistics routinely overlap while the paired difference is
    comfortably away from zero, and reading overlap as "no difference" would
    understate the result.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    model, base, delta = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:  # AP is undefined with no positives; skip the draw
            continue
        m = float(average_precision_score(yb, p_model[idx]))
        b = float(average_precision_score(yb, p_base[idx]))
        model.append(m)
        base.append(b)
        delta.append(m - b)

    if not model:
        return {"n_boot": 0, "note": "no resample contained a positive"}

    lo, hi = PCTILES

    def pct(v):
        return [round(float(np.percentile(v, lo)), 4), round(float(np.percentile(v, hi)), 4)]

    return {
        "n_boot": len(model),
        "percentiles": list(PCTILES),
        "pr_auc_model": pct(model),
        "pr_auc_size_only": pct(base),
        "pr_auc_delta": pct(delta),
        "p_model_beats_size": round(float(np.mean(np.asarray(delta) > 0)), 3),
    }


def _deterministic(df: pl.DataFrame) -> pl.DataFrame:
    """Impose a total order on rows before anything reads their positions.

    Row order is not a modelling choice, but three things silently depend on
    it: LightGBM's row subsampling, the tie-break inside the calibration
    split, and `argsort` itself. Left alone they make the score a function of
    how the parquet happened to be laid out -- adding a join upstream reorders
    rows and the headline number moves by ~0.03 PR-AUC with no change to the
    data at all. `(t0, national_fire_id)` is a total order because fire ids
    are unique, so the result is now reproducible across rebuilds.
    """
    keys = [c for c in ("t0", "national_fire_id") if c in df.columns]
    return df.sort(keys) if keys else df


def _fit_and_predict(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    drop: list[str] | None = None,
    seed: int = 17,
) -> dict:
    """Fit unweighted, isotonic-calibrate on a temporal tail, score the test set.

    `class_weight="balanced"` helps the trees split on a rare class but
    destroys the probability scale -- it optimises ranking at the cost of
    calibration, and these scores would be read as probabilities by anyone
    deciding where to send a crew. So: fit unweighted on the earlier part of
    the training window, then isotonic-calibrate on the later part. The
    calibration slice is temporal, never random, for the same reason the outer
    split is.
    """
    import lightgbm as lgb

    train, test = _deterministic(train), _deterministic(test)

    X_tr, levels = _feature_frame(train, drop=drop)
    X_te, _ = _feature_frame(test, categories=levels, drop=drop)
    y_tr = train[TARGET].to_numpy().astype(int)
    y_te = test[TARGET].to_numpy().astype(int)

    if y_tr.sum() == 0 or y_te.sum() == 0:
        raise RuntimeError(
            f"no positive examples in a split (train={y_tr.sum()}, test={y_te.sum()})"
        )

    cat_cols = [c for c in CATEGORICAL if c in X_tr.columns]
    # Stable, so fires sharing a t0 keep the total order imposed above rather
    # than whatever quicksort happens to do with them.
    order = np.argsort(train["t0"].to_numpy(), kind="stable")
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
        random_state=seed,
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

    size_only = np.nan_to_num(
        test["size_at_decision"].fill_null(0).to_numpy(), nan=0.0
    ).astype(float)

    return {
        "model": {"base": base, "calibrator": calibrator},
        "p": p,
        "p_raw": p_raw,
        "size_only": size_only,
        "y_tr": y_tr,
        "y_te": y_te,
        "columns": list(X_tr.columns),
        "importances": base.feature_importances_.astype(float),
        "calibrated": calibrator is not None,
        # Needed to score anything later. Rebuilding a feature frame without
        # these lets pandas assign category codes per frame, so a level absent
        # at serving time shifts every code after it and the model silently
        # reads one agency as another.
        "levels": levels,
        # The *reordered* frames. `_deterministic` runs on locals, so the
        # caller's copies are still in their original order -- anything that
        # lines a score vector up against a row must use these, not the frames
        # it passed in, or every prediction is attached to the wrong fire.
        "train_ordered": train,
        "test_ordered": test,
    }


def _years(df: pl.DataFrame) -> list[int]:
    if "fire_year" not in df.columns:
        raise RuntimeError("modelling table needs fire_year for the temporal split")
    return sorted(y for y in df["fire_year"].unique().to_list() if y is not None)


def train_and_backtest(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    n_boot: int = N_BOOT,
    spec=SPEC,
) -> Result:
    years = _years(df)
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

    fit = _fit_and_predict(train, test)
    p, p_raw, y_tr, y_te = fit["p"], fit["p_raw"], fit["y_tr"], fit["y_te"]

    prevalence = float(y_tr.mean())
    p_prev = np.full_like(p, prevalence, dtype=float)

    pr_model = float(average_precision_score(y_te, p))
    pr_size = float(average_precision_score(y_te, fit["size_only"]))
    pr_prev = float(y_te.mean())  # AP of a constant predictor is the base rate

    imp = sorted(
        zip(fit["columns"], fit["importances"]),
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
        trained_at=_now(),
        ci=_bootstrap(y_te, p, fit["size_only"], n_boot=n_boot),
    )

    out = config.MODELS / "escalation_backtest.json"
    out.write_text(json.dumps({"spec": asdict(spec), "result": asdict(res)}, indent=2))

    import joblib

    joblib.dump(
        {**fit["model"], "levels": fit["levels"], "columns": fit["columns"],
         "train_years": train_years, "trained_at": res.trained_at},
        config.MODELS / "escalation_lgbm.joblib",
    )

    # Persist the test-set scores. The bucketed `reliability` in the JSON is
    # enough for a table, but a reliability *curve* and a PR curve need the raw
    # vectors, and refitting just to draw them would be wasteful and would risk
    # drawing a different model than the one reported. Aligned against
    # `test_ordered`, not `test` -- see the note in `_fit_and_predict`.
    ordered = fit["test_ordered"]
    pl.DataFrame(
        {
            "national_fire_id": ordered["national_fire_id"],
            "fire_year": ordered["fire_year"],
            "agency_code": ordered["agency_code"],
            "y": y_te,
            "p": p,
            "p_uncalibrated": p_raw,
            "size_at_decision": fit["size_only"],
        }
    ).write_parquet(config.MODELS / "escalation_test_predictions.parquet")

    log.info("backtest written -> %s", out)
    return res


def fit_final(df: pl.DataFrame, *, spec=SPEC, seed: int = 17) -> dict:
    """Fit on every labelled season, for scoring fires that are burning now.

    The backtest model is deliberately handicapped -- it is trained on
    2023-24 so that 2025 can be held out honestly. Nothing should be *scored*
    with it: by the time you are forecasting a live fire, last season is
    history and withholding it is throwing away a third of the data for no
    reason. So the reported numbers come from the backtest model and the
    predictions come from this one, refit on the same pipeline.

    The split between "the model I measured" and "the model I deploy" is the
    honest arrangement, but it does mean the deployed model's accuracy is
    *inferred* from the backtest rather than directly observed.
    """
    import joblib

    labelled = df.filter(pl.col(TARGET).is_not_null())
    if labelled.is_empty():
        raise RuntimeError("no labelled rows to fit on")

    years = _years(labelled)
    ordered = _deterministic(labelled)

    y = ordered[TARGET].to_numpy().astype(int)
    if y.sum() < 20:
        raise RuntimeError(f"only {y.sum()} positives; refusing to fit a final model")

    fit = _fit_and_predict(ordered, ordered, seed=seed)
    payload = {
        **fit["model"],
        "levels": fit["levels"],
        "columns": fit["columns"],
        "train_years": years,
        "n_train": labelled.height,
        "n_positives": int(y.sum()),
        "spec": asdict(spec),
        "trained_at": _now(),
        "note": "fitted on all labelled seasons; accuracy is inferred from the "
                "held-out backtest, not measured on these rows",
    }
    joblib.dump(payload, config.MODELS / "escalation_final.joblib")
    log.info("final model fitted on %s (%s fires, %s positives)",
             years, labelled.height, int(y.sum()))
    return payload


CIFFC_CORE = ["ciffc_national_pl", "ciffc_agency_pl"]

FEATURE_SETS = {
    "full": [],
    "no CIFFC at all": ["ciffc"],
    "national + agency PL only": ["ciffc_detail"],
    "no satellite hotspots": ["hotspots"],
    "no geography": ["geography"],
}


def _drop_for(groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        if g == "ciffc":
            out += [c for c in NUMERIC if c.startswith("ciffc_")]
        elif g == "ciffc_detail":
            out += [c for c in NUMERIC if c.startswith("ciffc_") and c not in CIFFC_CORE]
        elif g == "hotspots":
            out += [c for c in NUMERIC + CATEGORICAL if c.startswith("hs_")]
        elif g == "geography":
            out += GEOGRAPHIC
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return sorted(set(out))


def ablation(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    n_boot: int = N_BOOT,
    sets: dict[str, list[str]] | None = None,
    seed: int = 17,
) -> dict:
    """Which blocks of features earn their place, paired on the same resample.

    The version of this that lived in a notebook compared point estimates
    across separately-fitted models, and the README carried the result with a
    warning that its absolute numbers predated the determinism fix. This is
    the same question asked properly: every variant is refit on the same
    split, and the *difference* between two score vectors is bootstrapped over
    the same resampled test fires.

    That pairing is what makes the table readable. With ~117 positives the
    marginal interval on any one PR-AUC swallows every other variant's point
    estimate, so a table of marginal intervals says "no difference" whatever
    is true. The paired difference removes the variance the two share.
    """
    sets = sets or FEATURE_SETS
    years = _years(df)
    if test_years is None:
        test_years = [years[-1]]
    cutoff = min(test_years)
    train = df.filter(pl.col("fire_year") < cutoff)
    test = df.filter(pl.col("fire_year").is_in(test_years))
    if train.is_empty() or test.is_empty():
        raise RuntimeError(f"empty split: train={train.height} test={test.height}")

    scores: dict[str, np.ndarray] = {}
    y = size_only = None
    for name, groups in sets.items():
        drop = _drop_for(groups) or None
        fit = _fit_and_predict(train, test, drop=drop, seed=seed)
        scores[name] = fit["p"]
        y, size_only = fit["y_te"], fit["size_only"]
        log.info("ablation %-26s PR-AUC %.4f", name,
                 average_precision_score(y, fit["p"]))

    reference = next(iter(sets))
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {k: [] for k in sets}
    deltas = {k: [] for k in sets if k != reference}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        vals = {k: float(average_precision_score(y[idx], v[idx]))
                for k, v in scores.items()}
        for k, v in vals.items():
            draws[k].append(v)
        for k in deltas:
            deltas[k].append(vals[k] - vals[reference])

    lo, hi = PCTILES

    def pct(v):
        return [round(float(np.percentile(v, lo)), 4),
                round(float(np.percentile(v, hi)), 4)]

    rows = []
    for name in sets:
        row = {
            "feature_set": name,
            "dropped": _drop_for(sets[name]),
            "pr_auc": round(float(average_precision_score(y, scores[name])), 4),
            "interval": pct(draws[name]),
        }
        if name != reference:
            d = np.asarray(deltas[name])
            row["delta_vs_full"] = round(float(np.mean(d)), 4)
            row["delta_interval"] = pct(d)
            row["p_full_is_better"] = round(float(np.mean(d < 0)), 3)
        rows.append(row)

    out = {
        "reference": reference,
        "test_years": test_years,
        "n_test": test.height,
        "n_positives_test": int(y.sum()),
        "pr_auc_size_only": round(float(average_precision_score(y, size_only)), 4),
        "n_boot": len(draws[reference]),
        "percentiles": list(PCTILES),
        "rows": rows,
        "trained_at": _now(),
    }
    dest = config.MODELS / "escalation_ablation.json"
    dest.write_text(json.dumps(out, indent=2))
    log.info("escalation ablation written -> %s", dest)
    return out


def geography_paired(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    min_test_positives: int = 10,
    n_boot: int = N_BOOT,
    seed: int = 17,
) -> dict:
    """Does dropping lat/lon/agency/region help or hurt out of region?

    Runs the leave-one-agency-out backtest twice -- with geography and
    without -- and bootstraps the difference *paired on the pooled
    out-of-region fires*, which is the only comparison that answers the
    question. The two runs score the same fires in the same fold order, so the
    vectors line up row for row.

    The direction of this result was never in doubt; the interval was, because
    the figure the README carried predated the feature-determinism fix.
    """
    with_geo = spatial_backtest(
        df, test_years=test_years, min_test_positives=min_test_positives,
        drop_geography=False, n_boot=0,
    )
    without = spatial_backtest(
        df, test_years=test_years, min_test_positives=min_test_positives,
        drop_geography=True, n_boot=0,
    )
    if with_geo.agencies != without.agencies:
        raise RuntimeError(
            "the two runs kept different folds, so their pooled vectors do not "
            f"line up: {with_geo.agencies} vs {without.agencies}"
        )

    y = with_geo.pooled_predictions["y"]
    p_geo = with_geo.pooled_predictions["p"]
    p_nogeo = without.pooled_predictions["p"]
    if not np.array_equal(y, without.pooled_predictions["y"]):
        raise RuntimeError("fold order differs between the two runs")

    rng = np.random.default_rng(seed)
    n = len(y)
    delta = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        delta.append(
            float(average_precision_score(y[idx], p_geo[idx]))
            - float(average_precision_score(y[idx], p_nogeo[idx]))
        )

    lo, hi = PCTILES
    out = {
        "agencies": with_geo.agencies,
        "n_pooled": int(n),
        "n_positives": int(y.sum()),
        "pr_auc_with_geography": round(float(average_precision_score(y, p_geo)), 4),
        "pr_auc_without_geography": round(float(average_precision_score(y, p_nogeo)), 4),
        "dropped": list(GEOGRAPHIC),
        "paired_delta": round(float(np.mean(delta)), 4) if delta else None,
        "percentiles": list(PCTILES),
        "delta_interval": (
            [round(float(np.percentile(delta, lo)), 4),
             round(float(np.percentile(delta, hi)), 4)] if delta else None
        ),
        "p_geography_helps": round(float(np.mean(np.asarray(delta) > 0)), 3) if delta else None,
        "n_boot": len(delta),
        "trained_at": _now(),
    }
    dest = config.MODELS / "escalation_geography_paired.json"
    dest.write_text(json.dumps(out, indent=2))
    log.info("paired geography comparison written -> %s", dest)
    return out


def spatial_backtest(
    df: pl.DataFrame,
    *,
    agencies: list[str] | None = None,
    test_years: list[int] | None = None,
    min_test_positives: int = 10,
    drop_geography: bool = False,
    n_boot: int = N_BOOT,
    spec=SPEC,
) -> SpatialResult:
    """Leave-one-agency-out: can it forecast fires in country it has never seen?

    The season-blocked backtest holds out *when*, never *where*, so it cannot
    separate "learned fire behaviour" from "memorised Alberta". Here each fold
    trains with one reporting agency removed entirely and tests on that
    agency's fires. `lat`, `lon`, `agency_code` and `region_code` are kept but
    are worthless-to-misleading on the held-out region by construction -- the
    held-out agency's category level is unseen in training and arrives as
    missing, and its lat/lon box sits outside the training range. That is the
    point of the test, not a flaw in it. Pass `drop_geography=True` to refit
    without them and see what transfers.

    Two blocking modes:

    * `test_years=None` -- region-blocked only. Every season appears on both
      sides of the split, so a fire in Saskatchewan and a fire in Manitoba
      burning under the same synoptic ridge can land in train and test
      respectively. That shared-weather confound is real and flatters the
      result; the mode exists because it is the only one with enough positives
      per fold to say anything at all.
    * `test_years=[...]` -- region *and* season blocked. Strictly honest, and
      thin: most agencies do not clear `min_test_positives` in a single
      season. Folds that do not are skipped rather than reported as noise.
    """
    years = _years(df)
    if test_years:
        cutoff = min(test_years)
        train_years = [y for y in years if y < cutoff]
        if not train_years:
            raise RuntimeError(f"no seasons before {cutoff} to train on; have {years}")
        mode = "region+season blocked"
    else:
        train_years = test_years = years
        mode = "region blocked (seasons shared)"

    train_pool = df.filter(pl.col("fire_year").is_in(train_years))
    test_pool = df.filter(pl.col("fire_year").is_in(test_years))

    counts = (
        test_pool.group_by("agency_code")
        .agg(pl.len().alias("n"), pl.col(TARGET).sum().alias("pos"))
        .sort("pos", descending=True)
    )
    available = {r["agency_code"]: r for r in counts.iter_rows(named=True)}

    if agencies is None:
        agencies = [a for a, r in available.items() if r["pos"] >= min_test_positives]
    agencies = [a for a in agencies if a is not None]

    drop = list(GEOGRAPHIC) if drop_geography else None

    folds: list[FoldResult] = [];  skipped: list[dict] = []
    pooled_y, pooled_p, pooled_size = [], [], []

    for agency in agencies:
        row = available.get(agency)
        if row is None:
            skipped.append({"agency": agency, "reason": "no test fires"})
            continue
        if row["pos"] < min_test_positives:
            skipped.append({
                "agency": agency, "n_test": row["n"], "positives": row["pos"],
                "reason": f"fewer than {min_test_positives} positives",
            })
            continue

        train = train_pool.filter(pl.col("agency_code") != agency)
        test = test_pool.filter(pl.col("agency_code") == agency)

        fit = _fit_and_predict(train, test, drop=drop)
        p, y_te, y_tr = fit["p"], fit["y_te"], fit["y_tr"]
        size_only = fit["size_only"]

        pr_model = float(average_precision_score(y_te, p))
        pr_size = float(average_precision_score(y_te, size_only))
        prevalence = float(y_tr.mean())

        folds.append(
            FoldResult(
                holdout_agency=agency,
                n_train=train.height,
                n_test=test.height,
                n_positives_test=int(y_te.sum()),
                positive_rate_train=round(prevalence, 4),
                positive_rate_test=round(float(y_te.mean()), 4),
                pr_auc_model=round(pr_model, 4),
                pr_auc_size_only=round(pr_size, 4),
                pr_auc_prevalence=round(float(y_te.mean()), 4),
                lift_over_size_baseline=(
                    round(pr_model / pr_size, 3) if pr_size > 0 else float("nan")
                ),
                roc_auc_model=round(float(roc_auc_score(y_te, p)), 4),
                brier_model=round(float(brier_score_loss(y_te, p)), 4),
                brier_prevalence=round(
                    float(brier_score_loss(y_te, np.full_like(p, prevalence))), 4
                ),
                calibrated=fit["calibrated"],
                ci=_bootstrap(y_te, p, size_only, n_boot=n_boot),
            )
        )
        pooled_y.append(y_te); pooled_p.append(p); pooled_size.append(size_only)
        log.info("fold %s: PR-AUC %.4f vs size baseline %.4f (n=%s, pos=%s)",
                 agency, pr_model, pr_size, test.height, int(y_te.sum()))

    if not folds:
        raise RuntimeError(
            "no agency cleared the positive threshold; lower --min-positives "
            "or drop --test-years to pool seasons"
        )

    # Pooled: every fire scored by a model that never saw its region. Folds
    # have different prevalences so this is not a plain average -- it is the
    # single number for "how well does this work out-of-region", weighted by
    # where the fires actually are.
    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_p)
    s_all = np.concatenate(pooled_size)
    pr_all = float(average_precision_score(y_all, p_all))
    pr_all_size = float(average_precision_score(y_all, s_all))

    pooled = {
        "n_test": int(len(y_all)),
        "n_positives": int(y_all.sum()),
        "pr_auc_model": round(pr_all, 4),
        "pr_auc_size_only": round(pr_all_size, 4),
        "pr_auc_prevalence": round(float(y_all.mean()), 4),
        "lift_over_size_baseline": round(pr_all / pr_all_size, 3) if pr_all_size > 0 else None,
        "roc_auc_model": round(float(roc_auc_score(y_all, p_all)), 4),
        "brier_model": round(float(brier_score_loss(y_all, p_all)), 4),
        "ci": _bootstrap(y_all, p_all, s_all, n_boot=n_boot),
        "reliability": _reliability(y_all, p_all),
    }
    macro = {
        "pr_auc_model": round(float(np.mean([f.pr_auc_model for f in folds])), 4),
        "pr_auc_size_only": round(float(np.mean([f.pr_auc_size_only for f in folds])), 4),
        "median_lift": round(float(np.median([f.lift_over_size_baseline for f in folds])), 3),
        "folds_beating_baseline": int(sum(f.pr_auc_model > f.pr_auc_size_only for f in folds)),
        "n_folds": len(folds),
    }

    res = SpatialResult(
        mode=mode,
        train_years=train_years,
        test_years=test_years,
        agencies=[f.holdout_agency for f in folds],
        skipped=skipped,
        folds=folds,
        pooled=pooled,
        macro=macro,
        dropped_features=list(drop or ()),
        trained_at=_now(),
    )

    # Both blocking modes and both feature sets write here, so the filename
    # has to say which run it was -- otherwise the strict run silently
    # overwrites the pooled one and the README ends up quoting a mix.
    tag = "_region_season" if test_years != train_years else "_region"
    if drop_geography:
        tag += "_nogeo"
    out = config.MODELS / f"escalation_spatial_backtest{tag}.json"
    out.write_text(json.dumps({"spec": asdict(spec), "result": asdict(res)}, indent=2))
    log.info("spatial backtest written -> %s", out)

    # The pooled score vectors, in fold order, for callers that need to pair
    # two runs against each other. Attached rather than declared as a field so
    # that `asdict(res)` -- which is what writes the JSON above -- never sees
    # a numpy array.
    res.pooled_predictions = {"y": y_all, "p": p_all, "size_only": s_all}
    return res
