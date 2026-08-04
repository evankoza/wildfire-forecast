"""Ignition model: will a new fire be reported in this cell today?

The second of the two models the project was designed around. It shares the
escalation model's evaluation discipline -- season- and region-blocked splits,
PR-AUC against real baselines, paired bootstrap intervals, calibration
reported rather than assumed -- and adds two problems that one did not have.

**Negative sampling, and undoing it.** One cell-day in roughly two thousand
carries an ignition, so the panel keeps every positive and a `neg_rate`
sample of the rest. That makes the model fittable and every raw number it
produces wrong by a known factor, in two distinct places:

* *Probabilities* are inflated, because the sample's prevalence is not the
  population's. `correct_prior` divides the sampling rate back out of the
  odds. This is the standard case-control correction and it is exact, not an
  approximation -- but it only holds if the model's scores are calibrated on
  the sample first, so isotonic calibration runs before it, not after.
* *PR-AUC* is inflated for the same reason: precision is a function of
  prevalence. Every metric here is therefore computed with sample weights --
  1 for a positive, `1/neg_rate` for a sampled negative -- which reconstructs
  the curve the full population panel would have given. Weighting is not a
  nicety: unweighted, this model's PR-AUC reads about 0.30, and the honest
  figure is an order of magnitude lower.

**Two baselines that are actually hard.** Prevalence is not interesting here.
The comparisons that matter are the two things a fire agency already has on
the wall at 6am: the fire-weather map (`wx_fwi`, interpolated from the station
network) and the knowledge of where fires keep starting (`ig_n_365d`, the
rolling per-cell climatology). A model that cannot beat both of those has
reproduced the status quo at greater expense.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .. import config
from ..features.ignition import NEG_RATE, TARGET

log = logging.getLogger(__name__)

CATEGORICAL = ["agency_code"]

NUMERIC = [
    # Interpolated station fire weather -- the exogenous leg. `wx_dist_km` is
    # in on purpose: it is a static property of the observing network, so it
    # cannot carry anything temporal, and it is the one honest indicator of
    # how much the FWI above should be trusted in a given cell. It does also
    # proxy remoteness, which is why `--drop-network` exists to refit without
    # it and the difference is reported rather than argued about.
    "wx_temp",
    "wx_rh",
    "wx_ws",
    "wx_precip",
    "wx_ffmc",
    "wx_dmc",
    "wx_dc",
    "wx_isi",
    "wx_bui",
    "wx_fwi",
    "wx_dsr",
    "wx_n_stations",
    "wx_dist_km",
    # Satellite activity in the cell, and separately in the ring around it.
    "hs_n_1d",
    "hs_n_3d",
    "hs_n_7d",
    "hs_fwi_max_1d",
    "hs_fwi_max_3d",
    "hs_fwi_max_7d",
    "hs_hfi_max_1d",
    "hs_hfi_max_3d",
    "hs_hfi_max_7d",
    "hs_ros_max_1d",
    "hs_ros_max_3d",
    "hs_ros_max_7d",
    "hs_ring_n_1d",
    "hs_ring_n_3d",
    "hs_ring_n_7d",
    "hs_ring_hfi_max_1d",
    "hs_ring_hfi_max_3d",
    "hs_ring_hfi_max_7d",
    "hs_days_since",
    # Ignition history: the rolling climatology and its short-window echo.
    "ig_n_7d",
    "ig_n_30d",
    "ig_n_365d",
    "ig_ring_n_7d",
    "ig_ring_n_365d",
    "ig_days_since",
    # Where and when.
    "lat",
    "lon",
    "doy",
    "month",
    # CIFFC preparedness. On escalation this block did nothing; here the unit
    # of prediction is an agency-day, which is the shape the covariate has,
    # and two of these columns are an agency's own forecast of tomorrow's
    # ignition load.
    "ciffc_national_pl",
    "ciffc_agency_pl",
    "ciffc_prep_hazard",
    "ciffc_prep_current_load",
    "ciffc_prep_expected_load",
    "ciffc_prep_resource_levels",
    "ciffc_prep_resource_availability",
    "ciffc_occurrence_pred_lightning",
    "ciffc_occurrence_pred_human",
]

GEOGRAPHIC = ["lat", "lon", "agency_code"]
NETWORK = ["wx_dist_km", "wx_n_stations"]

# Columns that encode the answer, or the row's identity.
FORBIDDEN = {TARGET, "n_ignitions", "pid", "day", "fire_year", "cell_x", "cell_y"}

N_BOOT = 1000
PCTILES = (5, 95)

BASELINES = {
    "fwi": "wx_fwi",
    "climatology": "ig_n_365d",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def correct_prior(p: np.ndarray, neg_rate: float) -> np.ndarray:
    """Undo negative subsampling on the probability scale.

    Keeping every positive and a fraction `r` of the negatives multiplies the
    odds of the positive class by 1/r. Dividing it back out is exact:

        odds_population = odds_sample * r

    Applied *after* isotonic calibration, never before -- the correction
    assumes the input is a calibrated probability on the sampled distribution,
    and applying it to a raw score would just move an uncalibrated number
    somewhere else.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    odds = p / (1.0 - p) * neg_rate
    return odds / (1.0 + odds)


def sample_weights(y: np.ndarray, neg_rate: float) -> np.ndarray:
    """One per positive, 1/r per sampled negative.

    Every metric in this module takes these. A sampled negative stands in for
    1/r population cell-days, and a precision computed without saying so is
    the precision of a world with fifty times fewer quiet days than exist.
    """
    return np.where(np.asarray(y) > 0, 1.0, 1.0 / neg_rate)


@dataclass
class IgnitionResult:
    n_train: int
    n_test: int
    train_years: list[int]
    test_years: list[int]
    neg_rate: float
    n_positives_train: int
    n_positives_test: int
    population_prevalence: float
    pr_auc_model: float
    pr_auc_fwi: float
    pr_auc_climatology: float
    pr_auc_prevalence: float
    lift_over_fwi: float
    lift_over_climatology: float
    roc_auc_model: float
    brier_model: float
    brier_prevalence: float
    top_features: list[tuple[str, float]]
    reliability: list[dict]
    dropped_features: list[str]
    trained_at: str
    ci: dict = field(default_factory=dict)


@dataclass
class IgnitionFold:
    holdout_agency: str
    n_train: int
    n_test: int
    n_positives_test: int
    pr_auc_model: float
    pr_auc_fwi: float
    pr_auc_climatology: float
    pr_auc_prevalence: float
    lift_over_fwi: float
    roc_auc_model: float
    ci: dict


@dataclass
class IgnitionSpatialResult:
    mode: str
    train_years: list[int]
    test_years: list[int]
    neg_rate: float
    folds: list[IgnitionFold]
    skipped: list[dict]
    pooled: dict
    macro: dict
    dropped_features: list[str]
    trained_at: str


def _deterministic(df: pl.DataFrame) -> pl.DataFrame:
    """Total order before anything reads a row position.

    Same reasoning as the escalation model, and the same cost if it is
    removed: LightGBM's subsampling and the calibration split both read
    positions, so without this the score is a function of how the parquet was
    laid out. `(day, cell_x, cell_y)` is total because a cell-day appears once.
    """
    keys = [c for c in ("day", "cell_x", "cell_y") if c in df.columns]
    return df.sort(keys) if keys else df


def _feature_frame(df: pl.DataFrame, *, categories: dict | None = None,
                   drop: list[str] | None = None):
    """Feature matrix plus the category levels it was built with.

    Identical contract to the escalation model's: `categories` pins a test
    frame to the training frame's levels, because pandas numbers categories
    per frame and a region-blocked fold has the held-out agency missing from
    training by construction.
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
            known = pdf[c].isin(list(categories[c]))
            pdf[c] = pd.Categorical(pdf[c].where(known), categories=categories[c])
        levels[c] = pdf[c].cat.categories
    return pdf, levels


def _weighted_reliability(y, p, w, bins: int = 5) -> list[dict]:
    """Predicted vs observed frequency, on the population scale.

    Buckets are cut on the score; the rates inside them are weighted, so
    `observed_rate` is the share of *population* cell-days in that bucket that
    ignited rather than the share of sampled ones.
    """
    out = []
    # Most cell-days score nearly zero, so several quantile edges land on the
    # same value and the naive buckets come out empty. Deduplicating the edges
    # gives fewer buckets than asked for, which is the honest outcome: the
    # score distribution genuinely has that much mass at one point.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    for i in range(len(edges) - 1):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        wm = w[m]
        out.append(
            {
                "bucket": i + 1,
                "n": int(m.sum()),
                "mean_predicted": round(float(np.average(p[m], weights=wm)), 6),
                "observed_rate": round(float(np.average(y[m], weights=wm)), 6),
            }
        )
    return out


def _ap_presorted(y, w, last_of_run) -> float:
    """Weighted average precision over arrays already in descending-score order.

    `last_of_run` marks the final index of each block of equal scores. Equal
    scores are one decision point, not several: without collapsing them a tied
    block's interior contributes precision steps that no threshold could
    actually produce.
    """
    tp = np.cumsum(w * y)
    fp = np.cumsum(w) - tp
    tp, fp = tp[last_of_run], fp[last_of_run]

    total = tp[-1]
    if total <= 0:
        return 0.0
    precision = tp / np.maximum(tp + fp, 1e-300)
    recall = tp / total
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _sorted_view(y, p, w):
    """(y, w, last_of_run, order) with everything in descending score order."""
    order = np.argsort(-np.asarray(p, dtype=float), kind="stable")
    ps = np.asarray(p, dtype=float)[order]
    return (
        np.asarray(y, dtype=float)[order],
        np.asarray(w, dtype=float)[order],
        np.append(np.diff(ps) != 0, True),
        order,
    )


def _ap(y, p, w) -> float:
    """Weighted average precision.

    Numerically identical to `sklearn.metrics.average_precision_score(y, p,
    sample_weight=w)` -- `tests/test_ignition.py` asserts that against inputs
    with heavy ties -- but written out here so the bootstrap below can skip the
    sort, which is where all the time goes.
    """
    ys, ws, runs, _ = _sorted_view(y, p, w)
    return _ap_presorted(ys, ws, runs)


def _bootstrap(y, p_model, p_base, w, *, n_boot: int = N_BOOT, seed: int = 17) -> dict:
    """Paired percentile bootstrap of the weighted PR-AUC.

    Resamples *rows of the sampled panel*, carrying their weights, which
    answers "how much of this number is the particular handful of ignitions we
    drew". It does not resample the negative sampling itself -- that is a
    second source of variance and a much smaller one, since the sample keeps
    hundreds of thousands of negatives.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # Resampling with replacement is a multinomial over the original rows, and
    # average precision depends only on the score *order* plus the weight
    # attached to each row. So drawing a row k times is exactly giving it k
    # times its weight -- which means the sort can be done once, up front, and
    # each draw is two cumulative sums. On the pooled region-blocked panel
    # (336,000 rows, a thousand draws, two score vectors) that is the
    # difference between this table taking half an hour and taking a minute.
    ym, wm, runs_m, om = _sorted_view(y, p_model, w)
    yb_, wb_, runs_b, ob = _sorted_view(y, p_base, w)
    probs = np.full(n, 1.0 / n)

    model, base, delta = [], [], []
    for _ in range(n_boot):
        counts = rng.multinomial(n, probs).astype(float)
        cm, cb = counts[om], counts[ob]
        if (ym * cm).sum() == 0:  # AP is undefined with no positives
            continue
        m = _ap_presorted(ym, wm * cm, runs_m)
        b = _ap_presorted(yb_, wb_ * cb, runs_b)
        model.append(m)
        base.append(b)
        delta.append(m - b)

    if not model:
        return {"n_boot": 0, "note": "no resample contained a positive"}

    lo, hi = PCTILES

    def pct(v):
        return [round(float(np.percentile(v, lo)), 5), round(float(np.percentile(v, hi)), 5)]

    return {
        "n_boot": len(model),
        "percentiles": list(PCTILES),
        "pr_auc_model": pct(model),
        "pr_auc_baseline": pct(base),
        "pr_auc_delta": pct(delta),
        "p_model_beats_baseline": round(float(np.mean(np.asarray(delta) > 0)), 3),
    }


def _fit_and_predict(train: pl.DataFrame, test: pl.DataFrame, *,
                     neg_rate: float, drop: list[str] | None = None,
                     seed: int = 17) -> dict:
    """Fit unweighted on the sample, isotonic-calibrate, then correct the prior.

    The order is load-bearing. Isotonic regression is fitted against the
    *sampled* labels, so it maps scores onto sampled-distribution
    probabilities; `correct_prior` then moves those onto the population scale.
    Doing it the other way round -- correcting first -- would ask the
    calibrator to fit a target it was never shown.

    No class weighting, for the reason the escalation model documents: it
    improves splits on the rare class and destroys the probability scale,
    which is the one thing these numbers are for.
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
    order = np.argsort(train["day"].to_numpy(), kind="stable")
    split = int(len(order) * 0.75)
    fit_idx, cal_idx = order[:split], order[split:]
    if y_tr[cal_idx].sum() < 25:
        log.warning("calibration slice has %s positives; skipping calibration",
                    int(y_tr[cal_idx].sum()))
        fit_idx, cal_idx = order, None

    base = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
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

        p_cal_in = base.predict_proba(X_tr.iloc[cal_idx])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p_cal_in, y_tr[cal_idx])
        p_sample = calibrator.predict(p_raw)
    else:
        p_sample = p_raw

    return {
        "model": {"base": base, "calibrator": calibrator},
        "p_sample": p_sample,
        "p": correct_prior(p_sample, neg_rate),
        "y_tr": y_tr,
        "y_te": y_te,
        "w_te": sample_weights(y_te, neg_rate),
        "columns": list(X_tr.columns),
        "importances": base.feature_importances_.astype(float),
        "levels": levels,
        "calibrated": calibrator is not None,
        "train_ordered": train,
        "test_ordered": test,
    }


def _baseline_vector(df: pl.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(df.height)
    return np.nan_to_num(df[column].fill_null(0).cast(pl.Float64).to_numpy(), nan=0.0)


def _years(df: pl.DataFrame) -> list[int]:
    if "fire_year" not in df.columns:
        raise RuntimeError("ignition table needs fire_year for the temporal split")
    return sorted(y for y in df["fire_year"].unique().to_list() if y is not None)


def train_and_backtest(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    neg_rate: float = NEG_RATE,
    n_boot: int = N_BOOT,
    drop_network: bool = False,
    drop_ciffc: bool = False,
) -> IgnitionResult:
    """Season-blocked backtest of the ignition model."""
    years = _years(df)
    if len(years) < 2:
        raise RuntimeError(f"need at least two seasons to hold one out; got {years}")
    if test_years is None:
        test_years = [years[-1]]

    cutoff = min(test_years)
    train_years = [y for y in years if y < cutoff]
    if not train_years:
        raise RuntimeError(f"no seasons before {cutoff} to train on; have {years}")

    train = df.filter(pl.col("fire_year").is_in(train_years))
    test = df.filter(pl.col("fire_year").is_in(test_years))
    if train.is_empty() or test.is_empty():
        raise RuntimeError(f"empty split: train={train.height} test={test.height}")

    drop = []
    if drop_network:
        drop += NETWORK
    if drop_ciffc:
        drop += [c for c in NUMERIC if c.startswith("ciffc_")]

    fit = _fit_and_predict(train, test, neg_rate=neg_rate, drop=drop or None)
    p, y_te, w = fit["p"], fit["y_te"], fit["w_te"]
    ordered = fit["test_ordered"]

    p_fwi = _baseline_vector(ordered, BASELINES["fwi"])
    p_clim = _baseline_vector(ordered, BASELINES["climatology"])

    # The population base rate, recovered from the weights: positives over
    # weighted total. This is the number PR-AUC has to be read against, and it
    # is roughly 1/50th of the sampled panel's positive rate.
    prevalence = float(np.average(y_te, weights=w))

    pr_model = _ap(y_te, p, w)
    pr_fwi = _ap(y_te, p_fwi, w)
    pr_clim = _ap(y_te, p_clim, w)

    imp = sorted(zip(fit["columns"], fit["importances"]), key=lambda t: t[1], reverse=True)[:15]

    # The paired interval is against the stronger of the two baselines, picked
    # on the point estimate and named in the artefact so the comparison is not
    # ambiguous when someone reads the JSON a month later.
    stronger = "climatology" if pr_clim >= pr_fwi else "fwi"
    ci = _bootstrap(y_te, p, p_clim if stronger == "climatology" else p_fwi,
                    w, n_boot=n_boot)
    ci["baseline"] = stronger

    res = IgnitionResult(
        n_train=train.height,
        n_test=test.height,
        train_years=train_years,
        test_years=test_years,
        neg_rate=neg_rate,
        n_positives_train=int(fit["y_tr"].sum()),
        n_positives_test=int(y_te.sum()),
        population_prevalence=round(prevalence, 6),
        pr_auc_model=round(pr_model, 5),
        pr_auc_fwi=round(pr_fwi, 5),
        pr_auc_climatology=round(pr_clim, 5),
        pr_auc_prevalence=round(prevalence, 6),
        lift_over_fwi=round(pr_model / pr_fwi, 3) if pr_fwi > 0 else float("nan"),
        lift_over_climatology=round(pr_model / pr_clim, 3) if pr_clim > 0 else float("nan"),
        roc_auc_model=round(float(roc_auc_score(y_te, p, sample_weight=w)), 4),
        brier_model=round(float(brier_score_loss(y_te, p, sample_weight=w)), 8),
        brier_prevalence=round(
            float(brier_score_loss(y_te, np.full_like(p, prevalence), sample_weight=w)), 8
        ),
        top_features=[(k, round(v, 1)) for k, v in imp],
        reliability=_weighted_reliability(y_te, p, w),
        dropped_features=list(drop),
        trained_at=_now(),
        ci=ci,
    )

    out = config.MODELS / "ignition_backtest.json"
    tag = ""
    if drop_network:
        tag += "_nonetwork"
    if drop_ciffc:
        tag += "_nociffc"
    if tag:
        out = config.MODELS / f"ignition_backtest{tag}.json"
    out.write_text(json.dumps({"result": asdict(res)}, indent=2))

    import joblib

    joblib.dump(
        {**fit["model"], "levels": fit["levels"], "columns": fit["columns"],
         "neg_rate": neg_rate, "train_years": train_years,
         "trained_at": res.trained_at},
        config.MODELS / "ignition_lgbm.joblib",
    )

    pl.DataFrame(
        {
            "cell_x": ordered["cell_x"],
            "cell_y": ordered["cell_y"],
            "day": ordered["day"],
            "agency_code": ordered["agency_code"],
            "y": y_te,
            "p": p,
            "w": w,
            "wx_fwi": p_fwi,
            "ig_n_365d": p_clim,
        }
    ).write_parquet(config.MODELS / "ignition_test_predictions.parquet")

    log.info("ignition backtest written -> %s", out)
    return res


FEATURE_SETS = {
    "full": [],
    "no CIFFC block": ["ciffc"],
    "no network columns": ["network"],
    "no fire weather": ["weather"],
    "no satellite history": ["hotspots"],
    "no ignition history": ["history"],
    "no geography": ["geography"],
    # The one that separates "fire weather is useless" from "fire weather is
    # redundant given where and when". Weather is a smooth function of place
    # and season, so `lat`/`lon`/`doy` can absorb most of it; dropping both
    # blocks at once is the only variant that says which it was.
    "no weather and no geography": ["weather", "geography"],
}


def _drop_for(groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        if g == "ciffc":
            out += [c for c in NUMERIC if c.startswith("ciffc_")]
        elif g == "network":
            out += NETWORK
        elif g == "weather":
            out += [c for c in NUMERIC if c.startswith("wx_")]
        elif g == "hotspots":
            out += [c for c in NUMERIC if c.startswith("hs_")]
        elif g == "history":
            out += [c for c in NUMERIC if c.startswith("ig_")]
        elif g == "geography":
            out += GEOGRAPHIC
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return sorted(set(out))


def ablation(
    df: pl.DataFrame,
    *,
    test_years: list[int] | None = None,
    neg_rate: float = NEG_RATE,
    n_boot: int = N_BOOT,
    sets: dict[str, list[str]] | None = None,
    seed: int = 17,
) -> dict:
    """Which blocks of features actually earn their place.

    **Paired, on the same resample.** Fitting each variant and comparing point
    estimates would be the obvious thing and it is not enough: these intervals
    are wide -- a couple of thousand positives -- and every variant's marginal
    interval swallows every other variant's point estimate. That reads as "no
    difference" no matter what is true. Bootstrapping the *difference* between
    two score vectors over the same resampled test rows removes the shared
    variance, which is the whole reason the escalation model's CIFFC table was
    readable at all.

    Each variant is refit from scratch on the same split, so a dropped block
    is genuinely absent rather than zeroed out.
    """
    sets = sets or FEATURE_SETS
    years = _years(df)
    if test_years is None:
        test_years = [years[-1]]
    cutoff = min(test_years)
    train = df.filter(pl.col("fire_year") < cutoff)
    test = df.filter(pl.col("fire_year").is_in(test_years))

    scores: dict[str, np.ndarray] = {}
    y = w = None
    for name, groups in sets.items():
        drop = _drop_for(groups) or None
        fit = _fit_and_predict(train, test, neg_rate=neg_rate, drop=drop, seed=seed)
        scores[name] = fit["p"]
        y, w = fit["y_te"], fit["w_te"]
        log.info("ablation %-22s PR-AUC %.5f", name, _ap(y, fit["p"], w))

    reference = next(iter(sets))
    rng = np.random.default_rng(seed)
    n = len(y)
    views = {k: _sorted_view(y, v, w) for k, v in scores.items()}
    probs = np.full(n, 1.0 / n)

    draws: dict[str, list[float]] = {k: [] for k in sets}
    deltas: dict[str, list[float]] = {k: [] for k in sets if k != reference}
    for _ in range(n_boot):
        counts = rng.multinomial(n, probs).astype(float)
        if (y * counts).sum() == 0:
            continue
        vals = {
            k: _ap_presorted(ys, ws * counts[order], runs)
            for k, (ys, ws, runs, order) in views.items()
        }
        for k, v in vals.items():
            draws[k].append(v)
        for k in deltas:
            deltas[k].append(vals[k] - vals[reference])

    lo, hi = PCTILES

    def pct(v, nd=5):
        return [round(float(np.percentile(v, lo)), nd),
                round(float(np.percentile(v, hi)), nd)]

    rows = []
    for name in sets:
        row = {
            "feature_set": name,
            "dropped": _drop_for(sets[name]),
            "pr_auc": round(_ap(y, scores[name], w), 5),
            "interval": pct(draws[name]),
        }
        if name != reference:
            d = np.asarray(deltas[name])
            row["delta_vs_full"] = round(float(np.mean(d)), 5)
            row["delta_interval"] = pct(d)
            # Read as "how often does dropping this block *cost* accuracy".
            row["p_full_is_better"] = round(float(np.mean(d < 0)), 3)
        rows.append(row)

    out = {
        "reference": reference,
        "test_years": test_years,
        "n_test": test.height,
        "n_positives_test": int(y.sum()),
        "population_prevalence": round(float(np.average(y, weights=w)), 6),
        "n_boot": len(draws[reference]),
        "percentiles": list(PCTILES),
        "rows": rows,
        "trained_at": _now(),
    }
    dest = config.MODELS / "ignition_ablation.json"
    dest.write_text(json.dumps(out, indent=2))
    log.info("ignition ablation written -> %s", dest)
    return out


def fit_final(df: pl.DataFrame, *, neg_rate: float = NEG_RATE, seed: int = 17) -> dict:
    """Refit on every season in the panel, for scoring tomorrow.

    Same arrangement as the escalation model: the measured model is
    handicapped so a season can be held out, and the deployed model is not the
    one the numbers came from.
    """
    import joblib

    ordered = _deterministic(df)
    y = ordered[TARGET].to_numpy().astype(int)
    if y.sum() < 100:
        raise RuntimeError(f"only {y.sum()} positives; refusing to fit a final model")

    fit = _fit_and_predict(ordered, ordered, neg_rate=neg_rate, seed=seed)
    payload = {
        **fit["model"],
        "levels": fit["levels"],
        "columns": fit["columns"],
        "neg_rate": neg_rate,
        "train_years": _years(ordered),
        "n_train": ordered.height,
        "n_positives": int(y.sum()),
        "trained_at": _now(),
        "note": "fitted on every season in the panel; accuracy is inferred from "
                "the held-out backtest, not measured on these rows",
    }
    joblib.dump(payload, config.MODELS / "ignition_final.joblib")
    log.info("final ignition model fitted on %s (%s rows, %s positives)",
             payload["train_years"], ordered.height, int(y.sum()))
    return payload


def spatial_backtest(
    df: pl.DataFrame,
    *,
    agencies: list[str] | None = None,
    test_years: list[int] | None = None,
    min_test_positives: int = 50,
    neg_rate: float = NEG_RATE,
    drop_geography: bool = False,
    n_boot: int = N_BOOT,
) -> IgnitionSpatialResult:
    """Leave-one-agency-out, so the test cells are in unseen country.

    A cell-day panel makes this test sharper than it is for escalation. Cells
    are fixed points that recur every day of the season, so a model with
    `lat`/`lon` available can memorise individual cells outright -- the same
    coordinates appear hundreds of times in training. Holding out an entire
    agency is the only split that makes that impossible.
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
    folds: list[IgnitionFold] = []
    skipped: list[dict] = []
    pooled_y, pooled_p, pooled_w, pooled_fwi, pooled_clim = [], [], [], [], []

    for agency in agencies:
        row = available.get(agency)
        if row is None or row["pos"] < min_test_positives:
            skipped.append({
                "agency": agency,
                "positives": None if row is None else int(row["pos"]),
                "reason": "no test cells" if row is None
                          else f"fewer than {min_test_positives} positives",
            })
            continue

        train = train_pool.filter(pl.col("agency_code") != agency)
        test = test_pool.filter(pl.col("agency_code") == agency)

        fit = _fit_and_predict(train, test, neg_rate=neg_rate, drop=drop)
        p, y_te, w = fit["p"], fit["y_te"], fit["w_te"]
        ordered = fit["test_ordered"]
        p_fwi = _baseline_vector(ordered, BASELINES["fwi"])
        p_clim = _baseline_vector(ordered, BASELINES["climatology"])

        pr_model = _ap(y_te, p, w)
        pr_fwi = _ap(y_te, p_fwi, w)
        pr_clim = _ap(y_te, p_clim, w)

        folds.append(
            IgnitionFold(
                holdout_agency=agency,
                n_train=train.height,
                n_test=test.height,
                n_positives_test=int(y_te.sum()),
                pr_auc_model=round(pr_model, 5),
                pr_auc_fwi=round(pr_fwi, 5),
                pr_auc_climatology=round(pr_clim, 5),
                pr_auc_prevalence=round(float(np.average(y_te, weights=w)), 6),
                lift_over_fwi=round(pr_model / pr_fwi, 3) if pr_fwi > 0 else float("nan"),
                roc_auc_model=round(float(roc_auc_score(y_te, p, sample_weight=w)), 4),
                ci=_bootstrap(y_te, p, np.maximum(p_fwi, p_clim), w, n_boot=n_boot),
            )
        )
        pooled_y.append(y_te); pooled_p.append(p); pooled_w.append(w)
        pooled_fwi.append(p_fwi); pooled_clim.append(p_clim)
        log.info("ignition fold %s: PR-AUC %.5f vs fwi %.5f / clim %.5f (pos=%s)",
                 agency, pr_model, pr_fwi, pr_clim, int(y_te.sum()))

    if not folds:
        raise RuntimeError("no agency cleared the positive threshold")

    y_all = np.concatenate(pooled_y)
    p_all = np.concatenate(pooled_p)
    w_all = np.concatenate(pooled_w)
    fwi_all = np.concatenate(pooled_fwi)
    clim_all = np.concatenate(pooled_clim)

    pr_all = _ap(y_all, p_all, w_all)
    pr_fwi = _ap(y_all, fwi_all, w_all)
    pr_clim = _ap(y_all, clim_all, w_all)
    stronger = clim_all if pr_clim >= pr_fwi else fwi_all

    pooled = {
        "n_test": int(len(y_all)),
        "n_positives": int(y_all.sum()),
        "population_prevalence": round(float(np.average(y_all, weights=w_all)), 6),
        "pr_auc_model": round(pr_all, 5),
        "pr_auc_fwi": round(pr_fwi, 5),
        "pr_auc_climatology": round(pr_clim, 5),
        "lift_over_best_baseline": round(pr_all / max(pr_fwi, pr_clim), 3),
        "roc_auc_model": round(float(roc_auc_score(y_all, p_all, sample_weight=w_all)), 4),
        "ci": _bootstrap(y_all, p_all, stronger, w_all, n_boot=n_boot),
        "reliability": _weighted_reliability(y_all, p_all, w_all),
    }
    macro = {
        "pr_auc_model": round(float(np.mean([f.pr_auc_model for f in folds])), 5),
        "median_lift_over_fwi": round(float(np.median([f.lift_over_fwi for f in folds])), 3),
        "folds_beating_fwi": int(sum(f.pr_auc_model > f.pr_auc_fwi for f in folds)),
        "folds_beating_climatology": int(
            sum(f.pr_auc_model > f.pr_auc_climatology for f in folds)
        ),
        "n_folds": len(folds),
    }

    res = IgnitionSpatialResult(
        mode=mode,
        train_years=train_years,
        test_years=test_years,
        neg_rate=neg_rate,
        folds=folds,
        skipped=skipped,
        pooled=pooled,
        macro=macro,
        dropped_features=list(drop or ()),
        trained_at=_now(),
    )

    tag = "_region_season" if test_years != train_years else "_region"
    if drop_geography:
        tag += "_nogeo"
    out = config.MODELS / f"ignition_spatial_backtest{tag}.json"
    out.write_text(json.dumps({"result": asdict(res)}, indent=2))
    log.info("ignition spatial backtest written -> %s", out)
    return res
