"""Score fires that are burning right now.

The backtest answers "would this have worked". This answers "what should
someone look at today", which is the only form the model takes that resembles
a product.

Two things keep it honest:

* Features come from `features.build.assemble_features` -- the same function
  the training table is built with. A second, serving-only feature path is
  how a model that backtests well quietly starts seeing a different
  distribution than it was fitted on.
* Category levels are restored from the fitted model rather than rebuilt from
  today's data. Rebuilt levels renumber themselves whenever an agency happens
  to have no active fires, and the model silently reads one agency as another.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from . import config
from .config import SPEC
from .features import asof
from .features.build import assemble_features

log = logging.getLogger(__name__)

# Agency-reported status codes, in the order a reader cares about.
STATUS_LABEL = {
    "OC": "out of control",
    "BH": "being held",
    "UC": "under control",
    "EX": "out",
    "UNK": "unknown",
}


def _load_model(prefer_final: bool = True) -> dict:
    import joblib

    final = config.MODELS / "escalation_final.joblib"
    backtest = config.MODELS / "escalation_lgbm.joblib"

    if prefer_final and final.exists():
        return joblib.load(final)
    if backtest.exists():
        payload = joblib.load(backtest)
        log.warning(
            "scoring with the BACKTEST model (trained on %s only). Run "
            "`wildfire fit-final` to fit on every labelled season.",
            payload.get("train_years"),
        )
        return payload
    raise RuntimeError("no fitted model on disk -- run `wildfire backtest` first")


def score(
    *,
    as_of: datetime | None = None,
    window_days: int = 14,
    spec=SPEC,
    prefer_final: bool = True,
) -> pl.DataFrame:
    """Rank recently-reported fires by probability of exceeding the threshold.

    `as_of` defaults to the newest transaction timestamp in the fire feed --
    not wall-clock now, which would silently score every fire against a stale
    ingest and report them all as quiet.
    """
    from .models.escalation import _feature_frame

    fires_path = config.CURATED / "reported_fires.parquet"
    if not fires_path.exists():
        raise RuntimeError("no fire data -- run `wildfire ingest-fires` first")
    fires = pl.read_parquet(fires_path)

    hs_path = config.CURATED / "hotspots.parquet"
    hotspots = pl.read_parquet(hs_path) if hs_path.exists() else pl.DataFrame()

    if as_of is None:
        as_of = fires["record_start"].max()
    log.info("scoring as of %s", as_of)

    offsets = asof.first_seen(fires)
    # Reached its decision instant, and recent enough to still be a live
    # question. A fire first reported three months ago has an answer already.
    offsets = offsets.filter(
        (pl.col("t0") + pl.duration(hours=spec.decision_hours) <= pl.lit(as_of))
        & (pl.col("t0") >= pl.lit(as_of) - pl.duration(days=window_days))
    )
    if offsets.is_empty():
        return pl.DataFrame()

    feats = assemble_features(fires, offsets, hotspots, spec=spec)

    # Same exclusion as training: the question is whether a fire *becomes*
    # large. One already past the threshold is not a forecast, it is a fact.
    feats = feats.filter(
        pl.col("size_at_decision").fill_null(0) < spec.size_threshold_ha
    )
    if feats.is_empty():
        return pl.DataFrame()

    payload = _load_model(prefer_final)
    X, _ = _feature_frame(feats, categories=payload["levels"])

    missing = [c for c in payload["columns"] if c not in X.columns]
    if missing:
        raise RuntimeError(
            f"the fitted model expects features that today's table lacks: {missing}. "
            "Re-run `wildfire build` (and re-fit) so both sides agree."
        )
    X = X[payload["columns"]]

    p_raw = payload["base"].predict_proba(X)[:, 1]
    p = payload["calibrator"].predict(p_raw) if payload.get("calibrator") is not None else p_raw

    # What the fire is doing *now*, for context beside the score. Not a
    # feature -- it is read at as_of, which is later than the decision instant.
    latest = (
        fires.filter(pl.col("record_start") <= pl.lit(as_of))
        .sort(["record_start", "id"])
        .group_by(asof.FIRE_KEY)
        .last()
        .select(
            asof.FIRE_KEY,
            pl.col("fire_size").alias("size_now"),
            pl.col("stage_of_control_status").alias("status_now"),
        )
    )

    out = (
        feats.select(
            asof.FIRE_KEY,
            "t0",
            "agency_code",
            "lat",
            "lon",
            "size_at_decision",
            "status_at_decision",
            "hs_count",
            "hs_hfi_max",
        )
        .with_columns(
            pl.Series("risk", p),
            (pl.lit(as_of) - pl.col("t0")).dt.total_hours().alias("age_hours"),
        )
        .join(latest, on=asof.FIRE_KEY, how="left")
        .sort("risk", descending=True)
    )

    dest = config.MODELS / "current_risk.parquet"
    out.write_parquet(dest)
    log.info("scored %s fires -> %s", out.height, dest)
    return out
