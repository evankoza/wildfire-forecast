"""HTTP serving layer.

`predict` writes parquet, which is fine for a person with a terminal and
useless for anything else. This is the same numbers over HTTP.

Three decisions shape it, and they are all about what a serving layer for a
*research model* should refuse to do:

* **It serves artefacts; it does not run the pipeline.** Every endpoint reads
  a file the CLI wrote. Nothing here ingests, refits or rebuilds. A rebuild is
  a five-minute job against half a dozen government endpoints, and putting
  that behind a request handler means the first curious visitor DDoSes NRCan
  on your behalf. Freshness is a scheduling problem, and `/v1/meta` reports
  it honestly rather than hiding it.

* **Every score carries its provenance.** A bare probability is the thing that
  ends up in a slide deck with no asterisk. So each risk payload names the
  model that produced it, the seasons it was fitted on, the instant the
  underlying data was current as of, and the held-out interval that is the
  only evidence about its accuracy -- which, because the deployed model is
  refit on every labelled season, is *inferred* rather than measured.

* **It says what it is.** A research model on a live fire feed is exactly the
  kind of thing that gets mistaken for an official product. The disclaimer is
  in the root document, in the OpenAPI description, and in every scored
  response, because someone consuming JSON will never see the HTML page.

Run it with `python -m wildfire serve`. FastAPI and uvicorn are an optional
extra (`pip install -e ".[serve]"`) so that the modelling side of the project
does not acquire a web framework it never imports.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from . import config
from .config import SPEC

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Research model. Not operational guidance, and not a product of any fire "
    "agency. For live official information see the Canadian Wildland Fire "
    "Information System at cwfis.cfs.nrcan.gc.ca."
)


class ArtefactMissing(RuntimeError):
    """A file the endpoint needs has not been produced yet."""


class _Cache:
    """Read-through cache keyed on path mtime.

    A restart-free refresh is the whole point: a scheduled `wildfire predict`
    rewrites the parquet underneath a running server, and the next request
    should see it. Comparing mtime costs a stat call and avoids both a stale
    process and a re-read per request.
    """

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[float, Any]] = {}

    def load(self, path: Path, reader):
        if not path.exists():
            raise ArtefactMissing(
                f"{path.name} has not been built yet. Run the CLI step that "
                f"produces it, then retry."
            )
        stamp = path.stat().st_mtime
        hit = self._entries.get(path)
        if hit is None or hit[0] != stamp:
            self._entries[path] = (stamp, reader(path))
        return self._entries[path][1]

    def parquet(self, path: Path) -> pl.DataFrame:
        return self.load(path, pl.read_parquet)

    def json(self, path: Path) -> dict:
        return self.load(path, lambda p: json.loads(p.read_text()))

    def clear(self) -> None:
        self._entries.clear()


CACHE = _Cache()


def _jsonable(value):
    """Polars gives back date/datetime objects; JSON does not take them."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _rows(df: pl.DataFrame) -> list[dict]:
    return [{k: _jsonable(v) for k, v in r.items()} for r in df.iter_rows(named=True)]


# --- the payloads ----------------------------------------------------------


def escalation_provenance() -> dict:
    """What produced the escalation scores, and what is known about them."""
    out: dict[str, Any] = {
        "target": (
            f"P(fire reaches >= {SPEC.size_threshold_ha:.0f} ha by T0+"
            f"{SPEC.horizon_hours}h), decided at T0+{SPEC.decision_hours}h"
        ),
        "spec": {
            "decision_hours": SPEC.decision_hours,
            "horizon_hours": SPEC.horizon_hours,
            "size_threshold_ha": SPEC.size_threshold_ha,
        },
    }
    try:
        import joblib

        payload = joblib.load(config.MODELS / "escalation_final.joblib")
        out["fitted_on_seasons"] = payload.get("train_years")
        out["fitted_at"] = payload.get("trained_at")
        out["n_training_fires"] = payload.get("n_train")
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        log.debug("no final escalation model to describe: %s", exc)

    try:
        bt = CACHE.json(config.MODELS / "escalation_backtest.json")["result"]
        out["accuracy"] = {
            "note": (
                "measured on a held-out season using a model trained only on "
                "earlier ones. The model serving these scores is refit on every "
                "labelled season, so its accuracy is inferred from this, not "
                "measured directly."
            ),
            "held_out_seasons": bt["test_years"],
            "pr_auc": bt["pr_auc_model"],
            "pr_auc_interval": bt["ci"].get("pr_auc_model"),
            "pr_auc_size_at_decision_baseline": bt["pr_auc_size_only"],
            "brier": bt["brier_model"],
        }
    except (ArtefactMissing, KeyError) as exc:
        log.debug("no escalation backtest to describe: %s", exc)
    return out


def ignition_provenance() -> dict:
    out: dict[str, Any] = {
        "target": "P(at least one new fire reported in this 10 km cell today)",
        "grid": None,
    }
    try:
        meta = CACHE.json(config.CURATED / "ignition_table_meta.json")
        out["grid"] = {
            "cell_km": meta["cell_km"],
            "projection": "Lambert azimuthal equal-area about 60N 96W",
            "study_area_cells": meta["n_cells"],
            "study_area_seasons": meta["area_years"],
            "study_area_coverage": meta["study_area_coverage"],
        }
        out["negative_sampling_rate"] = meta["neg_rate"]
    except (ArtefactMissing, KeyError) as exc:
        log.debug("no ignition panel metadata: %s", exc)

    try:
        bt = CACHE.json(config.MODELS / "ignition_backtest.json")["result"]
        out["accuracy"] = {
            "note": (
                "PR-AUC is sample-weighted back onto the true cell-day "
                "prevalence, so it is comparable with the baselines below and "
                "not with an unweighted figure."
            ),
            "held_out_seasons": bt["test_years"],
            "population_prevalence": bt["population_prevalence"],
            "pr_auc": bt["pr_auc_model"],
            "pr_auc_interval": bt["ci"].get("pr_auc_model"),
            "pr_auc_fire_weather_baseline": bt["pr_auc_fwi"],
            "pr_auc_climatology_baseline": bt["pr_auc_climatology"],
        }
    except (ArtefactMissing, KeyError) as exc:
        log.debug("no ignition backtest to describe: %s", exc)
    return out


def fire_risk(*, limit: int = 50, min_risk: float = 0.0,
              agency: str | None = None) -> dict:
    df = CACHE.parquet(config.MODELS / "current_risk.parquet")
    if agency:
        df = df.filter(pl.col("agency_code") == agency.upper())
    if min_risk > 0:
        df = df.filter(pl.col("risk") >= min_risk)
    total = df.height
    df = df.sort("risk", descending=True).head(limit)
    return {
        "as_of": _jsonable(_as_of(config.MODELS / "current_risk.parquet")),
        "n_matching": total,
        "n_returned": df.height,
        "model": escalation_provenance(),
        "disclaimer": DISCLAIMER,
        "fires": _rows(df),
    }


def one_fire(national_fire_id: str) -> dict | None:
    df = CACHE.parquet(config.MODELS / "current_risk.parquet")
    hit = df.filter(pl.col("national_fire_id") == national_fire_id)
    if hit.is_empty():
        return None
    return {
        "as_of": _jsonable(_as_of(config.MODELS / "current_risk.parquet")),
        "model": escalation_provenance(),
        "disclaimer": DISCLAIMER,
        "fire": _rows(hit)[0],
    }


def ignition_risk(*, limit: int = 100, min_risk: float = 0.0,
                  bbox: tuple[float, float, float, float] | None = None,
                  day: date | None = None) -> dict:
    """Ranked cells for a day. `bbox` is (min_lon, min_lat, max_lon, max_lat)."""
    path = config.MODELS / "ignition_risk.parquet"
    df = CACHE.parquet(path)
    available = sorted(df["day"].unique().to_list())
    target = day or (available[-1] if available else None)
    if target is None:
        raise ArtefactMissing("ignition_risk.parquet is empty")
    df = df.filter(pl.col("day") == target)

    if bbox:
        lo_lon, lo_lat, hi_lon, hi_lat = bbox
        df = df.filter(
            pl.col("lon").is_between(lo_lon, hi_lon)
            & pl.col("lat").is_between(lo_lat, hi_lat)
        )
    if min_risk > 0:
        df = df.filter(pl.col("risk") >= min_risk)
    total = df.height
    df = df.sort("risk", descending=True).head(limit)
    return {
        "day": _jsonable(target),
        "days_available": [_jsonable(d) for d in available],
        "n_matching": total,
        "n_returned": df.height,
        "model": ignition_provenance(),
        "disclaimer": DISCLAIMER,
        "cells": _rows(df),
    }


def _as_of(path: Path):
    df = CACHE.parquet(path)
    for col in ("t0", "day"):
        if col in df.columns and df.height:
            return df[col].max()
    return None


def meta() -> dict:
    """Freshness of every artefact an endpoint can serve.

    Deliberately reports staleness rather than concealing it. Nothing here
    refreshes on demand, so a client that cares needs to be able to see how old
    the answer is without inferring it from the data.
    """
    watched = {
        "reported_fires": config.CURATED / "reported_fires.parquet",
        "hotspots": config.CURATED / "hotspots.parquet",
        "station_fwi": config.CURATED / "station_fwi.parquet",
        "ciffc_sitreps": config.CURATED / "ciffc_sitreps.parquet",
        "modelling_table": config.CURATED / "modelling_table.parquet",
        "ignition_table": config.CURATED / "ignition_table.parquet",
        "escalation_scores": config.MODELS / "current_risk.parquet",
        "ignition_scores": config.MODELS / "ignition_risk.parquet",
    }
    out = {}
    for name, path in watched.items():
        if not path.exists():
            out[name] = {"built": False}
            continue
        stat = path.stat()
        out[name] = {
            "built": True,
            "written_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            "bytes": stat.st_size,
        }
    return {"artefacts": out, "disclaimer": DISCLAIMER}


# --- the app ---------------------------------------------------------------


def create_app():
    """Build the FastAPI application.

    Imported lazily so the CLI, the feature builder and the tests never pay
    for a web framework they do not use.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            'the serving layer needs its optional extra: pip install -e ".[serve]"'
        ) from exc

    app = FastAPI(
        title="wildfire-forecast",
        version="0.1.0",
        description=(
            "Escalation and ignition risk for Canadian wildland fire, from open "
            "government feeds.\n\n**" + DISCLAIMER + "**"
        ),
    )

    def _guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ArtefactMissing as exc:
            # 503, not 404: the route exists and will work once the pipeline
            # has run. A 404 would tell a client to stop asking.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "name": "wildfire-forecast",
            "disclaimer": DISCLAIMER,
            "endpoints": [
                "/health", "/v1/meta", "/v1/models",
                "/v1/fires", "/v1/fires/{national_fire_id}", "/v1/ignition",
            ],
            "docs": "/docs",
        }

    @app.get("/health")
    def health():
        """Liveness only. It does not assert that any artefact exists."""
        return {"status": "ok"}

    @app.get("/v1/meta")
    def meta_endpoint():
        return meta()

    @app.get("/v1/models")
    def models_endpoint():
        return {
            "escalation": escalation_provenance(),
            "ignition": ignition_provenance(),
            "disclaimer": DISCLAIMER,
        }

    @app.get("/v1/fires")
    def fires_endpoint(
        limit: int = Query(50, ge=1, le=1000),
        min_risk: float = Query(0.0, ge=0.0, le=1.0),
        agency: str | None = Query(None, max_length=4),
    ):
        return _guard(fire_risk, limit=limit, min_risk=min_risk, agency=agency)

    @app.get("/v1/fires/{national_fire_id}")
    def fire_endpoint(national_fire_id: str):
        hit = _guard(one_fire, national_fire_id)
        if hit is None:
            raise HTTPException(
                status_code=404,
                detail=f"{national_fire_id} is not in the current scoring window",
            )
        return hit

    @app.get("/v1/ignition")
    def ignition_endpoint(
        limit: int = Query(100, ge=1, le=5000),
        min_risk: float = Query(0.0, ge=0.0, le=1.0),
        day: date | None = Query(None),
        bbox: str | None = Query(
            None, description="min_lon,min_lat,max_lon,max_lat"
        ),
    ):
        box = None
        if bbox:
            try:
                parts = [float(v) for v in bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError
                box = (parts[0], parts[1], parts[2], parts[3])
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="bbox must be min_lon,min_lat,max_lon,max_lat",
                ) from exc
        return _guard(ignition_risk, limit=limit, min_risk=min_risk,
                      bbox=box, day=day)

    @app.exception_handler(ArtefactMissing)
    def _artefact_missing(_request, exc):  # pragma: no cover - safety net
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the app under uvicorn."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            'the serving layer needs its optional extra: pip install -e ".[serve]"'
        ) from exc

    if reload:
        uvicorn.run("wildfire.api:create_app", host=host, port=port,
                    reload=True, factory=True)
    else:
        uvicorn.run(create_app(), host=host, port=port)
