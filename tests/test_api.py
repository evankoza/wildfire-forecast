"""The HTTP layer.

Two things are worth testing here and they are not the happy path.

**It must fail honestly when the pipeline has not run.** A serving layer over
artefacts has a state the CLI does not: the file is not there yet. Answering
404 would tell a client to stop asking; answering 200 with an empty list would
be a lie. It has to be 503.

**Every scored response must carry its provenance and its disclaimer.** This
is a research model on a live fire feed. Someone consuming JSON never sees the
HTML page, so the caveat has to be in the payload -- and a probability without
the model that produced it is the thing that ends up in a slide with no
asterisk.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from wildfire import api, config  # noqa: E402


@pytest.fixture
def artefacts(tmp_path, monkeypatch):
    """Point the module at an empty artefact directory."""
    curated = tmp_path / "curated"
    models = tmp_path / "models"
    curated.mkdir()
    models.mkdir()
    monkeypatch.setattr(config, "CURATED", curated)
    monkeypatch.setattr(config, "MODELS", models)
    api.CACHE.clear()
    yield tmp_path
    api.CACHE.clear()


@pytest.fixture
def client(artefacts):
    return TestClient(api.create_app())


def _write_fires(models):
    pl.DataFrame(
        {
            "national_fire_id": ["2025_AB_A1", "2025_BC_B2", "2025_AB_A3"],
            "t0": [datetime(2025, 7, 1, 6), datetime(2025, 7, 2, 9), datetime(2025, 7, 3, 1)],
            "agency_code": ["AB", "BC", "AB"],
            "lat": [55.1, 52.0, 56.4],
            "lon": [-114.2, -122.0, -111.0],
            "size_at_decision": [2.0, 0.5, 12.0],
            "status_at_decision": ["OC", "BH", "OC"],
            "hs_count": [8, 0, 41],
            "hs_hfi_max": [4200.0, None, 15000.0],
            "risk": [0.12, 0.01, 0.47],
            "age_hours": [30, 26, 25],
            "size_now": [40.0, 0.5, 300.0],
            "status_now": ["OC", "UC", "OC"],
        }
    ).write_parquet(models / "current_risk.parquet")


def _write_cells(models):
    pl.DataFrame(
        {
            "cell_x": [10, 11, 12],
            "cell_y": [20, 21, 22],
            "day": [date(2025, 7, 3)] * 3,
            "lat": [55.0, 52.0, 49.5],
            "lon": [-114.0, -122.0, -97.0],
            "agency_code": ["AB", "BC", "MB"],
            "wx_fwi": [30.0, 12.0, 5.0],
            "wx_dist_km": [40.0, 90.0, 15.0],
            "hs_n_7d": [12, 0, 0],
            "ig_n_365d": [3, 0, 1],
            "risk": [0.031, 0.004, 0.0006],
        }
    ).write_parquet(models / "ignition_risk.parquet")


# --- failing honestly ----------------------------------------------------


def test_health_does_not_depend_on_any_artefact(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_artefacts_are_503_not_404(client):
    """The route exists and will work once the pipeline has run."""
    for url in ("/v1/fires", "/v1/ignition", "/v1/fires/2025_AB_A1"):
        r = client.get(url)
        assert r.status_code == 503, url
        assert "has not been built" in r.json()["detail"]


def test_meta_reports_what_is_missing_rather_than_failing(client):
    body = client.get("/v1/meta").json()
    assert body["artefacts"]["escalation_scores"] == {"built": False}
    assert body["disclaimer"]


def test_models_endpoint_works_with_no_models_on_disk(client):
    """Provenance is best-effort: it degrades, it does not 500."""
    body = client.get("/v1/models").json()
    assert "escalation" in body and "ignition" in body
    assert body["escalation"]["spec"]["size_threshold_ha"] == 100.0


# --- serving -------------------------------------------------------------


def test_fires_are_ranked_and_carry_provenance(client, artefacts):
    _write_fires(artefacts / "models")
    body = client.get("/v1/fires").json()

    assert body["n_matching"] == 3
    risks = [f["risk"] for f in body["fires"]]
    assert risks == sorted(risks, reverse=True)
    assert body["disclaimer"].startswith("Research model")
    assert "target" in body["model"]
    # Datetimes have to survive the trip as strings, not blow up the encoder.
    assert body["fires"][0]["t0"] == "2025-07-03T01:00:00"


def test_fires_filters_compose(client, artefacts):
    _write_fires(artefacts / "models")
    body = client.get("/v1/fires", params={"agency": "ab", "min_risk": 0.2}).json()
    assert [f["national_fire_id"] for f in body["fires"]] == ["2025_AB_A3"]
    assert body["n_matching"] == 1


def test_limit_reports_the_full_match_count(client, artefacts):
    """`n_matching` is the population; `n_returned` is the page."""
    _write_fires(artefacts / "models")
    body = client.get("/v1/fires", params={"limit": 1}).json()
    assert body["n_matching"] == 3
    assert body["n_returned"] == 1


def test_one_fire_by_id(client, artefacts):
    _write_fires(artefacts / "models")
    body = client.get("/v1/fires/2025_BC_B2").json()
    assert body["fire"]["agency_code"] == "BC"
    assert body["disclaimer"]


def test_unknown_fire_is_404(client, artefacts):
    _write_fires(artefacts / "models")
    r = client.get("/v1/fires/2025_XX_nope")
    assert r.status_code == 404


def test_ignition_defaults_to_the_newest_day_and_ranks(client, artefacts):
    _write_cells(artefacts / "models")
    body = client.get("/v1/ignition").json()
    assert body["day"] == "2025-07-03"
    assert [c["risk"] for c in body["cells"]] == [0.031, 0.004, 0.0006]
    assert body["disclaimer"]


def test_ignition_bbox_filters_by_coordinates(client, artefacts):
    _write_cells(artefacts / "models")
    body = client.get("/v1/ignition", params={"bbox": "-125,50,-110,60"}).json()
    assert {c["agency_code"] for c in body["cells"]} == {"AB", "BC"}


def test_malformed_bbox_is_400(client, artefacts):
    _write_cells(artefacts / "models")
    for bad in ("1,2,3", "north,south,east,west", "1,2,3,4,5"):
        r = client.get("/v1/ignition", params={"bbox": bad})
        assert r.status_code in (400, 422), bad
    # An empty bbox is absence, not a malformed filter.
    assert client.get("/v1/ignition", params={"bbox": ""}).status_code == 200


def test_root_lists_its_own_endpoints(client):
    body = client.get("/").json()
    assert "/v1/ignition" in body["endpoints"]
    assert body["disclaimer"]


# --- the cache -----------------------------------------------------------


def test_a_rewritten_artefact_is_picked_up_without_a_restart(client, artefacts):
    """A scheduled `predict` rewrites the parquet under a running server.

    The cache is keyed on mtime for exactly this: the next request has to see
    the new file, and must not re-read it on every request in between.
    """
    import os
    import time

    models = artefacts / "models"
    _write_fires(models)
    assert client.get("/v1/fires").json()["n_matching"] == 3

    pl.DataFrame(
        {
            "national_fire_id": ["2025_SK_S9"],
            "t0": [datetime(2025, 7, 9, 3)],
            "agency_code": ["SK"],
            "lat": [54.0], "lon": [-106.0],
            "size_at_decision": [1.0], "status_at_decision": ["OC"],
            "hs_count": [2], "hs_hfi_max": [900.0],
            "risk": [0.2], "age_hours": [25],
            "size_now": [3.0], "status_now": ["OC"],
        }
    ).write_parquet(models / "current_risk.parquet")
    # Filesystem timestamps are coarse enough that a fast rewrite can land on
    # the same mtime; nudge it rather than sleep.
    stamp = time.time() + 10
    os.utime(models / "current_risk.parquet", (stamp, stamp))

    body = client.get("/v1/fires").json()
    assert body["n_matching"] == 1
    assert body["fires"][0]["national_fire_id"] == "2025_SK_S9"


def test_nan_becomes_null_rather_than_invalid_json(artefacts):
    """`hs_hfi_max` is genuinely missing for a fire with no detections."""
    models = artefacts / "models"
    _write_fires(models)
    body = api.fire_risk()
    quiet = next(f for f in body["fires"] if f["national_fire_id"] == "2025_BC_B2")
    assert quiet["hs_hfi_max"] is None
