"""A nightly refresh must cost one season, not every season.

`load_seasons(..., merge=True)` exists so the dashboard can be rebuilt daily
without re-parsing the closed years' archive zips. That is only safe if a merge
is exact: the refreshed season is replaced wholesale, every other season is
untouched, and nothing is duplicated. If any of those slip, the model quietly
trains on a table that has either lost history or double-counted it.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from wildfire import config
from wildfire.sources import cwfis_hotspots


def _rows(year: int, n: int, *, hfi: float, extra: dict | None = None) -> pl.DataFrame:
    """n detections in one season, tagged by hfi so provenance is visible."""
    data = {
        "lat": [50.0 + i for i in range(n)],
        "lon": [-100.0 - i for i in range(n)],
        "rep_date": [datetime(year, 7, 1 + i) for i in range(n)],
        "source": ["test"] * n,
        "sensor": ["test"] * n,
        "fwi": [10.0] * n,
        "fuel": ["C2"] * n,
        "ros": [1.0] * n,
        "sfc": [1.0] * n,
        "tfc": [1.0] * n,
        "hfi": [hfi] * n,
    }
    if extra:
        data.update({k: [v] * n for k, v in extra.items()})
    return pl.DataFrame(data)


@pytest.fixture
def curated(tmp_path, monkeypatch):
    """Point CURATED at a temp dir so the real data dir is never touched."""
    monkeypatch.setattr(config, "CURATED", tmp_path)
    return tmp_path


def _seed(curated, frames: list[pl.DataFrame]) -> None:
    pl.concat(frames, how="vertical_relaxed").write_parquet(curated / "hotspots.parquet")


def _stub_fetch(monkeypatch, by_year: dict[int, pl.DataFrame]) -> None:
    """Serve canned seasons instead of hitting NRCan."""
    monkeypatch.setattr(
        cwfis_hotspots, "fetch_season", lambda y, force=False: by_year[y]
    )


def test_merge_replaces_only_the_named_season(curated, monkeypatch):
    _seed(curated, [_rows(2024, 3, hfi=1.0), _rows(2025, 2, hfi=2.0)])
    # 2025 comes back with a different row count and a new marker value.
    _stub_fetch(monkeypatch, {2025: _rows(2025, 5, hfi=99.0)})

    out = cwfis_hotspots.load_seasons([2025], merge=True)

    per_year = dict(
        out.with_columns(pl.col("rep_date").dt.year().alias("y"))
        .group_by("y")
        .len()
        .iter_rows()
    )
    assert per_year == {2024: 3, 2025: 5}, "2024 must survive untouched, 2025 replaced"
    # No trace of the old 2025 rows, and 2024's marker is intact.
    assert set(out.filter(pl.col("rep_date").dt.year() == 2025)["hfi"]) == {99.0}
    assert set(out.filter(pl.col("rep_date").dt.year() == 2024)["hfi"]) == {1.0}


def test_merge_does_not_duplicate_on_a_repeated_refresh(curated, monkeypatch):
    """Running the nightly job twice in a row must be a no-op, not a doubling."""
    _seed(curated, [_rows(2024, 3, hfi=1.0)])
    _stub_fetch(monkeypatch, {2025: _rows(2025, 4, hfi=7.0)})

    first = cwfis_hotspots.load_seasons([2025], merge=True)
    second = cwfis_hotspots.load_seasons([2025], merge=True)

    assert first.height == second.height == 7
    assert second.equals(first)


def test_replace_is_still_the_default(curated, monkeypatch):
    """Without merge=True the table is exactly the years asked for."""
    _seed(curated, [_rows(2024, 3, hfi=1.0), _rows(2025, 2, hfi=2.0)])
    _stub_fetch(monkeypatch, {2025: _rows(2025, 5, hfi=99.0)})

    out = cwfis_hotspots.load_seasons([2025])

    assert out.height == 5
    assert out["rep_date"].dt.year().unique().to_list() == [2025]


def test_merge_narrowing_the_schema_is_reported(curated, monkeypatch, caplog):
    """A column the incoming season lacks is dropped for EVERY year.

    That is the documented tradeoff (one feature set regardless of which source
    a season came from), but it must never happen silently.
    """
    _seed(curated, [_rows(2024, 3, hfi=1.0, extra={"estarea": 5.0})])
    _stub_fetch(monkeypatch, {2025: _rows(2025, 2, hfi=2.0)})  # no estarea

    with caplog.at_level("WARNING"):
        out = cwfis_hotspots.load_seasons([2025], merge=True)

    assert "estarea" not in out.columns
    assert any("estarea" in r.getMessage() for r in caplog.records)


def test_merge_without_an_existing_table_just_writes(curated, monkeypatch):
    _stub_fetch(monkeypatch, {2025: _rows(2025, 2, hfi=2.0)})

    out = cwfis_hotspots.load_seasons([2025], merge=True)

    assert out.height == 2
    assert (curated / "hotspots.parquet").exists()
