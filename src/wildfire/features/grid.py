"""A 10 km equal-area grid for Canada.

The escalation model's unit is a fire, which arrives with its own coordinates.
The ignition model's unit has to be invented: a patch of ground and a day.
This module defines that patch, once, so the panel builder, the model and the
map all agree about which cell is which.

**Equal-area, not the map's projection.** `docs/dashboard.html` draws Canada
in Lambert conformal conic, which is right for a map -- it preserves shape and
is what Canada is conventionally drawn in. It is the wrong choice here.
Conformal projections trade area for shape, so a conformal 10 km cell covers
noticeably more ground at 70 N than at 49 N, and the label -- "did at least
one fire start in this cell" -- would then mean something different at every
latitude. The model would learn that northern cells ignite more often when
all that happened is that they are bigger.

So: Lambert azimuthal equal-area about (60 N, 96 W), roughly the centre of the
country, and the grid is 10 km squares of the projected plane. Every cell has
the same area to within the round-off of a sphere-vs-ellipsoid approximation,
which at this cell size is under half a percent.

Cells are identified by integer `(cell_x, cell_y)` -- the floor of the
projected coordinate in units of `CELL_KM`. That keeps the key exact: derived
from floats, but never itself a float, so a cell can be joined on, grouped by
and round-tripped without an epsilon anywhere.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

# Authalic mean radius: the radius of the sphere with the same surface area as
# WGS-84. The right constant for an equal-area projection, and it differs from
# the more familiar 6371.0 by 9 m.
R_KM = 6371.0072

LAT0 = 60.0
LON0 = -96.0
CELL_KM = 10.0

_PHI0 = math.radians(LAT0)
_LAM0 = math.radians(LON0)
_SIN_PHI0 = math.sin(_PHI0)
_COS_PHI0 = math.cos(_PHI0)


def project(lat, lon):
    """Lambert azimuthal equal-area, in kilometres east/north of the origin.

    Accepts scalars or numpy arrays. The projection is singular only at the
    antipode of the origin (60 S, 84 E, in the Indian Ocean), which no
    Canadian coordinate approaches; the clamp is there so a corrupt row
    returns a finite number instead of a warning and a NaN.
    """
    phi = np.radians(np.asarray(lat, dtype=float))
    dlam = np.radians(np.asarray(lon, dtype=float) - LON0)

    cos_c = _SIN_PHI0 * np.sin(phi) + _COS_PHI0 * np.cos(phi) * np.cos(dlam)
    k = np.sqrt(2.0 / np.maximum(1.0 + cos_c, 1e-12))

    x = R_KM * k * np.cos(phi) * np.sin(dlam)
    y = R_KM * k * (_COS_PHI0 * np.sin(phi) - _SIN_PHI0 * np.cos(phi) * np.cos(dlam))
    return x, y


def unproject(x, y):
    """Inverse of `project`. Returns (lat, lon) in degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    safe = np.maximum(rho, 1e-12)
    c = 2.0 * np.arcsin(np.clip(rho / (2.0 * R_KM), -1.0, 1.0))

    sin_c, cos_c = np.sin(c), np.cos(c)
    lat = np.degrees(np.arcsin(cos_c * _SIN_PHI0 + y * sin_c * _COS_PHI0 / safe))
    lon = LON0 + np.degrees(
        np.arctan2(x * sin_c, safe * cos_c * _COS_PHI0 - y * sin_c * _SIN_PHI0)
    )
    # At the origin itself rho is 0 and the formula above is 0/0.
    at_origin = rho < 1e-12
    lat = np.where(at_origin, LAT0, lat)
    lon = np.where(at_origin, LON0, lon)
    return lat, ((lon + 180.0) % 360.0) - 180.0


def cell_of(lat, lon):
    """The `(cell_x, cell_y)` a coordinate falls in, as int32 arrays."""
    x, y = project(lat, lon)
    return (
        np.floor(x / CELL_KM).astype(np.int32),
        np.floor(y / CELL_KM).astype(np.int32),
    )


def cell_centre(cell_x, cell_y):
    """The (lat, lon) of a cell's centre."""
    x = (np.asarray(cell_x, dtype=float) + 0.5) * CELL_KM
    y = (np.asarray(cell_y, dtype=float) + 0.5) * CELL_KM
    return unproject(x, y)


def with_cells(df: pl.DataFrame, *, lat: str = "lat", lon: str = "lon") -> pl.DataFrame:
    """Attach `cell_x` / `cell_y` to a frame that carries coordinates.

    Rows with a null coordinate are dropped: a cell is the unit of analysis
    here, so a row that cannot be placed in one has nothing to contribute.
    """
    df = df.drop_nulls([lat, lon])
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("cell_x"),
            pl.lit(None, dtype=pl.Int32).alias("cell_y"),
        )
    cx, cy = cell_of(df[lat].to_numpy(), df[lon].to_numpy())
    return df.with_columns(
        pl.Series("cell_x", cx, dtype=pl.Int32),
        pl.Series("cell_y", cy, dtype=pl.Int32),
    )


def with_centres(cells: pl.DataFrame) -> pl.DataFrame:
    """Attach `lat` / `lon` of the cell centre to a frame of cell keys."""
    if cells.is_empty():
        return cells.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("lat"),
            pl.lit(None, dtype=pl.Float64).alias("lon"),
        )
    lat, lon = cell_centre(cells["cell_x"].to_numpy(), cells["cell_y"].to_numpy())
    return cells.with_columns(
        pl.Series("lat", lat, dtype=pl.Float64),
        pl.Series("lon", lon, dtype=pl.Float64),
    )


_PROVINCES: list[tuple[str, list]] | None = None


def _provinces() -> list[tuple[str, np.ndarray, np.ndarray, tuple]]:
    """Provincial outlines from the asset the dashboard already ships.

    Same file, same simplification -- 23 KB of Douglas-Peucker'd Natural Earth.
    It is far too coarse to draw a boundary dispute with and entirely good
    enough to answer "is this 10 km cell in Canada, and roughly whose".
    """
    global _PROVINCES
    if _PROVINCES is None:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "assets" / "canada_provinces.json"
        out = []
        for entry in json.loads(path.read_text()):
            for ring in entry["r"]:
                arr = np.asarray(ring, dtype=float)
                if len(arr) < 3:
                    continue
                bbox = (arr[:, 0].min(), arr[:, 0].max(),
                        arr[:, 1].min(), arr[:, 1].max())
                out.append((entry["p"], arr[:, 0], arr[:, 1], bbox))
        _PROVINCES = out
    return _PROVINCES


def _ray_cast(lon, lat, xs, ys) -> np.ndarray:
    """Even-odd point-in-polygon for one ring, vectorised over points."""
    inside = np.zeros(len(lon), dtype=bool)
    for xi, yi, xj, yj in zip(xs, ys, np.roll(xs, -1), np.roll(ys, -1)):
        # Half-open in y so a vertex is counted by exactly one of the two
        # edges meeting at it, which is what keeps the parity right.
        crosses = (yi > lat) != (yj > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = xi + (lat - yi) * (xj - xi) / np.where(yj != yi, yj - yi, np.nan)
        inside ^= crosses & (lon < xint)
    return inside


def province_of(lat, lon) -> np.ndarray:
    """The province code containing each point, or None outside Canada.

    This exists because the *study area* has to be bounded by a country. The
    CWFIS hotspot feed covers North America, not Canada, so cells derived from
    it alone reach down to 25 N -- and a cell in Florida is a guaranteed
    negative in a panel about Canadian fire reports, which quietly dilutes
    every rate in the table and hands the model a free 40% accuracy it has
    earned nothing for.

    Rings are tested largest-bounding-box last so that the mainland wins over
    an island whose simplified outline overlaps it.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.full(len(lat), None, dtype=object)

    for code, xs, ys, (lo_x, hi_x, lo_y, hi_y) in _provinces():
        candidate = (
            (lon >= lo_x - 0.01) & (lon <= hi_x + 0.01)
            & (lat >= lo_y - 0.01) & (lat <= hi_y + 0.01)
            & (out == None)  # noqa: E711 - object-array identity comparison
        )
        if not candidate.any():
            continue
        idx = np.flatnonzero(candidate)
        hit = _ray_cast(lon[idx], lat[idx], xs, ys)
        out[idx[hit]] = code
    return out


def nearest_stations(
    cells: pl.DataFrame,
    stations: pl.DataFrame,
    *,
    k: int = 12,
    chunk: int = 4096,
) -> pl.DataFrame:
    """The `k` nearest weather stations to each cell centre, with distances.

    This is a *static* neighbour table: the geometry never changes, so it is
    computed once and joined to whichever of those stations happened to report
    on a given day. Recomputing the true k-nearest *reporting* stations per day
    would be more faithful and ~200x more work; with k=12 a cell almost always
    has several of its neighbours reporting, and the interpolation reports how
    many it actually used so a thin day is visible rather than silent.

    Brute force, chunked: 57k cells x 2.8k stations is 160M distance
    evaluations, which numpy does in seconds. A KD-tree would need scipy for
    no measurable gain at this size.
    """
    cells = with_centres(cells.select("cell_x", "cell_y").unique().sort(["cell_x", "cell_y"]))
    sx, sy = project(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    ids = stations["aes"].to_numpy()
    k = min(k, len(ids))

    cx, cy = project(cells["lat"].to_numpy(), cells["lon"].to_numpy())

    out_cell_x, out_cell_y, out_aes, out_dist, out_rank = [], [], [], [], []
    keys_x = cells["cell_x"].to_numpy()
    keys_y = cells["cell_y"].to_numpy()

    for start in range(0, len(cx), chunk):
        end = min(start + chunk, len(cx))
        d = np.hypot(
            cx[start:end, None] - sx[None, :],
            cy[start:end, None] - sy[None, :],
        )
        # argpartition finds the k smallest without sorting the rest, then the
        # k are sorted among themselves. Ties are broken by station id so the
        # neighbour table does not depend on numpy's partition internals.
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        take = d[rows, idx]
        order = np.lexsort((ids[idx], take), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        take = np.take_along_axis(take, order, axis=1)

        n = end - start
        out_cell_x.append(np.repeat(keys_x[start:end], k))
        out_cell_y.append(np.repeat(keys_y[start:end], k))
        out_aes.append(ids[idx].reshape(-1))
        out_dist.append(take.reshape(-1))
        out_rank.append(np.tile(np.arange(k, dtype=np.int32), n))

    return pl.DataFrame(
        {
            "cell_x": pl.Series(np.concatenate(out_cell_x), dtype=pl.Int32),
            "cell_y": pl.Series(np.concatenate(out_cell_y), dtype=pl.Int32),
            "aes": np.concatenate(out_aes),
            "station_dist_km": np.concatenate(out_dist),
            "station_rank": pl.Series(np.concatenate(out_rank), dtype=pl.Int32),
        }
    )
