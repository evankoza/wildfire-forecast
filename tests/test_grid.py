"""The 10 km equal-area grid the ignition model is posed on.

Three properties matter and none of them is obvious from reading the formulas:
the projection round-trips, cells really do have equal area (which is the whole
reason it is not the map's conformal projection), and the country filter
actually excludes the United States -- which the hotspot feed is full of.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from wildfire.features import grid

# A spread of real Canadian places, deliberately including the far north where
# a conformal projection's area error is worst.
PLACES = {
    "Vancouver": (49.28, -123.12),
    "Edmonton": (53.55, -113.49),
    "Yellowknife": (62.45, -114.37),
    "Iqaluit": (63.75, -68.52),
    "Toronto": (43.65, -79.38),
    "St John's": (47.56, -52.71),
    "Whitehorse": (60.72, -135.06),
    "Resolute": (74.70, -94.83),
}


def test_projection_round_trips():
    lat = np.array([v[0] for v in PLACES.values()])
    lon = np.array([v[1] for v in PLACES.values()])
    x, y = grid.project(lat, lon)
    back_lat, back_lon = grid.unproject(x, y)
    assert np.allclose(back_lat, lat, atol=1e-9)
    assert np.allclose(back_lon, lon, atol=1e-9)


def test_origin_maps_to_origin():
    x, y = grid.project(grid.LAT0, grid.LON0)
    assert abs(float(x)) < 1e-9 and abs(float(y)) < 1e-9
    lat, lon = grid.unproject(0.0, 0.0)
    assert abs(float(lat) - grid.LAT0) < 1e-9
    assert abs(float(lon) - grid.LON0) < 1e-9


def test_cells_have_equal_area_everywhere():
    """The reason this is equal-area and the dashboard's map is not.

    A cell's ground area is measured by projecting its four corners back to
    the sphere and taking the spherical excess of the two triangles. If the
    projection were conformal these would grow with latitude; here they must
    not, because the label is "did a fire start in this cell" and a cell that
    covers more ground at 70 N would carry a mechanically higher rate.
    """
    areas = []
    for lat, lon in PLACES.values():
        cx, cy = grid.cell_of(lat, lon)
        x0, y0 = float(cx) * grid.CELL_KM, float(cy) * grid.CELL_KM
        corners = [
            (x0, y0),
            (x0 + grid.CELL_KM, y0),
            (x0 + grid.CELL_KM, y0 + grid.CELL_KM),
            (x0, y0 + grid.CELL_KM),
        ]
        lats, lons = grid.unproject(
            np.array([c[0] for c in corners]), np.array([c[1] for c in corners])
        )
        areas.append(_spherical_quad_area(lats, lons))

    areas = np.array(areas)
    nominal = grid.CELL_KM**2
    # Half a percent is the sphere-vs-ellipsoid approximation, not a latitude
    # trend; the point of the assertion is that the spread is tiny.
    assert np.all(np.abs(areas - nominal) / nominal < 0.005), areas
    assert (areas.max() - areas.min()) / nominal < 0.005


def _spherical_quad_area(lats, lons) -> float:
    """Area of a spherical quadrilateral, in km^2, via two triangles."""

    def tri(i, j, k):
        pts = [
            (math.radians(lats[n]), math.radians(lons[n])) for n in (i, j, k)
        ]
        vecs = [
            (math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p))
            for p, l in pts
        ]

        def angle(a, b, c):
            # Interior angle at vertex a of the spherical triangle abc.
            def cross(u, v):
                return (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )

            def dot(u, v):
                return sum(p * q for p, q in zip(u, v))

            def norm(u):
                n = math.sqrt(dot(u, u))
                return tuple(p / n for p in u)

            n1, n2 = norm(cross(a, b)), norm(cross(a, c))
            return math.acos(max(-1.0, min(1.0, dot(n1, n2))))

        a, b, c = vecs
        excess = angle(a, b, c) + angle(b, c, a) + angle(c, a, b) - math.pi
        return excess * grid.R_KM**2

    return tri(0, 1, 2) + tri(0, 2, 3)


def test_neighbouring_points_share_a_cell_and_distant_ones_do_not():
    here = grid.cell_of(55.0, -115.0)
    # 1 km away must be the same cell or an immediate neighbour; 100 km away
    # must not be either.
    near = grid.cell_of(55.009, -115.0)
    far = grid.cell_of(55.9, -115.0)
    assert abs(int(near[0]) - int(here[0])) <= 1
    assert abs(int(near[1]) - int(here[1])) <= 1
    assert abs(int(far[1]) - int(here[1])) >= 8


def test_cell_keys_are_integers_not_floats():
    """The key has to be exact so it can be joined, grouped and round-tripped."""
    cx, cy = grid.cell_of([49.28, 62.45], [-123.12, -114.37])
    assert cx.dtype == np.int32 and cy.dtype == np.int32


def test_cell_centre_falls_inside_its_own_cell():
    for lat, lon in PLACES.values():
        cx, cy = grid.cell_of(lat, lon)
        clat, clon = grid.cell_centre(cx, cy)
        again_x, again_y = grid.cell_of(clat, clon)
        assert int(again_x) == int(cx) and int(again_y) == int(cy)


def test_province_of_places_canadian_points_and_rejects_foreign_ones():
    canadian = {
        "Edmonton": (53.55, -113.49, "AB"),
        "Toronto": (43.65, -79.38, "ON"),
        "Saskatoon": (52.13, -106.67, "SK"),
        "Winnipeg": (49.90, -97.14, "MB"),
        "Yellowknife": (62.45, -114.37, "NT"),
        "Whitehorse": (60.72, -135.06, "YT"),
    }
    lats = [v[0] for v in canadian.values()]
    lons = [v[1] for v in canadian.values()]
    got = grid.province_of(lats, lons)
    for (name, (_, _, want)), have in zip(canadian.items(), got):
        assert have == want, f"{name} placed in {have}, expected {want}"


@pytest.mark.parametrize(
    "lat,lon,where",
    [
        (25.8, -80.2, "Miami"),
        (34.05, -118.24, "Los Angeles"),
        (47.6, -122.3, "Seattle"),
        (19.4, -99.1, "Mexico City"),
        (60.0, -30.0, "mid-Atlantic"),
    ],
)
def test_province_of_rejects_non_canadian_points(lat, lon, where):
    """The country filter, which is load-bearing.

    The CWFIS hotspot feed covers North America. An activity-derived study area
    that skips this test reaches to 25 N, and two fifths of its cell-days sit
    somewhere a Canadian fire report cannot happen.
    """
    assert grid.province_of([lat], [lon])[0] is None, where


def test_nearest_stations_matches_brute_force():
    rng = np.random.default_rng(3)
    stations = pl.DataFrame(
        {
            "aes": [f"S{i:03d}" for i in range(40)],
            "lat": rng.uniform(48.0, 62.0, 40),
            "lon": rng.uniform(-130.0, -60.0, 40),
        }
    )
    cells = pl.DataFrame(
        {
            "cell_x": pl.Series(rng.integers(-200, 200, 25), dtype=pl.Int32),
            "cell_y": pl.Series(rng.integers(-200, 200, 25), dtype=pl.Int32),
        }
    ).unique()

    got = grid.nearest_stations(cells, stations, k=5, chunk=7)
    assert got["station_rank"].max() == 4

    sx, sy = grid.project(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    placed = grid.with_centres(cells)
    cx, cy = grid.project(placed["lat"].to_numpy(), placed["lon"].to_numpy())

    aes = stations["aes"].to_list()
    for i, row in enumerate(placed.iter_rows(named=True)):
        d = np.hypot(cx[i] - sx, cy[i] - sy)
        want = [aes[int(j)] for j in np.argsort(d, kind="stable")[:5]]
        have = (
            got.filter(
                (pl.col("cell_x") == row["cell_x"]) & (pl.col("cell_y") == row["cell_y"])
            )
            .sort("station_rank")["aes"]
            .to_list()
        )
        assert have == want


def test_nearest_stations_distances_are_sorted():
    rng = np.random.default_rng(11)
    stations = pl.DataFrame(
        {
            "aes": [f"S{i}" for i in range(30)],
            "lat": rng.uniform(48.0, 62.0, 30),
            "lon": rng.uniform(-130.0, -60.0, 30),
        }
    )
    cells = pl.DataFrame(
        {
            "cell_x": pl.Series([10, -40, 300], dtype=pl.Int32),
            "cell_y": pl.Series([10, 220, -80], dtype=pl.Int32),
        }
    )
    got = grid.nearest_stations(cells, stations, k=6)
    for (_, _), part in got.group_by(["cell_x", "cell_y"], maintain_order=True):
        d = part.sort("station_rank")["station_dist_km"].to_list()
        assert d == sorted(d)
