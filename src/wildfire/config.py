"""Central configuration: paths, tunables, and the modelling contract.

Everything that another module might want to tweak lives here so that the
feature builder and the model never disagree about, say, the label horizon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("WILDFIRE_DATA_DIR", ROOT / "data"))

RAW = DATA / "raw"          # landing zone: responses exactly as received
CURATED = DATA / "curated"  # parsed, typed parquet
DB_PATH = DATA / "wildfire.duckdb"
MODELS = DATA / "models"

for _p in (RAW, CURATED, MODELS):
    _p.mkdir(parents=True, exist_ok=True)


# --- source endpoints ------------------------------------------------------

CWFIS_DOWNLOADS = "https://cwfis.cfs.nrcan.gc.ca/downloads"
CWFIF_WFS = "https://geoserver.cwfif.nrcan.gc.ca/geoserver/wfs"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FIRMS_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CIFFC_SITREP = "https://ciffc.net/situation/"          # rendered page (fallback)
CIFFC_API = "https://api.ciffc.net/v1"                 # what the page itself calls

# NASA FIRMS needs a free MAP_KEY. Everything else in this project works
# without credentials -- FIRMS is the optional near-real-time / global leg.
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "").strip()

# Be a good citizen: these are public-good government services.
USER_AGENT = os.getenv(
    "WILDFIRE_USER_AGENT",
    "wildfire-forecast/0.1 (research project; contact: set WILDFIRE_USER_AGENT)",
)
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 4
RATE_LIMIT_SLEEP = 0.34  # ~3 req/s ceiling against any single host


# --- modelling contract ----------------------------------------------------


@dataclass(frozen=True)
class EscalationSpec:
    """The escalation problem, defined once.

    A fire enters the national feed at T0. We stand at T0 + `decision_hours`
    and ask: at T0 + `horizon_hours`, will it be a big fire?

    Every feature must be derivable from rows whose transaction-time validity
    window had already opened at the decision time. That is enforced in
    `features.asof`, not here, but the numbers live here.
    """

    decision_hours: int = 24
    horizon_hours: int = 72
    size_threshold_ha: float = 100.0

    # Spatial window for pulling satellite hotspots around a fire.
    hotspot_radius_km: float = 10.0

    # Fires whose first report is inside this many hours of the data cutoff
    # cannot have a label yet, so they are dropped from training.
    min_observability_hours: int = 72


SPEC = EscalationSpec()

# Fire-season months. Off-season reports are overwhelmingly administrative
# (prescribed burns, carry-overs) and behave differently.
FIRE_SEASON_MONTHS = (4, 5, 6, 7, 8, 9, 10)
